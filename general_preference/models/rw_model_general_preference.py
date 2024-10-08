# Adapted from https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/models/model.py

from typing import Optional
import deepspeed
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig, AutoModelForCausalLM
from transformers.deepspeed import HfDeepSpeedConfig
from general_preference.utils.logging import init_logger
import torch.nn.functional as F

logger = init_logger(__name__)

# Construct reward model with a value head for sequence classification. (model also with a lm head) 
def get_reward_model(
    model_name_or_path: str,
    *,
    bf16=True,
    load_in_4bit=False,
    lora_rank=0,
    lora_alpha=16,
    target_modules=None,
    lora_dropout=0,
    use_flash_attention_2=False,
    ds_config: dict = None,
    init_value_head: bool = False,
    init_prompt_head: bool = False,
    add_prompt_head: bool = False,
    is_general_preference: bool = False,
    value_head_dim: int = 2,
    **kwargs,
) -> nn.Module:
    """Get reward model with a value head(linear layer) and a lm head.

    Args:
        model_name_or_path (str): Path to pretrained model.
        bf16 (bool, optional): Whether enable bfloat16. Defaults to True.
        use_flash_attention_2 (bool, optional): Whether use Flash Attention 2.0. Defaults to False.
        ds_config (dict, optional): Deepspeed config, used to automatically splitting the model onto multiple gpus during from_pretrained when ZeRO-3 enabled. Defaults to None.
        init_value_head (bool, optional): Whether to initialize the value head weights. Defaults to False.
        is_general_preference (bool, optional): Whether to use General Preference model. Defaults to False (Bradley Terry model by default).
        value_head_dim (int, optional): Dimension of value head for General Prefernce model. Ignored by the Bradley Terry model. Defaults to 2.

    Returns:
        nn.Module: pretrained transformer model.
    """

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"
    base_class = AutoModel._model_mapping[type(config)]
    base_causal_class = AutoModelForCausalLM._model_mapping.get(type(config), None)
    cls_class = _get_reward_model(base_causal_class, base_class, is_general_preference, add_prompt_head, value_head_dim)
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    if load_in_4bit:
        assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        nf4_config = None

    model = cls_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        quantization_config=nf4_config,
        **kwargs,
    )
    # LoRA
    if lora_rank > 0:
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        if load_in_4bit:
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module.to(torch.bfloat16)
                if "norm" in name:
                    module.to(torch.float32)
                if "value_head" in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        module.to(torch.bfloat16)

    if init_value_head:
        if dschf is not None:
            logger.info("Initialize value_head for ZeRO-3 reward model training.")
            with deepspeed.zero.GatheredParameters([model.value_head.weight], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    model.value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        else:
            model.value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
            
    if init_prompt_head and add_prompt_head:
        if dschf is not None:
            logger.info("Initialize prompt_head for ZeRO-3 reward model training.")
            with deepspeed.zero.GatheredParameters([model.prompt_head.weight], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    model.prompt_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        else:
            model.prompt_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        
    return model

def _get_reward_model(base_causal_model, base_llm_model, is_general_preference: bool=False, add_prompt_head: bool=False, value_head_dim: int=2):
    class CustomRewardModel(base_causal_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))
            if not is_general_preference:
                self.value_head = nn.Linear(config.hidden_size, 1, bias=False)
            else: 
                self.value_head = nn.Linear(config.hidden_size, value_head_dim, bias=False) 
                if add_prompt_head:
                    self.prompt_head = nn.Linear(config.hidden_size, value_head_dim // 2, bias=False) 
        
            self.is_general_preference = is_general_preference    
            
            self.post_init()

        def custom_forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
        ) -> torch.Tensor:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]
            
            if not self.is_general_preference:
                values = self.value_head(last_hidden_states).squeeze(-1)
                # left padding in training mode
                if self.training:
                    reward = values[:, -1]
                else:
                    eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                    reward = values.gather(dim=1, index=eos_indices).squeeze(1)
                if return_output:
                    return reward, outputs
                else:
                    return reward, None
            else:
                values = self.value_head(last_hidden_states)
                # left padding in training mode
                if self.training:
                    reward = values[:, -1, :]
                    # reward =  F.normalize(reward, p=2, dim=-1)  # Shape will be [batch_size, value_head_dim]
                    # if not hasattr(self, 'prompt_head'):
                    #     reward = F.normalize(reward, p=2, dim=-1) 
                else:
                    eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1)
                    eos_indices = eos_indices.unsqueeze(1)  # Change shape to [batch_size, 1]                  
                    reward_list = []
                    for dim in range(value_head_dim):
                        reward_list.append(values[:,:,dim].gather(dim=1, index=eos_indices))
                    reward = torch.cat(reward_list, dim=1)
                    # reward =  F.normalize(reward, p=2, dim=-1)  # Shape will be [batch_size, value_head_dim]
                    # if not hasattr(self, 'prompt_head'):
                    #     reward =  F.normalize(reward, p=2, dim=-1)  # Shape will be [batch_size, value_head_dim]
                if return_output:
                    return reward, outputs
                else:
                    return reward, None
        
        def create_skew_symmetric_block_matrix(self, dim, device, dtype, prompt_hidden_states):
            """
            Create a batch of skew-symmetric block matrices where each matrix is data-dependent on
            the corresponding prompt_hidden_states. Only the relevant block diagonal parts are generated.
            
            Args:
            - dim: Dimension of the square matrix (must be even).
            - prompt_hidden_states: Tensor of shape [batch_size, hidden_dim].
            
            Returns:
            - batch_R_matrices: Tensor of shape [batch_size, dim, dim], with skew-symmetric block entries.
            """
            if hasattr(self, 'prompt_head'):
                batch_size = prompt_hidden_states.shape[0]
                
                # Ensure that dim is even, as we're creating blocks of size 2x2
                assert dim % 2 == 0, "dim must be even for skew-symmetric block generation"

                # Pass through the linear layer to get the block diagonal entries (half of the matrix's off-diagonal blocks)
                block_values = self.prompt_head(prompt_hidden_states).view(batch_size, dim // 2)
                block_values = torch.softmax(block_values, dim=-1)
                
                # Create a batch of zero matrices [batch_size, dim, dim]
                batch_R_matrices = torch.zeros((batch_size, dim, dim), device=device, dtype=dtype)
                
                # Fill only the block diagonal entries with the learned values
                for i in range(0, dim, 2):
                    batch_R_matrices[:, i, i + 1] = -block_values[:, i // 2]
                    batch_R_matrices[:, i + 1, i] = block_values[:, i // 2]  # Skew-symmetric condition
            else:
                raise AttributeError("prompt_head is not defined. Ensure 'add_prompt_head' is set to True during initialization.")
                
            return batch_R_matrices
        
                
    return CustomRewardModel






