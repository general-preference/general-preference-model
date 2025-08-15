"""
    Abstract class for reward models.
"""
from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig, AutoModelForCausalLM
from transformers.deepspeed import HfDeepSpeedConfig
from typing import Optional
import deepspeed


class RewardModel:
    def __init__(self):
        raise NotImplementedError

    def pair_wise_scores(self, prompt: List[str], candidates: List[List[str]]):
        """
            Compute pairwise scores for prompt and candidates.
            Args:
            - prompt: (n, )
            - candidates: (n, num_candidates)

            Returns:
            - scores: (n, num_candidates, num_candidates) for each candidate pair
        """
        bsz = len(prompt)
        num_candidates = len(candidates[0])

        formatted = self.format_pair(prompt, candidates)    # size of (bsz, num_candidates * (num_candidates-1))
        formatted = self.regroup_formatted(formatted)       # size of (num_candidates * (num_candidates-1), bsz)

        inputs = self.format_input(formatted)   # size of bsz * dict

        scores = self.get_scores(inputs)

        scores_matrix = torch.zeros(bsz, num_candidates, num_candidates)
        # reshape scores as (bsz, num_candidates, (num_candidates-1))
        scores = scores.view(bsz, num_candidates, num_candidates-1)
        for i in range(num_candidates):
            scores_matrix[:, i, :i] = scores[:, i, :i]
            scores_matrix[:, i, i+1:] = scores[:, i, i:]

        return scores_matrix
    
    def format_pair(self, prompt: List[str], candidate: List[List[str]]):
        """
            Format a batch as model input.
            Args:
            - prompt: (bsz,)
            - candidate: (bsz, num_candidates)

            Returns:
            - formatted: (bsz, num_candidates * (num_candidates-1)) for the input contexts of all pairs
        """
        raise NotImplementedError

    def format_input(self, formatted: List[List[str]]):
        """
            Format input for model.
            Args:
            - formatted: (num_candidates * (num_candidates-1), bsz)

            Returns:
            - inputs: output of tokenizer of RMs
        """
        raise NotImplementedError

    def regroup_formatted(self, formatted: List[List[str]]):
        """
            Regroup formatted inputs.
            Args:
            - formatted: (bsz, num_candidates * (num_candidates-1))

            Returns:
            - regrouped: (num_candidates * (num_candidates-1), bsz)
        """
        # from group by batch to group by pair
        regrouped = []
        for i in range(len(formatted[0])):  # i: batch index
            regrouped.append([formatted[j][i] for j in range(len(formatted))])  # j: pair index
        return regrouped

    def get_scores(self, inputs):
        """
            Get scores from model.
            Args:
            - inputs: output of tokenizer of RMs, as (bsz, num_candidates * (num_candidates-1), ...)

            Returns:
            - scores: (bsz, num_candidates * (num_candidates-1))
        """
        raise NotImplementedError
    
def get_tokenizer(pretrain, model, padding_side="left", truncation_side="right", use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    tokenizer.truncation_side = truncation_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer

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
            with deepspeed.zero.GatheredParameters([model.value_head.weight], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    model.value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        else:
            model.value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
            
    if init_prompt_head and add_prompt_head:
        if dschf is not None:
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
                    reward = F.normalize(reward, p=2, dim=-1) 
                else:
                    eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1)
                    eos_indices = eos_indices.unsqueeze(1)  # Change shape to [batch_size, 1]                  
                    reward_list = []
                    for dim in range(value_head_dim):
                        reward_list.append(values[:,:,dim].gather(dim=1, index=eos_indices))
                    reward = torch.cat(reward_list, dim=1)
                    reward =  F.normalize(reward, p=2, dim=-1)  # Shape will be [batch_size, value_head_dim]
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

class CustomPairPreferenceModel(RewardModel):
    def __init__(self, model_name: str, use_flash_attn: bool=False, bf16: bool=True, is_general_preference: bool=False, value_head_dim: int=2, add_prompt_head: bool=False, tau: int=1, device: str = "cuda:0"):
        self.model =  get_reward_model(
            model_name,
            use_flash_attention_2=use_flash_attn,
            bf16=bf16,
            is_general_preference=is_general_preference,
            value_head_dim=value_head_dim,
            add_prompt_head=add_prompt_head
        )
        self.tokenizer = get_tokenizer(model_name, self.model, padding_side="left", truncation_side="right", use_fast=True)
        self.is_general_preference = is_general_preference
        self.value_head_dim = value_head_dim
        self.add_prompt_head = add_prompt_head
        self.tau = tau
    
        self.model.to(device)
        self.model.eval()

        print(f"Model loaded: {model_name}")
        print(f"Using General Preference model: {is_general_preference}")
        print(f"Value head dimension: {value_head_dim}")
        print(f"Add prompt head: {add_prompt_head}")
        print(f"Tau: {tau}")

    def format_pair(self, prompt: List[str], candidate: List[List[str]]):
        all_prompt = prompt
        formatted = []
        response_lengths = []
        for batch_idx in range(len(prompt)):
            prompt_text = all_prompt[batch_idx]
            candidates = candidate[batch_idx]
            batch_formatted = []
            batch_response_length = []
            for i in range(len(candidates)):
                context = [
                    {"role": "user", "content": prompt_text},
                    {"role": "assistant", "content": candidates[i]}
                ]
                context_str = self.tokenizer.apply_chat_template(context, tokenize=False).replace(self.tokenizer.bos_token, "")
                batch_formatted.append(context_str)
                if self.add_prompt_head:
                    prompt = [
                        {"role": "user", "content": prompt_text}
                    ]
                    prompt_str = self.tokenizer.apply_chat_template(prompt, tokenize=False).replace(self.tokenizer.bos_token, "")
                    context_length = self.tokenizer(context_str, padding=False, truncation=False, return_tensors="pt", add_special_tokens=False)["attention_mask"].sum()
                    prompt_length = self.tokenizer(prompt_str, padding=False, truncation=False, return_tensors="pt", add_special_tokens=False)["attention_mask"].sum()
                    response_length = context_length - prompt_length
                    batch_response_length.append(response_length)

            formatted.append(batch_formatted)
            response_lengths.append(batch_response_length)
            
        if self.add_prompt_head:
            return formatted, response_lengths
        else:
            return formatted
    
    def format_input(self, formatted: List[List[str]]):
        # apply left padding to align the output logits
        inputs = []
        for i, batch in enumerate(formatted):
            inputs.append(self.tokenizer(batch, padding=True, truncation=False, return_tensors="pt", add_special_tokens=False))
        
        return inputs

    def get_scores(self, inputs, response_lengths=None):
        with torch.no_grad():
            score_list = []
            rewards_cache = {}
            prompt_hidden_state_cache = {}

            for i in range(len(inputs)):
                input_i = {k: v.to(self.model.device) for k, v in inputs[i].items()}
                if hasattr(self.model, 'prompt_head'): 
                    reward_i, output_i = self.model.custom_forward(**input_i, return_output=True)
                    response_len_i = torch.tensor(response_lengths[i]).view(-1, 1).to("cuda")   
                    last_hidden_states_i = output_i["last_hidden_state"]
                    prompt_end_index = last_hidden_states_i.size(1) - response_len_i - 1
                    prompt_end_index_expanded = prompt_end_index.unsqueeze(-1).expand(-1, -1, last_hidden_states_i.size(-1))
                    prompt_hidden_state = torch.gather(last_hidden_states_i, dim=1, index=prompt_end_index_expanded).squeeze(1)
                    prompt_hidden_state_cache[i] = prompt_hidden_state
                else:
                    reward_i, _ = self.model.custom_forward(**input_i)
                rewards_cache[i] = reward_i

            for i in range(len(inputs)):
                for j in range(len(inputs)):
                    if i != j:
                        reward_i = rewards_cache[i]
                        reward_j = rewards_cache[j]
                        if hasattr(self.model, 'prompt_head'):
                            score = self.calculate_score(reward_i, reward_j, prompt_hidden_state_cache[i])
                        else:
                            score = self.calculate_score(reward_i, reward_j)
                        score_list.append(score)

            scores = torch.stack(score_list, dim=1)
        
        return scores
    
    def calculate_score(self, chosen, rejected, prompt_hidden_state=None):        
        if self.is_general_preference:
            if self.value_head_dim == 2:
                # chosen = chosen.squeeze(0)
                # rejected = rejected.squeeze(0)
                # score = chosen[0] * rejected[1] - chosen[1] * rejected[0]
                # xkp: 2024/9/22, handle bsz
                score = chosen[:, 0] * rejected[:, 1] - chosen[:, 1] * rejected[:, 0]
            elif not hasattr(self.model, 'prompt_head'):
                score = self.generate_high_dim_result(self.value_head_dim, chosen, rejected)
            else:    
                score = self.generate_high_dim_result_with_prompt(self.model, self.value_head_dim, chosen, rejected, prompt_hidden_state)
            score = score / self.tau
        else:
            score = chosen - rejected
        return score
    
    def pair_wise_scores(self, prompt: List[str], candidates: List[List[str]]):
        """
            Compute pairwise scores for prompt and candidates.
            Args:
            - prompt: (n, )
            - candidates: (n, num_candidates)

            Returns:
            - scores: (n, num_candidates, num_candidates) for each candidate pair
        """
        bsz = len(prompt)
        num_candidates = len(candidates[0])

        if not self.add_prompt_head:
            formatted = self.format_pair(prompt, candidates)    # size of (bsz, num_candidates * (num_candidates-1))
            formatted = self.regroup_formatted(formatted)       # size of (num_candidates * (num_candidates-1), bsz)

            inputs = self.format_input(formatted)   # size of bsz * dict

            scores = self.get_scores(inputs)
        else:
            formatted, response_lengths = self.format_pair(prompt, candidates)    # size of (bsz, num_candidates * (num_candidates-1))
            formatted = self.regroup_formatted(formatted)       # size of (num_candidates * (num_candidates-1), bsz)
            response_lengths = self.regroup_formatted(response_lengths)
            
            inputs = self.format_input(formatted)   # size of bsz * dict
            
            scores = self.get_scores(inputs, response_lengths)
            
        scores_matrix = torch.zeros(bsz, num_candidates, num_candidates)
        # reshape scores as (bsz, num_candidates, (num_candidates-1))
        scores = scores.view(bsz, num_candidates, num_candidates-1)
        for i in range(num_candidates):
            scores_matrix[:, i, :i] = scores[:, i, :i]
            scores_matrix[:, i, i+1:] = scores[:, i, i:]

        return scores_matrix
  
    def generate_high_dim_result(self, value_head_dim, chosen_reward, rejected_reward):
        R_matrix = torch.zeros((value_head_dim, value_head_dim), device=chosen_reward.device, dtype=chosen_reward.dtype)
        for i in range(0, value_head_dim, 2):
            R_matrix[i, i+1] = -1 
            R_matrix[i+1, i] = 1   
        if chosen_reward.device == rejected_reward.device == R_matrix.device:
            transformed_chosen = torch.matmul(chosen_reward, R_matrix.T)
            result = torch.bmm(transformed_chosen.view(chosen_reward.shape[0], 1, value_head_dim), rejected_reward.view(rejected_reward.shape[0], value_head_dim, 1))
            result = result.view(chosen_reward.shape[0])  
        return result
    
    def generate_high_dim_result_with_prompt(self, model, value_head_dim, chosen_reward, rejected_reward, prompt_hidden_states):
        R_matrix = model.create_skew_symmetric_block_matrix(value_head_dim, chosen_reward.device, chosen_reward.dtype, prompt_hidden_states)
        if chosen_reward.device == rejected_reward.device == R_matrix.device:
            transformed_chosen = torch.bmm(chosen_reward.view(chosen_reward.shape[0], 1, value_head_dim), R_matrix.transpose(1, 2))
            result = torch.bmm(transformed_chosen, rejected_reward.view(rejected_reward.shape[0], value_head_dim, 1))
            result = result.view(chosen_reward.shape[0])  
        return result
        


class PairPreferenceLlama(RewardModel):
    """
        Following format of https://huggingface.co/RLHFlow/pair-preference-model-LLaMA3-8B
    """
    PLAIN_TEMPLATE="\n{% for message in messages %}{% if loop.index0 % 2 == 0 %}\n\n<turn> user\n {{ message['content'] }}{% else %}\n\n<turn> assistant\n {{ message['content'] }}{% endif %}{% endfor %}\n\n\n"
    PROMPT_TEMPLATE="[CONTEXT] {context} [RESPONSE A] {response_A} [RESPONSE B] {response_B} \n"
    def __init__(self, model_name: str, device: str = "cuda:0"):
        # laod as fp16
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer_plain = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer_plain.chat_template = self.PLAIN_TEMPLATE    # for formatting the context
        self.tokenizer.padding_side = "left"    # to align the output logits

        # tokens
        self.token_id_A = self.tokenizer.encode("A", add_special_tokens=False)
        self.token_id_B = self.tokenizer.encode("B", add_special_tokens=False)
        assert len(self.token_id_A) == 1 and len(self.token_id_B) == 1
        self.token_id_A = self.token_id_A[0]
        self.token_id_B = self.token_id_B[0]

        self.model.to(device)
        self.model.eval()

    def format_pair(self, prompt: List[str], candidate: List[List[str]]):
        all_prompt = prompt
        formatted = []
        for batch_idx in range(len(prompt)):
            prompt_text = all_prompt[batch_idx]
            candidates = candidate[batch_idx]
            batch_formatted = []
            for i in range(len(candidates)):
                for j in range(len(candidates)):
                    if i != j:
                        context = [
                            {"role": "user", "content": prompt_text},
                        ]
                        context_str = self.tokenizer_plain.apply_chat_template(context, tokenize=False)
                        prompt = self.PROMPT_TEMPLATE.format(context=context_str, response_A=candidates[i], response_B=candidates[j])
                        message = [
                            {"role": "user", "content": prompt},
                        ]
                        message_str = self.tokenizer.apply_chat_template(message, tokenize=False).replace(self.tokenizer.bos_token, "")
                        batch_formatted.append(message_str)

            formatted.append(batch_formatted)

        return formatted

    def format_input(self, formatted: List[List[str]]):
        # apply left padding to align the output logits
        inputs = []
        for i, batch in enumerate(formatted):
            inputs.append(self.tokenizer(batch, padding=True, return_tensors="pt", add_special_tokens=False))
        return inputs

    def get_scores(self, inputs):
        # inputs: num_candidates * (num_candidates-1), bsz

        with torch.no_grad():
            scores = []
            for batch_inputs in inputs:
                batch_inputs = {k: v.to(self.model.device) for k, v in batch_inputs.items()}
                outputs = self.model(**batch_inputs)
                # take out the last token logits
                logits = outputs.logits[:, -1]
                # take out the logit for logits_A and logits_B
                logits_A = logits[:, self.token_id_A]
                logits_B = logits[:, self.token_id_B]   # (bsz,)
                scores.append(logits_A - logits_B)

            scores = torch.stack(scores, dim=1)
        return scores
