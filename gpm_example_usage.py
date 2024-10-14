from typing import Optional, List, Dict
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
import torch.nn.functional as F
from transformers import AutoTokenizer

def get_tokenizer(pretrain, model, padding_side="left", use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer

def get_reward_model(base_causal_model, base_llm_model, is_general_preference: bool=False, add_prompt_head: bool=False, value_head_dim: int=2):
    class CustomRewardModel(base_causal_model):

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
                    reward =  F.normalize(reward, p=2, dim=-1)  # Shape will be [batch_size, value_head_dim]
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

def generate_high_dim_result_with_prompt(model, value_head_dim, chosen_reward, rejected_reward, prompt_hidden_states):
    R_matrix = model.create_skew_symmetric_block_matrix(value_head_dim, chosen_reward.device, chosen_reward.dtype, prompt_hidden_states)
    if chosen_reward.device == rejected_reward.device == R_matrix.device:
        transformed_chosen = torch.bmm(chosen_reward.view(chosen_reward.shape[0], 1, value_head_dim), R_matrix.transpose(1, 2))
        result = torch.bmm(transformed_chosen, rejected_reward.view(rejected_reward.shape[0], value_head_dim, 1))
        result = result.view(chosen_reward.shape[0])  
    return result

class GPMPipeline:
    def __init__(self, model_name_or_path, device=torch.device("cuda:0"), is_general_preference: bool=True, add_prompt_head: bool=True, value_head_dim: int=2, bf16: bool=True, truncation: bool=True, max_length: int=4096, padding: bool=True, tau: float=0.1):
        self.device = device
        self.is_general_preference = is_general_preference
        self.add_prompt_head = add_prompt_head
        self.value_head_dim = value_head_dim
        self.truncation = truncation
        self.max_length = max_length
        self.padding = padding
        self.tau = tau
        
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        config._attn_implementation = "flash_attention_2" 
        base_class = AutoModel._model_mapping[type(config)]
        base_causal_class = AutoModelForCausalLM._model_mapping.get(type(config), None)
        cls_class = get_reward_model(base_causal_class, base_class, is_general_preference, add_prompt_head, value_head_dim)

        # configure model
        self.model = cls_class.from_pretrained(
            model_name_or_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if bf16 else "auto",
        )
        # configure tokenizer
        self.tokenizer = get_tokenizer(model_name_or_path, self.model, "left", use_fast=True)
        self.tokenizer.truncation_side = "right"
        
        # prepare model
        self.model.to(device)
        self.model.eval()

    def __call__(self, samples: List[List[Dict[str, str]]], return_prompt=False):
        input_texts = [self.tokenizer.apply_chat_template(sample, tokenize=False) for sample in samples]

        inputs = self.tokenizer(
            input_texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        inputs["input_ids"][:, -1] = self.tokenizer.eos_token_id
        inputs["attention_mask"][:, -1] = 1

        with torch.no_grad():
            rewards, outputs = self.model.custom_forward(**inputs, return_output=return_prompt)

        if return_prompt:
            # Compute prompt hidden states
            prompt_texts = [self.tokenizer.apply_chat_template([sample[0]], tokenize=False) for sample in samples]
            prompt_lengths = [len(self.tokenizer(prompt_text, padding=False, return_tensors="pt")["input_ids"][0]) for prompt_text in prompt_texts]
            prompt_lengths = torch.tensor(prompt_lengths, device=self.device)
            prompt_end_indices = prompt_lengths - 1

            last_hidden_states = outputs.last_hidden_state
            prompt_hidden_states = last_hidden_states[torch.arange(len(samples)), prompt_end_indices, :]

            return rewards, prompt_hidden_states

        return rewards


prompt_text = "Describe the importance of reading books in today's digital age."
response1 = "Books remain crucial in the digital era, offering in-depth knowledge and fostering critical thinking. They provide a unique, immersive experience that digital media can't replicate, contributing significantly to personal and intellectual growth."
response2 = "Books are still useful for learning new things. They help you relax and can be a good break from screens."

context1 = [
    {"role": "user", "content": prompt_text},
    {"role": "assistant", "content": response1}
]

context2 = [
    {"role": "user", "content": prompt_text},
    {"role": "assistant", "content": response2}
]

rm = GPMPipeline("general-preference/GPM-Llama-3.1-8B", value_head_dim=4)

reward1, prompt_hidden_state = rm([context1], return_prompt=True)
reward2 = rm([context2])

result = generate_high_dim_result_with_prompt(rm.model, rm.value_head_dim, reward1, reward2, prompt_hidden_state)

result_batch = result.float().cpu().detach().numpy().tolist()

results = []
[
    results.append(1) if result > 0 else results.append(0)
    for result in result_batch
]

print(result_batch)

