import torch

def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
    return model

class CustomLeftPadRewardBenchPipeline:
    def __init__(self, task, model, tokenizer):
        self.task = task
        self.model = disable_dropout_in_model(model).eval()
        self.tokenizer = tokenizer

    def __call__(self, samples, return_prompt: bool=False, **kwargs):
        _ = kwargs.get("batch_size", 1)
        self.truncation = kwargs.get("truncation", True)
        self.padding = kwargs.get("padding", True)
        self.max_length = kwargs.get("max_length", 2048)

        input_texts = [self.tokenizer.apply_chat_template(sample, tokenize=False) for sample in samples]

        inputs = self.tokenizer(
            input_texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        inputs["input_ids"][:, -1] = self.tokenizer.eos_token_id
        inputs["attention_mask"][:, -1] = 1

        with torch.no_grad():
            rewards, outputs = self.model.custom_forward(**inputs, return_output=return_prompt)

        chosen_response_len_list = []
        if return_prompt:
            prompt_texts = [self.tokenizer.apply_chat_template([sample[0]], tokenize=False) for sample in samples]
            for i in range(len(input_texts)):
                prompt_token = self.tokenizer(
                    prompt_texts[i],
                    max_length=self.max_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                )
                chosen_token = self.tokenizer(
                    input_texts[i],
                    max_length=self.max_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                )
                chosen_response_len = chosen_token["attention_mask"].sum() - prompt_token["attention_mask"].sum()
                chosen_response_len_list.append(chosen_response_len)
        chosen_response_len = torch.tensor(chosen_response_len_list).view(-1, 1).to("cuda")
        if return_prompt:   
            chosen_last_hidden_states = outputs["last_hidden_state"]
            prompt_end_index = chosen_last_hidden_states.size(1) - chosen_response_len - 1
            prompt_end_index_expanded = prompt_end_index.unsqueeze(-1).expand(-1, -1, chosen_last_hidden_states.size(-1))
            prompt_hidden_state = torch.gather(chosen_last_hidden_states, dim=1, index=prompt_end_index_expanded).squeeze(1)
            return rewards, prompt_hidden_state
        else:
            return rewards   

def generate_high_dim_result(value_head_dim, chosen_reward, rejected_reward):
    R_matrix = torch.zeros((value_head_dim, value_head_dim), device=chosen_reward.device, dtype=chosen_reward.dtype)
    for i in range(0, value_head_dim, 2):
        R_matrix[i, i+1] = -1 
        R_matrix[i+1, i] = 1   
    if chosen_reward.device == rejected_reward.device == R_matrix.device:
        transformed_chosen = torch.matmul(chosen_reward, R_matrix.T)
        result = torch.bmm(transformed_chosen.view(chosen_reward.shape[0], 1, value_head_dim), rejected_reward.view(rejected_reward.shape[0], value_head_dim, 1))
        result = result.view(chosen_reward.shape[0])  
    return result
    
def generate_high_dim_result_with_prompt(model, value_head_dim, chosen_reward, rejected_reward, prompt_hidden_states):
    R_matrix = model.create_skew_symmetric_block_matrix(value_head_dim, chosen_reward.device, chosen_reward.dtype, prompt_hidden_states)
    if chosen_reward.device == rejected_reward.device == R_matrix.device:
        transformed_chosen = torch.bmm(chosen_reward.view(chosen_reward.shape[0], 1, value_head_dim), R_matrix.transpose(1, 2))
        result = torch.bmm(transformed_chosen, rejected_reward.view(rejected_reward.shape[0], value_head_dim, 1))
        result = result.view(chosen_reward.shape[0])  
    return result
    

