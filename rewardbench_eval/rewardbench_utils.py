import torch

def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
    return model

# class CustomRewardBenchPipeline:
#     def __init__(self, task, model, tokenizer):
#         self.task = task
#         self.model = disable_dropout_in_model(model).eval()
#         self.tokenizer = tokenizer

#     def __call__(self, samples, return_inputs=False, **kwargs):
#         _ = kwargs.get("batch_size", 1)
#         truncation = kwargs.get("truncation", True)
#         padding = kwargs.get("padding", True)
#         max_length = kwargs.get("max_length", 2048)
#         inputs = self.tokenizer(
#             samples,
#             truncation=truncation,
#             max_length=max_length,
#             padding=padding,
#             # return_special_tokens_mask=True,
#             return_tensors="pt",
#         ).to("cuda")
#         with torch.no_grad():
#             reward, _ = self.model.custom_forward(**inputs)
#         return reward
    
class CustomRightPadRewardBenchPipeline:
    def __init__(self, task, model, tokenizer):
        self.task = task
        self.model = disable_dropout_in_model(model).eval()
        self.tokenizer = tokenizer

    def __call__(self, samples, return_prompt: bool=False, return_inputs=False, **kwargs):
        _ = kwargs.get("batch_size", 1)
        truncation = kwargs.get("truncation", True)
        padding = kwargs.get("padding", True)
        max_length = kwargs.get("max_length", 2048)

        input_samples = []
        input_prompts = []
        prompt_end_index_list = []

        for sample in samples:
            input_samples.append(self.tokenizer.apply_chat_template(sample, tokenize=False))
        if return_prompt:
            for sample in samples:
                input_prompts.append(self.tokenizer.apply_chat_template([sample[0]], tokenize=False))

        inputs = self.tokenizer(
            input_samples,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            # return_special_tokens_mask=True,
            return_tensors="pt",
        ).to("cuda")
        # inputs["input_ids"] = inputs["input_ids"][:, 1:]
        # inputs["attention_mask"] = inputs["attention_mask"][:, 1:]
        if inputs["attention_mask"][0][-1] == True:
            inputs["input_ids"][0][-1] = self.tokenizer.eos_token_id

        # print("inputs_shape", inputs["input_ids"].shape)

        with torch.no_grad():
            reward, outputs = self.model.custom_forward(**inputs, return_output=return_prompt)
        
        if return_prompt:
            for input_prompt in input_prompts:
                prompt_token = self.tokenizer(
                    input_prompt,
                    max_length=max_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                )
                # prompt_token["input_ids"] = prompt_token["input_ids"][:, 1:]
                # prompt_token["attention_mask"] = prompt_token["attention_mask"][:, 1:]
                prompt_end_index = prompt_token["attention_mask"].sum() - 1  
                prompt_end_index_list.append(prompt_end_index)
        prompt_end_index = torch.tensor(prompt_end_index_list).view(-1, 1).to("cuda")
        if return_prompt:   
            chosen_last_hidden_states = outputs["last_hidden_state"]
            prompt_end_index_expanded = prompt_end_index.unsqueeze(-1).expand(-1, -1, chosen_last_hidden_states.size(-1))
            prompt_hidden_state = torch.gather(chosen_last_hidden_states, dim=1, index=prompt_end_index_expanded).squeeze(1)
            return reward, prompt_hidden_state
        else:
            return reward       
    
class CustomLeftPadRewardBenchPipeline:
    def __init__(self, task, model, tokenizer):
        self.task = task
        self.model = disable_dropout_in_model(model).eval()
        self.tokenizer = tokenizer

    def __call__(self, samples, return_prompt: bool=False, return_inputs=False, **kwargs):
        _ = kwargs.get("batch_size", 1)
        truncation = kwargs.get("truncation", True)
        padding = kwargs.get("padding", True)
        max_length = kwargs.get("max_length", 2048)

        input_samples = []
        input_prompts = []
        chosen_response_len_list = []

        for sample in samples:
            input_samples.append(self.tokenizer.apply_chat_template(sample, tokenize=False))
        if return_prompt:
            for sample in samples:
                input_prompts.append(self.tokenizer.apply_chat_template([sample[0]], tokenize=False))

        inputs = self.tokenizer(
            input_samples,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            # return_special_tokens_mask=True,
            return_tensors="pt",
        ).to("cuda")
        inputs["input_ids"][0][-1] = self.tokenizer.eos_token_id
        inputs["attention_mask"][0][-1] = True

        # print("inputs_shape", inputs["input_ids"].shape)

        with torch.no_grad():
            reward, outputs = self.model.custom_forward(**inputs, return_output=return_prompt)
        
        if return_prompt:
            for i in range(len(input_prompts)):
                prompt_token = self.tokenizer(
                    input_prompts[i],
                    max_length=max_length,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                )
                chosen_token = self.tokenizer(
                    input_samples[i],
                    max_length=max_length,
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
            return reward, prompt_hidden_state
        else:
            return reward   

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
    

