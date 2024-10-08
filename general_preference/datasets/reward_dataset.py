from typing import Callable
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from .utils import exist_and_not_none, zero_pad_sequences

def preprocess_data(data, is_custom=False):
    if is_custom:
        if "chosen" in data and "response" in data["chosen"]:
            prompt = data["instruction"]
            chosen = data["chosen"]["response"]
            reject = data["reject"]["response"]
        else:
            prompt = data["instruction"]
            chosen = data["chosen"]
            reject = data["reject"]
        margin = data["margin"] if exist_and_not_none(data, "margin") else 0
        return prompt, chosen, reject, margin
    else:
        chosen = data["chosen"]
        reject = data["rejected"]
        margin = data["margin"] if exist_and_not_none(data, "margin") else 0
        return None, chosen, reject, margin    
class GeneralRewardDataset(Dataset):
    """
    General dataset for reward model, handling both custom and standard formats.

    Args:
        dataset: dataset for reward model
        tokenizer: tokenizer for reward model
        max_length: max length of input
        is_custom: flag indicating whether the dataset is in custom format
    """
    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        is_custom=False,
        return_prompt_length=False,
    ) -> None:
        super().__init__()
        self.prompts = []
        self.chosens = []
        self.rejects = []
        self.margins = []
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.is_custom = is_custom
        self.return_prompt_length = return_prompt_length

        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt, chosen, reject, margin = preprocess_data(data, is_custom=self.is_custom)
            self.prompts.append(prompt)
            self.chosens.append(chosen)
            self.rejects.append(reject)
            self.margins.append(margin)

    def __len__(self):
        return len(self.chosens)

    def __getitem__(self, idx):
        prompt, chosen, reject, margin = self.prompts[idx], self.chosens[idx], self.rejects[idx], self.margins[idx]

        if prompt:
            chosen = prompt + ' ' + chosen + ' ' + self.tokenizer.eos_token
            reject = prompt + ' ' + reject + ' ' + self.tokenizer.eos_token
        else:
            if self.tokenizer.chat_template is not None:
                if self.return_prompt_length:
                    prompt = self.tokenizer.apply_chat_template([chosen[0]], tokenize=False)
                chosen = self.tokenizer.apply_chat_template(chosen, tokenize=False)
                reject = self.tokenizer.apply_chat_template(reject, tokenize=False)
            else:
                if self.return_prompt_length:
                    prompt = chosen[0]['content']
                chosen = ' '.join([d['content'] for d in chosen]) + ' ' + self.tokenizer.eos_token
                reject = ' '.join([d['content'] for d in reject]) + ' ' + self.tokenizer.eos_token

        chosen_token = self.tokenizer(
            chosen,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        #### You may consider removing an additional bos for Llama3.1 8b Instruct
        # if not self.is_custom and self.tokenizer.chat_template is not None and self.tokenizer.bos_token_id == 128000:
        #     chosen_token["input_ids"] = chosen_token["input_ids"][:, 1:]
        #     chosen_token["attention_mask"] = chosen_token["attention_mask"][:, 1:]
        reject_token = self.tokenizer(
            reject,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        #### You may consider removing an additional bos for Llama3.1 8b Instruct
        # if not self.is_custom and self.tokenizer.chat_template is not None and self.tokenizer.bos_token_id == 128000:
        #     reject_token["input_ids"] = reject_token["input_ids"][:, 1:]
        #     reject_token["attention_mask"] = reject_token["attention_mask"][:, 1:]
        if self.return_prompt_length:
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
            )
            #### You may consider removing an additional bos for Llama3.1 8b Instruct
            # if not self.is_custom and self.tokenizer.chat_template is not None and self.tokenizer.bos_token_id == 128000:
            #     prompt_token["input_ids"] = prompt_token["input_ids"][:, 1:]
            #     prompt_token["attention_mask"] = prompt_token["attention_mask"][:, 1:]
            chosen_response_len = chosen_token["attention_mask"].sum() - prompt_token["attention_mask"].sum()
        else:
            chosen_response_len = 0

        # To avoid EOS_token truncation
        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        reject_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True
        reject_token["attention_mask"][0][-1] = True
        
        return (
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
            reject_token["input_ids"],
            reject_token["attention_mask"],
            margin,
            chosen_response_len,
        )

    def collate_fn(self, item_list):
        chosen_ids = []
        chosen_masks = []
        reject_ids = []
        reject_masks = []
        margins = []
        chosen_response_lens = []

        for chosen_id, chosen_mask, reject_id, reject_mask, margin, chosen_response_len in item_list:
            chosen_ids.append(chosen_id)
            chosen_masks.append(chosen_mask)
            reject_ids.append(reject_id)
            reject_masks.append(reject_mask)
            margins.append(margin)
            chosen_response_lens.append(chosen_response_len)

        chosen_ids = zero_pad_sequences(chosen_ids, value=self.tokenizer.pad_token_id)
        chosen_masks = zero_pad_sequences(chosen_masks)
        reject_ids = zero_pad_sequences(reject_ids, value=self.tokenizer.pad_token_id)
        reject_masks = zero_pad_sequences(reject_masks)

        return chosen_ids, chosen_masks, reject_ids, reject_masks, margins, chosen_response_lens