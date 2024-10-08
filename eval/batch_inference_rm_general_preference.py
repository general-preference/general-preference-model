import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from general_preference.datasets import GeneralRewardDataset
from general_preference.utils import blending_datasets, get_strategy, get_tokenizer
from general_preference.models import get_reward_model
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """
    def __init__(self, tau: float = 0.1):
        super().__init__()
        self.tau = tau
        
    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        if margin is not None:
            loss = -F.logsigmoid((chosen_reward - reject_reward - margin) / self.tau)
            prob = F.sigmoid((chosen_reward - reject_reward - margin) / self.tau)
            result = chosen_reward - reject_reward
        else:
            loss = -F.logsigmoid((chosen_reward - reject_reward) / self.tau)
            prob = F.sigmoid((chosen_reward - reject_reward) / self.tau)
            result= chosen_reward - reject_reward
        return loss, prob, result
class GeneralPreferenceLoss(nn.Module):
    """
    Loss for General Preference Reward Model
    """
    def __init__(self, tau: float = 1):
        super().__init__()
        self.tau = tau
        
    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        if margin is not None:
            result = chosen_reward[:, 0] * reject_reward[:, 1] - chosen_reward[:, 1] * reject_reward[:, 0]
            loss = -F.logsigmoid((result - margin) / self.tau)
            prob = F.sigmoid((result - margin) / self.tau)
        else:
            result = chosen_reward[:, 0] * reject_reward[:, 1] - chosen_reward[:, 1] * reject_reward[:, 0]
            loss = -F.logsigmoid(result / self.tau)
            prob = F.sigmoid(result / self.tau)
        return loss, prob, result
class HighDimGeneralPreferenceLoss(nn.Module):
    """
    Loss for General Preference Reward Model with high dimension value_head
    """
    def __init__(self, tau: float = 0.1, value_head_dim: int = 4, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tau = tau
        self.value_head_dim = value_head_dim
        
    def create_skew_symmetric_block_matrix(self, dim, device, dtype):
        matrix = torch.zeros((dim, dim), device=device, dtype=dtype)
        for i in range(0, dim, 2):
            matrix[i, i+1] = -1 
            matrix[i+1, i] = 1            
        return matrix

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None 
    ) -> torch.Tensor:
        if margin is not None:        
            R_matrix = self.create_skew_symmetric_block_matrix(self.value_head_dim, chosen_reward.device, chosen_reward.dtype)
            if chosen_reward.device == reject_reward.device == R_matrix.device:
                transformed_chosen = torch.matmul(chosen_reward, R_matrix.T)
                result = torch.bmm(transformed_chosen.view(chosen_reward.shape[0], 1, self.value_head_dim), reject_reward.view(reject_reward.shape[0], self.value_head_dim, 1))
                result = result.view(chosen_reward.shape[0])              
            loss = -F.logsigmoid((result - margin) / self.tau) 
            prob = F.sigmoid((result - margin) / self.tau)
        else:
            R_matrix = self.create_skew_symmetric_block_matrix(self.value_head_dim, chosen_reward.device, chosen_reward.dtype)
            if chosen_reward.device == reject_reward.device == R_matrix.device:
                transformed_chosen = torch.matmul(chosen_reward, R_matrix.T)
                result = torch.bmm(transformed_chosen.view(chosen_reward.shape[0], 1, self.value_head_dim), reject_reward.view(reject_reward.shape[0], self.value_head_dim, 1))
                result = result.view(chosen_reward.shape[0])  
            loss = -F.logsigmoid(result / self.tau)
            prob = F.sigmoid(result / self.tau)
        return loss, prob, result

def batch_rm_inference(args):
    
    class Empty:
        pass

    strategy = Empty()
    strategy.print = print
    strategy.is_rank_0 = lambda: True
    strategy.args = args

    device = torch.device("cuda:0")

    # configure model
    model = get_reward_model(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        is_general_preference=args.is_general_preference,
        value_head_dim=args.value_head_dim,
    )
    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, model, "left", strategy=None, use_fast=not args.disable_fast_tokenizer)
    tokenizer.truncation_side = "right"
    model.to(device)
    # prepare models
    model.eval()

    dataset = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
    )
    dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    dataset = GeneralRewardDataset(dataset, tokenizer, args.max_len, strategy, args.is_custom_dataset)
    sampler = DistributedSampler(
        dataset,
        num_replicas=1,
        rank=0,
        shuffle=False,
        seed=args.seed,
        drop_last=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        sampler=sampler,
        drop_last=False,
        collate_fn=dataset.collate_fn,
        pin_memory=False,
    )

    pbar = tqdm(
        dataloader,
        disable=not strategy.is_rank_0(),
    )

    if args.is_general_preference:
        if args.value_head_dim == 2:
            loss_fn = GeneralPreferenceLoss(tau=args.general_preference_tau)
            strategy.print("General Preference Loss")
        else:
            assert args.value_head_dim % 2 == 0, "Dimension of value head for general preference model can not be odd!"
            loss_fn = HighDimGeneralPreferenceLoss(tau=args.general_preference_tau, value_head_dim=args.value_head_dim)
            strategy.print("Loss for high-dimensional value head General Preference model.")
    else:
        loss_fn = PairWiseLoss(tau=args.general_preference_tau)
        strategy.print("LogSigmoid Loss")
                
    all_probs = []
    all_results = []
    
    with torch.no_grad():
        for chosen_ids, chosen_masks, reject_ids, reject_masks, _, _ in pbar:
            chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
            chosen_masks = chosen_masks.squeeze(1).to(torch.cuda.current_device())
            
            chosen_rewards, _ = model.custom_forward(chosen_ids, chosen_masks)
            
            reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
            reject_masks = reject_masks.squeeze(1).to(torch.cuda.current_device())
            
            reject_rewards, _ = model.custom_forward(reject_ids, reject_masks)
            
            preference_loss, prob, result = loss_fn(chosen_rewards, reject_rewards)
            print("chosen_rewards", chosen_rewards)
            print("reject_rewards", reject_rewards)
            print("preference loss", preference_loss)
            print("prob", prob)
            
            prob = prob.float().cpu().numpy()
            prob_array = [prob[index] for index in range(prob.shape[0])]
            result_array = [result[index] for index in range(result.shape[0])]
            all_probs.extend(prob_array)
            all_results.extend(result_array)


    greater_than_half = [x for x in all_probs if x > 0.5]
    count_greater_than_half = len(greater_than_half)
    total_count = len(all_probs)
    proportion = count_greater_than_half / total_count
    
    prob_mean = sum(all_probs)/len(all_probs)
    print("prob_mean", prob_mean)
    print("proportion", proportion) 
    
    greater_than_0 = [x for x in all_results if x > 0]
    results = len(greater_than_0)/len(all_results)
    print("results", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument("--micro_batch_size", type=int, default=16)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_probs", type=str, default="1.0")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=1234)
    
    # custom arguments added 
    parser.add_argument("--is_custom_dataset", action="store_true", default=False, help="Whether to use custom cyclic dataset. Default to False.")
    parser.add_argument("--is_general_preference", action="store_true", default=False, help="Whether to use General Preference model. Default to False (Bradley Terry model by default).")
    parser.add_argument("--general_preference_tau", type=float, default=0.1, help="Hyperparameter tau used in general preference loss.")
    parser.add_argument("--value_head_dim", type=int, default=2, help="Dimension of the value head in the general preference model. Ignored by the Bradley Terry model. Should be even.")


    args = parser.parse_args()
    
    batch_rm_inference(args)
    

