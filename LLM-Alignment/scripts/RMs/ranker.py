"""
    RewardModel and Ranker class for ranking candidates based on pairwise comparisons.
"""
from RMs.reward_model import RewardModel
from typing import List
import numpy as np
import torch

# tool copied from blender
def get_scores_from_cmps(cmp_results, policy="max_logits"):
    """
    Args:
        cmp_results: ndarray of shape (n, c, c) where n is the number of samples, c is the number of candidates
            for each element, >0 means the first candidate is better than the second one, <0 means the second one is better
    Returns:
        scores: ndarray of shape (n, c) where n is the number of samples, c is the number of candidates
    """
    if isinstance(cmp_results, list):
        cmp_results = np.array(cmp_results)
    bz, c, _ = cmp_results.shape
    scores = np.zeros((bz, c), dtype=np.float32)
    for i in range(bz):
        if policy == "max_logits":
            scores[i] = (cmp_results[i] - cmp_results[i].T).mean(axis=-1)
        elif policy == "max_wins":
            scores[i] = (cmp_results[i] > 0).sum(axis=-1) + (cmp_results[i] < 0).mean(axis=-2)
    return scores


from tqdm import tqdm
class Ranker:
    def __init__(self,
        model: RewardModel
    ):
        self.model = model
    
    def rank(self, 
        prompt: List[str], 
        candidates: List[List[str]],
        batch_size: int = 1,
        return_raw_table: bool = False
    ) -> np.ndarray:
        """
            Rank candidates based on prompt.
            prompt: (n, )
            candidates: (n, num_candidates)
        """
        # 1. to batches
        scores = []
        raw_table = []
        with torch.no_grad():
            for i in tqdm(range(0, len(prompt), batch_size), total=len(prompt)//batch_size):
                batch = prompt[i:i+batch_size]
                batch_candidates = candidates[i:i+batch_size]
                pair_wise_scores = self.model.pair_wise_scores(batch, batch_candidates)
                if return_raw_table:
                    raw_table.append(pair_wise_scores.cpu().numpy())

                scores.append(get_scores_from_cmps(pair_wise_scores.cpu().numpy()))

        if return_raw_table:
            return np.concatenate(scores, axis=0), np.concatenate(raw_table, axis=0)
        else:
            return np.concatenate(scores, axis=0)

                