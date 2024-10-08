import math
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

__all__ = ["GroupDistributedSampler", ]

T_co = TypeVar('T_co', covariant=True)


class GroupDistributedSampler(DistributedSampler):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, group_size: int = 1, sample_group_num: Optional[int] = None) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.group_size = group_size    
        self.num_groups = math.ceil(len(dataset) / group_size)  
        self.sample_group_num = sample_group_num if sample_group_num and sample_group_num <= self.num_groups else self.num_groups        
        
    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            group_indices = torch.randperm(self.num_groups, generator=g).tolist()  # type: ignore[arg-type]
            indices = []
            for group_idx in group_indices:
                start_idx = group_idx * self.group_size
                end_idx = min(start_idx + self.group_size, len(self.dataset))
                indices.extend(list(range(start_idx, end_idx)))
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        if self.sample_group_num < self.num_groups:
            total_sample_size = self.sample_group_num * self.group_size
            indices = indices[self.rank:total_sample_size:self.num_replicas]
        else:
            indices = indices[self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples

        return iter(indices)
    
    def __len__(self) -> int:
        if self.sample_group_num < self.num_groups:
            return math.ceil(self.sample_group_num * self.group_size / self.num_replicas)
        else:
            return self.num_samples

