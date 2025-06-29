from __future__ import absolute_import
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)
from tqdm import tqdm


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, num_instances=1):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        
        print(f"[SAMPLER] Building identity mapping for {len(data_source)} samples...")
        
        # Build identity to index mapping with progress bar
        for index, (_, target) in tqdm(enumerate(data_source), 
                                      total=len(data_source), 
                                      desc="Building identity mapping",
                                      unit="samples"):
            self.index_dic[target].append(index)
        
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)
        
        print(f"[SAMPLER] Found {self.num_samples} unique identities")
        print(f"[SAMPLER] Total samples per epoch: {self.num_samples * self.num_instances}")

    def __iter__(self):
        indices = torch.randperm(self.num_samples)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_samples * self.num_instances
