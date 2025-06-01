from __future__ import absolute_import
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, label) in enumerate(data_source):
            if index % 1000 == 0: # Print progress every 1000 samples
                print(f"DEBUG Sampler Init: Processing sample {index}")
            self.index_dic[label].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)
        print(f"DEBUG Sampler: Initialized with {len(self.data_source)} total samples, {self.num_identities} identities, {self.num_instances} instances per identity.")

    def __iter__(self):
        print("DEBUG Sampler: Starting __iter__")
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        print(f"DEBUG Sampler: __iter__ yielding {len(ret)} indices")
        return iter(ret)

    def __len__(self):
        length = self.num_identities * self.num_instances
        print(f"DEBUG Sampler: __len__ returning {length}")
        return length
