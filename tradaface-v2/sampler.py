from __future__ import absolute_import
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler, BatchSampler)
from tqdm import tqdm


class PKBatchSampler(Sampler):
    """
    PK Batch Sampler: Sample N identities, then for each identity sample K instances.
    This ensures each batch has exactly N*K samples with proper PK structure.
    """
    _instances = {}  # Singleton pattern to prevent multiple instances
    
    def __init__(self, data_source, n_identities=16, k_instances=4, **kwargs):
        # Handle DistributedSamplerWrapper - extract the underlying dataset
        actual_dataset = data_source
        if hasattr(data_source, 'dataset'):
            # This is a DistributedSamplerWrapper, get the underlying dataset
            actual_dataset = data_source.dataset
        
        # Create a unique key for this instance using the actual dataset
        instance_key = (id(actual_dataset), n_identities, k_instances)
        
        # Check if we already have an instance for this dataset
        if instance_key in PKBatchSampler._instances:
            print(f"[PKBatchSampler] Reusing existing instance for dataset {id(actual_dataset)}")
            existing_instance = PKBatchSampler._instances[instance_key]
            # Copy all attributes from existing instance
            for attr, value in existing_instance.__dict__.items():
                setattr(self, attr, value)
            return
        
        print(f"[PKBatchSampler] Creating new instance for dataset {id(actual_dataset)}")
        
        self.data_source = data_source
        self.actual_dataset = actual_dataset  # Store the actual dataset
        self.n_identities = n_identities
        self.k_instances = k_instances
        # Ignore batch_size parameter from PyTorch Lightning, calculate our own
        self.batch_size = n_identities * k_instances  # Add this for PyTorch Lightning compatibility
        # Use drop_last from PyTorch Lightning if provided, otherwise default to True
        self.drop_last = kwargs.get('drop_last', True)  # Add this for PyTorch Lightning compatibility
        
        # Add all other PyTorch Lightning expected attributes
        self.sampler = None  # PyTorch Lightning compatibility
        self.num_workers = 0  # PyTorch Lightning compatibility
        self.pin_memory = False  # PyTorch Lightning compatibility
        self.timeout = 0  # PyTorch Lightning compatibility
        self.worker_init_fn = None  # PyTorch Lightning compatibility
        self.multiprocessing_context = None  # PyTorch Lightning compatibility
        self.generator = None  # PyTorch Lightning compatibility
        self.prefetch_factor = 2  # PyTorch Lightning compatibility
        self.persistent_workers = False  # PyTorch Lightning compatibility
        
        self.index_dic = defaultdict(list)
        
        print(f"[PKBatchSampler] Building identity mapping for {len(data_source)} samples...")
        
        # Build identity to index mapping with robust error handling
        for index in range(len(actual_dataset)):
            try:
                item = actual_dataset[index]
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    # Standard format: (image, target)
                    _, target = item[0], item[1]
                elif hasattr(actual_dataset, 'targets') and index < len(actual_dataset.targets):
                    # Alternative format: dataset has targets attribute
                    target = actual_dataset.targets[index]
                else:
                    # Fallback: try to get target directly
                    target = item if isinstance(item, int) else getattr(item, 'target', index)
                
                self.index_dic[target].append(index)
            except Exception as e:
                print(f"[PKBatchSampler] Warning: Could not process sample {index}: {e}")
                # Skip this sample if we can't process it
                continue
        
        self.pids = list(self.index_dic.keys())
        self.num_identities_total = len(self.pids)
        
        print(f"[PKBatchSampler] Found {self.num_identities_total} total identities")
        print(f"[PKBatchSampler] Will sample {self.n_identities} identities per batch, {self.k_instances} instances each")
        print(f"[PKBatchSampler] Batch size: {self.batch_size}")
        
        # Store this instance for reuse
        PKBatchSampler._instances[instance_key] = self

    def __iter__(self):
        # Shuffle identities
        identity_indices = torch.randperm(self.num_identities_total)
        
        for i in range(0, self.num_identities_total, self.n_identities):
            batch_identities = identity_indices[i:i+self.n_identities]
            batch_indices = []
            
            for identity_idx in batch_identities:
                pid = self.pids[identity_idx]
                t = self.index_dic[pid]
                
                if len(t) >= self.k_instances:
                    t = np.random.choice(t, size=self.k_instances, replace=False)
                else:
                    t = np.random.choice(t, size=self.k_instances, replace=True)
                
                batch_indices.extend(t)
            
            yield batch_indices

    def __len__(self):
        return self.num_identities_total // self.n_identities


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
        
        print(f"[SAMPLER] Starting iteration with {self.num_samples} identities, {self.num_instances} instances each")
        
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
            
            # Debug: show first few batches
            if len(ret) <= 64:  # Only show first batch
                print(f"[SAMPLER] Identity {pid}: selected {len(t)} samples")
        
        print(f"[SAMPLER] Total samples generated: {len(ret)}")
        return iter(ret)

    def __len__(self):
        return self.num_samples * self.num_instances
