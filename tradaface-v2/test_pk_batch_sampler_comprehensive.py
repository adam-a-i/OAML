#!/usr/bin/env python3
"""
Comprehensive test for PKBatchSampler PyTorch Lightning compatibility
"""

import torch
import numpy as np
from sampler import PKBatchSampler
import inspect

class DummyDataset:
    def __init__(self, num_identities=100, samples_per_identity=10):
        self.samples = []
        self.targets = []
        
        for identity in range(num_identities):
            for sample in range(samples_per_identity):
                self.samples.append(f"sample_{identity}_{sample}")
                self.targets.append(identity)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]

def test_pytorch_lightning_compatibility():
    print("="*80)
    print("COMPREHENSIVE PYTORCH LIGHTNING COMPATIBILITY TEST")
    print("="*80)
    
    # Create dummy dataset
    dataset = DummyDataset(num_identities=100, samples_per_identity=10)
    
    # Create PK batch sampler
    n_identities = 16
    k_instances = 4
    batch_sampler = PKBatchSampler(dataset, n_identities=n_identities, k_instances=k_instances)
    
    print(f"PKBatchSampler type: {type(batch_sampler)}")
    print(f"PKBatchSampler attributes: {dir(batch_sampler)}")
    
    # Check all known PyTorch Lightning expected attributes
    expected_attributes = [
        'batch_size',
        'drop_last',
        'sampler',  # Sometimes expected
        'num_workers',  # Sometimes expected
        'pin_memory',  # Sometimes expected
        'timeout',  # Sometimes expected
        'worker_init_fn',  # Sometimes expected
        'multiprocessing_context',  # Sometimes expected
        'generator',  # Sometimes expected
        'prefetch_factor',  # Sometimes expected
        'persistent_workers',  # Sometimes expected
    ]
    
    print("\nChecking PyTorch Lightning compatibility attributes:")
    missing_attributes = []
    
    for attr in expected_attributes:
        if hasattr(batch_sampler, attr):
            value = getattr(batch_sampler, attr)
            print(f"✅ {attr}: {value}")
        else:
            print(f"❌ {attr}: MISSING")
            missing_attributes.append(attr)
    
    # Check if it's iterable (required for any sampler)
    try:
        iter(batch_sampler)
        print("✅ __iter__: Works")
    except Exception as e:
        print(f"❌ __iter__: FAILED - {e}")
        missing_attributes.append('__iter__')
    
    # Check if it has __len__ (required for any sampler)
    try:
        length = len(batch_sampler)
        print(f"✅ __len__: {length}")
    except Exception as e:
        print(f"❌ __len__: FAILED - {e}")
        missing_attributes.append('__len__')
    
    # Test actual sampling
    try:
        first_batch = next(iter(batch_sampler))
        print(f"✅ Sampling works: {len(first_batch)} samples")
        
        # Check PK structure
        batch_identities = [dataset.targets[idx] for idx in first_batch]
        unique_identities = list(set(batch_identities))
        
        if len(unique_identities) == n_identities:
            print(f"✅ PK structure correct: {len(unique_identities)} identities")
        else:
            print(f"❌ PK structure wrong: {len(unique_identities)} identities (expected {n_identities})")
            missing_attributes.append('pk_structure')
            
    except Exception as e:
        print(f"❌ Sampling failed: {e}")
        missing_attributes.append('sampling')
    
    print(f"\n{'='*80}")
    if missing_attributes:
        print(f"❌ MISSING ATTRIBUTES: {missing_attributes}")
        print("PyTorch Lightning compatibility issues found!")
    else:
        print("✅ ALL ATTRIBUTES PRESENT - PyTorch Lightning compatible!")
    print(f"{'='*80}")

if __name__ == "__main__":
    test_pytorch_lightning_compatibility() 