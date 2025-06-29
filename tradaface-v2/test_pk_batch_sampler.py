#!/usr/bin/env python3
"""
Test script for the new PKBatchSampler
"""

import torch
import numpy as np
from sampler import PKBatchSampler

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

def test_pk_batch_sampler():
    print("="*60)
    print("TESTING PK BATCH SAMPLER")
    print("="*60)
    
    # Create dummy dataset
    dataset = DummyDataset(num_identities=100, samples_per_identity=10)
    print(f"Dataset: {len(dataset)} samples, {len(set(dataset.targets))} identities")
    
    # Create PK batch sampler
    n_identities = 16
    k_instances = 4
    batch_sampler = PKBatchSampler(dataset, n_identities=n_identities, k_instances=k_instances)
    
    # Check if batch_size attribute exists (for PyTorch Lightning compatibility)
    print(f"Batch sampler batch_size attribute: {batch_sampler.batch_size}")
    print(f"Expected batch_size: {n_identities * k_instances}")
    
    if hasattr(batch_sampler, 'batch_size'):
        print("✅ batch_size attribute exists (PyTorch Lightning compatible)")
    else:
        print("❌ batch_size attribute missing (PyTorch Lightning incompatible)")
    
    # Check if drop_last attribute exists (for PyTorch Lightning compatibility)
    if hasattr(batch_sampler, 'drop_last'):
        print(f"✅ drop_last attribute exists: {batch_sampler.drop_last}")
    else:
        print("❌ drop_last attribute missing (PyTorch Lightning incompatible)")
    
    # Get first batch
    first_batch_indices = next(iter(batch_sampler))
    
    print(f"\nFirst batch indices (first 20): {first_batch_indices[:20]}...")
    print(f"First batch size: {len(first_batch_indices)}")
    
    # Check what identities we got
    batch_identities = [dataset.targets[idx] for idx in first_batch_indices]
    unique_identities = list(set(batch_identities))
    
    print(f"Unique identities in batch: {unique_identities}")
    print(f"Number of unique identities: {len(unique_identities)}")
    
    # Check samples per identity
    identity_counts = {}
    for identity in batch_identities:
        identity_counts[identity] = identity_counts.get(identity, 0) + 1
    
    print(f"Samples per identity: {identity_counts}")
    
    # Verify PK sampling
    expected_samples_per_identity = k_instances
    if all(count == expected_samples_per_identity for count in identity_counts.values()):
        print(f"✅ PK batch sampling working correctly: {expected_samples_per_identity} samples per identity")
    else:
        print(f"❌ PK batch sampling issue: expected {expected_samples_per_identity} samples per identity")
        print(f"   Actual: {identity_counts}")
    
    # Check total samples
    expected_total = n_identities * k_instances
    if len(first_batch_indices) == expected_total:
        print(f"✅ Batch size correct: {len(first_batch_indices)} == {expected_total}")
    else:
        print(f"❌ Batch size incorrect: {len(first_batch_indices)} != {expected_total}")
    
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)

if __name__ == "__main__":
    test_pk_batch_sampler() 