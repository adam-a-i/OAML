#!/usr/bin/env python3
"""
Test script to debug the RandomIdentitySampler
"""

import torch
import numpy as np
from sampler import RandomIdentitySampler

class DummyDataset:
    def __init__(self, num_identities=20, samples_per_identity=10):
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

def test_sampler():
    print("="*60)
    print("TESTING RANDOM IDENTITY SAMPLER")
    print("="*60)
    
    # Create dummy dataset
    dataset = DummyDataset(num_identities=20, samples_per_identity=10)
    print(f"Dataset: {len(dataset)} samples, {len(set(dataset.targets))} identities")
    
    # Create sampler
    num_instances = 4
    sampler = RandomIdentitySampler(dataset, num_instances=num_instances)
    
    # Get first batch
    batch_indices = list(sampler)[:16]  # First 16 identities * 4 samples = 64 samples
    
    print(f"\nFirst batch indices: {batch_indices}")
    
    # Check what identities we got
    batch_identities = [dataset.targets[idx] for idx in batch_indices]
    unique_identities = list(set(batch_identities))
    
    print(f"Unique identities in batch: {unique_identities}")
    print(f"Number of unique identities: {len(unique_identities)}")
    
    # Check samples per identity
    identity_counts = {}
    for identity in batch_identities:
        identity_counts[identity] = identity_counts.get(identity, 0) + 1
    
    print(f"Samples per identity: {identity_counts}")
    
    # Verify PK sampling
    expected_samples_per_identity = num_instances
    if all(count == expected_samples_per_identity for count in identity_counts.values()):
        print(f"✅ PK sampling working correctly: {expected_samples_per_identity} samples per identity")
    else:
        print(f"❌ PK sampling issue: expected {expected_samples_per_identity} samples per identity")
        print(f"   Actual: {identity_counts}")
    
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)

if __name__ == "__main__":
    test_sampler() 