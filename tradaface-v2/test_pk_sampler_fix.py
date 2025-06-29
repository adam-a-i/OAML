#!/usr/bin/env python3
"""
Test script to verify PK sampler fix
"""

import torch
import numpy as np
from data import DataModule
from config import get_args
import sys
import os

def test_pk_sampler():
    print("="*60)
    print("TESTING PK SAMPLER FIX")
    print("="*60)
    
    # Create args with proper PK sampling
    sys.argv = [
        'test_pk_sampler_fix.py',
        '--batch_size', '64',  # This will be overridden by PK sampler
        '--num_instances', '4',  # K=4 samples per identity
        '--pk_sampler',  # Enable PK sampling
        '--data_root', 'path/to/your/data',  # You'll need to set this
        '--train_data_path', 'faces_emore/imgs',
        '--val_data_path', 'faces_emore',
        '--prefix', 'pk_test'  # This will create the output directory
    ]
    
    args = get_args()
    
    # Override data root if needed
    if not os.path.exists(args.data_root):
        print(f"WARNING: Data root {args.data_root} doesn't exist. Using dummy data for testing.")
        # For testing without real data, we'll just check the configuration
        print(f"PK Sampler enabled: {args.pk_sampler}")
        print(f"Num instances (K): {args.num_instances}")
        print(f"Batch size: {args.batch_size}")
        print("Expected PK configuration: N=16, K=4, total_batch_size=64")
        return
    
    print(f"Creating DataModule with args:")
    print(f"  - pk_sampler: {args.pk_sampler}")
    print(f"  - num_instances: {args.num_instances}")
    print(f"  - batch_size: {args.batch_size}")
    
    # Create data module
    data_module = DataModule(**vars(args))
    data_module.setup(stage='fit')
    
    # Get train dataloader
    train_loader = data_module.train_dataloader()
    
    print(f"\nTrain loader batch size: {train_loader.batch_size}")
    print(f"Expected: 16 identities × 4 instances = 64 samples")
    
    # Test a few batches
    print("\nTesting batches...")
    for batch_idx, (images, labels) in enumerate(train_loader):
        if batch_idx >= 3:  # Test first 3 batches
            break
            
        print(f"\nBatch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        
        # Check unique identities
        unique_labels = torch.unique(labels)
        print(f"  Unique identities: {len(unique_labels)}")
        
        # Check samples per identity
        label_counts = {}
        for label in labels:
            label = label.item()
            label_counts[label] = label_counts.get(label, 0) + 1
        
        samples_per_identity = list(label_counts.values())
        print(f"  Samples per identity: {samples_per_identity[:10]}...")  # Show first 10
        
        # Verify PK sampling
        expected_samples_per_identity = args.num_instances
        if all(count == expected_samples_per_identity for count in samples_per_identity):
            print(f"  ✅ PK sampling working correctly: {expected_samples_per_identity} samples per identity")
        else:
            print(f"  ❌ PK sampling issue: expected {expected_samples_per_identity} samples per identity")
            print(f"     Actual: {samples_per_identity}")
        
        # Calculate positive/negative pair ratio
        total_pairs = len(labels) * (len(labels) - 1) // 2
        positive_pairs = 0
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if labels[i] == labels[j]:
                    positive_pairs += 1
        
        negative_pairs = total_pairs - positive_pairs
        print(f"  Positive pairs: {positive_pairs}")
        print(f"  Negative pairs: {negative_pairs}")
        print(f"  Positive/Negative ratio: {positive_pairs/negative_pairs:.3f}")
        
        # Expected: 16 identities × 4 samples = 64 samples
        # Positive pairs: 16 × (4×3)/2 = 96 positive pairs
        # Negative pairs: (64×63)/2 - 96 = 1920 negative pairs
        # Ratio: 96/1920 = 0.05
        expected_positive = len(unique_labels) * (args.num_instances * (args.num_instances - 1)) // 2
        expected_negative = total_pairs - expected_positive
        print(f"  Expected positive: {expected_positive}, negative: {expected_negative}")
    
    print("\n" + "="*60)
    print("PK SAMPLER TEST COMPLETED")
    print("="*60)

if __name__ == "__main__":
    test_pk_sampler() 