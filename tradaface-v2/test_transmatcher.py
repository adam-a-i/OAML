#!/usr/bin/env python3
"""
Test script to debug TransMatcher, PK sampler, and pairwise matching loss
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import DataModule
from net import build_model
from pairwise_matching_loss import PairwiseMatchingLoss
from sampler import RandomIdentitySampler


def test_pk_sampler():
    """Test if PK sampler is working correctly"""
    print("="*60)
    print("TESTING PK SAMPLER")
    print("="*60)
    
    # Create a simple test dataset
    class TestDataset:
        def __init__(self, num_identities=100, samples_per_identity=10):
            self.samples = []
            self.targets = []
            for identity in range(num_identities):
                for sample in range(samples_per_identity):
                    self.samples.append((f"fake_path_{identity}_{sample}", identity))
                    self.targets.append(identity)
            
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            return self.samples[idx]
    
    # Create test dataset
    test_dataset = TestDataset(num_identities=100, samples_per_identity=10)
    print(f"Test dataset: {len(test_dataset)} samples, {len(set(test_dataset.targets))} identities")
    
    # Test PK sampler
    num_instances = 4
    sampler = RandomIdentitySampler(test_dataset, num_instances=num_instances)
    
    # Get first batch indices
    batch_indices = list(sampler)[:16]  # First 16 samples (4 identities * 4 samples)
    print(f"First batch indices: {batch_indices}")
    
    # Check if we have 4 samples per identity
    batch_labels = [test_dataset.targets[idx] for idx in batch_indices]
    print(f"First batch labels: {batch_labels}")
    
    unique_labels, counts = np.unique(batch_labels, return_counts=True)
    print(f"Unique labels in first batch: {unique_labels}")
    print(f"Counts per label: {counts}")
    
    # Verify PK sampling
    if len(unique_labels) == 4 and all(count == 4 for count in counts):
        print("✅ PK sampler is working correctly!")
    else:
        print("❌ PK sampler is NOT working correctly!")
        return False
    
    return True


def test_transmatcher():
    """Test TransMatcher forward pass"""
    print("\n" + "="*60)
    print("TESTING TRANSMATCHER")
    print("="*60)
    
    # Create model
    model = build_model('ir_50')
    model.eval()
    
    # Create dummy input
    batch_size = 8
    feature_maps = torch.randn(batch_size, 512, 7, 7)  # [B, C, H, W]
    
    print(f"Input feature maps shape: {feature_maps.shape}")
    
    # Test TransMatcher
    transmatcher = model.transmatcher
    print(f"TransMatcher: {type(transmatcher)}")
    
    try:
        # Test forward pass
        with torch.no_grad():
            # Set memory
            transmatcher.make_kernel(feature_maps)
            
            # Forward pass
            scores = transmatcher(feature_maps)
            
            print(f"TransMatcher output shape: {scores.shape}")
            print(f"TransMatcher scores - min: {scores.min().item():.6f}, max: {scores.max().item():.6f}, mean: {scores.mean().item():.6f}")
            
            # Check if scores are reasonable
            if not torch.isnan(scores).any() and not torch.isinf(scores).any():
                print("✅ TransMatcher forward pass is working!")
                return True
            else:
                print("❌ TransMatcher output contains NaN or Inf!")
                return False
                
    except Exception as e:
        print(f"❌ TransMatcher forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pairwise_loss():
    """Test pairwise matching loss"""
    print("\n" + "="*60)
    print("TESTING PAIRWISE MATCHING LOSS")
    print("="*60)
    
    # Create model and get TransMatcher
    model = build_model('ir_50')
    transmatcher = model.transmatcher
    
    # Create dummy data with PK sampling (4 identities, 2 samples each)
    batch_size = 8
    feature_maps = torch.randn(batch_size, 512, 7, 7)
    
    # Create labels that simulate PK sampling: [0,0, 1,1, 2,2, 3,3]
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    
    print(f"Feature maps shape: {feature_maps.shape}")
    print(f"Labels: {labels.numpy()}")
    
    # Create pairwise loss
    pairwise_loss = PairwiseMatchingLoss(transmatcher)
    
    try:
        # Compute loss
        loss, acc = pairwise_loss(feature_maps, labels)
        
        print(f"Loss shape: {loss.shape}")
        print(f"Loss values: {loss.detach().cpu().numpy()}")
        print(f"Loss mean: {loss.mean().item():.6f}")
        
        print(f"Accuracy shape: {acc.shape}")
        print(f"Accuracy values: {acc.detach().cpu().numpy()}")
        print(f"Accuracy mean: {acc.mean().item():.6f}")
        
        # Check if loss is reasonable
        if not torch.isnan(loss).any() and not torch.isinf(loss).any() and loss.mean().item() > 0:
            print("✅ Pairwise loss is working!")
            return True
        else:
            print("❌ Pairwise loss is zero, NaN, or Inf!")
            return False
            
    except Exception as e:
        print(f"❌ Pairwise loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """Test the full pipeline with real data"""
    print("\n" + "="*60)
    print("TESTING FULL PIPELINE")
    print("="*60)
    
    # Create dummy config
    config = {
        'data_root': '/tmp',  # Dummy path
        'train_data_path': 'dummy',
        'val_data_path': 'dummy',
        'batch_size': 32,
        'num_workers': 0,
        'train_data_subset': False,
        'low_res_augmentation_prob': 0.0,
        'crop_augmentation_prob': 0.0,
        'photometric_augmentation_prob': 0.0,
        'swap_color_channel': False,
        'use_mxrecord': False,
        'pk_sampler': True,
        'num_instances': 4,
        'output_dir': '/tmp'
    }
    
    try:
        # Create data module
        data_module = DataModule(**config)
        print("✅ DataModule created successfully")
        
        # Test if we can create a dataloader (this will fail with dummy paths, but that's expected)
        print("Note: DataLoader test will fail with dummy paths - this is expected")
        
    except Exception as e:
        print(f"DataModule creation failed (expected with dummy paths): {e}")


def main():
    """Run all tests"""
    print("TRANSMATCHER DEBUG TEST SCRIPT")
    print("="*60)
    
    # Test 1: PK Sampler
    pk_ok = test_pk_sampler()
    
    # Test 2: TransMatcher
    transmatcher_ok = test_transmatcher()
    
    # Test 3: Pairwise Loss
    pairwise_ok = test_pairwise_loss()
    
    # Test 4: Full Pipeline (will fail with dummy paths)
    test_full_pipeline()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"PK Sampler: {'✅ PASS' if pk_ok else '❌ FAIL'}")
    print(f"TransMatcher: {'✅ PASS' if transmatcher_ok else '❌ FAIL'}")
    print(f"Pairwise Loss: {'✅ PASS' if pairwise_ok else '❌ FAIL'}")
    
    if pk_ok and transmatcher_ok and pairwise_ok:
        print("\n🎉 All core components are working!")
        print("The issue might be in the training loop or data loading.")
    else:
        print("\n🔧 Some components have issues that need to be fixed.")


if __name__ == "__main__":
    main() 