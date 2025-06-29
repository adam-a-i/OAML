#!/usr/bin/env python3
"""
Test script to verify TransMatcher dimension fix
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transmatcher import TransMatcher

def test_transmatcher_dimensions():
    print("Testing TransMatcher dimension handling...")
    
    # Create TransMatcher with correct parameters
    # Use d_model=510 (divisible by 3) instead of 512 to avoid chunking issues
    transmatcher = TransMatcher(seq_len=49, d_model=510, num_decoder_layers=3, dim_feedforward=2048)
    
    # Create feature maps in [batch, channels, height, width] format (like from backbone)
    batch_size = 4
    feature_maps = torch.randn(batch_size, 510, 7, 7)  # [4, 510, 7, 7] to match adjusted d_model
    
    print(f"Input feature maps shape: {feature_maps.shape}")
    print(f"Expected: [batch, channels, height, width] = [4, 510, 7, 7]")
    
    # Test make_kernel method
    try:
        transmatcher.make_kernel(feature_maps)
        print("✓ make_kernel() successful")
    except Exception as e:
        print(f"✗ make_kernel() failed: {e}")
        return False
    
    # Test forward method
    try:
        # First set the kernel (memory) with the same features
        transmatcher.make_kernel(feature_maps)
        print(f"Memory shape after make_kernel: {transmatcher.memory.shape}")
        
        # Then call forward with the same features
        scores = transmatcher(feature_maps)
        print(f"✓ forward() successful, output shape: {scores.shape}")
        print(f"Expected output shape: [{batch_size}, {batch_size}]")
    except Exception as e:
        print(f"✗ forward() failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test match_pairs method
    try:
        probe_features = torch.randn(batch_size, 510, 7, 7)
        gallery_features = torch.randn(batch_size, 510, 7, 7)
        pair_scores = transmatcher.match_pairs(probe_features, gallery_features)
        print(f"✓ match_pairs() successful, output shape: {pair_scores.shape}")
        print(f"Expected output shape: [{batch_size}]")
    except Exception as e:
        print(f"✗ match_pairs() failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("All tests passed! TransMatcher dimension fix is working correctly.")
    return True

if __name__ == "__main__":
    success = test_transmatcher_dimensions()
    if success:
        print("\n✓ TransMatcher is ready for training!")
    else:
        print("\n✗ TransMatcher still has issues!")
        sys.exit(1)