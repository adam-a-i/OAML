#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from transmatcher import TransMatcher

def test_transmatcher_dimensions():
    print("Testing TransMatcher with 512-channel feature maps...")
    
    # Create TransMatcher with 512 channels and 3 layers
    seq_len = 49  # 7x7
    d_model = 512
    num_layers = 3
    
    transmatcher = TransMatcher(seq_len, d_model, num_layers)
    print(f"TransMatcher created with d_model={d_model}, num_layers={num_layers}")
    
    # Create test feature maps with 512 channels
    batch_size = 4
    height, width = 7, 7
    channels = 512
    
    # Test feature maps in [B, C, H, W] format
    features = torch.randn(batch_size, channels, height, width)
    print(f"Input features shape: {features.shape}")
    
    # Test the forward pass
    try:
        # Set memory first
        transmatcher.make_kernel(features)
        print("Memory set successfully")
        
        # Forward pass
        scores = transmatcher.forward(features)
        print(f"Forward pass successful! Output shape: {scores.shape}")
        
        # Test match_pairs method
        probe_features = torch.randn(batch_size, channels, height, width)
        gallery_features = torch.randn(batch_size, channels, height, width)
        
        pair_scores = transmatcher.match_pairs(probe_features, gallery_features)
        print(f"match_pairs successful! Output shape: {pair_scores.shape}")
        
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_transmatcher_dimensions() 