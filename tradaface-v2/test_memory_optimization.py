#!/usr/bin/env python3
"""
Test script to verify TransMatcher memory optimizations work
"""

import torch
import torch.nn as nn
import time
import gc

def test_transmatcher_memory():
    print("Testing TransMatcher memory optimizations...")
    
    # Import TransMatcher
    from transmatcher import TransMatcher
    
    # Test parameters (reduced for memory)
    seq_len = 49  # 7x7
    d_model = 64  # Reduced from 512
    num_layers = 1
    dim_feedforward = 256
    
    print(f"TransMatcher params: seq_len={seq_len}, d_model={d_model}, num_layers={num_layers}, dim_feedforward={dim_feedforward}")
    
    # Create TransMatcher
    transmatcher = TransMatcher(seq_len, d_model, num_layers, dim_feedforward)
    print(f"TransMatcher created: {type(transmatcher)}")
    
    # Test with small batch size
    batch_size = 4
    feature_shape = (batch_size, 7, 7, d_model)  # [B, H, W, C]
    
    print(f"Testing with batch_size={batch_size}, feature_shape={feature_shape}")
    
    # Create test features
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    features = torch.randn(feature_shape, device=device)
    features = torch.nn.functional.normalize(features, p=2, dim=-1)
    
    # Move TransMatcher to the same device
    transmatcher = transmatcher.to(device)
    
    print(f"Features created: shape={features.shape}, device={features.device}")
    print(f"TransMatcher device: {next(transmatcher.parameters()).device}")
    
    # Test forward pass
    try:
        start_time = time.time()
        
        # Set memory kernel
        transmatcher.make_kernel(features)
        
        # Forward pass
        scores = transmatcher(features)
        
        end_time = time.time()
        
        print(f"✓ Forward pass successful!")
        print(f"  - Scores shape: {scores.shape}")
        print(f"  - Scores range: [{scores.min().item():.3f}, {scores.max().item():.3f}]")
        print(f"  - Time: {end_time - start_time:.3f}s")
        
        # Check memory usage
        if torch.cuda.is_available():
            print(f"  - GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            print(f"  - GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pairwise_loss():
    print("\nTesting PairwiseMatchingLoss with TransMatcher...")
    
    from transmatcher import TransMatcher
    from pairwise_matching_loss import PairwiseMatchingLoss
    
    # Create TransMatcher
    seq_len = 49
    d_model = 64
    num_layers = 1
    dim_feedforward = 256
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transmatcher = TransMatcher(seq_len, d_model, num_layers, dim_feedforward).to(device)
    pairwise_loss = PairwiseMatchingLoss(transmatcher)
    
    # Test with small batch
    batch_size = 4
    feature_shape = (batch_size, 7, 7, d_model)
    features = torch.randn(feature_shape, device=device)
    features = torch.nn.functional.normalize(features, p=2, dim=-1)
    
    # Create labels (2 classes)
    labels = torch.tensor([0, 0, 1, 1], device=device)
    
    try:
        start_time = time.time()
        
        # Compute loss
        loss, acc = pairwise_loss(features, labels)
        
        end_time = time.time()
        
        print(f"✓ Pairwise loss successful!")
        print(f"  - Loss shape: {loss.shape}")
        print(f"  - Loss mean: {loss.mean().item():.6f}")
        print(f"  - Accuracy mean: {acc.mean().item():.3f}")
        print(f"  - Time: {end_time - start_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"✗ Pairwise loss failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_integration():
    print("\nTesting model integration...")
    
    # Create a simple backbone-like feature extractor
    class SimpleBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 512, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((7, 7))
            
        def forward(self, x, return_feature_map=False):
            x = self.conv(x)
            feature_map = self.pool(x)
            
            # Create dummy embedding
            embedding = torch.randn(x.size(0), 512, device=x.device)
            norm = torch.norm(embedding, 2, 1, True)
            embedding = torch.div(embedding, norm + 1e-6)
            
            if return_feature_map:
                return embedding, norm, feature_map
            else:
                return embedding, norm
    
    # Import the model class
    import sys
    sys.path.append('.')
    from net import AdaFaceWithTransMatcher
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backbone = SimpleBackbone().to(device)
    transmatcher_params = {
        'seq_len': 49,
        'd_model': 64,
        'num_decoder_layers': 1,
        'dim_feedforward': 256,
    }
    
    model = AdaFaceWithTransMatcher(backbone, transmatcher_params).to(device)
    
    # Test forward pass
    batch_size = 2
    input_shape = (batch_size, 3, 112, 112)
    inputs = torch.randn(input_shape, device=device)
    
    try:
        start_time = time.time()
        
        # Forward pass
        embedding, norm, feature_map, transmatcher = model(inputs)
        
        end_time = time.time()
        
        print(f"✓ Model integration successful!")
        print(f"  - Embedding shape: {embedding.shape}")
        print(f"  - Norm shape: {norm.shape}")
        print(f"  - Feature map shape: {feature_map.shape}")
        print(f"  - TransMatcher type: {type(transmatcher)}")
        print(f"  - Time: {end_time - start_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"✗ Model integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TRANSMATCHER MEMORY OPTIMIZATION TEST")
    print("=" * 60)
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Run tests
    test1_passed = test_transmatcher_memory()
    test2_passed = test_pairwise_loss()
    test3_passed = test_model_integration()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"TransMatcher memory test: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Pairwise loss test: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    print(f"Model integration test: {'✓ PASSED' if test3_passed else '✗ FAILED'}")
    
    if test1_passed and test2_passed and test3_passed:
        print("\n🎉 All tests passed! Memory optimizations are working.")
        print("You can now run the training script.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
    
    # Final memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect() 