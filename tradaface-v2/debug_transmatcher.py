"""
Comprehensive TransMatcher Debugging Script
This script systematically tests each component of the TransMatcher integration
to identify why the loss is always zero and accuracy is stuck at 0.5.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transmatcher import TransMatcher
from pairwise_matching_loss import PairwiseMatchingLoss
import traceback

def debug_transmatcher_integration():
    """Main debugging function"""
    print("=" * 80)
    print("TRANSMATCHER DEBUGGING SESSION")
    print("=" * 80)
    
    # Test 1: Basic TransMatcher functionality
    print("\n1. TESTING BASIC TRANSMATCHER FUNCTIONALITY")
    test_basic_transmatcher()
    
    # Test 2: Feature map generation and normalization
    print("\n2. TESTING FEATURE MAP GENERATION")
    test_feature_maps()
    
    # Test 3: Pairwise loss computation
    print("\n3. TESTING PAIRWISE LOSS COMPUTATION")
    test_pairwise_loss()
    
    # Test 4: PK sampling simulation
    print("\n4. TESTING PK SAMPLING")
    test_pk_sampling()
    
    # Test 5: Full pipeline test
    print("\n5. TESTING FULL PIPELINE")
    test_full_pipeline()
    
    print("\n" + "=" * 80)
    print("DEBUGGING COMPLETE")
    print("=" * 80)

def test_basic_transmatcher():
    """Test basic TransMatcher forward pass"""
    print("  Testing TransMatcher initialization and forward pass...")
    
    try:
        # Initialize TransMatcher
        seq_len = 24 * 8  # 192 (typical feature map size)
        d_model = 512
        num_layers = 3
        transmatcher = TransMatcher(seq_len, d_model, num_layers)
        
        # Create dummy feature maps
        batch_size = 8
        height, width = 24, 8
        features = torch.randn(batch_size, height, width, d_model * num_layers)
        
        print(f"    TransMatcher initialized: {type(transmatcher)}")
        print(f"    Input features shape: {features.shape}")
        print(f"    Features range: [{features.min().item():.3f}, {features.max().item():.3f}]")
        
        # Test make_kernel and forward pass
        transmatcher.make_kernel(features)
        scores = transmatcher(features)
        
        print(f"    Output scores shape: {scores.shape}")
        print(f"    Scores range: [{scores.min().item():.3f}, {scores.max().item():.3f}]")
        print(f"    Scores mean: {scores.mean().item():.3f}")
        print(f"    Scores std: {scores.std().item():.3f}")
        
        # Check for NaN or Inf
        if torch.isnan(scores).any():
            print("    ERROR: Scores contain NaN values!")
        if torch.isinf(scores).any():
            print("    ERROR: Scores contain Inf values!")
        
        print("    ✓ Basic TransMatcher test PASSED")
        return True
        
    except Exception as e:
        print(f"    ✗ Basic TransMatcher test FAILED: {e}")
        print(f"    {traceback.format_exc()}")
        return False

def test_feature_maps():
    """Test feature map generation and normalization"""
    print("  Testing feature map generation and normalization...")
    
    try:
        # Simulate backbone feature maps
        batch_size = 8
        channels = 512
        height, width = 24, 8
        
        # Generate feature maps (simulating backbone output)
        feature_maps = torch.randn(batch_size, channels, height, width)
        
        print(f"    Raw feature maps shape: {feature_maps.shape}")
        print(f"    Raw feature maps range: [{feature_maps.min().item():.3f}, {feature_maps.max().item():.3f}]")
        
        # Normalize feature maps (L2 normalization)
        feature_maps_flat = feature_maps.view(feature_maps.size(0), -1)
        norms = torch.norm(feature_maps_flat, p=2, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-8)
        feature_maps_flat_norm = feature_maps_flat / norms
        feature_maps_norm = feature_maps_flat_norm.view_as(feature_maps)
        
        print(f"    Normalized feature maps range: [{feature_maps_norm.min().item():.3f}, {feature_maps_norm.max().item():.3f}]")
        
        # Verify normalization
        norms_check = torch.norm(feature_maps_norm.view(feature_maps_norm.size(0), -1), p=2, dim=1)
        print(f"    Norms after normalization: min={norms_check.min().item():.3f}, max={norms_check.max().item():.3f}")
        
        # Permute for TransMatcher (B, C, H, W) -> (B, H, W, C)
        feature_maps_perm = feature_maps_norm.permute(0, 2, 3, 1).contiguous()
        
        print(f"    Permuted feature maps shape: {feature_maps_perm.shape}")
        print(f"    Permuted feature maps range: [{feature_maps_perm.min().item():.3f}, {feature_maps_perm.max().item():.3f}]")
        
        # Test with TransMatcher
        seq_len = height * width
        d_model = channels
        num_layers = 3
        transmatcher = TransMatcher(seq_len, d_model, num_layers)
        
        # Repeat feature maps for multiple layers
        feature_maps_multi = feature_maps_perm.repeat(1, 1, 1, num_layers)
        
        transmatcher.make_kernel(feature_maps_multi)
        scores = transmatcher(feature_maps_multi)
        
        print(f"    TransMatcher scores shape: {scores.shape}")
        print(f"    TransMatcher scores range: [{scores.min().item():.3f}, {scores.max().item():.3f}]")
        
        print("    ✓ Feature map test PASSED")
        return True
        
    except Exception as e:
        print(f"    ✗ Feature map test FAILED: {e}")
        print(f"    {traceback.format_exc()}")
        return False

def test_pairwise_loss():
    """Test pairwise loss computation with TransMatcher"""
    print("  Testing pairwise loss computation...")
    
    try:
        # Create TransMatcher and loss function
        seq_len = 24 * 8
        d_model = 512
        num_layers = 3
        transmatcher = TransMatcher(seq_len, d_model, num_layers)
        pairwise_loss = PairwiseMatchingLoss(transmatcher)
        
        # Create feature maps and labels
        batch_size = 8
        height, width = 24, 8
        feature_maps = torch.randn(batch_size, height, width, d_model * num_layers)
        
        # Create PK sampling labels (4 identities, 2 samples each)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        
        print(f"    Feature maps shape: {feature_maps.shape}")
        print(f"    Labels: {labels}")
        print(f"    Unique labels: {torch.unique(labels)}")
        print(f"    Label counts: {torch.bincount(labels)}")
        
        # Test pairwise loss computation
        loss, acc = pairwise_loss(feature_maps, labels)
        
        print(f"    Loss shape: {loss.shape}")
        print(f"    Loss values: {loss}")
        print(f"    Loss mean: {loss.mean().item():.6f}")
        print(f"    Loss std: {loss.std().item():.6f}")
        print(f"    Accuracy: {acc.mean().item():.3f}")
        
        # Check for zero loss
        if loss.mean().item() == 0:
            print("    WARNING: Loss is zero! This indicates a problem.")
            
            # Debug the matcher output
            transmatcher.make_kernel(feature_maps)
            scores = transmatcher(feature_maps)
            print(f"    Matcher scores shape: {scores.shape}")
            print(f"    Matcher scores: {scores}")
            
            # Check pairwise mask
            target1 = labels.unsqueeze(1)
            mask = (target1 == target1.t())
            pair_labels = mask.float()
            print(f"    Pairwise mask sum: {pair_labels.sum().item()}")
            print(f"    Positive pairs: {(pair_labels == 1).sum().item()}")
            print(f"    Negative pairs: {(pair_labels == 0).sum().item()}")
            
            # Check binary cross entropy
            bce_loss = F.binary_cross_entropy_with_logits(scores, pair_labels, reduction='none')
            print(f"    BCE loss shape: {bce_loss.shape}")
            print(f"    BCE loss values: {bce_loss}")
            print(f"    BCE loss mean: {bce_loss.mean().item():.6f}")
        
        print("    ✓ Pairwise loss test PASSED")
        return True
        
    except Exception as e:
        print(f"    ✗ Pairwise loss test FAILED: {e}")
        print(f"    {traceback.format_exc()}")
        return False

def test_pk_sampling():
    """Test PK sampling simulation"""
    print("  Testing PK sampling simulation...")
    
    try:
        # Simulate PK sampling (P=4 identities, K=2 samples each)
        P, K = 4, 2
        batch_size = P * K
        
        # Create labels
        labels = torch.arange(P).repeat(K)
        print(f"    PK sampling: P={P}, K={K}, batch_size={batch_size}")
        print(f"    Labels: {labels}")
        print(f"    Unique labels: {torch.unique(labels)}")
        print(f"    Label counts: {torch.bincount(labels)}")
        
        # Create feature maps
        height, width = 24, 8
        d_model = 512
        num_layers = 3
        feature_maps = torch.randn(batch_size, height, width, d_model * num_layers)
        
        # Test with TransMatcher
        seq_len = height * width
        transmatcher = TransMatcher(seq_len, d_model, num_layers)
        pairwise_loss = PairwiseMatchingLoss(transmatcher)
        
        loss, acc = pairwise_loss(feature_maps, labels)
        
        print(f"    Loss mean: {loss.mean().item():.6f}")
        print(f"    Accuracy: {acc.mean().item():.3f}")
        
        # Check pairwise relationships
        target1 = labels.unsqueeze(1)
        mask = (target1 == target1.t())
        pair_labels = mask.float()
        
        print(f"    Total pairs: {pair_labels.numel()}")
        print(f"    Positive pairs: {(pair_labels == 1).sum().item()}")
        print(f"    Negative pairs: {(pair_labels == 0).sum().item()}")
        print(f"    Expected positive pairs: {P * K * (K-1) // 2}")
        print(f"    Expected negative pairs: {batch_size * batch_size - P * K * (K-1) // 2}")
        
        print("    ✓ PK sampling test PASSED")
        return True
        
    except Exception as e:
        print(f"    ✗ PK sampling test FAILED: {e}")
        print(f"    {traceback.format_exc()}")
        return False

def test_full_pipeline():
    """Test the complete pipeline"""
    print("  Testing complete pipeline...")
    
    try:
        # Setup
        P, K = 4, 2
        batch_size = P * K
        height, width = 24, 8
        d_model = 512
        num_layers = 3
        seq_len = height * width
        
        # Create model components
        transmatcher = TransMatcher(seq_len, d_model, num_layers)
        pairwise_loss = PairwiseMatchingLoss(transmatcher)
        
        # Create data
        labels = torch.arange(P).repeat(K)
        feature_maps = torch.randn(batch_size, height, width, d_model * num_layers)
        
        # Normalize feature maps
        feature_maps_flat = feature_maps.view(feature_maps.size(0), -1)
        norms = torch.norm(feature_maps_flat, p=2, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-8)
        feature_maps_flat_norm = feature_maps_flat / norms
        feature_maps_norm = feature_maps_flat_norm.view_as(feature_maps)
        
        print(f"    Normalized feature maps range: [{feature_maps_norm.min().item():.3f}, {feature_maps_norm.max().item():.3f}]")
        
        # Compute loss
        loss, acc = pairwise_loss(feature_maps_norm, labels)
        
        print(f"    Final loss: {loss.mean().item():.6f}")
        print(f"    Final accuracy: {acc.mean().item():.3f}")
        
        # Check gradients
        loss.mean().backward()
        
        grad_norm = 0
        for name, param in transmatcher.named_parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        
        print(f"    Gradient norm: {grad_norm:.6e}")
        
        if grad_norm == 0:
            print("    WARNING: Zero gradients! This explains why TransMatcher is not learning.")
        else:
            print("    ✓ Gradients are flowing properly.")
        
        print("    ✓ Full pipeline test PASSED")
        return True
        
    except Exception as e:
        print(f"    ✗ Full pipeline test FAILED: {e}")
        print(f"    {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    debug_transmatcher_integration() 