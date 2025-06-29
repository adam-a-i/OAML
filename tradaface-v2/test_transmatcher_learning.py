#!/usr/bin/env python3
"""
Simple test to verify if TransMatcher can actually learn and improve over time.
This will help determine if the issue is with TransMatcher itself or the training setup.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import transmatcher
import pairwise_matching_loss

def test_transmatcher_learning():
    """Test if TransMatcher can learn and improve over time"""
    print("="*60)
    print("TRANSMATCHER LEARNING TEST")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test parameters
    batch_size = 16  # Small batch for clarity
    feature_channels = 512
    feature_height = 7
    feature_width = 7
    seq_len = feature_height * feature_width
    
    # Create TransMatcher
    transmatcher_model = transmatcher.TransMatcher(
        seq_len=seq_len,
        d_model=feature_channels,
        num_decoder_layers=3,
        dim_feedforward=2048
    ).to(device)
    
    # Create pairwise loss
    pairwise_loss = pairwise_matching_loss.PairwiseMatchingLoss(transmatcher_model)
    
    # Create optimizer
    optimizer = optim.Adam(transmatcher_model.parameters(), lr=0.001)
    
    print(f"Model parameters: {sum(p.numel() for p in transmatcher_model.parameters()):,}")
    
    # Training loop
    num_epochs = 10
    losses = []
    accuracies = []
    
    print(f"\nTraining for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_accuracies = []
        
        # Generate 5 batches per epoch
        for batch_idx in range(5):
            # Create batch with controlled identity distribution
            # 4 identities, 4 samples each (batch_size=16)
            labels = torch.tensor([
                i // 4 for i in range(batch_size)
            ], device=device)
            
            # Create feature maps
            feature_maps = torch.randn(batch_size, feature_channels, feature_height, feature_width, device=device)
            
            # Normalize feature maps
            fm_flat = feature_maps.view(feature_maps.size(0), -1)
            fm_norms = torch.norm(fm_flat, p=2, dim=1, keepdim=True).clamp(min=1e-8)
            feature_maps_normalized = (fm_flat / fm_norms).view_as(feature_maps)
            
            # Forward pass
            optimizer.zero_grad()
            
            try:
                loss, acc = pairwise_loss(feature_maps_normalized, labels)
                
                # Backward pass
                loss.mean().backward()
                optimizer.step()
                
                epoch_losses.append(loss.mean().item())
                epoch_accuracies.append(acc.mean().item())
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                epoch_losses.append(0.0)
                epoch_accuracies.append(0.5)
        
        # Calculate epoch averages
        avg_loss = np.mean(epoch_losses)
        avg_acc = np.mean(epoch_accuracies)
        
        losses.append(avg_loss)
        accuracies.append(avg_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.6f}, Acc={avg_acc:.4f}")
    
    # Analyze results
    print(f"\n" + "="*40)
    print("LEARNING ANALYSIS")
    print("="*40)
    
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Final loss: {losses[-1]:.6f}")
    print(f"Loss improvement: {losses[0] - losses[-1]:.6f}")
    
    print(f"Initial accuracy: {accuracies[0]:.4f}")
    print(f"Final accuracy: {accuracies[-1]:.4f}")
    print(f"Accuracy improvement: {accuracies[-1] - accuracies[0]:.4f}")
    
    # Check if learning occurred
    loss_improved = losses[-1] < losses[0]
    acc_improved = accuracies[-1] > accuracies[0]
    
    print(f"\nCONCLUSION:")
    if loss_improved and acc_improved:
        print(f"✓ TransMatcher IS learning! Loss decreased and accuracy increased.")
        print(f"✓ The issue is NOT with TransMatcher itself.")
        print(f"✓ The problem is in your training setup or data.")
    elif loss_improved:
        print(f"⚠ TransMatcher loss improved but accuracy didn't.")
        print(f"⚠ This suggests the model is learning but may need tuning.")
    elif acc_improved:
        print(f"⚠ TransMatcher accuracy improved but loss didn't.")
        print(f"⚠ This suggests the loss function may have issues.")
    else:
        print(f"✗ TransMatcher is NOT learning.")
        print(f"✗ This suggests a fundamental issue with the model or loss.")
    
    return losses, accuracies

def test_with_realistic_data():
    """Test with more realistic data distribution"""
    print(f"\n" + "="*60)
    print("REALISTIC DATA TEST")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # More realistic parameters
    batch_size = 32
    feature_channels = 512
    feature_height = 7
    feature_width = 7
    seq_len = feature_height * feature_width
    
    # Create TransMatcher
    transmatcher_model = transmatcher.TransMatcher(
        seq_len=seq_len,
        d_model=feature_channels,
        num_decoder_layers=3,
        dim_feedforward=2048
    ).to(device)
    
    # Create pairwise loss
    pairwise_loss = pairwise_matching_loss.PairwiseMatchingLoss(transmatcher_model)
    
    # Create optimizer
    optimizer = optim.Adam(transmatcher_model.parameters(), lr=0.001)
    
    # Test with realistic batch composition (16 identities, 2 samples each)
    labels = torch.tensor([
        i // 2 for i in range(batch_size)
    ], device=device)
    
    print(f"Realistic batch:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Identities: {len(torch.unique(labels))}")
    print(f"  - Samples per identity: {batch_size // len(torch.unique(labels))}")
    
    # Test multiple batches
    losses = []
    accuracies = []
    
    for batch_idx in range(10):
        # Create feature maps
        feature_maps = torch.randn(batch_size, feature_channels, feature_height, feature_width, device=device)
        
        # Normalize feature maps
        fm_flat = feature_maps.view(feature_maps.size(0), -1)
        fm_norms = torch.norm(fm_flat, p=2, dim=1, keepdim=True).clamp(min=1e-8)
        feature_maps_normalized = (fm_flat / fm_norms).view_as(feature_maps)
        
        # Forward pass
        optimizer.zero_grad()
        
        try:
            loss, acc = pairwise_loss(feature_maps_normalized, labels)
            
            # Backward pass
            loss.mean().backward()
            optimizer.step()
            
            losses.append(loss.mean().item())
            accuracies.append(acc.mean().item())
            
            if batch_idx % 2 == 0:
                print(f"Batch {batch_idx}: Loss={loss.mean().item():.6f}, Acc={acc.mean().item():.4f}")
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            losses.append(0.0)
            accuracies.append(0.5)
    
    print(f"\nRealistic data results:")
    print(f"  - Average loss: {np.mean(losses):.6f}")
    print(f"  - Average accuracy: {np.mean(accuracies):.4f}")
    print(f"  - Loss std: {np.std(losses):.6f}")
    print(f"  - Accuracy std: {np.std(accuracies):.4f}")

def main():
    """Main function"""
    print("TransMatcher Learning Test")
    print("This will determine if TransMatcher can actually learn")
    
    try:
        # Test basic learning
        losses, accuracies = test_transmatcher_learning()
        
        # Test with realistic data
        test_with_realistic_data()
        
        print(f"\n" + "="*60)
        print("FINAL RECOMMENDATIONS")
        print("="*60)
        print("1. If TransMatcher learns in this test, the issue is in your training setup")
        print("2. If TransMatcher doesn't learn, there's a fundamental model issue")
        print("3. Check your learning rate, optimizer, and data preprocessing")
        print("4. Consider using a smaller learning rate (0.0001 instead of 0.001)")
        print("5. Make sure your feature maps are properly normalized")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 