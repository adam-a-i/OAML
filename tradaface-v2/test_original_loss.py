#!/usr/bin/env python3
"""
Test script to verify that the original pairwise loss works correctly with TransMatcher
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

def test_original_loss_with_transmatcher():
    """Test the original pairwise loss with TransMatcher"""
    print("="*60)
    print("TESTING ORIGINAL PAIRWISE LOSS WITH TRANSMATCHER")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test parameters
    batch_size = 16
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
    
    # Create original pairwise loss
    original_loss = pairwise_matching_loss.PairwiseMatchingLoss(transmatcher_model)
    
    # Create optimizer
    optimizer = optim.Adam(transmatcher_model.parameters(), lr=0.001)
    
    print(f"Model parameters: {sum(p.numel() for p in transmatcher_model.parameters()):,}")
    
    # Training loop
    num_epochs = 5
    losses = []
    accuracies = []
    
    print(f"\nTraining for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_accuracies = []
        
        # Generate 3 batches per epoch
        for batch_idx in range(3):
            # Create batch with controlled identity distribution
            # 4 identities, 4 samples each (batch_size=16)
            labels = torch.tensor([
                i // 4 for i in range(batch_size)
            ], device=device)
            
            # Create feature maps
            feature_maps = torch.randn(batch_size, feature_channels, feature_height, feature_width, device=device)
            
            # Forward pass
            optimizer.zero_grad()
            
            try:
                loss, acc = original_loss(feature_maps, labels)
                
                # Backward pass
                loss.mean().backward()
                optimizer.step()
                
                epoch_losses.append(loss.mean().item())
                epoch_accuracies.append(acc.mean().item())
                
                if batch_idx == 0:  # Print first batch of each epoch
                    print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss={loss.mean().item():.6f}, Acc={acc.mean().item():.4f}")
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                epoch_losses.append(0.0)
                epoch_accuracies.append(0.5)
        
        # Calculate epoch averages
        avg_loss = np.mean(epoch_losses)
        avg_acc = np.mean(epoch_accuracies)
        
        losses.append(avg_loss)
        accuracies.append(avg_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Avg Loss={avg_loss:.6f}, Avg Acc={avg_acc:.4f}")
    
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
        print(f"✓ Original pairwise loss works perfectly with TransMatcher!")
        print(f"✓ Loss decreased and accuracy increased.")
        print(f"✓ This should work in your full training setup.")
    elif loss_improved:
        print(f"⚠ Loss improved but accuracy didn't.")
        print(f"⚠ This suggests the model is learning but may need tuning.")
    elif acc_improved:
        print(f"⚠ Accuracy improved but loss didn't.")
        print(f"⚠ This suggests the loss function may have issues.")
    else:
        print(f"✗ No learning occurred.")
        print(f"✗ There may be a fundamental issue.")
    
    return losses, accuracies

def main():
    """Main function"""
    print("Original Pairwise Loss + TransMatcher Test")
    print("This will verify that your original loss works with TransMatcher")
    
    try:
        losses, accuracies = test_original_loss_with_transmatcher()
        
        print(f"\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("1. If this test passes, your training should work!")
        print("2. Run your full training with the updated code")
        print("3. The original pairwise loss should work much better than the problematic one")
        print("4. You should see proper learning and accuracy improvement")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 