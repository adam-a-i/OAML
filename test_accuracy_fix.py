"""
Test script to verify the accuracy calculation fix
"""

import torch
import torch.nn.functional as F
from transmatcher import TransMatcher
from pairwise_matching_loss import PairwiseMatchingLoss

def test_accuracy_calculation():
    print("Testing accuracy calculation fix...")
    
    # Initialize TransMatcher
    seq_len = 24 * 8
    d_model = 512
    num_layers = 3
    transmatcher = TransMatcher(seq_len, d_model, num_layers)
    pairwise_loss = PairwiseMatchingLoss(transmatcher)
    
    # Create test data with PK sampling (P=4, K=2)
    batch_size = 8
    height, width = 24, 8
    feature_maps = torch.randn(batch_size, height, width, d_model * num_layers)
    
    # Create labels: [0,0,1,1,2,2,3,3] - 4 identities, 2 samples each
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    
    print(f"Labels: {labels}")
    print(f"Unique labels: {torch.unique(labels)}")
    
    # Test the loss and accuracy
    loss, acc = pairwise_loss(feature_maps, labels)
    
    print(f"\nResults:")
    print(f"Loss mean: {loss.mean().item():.6f}")
    print(f"Accuracy mean: {acc.mean().item():.3f}")
    print(f"Individual accuracies: {acc}")
    
    # Manual verification of accuracy calculation
    print(f"\nManual verification:")
    
    # Get the scores from the matcher
    transmatcher.make_kernel(feature_maps)
    scores = transmatcher(feature_maps)
    
    # Create pairwise labels
    target1 = labels.unsqueeze(1)
    mask = (target1 == target1.t())
    pair_labels = mask.float()
    
    print(f"Scores shape: {scores.shape}")
    print(f"Pair labels shape: {pair_labels.shape}")
    
    # Check each sample manually
    for i in range(batch_size):
        sample_scores = scores[i]
        positive_mask = pair_labels[i] == 1
        negative_mask = pair_labels[i] == 0
        
        # Exclude self-comparison
        positive_mask[i] = False
        negative_mask[i] = False
        
        if positive_mask.sum() > 0 and negative_mask.sum() > 0:
            positive_scores = sample_scores[positive_mask]
            negative_scores = sample_scores[negative_mask]
            
            max_pos = positive_scores.max()
            max_neg = negative_scores.max()
            
            manual_acc = (max_pos > max_neg).float()
            print(f"Sample {i} (label {labels[i]}): max_pos={max_pos:.3f}, max_neg={max_neg:.3f}, acc={manual_acc:.3f}")
        else:
            print(f"Sample {i} (label {labels[i]}): no valid pairs")
    
    return acc.mean().item()

if __name__ == "__main__":
    accuracy = test_accuracy_calculation()
    print(f"\nFinal accuracy: {accuracy:.3f}")
    
    if accuracy > 0.1:  # Should be better than random (0.125)
        print("✓ Accuracy calculation fix appears to be working!")
    else:
        print("✗ Accuracy is still too low - more investigation needed.") 