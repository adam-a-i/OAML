#!/usr/bin/env python3
"""
Simplified TransMatcher loss function that should work better for training.
This addresses the issues we found in the pairwise loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTransMatcherLoss(nn.Module):
    def __init__(self, matcher, margin=1.0):
        """
        Simplified loss for TransMatcher training
        
        Args:
            matcher: TransMatcher instance
            margin: margin for triplet loss
        """
        super(SimpleTransMatcherLoss, self).__init__()
        self.matcher = matcher
        self.margin = margin
        
    def forward(self, features, labels):
        """
        Compute simplified loss for TransMatcher
        
        Args:
            features: [batch_size, channels, height, width] feature maps
            labels: [batch_size] identity labels
            
        Returns:
            loss: scalar loss value
            accuracy: scalar accuracy value
        """
        batch_size = features.size(0)
        device = features.device
        
        # Normalize features
        features_flat = features.view(batch_size, -1)
        features_norm = torch.norm(features_flat, p=2, dim=1, keepdim=True).clamp(min=1e-8)
        features_normalized = (features_flat / features_norm).view_as(features)
        
        # Get similarity scores from TransMatcher
        self.matcher.make_kernel(features_normalized)
        scores = self.matcher(features_normalized)  # [batch_size, batch_size]
        
        # Create positive/negative masks
        labels_expanded = labels.unsqueeze(1)
        positive_mask = (labels_expanded == labels_expanded.t()).float()
        negative_mask = (labels_expanded != labels_expanded.t()).float()
        
        # Remove self-comparisons
        eye_mask = torch.eye(batch_size, device=device)
        positive_mask = positive_mask * (1 - eye_mask)
        negative_mask = negative_mask * (1 - eye_mask)
        
        # Compute triplet loss
        total_loss = 0.0
        valid_triplets = 0
        
        for i in range(batch_size):
            # Get positive and negative scores for this anchor
            pos_scores = scores[i][positive_mask[i] > 0]
            neg_scores = scores[i][negative_mask[i] > 0]
            
            if len(pos_scores) > 0 and len(neg_scores) > 0:
                # Find hardest positive and negative
                hardest_pos = pos_scores.min()  # Lowest positive score
                hardest_neg = neg_scores.max()  # Highest negative score
                
                # Triplet loss
                triplet_loss = F.relu(hardest_neg - hardest_pos + self.margin)
                total_loss += triplet_loss
                valid_triplets += 1
        
        # Average loss
        if valid_triplets > 0:
            loss = total_loss / valid_triplets
        else:
            loss = torch.tensor(0.0, device=device)
        
        # Compute accuracy (simplified)
        correct = 0
        total = 0
        
        for i in range(batch_size):
            pos_scores = scores[i][positive_mask[i] > 0]
            neg_scores = scores[i][negative_mask[i] > 0]
            
            if len(pos_scores) > 0 and len(neg_scores) > 0:
                max_pos = pos_scores.max()
                max_neg = neg_scores.max()
                
                if max_pos > max_neg:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0.5
        
        return loss, torch.tensor(accuracy, device=device)

class ContrastiveTransMatcherLoss(nn.Module):
    def __init__(self, matcher, temperature=0.1):
        """
        Contrastive loss for TransMatcher training
        
        Args:
            matcher: TransMatcher instance
            temperature: temperature for contrastive loss
        """
        super(ContrastiveTransMatcherLoss, self).__init__()
        self.matcher = matcher
        self.temperature = temperature
        
    def forward(self, features, labels):
        """
        Compute contrastive loss for TransMatcher
        
        Args:
            features: [batch_size, channels, height, width] feature maps
            labels: [batch_size] identity labels
            
        Returns:
            loss: scalar loss value
            accuracy: scalar accuracy value
        """
        batch_size = features.size(0)
        device = features.device
        
        # Normalize features
        features_flat = features.view(batch_size, -1)
        features_norm = torch.norm(features_flat, p=2, dim=1, keepdim=True).clamp(min=1e-8)
        features_normalized = (features_flat / features_norm).view_as(features)
        
        # Get similarity scores from TransMatcher
        self.matcher.make_kernel(features_normalized)
        scores = self.matcher(features_normalized)  # [batch_size, batch_size]
        
        # Apply temperature scaling
        scores = scores / self.temperature
        
        # Create positive/negative masks
        labels_expanded = labels.unsqueeze(1)
        positive_mask = (labels_expanded == labels_expanded.t()).float()
        
        # Remove self-comparisons
        eye_mask = torch.eye(batch_size, device=device)
        positive_mask = positive_mask * (1 - eye_mask)
        
        # Compute contrastive loss
        total_loss = 0.0
        correct = 0
        total = 0
        
        for i in range(batch_size):
            # Get positive and negative scores for this anchor
            pos_mask = positive_mask[i] > 0
            neg_mask = ~pos_mask
            neg_mask[i] = False  # Remove self-comparison
            
            pos_scores = scores[i][pos_mask]
            neg_scores = scores[i][neg_mask]
            
            if len(pos_scores) > 0 and len(neg_scores) > 0:
                # Contrastive loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
                pos_term = torch.logsumexp(pos_scores, dim=0)
                neg_term = torch.logsumexp(neg_scores, dim=0)
                
                # Use log-sum-exp trick for numerical stability
                max_score = torch.max(torch.cat([pos_scores, neg_scores]))
                pos_term = max_score + torch.log(torch.sum(torch.exp(pos_scores - max_score)))
                neg_term = max_score + torch.log(torch.sum(torch.exp(neg_scores - max_score)))
                
                loss_i = -pos_term + neg_term
                total_loss += loss_i
                
                # Accuracy: check if max positive > max negative
                max_pos = pos_scores.max()
                max_neg = neg_scores.max()
                if max_pos > max_neg:
                    correct += 1
                total += 1
        
        # Average loss
        if total > 0:
            loss = total_loss / total
            accuracy = correct / total
        else:
            loss = torch.tensor(0.0, device=device)
            accuracy = torch.tensor(0.5, device=device)
        
        return loss, torch.tensor(accuracy, device=device)

def test_loss_functions():
    """Test the new loss functions"""
    import transmatcher
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    # Create test data
    features = torch.randn(batch_size, feature_channels, feature_height, feature_width, device=device)
    labels = torch.tensor([i // 4 for i in range(batch_size)], device=device)  # 4 identities, 4 samples each
    
    print("Testing SimpleTransMatcherLoss...")
    simple_loss = SimpleTransMatcherLoss(transmatcher_model, margin=1.0)
    loss1, acc1 = simple_loss(features, labels)
    print(f"  Loss: {loss1.item():.6f}, Accuracy: {acc1.item():.4f}")
    
    print("Testing ContrastiveTransMatcherLoss...")
    contrastive_loss = ContrastiveTransMatcherLoss(transmatcher_model, temperature=0.1)
    loss2, acc2 = contrastive_loss(features, labels)
    print(f"  Loss: {loss2.item():.6f}, Accuracy: {acc2.item():.4f}")
    
    return simple_loss, contrastive_loss

if __name__ == "__main__":
    test_loss_functions() 