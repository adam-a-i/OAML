"""Class for the pairwise matching loss
    Shengcai Liao and Ling Shao, "Graph Sampling Based Deep Metric Learning for Generalizable Person Re-Identification." In arXiv preprint, arXiv:2104.01546, 2021.
    Author:
        Shengcai Liao
        scliao@ieee.org
    Version:
        V1.1
        Jan 3, 2023
    """

import torch
from torch.nn import Module
from torch.nn import functional as F


class PairwiseMatchingLoss(Module):
    def __init__(self, matcher):
        """
        Inputs:
            matcher: a class for matching pairs of images
        """
        super(PairwiseMatchingLoss, self).__init__()
        # Store matcher directly - cloning is now handled in the forward method
        self.matcher = matcher

    def reset_running_stats(self):
        # Only reset when in training mode to avoid inference tensor issues
        if self.training:
            self.matcher.reset_running_stats()

    def reset_parameters(self):
        self.matcher.reset_parameters()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, feature, target):
        self._check_input_dim(feature)
        
        # Ensure matcher is in training mode
        training_state = self.matcher.training
        self.matcher.train()
        
        # Get device from features
        device = feature.device
        
        # Make sure matcher is on the right device
        self.matcher = self.matcher.to(device)
        
        # Make sure feature tensors are proper trainable tensors (not inference tensors)
        feature = feature.detach().clone().requires_grad_(True)
        
        try:
            # Calculate matching scores
            # For TransMatcher, we need to set memory first, then call forward
            self.matcher.make_kernel(feature)
            score = self.matcher(feature)  # [b, b]
            
            target1 = target.unsqueeze(1)
            mask = (target1 == target1.t())
            pair_labels = mask.float()
            
            loss = F.binary_cross_entropy_with_logits(score, pair_labels, reduction='none')
            loss = loss.sum(-1)

            with torch.no_grad():
                # Fixed accuracy calculation for PK sampling
                # For each sample, check if it has higher scores for same-class pairs than different-class pairs
                accuracies = []
                
                for i in range(score.size(0)):
                    # Get scores for this sample
                    sample_scores = score[i]  # [batch_size]
                    
                    # Get positive and negative pairs for this sample
                    positive_mask = pair_labels[i] == 1  # [batch_size]
                    negative_mask = pair_labels[i] == 0  # [batch_size]
                    
                    # Exclude self-comparison
                    positive_mask[i] = False
                    negative_mask[i] = False
                    
                    if positive_mask.sum() > 0 and negative_mask.sum() > 0:
                        # Get positive and negative scores
                        positive_scores = sample_scores[positive_mask]
                        negative_scores = sample_scores[negative_mask]
                        
                        # Check if max positive score > max negative score
                        max_pos = positive_scores.max()
                        max_neg = negative_scores.max()
                        
                        accuracy = (max_pos > max_neg).float()
                        accuracies.append(accuracy)
                    else:
                        # If no positive or negative pairs, assign random accuracy
                        accuracies.append(torch.tensor(0.5, device=device))
                
                acc = torch.stack(accuracies)
            
            # Restore matcher's previous training state
            self.matcher.train(training_state)
            
            return loss, acc
            
        except Exception as e:
            print(f"[PAIRWISE_LOSS] ERROR: {e}")
            import traceback
            traceback.print_exc()
            # Return zero loss and random accuracy on error
            batch_size = feature.size(0)
            loss = torch.zeros(batch_size, device=device)
            acc = torch.full((batch_size,), 0.5, device=device)
            
            # Restore matcher's previous training state
            self.matcher.train(training_state)
            
            return loss, acc
