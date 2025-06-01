"""Class for the Query-Adaptive Convolution (QAConv)
    QAConv is an effective image matching method proposed in
    Shengcai Liao and Ling Shao, "Interpretable and Generalizable Person Re-Identification with Query-Adaptive
    Convolution and Temporal Lifting." In The European Conference on Computer Vision (ECCV), 23-28 August, 2020.
    Author:
        Shengcai Liao
        scliao@ieee.org
    Version:
        V1.3.1
        July 1, 2021
    """

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F


class QAConv(Module):
    def __init__(self, num_features, height, width):
        """
        Inputs:
            num_features: the number of feature channels in the final feature map.
            height: height of the final feature map
            width: width of the final feature map
        """
        super(QAConv, self).__init__()
        self.num_features = num_features
        self.height = height
        self.width = width
        self.bn = nn.BatchNorm1d(1)
        self.fc = nn.Linear(self.height * self.width, 1)
        self.logit_bn = nn.BatchNorm1d(1)
        self.reset_parameters()

    def reset_running_stats(self):
        self.bn.reset_running_stats()
        self.logit_bn.reset_running_stats()

    def reset_parameters(self):
        self.bn.reset_parameters()
        self.logit_bn.reset_parameters()
        with torch.no_grad():
            self.fc.weight.fill_(1. / (self.height * self.width))
    
    def clone(self):
        """Create a fresh clone of this module"""
        # Get device from current module
        device = next(self.parameters()).device
        
        # Create a new instance with same parameters
        clone = QAConv(self.num_features, self.height, self.width).to(device)
        
        # Use state_dict for proper parameter copying without gradient issues
        clone.load_state_dict(self.state_dict())
        
        return clone

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, prob_fea, gal_fea, max_batch_size=128):
        """
        Forward pass for QAConv matching
        
        Args:
            prob_fea: Feature maps from probe images [batch_size, num_features, height, width]
            gal_fea: Feature maps from gallery images [batch_size, num_features, height, width]
            max_batch_size: Maximum batch size for processing to manage memory
            
        Returns:
            Similarity scores between probe and gallery features [prob_size, gal_size]
        """
        # Ensure inputs are properly shaped
        if prob_fea.dim() != 4 or gal_fea.dim() != 4:
            raise ValueError(f"Expected 4D input but got probe: {prob_fea.dim()}D, gallery: {gal_fea.dim()}D")
            
        # Get device
        device = prob_fea.device
        
        # Check for NaN values in input
        if torch.isnan(prob_fea).any() or torch.isnan(gal_fea).any():
            print("WARNING: Input features contain NaN values. Replacing with zeros.")
            prob_fea = torch.nan_to_num(prob_fea, nan=0.0)
            gal_fea = torch.nan_to_num(gal_fea, nan=0.0)
        
        # Ensure inputs are normalized
        prob_norm = torch.norm(prob_fea.view(prob_fea.size(0), -1), p=2, dim=1)
        gal_norm = torch.norm(gal_fea.view(gal_fea.size(0), -1), p=2, dim=1)
        
        if (prob_norm < 1e-8).any() or (gal_norm < 1e-8).any():
            print("WARNING: Some feature vectors have very small norms. Normalizing all features.")
            prob_fea = F.normalize(prob_fea, p=2, dim=1)
            gal_fea = F.normalize(gal_fea, p=2, dim=1)
            
        # Ensure inputs are contiguous
        prob_fea = prob_fea.contiguous()
        gal_fea = gal_fea.contiguous()
        
        hw = self.height * self.width
        prob_size = prob_fea.size(0)
        gal_size = gal_fea.size(0)
        
        # Reshape for correlation
        prob_fea = prob_fea.view(prob_size, self.num_features, hw)
        gal_fea = gal_fea.view(gal_size, self.num_features, hw)
        
        # Initialize output scores tensor
        all_scores = torch.zeros(prob_size, gal_size, device=device)
        
        try:
            # Process in batches to reduce memory usage
            for p_start in range(0, prob_size, max_batch_size):
                p_end = min(p_start + max_batch_size, prob_size)
                p_batch = prob_fea[p_start:p_end]
                
                for g_start in range(0, gal_size, max_batch_size):
                    g_end = min(g_start + max_batch_size, gal_size)
                    g_batch = gal_fea[g_start:g_end]
                    
                    # Calculate correlation between probe and gallery features in batches
                    score_batch = torch.einsum('p c s, g c r -> p g r s', p_batch, g_batch)
                    
                    # Check for NaN values after correlation
                    if torch.isnan(score_batch).any():
                        print(f"WARNING: NaN values after correlation. Batch shape: {score_batch.shape}")
                        score_batch = torch.nan_to_num(score_batch, nan=0.0)
                    
                    # Max pooling over spatial dimensions
                    max_r = score_batch.max(dim=2)[0]  # [p, g, s]
                    max_s = score_batch.max(dim=3)[0]  # [p, g, r]
                    score_batch = torch.cat((max_r, max_s), dim=-1)  # [p, g, r+s]
                    
                    # Check for NaN values after max pooling
                    if torch.isnan(score_batch).any():
                        print(f"WARNING: NaN values after max pooling. Replacing with zeros.")
                        score_batch = torch.nan_to_num(score_batch, nan=0.0)
                    
                    # Apply batch normalization to make training stable
                    score_batch = score_batch.reshape(-1, 1, hw)
                    
                    # Skip batch norm if there are NaN values
                    try:
                        score_batch = self.bn(score_batch).view(-1, hw)
                    except Exception as e:
                        print(f"WARNING: Error in batch norm: {e}. Using raw scores.")
                        score_batch = score_batch.view(-1, hw)
                    
                    # Apply FC layer for dimension reduction
                    score_batch = self.fc(score_batch)
                    
                    # Sum the scores
                    score_batch = score_batch.view(-1, 2).sum(dim=1, keepdim=True)
                    
                    # Apply batch normalization to scores, with safety check
                    try:
                        score_batch = self.logit_bn(score_batch)
                    except Exception as e:
                        print(f"WARNING: Error in logit batch norm: {e}. Using raw scores.")
                    
                    # Final NaN check
                    if torch.isnan(score_batch).any():
                        print(f"WARNING: Final scores contain NaN values. Replacing with zeros.")
                        score_batch = torch.nan_to_num(score_batch, nan=0.0)
                    
                    # Reshape to [batch_prob_size, batch_gal_size]
                    score_batch = score_batch.view(p_end - p_start, g_end - g_start)
                    
                    # Store in the full score matrix
                    all_scores[p_start:p_end, g_start:g_end] = score_batch
        
        except Exception as e:
            import traceback
            print(f"ERROR in QAConv forward pass: {e}")
            print(traceback.format_exc())
            # Return zeros as fallback
            return torch.zeros(prob_size, gal_size, device=device)
        
        # Final safety check
        if torch.isnan(all_scores).any():
            print("WARNING: Output scores contain NaN values. Replacing with zeros.")
            all_scores = torch.nan_to_num(all_scores, nan=0.0)
            
        return all_scores
    
    def match_pairs(self, probe_features, gallery_features):
        """
        Match only corresponding pairs directly (gallery[i] with probe[i])
        This is more efficient than the full match when we only need diagonal elements
        
        Args:
            probe_features: Feature maps of probe images [batch_size, num_features, height, width]
            gallery_features: Feature maps of gallery images [batch_size, num_features, height, width]
        
        Returns:
            Similarity scores between corresponding pairs [batch_size]
        """
        # Check that we have the same number of probes and galleries
        assert probe_features.size(0) == gallery_features.size(0), "Number of probe and gallery features must be the same"
        
        # Get device
        device = probe_features.device
        
        # Ensure model is in eval mode for matching
        training_state = self.training
        self.eval()
        
        # Ensure features are on the correct device and in the right format
        probe_features = probe_features.to(device, dtype=torch.float32).contiguous()
        gallery_features = gallery_features.to(device, dtype=torch.float32).contiguous()
        
        num_pairs = probe_features.size(0)
        hw = self.height * self.width
        
        # Initialize scores tensor
        scores = torch.zeros(num_pairs, device=device)
        
        # Process each pair individually
        with torch.no_grad():
            batch_size = 32  # Process in small batches
            
            for start in range(0, num_pairs, batch_size):
                end = min(start + batch_size, num_pairs)
                p_batch = probe_features[start:end]
                g_batch = gallery_features[start:end]
                
                # Reshape for correlation
                p_batch = p_batch.view(end-start, self.num_features, hw)
                g_batch = g_batch.view(end-start, self.num_features, hw)
                
                # Calculate correlation for each pair
                for i in range(end-start):
                    p_fea = p_batch[i:i+1]  # Keep dimension
                    g_fea = g_batch[i:i+1]  # Keep dimension
                    
                    # Calculate correlation using einsum for single pair
                    score = torch.einsum('p c s, g c r -> p g r s', p_fea, g_fea)
                    
                    # Max pooling over spatial dimensions
                    max_r = score.max(dim=2)[0]  # [1, 1, s]
                    max_s = score.max(dim=3)[0]  # [1, 1, r]
                    score = torch.cat((max_r, max_s), dim=-1)  # [1, 1, r+s]
                    
                    # Apply batch normalization
                    try:
                        score = self.bn(score.reshape(-1, 1, hw)).view(-1, hw)
                    except Exception as e:
                        score = score.reshape(-1, 1, hw).view(-1, hw)
                    
                    # Apply FC layer
                    score = self.fc(score)
                    
                    # Sum scores and apply logit_bn
                    score = score.view(-1, 2).sum(dim=1, keepdim=True)
                    
                    try:
                        score = self.logit_bn(score)
                    except Exception as e:
                        pass
                    
                    # Store score
                    scores[start + i] = score.view(-1)[0]
        
        # Restore previous training state
        self.train(training_state)
        
        # Check for NaNs
        if torch.isnan(scores).any():
            print("WARNING: Some scores are NaN. Replacing with zeros.")
            scores = torch.nan_to_num(scores, nan=0.0)
            
        return scores

    def match(self, probe_features, gallery_features):
        """
        Match probe features against gallery features using QAConv
        This is a convenience wrapper around the forward method
        
        Args:
            probe_features: Feature maps of probe images
            gallery_features: Feature maps of gallery images
        
        Returns:
            Similarity scores between probe and gallery features
        """
        # Ensure model is in eval mode for matching
        training_state = self.training
        self.eval()
        
        # Get device from self parameters
        device = next(self.parameters()).device
        
        # Ensure features are on the correct device and in the right format
        probe_features = probe_features.to(device, dtype=torch.float32).contiguous()
        gallery_features = gallery_features.to(device, dtype=torch.float32).contiguous()
        
        # Forward pass with no gradient tracking
        with torch.no_grad():
            # Direct use of forward method
            scores = self.forward(probe_features, gallery_features)
        
        # Restore previous training state
        self.train(training_state)
        
        return scores 