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
import math


class QAConv(Module):
    def __init__(self, num_features, height, width, num_classes=None, k_nearest=20):
        """
        Inputs:
            num_features: the number of feature channels in the final feature map.
            height: height of the final feature map
            width: width of the final feature map
            num_classes: number of identity classes (optional, for classification mode)
            k_nearest: number of nearest neighbors to consider for class neighbors
        """
        super(QAConv, self).__init__()
        self.num_features = num_features
        self.height = height
        self.width = width
        self.bn = nn.BatchNorm1d(1)
        self.fc = nn.Linear(self.height * self.width * 2, 1)
        self.logit_bn = nn.BatchNorm1d(1)
        
        # Initialize class embeddings if num_classes is provided
        self.class_embed = None
        if num_classes is not None:
            self.class_embed = nn.Parameter(
                torch.randn(num_classes, num_features, height, width) / num_features**0.5
            )
        
        # Graph sampling properties
        self.k_nearest = k_nearest
        self.class_neighbors = None
        self.chunk_size = 1024  # Process class embeddings in chunks

        # Kernel storage for pairwise matching (set by make_kernel)
        self._kernel = None

        self.reset_parameters()

    def reset_running_stats(self):
        self.bn.reset_running_stats()
        self.logit_bn.reset_running_stats()
        # Reset class neighbors at start of epoch
        self.class_neighbors = None

    def reset_parameters(self):
        self.bn.reset_parameters()
        self.logit_bn.reset_parameters()
        with torch.no_grad():
            self.fc.weight.fill_(1. / (self.height * self.width))
    
    def make_kernel(self, features):
        """
        Set up kernel/memory for matching. This method is expected by PairwiseMatchingLoss.
        For QAConv, we don't need to store anything, but we need this method for compatibility.
        
        Args:
            features: [B, C, H, W] feature maps
        """
        # QAConv doesn't need to store kernel/memory like TransMatcher
        # This method exists for compatibility with PairwiseMatchingLoss
        pass

    def clone(self):
        """Create a fresh clone of this module"""
        # Get device from current module
        device = next(self.parameters()).device
        
        # Create a new instance with same parameters
        clone = QAConv(
            self.num_features, 
            self.height, 
            self.width,
            num_classes=self.class_embed.size(0) if self.class_embed is not None else None
        ).to(device)
        
        # Use state_dict for proper parameter copying without gradient issues
        clone.load_state_dict(self.state_dict())
        
        return clone

    def make_kernel(self, feature):
        """Store features as kernel (gallery) for next forward call.

        This method is called by PairwiseMatchingLoss before forward() to set up
        the gallery features for self-matching (probe == gallery).

        Args:
            feature: [B, C, H, W] feature maps to use as gallery
        """
        self._kernel = feature.clone()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def compute_class_neighbors(self):
        """Compute k-nearest neighbor classes for each class based on class embeddings using chunked processing"""
        with torch.no_grad():
            num_classes = self.class_embed.size(0)
            device = self.class_embed.device
            
            # Initialize similarity matrix in chunks
            similarity_matrix = torch.zeros(num_classes, num_classes, device=device)
            
            # Normalize class embeddings
            class_embeds_flat = F.normalize(
                self.class_embed.view(num_classes, -1), p=2, dim=1
            )
            
            # Process in chunks to reduce memory usage
            for i in range(0, num_classes, self.chunk_size):
                end_i = min(i + self.chunk_size, num_classes)
                chunk_i = class_embeds_flat[i:end_i]
                
                for j in range(0, num_classes, self.chunk_size):
                    end_j = min(j + self.chunk_size, num_classes)
                    chunk_j = class_embeds_flat[j:end_j]
                    
                    # Compute similarity for this chunk
                    similarity_matrix[i:end_i, j:end_j] = torch.mm(chunk_i, chunk_j.t())
            
            # Get top-k neighbors for each class (excluding self)
            _, indices = similarity_matrix.topk(k=self.k_nearest + 1, dim=1)
            self.class_neighbors = indices[:, 1:].contiguous()

    def apply_occlusion_weight(self, similarity, query_occ, gallery_occ, method="scaling"):
        """
        Apply occlusion-aware weighting to similarity scores.
        
        Args:
            similarity: [B_probe, B_gallery, hw, hw] similarity tensor from correlation
            query_occ: [B_probe, 1, H, W] query occlusion confidence map (1=visible, 0=occluded)
            gallery_occ: [B_gallery, 1, H, W] gallery occlusion confidence map (1=visible, 0=occluded)
            method: "scaling" (efficient) or "outer" (full outer product)
        
        Returns:
            weighted_similarity: [B_probe, B_gallery, hw, hw] occlusion-weighted similarity
        """
        B_probe, B_gallery, hw_query, hw_gallery = similarity.shape
        
        # Flatten occlusion maps to match hw dimensions: [B, 1, H, W] -> [B, hw]
        query_flat = query_occ.view(B_probe, -1)      # [B_probe, hw]
        gallery_flat = gallery_occ.view(B_gallery, -1) # [B_gallery, hw]
        
        if method == "outer":
            # Full outer-product mask: each query location weighted by all gallery locations
            # Reshape for broadcasting: [B_probe, hw, 1] * [B_gallery, 1, hw] -> [B_probe, B_gallery, hw, hw]
            q_weight = query_flat.unsqueeze(1).unsqueeze(3)     # [B_probe, 1, hw, 1]
            g_weight = gallery_flat.unsqueeze(0).unsqueeze(2)   # [1, B_gallery, 1, hw]
            weight = q_weight * g_weight                         # [B_probe, B_gallery, hw, hw]
        else:  # scaling method (more efficient and stable)
            # Scale by query visibility and gallery visibility
            # Reshape for broadcasting: [B_probe, hw, 1] * [B_gallery, 1, hw] -> [B_probe, B_gallery, hw, hw]
            q_weight = query_flat.unsqueeze(1).unsqueeze(3)     # [B_probe, 1, hw, 1]
            g_weight = gallery_flat.unsqueeze(0).unsqueeze(2)   # [1, B_gallery, 1, hw]
            weight = q_weight * g_weight                         # [B_probe, B_gallery, hw, hw]
        
        return similarity * weight

    def _compute_similarity_batch(self, prob_fea, gal_fea, query_occ=None, gallery_occ=None, occlusion_method="scaling"):
        """Compute similarity between probe and gallery features"""
        # Get shapes and verify dimensions
        prob_size = prob_fea.size(0)
        gal_size = gal_fea.size(0)
        hw = self.height * self.width
        dtype = prob_fea.dtype
        
        # Ensure both inputs have same dtype
        gal_fea = gal_fea.to(dtype=dtype)
        
        # Reshape features for correlation - do this once outside loops
        prob_fea = prob_fea.view(prob_size, self.num_features, hw)
        gal_fea = gal_fea.view(gal_size, self.num_features, hw)
        
        # Compute correlation using efficient batched operations
        # Use chunking for memory efficiency on large inputs
        chunk_size = 64  # Adjust based on GPU memory
        score = torch.zeros(prob_size, gal_size, hw, hw, device=prob_fea.device, dtype=dtype)
        
        for i in range(0, prob_size, chunk_size):
            end_i = min(i + chunk_size, prob_size)
            prob_chunk = prob_fea[i:end_i]
            
            for j in range(0, gal_size, chunk_size):
                end_j = min(j + chunk_size, gal_size)
                gal_chunk = gal_fea[j:end_j]
                
                # Compute correlation for this chunk
                score[i:end_i, j:end_j] = torch.einsum('p c s, g c r -> p g r s', 
                                                       prob_chunk, gal_chunk)
        
        # Apply occlusion weighting if occlusion maps are provided
        if query_occ is not None and gallery_occ is not None:
            score = self.apply_occlusion_weight(score, query_occ, gallery_occ, occlusion_method)
        
        # Max pooling over spatial dimensions - operate on full tensor at once
        max_r = score.max(dim=2)[0]  # [p, g, s]
        max_s = score.max(dim=3)[0]  # [p, g, r]
        score = torch.cat((max_r, max_s), dim=-1)  # [p, g, hw*2]
        
        # Process through batch norm and FC layers
        orig_shape = score.shape
        score = score.view(-1, 1, hw * 2)  # [p*g, 1, hw*2]
        score = self.bn(score)
        score = score.view(-1, hw * 2)  # [p*g, hw*2]
        score = self.fc(score)  # [p*g, 1]
        score = self.logit_bn(score)
        score = score.view(prob_size, gal_size)  # [p, g]
        
        return score

    def _compute_similarity_batch_with_occlusion(self, prob_fea, gal_fea, prob_occ=None, gal_occ=None):
        """
        Compute similarity between probe and gallery features with occlusion-aware weighting.

        The occlusion maps weight the spatial correlation scores BEFORE max pooling,
        so that occluded regions contribute less to the final similarity score.

        Args:
            prob_fea: Probe features [B_p, C, H, W]
            gal_fea: Gallery features [B_g, C, H, W]
            prob_occ: Probe occlusion maps [B_p, 1, H, W] (optional, values in [0,1])
            gal_occ: Gallery occlusion maps [B_g, 1, H, W] (optional, values in [0,1])

        Returns:
            Similarity scores [B_p, B_g]

        Note:
            If occlusion maps are None, falls back to standard similarity computation.
            Occlusion values: 1 = visible, 0 = occluded
            Weight formula: weight[i,j] = prob_occ[i] * gal_occ[j]
        """
        # Fall back to standard computation if no occlusion maps provided
        if prob_occ is None or gal_occ is None:
            return self._compute_similarity_batch(prob_fea, gal_fea)

        # Get shapes and verify dimensions
        prob_size = prob_fea.size(0)
        gal_size = gal_fea.size(0)
        hw = self.height * self.width
        dtype = prob_fea.dtype

        # Ensure both inputs have same dtype
        gal_fea = gal_fea.to(dtype=dtype)

        # Reshape features for correlation
        prob_fea = prob_fea.view(prob_size, self.num_features, hw)
        gal_fea = gal_fea.view(gal_size, self.num_features, hw)

        # Flatten occlusion maps: [B, 1, H, W] -> [B, hw]
        prob_occ_flat = prob_occ.view(prob_size, -1).to(dtype=dtype)  # [B_p, hw]
        gal_occ_flat = gal_occ.view(gal_size, -1).to(dtype=dtype)     # [B_g, hw]

        # Compute correlation using efficient batched operations
        chunk_size = 64
        score = torch.zeros(prob_size, gal_size, hw, hw, device=prob_fea.device, dtype=dtype)

        for i in range(0, prob_size, chunk_size):
            end_i = min(i + chunk_size, prob_size)
            prob_chunk = prob_fea[i:end_i]

            for j in range(0, gal_size, chunk_size):
                end_j = min(j + chunk_size, gal_size)
                gal_chunk = gal_fea[j:end_j]

                # Compute correlation for this chunk: [chunk_p, chunk_g, hw, hw]
                score[i:end_i, j:end_j] = torch.einsum('p c s, g c r -> p g r s',
                                                        prob_chunk, gal_chunk)

        # Apply occlusion weighting BEFORE max pooling
        # Create weight matrix: weight[p, g, r, s] = prob_occ[p, r] * gal_occ[g, s]
        # where r indexes probe spatial locations and s indexes gallery spatial locations
        #
        # prob_occ_flat: [B_p, hw] -> [B_p, 1, hw, 1] for broadcasting
        # gal_occ_flat:  [B_g, hw] -> [1, B_g, 1, hw] for broadcasting
        prob_occ_expanded = prob_occ_flat.unsqueeze(1).unsqueeze(3)  # [B_p, 1, hw, 1]
        gal_occ_expanded = gal_occ_flat.unsqueeze(0).unsqueeze(2)    # [1, B_g, 1, hw]

        # Compute weights: [B_p, B_g, hw, hw]
        # weight[p, g, r, s] = prob_occ[p, r] * gal_occ[g, s]
        occlusion_weights = prob_occ_expanded * gal_occ_expanded

        # Apply weights to correlation scores
        # This downweights correlations where either probe or gallery location is occluded
        score = score * occlusion_weights

        # Max pooling over spatial dimensions
        # Note: After weighting, occluded locations will have lower scores and contribute less
        max_r = score.max(dim=2)[0]  # [p, g, s] - max over probe spatial dim
        max_s = score.max(dim=3)[0]  # [p, g, r] - max over gallery spatial dim
        score = torch.cat((max_r, max_s), dim=-1)  # [p, g, hw*2]

        # Process through batch norm and FC layers
        score = score.view(-1, 1, hw * 2)  # [p*g, 1, hw*2]
        score = self.bn(score)
        score = score.view(-1, hw * 2)     # [p*g, hw*2]
        score = self.fc(score)             # [p*g, 1]
        score = self.logit_bn(score)
        score = score.view(prob_size, gal_size)  # [p, g]

        return score

    def forward(self, prob_fea, gal_fea=None, labels=None, prob_occ=None, gal_occ=None):
        """
        Forward pass supporting both training and inference modes.

        Args:
            prob_fea: Probe features [batch_size, num_features, height, width]
            gal_fea: Gallery features (optional) [batch_size, num_features, height, width]
                    If None, use class embeddings (training mode)
            labels: Class labels for each sample in the batch [batch_size]
                    If provided, enables class-level neighbor computation
            prob_occ: Probe occlusion maps [batch_size, 1, height, width] (optional)
                    Values in [0, 1] where 1=visible, 0=occluded
            gal_occ: Gallery occlusion maps [batch_size, 1, height, width] (optional)
                    Values in [0, 1] where 1=visible, 0=occluded

        Returns:
            Similarity scores between probe and gallery features

        Note:
            When occlusion maps are provided, the spatial correlations are weighted
            by the product of probe and gallery occlusion values before max pooling.
            This ensures that occluded regions contribute less to matching scores.
        """
        self._check_input_dim(prob_fea)
        device = prob_fea.device
        dtype = prob_fea.dtype  # Get input dtype

        # Force normalization if needed - do this once up front
        prob_fea = F.normalize(prob_fea, p=2, dim=1)

        # Use stored kernel if set by make_kernel() and no gal_fea provided
        # This is used by PairwiseMatchingLoss for self-matching
        if gal_fea is None and self._kernel is not None:
            gal_fea = self._kernel
            self._kernel = None  # Clear after use

        # Training mode with class embeddings
        if self.training and gal_fea is None and self.class_embed is not None:
            # Compute class neighbors if not already done
            if self.class_neighbors is None:
                self.compute_class_neighbors()
                
            batch_size = prob_fea.size(0)
            
            # Initialize output scores with same dtype as input
            dtype = prob_fea.dtype
            all_scores = torch.zeros(batch_size, self.class_embed.size(0), device=device, dtype=dtype)
            
            # If labels are provided, we can optimize by grouping samples by class
            if labels is not None:
                # Group samples by label for class-level processing
                unique_labels, inverse_indices = torch.unique(labels, return_inverse=True)
                num_unique_classes = len(unique_labels)
                
                # Create a dict mapping class label to indices in the batch for faster lookup
                class_indices = {}
                for idx, label in enumerate(labels):
                    label_item = label.item()
                    if label_item not in class_indices:
                        class_indices[label_item] = []
                    class_indices[label_item].append(idx)
                
                # Pre-normalize all class embeddings once
                normalized_class_embed = F.normalize(self.class_embed, p=2, dim=1)
                
                # Process in larger chunks to increase parallelism
                chunk_size = min(32, num_unique_classes)  # Process more classes at once
                for chunk_start in range(0, num_unique_classes, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, num_unique_classes)
                    chunk_labels = unique_labels[chunk_start:chunk_end]
                    
                    # Get all neighbors for all classes in this chunk at once
                    all_neighbor_indices = self.class_neighbors[chunk_labels].reshape(-1)
                    all_neighbor_embeds = normalized_class_embed[all_neighbor_indices]
                    
                    # Process each class in the chunk
                    for i, class_label in enumerate(chunk_labels):
                        class_label_item = class_label.item()
                        if class_label_item not in class_indices:
                            continue
                            
                        # Get samples and indices for this class
                        sample_indices = class_indices[class_label_item]
                        if not sample_indices:
                            continue
                            
                        # Get all samples of this class at once
                        class_samples = prob_fea[sample_indices]
                        
                        # Get neighbors for this class
                        neighbors = self.class_neighbors[class_label]
                        neighbor_embeds = all_neighbor_embeds[i*self.k_nearest:(i+1)*self.k_nearest]
                        
                        # Process all samples against all neighbors in one batch operation
                        if len(sample_indices) == 1:
                            # Single sample case - simpler processing
                            sample = class_samples
                            sample_scores = self._compute_similarity_batch(sample, neighbor_embeds, None, None, occlusion_method)
                            # Ensure dtypes match when assigning
                            all_scores[sample_indices[0], neighbors] = sample_scores.to(dtype=all_scores.dtype).view(-1)
                        else:
                            # Multiple samples case - batch together
                            # Compute pairwise scores for all samples against all neighbors
                            sample_scores = self._compute_similarity_batch(class_samples, neighbor_embeds, None, None, occlusion_method)
                            
                            # Assign scores to each sample - ensure dtypes match
                            for j, idx in enumerate(sample_indices):
                                all_scores[idx, neighbors] = sample_scores[j].to(dtype=all_scores.dtype)
                
                return all_scores
            
            # Fall back to original sample-by-sample processing if no labels provided
            batch_labels = torch.arange(batch_size, device=device) % self.class_embed.size(0)
            batch_neighbors = self.class_neighbors[batch_labels]  # [batch_size, k_nearest]
            
            # Pre-normalize all class embeddings once
            normalized_class_embed = F.normalize(self.class_embed, p=2, dim=1)
            
            # Process in larger chunks to manage memory
            chunk_size = 64  # Larger chunks for better parallelism
            for start in range(0, batch_size, chunk_size):
                end = min(start + chunk_size, batch_size)
                curr_batch_size = end - start
                
                # Get current batch's probe features and neighbors
                curr_probe = prob_fea[start:end]  # [curr_batch_size, num_features, height, width]
                curr_neighbors = batch_neighbors[start:end]  # [curr_batch_size, k_nearest]
                
                # Flatten neighbors and get embeddings
                flat_neighbors = curr_neighbors.view(-1)
                neighbor_embeds = normalized_class_embed[flat_neighbors]
                
                # Reshape to match dimensions
                neighbor_embeds = neighbor_embeds.view(curr_batch_size, self.k_nearest, self.num_features, self.height, self.width)
                
                # Compute all scores in a single batch operation
                for i in range(curr_batch_size):
                    sample = curr_probe[i:i+1]
                    sample_neighbors = neighbor_embeds[i]
                    
                    # Compute similarity for this sample against its neighbors
                    scores = self._compute_similarity_batch(sample, sample_neighbors, None, None, occlusion_method)
                    
                    # Assign scores to correct positions
                    all_scores[start + i, curr_neighbors[i]] = scores.view(-1)
            
            return all_scores
        
        # Handle "same" case
        if gal_fea == "same":
            gal_fea = prob_fea
        
        # Regular forward pass for matching
        if gal_fea is not None:
            self._check_input_dim(gal_fea)
            # Ensure gallery features have same dtype as probe features
            gal_fea = gal_fea.to(dtype=dtype)

            # Force normalization if needed
            if gal_fea is not prob_fea:  # Skip if gal_fea is the same object as prob_fea
                gal_fea = F.normalize(gal_fea, p=2, dim=1)

            # Use occlusion-aware computation if occlusion maps are provided
            if prob_occ is not None and gal_occ is not None:
                return self._compute_similarity_batch_with_occlusion(prob_fea, gal_fea, prob_occ, gal_occ)
            else:
                return self._compute_similarity_batch(prob_fea, gal_fea)

        return None

    def match(self, probe_features, gallery_features, probe_occ=None, gallery_occ=None):
        """
        Wrapper for matching mode with optional occlusion-aware computation.

        Args:
            probe_features: Probe features [B_p, C, H, W]
            gallery_features: Gallery features [B_g, C, H, W]
            probe_occ: Probe occlusion maps [B_p, 1, H, W] (optional)
            gallery_occ: Gallery occlusion maps [B_g, 1, H, W] (optional)

        Returns:
            Similarity scores [B_p, B_g]
        """
        training_state = self.training
        self.eval()
        with torch.no_grad():
            scores = self.forward(probe_features, gallery_features,
                                  prob_occ=probe_occ, gal_occ=gallery_occ)
        self.train(training_state)
        return scores 

    def on_epoch_start(self):
        """Reset class neighbors at start of epoch to force recomputation"""
        if self.training and self.class_embed is not None:
            self.class_neighbors = None
            self.compute_class_neighbors() 

    def match_pairs(self, probe_features, gallery_features, probe_occ=None, gallery_occ=None):
        """
        Match probe-gallery pairs directly during validation.
        This is a simplified version of forward() specifically for validation.
        Optimized for memory efficiency with batch processing.
        
        Args:
            probe_features: Features from probe images [batch_size, num_features, height, width]
            gallery_features: Features from gallery images [batch_size, num_features, height, width]
            probe_occ: Probe occlusion maps [batch_size, 1, height, width] (optional)
            gallery_occ: Gallery occlusion maps [batch_size, 1, height, width] (optional)
            
        Returns:
            Similarity scores between probe and gallery pairs [batch_size]
        """
        self._check_input_dim(probe_features)
        self._check_input_dim(gallery_features)
        
        # Ensure we're in eval mode
        training_state = self.training
        self.eval()
        
        batch_size = probe_features.size(0)
        device = probe_features.device
        dtype = probe_features.dtype
        
        # Process in smaller chunks for memory efficiency
        chunk_size = 64  # Adjust based on available memory
        pair_scores = torch.zeros(batch_size, device=device, dtype=dtype)
        
        with torch.no_grad():
            for start in range(0, batch_size, chunk_size):
                end = min(start + chunk_size, batch_size)
                
                # Get current chunk
                probe_chunk = probe_features[start:end]
                gallery_chunk = gallery_features[start:end]
                
                # Force normalization to prevent warnings and issues
                probe_norms = torch.norm(probe_chunk.view(probe_chunk.size(0), -1), p=2, dim=1)
                gallery_norms = torch.norm(gallery_chunk.view(gallery_chunk.size(0), -1), p=2, dim=1)
                
                if (probe_norms < 0.99).any() or (probe_norms > 1.01).any():
                    probe_chunk = F.normalize(probe_chunk, p=2, dim=1)
                
                if (gallery_norms < 0.99).any() or (gallery_norms > 1.01).any():
                    gallery_chunk = F.normalize(gallery_chunk, p=2, dim=1)
                
                # Process each pair individually for memory efficiency
                for i in range(end - start):
                    # Extract single probe and gallery feature
                    single_probe = probe_chunk[i:i+1]  
                    single_gallery = gallery_chunk[i:i+1]
                    
                    # Extract occlusion maps if provided
                    single_probe_occ = None
                    single_gallery_occ = None
                    if probe_occ is not None and gallery_occ is not None:
                        single_probe_occ = probe_occ[start + i:start + i + 1]
                        single_gallery_occ = gallery_occ[start + i:start + i + 1]
                        # Ensure occlusion maps are on the same device/dtype as features
                        single_probe_occ = single_probe_occ.to(device=probe_features.device, dtype=dtype)
                        single_gallery_occ = single_gallery_occ.to(device=probe_features.device, dtype=dtype)
                    
                    # Direct score computation without full matrix calculation
                    hw = self.height * self.width
                    
                    # Reshape features
                    p_fea = single_probe.view(1, self.num_features, hw)
                    g_fea = single_gallery.view(1, self.num_features, hw)
                    
                    # Compute correlation efficiently
                    corr = torch.einsum('p c s, g c r -> p g r s', p_fea, g_fea)
                    
                    # Apply occlusion weighting if provided
                    if single_probe_occ is not None and single_gallery_occ is not None:
                        prob_occ_flat = single_probe_occ.view(1, -1)  # [1, hw]
                        gal_occ_flat = single_gallery_occ.view(1, -1)  # [1, hw]
                        prob_occ_expanded = prob_occ_flat.unsqueeze(1).unsqueeze(3)  # [1, 1, hw, 1]
                        gal_occ_expanded = gal_occ_flat.unsqueeze(0).unsqueeze(2)    # [1, 1, 1, hw]
                        occlusion_weights = prob_occ_expanded * gal_occ_expanded     # [1, 1, hw, hw]
                        corr = corr * occlusion_weights
                    
                    # Max pooling
                    max_r = corr.max(dim=2)[0]  # [1, 1, s]
                    max_s = corr.max(dim=3)[0]  # [1, 1, r]
                    features = torch.cat((max_r, max_s), dim=-1).view(1, 1, hw * 2)  # [1, 1, hw*2]
                    
                    # Forward through BN and FC
                    features = self.bn(features)
                    features = features.view(1, hw * 2)  # [1, hw*2]
                    features = self.fc(features)  # [1, 1]
                    score = self.logit_bn(features).item()  # Single score value
                    
                    # Store result
                    pair_scores[start + i] = score
        
        # Restore previous training state
        self.train(training_state)
        
        return pair_scores 