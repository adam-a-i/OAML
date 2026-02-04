"""Class for the hard triplet loss
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
from torch import nn


class SoftmaxTripletLoss(Module):
    def __init__(self, matcher, margin=1.0, triplet_weight=1.0):
        """
        Inputs:
            matcher: a class for matching pairs of images
            margin: margin parameter for the triplet loss
        """
        super(SoftmaxTripletLoss, self).__init__()
        self.matcher = matcher
        self.margin = margin
        self.triplet_weight = triplet_weight
        self.cls_loss = nn.CrossEntropyLoss(reduction='none')
        self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction='none')

    def reset_running_stats(self):
        self.matcher.reset_running_stats()

    def reset_parameters(self):
        self.matcher.reset_parameters()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input.dim()))

    def forward(self, feature, target, occlusion_maps=None):
        self._check_input_dim(feature)

        # Pass the target labels to QAConv with occlusion maps for occlusion-aware matching
        logits = self.matcher(feature, labels=target, prob_occ=occlusion_maps, gal_occ=occlusion_maps)

        # Handle NaN in logits before computing cls_loss
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)

        cls_loss = self.cls_loss(logits, target)

        # Handle NaN in cls_loss
        if torch.isnan(cls_loss).any() or torch.isinf(cls_loss).any():
            cls_loss = torch.nan_to_num(cls_loss, nan=0.0, posinf=0.0, neginf=0.0)

        score = self.matcher(feature, "same", prob_occ=occlusion_maps, gal_occ=occlusion_maps)  # [b, b]

        # Handle NaN/Inf in scores before computing triplet loss
        if torch.isnan(score).any() or torch.isinf(score).any():
            score = torch.nan_to_num(score, nan=0.0, posinf=100.0, neginf=-100.0)

        target1 = target.unsqueeze(1)
        mask = (target1 == target1.t())
        pair_labels = mask.float()

        # Use 1e9 instead of 1e15 for numerical stability (avoids overflow in mixed precision)
        INF_MASK = 1e9

        # Clamp scores to prevent extreme values before masking
        score_clamped = torch.clamp(score, min=-100, max=100)

        min_pos = torch.min(score_clamped * pair_labels +
                (1 - pair_labels + torch.eye(score.size(0), device=score.device)) * INF_MASK, dim=1)[0]
        max_neg = torch.max(score_clamped * (1 - pair_labels) - pair_labels * INF_MASK, dim=1)[0]

        # Handle edge cases where min_pos or max_neg are still extreme (no valid pairs)
        # This can happen if batch has all same class or all different classes
        valid_pos = min_pos < (INF_MASK / 2)  # Valid if not masked value
        valid_neg = max_neg > (-INF_MASK / 2)  # Valid if not masked value

        # Replace invalid values with reasonable defaults to avoid NaN
        min_pos = torch.where(valid_pos, min_pos, torch.zeros_like(min_pos))
        max_neg = torch.where(valid_neg, max_neg, torch.zeros_like(max_neg))

        # Compute ranking hinge loss
        triplet_loss = self.ranking_loss(min_pos, max_neg, torch.ones_like(target))

        # Additional NaN protection
        if torch.isnan(triplet_loss).any() or torch.isinf(triplet_loss).any():
            triplet_loss = torch.nan_to_num(triplet_loss, nan=0.0, posinf=0.0, neginf=0.0)
        loss = cls_loss + self.triplet_weight * triplet_loss.mean()

        with torch.no_grad():
            cls_acc = (logits.argmax(dim=1) == target).float()
            triplet_acc = (min_pos >= max_neg).float()

        return cls_loss, triplet_loss, loss, cls_acc, triplet_acc
