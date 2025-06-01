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

    def forward(self, feature, target):
        self._check_input_dim(feature)

        # Pass the target labels to QAConv
        logits = self.matcher(feature, labels=target)
        cls_loss = self.cls_loss(logits, target)

        score = self.matcher(feature, "same")  # [b, b]

        target1 = target.unsqueeze(1)
        mask = (target1 == target1.t())
        pair_labels = mask.float()

        min_pos = torch.min(score * pair_labels + 
                (1 - pair_labels + torch.eye(score.size(0), device=score.device)) * 1e15, dim=1)[0]
        max_neg = torch.max(score * (1 - pair_labels) - pair_labels * 1e15, dim=1)[0]

        # Compute ranking hinge loss
        triplet_loss = self.ranking_loss(min_pos, max_neg, torch.ones_like(target))
        loss = cls_loss + self.triplet_weight * triplet_loss.mean()

        with torch.no_grad():
            cls_acc = (logits.argmax(dim=1) == target).float()
            triplet_acc = (min_pos >= max_neg).float()

        return cls_loss, triplet_loss, loss, cls_acc, triplet_acc
