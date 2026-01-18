"""
Test to ensure QAConv losses do NOT backprop into OcclusionHead.

Run:
    python tests/test_occlusion_qaconv_grad_block.py
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from net import build_model
from pairwise_matching_loss import PairwiseMatchingLoss
from softmax_triplet_loss import SoftmaxTripletLoss


def _grad_is_zero_or_none(params):
    for p in params:
        if p.grad is None:
            continue
        if not torch.allclose(p.grad, torch.zeros_like(p.grad)):
            return False
    return True


def _grad_has_signal(params):
    for p in params:
        if p.grad is None:
            continue
        if torch.any(p.grad.abs() > 0):
            return True
    return False


def test_no_qaconv_grad_to_occlusion_head():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model("ir_18").to(device)
    model.train()

    qaconv = model.qaconv.to(device)
    pairwise_loss_fn = PairwiseMatchingLoss(qaconv).to(device)
    triplet_loss_fn = SoftmaxTripletLoss(qaconv).to(device)

    batch_size = 6
    images = torch.randn(batch_size, 3, 112, 112, device=device)
    labels = torch.tensor([0, 0, 1, 1, 2, 2], device=device)

    # Forward backbone
    x = model.input_layer(images)
    for layer in model.body:
        x = layer(x)
    feature_maps = F.normalize(x, p=2, dim=1)

    # Occlusion maps computed, then DETACHED for QAConv
    occlusion_maps = model.occlusion_head(x.detach())
    occlusion_maps_for_qaconv = occlusion_maps.detach()

    # Initialize QAConv class embeddings for triplet path
    num_classes = int(labels.max().item() + 1)
    qaconv.class_embed = torch.nn.Parameter(
        torch.randn(num_classes, qaconv.num_features, qaconv.height, qaconv.width, device=device)
        / (qaconv.num_features ** 0.5)
    )
    # Ensure k_nearest is valid for the small test class count
    qaconv.k_nearest = min(qaconv.k_nearest, num_classes - 1)
    qaconv.compute_class_neighbors()

    model.zero_grad()
    qaconv.zero_grad()
    pairwise_loss_fn.zero_grad()
    triplet_loss_fn.zero_grad()

    pairwise_loss, _ = pairwise_loss_fn(feature_maps, labels, occlusion_maps_for_qaconv)
    cls_loss, triplet_loss, _, _, _ = triplet_loss_fn(feature_maps, labels, occlusion_maps_for_qaconv)

    qaconv_loss = pairwise_loss.mean() + triplet_loss.mean() + cls_loss.mean()
    qaconv_loss.backward()

    assert _grad_is_zero_or_none(model.occlusion_head.parameters()), (
        "FAIL: QAConv loss is still backpropagating into OcclusionHead."
    )
    print("[PASS] QAConv loss does NOT backprop into OcclusionHead.")

    # Control: occlusion loss should still backprop into OcclusionHead
    model.zero_grad()
    niqab_x = x.detach().requires_grad_(True)
    occ_pred = model.occlusion_head(niqab_x)
    occ_loss = F.mse_loss(occ_pred, torch.zeros_like(occ_pred))
    occ_loss.backward()

    assert _grad_has_signal(model.occlusion_head.parameters()), (
        "FAIL: Occlusion loss did not backprop into OcclusionHead."
    )
    print("[PASS] Occlusion loss backprop into OcclusionHead works.")


if __name__ == "__main__":
    test_no_qaconv_grad_to_occlusion_head()
