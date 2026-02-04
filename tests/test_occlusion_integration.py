"""
Comprehensive test script to verify occlusion maps are properly integrated into QAConv.

This script tests ALL code paths to ensure:
1. Occlusion maps are passed correctly through all components
2. No NaN values are produced
3. BN statistics are consistent (all paths use occlusion weighting)
4. The occlusion weighting actually affects the output

Run with:
    python tests/test_occlusion_integration.py
"""

import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qaconv import QAConv
from pairwise_matching_loss import PairwiseMatchingLoss
from softmax_triplet_loss import SoftmaxTripletLoss


def print_test(name, passed, details=""):
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")
    if details and not passed:
        print(f"       {details}")
    return passed


def test_compute_similarity_with_occlusion():
    """Test _compute_similarity_batch_with_occlusion directly."""
    print("\n" + "="*60)
    print("TEST 1: _compute_similarity_batch_with_occlusion")
    print("="*60)

    qaconv = QAConv(num_features=512, height=7, width=7, num_classes=100)
    qaconv.eval()

    batch_p, batch_g = 4, 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    qaconv = qaconv.to(device)

    # Create test inputs
    prob_fea = torch.randn(batch_p, 512, 7, 7, device=device)
    gal_fea = torch.randn(batch_g, 512, 7, 7, device=device)
    prob_fea = F.normalize(prob_fea, p=2, dim=1)
    gal_fea = F.normalize(gal_fea, p=2, dim=1)

    # Create occlusion maps (values in [0, 1])
    prob_occ = torch.rand(batch_p, 1, 7, 7, device=device)
    gal_occ = torch.rand(batch_g, 1, 7, 7, device=device)

    all_passed = True

    # Test 1a: Both occlusion maps provided
    with torch.no_grad():
        scores = qaconv._compute_similarity_batch_with_occlusion(prob_fea, gal_fea, prob_occ, gal_occ)
    passed = not torch.isnan(scores).any() and scores.shape == (batch_p, batch_g)
    all_passed &= print_test("Both occlusion maps provided", passed,
                              f"shape={scores.shape}, has_nan={torch.isnan(scores).any()}")

    # Test 1b: Only probe occlusion (gallery assumed clean)
    with torch.no_grad():
        scores_probe_only = qaconv._compute_similarity_batch_with_occlusion(prob_fea, gal_fea, prob_occ, None)
    passed = not torch.isnan(scores_probe_only).any() and scores_probe_only.shape == (batch_p, batch_g)
    all_passed &= print_test("Probe occlusion only (gallery=None)", passed,
                              f"shape={scores_probe_only.shape}, has_nan={torch.isnan(scores_probe_only).any()}")

    # Test 1c: No occlusion maps (fallback to standard)
    with torch.no_grad():
        scores_no_occ = qaconv._compute_similarity_batch_with_occlusion(prob_fea, gal_fea, None, None)
    passed = not torch.isnan(scores_no_occ).any() and scores_no_occ.shape == (batch_p, batch_g)
    all_passed &= print_test("No occlusion maps (fallback)", passed,
                              f"shape={scores_no_occ.shape}, has_nan={torch.isnan(scores_no_occ).any()}")

    # Test 1d: Occlusion actually affects output
    # With all-zeros occlusion (fully occluded), scores should be different from all-ones
    prob_occ_zeros = torch.zeros(batch_p, 1, 7, 7, device=device)
    prob_occ_ones = torch.ones(batch_p, 1, 7, 7, device=device)
    gal_occ_ones = torch.ones(batch_g, 1, 7, 7, device=device)

    with torch.no_grad():
        scores_occluded = qaconv._compute_similarity_batch_with_occlusion(prob_fea, gal_fea, prob_occ_zeros, gal_occ_ones)
        scores_visible = qaconv._compute_similarity_batch_with_occlusion(prob_fea, gal_fea, prob_occ_ones, gal_occ_ones)

    # Occluded scores should be much lower (closer to 0 after weighting)
    passed = (scores_visible.abs().mean() > scores_occluded.abs().mean() * 2)
    all_passed &= print_test("Occlusion weighting affects output", passed,
                              f"visible_mean={scores_visible.abs().mean():.4f}, occluded_mean={scores_occluded.abs().mean():.4f}")

    return all_passed


def test_pairwise_matching_loss():
    """Test PairwiseMatchingLoss with occlusion maps."""
    print("\n" + "="*60)
    print("TEST 2: PairwiseMatchingLoss with occlusion")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    qaconv = QAConv(num_features=512, height=7, width=7, num_classes=100).to(device)
    loss_fn = PairwiseMatchingLoss(qaconv).to(device)
    loss_fn.train()

    batch_size = 8
    features = torch.randn(batch_size, 512, 7, 7, device=device)
    features = F.normalize(features, p=2, dim=1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device=device)
    occlusion_maps = torch.rand(batch_size, 1, 7, 7, device=device)

    all_passed = True

    # Test 2a: Forward with occlusion maps
    loss, acc = loss_fn(features, labels, occlusion_maps)
    passed = not torch.isnan(loss).any() and not torch.isnan(acc).any()
    all_passed &= print_test("Forward with occlusion maps", passed,
                              f"loss={loss.mean():.4f}, has_nan={torch.isnan(loss).any()}")

    # Test 2b: Forward without occlusion maps (backward compat)
    loss_no_occ, acc_no_occ = loss_fn(features, labels, None)
    passed = not torch.isnan(loss_no_occ).any()
    all_passed &= print_test("Forward without occlusion (backward compat)", passed,
                              f"loss={loss_no_occ.mean():.4f}")

    # Test 2c: Occlusion affects the loss
    # With heavy occlusion, loss should be different
    occ_heavy = torch.zeros(batch_size, 1, 7, 7, device=device)  # Fully occluded
    occ_light = torch.ones(batch_size, 1, 7, 7, device=device)   # Fully visible
    loss_heavy, _ = loss_fn(features, labels, occ_heavy)
    loss_light, _ = loss_fn(features, labels, occ_light)
    passed = abs(loss_heavy.mean() - loss_light.mean()) > 0.01
    all_passed &= print_test("Occlusion affects loss value", passed,
                              f"heavy={loss_heavy.mean():.4f}, light={loss_light.mean():.4f}")

    return all_passed


def test_softmax_triplet_loss():
    """Test SoftmaxTripletLoss with occlusion maps."""
    print("\n" + "="*60)
    print("TEST 3: SoftmaxTripletLoss with occlusion")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 100
    qaconv = QAConv(num_features=512, height=7, width=7, num_classes=num_classes).to(device)
    loss_fn = SoftmaxTripletLoss(qaconv).to(device)
    loss_fn.train()

    # Initialize class embeddings
    qaconv.class_embed = torch.nn.Parameter(
        torch.randn(num_classes, 512, 7, 7, device=device) / 512**0.5
    )
    qaconv.compute_class_neighbors()

    batch_size = 8
    features = torch.randn(batch_size, 512, 7, 7, device=device)
    features = F.normalize(features, p=2, dim=1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device=device)
    occlusion_maps = torch.rand(batch_size, 1, 7, 7, device=device)

    all_passed = True

    # Test 3a: Forward with occlusion maps
    cls_loss, triplet_loss, total_loss, cls_acc, triplet_acc = loss_fn(features, labels, occlusion_maps)
    passed = not torch.isnan(cls_loss).any() and not torch.isnan(triplet_loss).any()
    all_passed &= print_test("Forward with occlusion maps", passed,
                              f"cls_loss={cls_loss.mean():.4f}, triplet_loss={triplet_loss.mean():.4f}")

    # Test 3b: Forward without occlusion maps (backward compat)
    cls_loss_no, triplet_loss_no, _, _, _ = loss_fn(features, labels, None)
    passed = not torch.isnan(cls_loss_no).any() and not torch.isnan(triplet_loss_no).any()
    all_passed &= print_test("Forward without occlusion (backward compat)", passed,
                              f"cls_loss={cls_loss_no.mean():.4f}")

    # Test 3c: Occlusion affects the loss
    occ_heavy = torch.zeros(batch_size, 1, 7, 7, device=device)
    occ_light = torch.ones(batch_size, 1, 7, 7, device=device)
    cls_heavy, trip_heavy, _, _, _ = loss_fn(features, labels, occ_heavy)
    cls_light, trip_light, _, _, _ = loss_fn(features, labels, occ_light)
    passed = abs(cls_heavy.mean() - cls_light.mean()) > 0.01 or abs(trip_heavy.mean() - trip_light.mean()) > 0.01
    all_passed &= print_test("Occlusion affects loss value", passed,
                              f"heavy_cls={cls_heavy.mean():.4f}, light_cls={cls_light.mean():.4f}")

    return all_passed


def test_qaconv_forward_modes():
    """Test QAConv forward with different modes."""
    print("\n" + "="*60)
    print("TEST 4: QAConv forward modes with occlusion")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 100
    qaconv = QAConv(num_features=512, height=7, width=7, num_classes=num_classes).to(device)

    # Initialize class embeddings
    qaconv.class_embed = torch.nn.Parameter(
        torch.randn(num_classes, 512, 7, 7, device=device) / 512**0.5
    )
    qaconv.compute_class_neighbors()

    batch_size = 8
    features = torch.randn(batch_size, 512, 7, 7, device=device)
    features = F.normalize(features, p=2, dim=1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device=device)
    occlusion_maps = torch.rand(batch_size, 1, 7, 7, device=device)

    all_passed = True

    # Test 4a: Training mode with class embeddings and labels
    qaconv.train()
    with torch.no_grad():
        logits = qaconv(features, labels=labels, prob_occ=occlusion_maps, gal_occ=occlusion_maps)
    passed = logits is not None and not torch.isnan(logits).any() and logits.shape == (batch_size, num_classes)
    all_passed &= print_test("Training mode with class embeddings + labels", passed,
                              f"shape={logits.shape if logits is not None else None}")

    # Test 4b: Training mode with class embeddings, no labels
    with torch.no_grad():
        logits_no_labels = qaconv(features, prob_occ=occlusion_maps, gal_occ=occlusion_maps)
    passed = logits_no_labels is not None and not torch.isnan(logits_no_labels).any()
    all_passed &= print_test("Training mode with class embeddings, no labels", passed)

    # Test 4c: "same" mode (self-matching)
    with torch.no_grad():
        scores_same = qaconv(features, "same", prob_occ=occlusion_maps, gal_occ=occlusion_maps)
    passed = scores_same is not None and not torch.isnan(scores_same).any() and scores_same.shape == (batch_size, batch_size)
    all_passed &= print_test("'same' mode (self-matching)", passed,
                              f"shape={scores_same.shape if scores_same is not None else None}")

    # Test 4d: Explicit gallery features
    qaconv.eval()
    gallery = torch.randn(4, 512, 7, 7, device=device)
    gallery = F.normalize(gallery, p=2, dim=1)
    gallery_occ = torch.rand(4, 1, 7, 7, device=device)
    with torch.no_grad():
        scores_gal = qaconv(features, gallery, prob_occ=occlusion_maps, gal_occ=gallery_occ)
    passed = scores_gal is not None and not torch.isnan(scores_gal).any() and scores_gal.shape == (batch_size, 4)
    all_passed &= print_test("Explicit gallery features", passed,
                              f"shape={scores_gal.shape if scores_gal is not None else None}")

    # Test 4e: make_kernel + forward (PairwiseMatchingLoss path)
    qaconv.train()
    qaconv.make_kernel(features)
    with torch.no_grad():
        scores_kernel = qaconv(features, prob_occ=occlusion_maps, gal_occ=occlusion_maps)
    passed = scores_kernel is not None and not torch.isnan(scores_kernel).any() and scores_kernel.shape == (batch_size, batch_size)
    all_passed &= print_test("make_kernel + forward path", passed,
                              f"shape={scores_kernel.shape if scores_kernel is not None else None}")

    return all_passed


def test_match_pairs():
    """Test match_pairs method with occlusion."""
    print("\n" + "="*60)
    print("TEST 5: match_pairs with occlusion")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    qaconv = QAConv(num_features=512, height=7, width=7, num_classes=100).to(device)
    qaconv.eval()

    batch_size = 16
    probe = torch.randn(batch_size, 512, 7, 7, device=device)
    gallery = torch.randn(batch_size, 512, 7, 7, device=device)
    probe = F.normalize(probe, p=2, dim=1)
    gallery = F.normalize(gallery, p=2, dim=1)
    probe_occ = torch.rand(batch_size, 1, 7, 7, device=device)
    gallery_occ = torch.rand(batch_size, 1, 7, 7, device=device)

    all_passed = True

    # Test 5a: match_pairs with occlusion
    with torch.no_grad():
        scores = qaconv.match_pairs(probe, gallery, probe_occ, gallery_occ)
    passed = not torch.isnan(scores).any() and scores.shape == (batch_size,)
    all_passed &= print_test("match_pairs with occlusion", passed,
                              f"shape={scores.shape}, has_nan={torch.isnan(scores).any()}")

    # Test 5b: match_pairs without occlusion (backward compat)
    with torch.no_grad():
        scores_no_occ = qaconv.match_pairs(probe, gallery, None, None)
    passed = not torch.isnan(scores_no_occ).any() and scores_no_occ.shape == (batch_size,)
    all_passed &= print_test("match_pairs without occlusion", passed)

    # Test 5c: Occlusion affects match_pairs output
    occ_heavy = torch.zeros(batch_size, 1, 7, 7, device=device)
    occ_light = torch.ones(batch_size, 1, 7, 7, device=device)
    with torch.no_grad():
        scores_heavy = qaconv.match_pairs(probe, gallery, occ_heavy, occ_light)
        scores_light = qaconv.match_pairs(probe, gallery, occ_light, occ_light)
    passed = (scores_heavy - scores_light).abs().mean() > 0.1
    all_passed &= print_test("Occlusion affects match_pairs output", passed,
                              f"diff={abs(scores_heavy.mean() - scores_light.mean()):.4f}")

    return all_passed


def test_bn_statistics_consistency():
    """Test that BN statistics are consistent across all paths."""
    print("\n" + "="*60)
    print("TEST 6: BN statistics consistency")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 100
    qaconv = QAConv(num_features=512, height=7, width=7, num_classes=num_classes).to(device)

    # Initialize class embeddings
    qaconv.class_embed = torch.nn.Parameter(
        torch.randn(num_classes, 512, 7, 7, device=device) / 512**0.5
    )
    qaconv.compute_class_neighbors()

    qaconv.train()
    qaconv.reset_running_stats()

    batch_size = 32
    features = torch.randn(batch_size, 512, 7, 7, device=device)
    features = F.normalize(features, p=2, dim=1)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    occlusion_maps = torch.rand(batch_size, 1, 7, 7, device=device) * 0.5 + 0.25  # Range [0.25, 0.75]

    # Record initial BN stats
    initial_mean = qaconv.bn.running_mean.clone() if qaconv.bn.running_mean is not None else None
    initial_var = qaconv.bn.running_var.clone() if qaconv.bn.running_var is not None else None

    all_passed = True

    # Run multiple forward passes with occlusion
    for _ in range(10):
        # Class embedding path
        _ = qaconv(features, labels=labels, prob_occ=occlusion_maps, gal_occ=occlusion_maps)

        # Self-matching path
        qaconv.make_kernel(features)
        _ = qaconv(features, prob_occ=occlusion_maps, gal_occ=occlusion_maps)

        # "same" path
        _ = qaconv(features, "same", prob_occ=occlusion_maps, gal_occ=occlusion_maps)

    # Check BN stats updated
    final_mean = qaconv.bn.running_mean
    final_var = qaconv.bn.running_var

    passed = final_mean is not None and not torch.isnan(final_mean).any()
    all_passed &= print_test("BN running_mean valid after training", passed,
                              f"mean={final_mean.item() if final_mean is not None else 'None'}")

    passed = final_var is not None and not torch.isnan(final_var).any() and final_var.item() > 0
    all_passed &= print_test("BN running_var valid after training", passed,
                              f"var={final_var.item() if final_var is not None else 'None'}")

    # The key test: BN stats should reflect occlusion-weighted values
    # With occlusion weighting, max-pooled values should be smaller than without
    # So the running_mean should be less than ~0.3-0.4 (typical without occlusion)
    # This is a sanity check that occlusion IS being applied
    passed = final_mean is not None and final_mean.item() < 0.3
    all_passed &= print_test("BN mean reflects occlusion weighting (< 0.3)", passed,
                              f"mean={final_mean.item() if final_mean is not None else 'None'}")

    return all_passed


def test_gradient_flow():
    """Test that gradients flow correctly through occlusion-weighted paths."""
    print("\n" + "="*60)
    print("TEST 7: Gradient flow through occlusion weighting")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    qaconv = QAConv(num_features=512, height=7, width=7, num_classes=100).to(device)
    loss_fn = PairwiseMatchingLoss(qaconv).to(device)
    loss_fn.train()

    batch_size = 8
    features = torch.randn(batch_size, 512, 7, 7, device=device, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device=device)
    occlusion_maps = torch.rand(batch_size, 1, 7, 7, device=device)

    all_passed = True

    # Test gradient flow
    loss, acc = loss_fn(features, labels, occlusion_maps)
    print(f"\n  Loss shape: {loss.shape}")
    print(f"  Loss values: {loss}")
    print(f"  Loss mean: {loss.mean().item():.6f}")

    total_loss = loss.mean()
    total_loss.backward()

    # Check QAConv parameters have gradients - with detailed logging
    print("\n  QAConv parameter gradients:")
    params_with_grad = 0
    params_without_grad = 0
    params_with_nan = 0

    for name, p in qaconv.named_parameters():
        if p.requires_grad:
            if p.grad is not None:
                has_nan = torch.isnan(p.grad).any().item()
                grad_norm = p.grad.norm().item()
                grad_mean = p.grad.mean().item()
                grad_std = p.grad.std().item() if p.grad.numel() > 1 else 0
                print(f"    {name}: shape={tuple(p.shape)}, grad_norm={grad_norm:.6f}, "
                      f"grad_mean={grad_mean:.6f}, grad_std={grad_std:.6f}, has_nan={has_nan}")
                if has_nan:
                    params_with_nan += 1
                else:
                    params_with_grad += 1
            else:
                print(f"    {name}: shape={tuple(p.shape)}, grad=None")
                params_without_grad += 1

    print(f"\n  Summary: {params_with_grad} params with valid grads, "
          f"{params_without_grad} params without grads, {params_with_nan} params with NaN grads")

    # The test passes if at least the core parameters (bn, fc, logit_bn) have gradients
    # class_embed may not have gradients in this test since we're using PairwiseMatchingLoss
    core_params = ['bn.weight', 'bn.bias', 'fc.weight', 'fc.bias', 'logit_bn.weight', 'logit_bn.bias']
    core_have_grads = True
    for name, p in qaconv.named_parameters():
        if name in core_params:
            if p.grad is None or torch.isnan(p.grad).any():
                core_have_grads = False
                print(f"  WARNING: Core parameter {name} missing valid gradient!")

    all_passed &= print_test("Core QAConv parameters have valid gradients", core_have_grads)

    # Note: Input features should NOT have gradients because PairwiseMatchingLoss
    # intentionally detaches them (line 52: feature.detach().clone().requires_grad_(True))
    # This is by design - QAConv is trained separately from backbone
    features_have_grad = features.grad is not None
    print(f"\n  Input features gradient: exists={features.grad is not None}")
    print(f"  (Expected: False - PairwiseMatchingLoss detaches features by design)")
    all_passed &= print_test("Input features detached (expected behavior)", not features_have_grad)

    return all_passed


def test_end_to_end_training_simulation():
    """Simulate a training step to ensure everything works together."""
    print("\n" + "="*60)
    print("TEST 8: End-to-end training simulation")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 100

    # Create all components
    qaconv = QAConv(num_features=512, height=7, width=7, num_classes=num_classes).to(device)
    qaconv.class_embed = torch.nn.Parameter(
        torch.randn(num_classes, 512, 7, 7, device=device) / 512**0.5
    )
    qaconv.compute_class_neighbors()

    pairwise_loss = PairwiseMatchingLoss(qaconv).to(device)
    triplet_loss = SoftmaxTripletLoss(qaconv).to(device)

    # Simulate training
    qaconv.train()
    pairwise_loss.train()
    triplet_loss.train()

    optimizer = torch.optim.SGD(qaconv.parameters(), lr=0.01)

    all_passed = True

    # Run multiple training iterations
    for iteration in range(5):
        optimizer.zero_grad()

        # Generate fake batch
        batch_size = 16
        features = torch.randn(batch_size, 512, 7, 7, device=device)
        features = F.normalize(features, p=2, dim=1).detach().clone().requires_grad_(True)
        labels = torch.randint(0, num_classes, (batch_size,), device=device)
        occlusion_maps = torch.rand(batch_size, 1, 7, 7, device=device)

        # Compute losses (same as train_val.py)
        pw_loss, pw_acc = pairwise_loss(features, labels, occlusion_maps)
        cls_loss, trip_loss, _, cls_acc, trip_acc = triplet_loss(features, labels, occlusion_maps)

        total_loss = pw_loss.mean() + trip_loss.mean()

        # Check for NaN
        if torch.isnan(total_loss):
            all_passed &= print_test(f"Iteration {iteration}: No NaN loss", False,
                                      f"loss={total_loss.item()}")
            break

        total_loss.backward()
        optimizer.step()

    passed = not torch.isnan(total_loss)
    all_passed &= print_test("Training simulation completed without NaN", passed,
                              f"final_loss={total_loss.item():.4f}")

    # Check BN stats are reasonable
    bn_mean = qaconv.bn.running_mean.item() if qaconv.bn.running_mean is not None else None
    passed = bn_mean is not None and 0 < bn_mean < 0.5
    all_passed &= print_test("BN running_mean in expected range", passed,
                              f"mean={bn_mean}")

    return all_passed


def main():
    print("="*60)
    print("OCCLUSION INTEGRATION TEST SUITE")
    print("="*60)

    results = []

    results.append(("_compute_similarity_batch_with_occlusion", test_compute_similarity_with_occlusion()))
    results.append(("PairwiseMatchingLoss", test_pairwise_matching_loss()))
    results.append(("SoftmaxTripletLoss", test_softmax_triplet_loss()))
    results.append(("QAConv forward modes", test_qaconv_forward_modes()))
    results.append(("match_pairs", test_match_pairs()))
    results.append(("BN statistics consistency", test_bn_statistics_consistency()))
    results.append(("Gradient flow", test_gradient_flow()))
    results.append(("End-to-end training simulation", test_end_to_end_training_simulation()))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}")
        all_passed &= passed

    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED - Occlusion integration is complete!")
    else:
        print("SOME TESTS FAILED - Please review the failures above")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
