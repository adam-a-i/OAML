"""
Deterministic diagnostic - controls for dropout randomness.
Compares OLD vs NEW code paths with identical random state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from net import build_model
from head import build_head
import utils


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def simulate_old_forward(model, head, images, labels, cross_entropy_loss):
    """OLD code forward pass - NO occlusion code."""
    # Forward through backbone
    x = model.input_layer(images)
    for layer in model.body:
        x = layer(x)

    # Normalize for QAConv
    x_norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1, keepdim=True).view(x.size(0), 1, 1, 1)
    x_norm = torch.clamp(x_norm, min=1e-8)
    feature_maps = x / x_norm
    feature_maps = feature_maps.clone().detach().requires_grad_(True)

    # AdaFace
    embeddings = model.output_layer(x)
    embeddings, norms = utils.l2_norm(embeddings, axis=1)

    cos_thetas = head(embeddings, norms, labels)
    if isinstance(cos_thetas, tuple):
        cos_thetas, bad_grad = cos_thetas
        labels = labels.clone()
        labels[bad_grad.squeeze(-1)] = -100
    adaface_loss = cross_entropy_loss(cos_thetas, labels)

    return adaface_loss, x, embeddings


def simulate_new_forward_with_occlusion(model, head, images, labels, niqab_images, niqab_masks, cross_entropy_loss):
    """NEW code forward pass - WITH occlusion code in the middle."""
    # Forward through backbone (same as old)
    x = model.input_layer(images)
    for layer in model.body:
        x = layer(x)

    # Normalize for QAConv
    x_norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1, keepdim=True).view(x.size(0), 1, 1, 1)
    x_norm = torch.clamp(x_norm, min=1e-8)
    feature_maps = x / x_norm
    feature_maps = feature_maps.clone().detach().requires_grad_(True)

    # AdaFace (same as old)
    embeddings = model.output_layer(x)
    embeddings, norms = utils.l2_norm(embeddings, axis=1)

    cos_thetas = head(embeddings, norms, labels)
    if isinstance(cos_thetas, tuple):
        cos_thetas, bad_grad = cos_thetas
        labels = labels.clone()
        labels[bad_grad.squeeze(-1)] = -100
    adaface_loss = cross_entropy_loss(cos_thetas, labels)

    # ========== OCCLUSION CODE (this is what's new) ==========
    was_training = model.training
    model.eval()

    with torch.no_grad():
        niqab_x = model.input_layer(niqab_images)
        for layer in model.body:
            niqab_x = layer(niqab_x)

    model.train(was_training)

    niqab_occlusion_maps = model.occlusion_head(niqab_x)
    if niqab_masks.shape[-2:] != niqab_occlusion_maps.shape[-2:]:
        niqab_masks = F.interpolate(niqab_masks, size=niqab_occlusion_maps.shape[-2:],
                                    mode='bilinear', align_corners=False)
    occlusion_loss = F.mse_loss(niqab_occlusion_maps, niqab_masks)
    # ========== END OCCLUSION CODE ==========

    return adaface_loss, occlusion_loss, x, embeddings


def test_with_eval_mode():
    """Test in eval mode to eliminate dropout randomness."""
    print("=" * 80)
    print("TEST 1: Deterministic comparison (eval mode, no dropout)")
    print("=" * 80)

    set_seed(42)

    model = build_model('ir_18')
    model.eval()  # No dropout!

    head = build_head('adaface', embedding_size=512, class_num=100, m=0.4, h=0.333, s=64.0, t_alpha=0.01)
    head.eval()

    cross_entropy_loss = nn.CrossEntropyLoss()

    # Fixed data
    set_seed(42)
    batch_size = 8
    images = torch.randn(batch_size, 3, 112, 112)
    labels = torch.randint(0, 100, (batch_size,))
    niqab_images = torch.randn(batch_size, 3, 112, 112)
    niqab_masks = torch.rand(batch_size, 1, 7, 7)

    # Run OLD
    set_seed(42)
    old_loss, old_x, old_emb = simulate_old_forward(model, head, images, labels.clone(), cross_entropy_loss)

    # Run NEW (without occlusion)
    set_seed(42)
    new_loss, _, new_x, new_emb = simulate_new_forward_with_occlusion(
        model, head, images, labels.clone(), niqab_images, niqab_masks, cross_entropy_loss)

    print(f"\nOLD adaface_loss: {old_loss.item():.6f}")
    print(f"NEW adaface_loss: {new_loss.item():.6f}")
    print(f"Difference: {abs(old_loss.item() - new_loss.item()):.6e}")

    # Compare intermediate tensors
    x_diff = (old_x - new_x).abs().max().item()
    emb_diff = (old_emb - new_emb).abs().max().item()

    print(f"\nBackbone output (x) max diff: {x_diff:.6e}")
    print(f"Embeddings max diff: {emb_diff:.6e}")

    if x_diff < 1e-5 and emb_diff < 1e-5:
        print("\nPASS: OLD and NEW produce identical outputs")
        return True
    else:
        print("\nFAIL: OLD and NEW produce different outputs!")
        return False


def test_occlusion_code_isolation():
    """
    Test if the occlusion code section affects PREVIOUS computations.
    The occlusion code runs AFTER adaface loss is computed, so it should NOT affect it.
    """
    print("\n" + "=" * 80)
    print("TEST 2: Does occlusion code affect PREVIOUS computations?")
    print("=" * 80)

    set_seed(42)
    model = build_model('ir_18')
    model.train()  # Training mode

    head = build_head('adaface', embedding_size=512, class_num=100, m=0.4, h=0.333, s=64.0, t_alpha=0.01)
    cross_entropy_loss = nn.CrossEntropyLoss()

    set_seed(42)
    batch_size = 8
    images = torch.randn(batch_size, 3, 112, 112)
    labels = torch.randint(0, 100, (batch_size,))
    niqab_images = torch.randn(batch_size, 3, 112, 112)
    niqab_masks = torch.rand(batch_size, 1, 7, 7)

    # Compute adaface_loss WITHOUT occlusion code
    set_seed(42)
    x = model.input_layer(images)
    for layer in model.body:
        x = layer(x)
    embeddings = model.output_layer(x)
    embeddings_before, norms = utils.l2_norm(embeddings, axis=1)

    cos_thetas = head(embeddings_before, norms, labels.clone())
    if isinstance(cos_thetas, tuple):
        cos_thetas, _ = cos_thetas
    adaface_loss_before = cross_entropy_loss(cos_thetas, labels.clone())

    print(f"\nAdaFace loss BEFORE occlusion code: {adaface_loss_before.item():.6f}")

    # Now run occlusion code (simulating what happens in training_step)
    was_training = model.training
    model.eval()

    with torch.no_grad():
        niqab_x = model.input_layer(niqab_images)
        for layer in model.body:
            niqab_x = layer(niqab_x)

    model.train(was_training)

    niqab_occlusion_maps = model.occlusion_head(niqab_x)
    occlusion_loss = F.mse_loss(niqab_occlusion_maps, niqab_masks)

    print(f"Occlusion loss: {occlusion_loss.item():.6f}")

    # Now check if adaface_loss tensor value changed
    print(f"AdaFace loss AFTER occlusion code: {adaface_loss_before.item():.6f}")
    print("(Should be identical since it was computed BEFORE occlusion code)")

    # The key question: do the SAME images produce the same embeddings after occlusion code?
    # This tests if model state was properly restored
    set_seed(42)  # Reset seed
    x2 = model.input_layer(images)
    for layer in model.body:
        x2 = layer(x2)
    embeddings2 = model.output_layer(x2)
    embeddings_after, norms2 = utils.l2_norm(embeddings2, axis=1)

    emb_diff = (embeddings_before - embeddings_after).abs().max().item()
    print(f"\nEmbeddings difference after occlusion code: {emb_diff:.6e}")

    if emb_diff < 1e-5:
        print("PASS: Model state properly restored after occlusion code")
        return True
    else:
        print("FAIL: Model state CHANGED by occlusion code!")
        return False


def test_batchnorm_state():
    """
    Deep dive into BatchNorm state changes.
    """
    print("\n" + "=" * 80)
    print("TEST 3: BatchNorm state detailed analysis")
    print("=" * 80)

    model = build_model('ir_18')
    model.train()

    # Get all BatchNorm layers
    bn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            bn_layers.append((name, module))

    print(f"\nFound {len(bn_layers)} BatchNorm layers")

    # Store initial state
    initial_state = {}
    for name, bn in bn_layers:
        initial_state[name] = {
            'running_mean': bn.running_mean.clone(),
            'running_var': bn.running_var.clone(),
            'num_batches_tracked': bn.num_batches_tracked.clone(),
        }

    # Main forward pass
    images = torch.randn(8, 3, 112, 112)
    x = model.input_layer(images)
    for layer in model.body:
        x = layer(x)
    _ = model.output_layer(x)

    # Check how many BN layers changed
    changed_after_main = 0
    for name, bn in bn_layers:
        if not torch.allclose(initial_state[name]['running_mean'], bn.running_mean):
            changed_after_main += 1

    print(f"BN layers with changed running_mean after main forward: {changed_after_main}/{len(bn_layers)}")

    # Store state after main forward
    state_after_main = {}
    for name, bn in bn_layers:
        state_after_main[name] = {
            'running_mean': bn.running_mean.clone(),
            'running_var': bn.running_var.clone(),
        }

    # Occlusion forward (eval mode)
    model.eval()
    niqab_images = torch.randn(8, 3, 112, 112)
    with torch.no_grad():
        niqab_x = model.input_layer(niqab_images)
        for layer in model.body:
            niqab_x = layer(niqab_x)
    model.train()

    # Check which BN layers changed during eval forward
    changed_during_eval = []
    for name, bn in bn_layers:
        if not torch.allclose(state_after_main[name]['running_mean'], bn.running_mean):
            changed_during_eval.append(name)

    print(f"\nBN layers changed during EVAL forward: {len(changed_during_eval)}")
    if changed_during_eval:
        print("WARNING: These BN layers changed during eval mode (BUG!):")
        for name in changed_during_eval[:5]:
            print(f"  - {name}")

    # Occlusion head forward (train mode)
    niqab_occlusion_maps = model.occlusion_head(niqab_x)

    # Check OcclusionHead BN
    occ_bn = model.occlusion_head.bn1
    occ_changed = not torch.allclose(
        torch.zeros_like(occ_bn.running_mean),
        occ_bn.running_mean
    )
    print(f"\nOcclusionHead BN running_mean changed: {occ_changed}")
    print(f"  running_mean sample: {occ_bn.running_mean[:5].tolist()}")

    return len(changed_during_eval) == 0


def test_gradient_determinism():
    """Test if gradients are deterministic."""
    print("\n" + "=" * 80)
    print("TEST 4: Gradient determinism")
    print("=" * 80)

    set_seed(42)
    model = build_model('ir_18')
    model.train()
    head = build_head('adaface', embedding_size=512, class_num=100, m=0.4, h=0.333, s=64.0, t_alpha=0.01)
    cross_entropy_loss = nn.CrossEntropyLoss()

    set_seed(42)
    images = torch.randn(8, 3, 112, 112)
    labels = torch.randint(0, 100, (8,))

    # First run
    set_seed(42)
    model.zero_grad()
    x = model.input_layer(images)
    for layer in model.body:
        x = layer(x)
    embeddings = model.output_layer(x)
    embeddings, norms = utils.l2_norm(embeddings, axis=1)
    cos_thetas = head(embeddings, norms, labels.clone())
    if isinstance(cos_thetas, tuple):
        cos_thetas, _ = cos_thetas
    loss1 = cross_entropy_loss(cos_thetas, labels.clone())
    loss1.backward()
    grad1 = model.input_layer[0].weight.grad.clone()

    # Second run (same seed)
    set_seed(42)
    model.zero_grad()
    x = model.input_layer(images)
    for layer in model.body:
        x = layer(x)
    embeddings = model.output_layer(x)
    embeddings, norms = utils.l2_norm(embeddings, axis=1)
    cos_thetas = head(embeddings, norms, labels.clone())
    if isinstance(cos_thetas, tuple):
        cos_thetas, _ = cos_thetas
    loss2 = cross_entropy_loss(cos_thetas, labels.clone())
    loss2.backward()
    grad2 = model.input_layer[0].weight.grad.clone()

    grad_diff = (grad1 - grad2).abs().max().item()
    print(f"\nGradient difference between identical runs: {grad_diff:.6e}")

    if grad_diff < 1e-6:
        print("PASS: Gradients are deterministic")
        return True
    else:
        print("FAIL: Gradients are NOT deterministic!")
        return False


def main():
    print("=" * 80)
    print("DETERMINISTIC DIAGNOSTIC")
    print("=" * 80)

    results = {}
    results['eval_mode_test'] = test_with_eval_mode()
    results['occlusion_isolation'] = test_occlusion_code_isolation()
    results['batchnorm_state'] = test_batchnorm_state()
    results['gradient_determinism'] = test_gradient_determinism()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for test, passed in results.items():
        print(f"  {test}: {'PASS' if passed else 'FAIL'}")

    all_passed = all(results.values())
    if all_passed:
        print("\nAll tests passed - occlusion code is properly isolated")
        print("The issue might be elsewhere (data, hyperparams, etc.)")
    else:
        print("\nSome tests failed - there's a bug in occlusion code isolation!")


if __name__ == '__main__':
    main()
