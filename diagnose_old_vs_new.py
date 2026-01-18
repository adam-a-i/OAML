"""
Diagnostic script that simulates OLD (e0f384a) vs NEW training_step
to find exactly where they diverge.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from head import build_head
import utils


def build_old_model():
    """Build model as it was in e0f384a (without OcclusionHead)."""
    # We can't easily build the old model, so we'll use new model
    # but track what SHOULD happen in old code
    from net import build_model
    return build_model('ir_18')


def simulate_old_training_step(model, head, images, labels, qaconv_criterion=None):
    """
    Simulate the OLD training_step from commit e0f384a.
    Returns: total_loss, adaface_loss, qaconv_loss, and gradient info
    """
    device = images.device
    cross_entropy_loss = nn.CrossEntropyLoss()

    # Forward through backbone (exactly as old code)
    x = model.input_layer(images)
    for layer in model.body:
        x = layer(x)

    # Normalize for QAConv
    x_norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1, keepdim=True).view(x.size(0), 1, 1, 1)
    x_norm = torch.clamp(x_norm, min=1e-8)
    feature_maps = x / x_norm

    # DETACH for QAConv (old code had this)
    feature_maps = feature_maps.clone().detach().requires_grad_(True)

    # AdaFace embeddings
    embeddings = model.output_layer(x)
    embeddings, norms = utils.l2_norm(embeddings, axis=1)

    # AdaFace loss
    cos_thetas = head(embeddings, norms, labels)
    if isinstance(cos_thetas, tuple):
        cos_thetas, bad_grad = cos_thetas
        labels = labels.clone()
        labels[bad_grad.squeeze(-1)] = -100
    adaface_loss = cross_entropy_loss(cos_thetas, labels)

    # QAConv loss (simplified - just use feature_maps directly)
    qaconv_loss = torch.tensor(0.0, device=device)
    # In real code, qaconv_criterion would be called here

    # OLD total loss: no occlusion
    adaface_weight = 0.1
    qaconv_weight = 0.9
    total_loss = adaface_weight * adaface_loss + qaconv_weight * qaconv_loss

    return {
        'total_loss': total_loss,
        'adaface_loss': adaface_loss,
        'qaconv_loss': qaconv_loss,
        'x': x,  # backbone output before detach
        'feature_maps': feature_maps,
        'embeddings': embeddings,
    }


def simulate_new_training_step(model, head, images, labels, niqab_images=None, niqab_masks=None):
    """
    Simulate the NEW training_step with occlusion layer.
    Returns: total_loss, adaface_loss, qaconv_loss, occlusion_loss, and gradient info
    """
    device = images.device
    cross_entropy_loss = nn.CrossEntropyLoss()

    # Forward through backbone (same as old)
    x = model.input_layer(images)
    for layer in model.body:
        x = layer(x)

    # Normalize for QAConv
    x_norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1, keepdim=True).view(x.size(0), 1, 1, 1)
    x_norm = torch.clamp(x_norm, min=1e-8)
    feature_maps = x / x_norm

    # DETACH for QAConv (same as old)
    feature_maps = feature_maps.clone().detach().requires_grad_(True)

    # AdaFace embeddings (same as old)
    embeddings = model.output_layer(x)
    embeddings, norms = utils.l2_norm(embeddings, axis=1)

    # AdaFace loss (same as old)
    cos_thetas = head(embeddings, norms, labels)
    if isinstance(cos_thetas, tuple):
        cos_thetas, bad_grad = cos_thetas
        labels = labels.clone()
        labels[bad_grad.squeeze(-1)] = -100
    adaface_loss = cross_entropy_loss(cos_thetas, labels)

    # ========== NEW: OCCLUSION LOSS ==========
    occlusion_loss = torch.tensor(0.0, device=device)

    if niqab_images is not None and niqab_masks is not None:
        # Switch to eval mode
        was_training = model.training
        model.eval()

        # Forward niqab with no_grad
        with torch.no_grad():
            niqab_x = model.input_layer(niqab_images)
            for layer in model.body:
                niqab_x = layer(niqab_x)

        # Restore training mode
        model.train(was_training)

        # Occlusion head forward
        niqab_occlusion_maps = model.occlusion_head(niqab_x)

        # MSE loss
        if niqab_masks.shape[-2:] != niqab_occlusion_maps.shape[-2:]:
            niqab_masks = F.interpolate(niqab_masks, size=niqab_occlusion_maps.shape[-2:],
                                        mode='bilinear', align_corners=False)
        occlusion_loss = F.mse_loss(niqab_occlusion_maps, niqab_masks)

    # QAConv loss (simplified)
    qaconv_loss = torch.tensor(0.0, device=device)

    # NEW total loss: includes occlusion
    adaface_weight = 0.1
    qaconv_weight = 0.9
    occlusion_weight = 0.1
    total_loss = adaface_weight * adaface_loss + qaconv_weight * qaconv_loss + occlusion_weight * occlusion_loss

    return {
        'total_loss': total_loss,
        'adaface_loss': adaface_loss,
        'qaconv_loss': qaconv_loss,
        'occlusion_loss': occlusion_loss,
        'x': x,
        'feature_maps': feature_maps,
        'embeddings': embeddings,
    }


def compare_training_steps():
    """Compare old vs new training step outputs."""
    print("=" * 80)
    print("COMPARING OLD vs NEW TRAINING STEP")
    print("=" * 80)

    # Build model and head
    model = build_old_model()
    model.train()
    head = build_head('adaface', embedding_size=512, class_num=100, m=0.4, h=0.333, s=64.0, t_alpha=0.01)

    # Create test data
    torch.manual_seed(42)
    batch_size = 8
    images = torch.randn(batch_size, 3, 112, 112)
    labels = torch.randint(0, 100, (batch_size,))

    # Niqab data for new code
    niqab_images = torch.randn(batch_size, 3, 112, 112)
    niqab_masks = torch.rand(batch_size, 1, 7, 7)

    # Store initial model state
    initial_state = {name: param.clone() for name, param in model.named_parameters()}

    print("\n--- Running OLD training step ---")
    old_result = simulate_old_training_step(model, head, images, labels.clone())
    print(f"OLD total_loss: {old_result['total_loss'].item():.6f}")
    print(f"OLD adaface_loss: {old_result['adaface_loss'].item():.6f}")

    # Reset model state
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(initial_state[name])

    print("\n--- Running NEW training step (without niqab) ---")
    new_result_no_niqab = simulate_new_training_step(model, head, images, labels.clone())
    print(f"NEW (no niqab) total_loss: {new_result_no_niqab['total_loss'].item():.6f}")
    print(f"NEW (no niqab) adaface_loss: {new_result_no_niqab['adaface_loss'].item():.6f}")

    # Check if they match
    loss_diff = abs(old_result['total_loss'].item() - new_result_no_niqab['total_loss'].item())
    print(f"\nLoss difference (should be ~0): {loss_diff:.6e}")

    if loss_diff < 1e-5:
        print("PASS: OLD and NEW (without niqab) produce same loss")
    else:
        print("FAIL: OLD and NEW produce different losses!")

    # Reset model state
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(initial_state[name])

    print("\n--- Running NEW training step (with niqab) ---")
    new_result_with_niqab = simulate_new_training_step(model, head, images, labels.clone(),
                                                        niqab_images, niqab_masks)
    print(f"NEW (with niqab) total_loss: {new_result_with_niqab['total_loss'].item():.6f}")
    print(f"NEW (with niqab) adaface_loss: {new_result_with_niqab['adaface_loss'].item():.6f}")
    print(f"NEW (with niqab) occlusion_loss: {new_result_with_niqab['occlusion_loss'].item():.6f}")

    # The adaface_loss should be same regardless of niqab
    adaface_diff = abs(old_result['adaface_loss'].item() - new_result_with_niqab['adaface_loss'].item())
    print(f"\nAdaFace loss difference: {adaface_diff:.6e}")

    if adaface_diff < 1e-5:
        print("PASS: AdaFace loss unchanged by occlusion code")
    else:
        print("FAIL: AdaFace loss CHANGED by occlusion code!")
        print("This means the occlusion code is affecting the main training path!")

    return loss_diff < 1e-5 and adaface_diff < 1e-5


def check_gradient_flow():
    """Check if gradients flow correctly in both scenarios."""
    print("\n" + "=" * 80)
    print("CHECKING GRADIENT FLOW")
    print("=" * 80)

    model = build_old_model()
    model.train()
    head = build_head('adaface', embedding_size=512, class_num=100, m=0.4, h=0.333, s=64.0, t_alpha=0.01)
    cross_entropy_loss = nn.CrossEntropyLoss()

    torch.manual_seed(42)
    batch_size = 8
    images = torch.randn(batch_size, 3, 112, 112)
    labels = torch.randint(0, 100, (batch_size,))
    niqab_images = torch.randn(batch_size, 3, 112, 112)
    niqab_masks = torch.rand(batch_size, 1, 7, 7)

    # Test 1: OLD code gradient flow
    print("\n--- OLD code gradient flow ---")
    model.zero_grad()
    old_result = simulate_old_training_step(model, head, images, labels.clone())
    old_result['total_loss'].backward()

    backbone_grad_old = model.input_layer[0].weight.grad.clone() if model.input_layer[0].weight.grad is not None else None
    output_layer_grad_old = model.output_layer[3].weight.grad.clone() if model.output_layer[3].weight.grad is not None else None

    print(f"Backbone has gradient: {backbone_grad_old is not None}")
    print(f"Output layer has gradient: {output_layer_grad_old is not None}")
    if backbone_grad_old is not None:
        print(f"Backbone gradient magnitude: {backbone_grad_old.abs().mean().item():.6e}")

    # Test 2: NEW code gradient flow (with niqab)
    print("\n--- NEW code gradient flow (with niqab) ---")
    model.zero_grad()
    new_result = simulate_new_training_step(model, head, images, labels.clone(), niqab_images, niqab_masks)
    new_result['total_loss'].backward()

    backbone_grad_new = model.input_layer[0].weight.grad.clone() if model.input_layer[0].weight.grad is not None else None
    output_layer_grad_new = model.output_layer[3].weight.grad.clone() if model.output_layer[3].weight.grad is not None else None
    occlusion_grad = model.occlusion_head.conv1.weight.grad.clone() if model.occlusion_head.conv1.weight.grad is not None else None

    print(f"Backbone has gradient: {backbone_grad_new is not None}")
    print(f"Output layer has gradient: {output_layer_grad_new is not None}")
    print(f"OcclusionHead has gradient: {occlusion_grad is not None}")

    if backbone_grad_new is not None:
        print(f"Backbone gradient magnitude: {backbone_grad_new.abs().mean().item():.6e}")
    if occlusion_grad is not None:
        print(f"OcclusionHead gradient magnitude: {occlusion_grad.abs().mean().item():.6e}")

    # Compare backbone gradients
    print("\n--- Comparing backbone gradients ---")
    if backbone_grad_old is not None and backbone_grad_new is not None:
        grad_diff = (backbone_grad_old - backbone_grad_new).abs().mean().item()
        print(f"Backbone gradient difference: {grad_diff:.6e}")

        if grad_diff < 1e-6:
            print("PASS: Backbone gradients are IDENTICAL")
        else:
            print("FAIL: Backbone gradients are DIFFERENT!")
            print("The occlusion code is affecting backbone gradients!")

            # Find where they differ
            max_diff_idx = (backbone_grad_old - backbone_grad_new).abs().argmax()
            print(f"Max difference at index: {max_diff_idx}")
    else:
        print("Cannot compare - one or both gradients are None")


def check_model_state_after_occlusion():
    """Check if model state is properly restored after occlusion forward."""
    print("\n" + "=" * 80)
    print("CHECKING MODEL STATE AFTER OCCLUSION CODE")
    print("=" * 80)

    model = build_old_model()
    model.train()

    # Get BatchNorm layer
    bn = model.input_layer[1]

    print(f"\nInitial state:")
    print(f"  model.training: {model.training}")
    print(f"  bn.training: {bn.training}")
    print(f"  bn.running_mean[0]: {bn.running_mean[0].item():.6f}")
    print(f"  bn.running_var[0]: {bn.running_var[0].item():.6f}")

    # Run main forward
    images = torch.randn(8, 3, 112, 112)
    x = model.input_layer(images)
    for layer in model.body:
        x = layer(x)

    print(f"\nAfter main forward:")
    print(f"  model.training: {model.training}")
    print(f"  bn.running_mean[0]: {bn.running_mean[0].item():.6f}")
    print(f"  bn.running_var[0]: {bn.running_var[0].item():.6f}")

    running_mean_after_main = bn.running_mean.clone()
    running_var_after_main = bn.running_var.clone()

    # Simulate occlusion code
    was_training = model.training
    model.eval()

    print(f"\nAfter model.eval():")
    print(f"  model.training: {model.training}")
    print(f"  bn.training: {bn.training}")

    niqab_images = torch.randn(8, 3, 112, 112)
    with torch.no_grad():
        niqab_x = model.input_layer(niqab_images)
        for layer in model.body:
            niqab_x = layer(niqab_x)

    print(f"\nAfter niqab forward (in eval mode):")
    print(f"  bn.running_mean[0]: {bn.running_mean[0].item():.6f}")
    print(f"  bn.running_var[0]: {bn.running_var[0].item():.6f}")

    running_mean_after_niqab = bn.running_mean.clone()
    running_var_after_niqab = bn.running_var.clone()

    # Restore
    model.train(was_training)

    print(f"\nAfter model.train(was_training):")
    print(f"  model.training: {model.training}")
    print(f"  bn.training: {bn.training}")

    # Check if running stats changed during eval forward
    mean_changed = not torch.allclose(running_mean_after_main, running_mean_after_niqab)
    var_changed = not torch.allclose(running_var_after_main, running_var_after_niqab)

    print(f"\n--- VERDICT ---")
    print(f"Running mean changed during eval forward: {mean_changed}")
    print(f"Running var changed during eval forward: {var_changed}")

    if not mean_changed and not var_changed:
        print("PASS: BatchNorm stats NOT affected by eval-mode forward")
    else:
        print("FAIL: BatchNorm stats WERE affected!")


def main():
    print("=" * 80)
    print("OLD vs NEW TRAINING STEP COMPARISON")
    print("=" * 80)

    compare_training_steps()
    check_gradient_flow()
    check_model_state_after_occlusion()

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
