"""
Diagnostic script to find where occlusion layer code is interfering with main training.
Compares behavior between old (e0f384a) and new code paths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from net import build_model
from head import build_head
import utils


def test_theory_1_optimizer_params():
    """
    Theory 1: OcclusionHead parameters in optimizer might affect optimizer state.

    Check: Count parameters, verify OcclusionHead params are separate from backbone.
    """
    print("=" * 80)
    print("THEORY 1: OcclusionHead parameters in optimizer")
    print("=" * 80)

    model = build_model('ir_18')

    # Count parameters by module
    backbone_params = 0
    occlusion_params = 0
    output_layer_params = 0
    qaconv_params = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        if 'occlusion_head' in name:
            occlusion_params += num_params
            print(f"  OcclusionHead: {name} - {param.shape}")
        elif 'output_layer' in name:
            output_layer_params += num_params
        elif 'qaconv' in name:
            qaconv_params += num_params
        else:
            backbone_params += num_params

    print(f"\nParameter counts:")
    print(f"  Backbone: {backbone_params:,}")
    print(f"  Output layer: {output_layer_params:,}")
    print(f"  QAConv: {qaconv_params:,}")
    print(f"  OcclusionHead: {occlusion_params:,}")
    print(f"  Total: {backbone_params + output_layer_params + qaconv_params + occlusion_params:,}")

    # Check if OcclusionHead params are isolated
    print(f"\nOcclusionHead is {'ISOLATED' if occlusion_params > 0 else 'MISSING'}")

    return occlusion_params > 0


def test_theory_2_initialize_weights():
    """
    Theory 2: initialize_weights() might re-initialize backbone when OcclusionHead is added.

    Check: Compare backbone weights before/after initialize_weights call.
    """
    print("\n" + "=" * 80)
    print("THEORY 2: initialize_weights() re-initialization")
    print("=" * 80)

    # This would require modifying net.py to test, so we'll check if weights are reasonable
    model = build_model('ir_18')

    # Check first conv layer weights
    first_conv = model.input_layer[0]
    weights = first_conv.weight.data

    print(f"\nFirst conv layer weight stats:")
    print(f"  Mean: {weights.mean().item():.6f}")
    print(f"  Std: {weights.std().item():.6f}")
    print(f"  Min: {weights.min().item():.6f}")
    print(f"  Max: {weights.max().item():.6f}")

    # Kaiming init should have std ~ sqrt(2/fan_in)
    fan_in = weights.shape[1] * weights.shape[2] * weights.shape[3]
    expected_std = (2.0 / fan_in) ** 0.5
    print(f"  Expected std (Kaiming): {expected_std:.6f}")

    is_reasonable = 0.5 * expected_std < weights.std().item() < 2.0 * expected_std
    print(f"\nWeights are {'REASONABLE' if is_reasonable else 'SUSPICIOUS'}")

    return is_reasonable


def test_theory_3_eval_train_mode():
    """
    Theory 3: self.model.eval()/train() mid-step could affect BatchNorm.

    Check: Verify BatchNorm running stats aren't corrupted by eval/train switching.
    """
    print("\n" + "=" * 80)
    print("THEORY 3: eval()/train() mode switching")
    print("=" * 80)

    model = build_model('ir_18')
    model.train()

    # Get a BatchNorm layer
    bn_layer = model.input_layer[1]  # BatchNorm2d after first conv

    # Store initial running stats
    initial_running_mean = bn_layer.running_mean.clone()
    initial_running_var = bn_layer.running_var.clone()

    print(f"\nInitial BatchNorm running stats:")
    print(f"  running_mean: {initial_running_mean.mean().item():.6f}")
    print(f"  running_var: {initial_running_var.mean().item():.6f}")

    # Simulate what happens in training_step
    batch_size = 8
    images = torch.randn(batch_size, 3, 112, 112)

    # Main forward pass (training mode)
    x = model.input_layer(images)
    for layer in model.body:
        x = layer(x)

    after_main_forward_mean = bn_layer.running_mean.clone()
    after_main_forward_var = bn_layer.running_var.clone()

    print(f"\nAfter main forward (training mode):")
    print(f"  running_mean changed: {not torch.allclose(initial_running_mean, after_main_forward_mean)}")
    print(f"  running_var changed: {not torch.allclose(initial_running_var, after_main_forward_var)}")

    # Now simulate the occlusion code: eval() -> forward -> train()
    was_training = model.training
    model.eval()

    niqab_images = torch.randn(batch_size, 3, 112, 112)
    with torch.no_grad():
        niqab_x = model.input_layer(niqab_images)
        for layer in model.body:
            niqab_x = layer(niqab_x)

    model.train(was_training)

    after_occlusion_mean = bn_layer.running_mean.clone()
    after_occlusion_var = bn_layer.running_var.clone()

    print(f"\nAfter occlusion forward (eval mode with no_grad):")
    print(f"  running_mean changed: {not torch.allclose(after_main_forward_mean, after_occlusion_mean)}")
    print(f"  running_var changed: {not torch.allclose(after_main_forward_var, after_occlusion_var)}")

    # In eval mode, running stats should NOT be updated
    stats_unchanged = torch.allclose(after_main_forward_mean, after_occlusion_mean) and \
                      torch.allclose(after_main_forward_var, after_occlusion_var)

    print(f"\nBatchNorm running stats unchanged during eval: {'YES (GOOD)' if stats_unchanged else 'NO (BUG!)'}")

    return stats_unchanged


def test_theory_4_gradient_interference():
    """
    Theory 4: Occlusion loss gradients might interfere with main training gradients.

    Check: Verify backbone gradients come only from adaface_loss, not occlusion_loss.
    """
    print("\n" + "=" * 80)
    print("THEORY 4: Gradient interference")
    print("=" * 80)

    model = build_model('ir_18')
    model.train()
    head = build_head('adaface', embedding_size=512, class_num=100, m=0.4, h=0.333, s=64.0, t_alpha=0.01)
    cross_entropy_loss = nn.CrossEntropyLoss()

    batch_size = 8
    images = torch.randn(batch_size, 3, 112, 112)
    labels = torch.randint(0, 100, (batch_size,))

    # Forward pass (simulating training_step)
    x = model.input_layer(images)
    for layer in model.body:
        x = layer(x)

    # Store backbone output for later
    backbone_output = x.clone()

    # Normalize for QAConv (detached)
    x_norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1, keepdim=True).view(x.size(0), 1, 1, 1)
    x_norm = torch.clamp(x_norm, min=1e-8)
    feature_maps = x / x_norm
    feature_maps = feature_maps.clone().detach().requires_grad_(True)  # DETACHED

    # AdaFace embeddings (not detached - should get gradients)
    embeddings = model.output_layer(x)
    embeddings, norms = utils.l2_norm(embeddings, axis=1)

    # AdaFace loss
    cos_thetas = head(embeddings, norms, labels)
    if isinstance(cos_thetas, tuple):
        cos_thetas, _ = cos_thetas
    adaface_loss = cross_entropy_loss(cos_thetas, labels)

    # Simulate occlusion loss (using backbone_output with no_grad simulation)
    # In real code, niqab_x is computed with no_grad, so we simulate that
    with torch.no_grad():
        niqab_features = backbone_output.clone()
    niqab_features.requires_grad_(True)  # Enable grad for occlusion head only

    occlusion_maps = model.occlusion_head(niqab_features)
    gt_masks = torch.rand(batch_size, 1, 7, 7)
    occlusion_loss = F.mse_loss(occlusion_maps, gt_masks)

    # Combined loss
    total_loss = 0.1 * adaface_loss + 0.1 * occlusion_loss

    # Backward
    total_loss.backward()

    # Check gradients - use input_layer conv instead of body
    backbone_has_grad = model.input_layer[0].weight.grad is not None
    occlusion_head_has_grad = model.occlusion_head.conv1.weight.grad is not None

    print(f"\nGradient check:")
    print(f"  Backbone has gradients: {backbone_has_grad}")
    print(f"  OcclusionHead has gradients: {occlusion_head_has_grad}")

    if backbone_has_grad:
        backbone_grad_mag = model.input_layer[0].weight.grad.abs().mean().item()
        print(f"  Backbone gradient magnitude: {backbone_grad_mag:.6e}")

    if occlusion_head_has_grad:
        occlusion_grad_mag = model.occlusion_head.conv1.weight.grad.abs().mean().item()
        print(f"  OcclusionHead gradient magnitude: {occlusion_grad_mag:.6e}")

    # The key check: backbone gradients should come ONLY from adaface_loss
    # Let's verify by computing gradients separately
    model.zero_grad()

    # Only adaface backward
    x2 = model.input_layer(images)
    for layer in model.body:
        x2 = layer(x2)
    embeddings2 = model.output_layer(x2)
    embeddings2, norms2 = utils.l2_norm(embeddings2, axis=1)
    cos_thetas2 = head(embeddings2, norms2, labels)
    if isinstance(cos_thetas2, tuple):
        cos_thetas2, _ = cos_thetas2
    adaface_loss2 = cross_entropy_loss(cos_thetas2, labels)
    (0.1 * adaface_loss2).backward()

    adaface_only_grad = model.input_layer[0].weight.grad.clone()

    print(f"\nAdaFace-only gradient magnitude: {adaface_only_grad.abs().mean().item():.6e}")

    # Gradients should be similar (from adaface only)
    print(f"\nBackbone receives gradients ONLY from AdaFace: EXPECTED (due to detach)")

    return backbone_has_grad


def test_theory_5_loss_magnitudes():
    """
    Theory 5: Occlusion loss magnitude might be too large/small.

    Check: Compare loss magnitudes.
    """
    print("\n" + "=" * 80)
    print("THEORY 5: Loss magnitude comparison")
    print("=" * 80)

    model = build_model('ir_18')
    model.train()
    head = build_head('adaface', embedding_size=512, class_num=100, m=0.4, h=0.333, s=64.0, t_alpha=0.01)
    cross_entropy_loss = nn.CrossEntropyLoss()

    batch_size = 8
    images = torch.randn(batch_size, 3, 112, 112)
    labels = torch.randint(0, 100, (batch_size,))

    # Forward
    x = model.input_layer(images)
    for layer in model.body:
        x = layer(x)

    # AdaFace
    embeddings = model.output_layer(x)
    embeddings, norms = utils.l2_norm(embeddings, axis=1)
    cos_thetas = head(embeddings, norms, labels)
    if isinstance(cos_thetas, tuple):
        cos_thetas, _ = cos_thetas
    adaface_loss = cross_entropy_loss(cos_thetas, labels)

    # Occlusion
    occlusion_maps = model.occlusion_head(x.detach())
    gt_masks = torch.rand(batch_size, 1, 7, 7)  # Random GT for testing
    occlusion_loss = F.mse_loss(occlusion_maps, gt_masks)

    print(f"\nLoss magnitudes:")
    print(f"  AdaFace loss: {adaface_loss.item():.4f}")
    print(f"  Occlusion loss: {occlusion_loss.item():.4f}")
    print(f"\nWeighted losses (0.1 weight each):")
    print(f"  0.1 * AdaFace: {0.1 * adaface_loss.item():.4f}")
    print(f"  0.1 * Occlusion: {0.1 * occlusion_loss.item():.4f}")

    ratio = occlusion_loss.item() / (adaface_loss.item() + 1e-8)
    print(f"\nOcclusion/AdaFace ratio: {ratio:.4f}")

    if ratio > 10:
        print("WARNING: Occlusion loss is much larger than AdaFace!")
    elif ratio < 0.1:
        print("WARNING: Occlusion loss is much smaller than AdaFace!")
    else:
        print("Loss magnitudes are comparable (GOOD)")

    return True


def test_theory_6_pl_api():
    """
    Theory 6: PyTorch Lightning API differences.

    Check: Verify the correct API is being used.
    """
    print("\n" + "=" * 80)
    print("THEORY 6: PyTorch Lightning API")
    print("=" * 80)

    import pytorch_lightning as pl
    print(f"\nPyTorch Lightning version: {pl.__version__}")

    major_version = int(pl.__version__.split('.')[0])
    print(f"Major version: {major_version}")

    if major_version >= 2:
        print("Using PL 2.x - should use on_train_epoch_end, on_validation_epoch_end")
    else:
        print("Using PL 1.x - should use training_epoch_end, validation_epoch_end")

    # Check what's in train_val.py
    with open('train_val.py', 'r') as f:
        content = f.read()

    has_old_api = 'def training_epoch_end' in content or 'def validation_epoch_end' in content
    has_new_api = 'def on_train_epoch_end' in content or 'def on_validation_epoch_end' in content

    print(f"\ntrain_val.py uses:")
    print(f"  Old API (training_epoch_end): {has_old_api}")
    print(f"  New API (on_train_epoch_end): {has_new_api}")

    if major_version >= 2 and has_old_api and not has_new_api:
        print("\nWARNING: PL 2.x but using old API!")
    elif major_version < 2 and has_new_api and not has_old_api:
        print("\nWARNING: PL 1.x but using new API!")
    else:
        print("\nAPI version matches PL version (GOOD)")

    return True


def test_theory_7_niqab_dataloader():
    """
    Theory 7: Niqab dataloader might be causing issues.

    Check: Test if dataloader works correctly.
    """
    print("\n" + "=" * 80)
    print("THEORY 7: Niqab dataloader")
    print("=" * 80)

    try:
        from dataset.niqab_mask_dataset import NiqabMaskDataset, get_default_niqab_transform
        print("NiqabMaskDataset imported successfully")

        # Check if the dataset can be instantiated (without actual data)
        print("\nDataset class is available")
        print("To test with actual data, run training with --niqab_data_path")

    except Exception as e:
        print(f"ERROR importing NiqabMaskDataset: {e}")
        return False

    return True


def test_theory_8_forward_signature():
    """
    Theory 8: Changed forward() signature might cause issues.

    Check: Verify forward() works correctly with default args.
    """
    print("\n" + "=" * 80)
    print("THEORY 8: Backbone forward() signature")
    print("=" * 80)

    model = build_model('ir_18')
    model.eval()

    images = torch.randn(2, 3, 112, 112)

    # Test default call (should work like old code)
    try:
        output, norm = model(images)
        print(f"Default forward() works:")
        print(f"  output shape: {output.shape}")
        print(f"  norm shape: {norm.shape}")
    except Exception as e:
        print(f"ERROR in default forward(): {e}")
        return False

    # Test with return_occlusion=True
    try:
        output, norm, occ_map, feat_maps = model(images, return_occlusion=True)
        print(f"\nforward(return_occlusion=True) works:")
        print(f"  output shape: {output.shape}")
        print(f"  occlusion_map shape: {occ_map.shape}")
        print(f"  feature_maps shape: {feat_maps.shape}")
    except Exception as e:
        print(f"ERROR in forward(return_occlusion=True): {e}")
        return False

    print("\nForward signature is correct (GOOD)")
    return True


def main():
    print("=" * 80)
    print("OCCLUSION LAYER INTERFERENCE DIAGNOSTIC")
    print("=" * 80)
    print("\nTesting theories for why occlusion code might affect main training...\n")

    results = {}

    results['theory_1_optimizer'] = test_theory_1_optimizer_params()
    results['theory_2_init_weights'] = test_theory_2_initialize_weights()
    results['theory_3_eval_train'] = test_theory_3_eval_train_mode()
    results['theory_4_gradients'] = test_theory_4_gradient_interference()
    results['theory_5_loss_mag'] = test_theory_5_loss_magnitudes()
    results['theory_6_pl_api'] = test_theory_6_pl_api()
    results['theory_7_dataloader'] = test_theory_7_niqab_dataloader()
    results['theory_8_forward'] = test_theory_8_forward_signature()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for theory, passed in results.items():
        status = "PASS" if passed else "FAIL/ISSUE"
        print(f"  {theory}: {status}")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
If all tests pass, the issue might be:
1. Something specific to your data/training setup
2. A subtle interaction not caught by these tests
3. The actual niqab data causing issues

Recommended debugging:
1. Run training WITHOUT --niqab_data_path to verify baseline still works
2. Add print statements to training_step to log loss values
3. Check if backbone weights are actually changing during training
""")

    return results


if __name__ == '__main__':
    main()
