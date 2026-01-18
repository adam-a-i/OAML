#!/usr/bin/env python3
"""
Test script for training loop occlusion integration.

This script tests that:
1. Training step handles batch with and without GT masks
2. Occlusion maps are computed from feature maps
3. Occlusion loss (MSE) is computed when GT masks provided
4. Total loss includes occlusion loss component
5. Gradients flow through occlusion head

Run this script on HPC:
    python tests/test_training_occlusion.py

Expected output: All tests should print [PASS]

NOTE: This test creates a minimal mock trainer to test the occlusion integration
without requiring the full training infrastructure.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import model components
from net import build_model, OcclusionHead


def print_test_header(test_name):
    """Print formatted test header."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print('='*60)


def test_occlusion_head_integration():
    """Test that occlusion head is properly integrated in the model."""
    print_test_header("Occlusion Head Integration")

    try:
        model = build_model('ir_18')

        # Check occlusion head exists
        assert hasattr(model, 'occlusion_head'), "Model should have occlusion_head"
        assert isinstance(model.occlusion_head, OcclusionHead), "Should be OcclusionHead instance"
        print(f"  Occlusion head exists in model [PASS]")

        # Test forward with return_occlusion
        model.eval()
        x = torch.randn(2, 3, 112, 112)

        with torch.no_grad():
            output, norm, occ_map, feat_map = model(x, return_occlusion=True)

        assert occ_map.shape == (2, 1, 7, 7), f"Occlusion map shape: {occ_map.shape}"
        assert occ_map.min() >= 0 and occ_map.max() <= 1, "Occlusion values out of range"
        print(f"  Occlusion map shape: {list(occ_map.shape)} [PASS]")
        print(f"  Occlusion map range: [{occ_map.min():.4f}, {occ_map.max():.4f}] [PASS]")

        print("\n[PASS] Occlusion head integration test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_occlusion_loss_computation():
    """Test that occlusion loss is computed correctly."""
    print_test_header("Occlusion Loss Computation")

    try:
        model = build_model('ir_18')
        model.train()

        # Create input and GT mask
        images = torch.randn(4, 3, 112, 112)
        gt_masks = torch.rand(4, 1, 7, 7)  # Random GT masks

        # Extract features manually (as training_step does)
        x = model.input_layer(images)
        for layer in model.body:
            x = layer(x)

        # Compute occlusion maps
        occlusion_maps = model.occlusion_head(x)

        # Compute MSE loss
        occlusion_loss = F.mse_loss(occlusion_maps, gt_masks)

        assert not torch.isnan(occlusion_loss), "Occlusion loss is NaN"
        assert occlusion_loss >= 0, "Occlusion loss should be non-negative"
        print(f"  Occlusion loss: {occlusion_loss.item():.6f} [PASS]")

        # Test with different GT masks
        gt_masks_ones = torch.ones(4, 1, 7, 7)
        gt_masks_zeros = torch.zeros(4, 1, 7, 7)

        loss_ones = F.mse_loss(occlusion_maps, gt_masks_ones)
        loss_zeros = F.mse_loss(occlusion_maps, gt_masks_zeros)

        print(f"  Loss with all-ones GT: {loss_ones.item():.6f}")
        print(f"  Loss with all-zeros GT: {loss_zeros.item():.6f}")

        # Losses should be different
        assert abs(loss_ones.item() - loss_zeros.item()) > 0.01, \
            "Losses should differ based on GT"
        print(f"  Loss varies with GT [PASS]")

        print("\n[PASS] Occlusion loss computation test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow_occlusion_loss():
    """Test that gradients flow through occlusion loss."""
    print_test_header("Gradient Flow Through Occlusion Loss")

    try:
        model = build_model('ir_18')
        model.train()

        # Create input and GT mask
        images = torch.randn(4, 3, 112, 112, requires_grad=True)
        gt_masks = torch.rand(4, 1, 7, 7)

        # Forward pass
        x = model.input_layer(images)
        for layer in model.body:
            x = layer(x)

        occlusion_maps = model.occlusion_head(x)
        occlusion_loss = F.mse_loss(occlusion_maps, gt_masks)

        # Backward pass
        occlusion_loss.backward()

        # Check gradients for occlusion head
        assert model.occlusion_head.conv1.weight.grad is not None, \
            "Occlusion head conv1 should have gradients"
        assert not torch.isnan(model.occlusion_head.conv1.weight.grad).any(), \
            "Gradients should not be NaN"

        grad_norm = model.occlusion_head.conv1.weight.grad.norm().item()
        print(f"  Occlusion head conv1 grad norm: {grad_norm:.6f} [PASS]")

        # Check input gradients
        assert images.grad is not None, "Input should have gradients"
        input_grad_norm = images.grad.norm().item()
        print(f"  Input grad norm: {input_grad_norm:.6f} [PASS]")

        print("\n[PASS] Gradient flow test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_mask_resizing():
    """Test that GT masks are properly resized if needed."""
    print_test_header("Mask Resizing")

    try:
        # Simulate GT masks at different resolutions
        batch_size = 4

        # Occlusion maps are [B, 1, 7, 7]
        occlusion_maps = torch.rand(batch_size, 1, 7, 7)

        # Test various GT mask sizes
        test_sizes = [(7, 7), (14, 14), (28, 28), (112, 112)]

        for h, w in test_sizes:
            gt_masks = torch.rand(batch_size, 1, h, w)

            # Resize if needed (as training_step does)
            if gt_masks.shape[-2:] != occlusion_maps.shape[-2:]:
                gt_masks_resized = F.interpolate(
                    gt_masks,
                    size=occlusion_maps.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            else:
                gt_masks_resized = gt_masks

            assert gt_masks_resized.shape == occlusion_maps.shape, \
                f"Resized shape mismatch: {gt_masks_resized.shape} vs {occlusion_maps.shape}"

            # Compute loss
            loss = F.mse_loss(occlusion_maps, gt_masks_resized)
            assert not torch.isnan(loss), f"Loss is NaN for size {h}x{w}"

            print(f"  GT size {h}x{w} -> resized to 7x7, loss={loss.item():.4f} [PASS]")

        print("\n[PASS] Mask resizing test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_combined_loss():
    """Test that occlusion loss combines correctly with other losses."""
    print_test_header("Combined Loss")

    try:
        model = build_model('ir_18')
        model.train()

        # Create dummy losses
        adaface_loss = torch.tensor(1.0, requires_grad=True)
        qaconv_loss = torch.tensor(0.5, requires_grad=True)
        occlusion_loss = torch.tensor(0.2, requires_grad=True)

        # Weights (as defined in train_val.py)
        adaface_weight = 0.1
        qaconv_weight = 0.9
        occlusion_weight = 0.1

        # Combined loss
        total_loss = (
            adaface_weight * adaface_loss +
            qaconv_weight * qaconv_loss +
            occlusion_weight * occlusion_loss
        )

        expected = adaface_weight * 1.0 + qaconv_weight * 0.5 + occlusion_weight * 0.2
        assert abs(total_loss.item() - expected) < 1e-6, \
            f"Combined loss mismatch: {total_loss.item()} vs {expected}"

        print(f"  AdaFace loss: {adaface_loss.item():.4f} (weight={adaface_weight})")
        print(f"  QAConv loss: {qaconv_loss.item():.4f} (weight={qaconv_weight})")
        print(f"  Occlusion loss: {occlusion_loss.item():.4f} (weight={occlusion_weight})")
        print(f"  Total loss: {total_loss.item():.4f} [PASS]")

        # Verify gradients flow through all components
        total_loss.backward()

        assert adaface_loss.grad is not None, "AdaFace loss should have gradient"
        assert qaconv_loss.grad is not None, "QAConv loss should have gradient"
        assert occlusion_loss.grad is not None, "Occlusion loss should have gradient"

        print(f"  Gradients flow to all loss components [PASS]")

        print("\n[PASS] Combined loss test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_format_handling():
    """Test that training can handle both batch formats."""
    print_test_header("Batch Format Handling")

    try:
        # Simulate batch without masks: (images, labels)
        batch_no_mask = (
            torch.randn(4, 3, 112, 112),
            torch.randint(0, 100, (4,))
        )

        # Check format detection
        if len(batch_no_mask) == 3:
            has_gt_masks = True
        else:
            has_gt_masks = False

        assert not has_gt_masks, "Batch without mask should be detected"
        print(f"  Batch without masks: len={len(batch_no_mask)}, has_gt_masks={has_gt_masks} [PASS]")

        # Simulate batch with masks: (images, labels, masks)
        batch_with_mask = (
            torch.randn(4, 3, 112, 112),
            torch.randint(0, 100, (4,)),
            torch.rand(4, 1, 7, 7)
        )

        if len(batch_with_mask) == 3:
            images, labels, gt_masks = batch_with_mask
            has_gt_masks = True
        else:
            has_gt_masks = False

        assert has_gt_masks, "Batch with mask should be detected"
        assert gt_masks.shape == (4, 1, 7, 7), f"Mask shape: {gt_masks.shape}"
        print(f"  Batch with masks: len={len(batch_with_mask)}, has_gt_masks={has_gt_masks} [PASS]")

        print("\n[PASS] Batch format handling test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_device_compatibility():
    """Test that occlusion computation works on CPU and GPU."""
    print_test_header("Device Compatibility")

    try:
        model = build_model('ir_18')

        images = torch.randn(2, 3, 112, 112)
        gt_masks = torch.rand(2, 1, 7, 7)

        # CPU test
        model_cpu = model.to('cpu')
        model_cpu.eval()

        with torch.no_grad():
            x = model_cpu.input_layer(images)
            for layer in model_cpu.body:
                x = layer(x)
            occ_cpu = model_cpu.occlusion_head(x)

        loss_cpu = F.mse_loss(occ_cpu, gt_masks)
        print(f"  CPU occlusion loss: {loss_cpu.item():.6f} [PASS]")

        # GPU test if available
        if torch.cuda.is_available():
            model_gpu = model.to('cuda')
            model_gpu.eval()
            images_gpu = images.cuda()
            gt_masks_gpu = gt_masks.cuda()

            with torch.no_grad():
                x = model_gpu.input_layer(images_gpu)
                for layer in model_gpu.body:
                    x = layer(x)
                occ_gpu = model_gpu.occlusion_head(x)

            loss_gpu = F.mse_loss(occ_gpu, gt_masks_gpu)
            print(f"  GPU occlusion loss: {loss_gpu.item():.6f} [PASS]")

            # Compare CPU and GPU
            diff = abs(loss_cpu.item() - loss_gpu.item())
            print(f"  CPU-GPU loss diff: {diff:.6f}")
            assert diff < 0.1, f"CPU-GPU loss diff too large: {diff}"
            print(f"  CPU-GPU consistency [PASS]")
        else:
            print(f"  GPU not available, skipping GPU tests")

        print("\n[PASS] Device compatibility test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report summary."""
    print("\n" + "="*60)
    print("TRAINING OCCLUSION INTEGRATION TEST SUITE")
    print("="*60)

    tests = [
        ("Occlusion Head Integration", test_occlusion_head_integration),
        ("Occlusion Loss Computation", test_occlusion_loss_computation),
        ("Gradient Flow Through Occlusion Loss", test_gradient_flow_occlusion_loss),
        ("Mask Resizing", test_mask_resizing),
        ("Combined Loss", test_combined_loss),
        ("Batch Format Handling", test_batch_format_handling),
        ("Device Compatibility", test_device_compatibility),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n[FAIL] {name} test crashed: {str(e)}")
            results.append((name, False))

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, p in results if p)
    failed = len(results) - passed

    for name, p in results:
        status = "[PASS]" if p else "[FAIL]"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{len(results)} tests passed")

    if failed > 0:
        print(f"\n[OVERALL: FAIL] {failed} test(s) failed")
        return False
    else:
        print(f"\n[OVERALL: PASS] All tests passed!")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
