#!/usr/bin/env python3
"""
Test script for Backbone integration with OcclusionHead.

This script tests that:
1. Backbone correctly instantiates with OcclusionHead
2. Forward pass works with and without return_occlusion flag
3. Occlusion maps have correct shapes
4. Feature maps are shared between embedding and occlusion head
5. Gradient flow works correctly through both branches
6. Different backbone architectures work (IR-18, IR-50, etc.)

Run this script on HPC:
    python tests/test_backbone_occlusion.py

Expected output: All tests should print [PASS]
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np

# Import the backbone and build functions
from net import Backbone, build_model, OcclusionHead


def print_test_header(test_name):
    """Print formatted test header."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print('='*60)


def test_backbone_has_occlusion_head():
    """Test that Backbone correctly instantiates with OcclusionHead."""
    print_test_header("Backbone Has OcclusionHead")

    try:
        # Test with 112x112 input
        backbone_112 = Backbone(input_size=(112, 112), num_layers=18, mode='ir')
        assert hasattr(backbone_112, 'occlusion_head'), "Backbone should have occlusion_head attribute"
        assert isinstance(backbone_112.occlusion_head, OcclusionHead), "occlusion_head should be OcclusionHead instance"
        print(f"  Backbone 112x112: has occlusion_head [PASS]")

        # Test with 224x224 input
        backbone_224 = Backbone(input_size=(224, 224), num_layers=18, mode='ir')
        assert hasattr(backbone_224, 'occlusion_head'), "Backbone 224 should have occlusion_head"
        print(f"  Backbone 224x224: has occlusion_head [PASS]")

        # Test build_model function
        model = build_model('ir_18')
        assert hasattr(model, 'occlusion_head'), "Built model should have occlusion_head"
        print(f"  build_model('ir_18'): has occlusion_head [PASS]")

        print("\n[PASS] Backbone has OcclusionHead test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_without_occlusion():
    """Test that forward pass works without return_occlusion (backward compatible)."""
    print_test_header("Forward Without Occlusion Flag")

    try:
        model = build_model('ir_18')
        model.eval()

        # Create dummy input
        x = torch.randn(2, 3, 112, 112)

        with torch.no_grad():
            result = model(x)

        # Should return (output, norm) by default
        assert isinstance(result, tuple), "Forward should return tuple"
        assert len(result) == 2, f"Forward should return 2 elements, got {len(result)}"

        output, norm = result
        assert output.shape == (2, 512), f"Output shape should be [2, 512], got {output.shape}"
        assert norm.shape == (2, 1), f"Norm shape should be [2, 1], got {norm.shape}"

        print(f"  Output shape: {list(output.shape)} [PASS]")
        print(f"  Norm shape: {list(norm.shape)} [PASS]")
        print(f"  Backward compatible: [PASS]")

        print("\n[PASS] Forward without occlusion test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_with_occlusion():
    """Test that forward pass returns occlusion maps when requested."""
    print_test_header("Forward With Occlusion Flag")

    try:
        model = build_model('ir_18')
        model.eval()

        # Create dummy input
        x = torch.randn(4, 3, 112, 112)

        with torch.no_grad():
            result = model(x, return_occlusion=True)

        # Should return (output, norm, occlusion_map, feature_maps)
        assert isinstance(result, tuple), "Forward should return tuple"
        assert len(result) == 4, f"Forward with occlusion should return 4 elements, got {len(result)}"

        output, norm, occlusion_map, feature_maps = result

        # Check shapes
        assert output.shape == (4, 512), f"Output shape should be [4, 512], got {output.shape}"
        assert norm.shape == (4, 1), f"Norm shape should be [4, 1], got {norm.shape}"
        assert occlusion_map.shape == (4, 1, 7, 7), f"Occlusion map shape should be [4, 1, 7, 7], got {occlusion_map.shape}"
        assert feature_maps.shape == (4, 512, 7, 7), f"Feature maps shape should be [4, 512, 7, 7], got {feature_maps.shape}"

        print(f"  Output shape: {list(output.shape)} [PASS]")
        print(f"  Norm shape: {list(norm.shape)} [PASS]")
        print(f"  Occlusion map shape: {list(occlusion_map.shape)} [PASS]")
        print(f"  Feature maps shape: {list(feature_maps.shape)} [PASS]")

        # Check occlusion map values are in [0, 1]
        assert occlusion_map.min() >= 0.0, "Occlusion map has values < 0"
        assert occlusion_map.max() <= 1.0, "Occlusion map has values > 1"
        print(f"  Occlusion map range: [{occlusion_map.min():.4f}, {occlusion_map.max():.4f}] [PASS]")

        print("\n[PASS] Forward with occlusion test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_occlusion_spatial_resolution():
    """Test that occlusion maps have correct spatial resolution for different input sizes."""
    print_test_header("Occlusion Spatial Resolution")

    try:
        # Test 112x112 input -> 7x7 occlusion map
        model_112 = Backbone(input_size=(112, 112), num_layers=18, mode='ir')
        model_112.eval()
        x_112 = torch.randn(2, 3, 112, 112)

        with torch.no_grad():
            _, _, occ_112, feat_112 = model_112(x_112, return_occlusion=True)

        assert occ_112.shape == (2, 1, 7, 7), f"112x112 input should give 7x7 occlusion, got {occ_112.shape}"
        assert feat_112.shape == (2, 512, 7, 7), f"112x112 input should give 7x7 features, got {feat_112.shape}"
        print(f"  112x112 input -> {list(occ_112.shape)} occlusion [PASS]")

        # Test 224x224 input -> 14x14 occlusion map
        model_224 = Backbone(input_size=(224, 224), num_layers=18, mode='ir')
        model_224.eval()
        x_224 = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            _, _, occ_224, feat_224 = model_224(x_224, return_occlusion=True)

        assert occ_224.shape == (2, 1, 14, 14), f"224x224 input should give 14x14 occlusion, got {occ_224.shape}"
        assert feat_224.shape == (2, 512, 14, 14), f"224x224 input should give 14x14 features, got {feat_224.shape}"
        print(f"  224x224 input -> {list(occ_224.shape)} occlusion [PASS]")

        print("\n[PASS] Occlusion spatial resolution test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow_both_branches():
    """Test that gradients flow through both embedding and occlusion branches."""
    print_test_header("Gradient Flow Through Both Branches")

    try:
        model = build_model('ir_18')
        model.train()

        x = torch.randn(2, 3, 112, 112, requires_grad=True)

        # Forward with occlusion
        output, norm, occlusion_map, feature_maps = model(x, return_occlusion=True)

        # Create losses for both branches
        embedding_loss = output.mean()  # Simple loss on embeddings
        occlusion_loss = occlusion_map.mean()  # Simple loss on occlusion

        # Combined loss
        total_loss = embedding_loss + occlusion_loss
        total_loss.backward()

        # Check that input has gradients
        assert x.grad is not None, "Input should have gradients"
        assert not torch.isnan(x.grad).any(), "Input gradients contain NaN"
        print(f"  Input gradient norm: {x.grad.norm().item():.6f} [PASS]")

        # Check gradients for shared backbone layers (input_layer)
        input_conv = model.input_layer[0]
        assert input_conv.weight.grad is not None, "Input conv should have gradients"
        assert not torch.isnan(input_conv.weight.grad).any(), "Input conv gradients contain NaN"
        print(f"  Shared backbone gradient (input_layer): {input_conv.weight.grad.norm().item():.6f} [PASS]")

        # Check gradients for occlusion head
        assert model.occlusion_head.conv1.weight.grad is not None, "Occlusion head conv1 should have gradients"
        assert not torch.isnan(model.occlusion_head.conv1.weight.grad).any(), "Occlusion head gradients contain NaN"
        print(f"  Occlusion head gradient (conv1): {model.occlusion_head.conv1.weight.grad.norm().item():.6f} [PASS]")

        print("\n[PASS] Gradient flow test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_different_architectures():
    """Test that occlusion integration works with different backbone architectures."""
    print_test_header("Different Backbone Architectures")

    try:
        architectures = ['ir_18', 'ir_34', 'ir_50']

        for arch in architectures:
            model = build_model(arch)
            model.eval()

            x = torch.randn(1, 3, 112, 112)

            with torch.no_grad():
                result = model(x, return_occlusion=True)

            assert len(result) == 4, f"{arch}: Should return 4 elements with return_occlusion=True"
            output, norm, occ, feat = result

            assert output.shape == (1, 512), f"{arch}: Output shape mismatch"
            assert occ.shape == (1, 1, 7, 7), f"{arch}: Occlusion shape mismatch"

            print(f"  {arch}: output={list(output.shape)}, occlusion={list(occ.shape)} [PASS]")

        print("\n[PASS] Different architectures test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_sharing():
    """Test that occlusion head uses the same feature maps as embedding branch."""
    print_test_header("Feature Sharing Between Branches")

    try:
        model = build_model('ir_18')
        model.eval()

        x = torch.randn(2, 3, 112, 112)

        with torch.no_grad():
            output, norm, occlusion_map, feature_maps = model(x, return_occlusion=True)

        # The feature maps returned should be the same ones used by occlusion head
        # Verify by checking that occlusion_head(feature_maps) == occlusion_map
        occ_recomputed = model.occlusion_head(feature_maps)

        diff = (occlusion_map - occ_recomputed).abs().max().item()
        assert diff < 1e-6, f"Occlusion maps differ: max_diff={diff}"
        print(f"  Recomputed occlusion diff: {diff:.2e} [PASS]")

        # Verify feature maps shape matches what output_layer expects
        # For IR-18, output_layer expects [B, 512, 7, 7] flattened
        expected_flat_size = 512 * 7 * 7
        actual_flat_size = feature_maps.shape[1] * feature_maps.shape[2] * feature_maps.shape[3]
        assert actual_flat_size == expected_flat_size, f"Feature map size mismatch: {actual_flat_size} vs {expected_flat_size}"
        print(f"  Feature map size matches output_layer expectation [PASS]")

        print("\n[PASS] Feature sharing test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_sizes():
    """Test that backbone works with various batch sizes."""
    print_test_header("Various Batch Sizes")

    try:
        model = build_model('ir_18')
        model.eval()

        batch_sizes = [1, 2, 4, 8, 16]

        for bs in batch_sizes:
            x = torch.randn(bs, 3, 112, 112)

            with torch.no_grad():
                output, norm, occ, feat = model(x, return_occlusion=True)

            assert output.shape == (bs, 512), f"Batch {bs}: Output shape mismatch"
            assert occ.shape == (bs, 1, 7, 7), f"Batch {bs}: Occlusion shape mismatch"
            print(f"  Batch size {bs}: [PASS]")

        print("\n[PASS] Batch sizes test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_device_transfer():
    """Test backbone with occlusion works on CPU and GPU."""
    print_test_header("Device Transfer")

    try:
        model = build_model('ir_18')
        x = torch.randn(2, 3, 112, 112)

        # Test on CPU
        model_cpu = model.to('cpu')
        model_cpu.eval()
        x_cpu = x.to('cpu')

        with torch.no_grad():
            out_cpu, norm_cpu, occ_cpu, feat_cpu = model_cpu(x_cpu, return_occlusion=True)

        assert out_cpu.device.type == 'cpu', "CPU output should be on CPU"
        assert occ_cpu.device.type == 'cpu', "CPU occlusion should be on CPU"
        print(f"  CPU forward: [PASS]")

        # Test on GPU if available
        if torch.cuda.is_available():
            model_gpu = model.to('cuda')
            model_gpu.eval()
            x_gpu = x.to('cuda')

            with torch.no_grad():
                out_gpu, norm_gpu, occ_gpu, feat_gpu = model_gpu(x_gpu, return_occlusion=True)

            assert out_gpu.device.type == 'cuda', "GPU output should be on GPU"
            assert occ_gpu.device.type == 'cuda', "GPU occlusion should be on GPU"
            print(f"  GPU forward: [PASS]")

            # Check CPU-GPU consistency (relaxed tolerance due to numerical differences)
            out_diff = (out_cpu - out_gpu.cpu()).abs().max().item()
            occ_diff = (occ_cpu - occ_gpu.cpu()).abs().max().item()
            print(f"  Output CPU-GPU diff: {out_diff:.2e}")
            print(f"  Occlusion CPU-GPU diff: {occ_diff:.2e}")

            assert out_diff < 0.1, f"Output differs too much: {out_diff}"  # Relaxed for BN differences
            assert occ_diff < 0.1, f"Occlusion differs too much: {occ_diff}"
            print(f"  CPU-GPU consistency: [PASS]")
        else:
            print(f"  GPU not available, skipping GPU tests")

        print("\n[PASS] Device transfer test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report summary."""
    print("\n" + "="*60)
    print("BACKBONE OCCLUSION INTEGRATION TEST SUITE")
    print("="*60)

    tests = [
        ("Backbone Has OcclusionHead", test_backbone_has_occlusion_head),
        ("Forward Without Occlusion", test_forward_without_occlusion),
        ("Forward With Occlusion", test_forward_with_occlusion),
        ("Occlusion Spatial Resolution", test_occlusion_spatial_resolution),
        ("Gradient Flow Both Branches", test_gradient_flow_both_branches),
        ("Different Architectures", test_different_architectures),
        ("Feature Sharing", test_feature_sharing),
        ("Batch Sizes", test_batch_sizes),
        ("Device Transfer", test_device_transfer),
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
