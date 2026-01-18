#!/usr/bin/env python3
"""
Test script for OcclusionHead class.

This script tests the OcclusionHead module to ensure:
1. Correct instantiation with different parameters
2. Input/output shape correctness
3. Output values are in [0, 1] range (sigmoid constraint)
4. Gradient flow during backpropagation
5. Behavior with different batch sizes
6. Spatial resolution preservation

Run this script on HPC:
    python tests/test_occlusion_head.py

Expected output: All tests should print [PASS]
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np

# Import the OcclusionHead class
from net import OcclusionHead


def print_test_header(test_name):
    """Print formatted test header."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print('='*60)


def test_instantiation():
    """Test that OcclusionHead instantiates correctly with default and custom parameters."""
    print_test_header("Instantiation")

    try:
        # Test default parameters
        head_default = OcclusionHead()
        assert head_default.conv1.in_channels == 512, "Default in_channels should be 512"
        assert head_default.conv1.out_channels == 128, "Default hidden_channels should be 128"
        assert head_default.conv2.out_channels == 1, "Output should have 1 channel"
        print(f"  Default instantiation: [PASS]")

        # Test custom parameters
        head_custom = OcclusionHead(in_channels=256, hidden_channels=64)
        assert head_custom.conv1.in_channels == 256, "Custom in_channels not set correctly"
        assert head_custom.conv1.out_channels == 64, "Custom hidden_channels not set correctly"
        print(f"  Custom parameters (in=256, hidden=64): [PASS]")

        # Test for IR-101/152 with 2048 channels
        head_large = OcclusionHead(in_channels=2048, hidden_channels=256)
        assert head_large.conv1.in_channels == 2048, "Large model in_channels not set correctly"
        print(f"  Large model parameters (in=2048, hidden=256): [PASS]")

        print("\n[PASS] Instantiation test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Instantiation test failed: {str(e)}")
        return False


def test_output_shape():
    """Test that output shape matches specification for various input sizes."""
    print_test_header("Output Shape")

    try:
        head = OcclusionHead(in_channels=512, hidden_channels=128)

        # Test case 1: Standard 112x112 input -> 7x7 feature maps
        batch_sizes = [1, 4, 16, 32]
        spatial_sizes = [(7, 7), (14, 14), (3, 3)]  # Different feature map sizes

        for batch_size in batch_sizes:
            for h, w in spatial_sizes:
                x = torch.randn(batch_size, 512, h, w)
                out = head(x)

                expected_shape = (batch_size, 1, h, w)
                assert out.shape == expected_shape, \
                    f"Shape mismatch: got {out.shape}, expected {expected_shape}"
                print(f"  Input [{batch_size}, 512, {h}, {w}] -> Output {list(out.shape)}: [PASS]")

        print("\n[PASS] Output shape test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Output shape test failed: {str(e)}")
        return False


def test_output_range():
    """Test that output values are in [0, 1] range due to sigmoid."""
    print_test_header("Output Range")

    try:
        head = OcclusionHead()

        # Test with various input distributions
        test_cases = [
            ("Normal distribution", torch.randn(8, 512, 7, 7)),
            ("Uniform [0, 1]", torch.rand(8, 512, 7, 7)),
            ("Uniform [-1, 1]", torch.rand(8, 512, 7, 7) * 2 - 1),
            ("Large values", torch.randn(8, 512, 7, 7) * 100),
            ("Small values", torch.randn(8, 512, 7, 7) * 0.01),
            ("All zeros", torch.zeros(8, 512, 7, 7)),
            ("All ones", torch.ones(8, 512, 7, 7)),
        ]

        for name, x in test_cases:
            out = head(x)
            min_val = out.min().item()
            max_val = out.max().item()

            assert min_val >= 0.0, f"Output contains values < 0: min={min_val}"
            assert max_val <= 1.0, f"Output contains values > 1: max={max_val}"
            print(f"  {name}: range=[{min_val:.4f}, {max_val:.4f}]: [PASS]")

        print("\n[PASS] Output range test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Output range test failed: {str(e)}")
        return False


def test_gradient_flow():
    """Test that gradients flow correctly during backpropagation."""
    print_test_header("Gradient Flow")

    try:
        head = OcclusionHead()
        head.train()

        # Create input with requires_grad
        x = torch.randn(4, 512, 7, 7, requires_grad=True)

        # Forward pass
        out = head(x)

        # Create dummy loss (MSE with target of all 0.5)
        target = torch.ones_like(out) * 0.5
        loss = nn.functional.mse_loss(out, target)

        # Backward pass
        loss.backward()

        # Check gradients exist for input
        assert x.grad is not None, "Input gradient is None"
        assert not torch.isnan(x.grad).any(), "Input gradient contains NaN"
        assert not torch.isinf(x.grad).any(), "Input gradient contains Inf"
        print(f"  Input gradient: shape={list(x.grad.shape)}, norm={x.grad.norm().item():.6f}: [PASS]")

        # Check gradients exist for all parameters
        for name, param in head.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient for {name} is None"
                assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN"
                assert not torch.isinf(param.grad).any(), f"Gradient for {name} contains Inf"
                print(f"  {name}: grad_norm={param.grad.norm().item():.6f}: [PASS]")

        print("\n[PASS] Gradient flow test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Gradient flow test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_parameter_count():
    """Test that parameter count is reasonable (lightweight design)."""
    print_test_header("Parameter Count")

    try:
        head = OcclusionHead(in_channels=512, hidden_channels=128)

        total_params = sum(p.numel() for p in head.parameters())
        trainable_params = sum(p.numel() for p in head.parameters() if p.requires_grad)

        # Expected parameter count:
        # conv1: 512 * 128 * 3 * 3 = 589,824 (no bias)
        # bn1: 128 * 2 = 256 (weight + bias)
        # conv2: 128 * 1 * 1 * 1 + 1 = 129 (with bias)
        # Total expected: ~590,209

        expected_approx = 590209
        tolerance = 1000  # Allow small variance

        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Expected (approx): {expected_approx:,}")

        assert abs(total_params - expected_approx) < tolerance, \
            f"Parameter count {total_params} differs significantly from expected {expected_approx}"

        # Verify it's lightweight (< 1M parameters)
        assert total_params < 1_000_000, "OcclusionHead should be lightweight (< 1M params)"
        print(f"  Lightweight check (< 1M params): [PASS]")

        print("\n[PASS] Parameter count test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Parameter count test failed: {str(e)}")
        return False


def test_determinism():
    """Test that the module produces deterministic output with same input."""
    print_test_header("Determinism")

    try:
        # Set seed for reproducibility
        torch.manual_seed(42)
        head = OcclusionHead()
        head.eval()  # Set to eval mode to disable dropout (if any)

        # Create fixed input
        x = torch.randn(4, 512, 7, 7)

        # Multiple forward passes should give same result
        with torch.no_grad():
            out1 = head(x)
            out2 = head(x)
            out3 = head(x)

        assert torch.allclose(out1, out2, atol=1e-6), "Outputs differ between runs 1 and 2"
        assert torch.allclose(out2, out3, atol=1e-6), "Outputs differ between runs 2 and 3"

        max_diff_12 = (out1 - out2).abs().max().item()
        max_diff_23 = (out2 - out3).abs().max().item()
        print(f"  Max diff (run 1 vs 2): {max_diff_12:.2e}: [PASS]")
        print(f"  Max diff (run 2 vs 3): {max_diff_23:.2e}: [PASS]")

        print("\n[PASS] Determinism test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Determinism test failed: {str(e)}")
        return False


def test_device_transfer():
    """Test that the module works correctly on CPU and GPU (if available)."""
    print_test_header("Device Transfer")

    try:
        head = OcclusionHead()
        x_cpu = torch.randn(4, 512, 7, 7)

        # Test on CPU
        head_cpu = head.to('cpu')
        out_cpu = head_cpu(x_cpu)
        assert out_cpu.device.type == 'cpu', "Output should be on CPU"
        print(f"  CPU forward pass: [PASS]")

        # Test on GPU if available
        if torch.cuda.is_available():
            head_gpu = head.to('cuda')
            x_gpu = x_cpu.to('cuda')
            out_gpu = head_gpu(x_gpu)

            assert out_gpu.device.type == 'cuda', "Output should be on GPU"
            print(f"  GPU forward pass: [PASS]")

            # Compare CPU and GPU outputs (should be very close)
            # Note: CPU vs GPU differences up to 1e-2 are normal due to different
            # floating point implementations and operation ordering
            out_gpu_cpu = out_gpu.to('cpu')
            max_diff = (out_cpu - out_gpu_cpu).abs().max().item()
            print(f"  CPU vs GPU max diff: {max_diff:.2e}")
            assert max_diff < 1e-2, f"CPU and GPU outputs differ significantly: {max_diff}"
            print(f"  CPU-GPU consistency: [PASS]")
        else:
            print(f"  GPU not available, skipping GPU tests")

        print("\n[PASS] Device transfer test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Device transfer test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_independence():
    """Test that samples in a batch are processed independently (except for BN)."""
    print_test_header("Batch Independence")

    try:
        head = OcclusionHead()
        head.eval()  # Use eval mode to make BN deterministic

        # Create input where each sample is different
        x = torch.randn(4, 512, 7, 7)

        with torch.no_grad():
            # Process full batch
            out_batch = head(x)

            # Process each sample individually
            out_individual = []
            for i in range(4):
                out_i = head(x[i:i+1])
                out_individual.append(out_i)
            out_individual = torch.cat(out_individual, dim=0)

        # In eval mode, batch processing should match individual processing
        max_diff = (out_batch - out_individual).abs().max().item()
        print(f"  Max diff (batch vs individual): {max_diff:.2e}")

        # Allow small tolerance due to numerical precision
        assert max_diff < 1e-5, f"Batch and individual processing differ: {max_diff}"
        print(f"  Batch independence in eval mode: [PASS]")

        print("\n[PASS] Batch independence test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Batch independence test failed: {str(e)}")
        return False


def run_all_tests():
    """Run all tests and report summary."""
    print("\n" + "="*60)
    print("OCCLUSION HEAD TEST SUITE")
    print("="*60)

    tests = [
        ("Instantiation", test_instantiation),
        ("Output Shape", test_output_shape),
        ("Output Range", test_output_range),
        ("Gradient Flow", test_gradient_flow),
        ("Parameter Count", test_parameter_count),
        ("Determinism", test_determinism),
        ("Device Transfer", test_device_transfer),
        ("Batch Independence", test_batch_independence),
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
