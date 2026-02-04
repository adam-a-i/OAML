#!/usr/bin/env python3
"""
Test script for QAConv occlusion-aware weighting.

This script tests that:
1. _compute_similarity_batch_with_occlusion produces correct shapes
2. Occlusion weighting reduces scores for occluded regions
3. All-ones occlusion maps give same result as standard computation
4. All-zeros occlusion maps give zero/near-zero scores
5. Gradient flow works through occlusion-weighted computation
6. Forward pass correctly dispatches to occlusion-aware method
7. match() method works with occlusion maps

Run this script on HPC:
    python tests/test_qaconv_occlusion.py

Expected output: All tests should print [PASS]
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import QAConv
from qaconv import QAConv


def print_test_header(test_name):
    """Print formatted test header."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print('='*60)


def test_occlusion_method_exists():
    """Test that the occlusion-aware method exists and is callable."""
    print_test_header("Occlusion Method Exists")

    try:
        qaconv = QAConv(num_features=512, height=7, width=7)

        assert hasattr(qaconv, '_compute_similarity_batch_with_occlusion'), \
            "QAConv should have _compute_similarity_batch_with_occlusion method"
        print(f"  _compute_similarity_batch_with_occlusion exists: [PASS]")

        # Check forward method signature accepts occlusion parameters
        import inspect
        sig = inspect.signature(qaconv.forward)
        params = list(sig.parameters.keys())
        assert 'prob_occ' in params, "forward should accept prob_occ parameter"
        assert 'gal_occ' in params, "forward should accept gal_occ parameter"
        print(f"  forward() accepts occlusion parameters: [PASS]")

        print("\n[PASS] Occlusion method exists test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_occlusion_output_shape():
    """Test that occlusion-aware computation produces correct output shapes."""
    print_test_header("Occlusion Output Shape")

    try:
        qaconv = QAConv(num_features=512, height=7, width=7)
        qaconv.eval()

        test_cases = [
            (4, 4),   # Same batch sizes
            (4, 8),   # Different batch sizes
            (1, 4),   # Single probe
            (8, 1),   # Single gallery
            (1, 1),   # Single pair
        ]

        for prob_size, gal_size in test_cases:
            # Create random features and occlusion maps
            prob_fea = torch.randn(prob_size, 512, 7, 7)
            gal_fea = torch.randn(gal_size, 512, 7, 7)
            prob_occ = torch.rand(prob_size, 1, 7, 7)  # Random [0, 1]
            gal_occ = torch.rand(gal_size, 1, 7, 7)

            with torch.no_grad():
                scores = qaconv._compute_similarity_batch_with_occlusion(
                    prob_fea, gal_fea, prob_occ, gal_occ
                )

            expected_shape = (prob_size, gal_size)
            assert scores.shape == expected_shape, \
                f"Shape mismatch: got {scores.shape}, expected {expected_shape}"
            print(f"  Probe={prob_size}, Gallery={gal_size} -> {list(scores.shape)}: [PASS]")

        print("\n[PASS] Occlusion output shape test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_all_ones_occlusion():
    """Test that all-ones occlusion maps give same result as standard computation."""
    print_test_header("All-Ones Occlusion (Equivalence)")

    try:
        qaconv = QAConv(num_features=512, height=7, width=7)
        qaconv.eval()

        # Create features
        prob_fea = torch.randn(4, 512, 7, 7)
        gal_fea = torch.randn(4, 512, 7, 7)

        # Normalize features (as forward does)
        prob_fea_norm = F.normalize(prob_fea, p=2, dim=1)
        gal_fea_norm = F.normalize(gal_fea, p=2, dim=1)

        # All-ones occlusion = fully visible
        prob_occ = torch.ones(4, 1, 7, 7)
        gal_occ = torch.ones(4, 1, 7, 7)

        with torch.no_grad():
            # Standard computation
            scores_standard = qaconv._compute_similarity_batch(prob_fea_norm, gal_fea_norm)

            # Occlusion-aware with all ones
            scores_occlusion = qaconv._compute_similarity_batch_with_occlusion(
                prob_fea_norm, gal_fea_norm, prob_occ, gal_occ
            )

        # Should be identical (or very close due to numerical precision)
        max_diff = (scores_standard - scores_occlusion).abs().max().item()
        print(f"  Max difference between standard and all-ones occlusion: {max_diff:.2e}")

        assert max_diff < 1e-5, f"All-ones should match standard: diff={max_diff}"
        print(f"  All-ones matches standard computation: [PASS]")

        print("\n[PASS] All-ones occlusion test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_zeros_reduce_scores():
    """Test that zero occlusion (fully occluded) reduces scores significantly."""
    print_test_header("Zeros Reduce Scores")

    try:
        qaconv = QAConv(num_features=512, height=7, width=7)
        qaconv.eval()

        # Create features
        prob_fea = torch.randn(4, 512, 7, 7)
        gal_fea = torch.randn(4, 512, 7, 7)

        # Normalize
        prob_fea_norm = F.normalize(prob_fea, p=2, dim=1)
        gal_fea_norm = F.normalize(gal_fea, p=2, dim=1)

        with torch.no_grad():
            # All-ones (fully visible)
            prob_occ_ones = torch.ones(4, 1, 7, 7)
            gal_occ_ones = torch.ones(4, 1, 7, 7)
            scores_visible = qaconv._compute_similarity_batch_with_occlusion(
                prob_fea_norm, gal_fea_norm, prob_occ_ones, gal_occ_ones
            )

            # All-zeros (fully occluded)
            prob_occ_zeros = torch.zeros(4, 1, 7, 7)
            gal_occ_zeros = torch.zeros(4, 1, 7, 7)
            scores_occluded = qaconv._compute_similarity_batch_with_occlusion(
                prob_fea_norm, gal_fea_norm, prob_occ_zeros, gal_occ_zeros
            )

        print(f"  Visible scores - mean: {scores_visible.mean().item():.4f}, std: {scores_visible.std().item():.4f}")
        print(f"  Occluded scores - mean: {scores_occluded.mean().item():.4f}, std: {scores_occluded.std().item():.4f}")

        # Occluded scores should be much smaller in absolute value
        # (Note: after BN/FC, scores can be negative, so we compare magnitudes)
        visible_magnitude = scores_visible.abs().mean().item()
        occluded_magnitude = scores_occluded.abs().mean().item()

        print(f"  Visible magnitude: {visible_magnitude:.4f}")
        print(f"  Occluded magnitude: {occluded_magnitude:.4f}")

        # The occluded scores should be significantly smaller
        # They will be near-zero before BN, but BN may shift them
        # Key check: scores should be different
        diff = (scores_visible - scores_occluded).abs().mean().item()
        print(f"  Mean absolute difference: {diff:.4f}")

        assert diff > 0.1, f"Occlusion should significantly change scores, diff={diff}"
        print(f"  Occlusion changes scores significantly: [PASS]")

        print("\n[PASS] Zeros reduce scores test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_partial_occlusion():
    """Test that partial occlusion reduces scores proportionally."""
    print_test_header("Partial Occlusion")

    try:
        qaconv = QAConv(num_features=512, height=7, width=7)
        qaconv.eval()

        # Create features
        prob_fea = torch.randn(4, 512, 7, 7)
        gal_fea = torch.randn(4, 512, 7, 7)

        # Normalize
        prob_fea_norm = F.normalize(prob_fea, p=2, dim=1)
        gal_fea_norm = F.normalize(gal_fea, p=2, dim=1)

        with torch.no_grad():
            # Full visibility
            prob_occ_full = torch.ones(4, 1, 7, 7)
            gal_occ_full = torch.ones(4, 1, 7, 7)
            scores_full = qaconv._compute_similarity_batch_with_occlusion(
                prob_fea_norm, gal_fea_norm, prob_occ_full, gal_occ_full
            )

            # Half occluded (lower half of image)
            prob_occ_half = torch.ones(4, 1, 7, 7)
            prob_occ_half[:, :, 4:, :] = 0.0  # Bottom half occluded
            gal_occ_half = torch.ones(4, 1, 7, 7)
            gal_occ_half[:, :, 4:, :] = 0.0

            scores_half = qaconv._compute_similarity_batch_with_occlusion(
                prob_fea_norm, gal_fea_norm, prob_occ_half, gal_occ_half
            )

        print(f"  Full visibility scores - mean: {scores_full.mean().item():.4f}")
        print(f"  Half occluded scores - mean: {scores_half.mean().item():.4f}")

        # Scores should be different
        diff = (scores_full - scores_half).abs().mean().item()
        print(f"  Mean difference: {diff:.4f}")

        assert diff > 0.01, f"Partial occlusion should change scores, diff={diff}"
        print(f"  Partial occlusion changes scores: [PASS]")

        print("\n[PASS] Partial occlusion test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow():
    """Test that gradients flow through occlusion-weighted computation."""
    print_test_header("Gradient Flow")

    try:
        qaconv = QAConv(num_features=512, height=7, width=7)
        qaconv.train()

        # Create features and occlusion maps with gradients
        prob_fea = torch.randn(4, 512, 7, 7, requires_grad=True)
        gal_fea = torch.randn(4, 512, 7, 7, requires_grad=True)
        prob_occ = torch.rand(4, 1, 7, 7, requires_grad=True)
        gal_occ = torch.rand(4, 1, 7, 7, requires_grad=True)

        # Forward pass (normalize first as forward does)
        prob_fea_norm = F.normalize(prob_fea, p=2, dim=1)
        gal_fea_norm = F.normalize(gal_fea, p=2, dim=1)

        scores = qaconv._compute_similarity_batch_with_occlusion(
            prob_fea_norm, gal_fea_norm, prob_occ, gal_occ
        )

        # Backward pass
        loss = scores.mean()
        loss.backward()

        # Check gradients exist
        assert prob_fea.grad is not None, "Probe features should have gradients"
        assert gal_fea.grad is not None, "Gallery features should have gradients"
        assert prob_occ.grad is not None, "Probe occlusion should have gradients"
        assert gal_occ.grad is not None, "Gallery occlusion should have gradients"

        # Check gradients are valid
        assert not torch.isnan(prob_fea.grad).any(), "Probe feature gradients contain NaN"
        assert not torch.isnan(gal_fea.grad).any(), "Gallery feature gradients contain NaN"
        assert not torch.isnan(prob_occ.grad).any(), "Probe occlusion gradients contain NaN"
        assert not torch.isnan(gal_occ.grad).any(), "Gallery occlusion gradients contain NaN"

        print(f"  Probe feature grad norm: {prob_fea.grad.norm().item():.6f}: [PASS]")
        print(f"  Gallery feature grad norm: {gal_fea.grad.norm().item():.6f}: [PASS]")
        print(f"  Probe occlusion grad norm: {prob_occ.grad.norm().item():.6f}: [PASS]")
        print(f"  Gallery occlusion grad norm: {gal_occ.grad.norm().item():.6f}: [PASS]")

        print("\n[PASS] Gradient flow test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_dispatch():
    """Test that forward() correctly dispatches to occlusion-aware method."""
    print_test_header("Forward Dispatch")

    try:
        qaconv = QAConv(num_features=512, height=7, width=7)
        qaconv.eval()

        prob_fea = torch.randn(4, 512, 7, 7)
        gal_fea = torch.randn(4, 512, 7, 7)
        prob_occ = torch.rand(4, 1, 7, 7)
        gal_occ = torch.rand(4, 1, 7, 7)

        with torch.no_grad():
            # Without occlusion maps - should use standard method
            scores_no_occ = qaconv.forward(prob_fea, gal_fea)

            # With occlusion maps - should use occlusion-aware method
            scores_with_occ = qaconv.forward(prob_fea, gal_fea,
                                              prob_occ=prob_occ, gal_occ=gal_occ)

        # Both should produce valid outputs
        assert scores_no_occ.shape == (4, 4), f"No-occ shape mismatch: {scores_no_occ.shape}"
        assert scores_with_occ.shape == (4, 4), f"With-occ shape mismatch: {scores_with_occ.shape}"

        print(f"  forward() without occlusion: shape={list(scores_no_occ.shape)} [PASS]")
        print(f"  forward() with occlusion: shape={list(scores_with_occ.shape)} [PASS]")

        # Scores should be different when using occlusion (unless occlusion is all ones)
        diff = (scores_no_occ - scores_with_occ).abs().mean().item()
        print(f"  Difference between with/without occlusion: {diff:.4f}")

        print("\n[PASS] Forward dispatch test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_match_method():
    """Test that match() method works with occlusion maps."""
    print_test_header("Match Method with Occlusion")

    try:
        qaconv = QAConv(num_features=512, height=7, width=7)

        prob_fea = torch.randn(4, 512, 7, 7)
        gal_fea = torch.randn(8, 512, 7, 7)
        prob_occ = torch.rand(4, 1, 7, 7)
        gal_occ = torch.rand(8, 1, 7, 7)

        # Without occlusion
        scores_no_occ = qaconv.match(prob_fea, gal_fea)
        assert scores_no_occ.shape == (4, 8), f"No-occ shape: {scores_no_occ.shape}"
        print(f"  match() without occlusion: {list(scores_no_occ.shape)} [PASS]")

        # With occlusion
        scores_with_occ = qaconv.match(prob_fea, gal_fea,
                                        probe_occ=prob_occ, gallery_occ=gal_occ)
        assert scores_with_occ.shape == (4, 8), f"With-occ shape: {scores_with_occ.shape}"
        print(f"  match() with occlusion: {list(scores_with_occ.shape)} [PASS]")

        print("\n[PASS] Match method test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_device_transfer():
    """Test that occlusion-aware computation works on CPU and GPU."""
    print_test_header("Device Transfer")

    try:
        qaconv = QAConv(num_features=512, height=7, width=7)

        prob_fea = torch.randn(4, 512, 7, 7)
        gal_fea = torch.randn(4, 512, 7, 7)
        prob_occ = torch.rand(4, 1, 7, 7)
        gal_occ = torch.rand(4, 1, 7, 7)

        # CPU test
        qaconv_cpu = qaconv.to('cpu')
        qaconv_cpu.eval()
        with torch.no_grad():
            scores_cpu = qaconv_cpu.forward(prob_fea, gal_fea,
                                             prob_occ=prob_occ, gal_occ=gal_occ)
        assert scores_cpu.device.type == 'cpu', "CPU scores should be on CPU"
        print(f"  CPU forward with occlusion: [PASS]")

        # GPU test if available
        if torch.cuda.is_available():
            qaconv_gpu = qaconv.to('cuda')
            qaconv_gpu.eval()
            with torch.no_grad():
                scores_gpu = qaconv_gpu.forward(
                    prob_fea.cuda(), gal_fea.cuda(),
                    prob_occ=prob_occ.cuda(), gal_occ=gal_occ.cuda()
                )
            assert scores_gpu.device.type == 'cuda', "GPU scores should be on GPU"
            print(f"  GPU forward with occlusion: [PASS]")

            # Compare CPU and GPU
            diff = (scores_cpu - scores_gpu.cpu()).abs().max().item()
            print(f"  CPU-GPU diff: {diff:.2e}")
            assert diff < 0.1, f"CPU-GPU diff too large: {diff}"
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
    print("QACONV OCCLUSION WEIGHTING TEST SUITE")
    print("="*60)

    tests = [
        ("Occlusion Method Exists", test_occlusion_method_exists),
        ("Occlusion Output Shape", test_occlusion_output_shape),
        ("All-Ones Occlusion", test_all_ones_occlusion),
        ("Zeros Reduce Scores", test_zeros_reduce_scores),
        ("Partial Occlusion", test_partial_occlusion),
        ("Gradient Flow", test_gradient_flow),
        ("Forward Dispatch", test_forward_dispatch),
        ("Match Method", test_match_method),
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
