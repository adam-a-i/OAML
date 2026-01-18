"""
Diagnose QAConv make_kernel behavior.
Compare old vs new calling patterns.
"""

import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qaconv import QAConv


def test_old_pattern():
    """Test OLD calling pattern: matcher(feature, feature)"""
    print("=" * 80)
    print("TEST: Old pattern - matcher(feature, feature)")
    print("=" * 80)

    torch.manual_seed(42)

    qaconv = QAConv(num_features=512, height=7, width=7, num_classes=100)
    qaconv.train()

    # Create test features
    feature = torch.randn(8, 512, 7, 7)
    feature = F.normalize(feature, p=2, dim=1)

    # Old pattern: pass feature as both probe and gallery
    score_old = qaconv(feature, feature)

    print(f"Score shape: {score_old.shape}")
    print(f"Score sample (diagonal): {torch.diag(score_old)[:5]}")
    print(f"Score sample (off-diagonal row 0): {score_old[0, 1:5]}")

    return score_old


def test_new_pattern():
    """Test NEW calling pattern: make_kernel(feature), then matcher(feature)"""
    print("\n" + "=" * 80)
    print("TEST: New pattern - make_kernel(feature), then matcher(feature)")
    print("=" * 80)

    torch.manual_seed(42)

    qaconv = QAConv(num_features=512, height=7, width=7, num_classes=100)
    qaconv.train()

    # Create test features
    feature = torch.randn(8, 512, 7, 7)
    feature = F.normalize(feature, p=2, dim=1)

    # New pattern: set kernel first, then call forward without gallery
    qaconv.make_kernel(feature)
    score_new = qaconv(feature)  # No gallery - uses stored kernel

    print(f"Score shape: {score_new.shape}")
    print(f"Score sample (diagonal): {torch.diag(score_new)[:5]}")
    print(f"Score sample (off-diagonal row 0): {score_new[0, 1:5]}")

    return score_new


def test_pattern_equivalence():
    """Test if old and new patterns produce the same results."""
    print("\n" + "=" * 80)
    print("TEST: Pattern Equivalence")
    print("=" * 80)

    torch.manual_seed(42)

    qaconv = QAConv(num_features=512, height=7, width=7, num_classes=100)
    qaconv.eval()  # Use eval mode to eliminate any training-specific behavior

    # Create test features
    feature = torch.randn(8, 512, 7, 7)
    feature = F.normalize(feature, p=2, dim=1)

    # Clone for second test to ensure no state sharing
    feature_clone = feature.clone()

    # Old pattern
    with torch.no_grad():
        score_old = qaconv(feature, feature)

    # Reset any state
    qaconv._kernel = None

    # New pattern
    with torch.no_grad():
        qaconv.make_kernel(feature_clone)
        score_new = qaconv(feature_clone)

    # Compare
    diff = (score_old - score_new).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")

    if max_diff < 1e-5:
        print("PASS: Old and new patterns produce identical results")
        return True
    else:
        print("FAIL: Old and new patterns produce DIFFERENT results!")
        print(f"Old diagonal: {torch.diag(score_old)[:5]}")
        print(f"New diagonal: {torch.diag(score_new)[:5]}")
        return False


def test_kernel_state_leak():
    """Test if kernel state can leak between iterations."""
    print("\n" + "=" * 80)
    print("TEST: Kernel State Leak")
    print("=" * 80)

    torch.manual_seed(42)

    qaconv = QAConv(num_features=512, height=7, width=7, num_classes=100)
    qaconv.train()

    # Iteration 1: normal usage
    feature1 = torch.randn(8, 512, 7, 7)
    feature1 = F.normalize(feature1, p=2, dim=1)

    qaconv.make_kernel(feature1)
    score1 = qaconv(feature1)

    print(f"After iteration 1:")
    print(f"  _kernel is None: {qaconv._kernel is None}")

    # Iteration 2: normal usage
    feature2 = torch.randn(8, 512, 7, 7)
    feature2 = F.normalize(feature2, p=2, dim=1)

    qaconv.make_kernel(feature2)
    score2 = qaconv(feature2)

    print(f"After iteration 2:")
    print(f"  _kernel is None: {qaconv._kernel is None}")

    # Test: what if make_kernel is called but forward is NOT called?
    print("\n--- Testing state leak scenario ---")
    feature3 = torch.randn(8, 512, 7, 7)
    feature3 = F.normalize(feature3, p=2, dim=1)

    qaconv.make_kernel(feature3)
    print(f"After make_kernel (no forward): _kernel is None: {qaconv._kernel is None}")

    # Now on next iteration, if we call forward without make_kernel, it would use stale kernel
    feature4 = torch.randn(8, 512, 7, 7)
    feature4 = F.normalize(feature4, p=2, dim=1)

    # This would use the stale kernel from feature3!
    score_wrong = qaconv(feature4)  # No make_kernel call, but _kernel is set!
    print(f"Forward without make_kernel: _kernel is None: {qaconv._kernel is None}")

    # Compare to correct behavior
    qaconv.make_kernel(feature4)
    score_correct = qaconv(feature4)

    diff = (score_wrong - score_correct).abs().max().item()
    print(f"Score difference (wrong vs correct): {diff:.4f}")

    if diff > 0.1:
        print("WARNING: Stale kernel causes significant score differences!")
        print("This could cause training issues if make_kernel is skipped!")

    return qaconv._kernel is None  # Should be None after normal usage


def test_gradient_flow():
    """Test gradient flow through make_kernel pattern."""
    print("\n" + "=" * 80)
    print("TEST: Gradient Flow")
    print("=" * 80)

    torch.manual_seed(42)

    qaconv = QAConv(num_features=512, height=7, width=7, num_classes=100)
    qaconv.train()

    # Create features with grad
    feature = torch.randn(8, 512, 7, 7, requires_grad=True)
    feature_normalized = F.normalize(feature, p=2, dim=1)

    # Use detach+clone like pairwise_matching_loss does
    feature_for_qaconv = feature_normalized.detach().clone().requires_grad_(True)

    qaconv.make_kernel(feature_for_qaconv)
    score = qaconv(feature_for_qaconv)

    # Compute a dummy loss
    loss = score.mean()
    loss.backward()

    # Check gradients
    print(f"QAConv fc has gradient: {qaconv.fc.weight.grad is not None}")
    print(f"QAConv bn has gradient: {qaconv.bn.weight.grad is not None}")
    print(f"feature_for_qaconv has gradient: {feature_for_qaconv.grad is not None}")
    print(f"original feature has gradient: {feature.grad is not None}")  # Should be None due to detach

    if qaconv.fc.weight.grad is not None:
        print(f"QAConv fc gradient magnitude: {qaconv.fc.weight.grad.abs().mean().item():.6e}")

    return qaconv.fc.weight.grad is not None


def main():
    results = {}

    score_old = test_old_pattern()
    score_new = test_new_pattern()

    results['pattern_equivalence'] = test_pattern_equivalence()
    results['kernel_state'] = test_kernel_state_leak()
    results['gradient_flow'] = test_gradient_flow()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for test, passed in results.items():
        print(f"  {test}: {'PASS' if passed else 'FAIL/WARNING'}")

    if all(results.values()):
        print("\nAll tests passed - QAConv make_kernel pattern is working correctly")
    else:
        print("\nSome tests failed - there may be issues with QAConv")


if __name__ == '__main__':
    main()
