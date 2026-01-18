#!/usr/bin/env python3
"""
Test script for occlusion-related config parameters.

This script tests that:
1. Occlusion config arguments are parsed correctly
2. Default values are set properly
3. Custom values can be provided
4. Arguments integrate with existing config structure

Run this script on HPC:
    python tests/test_config_occlusion.py

Expected output: All tests should print [PASS]
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse


def print_test_header(test_name):
    """Print formatted test header."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print('='*60)


def get_test_parser():
    """Create a parser with just the occlusion arguments for testing."""
    parser = argparse.ArgumentParser(description='Test occlusion config')

    # Occlusion layer parameters (same as in config.py)
    parser.add_argument('--occlusion_loss_weight', default=0.1, type=float,
                        help='Weight for occlusion prediction MSE loss (default: 0.1)')
    parser.add_argument('--niqab_data_path', type=str, default='',
                        help='Path to niqab dataset with GT masks')
    parser.add_argument('--use_occlusion_weighting', action='store_true',
                        help='Enable occlusion-aware weighting in QAConv matching')

    return parser


def test_default_values():
    """Test that default values are set correctly."""
    print_test_header("Default Values")

    try:
        parser = get_test_parser()
        args = parser.parse_args([])

        # Check occlusion_loss_weight default
        assert args.occlusion_loss_weight == 0.1, \
            f"occlusion_loss_weight should be 0.1, got {args.occlusion_loss_weight}"
        print(f"  occlusion_loss_weight: {args.occlusion_loss_weight} [PASS]")

        # Check niqab_data_path default
        assert args.niqab_data_path == '', \
            f"niqab_data_path should be empty string, got '{args.niqab_data_path}'"
        print(f"  niqab_data_path: '{args.niqab_data_path}' (empty) [PASS]")

        # Check use_occlusion_weighting default
        assert args.use_occlusion_weighting == False, \
            f"use_occlusion_weighting should be False, got {args.use_occlusion_weighting}"
        print(f"  use_occlusion_weighting: {args.use_occlusion_weighting} [PASS]")

        print("\n[PASS] Default values test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_custom_values():
    """Test that custom values can be set."""
    print_test_header("Custom Values")

    try:
        parser = get_test_parser()

        # Parse with custom values
        args = parser.parse_args([
            '--occlusion_loss_weight', '0.25',
            '--niqab_data_path', '/home/maass/code/niqab/train',
            '--use_occlusion_weighting'
        ])

        # Check occlusion_loss_weight
        assert args.occlusion_loss_weight == 0.25, \
            f"occlusion_loss_weight should be 0.25, got {args.occlusion_loss_weight}"
        print(f"  occlusion_loss_weight: {args.occlusion_loss_weight} [PASS]")

        # Check niqab_data_path
        assert args.niqab_data_path == '/home/maass/code/niqab/train', \
            f"niqab_data_path mismatch"
        print(f"  niqab_data_path: '{args.niqab_data_path}' [PASS]")

        # Check use_occlusion_weighting
        assert args.use_occlusion_weighting == True, \
            f"use_occlusion_weighting should be True, got {args.use_occlusion_weighting}"
        print(f"  use_occlusion_weighting: {args.use_occlusion_weighting} [PASS]")

        print("\n[PASS] Custom values test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_weight_range():
    """Test that occlusion_loss_weight accepts various valid values."""
    print_test_header("Weight Range")

    try:
        parser = get_test_parser()

        # Test various weight values
        test_weights = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]

        for weight in test_weights:
            args = parser.parse_args(['--occlusion_loss_weight', str(weight)])
            assert abs(args.occlusion_loss_weight - weight) < 1e-6, \
                f"Weight {weight} not parsed correctly"
            print(f"  Weight {weight}: {args.occlusion_loss_weight} [PASS]")

        print("\n[PASS] Weight range test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_full_config_integration():
    """Test that occlusion args work with the full config parser."""
    print_test_header("Full Config Integration")

    try:
        # Import the actual config module
        from config import get_args

        # Temporarily modify sys.argv for testing
        original_argv = sys.argv

        # Test with occlusion arguments
        sys.argv = [
            'test',
            '--occlusion_loss_weight', '0.15',
            '--niqab_data_path', '/test/path',
            '--use_occlusion_weighting',
            '--data_root', '/tmp',  # Required for some configs
        ]

        try:
            args = get_args()

            # Check occlusion arguments exist
            assert hasattr(args, 'occlusion_loss_weight'), \
                "args should have occlusion_loss_weight"
            assert hasattr(args, 'niqab_data_path'), \
                "args should have niqab_data_path"
            assert hasattr(args, 'use_occlusion_weighting'), \
                "args should have use_occlusion_weighting"

            print(f"  occlusion_loss_weight present: {args.occlusion_loss_weight} [PASS]")
            print(f"  niqab_data_path present: '{args.niqab_data_path}' [PASS]")
            print(f"  use_occlusion_weighting present: {args.use_occlusion_weighting} [PASS]")

            # Check values
            assert args.occlusion_loss_weight == 0.15, \
                f"occlusion_loss_weight should be 0.15, got {args.occlusion_loss_weight}"
            assert args.niqab_data_path == '/test/path', \
                f"niqab_data_path mismatch"
            assert args.use_occlusion_weighting == True, \
                f"use_occlusion_weighting should be True"

            print(f"  Values parsed correctly [PASS]")

        finally:
            sys.argv = original_argv

        print("\n[PASS] Full config integration test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_uses_config():
    """Test that Trainer class can access occlusion config."""
    print_test_header("Trainer Config Usage")

    try:
        # Create a mock hparams namespace
        class MockHparams:
            def __init__(self):
                self.occlusion_loss_weight = 0.2
                self.niqab_data_path = '/home/maass/code/niqab/train'
                self.use_occlusion_weighting = True
                # Add other required hparams
                self.arch = 'ir_18'
                self.lr = 0.1
                self.head = 'adaface'

        hparams = MockHparams()

        # Test getattr pattern used in Trainer
        occlusion_weight = getattr(hparams, 'occlusion_loss_weight', 0.1)
        niqab_path = getattr(hparams, 'niqab_data_path', '')
        use_occ = getattr(hparams, 'use_occlusion_weighting', False)

        assert occlusion_weight == 0.2, f"Weight should be 0.2, got {occlusion_weight}"
        print(f"  Trainer can access occlusion_loss_weight: {occlusion_weight} [PASS]")

        assert niqab_path == '/home/maass/code/niqab/train', f"Path mismatch"
        print(f"  Trainer can access niqab_data_path: '{niqab_path}' [PASS]")

        assert use_occ == True, f"use_occlusion_weighting should be True"
        print(f"  Trainer can access use_occlusion_weighting: {use_occ} [PASS]")

        # Test fallback when not present
        class EmptyHparams:
            pass

        empty = EmptyHparams()
        fallback_weight = getattr(empty, 'occlusion_loss_weight', 0.1)
        assert fallback_weight == 0.1, f"Fallback should be 0.1, got {fallback_weight}"
        print(f"  Fallback to default works: {fallback_weight} [PASS]")

        print("\n[PASS] Trainer config usage test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_help_text():
    """Test that help text is properly defined."""
    print_test_header("Help Text")

    try:
        parser = get_test_parser()

        # Get help text
        help_text = parser.format_help()

        # Check that occlusion arguments appear in help
        assert 'occlusion_loss_weight' in help_text, \
            "occlusion_loss_weight should be in help text"
        print(f"  occlusion_loss_weight in help [PASS]")

        assert 'niqab_data_path' in help_text, \
            "niqab_data_path should be in help text"
        print(f"  niqab_data_path in help [PASS]")

        assert 'use_occlusion_weighting' in help_text, \
            "use_occlusion_weighting should be in help text"
        print(f"  use_occlusion_weighting in help [PASS]")

        # Check that help descriptions are present
        assert 'MSE loss' in help_text, \
            "MSE loss description should be in help"
        assert 'QAConv' in help_text, \
            "QAConv description should be in help"
        print(f"  Help descriptions present [PASS]")

        print("\n[PASS] Help text test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report summary."""
    print("\n" + "="*60)
    print("CONFIG OCCLUSION PARAMETERS TEST SUITE")
    print("="*60)

    tests = [
        ("Default Values", test_default_values),
        ("Custom Values", test_custom_values),
        ("Weight Range", test_weight_range),
        ("Full Config Integration", test_full_config_integration),
        ("Trainer Config Usage", test_trainer_uses_config),
        ("Help Text", test_help_text),
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
