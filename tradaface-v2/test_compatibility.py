#!/usr/bin/env python3
"""
Test script to verify compatibility between net.py, train_val.py, and main.py
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_compatibility():
    """Test that the model can be created and run forward pass"""
    print("="*60)
    print("TESTING MODEL COMPATIBILITY")
    print("="*60)
    
    try:
        # Test imports
        print("[TEST] Testing imports...")
        import net
        import train_val
        import pairwise_matching_loss
        from transmatcher import TransMatcher
        print("[TEST] ✓ All imports successful")
        
        # Test model creation
        print("[TEST] Testing model creation...")
        backbone = net.build_model(model_name='ir_50')
        transmatcher_params = {
            'seq_len': 49,
            'd_model': 512,
            'num_decoder_layers': 3,
            'dim_feedforward': 2048,
        }
        model = net.AdaFaceWithTransMatcher(backbone, transmatcher_params)
        print("[TEST] ✓ Model created successfully")
        
        # Test forward pass
        print("[TEST] Testing forward pass...")
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 112, 112)
        
        with torch.no_grad():
            embedding, norm, feature_maps, transmatcher = model(input_tensor)
        
        print(f"[TEST] ✓ Forward pass successful")
        print(f"[TEST]   - Embedding shape: {embedding.shape}")
        print(f"[TEST]   - Norm shape: {norm.shape}")
        print(f"[TEST]   - Feature maps shape: {feature_maps.shape}")
        print(f"[TEST]   - TransMatcher type: {type(transmatcher)}")
        
        # Test feature map format
        expected_shape = (batch_size, 512, 7, 7)
        if feature_maps.shape == expected_shape:
            print(f"[TEST] ✓ Feature maps have correct shape: {expected_shape}")
        else:
            print(f"[TEST] ✗ Feature maps have wrong shape: {feature_maps.shape}, expected: {expected_shape}")
            return False
        
        return True
        
    except Exception as e:
        print(f"[TEST] ✗ Error during model compatibility test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trainer_compatibility():
    """Test that the trainer can be created"""
    print("\n" + "="*60)
    print("TESTING TRAINER COMPATIBILITY")
    print("="*60)
    
    try:
        # Import train_val inside the function to avoid scope issues
        import train_val
        
        # Create mock hparams
        class MockHParams:
            def __init__(self):
                self.arch = 'ir_50'
                self.head = 'adaface'
                self.m = 0.4
                self.h = 0.333
                self.t_alpha = 0.01
                self.s = 64.0
                self.lr = 0.1
                self.momentum = 0.9
                self.lr_milestones = [10, 20, 30]
                self.lr_gamma = 0.1
                self.batch_size = 4
                self.distributed_backend = None
                self.start_from_model_statedict = None
                self.data_root = '/tmp'  # Mock path
                self.class_num = 1000  # Mock class number
                # Add missing parameters that utils.get_num_class expects
                self.custom_num_class = -1
                self.train_data_path = '/tmp/faces_emore'  # Mock path
                self.train_data_subset = False
        
        hparams = MockHParams()
        
        # Test trainer creation
        print("[TEST] Testing trainer creation...")
        trainer = train_val.Trainer(**vars(hparams))
        print("[TEST] ✓ Trainer created successfully")
        
        # Test that model has correct components
        print("[TEST] Testing trainer model components...")
        if hasattr(trainer, 'model'):
            print("[TEST] ✓ Trainer has model attribute")
        else:
            print("[TEST] ✗ Trainer missing model attribute")
            return False
            
        if hasattr(trainer.model, 'backbone'):
            print("[TEST] ✓ Model has backbone")
        else:
            print("[TEST] ✗ Model missing backbone")
            return False
            
        if hasattr(trainer.model, 'transmatcher'):
            print("[TEST] ✓ Model has transmatcher")
        else:
            print("[TEST] ✗ Model missing transmatcher")
            return False
            
        if hasattr(trainer, 'head'):
            print("[TEST] ✓ Trainer has head")
        else:
            print("[TEST] ✗ Trainer missing head")
            return False
            
        if hasattr(trainer, 'pairwise_matching_loss'):
            print("[TEST] ✓ Trainer has pairwise_matching_loss")
        else:
            print("[TEST] ✗ Trainer missing pairwise_matching_loss")
            return False
        
        return True
        
    except Exception as e:
        print(f"[TEST] ✗ Error during trainer compatibility test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_map_format():
    """Test that feature maps are in the correct format for both training and validation"""
    print("\n" + "="*60)
    print("TESTING FEATURE MAP FORMAT")
    print("="*60)
    
    try:
        import net
        import train_val
        
        # Create model
        backbone = net.build_model(model_name='ir_50')
        transmatcher_params = {
            'seq_len': 49,
            'd_model': 512,
            'num_decoder_layers': 3,
            'dim_feedforward': 2048,
        }
        model = net.AdaFaceWithTransMatcher(backbone, transmatcher_params)
        
        # Test forward pass
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 112, 112)
        
        with torch.no_grad():
            embedding, norm, feature_maps, transmatcher = model(input_tensor)
        
        print(f"[TEST] Feature maps shape: {feature_maps.shape}")
        
        # Test that feature maps are in standard format (B, C, H, W)
        if len(feature_maps.shape) == 4 and feature_maps.shape[1] == 512:
            print("[TEST] ✓ Feature maps in correct format (B, C, H, W)")
        else:
            print("[TEST] ✗ Feature maps in wrong format")
            return False
        
        # Test permutation for TransMatcher
        feature_maps_perm = feature_maps.permute(0, 2, 3, 1).contiguous()
        print(f"[TEST] Permuted feature maps shape: {feature_maps_perm.shape}")
        
        if feature_maps_perm.shape == (batch_size, 7, 7, 512):
            print("[TEST] ✓ Permuted feature maps in correct format (B, H, W, C)")
        else:
            print("[TEST] ✗ Permuted feature maps in wrong format")
            return False
        
        return True
        
    except Exception as e:
        print(f"[TEST] ✗ Error during feature map format test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all compatibility tests"""
    print("STARTING COMPATIBILITY TESTS")
    print("="*80)
    
    tests = [
        test_model_compatibility,
        test_trainer_compatibility,
        test_feature_map_format,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"[TEST] ✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "="*80)
    print("COMPATIBILITY TEST RESULTS")
    print("="*80)
    
    test_names = [
        "Model Compatibility",
        "Trainer Compatibility", 
        "Feature Map Format"
    ]
    
    all_passed = True
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"[{i+1}/{len(tests)}] {name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED - COMPATIBILITY CONFIRMED!")
        print("The project components are compatible and ready for training.")
    else:
        print("❌ SOME TESTS FAILED - COMPATIBILITY ISSUES DETECTED!")
        print("Please fix the failing tests before proceeding with training.")
    print("="*80)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 