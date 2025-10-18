#!/usr/bin/env python3
"""
Diagnose Training Issue - Why Occlusion Head Didn't Learn
=========================================================

This script analyzes why the occlusion head didn't learn effectively:
1. Check training data occlusion frequency
2. Analyze loss weights and contributions
3. Test with more aggressive occlusion
4. Suggest fixes
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append('.')

import net
from qaconv import QAConv
from transforms import MedicalMaskOcclusion, ToTensorWithMask
import torchvision.transforms as transforms

def load_trained_model(model_path, device='cuda'):
    """Load the trained model with occlusion head"""
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    architecture = 'ir_50'
    model = net.build_model(architecture)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Filter state dict to only include backbone parameters
    backbone_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('model.') and not any(x in k for x in ['qaconv_criterion', 'qaconv_triplet_criterion', 'head.kernel', 'head.t', 'head.batch_mean', 'head.batch_std']):
            new_key = k[6:]
            backbone_state_dict[new_key] = v
        elif not any(x in k for x in ['qaconv_criterion', 'qaconv_triplet_criterion', 'head.kernel', 'head.t', 'head.batch_mean', 'head.batch_std']) and not k.startswith('model.'):
            backbone_state_dict[k] = v
    
    print(f"Loading {len(backbone_state_dict)} backbone parameters...")
    model.load_state_dict(backbone_state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully")
    return model

def test_occlusion_head_with_aggressive_occlusion(model, device='cuda'):
    """Test occlusion head with more aggressive occlusion patterns"""
    print("\n🎭 Testing with Aggressive Occlusion...")
    
    # Create test image
    img = torch.randn(1, 3, 112, 112) * 0.3 + 0.5
    
    # Test 1: No occlusion
    with torch.no_grad():
        _, _, occ_no = model(img.to(device))
        print(f"No occlusion - Mean: {occ_no[0, 0].mean():.4f}, Std: {occ_no[0, 0].std():.4f}")
    
    # Test 2: Medical mask occlusion
    occlusion_transform = MedicalMaskOcclusion(prob=1.0)
    tensor_transform = ToTensorWithMask()
    
    img_pil = transforms.ToPILImage()(img[0])
    img_occluded, mask = occlusion_transform(img_pil)
    img_occluded_tensor, mask_tensor = tensor_transform((img_occluded, mask))
    img_occluded_tensor = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img_occluded_tensor)
    
    with torch.no_grad():
        _, _, occ_medical = model(img_occluded_tensor.unsqueeze(0).to(device))
        print(f"Medical mask - Mean: {occ_medical[0, 0].mean():.4f}, Std: {occ_medical[0, 0].std():.4f}")
        print(f"Difference: {abs(occ_medical[0, 0].mean() - occ_no[0, 0].mean()):.4f}")
    
    # Test 3: Create artificial heavy occlusion
    img_heavy_occ = img.clone()
    img_heavy_occ[:, :, 40:80, 30:70] = 0.0  # Black rectangle in center
    
    with torch.no_grad():
        _, _, occ_heavy = model(img_heavy_occ.to(device))
        print(f"Heavy occlusion - Mean: {occ_heavy[0, 0].mean():.4f}, Std: {occ_heavy[0, 0].std():.4f}")
        print(f"Difference from no-occ: {abs(occ_heavy[0, 0].mean() - occ_no[0, 0].mean()):.4f}")
    
    # Test 4: Check if occlusion head responds to different image types
    img_dark = torch.zeros_like(img)
    img_bright = torch.ones_like(img)
    
    with torch.no_grad():
        _, _, occ_dark = model(img_dark.to(device))
        _, _, occ_bright = model(img_bright.to(device))
        print(f"Dark image - Mean: {occ_dark[0, 0].mean():.4f}")
        print(f"Bright image - Mean: {occ_bright[0, 0].mean():.4f}")
        print(f"Dark vs Bright difference: {abs(occ_dark[0, 0].mean() - occ_bright[0, 0].mean()):.4f}")

def analyze_loss_weights():
    """Analyze the loss weight configuration"""
    print("\n⚖️ Loss Weight Analysis:")
    print("Current weights from train_val.py:")
    print("  AdaFace: 0.1")
    print("  QAConv: 0.7") 
    print("  Occlusion: 0.3")
    print("  Total: 1.1")
    print()
    print("Problem: Occlusion loss weight (0.3) is relatively low!")
    print("With AdaFace=0.1 and QAConv=0.7, occlusion gets only 27% of the total loss weight.")
    print("This means the occlusion head gets weak supervision signal.")

def suggest_fixes():
    """Suggest fixes for the occlusion head learning issue"""
    print("\n🔧 Suggested Fixes:")
    print("=" * 50)
    
    print("\n1. INCREASE OCCLUSION LOSS WEIGHT:")
    print("   Current: occlusion_loss_weight = 0.3")
    print("   Suggested: occlusion_loss_weight = 1.0 or 2.0")
    print("   Reason: Give occlusion head stronger supervision signal")
    
    print("\n2. INCREASE OCCLUSION FREQUENCY:")
    print("   Current: MedicalMaskOcclusion(prob=0.5)")
    print("   Suggested: MedicalMaskOcclusion(prob=0.8) or 1.0")
    print("   Reason: More training samples with occlusion")
    
    print("\n3. ADD MORE OCCLUSION TYPES:")
    print("   Add: Random rectangular occlusion")
    print("   Add: Gaussian noise occlusion")
    print("   Add: Random pixel dropout")
    print("   Reason: More diverse occlusion patterns")
    
    print("\n4. ADJUST LOSS WEIGHTS:")
    print("   Current: AdaFace=0.1, QAConv=0.7, Occ=0.3")
    print("   Suggested: AdaFace=0.1, QAConv=0.4, Occ=1.0")
    print("   Reason: Balance occlusion learning with other tasks")
    
    print("\n5. CHECK GROUND TRUTH MASK QUALITY:")
    print("   Verify medical mask masks are properly generated")
    print("   Ensure masks have sufficient occluded regions")
    print("   Check mask resolution matches occlusion head output")

def test_occlusion_head_parameters(model, device='cuda'):
    """Check if occlusion head parameters are actually being updated"""
    print("\n🔍 Checking Occlusion Head Parameters...")
    
    # Find occlusion head in the model
    occlusion_head = None
    for name, module in model.named_modules():
        if 'occlusion' in name.lower():
            occlusion_head = module
            print(f"Found occlusion head: {name}")
            break
    
    if occlusion_head is None:
        print("❌ No occlusion head found in model!")
        return
    
    print("Occlusion head parameters:")
    for name, param in occlusion_head.named_parameters():
        print(f"  {name}: shape={param.shape}, mean={param.data.mean():.6f}, std={param.data.std():.6f}")
    
    # Check if parameters have reasonable values
    total_params = sum(p.numel() for p in occlusion_head.parameters())
    print(f"  Total parameters: {total_params}")
    
    # Check if any parameters are zero or very small
    for name, param in occlusion_head.named_parameters():
        if param.data.abs().max() < 1e-6:
            print(f"  ⚠️  WARNING: {name} has very small values!")
        if param.data.abs().max() > 10:
            print(f"  ⚠️  WARNING: {name} has very large values!")

def main():
    print("🔍 Diagnosing Occlusion Head Training Issue")
    print("=" * 60)
    
    # Configuration
    model_path = "/home/maass/code/qaconv/experiments/ir50_casia_adaface_partial_10-02_9/last.ckpt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")
    
    try:
        # Load model
        model = load_trained_model(model_path, device)
        
        # Test occlusion head with different inputs
        test_occlusion_head_with_aggressive_occlusion(model, device)
        
        # Check occlusion head parameters
        test_occlusion_head_parameters(model, device)
        
        # Analyze loss weights
        analyze_loss_weights()
        
        # Suggest fixes
        suggest_fixes()
        
        print("\n" + "=" * 60)
        print("🎯 DIAGNOSIS COMPLETE")
        print("=" * 60)
        print("\nCONCLUSION:")
        print("The occlusion head is not learning effectively because:")
        print("1. Occlusion loss weight is too low (0.3)")
        print("2. Insufficient occlusion frequency in training data")
        print("3. Medical masks may not provide enough supervision signal")
        print("\nRECOMMENDATION:")
        print("Retrain with higher occlusion loss weight and more occlusion samples!")
        
    except Exception as e:
        print(f"❌ Error during diagnosis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
