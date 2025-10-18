#!/usr/bin/env python3
"""
Simple Occlusion-Aware QAConv Analysis
=====================================

This script analyzes whether the occlusion head and QAConv weighting are working:
1. Load the trained model
2. Test with sample images
3. Compare QAConv scores with and without occlusion weighting
4. Check if occlusion maps are meaningful
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
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Build model (assuming ir_50 based on your path)
    architecture = 'ir_50'
    model = net.build_model(architecture)
    
    # Load state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Filter state dict to only include backbone parameters
    backbone_state_dict = {}
    for k, v in state_dict.items():
        # Only keep keys that start with 'model.' and are backbone parameters
        if k.startswith('model.') and not any(x in k for x in ['qaconv_criterion', 'qaconv_triplet_criterion', 'head.kernel', 'head.t', 'head.batch_mean', 'head.batch_std']):
            # Remove 'model.' prefix
            new_key = k[6:]
            backbone_state_dict[new_key] = v
        elif not any(x in k for x in ['qaconv_criterion', 'qaconv_triplet_criterion', 'head.kernel', 'head.t', 'head.batch_mean', 'head.batch_std']) and not k.startswith('model.'):
            # Direct backbone parameters
            backbone_state_dict[k] = v
    
    print(f"Loading {len(backbone_state_dict)} backbone parameters...")
    model.load_state_dict(backbone_state_dict, strict=False)
    
    model = model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully")
    return model

def create_test_images(device='cuda'):
    """Create test images with known patterns"""
    print("Creating test images...")
    
    # Create 4 test images with different patterns
    images = []
    
    # Image 1: Random face-like pattern
    img1 = torch.randn(3, 112, 112) * 0.3 + 0.5
    images.append(img1)
    
    # Image 2: Another random pattern
    img2 = torch.randn(3, 112, 112) * 0.3 + 0.5
    images.append(img2)
    
    # Image 3: Similar to img1 (should have high similarity)
    img3 = img1 + torch.randn(3, 112, 112) * 0.1
    images.append(img3)
    
    # Image 4: Very different pattern
    img4 = torch.randn(3, 112, 112) * 0.5
    images.append(img4)
    
    images = torch.stack(images)
    print(f"✅ Created {len(images)} test images")
    return images

def analyze_occlusion_maps(model, images, device='cuda'):
    """Analyze occlusion maps predicted by the model"""
    print("\n🔍 Analyzing Occlusion Maps...")
    
    with torch.no_grad():
        # Get model outputs
        embeddings, norms, occlusion_maps = model(images.to(device))
        
        print(f"Occlusion maps shape: {occlusion_maps.shape}")
        print(f"Occlusion maps range: [{occlusion_maps.min():.4f}, {occlusion_maps.max():.4f}]")
        print(f"Occlusion maps mean: {occlusion_maps.mean():.4f}")
        print(f"Occlusion maps std: {occlusion_maps.std():.4f}")
        
        # Analyze each sample
        for i in range(len(images)):
            occ_map = occlusion_maps[i, 0].cpu().numpy()  # [H, W]
            print(f"\nSample {i+1}:")
            print(f"  Occlusion map shape: {occ_map.shape}")
            print(f"  Min: {occ_map.min():.4f}, Max: {occ_map.max():.4f}")
            print(f"  Mean: {occ_map.mean():.4f}, Std: {occ_map.std():.4f}")
            print(f"  Values < 0.5 (likely occluded): {(occ_map < 0.5).sum()} / {occ_map.size} ({100*(occ_map < 0.5).sum()/occ_map.size:.1f}%)")
            print(f"  Values > 0.8 (likely visible): {(occ_map > 0.8).sum()} / {occ_map.size} ({100*(occ_map > 0.8).sum()/occ_map.size:.1f}%)")
        
        return occlusion_maps

def test_qaconv_weighting(model, images, device='cuda'):
    """Test QAConv with and without occlusion weighting"""
    print("\n🧪 Testing QAConv Weighting...")
    
    # Get feature maps for QAConv
    with torch.no_grad():
        # Get intermediate features
        feature_maps = model.get_feature_maps(images.to(device))
        print(f"Feature maps shape: {feature_maps.shape}")
        
        # Get occlusion maps
        _, _, occlusion_maps = model(images.to(device))
        print(f"Occlusion maps shape: {occlusion_maps.shape}")
        
        # Initialize QAConv
        qaconv = QAConv(feature_maps.shape[1], height=7, width=7, k_nearest=32).to(device)
        
        # Test with different occlusion methods
        print("\n📊 QAConv Similarity Analysis:")
        
        # Test 1: No occlusion weighting
        scores_no_occ = qaconv(feature_maps, feature_maps, query_occ=None, gallery_occ=None, occlusion_method="none")
        print(f"No occlusion weighting:")
        print(f"  Scores shape: {scores_no_occ.shape}")
        print(f"  Min: {scores_no_occ.min():.4f}, Max: {scores_no_occ.max():.4f}")
        print(f"  Mean: {scores_no_occ.mean():.4f}, Std: {scores_no_occ.std():.4f}")
        
        # Test 2: With occlusion weighting (scaling)
        scores_scaling = qaconv(feature_maps, feature_maps, query_occ=occlusion_maps, gallery_occ=occlusion_maps, occlusion_method="scaling")
        print(f"\nWith occlusion weighting (scaling):")
        print(f"  Scores shape: {scores_scaling.shape}")
        print(f"  Min: {scores_scaling.min():.4f}, Max: {scores_scaling.max():.4f}")
        print(f"  Mean: {scores_scaling.mean():.4f}, Std: {scores_scaling.std():.4f}")
        
        # Test 3: With occlusion weighting (outer)
        scores_outer = qaconv(feature_maps, feature_maps, query_occ=occlusion_maps, gallery_occ=occlusion_maps, occlusion_method="outer")
        print(f"\nWith occlusion weighting (outer):")
        print(f"  Scores shape: {scores_outer.shape}")
        print(f"  Min: {scores_outer.min():.4f}, Max: {scores_outer.max():.4f}")
        print(f"  Mean: {scores_outer.mean():.4f}, Std: {scores_outer.std():.4f}")
        
        # Compare differences
        diff_scaling = (scores_scaling - scores_no_occ).abs().mean()
        diff_outer = (scores_outer - scores_no_occ).abs().mean()
        
        print(f"\n📈 Impact Analysis:")
        print(f"  Scaling vs No-occ difference: {diff_scaling:.6f}")
        print(f"  Outer vs No-occ difference: {diff_outer:.6f}")
        
        if diff_scaling < 0.001:
            print("  ⚠️  WARNING: Occlusion weighting has minimal impact!")
        else:
            print("  ✅ Occlusion weighting is having an effect")
        
        # Check if occlusion maps are all similar (bad sign)
        occ_std = occlusion_maps.std()
        print(f"  Occlusion map diversity (std): {occ_std:.4f}")
        if occ_std < 0.01:
            print("  ⚠️  WARNING: All occlusion maps are very similar - head may not be learning!")
        else:
            print("  ✅ Occlusion maps show diversity")
        
        return scores_no_occ, scores_scaling, scores_outer

def test_with_synthetic_occlusion(model, images, device='cuda'):
    """Test with synthetic occlusion to see if the system responds"""
    print("\n🎭 Testing with Synthetic Occlusion...")
    
    # Create synthetic occlusion transform
    occlusion_transform = MedicalMaskOcclusion(prob=1.0)  # Always apply occlusion
    tensor_transform = ToTensorWithMask()
    
    # Apply occlusion to first image
    img_pil = transforms.ToPILImage()(images[0])
    img_occluded, mask = occlusion_transform(img_pil)
    img_occluded_tensor, mask_tensor = tensor_transform((img_occluded, mask))
    img_occluded_tensor = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(img_occluded_tensor)
    
    # Create batch with original and occluded versions
    test_images = torch.stack([images[0], img_occluded_tensor])
    
    with torch.no_grad():
        # Get model outputs
        embeddings, norms, occlusion_maps = model(test_images.to(device))
        
        print(f"Original vs Occluded Analysis:")
        print(f"  Original occlusion map mean: {occlusion_maps[0, 0].mean():.4f}")
        print(f"  Occluded occlusion map mean: {occlusion_maps[1, 0].mean():.4f}")
        
        # Test QAConv similarity
        feature_maps = model.get_feature_maps(test_images.to(device))
        qaconv = QAConv(feature_maps.shape[1], height=7, width=7, k_nearest=32).to(device)
        
        # Similarity between original and occluded
        similarity_no_occ = qaconv(feature_maps[0:1], feature_maps[1:2], query_occ=None, gallery_occ=None, occlusion_method="none")
        similarity_with_occ = qaconv(feature_maps[0:1], feature_maps[1:2], query_occ=occlusion_maps[0:1], gallery_occ=occlusion_maps[1:2], occlusion_method="scaling")
        
        print(f"  Similarity without occlusion weighting: {similarity_no_occ.item():.4f}")
        print(f"  Similarity with occlusion weighting: {similarity_with_occ.item():.4f}")
        print(f"  Difference: {abs(similarity_with_occ.item() - similarity_no_occ.item()):.4f}")
        
        if abs(similarity_with_occ.item() - similarity_no_occ.item()) < 0.01:
            print("  ⚠️  WARNING: Occlusion weighting has minimal impact on similarity!")
        else:
            print("  ✅ Occlusion weighting is affecting similarity scores")

def check_occlusion_head_learning(model, device='cuda'):
    """Check if the occlusion head has actually learned anything"""
    print("\n🧠 Checking Occlusion Head Learning...")
    
    # Create two very different inputs
    img1 = torch.randn(1, 3, 112, 112) * 0.5 + 0.5
    img2 = torch.randn(1, 3, 112, 112) * 0.5 + 0.5
    
    with torch.no_grad():
        # Get occlusion maps for both images
        _, _, occ1 = model(img1.to(device))
        _, _, occ2 = model(img2.to(device))
        
        print(f"Image 1 occlusion map mean: {occ1[0, 0].mean():.4f}")
        print(f"Image 2 occlusion map mean: {occ2[0, 0].mean():.4f}")
        print(f"Difference between images: {abs(occ1[0, 0].mean() - occ2[0, 0].mean()):.4f}")
        
        # Check if occlusion head is just outputting constant values
        if abs(occ1[0, 0].mean() - occ2[0, 0].mean()) < 0.01:
            print("  ⚠️  WARNING: Occlusion head outputs are very similar - may not be learning!")
        else:
            print("  ✅ Occlusion head shows variation between different inputs")

def main():
    print("🔍 Simple Occlusion-Aware QAConv Analysis")
    print("=" * 60)
    
    # Configuration
    model_path = "/home/maass/code/qaconv/experiments/ir50_casia_adaface_partial_10-02_9/last.ckpt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")
    
    try:
        # Load model
        model = load_trained_model(model_path, device)
        
        # Create test images
        images = create_test_images(device)
        
        # Check occlusion head learning
        check_occlusion_head_learning(model, device)
        
        # Analyze occlusion maps
        occlusion_maps = analyze_occlusion_maps(model, images, device)
        
        # Test QAConv weighting
        scores_no_occ, scores_scaling, scores_outer = test_qaconv_weighting(model, images, device)
        
        # Test with synthetic occlusion
        test_with_synthetic_occlusion(model, images, device)
        
        print("\n" + "=" * 60)
        print("🎯 ANALYSIS COMPLETE")
        print("=" * 60)
        print("\nKey Findings:")
        print("1. Check if occlusion maps show diversity between different images")
        print("2. Compare the similarity score differences to see if occlusion weighting has impact")
        print("3. If differences are small, the occlusion head may not have learned effectively")
        print("4. If differences are large, the weighting is working but may need tuning")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
