#!/usr/bin/env python3
"""
Analyze Occlusion-Aware QAConv Effectiveness
============================================

This script analyzes whether the occlusion head and QAConv weighting are actually working:
1. Load the trained model
2. Extract occlusion maps from sample images
3. Compare QAConv scores with and without occlusion weighting
4. Visualize occlusion predictions
5. Analyze the impact on similarity scores
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
from data import get_dataloader
from qaconv import QAConv
from transforms import MedicalMaskOcclusion, ToTensorWithMask
import torchvision.transforms as transforms

def load_trained_model(model_path, device='cuda'):
    """Load the trained model with occlusion head"""
    print(f"Loading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract architecture info (assuming ir_50 based on your path)
    architecture = 'ir_50'
    
    # Build model
    model = net.build_model(architecture)
    
    # Load state dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully")
    return model

def get_sample_images(data_path, num_samples=8):
    """Get sample images from the dataset"""
    print(f"Loading sample images from: {data_path}")
    
    # Create a simple transform for testing
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Get sample images
    image_paths = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
                if len(image_paths) >= num_samples:
                    break
        if len(image_paths) >= num_samples:
            break
    
    images = []
    for path in image_paths[:num_samples]:
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img)
        images.append(img_tensor)
    
    images = torch.stack(images)
    print(f"✅ Loaded {len(images)} sample images")
    return images, image_paths

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
        qaconv = QAConv(feature_maps.shape[1], k_nearest=32).to(device)
        
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
        qaconv = QAConv(feature_maps.shape[1], k_nearest=32).to(device)
        
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

def visualize_occlusion_maps(images, occlusion_maps, save_path="occlusion_analysis.png"):
    """Visualize occlusion maps"""
    print(f"\n🎨 Creating visualization: {save_path}")
    
    num_samples = min(4, len(images))
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    
    for i in range(num_samples):
        # Original image
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img + 1) / 2  # Denormalize
        img = np.clip(img, 0, 1)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis('off')
        
        # Occlusion map
        occ_map = occlusion_maps[i, 0].cpu().numpy()
        im = axes[1, i].imshow(occ_map, cmap='viridis', vmin=0, vmax=1)
        axes[1, i].set_title(f"Occlusion Map {i+1}\n(mean: {occ_map.mean():.3f})")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Visualization saved to: {save_path}")

def main():
    print("🔍 Occlusion-Aware QAConv Effectiveness Analysis")
    print("=" * 60)
    
    # Configuration
    model_path = "/home/maass/code/qaconv/experiments/ir50_casia_adaface_partial_10-02_9/last.ckpt"
    data_path = "/home/maass/code/VPI dataset resized 25%/"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")
    
    # Load model
    model = load_trained_model(model_path, device)
    
    # Get sample images
    images, image_paths = get_sample_images(data_path, num_samples=8)
    
    # Analyze occlusion maps
    occlusion_maps = analyze_occlusion_maps(model, images, device)
    
    # Test QAConv weighting
    scores_no_occ, scores_scaling, scores_outer = test_qaconv_weighting(model, images, device)
    
    # Test with synthetic occlusion
    test_with_synthetic_occlusion(model, images, device)
    
    # Create visualization
    visualize_occlusion_maps(images, occlusion_maps)
    
    print("\n" + "=" * 60)
    print("🎯 ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nKey Findings:")
    print("1. Check the occlusion map visualization to see if predictions look reasonable")
    print("2. Compare the similarity score differences to see if occlusion weighting has impact")
    print("3. If differences are small, the occlusion head may not have learned effectively")
    print("4. If differences are large, the weighting is working but may need tuning")

if __name__ == "__main__":
    main()
