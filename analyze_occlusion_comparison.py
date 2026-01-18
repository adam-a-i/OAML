"""
Quick script to analyze occlusion maps on 3 VPI images and 3 clean images.
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import transforms
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import net
import utils


def load_model_with_occlusion_head(checkpoint_path, device):
    """Load trained model with occlusion head."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    hparams = ckpt['hyper_parameters']
    
    model = net.build_model(model_name=hparams['arch'])
    
    state_dict = {}
    for key, val in ckpt['state_dict'].items():
        if key.startswith('model.'):
            state_dict[key.replace('model.', '')] = val
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    if hasattr(model, 'occlusion_head') and model.occlusion_head is not None:
        occ_keys = [k for k in state_dict.keys() if 'occlusion_head' in k]
        print(f"✓ OcclusionHead found and loaded ({len(occ_keys)} parameters)")
    else:
        print("✗ OcclusionHead NOT found in model")
        return None, None
    
    return model, hparams


def get_occlusion_prediction(model, image_tensor, device):
    """Get occlusion map prediction from the model."""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        x = model.input_layer(image_tensor)
        for layer in model.body:
            x = layer(x)
        occlusion_map = model.occlusion_head(x)
        occlusion_map = occlusion_map.squeeze().cpu().numpy()
    return occlusion_map


def visualize_comparison(model, vpi_paths, clean_paths, device, output_dir):
    """Visualize occlusion maps for VPI and clean images side by side."""
    os.makedirs(output_dir, exist_ok=True)
    
    tensor_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    raw_transform = transforms.Compose([
        transforms.Resize((112, 112)),
    ])
    
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 6, figure=fig, hspace=0.3, wspace=0.3)
    
    all_images = []
    all_maps = []
    all_labels = []
    
    # Process VPI images
    for i, img_path in enumerate(vpi_paths[:3]):
        try:
            raw_image = Image.open(img_path).convert('RGB')
            raw_image_resized = raw_transform(raw_image)
            image_tensor = tensor_transform(raw_image)
            predicted_map = get_occlusion_prediction(model, image_tensor, device)
            
            all_images.append(np.array(raw_image_resized))
            all_maps.append(predicted_map)
            all_labels.append(f"VPI {i+1}\n{os.path.basename(img_path)[:20]}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Process clean images
    for i, img_path in enumerate(clean_paths[:3]):
        try:
            raw_image = Image.open(img_path).convert('RGB')
            raw_image_resized = raw_transform(raw_image)
            image_tensor = tensor_transform(raw_image)
            predicted_map = get_occlusion_prediction(model, image_tensor, device)
            
            all_images.append(np.array(raw_image_resized))
            all_maps.append(predicted_map)
            all_labels.append(f"Clean {i+1}\n{os.path.basename(img_path)[:20]}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Plot: 3 rows (VPI, Clean, Clean), 6 cols (Original, Overlay, Map for each)
    for idx, (img, occ_map, label) in enumerate(zip(all_images, all_maps, all_labels)):
        row = idx // 2
        col_base = (idx % 2) * 3
        
        # Original image
        ax1 = fig.add_subplot(gs[row, col_base])
        ax1.imshow(img)
        ax1.set_title(f"{label}\nOriginal", fontsize=10)
        ax1.axis('off')
        
        # Overlay
        ax2 = fig.add_subplot(gs[row, col_base + 1])
        ax2.imshow(img)
        h, w = img.shape[:2]
        occ_resized = np.array(Image.fromarray((occ_map * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
        occlusion_overlay = 1 - occ_resized
        ax2.imshow(occlusion_overlay, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
        ax2.set_title(f"Overlay\n(Red=Occluded)", fontsize=10)
        ax2.axis('off')
        
        # Standalone map
        ax3 = fig.add_subplot(gs[row, col_base + 2])
        im3 = ax3.imshow(occ_resized, cmap='RdYlGn', vmin=0, vmax=1)
        ax3.set_title(f"Map\n(Green=Visible)", fontsize=10)
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        # Add stats
        mean_vis = occ_map.mean()
        ax3.text(0.5, -0.15, f"μ={mean_vis:.3f}",
                 transform=ax3.transAxes, fontsize=9, ha='center',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Occlusion Head Predictions: VPI (Niqab) vs Clean Faces', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'occlusion_comparison.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Print summary
    print("\n" + "="*60)
    print("OCCLUSION MAP COMPARISON SUMMARY")
    print("="*60)
    vpi_means = [all_maps[i].mean() for i in range(len(vpi_paths[:3]))]
    clean_means = [all_maps[i+len(vpi_paths[:3])].mean() for i in range(len(clean_paths[:3]))]
    print(f"VPI images (niqab):")
    for i, mean in enumerate(vpi_means):
        print(f"  Image {i+1}: mean visibility = {mean:.4f}")
    print(f"  Average: {np.mean(vpi_means):.4f}")
    print(f"\nClean images:")
    for i, mean in enumerate(clean_means):
        print(f"  Image {i+1}: mean visibility = {mean:.4f}")
    print(f"  Average: {np.mean(clean_means):.4f}")
    print("="*60)
    print(f"\nSaved comparison to: {os.path.join(output_dir, 'occlusion_comparison.png')}")


def find_images(path, num=3):
    """Find image files in a directory."""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = []
    
    for root, dirs, files in os.walk(path):
        for f in files:
            if os.path.splitext(f.lower())[1] in valid_extensions:
                image_paths.append(os.path.join(root, f))
        if len(image_paths) >= num:
            break
    
    return image_paths[:num]


def main():
    parser = argparse.ArgumentParser(description='Compare occlusion maps on VPI and clean images')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--vpi_path', type=str, default='/home/maass/code/VPI dataset resized 25%',
                        help='Path to VPI dataset')
    parser.add_argument('--clean_path', type=str, default='/home/maass/code/faces_webface_112x112',
                        help='Path to clean faces dataset')
    parser.add_argument('--output_dir', type=str, default='./occlusion_comparison',
                        help='Output directory')
    parser.add_argument('--num_per_type', type=int, default=3,
                        help='Number of images per type to analyze')
    
    args = parser.parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model, hparams = load_model_with_occlusion_head(args.checkpoint, device)
    if model is None:
        return
    
    print(f"\nFinding images...")
    vpi_paths = find_images(args.vpi_path, args.num_per_type)
    clean_paths = find_images(args.clean_path, args.num_per_type)
    
    print(f"Found {len(vpi_paths)} VPI images, {len(clean_paths)} clean images")
    
    if vpi_paths and clean_paths:
        visualize_comparison(model, vpi_paths, clean_paths, device, args.output_dir)
    else:
        print("ERROR: Could not find enough images")


if __name__ == "__main__":
    main()
