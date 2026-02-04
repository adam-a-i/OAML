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
import matplotlib.patches as mpatches
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
    """Visualize occlusion maps for VPI and clean images in a paper-friendly layout."""
    os.makedirs(output_dir, exist_ok=True)
    
    tensor_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    raw_transform = transforms.Compose([
        transforms.Resize((112, 112)),
    ])
    
    # Collect samples (we'll render as columns: Original (top) -> Map (bottom))
    fig = plt.figure(figsize=(3.2 * 6, 6.2))  # tuned for up to ~6 samples
    
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
            all_labels.append("")  # no dataset/index labels for paper figure
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
            all_labels.append("")  # no dataset/index labels for paper figure
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Render samples: N rows x 2 columns (Original -> Map), with clear horizontal arrows
    n = len(all_images)
    if n == 0:
        print("No images to visualize.")
        return

    # More spacing for paper readability
    gs = gridspec.GridSpec(n, 2, figure=fig, hspace=0.55, wspace=0.35)

    # No title/subtitle at top (paper figure will add caption externally)

    # Shared colorbar axis on the right (tight to plots)
    # Note: values are figure-relative: [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.875, 0.18, 0.012, 0.62])
    last_im = None

    for idx, (img, occ_map, label) in enumerate(zip(all_images, all_maps, all_labels)):
        # Resize occlusion map to match image size (same as analyze_occlusion_maps.py)
        h, w = img.shape[:2]
        occ_resized = np.array(
            Image.fromarray((occ_map * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
        ) / 255.0

        ax_left = fig.add_subplot(gs[idx, 0])
        ax_right = fig.add_subplot(gs[idx, 1])

        ax_left.imshow(img)
        if label:
            ax_left.set_title(label, fontsize=11)
        ax_left.axis('off')

        last_im = ax_right.imshow(occ_resized, cmap='RdYlGn', vmin=0, vmax=1)
        ax_right.axis('off')

        # Mean visibility annotation (small, below map)
        mean_vis = float(occ_map.mean())
        ax_right.text(
            0.5,
            -0.10,
            f"μ={mean_vis:.3f}",
            transform=ax_right.transAxes,
            fontsize=9,
            ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, linewidth=0.5),
        )

        # Arrow from original -> map (figure coordinates for a clean horizontal arrow)
        left_bb = ax_left.get_position()
        right_bb = ax_right.get_position()
        y_mid = (left_bb.y0 + left_bb.y1) / 2.0
        # Even shorter arrow: draw only a small segment centered in the gap (gap-relative),
        # so it stays short regardless of figure size/spacing.
        gap = right_bb.x0 - left_bb.x1
        x_start = left_bb.x1 + gap * 0.48
        x_end = right_bb.x0 - gap * 0.48
        if x_end <= x_start:
            # Fallback if layout is extremely tight
            x_start = left_bb.x1 + gap * 0.40
            x_end = right_bb.x0 - gap * 0.40
        ax_left.annotate(
            "",
            xy=(x_end, y_mid),
            xytext=(x_start, y_mid),
            xycoords=fig.transFigure,
            textcoords=fig.transFigure,
            arrowprops=dict(arrowstyle="-|>", lw=3.0, color="#111111", mutation_scale=18),
        )

    if last_im is not None:
        cbar = plt.colorbar(last_im, cax=cbar_ax)
        cbar.set_ticks([0, 0.5, 1.0])
        cbar.set_ticklabels(["Occ", "0.5", "Vis"])

    out_path = os.path.join(output_dir, 'occlusion_comparison.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
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
    print(f"\nSaved comparison to: {out_path}")


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
    parser.add_argument(
        '--image_dir',
        type=str,
        default=None,
        help='If set, ignore --vpi_path/--clean_path and visualize images from this single directory.',
    )
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
    if args.image_dir is not None:
        # Single-directory mode (use same visualization layout, but label as "Custom")
        paths = find_images(args.image_dir, args.num_per_type * 2)
        if len(paths) < max(1, args.num_per_type):
            print("ERROR: Could not find enough images in --image_dir")
            return
        # Split into two groups just to reuse the existing grid layout
        vpi_paths = paths[: args.num_per_type]
        clean_paths = paths[args.num_per_type : args.num_per_type * 2]
        print(f"Found {len(paths)} images in {args.image_dir}")
        visualize_comparison(model, vpi_paths, clean_paths, device, args.output_dir)
        return

    vpi_paths = find_images(args.vpi_path, args.num_per_type)
    clean_paths = find_images(args.clean_path, args.num_per_type)

    print(f"Found {len(vpi_paths)} VPI images, {len(clean_paths)} clean images")

    if vpi_paths and clean_paths:
        visualize_comparison(model, vpi_paths, clean_paths, device, args.output_dir)
    else:
        print("ERROR: Could not find enough images")


if __name__ == "__main__":
    main()
