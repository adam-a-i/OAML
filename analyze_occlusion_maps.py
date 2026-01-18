"""
Script 1: Occlusion Map Visualization

Visualizes the occlusion head's predictions on niqab images.
Creates side-by-side comparisons of:
- Original image
- Predicted occlusion map (heatmap overlay)
- Ground truth mask (if available)

Usage:
    python analyze_occlusion_maps.py \
        --checkpoint /path/to/checkpoint.ckpt \
        --niqab_data_path /home/maass/code/VPI \
        --output_dir ./occlusion_analysis \
        --num_samples 10
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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import net
from dataset.niqab_mask_dataset import NiqabMaskDataset


def load_model_with_occlusion_head(checkpoint_path, device):
    """Load trained model with occlusion head."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    hparams = ckpt['hyper_parameters']
    
    # Build model
    model = net.build_model(model_name=hparams['arch'])
    
    # Load model weights
    state_dict = {}
    for key, val in ckpt['state_dict'].items():
        if key.startswith('model.'):
            state_dict[key.replace('model.', '')] = val
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    # Check if occlusion head exists and has weights
    if hasattr(model, 'occlusion_head') and model.occlusion_head is not None:
        print("✓ OcclusionHead found in model")
        
        # Check if occlusion head weights were loaded
        occ_keys = [k for k in state_dict.keys() if 'occlusion_head' in k]
        print(f"  Loaded {len(occ_keys)} occlusion head parameters")
    else:
        print("✗ OcclusionHead NOT found in model")
        return None, None
    
    return model, hparams


def get_occlusion_prediction(model, image_tensor, device):
    """
    Get occlusion map prediction from the model.
    
    Returns:
        occlusion_map: [H, W] numpy array with values in [0, 1]
                      1 = visible, 0 = occluded
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]
        
        # Extract feature maps through backbone
        x = model.input_layer(image_tensor)
        for layer in model.body:
            x = layer(x)
        
        # Get occlusion prediction
        occlusion_map = model.occlusion_head(x)  # [1, 1, 7, 7]
        
        # Convert to numpy
        occlusion_map = occlusion_map.squeeze().cpu().numpy()  # [7, 7]
    
    return occlusion_map


def visualize_single_sample(
    original_image,
    predicted_map,
    ground_truth_mask=None,
    title="",
    save_path=None
):
    """
    Create visualization for a single sample.
    
    Args:
        original_image: PIL Image or numpy array [H, W, 3]
        predicted_map: numpy array [h, w] with values in [0, 1]
        ground_truth_mask: optional numpy array [h, w] with values in [0, 1]
        title: title for the figure
        save_path: path to save figure
    """
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)
    
    # Determine number of columns:
    # With GT: Original | Pred Overlay | Pred Standalone | GT Overlay | GT Standalone
    # Without GT: Original | Pred Overlay | Pred Standalone
    n_cols = 5 if ground_truth_mask is not None else 3
    
    fig = plt.figure(figsize=(4 * n_cols, 4))
    gs = gridspec.GridSpec(1, n_cols, figure=fig)
    
    # Resize occlusion map to match image size
    h, w = original_image.shape[:2]
    predicted_resized = np.array(Image.fromarray(
        (predicted_map * 255).astype(np.uint8)
    ).resize((w, h), Image.BILINEAR)) / 255.0
    
    # 1. Original image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_image)
    ax1.set_title("Original", fontsize=11)
    ax1.axis('off')
    
    # 2. Predicted occlusion map OVERLAY on image
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(original_image)
    # Invert: 1 = occluded (red), 0 = visible (transparent)
    occlusion_overlay = 1 - predicted_resized
    ax2.imshow(occlusion_overlay, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
    ax2.set_title("Pred Overlay\n(Red=Occluded)", fontsize=11)
    ax2.axis('off')
    
    # 3. Predicted occlusion map STANDALONE (white background)
    ax3 = fig.add_subplot(gs[0, 2])
    # Use RdYlGn colormap: Red = occluded (0), Green = visible (1)
    im3 = ax3.imshow(predicted_resized, cmap='RdYlGn', vmin=0, vmax=1)
    ax3.set_title("Pred Map\n(Green=Visible)", fontsize=11)
    ax3.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Occ', '0.5', 'Vis'])
    
    # Add statistics text
    pred_mean = predicted_map.mean()
    pred_min = predicted_map.min()
    pred_max = predicted_map.max()
    ax3.text(0.5, -0.15, f"μ={pred_mean:.2f} [{pred_min:.2f}-{pred_max:.2f}]",
             transform=ax3.transAxes, fontsize=9, ha='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4 & 5. Ground truth (if available)
    if ground_truth_mask is not None:
        # Resize ground truth to match image
        gt_resized = np.array(Image.fromarray(
            (ground_truth_mask * 255).astype(np.uint8)
        ).resize((w, h), Image.BILINEAR)) / 255.0
        
        # 4. GT overlay on image
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(original_image)
        gt_overlay = 1 - gt_resized  # 1 = occluded
        ax4.imshow(gt_overlay, cmap='Blues', alpha=0.6, vmin=0, vmax=1)
        ax4.set_title("GT Overlay\n(Blue=Occluded)", fontsize=11)
        ax4.axis('off')
        
        # 5. GT standalone (white background)
        ax5 = fig.add_subplot(gs[0, 4])
        im5 = ax5.imshow(gt_resized, cmap='RdYlGn', vmin=0, vmax=1)
        ax5.set_title("GT Map\n(Green=Visible)", fontsize=11)
        ax5.axis('off')
        
        # Add colorbar
        cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        cbar5.set_ticks([0, 0.5, 1])
        cbar5.set_ticklabels(['Occ', '0.5', 'Vis'])
        
        # Compute IoU
        pred_binary = (predicted_resized > 0.5).astype(float)
        gt_binary = (gt_resized > 0.5).astype(float)
        intersection = (pred_binary * gt_binary).sum()
        union = ((pred_binary + gt_binary) > 0).sum()
        iou = intersection / (union + 1e-8)
        ax5.text(0.5, -0.15, f"IoU={iou:.3f}",
                 transform=ax5.transAxes, fontsize=9, ha='center',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {save_path}")
    
    plt.close()


def visualize_occlusion_map_grid(
    model,
    dataset,
    device,
    num_samples=10,
    output_dir="./occlusion_analysis",
    has_ground_truth=True
):
    """
    Create visualization grid for multiple samples.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get transforms for loading raw images
    raw_transform = transforms.Compose([
        transforms.Resize((112, 112)),
    ])
    
    tensor_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    num_samples = min(num_samples, len(dataset))
    print(f"\nProcessing {num_samples} samples...")
    
    all_ious = []
    all_pred_means = []
    
    for i in range(num_samples):
        sample_info = dataset.get_sample_info(i)
        
        # Load raw image for visualization
        raw_image = Image.open(sample_info['image_path']).convert('RGB')
        raw_image_resized = raw_transform(raw_image)
        
        # Load image tensor for model
        image_tensor = tensor_transform(raw_image)
        
        # Get prediction
        predicted_map = get_occlusion_prediction(model, image_tensor, device)
        all_pred_means.append(predicted_map.mean())
        
        # Load ground truth if available
        ground_truth = None
        if has_ground_truth:
            try:
                gt_image = Image.open(sample_info['mask_path']).convert('L')
                gt_resized = gt_image.resize((7, 7), Image.BILINEAR)
                ground_truth = np.array(gt_resized) / 255.0
                
                # Compute IoU
                pred_binary = (predicted_map > 0.5).astype(float)
                gt_binary = (ground_truth > 0.5).astype(float)
                intersection = (pred_binary * gt_binary).sum()
                union = ((pred_binary + gt_binary) > 0).sum()
                iou = intersection / (union + 1e-8)
                all_ious.append(iou)
            except Exception as e:
                print(f"  Warning: Could not load ground truth for sample {i}: {e}")
                has_ground_truth = False
        
        # Create visualization
        save_path = os.path.join(output_dir, f"sample_{i:03d}_{sample_info['base_name']}.png")
        visualize_single_sample(
            original_image=raw_image_resized,
            predicted_map=predicted_map,
            ground_truth_mask=ground_truth,
            title=f"Sample {i}: {sample_info['base_name']}",
            save_path=save_path
        )
    
    # Print summary statistics
    print("\n" + "="*60)
    print("OCCLUSION MAP ANALYSIS SUMMARY")
    print("="*60)
    print(f"  Samples analyzed: {num_samples}")
    print(f"  Mean visibility score: {np.mean(all_pred_means):.4f}")
    print(f"  Std visibility score: {np.std(all_pred_means):.4f}")
    
    if all_ious:
        print(f"  Mean IoU with ground truth: {np.mean(all_ious):.4f}")
        print(f"  Min IoU: {np.min(all_ious):.4f}")
        print(f"  Max IoU: {np.max(all_ious):.4f}")
    
    print("="*60)
    
    # Create summary grid
    create_summary_grid(output_dir, num_samples)
    
    return all_ious, all_pred_means


def create_summary_grid(output_dir, num_samples):
    """Create a single image with all samples in a grid."""
    from PIL import Image as PILImage
    
    sample_images = []
    for i in range(min(num_samples, 12)):  # Limit to 12 for grid
        path = os.path.join(output_dir, f"sample_{i:03d}_*.png")
        import glob
        matches = glob.glob(path.replace('*', '*'))
        if matches:
            img = PILImage.open(matches[0])
            sample_images.append(img)
    
    if not sample_images:
        return
    
    # Create grid
    n_cols = min(3, len(sample_images))
    n_rows = (len(sample_images) + n_cols - 1) // n_cols
    
    # Get size of individual images
    w, h = sample_images[0].size
    
    # Create large canvas
    grid_w = w * n_cols
    grid_h = h * n_rows
    grid = PILImage.new('RGB', (grid_w, grid_h), color='white')
    
    for idx, img in enumerate(sample_images):
        row = idx // n_cols
        col = idx % n_cols
        grid.paste(img, (col * w, row * h))
    
    grid_path = os.path.join(output_dir, "summary_grid.png")
    grid.save(grid_path)
    print(f"\nSummary grid saved to: {grid_path}")


def analyze_without_dataset(model, image_paths, device, output_dir):
    """
    Analyze images without ground truth masks.
    Useful for arbitrary VPI images.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    tensor_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    raw_transform = transforms.Compose([
        transforms.Resize((112, 112)),
    ])
    
    all_pred_means = []
    
    for i, img_path in enumerate(image_paths):
        print(f"Processing: {img_path}")
        
        try:
            # Load image
            raw_image = Image.open(img_path).convert('RGB')
            raw_image_resized = raw_transform(raw_image)
            image_tensor = tensor_transform(raw_image)
            
            # Get prediction
            predicted_map = get_occlusion_prediction(model, image_tensor, device)
            all_pred_means.append(predicted_map.mean())
            
            # Create visualization (no ground truth)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(output_dir, f"sample_{i:03d}_{base_name}.png")
            
            visualize_single_sample(
                original_image=raw_image_resized,
                predicted_map=predicted_map,
                ground_truth_mask=None,
                title=f"{base_name}",
                save_path=save_path
            )
            
        except Exception as e:
            print(f"  Error processing {img_path}: {e}")
    
    print(f"\nMean visibility score across images: {np.mean(all_pred_means):.4f}")
    return all_pred_means


def main():
    parser = argparse.ArgumentParser(description='Occlusion Map Visualization')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--niqab_data_path', type=str, default='/home/maass/code/VPI',
                        help='Path to niqab dataset (with kept_faces/ and masks/ subdirs)')
    parser.add_argument('--output_dir', type=str, default='./occlusion_analysis',
                        help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--no_ground_truth', action='store_true',
                        help='Skip ground truth comparison (for images without masks)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, hparams = load_model_with_occlusion_head(args.checkpoint, device)
    if model is None:
        print("ERROR: Could not load model with occlusion head")
        return
    
    # Check if dataset has proper structure
    image_dir = os.path.join(args.niqab_data_path, 'kept_faces')
    mask_dir = os.path.join(args.niqab_data_path, 'masks')
    
    has_structure = os.path.isdir(image_dir) and os.path.isdir(mask_dir)
    
    if has_structure and not args.no_ground_truth:
        print(f"\nFound dataset structure at {args.niqab_data_path}")
        print(f"  Images: {image_dir}")
        print(f"  Masks: {mask_dir}")
        
        # Load dataset with ground truth
        dataset = NiqabMaskDataset(
            root_dir=args.niqab_data_path,
            mask_target_size=7
        )
        
        visualize_occlusion_map_grid(
            model=model,
            dataset=dataset,
            device=device,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            has_ground_truth=True
        )
    else:
        # No dataset structure - just find images
        print(f"\nNo dataset structure found. Looking for images in {args.niqab_data_path}")
        
        # Find image files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_paths = []
        
        for root, dirs, files in os.walk(args.niqab_data_path):
            for f in files:
                if os.path.splitext(f.lower())[1] in valid_extensions:
                    image_paths.append(os.path.join(root, f))
        
        image_paths = image_paths[:args.num_samples]
        print(f"Found {len(image_paths)} images")
        
        if image_paths:
            analyze_without_dataset(model, image_paths, device, args.output_dir)
        else:
            print("ERROR: No images found")


if __name__ == "__main__":
    main()
