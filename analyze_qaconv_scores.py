"""
Script 2: QAConv Score Distribution Analysis

Analyzes how QAConv scores distribute across different pair types:
- Genuine pairs (same identity) vs Imposter pairs (different identity)
- Clean-Clean, Niqab-Niqab, and Mixed pairs

Usage:
    python analyze_qaconv_scores.py \
        --checkpoint /path/to/checkpoint.ckpt \
        --vpi_data_path "/home/maass/code/VPI dataset resized 25%" \
        --output_dir ./qaconv_score_analysis \
        --num_pairs 500
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from collections import defaultdict
import re

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import net
import utils
from qaconv import QAConv


def load_model_and_qaconv(checkpoint_path, device):
    """Load trained model and QAConv."""
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
    
    # Get QAConv
    qaconv = model.qaconv
    num_features = qaconv.num_features
    height = qaconv.height
    width = qaconv.width
    class_num = utils.get_num_class(argparse.Namespace(**hparams))
    k_nearest = hparams.get('k_nearest', 20)
    
    qaconv = QAConv(num_features, height, width, 
                    num_classes=class_num,
                    k_nearest=k_nearest)
    
    # Load QAConv weights
    qaconv_state_dict = {}
    for key, val in ckpt['state_dict'].items():
        if key.startswith('qaconv.'):
            qaconv_state_dict[key.replace('qaconv.', '')] = val
    if qaconv_state_dict:
        qaconv.load_state_dict(qaconv_state_dict, strict=False)
        print(f"✓ Loaded QAConv weights ({len(qaconv_state_dict)} keys)")
    
    model = model.to(device)
    qaconv = qaconv.to(device)
    model.eval()
    qaconv.eval()
    
    return model, qaconv, hparams


def extract_identity_from_filename(filename):
    """
    Extract identity from VPI filename.
    VPI naming convention: S{session}-P{person_id}-{condition}-{angle}-{number}
    Example: S1-P67-M-17-1.jpg -> person_id = 67
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    
    # Try to match VPI pattern
    match = re.match(r'S\d+-P(\d+)-', base)
    if match:
        return int(match.group(1))
    
    # Fallback: use first number in filename
    numbers = re.findall(r'\d+', base)
    if numbers:
        return int(numbers[0])
    
    return hash(base) % 10000  # Fallback hash


def is_niqab_image(filename):
    """
    Check if image is a niqab image based on filename.
    VPI convention: M = masked/niqab, U = unmasked/clean
    """
    base = os.path.basename(filename).upper()
    
    # Check for VPI naming convention
    if '-M-' in base:
        return True
    if '-U-' in base:
        return False
    
    # Check for common niqab keywords
    niqab_keywords = ['niqab', 'mask', 'covered', 'occluded']
    for keyword in niqab_keywords:
        if keyword in base.lower():
            return True
    
    return False  # Default to clean


def get_feature_maps_and_occlusion(model, image_tensor, device):
    """Extract feature maps and occlusion map from model."""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        x = model.input_layer(image_tensor)
        for layer in model.body:
            x = layer(x)
        feature_maps = F.normalize(x, p=2, dim=1)
        occlusion_map = None
        if hasattr(model, 'occlusion_head') and model.occlusion_head is not None:
            occlusion_map = model.occlusion_head(x)
    return feature_maps, occlusion_map


def compute_qaconv_score(qaconv, features1, features2, occ1=None, occ2=None):
    """Compute QAConv similarity score between two feature maps."""
    with torch.no_grad():
        score = qaconv.match_pairs(features1, features2, probe_occ=occ1, gallery_occ=occ2)
    return score.item()


def load_vpi_images(vpi_path, max_images=1000):
    """
    Load VPI images and organize by identity.
    Returns dict: {identity: [(filepath, is_niqab), ...]}
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    images_by_identity = defaultdict(list)
    
    # Find all images
    for root, dirs, files in os.walk(vpi_path):
        for f in files:
            if os.path.splitext(f.lower())[1] in valid_extensions:
                filepath = os.path.join(root, f)
                identity = extract_identity_from_filename(f)
                is_niqab = is_niqab_image(f)
                images_by_identity[identity].append((filepath, is_niqab))
    
    # Filter identities with at least 2 images
    valid_identities = {k: v for k, v in images_by_identity.items() if len(v) >= 2}
    
    print(f"Found {len(valid_identities)} identities with 2+ images")
    
    # Count image types
    total_niqab = sum(1 for imgs in valid_identities.values() for _, is_niqab in imgs if is_niqab)
    total_clean = sum(1 for imgs in valid_identities.values() for _, is_niqab in imgs if not is_niqab)
    print(f"  Niqab images: {total_niqab}")
    print(f"  Clean images: {total_clean}")
    
    return valid_identities


def create_pairs(images_by_identity, num_genuine=200, num_imposter=200):
    """
    Create genuine and imposter pairs categorized by type.
    
    Returns:
        genuine_pairs: List of (img1_path, img2_path, pair_type)
        imposter_pairs: List of (img1_path, img2_path, pair_type)
        
    pair_type: 'clean-clean', 'niqab-niqab', 'mixed'
    """
    genuine_pairs = {'clean-clean': [], 'niqab-niqab': [], 'mixed': []}
    imposter_pairs = {'clean-clean': [], 'niqab-niqab': [], 'mixed': []}
    
    identities = list(images_by_identity.keys())
    
    # Create genuine pairs (same identity)
    for identity, images in images_by_identity.items():
        if len(images) < 2:
            continue
        
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                img1_path, is_niqab1 = images[i]
                img2_path, is_niqab2 = images[j]
                
                # Determine pair type
                if is_niqab1 and is_niqab2:
                    pair_type = 'niqab-niqab'
                elif not is_niqab1 and not is_niqab2:
                    pair_type = 'clean-clean'
                else:
                    pair_type = 'mixed'
                
                genuine_pairs[pair_type].append((img1_path, img2_path))
    
    # Create imposter pairs (different identities)
    np.random.seed(42)
    all_images = [(img, is_niqab, identity) 
                  for identity, images in images_by_identity.items() 
                  for img, is_niqab in images]
    
    attempts = 0
    max_attempts = num_imposter * 10
    
    while sum(len(v) for v in imposter_pairs.values()) < num_imposter * 3 and attempts < max_attempts:
        attempts += 1
        
        # Random sample two images
        idx1, idx2 = np.random.choice(len(all_images), 2, replace=False)
        img1_path, is_niqab1, id1 = all_images[idx1]
        img2_path, is_niqab2, id2 = all_images[idx2]
        
        # Must be different identities
        if id1 == id2:
            continue
        
        # Determine pair type
        if is_niqab1 and is_niqab2:
            pair_type = 'niqab-niqab'
        elif not is_niqab1 and not is_niqab2:
            pair_type = 'clean-clean'
        else:
            pair_type = 'mixed'
        
        imposter_pairs[pair_type].append((img1_path, img2_path))
    
    # Limit to requested number
    for pair_type in genuine_pairs:
        np.random.shuffle(genuine_pairs[pair_type])
        genuine_pairs[pair_type] = genuine_pairs[pair_type][:num_genuine]
    
    for pair_type in imposter_pairs:
        np.random.shuffle(imposter_pairs[pair_type])
        imposter_pairs[pair_type] = imposter_pairs[pair_type][:num_imposter]
    
    return genuine_pairs, imposter_pairs


def compute_scores_for_pairs(model, qaconv, pairs, device, transform, pair_label):
    """Compute QAConv scores for a list of pairs."""
    scores = []
    
    for i, (img1_path, img2_path) in enumerate(pairs):
        try:
            # Load images
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
            
            # Transform
            tensor1 = transform(img1)
            tensor2 = transform(img2)
            
            # Get features
            features1, occ1 = get_feature_maps_and_occlusion(model, tensor1, device)
            features2, occ2 = get_feature_maps_and_occlusion(model, tensor2, device)
            
            # Compute score
            score = compute_qaconv_score(qaconv, features1, features2, occ1=occ1, occ2=occ2)
            scores.append(score)
            
            if (i + 1) % 50 == 0:
                print(f"    Processed {i + 1}/{len(pairs)} {pair_label} pairs")
                
        except Exception as e:
            print(f"    Error processing pair: {e}")
            continue
    
    return np.array(scores)


def plot_score_distributions(results, output_dir):
    """Create visualization plots for score distributions."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Overall genuine vs imposter
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    pair_types = ['clean-clean', 'niqab-niqab', 'mixed']
    colors = {'genuine': '#2ecc71', 'imposter': '#e74c3c'}
    
    for idx, pair_type in enumerate(pair_types):
        ax = axes[idx]
        
        genuine_scores = results[pair_type]['genuine']
        imposter_scores = results[pair_type]['imposter']
        
        if len(genuine_scores) > 0:
            ax.hist(genuine_scores, bins=30, alpha=0.7, label=f'Genuine (n={len(genuine_scores)})', 
                    color=colors['genuine'], density=True)
        if len(imposter_scores) > 0:
            ax.hist(imposter_scores, bins=30, alpha=0.7, label=f'Imposter (n={len(imposter_scores)})', 
                    color=colors['imposter'], density=True)
        
        ax.set_xlabel('QAConv Score', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{pair_type.upper()}\nPairs', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        
        # Add statistics
        if len(genuine_scores) > 0 and len(imposter_scores) > 0:
            gen_mean = np.mean(genuine_scores)
            imp_mean = np.mean(imposter_scores)
            separation = gen_mean - imp_mean
            
            # Compute d-prime (discriminability)
            gen_std = np.std(genuine_scores) + 1e-8
            imp_std = np.std(imposter_scores) + 1e-8
            d_prime = (gen_mean - imp_mean) / np.sqrt(0.5 * (gen_std**2 + imp_std**2))
            
            stats_text = f"Gen μ={gen_mean:.2f}\nImp μ={imp_mean:.2f}\nd'={d_prime:.2f}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_distributions_by_type.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: score_distributions_by_type.png")
    
    # Plot 2: Box plots comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    box_data = []
    box_labels = []
    box_colors = []
    
    for pair_type in pair_types:
        for match_type in ['genuine', 'imposter']:
            scores = results[pair_type][match_type]
            if len(scores) > 0:
                box_data.append(scores)
                box_labels.append(f"{pair_type}\n({match_type})")
                box_colors.append(colors[match_type])
    
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('QAConv Score', fontsize=12)
    ax.set_title('QAConv Score Distribution by Pair Type', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'score_boxplots.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: score_boxplots.png")
    
    # Plot 3: ROC-like curves (genuine acceptance vs imposter rejection)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for pair_type in pair_types:
        genuine_scores = results[pair_type]['genuine']
        imposter_scores = results[pair_type]['imposter']
        
        if len(genuine_scores) < 10 or len(imposter_scores) < 10:
            continue
        
        # Compute ROC
        thresholds = np.linspace(
            min(genuine_scores.min(), imposter_scores.min()),
            max(genuine_scores.max(), imposter_scores.max()),
            100
        )
        
        tpr = []  # True positive rate (genuine acceptance)
        fpr = []  # False positive rate (imposter acceptance)
        
        for thresh in thresholds:
            tpr.append(np.mean(genuine_scores >= thresh))
            fpr.append(np.mean(imposter_scores >= thresh))
        
        ax.plot(fpr, tpr, label=f'{pair_type}', linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate (Imposter Acceptance)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Genuine Acceptance)', fontsize=12)
    ax.set_title('ROC Curves by Pair Type', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: roc_curves.png")


def print_summary(results):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("QACONV SCORE DISTRIBUTION ANALYSIS")
    print("="*70)
    
    pair_types = ['clean-clean', 'niqab-niqab', 'mixed']
    
    for pair_type in pair_types:
        genuine_scores = results[pair_type]['genuine']
        imposter_scores = results[pair_type]['imposter']
        
        print(f"\n{pair_type.upper()} PAIRS:")
        print("-" * 40)
        
        if len(genuine_scores) > 0:
            print(f"  Genuine (n={len(genuine_scores)}):")
            print(f"    Mean: {np.mean(genuine_scores):.4f}")
            print(f"    Std:  {np.std(genuine_scores):.4f}")
            print(f"    Min:  {np.min(genuine_scores):.4f}")
            print(f"    Max:  {np.max(genuine_scores):.4f}")
        else:
            print(f"  Genuine: No pairs found")
        
        if len(imposter_scores) > 0:
            print(f"  Imposter (n={len(imposter_scores)}):")
            print(f"    Mean: {np.mean(imposter_scores):.4f}")
            print(f"    Std:  {np.std(imposter_scores):.4f}")
            print(f"    Min:  {np.min(imposter_scores):.4f}")
            print(f"    Max:  {np.max(imposter_scores):.4f}")
        else:
            print(f"  Imposter: No pairs found")
        
        if len(genuine_scores) > 0 and len(imposter_scores) > 0:
            gen_mean = np.mean(genuine_scores)
            imp_mean = np.mean(imposter_scores)
            gen_std = np.std(genuine_scores) + 1e-8
            imp_std = np.std(imposter_scores) + 1e-8
            d_prime = (gen_mean - imp_mean) / np.sqrt(0.5 * (gen_std**2 + imp_std**2))
            
            print(f"  SEPARATION:")
            print(f"    Mean diff: {gen_mean - imp_mean:.4f}")
            print(f"    d-prime:   {d_prime:.4f}")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='QAConv Score Distribution Analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--vpi_data_path', type=str, required=True,
                        help='Path to VPI dataset')
    parser.add_argument('--output_dir', type=str, default='./qaconv_score_analysis',
                        help='Output directory for plots')
    parser.add_argument('--num_pairs', type=int, default=200,
                        help='Number of pairs per category')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, qaconv, hparams = load_model_and_qaconv(args.checkpoint, device)
    
    # Image transform
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load VPI images
    print(f"\nLoading images from: {args.vpi_data_path}")
    images_by_identity = load_vpi_images(args.vpi_data_path)
    
    if not images_by_identity:
        print("ERROR: No images found")
        return
    
    # Create pairs
    print(f"\nCreating pairs (target: {args.num_pairs} per category)...")
    genuine_pairs, imposter_pairs = create_pairs(
        images_by_identity, 
        num_genuine=args.num_pairs,
        num_imposter=args.num_pairs
    )
    
    # Print pair counts
    print("\nPair counts:")
    for pair_type in genuine_pairs:
        print(f"  {pair_type}: {len(genuine_pairs[pair_type])} genuine, {len(imposter_pairs[pair_type])} imposter")
    
    # Compute scores
    results = {}
    pair_types = ['clean-clean', 'niqab-niqab', 'mixed']
    
    for pair_type in pair_types:
        print(f"\nProcessing {pair_type} pairs...")
        results[pair_type] = {}
        
        # Genuine scores
        if genuine_pairs[pair_type]:
            print(f"  Computing genuine scores...")
            results[pair_type]['genuine'] = compute_scores_for_pairs(
                model, qaconv, genuine_pairs[pair_type], device, transform, f"{pair_type} genuine"
            )
        else:
            results[pair_type]['genuine'] = np.array([])
        
        # Imposter scores
        if imposter_pairs[pair_type]:
            print(f"  Computing imposter scores...")
            results[pair_type]['imposter'] = compute_scores_for_pairs(
                model, qaconv, imposter_pairs[pair_type], device, transform, f"{pair_type} imposter"
            )
        else:
            results[pair_type]['imposter'] = np.array([])
    
    # Print summary
    print_summary(results)
    
    # Create plots
    print("\nGenerating plots...")
    plot_score_distributions(results, args.output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
