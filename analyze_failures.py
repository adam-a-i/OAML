"""
Script 3: Failure Case Analysis

Identifies and visualizes cases where QAConv makes errors:
- False Rejects: Genuine pairs with low scores (missed matches)
- False Accepts: Imposter pairs with high scores (wrong matches)

For each failure, shows:
- Both images side by side
- Occlusion maps for both
- QAConv score and expected outcome

Usage:
    python analyze_failures.py \
        --checkpoint /path/to/checkpoint.ckpt \
        --vpi_data_path "/home/maass/code/VPI dataset resized 25%" \
        --output_dir ./failure_analysis \
        --num_failures 20
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
    
    model = net.build_model(model_name=hparams['arch'])
    
    state_dict = {}
    for key, val in ckpt['state_dict'].items():
        if key.startswith('model.'):
            state_dict[key.replace('model.', '')] = val
    model.load_state_dict(state_dict, strict=False)
    
    qaconv = model.qaconv
    num_features = qaconv.num_features
    height = qaconv.height
    width = qaconv.width
    class_num = utils.get_num_class(argparse.Namespace(**hparams))
    k_nearest = hparams.get('k_nearest', 20)
    
    qaconv = QAConv(num_features, height, width, 
                    num_classes=class_num,
                    k_nearest=k_nearest)
    
    qaconv_state_dict = {}
    for key, val in ckpt['state_dict'].items():
        if key.startswith('qaconv.'):
            qaconv_state_dict[key.replace('qaconv.', '')] = val
    if qaconv_state_dict:
        qaconv.load_state_dict(qaconv_state_dict, strict=False)
    
    model = model.to(device)
    qaconv = qaconv.to(device)
    model.eval()
    qaconv.eval()
    
    return model, qaconv, hparams


def extract_identity_from_filename(filename):
    """Extract identity from VPI filename."""
    base = os.path.splitext(os.path.basename(filename))[0]
    match = re.match(r'S\d+-P(\d+)-', base)
    if match:
        return int(match.group(1))
    numbers = re.findall(r'\d+', base)
    if numbers:
        return int(numbers[0])
    return hash(base) % 10000


def is_niqab_image(filename):
    """Check if image is a niqab image."""
    base = os.path.basename(filename).upper()
    if '-M-' in base:
        return True
    if '-U-' in base:
        return False
    return False


def get_feature_maps_and_occlusion(model, image_tensor, device):
    """Extract feature maps and occlusion map from model."""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        x = model.input_layer(image_tensor)
        for layer in model.body:
            x = layer(x)
        feature_maps = F.normalize(x, p=2, dim=1)
        
        # Get occlusion map if available
        occlusion_map = None
        if hasattr(model, 'occlusion_head') and model.occlusion_head is not None:
            occlusion_map = model.occlusion_head(x)
    
    return feature_maps, occlusion_map


def compute_qaconv_score(qaconv, features1, features2, occ1=None, occ2=None):
    """Compute QAConv similarity score."""
    with torch.no_grad():
        score = qaconv.match_pairs(features1, features2, probe_occ=occ1, gallery_occ=occ2)
    return score.item()


def load_vpi_images(vpi_path):
    """Load VPI images organized by identity."""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images_by_identity = defaultdict(list)
    
    for root, dirs, files in os.walk(vpi_path):
        for f in files:
            if os.path.splitext(f.lower())[1] in valid_extensions:
                filepath = os.path.join(root, f)
                identity = extract_identity_from_filename(f)
                is_niqab = is_niqab_image(f)
                images_by_identity[identity].append((filepath, is_niqab))
    
    return {k: v for k, v in images_by_identity.items() if len(v) >= 2}


def create_all_pairs(images_by_identity, max_genuine=500, max_imposter=500):
    """Create genuine and imposter pairs with scores placeholder."""
    genuine_pairs = []
    imposter_pairs = []
    
    identities = list(images_by_identity.keys())
    
    # Genuine pairs
    for identity, images in images_by_identity.items():
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                img1_path, is_niqab1 = images[i]
                img2_path, is_niqab2 = images[j]
                
                if is_niqab1 and is_niqab2:
                    pair_type = 'niqab-niqab'
                elif not is_niqab1 and not is_niqab2:
                    pair_type = 'clean-clean'
                else:
                    pair_type = 'mixed'
                
                genuine_pairs.append({
                    'img1': img1_path,
                    'img2': img2_path,
                    'identity1': identity,
                    'identity2': identity,
                    'is_niqab1': is_niqab1,
                    'is_niqab2': is_niqab2,
                    'pair_type': pair_type,
                    'is_genuine': True
                })
    
    # Imposter pairs
    np.random.seed(42)
    all_images = [(img, is_niqab, identity) 
                  for identity, images in images_by_identity.items() 
                  for img, is_niqab in images]
    
    attempts = 0
    while len(imposter_pairs) < max_imposter and attempts < max_imposter * 10:
        attempts += 1
        idx1, idx2 = np.random.choice(len(all_images), 2, replace=False)
        img1_path, is_niqab1, id1 = all_images[idx1]
        img2_path, is_niqab2, id2 = all_images[idx2]
        
        if id1 == id2:
            continue
        
        if is_niqab1 and is_niqab2:
            pair_type = 'niqab-niqab'
        elif not is_niqab1 and not is_niqab2:
            pair_type = 'clean-clean'
        else:
            pair_type = 'mixed'
        
        imposter_pairs.append({
            'img1': img1_path,
            'img2': img2_path,
            'identity1': id1,
            'identity2': id2,
            'is_niqab1': is_niqab1,
            'is_niqab2': is_niqab2,
            'pair_type': pair_type,
            'is_genuine': False
        })
    
    # Limit genuine pairs
    np.random.shuffle(genuine_pairs)
    genuine_pairs = genuine_pairs[:max_genuine]
    
    return genuine_pairs, imposter_pairs


def compute_all_scores(model, qaconv, pairs, device, transform):
    """Compute QAConv scores for all pairs."""
    print(f"Computing scores for {len(pairs)} pairs...")
    
    for i, pair in enumerate(pairs):
        try:
            img1 = Image.open(pair['img1']).convert('RGB')
            img2 = Image.open(pair['img2']).convert('RGB')
            
            tensor1 = transform(img1)
            tensor2 = transform(img2)
            
            features1, occ1 = get_feature_maps_and_occlusion(model, tensor1, device)
            features2, occ2 = get_feature_maps_and_occlusion(model, tensor2, device)
            
            score = compute_qaconv_score(qaconv, features1, features2, occ1=occ1, occ2=occ2)
            
            pair['score'] = score
            pair['occlusion1'] = occ1.squeeze().cpu().numpy() if occ1 is not None else None
            pair['occlusion2'] = occ2.squeeze().cpu().numpy() if occ2 is not None else None
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(pairs)}")
                
        except Exception as e:
            pair['score'] = None
            pair['occlusion1'] = None
            pair['occlusion2'] = None
            print(f"  Error: {e}")
    
    # Filter out failed pairs
    pairs = [p for p in pairs if p['score'] is not None]
    return pairs


def find_threshold(genuine_pairs, imposter_pairs):
    """Find optimal threshold that balances FAR and FRR."""
    genuine_scores = [p['score'] for p in genuine_pairs]
    imposter_scores = [p['score'] for p in imposter_pairs]
    
    all_scores = genuine_scores + imposter_scores
    thresholds = np.linspace(min(all_scores), max(all_scores), 100)
    
    best_threshold = 0
    best_accuracy = 0
    
    for thresh in thresholds:
        # True positives: genuine above threshold
        tp = sum(1 for s in genuine_scores if s >= thresh)
        # True negatives: imposter below threshold
        tn = sum(1 for s in imposter_scores if s < thresh)
        
        accuracy = (tp + tn) / (len(genuine_scores) + len(imposter_scores))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = thresh
    
    return best_threshold, best_accuracy


def find_failures(genuine_pairs, imposter_pairs, threshold):
    """
    Find failure cases:
    - False Rejects: genuine pairs with score < threshold
    - False Accepts: imposter pairs with score >= threshold
    """
    false_rejects = []
    false_accepts = []
    
    for pair in genuine_pairs:
        if pair['score'] < threshold:
            pair['error_type'] = 'false_reject'
            pair['margin'] = threshold - pair['score']  # How far below threshold
            false_rejects.append(pair)
    
    for pair in imposter_pairs:
        if pair['score'] >= threshold:
            pair['error_type'] = 'false_accept'
            pair['margin'] = pair['score'] - threshold  # How far above threshold
            false_accepts.append(pair)
    
    # Sort by margin (worst errors first)
    false_rejects.sort(key=lambda x: x['margin'], reverse=True)
    false_accepts.sort(key=lambda x: x['margin'], reverse=True)
    
    return false_rejects, false_accepts


def visualize_failure(pair, output_path, transform_display, threshold):
    """Create detailed visualization for a failure case."""
    fig = plt.figure(figsize=(16, 6))
    
    # Load images
    img1 = Image.open(pair['img1']).convert('RGB')
    img2 = Image.open(pair['img2']).convert('RGB')
    img1_display = transform_display(img1)
    img2_display = transform_display(img2)
    
    # Create grid: 2 rows x 4 cols
    # Row 1: Img1, Occ1, Img2, Occ2
    # Row 2: Analysis text spanning all columns
    
    gs = gridspec.GridSpec(2, 4, height_ratios=[4, 1], figure=fig)
    
    # Image 1
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img1_display)
    name1 = os.path.basename(pair['img1'])
    type1 = "NIQAB" if pair['is_niqab1'] else "CLEAN"
    ax1.set_title(f"Image 1: {type1}\n{name1[:25]}...", fontsize=10)
    ax1.axis('off')
    
    # Occlusion map 1
    ax2 = fig.add_subplot(gs[0, 1])
    if pair['occlusion1'] is not None:
        im2 = ax2.imshow(pair['occlusion1'], cmap='RdYlGn', vmin=0, vmax=1)
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        vis_mean1 = pair['occlusion1'].mean()
        ax2.set_title(f"Occlusion Map 1\nVis: {vis_mean1:.2f}", fontsize=10)
    else:
        ax2.text(0.5, 0.5, "No occlusion\nmap", ha='center', va='center')
        ax2.set_title("Occlusion Map 1", fontsize=10)
    ax2.axis('off')
    
    # Image 2
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(img2_display)
    name2 = os.path.basename(pair['img2'])
    type2 = "NIQAB" if pair['is_niqab2'] else "CLEAN"
    ax3.set_title(f"Image 2: {type2}\n{name2[:25]}...", fontsize=10)
    ax3.axis('off')
    
    # Occlusion map 2
    ax4 = fig.add_subplot(gs[0, 3])
    if pair['occlusion2'] is not None:
        im4 = ax4.imshow(pair['occlusion2'], cmap='RdYlGn', vmin=0, vmax=1)
        plt.colorbar(im4, ax=ax4, fraction=0.046)
        vis_mean2 = pair['occlusion2'].mean()
        ax4.set_title(f"Occlusion Map 2\nVis: {vis_mean2:.2f}", fontsize=10)
    else:
        ax4.text(0.5, 0.5, "No occlusion\nmap", ha='center', va='center')
        ax4.set_title("Occlusion Map 2", fontsize=10)
    ax4.axis('off')
    
    # Analysis text at bottom
    ax5 = fig.add_subplot(gs[1, :])
    ax5.axis('off')
    
    error_type = pair['error_type'].upper().replace('_', ' ')
    is_genuine = "SAME PERSON" if pair['is_genuine'] else "DIFFERENT PEOPLE"
    
    if pair['is_genuine']:
        expected = f"Expected: MATCH (score ≥ {threshold:.2f})"
        got = f"Got: NO MATCH (score = {pair['score']:.2f})"
        color = '#e74c3c'  # Red for false reject
    else:
        expected = f"Expected: NO MATCH (score < {threshold:.2f})"
        got = f"Got: MATCH (score = {pair['score']:.2f})"
        color = '#e67e22'  # Orange for false accept
    
    analysis_text = (
        f"ERROR TYPE: {error_type}  |  {is_genuine}  |  Pair Type: {pair['pair_type'].upper()}\n"
        f"Identity 1: P{pair['identity1']}  |  Identity 2: P{pair['identity2']}\n"
        f"{expected}  →  {got}  |  Margin: {pair['margin']:.2f}"
    )
    
    ax5.text(0.5, 0.5, analysis_text, ha='center', va='center', fontsize=11,
             bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3),
             family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_summary_plot(false_rejects, false_accepts, genuine_pairs, imposter_pairs, threshold, output_dir):
    """Create summary visualization of all failures."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Score distribution with threshold
    ax1 = axes[0, 0]
    genuine_scores = [p['score'] for p in genuine_pairs]
    imposter_scores = [p['score'] for p in imposter_pairs]
    
    ax1.hist(genuine_scores, bins=30, alpha=0.7, label='Genuine', color='#2ecc71', density=True)
    ax1.hist(imposter_scores, bins=30, alpha=0.7, label='Imposter', color='#e74c3c', density=True)
    ax1.axvline(x=threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold={threshold:.2f}')
    ax1.set_xlabel('QAConv Score')
    ax1.set_ylabel('Density')
    ax1.set_title('Score Distribution with Threshold')
    ax1.legend()
    
    # Plot 2: Error breakdown by pair type
    ax2 = axes[0, 1]
    pair_types = ['clean-clean', 'niqab-niqab', 'mixed']
    fr_counts = [sum(1 for p in false_rejects if p['pair_type'] == pt) for pt in pair_types]
    fa_counts = [sum(1 for p in false_accepts if p['pair_type'] == pt) for pt in pair_types]
    
    x = np.arange(len(pair_types))
    width = 0.35
    ax2.bar(x - width/2, fr_counts, width, label='False Rejects', color='#e74c3c')
    ax2.bar(x + width/2, fa_counts, width, label='False Accepts', color='#e67e22')
    ax2.set_xticks(x)
    ax2.set_xticklabels([pt.upper() for pt in pair_types])
    ax2.set_ylabel('Count')
    ax2.set_title('Errors by Pair Type')
    ax2.legend()
    
    # Plot 3: Error margin distribution
    ax3 = axes[1, 0]
    if false_rejects:
        fr_margins = [p['margin'] for p in false_rejects]
        ax3.hist(fr_margins, bins=20, alpha=0.7, label=f'False Rejects (n={len(false_rejects)})', color='#e74c3c')
    if false_accepts:
        fa_margins = [p['margin'] for p in false_accepts]
        ax3.hist(fa_margins, bins=20, alpha=0.7, label=f'False Accepts (n={len(false_accepts)})', color='#e67e22')
    ax3.set_xlabel('Error Margin (distance from threshold)')
    ax3.set_ylabel('Count')
    ax3.set_title('Error Margin Distribution')
    ax3.legend()
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    total_genuine = len(genuine_pairs)
    total_imposter = len(imposter_pairs)
    frr = len(false_rejects) / total_genuine * 100 if total_genuine > 0 else 0
    far = len(false_accepts) / total_imposter * 100 if total_imposter > 0 else 0
    
    summary_text = f"""
    FAILURE ANALYSIS SUMMARY
    ========================
    
    Threshold: {threshold:.2f}
    
    GENUINE PAIRS: {total_genuine}
      - Correct: {total_genuine - len(false_rejects)}
      - False Rejects: {len(false_rejects)}
      - FRR: {frr:.2f}%
    
    IMPOSTER PAIRS: {total_imposter}
      - Correct: {total_imposter - len(false_accepts)}
      - False Accepts: {len(false_accepts)}
      - FAR: {far:.2f}%
    
    OVERALL ACCURACY: {100 - (frr + far)/2:.2f}%
    """
    
    ax4.text(0.1, 0.5, summary_text, ha='left', va='center', fontsize=11,
             family='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'failure_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: failure_summary.png")


def print_failure_patterns(false_rejects, false_accepts):
    """Analyze and print patterns in failures."""
    print("\n" + "="*70)
    print("FAILURE PATTERN ANALYSIS")
    print("="*70)
    
    # False Rejects analysis
    print("\n📉 FALSE REJECTS (Genuine pairs wrongly rejected):")
    print("-" * 50)
    if false_rejects:
        fr_by_type = defaultdict(list)
        for p in false_rejects:
            fr_by_type[p['pair_type']].append(p)
        
        for pair_type, pairs in fr_by_type.items():
            print(f"\n  {pair_type.upper()}: {len(pairs)} errors")
            avg_margin = np.mean([p['margin'] for p in pairs])
            print(f"    Avg margin from threshold: {avg_margin:.2f}")
            
            # Check occlusion patterns
            if pairs[0]['occlusion1'] is not None:
                avg_vis1 = np.mean([p['occlusion1'].mean() for p in pairs if p['occlusion1'] is not None])
                avg_vis2 = np.mean([p['occlusion2'].mean() for p in pairs if p['occlusion2'] is not None])
                print(f"    Avg visibility: {avg_vis1:.2f} vs {avg_vis2:.2f}")
    else:
        print("  No false rejects found!")
    
    # False Accepts analysis
    print("\n📈 FALSE ACCEPTS (Imposter pairs wrongly accepted):")
    print("-" * 50)
    if false_accepts:
        fa_by_type = defaultdict(list)
        for p in false_accepts:
            fa_by_type[p['pair_type']].append(p)
        
        for pair_type, pairs in fa_by_type.items():
            print(f"\n  {pair_type.upper()}: {len(pairs)} errors")
            avg_margin = np.mean([p['margin'] for p in pairs])
            print(f"    Avg margin above threshold: {avg_margin:.2f}")
            
            # Check if same-looking people
            print(f"    Sample identity pairs: ", end="")
            for p in pairs[:3]:
                print(f"P{p['identity1']}-P{p['identity2']}, ", end="")
            print()
    else:
        print("  No false accepts found!")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='Failure Case Analysis')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--vpi_data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./failure_analysis')
    parser.add_argument('--num_failures', type=int, default=20,
                        help='Number of failure cases to visualize')
    parser.add_argument('--max_pairs', type=int, default=500,
                        help='Max pairs to evaluate')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, qaconv, hparams = load_model_and_qaconv(args.checkpoint, device)
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    transform_display = transforms.Compose([
        transforms.Resize((112, 112)),
    ])
    
    # Load images
    print(f"\nLoading images from: {args.vpi_data_path}")
    images_by_identity = load_vpi_images(args.vpi_data_path)
    print(f"Found {len(images_by_identity)} identities")
    
    # Create pairs
    print("\nCreating pairs...")
    genuine_pairs, imposter_pairs = create_all_pairs(
        images_by_identity, 
        max_genuine=args.max_pairs,
        max_imposter=args.max_pairs
    )
    print(f"Created {len(genuine_pairs)} genuine, {len(imposter_pairs)} imposter pairs")
    
    # Compute scores
    print("\nComputing genuine scores...")
    genuine_pairs = compute_all_scores(model, qaconv, genuine_pairs, device, transform)
    
    print("\nComputing imposter scores...")
    imposter_pairs = compute_all_scores(model, qaconv, imposter_pairs, device, transform)
    
    # Find optimal threshold
    threshold, accuracy = find_threshold(genuine_pairs, imposter_pairs)
    print(f"\nOptimal threshold: {threshold:.2f} (accuracy: {accuracy:.2%})")
    
    # Find failures
    false_rejects, false_accepts = find_failures(genuine_pairs, imposter_pairs, threshold)
    print(f"\nFound {len(false_rejects)} false rejects, {len(false_accepts)} false accepts")
    
    # Print pattern analysis
    print_failure_patterns(false_rejects, false_accepts)
    
    # Create summary plot
    print("\nGenerating summary plot...")
    create_summary_plot(false_rejects, false_accepts, genuine_pairs, imposter_pairs, threshold, args.output_dir)
    
    # Visualize worst failures
    print(f"\nVisualizing top {args.num_failures} failures...")
    
    # False rejects
    fr_dir = os.path.join(args.output_dir, 'false_rejects')
    os.makedirs(fr_dir, exist_ok=True)
    for i, pair in enumerate(false_rejects[:args.num_failures]):
        output_path = os.path.join(fr_dir, f'fr_{i:03d}_margin{pair["margin"]:.1f}.png')
        visualize_failure(pair, output_path, transform_display, threshold)
        print(f"  Saved: {os.path.basename(output_path)}")
    
    # False accepts
    fa_dir = os.path.join(args.output_dir, 'false_accepts')
    os.makedirs(fa_dir, exist_ok=True)
    for i, pair in enumerate(false_accepts[:args.num_failures]):
        output_path = os.path.join(fa_dir, f'fa_{i:03d}_margin{pair["margin"]:.1f}.png')
        visualize_failure(pair, output_path, transform_display, threshold)
        print(f"  Saved: {os.path.basename(output_path)}")
    
    print(f"\n✓ Analysis complete! Results saved to: {args.output_dir}")
    print(f"  - failure_summary.png")
    print(f"  - false_rejects/ ({min(len(false_rejects), args.num_failures)} images)")
    print(f"  - false_accepts/ ({min(len(false_accepts), args.num_failures)} images)")


if __name__ == "__main__":
    main()
