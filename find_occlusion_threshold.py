"""
Find Optimal Occlusion Threshold on Niqab Training Dataset

This script evaluates occlusion head predictions on the niqab training dataset
and finds the optimal threshold for binary classification (occluded vs visible).

Usage:
    python find_occlusion_threshold.py --checkpoint /path/to/checkpoint.ckpt --niqab_path /home/maass/code/niqab/train
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, accuracy_score
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import net
from dataset.niqab_mask_dataset import NiqabMaskDataset
from torchvision import transforms


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    hparams = ckpt['hyper_parameters']
    
    # Build model
    model = net.build_model(hparams['arch'])
    
    # Load weights
    state_dict = ckpt['state_dict']
    if any(k.startswith('model.') for k in state_dict.keys()):
        model_state = {k[6:]: v for k, v in state_dict.items() if k.startswith('model.')}
        model.load_state_dict(model_state, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    
    # Check for occlusion head
    if not hasattr(model, 'occlusion_head') or model.occlusion_head is None:
        raise ValueError("Model does not have occlusion_head!")
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded with occlusion head")
    return model, hparams


def evaluate_occlusion_thresholds(model, dataloader, device):
    """
    Evaluate occlusion predictions at different thresholds and find optimal one.
    
    Returns:
        Dictionary with threshold metrics and optimal threshold
    """
    all_predictions = []
    all_ground_truth = []
    gt_mean_visibility_per_image_list = []  # Store mean visibility per image
    
    print("Extracting occlusion predictions...")
    with torch.no_grad():
        for images, masks, _ in tqdm(dataloader, desc="Processing"):
            images = images.to(device)
            masks = masks.to(device)
            
            # Get feature maps (matching training structure)
            if hasattr(model, 'body_early') and hasattr(model, 'body_late'):
                # New structure: use body_early for 14x14 intermediate features
                x = model.input_layer(images)
                for layer in model.body_early:
                    x = layer(x)
            else:
                # Old structure: use body for 7x7 features
                x = model.input_layer(images)
                for layer in model.body:
                    x = layer(x)
            
            # Get occlusion predictions
            pred_maps = model.occlusion_head(x)  # [B, 1, H, W]
            
            # Resize GT masks to match prediction size if needed
            if masks.shape[-2:] != pred_maps.shape[-2:]:
                masks = F.interpolate(
                    masks,
                    size=pred_maps.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # Compute mean visibility per image from ground truth
            # masks shape: [B, 1, H, W]
            batch_mean_visibility = masks.cpu().numpy().mean(axis=(1, 2, 3))  # [B]
            gt_mean_visibility_per_image_list.extend(batch_mean_visibility)
            
            # Flatten spatial dimensions for pixel-wise evaluation
            pred_flat = pred_maps.cpu().numpy().flatten()
            gt_flat = masks.cpu().numpy().flatten()
            
            all_predictions.extend(pred_flat)
            all_ground_truth.extend(gt_flat)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_ground_truth = np.array(all_ground_truth)
    
    print(f"\nTotal pixels evaluated: {len(all_predictions):,}")
    print(f"Ground truth - Visible (1): {np.sum(all_ground_truth == 1):,} ({100*np.mean(all_ground_truth == 1):.2f}%)")
    print(f"Ground truth - Occluded (0): {np.sum(all_ground_truth == 0):,} ({100*np.mean(all_ground_truth == 0):.2f}%)")
    print(f"Prediction range: [{all_predictions.min():.4f}, {all_predictions.max():.4f}], Mean: {all_predictions.mean():.4f}")
    
    # Analyze ground truth mean visibility per image
    gt_mean_visibility_per_image = np.array(gt_mean_visibility_per_image_list)
    num_images = len(gt_mean_visibility_per_image)
    
    print("\n" + "="*80)
    print("GROUND TRUTH MASK ANALYSIS (Mean Visibility Per Image)")
    print("="*80)
    print(f"Number of images: {num_images}")
    print(f"\nGround truth mean visibility per image:")
    print(f"  Min: {gt_mean_visibility_per_image.min():.4f}")
    print(f"  Max: {gt_mean_visibility_per_image.max():.4f}")
    print(f"  Mean: {gt_mean_visibility_per_image.mean():.4f}")
    print(f"  Median: {np.median(gt_mean_visibility_per_image):.4f}")
    print(f"  Std: {gt_mean_visibility_per_image.std():.4f}")
    print(f"\nPercentiles:")
    print(f"  5th: {np.percentile(gt_mean_visibility_per_image, 5):.4f}")
    print(f"  10th: {np.percentile(gt_mean_visibility_per_image, 10):.4f}")
    print(f"  25th: {np.percentile(gt_mean_visibility_per_image, 25):.4f}")
    print(f"  50th: {np.percentile(gt_mean_visibility_per_image, 50):.4f}")
    print(f"  75th: {np.percentile(gt_mean_visibility_per_image, 75):.4f}")
    print(f"  90th: {np.percentile(gt_mean_visibility_per_image, 90):.4f}")
    print(f"  95th: {np.percentile(gt_mean_visibility_per_image, 95):.4f}")
    print(f"\n🎯 LOWEST mean visibility in ground truth: {gt_mean_visibility_per_image.min():.4f}")
    print(f"   This is the minimum visibility across all {num_images} images")
    print(f"   (i.e., the most occluded image has {gt_mean_visibility_per_image.min()*100:.2f}% visible pixels)")
    print("="*80)
    
    # Evaluate at different thresholds
    thresholds = np.arange(0.0, 1.01, 0.01)
    metrics = {
        'thresholds': thresholds,
        'accuracy': [],
        'f1': [],
        'precision': [],
        'recall': [],
        'iou': [],
        'dice': []
    }
    
    print("\nEvaluating thresholds...")
    # Binarize ground truth (threshold at 0.5 to handle any floating point issues)
    gt_binary = (all_ground_truth >= 0.5).astype(np.int32)
    
    for threshold in tqdm(thresholds, desc="Thresholds"):
        # Binary predictions
        pred_binary = (all_predictions >= threshold).astype(np.int32)
        
        # Compute metrics
        acc = accuracy_score(gt_binary, pred_binary)
        f1 = f1_score(gt_binary, pred_binary, zero_division=0)
        
        # Precision and recall
        tp = np.sum((pred_binary == 1) & (gt_binary == 1))
        fp = np.sum((pred_binary == 1) & (gt_binary == 0))
        fn = np.sum((pred_binary == 0) & (gt_binary == 1))
        tn = np.sum((pred_binary == 0) & (gt_binary == 0))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        # IoU (Intersection over Union)
        intersection = np.sum((pred_binary == 1) & (gt_binary == 1))
        union = np.sum((pred_binary == 1) | (gt_binary == 1))
        iou = intersection / (union + 1e-8)
        
        # Dice coefficient (F1 for segmentation)
        dice = (2 * intersection) / (np.sum(pred_binary == 1) + np.sum(gt_binary == 1) + 1e-8)
        
        metrics['accuracy'].append(acc)
        metrics['f1'].append(f1)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['iou'].append(iou)
        metrics['dice'].append(dice)
    
    # Find optimal thresholds for different metrics
    metrics['accuracy'] = np.array(metrics['accuracy'])
    metrics['f1'] = np.array(metrics['f1'])
    metrics['precision'] = np.array(metrics['precision'])
    metrics['recall'] = np.array(metrics['recall'])
    metrics['iou'] = np.array(metrics['iou'])
    metrics['dice'] = np.array(metrics['dice'])
    
    # Optimal thresholds (maximize each metric)
    optimal = {
        'accuracy': thresholds[np.argmax(metrics['accuracy'])],
        'f1': thresholds[np.argmax(metrics['f1'])],
        'iou': thresholds[np.argmax(metrics['iou'])],
        'dice': thresholds[np.argmax(metrics['dice'])],
    }
    
    # Binarize ground truth for ROC/PR curves
    gt_binary_for_roc = (all_ground_truth >= 0.5).astype(np.int32)
    
    # Compute ROC curve
    fpr, tpr, roc_thresholds = roc_curve(gt_binary_for_roc, all_predictions)
    roc_auc = auc(fpr, tpr)
    
    # Optimal threshold from ROC (Youden's J statistic)
    youden_j = tpr - fpr
    optimal_roc_idx = np.argmax(youden_j)
    optimal['roc'] = roc_thresholds[optimal_roc_idx]
    
    # Compute PR curve
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(gt_binary_for_roc, all_predictions)
    pr_auc = auc(recall_curve, precision_curve)
    
    # Optimal threshold from PR curve (F1-maximizing)
    f1_scores_pr = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-8)
    optimal_pr_idx = np.argmax(f1_scores_pr)
    optimal['pr'] = pr_thresholds[optimal_pr_idx] if optimal_pr_idx < len(pr_thresholds) else pr_thresholds[-1]
    
    # Print results
    print("\n" + "="*80)
    print("OPTIMAL THRESHOLDS")
    print("="*80)
    print(f"Best Accuracy:  {optimal['accuracy']:.4f} (Acc: {metrics['accuracy'][np.argmax(metrics['accuracy'])]:.4f})")
    print(f"Best F1 Score:  {optimal['f1']:.4f} (F1: {metrics['f1'][np.argmax(metrics['f1'])]:.4f})")
    print(f"Best IoU:       {optimal['iou']:.4f} (IoU: {metrics['iou'][np.argmax(metrics['iou'])]:.4f})")
    print(f"Best Dice:      {optimal['dice']:.4f} (Dice: {metrics['dice'][np.argmax(metrics['dice'])]:.4f})")
    print(f"ROC Optimal:    {optimal['roc']:.4f} (AUC: {roc_auc:.4f})")
    print(f"PR Optimal:     {optimal['pr']:.4f} (AUC: {pr_auc:.4f})")
    print("="*80)
    
    # Recommend lowest threshold that still gives good performance
    # Use IoU as primary metric (standard for segmentation)
    best_iou_idx = np.argmax(metrics['iou'])
    best_iou_threshold = thresholds[best_iou_idx]
    best_iou_value = metrics['iou'][best_iou_idx]
    
    print(f"\n🎯 RECOMMENDED THRESHOLD (based on IoU): {best_iou_threshold:.4f}")
    print(f"   IoU: {best_iou_value:.4f}")
    print(f"   Accuracy: {metrics['accuracy'][best_iou_idx]:.4f}")
    print(f"   F1: {metrics['f1'][best_iou_idx]:.4f}")
    print(f"   Precision: {metrics['precision'][best_iou_idx]:.4f}")
    print(f"   Recall: {metrics['recall'][best_iou_idx]:.4f}")
    
    # Find lowest threshold with IoU > 0.9 * best_iou
    iou_threshold = best_iou_value * 0.9
    valid_indices = np.where(metrics['iou'] >= iou_threshold)[0]
    if len(valid_indices) > 0:
        lowest_good_threshold = thresholds[valid_indices[0]]
        print(f"\n📉 LOWEST THRESHOLD with IoU >= {iou_threshold:.4f}: {lowest_good_threshold:.4f}")
        print(f"   IoU: {metrics['iou'][valid_indices[0]]:.4f}")
        print(f"   Accuracy: {metrics['accuracy'][valid_indices[0]]:.4f}")
        print(f"   F1: {metrics['f1'][valid_indices[0]]:.4f}")
    
    # Also find lowest threshold with reasonable performance (IoU > 0.8)
    iou_threshold_80 = 0.8
    valid_indices_80 = np.where(metrics['iou'] >= iou_threshold_80)[0]
    if len(valid_indices_80) > 0:
        lowest_80_threshold = thresholds[valid_indices_80[0]]
        print(f"\n🔻 LOWEST THRESHOLD with IoU >= {iou_threshold_80:.4f}: {lowest_80_threshold:.4f}")
        print(f"   IoU: {metrics['iou'][valid_indices_80[0]]:.4f}")
        print(f"   Accuracy: {metrics['accuracy'][valid_indices_80[0]]:.4f}")
        print(f"   F1: {metrics['f1'][valid_indices_80[0]]:.4f}")
        print(f"\n💡 FOR HYBRID EVALUATION: Use threshold = {lowest_80_threshold:.4f} (or lower)")
        print(f"   This is the lowest threshold that maintains IoU >= 0.8")
    
    return {
        'metrics': metrics,
        'optimal': optimal,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'recommended': best_iou_threshold,
        'predictions': all_predictions,
        'ground_truth': all_ground_truth
    }


def main():
    parser = argparse.ArgumentParser(description='Find optimal occlusion threshold on niqab dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--niqab_path', type=str, default='/home/maass/code/niqab/train',
                        help='Path to niqab training dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model, hparams = load_model(args.checkpoint, device)
    
    # Determine mask target size based on model structure
    if hasattr(model, 'body_early'):
        mask_target_size = 14  # body_early produces 14x14 features
    else:
        mask_target_size = 7   # body produces 7x7 features
    
    # Load niqab dataset
    print(f"\nLoading niqab dataset from: {args.niqab_path}")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = NiqabMaskDataset(
        root_dir=args.niqab_path,
        image_transform=transform,
        mask_target_size=mask_target_size
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Evaluate thresholds
    results = evaluate_occlusion_thresholds(model, dataloader, device)
    
    print("\n✅ Evaluation complete!")
    print(f"Recommended threshold: {results['recommended']:.4f}")


if __name__ == '__main__':
    main()
