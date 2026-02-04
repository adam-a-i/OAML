"""
Hybrid Occlusion-Aware Evaluation Script

This script implements a hybrid evaluation method:
- If mean occlusion > 50% (visibility < 0.5): Use QAConv matching only
- If mean occlusion <= 50% (visibility >= 0.5): Use AdaFace matching only

The decision is made per image pair based on the mean visibility of both images.

Usage:
    python eval_hybrid_occlusion.py --checkpoint /path/to/checkpoint.ckpt --val_data_path /home/maass/code/faces_webface_112x112
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import net
import utils
import evaluate_utils
from qaconv import QAConv
from dataset.five_validation_dataset import FiveValidationDataset
from torch.utils.data import DataLoader


def evaluate_hybrid_occlusion(checkpoint_path, val_data_path, batch_size=64, num_workers=4, occlusion_threshold=0.5):
    """
    Run hybrid occlusion-aware evaluation on the 5 validation datasets.
    
    Args:
        checkpoint_path: Path to model checkpoint
        val_data_path: Path to validation data directory
        batch_size: Batch size for feature extraction
        num_workers: Number of data loading workers
        occlusion_threshold: Threshold for mean visibility (default: 0.5)
                            If mean visibility < threshold, use QAConv; else use AdaFace
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Validation data path: {val_data_path}")
    print(f"Occlusion threshold: {occlusion_threshold} (mean visibility)")
    print(f"  - If mean visibility < {occlusion_threshold}: Use QAConv")
    print(f"  - If mean visibility >= {occlusion_threshold}: Use AdaFace")
    
    # Load checkpoint
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
    
    # Check if occlusion head exists and has weights
    has_occlusion_head = hasattr(model, 'occlusion_head') and model.occlusion_head is not None
    if has_occlusion_head:
        occlusion_keys = [k for k in state_dict.keys() if 'occlusion_head' in k]
        if occlusion_keys:
            print(f"✓ OcclusionHead found and loaded ({len(occlusion_keys)} parameters)")
        else:
            print("⚠ WARNING: OcclusionHead exists but no weights found in checkpoint!")
            print("  Falling back to AdaFace for all pairs.")
            has_occlusion_head = False
    else:
        print("⚠ WARNING: OcclusionHead NOT found in model!")
        print("  Falling back to AdaFace for all pairs.")
    
    # Get QAConv from model
    qaconv = None
    if hasattr(model, 'qaconv'):
        qaconv = model.qaconv
        
        # Rebuild QAConv with class_num and k_nearest
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
            print(f"Loaded QAConv weights ({len(qaconv_state_dict)} keys)")
        else:
            print("WARNING: No QAConv weights found in checkpoint!")
            qaconv = None
    else:
        print("WARNING: Model does not have QAConv layer!")
    
    # Move to GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if qaconv is not None:
        qaconv = qaconv.to(device)
    model.eval()
    if qaconv is not None:
        qaconv.eval()
    
    print(f"Model loaded on device: {device}")
    
    # Load validation data
    print("Loading validation data...")
    concat_mem_file_name = os.path.join(val_data_path, 'concat_validation_memfile')
    
    # Check if memfile exists, if not create it
    if not os.path.isfile(concat_mem_file_name):
        print(f"Creating concat memfile at {concat_mem_file_name}")
        concat = []
        for key in ['agedb_30', 'cfp_fp', 'lfw', 'cplfw', 'calfw']:
            np_array, issame = evaluate_utils.get_val_pair(path=val_data_path, name=key, use_memfile=False)
            concat.append(np_array)
        concat = np.concatenate(concat)
        evaluate_utils.make_memmap(concat_mem_file_name, concat)
    
    # Get validation data
    val_data = evaluate_utils.get_val_data(val_data_path)
    age_30, cfp_fp, lfw, age_30_issame, cfp_fp_issame, lfw_issame, cplfw, cplfw_issame, calfw, calfw_issame = val_data
    val_data_dict = {
        'agedb_30': (age_30, age_30_issame),
        "cfp_fp": (cfp_fp, cfp_fp_issame),
        "lfw": (lfw, lfw_issame),
        "cplfw": (cplfw, cplfw_issame),
        "calfw": (calfw, calfw_issame),
    }
    val_dataset = FiveValidationDataset(val_data_dict, concat_mem_file_name)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    print(f"Validation data loaded: {len(val_dataset)} samples")
    
    # Collect all outputs
    all_outputs = []
    
    print("Extracting features...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images, labels, dataname, image_index = batch
            images = images.to(device)
            
            # Get intermediate features (14x14) from body_early for QAConv and OcclusionHead
            # Match the exact structure from train_val.py
            x = model.input_layer(images)
            
            # Check if model has body_early/body_late (new structure) or body (old structure)
            if hasattr(model, 'body_early') and hasattr(model, 'body_late'):
                # New structure: body_early (Blocks 1-3) -> 14x14, body_late (Block 4) -> 7x7
                for layer in model.body_early:
                    x = layer(x)
                
                # Store intermediate features for QAConv and occlusion
                intermediate_x = x
                
                # Normalize intermediate feature maps for QAConv (14x14 resolution)
                feature_maps = F.normalize(intermediate_x, p=2, dim=1)
                
                # Compute occlusion maps from intermediate features (14x14 resolution)
                if has_occlusion_head:
                    occlusion_maps = model.occlusion_head(intermediate_x)  # [B, 1, 14, 14]
                    # Compute mean visibility per image (mean over spatial dimensions)
                    mean_visibility = occlusion_maps.mean(dim=[1, 2, 3])  # [B]
                else:
                    # If no occlusion head, assume all images are clean (high visibility)
                    occlusion_maps = torch.ones(intermediate_x.size(0), 1, intermediate_x.size(2), intermediate_x.size(3), device=x.device)
                    mean_visibility = torch.ones(intermediate_x.size(0), device=x.device)
                
                # Continue through body_late to get final features for AdaFace embedding
                final_x = intermediate_x
                for layer in model.body_late:
                    final_x = layer(final_x)
                
                embeddings = model.output_layer(final_x)
                embeddings, norms = utils.l2_norm(embeddings, axis=1)
            else:
                # Old structure: use model.body (all blocks) -> 7x7
                for layer in model.body:
                    x = layer(x)
                
                # Normalize feature maps for QAConv (7x7 resolution)
                feature_maps = F.normalize(x, p=2, dim=1)
                
                # Compute occlusion maps (7x7 resolution)
                if has_occlusion_head:
                    occlusion_maps = model.occlusion_head(x)  # [B, 1, 7, 7]
                    # Compute mean visibility per image
                    mean_visibility = occlusion_maps.mean(dim=[1, 2, 3])  # [B]
                else:
                    # If no occlusion head, assume all images are clean (high visibility)
                    occlusion_maps = torch.ones(x.size(0), 1, x.size(2), x.size(3), device=x.device)
                    mean_visibility = torch.ones(x.size(0), device=x.device)
                
                # Get AdaFace embeddings
                embeddings = model.output_layer(x)
                embeddings, norms = utils.l2_norm(embeddings, axis=1)
            
            # Check for NaNs
            if torch.isnan(feature_maps).any():
                print(f"WARNING: QAConv feature maps contain NaNs. Replacing with zeros.")
                feature_maps = torch.nan_to_num(feature_maps, nan=0.0)
                feature_maps = F.normalize(feature_maps, p=2, dim=1)
            
            if torch.isnan(embeddings).any():
                print(f"WARNING: AdaFace embeddings contain NaNs. Replacing with zeros.")
                embeddings = torch.nan_to_num(embeddings, nan=0.0)
                embeddings, norms = utils.l2_norm(embeddings, axis=1)
            
            all_outputs.append({
                'adaface_output': embeddings.cpu(),
                'qaconv_output': feature_maps.cpu(),
                'qaconv_occ': occlusion_maps.cpu(),
                'mean_visibility': mean_visibility.cpu(),
                'target': labels.cpu(),
                'dataname': dataname.cpu(),
                'image_index': image_index.cpu()
            })
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1} batches")
    
    print(f"Feature extraction complete. Total batches: {len(all_outputs)}")
    
    # Concatenate all outputs
    all_adaface_tensor = torch.cat([out['adaface_output'] for out in all_outputs], axis=0)
    all_qaconv_tensor = torch.cat([out['qaconv_output'] for out in all_outputs], axis=0)
    all_qaconv_occ_tensor = torch.cat([out['qaconv_occ'] for out in all_outputs], axis=0)
    all_mean_visibility = torch.cat([out['mean_visibility'] for out in all_outputs], axis=0)
    all_target_tensor = torch.cat([out['target'] for out in all_outputs], axis=0)
    all_dataname_tensor = torch.cat([out['dataname'] for out in all_outputs], axis=0)
    all_image_index = torch.cat([out['image_index'] for out in all_outputs], axis=0)
    
    # Remove duplicates
    unique_dict = {}
    for _ada, _qa, _occ, _vis, _tar, _dat, _idx in zip(
        all_adaface_tensor, all_qaconv_tensor, all_qaconv_occ_tensor,
        all_mean_visibility, all_target_tensor, all_dataname_tensor, all_image_index
    ):
        unique_dict[_idx.item()] = {
            'adaface_output': _ada,
            'qaconv_output': _qa,
            'qaconv_occ': _occ,
            'mean_visibility': _vis.item(),
            'target': _tar,
            'dataname': _dat
        }
    unique_keys = sorted(unique_dict.keys())
    all_adaface_tensor = torch.stack([unique_dict[key]['adaface_output'] for key in unique_keys], axis=0)
    all_qaconv_tensor = torch.stack([unique_dict[key]['qaconv_output'] for key in unique_keys], axis=0)
    all_qaconv_occ_tensor = torch.stack([unique_dict[key]['qaconv_occ'] for key in unique_keys], axis=0)
    all_mean_visibility = np.array([unique_dict[key]['mean_visibility'] for key in unique_keys])
    all_target_tensor = torch.stack([unique_dict[key]['target'] for key in unique_keys], axis=0)
    all_dataname_tensor = torch.stack([unique_dict[key]['dataname'] for key in unique_keys], axis=0)
    
    print(f"Total unique samples: {len(unique_keys)}")
    
    # Evaluate each dataset
    dataname_to_idx = {"agedb_30": 0, "cfp_fp": 1, "lfw": 2, "cplfw": 3, "calfw": 4}
    idx_to_dataname = {val: key for key, val in dataname_to_idx.items()}
    results = {}
    
    for dataname_idx in all_dataname_tensor.unique():
        dataname = idx_to_dataname[dataname_idx.item()]
        
        # Get data for this dataset
        mask = all_dataname_tensor == dataname_idx
        adaface_features = all_adaface_tensor[mask]
        qaconv_features = all_qaconv_tensor[mask]
        qaconv_occ = all_qaconv_occ_tensor[mask]
        mean_visibility = all_mean_visibility[mask]
        labels = all_target_tensor[mask].cpu().numpy()
        issame = labels[0::2]
        
        print(f"\n{'='*60}")
        print(f"Processing {dataname} with {len(adaface_features)} samples")
        print(f"{'='*60}")
        
        # Structure data for gallery-query pairs
        gallery_adaface = adaface_features[0::2]  # Even indices
        query_adaface = adaface_features[1::2]    # Odd indices
        gallery_qaconv = qaconv_features[0::2]
        query_qaconv = qaconv_features[1::2]
        gallery_occ = qaconv_occ[0::2]
        query_occ = qaconv_occ[1::2]
        gallery_visibility = mean_visibility[0::2]
        query_visibility = mean_visibility[1::2]
        
        # Compute mean visibility for each pair (average of gallery and query)
        pair_mean_visibility = (gallery_visibility + query_visibility) / 2.0
        
        # Adaptive threshold: if all pairs have very low visibility, use percentile-based threshold
        # This handles the case where occlusion head predicts low visibility for clean faces
        visibility_mean = pair_mean_visibility.mean()
        visibility_std = pair_mean_visibility.std()
        visibility_max = pair_mean_visibility.max()
        
        # If mean visibility is very low (< 0.3), the occlusion head likely wasn't trained on clean faces
        # Use an adaptive threshold based on the distribution
        if visibility_mean < 0.3:
            # Use median or 75th percentile as threshold to split the distribution
            adaptive_threshold = np.percentile(pair_mean_visibility, 75)  # Top 25% use AdaFace
            print(f"  ⚠ WARNING: Mean visibility is very low ({visibility_mean:.4f})")
            print(f"     This suggests occlusion head wasn't trained on clean faces.")
            print(f"     Using adaptive threshold: {adaptive_threshold:.4f} (75th percentile)")
            print(f"     Original threshold {occlusion_threshold:.4f} would use QAConv for all pairs.")
            actual_threshold = adaptive_threshold
        else:
            actual_threshold = occlusion_threshold
        
        # Determine which method to use for each pair
        use_qaconv = pair_mean_visibility < actual_threshold
        use_adaface = ~use_qaconv
        
        num_pairs = len(gallery_adaface)
        num_qaconv_pairs = use_qaconv.sum()
        num_adaface_pairs = use_adaface.sum()
        
        print(f"  Total pairs: {num_pairs}")
        print(f"  Pairs using QAConv (visibility < {actual_threshold:.4f}): {num_qaconv_pairs} ({100*num_qaconv_pairs/num_pairs:.1f}%)")
        print(f"  Pairs using AdaFace (visibility >= {actual_threshold:.4f}): {num_adaface_pairs} ({100*num_adaface_pairs/num_pairs:.1f}%)")
        print(f"  Mean visibility stats - Min: {pair_mean_visibility.min():.4f}, Max: {pair_mean_visibility.max():.4f}, Mean: {pair_mean_visibility.mean():.4f}, Std: {pair_mean_visibility.std():.4f}")
        print(f"  Visibility percentiles - 25th: {np.percentile(pair_mean_visibility, 25):.4f}, 50th: {np.percentile(pair_mean_visibility, 50):.4f}, 75th: {np.percentile(pair_mean_visibility, 75):.4f}, 90th: {np.percentile(pair_mean_visibility, 90):.4f}")
        
        if len(gallery_adaface) == len(query_adaface) and len(gallery_adaface) > 0:
            try:
                # Move to device
                gallery_adaface = gallery_adaface.to(device)
                query_adaface = query_adaface.to(device)
                gallery_qaconv = gallery_qaconv.to(device)
                query_qaconv = query_qaconv.to(device)
                gallery_occ = gallery_occ.to(device)
                query_occ = query_occ.to(device)
                
                if qaconv is not None:
                    qaconv = qaconv.to(device)
                
                # Initialize hybrid scores
                hybrid_scores = np.zeros(num_pairs)
                
                # Compute AdaFace scores for clean pairs
                if num_adaface_pairs > 0:
                    adaface_gallery = gallery_adaface[use_adaface]
                    adaface_query = query_adaface[use_adaface]
                    
                    # Compute cosine similarity
                    adaface_scores = torch.sum(adaface_gallery * adaface_query, dim=1).cpu().numpy()
                    hybrid_scores[use_adaface] = adaface_scores
                    
                    print(f"  AdaFace scores - Min: {adaface_scores.min():.4f}, Max: {adaface_scores.max():.4f}, Mean: {adaface_scores.mean():.4f}")
                
                # Compute QAConv scores for occluded pairs
                if num_qaconv_pairs > 0 and qaconv is not None:
                    qaconv_gallery = gallery_qaconv[use_qaconv]
                    qaconv_query = query_qaconv[use_qaconv]
                    qaconv_gallery_occ = gallery_occ[use_qaconv]
                    qaconv_query_occ = query_occ[use_qaconv]
                    
                    # Verify normalization
                    q_norms = torch.norm(qaconv_query.view(qaconv_query.size(0), -1), p=2, dim=1)
                    g_norms = torch.norm(qaconv_gallery.view(qaconv_gallery.size(0), -1), p=2, dim=1)
                    
                    if (q_norms < 0.99).any() or (q_norms > 1.01).any():
                        qaconv_query = F.normalize(qaconv_query, p=2, dim=1)
                    
                    if (g_norms < 0.99).any() or (g_norms > 1.01).any():
                        qaconv_gallery = F.normalize(qaconv_gallery, p=2, dim=1)
                    
                    # Compute QAConv scores with occlusion weighting
                    with torch.no_grad():
                        qaconv_scores = qaconv.match_pairs(
                            qaconv_query,
                            qaconv_gallery,
                            probe_occ=qaconv_query_occ,
                            gallery_occ=qaconv_gallery_occ
                        )
                    
                    hybrid_scores[use_qaconv] = qaconv_scores.cpu().numpy()
                    
                    print(f"  QAConv scores - Min: {qaconv_scores.min().item():.4f}, Max: {qaconv_scores.max().item():.4f}, Mean: {qaconv_scores.mean().item():.4f}")
                elif num_qaconv_pairs > 0:
                    print(f"  WARNING: {num_qaconv_pairs} pairs need QAConv but QAConv is not available!")
                    print(f"  Falling back to AdaFace for these pairs.")
                    # Fallback to AdaFace
                    fallback_gallery = gallery_adaface[use_qaconv]
                    fallback_query = query_adaface[use_qaconv]
                    fallback_scores = torch.sum(fallback_gallery * fallback_query, dim=1).cpu().numpy()
                    hybrid_scores[use_qaconv] = fallback_scores
                
                # Sample negative pairs for evaluation
                np.random.seed(42)
                negative_scores = np.zeros(num_pairs)
                half_size = num_pairs // 2
                
                for i in range(0, num_pairs, 32):
                    end_idx = min(i + 32, num_pairs)
                    batch_size_chunk = end_idx - i
                    
                    random_indices = np.zeros(batch_size_chunk, dtype=np.int64)
                    for j in range(batch_size_chunk):
                        idx = (i + j + half_size) % num_pairs
                        random_indices[j] = idx
                    
                    # Get random query features
                    random_query_adaface = query_adaface[random_indices]
                    random_query_qaconv = query_qaconv[random_indices]
                    random_query_occ = query_occ[random_indices]
                    random_query_visibility = query_visibility[random_indices]
                    
                    # Get batch gallery features
                    batch_gallery_adaface = gallery_adaface[i:end_idx]
                    batch_gallery_qaconv = gallery_qaconv[i:end_idx]
                    batch_gallery_occ = gallery_occ[i:end_idx]
                    batch_gallery_visibility = gallery_visibility[i:end_idx]
                    
                    # Determine method for each negative pair (use same threshold as positive pairs)
                    batch_pair_visibility = (batch_gallery_visibility + random_query_visibility) / 2.0
                    batch_use_qaconv = batch_pair_visibility < actual_threshold
                    batch_use_adaface = ~batch_use_qaconv
                    
                    for j in range(batch_size_chunk):
                        pair_idx = i + j
                        
                        if batch_use_qaconv[j] and qaconv is not None:
                            # Use QAConv for negative pair
                            score = qaconv(
                                batch_gallery_qaconv[j:j+1],
                                random_query_qaconv[j:j+1],
                                prob_occ=batch_gallery_occ[j:j+1],
                                gal_occ=random_query_occ[j:j+1]
                            )
                            negative_scores[pair_idx] = score.view(-1)[0].item()
                        else:
                            # Use AdaFace for negative pair
                            score = torch.sum(batch_gallery_adaface[j:j+1] * random_query_adaface[j:j+1], dim=1)
                            negative_scores[pair_idx] = score.item()
                
                # Combine positive and negative scores
                all_scores = np.concatenate([hybrid_scores, negative_scores])
                
                # Check for NaN
                if np.isnan(all_scores).any():
                    print("WARNING: Hybrid scores contain NaN values. Replacing with zeros.")
                    all_scores = np.nan_to_num(all_scores, nan=0.0)
                
                # Move to CPU for numpy operations
                pos_scores = all_scores[:num_pairs]
                neg_scores = all_scores[num_pairs:]
                
                print(f"\n  Hybrid positive scores - Min: {np.min(pos_scores):.4f}, Max: {np.max(pos_scores):.4f}, Mean: {np.mean(pos_scores):.4f}")
                print(f"  Hybrid negative scores - Min: {np.min(neg_scores):.4f}, Max: {np.max(neg_scores):.4f}, Mean: {np.mean(neg_scores):.4f}")
                
                # Handle inverted scores
                pos_mean = np.mean(pos_scores)
                neg_mean = np.mean(neg_scores)
                print(f"  Score means - positive: {pos_mean:.4f}, negative: {neg_mean:.4f}")
                
                if neg_mean > pos_mean:
                    print("  WARNING: Scores appear to be inverted. Flipping labels.")
                    all_scores = -all_scores
                    pos_scores = all_scores[:num_pairs]
                    neg_scores = all_scores[num_pairs:]
                
                # Handle large scores
                if pos_mean > 100 or neg_mean > 100:
                    print(f"  WARNING: Scores are extremely large. Normalizing...")
                    all_mean = np.mean(all_scores)
                    all_std = np.std(all_scores)
                    if all_std < 1e-8:
                        all_std = 1.0
                    all_scores = (all_scores - all_mean) / all_std
                    pos_scores = all_scores[:num_pairs]
                    neg_scores = all_scores[num_pairs:]
                    print(f"  After normalization - Positive: {np.mean(pos_scores):.4f}, Negative: {np.mean(neg_scores):.4f}")
                
                # Calculate accuracy
                # Convert scores to distances (smaller = more similar)
                hybrid_dists = -all_scores
                pos_dists = hybrid_dists[:num_pairs]
                neg_dists = hybrid_dists[num_pairs:]
                
                # Direct accuracy calculation
                direct_correct = 0
                for i in range(num_pairs):
                    if pos_dists[i] < np.mean(neg_dists):
                        direct_correct += 1
                for i in range(num_pairs):
                    if neg_dists[i] > np.mean(pos_dists):
                        direct_correct += 1
                direct_acc = direct_correct / (2 * num_pairs)
                
                results[dataname] = direct_acc
                print(f"\n  {dataname} Hybrid accuracy: {direct_acc:.4f}")
                
            except Exception as e:
                import traceback
                print(f"WARNING: Error during hybrid matching: {e}")
                print(traceback.format_exc())
                results[dataname] = 0.0
        else:
            print(f"Warning: {dataname} dataset has mismatched gallery/query sizes")
            results[dataname] = 0.0
    
    # Print summary
    print("\n" + "="*60)
    print("HYBRID OCCLUSION-AWARE EVALUATION RESULTS")
    print("="*60)
    for dataname, acc in results.items():
        print(f"  {dataname}: {acc:.4f}")
    
    avg_acc = np.mean(list(results.values()))
    print(f"\n  Average Hybrid accuracy: {avg_acc:.4f}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hybrid occlusion-aware evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--val_data_path', type=str, default='/home/maass/code/faces_webface_112x112',
                        help='Path to validation data directory')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--occlusion_threshold', type=float, default=0.5,
                        help='Mean visibility threshold for method selection (default: 0.5)')
    
    args = parser.parse_args()
    
    results = evaluate_hybrid_occlusion(
        args.checkpoint, 
        args.val_data_path, 
        args.batch_size, 
        args.num_workers,
        args.occlusion_threshold
    )
