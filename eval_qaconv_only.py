"""
Standalone QAConv-only evaluation script.
Uses the exact same evaluation logic from commit e0f384a.
Runs on single GPU to avoid DDP/OOM issues.

Usage:
    python eval_qaconv_only.py --checkpoint /path/to/checkpoint.ckpt --val_data_path /home/maass/code/faces_webface_112x112
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


def evaluate_qaconv(checkpoint_path, val_data_path, batch_size=64, num_workers=4):
    """
    Run QAConv-only evaluation on the 5 validation datasets.
    Uses the exact same logic as the original working code from e0f384a.
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Validation data path: {val_data_path}")
    
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
            print("  Occlusion maps will be random/uninitialized - this will hurt QAConv performance!")
    else:
        print("⚠ WARNING: OcclusionHead NOT found in model!")
        print("  QAConv will run without occlusion weighting - performance may be degraded!")
    
    # Get QAConv from model
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
    
    # Move to GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    qaconv = qaconv.to(device)
    model.eval()
    qaconv.eval()
    
    print(f"Model loaded on device: {device}")
    
    # Load validation data using exact same method as training code
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
            
            # Extract feature maps (exact same logic as e0f384a)
            x = model.input_layer(images)
            for layer in model.body:
                x = layer(x)
            
            # Normalize feature maps
            feature_maps = F.normalize(x, p=2, dim=1)

            # Compute occlusion maps for QAConv weighting
            if has_occlusion_head:
                occlusion_maps = model.occlusion_head(x)
                # Check if occlusion maps are valid (not all zeros/ones)
                if batch_idx == 0:
                    occ_mean = occlusion_maps.mean().item()
                    occ_std = occlusion_maps.std().item()
                    print(f"  Occlusion maps stats: mean={occ_mean:.4f}, std={occ_std:.4f}")
                    if occ_std < 0.01:
                        print(f"  ⚠ WARNING: Occlusion maps have very low variance (std={occ_std:.4f})")
                        print(f"     This suggests occlusion head may not be trained properly!")
            else:
                # Create dummy occlusion maps (all ones = fully visible)
                occlusion_maps = torch.ones(x.size(0), 1, x.size(2), x.size(3), device=x.device)
            
            # Check for NaNs
            if torch.isnan(feature_maps).any():
                print(f"WARNING: QAConv feature maps contain NaNs. Replacing with zeros.")
                feature_maps = torch.nan_to_num(feature_maps, nan=0.0)
                feature_maps = F.normalize(feature_maps, p=2, dim=1)
            
            all_outputs.append({
                'qaconv_output': feature_maps.cpu(),
                'qaconv_occ': occlusion_maps.cpu(),
                'target': labels.cpu(),
                'dataname': dataname.cpu(),
                'image_index': image_index.cpu()
            })
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1} batches")
    
    print(f"Feature extraction complete. Total batches: {len(all_outputs)}")
    
    # Concatenate all outputs
    all_qaconv_tensor = torch.cat([out['qaconv_output'] for out in all_outputs], axis=0)
    all_qaconv_occ_tensor = torch.cat([out['qaconv_occ'] for out in all_outputs], axis=0)
    all_target_tensor = torch.cat([out['target'] for out in all_outputs], axis=0)
    all_dataname_tensor = torch.cat([out['dataname'] for out in all_outputs], axis=0)
    all_image_index = torch.cat([out['image_index'] for out in all_outputs], axis=0)
    
    # Remove duplicates (exact same logic as e0f384a)
    unique_dict = {}
    for _qa, _occ, _tar, _dat, _idx in zip(all_qaconv_tensor, all_qaconv_occ_tensor, all_target_tensor, all_dataname_tensor, all_image_index):
        unique_dict[_idx.item()] = {
            'qaconv_output': _qa,
            'qaconv_occ': _occ,
            'target': _tar,
            'dataname': _dat
        }
    unique_keys = sorted(unique_dict.keys())
    all_qaconv_tensor = torch.stack([unique_dict[key]['qaconv_output'] for key in unique_keys], axis=0)
    all_qaconv_occ_tensor = torch.stack([unique_dict[key]['qaconv_occ'] for key in unique_keys], axis=0)
    all_target_tensor = torch.stack([unique_dict[key]['target'] for key in unique_keys], axis=0)
    all_dataname_tensor = torch.stack([unique_dict[key]['dataname'] for key in unique_keys], axis=0)
    
    print(f"Total unique samples: {len(unique_keys)}")
    
    # Evaluate each dataset (exact same logic as e0f384a)
    dataname_to_idx = {"agedb_30": 0, "cfp_fp": 1, "lfw": 2, "cplfw": 3, "calfw": 4}
    idx_to_dataname = {val: key for key, val in dataname_to_idx.items()}
    results = {}
    
    for dataname_idx in all_dataname_tensor.unique():
        dataname = idx_to_dataname[dataname_idx.item()]
        
        # Get data for this dataset
        mask = all_dataname_tensor == dataname_idx
        qaconv_features = all_qaconv_tensor[mask]
        qaconv_occ = all_qaconv_occ_tensor[mask]
        labels = all_target_tensor[mask].cpu().numpy()
        issame = labels[0::2]
        
        print(f"\nProcessing {dataname} with {len(qaconv_features)} samples")
        
        # Structure data for gallery-query pairs (exact same as e0f384a)
        gallery_features = qaconv_features[0::2]  # Even indices
        query_features = qaconv_features[1::2]    # Odd indices
        gallery_occ = qaconv_occ[0::2]
        query_occ = qaconv_occ[1::2]
        
        if len(gallery_features) == len(query_features) and len(gallery_features) > 0:
            try:
                # Move to device
                gallery_features = gallery_features.to(device)
                query_features = query_features.to(device)
                gallery_occ = gallery_occ.to(device)
                query_occ = query_occ.to(device)
                qaconv = qaconv.to(device)
                
                num_pairs = len(gallery_features)
                print(f"Computing scores for {num_pairs} gallery-query pairs")
                
                # Verify normalization (exact same as e0f384a)
                q_norms = torch.norm(query_features.view(query_features.size(0), -1), p=2, dim=1)
                g_norms = torch.norm(gallery_features.view(gallery_features.size(0), -1), p=2, dim=1)
                
                if (q_norms < 0.99).any() or (q_norms > 1.01).any():
                    print(f"WARNING: Query features not properly normalized. Min: {q_norms.min().item():.4f}, Max: {q_norms.max().item():.4f}")
                    query_features = F.normalize(query_features, p=2, dim=1)
                
                if (g_norms < 0.99).any() or (g_norms > 1.01).any():
                    print(f"WARNING: Gallery features not properly normalized. Min: {g_norms.min().item():.4f}, Max: {g_norms.max().item():.4f}")
                    gallery_features = F.normalize(gallery_features, p=2, dim=1)
                
                with torch.no_grad():
                    # Use occlusion maps for QAConv weighting
                    # The occlusion head is now trained on both niqab (occluded) and clean faces,
                    # so it should correctly predict high visibility for clean faces
                    positive_scores = qaconv.match_pairs(
                        query_features,
                        gallery_features,
                        probe_occ=query_occ,
                        gallery_occ=gallery_occ
                    )
                    
                    # Sample negative pairs (exact same as e0f384a)
                    np.random.seed(42)
                    negative_scores = torch.zeros(num_pairs, device=device)
                    half_size = num_pairs // 2
                    
                    for i in range(0, num_pairs, 32):
                        end_idx = min(i + 32, num_pairs)
                        batch_size_chunk = end_idx - i
                        
                        random_indices = np.zeros(batch_size_chunk, dtype=np.int64)
                        for j in range(batch_size_chunk):
                            idx = (i + j + half_size) % num_pairs
                            random_indices[j] = idx
                        
                        selected_queries = query_features[random_indices]
                        selected_queries_occ = query_occ[random_indices]
                        batch_galleries = gallery_features[i:end_idx]
                        batch_galleries_occ = gallery_occ[i:end_idx]
                        
                        for j in range(batch_size_chunk):
                            # Use occlusion maps for QAConv weighting
                            score = qaconv(
                                batch_galleries[j:j+1],
                                selected_queries[j:j+1],
                                prob_occ=selected_queries_occ[j:j+1],
                                gal_occ=batch_galleries_occ[j:j+1]
                            )
                            negative_scores[i + j] = score.view(-1)[0]
                    
                    all_scores = torch.cat([positive_scores, negative_scores])
                    
                    if torch.isnan(all_scores).any():
                        print("WARNING: QAConv scores contain NaN values. Replacing with zeros.")
                        all_scores = torch.nan_to_num(all_scores, nan=0.0)
                
                # Move to CPU (exact same as e0f384a)
                qaconv_scores = all_scores.cpu().numpy()
                
                pos_scores = qaconv_scores[:num_pairs]
                neg_scores = qaconv_scores[num_pairs:]
                print(f"QAConv positive scores - min: {np.min(pos_scores):.4f}, max: {np.max(pos_scores):.4f}")
                print(f"QAConv negative scores - min: {np.min(neg_scores):.4f}, max: {np.max(neg_scores):.4f}")
                
                pos_mean = np.mean(pos_scores)
                neg_mean = np.mean(neg_scores)
                print(f"QAConv score means - positive: {pos_mean:.4f}, negative: {neg_mean:.4f}")
                
                # Handle inverted scores (exact same as e0f384a)
                if neg_mean > pos_mean:
                    print("WARNING: QAConv scores appear to be inverted. Flipping labels.")
                
                # Handle large scores (exact same as e0f384a)
                if pos_mean > 100 or neg_mean > 100:
                    print(f"WARNING: QAConv scores are extremely large. Normalizing...")
                    all_mean = np.mean(qaconv_scores)
                    all_std = np.std(qaconv_scores)
                    if all_std < 1e-8:
                        all_std = 1.0
                    qaconv_scores = (qaconv_scores - all_mean) / all_std
                    pos_scores = qaconv_scores[:num_pairs]
                    neg_scores = qaconv_scores[num_pairs:]
                    print(f"After normalization - Positive: {np.mean(pos_scores):.4f}, Negative: {np.mean(neg_scores):.4f}")
                
                # Calculate accuracy (exact same as e0f384a)
                qaconv_dists = -qaconv_scores
                qaconv_pos_dists = qaconv_dists[:num_pairs]
                qaconv_neg_dists = qaconv_dists[num_pairs:]
                
                direct_correct = 0
                for i in range(num_pairs):
                    if qaconv_pos_dists[i] < np.mean(qaconv_neg_dists):
                        direct_correct += 1
                for i in range(num_pairs):
                    if qaconv_neg_dists[i] > np.mean(qaconv_pos_dists):
                        direct_correct += 1
                direct_qaconv_acc = direct_correct / (2 * num_pairs)
                
                results[dataname] = direct_qaconv_acc
                print(f"{dataname} QAConv accuracy: {direct_qaconv_acc:.4f}")
                
            except Exception as e:
                import traceback
                print(f"WARNING: Error during QAConv matching: {e}")
                print(traceback.format_exc())
                results[dataname] = 0.0
        else:
            print(f"Warning: {dataname} dataset has mismatched gallery/query sizes")
            results[dataname] = 0.0
    
    # Print summary
    print("\n" + "="*60)
    print("QACONV EVALUATION RESULTS")
    print("="*60)
    for dataname, acc in results.items():
        print(f"  {dataname}: {acc:.4f}")
    
    avg_acc = np.mean(list(results.values()))
    print(f"\n  Average QAConv accuracy: {avg_acc:.4f}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QAConv-only evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--val_data_path', type=str, default='/home/maass/code/faces_webface_112x112',
                        help='Path to validation data directory')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    results = evaluate_qaconv(args.checkpoint, args.val_data_path, args.batch_size, args.num_workers)
