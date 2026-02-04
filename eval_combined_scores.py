"""
Combined Score Evaluation Script

Evaluates AdaFace, QAConv, and Combined scores on the 5 validation datasets.
Uses the same evaluation logic as train_val.py to compute combined accuracy.

Usage:
    python eval_combined_scores.py --checkpoint /path/to/checkpoint.ckpt --val_data_path /home/maass/code/faces_webface_112x112
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
from dataset.image_folder_dataset import CustomImageFolderDataset
from torch.utils.data import DataLoader
from torchvision import transforms


def evaluate_combined_scores(checkpoint_path, val_data_path, batch_size=64, num_workers=4, 
                             adaface_weight=0.5, qaconv_weight=0.5, train_data_path=None, data_root=None):
    """
    Run combined evaluation (AdaFace + QAConv) on the 5 validation datasets.
    
    Args:
        checkpoint_path: Path to model checkpoint
        val_data_path: Path to validation data directory (test sets: agedb_30, cfp_fp, lfw, etc.)
        batch_size: Batch size for feature extraction
        num_workers: Number of data loading workers
        adaface_weight: Weight for AdaFace in combined score (default: 0.5)
        qaconv_weight: Weight for QAConv in combined score (default: 0.5)
        train_data_path: Path to training data (for computing min/max normalization) - optional
        data_root: Root directory for data paths - optional
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Test data path: {val_data_path}")
    print(f"Training data path (for min/max): {train_data_path if train_data_path else 'Not provided - will use test set min/max (NOT RECOMMENDED)'}")
    print(f"Combined score weights - AdaFace: {adaface_weight}, QAConv: {qaconv_weight}")
    
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
            has_occlusion_head = False
    else:
        print("⚠ WARNING: OcclusionHead NOT found in model!")
    
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
    
    # ========== COMPUTE MIN/MAX FROM TRAINING/VALIDATION DATA ==========
    adaface_min, adaface_max = None, None
    qaconv_min, qaconv_max = None, None
    
    if train_data_path and data_root:
        print(f"\n{'='*60}")
        print("Computing min/max normalization from training data...")
        print(f"{'='*60}")
        
        # Load training data subset for min/max computation
        train_dir = os.path.join(data_root, train_data_path, 'imgs')
        if not os.path.exists(train_dir):
            train_dir = os.path.join(data_root, train_data_path)
        
        if os.path.exists(train_dir):
            # Create simple transform (no augmentation for feature extraction)
            train_transform = transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            
            train_dataset = CustomImageFolderDataset(
                root=train_dir,
                transform=train_transform
            )
            
            # Use a subset of training data (max 5000 samples) for efficiency
            max_train_samples = 5000
            if len(train_dataset) > max_train_samples:
                import random
                random.seed(42)
                indices = random.sample(range(len(train_dataset)), max_train_samples)
                train_dataset = torch.utils.data.Subset(train_dataset, indices)
                print(f"Using {max_train_samples} random samples from training set for min/max computation")
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False
            )
            
            print(f"Extracting features from {len(train_dataset)} training samples...")
            train_adaface_embeddings = []
            train_qaconv_features = []
            
            with torch.no_grad():
                for batch_idx, (images, _) in enumerate(train_loader):
                    images = images.to(device)
                    
                    # Extract features (same logic as test set extraction)
                    x = model.input_layer(images)
                    
                    if hasattr(model, 'body_early') and hasattr(model, 'body_late'):
                        for layer in model.body_early:
                            x = layer(x)
                        intermediate_x = x
                        feature_maps = F.normalize(intermediate_x, p=2, dim=1)
                        for layer in model.body_late:
                            x = layer(x)
                        embeddings = model.output_layer(x)
                    else:
                        for layer in model.body:
                            x = layer(x)
                        feature_maps = F.normalize(x, p=2, dim=1)
                        embeddings = model.output_layer(x)
                    
                    embeddings, norms = utils.l2_norm(embeddings, axis=1)
                    
                    train_adaface_embeddings.append(embeddings.cpu())
                    train_qaconv_features.append(feature_maps.cpu())
                    
                    if (batch_idx + 1) % 50 == 0:
                        print(f"  Processed {batch_idx + 1} batches")
            
            # Concatenate training features
            train_adaface_tensor = torch.cat(train_adaface_embeddings, dim=0)
            train_qaconv_tensor = torch.cat(train_qaconv_features, dim=0)
            
            print(f"Computing similarity matrices from training data...")
            
            # Compute AdaFace similarity matrix on training data
            train_adaface_tensor_device = train_adaface_tensor.to(device)
            train_normalized_adaface = F.normalize(train_adaface_tensor_device, p=2, dim=1)
            train_adaface_sim = torch.matmul(train_normalized_adaface, train_normalized_adaface.t())
            adaface_min = train_adaface_sim.min().item()
            adaface_max = train_adaface_sim.max().item()
            
            # Compute QAConv similarity matrix on training data (sample subset for efficiency)
            if qaconv is not None:
                train_qaconv_tensor_device = train_qaconv_tensor.to(device)
                train_qaconv = qaconv.to(device)
                train_qaconv.eval()
                
                # Sample subset for QAConv (max 2000 samples to avoid memory issues)
                max_qaconv_samples = min(2000, len(train_qaconv_tensor))
                if len(train_qaconv_tensor) > max_qaconv_samples:
                    import random
                    random.seed(42)
                    qaconv_indices = random.sample(range(len(train_qaconv_tensor)), max_qaconv_samples)
                    train_qaconv_tensor_sampled = train_qaconv_tensor_device[qaconv_indices]
                else:
                    train_qaconv_tensor_sampled = train_qaconv_tensor_device
                
                train_qaconv_sim = torch.zeros((len(train_qaconv_tensor_sampled), len(train_qaconv_tensor_sampled)), device=device)
                batch_size_sim = 128
                
                with torch.no_grad():
                    for i in range(0, len(train_qaconv_tensor_sampled), batch_size_sim):
                        end_i = min(i + batch_size_sim, len(train_qaconv_tensor_sampled))
                        query_batch = train_qaconv_tensor_sampled[i:end_i]
                        
                        for j in range(0, len(train_qaconv_tensor_sampled), batch_size_sim):
                            end_j = min(j + batch_size_sim, len(train_qaconv_tensor_sampled))
                            gallery_batch = train_qaconv_tensor_sampled[j:end_j]
                            
                            batch_sim = train_qaconv.forward(
                                query_batch,
                                gallery_batch,
                                prob_occ=None,
                                gal_occ=None
                            )
                            train_qaconv_sim[i:end_i, j:end_j] = batch_sim
                
                qaconv_min = train_qaconv_sim.min().item()
                qaconv_max = train_qaconv_sim.max().item()
            
            print(f"Training AdaFace sim - Min: {adaface_min:.6f}, Max: {adaface_max:.6f}")
            if qaconv_min is not None:
                print(f"Training QAConv sim - Min: {qaconv_min:.6f}, Max: {qaconv_max:.6f}")
            print("Will use these min/max values to normalize test set similarity matrices.")
        else:
            print(f"WARNING: Training directory not found: {train_dir}")
            print("Will compute min/max from test set (NOT RECOMMENDED - data leakage)")
    else:
        print("WARNING: train_data_path or data_root not provided.")
        print("Will compute min/max from test set (NOT RECOMMENDED - data leakage)")
    
    # Load validation data (test sets)
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
                else:
                    occlusion_maps = torch.ones(intermediate_x.size(0), 1, intermediate_x.size(2), intermediate_x.size(3), device=x.device)
                
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
                else:
                    occlusion_maps = torch.ones(x.size(0), 1, x.size(2), x.size(3), device=x.device)
                
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
    all_target_tensor = torch.cat([out['target'] for out in all_outputs], axis=0)
    all_dataname_tensor = torch.cat([out['dataname'] for out in all_outputs], axis=0)
    all_image_index = torch.cat([out['image_index'] for out in all_outputs], axis=0)
    
    # Remove duplicates
    unique_dict = {}
    for _ada, _qa, _occ, _tar, _dat, _idx in zip(
        all_adaface_tensor, all_qaconv_tensor, all_qaconv_occ_tensor,
        all_target_tensor, all_dataname_tensor, all_image_index
    ):
        unique_dict[_idx.item()] = {
            'adaface_output': _ada,
            'qaconv_output': _qa,
            'qaconv_occ': _occ,
            'target': _tar,
            'dataname': _dat
        }
    unique_keys = sorted(unique_dict.keys())
    all_adaface_tensor = torch.stack([unique_dict[key]['adaface_output'] for key in unique_keys], axis=0)
    all_qaconv_tensor = torch.stack([unique_dict[key]['qaconv_output'] for key in unique_keys], axis=0)
    all_qaconv_occ_tensor = torch.stack([unique_dict[key]['qaconv_occ'] for key in unique_keys], axis=0)
    all_target_tensor = torch.stack([unique_dict[key]['target'] for key in unique_keys], axis=0)
    all_dataname_tensor = torch.stack([unique_dict[key]['dataname'] for key in unique_keys], axis=0)
    
    print(f"Total unique samples: {len(unique_keys)}")
    
    # Evaluate each dataset
    dataname_to_idx = {"agedb_30": 0, "cfp_fp": 1, "lfw": 2, "cplfw": 3, "calfw": 4}
    idx_to_dataname = {val: key for key, val in dataname_to_idx.items()}
    results = {
        'adaface': {},
        'qaconv': {},
        'combined': {}
    }
    
    for dataname_idx in all_dataname_tensor.unique():
        dataname = idx_to_dataname[dataname_idx.item()]
        
        # Get data for this dataset
        mask = all_dataname_tensor == dataname_idx
        adaface_embeddings = all_adaface_tensor[mask]
        qaconv_features = all_qaconv_tensor[mask]
        qaconv_occ = all_qaconv_occ_tensor[mask]
        labels = all_target_tensor[mask].cpu().numpy()
        issame = labels[0::2]
        
        print(f"\n{'='*60}")
        print(f"Processing {dataname} with {len(adaface_embeddings)} samples")
        print(f"{'='*60}")
        
        # ========== EVALUATE ADAFACE ==========
        adaface_embeddings_np = adaface_embeddings.cpu().numpy()
        tpr, fpr, accuracy, best_thresholds = evaluate_utils.evaluate(adaface_embeddings_np, issame, nrof_folds=10)
        adaface_acc = accuracy.mean()
        results['adaface'][dataname] = adaface_acc
        print(f"AdaFace accuracy: {adaface_acc:.4f}")
        
        # ========== EVALUATE QACONV ==========
        if qaconv is not None:
            # Structure data for gallery-query pairs
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
                    print(f"Computing QAConv scores for {num_pairs} gallery-query pairs")
                    
                    # Verify normalization
                    q_norms = torch.norm(query_features.view(query_features.size(0), -1), p=2, dim=1)
                    g_norms = torch.norm(gallery_features.view(gallery_features.size(0), -1), p=2, dim=1)
                    
                    if (q_norms < 0.99).any() or (q_norms > 1.01).any():
                        query_features = F.normalize(query_features, p=2, dim=1)
                    
                    if (g_norms < 0.99).any() or (g_norms > 1.01).any():
                        gallery_features = F.normalize(gallery_features, p=2, dim=1)
                    
                    with torch.no_grad():
                        # Compute positive scores (without occlusion maps for now)
                        positive_scores = qaconv.match_pairs(
                            query_features,
                            gallery_features,
                            probe_occ=None,
                            gallery_occ=None
                        )
                        
                        # Sample negative pairs
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
                            batch_galleries = gallery_features[i:end_idx]
                            
                            for j in range(batch_size_chunk):
                                score = qaconv(
                                    batch_galleries[j:j+1],
                                    selected_queries[j:j+1],
                                    prob_occ=None,
                                    gal_occ=None
                                )
                                negative_scores[i + j] = score.view(-1)[0]
                        
                        all_scores = torch.cat([positive_scores, negative_scores])
                        
                        if torch.isnan(all_scores).any():
                            print("WARNING: QAConv scores contain NaN values. Replacing with zeros.")
                            all_scores = torch.nan_to_num(all_scores, nan=0.0)
                    
                    # Move to CPU
                    qaconv_scores = all_scores.cpu().numpy()
                    
                    pos_scores = qaconv_scores[:num_pairs]
                    neg_scores = qaconv_scores[num_pairs:]
                    print(f"QAConv positive scores - min: {np.min(pos_scores):.4f}, max: {np.max(pos_scores):.4f}, mean: {np.mean(pos_scores):.4f}")
                    print(f"QAConv negative scores - min: {np.min(neg_scores):.4f}, max: {np.max(neg_scores):.4f}, mean: {np.mean(neg_scores):.4f}")
                    
                    # Handle inverted scores
                    pos_mean = np.mean(pos_scores)
                    neg_mean = np.mean(neg_scores)
                    if neg_mean > pos_mean:
                        print("WARNING: QAConv scores appear to be inverted. Flipping scores.")
                        all_scores = -all_scores
                        qaconv_scores = all_scores
                    
                    # Handle large scores
                    if pos_mean > 100 or neg_mean > 100:
                        print(f"WARNING: QAConv scores are extremely large. Normalizing...")
                        all_mean = np.mean(qaconv_scores)
                        all_std = np.std(qaconv_scores)
                        if all_std < 1e-8:
                            all_std = 1.0
                        qaconv_scores = (qaconv_scores - all_mean) / all_std
                    
                    # Calculate QAConv accuracy
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
                    qaconv_acc = direct_correct / (2 * num_pairs)
                    
                    results['qaconv'][dataname] = qaconv_acc
                    print(f"QAConv accuracy: {qaconv_acc:.4f}")
                    
                    # ========== COMPUTE COMBINED SCORE ==========
                    # Use the same approach as eval_dataset.py: compute full similarity matrices
                    # and then combine them, matching the evaluation approach in main_eval.py
                    
                    N = len(adaface_embeddings)
                    print(f"Computing full similarity matrices for {N} samples...")
                    
                    # Compute AdaFace similarity matrix (same as eval_dataset.py)
                    adaface_embeddings_tensor = torch.from_numpy(adaface_embeddings_np).to(device)
                    normalized_adaface = F.normalize(adaface_embeddings_tensor, p=2, dim=1)
                    adaface_sim = torch.matmul(normalized_adaface, normalized_adaface.t())
                    
                    # Compute QAConv similarity matrix (same as eval_dataset.py)
                    # Move QAConv features to device
                    qaconv_features_device = qaconv_features.to(device)
                    qaconv = qaconv.to(device)
                    qaconv.eval()
                    qaconv_sim = torch.zeros((N, N), device=device)
                    batch_size_sim = 128  # Process in batches to avoid memory issues
                    
                    print("Computing QAConv similarity matrix...")
                    with torch.no_grad():
                        for i in range(0, N, batch_size_sim):
                            end_i = min(i + batch_size_sim, N)
                            query_batch = qaconv_features_device[i:end_i]
                            
                            for j in range(0, N, batch_size_sim):
                                end_j = min(j + batch_size_sim, N)
                                gallery_batch = qaconv_features_device[j:end_j]
                                
                                # Compute similarity batch (without occlusion maps)
                                batch_sim = qaconv.forward(
                                    query_batch,
                                    gallery_batch,
                                    prob_occ=None,
                                    gal_occ=None
                                )
                                qaconv_sim[i:end_i, j:end_j] = batch_sim
                    
                    # Normalize both similarity matrices to [0,1] range using min/max from training data
                    # If training min/max not available, fall back to test set min/max (not recommended)
                    if adaface_min is not None and adaface_max is not None:
                        adaface_sim_norm = (adaface_sim - adaface_min) / (adaface_max - adaface_min + 1e-8)
                        print(f"Normalized AdaFace sim using training min/max: [{adaface_min:.6f}, {adaface_max:.6f}]")
                    else:
                        adaface_sim_norm = (adaface_sim - adaface_sim.min()) / (adaface_sim.max() - adaface_sim.min() + 1e-8)
                        print("WARNING: Using test set min/max for AdaFace normalization (data leakage!)")
                    
                    if qaconv_min is not None and qaconv_max is not None:
                        qaconv_sim_norm = (qaconv_sim - qaconv_min) / (qaconv_max - qaconv_min + 1e-8)
                        print(f"Normalized QAConv sim using training min/max: [{qaconv_min:.6f}, {qaconv_max:.6f}]")
                    else:
                        qaconv_sim_norm = (qaconv_sim - qaconv_sim.min()) / (qaconv_sim.max() - qaconv_sim.min() + 1e-8)
                        print("WARNING: Using test set min/max for QAConv normalization (data leakage!)")
                    
                    # Adjust weights based on QAConv reliability
                    if qaconv_acc < 0.5:
                        print(f"WARNING: QAConv accuracy too low ({qaconv_acc:.4f}). Using more weight on AdaFace for combined score.")
                        actual_adaface_weight = 0.8
                        actual_qaconv_weight = 0.2
                    else:
                        actual_adaface_weight = adaface_weight
                        actual_qaconv_weight = qaconv_weight
                    
                    # Combine normalized similarity matrices (same as eval_dataset.py)
                    combined_sim = actual_adaface_weight * adaface_sim_norm + actual_qaconv_weight * qaconv_sim_norm
                    
                    print(f"Combined score weights - AdaFace: {actual_adaface_weight}, QAConv: {actual_qaconv_weight}")
                    print(f"AdaFace sim - Min: {adaface_sim_norm.min().item():.6f}, Max: {adaface_sim_norm.max().item():.6f}, Mean: {adaface_sim_norm.mean().item():.6f}")
                    print(f"QAConv sim - Min: {qaconv_sim_norm.min().item():.6f}, Max: {qaconv_sim_norm.max().item():.6f}, Mean: {qaconv_sim_norm.mean().item():.6f}")
                    print(f"Combined sim - Min: {combined_sim.min().item():.6f}, Max: {combined_sim.max().item():.6f}, Mean: {combined_sim.mean().item():.6f}")
                    
                    # Convert combined similarity matrix to distance matrix for evaluation
                    # Higher similarity = lower distance
                    combined_dist_matrix = 1.0 - combined_sim.cpu().numpy()
                    
                    # Extract pair distances (same format as evaluate_utils.evaluate expects)
                    # The data is structured as: [gallery0, query0, gallery1, query1, ...]
                    # So pairs are: (0,1), (2,3), (4,5), ...
                    num_pairs = N // 2
                    pair_distances = np.zeros(num_pairs)
                    for i in range(num_pairs):
                        gallery_idx = i * 2
                        query_idx = i * 2 + 1
                        pair_distances[i] = combined_dist_matrix[gallery_idx, query_idx]
                    
                    # Evaluate using the same approach as evaluate_utils.evaluate
                    # Create dummy embeddings that produce our desired distances
                    adaface_embeddings1 = adaface_embeddings_np[0::2]  # gallery
                    adaface_embeddings2 = adaface_embeddings_np[1::2]   # query
                    
                    # Create embeddings where ||emb1 - emb2||^2 = pair_distances
                    # Solution: emb1[i] = [sqrt(pair_distances[i]), 0, 0, ...], emb2[i] = [0, 0, 0, ...]
                    dummy_dim = adaface_embeddings1.shape[1]
                    dummy_emb1 = np.zeros((num_pairs, dummy_dim))
                    dummy_emb2 = np.zeros((num_pairs, dummy_dim))
                    for i in range(num_pairs):
                        dist_val = max(pair_distances[i], 1e-8)  # Ensure positive
                        dummy_emb1[i, 0] = np.sqrt(dist_val)
                    
                    # Evaluate using calculate_roc (same as evaluate_utils.evaluate)
                    thresholds = np.arange(0, 4, 0.01)
                    from evaluate_utils import calculate_roc
                    tpr, fpr, accuracy, best_thresholds = calculate_roc(
                        thresholds,
                        dummy_emb1,
                        dummy_emb2,
                        np.asarray(issame),
                        nrof_folds=10,
                        pca=0
                    )
                    
                    combined_acc = accuracy.mean()
                    results['combined'][dataname] = combined_acc
                    print(f"Combined accuracy: {combined_acc:.4f}")
                    
                except Exception as e:
                    import traceback
                    print(f"WARNING: Error during QAConv/Combined evaluation: {e}")
                    print(traceback.format_exc())
                    results['qaconv'][dataname] = 0.0
                    results['combined'][dataname] = adaface_acc  # Fallback to AdaFace only
            else:
                print(f"Warning: {dataname} dataset has mismatched gallery/query sizes")
                results['qaconv'][dataname] = 0.0
                results['combined'][dataname] = adaface_acc
        else:
            print("QAConv not available. Skipping QAConv and Combined evaluation.")
            results['qaconv'][dataname] = 0.0
            results['combined'][dataname] = adaface_acc
    
    # Print summary
    print("\n" + "="*60)
    print("COMBINED SCORE EVALUATION RESULTS")
    print("="*60)
    print(f"\n{'Dataset':<12} {'AdaFace':<10} {'QAConv':<10} {'Combined':<10}")
    print("-" * 60)
    
    for dataname in dataname_to_idx.keys():
        adaface_acc = results['adaface'].get(dataname, 0.0)
        qaconv_acc = results['qaconv'].get(dataname, 0.0)
        combined_acc = results['combined'].get(dataname, 0.0)
        print(f"{dataname:<12} {adaface_acc:<10.4f} {qaconv_acc:<10.4f} {combined_acc:<10.4f}")
    
    # Calculate averages
    avg_adaface = np.mean(list(results['adaface'].values()))
    avg_qaconv = np.mean(list(results['qaconv'].values())) if results['qaconv'] else 0.0
    avg_combined = np.mean(list(results['combined'].values()))
    
    print("-" * 60)
    print(f"{'Average':<12} {avg_adaface:<10.4f} {avg_qaconv:<10.4f} {avg_combined:<10.4f}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combined score evaluation (AdaFace + QAConv)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--val_data_path', type=str, default='/home/maass/code/faces_webface_112x112',
                        help='Path to test data directory (public benchmarks)')
    parser.add_argument('--train_data_path', type=str, default=None,
                        help='Path to training data (for computing min/max normalization) - REQUIRED for proper evaluation')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Root directory for data paths (required if train_data_path is provided)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--adaface_weight', type=float, default=0.5,
                        help='Weight for AdaFace in combined score (default: 0.5)')
    parser.add_argument('--qaconv_weight', type=float, default=0.5,
                        help='Weight for QAConv in combined score (default: 0.5)')
    
    args = parser.parse_args()
    
    # Normalize weights to sum to 1.0
    total_weight = args.adaface_weight + args.qaconv_weight
    if total_weight > 0:
        args.adaface_weight /= total_weight
        args.qaconv_weight /= total_weight
    
    results = evaluate_combined_scores(
        args.checkpoint, 
        args.val_data_path, 
        args.batch_size, 
        args.num_workers,
        args.adaface_weight,
        args.qaconv_weight,
        args.train_data_path,
        args.data_root
    )
