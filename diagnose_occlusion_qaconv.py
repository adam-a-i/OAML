"""
Diagnostic script to find root cause of QAConv NaN scores when occlusion maps are connected.

Usage:
    python diagnose_occlusion_qaconv.py --checkpoint /path/to/checkpoint.ckpt --val_data_path /path/to/val/data
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import net
import utils
import evaluate_utils
from qaconv import QAConv
from dataset.five_validation_dataset import FiveValidationDataset
from torch.utils.data import DataLoader


def print_tensor_stats(name, tensor):
    """Print comprehensive statistics for a tensor."""
    if tensor is None:
        print(f"  {name}: None")
        return

    t = tensor.float()
    print(f"  {name}:")
    print(f"    shape: {tuple(tensor.shape)}")
    print(f"    dtype: {tensor.dtype}")
    print(f"    min: {t.min().item():.6f}, max: {t.max().item():.6f}")
    print(f"    mean: {t.mean().item():.6f}, std: {t.std().item():.6f}")
    print(f"    has_nan: {torch.isnan(t).any().item()}, has_inf: {torch.isinf(t).any().item()}")
    print(f"    num_nan: {torch.isnan(t).sum().item()}, num_inf: {torch.isinf(t).sum().item()}")
    print(f"    num_zeros: {(t == 0).sum().item()} / {t.numel()}")

    # Check for extreme values
    if t.numel() > 0 and not torch.isnan(t).all():
        percentiles = [0, 1, 25, 50, 75, 99, 100]
        try:
            vals = torch.quantile(t[~torch.isnan(t)].flatten().float(),
                                  torch.tensor([p/100 for p in percentiles]))
            print(f"    percentiles: " + ", ".join([f"p{p}={v:.4f}" for p, v in zip(percentiles, vals)]))
        except:
            pass


def diagnose(checkpoint_path, val_data_path, batch_size=8):
    """Run comprehensive diagnostics."""

    print("=" * 70)
    print("STEP 1: CHECKPOINT ANALYSIS")
    print("=" * 70)

    ckpt = torch.load(checkpoint_path, map_location='cpu')

    print(f"\nCheckpoint keys: {list(ckpt.keys())}")
    print(f"\nHyperparameters: {ckpt.get('hyper_parameters', 'NOT FOUND')}")

    # Check state_dict keys
    state_dict = ckpt['state_dict']
    print(f"\nTotal state_dict keys: {len(state_dict)}")

    # Group keys by prefix
    prefixes = {}
    for key in state_dict.keys():
        prefix = key.split('.')[0]
        if prefix not in prefixes:
            prefixes[prefix] = []
        prefixes[prefix].append(key)

    print("\nState dict key prefixes:")
    for prefix, keys in sorted(prefixes.items()):
        print(f"  {prefix}: {len(keys)} keys")

    # Check for occlusion_head keys specifically
    occ_keys = [k for k in state_dict.keys() if 'occlusion' in k.lower()]
    print(f"\nOcclusion-related keys ({len(occ_keys)}):")
    for k in occ_keys[:20]:  # Show first 20
        print(f"  {k}: {state_dict[k].shape}")
    if len(occ_keys) > 20:
        print(f"  ... and {len(occ_keys) - 20} more")

    if len(occ_keys) == 0:
        print("\n*** WARNING: NO OCCLUSION HEAD WEIGHTS IN CHECKPOINT! ***")
        print("    This means the occlusion head has RANDOM UNTRAINED weights!")

    print("\n" + "=" * 70)
    print("STEP 2: MODEL LOADING")
    print("=" * 70)

    hparams = ckpt['hyper_parameters']
    model = net.build_model(model_name=hparams['arch'])

    # Check if model has occlusion head
    has_occ = hasattr(model, 'occlusion_head') and model.occlusion_head is not None
    print(f"\nModel has occlusion_head: {has_occ}")

    if has_occ:
        print(f"OcclusionHead architecture: {model.occlusion_head}")

    # Load weights
    model_state = {}
    for key, val in state_dict.items():
        if key.startswith('model.'):
            model_state[key.replace('model.', '')] = val

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print(f"\nModel weight loading:")
    print(f"  Missing keys: {len(missing)}")
    if missing:
        print(f"    First 10: {missing[:10]}")
    print(f"  Unexpected keys: {len(unexpected)}")
    if unexpected:
        print(f"    First 10: {unexpected[:10]}")

    # Check if occlusion_head weights were loaded
    occ_loaded = [k for k in model_state.keys() if 'occlusion' in k.lower()]
    print(f"\nOcclusion weights loaded: {len(occ_loaded)}")

    # Load QAConv
    qaconv = QAConv(model.qaconv.num_features, model.qaconv.height, model.qaconv.width,
                    num_classes=utils.get_num_class(argparse.Namespace(**hparams)),
                    k_nearest=hparams.get('k_nearest', 20))

    qaconv_state = {}
    for key, val in state_dict.items():
        if key.startswith('qaconv.'):
            qaconv_state[key.replace('qaconv.', '')] = val

    if qaconv_state:
        qaconv.load_state_dict(qaconv_state, strict=False)
        print(f"QAConv weights loaded: {len(qaconv_state)} keys")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    qaconv = qaconv.to(device).eval()

    print("\n" + "=" * 70)
    print("STEP 3: FORWARD PASS ANALYSIS")
    print("=" * 70)

    # Load a small batch of validation data
    val_data = evaluate_utils.get_val_data(val_data_path)
    age_30, cfp_fp, lfw, age_30_issame, cfp_fp_issame, lfw_issame, cplfw, cplfw_issame, calfw, calfw_issame = val_data

    # Take first few images from LFW
    images = torch.from_numpy(lfw[:batch_size]).float()
    print(f"\nInput images shape: {images.shape}")
    images = images.to(device)

    with torch.no_grad():
        # Step-by-step forward pass
        print("\n--- Backbone Forward Pass ---")

        x = model.input_layer(images)
        print_tensor_stats("After input_layer", x)

        for i, layer in enumerate(model.body):
            x = layer(x)
            if i == len(model.body) - 1:
                print_tensor_stats(f"After body (final)", x)

        # This is the feature map before normalization
        raw_features = x.clone()
        print_tensor_stats("Raw feature maps", raw_features)

        # Normalize for QAConv
        feature_maps = F.normalize(x, p=2, dim=1)
        print_tensor_stats("Normalized feature maps (dim=1)", feature_maps)

        # Check norm per spatial location
        norms_per_location = torch.norm(feature_maps, p=2, dim=1)  # [B, H, W]
        print_tensor_stats("Norm per spatial location", norms_per_location)

        # Check total norm (what the eval script computes)
        total_norms = torch.norm(feature_maps.view(feature_maps.size(0), -1), p=2, dim=1)
        print(f"\n  Total norm across all dims: {total_norms}")
        print(f"  Expected (sqrt(H*W)): {np.sqrt(feature_maps.shape[2] * feature_maps.shape[3]):.4f}")

        print("\n--- Occlusion Head Forward Pass ---")

        if has_occ:
            # Check occlusion head weights
            print("\nOcclusionHead weight statistics:")
            for name, param in model.occlusion_head.named_parameters():
                print_tensor_stats(f"  {name}", param.data)

            # Run occlusion head
            occlusion_maps = model.occlusion_head(raw_features)
            print_tensor_stats("Occlusion maps output", occlusion_maps)

            # Check if values are in expected range [0, 1]
            in_range = ((occlusion_maps >= 0) & (occlusion_maps <= 1)).float().mean()
            print(f"\n  Fraction in [0,1] range: {in_range.item():.4f}")

            # Show sample occlusion map
            print(f"\n  Sample occlusion map [0]:\n{occlusion_maps[0, 0].cpu().numpy()}")
        else:
            occlusion_maps = None
            print("  No occlusion head - skipping")

        print("\n--- QAConv Forward Pass (WITH occlusion) ---")

        # Test QAConv with occlusion
        if has_occ and occlusion_maps is not None:
            # Take first 2 samples as probe/gallery pair
            probe = feature_maps[0:1]
            gallery = feature_maps[1:2]
            probe_occ = occlusion_maps[0:1]
            gallery_occ = occlusion_maps[1:2]

            print_tensor_stats("Probe features", probe)
            print_tensor_stats("Gallery features", gallery)
            print_tensor_stats("Probe occlusion", probe_occ)
            print_tensor_stats("Gallery occlusion", gallery_occ)

            # Manual step-through of match_pairs logic
            hw = qaconv.height * qaconv.width

            p_fea = probe.view(1, qaconv.num_features, hw)
            g_fea = gallery.view(1, qaconv.num_features, hw)

            print(f"\n  Reshaped probe: {p_fea.shape}")
            print(f"  Reshaped gallery: {g_fea.shape}")

            # Compute correlation
            corr = torch.einsum('p c s, g c r -> p g r s', p_fea, g_fea)
            print_tensor_stats("Correlation (before occlusion)", corr)

            # Apply occlusion weighting
            p_occ = probe_occ.view(-1)
            g_occ = gallery_occ.view(-1)
            print_tensor_stats("Probe occ flattened", p_occ)
            print_tensor_stats("Gallery occ flattened", g_occ)

            occ_weight = p_occ.unsqueeze(1) * g_occ.unsqueeze(0)
            print_tensor_stats("Occlusion weight matrix", occ_weight)

            corr_weighted = corr * occ_weight.unsqueeze(0).unsqueeze(0)
            print_tensor_stats("Correlation (after occlusion weighting)", corr_weighted)

            # Max pooling
            max_r = corr_weighted.max(dim=2)[0]
            max_s = corr_weighted.max(dim=3)[0]
            pooled = torch.cat((max_r, max_s), dim=-1).view(1, 1, hw * 2)
            print_tensor_stats("After max pooling", pooled)

            # BN
            bn_out = qaconv.bn(pooled)
            print_tensor_stats("After BN", bn_out)

            # Check BN statistics
            print(f"\n  BN running_mean: {qaconv.bn.running_mean[:5] if qaconv.bn.running_mean is not None else 'None'}")
            print(f"  BN running_var: {qaconv.bn.running_var[:5] if qaconv.bn.running_var is not None else 'None'}")

            # FC
            fc_in = bn_out.view(1, hw * 2)
            fc_out = qaconv.fc(fc_in)
            print_tensor_stats("After FC", fc_out)

            # Logit BN
            final_out = qaconv.logit_bn(fc_out)
            print_tensor_stats("Final score (after logit_bn)", final_out)

            # Now try WITHOUT occlusion for comparison
            print("\n--- QAConv Forward Pass (WITHOUT occlusion) ---")

            corr_no_occ = torch.einsum('p c s, g c r -> p g r s', p_fea, g_fea)
            max_r_no = corr_no_occ.max(dim=2)[0]
            max_s_no = corr_no_occ.max(dim=3)[0]
            pooled_no = torch.cat((max_r_no, max_s_no), dim=-1).view(1, 1, hw * 2)
            print_tensor_stats("After max pooling (no occ)", pooled_no)

            bn_out_no = qaconv.bn(pooled_no)
            print_tensor_stats("After BN (no occ)", bn_out_no)

            fc_in_no = bn_out_no.view(1, hw * 2)
            fc_out_no = qaconv.fc(fc_in_no)
            print_tensor_stats("After FC (no occ)", fc_out_no)

            final_out_no = qaconv.logit_bn(fc_out_no)
            print_tensor_stats("Final score (no occ)", final_out_no)

            # Compare
            print("\n--- Comparison ---")
            print(f"  Score WITH occlusion: {final_out.item():.6f}")
            print(f"  Score WITHOUT occlusion: {final_out_no.item():.6f}")

    print("\n" + "=" * 70)
    print("STEP 4: BATCH EVALUATION TEST")
    print("=" * 70)

    # Test match_pairs with a small batch
    with torch.no_grad():
        probe_batch = feature_maps[0::2]  # Even indices
        gallery_batch = feature_maps[1::2]  # Odd indices

        if has_occ and occlusion_maps is not None:
            probe_occ_batch = occlusion_maps[0::2]
            gallery_occ_batch = occlusion_maps[1::2]

            print("\nTesting match_pairs WITH occlusion:")
            scores_with_occ = qaconv.match_pairs(probe_batch, gallery_batch, probe_occ_batch, gallery_occ_batch)
            print_tensor_stats("Scores with occlusion", scores_with_occ)

            print("\nTesting match_pairs WITHOUT occlusion:")
            scores_without_occ = qaconv.match_pairs(probe_batch, gallery_batch, None, None)
            print_tensor_stats("Scores without occlusion", scores_without_occ)
        else:
            print("\nTesting match_pairs (no occlusion head):")
            scores = qaconv.match_pairs(probe_batch, gallery_batch, None, None)
            print_tensor_stats("Scores", scores)

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Diagnose QAConv + Occlusion issues')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--val_data_path', type=str, required=True,
                        help='Path to validation data directory')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for testing')

    args = parser.parse_args()
    diagnose(args.checkpoint, args.val_data_path, args.batch_size)
