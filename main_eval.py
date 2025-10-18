import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from scipy import interpolate

import net
from eval_dataset import EvaluationDataset
from qaconv import QAConv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def evaluate_verification_and_plot(score_mat, labels,
                                   far_list=(1e-3, 1e-4),
                                   hist_path='verif_hist.png',
                                   roc_path='verif_roc.png'):
    """
    Evaluate verification performance and create plots
    Returns: Dictionary mapping FAR values to corresponding VR (verification rate)
    """
    # Get genuine and imposter scores
    same_id = labels[:, None] == labels[None, :]
    diff_id = ~same_id
    
    genuine_scores = score_mat[same_id]
    imposter_scores = score_mat[diff_id]
    print(f" - Genuine pairs: {len(genuine_scores)}, Imposter pairs: {len(imposter_scores)}")

    # Plot score distributions
    plt.figure(figsize=(10, 6))
    plt.hist(genuine_scores, bins=100, density=True, alpha=0.5, label='Genuine')
    plt.hist(imposter_scores, bins=100, density=True, alpha=0.5, label='Imposter')
    plt.legend()
    plt.title('Score Distribution')
    plt.xlabel('Similarity Score')
    plt.ylabel('Density')
    plt.savefig(hist_path)
    plt.close()

    # Compute ROC
    y_true = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(imposter_scores)])
    y_score = np.concatenate([genuine_scores, imposter_scores])
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    # Plot ROC
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr)
    plt.xscale('log')
    plt.grid(True)
    plt.xlabel('False Accept Rate')
    plt.ylabel('True Accept Rate (Verification Rate)')
    plt.title('ROC Curve')
    
    # Mark requested FAR points
    results = {}
    for far in far_list:
        idx = np.searchsorted(fpr, far)
        if idx > 0:  # Ensure we don't go out of bounds
            vr = tpr[idx-1]
            results[far] = vr
            plt.plot(far, vr, 'ro')
            plt.annotate(f'VR@FAR={far:.0e}: {vr:.2%}', 
                        xy=(far, vr), xytext=(far*2, vr-0.1),
                        arrowprops=dict(arrowstyle='->'))
    
    plt.savefig(roc_path)
    plt.close()
    
    return results


def load_pretrained_model(model_file, architecture='ir_50', k_nearest=32):
    # Load model and pretrained state dict
    state = torch.load(model_file, map_location='cpu')
    state_dict = state['state_dict'] if 'state_dict' in state else state
    # Infer num_classes from the checkpoint
    if 'qaconv.class_embed' in state_dict:
        num_classes = state_dict['qaconv.class_embed'].shape[0]
    else:
        num_classes = None  # fallback or raise error
    # Build model
    model = net.build_model(architecture)
    # Re-initialize qaconv with correct num_classes and k_nearest
    if hasattr(model, 'qaconv') and num_classes is not None:
        num_features = model.qaconv.num_features
        height = model.qaconv.height
        width = model.qaconv.width
        model.qaconv = type(model.qaconv)(num_features, height, width, num_classes=num_classes, k_nearest=k_nearest)
    # Load model weights
    # If checkpoint uses 'model.' prefix, strip it
    if any(k.startswith('model.') for k in state_dict.keys()):
        model_state = {k[6:]: v for k, v in state_dict.items() if k.startswith('model.')}
        model.load_state_dict(model_state, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Face Verification Evaluation')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the pretrained model checkpoint')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the dataset directory')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for feature extraction')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--adaface-weight', type=float, default=0.5,
                        help='Weight for AdaFace scores in combined evaluation')
    parser.add_argument('--qaconv-weight', type=float, default=0.5,
                        help='Weight for QAConv scores in combined evaluation')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load model
    print("Loading pretrained model...")
    model = load_pretrained_model(args.model_path, k_nearest=32)
    model.to(device)

    # Get QAConv matcher from model
    qaconv_matcher = model.qaconv if hasattr(model, 'qaconv') else None
    if qaconv_matcher is None:
        print("Warning: Model does not have QAConv matcher")

    # load evaluation dataset
    print("Loading evaluation dataset...")
    dataset = EvaluationDataset(args.data_path)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True)
    print(f"Number of samples: {len(dataset)}, Number of identities: {dataset.num_ids}")

    # feature extraction
    print("Extracting features...")
    adaface_feats_list, qaconv_feats_list, labels_list = [], [], []
    with torch.no_grad():
        for imgs, labs in tqdm(loader, desc="Feature Extraction"):
            imgs = imgs.to(device)
            # Get AdaFace embeddings (model now returns 3 values: embeddings, norms, occlusion_maps)
            embeddings, norms, _ = model(imgs)  # Ignore occlusion maps for evaluation
            adaface_feats_list.append(embeddings.cpu())
            
            # Get QAConv feature maps if available
            if qaconv_matcher is not None:
                # Get feature maps before final layer for QAConv
                feature_maps = model.body(model.input_layer(imgs))
                qaconv_feats_list.append(feature_maps.cpu())
            
            labels_list.extend(labs.numpy())
            
    all_adaface_feats = torch.cat(adaface_feats_list, dim=0)
    all_labels = np.array(labels_list)
    print(f"Extracted AdaFace features shape: {all_adaface_feats.shape}")
    
    if qaconv_matcher is not None:
        all_qaconv_feats = torch.cat(qaconv_feats_list, dim=0)
        print(f"Extracted QAConv features shape: {all_qaconv_feats.shape}")
    else:
        all_qaconv_feats = None

    # Evaluate using all methods
    print("\nComputing similarity matrices and evaluating...")
    results = dataset.evaluate_features(
        adaface_features=all_adaface_feats.to(device),
        qaconv_features=all_qaconv_feats.to(device) if qaconv_matcher is not None else None,
        qaconv_matcher=qaconv_matcher,
        weights=(args.adaface_weight, args.qaconv_weight)
    )

    # Evaluate each method
    methods_to_evaluate = ['adaface']
    if qaconv_matcher is not None:
        methods_to_evaluate.extend(['qaconv', 'combined'])

    method_names = {
        'adaface': 'AdaFace',
        'qaconv': 'QAConv',
        'combined': 'Combined'
    }

    print("\nVerification Results:")
    print("------------------------------------------------------------")
    print("Method          VR@FAR=0.001    VR@FAR=0.0001")
    print("------------------------------------------------------------")

    # Save detailed results to file
    result_txt = os.path.join(args.output_dir, 'verification_results.txt')
    with open(result_txt, 'w') as f:
        for method in methods_to_evaluate:
            vr_far = evaluate_verification_and_plot(
                score_mat=results[method],
                labels=all_labels,
                hist_path=os.path.join(args.output_dir, f'verif_hist_{method_names[method]}.png'),
                roc_path=os.path.join(args.output_dir, f'verif_roc_{method_names[method]}.png')
            )
            
            # Print summary
            print(f"{method_names[method]:<15} {vr_far[1e-3]:.2%}      {vr_far[1e-4]:.2%}")
            
            # Save detailed results
            f.write(f"\n{method_names[method]} Results:\n")
            for far, vr in vr_far.items():
                f.write(f"FAR={far:.0e}\tVR={vr:.4%}\n")

    print(f"\nDetailed results saved to: {result_txt}")


if __name__ == '__main__':
    main()