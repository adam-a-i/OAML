import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

import net
from eval_dataset import EvaluationDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


def evaluate_verification_and_plot(score_mat, labels,
                                   far_list=(1e-3, 1e-4, 1e-5, 1e-6),
                                   hist_path='verif_hist.png',
                                   roc_path='verif_roc.png'):
    """
    Inputs:
      score_mat: N×N similarity matrix (np.ndarray)
      labels: length-N array of ground-truth IDs (np.ndarray)
    Outputs:
      Saves genuine vs. imposter histogram and ROC curve plots,
      Returns dict mapping FAR -> VR.
    """

    if np.isnan(score_mat).any() or np.isinf(score_mat).any():
        print(f"Warning: score_matrix contains invalid values: nan={np.isnan(score_mat).any()}, inf={np.isinf(score_mat).any()}. Skip evaluation with zero scores.")
        score_mat = np.zeros_like(score_mat)

    # build genuine and imposter masks
    same_id = labels[:, None] == labels[None, :]
    diff_id = ~same_id

    genuine_scores = score_mat[same_id]
    imposter_scores = score_mat[diff_id]
    print(f"Genuine pairs: {len(genuine_scores)}, Imposter pairs: {len(imposter_scores)}")

    # plot score distributions
    plt.figure()
    plt.hist(genuine_scores, bins=100, density=True, histtype='step', label='Genuine')
    plt.hist(imposter_scores, bins=100, density=True, histtype='step', label='Imposter')
    plt.legend()
    plt.title('Score Distribution')
    plt.xlabel('Similarity')
    plt.savefig(hist_path)
    plt.close()

    # compute ROC
    y_true = np.concatenate([np.ones_like(genuine_scores),
                             np.zeros_like(imposter_scores)])
    y_score = np.concatenate([genuine_scores, imposter_scores])
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)

    # plot ROC and mark FAR points
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label='ROC curve')
    vr_dict = {}
    for far in far_list:
        idx = np.searchsorted(fpr, far, side='right') - 1
        idx = np.clip(idx, 0, len(tpr) - 1)
        vr = tpr[idx]
        vr_dict[far] = vr
        plt.scatter(far, vr,
                    label=f'FAR={far:.0e}, VR={vr:.2%}, thr={thresholds[idx]:.3f}')
    plt.xscale('log')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(roc_path)
    plt.close()

    return vr_dict


def load_pretrained_model(model_file, architecture='ir_50'):
    # Load model and pretrained state dict
    model = net.build_model(architecture)
    state = torch.load(model_file)['state_dict']
    model_state = {k[6:]: v for k, v in state.items() if k.startswith('model.')}
    model.load_state_dict(model_state)
    model.eval()
    return model


def main():
    # Configuration
    data_root = 'data'  # TODO: Path to the dataset
    model_architecture = 'ir_50'
    batch_size = 64
    num_workers = 4
    model_file = 'last.ckpt'  # TODO: Path to the pretrained model file
    output_dir = './'  # TODO: Path to save the output files
    os.makedirs(output_dir, exist_ok=True)

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load model
    print("Loading pretrained model...")
    model = load_pretrained_model(model_file, model_architecture)
    model.to(device)

    # load evaluation dataset
    print("Loading evaluation dataset...")
    dataset = EvaluationDataset(data_root)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=True)
    print(f"Number of samples: {len(dataset)}, Number of identities: {dataset.num_ids}")

    # feature extraction
    print("Extracting features...")
    feats_list, labels_list = [], []
    with torch.no_grad():
        for imgs, labs in tqdm(loader, desc="Feature Extraction"):
            imgs = imgs.to(device)
            out = model(imgs)
            feats = out[0] if isinstance(out, tuple) else out
            feats_list.append(feats.cpu())
            labels_list.extend(labs.numpy())
    all_feats = torch.cat(feats_list, dim=0)
    all_labels = np.array(labels_list)
    print(f"Extracted features shape: {all_feats.shape}, labels shape: {all_labels.shape}")

    # compute similarity matrix
    print("Computing similarity matrix...")
    feats_norm = torch.nn.functional.normalize(all_feats.float(), p=2, dim=1)
    score_matrix = torch.matmul(feats_norm, feats_norm.t()).numpy()
    print(f"Score matrix shape: {score_matrix.shape}")

    # evaluation and plotting
    print("Generating histogram and ROC curve...")
    vr = evaluate_verification_and_plot(
        score_mat=score_matrix,
        labels=all_labels,
        far_list=(1e-3, 1e-4, 1e-5, 1e-6),
        hist_path=os.path.join(output_dir, 'verif_hist.png'),
        roc_path=os.path.join(output_dir, 'verif_roc.png')
    )
    print("Verification results (VR @ FAR):")
    for far, v in vr.items():
        print(f"  FAR={far:.0e}: VR={v:.2%}")

    # save results to file
    result_txt = os.path.join(output_dir, 'verification_results.txt')
    with open(result_txt, 'w') as f:
        f.write("FAR\tVR\n")
        for far, v in vr.items():
            f.write(f"{far:.0e}\t{v:.4f}\n")
    print(f"Verification results saved to: {result_txt}")


if __name__ == '__main__':
    main()