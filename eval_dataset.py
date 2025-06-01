import os.path as osp
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from face import Face

class EvaluationDataset(Dataset):
    """
    Dataset for evaluation that returns individual images and their labels.
    """
    def __init__(self, root):
        print(f"Loading evaluation data from: {root}")
        dataset = Face(root)
        self.dataset = dataset.data
        self.num_ids = dataset.num_ids
        self.root = root

        if not self.dataset:
            raise RuntimeError(f"No data found in {self.root}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        fname, pid = self.dataset[index]
        fpath = osp.join(self.root, fname)
        bgr_image = cv2.imread(fpath)
        # Resize to 112x112 which is standard for face recognition
        bgr_image = cv2.resize(bgr_image, (112, 112))
        # Normalize to [-1, 1] range
        bgr_image = ((bgr_image.astype(np.float32) / 255.) - 0.5) / 0.5
        # Convert to tensor with shape [3, 112, 112]
        tensor = torch.from_numpy(bgr_image.transpose(2, 0, 1)).float()
        return tensor, pid

    def get_all_pairs(self):
        """Get all possible pairs of images and their labels"""
        n = len(self)
        # Create label matrix where (i,j) is 1 if images i and j are same person
        pids = np.array([pid for _, pid in self.dataset])
        label_matrix = pids[:, None] == pids[None, :]
        
        # Convert to indices and labels
        indices = np.triu_indices(n, k=1)  # Get upper triangular indices, excluding diagonal
        labels = label_matrix[indices]
        
        return indices[0], indices[1], labels

    def evaluate_features(self, adaface_features, qaconv_features=None, qaconv_matcher=None, weights=(0.5, 0.5)):
        """Evaluate feature vectors using multiple methods
        Args:
            adaface_features: N x D matrix of AdaFace embeddings
            qaconv_features: N x C x H x W tensor of QAConv feature maps (optional)
            qaconv_matcher: QAConv matcher module (optional)
            weights: Tuple of (adaface_weight, qaconv_weight) for score combination
        Returns:
            Dictionary containing similarity matrices for each method
        """
        N = len(adaface_features)
        device = adaface_features.device
        
        # Process AdaFace similarities
        print("Computing AdaFace similarities...")
        normalized_features = torch.nn.functional.normalize(adaface_features, p=2, dim=1)
        adaface_sim = torch.matmul(normalized_features, normalized_features.t())
        
        # Process QAConv similarities
        print("Computing QAConv similarities...")
        if qaconv_features is not None and qaconv_matcher is not None:
            qaconv_sim = torch.zeros((N, N), device=device)
            batch_size = 128  # Process in smaller batches
            
            # Ensure QAConv matcher is in eval mode
            qaconv_matcher.eval()
            
            with torch.no_grad():
                for i in range(0, N, batch_size):
                    end_i = min(i + batch_size, N)
                    query_batch = qaconv_features[i:end_i]  # Query batch
                    
                    for j in range(0, N, batch_size):
                        end_j = min(j + batch_size, N)
                        print(f"Processing batch {i//batch_size + 1}/{(N-1)//batch_size + 1} x {j//batch_size + 1}/{(N-1)//batch_size + 1}")
                        gallery_batch = qaconv_features[j:end_j]  # Gallery batch
                        
                        # Use QAConv's forward method which can handle different batch sizes
                        batch_sim = qaconv_matcher(query_batch, gallery_batch)
                        qaconv_sim[i:end_i, j:end_j] = batch_sim
            
            # Analyze QAConv score distribution
            print("\nQAConv Score Statistics:")
            print(f"Min score: {qaconv_sim.min().item():.4f}")
            print(f"Max score: {qaconv_sim.max().item():.4f}")
            print(f"Mean score: {qaconv_sim.mean().item():.4f}")
            print(f"Std score: {qaconv_sim.std().item():.4f}")
            
            # Normalize both similarity matrices to [0,1] range for fair combination
            adaface_sim = (adaface_sim - adaface_sim.min()) / (adaface_sim.max() - adaface_sim.min())
            qaconv_sim = (qaconv_sim - qaconv_sim.min()) / (qaconv_sim.max() - qaconv_sim.min())
            
            # Compute combined similarities with normalized scores
            print("\nComputing combined similarities...")
            adaface_weight, qaconv_weight = weights
            print(f"Using weights: AdaFace={adaface_weight:.2f}, QAConv={qaconv_weight:.2f}")
            combined_sim = adaface_weight * adaface_sim + qaconv_weight * qaconv_sim
        else:
            qaconv_sim = torch.zeros_like(adaface_sim)
            
        # Convert to numpy arrays
        results = {
            'adaface': adaface_sim.cpu().numpy(),
            'qaconv': qaconv_sim.cpu().numpy(),
            'combined': combined_sim.cpu().numpy()
        }
        
        return results