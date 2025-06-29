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

    def evaluate_features(self, features):
        """Evaluate feature vectors for the entire dataset
        Args:
            features: N x D matrix of features
        Returns:
            scores: N x N similarity matrix
            labels: N x N ground truth matrix (1 for same person, 0 for different)
        """
        # Normalize features
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(features, features.t())
        
        # Create label matrix
        pids = np.array([pid for _, pid in self.dataset])
        label_matrix = pids[:, None] == pids[None, :]
        
        return similarity_matrix.cpu().numpy(), label_matrix