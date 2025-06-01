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
        bgr_image = ((bgr_image.astype(np.float32) / 255.) - 0.5) / 0.5
        tensor = torch.tensor([bgr_image.transpose(2, 0, 1)]).float() # (3, H, W)
        return tensor, pid