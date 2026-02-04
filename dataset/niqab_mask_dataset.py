"""
NiqabMaskDataset: Dataset for loading niqab face images with ground truth occlusion masks.

This dataset is used for training the occlusion prediction head with supervised MSE loss.

Directory structure expected:
    root/
        kept_faces/
            image1.jpg
            image2.jpg
            ...
        masks/
            image1_mask.png
            image2_mask.png
            ...

Naming convention:
    Image: {name}.{ext}
    Mask:  {name}_mask.png

Mask format:
    - Grayscale PNG
    - 255 = visible (face), 0 = occluded (niqab)
    - Will be normalized to [0, 1] where 1 = visible, 0 = occluded
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.nn.functional as F


class NiqabMaskDataset(Dataset):
    """
    Dataset for niqab face images with ground truth occlusion masks.

    Args:
        root_dir: Root directory containing 'kept_faces' and 'masks' subdirectories
        image_transform: Transforms to apply to images (e.g., resize, normalize)
        mask_target_size: Target spatial size for masks (default: 7 for 112x112 input)
        image_subdir: Subdirectory name for images (default: 'kept_faces')
        mask_subdir: Subdirectory name for masks (default: 'masks')
        mask_suffix: Suffix added to image name for mask filename (default: '_mask')

    Returns:
        tuple: (image, mask, index)
            - image: Transformed image tensor [3, H, W]
            - mask: Occlusion mask tensor [1, mask_target_size, mask_target_size]
                   Values in [0, 1] where 1 = visible, 0 = occluded
            - index: Sample index (can be used as pseudo-label)
    """

    def __init__(
        self,
        root_dir,
        image_transform=None,
        mask_target_size=7,
        image_subdir='kept_faces',
        mask_subdir='masks',
        mask_suffix='_mask'
    ):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, image_subdir)
        self.mask_dir = os.path.join(root_dir, mask_subdir)
        self.image_transform = image_transform
        self.mask_target_size = mask_target_size
        self.mask_suffix = mask_suffix

        # Validate directories exist
        if not os.path.isdir(self.image_dir):
            raise ValueError(f"Image directory not found: {self.image_dir}")
        if not os.path.isdir(self.mask_dir):
            raise ValueError(f"Mask directory not found: {self.mask_dir}")

        # Find all valid image-mask pairs
        self.samples = self._find_samples()

        if len(self.samples) == 0:
            raise ValueError(f"No valid image-mask pairs found in {root_dir}")

        print(f"NiqabMaskDataset: Found {len(self.samples)} image-mask pairs")

    def _find_samples(self):
        """Find all valid image-mask pairs."""
        samples = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

        # Get all images
        image_files = [f for f in os.listdir(self.image_dir)
                       if os.path.splitext(f.lower())[1] in valid_extensions]

        for img_file in image_files:
            # Get base name without extension
            base_name, ext = os.path.splitext(img_file)

            # Construct mask filename: {base_name}_mask.png
            mask_file = f"{base_name}{self.mask_suffix}.png"
            mask_path = os.path.join(self.mask_dir, mask_file)

            # Check if mask exists
            if os.path.exists(mask_path):
                img_path = os.path.join(self.image_dir, img_file)
                samples.append({
                    'image_path': img_path,
                    'mask_path': mask_path,
                    'base_name': base_name
                })
            else:
                # Try with same extension as image for mask
                mask_file_alt = f"{base_name}{self.mask_suffix}{ext}"
                mask_path_alt = os.path.join(self.mask_dir, mask_file_alt)
                if os.path.exists(mask_path_alt):
                    img_path = os.path.join(self.image_dir, img_file)
                    samples.append({
                        'image_path': img_path,
                        'mask_path': mask_path_alt,
                        'base_name': base_name
                    })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')

        # Load mask (grayscale)
        mask = Image.open(sample['mask_path']).convert('L')

        # Apply image transforms
        if self.image_transform is not None:
            image = self.image_transform(image)
        else:
            # Default transform: resize to 112x112 and convert to tensor
            image = transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])(image)

        # Process mask:
        # 1. Convert to tensor [1, H, W]
        # 2. Resize to mask_target_size
        # 3. Normalize to [0, 1] where 1 = visible
        mask = transforms.ToTensor()(mask)  # [1, H, W], values in [0, 1]

        # Resize mask to target size using bilinear interpolation
        # Add batch dimension for F.interpolate
        mask = F.interpolate(
            mask.unsqueeze(0),
            size=(self.mask_target_size, self.mask_target_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # Back to [1, H, W]

        # Ensure values are in [0, 1]
        mask = mask.clamp(0, 1)

        return image, mask, idx

    def get_sample_info(self, idx):
        """Get metadata for a sample (useful for debugging)."""
        return self.samples[idx]


class NiqabMaskDataModule:
    """
    Data module wrapper for NiqabMaskDataset with train/val split support.

    Args:
        root_dir: Root directory for niqab dataset
        batch_size: Batch size for dataloaders
        num_workers: Number of dataloader workers
        val_split: Fraction of data to use for validation (default: 0.1)
        image_size: Input image size (default: 112)
        mask_target_size: Target mask size (default: 7)
    """

    def __init__(
        self,
        root_dir,
        batch_size=32,
        num_workers=4,
        val_split=0.1,
        image_size=112,
        mask_target_size=7,
        seed=42
    ):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.image_size = image_size
        self.mask_target_size = mask_target_size
        self.seed = seed

        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def setup(self):
        """Set up train and validation datasets."""
        # Create full dataset
        full_dataset = NiqabMaskDataset(
            root_dir=self.root_dir,
            image_transform=self.train_transform,
            mask_target_size=self.mask_target_size
        )

        # Split into train/val
        total_size = len(full_dataset)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size

        # Use random split with seed
        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

        # Update val dataset transform
        # Note: This is a workaround since random_split creates Subset objects
        # In practice, you might want to create separate datasets for train/val

        print(f"NiqabMaskDataModule: {train_size} train, {val_size} val samples")

        return self

    def train_dataloader(self):
        """Get training dataloader."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        """Get validation dataloader."""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


def get_default_niqab_transform(image_size=112):
    """Get default transform for niqab images."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
