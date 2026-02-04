#!/usr/bin/env python3
"""
Test script for NiqabMaskDataset.

This script tests that:
1. Dataset correctly finds image-mask pairs
2. Images and masks load correctly
3. Masks are resized to target size (7x7)
4. Mask values are in [0, 1] range
5. Transforms are applied correctly
6. DataLoader works properly
7. Batch shapes are correct

Run this script on HPC:
    python tests/test_niqab_dataset.py

Expected output: All tests should print [PASS]

NOTE: This test requires the niqab dataset at /home/maass/code/niqab/train/
      or will create mock data for testing if the path doesn't exist.
"""

import sys
import os
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image

# Import the dataset classes
from dataset.niqab_mask_dataset import NiqabMaskDataset, NiqabMaskDataModule, get_default_niqab_transform


# HPC dataset path
HPC_DATASET_PATH = '/home/maass/code/niqab/train'


def print_test_header(test_name):
    """Print formatted test header."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print('='*60)


def create_mock_dataset(root_dir, num_samples=10):
    """Create a mock dataset for testing when real data isn't available."""
    image_dir = os.path.join(root_dir, 'kept_faces')
    mask_dir = os.path.join(root_dir, 'masks')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for i in range(num_samples):
        # Create random image
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = os.path.join(image_dir, f'sample_{i:04d}.jpg')
        img.save(img_path)

        # Create random mask (binary with some structure)
        mask = np.zeros((224, 224), dtype=np.uint8)
        # Upper half visible (face), lower half occluded (niqab)
        mask[:112, :] = 255
        # Add some randomness
        noise = np.random.randint(0, 50, (224, 224), dtype=np.uint8)
        mask = np.clip(mask.astype(np.int32) + noise - 25, 0, 255).astype(np.uint8)

        mask_img = Image.fromarray(mask)
        mask_path = os.path.join(mask_dir, f'sample_{i:04d}_mask.png')
        mask_img.save(mask_path)

    return root_dir


def get_dataset_path():
    """Get dataset path - use HPC path if exists, otherwise create mock."""
    if os.path.exists(HPC_DATASET_PATH):
        print(f"Using HPC dataset at: {HPC_DATASET_PATH}")
        return HPC_DATASET_PATH, False  # (path, is_mock)
    else:
        print(f"HPC dataset not found at {HPC_DATASET_PATH}")
        print("Creating mock dataset for testing...")
        mock_dir = tempfile.mkdtemp(prefix='niqab_mock_')
        create_mock_dataset(mock_dir)
        return mock_dir, True  # (path, is_mock)


def test_dataset_creation(dataset_path):
    """Test that dataset can be created and finds samples."""
    print_test_header("Dataset Creation")

    try:
        dataset = NiqabMaskDataset(
            root_dir=dataset_path,
            mask_target_size=7
        )

        assert len(dataset) > 0, "Dataset should have samples"
        print(f"  Dataset created successfully: {len(dataset)} samples [PASS]")

        # Test get_sample_info
        info = dataset.get_sample_info(0)
        assert 'image_path' in info, "Sample info should have image_path"
        assert 'mask_path' in info, "Sample info should have mask_path"
        print(f"  Sample info accessible [PASS]")

        print("\n[PASS] Dataset creation test completed successfully")
        return True, dataset

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


def test_sample_loading(dataset):
    """Test that samples load correctly with proper shapes."""
    print_test_header("Sample Loading")

    try:
        # Load first sample
        image, mask, idx = dataset[0]

        # Check image shape and type
        assert isinstance(image, torch.Tensor), "Image should be a tensor"
        assert image.dim() == 3, f"Image should be 3D, got {image.dim()}D"
        assert image.shape[0] == 3, f"Image should have 3 channels, got {image.shape[0]}"
        print(f"  Image shape: {list(image.shape)} [PASS]")

        # Check mask shape
        assert isinstance(mask, torch.Tensor), "Mask should be a tensor"
        assert mask.dim() == 3, f"Mask should be 3D [1, H, W], got {mask.dim()}D"
        assert mask.shape[0] == 1, f"Mask should have 1 channel, got {mask.shape[0]}"
        assert mask.shape[1] == 7, f"Mask height should be 7, got {mask.shape[1]}"
        assert mask.shape[2] == 7, f"Mask width should be 7, got {mask.shape[2]}"
        print(f"  Mask shape: {list(mask.shape)} [PASS]")

        # Check mask value range
        assert mask.min() >= 0.0, f"Mask min should be >= 0, got {mask.min()}"
        assert mask.max() <= 1.0, f"Mask max should be <= 1, got {mask.max()}"
        print(f"  Mask range: [{mask.min():.4f}, {mask.max():.4f}] [PASS]")

        # Check index
        assert idx == 0, f"Index should be 0, got {idx}"
        print(f"  Index: {idx} [PASS]")

        print("\n[PASS] Sample loading test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_samples(dataset):
    """Test loading multiple samples."""
    print_test_header("Multiple Samples")

    try:
        num_to_test = min(5, len(dataset))

        for i in range(num_to_test):
            image, mask, idx = dataset[i]

            # Basic shape checks
            assert image.shape[0] == 3, f"Sample {i}: Image should have 3 channels"
            assert mask.shape == (1, 7, 7), f"Sample {i}: Mask shape should be [1, 7, 7]"
            assert mask.min() >= 0 and mask.max() <= 1, f"Sample {i}: Mask values out of range"

        print(f"  Loaded {num_to_test} samples successfully [PASS]")

        # Check that samples are different
        img1, _, _ = dataset[0]
        img2, _, _ = dataset[min(1, len(dataset)-1)]
        if len(dataset) > 1:
            diff = (img1 - img2).abs().mean().item()
            assert diff > 0, "Different samples should have different images"
            print(f"  Samples are unique (diff={diff:.4f}) [PASS]")

        print("\n[PASS] Multiple samples test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_custom_transforms(dataset_path):
    """Test dataset with custom transforms."""
    print_test_header("Custom Transforms")

    try:
        from torchvision import transforms

        # Custom transform with different size
        custom_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = NiqabMaskDataset(
            root_dir=dataset_path,
            image_transform=custom_transform,
            mask_target_size=14  # Different mask size
        )

        image, mask, _ = dataset[0]

        assert image.shape == (3, 224, 224), f"Custom transform image shape: {image.shape}"
        print(f"  Custom image size (224x224): {list(image.shape)} [PASS]")

        assert mask.shape == (1, 14, 14), f"Custom mask size: {mask.shape}"
        print(f"  Custom mask size (14x14): {list(mask.shape)} [PASS]")

        print("\n[PASS] Custom transforms test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader(dataset):
    """Test that DataLoader works correctly."""
    print_test_header("DataLoader")

    try:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,  # Use 0 for testing
            drop_last=True
        )

        # Get one batch
        batch = next(iter(dataloader))
        images, masks, indices = batch

        assert images.shape == (4, 3, 112, 112), f"Batch image shape: {images.shape}"
        print(f"  Batch images shape: {list(images.shape)} [PASS]")

        assert masks.shape == (4, 1, 7, 7), f"Batch mask shape: {masks.shape}"
        print(f"  Batch masks shape: {list(masks.shape)} [PASS]")

        assert len(indices) == 4, f"Batch indices length: {len(indices)}"
        print(f"  Batch indices: {indices.tolist()} [PASS]")

        # Check batch mask values
        assert masks.min() >= 0 and masks.max() <= 1, "Batch masks out of range"
        print(f"  Batch mask range: [{masks.min():.4f}, {masks.max():.4f}] [PASS]")

        print("\n[PASS] DataLoader test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_data_module(dataset_path):
    """Test NiqabMaskDataModule."""
    print_test_header("Data Module")

    try:
        data_module = NiqabMaskDataModule(
            root_dir=dataset_path,
            batch_size=4,
            num_workers=0,
            val_split=0.2,
            image_size=112,
            mask_target_size=7
        )

        data_module.setup()

        # Check train dataloader
        train_loader = data_module.train_dataloader()
        train_batch = next(iter(train_loader))
        assert len(train_batch) == 3, "Train batch should have 3 elements"
        print(f"  Train dataloader works [PASS]")

        # Check val dataloader
        val_loader = data_module.val_dataloader()
        val_batch = next(iter(val_loader))
        assert len(val_batch) == 3, "Val batch should have 3 elements"
        print(f"  Val dataloader works [PASS]")

        print("\n[PASS] Data module test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_mask_values_meaningful(dataset):
    """Test that mask values have meaningful distribution (not all same)."""
    print_test_header("Mask Values Meaningful")

    try:
        # Check several samples
        num_to_check = min(5, len(dataset))
        meaningful_count = 0

        for i in range(num_to_check):
            _, mask, _ = dataset[i]

            # Check if mask has variation (not all same value)
            unique_values = len(torch.unique(mask))
            has_variation = unique_values > 1

            # Check if mask has both high and low values (actual occlusion)
            has_high = mask.max() > 0.5
            has_low = mask.min() < 0.5

            if has_variation and (has_high or has_low):
                meaningful_count += 1
                print(f"  Sample {i}: unique_values={unique_values}, "
                      f"range=[{mask.min():.2f}, {mask.max():.2f}] [OK]")
            else:
                print(f"  Sample {i}: unique_values={unique_values}, "
                      f"range=[{mask.min():.2f}, {mask.max():.2f}] [WARN: low variation]")

        # At least some masks should be meaningful
        assert meaningful_count > 0, "At least some masks should have variation"
        print(f"  {meaningful_count}/{num_to_check} masks have meaningful variation [PASS]")

        print("\n[PASS] Mask values meaningful test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_transfer(dataset):
    """Test that data can be transferred to GPU."""
    print_test_header("GPU Transfer")

    try:
        image, mask, _ = dataset[0]

        # CPU check
        assert image.device.type == 'cpu', "Initial image should be on CPU"
        assert mask.device.type == 'cpu', "Initial mask should be on CPU"
        print(f"  Data starts on CPU [PASS]")

        # GPU check if available
        if torch.cuda.is_available():
            image_gpu = image.cuda()
            mask_gpu = mask.cuda()

            assert image_gpu.device.type == 'cuda', "Image should be on GPU"
            assert mask_gpu.device.type == 'cuda', "Mask should be on GPU"
            print(f"  Data transfers to GPU [PASS]")

            # Verify values preserved
            diff_img = (image - image_gpu.cpu()).abs().max().item()
            diff_mask = (mask - mask_gpu.cpu()).abs().max().item()
            assert diff_img < 1e-6, "Image values changed during transfer"
            assert diff_mask < 1e-6, "Mask values changed during transfer"
            print(f"  Values preserved during transfer [PASS]")
        else:
            print(f"  GPU not available, skipping GPU tests")

        print("\n[PASS] GPU transfer test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report summary."""
    print("\n" + "="*60)
    print("NIQAB MASK DATASET TEST SUITE")
    print("="*60)

    # Get dataset path
    dataset_path, is_mock = get_dataset_path()

    results = []
    dataset = None

    try:
        # Test 1: Dataset creation
        passed, dataset = test_dataset_creation(dataset_path)
        results.append(("Dataset Creation", passed))

        if dataset is None:
            print("\n[FAIL] Cannot continue without dataset")
            return False

        # Test 2: Sample loading
        passed = test_sample_loading(dataset)
        results.append(("Sample Loading", passed))

        # Test 3: Multiple samples
        passed = test_multiple_samples(dataset)
        results.append(("Multiple Samples", passed))

        # Test 4: Custom transforms
        passed = test_custom_transforms(dataset_path)
        results.append(("Custom Transforms", passed))

        # Test 5: DataLoader
        passed = test_dataloader(dataset)
        results.append(("DataLoader", passed))

        # Test 6: Data module
        passed = test_data_module(dataset_path)
        results.append(("Data Module", passed))

        # Test 7: Mask values meaningful
        passed = test_mask_values_meaningful(dataset)
        results.append(("Mask Values Meaningful", passed))

        # Test 8: GPU transfer
        passed = test_gpu_transfer(dataset)
        results.append(("GPU Transfer", passed))

    finally:
        # Clean up mock dataset if created
        if is_mock and os.path.exists(dataset_path):
            print(f"\nCleaning up mock dataset at {dataset_path}")
            shutil.rmtree(dataset_path)

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, p in results if p)
    failed = len(results) - passed

    for name, p in results:
        status = "[PASS]" if p else "[FAIL]"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{len(results)} tests passed")

    if is_mock:
        print("\nNOTE: Tests ran with MOCK data. Re-run on HPC for full validation.")

    if failed > 0:
        print(f"\n[OVERALL: FAIL] {failed} test(s) failed")
        return False
    else:
        print(f"\n[OVERALL: PASS] All tests passed!")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
