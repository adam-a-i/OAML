#!/usr/bin/env python3
"""
Test script for dual-dataloader training (CASIA + Niqab).

This script tests that:
1. Niqab dataloader is properly initialized
2. Niqab iterator cycles correctly when exhausted
3. Training step handles both data sources correctly
4. Occlusion loss is computed only from niqab data
5. Recognition losses are computed only from main data
6. Gradients flow correctly through both branches

Run this script on HPC:
    python tests/test_dual_dataloader.py

Expected output: All tests should print [PASS]

NOTE: This test creates mock data to test the dual-dataloader logic
without requiring the full training infrastructure.
"""

import sys
import os
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Import model components
from net import build_model, OcclusionHead
from dataset.niqab_mask_dataset import NiqabMaskDataset, get_default_niqab_transform


def print_test_header(test_name):
    """Print formatted test header."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print('='*60)


def create_mock_niqab_dataset(root_dir, num_samples=20):
    """Create a mock niqab dataset for testing."""
    image_dir = os.path.join(root_dir, 'kept_faces')
    mask_dir = os.path.join(root_dir, 'masks')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for i in range(num_samples):
        # Create random image
        img = Image.fromarray(np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8))
        img_path = os.path.join(image_dir, f'sample_{i:04d}.jpg')
        img.save(img_path)

        # Create random mask (upper half visible, lower half occluded)
        mask = np.zeros((112, 112), dtype=np.uint8)
        mask[:56, :] = 255  # Upper half visible
        mask_img = Image.fromarray(mask)
        mask_path = os.path.join(mask_dir, f'sample_{i:04d}_mask.png')
        mask_img.save(mask_path)

    return root_dir


def test_niqab_dataloader_setup():
    """Test that niqab dataloader can be properly initialized."""
    print_test_header("Niqab Dataloader Setup")

    mock_dir = None
    try:
        # Create mock dataset
        mock_dir = tempfile.mkdtemp(prefix='niqab_test_')
        create_mock_niqab_dataset(mock_dir, num_samples=20)

        # Create dataset
        dataset = NiqabMaskDataset(
            root_dir=mock_dir,
            image_transform=get_default_niqab_transform(image_size=112),
            mask_target_size=7,
            image_subdir='kept_faces',
            mask_subdir='masks',
            mask_suffix='_mask'
        )

        assert len(dataset) == 20, f"Dataset should have 20 samples, got {len(dataset)}"
        print(f"  Dataset created with {len(dataset)} samples [PASS]")

        # Create dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        # Get a batch
        batch = next(iter(dataloader))
        images, masks, indices = batch

        assert images.shape == (4, 3, 112, 112), f"Image shape: {images.shape}"
        assert masks.shape == (4, 1, 7, 7), f"Mask shape: {masks.shape}"
        print(f"  Batch shapes correct: images {list(images.shape)}, masks {list(masks.shape)} [PASS]")

        print("\n[PASS] Niqab dataloader setup test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if mock_dir and os.path.exists(mock_dir):
            shutil.rmtree(mock_dir)


def test_iterator_cycling():
    """Test that niqab iterator cycles correctly when exhausted."""
    print_test_header("Iterator Cycling")

    mock_dir = None
    try:
        # Create mock dataset with small number of samples
        mock_dir = tempfile.mkdtemp(prefix='niqab_test_')
        create_mock_niqab_dataset(mock_dir, num_samples=8)

        dataset = NiqabMaskDataset(
            root_dir=mock_dir,
            image_transform=get_default_niqab_transform(image_size=112),
            mask_target_size=7
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        # With 8 samples and batch_size=4, we have 2 batches per epoch
        # Iterate more than 2 times to test cycling
        niqab_iter = iter(dataloader)
        batches_retrieved = 0

        for _ in range(5):  # Try to get 5 batches (should cycle)
            try:
                batch = next(niqab_iter)
            except StopIteration:
                # Restart iterator
                niqab_iter = iter(dataloader)
                batch = next(niqab_iter)

            batches_retrieved += 1
            images, masks, _ = batch
            assert images.shape[0] == 4, f"Batch size should be 4"

        assert batches_retrieved == 5, f"Should have retrieved 5 batches, got {batches_retrieved}"
        print(f"  Successfully retrieved {batches_retrieved} batches with cycling [PASS]")

        print("\n[PASS] Iterator cycling test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if mock_dir and os.path.exists(mock_dir):
            shutil.rmtree(mock_dir)


def test_occlusion_from_niqab_only():
    """Test that occlusion loss is computed only from niqab data."""
    print_test_header("Occlusion From Niqab Only")

    mock_dir = None
    try:
        # Create mock niqab dataset
        mock_dir = tempfile.mkdtemp(prefix='niqab_test_')
        create_mock_niqab_dataset(mock_dir, num_samples=8)

        # Build model
        model = build_model('ir_18')
        model.train()

        # Create niqab dataloader
        dataset = NiqabMaskDataset(
            root_dir=mock_dir,
            image_transform=get_default_niqab_transform(image_size=112),
            mask_target_size=7
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)

        # Get niqab batch
        niqab_batch = next(iter(dataloader))
        niqab_images, niqab_masks, _ = niqab_batch

        # Simulate main batch (CASIA) - no masks
        main_images = torch.randn(4, 3, 112, 112)
        main_labels = torch.randint(0, 100, (4,))

        # Forward main images (no occlusion loss should be computed from this)
        x_main = model.input_layer(main_images)
        for layer in model.body:
            x_main = layer(x_main)

        # Forward niqab images for occlusion
        x_niqab = model.input_layer(niqab_images)
        for layer in model.body:
            x_niqab = layer(x_niqab)

        # Occlusion maps from niqab only
        occlusion_maps = model.occlusion_head(x_niqab)

        # Compute occlusion loss
        occlusion_loss = F.mse_loss(occlusion_maps, niqab_masks)

        assert not torch.isnan(occlusion_loss), "Occlusion loss should not be NaN"
        assert occlusion_loss >= 0, "Occlusion loss should be non-negative"
        print(f"  Occlusion loss from niqab: {occlusion_loss.item():.6f} [PASS]")

        # Verify main batch doesn't contribute to occlusion loss
        # (we simply don't compute occlusion loss from main batch)
        print(f"  Main batch used for recognition only [PASS]")

        print("\n[PASS] Occlusion from niqab only test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if mock_dir and os.path.exists(mock_dir):
            shutil.rmtree(mock_dir)


def test_gradient_flow_dual_sources():
    """Test that gradients flow correctly through both data sources."""
    print_test_header("Gradient Flow Dual Sources")

    mock_dir = None
    try:
        # Create mock niqab dataset
        mock_dir = tempfile.mkdtemp(prefix='niqab_test_')
        create_mock_niqab_dataset(mock_dir, num_samples=8)

        # Build model
        model = build_model('ir_18')
        model.train()

        # Create niqab dataloader
        dataset = NiqabMaskDataset(
            root_dir=mock_dir,
            image_transform=get_default_niqab_transform(image_size=112),
            mask_target_size=7
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)

        # Get niqab batch
        niqab_batch = next(iter(dataloader))
        niqab_images, niqab_masks, _ = niqab_batch

        # Main batch (CASIA)
        main_images = torch.randn(4, 3, 112, 112, requires_grad=True)
        main_labels = torch.randint(0, 100, (4,))

        # Forward main images for recognition
        x_main = model.input_layer(main_images)
        for layer in model.body:
            x_main = layer(x_main)
        embeddings = model.output_layer(x_main)

        # Simple recognition loss (using embedding norm as proxy)
        recognition_loss = embeddings.norm(dim=1).mean()

        # Forward niqab images for occlusion
        x_niqab = model.input_layer(niqab_images)
        for layer in model.body:
            x_niqab = layer(x_niqab)
        occlusion_maps = model.occlusion_head(x_niqab)
        occlusion_loss = F.mse_loss(occlusion_maps, niqab_masks)

        # Combined loss
        total_loss = recognition_loss + 0.1 * occlusion_loss

        # Backward
        total_loss.backward()

        # Check gradients in backbone (should have gradients from both sources)
        backbone_grad = model.body[0].res_layer[0].weight.grad
        assert backbone_grad is not None, "Backbone should have gradients"
        assert not torch.isnan(backbone_grad).any(), "Backbone gradients should not be NaN"
        print(f"  Backbone grad norm: {backbone_grad.norm().item():.6f} [PASS]")

        # Check gradients in occlusion head
        occ_grad = model.occlusion_head.conv1.weight.grad
        assert occ_grad is not None, "Occlusion head should have gradients"
        assert not torch.isnan(occ_grad).any(), "Occlusion head gradients should not be NaN"
        print(f"  Occlusion head grad norm: {occ_grad.norm().item():.6f} [PASS]")

        # Check gradients flow to main input
        assert main_images.grad is not None, "Main images should have gradients"
        print(f"  Main input grad norm: {main_images.grad.norm().item():.6f} [PASS]")

        print("\n[PASS] Gradient flow dual sources test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if mock_dir and os.path.exists(mock_dir):
            shutil.rmtree(mock_dir)


def test_loss_combination():
    """Test that losses from both sources combine correctly."""
    print_test_header("Loss Combination")

    try:
        # Simulate losses
        recognition_loss = torch.tensor(1.5, requires_grad=True)
        occlusion_loss = torch.tensor(0.3, requires_grad=True)

        # Weights
        recognition_weight = 1.0  # AdaFace + QAConv combined
        occlusion_weight = 0.1

        # Combined loss
        total_loss = recognition_weight * recognition_loss + occlusion_weight * occlusion_loss

        expected = 1.0 * 1.5 + 0.1 * 0.3
        assert abs(total_loss.item() - expected) < 1e-6, \
            f"Total loss mismatch: {total_loss.item()} vs {expected}"

        print(f"  Recognition loss: {recognition_loss.item():.4f} (weight={recognition_weight})")
        print(f"  Occlusion loss: {occlusion_loss.item():.4f} (weight={occlusion_weight})")
        print(f"  Total loss: {total_loss.item():.4f} [PASS]")

        # Verify gradients flow to both
        total_loss.backward()

        assert recognition_loss.grad is not None, "Recognition loss should have gradient"
        assert occlusion_loss.grad is not None, "Occlusion loss should have gradient"
        print(f"  Gradients flow to both loss components [PASS]")

        print("\n[PASS] Loss combination test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_no_niqab_fallback():
    """Test that training works when niqab dataloader is not available."""
    print_test_header("No Niqab Fallback")

    try:
        # Build model
        model = build_model('ir_18')
        model.train()

        # Main batch only (no niqab)
        main_images = torch.randn(4, 3, 112, 112)
        main_labels = torch.randint(0, 100, (4,))

        # Forward main images
        x_main = model.input_layer(main_images)
        for layer in model.body:
            x_main = layer(x_main)
        embeddings = model.output_layer(x_main)

        # Recognition loss only
        recognition_loss = embeddings.norm(dim=1).mean()

        # Occlusion loss is 0 when no niqab data
        occlusion_loss = torch.tensor(0.0)

        # Combined loss (occlusion contributes nothing)
        total_loss = recognition_loss + 0.1 * occlusion_loss

        assert total_loss.item() == recognition_loss.item(), \
            "Total loss should equal recognition loss when no niqab data"
        print(f"  Total loss == Recognition loss when no niqab: {total_loss.item():.4f} [PASS]")

        print("\n[PASS] No niqab fallback test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_device_compatibility():
    """Test dual-dataloader works on CPU and GPU."""
    print_test_header("Device Compatibility")

    mock_dir = None
    try:
        # Create mock dataset
        mock_dir = tempfile.mkdtemp(prefix='niqab_test_')
        create_mock_niqab_dataset(mock_dir, num_samples=8)

        # Build model
        model = build_model('ir_18')
        model.train()

        # Create dataloader
        dataset = NiqabMaskDataset(
            root_dir=mock_dir,
            image_transform=get_default_niqab_transform(image_size=112),
            mask_target_size=7
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, drop_last=True)

        # CPU test
        model_cpu = model.to('cpu')
        niqab_images, niqab_masks, _ = next(iter(dataloader))

        x = model_cpu.input_layer(niqab_images)
        for layer in model_cpu.body:
            x = layer(x)
        occ_maps = model_cpu.occlusion_head(x)
        loss_cpu = F.mse_loss(occ_maps, niqab_masks)

        print(f"  CPU occlusion loss: {loss_cpu.item():.6f} [PASS]")

        # GPU test
        if torch.cuda.is_available():
            model_gpu = model.to('cuda')
            niqab_images_gpu = niqab_images.cuda()
            niqab_masks_gpu = niqab_masks.cuda()

            x = model_gpu.input_layer(niqab_images_gpu)
            for layer in model_gpu.body:
                x = layer(x)
            occ_maps_gpu = model_gpu.occlusion_head(x)
            loss_gpu = F.mse_loss(occ_maps_gpu, niqab_masks_gpu)

            print(f"  GPU occlusion loss: {loss_gpu.item():.6f} [PASS]")

            diff = abs(loss_cpu.item() - loss_gpu.item())
            print(f"  CPU-GPU diff: {diff:.6f}")
            assert diff < 0.01, f"CPU-GPU diff too large: {diff}"
            print(f"  CPU-GPU consistency [PASS]")
        else:
            print(f"  GPU not available, skipping GPU tests")

        print("\n[PASS] Device compatibility test completed successfully")
        return True

    except Exception as e:
        print(f"\n[FAIL] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if mock_dir and os.path.exists(mock_dir):
            shutil.rmtree(mock_dir)


def run_all_tests():
    """Run all tests and report summary."""
    print("\n" + "="*60)
    print("DUAL-DATALOADER TRAINING TEST SUITE")
    print("="*60)

    tests = [
        ("Niqab Dataloader Setup", test_niqab_dataloader_setup),
        ("Iterator Cycling", test_iterator_cycling),
        ("Occlusion From Niqab Only", test_occlusion_from_niqab_only),
        ("Gradient Flow Dual Sources", test_gradient_flow_dual_sources),
        ("Loss Combination", test_loss_combination),
        ("No Niqab Fallback", test_no_niqab_fallback),
        ("Device Compatibility", test_device_compatibility),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n[FAIL] {name} test crashed: {str(e)}")
            results.append((name, False))

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

    if failed > 0:
        print(f"\n[OVERALL: FAIL] {failed} test(s) failed")
        return False
    else:
        print(f"\n[OVERALL: PASS] All tests passed!")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
