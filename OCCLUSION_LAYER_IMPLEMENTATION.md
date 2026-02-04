# Occlusion Layer Implementation Documentation

This document tracks the implementation progress of the occlusion-aware layer for OAML (Occluded Face Recognition).

---

## Overview

The occlusion layer is a spatial occlusion prediction module that:
1. Predicts which regions of a face are visible vs. occluded from CNN feature maps
2. Uses these predictions to weight QAConv matching scores
3. Is trained with ground truth masks from niqab dataset

---

## Implementation Progress

| Component | Status | Test Status |
|-----------|--------|-------------|
| 1. OcclusionHead class | **DONE** | **PASSED** (8/8) |
| 2. Backbone integration | **DONE** | **PASSED** (9/9) |
| 3. QAConv occlusion weighting | **DONE** | **PASSED** (9/9) |
| 4. NiqabMaskDataset | **DONE** | **PASSED** (8/8) |
| 5. Training loop updates | **DONE** | **PASSED** (7/7) |
| 6. Config parameters | **DONE** | Pending |
| 7. Dual-dataloader training | **DONE** | Pending |

---

## Component 1: OcclusionHead Class

### Status: **IMPLEMENTED**

### Description
Lightweight CNN head that predicts spatial occlusion maps from backbone feature maps.

### Architecture
```
Input:  Feature Maps [B, 512, 7, 7]
    |
Conv2d(512 -> 128, kernel=3, padding=1, bias=False)
    |
BatchNorm2d(128)
    |
ReLU(inplace=True)
    |
Conv2d(128 -> 1, kernel=1, bias=True)
    |
Sigmoid
    |
Output: Occlusion Map [B, 1, 7, 7] (values in [0, 1])
```

### Key Design Choices
- **Single channel output**: One confidence value per spatial location
- **Sigmoid activation**: Ensures output in [0, 1] range (1=visible, 0=occluded)
- **Lightweight**: Only 2 conv layers (~590K parameters) to avoid overfitting
- **Kaiming initialization**: For proper gradient flow
- **No bias in conv1**: Following ResNet convention for conv+BN pairs
- **Bias in conv2**: Final prediction layer has bias for flexibility

### File Modified
- `net.py`: Added `OcclusionHead` class at lines 143-204

### Implementation Details

**Location in net.py**: Lines 143-204 (after SEModule, before BasicBlockIR)

**Class Structure**:
```python
class OcclusionHead(Module):
    def __init__(self, in_channels=512, hidden_channels=128):
        # conv1: 3x3 conv for feature extraction (no bias)
        # bn1: batch normalization
        # relu: ReLU activation
        # conv2: 1x1 conv for prediction (with bias)
        # sigmoid: ensure output in [0, 1]

    def _initialize_weights(self):
        # Kaiming normal for conv layers
        # Constant 1/0 for BN weight/bias

    def forward(self, x):
        # Returns occlusion map [B, 1, H, W]
```

**Parameter Count**:
- conv1: 512 × 128 × 3 × 3 = 589,824
- bn1: 128 × 2 = 256
- conv2: 128 × 1 × 1 × 1 + 1 = 129
- **Total: ~590,209 parameters**

### Test Script
**File**: `tests/test_occlusion_head.py`

**Tests Included**:
1. **Instantiation**: Default and custom parameter initialization
2. **Output Shape**: Various batch sizes and spatial dimensions
3. **Output Range**: Verifies sigmoid constraint [0, 1]
4. **Gradient Flow**: Backprop through all parameters
5. **Parameter Count**: Verifies lightweight design
6. **Determinism**: Consistent outputs for same input
7. **Device Transfer**: CPU and GPU compatibility
8. **Batch Independence**: Samples processed independently in eval mode

**How to Run**:
```bash
cd /path/to/OAML
python tests/test_occlusion_head.py
```

### Test Results
**Date**: Component 1 tests completed

```
============================================================
TEST SUMMARY
============================================================
  Instantiation: [PASS]
  Output Shape: [PASS]
  Output Range: [PASS]
  Gradient Flow: [PASS]
  Parameter Count: [PASS]  (590,209 params)
  Determinism: [PASS]
  Device Transfer: [PASS]  (CPU-GPU diff ~1e-3, within tolerance)
  Batch Independence: [PASS]

Total: 8/8 tests passed
[OVERALL: PASS]
```

---

## Component 2: Backbone Integration

### Status: **IMPLEMENTED**

### Description
Integrate OcclusionHead into the Backbone class to produce occlusion maps during forward pass.

### Changes Made
1. Added `self.output_channel` to store channel count for reference
2. Added `self.occlusion_head` in `Backbone.__init__()` for both 112x112 and 224x224 input sizes
3. Modified `Backbone.forward()` to accept `return_occlusion` parameter
4. When `return_occlusion=True`, returns `(output, norm, occlusion_map, feature_maps)`
5. Backward compatible: default behavior unchanged

### File Modified
- `net.py`: Lines 370-490

### Implementation Details

**Changes to `__init__()` (lines 370-390)**:
```python
# Store output_channel for reference
self.output_channel = output_channel

# For 112x112 input:
self.occlusion_head = OcclusionHead(in_channels=output_channel, hidden_channels=128)

# For 224x224 input:
self.occlusion_head = OcclusionHead(in_channels=output_channel, hidden_channels=128)
```

**Changes to `forward()` (lines 405-490)**:
```python
def forward(self, x, return_occlusion=False):
    # ... existing backbone processing ...

    feature_maps = x  # Store for both branches

    # Compute occlusion map (always computed, but only returned if requested)
    occlusion_map = self.occlusion_head(feature_maps)

    # Get embedding
    embedding = self.output_layer(feature_maps)
    # ... normalization ...

    if return_occlusion:
        return output, norm, occlusion_map, feature_maps

    # ... existing inference logic ...
    return output, norm
```

**Return Values**:
- `return_occlusion=False` (default): `(output, norm)` - backward compatible
- `return_occlusion=True`: `(output, norm, occlusion_map, feature_maps)`

**Key Design Decisions**:
- Occlusion map is always computed (minimal overhead due to lightweight head)
- Feature maps are shared between embedding and occlusion branches (multi-task learning)
- Backward compatible: existing code works without modification

### Test Script
**File**: `tests/test_backbone_occlusion.py`

**Tests Included**:
1. **Backbone Has OcclusionHead**: Verifies occlusion_head attribute exists
2. **Forward Without Occlusion**: Backward compatibility check
3. **Forward With Occlusion**: Shape verification for all returned tensors
4. **Occlusion Spatial Resolution**: 112->7x7 and 224->14x14
5. **Gradient Flow Both Branches**: Gradients through shared backbone
6. **Different Architectures**: IR-18, IR-34, IR-50 support
7. **Feature Sharing**: Same features used by both branches
8. **Batch Sizes**: Various batch sizes work correctly
9. **Device Transfer**: CPU and GPU compatibility

**How to Run**:
```bash
cd /path/to/OAML
python tests/test_backbone_occlusion.py
```

### Test Results
**Date**: Component 2 tests completed

```
============================================================
TEST SUMMARY
============================================================
  Backbone Has OcclusionHead: [PASS]
  Forward Without Occlusion: [PASS]
  Forward With Occlusion: [PASS]
  Occlusion Spatial Resolution: [PASS]
  Gradient Flow Both Branches: [PASS]
  Different Architectures: [PASS]
  Feature Sharing: [PASS]
  Batch Sizes: [PASS]
  Device Transfer: [PASS]

Total: 9/9 tests passed
[OVERALL: PASS]
```

**Notes**:
- Feature map norms vary by architecture (IR-18: ~1500, IR-50: ~8M) - expected for unnormalized features
- CPU-GPU occlusion diff: 3.25e-03 - within tolerance
- All shapes verified correct

---

## Component 3: QAConv Occlusion Weighting

### Status: **IMPLEMENTED**

### Description
Add occlusion-aware weighting to QAConv's spatial correlation computation.

### Changes Made
1. Added `_compute_similarity_batch_with_occlusion()` method
2. Applied occlusion weights BEFORE max pooling
3. Weight formula: `weight[p, g, r, s] = prob_occ[p, r] * gal_occ[g, s]`
4. Updated `forward()` docstring with new occlusion parameters
5. Updated `forward()` to dispatch to occlusion-aware method when maps provided
6. Updated `match()` method to accept occlusion maps

### File Modified
- `qaconv.py`: Lines 166-255 (new method), Lines 257-279 (docstring), Lines 413-417 (dispatch), Lines 421-440 (match method)

### Implementation Details

**New Method: `_compute_similarity_batch_with_occlusion()`** (lines 166-255):
```python
def _compute_similarity_batch_with_occlusion(self, prob_fea, gal_fea, prob_occ=None, gal_occ=None):
    # Falls back to standard if no occlusion maps provided

    # 1. Compute spatial correlation: score[p, g, r, s]
    score = torch.einsum('p c s, g c r -> p g r s', prob_chunk, gal_chunk)

    # 2. Create occlusion weight matrix
    # prob_occ_flat: [B_p, hw] -> [B_p, 1, hw, 1]
    # gal_occ_flat:  [B_g, hw] -> [1, B_g, 1, hw]
    occlusion_weights = prob_occ_expanded * gal_occ_expanded  # [B_p, B_g, hw, hw]

    # 3. Apply weights BEFORE max pooling
    score = score * occlusion_weights

    # 4. Max pool and process through BN/FC (unchanged)
```

**Updated forward() signature**:
```python
def forward(self, prob_fea, gal_fea=None, labels=None, prob_occ=None, gal_occ=None):
    # ...
    if prob_occ is not None and gal_occ is not None:
        return self._compute_similarity_batch_with_occlusion(...)
    else:
        return self._compute_similarity_batch(...)
```

**Updated match() signature**:
```python
def match(self, probe_features, gallery_features, probe_occ=None, gallery_occ=None):
    # Passes occlusion maps to forward()
```

**Key Design Decisions**:
- Weighting before max pooling ensures occluded regions don't dominate max selection
- Multiplicative weighting: visible × visible = 1, occluded × anything = 0
- Backward compatible: no occlusion maps = standard computation
- Gradients flow through occlusion maps for end-to-end training

### Test Script
**File**: `tests/test_qaconv_occlusion.py`

**Tests Included**:
1. **Occlusion Method Exists**: Verifies method and parameters exist
2. **Occlusion Output Shape**: Various probe/gallery batch size combinations
3. **All-Ones Occlusion**: Should match standard computation
4. **Zeros Reduce Scores**: Full occlusion significantly changes scores
5. **Partial Occlusion**: Half-occluded images show intermediate effect
6. **Gradient Flow**: Gradients through features AND occlusion maps
7. **Forward Dispatch**: Correct method called based on parameters
8. **Match Method**: match() works with occlusion maps
9. **Device Transfer**: CPU and GPU compatibility

**How to Run**:
```bash
cd /path/to/OAML
python tests/test_qaconv_occlusion.py
```

### Test Results
**Date**: Component 3 tests completed

```
============================================================
TEST SUMMARY
============================================================
  Occlusion Method Exists: [PASS]
  Occlusion Output Shape: [PASS]
  All-Ones Occlusion: [PASS]  (diff = 0.00e+00)
  Zeros Reduce Scores: [PASS]  (0.2351 → 0.0361, 85% reduction)
  Partial Occlusion: [PASS]  (0.1860 → 0.0896, ~52% reduction)
  Gradient Flow: [PASS]
  Forward Dispatch: [PASS]
  Match Method: [PASS]
  Device Transfer: [PASS]  (CPU-GPU diff = 2.98e-08)

Total: 9/9 tests passed
[OVERALL: PASS]
```

---

## Component 4: NiqabMaskDataset

### Status: **IMPLEMENTED**

### Description
Dataset class for loading niqab images with ground truth occlusion masks.

### Actual Directory Structure (HPC)
```
/home/maass/code/niqab/train/
    kept_faces/
        {name}.jpg
        ...
    masks/
        {name}_mask.png  (grayscale: 255=visible, 0=occluded)
        ...
```

### Files Created
- `dataset/__init__.py`: Module exports
- `dataset/niqab_mask_dataset.py`: Dataset classes

### Implementation Details

**Classes Implemented**:

1. **`NiqabMaskDataset`**: Main dataset class
   ```python
   dataset = NiqabMaskDataset(
       root_dir='/home/maass/code/niqab/train',
       image_transform=transform,      # Optional, default resize to 112x112
       mask_target_size=7,             # Resize masks to 7x7
       image_subdir='kept_faces',      # Subdirectory for images
       mask_subdir='masks',            # Subdirectory for masks
       mask_suffix='_mask'             # Suffix for mask files
   )
   ```

2. **`NiqabMaskDataModule`**: DataLoader wrapper with train/val split
   ```python
   data_module = NiqabMaskDataModule(
       root_dir='/home/maass/code/niqab/train',
       batch_size=32,
       val_split=0.1,
       mask_target_size=7
   )
   data_module.setup()
   train_loader = data_module.train_dataloader()
   ```

**Return Format**:
```python
image, mask, index = dataset[i]
# image: [3, 112, 112] - normalized tensor
# mask: [1, 7, 7] - values in [0, 1], 1=visible, 0=occluded
# index: int - sample index
```

**Key Features**:
- Auto-discovers image-mask pairs by naming convention
- Resizes masks to match feature map resolution (7x7 for 112x112 input)
- Normalizes mask values to [0, 1]
- Configurable transforms
- Train/val split support via DataModule

### Test Script
**File**: `tests/test_niqab_dataset.py`

**Tests Included**:
1. **Dataset Creation**: Finds image-mask pairs
2. **Sample Loading**: Correct shapes and types
3. **Multiple Samples**: Uniqueness verification
4. **Custom Transforms**: Different sizes work
5. **DataLoader**: Batching works correctly
6. **Data Module**: Train/val split works
7. **Mask Values Meaningful**: Masks have variation
8. **GPU Transfer**: Data can move to GPU

**How to Run**:
```bash
cd /path/to/OAML
python tests/test_niqab_dataset.py
```

**Note**: Test auto-detects HPC path `/home/maass/code/niqab/train` or creates mock data.

### Test Results
**Date**: Component 4 tests completed

```
============================================================
TEST SUMMARY
============================================================
  Dataset Creation: [PASS]  (205 image-mask pairs)
  Sample Loading: [PASS]  (image [3,112,112], mask [1,7,7])
  Multiple Samples: [PASS]  (diff=0.4911)
  Custom Transforms: [PASS]
  DataLoader: [PASS]
  Data Module: [PASS]  (164 train, 41 val)
  Mask Values Meaningful: [PASS]  (2-4 unique values)
  GPU Transfer: [PASS]

Total: 8/8 tests passed
[OVERALL: PASS]
```

---

## Component 5: Training Loop Updates

### Status: **IMPLEMENTED**

### Description
Update training loop to compute occlusion loss and integrate with existing loss structure.

### Changes Made
1. Added `occlusion_loss_weight` parameter to `Trainer.__init__()` (default: 0.1)
2. Modified `training_step()` to handle batch format with/without GT masks
3. Extract feature maps and compute occlusion maps using `model.occlusion_head()`
4. Compute MSE loss between predicted and GT occlusion maps
5. Add occlusion loss to total loss with configurable weight
6. Log occlusion loss to PyTorch Lightning and wandb

### File Modified
- `train_val.py`: Modified `Trainer` class

### Implementation Details

**Changes to `__init__()` (around line 55)**:
```python
self.occlusion_loss_weight = occlusion_loss_weight  # default 0.1
```

**Changes to `training_step()` (around lines 120-200)**:
```python
def training_step(self, batch, batch_idx):
    # Handle batch format: (images, labels) or (images, labels, gt_masks)
    if len(batch) == 3:
        images, labels, gt_masks = batch
        has_gt_masks = True
    else:
        images, labels = batch
        gt_masks = None
        has_gt_masks = False

    # Extract features manually to get feature maps for occlusion
    x = self.model.input_layer(images)
    for layer in self.model.body:
        x = layer(x)

    feature_maps = x  # [B, 512, 7, 7]

    # Compute occlusion maps
    occlusion_maps = self.model.occlusion_head(feature_maps)  # [B, 1, 7, 7]

    # Compute occlusion loss if GT masks provided
    if has_gt_masks:
        # Resize GT masks if needed
        if gt_masks.shape[-2:] != occlusion_maps.shape[-2:]:
            gt_masks = F.interpolate(gt_masks, size=occlusion_maps.shape[-2:],
                                     mode='bilinear', align_corners=False)
        occlusion_loss = F.mse_loss(occlusion_maps, gt_masks)
    else:
        occlusion_loss = torch.tensor(0.0, device=images.device)

    # ... existing loss computation for adaface_loss, qaconv_loss, etc. ...

    # Total loss with occlusion component
    total_loss = (
        adaface_weight * adaface_loss +
        qaconv_weight * qaconv_loss +
        self.occlusion_loss_weight * occlusion_loss
    )

    # Log occlusion loss
    self.log('occlusion_loss', occlusion_loss, prog_bar=True)
    if self.use_wandb:
        wandb.log({'occlusion_loss': occlusion_loss.item()})
```

**Key Design Decisions**:
- Batch format detection: `len(batch) == 3` indicates GT masks present
- Feature map extraction done before embedding to enable occlusion head computation
- GT masks resized to match occlusion map resolution (7x7) if needed
- Occlusion loss is 0 when no GT masks provided (backward compatible)
- Loss weights: adaface=0.1, qaconv=0.9, occlusion=0.1 (configurable)

### Test Script
**File**: `tests/test_training_occlusion.py`

**Tests Included**:
1. **Occlusion Head Integration**: Model has occlusion_head attribute
2. **Occlusion Loss Computation**: MSE loss computed correctly
3. **Gradient Flow Through Occlusion Loss**: Gradients propagate to occlusion head
4. **Mask Resizing**: GT masks resized from various sizes to 7x7
5. **Combined Loss**: Total loss includes occlusion component with correct weights
6. **Batch Format Handling**: Both (images, labels) and (images, labels, masks) formats
7. **Device Compatibility**: Works on CPU and GPU

**How to Run**:
```bash
cd /path/to/OAML
python tests/test_training_occlusion.py
```

### Test Results
**Date**: Component 5 tests completed

```
============================================================
TEST SUMMARY
============================================================
  Occlusion Head Integration: [PASS]
  Occlusion Loss Computation: [PASS]  (loss=0.316, varies with GT)
  Gradient Flow Through Occlusion Loss: [PASS]  (conv1 grad=1.07, input grad=0.07)
  Mask Resizing: [PASS]  (7x7, 14x14, 28x28, 112x112 all work)
  Combined Loss: [PASS]  (0.1*1.0 + 0.9*0.5 + 0.1*0.2 = 0.57)
  Batch Format Handling: [PASS]
  Device Compatibility: [PASS]  (CPU-GPU diff = 0.0006)

Total: 7/7 tests passed
[OVERALL: PASS]
```

---

## Component 6: Config Parameters

### Status: **IMPLEMENTED**

### Description
Add configuration arguments for occlusion layer.

### New Arguments
- `--occlusion_loss_weight`: Weight for occlusion prediction MSE loss (default: 0.1)
- `--niqab_data_path`: Path to niqab dataset with GT masks
- `--use_occlusion_weighting`: Enable occlusion-aware weighting in QAConv matching

### File Modified
- `config.py`: Added new arguments in `add_task_arguments()` (lines 79-85)

### Implementation Details

**Changes to `add_task_arguments()` (lines 79-85)**:
```python
# Occlusion layer parameters
parser.add_argument('--occlusion_loss_weight', default=0.1, type=float,
                    help='Weight for occlusion prediction MSE loss (default: 0.1)')
parser.add_argument('--niqab_data_path', type=str, default='',
                    help='Path to niqab dataset with GT masks (e.g., /home/maass/code/niqab/train)')
parser.add_argument('--use_occlusion_weighting', action='store_true',
                    help='Enable occlusion-aware weighting in QAConv matching')
```

**Usage Examples**:
```bash
# Train with occlusion loss using niqab data
python main.py --niqab_data_path /home/maass/code/niqab/train --occlusion_loss_weight 0.1

# Enable occlusion-weighted QAConv matching
python main.py --use_occlusion_weighting

# Full occlusion training
python main.py \
    --niqab_data_path /home/maass/code/niqab/train \
    --occlusion_loss_weight 0.15 \
    --use_occlusion_weighting
```

**Integration with Trainer**:
The Trainer class in `train_val.py` accesses these via `self.hparams`:
```python
self.occlusion_loss_weight = getattr(self.hparams, 'occlusion_loss_weight', 0.1)
```

### Test Script
**File**: `tests/test_config_occlusion.py`

**Tests Included**:
1. **Default Values**: All occlusion args have correct defaults
2. **Custom Values**: Args accept custom values correctly
3. **Weight Range**: Various weight values (0.0, 0.05, 0.1, 0.2, 0.5, 1.0) work
4. **Full Config Integration**: Args work with full config.py parser
5. **Trainer Config Usage**: Trainer can access occlusion config via getattr
6. **Help Text**: Help descriptions are properly defined

**How to Run**:
```bash
cd /path/to/OAML
python tests/test_config_occlusion.py
```

### Test Results
*(To be filled after user runs tests)*

---

## Component 7: Dual-Dataloader Training

### Status: **IMPLEMENTED**

### Description
Separate training data streams for recognition (CASIA-WebFace) and occlusion (Niqab).

### Training Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                     Training Step                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CASIA-WebFace Batch          Niqab Batch (cycling)             │
│  (images, labels)             (images, masks)                   │
│         │                            │                          │
│         ▼                            ▼                          │
│    ┌─────────┐                 ┌─────────┐                      │
│    │Backbone │                 │Backbone │                      │
│    └────┬────┘                 └────┬────┘                      │
│         │                            │                          │
│    ┌────┴────┐                 ┌────┴────┐                      │
│    │         │                 │Occlusion│                      │
│    │ AdaFace │                 │  Head   │                      │
│    │   +     │                 └────┬────┘                      │
│    │ QAConv  │                      │                           │
│    └────┬────┘                      ▼                           │
│         │                    MSE Loss (pred vs GT)              │
│         ▼                            │                          │
│  Recognition Loss                    │                          │
│         │                            │                          │
│         └──────────┬─────────────────┘                          │
│                    ▼                                             │
│              Total Loss                                          │
│  = 0.1*AdaFace + 0.9*QAConv + 0.1*Occlusion                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Changes Made
1. Added `_setup_niqab_dataloader()` method to initialize niqab dataloader
2. Added `_get_next_niqab_batch()` method with cycling iterator
3. Modified `training_step()` to:
   - Process main batch (CASIA) for recognition losses only
   - Get niqab batch separately for occlusion loss only
   - Combine both losses

### File Modified
- `train_val.py`: Added niqab dataloader methods, modified training_step

### Implementation Details

**New method `_setup_niqab_dataloader()` (lines 115-141)**:
```python
def _setup_niqab_dataloader(self, niqab_path):
    # Create NiqabMaskDataset with 7x7 mask target
    self.niqab_dataset = NiqabMaskDataset(
        root_dir=niqab_path,
        mask_target_size=7,
        image_subdir='kept_faces',
        mask_subdir='masks',
        mask_suffix='_mask'
    )
    # Create cycling dataloader
    self.niqab_dataloader = DataLoader(...)
    self.niqab_iter = iter(self.niqab_dataloader)
```

**New method `_get_next_niqab_batch()` (lines 143-157)**:
```python
def _get_next_niqab_batch(self):
    try:
        batch = next(self.niqab_iter)
    except StopIteration:
        # Cycle when exhausted
        self.niqab_iter = iter(self.niqab_dataloader)
        batch = next(self.niqab_iter)
    images, masks, _ = batch
    return images, masks
```

**Modified `training_step()` (lines 274-310)**:
```python
# ========== OCCLUSION LOSS FROM NIQAB BATCH ==========
if self.niqab_dataloader is not None:
    niqab_images, niqab_masks = self._get_next_niqab_batch()

    # Forward niqab images through backbone
    niqab_x = self.model.input_layer(niqab_images)
    for layer in self.model.body:
        niqab_x = layer(niqab_x)

    # Compute occlusion maps and loss
    niqab_occlusion_maps = self.model.occlusion_head(niqab_x)
    occlusion_loss = F.mse_loss(niqab_occlusion_maps, niqab_masks)
```

**Key Design Decisions**:
- Niqab data cycles independently (small dataset, ~180 images)
- Main batch (CASIA) has no GT masks → no occlusion loss computed
- Niqab batch has GT masks but no identity labels → no recognition loss computed
- Both losses combined with configurable weights
- Backward compatible: works without niqab data (occlusion loss = 0)

### Test Script
**File**: `tests/test_dual_dataloader.py`

**Tests Included**:
1. **Niqab Dataloader Setup**: Dataset and dataloader initialize correctly
2. **Iterator Cycling**: Iterator restarts when exhausted
3. **Occlusion From Niqab Only**: Occlusion loss computed from niqab batch
4. **Gradient Flow Dual Sources**: Gradients flow through both data streams
5. **Loss Combination**: Total loss combines recognition + occlusion correctly
6. **No Niqab Fallback**: Training works when niqab data unavailable
7. **Device Compatibility**: Works on CPU and GPU

**How to Run**:
```bash
cd /path/to/OAML
python tests/test_dual_dataloader.py
```

### Test Results
*(To be filled after user runs tests)*

---

## Testing Strategy

Each component has a dedicated test script that verifies:
1. **Shape correctness**: Input/output tensor shapes match specification
2. **Value ranges**: Outputs are within expected ranges
3. **Gradient flow**: Gradients propagate correctly during training
4. **Integration**: Component works with existing modules

Test scripts are designed to run on HPC without GUI dependencies.

---

## Notes

### PyTorch Lightning 2.0 Compatibility Fixes

The following changes were made to `train_val.py` for PL 2.0 compatibility:

1. **Hook renames**:
   - `training_epoch_end` → `on_train_epoch_end`
   - `validation_epoch_end` → `on_validation_epoch_end`
   - `test_epoch_end` → `on_test_epoch_end`

2. **Output collection**: PL 2.0 removed `outputs` parameter from epoch end hooks.
   - Added `self.validation_step_outputs = []` and `self.test_step_outputs = []`
   - Modified step methods to append outputs to these lists
   - Modified epoch end hooks to use stored outputs and clear after processing

3. **main.py changes**:
   - `gpus` → `devices` parameter
   - `precision=16` → `precision='16-mixed'`
   - `resume_from_checkpoint` moved from Trainer() to trainer.fit(ckpt_path=...)

4. **Scheduler compatibility in `get_current_lr()`**:
   - Changed `isinstance(scheduler, lr_scheduler._LRScheduler)` to `hasattr(scheduler, 'get_last_lr')`
   - `lr_scheduler._LRScheduler` was deprecated/renamed in newer PyTorch versions
   - Now uses `hasattr` checks to detect scheduler type:
     - PyTorch schedulers (MultiStepLR, etc.) → `get_last_lr()`
     - timm schedulers → `get_epoch_values()`
     - Fallback → read from `optimizer.param_groups[0]['lr']`

5. **DDP unused parameters** (in `main.py`):
   - OcclusionHead parameters aren't used every training step (only with niqab data)
   - DDP requires all parameters to participate in gradient computation by default
   - Changed `strategy='ddp'` → `strategy='ddp_find_unused_parameters_true'`
   - This allows DDP to handle parameters that don't contribute to every loss computation

### QAConv make_kernel Method

Added `make_kernel(feature)` method to `qaconv.py` for compatibility with `PairwiseMatchingLoss`:
- Stores features as kernel (gallery) for next forward call
- Used by PairwiseMatchingLoss for self-matching (probe == gallery)
- Cleared after use in forward() to avoid stale data

### DDP Validation Gathering Fix

**Issue**: The `gather_outputs()` function in `train_val.py` was modified to skip `utils.all_gather()` to avoid OOM during validation. This caused validation metrics to be computed on only local GPU data (~50% with 2 GPUs).

**Problem**: Validation datasets (LFW, AgeDB, etc.) use sequential pair structure:
- Pairs are `(image0, image1), (image2, image3), ...`
- AdaFace/QAConv compute: `embeddings1 = embeddings[0::2]`, `embeddings2 = embeddings[1::2]`
- Without gathering, each GPU has random subsets → **pair structure is broken**
- This caused AdaFace validation accuracy to appear much lower than actual

**Root Cause**: The `utils.all_gather()` was already updated to use CPU-based gathering (gloo backend) to avoid GPU OOM, but `gather_outputs()` was changed to skip it entirely instead of using the improved version.

**Fix Applied** (lines 1087-1100 in `train_val.py`):
```python
# BEFORE (broken):
def gather_outputs(self, outputs):
    if self.hparams.distributed_backend == 'ddp':
        # Skip gathering - each GPU uses local data only
        outputs_list = outputs  # <-- WRONG: breaks pair structure!

# AFTER (fixed):
def gather_outputs(self, outputs):
    if self.hparams.distributed_backend == 'ddp':
        torch.cuda.empty_cache()
        # Gather from ALL GPUs using CPU-based gloo backend
        outputs_list = []
        _outputs_list = utils.all_gather(outputs)
        for _outputs in _outputs_list:
            outputs_list.extend(_outputs)
        torch.cuda.empty_cache()
```

**Why AdaFace was more affected than QAConv**:
- Both metrics rely on the same pair structure
- Different sensitivity to pair distribution may cause different degradation
- QAConv's local feature matching may be more robust to partial data than AdaFace's global embeddings

**Key Learning**: When using DDP, validation data must be gathered across all GPUs to preserve the pair structure required by face recognition evaluation metrics.
