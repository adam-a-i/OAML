# Occlusion Layer Specification

## High-Level Purpose

The occlusion layer is a **spatial occlusion prediction module** that learns to identify which regions of a face are visible vs. occluded directly from CNN feature maps. Its primary purpose is to enable **occlusion-aware face matching** by providing spatial confidence maps that guide the matching process to focus on visible facial regions.

### Core Motivation

Traditional face recognition systems struggle when faces are partially occluded (masks, sunglasses, hands, etc.) because:
- Occluded regions provide unreliable or misleading features
- Matching algorithms treat all regions equally, diluting the signal from visible regions
- The model has no explicit understanding of which features are trustworthy

The occlusion layer solves this by:
1. **Predicting occlusion** at the spatial feature level (not just globally)
2. **Weighting similarity scores** during matching based on predicted visibility
3. **Learning from real-world occlusions** (niqab) during training with ground truth masks

---

## Architecture Overview

### Component: OcclusionHead

A lightweight CNN head that operates on the backbone's feature maps.

```
Input:  Feature Maps [B, C, H, W]  (e.g., [B, 512, 7, 7] for 112x112 input)
    ↓
Conv2d(C → hidden_channels, kernel=3, padding=1)
    ↓
BatchNorm2d(hidden_channels)
    ↓
ReLU
    ↓
Conv2d(hidden_channels → 1, kernel=1)
    ↓
Sigmoid
    ↓
Output: Occlusion Map [B, 1, 7, 7]  (values in [0, 1])
```

**Key Design Choices:**
- **Spatial preservation**: Maintains spatial dimensions (H, W) to provide per-location predictions
- **Single channel output**: One confidence value per spatial location
- **Sigmoid activation**: Ensures output is in [0, 1] range (1=visible, 0=occluded)
- **Lightweight**: Only 2 conv layers to avoid overfitting and maintain efficiency

### Integration with Backbone

The occlusion head branches off from the backbone's feature maps **at the same resolution as QAConv**:

```
Backbone CNN
    ↓
Feature Maps [B, 512, 7, 7]  (for 112x112 input)
    ├──→ Output Layer → Embedding [B, 512] (for AdaFace)
    └──→ OcclusionHead → Occlusion Map [B, 1, 7, 7]
```

**Spatial Resolution Strategy:**
- **Occlusion maps**: 7x7 (matching QAConv feature map resolution)
- **QAConv feature maps**: 7x7
- **Direct compatibility**: No downsampling needed - occlusion maps match QAConv resolution exactly

**Why 7x7?**
- **No information loss**: Avoids downsampling artifacts and information loss
- **Simpler implementation**: Direct compatibility, no interpolation step
- **Sufficient granularity**: For 112x112 input, each 7x7 location represents ~16x16 pixels, which is adequate for niqab occlusion patterns
- **Consistent training/inference**: Model learns and operates at the same resolution

This allows the occlusion head to:
- Share the same feature representation as the recognition task
- Benefit from multi-task learning
- Operate at the same spatial resolution as QAConv matching (no downsampling needed)
- Maintain direct compatibility without any resolution mismatch

---

## Training Methodology

### Supervised Learning with Real-World Occlusions

The occlusion head is trained using **ground truth occlusion masks** from a dataset of niqabi women. Each training sample consists of:
- **Image**: Face image with niqab occlusion
- **Ground Truth Mask**: Binary or continuous mask indicating visible (1.0) vs occluded (0.0) regions

#### 1. Dataset Structure

**Niqab Dataset:**
- Real-world images of women wearing niqab
- Manually annotated or automatically generated ground truth masks
- Masks indicate which facial regions are visible vs occluded by the niqab
- No synthetic data augmentation - only real niqab occlusions

**Key Properties:**
- Ground truth masks are provided at image resolution (e.g., 112x112 or 224x224)
- Masks are downsampled to match occlusion head output resolution (7x7 for 112x112 input, 14x14 for 224x224 input) for training
- Binary or continuous values: ground truth can be binary (0/1) or continuous (0.0-1.0)
- All training samples have corresponding ground truth masks (no missing masks)

#### 2. Loss Function

**Mean Squared Error (MSE) Loss:**

```python
occlusion_loss = F.mse_loss(predicted_occlusion_map, ground_truth_mask)
```

**Why MSE?**
- Smooth gradient flow (unlike binary cross-entropy for continuous predictions)
- Handles continuous confidence values naturally
- Simple and stable

**Shape Alignment:**
- Predicted: `[B, 1, 7, 7]` (from OcclusionHead for 112x112 input)
- Ground Truth: `[B, 1, H, W]` (from dataset, e.g., 112x112 or 224x224)
- Solution: Downsample GT to `[B, 1, 7, 7]` using bilinear interpolation (for 112x112 input)

#### 3. Multi-Task Learning

The occlusion head is trained jointly with:
- **AdaFace loss**: Face recognition/classification
- **QAConv loss**: Pairwise matching
- **Occlusion loss**: Occlusion prediction

**Loss Weighting:**
```python
total_loss = (
    adaface_loss_weight * adaface_loss +
    qaconv_loss_weight * qaconv_loss +
    occlusion_loss_weight * occlusion_loss
)
```

Typical weights: `adaface_loss_weight=1.0`, `qaconv_loss_weight=1.0`, `occlusion_loss_weight=0.1-1.0`

---

## Integration with QAConv Matching

### Occlusion-Aware Similarity Weighting

The occlusion maps are used to **weight the correlation scores** in QAConv's spatial matching process.

#### QAConv Matching Process

1. **Spatial Correlation**: Compute correlation between all spatial locations:
   ```
   correlation[i, j] = similarity(query_location_i, gallery_location_j)
   ```
   Shape: `[B_query, B_gallery, H*W, H*W]`

2. **Occlusion Weighting**: Apply occlusion maps to weight correlations:
   ```python
   weight[i, j] = query_occ[i] * gallery_occ[j]
   weighted_correlation = correlation * weight
   ```
   - If either location is occluded (low confidence), the correlation gets low weight
   - Only visible-to-visible matches contribute strongly
   - Occlusion maps are already at 7x7 resolution, matching QAConv exactly (no downsampling needed)

3. **Pooling**: Max pooling over spatial dimensions to get final similarity score

#### Implementation Details

**Method: "scaling" (recommended)**
```python
# Occlusion maps are already at 7x7 resolution (matching QAConv)
# Flatten occlusion maps: [B, 1, 7, 7] -> [B, 49]
query_flat = query_occ.view(B_query, -1)
gallery_flat = gallery_occ.view(B_gallery, -1)

# Create weight matrix via broadcasting
q_weight = query_flat.unsqueeze(1).unsqueeze(3)  # [B_query, 1, 49, 1]
g_weight = gallery_flat.unsqueeze(0).unsqueeze(2)  # [1, B_gallery, 1, 49]
weight = q_weight * g_weight  # [B_query, B_gallery, 49, 49]

# Apply weighting
weighted_similarity = similarity * weight
```

**Key Insight:**
- Each query location is weighted by its visibility
- Each gallery location is weighted by its visibility
- The product ensures both must be visible for high weight
- This naturally downweights occluded regions in matching

---

## Expected Behavior

### During Training

1. **Occlusion Prediction Learning**:
   - Model sees real niqab-occluded images with ground truth masks
   - Learns to predict occlusion from feature patterns
   - Loss decreases as predictions match ground truth masks
   - 7x7 resolution provides adequate spatial detail for niqab occlusion patterns

2. **Feature Learning**:
   - Backbone learns features that are informative for both recognition AND occlusion detection
   - Multi-task learning improves generalization

3. **Matching Improvement**:
   - QAConv learns to rely more on visible regions
   - Matching becomes more robust to occlusions

### During Inference

1. **Occlusion Detection**:
   - Model predicts occlusion map for any input face
   - Works even without explicit occlusion (predicts all visible = 1.0)

2. **Adaptive Matching**:
   - If face is partially occluded, matching focuses on visible regions
   - If face is fully visible, all regions contribute equally

---

## Implementation Checklist

### Core Components

- [ ] **OcclusionHead class**
  - Input: `[B, in_channels, H, W]` feature maps
  - Architecture: Conv → BN → ReLU → Conv → Sigmoid
  - Output: `[B, 1, H, W]` occlusion confidence map
  - Weight initialization: Kaiming normal for conv layers

- [ ] **Integration in Backbone**
  - Add occlusion head as a module
  - Forward pass returns: `(embedding, norm, occlusion_map)`
  - Feature maps shared between embedding and occlusion head

- [ ] **Dataset Loading**
  - Niqab dataset that returns `(image, mask)` tuples
  - Ground truth masks at image resolution
  - Mask downsampling to match occlusion head output resolution (14x14)

- [ ] **Loss Computation**
  - MSE loss between predicted and ground truth masks
  - Shape alignment (interpolation if needed)
  - NaN/infinity checks

- [ ] **QAConv Integration**
  - `apply_occlusion_weight()` method
  - Occlusion maps are already at 7x7 (matching QAConv) - no downsampling needed
  - Pass occlusion maps directly to `_compute_similarity_batch()`
  - Weight correlation scores before pooling

### Training Considerations

- [ ] **Loss Weighting**
  - Configurable weights for each loss component
  - Start with small occlusion loss weight (0.1) and increase gradually

- [ ] **Ground Truth Availability**
  - All training samples have ground truth masks (niqab dataset)
  - Ensure masks are properly loaded and aligned with images

- [ ] **Numerical Stability**
  - Clamp occlusion predictions to [0, 1]
  - Handle edge cases (all zeros, all ones)

### Validation/Testing

- [ ] **Occlusion Map Visualization**
  - Visualize predicted occlusion maps during training
  - Compare with ground truth masks

- [ ] **Matching Evaluation**
  - Test matching accuracy with and without occlusion weighting
  - Evaluate on occluded face datasets

---

## Key Design Principles

1. **Spatial Awareness**: Predict occlusion per spatial location, not globally (7x7 resolution matching QAConv)
2. **Lightweight**: Minimal parameters to avoid overfitting
3. **Multi-Task Learning**: Share features with recognition task
4. **Real-World Training**: Learn from real niqab occlusions with ground truth masks
5. **Direct Integration**: Occlusion maps directly influence matching scores at the same resolution as QAConv (7x7)
6. **Robustness**: Works even when no occlusion is present (predicts all visible)
7. **No Information Loss**: Same resolution as QAConv avoids downsampling artifacts and maintains consistency

---

## Mathematical Formulation

### Occlusion Prediction

Given feature maps `F ∈ R^(B×C×H×W)`, the occlusion head predicts:

```
O = σ(Conv2(ReLU(BN(Conv1(F)))))
```

where `O ∈ R^(B×1×7×7)` (for 112x112 input) and `σ` is the sigmoid function.

**Note**: The occlusion head operates on feature maps at the same resolution as QAConv (7x7), ensuring direct compatibility without any downsampling.

### Occlusion-Aware Matching

For query features `F_q` and gallery features `F_g` with occlusion maps `O_q` and `O_g`:

1. Compute spatial correlation:
   ```
   S[i,j] = F_q[i] · F_g[j]  for all spatial locations i, j
   ```

2. Apply occlusion weighting (occlusion maps already match QAConv resolution):
   ```
   S_weighted[i,j] = S[i,j] · O_q[i] · O_g[j]
   ```

3. Pool to get final similarity:
   ```
   similarity = pool(S_weighted)
   ```

This ensures that only visible regions contribute significantly to the matching score.

---

## Notes for Clean Re-implementation

1. **Separate Concerns**: Keep occlusion head, data augmentation, and matching integration as separate, well-defined modules

2. **Clear Interfaces**: Define clear input/output shapes and data formats

3. **Error Handling**: Handle missing masks, shape mismatches, and edge cases gracefully

4. **Documentation**: Document the spatial resolution assumptions:
   - Occlusion head output: 7x7 (for 112x112 input), 14x14 (for 224x224 input)
   - QAConv feature maps: 7x7 (for 112x112 input), 14x14 (for 224x224 input)
   - Direct compatibility: Occlusion maps match QAConv resolution exactly (no downsampling needed)

5. **Testing**: Test each component independently before integration

6. **Configuration**: Make occlusion probability, loss weights, and architecture parameters configurable

---

## Plan: Upgrading to 14x14 Feature Maps for QAConv and Occlusion Head

### Motivation

The current 7x7 spatial resolution (49 locations) may not be expressive enough for fine-grained occlusion detection and spatial matching. By tapping into the **penultimate ResNet block** (Block 3), we can use **14x14 feature maps** (196 locations) - a 4x increase in spatial granularity.

### Current Architecture (7x7)

For IR-50 with 112x112 input:
```
Input (112x112)
    ↓
Input Layer → 112x112, 64ch
    ↓
Block 1 (3 units) → 56x56, 64ch
    ↓
Block 2 (4 units) → 28x28, 128ch
    ↓
Block 3 (14 units) → 14x14, 256ch
    ↓
Block 4 (3 units) → 7x7, 512ch  ← Current: QAConv + OcclusionHead tap here
    ↓
Output Layer → 512-dim embedding
```

### Proposed Architecture (14x14)

```
Input (112x112)
    ↓
Input Layer → 112x112, 64ch
    ↓
Block 1 (3 units) → 56x56, 64ch
    ↓
Block 2 (4 units) → 28x28, 128ch
    ↓
Block 3 (14 units) → 14x14, 256ch  ← NEW: QAConv + OcclusionHead tap here
    ↓                               (body indices 7-20)
Block 4 (3 units) → 7x7, 512ch
    ↓
Output Layer → 512-dim embedding (unchanged)
```

### ResNet Body Index Mapping (IR-50)

| Block | Units | Body Indices | Output Channels | Output Resolution |
|-------|-------|--------------|-----------------|-------------------|
| 1 | 3 | 0-2 | 64 | 56x56 |
| 2 | 4 | 3-6 | 128 | 28x28 |
| 3 | 14 | 7-20 | 256 | 14x14 |
| 4 | 3 | 21-23 | 512 | 7x7 |

**Key insight**: After body index 20 (last unit of Block 3), we have 14x14 feature maps with 256 channels.

### Implementation Changes

#### 1. net.py - Backbone Class

**Split the body into two parts:**
```python
# In __init__:
# Split body: body_early = Blocks 1-3, body_late = Block 4
block_counts = {18: [2,2,2,2], 34: [3,4,6,3], 50: [3,4,14,3], 100: [3,13,30,3]}
counts = block_counts[num_layers]
split_idx = sum(counts[:3])  # After Block 3

self.body_early = Sequential(*modules[:split_idx])  # Blocks 1-3 → 14x14
self.body_late = Sequential(*modules[split_idx:])   # Block 4 → 7x7

# For 112x112 input with 14x14 intermediate features:
intermediate_channels = 256  # Block 3 output channels
self.qaconv = QAConv(num_features=intermediate_channels, height=14, width=14)
self.occlusion_head = OcclusionHead(in_channels=intermediate_channels, hidden_channels=128)

# Output layer still uses final 7x7 features (512 channels)
self.output_layer = Sequential(
    BatchNorm2d(512),
    Dropout(0.4),
    Flatten(),
    Linear(512 * 7 * 7, 512),
    BatchNorm1d(512, affine=False)
)
```

**Modify forward pass:**
```python
def forward(self, x, return_occlusion=False):
    x = self.input_layer(x)

    # Process through Blocks 1-3 → 14x14 features
    intermediate_features = self.body_early(x)  # [B, 256, 14, 14]

    # Occlusion prediction from 14x14 features
    occlusion_map = None
    if return_occlusion:
        occlusion_map = self.occlusion_head(intermediate_features)  # [B, 1, 14, 14]

    # Continue through Block 4 → 7x7 features for embedding
    final_features = self.body_late(intermediate_features)  # [B, 512, 7, 7]

    # Output embedding (unchanged)
    embedding = self.output_layer(final_features)
    norm = torch.norm(embedding, 2, 1, True).clamp(min=1e-6)
    output = torch.div(embedding, norm)

    if return_occlusion:
        return output, norm, occlusion_map, intermediate_features  # 14x14 features

    return output, norm
```

#### 2. qaconv.py - QAConv Class

**Dimension changes:**
- `num_features`: 512 → 256
- `height`: 7 → 14
- `width`: 7 → 14
- `fc` input: 7×7×2 = 98 → 14×14×2 = 392

```python
# The FC layer will automatically adjust:
self.fc = nn.Linear(self.height * self.width * 2, 1)  # 392 → 1
```

**Memory consideration:**
- Correlation tensor: [B_p, B_g, H×W, H×W]
- Old: [B_p, B_g, 49, 49] = 2401 elements per pair
- New: [B_p, B_g, 196, 196] = 38,416 elements per pair (16x increase)
- May need to reduce chunk_size for memory efficiency

#### 3. OcclusionHead

**Dimension changes:**
- `in_channels`: 512 → 256
- Output: [B, 1, 7, 7] → [B, 1, 14, 14]

No architectural changes needed - just different input dimensions.

#### 4. Training Code (train_val.py)

**Ground truth mask downsampling:**
```python
# Old: downsample to 7x7
gt_mask_downsampled = F.interpolate(gt_mask, size=(7, 7), mode='bilinear')

# New: downsample to 14x14
gt_mask_downsampled = F.interpolate(gt_mask, size=(14, 14), mode='bilinear')
```

#### 5. Configuration (config.py)

**Add configuration options:**
```python
# Spatial resolution for QAConv/Occlusion (use intermediate features)
USE_INTERMEDIATE_FEATURES = True  # False = 7x7 (Block 4), True = 14x14 (Block 3)
INTERMEDIATE_FEATURE_RESOLUTION = 14  # Spatial resolution when using intermediate features
INTERMEDIATE_FEATURE_CHANNELS = 256   # Channel count for Block 3 output
```

### Benefits

1. **4x spatial granularity**: 196 vs 49 spatial locations for occlusion detection
2. **Finer occlusion boundaries**: Better representation of niqab edge regions
3. **More expressive matching**: QAConv can find more precise spatial correspondences
4. **Richer features for occlusion**: 256 channels still provide strong semantic information

### Trade-offs

1. **Memory increase**: Correlation tensor grows 16x (may need smaller batch sizes)
2. **Computation increase**: More spatial locations to process
3. **Earlier features**: Block 3 features are less semantically refined than Block 4
4. **FC layer size**: Grows from 98 to 392 parameters (minimal impact)

### Backwards Compatibility

- Keep ability to switch between 7x7 and 14x14 via config flag
- Existing checkpoints won't be directly compatible (different layer shapes)
- May need checkpoint conversion script or fresh training

### Testing Plan

1. Verify body split produces correct intermediate feature shapes
2. Test QAConv with 14x14 inputs (check memory usage)
3. Test OcclusionHead with 256-channel, 14x14 inputs
4. Validate ground truth mask downsampling to 14x14
5. End-to-end training run to verify gradients flow correctly
6. Compare occlusion map quality: 7x7 vs 14x14

