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
3. **Learning from synthetic occlusions** during training to generalize to real occlusions

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
Output: Occlusion Map [B, 1, H, W]  (values in [0, 1])
```

**Key Design Choices:**
- **Spatial preservation**: Maintains spatial dimensions (H, W) to provide per-location predictions
- **Single channel output**: One confidence value per spatial location
- **Sigmoid activation**: Ensures output is in [0, 1] range (1=visible, 0=occluded)
- **Lightweight**: Only 2 conv layers to avoid overfitting and maintain efficiency

### Integration with Backbone

The occlusion head branches off from the backbone's feature maps **before** the final embedding layer:

```
Backbone CNN
    ↓
Feature Maps [B, 512, 7, 7]
    ├──→ Output Layer → Embedding [B, 512] (for AdaFace)
    └──→ OcclusionHead → Occlusion Map [B, 1, 7, 7]
```

This allows the occlusion head to:
- Share the same feature representation as the recognition task
- Benefit from multi-task learning
- Operate at the same spatial resolution as QAConv matching

---

## Training Methodology

### Supervised Learning with Synthetic Occlusions

The occlusion head is trained using **ground truth occlusion masks** generated during data augmentation.

#### 1. Data Augmentation Strategy

**Two types of synthetic occlusions:**

**A. Medical Mask Occlusion** (`MedicalMaskOcclusion`)
- Applies a realistic medical mask image to faces
- Uses facial landmarks (nose, mouth corners) for positioning
- Returns: `(occluded_image, binary_mask)` where mask is `[H, W]` with values 0 (occluded) or 255 (visible)

**B. Synthetic Rectangular Occlusion** (`SyntheticOcclusionMask`)
- Creates random rectangular patches
- Multiple patches possible (1-3 patches per image)
- Returns: `(occluded_image, feature_map_mask)` where mask is `[7, 7]` matching feature map resolution

**Key Properties:**
- Probability-based application (e.g., 50% chance of occlusion)
- Spatial alignment: masks are downsampled to match feature map resolution (7x7 for 112x112 input)
- Binary or continuous values: ground truth can be binary (0/1) or continuous (0.0-1.0)

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
- Predicted: `[B, 1, 7, 7]` (from OcclusionHead)
- Ground Truth: `[B, 1, H, W]` (from augmentation, e.g., 112x112)
- Solution: Downsample GT to `[B, 1, 7, 7]` using bilinear interpolation

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
   weight[i, j] = query_occlusion[i] * gallery_occlusion[j]
   weighted_correlation = correlation * weight
   ```
   - If either location is occluded (low confidence), the correlation gets low weight
   - Only visible-to-visible matches contribute strongly

3. **Pooling**: Max pooling over spatial dimensions to get final similarity score

#### Implementation Details

**Method: "scaling" (recommended)**
```python
# Flatten occlusion maps: [B, 1, H, W] -> [B, H*W]
query_flat = query_occ.view(B_query, -1)
gallery_flat = gallery_occ.view(B_gallery, -1)

# Create weight matrix via broadcasting
q_weight = query_flat.unsqueeze(1).unsqueeze(3)  # [B_query, 1, H*W, 1]
g_weight = gallery_flat.unsqueeze(0).unsqueeze(2)  # [1, B_gallery, 1, H*W]
weight = q_weight * g_weight  # [B_query, B_gallery, H*W, H*W]

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
   - Model sees occluded images (masks, rectangles)
   - Learns to predict occlusion from feature patterns
   - Loss decreases as predictions match ground truth masks

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

- [ ] **Data Augmentation**
  - Synthetic occlusion transforms that return `(image, mask)` tuples
  - Mask downsampling to match feature map resolution
  - Probability-based application

- [ ] **Loss Computation**
  - MSE loss between predicted and ground truth masks
  - Shape alignment (interpolation if needed)
  - NaN/infinity checks

- [ ] **QAConv Integration**
  - `apply_occlusion_weight()` method
  - Pass occlusion maps to `_compute_similarity_batch()`
  - Weight correlation scores before pooling

### Training Considerations

- [ ] **Loss Weighting**
  - Configurable weights for each loss component
  - Start with small occlusion loss weight (0.1) and increase gradually

- [ ] **Ground Truth Availability**
  - Handle batches with and without occlusion masks
  - Skip occlusion loss if masks not available

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

1. **Spatial Awareness**: Predict occlusion per spatial location, not globally
2. **Lightweight**: Minimal parameters to avoid overfitting
3. **Multi-Task Learning**: Share features with recognition task
4. **Self-Supervised Signal**: Use synthetic occlusions for training
5. **Direct Integration**: Occlusion maps directly influence matching scores
6. **Robustness**: Works even when no occlusion is present (predicts all visible)

---

## Mathematical Formulation

### Occlusion Prediction

Given feature maps `F ∈ R^(B×C×H×W)`, the occlusion head predicts:

```
O = σ(Conv2(ReLU(BN(Conv1(F)))))
```

where `O ∈ R^(B×1×H×W)` and `σ` is the sigmoid function.

### Occlusion-Aware Matching

For query features `F_q` and gallery features `F_g` with occlusion maps `O_q` and `O_g`:

1. Compute spatial correlation:
   ```
   S[i,j] = F_q[i] · F_g[j]  for all spatial locations i, j
   ```

2. Apply occlusion weighting:
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

4. **Documentation**: Document the spatial resolution assumptions (7x7 for 112x112 input, 14x14 for 224x224)

5. **Testing**: Test each component independently before integration

6. **Configuration**: Make occlusion probability, loss weights, and architecture parameters configurable

