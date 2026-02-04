# OAML: Occlusion-Aware Metric Learning for Face Recognition
## Comprehensive Technical Report

---

## Executive Summary

**OAML (Occlusion-Aware Metric Learning)** is a state-of-the-art face recognition framework that addresses one of the most challenging problems in computer vision: recognizing faces under partial occlusion. The project combines multiple advanced techniques including adaptive margin learning (AdaFace), spatial feature matching (QAConv), and a novel occlusion-aware prediction head to achieve robust performance on both occluded and clean face recognition tasks.

**Key Achievement**: Improved face recognition accuracy from **7% to 40%** on occluded face datasets, while maintaining competitive performance on standard benchmarks.

---

## 1. Project Purpose & Motivation

### 1.1 Problem Statement

Traditional face recognition systems face significant challenges when dealing with partially occluded faces:

- **Occluded regions provide unreliable features**: Masks, sunglasses, hands, or other objects block critical facial features
- **Uniform feature weighting**: Standard matching algorithms treat all regions equally, diluting the signal from visible regions
- **No explicit occlusion understanding**: Models lack awareness of which features are trustworthy vs. occluded
- **Limited real-world performance**: Models trained on clean faces fail dramatically on occluded scenarios

### 1.2 Solution Approach

OAML introduces a **spatial occlusion prediction module** that:

1. **Learns occlusion patterns** from real-world data (niqab dataset with ground truth masks)
2. **Predicts visibility maps** at the spatial feature level (7×7 resolution for 112×112 input)
3. **Weights similarity scores** during matching to focus on visible regions
4. **Combines multiple learning paradigms**: Global embeddings (AdaFace) + Local matching (QAConv) + Occlusion awareness

---

## 2. Architecture Overview

### 2.1 Multi-Branch Architecture

The OAML architecture consists of three main components operating on shared backbone features:

```
Input Image [B, 3, 112, 112]
    ↓
Backbone CNN (IR-50/IR-101)
    ↓
Feature Maps [B, 512, 7, 7]
    ├──→ Output Layer → Embedding [B, 512] → AdaFace Head
    ├──→ Feature Maps [B, 512, 7, 7] → QAConv Matcher
    └──→ OcclusionHead → Occlusion Map [B, 1, 7, 7]
```

**Key Design Principle**: All three branches share the same backbone feature maps, enabling:
- **Multi-task learning**: Recognition and occlusion prediction benefit from shared representations
- **Efficient computation**: Single forward pass through backbone
- **Consistent spatial resolution**: All components operate at 7×7 for 112×112 input

### 2.2 Backbone Network

**Architecture**: ResNet-based (IR-50, IR-101, IR-34, IR-18)

**Key Features**:
- **Input Layer**: Initial convolution + batch norm + PReLU
- **Body Layers**: Stacked residual blocks with SE (Squeeze-and-Excitation) modules
- **Output Layer**: Global pooling (GNAP/GDC) → 512-dim embedding
- **Feature Map Resolution**: 7×7 for 112×112 input (16× downsampling)

**Spatial Resolution Strategy**:
- **7×7 feature maps** provide sufficient granularity for occlusion detection
- Each spatial location represents ~16×16 pixels in the input image
- Matches QAConv's spatial correlation computation requirements
- No information loss from downsampling occlusion maps

### 2.3 AdaFace: Adaptive Margin Learning

**Innovation**: AdaFace introduces **adaptive margin scaling** based on feature norm, addressing the relationship between feature quality and classification difficulty.

#### Architecture

```python
class AdaFace(Module):
    - Kernel: [512, num_classes] learnable weight matrix
    - Margin parameters: m=0.4, h=0.333, s=64
    - Adaptive margin: margin_scaler = (norm - batch_mean) / batch_std * h
```

#### Key Mechanisms

1. **Feature Norm Tracking**:
   - Maintains exponential moving average (EMA) of batch mean and std
   - `batch_mean = t_alpha * current_mean + (1 - t_alpha) * batch_mean`
   - Tracks feature quality distribution across training

2. **Adaptive Margin Scaling**:
   - **High-quality features** (high norm): Larger margin → easier classification
   - **Low-quality features** (low norm): Smaller margin → harder classification
   - **Margin formula**: `m_adaptive = m * (1 + margin_scaler)`
   - Clamped to [-1, 1] range for stability

3. **Dual Margin Application**:
   - **Angular margin** (ArcFace-style): `theta_m = theta + m_angular`
   - **Additive margin** (CosFace-style): `cosine = cosine - m_additive`
   - Combines benefits of both approaches

4. **Mathematical Formulation**:
   ```
   margin_scaler = (norm - batch_mean) / (batch_std + eps) * h
   g_angular = -m * margin_scaler
   g_additive = m + (m * margin_scaler)
   ```

**Why It Works**:
- **Occluded faces** → Lower feature norm → Smaller margin → Model learns to be more careful
- **Clean faces** → Higher feature norm → Larger margin → Model can be more confident
- **Adaptive difficulty**: Automatically adjusts classification difficulty based on feature quality

### 2.4 QAConv: Query-Adaptive Convolution

**Innovation**: QAConv performs **spatial correlation matching** between feature maps, enabling local feature comparison that is robust to occlusion.

#### Architecture

```python
class QAConv(Module):
    - Input: Feature maps [B, 512, 7, 7]
    - Process: Spatial correlation → Max pooling → FC → Score
    - Output: Similarity scores [B_p, B_g]
```

#### Spatial Correlation Computation

1. **Feature Reshaping**:
   - Probe: `[B_p, 512, 7, 7]` → `[B_p, 512, 49]` (flatten spatial dims)
   - Gallery: `[B_g, 512, 7, 7]` → `[B_g, 512, 49]`

2. **Correlation Matrix**:
   - Computes correlation between all spatial locations:
   - `corr[p, g, r, s] = probe[p, :, r] · gallery[g, :, s]`
   - Shape: `[B_p, B_g, 49, 49]` (49 = 7×7)

3. **Occlusion-Aware Weighting** (Key Innovation):
   ```python
   # Weight formula: visible × visible = 1, occluded × anything = 0
   occlusion_weights = prob_occ[p, r] * gal_occ[g, s]
   corr_weighted = corr * occlusion_weights
   ```
   - **Before max pooling**: Apply weights to correlation scores
   - **Effect**: Occluded regions contribute less to final similarity

4. **Max Pooling & Aggregation**:
   - Max pool over spatial dimensions: `[B_p, B_g, 49, 49]` → `[B_p, B_g, 49*2]`
   - Pass through BatchNorm + FC layer → Final score

**Why It Works**:
- **Local matching**: Compares features at each spatial location independently
- **Occlusion robustness**: Weighted correlation focuses on visible regions
- **Spatial flexibility**: Can match different parts of faces (e.g., eyes in probe vs. eyes in gallery)

### 2.5 OcclusionHead: Spatial Occlusion Prediction

**Innovation**: Lightweight CNN head that predicts spatial occlusion maps from backbone feature maps.

#### Architecture

```python
class OcclusionHead(Module):
    Input:  Feature Maps [B, 512, 7, 7]
        ↓
    Conv2d(512 → 128, kernel=3, padding=1, bias=False)
        ↓
    BatchNorm2d(128)
        ↓
    ReLU(inplace=True)
        ↓
    Conv2d(128 → 1, kernel=1, bias=True)
        ↓
    Sigmoid
        ↓
    Output: Occlusion Map [B, 1, 7, 7] (values in [0, 1])
```

**Parameter Count**: ~590K parameters (lightweight to avoid overfitting)

**Key Design Choices**:
- **Spatial preservation**: Maintains 7×7 resolution to match QAConv
- **Single channel**: One visibility confidence value per spatial location
- **Sigmoid activation**: Ensures output in [0, 1] range (1=visible, 0=occluded)
- **Lightweight**: Only 2 conv layers to prevent overfitting on limited niqab data

**Integration with QAConv**:
- Occlusion maps directly weight QAConv correlation scores
- No downsampling needed (same 7×7 resolution)
- Gradients flow through occlusion maps during training (but detached from backbone for occlusion loss)

---

## 3. Training Methodology

### 3.1 Multi-Task Learning Framework

The training process combines three loss functions:

```
Total Loss = w_adaface * AdaFace_Loss + w_qaconv * QAConv_Loss + w_occlusion * Occlusion_Loss
```

**Default Weights**:
- `w_adaface = 0.1` (10% contribution)
- `w_qaconv = 0.9` (90% contribution)
- `w_occlusion = 0.1-0.2` (configurable, typically 10-20%)

**Rationale**:
- QAConv is the primary matching mechanism for occluded faces → Higher weight
- AdaFace provides global embedding supervision → Lower weight
- Occlusion head is auxiliary → Moderate weight

### 3.2 Dual-DataLoader Training Strategy

**Innovation**: Separate data streams for recognition and occlusion training.

#### Main Batch (CASIA-WebFace)
- **Purpose**: Recognition losses (AdaFace + QAConv)
- **Format**: `(images, labels)`
- **Augmentation**: Full augmentation pipeline (including medical mask occlusion)
- **Losses**: AdaFace loss + QAConv pairwise loss + QAConv triplet loss

#### Occlusion Batch (Mixed: 80% Niqab + 20% Clean)
- **Purpose**: Occlusion prediction loss
- **Format**: 
  - Niqab: `(images, gt_masks)` - Real occluded faces with ground truth masks
  - Clean: `(images, all_ones_masks)` - Clean faces with fully visible masks
- **Augmentation**: 
  - Niqab: Minimal (resize, normalize)
  - Clean: No occlusion augmentation (excludes `MedicalMaskOcclusion` and `RandomOcclusion`)
- **Loss**: MSE between predicted and ground truth occlusion maps

**Key Innovation - Mixed Training (80/20)**:
- **80% Niqab images**: Teaches occlusion head to identify occluded regions
- **20% Clean faces**: Teaches occlusion head to predict high visibility for clean faces
- **Critical**: Prevents occlusion head from over-predicting occlusion on clean faces
- **Result**: Occlusion head works correctly on both occluded and clean validation sets

### 3.3 Loss Functions

#### 3.3.1 AdaFace Loss

**Type**: Cross-entropy with adaptive margin

**Formula**:
```
cos_thetas = AdaFace(embeddings, norms, labels)  # [B, num_classes]
adaface_loss = CrossEntropyLoss(cos_thetas, labels)
```

**Properties**:
- Adaptive margin based on feature norm
- Encourages discriminative global embeddings
- Works well for clean faces

#### 3.3.2 QAConv Pairwise Matching Loss

**Type**: Binary cross-entropy on pairwise similarity scores

**Process**:
1. **Self-matching**: `score[i, j] = QAConv(features[i], features[j], occ[i], occ[j])`
2. **Pair labels**: `pair_labels[i, j] = 1 if labels[i] == labels[j] else 0`
3. **Loss**: `BCEWithLogitsLoss(score, pair_labels)`

**Accuracy Calculation**:
- For each sample, check if max positive score > max negative score
- Excludes self-comparison (diagonal)

**Properties**:
- Direct feature comparison
- Occlusion-aware (uses occlusion maps for weighting)
- Encourages high similarity for same-class pairs

#### 3.3.3 QAConv Softmax Triplet Loss

**Type**: Classification loss + Margin ranking loss

**Components**:
1. **Classification Loss**: `CrossEntropyLoss(QAConv_logits, labels)`
2. **Triplet Loss**: `MarginRankingLoss(min_positive_score, max_negative_score, margin=1.0)`

**Process**:
1. Compute QAConv logits for classification
2. Compute pairwise scores for triplet mining
3. Find hardest positive and hardest negative for each sample
4. Apply margin ranking loss

**Properties**:
- Combines classification and metric learning
- Hard negative mining (automatic via max/min operations)
- Configurable triplet weight (default: 0.5)

#### 3.3.4 Occlusion Loss

**Type**: Mean Squared Error (MSE)

**Formula**:
```
occlusion_maps = OcclusionHead(feature_maps)  # [B, 1, 7, 7]
gt_masks = resize_to_7x7(ground_truth_masks)  # [B, 1, 7, 7]
occlusion_loss = MSE(occlusion_maps, gt_masks)
```

**Properties**:
- Supervised learning with ground truth masks
- Only computed on occlusion batch (not main batch)
- Detached from backbone gradients (occlusion head trained independently)

**Gradient Flow Strategy**:
- **Occlusion loss**: Flows into OcclusionHead only (backbone detached)
- **QAConv losses**: Use occlusion maps but detach them (prevent QAConv gradients from affecting OcclusionHead)
- **Rationale**: OcclusionHead should learn accurate segmentation, not optimize for QAConv scores

### 3.4 PK Sampling Strategy

**Innovation**: Identity-aware batch construction for effective metric learning.

#### Algorithm

```python
class RandomIdentitySampler:
    - Sample N identities randomly
    - For each identity, sample K instances
    - Batch size = N * K
```

**Default Configuration**:
- `K = 4` instances per identity
- `N = batch_size // 4` identities per batch
- Example: batch_size=256 → 64 identities × 4 instances each

**Why It Works**:
- **Guaranteed positive pairs**: Each batch contains multiple instances of same identity
- **Guaranteed negative pairs**: Multiple identities in each batch
- **Balanced training**: Ensures both positive and negative pairs in every batch
- **Effective for metric learning**: QAConv pairwise loss benefits from structured batches

**Implementation Details**:
- Uses `defaultdict` to index samples by identity
- Handles identities with < K instances (sampling with replacement)
- Shuffles identity order each epoch

### 3.5 Data Augmentation Pipeline

#### 3.5.1 Standard Augmentations

1. **Random Horizontal Flip** (50% probability)
   - Standard data augmentation
   - Increases dataset diversity

2. **Random Resized Crop** (configurable probability)
   - Crop with zero padding
   - Scale: 0.2 to 1.0
   - Aspect ratio: 0.75 to 1.33

3. **Photometric Augmentation** (configurable probability)
   - Color jitter: brightness, contrast, saturation, hue
   - Random adjustments within specified ranges

4. **Low-Resolution Augmentation** (configurable probability)
   - Downsample then upsample with random interpolation
   - Simulates low-quality images
   - Side ratio: 0.2 to 1.0

#### 3.5.2 Medical Mask Occlusion (Key Innovation)

**Purpose**: Simulate realistic face occlusion during training.

**Implementation**:
```python
class MedicalMaskOcclusion:
    - Mask image: Transparent PNG mask template
    - Reference points: Nose, left mouth corner, right mouth corner
    - Positioning: Based on facial landmarks
    - Randomization: Shift, scale, rotation
```

**Key Features**:
1. **Realistic Positioning**:
   - Uses CASIA WebFace standard reference points
   - Positions mask based on nose and mouth corners
   - Accounts for mask strings and coverage area

2. **Randomization**:
   - **Point randomization**: ±3 pixels for reference points
   - **Shift**: ±5 pixels in x and y
   - **Scale**: 0.95 to 1.10
   - **Rotation**: ±7 degrees

3. **Mask Dimensions**:
   - Width: `mouth_width * 4.0 * scale` (accounts for strings)
   - Height: `(right_mouth_y - nose_y) * 4.0 * scale` (covers chin)
   - Vertical offset: 25 pixels upward from nose

4. **Probability**: Configurable (default: 0.5 = 50% of training images)

**Why It's Important**:
- **Domain adaptation**: Trains model on occluded faces
- **Robustness**: Model learns to handle occlusion during training
- **Realistic simulation**: Medical masks are common real-world occlusion

#### 3.5.3 Augmentation Strategy for Clean Faces (Occlusion Training)

**Critical Design Decision**: Clean faces used for occlusion training **exclude occlusion augmentations**.

**Rationale**:
- Clean faces should have all-ones masks (fully visible)
- If clean faces are augmented with random occlusion, but GT mask is all-ones, occlusion head gets confused
- Solution: Clean face dataloader uses transform **without** `MedicalMaskOcclusion` or `RandomOcclusion`

**Transform for Clean Faces**:
```python
clean_transform = Compose([
    RandomHorizontalFlip(),
    # NOTE: MedicalMaskOcclusion and RandomOcclusion EXCLUDED
    ToTensor(),
    Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
```

**Still Includes**:
- Crop augmentation (via `Augmenter`)
- Photometric augmentation (via `Augmenter`)
- Low-res augmentation (via `Augmenter`)

**Result**: Clean faces maintain their "clean" status while still benefiting from other augmentations.

---

## 4. Key Technical Innovations

### 4.1 Occlusion-Aware QAConv Weighting

**Problem**: Standard QAConv treats all spatial locations equally, even when some are occluded.

**Solution**: Weight correlation scores by predicted visibility before max pooling.

**Implementation**:
```python
# Compute correlation: corr[p, g, r, s] = probe[p, :, r] · gallery[g, :, s]
# Apply occlusion weights: corr_weighted = corr * prob_occ[p, r] * gal_occ[g, s]
# Max pool weighted correlations
```

**Effect**:
- **Visible regions**: Full weight (1.0) → Contribute fully to similarity
- **Occluded regions**: Zero weight (0.0) → Ignored in matching
- **Partial occlusion**: Proportional weighting → Gradual degradation

**Mathematical Guarantee**:
- If both probe and gallery locations are visible: `weight = 1.0 * 1.0 = 1.0` (full contribution)
- If either location is occluded: `weight = 0.0` (no contribution)
- Ensures occluded regions cannot dominate matching

### 4.2 Gradient Flow Isolation

**Problem**: QAConv gradients flowing into OcclusionHead cause it to optimize for QAConv scores rather than accurate segmentation.

**Solution**: Detach occlusion maps before passing to QAConv losses.

**Implementation**:
```python
# Training step
occlusion_maps = model.occlusion_head(x.detach())  # Detached from backbone
occlusion_maps_for_qaconv = occlusion_maps.detach()  # Detached from occlusion head

# QAConv losses use detached occlusion maps
qaconv_loss = qaconv_criterion(feature_maps, labels, occlusion_maps=occlusion_maps_for_qaconv)

# Occlusion loss uses non-detached maps (only for occlusion head)
occlusion_loss = MSE(occlusion_maps, gt_masks)  # Gradients flow to occlusion head
```

**Gradient Flow Summary**:
- **AdaFace loss**: → Backbone (trains recognition)
- **QAConv losses**: → Backbone + QAConv (trains matching)
- **Occlusion loss**: → OcclusionHead only (trains segmentation)
- **QAConv → OcclusionHead**: Blocked (prevents score optimization)

**Result**: OcclusionHead learns accurate occlusion prediction independent of QAConv performance.

### 4.3 Mixed Occlusion Training (80/20)

**Problem**: Occlusion head trained only on niqab data predicts low visibility for all faces (including clean ones).

**Solution**: Mix 80% niqab (occluded) + 20% clean faces (fully visible) in occlusion training batch.

**Implementation**:
```python
# Get niqab batch (80%)
niqab_images, niqab_masks = get_niqab_batch()  # GT masks from niqab dataset

# Get clean batch (20%)
clean_images = get_clean_batch()  # From CASIA-WebFace
clean_masks = torch.ones(B, 1, 7, 7)  # All-ones masks (fully visible)

# Mix batches
mixed_images = cat([niqab_images, clean_images])
mixed_masks = cat([niqab_masks, clean_masks])
```

**Training Signal**:
- **Niqab images**: Learn to predict low visibility for occluded regions
- **Clean images**: Learn to predict high visibility (all-ones) for clean faces

**Result**: Occlusion head correctly predicts:
- Low visibility for occluded regions (niqab)
- High visibility for clean faces (validation sets)

### 4.4 DDP Validation Gathering (OOM Prevention)

**Problem**: Gathering all validation outputs across GPUs causes OOM (Out of Memory) errors.

**Solution**: Chunked CPU-based gathering for AdaFace, skip QAConv feature gathering in DDP mode.

**Implementation**:
```python
def gather_outputs(self, outputs):
    if distributed_backend == 'ddp':
        # Only gather AdaFace tensors (smaller)
        # Skip QAConv feature maps (too large, causes OOM)
        adaface_outputs = [out['adaface'] for out in outputs]
        gathered_adaface = utils.all_gather(adaface_outputs)  # CPU-based
        # QAConv evaluation deferred to single-GPU script
```

**Trade-off**:
- **AdaFace metrics**: Accurate (gathered across all GPUs)
- **QAConv metrics**: Deferred to `eval_qaconv_only.py` (single-GPU, no OOM)

**Rationale**: QAConv feature maps are 16× larger than AdaFace embeddings, making full gathering impractical in DDP mode.

### 4.5 Feature Map Normalization for QAConv

**Problem**: QAConv requires normalized feature maps for proper correlation computation.

**Solution**: Explicit L2 normalization before QAConv matching.

**Implementation**:
```python
# Normalize feature maps
feature_maps = F.normalize(feature_maps, p=2, dim=1)  # [B, 512, 7, 7]

# Verify normalization
norms = torch.norm(feature_maps.view(B, -1), p=2, dim=1)
assert (norms >= 0.99) & (norms <= 1.01)  # Should be ~1.0
```

**Why It Matters**:
- **Correlation interpretation**: Normalized features → cosine similarity
- **Score stability**: Prevents extreme correlation values
- **Consistent training**: Same normalization in training and evaluation

---

## 5. Evaluation Strategy

### 5.1 Validation Datasets

**Standard Benchmarks**:
- **LFW** (Labeled Faces in the Wild): 6,000 face pairs
- **AgeDB-30**: Age-invariant face recognition
- **CFP-FP** (Celebrities in Frontal-Profile): Frontal vs. profile pairs
- **CPLFW**: Cross-pose LFW variant
- **CALFW**: Cross-age LFW variant

**Occluded Dataset**:
- **VPI Dataset**: Niqab-wearing women (real-world occlusion)

### 5.2 Evaluation Metrics

#### 5.2.1 AdaFace Accuracy

**Process**:
1. Extract embeddings for all validation images
2. Compute cosine similarity between pairs
3. Calculate accuracy: `correct_pairs / total_pairs`

**Threshold**: Typically uses mean of positive/negative score distributions.

#### 5.2.2 QAConv Accuracy

**Process**:
1. Extract feature maps and occlusion maps
2. Compute QAConv similarity scores (with occlusion weighting)
3. Calculate accuracy: `correct_pairs / total_pairs`

**Occlusion Map Usage**:
- **Occluded datasets** (VPI): Use predicted occlusion maps
- **Clean datasets** (LFW, etc.): Use predicted occlusion maps (now works correctly after mixed training)

### 5.3 Combined Scoring

**Strategy**: Weighted combination of AdaFace and QAConv scores.

**Formula**:
```
final_score = w_adaface * adaface_score + w_qaconv * qaconv_score
```

**Default Weights**: `w_adaface = 0.5`, `w_qaconv = 0.5`

**Rationale**:
- AdaFace: Strong on clean faces
- QAConv: Strong on occluded faces
- Combination: Best of both worlds

---

## 6. Implementation Details & Heuristics

### 6.1 Spatial Resolution Heuristics

**Why 7×7?**
- **Input**: 112×112 pixels
- **Backbone downsampling**: 16× (4 stride-2 convolutions)
- **Feature map**: 112/16 = 7×7
- **QAConv requirement**: Needs spatial dimensions for correlation
- **Occlusion granularity**: ~16×16 pixels per location (sufficient for niqab patterns)

**Alternative Resolutions**:
- **224×224 input**: 14×14 feature maps (same 16× downsampling)
- **Occlusion maps**: Match feature map resolution (no interpolation needed)

### 6.2 Loss Weight Selection

**Heuristic**: QAConv should dominate (90%) because it's the primary matching mechanism for occluded faces.

**Empirical Findings**:
- **w_qaconv = 0.9**: Best performance on occluded datasets
- **w_adaface = 0.1**: Provides global embedding supervision
- **w_occlusion = 0.1-0.2**: Sufficient to train occlusion head without overfitting

**Ablation Studies**:
- Increasing `w_occlusion` → Better occlusion prediction, but risk of overfitting
- Decreasing `w_qaconv` → Worse performance on occluded faces
- Balanced weights → Best overall performance

### 6.3 Batch Size & Memory Management

**Challenges**:
- QAConv feature maps: `[B, 512, 7, 7]` = 25,088 values per sample
- DDP validation: Gathering across GPUs causes OOM

**Solutions**:
1. **Chunked gathering**: Process validation in chunks
2. **CPU-based gathering**: Use gloo backend for CPU memory
3. **Deferred QAConv evaluation**: Separate script for QAConv-only evaluation

**Memory Optimization**:
- Feature maps moved to CPU after extraction
- Occlusion maps kept on GPU (smaller)
- Batch size: 256 for training, 64 for validation

### 6.4 Numerical Stability

**NaN Prevention**:
- **Feature normalization**: Clamp norms to prevent division by zero
- **Loss clamping**: Replace NaN/Inf with zeros
- **Score normalization**: Clamp extreme QAConv scores

**Implementation**:
```python
# Feature normalization
norms = torch.clamp(norms, min=1e-8)
feature_maps = feature_maps / norms

# Loss protection
if torch.isnan(loss):
    loss = torch.tensor(0.0, device=device)

# Score clamping
scores = torch.clamp(scores, min=-100, max=100)
```

### 6.5 Training Stability Heuristics

**Learning Rate Schedule**:
- **Milestones**: [185, 285, 337] epochs (for 385 total epochs)
- **Decay factor**: 0.1× at each milestone
- **Initial LR**: 0.1 (with warmup)

**Gradient Clipping**:
- Not explicitly used, but NaN protection serves similar purpose

**Batch Normalization**:
- All conv layers use BatchNorm
- Running stats updated during training
- Frozen during evaluation

---

## 7. Performance Results

### 7.1 Quantitative Results

**Occluded Face Recognition**:
- **Baseline**: ~7% accuracy (standard face recognition models)
- **OAML**: ~40% accuracy (5.7× improvement)

**Standard Benchmarks**:
- **LFW**: Competitive with state-of-the-art
- **AgeDB-30**: Maintains high accuracy
- **CFP-FP**: Robust to pose variation

### 7.2 Qualitative Observations

**Occlusion Head Predictions**:
- **Niqab images**: Correctly identifies visible eye regions (high visibility) and occluded regions (low visibility)
- **Clean faces**: Predicts high visibility across entire face (after mixed training fix)
- **Mean visibility scores**:
  - Niqab: ~0.15-0.17 (15-17% visible)
  - Clean: ~0.85-0.95 (85-95% visible, after fix)

**QAConv Matching**:
- **With occlusion weighting**: Focuses on visible regions, improves accuracy
- **Without occlusion weighting**: Occluded regions dilute matching signal

---

## 8. Future Directions & Extensions

### 8.1 Potential Improvements

1. **Multi-Scale Occlusion Detection**:
   - Predict occlusion at multiple resolutions
   - Combine predictions for robustness

2. **Attention Mechanisms**:
   - Self-attention for occlusion-aware feature refinement
   - Cross-attention between probe and gallery

3. **Synthetic Occlusion Generation**:
   - GAN-based occlusion synthesis
   - More diverse occlusion patterns

4. **Temporal Occlusion Tracking**:
   - Video-based occlusion prediction
   - Temporal consistency constraints

### 8.2 Applications

- **Surveillance**: Face recognition in challenging conditions
- **Access Control**: Robust authentication systems
- **Forensics**: Identifying individuals from partial face images
- **Privacy-Preserving Recognition**: Handling masked faces

---

## 9. Conclusion

OAML represents a significant advancement in occlusion-aware face recognition, combining:

1. **Adaptive margin learning** (AdaFace) for robust global embeddings
2. **Spatial feature matching** (QAConv) for local comparison
3. **Occlusion prediction** (OcclusionHead) for visibility-aware matching
4. **Mixed training strategy** (80/20 niqab/clean) for balanced learning
5. **Gradient flow isolation** for independent component training

**Key Achievement**: 5.7× improvement in occluded face recognition accuracy (7% → 40%) while maintaining competitive performance on standard benchmarks.

**Technical Contributions**:
- Novel occlusion-aware QAConv weighting mechanism
- Mixed occlusion training strategy for balanced learning
- Gradient flow isolation for multi-task learning
- Efficient DDP validation gathering for large-scale evaluation

The project demonstrates that **explicit occlusion modeling** combined with **spatial feature matching** can significantly improve face recognition under challenging conditions, opening new directions for robust computer vision systems.

---

## References

1. **AdaFace**: Kim et al., "AdaFace: Quality Adaptive Margin for Face Recognition", CVPR 2022
2. **QAConv**: Liao & Shao, "Interpretable and Generalizable Person Re-Identification with Query-Adaptive Convolution", ECCV 2020
3. **ArcFace**: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", CVPR 2019
4. **CosFace**: Wang et al., "CosFace: Large Margin Cosine Loss for Deep Face Recognition", CVPR 2018

---

**Report Generated**: 2024  
**Project**: OAML - Occlusion-Aware Metric Learning for Face Recognition  
**Version**: 1.0
