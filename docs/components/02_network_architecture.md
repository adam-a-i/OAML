# Network Architecture Components

## Overview
The OAML framework implements a dual-branch architecture combining global face embeddings (AdaFace) with local feature matching (QAConv) for robust face recognition under occlusion. The architecture consists of CNN backbones, loss heads, QAConv matchers, and an occlusion prediction head.

## Architecture Diagram

```
Input Image [3, 112, 112]
      │
      ▼
┌─────────────────────────────────────────────┐
│         Backbone (IR-18/50/101)             │
│  ┌────────────┐                             │
│  │ Input Layer│ [64, 56, 56]                │
│  └─────┬──────┘                             │
│        │                                     │
│  ┌─────▼──────┐                             │
│  │  Body      │ [512, 7, 7]                 │
│  │ (Residual  │                             │
│  │  Blocks)   │                             │
│  └─────┬──────┘                             │
│        │                                     │
│        ├──────────────┬─────────────────────┤
│        │              │                     │
└────────┼──────────────┼─────────────────────┘
         │              │                     │
         │              │              ┌──────▼───────┐
         │              │              │ Occlusion    │
         │              │              │ Head         │
         │              │              │ [1, 7, 7]    │
         │              │              └──────────────┘
         │              │
    ┌────▼─────┐   ┌───▼──────┐
    │ Output   │   │  QAConv  │
    │ Layer    │   │ Matcher  │
    │ [512]    │   │ [512,7,7]│
    └────┬─────┘   └────┬─────┘
         │              │
    ┌────▼─────┐   ┌───▼──────┐
    │ AdaFace  │   │ Pairwise │
    │ Head     │   │ Matching │
    │ [classes]│   │ [B, B]   │
    └──────────┘   └──────────┘
```

---

## Component 1: net.py - Network Backbones

### Purpose
Implements ResNet-based backbones (IR-18, IR-34, IR-50, IR-101) with integrated QAConv and occlusion prediction capabilities.

### Key Classes

#### 1. OcclusionHead(Module)

**Purpose**: Predicts per-location occlusion confidence maps from feature maps.

**Architecture**:
```python
OcclusionHead(in_channels=512, hidden_channels=256)
  ├── Conv2d(512 → 256, kernel=3, padding=1)
  ├── BatchNorm2d(256)
  ├── ReLU
  ├── Conv2d(256 → 1, kernel=1)
  └── Sigmoid  # Output in [0,1]
```

**Input/Output**:
- Input: Feature maps `[B, in_channels, H, W]` (e.g., [B, 512, 7, 7])
- Output: Occlusion map `[B, 1, H, W]` where 1=visible, 0=occluded

**Weight Initialization**:
- Conv weights: Kaiming normal initialization
- Biases: Zero initialization
- BatchNorm: weights=1, bias=0

**Usage**:
```python
occlusion_head = OcclusionHead(in_channels=512)
feature_maps = backbone.body(x)  # [B, 512, 7, 7]
occlusion_map = occlusion_head(feature_maps)  # [B, 1, 7, 7]
```

#### 2. Backbone(Module)

**Purpose**: Core CNN architecture with residual blocks, occlusion head, and QAConv integration.

**Supported Architectures**:
- **IR-18**: 18 layers, BasicBlockIR, 512 channels
- **IR-34**: 34 layers, BasicBlockIR, 512 channels
- **IR-50**: 50 layers, BasicBlockIR, 512 channels
- **IR-101**: 100 layers, BasicBlockIR, 512 channels
- **IR-SE variants**: With Squeeze-and-Excitation blocks

**Architecture Components**:

##### Input Layer
```python
Sequential(
  Conv2d(3 → 64, kernel=3, stride=1, padding=1),
  BatchNorm2d(64),
  PReLU(64)
)
```
- Input: [B, 3, 112, 112] or [B, 3, 224, 224]
- Output: [B, 64, 112, 112] or [B, 64, 224, 224]

##### Body (Residual Blocks)
Block configuration varies by architecture:

**IR-50 Example**:
```python
Block 1: [64 → 64] × 3 units
Block 2: [64 → 128] × 4 units (stride=2)
Block 3: [128 → 256] × 14 units (stride=2)
Block 4: [256 → 512] × 3 units (stride=2)
```

Final output: [B, 512, 7, 7] for 112×112 input

##### Output Layer (Global Embedding)
```python
Sequential(
  BatchNorm2d(512),
  Dropout(0.4),
  Flatten(),
  Linear(512 × 7 × 7 → 512),
  BatchNorm1d(512, affine=False)
)
```

##### QAConv Layer
```python
QAConv(
  num_features=512,
  height=7,
  width=7,
  num_classes=None,  # Set during training
  k_nearest=32
)
```

##### Occlusion Head
```python
OcclusionHead(
  in_channels=512,
  hidden_channels=256
)
```

**Forward Pass**:
```python
def forward(x):
    # 1. Input layer
    x = input_layer(x)  # [B, 64, H, W]
    
    # 2. Residual blocks
    for layer in body:
        x = layer(x)  # [B, 512, 7, 7]
    
    # 3. NaN handling and stabilization
    if torch.isnan(x).any():
        x = torch.nan_to_num(x, nan=0.0)
    
    feature_maps = x
    
    # 4. Occlusion prediction
    occlusion_map = occlusion_head(feature_maps)  # [B, 1, 7, 7]
    
    # 5. Global embedding
    embedding = output_layer(feature_maps)  # [B, 512]
    norm = torch.norm(embedding, 2, 1, True)
    output = torch.div(embedding, norm)  # L2-normalized
    
    return output, norm, occlusion_map
```

**NaN Handling & Numerical Stability**:
The backbone implements comprehensive NaN detection and handling:

```python
# Layer-by-layer checking
for idx, module in enumerate(self.body):
    x = module(x)
    if torch.isnan(x).any():
        print(f"WARNING: NaN at layer {idx}")
        x = torch.nan_to_num(x, nan=0.0)
    
    # Special handling for problematic layers (21-23)
    if idx in [21, 22, 23]:
        # Stabilize small values
        small_mask = (x.abs() < 1e-6) & (x != 0)
        x = torch.where(small_mask, torch.sign(x) * 1e-6, x)
        
        # Channel-wise normalization if needed
        norms = torch.norm(x.view(x.size(0), x.size(1), -1), dim=2)
        if (norms < 1e-7).any():
            x = F.layer_norm(x, [x.size(2), x.size(3)])
```

**Helper Methods**:

##### `get_feature_maps(x)`
Extracts intermediate feature maps for QAConv:
```python
feature_maps = model.get_feature_maps(images)  # [B, 512, 7, 7]
```

##### `set_gallery_features(gallery_features)`
Stores gallery features for inference-time matching:
```python
model.set_gallery_features(gallery_features)
embeddings, norms, occlusion_maps, scores = model(query_images)
```

**Model Building**:
```python
def build_model(model_name='ir_50'):
    if model_name == 'ir_18':
        return IR_18(input_size=(112, 112))
    elif model_name == 'ir_50':
        return IR_50(input_size=(112, 112))
    elif model_name == 'ir_101':
        return IR_101(input_size=(112, 112))
    # ... other variants
```

### Building Blocks

#### BasicBlockIR
Basic residual block without bottleneck:
```python
BasicBlockIR(in_channel, depth, stride)
  ├── Shortcut: MaxPool2d or Conv2d+BN
  └── Residual:
      ├── BatchNorm2d
      ├── Conv2d(in_channel → depth, 3×3)
      ├── BatchNorm2d
      ├── PReLU
      ├── Conv2d(depth → depth, 3×3, stride)
      └── BatchNorm2d
```

#### BottleneckIR
Bottleneck residual block for deeper networks:
```python
BottleneckIR(in_channel, depth, stride)
  ├── Shortcut: MaxPool2d or Conv2d+BN
  └── Residual:
      ├── BatchNorm2d
      ├── Conv2d(in_channel → depth//4, 1×1)
      ├── BatchNorm2d + PReLU
      ├── Conv2d(depth//4 → depth//4, 3×3)
      ├── BatchNorm2d + PReLU
      ├── Conv2d(depth//4 → depth, 1×1, stride)
      └── BatchNorm2d
```

#### BasicBlockIRSE / BottleneckIRSE
Variants with Squeeze-and-Excitation:
```python
# Adds SEModule after residual path
SEModule(channels, reduction=16)
  ├── AdaptiveAvgPool2d(1)
  ├── Conv2d(channels → channels//16, 1×1)
  ├── ReLU
  ├── Conv2d(channels//16 → channels, 1×1)
  ├── Sigmoid
  └── Multiply with input
```

---

## Component 2: head.py - Loss Heads

### Purpose
Implements various angular margin-based loss heads for face recognition.

### Supported Loss Heads

#### 1. AdaFace

**Purpose**: Adaptive margin loss that adjusts margin based on image quality (measured by embedding norm).

**Formula**:
```
AdaFace = s * cos(θ + m * g_angular - m_additive)

where:
  g_angular = m * margin_scaler * -1  (angular margin)
  g_additive = m + m * margin_scaler  (additive margin)
  margin_scaler = (norm - batch_mean) / (batch_std + eps) * h
```

**Key Parameters**:
- `m`: Base margin (default: 0.4)
- `h`: Margin scaling factor (default: 0.333)
- `s`: Scale factor (default: 64.0)
- `t_alpha`: EMA alpha for batch statistics (default: 1.0)

**Implementation**:
```python
class AdaFace(Module):
    def __init__(self, embedding_size=512, classnum=70722, 
                 m=0.4, h=0.333, s=64., t_alpha=1.0):
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        self.m = m
        self.h = h
        self.s = s
        self.t_alpha = t_alpha
        
        # EMA batch statistics
        self.register_buffer('batch_mean', torch.ones(1) * 20)
        self.register_buffer('batch_std', torch.ones(1) * 100)
    
    def forward(self, embeddings, norms, label):
        # 1. Cosine similarity
        kernel_norm = l2_norm(self.kernel, axis=0)
        cosine = torch.mm(embeddings, kernel_norm)
        
        # 2. Adaptive margin scaling
        safe_norms = torch.clip(norms, min=0.001, max=100)
        margin_scaler = (safe_norms - self.batch_mean) / 
                        (self.batch_std + eps) * self.h
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        
        # 3. Angular margin
        g_angular = self.m * margin_scaler * -1
        m_arc = scatter_margin(label, g_angular)
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=eps, max=π-eps)
        cosine = theta_m.cos()
        
        # 4. Additive margin
        g_add = self.m + (self.m * margin_scaler)
        m_cos = scatter_margin(label, g_add)
        cosine = cosine - m_cos
        
        # 5. Scale
        return cosine * self.s
```

**Advantages**:
- Adaptive to image quality (higher quality → larger margin)
- More robust to low-quality images
- Better generalization on challenging benchmarks

#### 2. ArcFace

**Purpose**: Angular margin-based loss with additive angular margin.

**Formula**:
```
ArcFace = s * cos(θ + m)
```

**Parameters**:
- `m`: Angular margin (default: 0.5)
- `s`: Scale factor (default: 64.0)

**Implementation**:
```python
class ArcFace(Module):
    def forward(self, embeddings, norms, label):
        kernel_norm = l2_norm(self.kernel, axis=0)
        cosine = torch.mm(embeddings, kernel_norm)
        
        m_hot = scatter_margin(label, self.m)
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_hot, min=eps, max=π-eps)
        cosine_m = theta_m.cos()
        
        return cosine_m * self.s
```

#### 3. CosFace

**Purpose**: Cosine margin-based loss with additive cosine margin.

**Formula**:
```
CosFace = s * (cos(θ) - m)
```

**Parameters**:
- `m`: Cosine margin (default: 0.4)
- `s`: Scale factor (default: 64.0)

**Implementation**:
```python
class CosFace(Module):
    def forward(self, embeddings, norms, label):
        kernel_norm = l2_norm(self.kernel, axis=0)
        cosine = torch.mm(embeddings, kernel_norm)
        
        m_hot = scatter_margin(label, self.m)
        cosine = cosine - m_hot
        
        return cosine * self.s
```

**Comparison**:

| Loss Head | Margin Type | Formula | Adaptive | Best For |
|-----------|------------|---------|----------|----------|
| AdaFace | Angular + Additive | Adaptive | ✓ | Low-quality, diverse datasets |
| ArcFace | Angular | Fixed | ✗ | High-quality images |
| CosFace | Additive | Fixed | ✗ | Simple, fast training |

**Builder Function**:
```python
def build_head(head_type, embedding_size, class_num, m, t_alpha, h, s):
    if head_type == 'adaface':
        return AdaFace(embedding_size, class_num, m, h, s, t_alpha)
    elif head_type == 'arcface':
        return ArcFace(embedding_size, class_num, m, s)
    elif head_type == 'cosface':
        return CosFace(embedding_size, class_num, m, s)
```

---

## Component 3: qaconv.py - Query-Adaptive Convolution

### Purpose
Implements QAConv for local feature matching, enabling robust recognition under occlusion by comparing spatial feature maps.

### Class: QAConv(Module)

**Architecture**:
```
Query Features [B_q, C, H, W]    Gallery Features [B_g, C, H, W]
        │                                │
        └────────────┬───────────────────┘
                     │
            ┌────────▼─────────┐
            │  Correlation     │
            │  [B_q, B_g, hw, hw]
            └────────┬──────────┘
                     │
            ┌────────▼─────────┐
            │  Occlusion       │
            │  Weighting       │ (optional)
            └────────┬──────────┘
                     │
            ┌────────▼─────────┐
            │  Max Pooling     │
            │  [B_q, B_g, hw*2]│
            └────────┬──────────┘
                     │
            ┌────────▼─────────┐
            │  BatchNorm1d(1)  │
            └────────┬──────────┘
                     │
            ┌────────▼─────────┐
            │  Linear(hw*2 → 1)│
            └────────┬──────────┘
                     │
            ┌────────▼─────────┐
            │  BatchNorm1d(1)  │
            └────────┬──────────┘
                     │
              Similarity Scores
              [B_q, B_g]
```

**Key Parameters**:
- `num_features`: Feature channels (512 for IR-50)
- `height`, `width`: Feature map spatial dimensions (7×7 for 112×112 input)
- `num_classes`: Number of identity classes (for training)
- `k_nearest`: Number of nearest neighbor classes (default: 32)

**Initialization**:
```python
QAConv(
    num_features=512,
    height=7,
    width=7,
    num_classes=70722,  # Optional, for training
    k_nearest=32
)
```

### Core Methods

#### 1. `forward(prob_fea, gal_fea, labels, query_occ, gallery_occ, occlusion_method)`

**Mode 1: Training with Class Embeddings**
```python
# When self.training and gal_fea is None
scores = qaconv(probe_features, gal_fea=None, labels=batch_labels)
# Returns: [B, num_classes]
```

Process:
1. Compute k-nearest neighbor classes based on class embeddings
2. For each sample, match against its class neighbors
3. Return classification scores

**Mode 2: Pairwise Matching**
```python
# When gal_fea is provided
scores = qaconv(probe_features, gallery_features)
# Returns: [B_probe, B_gallery]
```

Process:
1. Normalize features
2. Compute correlation (einsum)
3. Apply occlusion weighting (optional)
4. Max pooling over spatial dimensions
5. Linear transformation and batch normalization

#### 2. `_compute_similarity_batch(prob_fea, gal_fea, query_occ, gallery_occ, occlusion_method)`

Core similarity computation:

```python
# 1. Reshape features
prob_fea = prob_fea.view(B_p, C, hw)  # [B_p, 512, 49]
gal_fea = gal_fea.view(B_g, C, hw)    # [B_g, 512, 49]

# 2. Compute correlation (chunked for memory efficiency)
score = torch.zeros(B_p, B_g, hw, hw)
for i in range(0, B_p, chunk_size):
    for j in range(0, B_g, chunk_size):
        score[i:end_i, j:end_j] = torch.einsum(
            'p c s, g c r -> p g r s',
            prob_chunk, gal_chunk
        )

# 3. Apply occlusion weighting (if provided)
if query_occ is not None and gallery_occ is not None:
    score = apply_occlusion_weight(score, query_occ, gallery_occ)

# 4. Max pooling
max_r = score.max(dim=2)[0]  # [B_p, B_g, hw]
max_s = score.max(dim=3)[0]  # [B_p, B_g, hw]
score = torch.cat((max_r, max_s), dim=-1)  # [B_p, B_g, hw*2]

# 5. Batch norm + Linear + Batch norm
score = bn(score.view(-1, 1, hw*2))  # [B_p*B_g, 1, hw*2]
score = fc(score.view(-1, hw*2))     # [B_p*B_g, 1]
score = logit_bn(score)
score = score.view(B_p, B_g)

return score
```

#### 3. `apply_occlusion_weight(similarity, query_occ, gallery_occ, method)`

**Purpose**: Weight similarity scores by occlusion confidence.

**Method 1: Scaling (Efficient)**
```python
# Query occlusion: [B_p, 1, H, W] → [B_p, hw]
# Gallery occlusion: [B_g, 1, H, W] → [B_g, hw]

q_weight = query_flat.unsqueeze(1).unsqueeze(3)   # [B_p, 1, hw, 1]
g_weight = gallery_flat.unsqueeze(0).unsqueeze(2) # [1, B_g, 1, hw]
weight = q_weight * g_weight  # [B_p, B_g, hw, hw]

return similarity * weight
```

**Method 2: Outer Product (Full)**
```python
# Full outer product weighting
# Same as scaling but with explicit outer product
```

**Rationale**:
- Downweights occluded regions in matching
- Focuses on visible facial features
- Improves robustness under occlusion

#### 4. `compute_class_neighbors()`

**Purpose**: Precompute k-nearest neighbor classes for efficient training.

```python
# 1. Normalize all class embeddings
class_embeds_flat = F.normalize(
    self.class_embed.view(num_classes, -1), p=2, dim=1
)

# 2. Compute pairwise similarities (chunked)
similarity_matrix = torch.zeros(num_classes, num_classes)
for i in range(0, num_classes, chunk_size):
    for j in range(0, num_classes, chunk_size):
        chunk_i = class_embeds_flat[i:end_i]
        chunk_j = class_embeds_flat[j:end_j]
        similarity_matrix[i:end_i, j:end_j] = torch.mm(chunk_i, chunk_j.t())

# 3. Get top-k neighbors (excluding self)
_, indices = similarity_matrix.topk(k=k_nearest + 1, dim=1)
self.class_neighbors = indices[:, 1:]  # [num_classes, k_nearest]
```

#### 5. `match_pairs(probe_features, gallery_features, query_occ, gallery_occ)`

**Purpose**: Efficient pairwise matching for validation (memory-optimized).

```python
# Process in chunks for memory efficiency
pair_scores = torch.zeros(batch_size)

for start in range(0, batch_size, chunk_size):
    probe_chunk = probe_features[start:end]
    gallery_chunk = gallery_features[start:end]
    
    # Normalize
    probe_chunk = F.normalize(probe_chunk, p=2, dim=1)
    gallery_chunk = F.normalize(gallery_chunk, p=2, dim=1)
    
    # Match each pair individually
    for i in range(len(probe_chunk)):
        single_probe = probe_chunk[i:i+1]
        single_gallery = gallery_chunk[i:i+1]
        
        # Compute correlation
        corr = torch.einsum('p c s, g c r -> p g r s', 
                           single_probe, single_gallery)
        
        # Apply occlusion weighting
        if query_occ is not None:
            corr = apply_occlusion_weight(corr, ...)
        
        # Max pooling + FC
        features = torch.cat((corr.max(dim=2)[0], 
                            corr.max(dim=3)[0]), dim=-1)
        score = logit_bn(fc(bn(features)))
        pair_scores[start + i] = score.item()

return pair_scores
```

### Class Embeddings

**Purpose**: Learnable prototypes for each identity class.

```python
# Initialization
self.class_embed = nn.Parameter(
    torch.randn(num_classes, num_features, height, width) / num_features**0.5
)
# Shape: [num_classes, 512, 7, 7]
```

**Training**:
- Updated via backpropagation like other parameters
- Represents typical feature maps for each class
- Used for classification and neighbor computation

### Graph Sampling

**Purpose**: Construct batches based on class similarity for better metric learning.

**Process**:
1. Compute class neighbors at epoch start
2. Sample K instances from each class
3. Include instances from neighbor classes in batch
4. Creates hard positive/negative pairs

**Benefits**:
- More informative batches
- Better gradient signals
- Faster convergence

---

## Component 4: face.py - Dataset Wrapper

### Purpose
Simple wrapper class for loading face recognition datasets with consistent interface.

### Class: Face

```python
class Face:
    def __init__(self, root):
        self.images_dir = root
        self.data = []
        self.num_ids = 0
        self.load()
    
    def preprocess(self):
        # Scan directory structure
        # Extract person IDs from filenames
        # Build dataset index
        return data, num_ids
    
    def get_image_path(self, fname):
        return os.path.join(self.images_dir, fname)
```

**Directory Structure**:
```
root/
  ├── class1/
  │   ├── S1-P1-F-39-1.jpg
  │   ├── S1-P1-F-39-2.jpg
  │   └── ...
  ├── class2/
  │   └── ...
  └── ...
```

**Person ID Extraction**:
```python
# From filename: S2-P1-F-39-1.jpg
# Extract: S2-P1 (first two hyphen-separated parts)
pid = '-'.join(fname.split('-')[:2])
```

**Usage**:
```python
dataset = Face(root='/path/to/images')
print(f"Loaded {dataset.num_ids} identities")
print(f"Total images: {len(dataset.data)}")

for img_path, label in dataset.data:
    image = Image.open(dataset.get_image_path(img_path))
```

---

## Integration & Data Flow

### Training Flow
```
Batch [B, 3, 112, 112]
  │
  ├──> Backbone (net.py)
  │    ├──> input_layer: [B, 64, 112, 112]
  │    ├──> body: [B, 512, 7, 7]
  │    ├──> output_layer: [B, 512]  (AdaFace embeddings)
  │    ├──> QAConv: [B, 512, 7, 7]  (feature maps)
  │    └──> OcclusionHead: [B, 1, 7, 7]
  │
  ├──> AdaFace Head (head.py)
  │    └──> scores: [B, num_classes]
  │
  └──> QAConv Matching (qaconv.py)
       ├──> Pairwise: [B, B]
       └──> Triplet: classification + ranking
```

### Inference Flow
```
Query Image [1, 3, 112, 112]
  │
  ├──> Backbone
  │    ├──> AdaFace embedding: [1, 512]
  │    └──> QAConv features: [1, 512, 7, 7]
  │
  ├──> AdaFace Matching
  │    └──> L2 distance with gallery
  │
  └──> QAConv Matching
       └──> Correlation with gallery features
```

---

## Design Decisions

### 1. Dual-Branch Architecture
**Rationale**: Combines global (AdaFace) and local (QAConv) features for complementary strengths.

### 2. Occlusion Prediction
**Rationale**: Provides interpretable occlusion maps and enables occlusion-aware matching.

### 3. Residual Connections
**Rationale**: Enables training very deep networks (50-101 layers) without degradation.

### 4. Adaptive Margins (AdaFace)
**Rationale**: Adjusts difficulty based on image quality, improving generalization.

### 5. Class Embeddings in QAConv
**Rationale**: Enables efficient classification-based training without storing all feature maps.

### 6. Graph Sampling
**Rationale**: Constructs informative batches based on class similarity for better metric learning.

---

## Performance Considerations

### Memory Optimization
- **Chunked Processing**: QAConv processes large batches in chunks
- **Gradient Checkpointing**: Can be enabled for very deep networks
- **Mixed Precision**: FP16 training reduces memory by ~50%

### Computational Efficiency
- **Einsum Operations**: Optimized tensor contractions for correlation
- **Batch Normalization**: Accelerates training and improves stability
- **Cached Class Neighbors**: Computed once per epoch, not per batch

### Numerical Stability
- **NaN Detection**: Multi-stage checking and replacement
- **Norm Clipping**: Prevents extreme values in AdaFace
- **Epsilon Values**: Added to prevent division by zero
- **Layer Normalization**: Applied when channel norms collapse

---

## Usage Examples

### Load Pretrained Model
```python
model = net.build_model('ir_50')
checkpoint = torch.load('model.ckpt')
model.load_state_dict(checkpoint['state_dict'])
model.eval()
```

### Extract Features
```python
# AdaFace embeddings
embeddings, norms, occlusion_maps = model(images)

# QAConv feature maps
feature_maps = model.body(model.input_layer(images))
feature_maps = F.normalize(feature_maps, p=2, dim=1)
```

### Matching
```python
# AdaFace: cosine similarity
similarity = torch.mm(embeddings, gallery_embeddings.t())

# QAConv: pairwise matching
qaconv = model.qaconv
similarity = qaconv(feature_maps, gallery_feature_maps)
```

### Occlusion Analysis
```python
# Get occlusion maps
_, _, occlusion_maps = model(images)

# Visualize
import matplotlib.pyplot as plt
plt.imshow(occlusion_maps[0, 0].cpu().numpy())
plt.colorbar()
plt.title('Occlusion Confidence (1=visible, 0=occluded)')
```

---

## Notes

### Backbone Selection
- **IR-18/34**: Fast, suitable for edge devices
- **IR-50**: Good balance of accuracy and speed
- **IR-101**: Best accuracy, higher computational cost
- **IR-SE variants**: Better performance with SE modules

### Loss Head Selection
- **AdaFace**: Best for diverse/challenging datasets
- **ArcFace**: Good baseline, widely used
- **CosFace**: Simpler, faster convergence

### Common Issues
1. **OOM**: Reduce batch size or use gradient checkpointing
2. **NaN Losses**: Lower learning rate, check normalization
3. **Slow Training**: Enable mixed precision, reduce validation frequency

