# Loss Functions Components

## Overview
Implements metric learning losses for training QAConv: pairwise matching loss and softmax-triplet loss. These losses operate on feature maps rather than global embeddings, enabling local feature matching.

---

## 1. pairwise_matching_loss.py - Pairwise Matching Loss

### Purpose
Implements binary classification loss for pairwise feature matching using QAConv.

### Class: PairwiseMatchingLoss(Module)

**Formula**:
```
Loss = BCE_logits(scores, pair_labels)

where:
  scores[i,j] = QAConv(feature_i, feature_j)
  pair_labels[i,j] = 1 if label_i == label_j else 0
```

**Implementation**:
```python
class PairwiseMatchingLoss(Module):
    def __init__(self, matcher):
        super().__init__()
        self.matcher = matcher  # QAConv instance
    
    def forward(self, feature, target):
        # Ensure proper tensor format
        feature = feature.detach().clone().requires_grad_(True)
        
        # Compute pairwise scores
        self.matcher.make_kernel(feature)
        score = self.matcher(feature, gal_fea=feature)  # [B, B]
        
        # Create pair labels
        target1 = target.unsqueeze(1)
        mask = (target1 == target1.t())  # [B, B]
        pair_labels = mask.float()
        
        # Binary cross entropy with logits
        loss = F.binary_cross_entropy_with_logits(
            score, pair_labels, reduction='none'
        )
        loss = loss.sum(-1)  # [B]
        
        # Compute accuracy
        with torch.no_grad():
            accuracies = []
            for i in range(score.size(0)):
                sample_scores = score[i]
                positive_mask = pair_labels[i] == 1
                negative_mask = pair_labels[i] == 0
                
                # Exclude self-comparison
                positive_mask[i] = False
                negative_mask[i] = False
                
                if positive_mask.sum() > 0 and negative_mask.sum() > 0:
                    pos_scores = sample_scores[positive_mask]
                    neg_scores = sample_scores[negative_mask]
                    accuracy = (pos_scores.max() > neg_scores.max()).float()
                    accuracies.append(accuracy)
                else:
                    accuracies.append(torch.tensor(0.5))
            
            acc = torch.stack(accuracies)
        
        return loss, acc
```

**Key Features**:
- **Per-Sample Loss**: Returns [B] loss tensor for flexible weighting
- **Accuracy Metric**: Checks if max positive score > max negative score
- **Gradient Handling**: Ensures proper gradient flow through matcher
- **Device Management**: Automatically moves matcher to correct device

**Usage**:
```python
qaconv = QAConv(num_features=512, height=7, width=7)
criterion = PairwiseMatchingLoss(qaconv)

feature_maps = backbone.body(images)  # [B, 512, 7, 7]
loss, accuracy = criterion(feature_maps, labels)

loss = loss.mean()  # Average across batch
loss.backward()
```

**PK Sampling Benefit**:
With P identities × K instances:
- Positive pairs: K-1 pairs per sample within same identity
- Negative pairs: (P-1)×K pairs with different identities
- Total comparisons: B×B where B=P×K

---

## 2. softmax_triplet_loss.py - Softmax-Triplet Loss

### Purpose
Combines classification loss and triplet ranking loss for robust metric learning.

### Class: SoftmaxTripletLoss(Module)

**Formula**:
```
Loss = CrossEntropy(logits, target) + λ * TripletLoss

TripletLoss = max(0, margin + hardest_negative - easiest_positive)

where:
  logits = QAConv(features, labels=target)  # Classification
  hardest_negative = max(score[i,j]) for label[i] ≠ label[j]
  easiest_positive = min(score[i,j]) for label[i] = label[j], i ≠ j
```

**Implementation**:
```python
class SoftmaxTripletLoss(Module):
    def __init__(self, matcher, margin=1.0, triplet_weight=1.0):
        super().__init__()
        self.matcher = matcher
        self.margin = margin
        self.triplet_weight = triplet_weight
        self.cls_loss = nn.CrossEntropyLoss(reduction='none')
        self.ranking_loss = nn.MarginRankingLoss(
            margin=margin, reduction='none'
        )
    
    def forward(self, feature, target):
        # Classification loss
        logits = self.matcher(feature, labels=target)  # [B, num_classes]
        cls_loss = self.cls_loss(logits, target)
        
        # Pairwise similarity scores
        score = self.matcher(feature, "same")  # [B, B]
        
        # Create pair labels
        target1 = target.unsqueeze(1)
        mask = (target1 == target1.t())
        pair_labels = mask.float()
        
        # Find hardest positive (minimum positive score)
        min_pos = torch.min(
            score * pair_labels + 
            (1 - pair_labels + torch.eye(B)) * 1e15,
            dim=1
        )[0]
        
        # Find hardest negative (maximum negative score)
        max_neg = torch.max(
            score * (1 - pair_labels) - pair_labels * 1e15,
            dim=1
        )[0]
        
        # Triplet ranking loss
        triplet_loss = self.ranking_loss(
            min_pos, max_neg, torch.ones_like(target)
        )
        
        # Combined loss
        loss = cls_loss + self.triplet_weight * triplet_loss.mean()
        
        # Compute accuracies
        with torch.no_grad():
            cls_acc = (logits.argmax(dim=1) == target).float()
            triplet_acc = (min_pos >= max_neg).float()
        
        return cls_loss, triplet_loss, loss, cls_acc, triplet_acc
```

**Components**:
1. **Classification Loss**: Standard cross-entropy for class prediction
2. **Triplet Loss**: Margin-based ranking loss
   - Ensures positive pairs have higher scores than negative pairs
   - Margin pushes apart positive and negative distributions

**Key Features**:
- **Dual Objectives**: Classification + metric learning
- **Hard Mining**: Automatically mines hardest examples
- **Flexible Weighting**: Adjustable triplet_weight for balance
- **Multiple Metrics**: Tracks both classification and triplet accuracy

**Usage**:
```python
qaconv = QAConv(num_features=512, height=7, width=7, 
                num_classes=70722, k_nearest=32)
criterion = SoftmaxTripletLoss(
    qaconv, margin=1.0, triplet_weight=0.5
)

feature_maps = backbone.body(images)
cls_loss, triplet_loss, total_loss, cls_acc, triplet_acc = criterion(
    feature_maps, labels
)

total_loss.backward()
```

---

## Loss Combination Strategy

### In train_val.py

**Combined Loss**:
```python
# Loss weights (from train_val.py)
qaconv_loss_weight = 0.7
adaface_loss_weight = 0.1
occlusion_loss_weight = 0.3
triplet_weight = 0.5  # within QAConv loss

# Compute individual losses
adaface_loss = cross_entropy(adaface_scores, labels)

pairwise_loss, pairwise_acc = pairwise_matching_loss(
    feature_maps, labels
)

cls_loss, triplet_loss, _, cls_acc, triplet_acc = softmax_triplet_loss(
    feature_maps, labels
)

# Combine QAConv losses
qaconv_loss = pairwise_loss.mean() + triplet_weight * triplet_loss.mean()

# Occlusion supervision
occlusion_loss = F.mse_loss(pred_occlusion_maps, gt_occlusion_masks)

# Total loss
total_loss = (
    adaface_loss_weight * adaface_loss +
    qaconv_loss_weight * qaconv_loss +
    occlusion_loss_weight * occlusion_loss
)
```

**Rationale**:
- **AdaFace (0.1)**: Baseline face recognition
- **QAConv (0.7)**: Primary metric learning objective
  - Pairwise: Direct matching
  - Triplet (0.5×): Ranking with margin
- **Occlusion (0.3)**: Supervision for interpretability

**Total Effective Weight**: 1.1 (slightly over 1.0 to emphasize QAConv)

---

## Comparison with Standard Losses

### vs. Standard Triplet Loss

| Feature | Standard Triplet | Our Approach |
|---------|-----------------|--------------|
| Input | Global embeddings [B, 512] | Feature maps [B, 512, 7, 7] |
| Matching | Euclidean distance | QAConv correlation |
| Mining | Random or semi-hard | Hardest within batch |
| Occlusion | No handling | Explicit weighting |
| Classification | Separate loss | Integrated |

### vs. Contrastive Loss

| Feature | Contrastive | Our Approach |
|---------|------------|--------------|
| Pairs | Binary (same/different) | Pairwise + classification |
| Margin | Fixed | Adaptive (through QAConv) |
| Spatial | Global | Local (spatial matching) |
| Ranking | No | Yes (triplet component) |

---

## Design Decisions

### 1. Pairwise vs. Triplet
**Decision**: Use both
**Rationale**: 
- Pairwise: Direct similarity learning
- Triplet: Explicit margin enforcement
- Together: Complementary strengths

### 2. Hard Mining
**Decision**: Hardest negative within batch
**Rationale**: 
- More informative gradients
- Faster convergence
- No need for external mining

### 3. Per-Sample Loss
**Decision**: Return [B] tensor, not scalar
**Rationale**:
- Enables sample weighting
- Supports advanced sampling strategies
- Better debugging

### 4. Integrated Classification
**Decision**: QAConv performs classification in triplet loss
**Rationale**:
- Shares computation
- Enforces class structure
- Improves convergence

---

## Training Tips

### Loss Weights Tuning
```python
# Balanced (default)
qaconv_weight = 0.7
adaface_weight = 0.1

# Emphasize AdaFace (when QAConv unstable)
qaconv_weight = 0.3
adaface_weight = 0.5

# Pure metric learning
qaconv_weight = 1.0
adaface_weight = 0.0
```

### Triplet Margin
```python
# Smaller margin: Easier optimization
margin = 0.5

# Larger margin: Better separation (default)
margin = 1.0

# Very large: May slow convergence
margin = 2.0
```

### Monitoring
Key metrics to track:
- Pairwise accuracy: Should reach >70% by end of training
- Triplet accuracy: Should reach >80%
- Classification accuracy: Should reach >60%
- Combined validation accuracy: Primary metric

### Common Issues

**Issue**: QAConv loss is NaN
- **Cause**: Features not normalized
- **Solution**: Ensure `F.normalize(features, p=2, dim=1)`

**Issue**: Accuracy stuck at 50%
- **Cause**: All scores similar, no discrimination
- **Solution**: 
  - Increase learning rate
  - Check feature map quality
  - Verify PK sampling

**Issue**: Loss diverges
- **Cause**: Learning rate too high
- **Solution**: Reduce LR or use warmup

---

## Code Examples

### Basic Usage
```python
# Setup
backbone = net.build_model('ir_50')
qaconv = backbone.qaconv
pairwise_criterion = PairwiseMatchingLoss(qaconv)
triplet_criterion = SoftmaxTripletLoss(qaconv, margin=1.0, triplet_weight=0.5)

# Training step
images, labels = batch
feature_maps = backbone.body(backbone.input_layer(images))

# Normalize features
feature_maps = F.normalize(feature_maps, p=2, dim=1)

# Compute losses
pw_loss, pw_acc = pairwise_criterion(feature_maps, labels)
cls_loss, tri_loss, _, cls_acc, tri_acc = triplet_criterion(feature_maps, labels)

# Combine
qaconv_loss = pw_loss.mean() + 0.5 * tri_loss.mean()
qaconv_loss.backward()
```

### Custom Loss Combination
```python
class CustomLoss(nn.Module):
    def __init__(self, matcher, weights={'pw': 0.5, 'cls': 0.3, 'tri': 0.2}):
        super().__init__()
        self.pairwise = PairwiseMatchingLoss(matcher)
        self.triplet = SoftmaxTripletLoss(matcher)
        self.weights = weights
    
    def forward(self, features, labels):
        pw_loss, pw_acc = self.pairwise(features, labels)
        cls_loss, tri_loss, _, cls_acc, tri_acc = self.triplet(features, labels)
        
        total = (
            self.weights['pw'] * pw_loss.mean() +
            self.weights['cls'] * cls_loss.mean() +
            self.weights['tri'] * tri_loss.mean()
        )
        
        return total, {
            'pw_acc': pw_acc.mean(),
            'cls_acc': cls_acc.mean(),
            'tri_acc': tri_acc.mean()
        }
```

---

## Performance Considerations

### Memory
- **Pairwise Scores**: [B, B] matrix can be large
  - For B=256: 256×256 = 65K scores
  - Memory: ~260KB per batch (fp32)
- **Solution**: Process in chunks if needed

### Speed
- **QAConv Forward**: Dominant computation
  - Correlation: O(B² × C × H × W)
  - For B=64, C=512, H=W=7: ~115M operations
- **Optimization**: Chunked processing in QAConv

### Numerical Stability
- **Large Negative Values**: Use 1e15 for masking, not infinity
- **Division by Zero**: Add epsilon in normalization
- **Gradient Clipping**: Recommended for stability

