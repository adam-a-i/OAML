# Occlusion Layer Implementation Task

## Project Context

**OAML (Occluded Face Recognition)** is a research project focused on improving facial recognition accuracy for occluded faces—an underserved area in the facial recognition research community.

### Current State
- **Architecture**: Combines AdaFace and QAConv for occluded face recognition
- **Performance**: Achieved significant improvement from 7% to 40% accuracy
- **Problem**: Performance has plateaued—AdaFace and QAConv show similar accuracy when evaluated on occluded face datasets (IJB datasets)

### Solution Goal
Implement an **occlusion-aware layer** that predicts spatial occlusion maps and uses them to weight QAConv matching scores, focusing the matching process on visible facial regions.

---

## Task: Implement Occlusion Layer

### Objective
Study the codebase architecture and implement the occlusion layer according to the specification document.

### Key Requirements

1. **Study the Codebase**
   - Understand how AdaFace and QAConv are integrated
   - Understand the feature map flow through the backbone network
   - Understand how QAConv performs spatial matching
   - Understand the training loop and loss computation

2. **Review the Specification**
   - Read and understand: `OCCLUSION_LAYER_SPEC.md`
   - The spec defines:
     - Architecture: Lightweight CNN head (OcclusionHead) that outputs 7x7 occlusion maps
     - Training: Supervised learning on niqab dataset with ground truth masks
     - Integration: Occlusion maps weight QAConv correlation scores
     - Loss: MSE loss between predicted and ground truth occlusion maps

3. **Implementation Plan**
   Create a detailed, step-by-step implementation plan that covers:
   - **OcclusionHead module** (in `net.py`)
   - **Integration with backbone** (modify forward pass)
   - **Dataset loading** (niqab dataset with masks)
   - **Loss computation** (MSE loss for occlusion prediction)
   - **QAConv integration** (occlusion-aware weighting)
   - **Training loop updates** (multi-task learning with occlusion loss)

---

## Key Files to Study

### Core Architecture Files
- **`net.py`**: Backbone network (ResNet-based), feature map extraction, QAConv integration
- **`qaconv.py`**: QAConv matching module, spatial correlation computation
- **`train_val.py`**: Training loop, loss computation, multi-task learning setup
- **`data.py`**: Dataset loading and data pipeline
- **`main.py`**: Training script entry point

### Specification Document
- **`OCCLUSION_LAYER_SPEC.md`**: Complete specification for occlusion layer implementation

---

## Implementation Checklist

Before implementing, ensure you understand:

- [ ] How feature maps flow from backbone → QAConv
- [ ] Current QAConv matching process (spatial correlation computation)
- [ ] How AdaFace and QAConv losses are combined
- [ ] Dataset structure and loading pipeline
- [ ] The specification document requirements

Then implement:

- [ ] **OcclusionHead class** in `net.py`
  - Input: Feature maps `[B, 512, 7, 7]`
  - Output: Occlusion maps `[B, 1, 7, 7]` (values in [0, 1])
  - Architecture: Conv → BN → ReLU → Conv → Sigmoid

- [ ] **Backbone integration** in `net.py`
  - Add occlusion head as module
  - Modify forward pass to return occlusion maps
  - Ensure feature maps are shared between embedding and occlusion head

- [ ] **Dataset support** in `data.py`
  - Load niqab dataset with ground truth masks
  - Downsample masks to 7x7 for training
  - Handle mask format (binary or continuous)

- [ ] **Loss computation** in `train_val.py`
  - MSE loss between predicted and ground truth occlusion maps
  - Shape alignment (interpolation if needed)
  - Add occlusion loss to multi-task learning

- [ ] **QAConv integration** in `qaconv.py`
  - Add `apply_occlusion_weight()` method
  - Modify `_compute_similarity_batch()` to accept occlusion maps
  - Weight correlation scores before pooling

- [ ] **Training loop updates** in `train_val.py`
  - Extract occlusion maps from forward pass
  - Compute occlusion loss when ground truth masks available
  - Combine losses: `total_loss = adaface_loss + qaconv_loss + occlusion_loss`

---

## Important Notes

1. **Spatial Resolution**: Occlusion maps are 7x7 (matching QAConv feature map resolution for 112x112 input)
2. **No Synthetic Data**: Training uses real niqab dataset with ground truth masks
3. **Multi-Task Learning**: Occlusion head shares features with recognition task
4. **Direct Integration**: Occlusion maps directly weight QAConv correlation scores
5. **Clean Implementation**: Keep components modular and well-documented

---

## Expected Outcome

After implementation:
- Model predicts occlusion maps for each input face
- QAConv matching focuses on visible regions (weighted by occlusion maps)
- Improved accuracy on occluded face datasets
- Clean, maintainable code following the specification

---

## Questions to Consider

1. Where in the backbone should the occlusion head branch off?
2. How should occlusion maps be integrated into QAConv's correlation computation?
3. What loss weight should be used for occlusion loss?
4. How to handle batches without ground truth masks?
5. How to visualize occlusion maps during training for debugging?

---

**Start by thoroughly studying the codebase, then create a detailed implementation plan before writing any code.**

