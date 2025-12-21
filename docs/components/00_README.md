# OAML Component Documentation

## Overview
This directory contains comprehensive documentation for all components of the OAML (Occlusion-Aware Metric Learning) face recognition framework.

## Contents

### [01_core_training.md](01_core_training.md)
**Core Training & Evaluation Framework**
- `main.py` - Training entry point with PyTorch Lightning
- `main_eval.py` - Standalone evaluation script
- `config.py` - Configuration management
- `train_val.py` - Training logic (PyTorch Lightning Module)
- `data.py` - Data module for loading and preprocessing

**Key Topics**:
- Training pipeline architecture
- Loss combination strategies
- Validation protocols
- Multi-GPU training (DDP)
- Configuration management

---

### [02_network_architecture.md](02_network_architecture.md)
**Network Architecture Components**
- `net.py` - CNN backbones (IR-18, IR-50, IR-101) with QAConv and occlusion head
- `head.py` - Loss heads (AdaFace, ArcFace, CosFace)
- `qaconv.py` - Query-Adaptive Convolution for local feature matching
- `face.py` - Dataset wrapper for face recognition

**Key Topics**:
- Dual-branch architecture (global + local)
- Residual network architectures
- Occlusion prediction head
- QAConv correlation and matching
- Class embeddings and graph sampling

---

### [03_dataset_processing.md](03_dataset_processing.md)
**Dataset & Data Processing**
- `transforms.py` - Data transforms including medical mask occlusion
- `sampler.py` - PK sampling for metric learning
- `dataset/` - Dataset implementations
  - `image_folder_dataset.py` - Image folder loading
  - `record_dataset.py` - MXNet record format
  - `five_validation_dataset.py` - Validation benchmarks
  - `augmenter.py` - Quality-degrading augmentations

**Key Topics**:
- Medical mask occlusion simulation
- Synthetic occlusion generation
- PK sampling strategy
- Data augmentation pipeline
- Memory-mapped validation data

---

### [04_loss_functions.md](04_loss_functions.md)
**Loss Function Implementations**
- `pairwise_matching_loss.py` - Binary classification loss for QAConv
- `softmax_triplet_loss.py` - Combined classification and triplet loss

**Key Topics**:
- Pairwise matching loss
- Hard triplet mining
- Loss combination strategies
- Metric learning objectives
- Training tips and best practices

---

### [05_face_alignment.md](05_face_alignment.md)
**Face Alignment Components**
- `face_alignment/align.py` - Simple alignment interface
- `face_alignment/mtcnn.py` - MTCNN implementation
- `face_alignment/mtcnn_pytorch/` - MTCNN network implementations

**Key Topics**:
- Multi-stage face detection (PNet, RNet, ONet)
- Facial landmark detection
- Affine transformation and alignment
- Configuration for challenging conditions
- Batch processing strategies

---

### [06_evaluation_validation.md](06_evaluation_validation.md)
**Evaluation & Validation**
- `evaluate_utils.py` - Core evaluation utilities
- `validation_hq/` - High-quality benchmarks (LFW, AgeDB-30, etc.)
- `validation_lq/` - Low-quality benchmarks (IJB-S, TinyFace)
- `validation_mixed/` - Mixed-quality benchmarks (IJB-B, IJB-C)

**Key Topics**:
- ROC curve computation
- 10-fold cross-validation
- VR@FAR metrics
- Memory-mapped file utilities
- Standard face recognition benchmarks

---

### [07_utilities_inference.md](07_utilities_inference.md)
**Utilities & Inference**
- `utils.py` - General utilities (normalization, distributed training)
- `inference.py` - Model loading and feature extraction
- `graph_sampler.py` - Graph-based sampling for metric learning

**Key Topics**:
- L2 normalization and feature fusion
- Distributed training utilities (DDP)
- Dataset configuration helpers
- Model inference pipeline
- Graph sampling strategy

---

### [08_data_processing_scripts.md](08_data_processing_scripts.md)
**Data Processing Scripts**
- `align_niqab_*.py` - Various alignment strategies for niqab dataset
- `analyze_occlusion_*.py` - Occlusion analysis tools
- `filter_*.py` - Data filtering scripts
- `review_*.py` - Interactive review tools
- `cleanup_*.py` - Cleanup utilities
- `diagnose_training_issue.py` - Training diagnostics

**Key Topics**:
- Robust alignment for challenging faces
- Quality filtering
- Interactive data review
- Occlusion effectiveness analysis
- Data pipeline recommendations

---

## Quick Start

### Understanding the Framework

1. **Start with Core Training** ([01_core_training.md](01_core_training.md))
   - Understand the training pipeline
   - Learn configuration options
   - See data flow diagrams

2. **Explore Network Architecture** ([02_network_architecture.md](02_network_architecture.md))
   - Understand dual-branch design
   - Learn QAConv mechanism
   - Review loss head implementations

3. **Study Data Processing** ([03_dataset_processing.md](03_dataset_processing.md))
   - Learn PK sampling
   - Understand occlusion simulation
   - Review transform pipeline

4. **Review Loss Functions** ([04_loss_functions.md](04_loss_functions.md))
   - Understand metric learning losses
   - Learn loss combination
   - See training tips

### For Common Tasks

#### Training a Model
See: [01_core_training.md](01_core_training.md) → Usage Examples

#### Evaluating Performance
See: [06_evaluation_validation.md](06_evaluation_validation.md) → Usage Examples

#### Processing New Data
See: [08_data_processing_scripts.md](08_data_processing_scripts.md) → Data Processing Pipeline

#### Running Inference
See: [07_utilities_inference.md](07_utilities_inference.md) → inference.py section

#### Aligning Faces
See: [05_face_alignment.md](05_face_alignment.md) → MTCNN Usage

---

## Architecture Overview

```
OAML Framework Architecture

Input: Face Image [3, 112, 112]
    │
    ├──> Backbone (net.py)
    │    ├──> IR-18/50/101 ResNet
    │    ├──> Feature Maps [512, 7, 7]
    │    ├──> Global Embedding [512]
    │    └──> Occlusion Map [1, 7, 7]
    │
    ├──> Loss Heads
    │    ├──> AdaFace (head.py)
    │    │    └──> Classification Loss
    │    │
    │    └──> QAConv (qaconv.py)
    │         ├──> Pairwise Matching Loss
    │         └──> Triplet Loss
    │
    └──> Evaluation
         ├──> AdaFace Matching (L2 distance)
         ├──> QAConv Matching (correlation)
         └──> Combined Score
```

---

## Key Concepts

### Dual-Branch Architecture
- **Global Branch**: AdaFace embeddings for overall face representation
- **Local Branch**: QAConv feature maps for occlusion-robust matching
- **Combination**: Weighted fusion of both for optimal performance

### Occlusion Handling
1. **Simulation**: Medical mask occlusion during training
2. **Prediction**: Occlusion confidence map from network
3. **Weighting**: Occlusion-aware matching in QAConv
4. **Supervision**: MSE loss between predicted and ground truth masks

### PK Sampling
- **P**: Number of identities per batch
- **K**: Number of instances per identity
- **Benefit**: Creates informative batches for metric learning
- **Example**: P=64, K=4 → batch size = 256

### QAConv Matching
1. Compute spatial correlation between query and gallery
2. Apply occlusion weighting (optional)
3. Max pooling over spatial dimensions
4. Linear transformation to similarity score

---

## Dependencies

### Core Dependencies
- PyTorch (≤1.13.1)
- PyTorch Lightning (1.8.6)
- Weights & Biases (experiment tracking)

### Data Processing
- PIL/Pillow (image handling)
- OpenCV (image processing)
- NumPy (numerical operations)
- bcolz (efficient data storage)

### Evaluation
- scikit-learn (KFold, metrics)
- matplotlib (visualization)
- scipy (interpolation)

---

## Performance Benchmarks

### Training Speed
- **IR-50, Batch 256, 4×V100**: ~2 hours/epoch on MS1MV2
- **Mixed Precision (FP16)**: ~1.5× speedup
- **DDP vs DP**: ~3× faster with 4 GPUs

### Accuracy (MS1MV2 trained)
- **LFW**: ~99.8%
- **AgeDB-30**: ~98.1%
- **CFP-FP**: ~98.7%
- **CPLFW**: ~92.4%
- **CALFW**: ~96.2%

### Inference Speed
- **Single Image (GPU)**: ~5ms
- **Batch 128 (GPU)**: ~80ms (0.6ms/image)
- **Feature Dimension**: 512
- **Model Size**: ~100MB (IR-50)

---

## Troubleshooting

### Common Issues

1. **NaN Losses**
   - See: [01_core_training.md](01_core_training.md) → Common Issues
   - Check: Feature normalization, learning rate

2. **Low QAConv Accuracy**
   - See: [04_loss_functions.md](04_loss_functions.md) → Training Tips
   - Normal in early epochs, improves after LR drop

3. **Face Detection Failures**
   - See: [05_face_alignment.md](05_face_alignment.md) → Common Issues
   - Solution: Adjust MTCNN thresholds, use manual fallback

4. **OOM Errors**
   - See: [03_dataset_processing.md](03_dataset_processing.md) → Performance
   - Solution: Reduce batch size, use gradient accumulation

5. **Slow Data Loading**
   - See: [03_dataset_processing.md](03_dataset_processing.md) → Speed
   - Solution: Increase num_workers, use memfiles

---

## Contributing

When adding new components:
1. Follow existing code structure
2. Document key functions and classes
3. Add usage examples
4. Update relevant component documentation
5. Include performance considerations

---

## References

### Paper
- AdaFace: [https://github.com/mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace)
- QAConv: [https://github.com/scliao/QAConv](https://github.com/scliao/QAConv)

### Benchmarks
- LFW: [http://vis-www.cs.umass.edu/lfw/](http://vis-www.cs.umass.edu/lfw/)
- AgeDB: [https://ibug.doc.ic.ac.uk/resources/agedb/](https://ibug.doc.ic.ac.uk/resources/agedb/)
- IJB-B/C: [https://www.nist.gov/programs-projects/face-challenges](https://www.nist.gov/programs-projects/face-challenges)

---

## Document Navigation

Each component document includes:
- **Overview**: Purpose and scope
- **Architecture**: Structure and design
- **Key Methods**: Important functions/classes
- **Usage Examples**: Practical code examples
- **Configuration**: Parameters and options
- **Performance**: Speed and accuracy considerations
- **Troubleshooting**: Common issues and solutions

For detailed information on any component, see the respective markdown file.

