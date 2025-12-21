# Core Training & Evaluation Framework

## Overview
The core training and evaluation framework provides the foundation for training face recognition models using PyTorch Lightning. It implements an occlusion-aware metric learning approach that combines multiple loss functions and evaluation strategies.

## Components

### 1. main.py - Training Entry Point

**Purpose**: Initializes and orchestrates the training pipeline using PyTorch Lightning.

**Key Functions**:
- `main(args)`: Sets up trainer, data module, and callbacks
  - Creates PyTorch Lightning `Trainer` with DDP strategy support
  - Configures model checkpointing based on validation accuracy
  - Handles both training and evaluation modes
  - Supports mixed precision (FP16) training

**Architecture**:
```python
main(args)
  ├── Setup hparams (config)
  ├── Initialize train_val.Trainer (Lightning Module)
  ├── Initialize data.DataModule (Lightning Data Module)
  ├── Setup ModelCheckpoint callback
  ├── Create CSV logger
  ├── Configure PyTorch Lightning Trainer
  │   ├── DDP/DP strategy
  │   ├── Mixed precision
  │   └── Gradient accumulation
  └── Execute training/evaluation
```

**Key Configuration**:
- **Distributed Backend**: DDP with find_unused_parameters_true for complex architectures
- **Checkpointing**: Monitors `val_acc`, saves best model and last checkpoint
- **Batch Size Adjustment**: Automatically divides batch size across GPUs in DDP mode
- **Validation**: Runs at specified intervals (1.0 for long training, 0.1 for short)

**Usage Example**:
```bash
python main.py \
  --data_root /path/to/data \
  --train_data_path faces_emore \
  --val_data_path faces_emore \
  --arch ir_50 \
  --head adaface \
  --batch_size 256 \
  --epochs 24 \
  --gpus 4
```

---

### 2. config.py - Configuration Management

**Purpose**: Centralizes all hyperparameters and command-line argument parsing.

**Key Functions**:
- `get_args()`: Parses and validates command-line arguments
- `add_task_arguments()`: Adds task-specific parameters

**Configuration Categories**:

#### Data Parameters
- `data_root`: Root directory for datasets
- `train_data_path`: Path to training data
- `val_data_path`: Path to validation data
- `use_mxrecord`: Use MXNet record format
- `train_data_subset`: Use subset of MS1MV2 for ablation studies

#### Training Parameters
- `epochs`: Number of training epochs (default: 24)
- `batch_size`: Total batch size across all GPUs (default: 256)
- `lr`: Learning rate (default: 0.1)
- `lr_milestones`: Epochs for LR reduction (default: "8,12,14")
- `lr_gamma`: LR multiplication factor (default: 0.1)
- `momentum`: SGD momentum (default: 0.9)
- `weight_decay`: L2 regularization (default: 1e-4)

#### Model Architecture
- `arch`: Backbone architecture (ir_18, ir_34, ir_50, ir_101)
- `head`: Loss head type (adaface, arcface, cosface)
- `m`: Margin parameter for loss head (default: 0.5)
- `h`: AdaFace h parameter (default: 0.0)
- `s`: Scale parameter (default: 64.0)
- `t_alpha`: AdaFace EMA alpha (default: 0.01)

#### QAConv Parameters
- `k_nearest`: Number of nearest neighbors for class graph (default: 32)

#### Augmentation
- `low_res_augmentation_prob`: Probability of low-resolution augmentation (default: 0.0)
- `crop_augmentation_prob`: Probability of crop augmentation (default: 0.0)
- `photometric_augmentation_prob`: Probability of photometric augmentation (default: 0.0)

#### System Parameters
- `gpus`: Number of GPUs to use
- `distributed_backend`: DDP, DP, or DDP2
- `use_16bit`: Enable mixed precision training
- `num_workers`: Data loading workers (default: 16)
- `accumulate_grad_batches`: Gradient accumulation steps (default: 1)

**Output Directory Naming**:
Automatically generates unique experiment directories based on prefix and timestamp:
```
experiments/{prefix}_{month}-{day}_{counter}/
```

---

### 3. train_val.py - Training Logic (PyTorch Lightning Module)

**Purpose**: Implements the core training loop, validation, and loss computation using PyTorch Lightning.

**Class: Trainer(LightningModule)**

**Architecture Overview**:
```
Trainer
  ├── Model Components
  │   ├── CNN Backbone (net.build_model)
  │   ├── Loss Head (head.build_head)
  │   └── QAConv Matcher
  ├── Loss Functions
  │   ├── AdaFace Cross-Entropy
  │   ├── QAConv Pairwise Matching
  │   ├── QAConv Triplet Loss
  │   └── Occlusion Supervision (MSE)
  └── Evaluation
      ├── AdaFace embeddings
      ├── QAConv matching scores
      └── Combined scores
```

**Key Attributes**:
- **Loss Weights**:
  - `qaconv_loss_weight = 0.7`: Weight for QAConv loss
  - `adaface_loss_weight = 0.1`: Weight for AdaFace loss
  - `occlusion_loss_weight = 0.3`: Weight for occlusion supervision
  
- **Evaluation Weights**:
  - `adaface_eval_weight = 0.5`: Weight for AdaFace scores
  - `qaconv_eval_weight = 0.5`: Weight for QAConv scores

- **Occlusion Method**: "scaling" for occlusion-aware weighting

**Key Methods**:

#### 1. `__init__(**kwargs)`
Initializes all model components:
- Builds CNN backbone using `net.build_model()`
- Creates loss head (AdaFace/ArcFace/CosFace)
- Initializes QAConv with class embeddings
- Sets up loss functions (PairwiseMatchingLoss, SoftmaxTripletLoss)
- Initializes Weights & Biases logging

#### 2. `training_step(batch, batch_idx)`
Main training loop for each batch:

**Process Flow**:
```
1. Unpack batch → (images, labels, [optional_masks])
2. Extract feature maps → CNN backbone
3. Generate AdaFace embeddings → output_layer
4. Compute AdaFace loss → cross_entropy_loss
5. Compute QAConv losses:
   a. Pairwise matching loss
   b. Triplet loss (classification + ranking)
6. Compute occlusion supervision loss (if masks available)
7. Combine weighted losses
8. Log metrics to PyTorch Lightning & Weights & Biases
```

**Loss Combination**:
```python
total_loss = (adaface_weight * adaface_loss + 
              qaconv_weight * (pairwise_loss + triplet_weight * triplet_loss) +
              occlusion_weight * occlusion_loss)
```

**NaN Handling**:
- Checks for NaN values at multiple stages
- Replaces NaN with zeros and logs warnings
- Adds epsilon values to prevent division by zero
- Verifies feature map normalization

#### 3. `validation_step(batch, batch_idx)`
Evaluates model on validation data:

**Features**:
- Extracts both AdaFace embeddings and QAConv feature maps
- Uses test-time augmentation (horizontal flip)
- Fuses original and flipped features using norm-weighted averaging
- Returns embeddings for 5 validation datasets: agedb_30, cfp_fp, lfw, cplfw, calfw

#### 4. `on_validation_epoch_end()`
Computes validation metrics across all datasets:

**Evaluation Strategy**:
1. **AdaFace Evaluation**: Standard L2 distance-based verification
2. **QAConv Evaluation**: Direct pairwise matching with positive/negative pairs
3. **Combined Evaluation**: Weighted combination of normalized distances

**QAConv Matching**:
- Structures data as gallery-query pairs (even/odd indices)
- Computes matching scores for positive pairs (same identity)
- Samples negative pairs (different identities) for comparison
- Calculates direct accuracy based on score separation

**Dynamic Weight Adjustment**:
```python
if qaconv_acc < 0.5:
    adaface_weight = 0.8
    qaconv_weight = 0.2
else:
    # Use configured weights
    adaface_weight = 0.5
    qaconv_weight = 0.5
```

**Monitored Metric**: `val_acc = val_combined_acc` for model checkpointing

#### 5. `test_step()` and `on_test_epoch_end()`
Mirror validation logic for final model testing.

#### 6. `configure_optimizers()`
Sets up optimizer and learning rate scheduler:

**Optimizer Configuration**:
```python
SGD(
    params=[
        {params: backbone + QAConv + occlusion_head + kernel, 
         weight_decay: 5e-4},
        {params: batch_norm_params, 
         weight_decay: 0}
    ],
    lr=hparams.lr,
    momentum=hparams.momentum
)
```

**LR Scheduler**: MultiStepLR with milestones at epochs [8, 12, 14] and gamma=0.1

#### 7. `split_parameters(module)`
Separates parameters into weight decay and no-decay groups:
- **Weight Decay**: Conv weights, Linear weights, QAConv class embeddings
- **No Weight Decay**: Batch normalization parameters

**Logging and Monitoring**:
- PyTorch Lightning metrics: loss components, accuracies, learning rate
- Weights & Biases: comprehensive metrics including QAConv-specific statistics
- Occlusion head norm tracking every 1000 steps
- Sample image saving in first epoch for debugging

**Debug Features**:
- NaN detection at multiple stages
- Feature map norm verification
- Loss component validation
- Score distribution analysis for QAConv

---

### 4. data.py - Data Module (PyTorch Lightning)

**Purpose**: Manages data loading, preprocessing, and augmentation for training and validation.

**Class: DataModule(pl.LightningDataModule)**

**Key Methods**:

#### 1. `__init__(**kwargs)`
Stores configuration parameters:
- Data paths and batch size
- Augmentation probabilities
- MXRecord usage flag

#### 2. `prepare_data()`
One-time data preparation:
- Converts validation datasets to memory-mapped files for efficiency
- Creates concatenated memfile for all 5 validation sets

#### 3. `setup(stage)`
Creates dataset instances:
- **Training**: `train_dataset()` with augmentation
- **Validation/Test**: `val_dataset()` and `test_dataset()` with memfiles

#### 4. `train_dataloader()`
**PK Sampling Strategy**:
```python
K = 4  # instances per identity
P = batch_size // K  # identities per batch
sampler = RandomIdentitySampler(dataset, num_instances=K)
```
This ensures each batch contains multiple samples from the same identities, crucial for metric learning.

#### 5. `val_dataloader()` and `test_dataloader()`
Standard sequential dataloaders for validation/test sets.

**Training Data Transform Pipeline**:
```python
1. Random Horizontal Flip (p=0.5)
2. Medical Mask Occlusion (p=0.5)
   ├── Randomize facial landmarks
   ├── Apply realistic mask with proper positioning
   └── Generate binary occlusion mask
3. ToTensorWithMask
   ├── Convert image to tensor
   └── Convert mask to tensor [1, H, W]
4. Normalize image (mean=0.5, std=0.5)
```

**Dataset Support**:
- **Image Folder**: `CustomImageFolderDataset` for standard folder structure
- **MXRecord**: `AugmentRecordDataset` for MXNet binary records
- **Validation**: `FiveValidationDataset` for concatenated validation sets

**Data Augmentation**:
Uses `Augmenter` class for additional augmentations:
- Low-resolution simulation
- Random crops
- Photometric distortions

**MS1MV2 Subsetting**:
For ablation studies, supports subsetting MS1MV2 dataset:
- Uses predefined subset indices from `assets/ms1mv2_train_subset_index.txt`
- Removes identities with < 5 samples
- Adjusts label indices to be contiguous

---

### 5. main_eval.py - Standalone Evaluation Script

**Purpose**: Provides a standalone evaluation script for testing trained models on custom datasets.

**Key Functions**:

#### 1. `load_pretrained_model(model_file, architecture, k_nearest)`
Loads a pretrained checkpoint:
- Infers number of classes from checkpoint
- Rebuilds model architecture
- Reinitializes QAConv with correct parameters
- Loads weights with proper prefix handling

#### 2. `main()`
Evaluation pipeline:

**Process**:
```
1. Load pretrained model
2. Create evaluation dataset (EvaluationDataset)
3. Extract features in batches:
   a. AdaFace embeddings
   b. QAConv feature maps
4. Compute similarity matrices:
   a. AdaFace: cosine similarity
   b. QAConv: pairwise matching
   c. Combined: weighted average
5. Evaluate verification performance:
   a. ROC curves
   b. VR@FAR metrics (1e-3, 1e-4)
   c. Score distributions
6. Generate visualizations and reports
```

**Command-Line Arguments**:
- `--model-path`: Path to checkpoint
- `--data-path`: Dataset directory
- `--output-dir`: Results directory
- `--batch-size`: Feature extraction batch size
- `--adaface-weight`: Weight for AdaFace scores (default: 0.5)
- `--qaconv-weight`: Weight for QAConv scores (default: 0.5)

**Evaluation Outputs**:
- Verification ROC curves for each method
- Score distribution histograms
- VR@FAR tables
- Detailed results in text format

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         main.py                              │
│  ┌────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  config.py │───▶│ train_val.py │◀──▶│   data.py    │   │
│  └────────────┘    │  (Trainer)   │    │ (DataModule) │   │
│                     └──────────────┘    └──────────────┘   │
│                            │                     │           │
│                            │                     │           │
└────────────────────────────┼─────────────────────┼──────────┘
                             │                     │
                    ┌────────▼─────────┐  ┌───────▼────────┐
                    │   Model (net.py) │  │ Datasets       │
                    │   Head (head.py) │  │ Transforms     │
                    │   QAConv         │  │ Samplers       │
                    └──────────────────┘  └────────────────┘
```

---

## Dependencies

**External Modules**:
- PyTorch Lightning (training framework)
- Weights & Biases (experiment tracking)
- PyTorch (deep learning)
- NumPy (numerical operations)

**Internal Modules**:
- `net.py`: Network architectures
- `head.py`: Loss heads
- `qaconv.py`: QAConv matcher
- `evaluate_utils.py`: Validation utilities
- `utils.py`: Helper functions
- `pairwise_matching_loss.py`: QAConv pairwise loss
- `softmax_triplet_loss.py`: QAConv triplet loss
- Dataset modules: `image_folder_dataset`, `record_dataset`, `five_validation_dataset`
- Transform modules: `transforms`, `sampler`

---

## Configuration Notes

### Training Hyperparameters
The default configuration is optimized for MS1MV2 dataset:
- **Learning Rate**: 0.1 with MultiStepLR schedule
- **Batch Size**: 256 (automatically divided across GPUs)
- **Epochs**: 24 with LR drops at [8, 12, 14]
- **Optimizer**: SGD with momentum=0.9, weight_decay=5e-4

### Loss Weights
Carefully tuned for balanced training:
- **AdaFace**: 0.1 (primary face recognition objective)
- **QAConv**: 0.7 (pairwise + triplet, for local feature matching)
- **Occlusion**: 0.3 (supervision signal for occlusion prediction)

Total effective weight ≈ 1.1, slightly favoring QAConv for robust occlusion handling.

### Validation Strategy
- **5 Benchmarks**: agedb_30, cfp_fp, lfw, cplfw, calfw
- **Metrics**: Accuracy at optimal threshold (10-fold cross-validation)
- **Combined Score**: Weighted average of AdaFace + QAConv
- **Checkpoint**: Saves best model based on combined validation accuracy

---

## Design Decisions

### 1. PyTorch Lightning Framework
**Rationale**: Simplifies distributed training, checkpointing, and logging while maintaining flexibility.

### 2. Dual-Branch Evaluation
**Rationale**: Combines global (AdaFace) and local (QAConv) features for robust performance under occlusion.

### 3. PK Sampling
**Rationale**: Essential for metric learning losses (pairwise, triplet) to ensure meaningful comparisons within batches.

### 4. Occlusion Supervision
**Rationale**: Provides explicit supervision for learning occlusion patterns, improving interpretability and performance.

### 5. Dynamic Weight Adjustment
**Rationale**: Prevents poor QAConv performance from degrading overall results during early training.

---

## Usage Examples

### Basic Training
```bash
python main.py \
  --data_root /data \
  --train_data_path faces_emore \
  --val_data_path faces_emore \
  --arch ir_50 \
  --batch_size 256 \
  --epochs 24 \
  --gpus 4
```

### Training with Subset (Ablation)
```bash
python main.py \
  --data_root /data \
  --train_data_path faces_emore \
  --val_data_path faces_emore \
  --train_data_subset \
  --arch ir_50 \
  --batch_size 128 \
  --epochs 24 \
  --gpus 2
```

### Evaluation Only
```bash
python main.py \
  --data_root /data \
  --val_data_path faces_emore \
  --evaluate \
  --start_from_model_statedict experiments/model/best.ckpt \
  --arch ir_50
```

### Resume Training
```bash
python main.py \
  --resume_from_checkpoint experiments/model/last.ckpt \
  --data_root /data \
  --train_data_path faces_emore \
  --val_data_path faces_emore
```

### Custom Evaluation
```bash
python main_eval.py \
  --model-path experiments/model/best.ckpt \
  --data-path /test/dataset \
  --output-dir results \
  --batch-size 64 \
  --adaface-weight 0.5 \
  --qaconv-weight 0.5
```

---

## Notes

### Performance Considerations
- **Mixed Precision**: Use `--use_16bit` for faster training with minimal accuracy loss
- **Gradient Accumulation**: Use `--accumulate_grad_batches N` for larger effective batch sizes
- **Memory Optimization**: Memfiles reduce validation memory footprint significantly

### Common Issues
1. **NaN Losses**: Check learning rate (may be too high), ensure proper normalization
2. **QAConv Accuracy Low**: Normal in early epochs, should improve after LR drop
3. **OOM Errors**: Reduce batch size or enable gradient accumulation

### Best Practices
- Always use DDP for multi-GPU training (more efficient than DP)
- Monitor both individual and combined metrics
- Save sample images from first epoch to verify augmentation
- Use Weights & Biases for comprehensive experiment tracking

