# Evaluation & Validation Components

## Overview
Provides utilities for evaluating face recognition performance on standard benchmarks using ROC curves, accuracy metrics, and verification protocols.

---

## 1. evaluate_utils.py - Core Evaluation Utilities

### Purpose
Implements evaluation protocols for face verification benchmarks.

### Key Functions

#### `evaluate(embeddings, actual_issame, nrof_folds=10, pca=0)`

**Purpose**: Evaluate face verification using 10-fold cross-validation.

```python
def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    """
    Args:
        embeddings: [N, D] array of face embeddings
        actual_issame: [N//2] array of boolean labels
        nrof_folds: Number of cross-validation folds
        pca: PCA dimensions (0 = no PCA)
    
    Returns:
        tpr: True positive rate per threshold
        fpr: False positive rate per threshold
        accuracy: Accuracy per fold
        best_thresholds: Best threshold per fold
    """
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]  # Even indices
    embeddings2 = embeddings[1::2]  # Odd indices
    
    tpr, fpr, accuracy, best_thresholds = calculate_roc(
        thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds, pca
    )
    
    return tpr, fpr, accuracy, best_thresholds
```

**Usage**:
```python
# Extract embeddings for all images in validation set
embeddings = extract_embeddings(model, val_loader)
issame_labels = load_pair_labels('lfw_pairs.txt')

# Evaluate
tpr, fpr, accuracy, thresholds = evaluate_utils.evaluate(
    embeddings, issame_labels, nrof_folds=10
)

print(f"Accuracy: {accuracy.mean():.4f} ± {accuracy.std():.4f}")
```

#### `calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds, pca)`

**Purpose**: Compute ROC curve with K-fold cross-validation.

**Process**:
```python
# 1. Compute distances
diff = np.subtract(embeddings1, embeddings2)
dist = np.sum(np.square(diff), axis=1)

# 2. K-fold split
k_fold = KFold(n_splits=nrof_folds, shuffle=False)

for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
    # 3. Find best threshold on train set
    for threshold in thresholds:
        _, _, acc = calculate_accuracy(threshold, dist[train_set], issame[train_set])
        acc_train[threshold_idx] = acc
    
    best_threshold = thresholds[np.argmax(acc_train)]
    
    # 4. Evaluate on test set
    for threshold in thresholds:
        tpr, fpr, _ = calculate_accuracy(threshold, dist[test_set], issame[test_set])
        tprs[fold_idx, threshold_idx] = tpr
        fprs[fold_idx, threshold_idx] = fpr
    
    # 5. Compute accuracy with best threshold
    _, _, accuracy[fold_idx] = calculate_accuracy(
        best_threshold, dist[test_set], issame[test_set]
    )

# 6. Average across folds
tpr = np.mean(tprs, axis=0)
fpr = np.mean(fprs, axis=0)

return tpr, fpr, accuracy, best_thresholds
```

#### `calculate_accuracy(threshold, dist, actual_issame)`

**Purpose**: Compute TPR, FPR, and accuracy for a given threshold.

```python
def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), 
                                np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
    
    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    
    return tpr, fpr, acc
```

#### `calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds)`

**Purpose**: Calculate verification rate at specific FAR (False Accept Rate).

```python
def calculate_val(thresholds, embeddings1, embeddings2, 
                  actual_issame, far_target=1e-3, nrof_folds=10):
    """
    Returns:
        val_mean: Mean verification rate at FAR target
        val_std: Standard deviation of verification rate
        far_mean: Actual FAR achieved
    """
    # For each fold, find threshold that gives FAR = far_target
    # Then compute verification rate on test set
```

**Common FAR Targets**:
- 1e-3 (0.1%): Standard benchmark
- 1e-4 (0.01%): Stricter security
- 1e-5 (0.001%): Very strict

### Memory-Mapped File Utilities

#### `get_val_data(data_path)`

**Purpose**: Load all validation datasets.

```python
def get_val_data(data_path):
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    cplfw, cplfw_issame = get_val_pair(data_path, 'cplfw')
    calfw, calfw_issame = get_val_pair(data_path, 'calfw')
    
    return (agedb_30, cfp_fp, lfw, agedb_30_issame, 
            cfp_fp_issame, lfw_issame, cplfw, cplfw_issame, 
            calfw, calfw_issame)
```

#### `get_val_pair(path, name, use_memfile=True)`

**Purpose**: Load single validation dataset.

```python
def get_val_pair(path, name, use_memfile=True):
    if use_memfile:
        mem_file = os.path.join(path, name, 'memfile', 'mem_file.dat')
        
        if os.path.exists(mem_file):
            np_array = read_memmap(mem_file)
        else:
            # Convert from bcolz to memfile
            carray = bcolz.carray(rootdir=os.path.join(path, name), mode='r')
            np_array = np.array(carray)
            make_memmap(mem_file, np_array)
            np_array = read_memmap(mem_file)
    else:
        np_array = bcolz.carray(rootdir=os.path.join(path, name), mode='r')
    
    issame = np.load(os.path.join(path, f'{name}_list.npy'))
    
    return np_array, issame
```

#### `make_memmap(mem_file_name, np_to_copy)`

**Purpose**: Create memory-mapped file from numpy array.

```python
def make_memmap(mem_file_name, np_to_copy):
    # Save config
    config = {
        'shape': tuple(np_to_copy.shape),
        'dtype': str(np_to_copy.dtype)
    }
    json.dump(config, open(mem_file_name + '.conf', 'w'))
    
    # Create memmap
    mm = np.memmap(mem_file_name, mode='w+', 
                  shape=config['shape'], dtype=config['dtype'])
    mm[:] = np_to_copy[:]
    mm.flush()
    
    return mm
```

#### `read_memmap(mem_file_name)`

**Purpose**: Read existing memory-mapped file.

```python
def read_memmap(mem_file_name):
    with open(mem_file_name + '.conf', 'r') as f:
        config = json.load(f)
    
    return np.memmap(mem_file_name, mode='r+',
                    shape=tuple(config['shape']),
                    dtype=config['dtype'])
```

**Benefits**:
- **Memory Efficient**: Only load required data
- **Fast Access**: No need to decompress
- **Shared Memory**: Multiple processes can access

---

## 2. Validation Datasets

### Standard Benchmarks

#### LFW (Labeled Faces in the Wild)
- **Images**: 13,233 faces, 5,749 identities
- **Pairs**: 6,000 (3,000 same, 3,000 different)
- **Protocol**: 10-fold cross-validation
- **Difficulty**: Unconstrained faces, variations in pose/lighting

#### AgeDB-30
- **Images**: 16,488 faces, 568 identities
- **Pairs**: 6,000 (3,000 same, 3,000 different)
- **Challenge**: Age variations (30+ years between pairs)
- **Protocol**: 10-fold cross-validation

#### CFP-FP (Celebrities in Frontal-Profile)
- **Images**: 7,000 faces, 500 identities
- **Pairs**: 7,000 (3,500 same, 3,500 different)
- **Challenge**: Frontal vs. profile poses
- **Protocol**: 10-fold cross-validation

#### CPLFW (Cross-Pose LFW)
- **Images**: 13,233 faces (LFW subset)
- **Pairs**: 6,000 with pose variations
- **Challenge**: Large pose variations
- **Protocol**: 10-fold cross-validation

#### CALFW (Cross-Age LFW)
- **Images**: 13,233 faces (LFW subset)
- **Pairs**: 6,000 with age variations
- **Challenge**: Age variations
- **Protocol**: 10-fold cross-validation

### Dataset Structure
```
validation_data/
  ├── lfw/
  │   ├── lfw.bin  (bcolz format)
  │   ├── memfile/
  │   │   ├── mem_file.dat
  │   │   └── mem_file.dat.conf
  │   └── ...
  ├── lfw_list.npy  (issame labels)
  ├── agedb_30/
  ├── cfp_fp/
  ├── cplfw/
  └── calfw/
```

---

## 3. validation_lq/ - Low-Quality Benchmarks

### IJB-S (IARPA Janus Benchmark - Surveillance)
- **Images**: Surveillance video frames
- **Protocols**: 1:1 verification, 1:N identification
- **Challenge**: Low resolution, motion blur, occlusions

### TinyFace
- **Images**: Low-resolution faces
- **Challenge**: Very small face sizes (< 20×20 pixels)

### Scripts
- `validate_IJB_S.py`: Evaluation on IJB-S
- `validate_tinyface.py`: Evaluation on TinyFace
- `tinyface_helper.py`: Helper functions for TinyFace

---

## 4. validation_mixed/ - Mixed-Quality Benchmarks

### IJB-B & IJB-C
- **Images**: Mix of still images and video frames
- **Protocols**: 1:1 verification, 1:N identification
- **Templates**: Multiple images per identity
- **Challenge**: Unconstrained, large-scale

### Scripts
- `validate_IJB_BC.py`: Evaluation on IJB-B and IJB-C
- `insightface_ijb_helper/`: Helper modules
  - `dataloader.py`: Load IJB datasets
  - `eval_helper.py`: Verification evaluation
  - `eval_helper_identification.py`: Identification evaluation
  - `infer_helper.py`: Feature extraction

---

## 5. validation_hq/ - High-Quality Benchmarks

### 5 Validation Datasets
Uses the standard benchmarks (LFW, AgeDB-30, CFP-FP, CPLFW, CALFW) with high-quality images.

### Script
- `eval_5valsets.sh`: Batch evaluation script

---

## Usage Examples

### Evaluate on Standard Benchmarks
```python
import evaluate_utils
from dataset.five_validation_dataset import FiveValidationDataset

# Load validation data
val_data = evaluate_utils.get_val_data('data/faces_emore')
val_data_dict = {
    'agedb_30': (val_data[0], val_data[3]),
    'cfp_fp': (val_data[1], val_data[4]),
    'lfw': (val_data[2], val_data[5]),
    'cplfw': (val_data[6], val_data[7]),
    'calfw': (val_data[8], val_data[9]),
}

# Extract embeddings
model.eval()
embeddings = {}

for name, (imgs, issame) in val_data_dict.items():
    embs = []
    for i in range(0, len(imgs), batch_size):
        batch = torch.tensor(imgs[i:i+batch_size])
        with torch.no_grad():
            emb, _, _ = model(batch.cuda())
        embs.append(emb.cpu().numpy())
    
    embeddings[name] = np.concatenate(embs)

# Evaluate each dataset
for name in val_data_dict.keys():
    embs = embeddings[name]
    issame = val_data_dict[name][1]
    
    tpr, fpr, accuracy, _ = evaluate_utils.evaluate(embs, issame)
    
    print(f"{name}: {accuracy.mean():.4f} ± {accuracy.std():.4f}")
```

### Compute ROC Curve
```python
import matplotlib.pyplot as plt

tpr, fpr, accuracy, _ = evaluate_utils.evaluate(embeddings, issame)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('False Accept Rate')
plt.ylabel('True Accept Rate')
plt.title(f'ROC Curve (Accuracy: {accuracy.mean():.4f})')
plt.xscale('log')
plt.grid(True)
plt.legend()
plt.savefig('roc_curve.png')
```

### Verification at Specific FAR
```python
# Compute verification rate at FAR = 1e-3
embeddings1 = embeddings[0::2]
embeddings2 = embeddings[1::2]

val_mean, val_std, far_mean = evaluate_utils.calculate_val(
    np.arange(0, 4, 0.01),
    embeddings1,
    embeddings2,
    issame,
    far_target=1e-3,
    nrof_folds=10
)

print(f"VR@FAR=1e-3: {val_mean:.4f} ± {val_std:.4f}")
print(f"Actual FAR: {far_mean:.6f}")
```

---

## Performance Metrics

### Standard Metrics
1. **Accuracy**: Optimal threshold determined on each fold
2. **TPR/FPR**: For plotting ROC curves
3. **VR@FAR**: Verification rate at specific false accept rates
4. **EER**: Equal error rate (TPR = 1 - FPR)

### Computing EER
```python
# Find point where TPR = 1 - FPR
eer_idx = np.nanargmin(np.abs(tpr - (1 - fpr)))
eer = fpr[eer_idx]
print(f"EER: {eer:.4f}")
```

---

## Best Practices

### 1. Use Memfiles
- Convert validation data to memfiles once
- Saves memory and speeds up loading
- Essential for large-scale validation

### 2. Batch Processing
- Extract embeddings in batches
- Use GPU for feature extraction
- Save extracted features to avoid recomputation

### 3. Multiple Metrics
- Report accuracy, VR@FAR, and EER
- Compare across different thresholds
- Use ROC curves for visualization

### 4. Cross-Validation
- Always use 10-fold CV for consistency
- Report mean and standard deviation
- Don't cherry-pick folds

---

## Common Issues

### Issue 1: Memory Error
**Solution**: Use memfiles and smaller batch sizes
```python
batch_size = 64  # Reduce if needed
use_memfile = True
```

### Issue 2: Slow Evaluation
**Solution**: Extract and cache features
```python
# Extract once
features = extract_features(model, dataset)
np.save('cached_features.npy', features)

# Load for evaluation
features = np.load('cached_features.npy')
```

### Issue 3: Inconsistent Results
**Cause**: Different random seeds in K-fold
**Solution**: Use `shuffle=False` in KFold
```python
k_fold = KFold(n_splits=10, shuffle=False)
```

---

## Integration with Training

### In train_val.py
```python
def on_validation_epoch_end(self):
    # Gather embeddings from all validation steps
    all_embeddings = gather_outputs(self.validation_step_outputs)
    
    # Evaluate on each dataset
    for dataset_name in ['agedb_30', 'cfp_fp', 'lfw', 'cplfw', 'calfw']:
        mask = dataname == dataset_name
        embs = all_embeddings[mask]
        issame = labels[mask]
        
        tpr, fpr, acc, _ = evaluate_utils.evaluate(embs, issame)
        
        self.log(f'{dataset_name}_acc', acc.mean())
    
    # Log average accuracy
    avg_acc = np.mean([acc for acc in all_accs])
    self.log('val_acc', avg_acc)  # For checkpoint monitoring
```

