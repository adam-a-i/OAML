# Data Processing Scripts

## Overview
Collection of scripts for preprocessing, filtering, and analyzing face datasets, particularly for handling niqab images and occlusion scenarios.

---

## 1. Face Alignment Scripts (align_niqab_*.py)

### Purpose
Multiple variants of alignment scripts for niqab dataset with different strategies.

### align_niqab_robust.py - Robust Alignment

**Purpose**: Most robust alignment with multiple fallback strategies.

**Features**:
- Lenient MTCNN thresholds for niqab detection
- Manual cropping fallback when detection fails
- Progress tracking and failure logging
- Automatic retry mechanisms

**Key Configuration**:
```python
mtcnn_model.thresholds = [0.3, 0.4, 0.5]  # Very lenient
mtcnn_model.min_face_size = 8  # Very small minimum
mtcnn_model.nms_thresholds = [0.5, 0.5, 0.5]  # More lenient NMS
```

**Manual Crop Function**:
```python
def manual_crop_niqab(img, crop_ratio=0.3):
    """
    Manually crop niqab images when detection fails
    Assumes face is in center-upper portion
    """
    width, height = img.size
    
    crop_width = int(width * crop_ratio)
    crop_height = int(height * crop_ratio)
    
    # Center horizontally, upper portion vertically
    left = (width - crop_width) // 2
    top = int(height * 0.1)  # Start 10% from top
    
    cropped = img.crop((left, top, left + crop_width, top + crop_height))
    resized = cropped.resize((112, 112), Image.LANCZOS)
    
    return resized
```

**Usage**:
```bash
python align_niqab_robust.py \
    --input-dir raw_niqab_images \
    --output-dir aligned_niqab \
    --device cuda:0 \
    --crop-size 112 112
```

### Other Variants

#### align_niqab_adjusted_crop.py
- Custom crop ratios for niqab images
- Adjustable vertical/horizontal offsets

#### align_niqab_wide_crop.py
- Wider crop windows for better context
- Suitable when eye region is primary feature

#### align_niqab_improved.py
- Enhanced detection parameters
- Better landmark estimation

#### align_niqab_no_manual.py
- Pure MTCNN without fallback
- For high-quality images only

#### align_niqab_dataset.py
- Batch processing for entire datasets
- Maintains directory structure
- Parallel processing support

---

## 2. Occlusion Analysis Scripts

### analyze_occlusion_effectiveness.py

**Purpose**: Comprehensive analysis of occlusion prediction effectiveness.

**Features**:
- Compares predicted vs. ground truth occlusion masks
- Computes IoU, precision, recall metrics
- Visualizes occlusion maps
- Analyzes model performance under various occlusions

**Key Metrics**:
```python
# Intersection over Union
iou = intersection / union

# Pixel-wise accuracy
accuracy = (tp + tn) / total_pixels

# Precision/Recall
precision = tp / (tp + fp)
recall = tp / (tp + fn)
```

**Usage**:
```bash
python analyze_occlusion_effectiveness.py \
    --model-path trained_model.ckpt \
    --test-images test_set/ \
    --output-dir analysis_results/
```

### analyze_occlusion_simple.py

**Purpose**: Simplified occlusion analysis for quick checks.

**Features**:
- Basic occlusion detection
- Simple visualization
- Faster execution
- Good for debugging

---

## 3. Data Filtering Scripts

### Purpose
Filter and clean datasets based on quality criteria.

### filter_bad_alignments.py

**Purpose**: Remove poorly aligned faces from dataset.

**Criteria**:
- Blurriness detection
- Face size validation
- Alignment quality check
- Landmark confidence

**Process**:
```python
def is_bad_alignment(img_path):
    img = Image.open(img_path)
    
    # Check 1: Image size
    if img.size != (112, 112):
        return True
    
    # Check 2: Blurriness (Laplacian variance)
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 100:  # Too blurry
        return True
    
    # Check 3: Brightness
    brightness = np.mean(gray)
    if brightness < 30 or brightness > 225:  # Too dark or bright
        return True
    
    return False
```

**Usage**:
```bash
python filter_bad_alignments.py \
    --input-dir aligned_faces/ \
    --output-dir filtered_faces/ \
    --threshold 100
```

### filter_niqab_specific.py

**Purpose**: Filter niqab-specific issues.

**Criteria**:
- Eye region visibility check
- Minimum visible area threshold
- Face orientation validation

### keep_good_niqab.py

**Purpose**: Keep only high-quality niqab faces.

**Features**:
- Positive filtering (keep good ones)
- Quality scoring system
- Multiple quality criteria

---

## 4. Data Review Scripts

### Purpose
Interactive tools for reviewing and validating datasets.

### review_niqab_faces.py

**Purpose**: Interactive review of niqab face alignments.

**Features**:
- Visual inspection interface
- Keyboard controls for accept/reject
- Batch processing with checkpoints
- Statistics tracking

**Usage**:
```bash
python review_niqab_faces.py \
    --data-dir aligned_niqab/ \
    --output good_faces/ \
    --reject-dir bad_faces/
```

**Interface**:
```
Controls:
  'a' - Accept current image
  'r' - Reject current image
  'n' - Next image
  'p' - Previous image
  'q' - Quit and save progress
  
Statistics:
  Reviewed: 150/1000
  Accepted: 120
  Rejected: 30
  Acceptance Rate: 80%
```

### review_niqab_web.py

**Purpose**: Web-based review interface.

**Features**:
- Browser-based UI
- Multi-user support
- Progress persistence
- Export functionality

### review_niqab_web_mask.py

**Purpose**: Review with occlusion mask visualization.

**Features**:
- Shows original and mask overlays
- Side-by-side comparison
- Mask quality assessment

---

## 5. Cleanup Scripts

### cleanup_failed_log.py

**Purpose**: Clean up failed alignment logs and retry.

**Features**:
- Parse failure logs
- Identify patterns
- Retry with different parameters
- Archive failed cases

### delete_failed_alignments.py

**Purpose**: Remove failed alignment results.

```python
def delete_failed_alignments(data_dir):
    failed_count = 0
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            path = os.path.join(root, file)
            
            # Check if valid image
            try:
                img = Image.open(path)
                img.verify()
            except:
                os.remove(path)
                failed_count += 1
    
    print(f"Removed {failed_count} failed alignments")
```

### remove_bad_alignments.py

**Purpose**: Remove alignments below quality threshold.

### delete_shopping_mall.py

**Purpose**: Remove specific unwanted categories.

---

## 6. Diagnostic Scripts

### diagnose_training_issue.py

**Purpose**: Debug training problems.

**Features**:
- Check data loading
- Validate transforms
- Test model forward pass
- Identify NaN sources
- Profile memory usage

**Common Checks**:
```python
# 1. Data loading
loader = create_dataloader()
batch = next(iter(loader))
print(f"Batch shape: {batch[0].shape}")

# 2. Check for NaN
if torch.isnan(batch[0]).any():
    print("WARNING: NaN in data!")

# 3. Check value ranges
print(f"Min: {batch[0].min()}, Max: {batch[0].max()}")

# 4. Test forward pass
model = load_model()
try:
    output = model(batch[0].cuda())
    print("Forward pass successful")
except Exception as e:
    print(f"Forward pass failed: {e}")

# 5. Memory profile
import torch.cuda
torch.cuda.reset_peak_memory_stats()
output = model(batch[0].cuda())
peak_mem = torch.cuda.max_memory_allocated() / 1024**2
print(f"Peak memory: {peak_mem:.2f} MB")
```

### run_all_tests.py

**Purpose**: Run comprehensive test suite.

**Features**:
- Unit tests for all modules
- Integration tests
- Performance benchmarks
- Coverage reporting

---

## 7. Example Scripts

### example_align_niqab.py

**Purpose**: Simple example demonstrating niqab alignment.

**Code**:
```python
from face_alignment import align
from PIL import Image
import os

input_dir = 'raw_niqab'
output_dir = 'aligned_niqab'
os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if not fname.endswith(('.jpg', '.png')):
        continue
    
    input_path = os.path.join(input_dir, fname)
    output_path = os.path.join(output_dir, fname)
    
    # Try alignment
    aligned = align.get_aligned_face(input_path)
    
    if aligned:
        aligned.save(output_path)
        print(f"✓ {fname}")
    else:
        print(f"✗ {fname} - Failed to align")
```

---

## Data Processing Pipeline

### Recommended Workflow

```
Raw Images
    │
    ├──> 1. align_niqab_robust.py
    │    └──> Aligned faces + failed log
    │
    ├──> 2. filter_bad_alignments.py
    │    └──> High-quality alignments
    │
    ├──> 3. review_niqab_faces.py (optional)
    │    └──> Manually verified faces
    │
    ├──> 4. analyze_occlusion_simple.py (optional)
    │    └──> Quality statistics
    │
    └──> Clean Dataset for Training
```

### Batch Processing Example

```bash
#!/bin/bash

# 1. Align all images
python align_niqab_robust.py \
    --input-dir raw_data/ \
    --output-dir aligned_data/ \
    --device cuda:0

# 2. Filter bad alignments
python filter_bad_alignments.py \
    --input-dir aligned_data/ \
    --output-dir filtered_data/ \
    --threshold 100

# 3. Review (optional)
python review_niqab_faces.py \
    --data-dir filtered_data/ \
    --output reviewed_data/

# 4. Analyze quality
python analyze_occlusion_simple.py \
    --data-dir reviewed_data/ \
    --output-dir analysis/

echo "Processing complete!"
```

---

## Configuration Best Practices

### For Niqab Dataset

**MTCNN Settings**:
```python
# Lenient settings for challenging faces
min_face_size = 8
thresholds = [0.3, 0.4, 0.5]
nms_thresholds = [0.5, 0.5, 0.5]
```

**Manual Crop Parameters**:
```python
crop_ratio = 0.3  # 30% of image
vertical_offset = 0.1  # Start 10% from top
```

### For General Dataset

**MTCNN Settings**:
```python
# Standard settings
min_face_size = 40
thresholds = [0.6, 0.7, 0.9]
nms_thresholds = [0.7, 0.7, 0.7]
```

---

## Common Issues & Solutions

### Issue 1: Low Alignment Success Rate

**Possible Causes**:
- Images too low resolution
- Extreme poses or occlusions
- Poor lighting

**Solutions**:
1. Lower MTCNN thresholds
2. Use manual crop fallback
3. Pre-process images (enhance contrast, denoise)

### Issue 2: Poor Alignment Quality

**Possible Causes**:
- Incorrect landmarks
- Low-quality source images
- Wrong reference points

**Solutions**:
1. Verify reference points for dataset
2. Use stricter filtering criteria
3. Manual review critical samples

### Issue 3: Inconsistent Face Sizes

**Possible Causes**:
- Variable crop windows
- Different alignment methods

**Solutions**:
1. Standardize crop size (112×112)
2. Use same alignment script for all images
3. Post-process to verify sizes

### Issue 4: Batch Processing Errors

**Possible Causes**:
- Out of memory
- Corrupted images
- Missing files

**Solutions**:
1. Process in smaller batches
2. Add error handling and logging
3. Skip corrupted files, log them

---

## Script Selection Guide

| Task | Recommended Script | Alternative |
|------|-------------------|-------------|
| Align niqab faces | `align_niqab_robust.py` | `align_niqab_improved.py` |
| Align standard faces | `align.py` (from face_alignment) | N/A |
| Filter quality | `filter_bad_alignments.py` | `keep_good_niqab.py` |
| Review dataset | `review_niqab_faces.py` | `review_niqab_web.py` |
| Analyze occlusion | `analyze_occlusion_effectiveness.py` | `analyze_occlusion_simple.py` |
| Cleanup | `delete_failed_alignments.py` | `remove_bad_alignments.py` |
| Debug | `diagnose_training_issue.py` | `run_all_tests.py` |

---

## Performance Metrics

### Alignment Success Rates

**Standard Faces** (LFW, MS1MV2):
- MTCNN: ~99.5% success rate
- Processing: ~50 images/second (GPU)

**Niqab Faces**:
- MTCNN alone: ~30-50% success rate
- With manual fallback: ~95% success rate
- Processing: ~20 images/second (GPU)

### Quality Metrics

**Good Alignment** (suggested criteria):
- Laplacian variance > 100 (not blurry)
- Brightness: 30-225 (not too dark/bright)
- Face size: ≥ 90% of target size
- Landmark confidence: > 0.9

---

## Extending Scripts

### Adding New Filter Criteria

```python
# In filter_bad_alignments.py
def custom_quality_check(img_path):
    img = Image.open(img_path)
    
    # Your custom criteria
    if some_condition(img):
        return False  # Reject
    
    return True  # Accept

# Register filter
filters = [
    is_bad_alignment,
    custom_quality_check
]

for img_path in image_paths:
    if all(f(img_path) for f in filters):
        keep_image(img_path)
```

### Adding New Review Interface

```python
# Custom review script
class CustomReviewer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.results = []
    
    def review_batch(self, batch_size=10):
        for i in range(batch_size):
            img = self.load_next()
            quality_score = self.compute_quality(img)
            
            if quality_score > threshold:
                self.accept(img)
            else:
                self.reject(img)
    
    def save_results(self):
        with open('review_results.json', 'w') as f:
            json.dump(self.results, f)
```

---

## Maintenance

### Regular Tasks

1. **Monthly**: Review alignment success rates
2. **Per Dataset**: Run quality analysis
3. **Before Training**: Verify data integrity
4. **After Filtering**: Check class balance

### Monitoring

```bash
# Count successful alignments
find aligned_data/ -name "*.jpg" | wc -l

# Check average file size
du -sh aligned_data/

# Verify all images loadable
python -c "
from PIL import Image
import os
count = 0
for root, dirs, files in os.walk('aligned_data/'):
    for f in files:
        if f.endswith('.jpg'):
            try:
                Image.open(os.path.join(root, f)).verify()
                count += 1
            except:
                print(f'Failed: {f}')
print(f'Verified {count} images')
"
```

