# Dataset & Data Processing Components

## Overview
The dataset and data processing module handles data loading, augmentation, sampling strategies, and transforms for training and validation. It implements specialized transforms for occlusion simulation and PK sampling for metric learning.

## Component Architecture

```
Data Pipeline
┌─────────────────────────────────────────────────────────────┐
│                    DataModule (data.py)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Train Dataset│  │ Val Dataset  │  │  RandomIdentity  │  │
│  │              │  │              │  │  Sampler         │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
└─────────┼──────────────────┼───────────────────┼────────────┘
          │                  │                   │
    ┌─────▼────┐      ┌──────▼───────┐   ┌──────▼─────┐
    │ Image    │      │  FiveVal     │   │ PK Batches │
    │ Folder   │      │  Dataset     │   │ (P×K)      │
    └─────┬────┘      └──────┬───────┘   └────────────┘
          │                  │
    ┌─────▼────┐      ┌──────▼───────┐
    │ Augment  │      │  Memfiles    │
    └─────┬────┘      └──────────────┘
          │
    ┌─────▼────┐
    │Transform │
    └──────────┘
```

---

## 1. transforms.py - Data Transforms

### Purpose
Implements specialized transforms for face recognition training, including realistic occlusion simulation with medical masks.

### Key Transforms

#### MedicalMaskOcclusion

**Purpose**: Simulates realistic medical mask occlusion on faces.

**Features**:
- Realistic mask positioning using facial landmarks
- Random variations in mask size, position, and rotation
- Generates binary occlusion masks for supervision
- Standard reference points for CASIA WebFace alignment

**Architecture**:
```python
MedicalMaskOcclusion(
    mask_path='maskPic.jpg',
    prob=0.5,
    ref_points=None  # Default: CASIA WebFace standard
)
```

**Reference Points** (112×112 images):
```python
ref_points = [
    (48.0252, 71.7366),   # Nose
    (33.5493, 92.3655),   # Left mouth corner
    (62.7299, 92.2041)    # Right mouth corner
]
```

**Transform Process**:
```python
def __call__(self, img):
    if random() > prob:
        return img, all_ones_mask  # No occlusion
    
    # 1. Randomize facial landmarks
    nose = randomize_point(ref_points[0], max_offset=3)
    left_mouth = randomize_point(ref_points[1], max_offset=3)
    right_mouth = randomize_point(ref_points[2], max_offset=3)
    
    # 2. Random disturbance
    shift_x = uniform(-5, 5)
    shift_y = uniform(-5, 5)
    scale = uniform(0.95, 1.10)
    angle = uniform(-7, 7)
    
    # 3. Compute mask dimensions
    mouth_width = norm(right_mouth - left_mouth)
    mask_width = int(mouth_width * 4.0 * scale)   # Wide coverage
    mask_height = int((right_mouth[1] - nose[1]) * 4.0 * scale)  # Cover chin
    
    # 4. Resize and rotate mask
    mask = mask_img.resize((mask_width, mask_height))
    mask = mask.rotate(angle, expand=True)
    
    # 5. Position mask
    center_x = int(nose[0] + shift_x)
    paste_y = int(nose[1] - vertical_offset + shift_y)
    paste_x = center_x - mask.width // 2
    
    # 6. Paste mask onto image
    img.paste(mask, (paste_x, paste_y), mask)
    
    # 7. Generate binary occlusion mask
    binary_mask = create_occlusion_mask(mask, paste_x, paste_y)
    
    return img, binary_mask
```

**Mask Positioning**:
- Horizontal: Centered on nose
- Vertical: 25px above nose to cover nose and mouth
- Size: 4× mouth width for realistic coverage including strings

**Output**:
- Modified PIL Image with medical mask
- Binary mask (PIL Image 'L' mode): 255=visible, 0=occluded

#### SyntheticOcclusionMask

**Purpose**: Generates random rectangular occlusions for training robustness.

**Features**:
- Multiple random occlusion patches per image
- Variable patch sizes and aspect ratios
- Direct generation of feature map-sized masks (7×7)
- Random colors or black patches

**Architecture**:
```python
SyntheticOcclusionMask(
    prob=0.5,
    min_size=0.1,      # 10% of image dimension
    max_size=0.4,      # 40% of image dimension
    max_patches=3,     # Up to 3 occlusion patches
    feature_map_size=7 # Target feature map resolution
)
```

**Process**:
```python
def __call__(self, img):
    if random() > prob:
        return img, all_ones_mask
    
    occluded_img = img.copy()
    mask = ones((feature_map_size, feature_map_size))
    
    num_patches = randint(1, max_patches + 1)
    
    for _ in range(num_patches):
        # 1. Random size
        patch_size = int(min(H, W) * uniform(min_size, max_size))
        aspect_ratio = uniform(0.7, 1.4)
        patch_h = int(patch_size / sqrt(aspect_ratio))
        patch_w = int(patch_size * sqrt(aspect_ratio))
        
        # 2. Random position
        x0, y0 = random_position()
        
        # 3. Apply occlusion
        color = random_color() if random() < 0.5 else (0, 0, 0)
        occluded_img.paste(block, (x0, y0, x1, y1))
        
        # 4. Update feature map mask
        fm_x0, fm_y0, fm_x1, fm_y1 = map_to_feature_coords()
        mask[fm_y0:fm_y1, fm_x0:fm_x1] = 0.0  # Mark occluded
    
    return occluded_img, mask
```

**Coordinate Mapping**:
```python
# Map image coordinates to feature map coordinates
fm_x = int((x / image_width) * feature_map_size)
fm_y = int((y / image_height) * feature_map_size)
```

#### OcclusionMaskWrapper

**Purpose**: Wraps existing transforms to work with (image, mask) tuples.

```python
class OcclusionMaskWrapper:
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, input_data):
        if isinstance(input_data, tuple):
            img, mask = input_data
            return self.transform(img), mask  # Transform image, keep mask
        else:
            return self.transform(input_data)  # Regular transform
```

**Usage**:
```python
transform = Compose([
    MedicalMaskOcclusion(prob=0.5),
    OcclusionMaskWrapper(RandomHorizontalFlip()),
    ToTensorWithMask(),
    OcclusionMaskWrapper(Normalize(mean=[0.5]*3, std=[0.5]*3))
])
```

#### ToTensorWithMask

**Purpose**: Converts PIL Images and masks to PyTorch tensors.

```python
class ToTensorWithMask:
    def __call__(self, input_data):
        if isinstance(input_data, tuple):
            img, mask = input_data
            img_tensor = to_tensor(img)  # [3, H, W]
            
            if isinstance(mask, PIL.Image):
                mask_array = np.array(mask)
            mask_tensor = torch.from_numpy(mask_array).float()
            mask_tensor = mask_tensor.unsqueeze(0) / 255.0  # [1, H, W], [0,1]
            
            return img_tensor, mask_tensor
        else:
            return to_tensor(input_data)
```

#### Other Transforms

##### RandomOcclusion
```python
# Simple random square occlusion
RandomOcclusion(min_size=0.2, max_size=1.0)
```

##### RectScale
```python
# Resize to specific dimensions
RectScale(height=112, width=112, interpolation=Image.BILINEAR)
```

##### RandomSizedRectCrop
```python
# Random crop with resizing
RandomSizedRectCrop(height=112, width=112, crop_factor=0.8)
```

##### RandomErasing
```python
# Random erasing augmentation
RandomErasing(EPSILON=0.5, mean=[0.485, 0.456, 0.406])
```

---

## 2. sampler.py - Identity-Based Sampling

### Purpose
Implements PK sampling strategy for metric learning: P identities × K instances per identity.

### Class: RandomIdentitySampler

**Purpose**: Samples batches containing multiple instances of the same identities for metric learning.

**PK Sampling Strategy**:
```
Batch Structure:
┌──────────────────────────────────────┐
│ Identity 1: [img1, img2, img3, img4] │  K=4 instances
│ Identity 2: [img5, img6, img7, img8] │
│ Identity 3: [img9, imgA, imgB, imgC] │
│     ...                               │
│ Identity P: [imgX, imgY, imgZ, imgW] │
└──────────────────────────────────────┘
  Total batch size = P × K
```

**Implementation**:
```python
class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.num_instances = num_instances  # K
        
        # Build index: label → [sample_indices]
        self.index_dic = defaultdict(list)
        for index, sample in enumerate(data_source):
            if len(sample) == 2:
                _, label = sample
            elif len(sample) == 3:
                _, label, _ = sample  # Handle (img, label, mask)
            
            self.index_dic[label].append(index)
        
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)  # P
    
    def __iter__(self):
        # Shuffle identities
        indices = torch.randperm(self.num_identities)
        ret = []
        
        for i in indices:
            pid = self.pids[i]
            samples = self.index_dic[pid]
            
            # Sample K instances (with replacement if needed)
            replace = len(samples) < self.num_instances
            selected = np.random.choice(
                samples, 
                size=self.num_instances, 
                replace=replace
            )
            ret.extend(selected)
        
        return iter(ret)
    
    def __len__(self):
        return self.num_identities * self.num_instances
```

**Benefits for Metric Learning**:
1. **Positive Pairs**: Multiple samples from same identity in batch
2. **Negative Pairs**: Samples from different identities
3. **Hard Mining**: Can implement hard negative mining within batch
4. **Efficient**: Avoids need to sample pairs separately

**Usage with DataLoader**:
```python
K = 4  # instances per identity
P = batch_size // K  # identities per batch

sampler = RandomIdentitySampler(dataset, num_instances=K)

dataloader = DataLoader(
    dataset,
    batch_size=P * K,
    sampler=sampler,
    shuffle=False  # Sampler handles shuffling
)
```

**Example Batch** (P=16, K=4, batch_size=64):
```
Indices: [3, 7, 12, 15,  # Identity A
          22, 45, 67, 89, # Identity B
          ...              # 14 more identities
          105, 123, 145, 167]  # Identity P
```

---

## 3. Dataset Classes

### CustomImageFolderDataset

**Purpose**: Loads images from folder structure with augmentation support.

**Directory Structure**:
```
root/
  ├── class_0/
  │   ├── img_001.jpg
  │   ├── img_002.jpg
  │   └── ...
  ├── class_1/
  │   └── ...
  └── ...
```

**Features**:
- Inherits from `torchvision.datasets.ImageFolder`
- Integrates with `Augmenter` for advanced augmentations
- Supports (image, mask) tuple returns
- Saves sample images for debugging

**Implementation**:
```python
class CustomImageFolderDataset(datasets.ImageFolder):
    def __init__(self, root, transform, augmenter_args, ...):
        super().__init__(root, transform=transform)
        self.augmenter = Augmenter(
            crop_prob, photometric_prob, low_res_prob
        )
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        
        # Convert RGB to BGR if needed
        if self.swap_color_channel:
            sample = Image.fromarray(np.asarray(sample)[:,:,::-1])
        
        # Apply augmentations
        sample = self.augmenter.augment(sample)
        
        # Apply transforms
        if self.transform:
            result = self.transform(sample)
            
            # Handle (image, mask) tuples
            if isinstance(result, tuple) and len(result) == 2:
                sample, mask = result
                return sample, target, mask
            else:
                sample = result
        
        return sample, target
```

### AugmentRecordDataset

**Purpose**: Loads images from MXNet record format (binary).

**Features**:
- Efficient binary format
- Integrated augmentation
- Supports large-scale datasets (MS1MV2, WebFace)

**Implementation**:
```python
class AugmentRecordDataset(BaseMXDataset):
    def __init__(self, root_dir, transform, augmenter_args, ...):
        super().__init__(root_dir, swap_color_channel)
        self.augmenter = Augmenter(...)
        self.transform = transform
    
    def __getitem__(self, index):
        # Read from MXNet record
        sample, target = self.read_sample(index)
        
        # Augment
        sample = self.augmenter.augment(sample)
        
        # Transform
        if self.transform:
            result = self.transform(sample)
            if isinstance(result, tuple) and len(result) == 2:
                return result[0], target, result[1]  # img, label, mask
            return result, target
        
        return sample, target
```

### FiveValidationDataset

**Purpose**: Concatenated validation dataset for 5 face recognition benchmarks.

**Benchmarks**:
1. **agedb_30**: Age variation
2. **cfp_fp**: Frontal-profile poses
3. **lfw**: Labeled Faces in the Wild
4. **cplfw**: Cross-pose LFW
5. **calfw**: Cross-age LFW

**Features**:
- Memory-mapped file access for efficiency
- Concatenates all benchmarks into single dataset
- Returns dataname for per-benchmark evaluation

**Structure**:
```python
class FiveValidationDataset(Dataset):
    def __init__(self, val_data_dict, concat_mem_file_name):
        self.dataname_to_idx = {
            "agedb_30": 0, "cfp_fp": 1, "lfw": 2, 
            "cplfw": 3, "calfw": 4
        }
        
        # Concatenate all datasets
        all_imgs = []
        all_issame = []
        all_dataname = []
        
        for key, (imgs, issame) in val_data_dict.items():
            all_imgs.append(imgs)
            # Duplicate issame for each image in pair
            dup_issame = [val for val in issame for _ in (0, 1)]
            all_issame.append(dup_issame)
            all_dataname.append([self.dataname_to_idx[key]] * len(imgs))
        
        # Use memmap for efficiency
        self.all_imgs = read_memmap(concat_mem_file_name)
        self.all_issame = np.concatenate(all_issame)
        self.all_dataname = np.concatenate(all_dataname)
    
    def __getitem__(self, index):
        x = torch.tensor(self.all_imgs[index].copy())
        y = self.all_issame[index]
        dataname = self.all_dataname[index]
        return x, y, dataname, index
```

**Memfile Format**:
```python
# Create memfile
memmap_configs = {
    'shape': np_array.shape,
    'dtype': str(np_array.dtype)
}
mm = np.memmap(filename, mode='w+', shape=shape, dtype=dtype)
mm[:] = np_array[:]
mm.flush()

# Read memfile
with open(filename + '.conf', 'r') as f:
    config = json.load(f)
    mm = np.memmap(filename, mode='r+', 
                  shape=tuple(config['shape']),
                  dtype=config['dtype'])
```

---

## 4. dataset/augmenter.py

### Purpose
Provides quality-degrading augmentations for robust training.

### Class: Augmenter

**Augmentation Types**:
1. **Low-Resolution**: Simulates low-quality images
2. **Crop**: Random cropping augmentation
3. **Photometric**: Color/brightness variations

**Implementation**:
```python
class Augmenter:
    def __init__(self, crop_prob, photometric_prob, low_res_prob):
        self.crop_prob = crop_prob
        self.photometric_prob = photometric_prob
        self.low_res_prob = low_res_prob
    
    def augment(self, img):
        if random() < self.crop_prob:
            img = self.random_crop(img)
        
        if random() < self.photometric_prob:
            img = self.photometric_augment(img)
        
        if random() < self.low_res_prob:
            img = self.low_resolution(img)
        
        return img
    
    def low_resolution(self, img):
        # Downsample and upsample
        small_size = int(img.size[0] * random.uniform(0.25, 0.75))
        img = img.resize((small_size, small_size), Image.BILINEAR)
        img = img.resize((112, 112), Image.BILINEAR)
        return img
    
    def random_crop(self, img):
        # Random crop with resize
        scale = random.uniform(0.8, 1.0)
        # ... crop logic
        return img
    
    def photometric_augment(self, img):
        # Adjust brightness, contrast, saturation
        # ... photometric logic
        return img
```

---

## Data Flow

### Training Pipeline
```
Image File
  │
  ├──> Load (PIL Image)
  │
  ├──> Color Channel Swap (if needed)
  │
  ├──> Augmenter
  │    ├──> Random Crop
  │    ├──> Photometric
  │    └──> Low Resolution
  │
  ├──> Transform Pipeline
  │    ├──> Random Horizontal Flip
  │    ├──> MedicalMaskOcclusion
  │    │    └──> Returns (image, mask)
  │    ├──> ToTensorWithMask
  │    └──> Normalize
  │
  └──> Batch via PK Sampler
       └──> [P×K, 3, 112, 112], [P×K], [P×K, 1, 112, 112]
```

### Validation Pipeline
```
Memfile
  │
  ├──> Read numpy array
  │
  ├──> Convert to tensor
  │
  └──> Batch sequentially
       └──> [B, 3, 112, 112], [B], [B], [B]
            (image, issame, dataname, index)
```

---

## Configuration Examples

### Standard Training
```python
train_transform = Compose([
    lambda img: img.transpose(Image.FLIP_LEFT_RIGHT) 
                if random() > 0.5 else img,
    MedicalMaskOcclusion(prob=0.5),
    ToTensorWithMask(),
    lambda x: (Normalize(mean=[0.5]*3, std=[0.5]*3)(x[0]), x[1])
])

dataset = CustomImageFolderDataset(
    root='data/train',
    transform=train_transform,
    low_res_augmentation_prob=0.2,
    crop_augmentation_prob=0.1,
    photometric_augmentation_prob=0.3
)

sampler = RandomIdentitySampler(dataset, num_instances=4)
loader = DataLoader(dataset, batch_size=64, sampler=sampler)
```

### Validation
```python
val_dataset = FiveValidationDataset(
    val_data_dict={'agedb_30': (imgs, issame), ...},
    concat_mem_file_name='memfile.dat'
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=4
)
```

---

## Design Decisions

### 1. PK Sampling
**Rationale**: Essential for metric learning losses (pairwise, triplet) that require multiple samples per identity in each batch.

### 2. Medical Mask Simulation
**Rationale**: Realistic occlusion is crucial for training robust models. Medical masks are common real-world occlusions.

### 3. Binary Occlusion Masks
**Rationale**: Provides supervision signal for occlusion prediction head, enabling interpretable and effective occlusion handling.

### 4. Memfiles for Validation
**Rationale**: Memory-mapped files reduce memory footprint for large validation datasets while maintaining fast access.

### 5. Tuple-Based Transform Pipeline
**Rationale**: Allows seamless integration of occlusion masks through the transform pipeline without breaking existing transforms.

---

## Performance Considerations

### Memory
- **Memfiles**: Reduce RAM usage by ~10×
- **Lazy Loading**: Images loaded on-demand
- **Batch Size**: Typically 256 (P=64, K=4)

### Speed
- **Num Workers**: 16 workers for training, 4 for validation
- **Pin Memory**: Enabled for GPU transfer
- **Augmentation**: Applied in dataset __getitem__ (parallelized across workers)

### Quality
- **Augmentation Probabilities**: Tuned for generalization
  - Low-res: 0.0-0.2 (optional, for robustness)
  - Crop: 0.0-0.1 (optional)
  - Photometric: 0.0-0.3 (optional)
  - Mask: 0.5 (recommended)

---

## Usage Notes

### Setting Up Validation Data
```python
# Convert validation data to memfiles (one-time)
python -c "
from evaluate_utils import get_val_data
get_val_data('path/to/faces_emore')
"
```

### Custom Occlusion Masks
```python
# Use custom mask image
mask_occlusion = MedicalMaskOcclusion(
    mask_path='custom_mask.png',
    prob=0.5,
    ref_points=[(48, 72), (34, 92), (63, 92)]  # Custom landmarks
)
```

### Debugging Augmentations
```python
# Save sample images
dataset = CustomImageFolderDataset(
    ...,
    output_dir='experiments/debug'
)

# Sample saved to: experiments/debug/training_samples/sample.jpg
```

---

## Common Issues

### Issue 1: OOM During Training
**Solution**: Reduce batch size or use gradient accumulation
```python
# Option 1: Smaller batch size
batch_size = 128  # instead of 256

# Option 2: Gradient accumulation
accumulate_grad_batches = 2  # effective batch size = 256
```

### Issue 2: Slow Data Loading
**Solution**: Increase num_workers or use SSD storage
```python
num_workers = 16  # Increase if CPU cores available
```

### Issue 3: Masks Not Generated
**Check**: Transform pipeline returns tuples
```python
# Ensure MedicalMaskOcclusion is used correctly
result = transform(img)
assert isinstance(result, tuple), "Expected (image, mask)"
```

### Issue 4: Validation Data Not Found
**Solution**: Run prepare_data() once
```python
data_module = DataModule(...)
data_module.prepare_data()  # Creates memfiles
```

