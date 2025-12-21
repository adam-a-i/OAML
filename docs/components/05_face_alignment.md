# Face Alignment Components

## Overview
Implements MTCNN-based face detection and alignment for preprocessing face images to standard 112×112 format.

---

## 1. face_alignment/align.py - Simple Alignment Interface

### Purpose
Provides simple API for face alignment using MTCNN.

### Key Functions

#### `get_aligned_face(image_path, rgb_pil_image=None)`

**Purpose**: Detect and align face from image.

```python
def get_aligned_face(image_path, rgb_pil_image=None):
    """
    Args:
        image_path: Path to image file (optional)
        rgb_pil_image: PIL Image in RGB format (optional)
    
    Returns:
        PIL Image: Aligned face (112×112) or None if detection failed
    """
    if rgb_pil_image is None:
        img = Image.open(image_path).convert('RGB')
    else:
        img = rgb_pil_image
    
    try:
        bboxes, faces = mtcnn_model.align_multi(img, limit=1)
        return faces[0]  # Return first detected face
    except Exception as e:
        print(f'Face detection failed: {e}')
        return None
```

**Global MTCNN Instance**:
```python
mtcnn_model = mtcnn.MTCNN(device='cuda:0', crop_size=(112, 112))
```

**Usage**:
```python
from face_alignment import align

# From file
aligned = align.get_aligned_face('photo.jpg')

# From PIL Image
from PIL import Image
img = Image.open('photo.jpg')
aligned = align.get_aligned_face(rgb_pil_image=img)

# Save
if aligned:
    aligned.save('aligned.jpg')
```

---

## 2. face_alignment/mtcnn.py - MTCNN Implementation

### Purpose
Implements Multi-Task Cascaded Convolutional Networks for face detection and alignment.

### Class: MTCNN

**Architecture**: 3-stage cascade
```
Input Image
  │
  ├──> Stage 1: PNet (Proposal Network)
  │    └──> Candidate face regions
  │
  ├──> Stage 2: RNet (Refinement Network)
  │    └──> Refined face regions
  │
  └──> Stage 3: ONet (Output Network)
       ├──> Final bounding boxes
       └──> 5 facial landmarks
```

**Initialization**:
```python
MTCNN(device='cuda:0', crop_size=(112, 112))

Parameters:
  - device: 'cuda:0' or 'cpu'
  - crop_size: (112, 112) or (96, 112)
```

**Default Settings**:
```python
self.min_face_size = 20
self.thresholds = [0.6, 0.7, 0.9]  # Per-stage confidence thresholds
self.nms_thresholds = [0.7, 0.7, 0.7]  # Per-stage NMS thresholds
self.factor = 0.85  # Scale factor for pyramid
```

### Key Methods

#### `detect_faces(image, min_face_size, thresholds, nms_thresholds, factor)`

**Purpose**: Detect faces and facial landmarks.

**Process**:
```python
# 1. Build image pyramid
scales = []
m = min_detection_size / min_face_size
min_length = min(height, width) * m

while min_length > min_detection_size:
    scales.append(m * factor**factor_count)
    min_length *= factor
    factor_count += 1

# 2. Stage 1: PNet on multiple scales
for s in scales:
    boxes = run_first_stage(image, pnet, scale=s, threshold=thresholds[0])
    bounding_boxes.append(boxes)

# NMS and calibration
bounding_boxes = nms(bounding_boxes, nms_thresholds[0])
bounding_boxes = calibrate_box(bounding_boxes)
bounding_boxes = convert_to_square(bounding_boxes)

# 3. Stage 2: RNet
img_boxes = get_image_boxes(bounding_boxes, image, size=24)
output = rnet(img_boxes)
# Filter by threshold and apply NMS
bounding_boxes = filter_and_nms(output, thresholds[1], nms_thresholds[1])

# 4. Stage 3: ONet
img_boxes = get_image_boxes(bounding_boxes, image, size=48)
output = onet(img_boxes)
landmarks, offsets, probs = output

# Compute landmark coordinates
width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
landmarks[:, 0:5] = xmin + width * landmarks[:, 0:5]
landmarks[:, 5:10] = ymin + height * landmarks[:, 5:10]

return bounding_boxes, landmarks
```

**Output**:
- `bounding_boxes`: [N, 5] - (x1, y1, x2, y2, confidence)
- `landmarks`: [N, 10] - (x1,...,x5, y1,...,y5) for 5 points:
  - Left eye, Right eye, Nose, Left mouth, Right mouth

#### `align(img)`

**Purpose**: Detect and align single face.

```python
def align(img):
    _, landmarks = self.detect_faces(
        img, self.min_face_size, self.thresholds, 
        self.nms_thresholds, self.factor
    )
    
    # Convert landmarks to 5-point format
    facial5points = [[landmarks[0][j], landmarks[0][j + 5]] 
                     for j in range(5)]
    
    # Warp and crop face
    warped_face = warp_and_crop_face(
        np.array(img), facial5points, 
        self.refrence, crop_size=self.crop_size
    )
    
    return Image.fromarray(warped_face)
```

#### `align_multi(img, limit=None)`

**Purpose**: Detect and align multiple faces.

```python
def align_multi(img, limit=None):
    boxes, landmarks = self.detect_faces(...)
    
    if limit:
        boxes = boxes[:limit]
        landmarks = landmarks[:limit]
    
    faces = []
    for landmark in landmarks:
        facial5points = [[landmark[j], landmark[j + 5]] 
                        for j in range(5)]
        warped_face = warp_and_crop_face(
            np.array(img), facial5points, 
            self.refrence, crop_size=self.crop_size
        )
        faces.append(Image.fromarray(warped_face))
    
    return boxes, faces
```

---

## 3. face_alignment/mtcnn_pytorch/ - MTCNN Implementation Details

### Network Architectures

#### PNet (Proposal Network)
```
Input: Various scales of image patches (12×12)
Conv1: 3×3, 10 channels + PReLU + MaxPool
Conv2: 3×3, 16 channels + PReLU
Conv3: 3×3, 32 channels + PReLU
Outputs:
  - Face classification: 2 channels (face/non-face)
  - Bounding box regression: 4 channels (dx, dy, dw, dh)
```

#### RNet (Refinement Network)
```
Input: 24×24 image patches
Conv1: 3×3, 28 channels + PReLU + MaxPool
Conv2: 3×3, 48 channels + PReLU + MaxPool
Conv3: 2×2, 64 channels + PReLU
FC: 128 units + PReLU
Outputs:
  - Face classification: 2 units
  - Bounding box regression: 4 units
```

#### ONet (Output Network)
```
Input: 48×48 image patches
Conv1: 3×3, 32 channels + PReLU + MaxPool
Conv2: 3×3, 64 channels + PReLU + MaxPool
Conv3: 3×3, 64 channels + PReLU + MaxPool
Conv4: 2×2, 128 channels + PReLU
FC: 256 units + PReLU
Outputs:
  - Face classification: 2 units
  - Bounding box regression: 4 units
  - Facial landmarks: 10 units (5 points × 2 coordinates)
```

### Helper Modules

#### align_trans.py
- `get_reference_facial_points()`: Standard facial landmark positions
- `warp_and_crop_face()`: Affine transformation for alignment

#### box_utils.py
- `nms()`: Non-maximum suppression
- `calibrate_box()`: Apply regression offsets
- `convert_to_square()`: Convert rectangles to squares
- `get_image_boxes()`: Extract image patches

#### detector.py
- Integration and orchestration of 3-stage detection

---

## Alignment Process

### Standard Pipeline
```
Input Image (any size)
  │
  ├──> Build Image Pyramid
  │    └──> Multiple scales (factor=0.85)
  │
  ├──> Stage 1: PNet
  │    ├──> Detect candidate regions at each scale
  │    ├──> NMS across scales
  │    └──> ~Thousands of candidates
  │
  ├──> Stage 2: RNet
  │    ├──> Refine candidates
  │    ├──> Filter by confidence
  │    └──> ~Hundreds of candidates
  │
  ├──> Stage 3: ONet
  │    ├──> Final detection
  │    ├──> Predict landmarks
  │    └──> ~Tens of faces
  │
  ├──> Facial Landmark Detection
  │    └──> 5 points: eyes, nose, mouth corners
  │
  └──> Affine Transformation
       ├──> Align landmarks to standard positions
       ├──> Crop to target size (112×112)
       └──> Output aligned face
```

### Reference Facial Points (112×112)
```python
# Standard positions for aligned faces
reference_points = np.array([
    [30.2946, 51.6963],  # Left eye
    [65.5318, 51.5014],  # Right eye
    [48.0252, 71.7366],  # Nose
    [33.5493, 92.3655],  # Left mouth
    [62.7299, 92.2041]   # Right mouth
], dtype=np.float32)
```

---

## Configuration & Tuning

### For High-Quality Images
```python
mtcnn = MTCNN(device='cuda:0', crop_size=(112, 112))
mtcnn.min_face_size = 40
mtcnn.thresholds = [0.6, 0.7, 0.9]  # Default
```

### For Challenging Conditions (e.g., Niqab)
```python
mtcnn = MTCNN(device='cuda:0', crop_size=(112, 112))
mtcnn.min_face_size = 8  # Much smaller minimum
mtcnn.thresholds = [0.3, 0.4, 0.5]  # Very lenient
mtcnn.nms_thresholds = [0.5, 0.5, 0.5]  # More lenient NMS
```

### For Speed
```python
mtcnn.min_face_size = 60  # Larger minimum (fewer scales)
mtcnn.factor = 0.9  # Fewer pyramid levels
```

---

## Usage Examples

### Basic Face Alignment
```python
from face_alignment import align

# Single face
aligned = align.get_aligned_face('photo.jpg')
if aligned:
    aligned.save('output.jpg')

# Multiple faces
from face_alignment.mtcnn import MTCNN
mtcnn = MTCNN()
boxes, faces = mtcnn.align_multi(Image.open('group.jpg'))

for i, face in enumerate(faces):
    face.save(f'face_{i}.jpg')
```

### Batch Processing
```python
import os
from tqdm import tqdm

input_dir = 'raw_images'
output_dir = 'aligned_faces'
os.makedirs(output_dir, exist_ok=True)

failed = []
for fname in tqdm(os.listdir(input_dir)):
    if not fname.endswith(('.jpg', '.png')):
        continue
    
    path = os.path.join(input_dir, fname)
    aligned = align.get_aligned_face(path)
    
    if aligned:
        aligned.save(os.path.join(output_dir, fname))
    else:
        failed.append(fname)

print(f"Failed to align {len(failed)} images")
```

### Manual Fallback (for Niqab)
```python
def align_with_fallback(image_path, crop_ratio=0.3):
    # Try MTCNN first
    aligned = align.get_aligned_face(image_path)
    
    if aligned:
        return aligned
    
    # Fallback: manual crop
    img = Image.open(image_path)
    width, height = img.size
    
    crop_width = int(width * crop_ratio)
    crop_height = int(height * crop_ratio)
    
    left = (width - crop_width) // 2
    top = int(height * 0.1)  # Upper portion
    
    cropped = img.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((112, 112), Image.LANCZOS)
```

---

## Performance Considerations

### Speed
- **GPU**: ~30-50ms per image (single face)
- **CPU**: ~200-500ms per image
- **Batch**: Can process multiple images in parallel

### Accuracy
- **LFW Dataset**: ~99.8% detection rate
- **Challenging Poses**: May fail on extreme profiles
- **Occlusions**: May fail with heavy occlusions (e.g., niqab)

### Memory
- **GPU**: ~500MB for model weights
- **Image Pyramid**: Depends on input size
  - 640×480: ~10 scales
  - Memory per scale: ~Width × Height × 3 bytes

---

## Common Issues & Solutions

### Issue 1: No Faces Detected
**Solutions**:
- Lower `min_face_size` (try 20 or even 8)
- Lower `thresholds` (e.g., [0.4, 0.5, 0.6])
- Check image quality and face visibility
- Try manual fallback for extreme cases

### Issue 2: Wrong Face Detected
**Solution**: Use `align_multi()` and select based on:
- Largest bounding box
- Highest confidence
- Centroid closest to image center

### Issue 3: Poor Alignment Quality
**Solutions**:
- Increase input image resolution
- Adjust `crop_size` if needed
- Check if detected landmarks are reasonable

### Issue 4: Slow Processing
**Solutions**:
- Use GPU (`device='cuda:0'`)
- Increase `min_face_size`
- Increase `factor` (fewer pyramid levels)
- Process multiple images in batch

---

## Integration with Training Pipeline

### Preprocessing Dataset
```bash
# Align all images in dataset
python align_niqab_robust.py \
  --input-dir raw_dataset \
  --output-dir aligned_dataset \
  --device cuda:0
```

### On-the-Fly Alignment (Not Recommended)
```python
# Slow but flexible
from face_alignment import align

class AlignedDataset(Dataset):
    def __getitem__(self, idx):
        img_path = self.paths[idx]
        aligned = align.get_aligned_face(img_path)
        # Apply transforms...
        return aligned, label
```

**Recommendation**: Pre-align all images and save to disk for faster training.

