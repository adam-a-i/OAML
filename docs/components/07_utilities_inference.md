# Utilities & Inference Components

## Overview
Helper utilities for distributed training, data processing, model inference, and graph-based sampling.

---

## 1. utils.py - General Utilities

### Purpose
Provides helper functions for normalization, distributed training, and dataset configuration.

### Key Components

#### `dotdict` - Dictionary with Dot Notation Access
```python
class dotdict(dict):
    """Access dictionary attributes with dot notation"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# Usage
args = dotdict({'lr': 0.1, 'batch_size': 256})
print(args.lr)  # 0.1
args.epochs = 24
```

#### `l2_norm(input, axis=1)` - L2 Normalization
```python
def l2_norm(input, axis=1):
    """L2 normalize tensor along specified axis"""
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output, norm

# Usage
embeddings, norms = l2_norm(features, axis=1)
# embeddings: [B, D] with unit norm
# norms: [B, 1] original norms
```

#### `fuse_features_with_norm(stacked_embeddings, stacked_norms)` - Feature Fusion
```python
def fuse_features_with_norm(stacked_embeddings, stacked_norms):
    """
    Fuse multiple feature sets using norm-weighted averaging
    
    Args:
        stacked_embeddings: [N, B, D] - N feature sets
        stacked_norms: [N, B, 1] - Corresponding norms
    
    Returns:
        fused: [B, D] - Fused features
        fused_norm: [B, 1] - Fused norms
    """
    # Recover original magnitudes
    pre_norm_embeddings = stacked_embeddings * stacked_norms
    
    # Sum and renormalize
    fused = pre_norm_embeddings.sum(dim=0)
    fused, fused_norm = l2_norm(fused, axis=1)
    
    return fused, fused_norm

# Usage (flip augmentation)
original_emb, original_norm = model(images)
flipped_emb, flipped_norm = model(flip(images))
stacked = torch.stack([original_emb, flipped_emb], dim=0)
stacked_norms = torch.stack([original_norm, flipped_norm], dim=0)
fused_emb, fused_norm = fuse_features_with_norm(stacked, stacked_norms)
```

### Distributed Training Utilities

#### `is_dist_avail_and_initialized()` - Check DDP Status
```python
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
```

#### `get_world_size()` - Number of Processes
```python
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()
```

#### `get_local_rank()` - Current Process Rank
```python
def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ["LOCAL_RANK"])
```

#### `all_gather(data)` - Gather Data Across Processes
```python
def all_gather(data):
    """
    Gather arbitrary picklable data across all processes
    
    Args:
        data: Any picklable object
    
    Returns:
        list[data]: List of data from all processes
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    
    # Serialize data
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(get_local_rank())
    
    # Gather sizes
    local_size = torch.tensor([tensor.numel()], device=tensor.device)
    size_list = [torch.tensor([0], device=tensor.device) 
                 for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    max_size = max([int(size.item()) for size in size_list])
    
    # Pad and gather tensors
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,), 
            dtype=torch.uint8, device=tensor.device
        )
        tensor = torch.cat((tensor, padding), dim=0)
    
    tensor_list = [torch.empty((max_size,), dtype=torch.uint8, 
                               device=tensor.device) 
                   for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    
    # Deserialize
    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:int(size.item())]
        data_list.append(pickle.loads(buffer))
    
    return data_list

# Usage in validation
outputs = self.validation_step_outputs  # Local outputs
all_outputs = all_gather(outputs)  # Gather from all GPUs
# Now all_outputs contains outputs from all processes
```

### Dataset Configuration

#### `get_num_class(hparams)` - Infer Number of Classes
```python
def get_num_class(hparams):
    """
    Infer number of classes from dataset name
    
    Supported datasets:
    - MS1MV2 (faces_emore): 85,742 classes (70,722 with subset)
    - MS1MV3 (ms1m-retinaface-t1): 93,431 classes
    - VGGFace2: 9,131 classes
    - CASIA-WebFace: 10,572 classes
    - WebFace4M: 205,990 classes
    - WebFace12M: 617,970 classes
    - WebFace42M: 2,059,906 classes
    """
    if hparams.custom_num_class != -1:
        return hparams.custom_num_class
    
    if 'faces_emore' in hparams.train_data_path.lower():
        return 70722 if hparams.train_data_subset else 85742
    elif 'ms1m-retinaface-t1' in hparams.train_data_path.lower():
        return 93431
    elif 'faces_vgg_112x112' in hparams.train_data_path.lower():
        return 9131
    elif 'faces_webface_112x112' in hparams.train_data_path.lower():
        return 10572
    elif 'webface4m' in hparams.train_data_path.lower():
        return 205990
    elif 'webface12m' in hparams.train_data_path.lower():
        return 617970
    elif 'webface42m' in hparams.train_data_path.lower():
        return 2059906
    else:
        raise ValueError(f'Unknown dataset: {hparams.train_data_path}')
```

---

## 2. inference.py - Model Inference

### Purpose
Provides utilities for loading pretrained models and extracting features for inference.

### Key Functions

#### `load_pretrained_model(architecture='ir_50')` - Load Model
```python
def load_pretrained_model(architecture='ir_50'):
    """
    Load pretrained AdaFace model
    
    Args:
        architecture: 'ir_18', 'ir_50', 'ir_101', etc.
    
    Returns:
        model: Loaded model in eval mode
    """
    assert architecture in adaface_models.keys()
    
    # Build model
    model = net.build_model(architecture)
    
    # Load checkpoint
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {
        key[6:]: val for key, val in statedict.items() 
        if key.startswith('model.')
    }
    model.load_state_dict(model_statedict)
    model.eval()
    
    return model

# Usage
model = load_pretrained_model('ir_50')
model = model.cuda()
```

#### `to_input(pil_rgb_image)` - Preprocess Image
```python
def to_input(pil_rgb_image):
    """
    Convert PIL RGB image to model input tensor
    
    Args:
        pil_rgb_image: PIL Image in RGB format
    
    Returns:
        tensor: [1, 3, 112, 112] tensor ready for model
    """
    np_img = np.array(pil_rgb_image)
    
    # RGB -> BGR and normalize
    brg_img = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5
    
    # To tensor [1, 3, H, W]
    tensor = torch.tensor([brg_img.transpose(2, 0, 1)]).float()
    
    return tensor

# Usage
from PIL import Image
img = Image.open('face.jpg')
aligned = align.get_aligned_face(img)  # 112x112
tensor = to_input(aligned)
feature, norm, _ = model(tensor.cuda())
```

### Complete Inference Pipeline

```python
import net
import torch
from face_alignment import align
from PIL import Image

# 1. Load model
model = net.build_model('ir_50')
checkpoint = torch.load('model.ckpt')
model.load_state_dict(checkpoint['state_dict'])
model = model.cuda().eval()

# 2. Load and align image
img_path = 'photo.jpg'
aligned_face = align.get_aligned_face(img_path)

# 3. Preprocess
np_img = np.array(aligned_face)
bgr_img = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5
tensor = torch.tensor([bgr_img.transpose(2, 0, 1)]).float()

# 4. Extract features
with torch.no_grad():
    embedding, norm, occlusion_map = model(tensor.cuda())

# 5. Compare with gallery
gallery_embeddings = load_gallery()  # [N, 512]
similarities = torch.mm(embedding, gallery_embeddings.t())
best_match_idx = similarities.argmax().item()
print(f"Best match: {gallery_labels[best_match_idx]}")
print(f"Similarity: {similarities[0, best_match_idx].item():.4f}")
```

### Batch Inference

```python
import os
from tqdm import tqdm

def extract_features_from_folder(folder_path, model, batch_size=32):
    """Extract features for all images in folder"""
    image_files = [f for f in os.listdir(folder_path) 
                   if f.endswith(('.jpg', '.png'))]
    
    all_features = []
    all_labels = []
    
    for i in tqdm(range(0, len(image_files), batch_size)):
        batch_files = image_files[i:i+batch_size]
        batch_tensors = []
        
        for fname in batch_files:
            path = os.path.join(folder_path, fname)
            aligned = align.get_aligned_face(path)
            
            if aligned:
                tensor = to_input(aligned)
                batch_tensors.append(tensor)
                all_labels.append(fname)
        
        if batch_tensors:
            batch = torch.cat(batch_tensors, dim=0).cuda()
            
            with torch.no_grad():
                features, _, _ = model(batch)
            
            all_features.append(features.cpu())
    
    return torch.cat(all_features, dim=0), all_labels

# Usage
features, labels = extract_features_from_folder('faces/', model)
# Save for later use
torch.save({'features': features, 'labels': labels}, 'gallery.pth')
```

---

## 3. graph_sampler.py - Graph-Based Sampling

### Purpose
Implements graph sampling for metric learning: samples identities based on similarity in feature space.

### Class: GraphSampler(Sampler)

**Concept**: Sample batches where identities are neighbors in the feature space graph, creating more informative batches for metric learning.

**Process**:
```
1. Extract features for one sample per identity
2. Compute pairwise similarities
3. For each identity, find K nearest neighbor identities
4. Sample instances from identity + its neighbors
5. Construct batches with similar identities
```

**Implementation**:
```python
class GraphSampler(Sampler):
    def __init__(self, data_source, img_path, transformer, 
                 model, matcher, batch_size=64, num_instance=4,
                 gal_batch_size=256, prob_batch_size=256):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instance = num_instance
        self.model = model
        self.matcher = matcher
        
        # Build identity index
        self.index_dic = defaultdict(list)
        for index, (_, pid, _, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        
        self.pids = list(self.index_dic.keys())
    
    def make_index(self):
        """Construct graph-based sampling index"""
        # 1. Sample one image per identity
        sam_index = []
        for pid in self.pids:
            index = np.random.choice(self.index_dic[pid], size=1)[0]
            sam_index.append(index)
        
        dataset = [self.data_source[i] for i in sam_index]
        
        # 2. Extract features and compute distances
        dist = self.calc_distance(dataset)
        
        # 3. Find K nearest neighbors for each identity
        num_ids = len(self.pids)
        dist = dist + torch.eye(num_ids) * 1e15  # Exclude self
        
        topk = self.batch_size // self.num_instance - 1
        _, topk_index = torch.topk(dist, topk, largest=False)
        
        # 4. Sample instances based on graph
        sam_index = []
        for i, pid in enumerate(self.pids):
            # Get neighbor identities
            neighbor_indices = topk_index[i, :].tolist()
            neighbor_indices.append(i)  # Include self
            
            # Sample num_instance from each neighbor
            for j in neighbor_indices:
                neighbor_pid = self.pids[j]
                instances = self.index_dic[neighbor_pid]
                
                # Circular sampling from instances
                selected = []
                while len(selected) < self.num_instance:
                    idx = instances[self.sam_pointer[neighbor_pid]]
                    selected.append(idx)
                    self.sam_pointer[neighbor_pid] += 1
                    
                    if self.sam_pointer[neighbor_pid] >= len(instances):
                        shuffle(instances)
                        self.sam_pointer[neighbor_pid] = 0
                
                sam_index.extend(selected)
        
        # 5. Shuffle and return
        sam_index = np.array(sam_index).reshape((-1, self.batch_size))
        np.random.shuffle(sam_index)
        self.sam_index = list(sam_index.flatten())
    
    def __iter__(self):
        self.make_index()
        return iter(self.sam_index)
    
    def __len__(self):
        return len(self.sam_index) if self.sam_index else self.num_pids
```

**Benefits**:
- **Harder Batches**: Identities are similar, creating harder negatives
- **Better Gradients**: More informative comparisons
- **Faster Convergence**: Learns discriminative features faster

**Usage**:
```python
from graph_sampler import GraphSampler

# Create sampler
sampler = GraphSampler(
    data_source=dataset,
    img_path='data/images',
    transformer=transform,
    model=model,
    matcher=qaconv,
    batch_size=64,
    num_instance=4
)

# Use with DataLoader
loader = DataLoader(
    dataset,
    batch_size=64,
    sampler=sampler,
    num_workers=8
)
```

**Comparison with RandomIdentitySampler**:

| Aspect | RandomIdentitySampler | GraphSampler |
|--------|----------------------|--------------|
| Sampling | Random identities | Similar identities |
| Batch Difficulty | Mixed | Harder (similar faces) |
| Convergence | Slower | Faster |
| Overhead | None | Feature extraction per epoch |
| Use Case | General training | Advanced metric learning |

---

## Usage Guidelines

### When to Use Utils

#### Distributed Training
```python
# Check if distributed
if utils.is_dist_avail_and_initialized():
    world_size = utils.get_world_size()
    rank = utils.get_local_rank()
    
    # Gather validation outputs
    outputs = utils.all_gather(local_outputs)
```

#### Feature Fusion
```python
# Test-time augmentation
original = model(images)
flipped = model(images.flip(-1))
stacked_embs = torch.stack([original[0], flipped[0]])
stacked_norms = torch.stack([original[1], flipped[1]])
fused_emb, fused_norm = utils.fuse_features_with_norm(
    stacked_embs, stacked_norms
)
```

### When to Use GraphSampler

**Recommended**:
- Advanced metric learning training
- After initial convergence with RandomIdentitySampler
- When computational resources allow

**Not Recommended**:
- Initial training (use RandomIdentitySampler first)
- Very large datasets (overhead too high)
- Limited GPU memory

---

## Performance Tips

### Inference Optimization
```python
# 1. Use half precision
model = model.half()
images = images.half()

# 2. Batch processing
batch_size = 128  # Larger for inference

# 3. Disable gradient computation
with torch.no_grad():
    features = model(images)

# 4. Use TorchScript
model = torch.jit.script(model)
```

### Distributed Training
```python
# DDP settings
torch.distributed.init_process_group(backend='nccl')
model = torch.nn.parallel.DistributedDataParallel(model)

# Efficient validation
if rank == 0:
    # Only rank 0 logs
    logger.log(metrics)
```

### Memory Efficiency
```python
# Use memory-mapped files
features = np.memmap('features.dat', dtype='float32', 
                     mode='r', shape=(N, 512))

# Chunk large computations
for i in range(0, N, chunk_size):
    chunk = features[i:i+chunk_size]
    process(chunk)
```

