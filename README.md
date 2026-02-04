# OAML: Occlusion-Aware Metric Learning for Face Recognition

OAML is a face recognition framework that combines global and local feature matching with occlusion-aware training to achieve robust performance under challenging conditions, particularly for partially occluded faces.

## Architecture

![OAML Architecture](image%20(6).png)

The model uses a multi-task learning framework with a shared IR-50 backbone and three branches:

1. **AdaFace Branch**: Global feature-based recognition using adaptive margin softmax loss
2. **QAConv Branch**: Local feature matching with occlusion-aware spatial correlation
3. **Occlusion Prediction Branch**: Lightweight CNN head for spatial occlusion prediction

The total loss combines all three branches:
```
L_total = λ_ada × L_AdaFace + λ_qa × L_QAConv + λ_occ × L_Occlusion
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python main.py --data_root /path/to/data --train_data_path faces_emore --val_data_path faces_emore
```

### Evaluation

```bash
python main_eval.py --model-path /path/to/model --data-path /path/to/test/data
```

## Key Components

### Backbone
- IR-50 (ResNet-50 variant) extracts 7×7×512 feature maps
- Shared features feed all three branches

### AdaFace Branch
- Global pooling (GNAP/GDC) aggregates spatial features
- Produces 512-dimensional embeddings
- Uses adaptive margin softmax loss based on image quality

### QAConv Branch
- Local feature matching via spatial correlation
- Occlusion maps weight correlation scores
- Graph sampling (k-nearest neighbors) for efficient training
- Combines classification loss and triplet loss

### Occlusion Prediction
- Lightweight CNN head (OcclusionHead) predicts 7×7 occlusion maps
- Values: 1 = visible, 0 = occluded
- Trained with MSE loss against ground truth masks

## Project Structure

```
OAML/
├── dataset/              # Dataset handling code
├── face_alignment/       # Face alignment utilities
├── assets/              # Model weights and resources
├── scripts/             # Utility scripts
├── validation_*/        # Validation datasets
├── data.py             # Data loading and preprocessing
├── net.py              # Network architecture
├── qaconv.py           # QAConv local feature matching
├── transforms.py        # Data augmentation transforms
├── train_val.py        # Training and validation logic
├── main.py             # Training entry point
├── main_eval.py        # Evaluation entry point
└── requirements.txt     # Project dependencies
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
