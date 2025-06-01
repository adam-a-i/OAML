# OAML: Occlusion-Aware Metric Learning for Face Recognition

OAML is a novel face recognition framework that combines metric learning with occlusion-aware training to achieve robust performance under challenging conditions. Our solution is particularly effective for handling partial face occlusions, such as medical masks, while maintaining high accuracy on standard face recognition benchmarks.

## Key Features

- **Occlusion-Aware Training**: 
  - Medical mask occlusion simulation with realistic positioning
  - Configurable occlusion probability and mask parameters
  - Standard reference points for consistent alignment

- **Advanced Metric Learning**:
  - Dual-branch architecture combining global and local features
  - Pairwise matching loss for direct feature comparison
  - Triplet loss for improved feature discrimination
  - PK sampling strategy for effective batch construction

- **Robust Evaluation**:
  - Support for multiple face recognition benchmarks
  - Comprehensive metrics for occlusion scenarios
  - Combined scoring system for optimal performance

## Installation

```bash
pip install -r requirements.txt
```

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
├── qaconv.py           # Local feature matching
├── transforms.py        # Data augmentation transforms
├── train_val.py        # Training and validation logic
├── main.py             # Training entry point
├── main_eval.py        # Evaluation entry point
└── requirements.txt     # Project dependencies
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

### Medical Mask Occlusion
The `MedicalMaskOcclusion` transform simulates face occlusions using medical masks:
- Random mask positioning and scaling for realistic simulation
- Configurable occlusion probability (default: 0.5)
- Standard reference points for CASIA WebFace alignment
- Adjustable mask dimensions for different occlusion patterns

### Metric Learning Architecture
Our dual-branch architecture combines:
- Global feature branch for overall face representation
- Local feature branch for handling occlusions
- Feature fusion for robust recognition

### Loss Functions
- Pairwise Matching Loss: Direct comparison of feature pairs
- Triplet Loss: Enforces margin between positive and negative pairs
- Combined loss function with configurable weights

### PK Sampling
- Identity-aware batch sampling strategy
- Ensures K instances per identity in each batch
- Optimized for metric learning objectives
- Configurable batch size and instances per identity

## Citation

If you use this code in your research, please cite:

```bibtex
@article{oaml2024,
  title={OAML: Occlusion-Aware Metric Learning for Face Recognition},
  author={Adam Ibrahim, Shengcia Liao},
  journal={[Journal/Conference]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on AdaFace: [https://github.com/mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace)
- QAConv implementation inspired by: [https://github.com/scliao/QAConv](https://github.com/scliao/QAConv)


