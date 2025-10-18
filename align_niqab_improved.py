#!/usr/bin/env python3
"""
Improved Niqab Dataset Face Alignment Script
============================================

This script optimizes MTCNN specifically for niqab face detection by:
1. Using more lenient detection thresholds
2. Trying multiple detection strategies
3. Preprocessing images to improve detection
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
import torch
from tqdm import tqdm
import numpy as np

# Add current directory to path
sys.path.append('.')

from face_alignment.mtcnn import MTCNN

def preprocess_for_niqab(img):
    """
    Preprocess image to improve niqab face detection
    """
    # Convert to numpy for processing
    img_array = np.array(img)
    
    # Enhance contrast to make eyes more visible
    img_enhanced = ImageEnhance.Contrast(img).enhance(1.2)
    
    # Slight brightness adjustment
    img_enhanced = ImageEnhance.Brightness(img_enhanced).enhance(1.1)
    
    return img_enhanced

def try_multiple_detection_strategies(mtcnn_model, img):
    """
    Try multiple detection strategies for niqab images
    """
    strategies = [
        # Strategy 1: Original image
        lambda x: mtcnn_model.align(x),
        
        # Strategy 2: Enhanced contrast
        lambda x: mtcnn_model.align(preprocess_for_niqab(x)),
        
        # Strategy 3: Slightly larger image (helps with small faces)
        lambda x: mtcnn_model.align(x.resize((int(x.width*1.2), int(x.height*1.2)), Image.LANCZOS)),
        
        # Strategy 4: Enhanced + larger
        lambda x: mtcnn_model.align(preprocess_for_niqab(x).resize((int(x.width*1.2), int(x.height*1.2)), Image.LANCZOS)),
    ]
    
    for i, strategy in enumerate(strategies):
        try:
            aligned_face = strategy(img)
            if aligned_face is not None:
                return aligned_face, f"Strategy {i+1}"
        except (IndexError, ValueError, Exception):
            continue
    
    return None, "All strategies failed"

def align_niqab_improved(input_dir, output_dir, crop_size=(112, 112), device='cuda:0'):
    """
    Improved alignment for niqab dataset with optimized MTCNN settings
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize MTCNN with optimized settings for niqab
    print(f"Initializing MTCNN on {device}...")
    mtcnn_model = MTCNN(device=device, crop_size=crop_size)
    
    # Optimized thresholds for niqab detection
    mtcnn_model.thresholds = [0.3, 0.4, 0.5]  # Very lenient
    mtcnn_model.min_face_size = 8  # Very small minimum face size
    mtcnn_model.nms_thresholds = [0.4, 0.4, 0.4]  # More lenient NMS
    mtcnn_model.factor = 0.8  # Smaller factor for more scales
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_files)} images to process")
    
    if len(image_files) == 0:
        print("No images found in input directory!")
        return
    
    # Process each image
    successful_alignments = 0
    failed_alignments = 0
    strategy_stats = {}
    failed_files = []
    
    for img_path in tqdm(image_files, desc="Aligning faces"):
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Try multiple detection strategies
            aligned_face, strategy_used = try_multiple_detection_strategies(mtcnn_model, img)
            
            if aligned_face is not None:
                # Save aligned face
                output_path = Path(output_dir) / img_path.name
                aligned_face.save(output_path)
                successful_alignments += 1
                
                # Track which strategy worked
                if strategy_used not in strategy_stats:
                    strategy_stats[strategy_used] = 0
                strategy_stats[strategy_used] += 1
            else:
                failed_alignments += 1
                failed_files.append(img_path.name)
                
        except Exception as e:
            failed_alignments += 1
            failed_files.append(img_path.name)
            print(f"Error processing {img_path.name}: {e}")
    
    print(f"\nImproved alignment complete!")
    print(f"Successfully aligned: {successful_alignments}")
    print(f"Failed alignments: {failed_alignments}")
    print(f"Success rate: {successful_alignments/(successful_alignments+failed_alignments)*100:.1f}%")
    print(f"Aligned faces saved to: {output_dir}")
    
    # Print strategy statistics
    if strategy_stats:
        print(f"\nDetection strategy statistics:")
        for strategy, count in strategy_stats.items():
            print(f"  {strategy}: {count} images")
    
    # Save list of failed files for debugging
    if failed_files:
        failed_log_path = Path(output_dir) / "failed_alignments.txt"
        with open(failed_log_path, 'w') as f:
            for filename in failed_files:
                f.write(f"{filename}\n")
        print(f"List of failed files saved to: {failed_log_path}")

def main():
    parser = argparse.ArgumentParser(description='Improved face alignment for niqab dataset')
    parser.add_argument('input_dir', help='Directory containing niqab images')
    parser.add_argument('output_dir', help='Directory to save aligned faces')
    parser.add_argument('--crop_size', nargs=2, type=int, default=[112, 112], 
                       help='Size of aligned face (width height)')
    parser.add_argument('--device', default='cuda:0', choices=['cuda:0', 'cpu'],
                       help='Device to use for MTCNN')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist!")
        return
    
    # Check device availability
    if args.device == 'cuda:0' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU...")
        args.device = 'cpu'
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Crop size: {args.crop_size}")
    print(f"Device: {args.device}")
    print("Using improved MTCNN detection with multiple strategies")
    
    align_niqab_improved(
        args.input_dir, 
        args.output_dir, 
        tuple(args.crop_size), 
        args.device
    )

if __name__ == "__main__":
    main()
