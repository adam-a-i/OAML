#!/usr/bin/env python3
"""
Robust Niqab Dataset Face Alignment Script
==========================================

This script uses multiple strategies to align faces in niqab images:
1. Standard MTCNN with lenient thresholds
2. Manual cropping if face detection fails
3. Alternative face detection methods
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image, ImageOps
import torch
from tqdm import tqdm
import numpy as np

# Add current directory to path
sys.path.append('.')

from face_alignment.mtcnn import MTCNN

def manual_crop_niqab(img, crop_ratio=0.3):
    """
    Manually crop niqab images when face detection fails.
    Assumes the face is in the center-upper portion of the image.
    
    Args:
        img: PIL Image
        crop_ratio: Ratio of image to crop (0.3 = crop to 30% of original)
    
    Returns:
        Cropped PIL Image
    """
    width, height = img.size
    
    # Calculate crop dimensions
    crop_width = int(width * crop_ratio)
    crop_height = int(height * crop_ratio)
    
    # Center the crop horizontally, position it in upper portion
    left = (width - crop_width) // 2
    top = int(height * 0.1)  # Start from 10% down from top
    right = left + crop_width
    bottom = top + crop_height
    
    # Ensure we don't go out of bounds
    right = min(right, width)
    bottom = min(bottom, height)
    
    # Crop and resize to standard size
    cropped = img.crop((left, top, right, bottom))
    resized = cropped.resize((112, 112), Image.LANCZOS)
    
    return resized

def align_niqab_robust(input_dir, output_dir, crop_size=(112, 112), device='cuda:0'):
    """
    Robust alignment for niqab dataset with multiple fallback strategies
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize MTCNN with very lenient settings
    print(f"Initializing MTCNN on {device}...")
    mtcnn_model = MTCNN(device=device, crop_size=crop_size)
    
    # Very lenient thresholds for niqab detection
    mtcnn_model.thresholds = [0.3, 0.4, 0.5]  # Very lenient
    mtcnn_model.min_face_size = 8  # Very small minimum face size
    mtcnn_model.nms_thresholds = [0.5, 0.5, 0.5]  # More lenient NMS
    
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
    manual_crops = 0
    failed_files = []
    
    for img_path in tqdm(image_files, desc="Aligning faces"):
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Strategy 1: Try MTCNN alignment
            aligned_face = None
            try:
                aligned_face = mtcnn_model.align(img)
            except (IndexError, ValueError, Exception):
                # MTCNN failed, try manual cropping
                pass
            
            if aligned_face is not None:
                # MTCNN succeeded
                output_path = Path(output_dir) / img_path.name
                aligned_face.save(output_path)
                successful_alignments += 1
            else:
                # Strategy 2: Manual cropping
                try:
                    manual_cropped = manual_crop_niqab(img)
                    output_path = Path(output_dir) / img_path.name
                    manual_cropped.save(output_path)
                    successful_alignments += 1
                    manual_crops += 1
                except Exception as e:
                    failed_alignments += 1
                    failed_files.append(img_path.name)
                    print(f"Failed to process {img_path.name}: {e}")
                
        except Exception as e:
            failed_alignments += 1
            failed_files.append(img_path.name)
            print(f"Error processing {img_path.name}: {e}")
    
    print(f"\nRobust alignment complete!")
    print(f"Successfully aligned: {successful_alignments}")
    print(f"  - MTCNN alignments: {successful_alignments - manual_crops}")
    print(f"  - Manual crops: {manual_crops}")
    print(f"Failed alignments: {failed_alignments}")
    print(f"Success rate: {successful_alignments/(successful_alignments+failed_alignments)*100:.1f}%")
    print(f"Aligned faces saved to: {output_dir}")
    
    # Save list of failed files for debugging
    if failed_files:
        failed_log_path = Path(output_dir) / "failed_alignments.txt"
        with open(failed_log_path, 'w') as f:
            for filename in failed_files:
                f.write(f"{filename}\n")
        print(f"List of failed files saved to: {failed_log_path}")

def main():
    parser = argparse.ArgumentParser(description='Robust face alignment for niqab dataset')
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
    print("Using robust alignment with MTCNN + manual cropping fallback")
    
    align_niqab_robust(
        args.input_dir, 
        args.output_dir, 
        tuple(args.crop_size), 
        args.device
    )

if __name__ == "__main__":
    main()
