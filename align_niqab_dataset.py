#!/usr/bin/env python3
"""
Niqab Dataset Face Alignment Script
====================================

This script uses the MTCNN pipeline to align faces in a niqab dataset.
It processes all images in a directory and saves aligned faces to an output directory.
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
import shutil

# Add current directory to path
sys.path.append('.')

from face_alignment.mtcnn import MTCNN

def align_niqab_dataset(input_dir, output_dir, crop_size=(112, 112), device='cuda:0'):
    """
    Align faces in niqab dataset using MTCNN
    
    Args:
        input_dir: Directory containing niqab images
        output_dir: Directory to save aligned faces
        crop_size: Size of aligned face (width, height)
        device: Device to use for MTCNN ('cuda:0' or 'cpu')
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize MTCNN with more lenient thresholds for niqab images
    print(f"Initializing MTCNN on {device}...")
    mtcnn_model = MTCNN(device=device, crop_size=crop_size)
    
    # Lower thresholds for better niqab detection
    mtcnn_model.thresholds = [0.4, 0.5, 0.6]  # More lenient
    mtcnn_model.min_face_size = 10  # Smaller minimum face size
    
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
    failed_files = []
    
    for img_path in tqdm(image_files, desc="Aligning faces"):
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Try to align face with error handling
            try:
                aligned_face = mtcnn_model.align(img)
                
                if aligned_face is not None:
                    # Save aligned face
                    output_path = Path(output_dir) / img_path.name
                    aligned_face.save(output_path)
                    successful_alignments += 1
                else:
                    failed_alignments += 1
                    failed_files.append(img_path.name)
                    
            except (IndexError, ValueError) as e:
                # Handle the specific "list index out of range" errors
                failed_alignments += 1
                failed_files.append(img_path.name)
                # Don't print every error to avoid spam
                
        except Exception as e:
            failed_alignments += 1
            failed_files.append(img_path.name)
            print(f"Error processing {img_path.name}: {e}")
    
    print(f"\nAlignment complete!")
    print(f"Successfully aligned: {successful_alignments}")
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

def align_with_multiple_faces(input_dir, output_dir, crop_size=(112, 112), device='cuda:0', max_faces=1):
    """
    Align faces in niqab dataset, handling multiple faces per image
    
    Args:
        input_dir: Directory containing niqab images
        output_dir: Directory to save aligned faces
        crop_size: Size of aligned face (width, height)
        device: Device to use for MTCNN ('cuda:0' or 'cpu')
        max_faces: Maximum number of faces to extract per image
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize MTCNN
    print(f"Initializing MTCNN on {device}...")
    mtcnn_model = MTCNN(device=device, crop_size=crop_size)
    
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
    
    for img_path in tqdm(image_files, desc="Aligning faces"):
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Detect and align multiple faces
            boxes, faces = mtcnn_model.align_multi(img, limit=max_faces)
            
            if len(faces) > 0:
                # Save each detected face
                base_name = img_path.stem
                ext = img_path.suffix
                
                for i, face in enumerate(faces):
                    if i == 0:
                        # First face keeps original name
                        output_path = Path(output_dir) / img_path.name
                    else:
                        # Additional faces get numbered
                        output_path = Path(output_dir) / f"{base_name}_face_{i+1}{ext}"
                    
                    face.save(output_path)
                
                successful_alignments += 1
            else:
                print(f"No faces detected in: {img_path.name}")
                failed_alignments += 1
                
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            failed_alignments += 1
    
    print(f"\nAlignment complete!")
    print(f"Successfully processed: {successful_alignments}")
    print(f"Failed alignments: {failed_alignments}")
    print(f"Aligned faces saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Align faces in niqab dataset using MTCNN')
    parser.add_argument('input_dir', help='Directory containing niqab images')
    parser.add_argument('output_dir', help='Directory to save aligned faces')
    parser.add_argument('--crop_size', nargs=2, type=int, default=[112, 112], 
                       help='Size of aligned face (width height)')
    parser.add_argument('--device', default='cuda:0', choices=['cuda:0', 'cpu'],
                       help='Device to use for MTCNN')
    parser.add_argument('--multi_face', action='store_true',
                       help='Extract multiple faces per image')
    parser.add_argument('--max_faces', type=int, default=1,
                       help='Maximum number of faces to extract per image')
    
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
    
    if args.multi_face:
        print(f"Multi-face mode: extracting up to {args.max_faces} faces per image")
        align_with_multiple_faces(
            args.input_dir, 
            args.output_dir, 
            tuple(args.crop_size), 
            args.device,
            args.max_faces
        )
    else:
        align_niqab_dataset(
            args.input_dir, 
            args.output_dir, 
            tuple(args.crop_size), 
            args.device
        )

if __name__ == "__main__":
    main()
