#!/usr/bin/env python3
"""
Niqab Alignment with Adjusted Crop
=================================

This script detects faces and then manually adjusts the crop area
to include more of the niqab and avoid zooming too much into just the eyes.
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

def preprocess_for_niqab_detection(img):
    """
    Preprocess image specifically for niqab face detection
    """
    # Convert to numpy for processing
    img_array = np.array(img)
    
    # Strategy 1: Enhance contrast to make eyes more visible
    img_enhanced = ImageEnhance.Contrast(img).enhance(1.3)
    
    # Strategy 2: Slight brightness adjustment
    img_enhanced = ImageEnhance.Brightness(img_enhanced).enhance(1.1)
    
    # Strategy 3: Sharpen to enhance edges
    img_enhanced = ImageEnhance.Sharpness(img_enhanced).enhance(1.2)
    
    return img_enhanced

def get_face_bbox_with_landmarks(mtcnn_model, img):
    """
    Get face bounding box and landmarks for manual crop adjustment
    """
    try:
        # Get bounding boxes and landmarks
        boxes, landmarks = mtcnn_model.detect_faces(
            img, 
            mtcnn_model.min_face_size, 
            mtcnn_model.thresholds, 
            mtcnn_model.nms_thresholds, 
            mtcnn_model.factor
        )
        
        if len(boxes) > 0 and len(landmarks) > 0:
            return boxes[0], landmarks[0]
        else:
            return None, None
    except:
        return None, None

def adjust_crop_for_niqab(img, bbox, landmarks):
    """
    Adjust the crop area to include more of the niqab
    """
    if bbox is None or landmarks is None:
        return None
    
    x1, y1, x2, y2, confidence = bbox
    width = x2 - x1
    height = y2 - y1
    
    # Calculate expansion factors for niqab
    # Expand more vertically to include niqab
    vertical_expansion = 0.4  # 40% more height
    horizontal_expansion = 0.2  # 20% more width
    
    # Calculate new crop area
    new_width = width * (1 + horizontal_expansion)
    new_height = height * (1 + vertical_expansion)
    
    # Center the expanded area
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    new_x1 = max(0, center_x - new_width / 2)
    new_y1 = max(0, center_y - new_height / 2)
    new_x2 = min(img.width, center_x + new_width / 2)
    new_y2 = min(img.height, center_y + new_height / 2)
    
    # Crop the adjusted area
    adjusted_crop = img.crop((new_x1, new_y1, new_x2, new_y2))
    
    # Resize to target size
    return adjusted_crop.resize((112, 112), Image.LANCZOS)

def try_niqab_adjusted_strategies(mtcnn_model, img):
    """
    Try multiple strategies with adjusted crops for niqab
    """
    strategies = [
        # Strategy 1: Original image with adjusted crop
        ("Original+Adjusted", lambda x: try_adjusted_crop(mtcnn_model, x)),
        
        # Strategy 2: Enhanced contrast with adjusted crop
        ("Enhanced+Adjusted", lambda x: try_adjusted_crop(mtcnn_model, preprocess_for_niqab_detection(x))),
        
        # Strategy 3: Upper portion with adjusted crop
        ("Upper+Adjusted", lambda x: try_adjusted_crop(mtcnn_model, x.crop((0, 0, x.width, int(x.height*0.8))))),
        
        # Strategy 4: Enhanced upper portion with adjusted crop
        ("EnhancedUpper+Adjusted", lambda x: try_adjusted_crop(mtcnn_model, preprocess_for_niqab_detection(x).crop((0, 0, x.width, int(x.height*0.8))))),
    ]
    
    for strategy_name, strategy_func in strategies:
        try:
            aligned_face = strategy_func(img)
            if aligned_face is not None:
                return aligned_face, strategy_name
        except (IndexError, ValueError, Exception):
            continue
    
    return None, "All strategies failed"

def try_adjusted_crop(mtcnn_model, img):
    """
    Try to get face detection and then adjust the crop area
    """
    try:
        # Get face bounding box and landmarks
        bbox, landmarks = get_face_bbox_with_landmarks(mtcnn_model, img)
        
        if bbox is not None and landmarks is not None:
            # Adjust crop for niqab
            return adjust_crop_for_niqab(img, bbox, landmarks)
        else:
            return None
    except:
        return None

def align_niqab_adjusted_crop(input_dir, output_dir, crop_size=(112, 112), device='cuda:0'):
    """
    Align niqab images with adjusted crops to include more niqab area
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize MTCNN with settings optimized for niqab
    print(f"Initializing MTCNN on {device}...")
    mtcnn_model = MTCNN(device=device, crop_size=crop_size)
    
    # Lenient thresholds for niqab detection
    mtcnn_model.thresholds = [0.2, 0.3, 0.4]  # Very lenient
    mtcnn_model.min_face_size = 8  # Small minimum face size
    mtcnn_model.nms_thresholds = [0.3, 0.3, 0.3]  # More lenient NMS
    mtcnn_model.factor = 0.7  # Smaller factor for more scales
    
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
    
    for img_path in tqdm(image_files, desc="Aligning faces (adjusted crop)"):
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Try multiple strategies with adjusted crops
            aligned_face, strategy_used = try_niqab_adjusted_strategies(mtcnn_model, img)
            
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
    
    print(f"\nAdjusted crop alignment complete!")
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
    parser = argparse.ArgumentParser(description='Align niqab images with adjusted crops')
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
    print("Using MTCNN with adjusted crop strategies for niqab images")
    
    align_niqab_adjusted_crop(
        args.input_dir, 
        args.output_dir, 
        tuple(args.crop_size), 
        args.device
    )

if __name__ == "__main__":
    main()
