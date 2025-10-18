#!/usr/bin/env python3
"""
Simple Bad Alignment Removal Script
==================================

This script removes obviously bad aligned images by checking for:
1. Images that are too dark (no face visible)
2. Images that are too uniform (no features)
3. Images that are too bright (overexposed)
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageStat
import numpy as np
from tqdm import tqdm

def is_bad_alignment(image_path):
    """
    Check if an aligned image is bad quality
    Returns True if image should be removed
    """
    try:
        img = Image.open(image_path).convert('RGB')
        
        # Calculate basic statistics
        stat = ImageStat.Stat(img)
        mean_brightness = stat.mean[0]
        std_dev = stat.stddev[0]
        
        # Check if image is too dark (likely no face visible)
        if mean_brightness < 40:
            return True, "Too dark"
        
        # Check if image is too bright (overexposed)
        if mean_brightness > 180:
            return True, "Too bright"
        
        # Check if image has too low variation (likely uniform/empty)
        if std_dev < 20:
            return True, "Too uniform"
        
        # Check if image is mostly one color (likely bad alignment)
        img_array = np.array(img)
        unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[2]), axis=0))
        if unique_colors < 100:  # Very few unique colors
            return True, "Too few colors"
        
        return False, "Good"
        
    except Exception as e:
        return True, f"Error: {e}"

def remove_bad_alignments(input_dir, min_quality_threshold=40):
    """
    Remove bad aligned images from directory
    
    Args:
        input_dir: Directory containing aligned images
        min_quality_threshold: Minimum brightness threshold
    """
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_files)} aligned images to check")
    
    if len(image_files) == 0:
        print("No images found in input directory!")
        return
    
    # Process each image
    good_images = 0
    bad_images = 0
    removal_reasons = {}
    
    for img_path in tqdm(image_files, desc="Checking images"):
        try:
            is_bad, reason = is_bad_alignment(img_path)
            
            if is_bad:
                # Remove bad image
                os.remove(img_path)
                bad_images += 1
                
                if reason not in removal_reasons:
                    removal_reasons[reason] = 0
                removal_reasons[reason] += 1
                
                print(f"Removed: {img_path.name} - {reason}")
            else:
                good_images += 1
                
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
    
    # Print statistics
    print(f"\nCleanup complete!")
    print(f"Good images kept: {good_images}")
    print(f"Bad images removed: {bad_images}")
    
    if removal_reasons:
        print(f"\nRemoval reasons:")
        for reason, count in removal_reasons.items():
            print(f"  {reason}: {count} images")
    
    print(f"Remaining images in: {input_dir}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Remove bad aligned face images')
    parser.add_argument('input_dir', help='Directory containing aligned images')
    parser.add_argument('--threshold', type=int, default=40,
                       help='Minimum brightness threshold (default: 40)')
    parser.add_argument('--preview', action='store_true',
                       help='Preview mode - show what would be removed without deleting')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist!")
        return
    
    print(f"Input directory: {args.input_dir}")
    print(f"Brightness threshold: {args.threshold}")
    
    if args.preview:
        print("PREVIEW MODE - No files will be deleted")
        # Just analyze and report
        image_files = list(Path(args.input_dir).glob("*.jpg")) + list(Path(args.input_dir).glob("*.jpeg"))
        bad_count = 0
        for img_path in tqdm(image_files, desc="Previewing images"):
            is_bad, reason = is_bad_alignment(img_path)
            if is_bad:
                print(f"Would remove: {img_path.name} - {reason}")
                bad_count += 1
        
        print(f"\nPreview complete: {bad_count} images would be removed")
    else:
        remove_bad_alignments(args.input_dir, args.threshold)

if __name__ == "__main__":
    main()
