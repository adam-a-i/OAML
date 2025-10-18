#!/usr/bin/env python3
"""
Keep Good Niqab Images Script
============================

Very lenient filtering for niqab images - only removes obviously bad ones.
Focuses on basic image quality without requiring face feature detection.
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageStat
import numpy as np
from tqdm import tqdm

def is_obviously_bad(image_path):
    """
    Check if image is obviously bad - very lenient criteria
    Only removes images that are clearly not faces
    """
    try:
        img = Image.open(image_path).convert('RGB')
        
        # Calculate basic statistics
        stat = ImageStat.Stat(img)
        mean_brightness = stat.mean[0]
        std_dev = stat.stddev[0]
        
        # Only remove if image is extremely dark (likely no content)
        if mean_brightness < 20:
            return True, "Extremely dark"
        
        # Only remove if image is extremely bright (likely overexposed)
        if mean_brightness > 220:
            return True, "Extremely bright"
        
        # Only remove if image has almost no variation (likely empty)
        if std_dev < 10:
            return True, "No variation"
        
        # Check if image is mostly one color (likely bad alignment)
        img_array = np.array(img)
        unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[2]), axis=0))
        if unique_colors < 20:  # Very few unique colors
            return True, "Too few colors"
        
        # Check for reasonable aspect ratio (not too extreme)
        width, height = img.size
        aspect_ratio = width / height
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:  # Very extreme ratios
            return True, "Bad aspect ratio"
        
        # Check if image is too small (likely bad alignment)
        if width < 50 or height < 50:
            return True, "Too small"
        
        return False, "Keep"
        
    except Exception as e:
        return True, f"Error: {e}"

def keep_good_niqab_images(input_dir, output_dir):
    """
    Keep only obviously good niqab images - very lenient filtering
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_files)} niqab aligned images to filter")
    
    if len(image_files) == 0:
        print("No images found in input directory!")
        return
    
    # Process each image
    kept_images = 0
    removed_images = 0
    removal_reasons = {}
    
    for img_path in tqdm(image_files, desc="Filtering niqab images"):
        try:
            is_bad, reason = is_obviously_bad(img_path)
            
            if is_bad:
                # Remove bad image
                os.remove(img_path)
                removed_images += 1
                
                if reason not in removal_reasons:
                    removal_reasons[reason] = 0
                removal_reasons[reason] += 1
                
                print(f"Removed: {img_path.name} - {reason}")
            else:
                # Keep good image - copy to output directory
                output_path = Path(output_dir) / img_path.name
                import shutil
                shutil.copy2(img_path, output_path)
                kept_images += 1
                
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            removed_images += 1
    
    # Print statistics
    print(f"\nNiqab filtering complete!")
    print(f"Images kept: {kept_images}")
    print(f"Images removed: {removed_images}")
    
    if removal_reasons:
        print(f"\nRemoval reasons:")
        for reason, count in removal_reasons.items():
            print(f"  {reason}: {count} images")
    
    print(f"Good images saved to: {output_dir}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Keep good niqab images with very lenient filtering')
    parser.add_argument('input_dir', help='Directory containing aligned images')
    parser.add_argument('output_dir', help='Directory to save good images')
    parser.add_argument('--preview', action='store_true',
                       help='Preview mode - show what would be removed without deleting')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist!")
        return
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print("Using very lenient filtering for niqab images")
    
    if args.preview:
        print("PREVIEW MODE - No files will be moved or deleted")
        # Just analyze and report
        image_files = list(Path(args.input_dir).glob("*.jpg")) + list(Path(args.input_dir).glob("*.jpeg"))
        bad_count = 0
        for img_path in tqdm(image_files, desc="Previewing niqab images"):
            is_bad, reason = is_obviously_bad(img_path)
            if is_bad:
                print(f"Would remove: {img_path.name} - {reason}")
                bad_count += 1
        
        print(f"\nPreview complete: {bad_count} images would be removed")
    else:
        keep_good_niqab_images(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
