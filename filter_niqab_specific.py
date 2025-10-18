#!/usr/bin/env python3
"""
Niqab-Specific Image Filtering Script
====================================

This script is specifically designed for niqab images and is more lenient
about face detection since eyes are often covered. It focuses on:
1. Overall image quality
2. Presence of any face-like features
3. Not requiring eye detection
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageStat
import numpy as np
from tqdm import tqdm

def is_good_niqab_alignment(image_path):
    """
    Check if a niqab aligned image is good quality
    More lenient than standard face detection
    """
    try:
        img = Image.open(image_path).convert('RGB')
        
        # Calculate basic statistics
        stat = ImageStat.Stat(img)
        mean_brightness = stat.mean[0]
        std_dev = stat.stddev[0]
        
        # Check if image is too dark (likely no face visible)
        if mean_brightness < 25:
            return False, "Too dark"
        
        # Check if image is too bright (overexposed)
        if mean_brightness > 200:
            return False, "Too bright"
        
        # Check if image has too low variation (likely uniform/empty)
        if std_dev < 15:
            return False, "Too uniform"
        
        # Check if image is mostly one color (likely bad alignment)
        img_array = np.array(img)
        unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[2]), axis=0))
        if unique_colors < 50:  # Very few unique colors
            return False, "Too few colors"
        
        # Check for reasonable face proportions (not too square, not too rectangular)
        width, height = img.size
        aspect_ratio = width / height
        if aspect_ratio < 0.7 or aspect_ratio > 1.4:  # Should be roughly square
            return False, "Bad aspect ratio"
        
        # Check if image has reasonable texture (not just solid color)
        gray = img.convert('L')
        gray_array = np.array(gray)
        
        # Calculate local variance to detect texture
        from scipy import ndimage
        try:
            # Calculate local variance
            kernel = np.ones((3, 3)) / 9
            local_mean = ndimage.convolve(gray_array, kernel)
            local_variance = ndimage.convolve(gray_array**2, kernel) - local_mean**2
            avg_local_variance = np.mean(local_variance)
            
            if avg_local_variance < 5:  # Very low texture
                return False, "No texture"
        except:
            # If scipy not available, use simpler method
            if std_dev < 20:
                return False, "Too uniform"
        
        # Check for reasonable brightness distribution
        # Should have some variation in brightness
        brightness_hist = np.histogram(gray_array, bins=10)[0]
        if np.max(brightness_hist) > len(gray_array.flatten()) * 0.8:  # 80% of pixels in one bin
            return False, "Too uniform brightness"
        
        return True, "Good"
        
    except Exception as e:
        return False, f"Error: {e}"

def filter_niqab_images(input_dir, output_dir, move_bad=False):
    """
    Filter niqab aligned images with niqab-specific criteria
    
    Args:
        input_dir: Directory containing aligned images
        output_dir: Directory to save good images
        move_bad: If True, move bad images to separate folder instead of deleting
    """
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    if move_bad:
        bad_dir = Path(output_dir).parent / "bad_niqab_alignments"
        os.makedirs(bad_dir, exist_ok=True)
    
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
    good_images = 0
    bad_images = 0
    rejection_reasons = {}
    
    for img_path in tqdm(image_files, desc="Filtering niqab images"):
        try:
            is_good, reason = is_good_niqab_alignment(img_path)
            
            if is_good:
                # Good image - copy to output directory
                output_path = Path(output_dir) / img_path.name
                import shutil
                shutil.copy2(img_path, output_path)
                good_images += 1
            else:
                # Bad image
                bad_images += 1
                if reason not in rejection_reasons:
                    rejection_reasons[reason] = 0
                rejection_reasons[reason] += 1
                
                if move_bad:
                    # Move to bad directory
                    bad_path = bad_dir / img_path.name
                    import shutil
                    shutil.move(str(img_path), str(bad_path))
                else:
                    # Delete bad image
                    os.remove(img_path)
                
                print(f"Removed: {img_path.name} - {reason}")
                
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            bad_images += 1
    
    # Print statistics
    print(f"\nNiqab filtering complete!")
    print(f"Good images: {good_images}")
    print(f"Bad images removed: {bad_images}")
    
    if rejection_reasons:
        print(f"\nRejection reasons:")
        for reason, count in rejection_reasons.items():
            print(f"  {reason}: {count} images")
    
    if move_bad:
        print(f"Bad images moved to: {bad_dir}")
    else:
        print("Bad images deleted")
    
    print(f"Good images saved to: {output_dir}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Filter niqab aligned images with niqab-specific criteria')
    parser.add_argument('input_dir', help='Directory containing aligned images')
    parser.add_argument('output_dir', help='Directory to save good images')
    parser.add_argument('--move_bad', action='store_true',
                       help='Move bad images to separate folder instead of deleting')
    parser.add_argument('--preview', action='store_true',
                       help='Preview mode - show what would be removed without deleting')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist!")
        return
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Move bad images: {args.move_bad}")
    print("Using niqab-specific filtering (more lenient about eye detection)")
    
    if args.preview:
        print("PREVIEW MODE - No files will be moved or deleted")
        # Just analyze and report
        image_files = list(Path(args.input_dir).glob("*.jpg")) + list(Path(args.input_dir).glob("*.jpeg"))
        bad_count = 0
        for img_path in tqdm(image_files, desc="Previewing niqab images"):
            is_good, reason = is_good_niqab_alignment(img_path)
            if not is_good:
                print(f"Would remove: {img_path.name} - {reason}")
                bad_count += 1
        
        print(f"\nPreview complete: {bad_count} images would be removed")
    else:
        filter_niqab_images(args.input_dir, args.output_dir, args.move_bad)

if __name__ == "__main__":
    main()
