#!/usr/bin/env python3
"""
Filter Bad Alignments Script
============================

This script analyzes aligned face images and removes those that are poorly aligned
or contain no visible face features. It uses multiple quality metrics to detect
bad alignments automatically.
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image, ImageStat
import numpy as np
import shutil
from tqdm import tqdm
import cv2

def calculate_brightness(image):
    """Calculate average brightness of image"""
    stat = ImageStat.Stat(image)
    return stat.mean[0]  # Average of all channels

def calculate_contrast(image):
    """Calculate contrast of image"""
    stat = ImageStat.Stat(image)
    return stat.stddev[0]  # Standard deviation

def detect_face_features(image):
    """
    Detect if image contains visible face features using OpenCV
    """
    # Convert PIL to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Try to detect eyes
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 3)
    
    # Try to detect face
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    
    return len(eyes) > 0 or len(faces) > 0

def analyze_image_quality(image_path):
    """
    Analyze the quality of an aligned face image
    Returns a quality score and reasons for rejection
    """
    try:
        img = Image.open(image_path).convert('RGB')
        
        # Basic quality metrics
        brightness = calculate_brightness(img)
        contrast = calculate_contrast(img)
        
        # Check if image is too dark (likely no face visible)
        if brightness < 30:
            return 0, "Too dark"
        
        # Check if image is too bright (likely overexposed)
        if brightness > 200:
            return 0, "Too bright"
        
        # Check if image has too low contrast (likely uniform/empty)
        if contrast < 15:
            return 0, "Too low contrast"
        
        # Check if image is too uniform (likely no face)
        img_array = np.array(img)
        if np.std(img_array) < 10:
            return 0, "Too uniform"
        
        # Try to detect face features
        has_features = detect_face_features(img)
        if not has_features:
            return 0, "No face features detected"
        
        # Calculate quality score (0-100)
        quality_score = 0
        
        # Brightness score (optimal range 50-150)
        if 50 <= brightness <= 150:
            quality_score += 30
        elif 30 <= brightness < 50 or 150 < brightness <= 200:
            quality_score += 20
        else:
            quality_score += 10
        
        # Contrast score
        if contrast > 30:
            quality_score += 30
        elif contrast > 20:
            quality_score += 20
        else:
            quality_score += 10
        
        # Face features score
        if has_features:
            quality_score += 40
        
        return quality_score, "Good"
        
    except Exception as e:
        return 0, f"Error: {e}"

def filter_aligned_images(input_dir, output_dir, min_quality=50, move_bad=False):
    """
    Filter aligned images based on quality metrics
    
    Args:
        input_dir: Directory containing aligned images
        output_dir: Directory to save good images
        min_quality: Minimum quality score (0-100)
        move_bad: If True, move bad images to separate folder instead of deleting
    """
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    if move_bad:
        bad_dir = Path(output_dir).parent / "bad_alignments"
        os.makedirs(bad_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f"*{ext}"))
        image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_files)} aligned images to filter")
    
    if len(image_files) == 0:
        print("No images found in input directory!")
        return
    
    # Process each image
    good_images = 0
    bad_images = 0
    quality_scores = []
    rejection_reasons = {}
    
    for img_path in tqdm(image_files, desc="Filtering images"):
        try:
            # Analyze image quality
            quality_score, reason = analyze_image_quality(img_path)
            quality_scores.append(quality_score)
            
            if quality_score >= min_quality:
                # Good image - copy to output directory
                output_path = Path(output_dir) / img_path.name
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
                    shutil.move(str(img_path), str(bad_path))
                else:
                    # Delete bad image
                    os.remove(img_path)
                
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            bad_images += 1
    
    # Print statistics
    print(f"\nFiltering complete!")
    print(f"Good images: {good_images}")
    print(f"Bad images removed: {bad_images}")
    print(f"Quality threshold: {min_quality}")
    
    if quality_scores:
        print(f"Average quality score: {np.mean(quality_scores):.1f}")
        print(f"Quality score range: {min(quality_scores):.1f} - {max(quality_scores):.1f}")
    
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
    parser = argparse.ArgumentParser(description='Filter bad aligned face images')
    parser.add_argument('input_dir', help='Directory containing aligned images')
    parser.add_argument('output_dir', help='Directory to save good images')
    parser.add_argument('--min_quality', type=int, default=50, 
                       help='Minimum quality score (0-100, default: 50)')
    parser.add_argument('--move_bad', action='store_true',
                       help='Move bad images to separate folder instead of deleting')
    parser.add_argument('--preview', action='store_true',
                       help='Preview mode - analyze but don\'t move/delete files')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist!")
        return
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Minimum quality: {args.min_quality}")
    print(f"Move bad images: {args.move_bad}")
    
    if args.preview:
        print("PREVIEW MODE - No files will be moved or deleted")
        # Just analyze and report
        image_files = list(Path(args.input_dir).glob("*.jpg")) + list(Path(args.input_dir).glob("*.jpeg"))
        quality_scores = []
        for img_path in tqdm(image_files[:50], desc="Previewing first 50 images"):
            quality_score, reason = analyze_image_quality(img_path)
            quality_scores.append(quality_score)
            print(f"{img_path.name}: {quality_score} - {reason}")
        
        if quality_scores:
            print(f"\nPreview statistics:")
            print(f"Average quality: {np.mean(quality_scores):.1f}")
            print(f"Range: {min(quality_scores):.1f} - {max(quality_scores):.1f}")
    else:
        filter_aligned_images(
            args.input_dir, 
            args.output_dir, 
            args.min_quality, 
            args.move_bad
        )

if __name__ == "__main__":
    main()
