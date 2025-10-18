#!/usr/bin/env python3
"""
Example script to align niqab dataset
=====================================

Simple example showing how to use the MTCNN alignment pipeline
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from face_alignment.mtcnn import MTCNN
from PIL import Image

def simple_align_example():
    """Simple example of aligning a single niqab image"""
    
    # Initialize MTCNN
    print("Initializing MTCNN...")
    mtcnn_model = MTCNN(device='cuda:0', crop_size=(112, 112))
    
    # Example image path (replace with your actual image)
    image_path = "path/to/your/niqab_image.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print("Please update the image_path variable with your actual image path")
        return
    
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        print(f"Loaded image: {image_path}")
        
        # Align face
        aligned_face = mtcnn_model.align(img)
        
        if aligned_face is not None:
            # Save aligned face
            output_path = "aligned_face.jpg"
            aligned_face.save(output_path)
            print(f"Aligned face saved to: {output_path}")
        else:
            print("Failed to detect face in the image")
            
    except Exception as e:
        print(f"Error: {e}")

def batch_align_example():
    """Example of aligning multiple images in a directory"""
    
    # Set your directories
    input_dir = "path/to/your/niqab/images"
    output_dir = "path/to/save/aligned/faces"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize MTCNN
    print("Initializing MTCNN...")
    mtcnn_model = MTCNN(device='cuda:0', crop_size=(112, 112))
    
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
    successful = 0
    failed = 0
    
    for img_path in image_files:
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Align face
            aligned_face = mtcnn_model.align(img)
            
            if aligned_face is not None:
                # Save aligned face
                output_path = Path(output_dir) / img_path.name
                aligned_face.save(output_path)
                successful += 1
                print(f"✓ Aligned: {img_path.name}")
            else:
                print(f"✗ Failed: {img_path.name}")
                failed += 1
                
        except Exception as e:
            print(f"✗ Error processing {img_path.name}: {e}")
            failed += 1
    
    print(f"\nBatch alignment complete!")
    print(f"Successfully aligned: {successful}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    print("Niqab Dataset Alignment Example")
    print("=" * 40)
    print()
    print("Choose an example:")
    print("1. Simple single image alignment")
    print("2. Batch directory alignment")
    print()
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        simple_align_example()
    elif choice == "2":
        batch_align_example()
    else:
        print("Invalid choice. Please run the script again and choose 1 or 2.")
