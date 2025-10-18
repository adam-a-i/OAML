#!/usr/bin/env python3
"""
Delete Shopping Mall Images Script
=================================

This script deletes all images that start with "Shopping-mall" from the directory.
"""

import os
import sys
from pathlib import Path

def delete_shopping_mall_images(alignments_dir):
    """
    Delete all images that start with "Shopping-mall"
    
    Args:
        alignments_dir: Directory containing the aligned images
    """
    
    alignments_path = Path(alignments_dir)
    
    if not alignments_path.exists():
        print(f"Error: Directory '{alignments_dir}' does not exist!")
        return
    
    # Get all image files that start with "Shopping-mall"
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    shopping_mall_images = []
    
    for ext in image_extensions:
        # Look for files starting with "Shopping-mall"
        shopping_mall_images.extend(alignments_path.glob(f"Shopping-mall*{ext}"))
        shopping_mall_images.extend(alignments_path.glob(f"Shopping-mall*{ext.upper()}"))
    
    print(f"Found {len(shopping_mall_images)} Shopping-mall images to delete")
    
    if len(shopping_mall_images) == 0:
        print("No Shopping-mall images found")
        return
    
    # Delete each Shopping-mall image
    deleted_count = 0
    error_count = 0
    
    for image_path in shopping_mall_images:
        try:
            os.remove(image_path)
            deleted_count += 1
            print(f"Deleted: {image_path.name}")
        except Exception as e:
            error_count += 1
            print(f"Error deleting {image_path.name}: {e}")
    
    print(f"\nDeletion complete!")
    print(f"Successfully deleted: {deleted_count}")
    print(f"Errors: {error_count}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Delete Shopping-mall images')
    parser.add_argument('alignments_dir', help='Directory containing aligned images')
    parser.add_argument('--preview', action='store_true',
                       help='Preview mode - show what would be deleted without deleting')
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.exists(args.alignments_dir):
        print(f"Error: Directory '{args.alignments_dir}' does not exist!")
        return
    
    if args.preview:
        print("PREVIEW MODE - No files will be deleted")
        alignments_path = Path(args.alignments_dir)
        
        # Get all image files that start with "Shopping-mall"
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        shopping_mall_images = []
        
        for ext in image_extensions:
            shopping_mall_images.extend(alignments_path.glob(f"Shopping-mall*{ext}"))
            shopping_mall_images.extend(alignments_path.glob(f"Shopping-mall*{ext.upper()}"))
        
        print(f"Found {len(shopping_mall_images)} Shopping-mall images that would be deleted:")
        for image_path in shopping_mall_images:
            print(f"  Would delete: {image_path.name}")
    else:
        delete_shopping_mall_images(args.alignments_dir)

if __name__ == "__main__":
    main()
