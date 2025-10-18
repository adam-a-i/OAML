#!/usr/bin/env python3
"""
Delete Failed Alignments Script
==============================

This script reads the failed_alignments.txt file and deletes all the listed images.
"""

import os
import sys
from pathlib import Path

def delete_failed_alignments(alignments_dir, failed_log_file="failed_alignments.txt"):
    """
    Delete all images listed in the failed_alignments.txt file
    
    Args:
        alignments_dir: Directory containing the aligned images
        failed_log_file: Name of the file containing failed image names
    """
    
    alignments_path = Path(alignments_dir)
    failed_log_path = alignments_path / failed_log_file
    
    if not failed_log_path.exists():
        print(f"Error: {failed_log_path} does not exist!")
        return
    
    # Read the failed image names
    with open(failed_log_path, 'r') as f:
        failed_images = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(failed_images)} failed images to delete")
    
    if len(failed_images) == 0:
        print("No failed images to delete")
        return
    
    # Delete each failed image
    deleted_count = 0
    not_found_count = 0
    
    for image_name in failed_images:
        image_path = alignments_path / image_name
        
        if image_path.exists():
            try:
                os.remove(image_path)
                deleted_count += 1
                print(f"Deleted: {image_path}")
            except Exception as e:
                print(f"Error deleting {image_name}: {e}")
        else:
            not_found_count += 1
            print(f"Not found: {image_path}")
    
    print(f"\nDeletion complete!")
    print(f"Successfully deleted: {deleted_count}")
    print(f"Not found: {not_found_count}")
    
    # Optionally delete the failed_alignments.txt file itself
    if deleted_count > 0:
        try:
            os.remove(failed_log_path)
            print(f"Also deleted: {failed_log_file}")
        except Exception as e:
            print(f"Could not delete {failed_log_file}: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Delete failed alignment images')
    parser.add_argument('alignments_dir', help='Directory containing aligned images')
    parser.add_argument('--log_file', default='failed_alignments.txt',
                       help='Name of the failed alignments log file')
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
        failed_log_path = alignments_path / args.log_file
        
        if not failed_log_path.exists():
            print(f"Error: {failed_log_path} does not exist!")
            return
        
        # Read the failed image names
        with open(failed_log_path, 'r') as f:
            failed_images = [line.strip() for line in f if line.strip()]
        
        print(f"Found {len(failed_images)} failed images that would be deleted:")
        for image_name in failed_images:
            image_path = alignments_path / image_name
            if image_path.exists():
                print(f"  Would delete: {image_path}")
            else:
                print(f"  Not found: {image_path}")
    else:
        delete_failed_alignments(args.alignments_dir, args.log_file)

if __name__ == "__main__":
    main()
