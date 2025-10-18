#!/usr/bin/env python3
"""
Cleanup Failed Alignments Log
============================

This script removes the failed_alignments.txt file since the images
are already gone or were never created.
"""

import os
import sys
from pathlib import Path

def cleanup_failed_log(alignments_dir, failed_log_file="failed_alignments.txt"):
    """
    Remove the failed_alignments.txt file
    
    Args:
        alignments_dir: Directory containing the aligned images
        failed_log_file: Name of the file containing failed image names
    """
    
    alignments_path = Path(alignments_dir)
    failed_log_path = alignments_path / failed_log_file
    
    if not failed_log_path.exists():
        print(f"File {failed_log_path} does not exist!")
        return
    
    try:
        os.remove(failed_log_path)
        print(f"Successfully deleted: {failed_log_path}")
    except Exception as e:
        print(f"Error deleting {failed_log_path}: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Cleanup failed alignments log file')
    parser.add_argument('alignments_dir', help='Directory containing aligned images')
    parser.add_argument('--log_file', default='failed_alignments.txt',
                       help='Name of the failed alignments log file')
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.exists(args.alignments_dir):
        print(f"Error: Directory '{args.alignments_dir}' does not exist!")
        return
    
    cleanup_failed_log(args.alignments_dir, args.log_file)

if __name__ == "__main__":
    main()
