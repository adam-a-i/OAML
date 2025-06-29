#!/usr/bin/env python3
"""
Debug script to check PK sampler configuration
"""

import sys
import os
from config import get_args

def debug_pk_sampler():
    print("="*60)
    print("DEBUGGING PK SAMPLER CONFIGURATION")
    print("="*60)
    
    # Simulate your command line arguments
    sys.argv = [
        'debug_pk_sampler.py',
        '--data_root', '/home/maass/code',
        '--train_data_path', 'faces_webface_112x112',
        '--val_data_path', 'faces_webface_112x112',
        '--prefix', 'ir50_casia_adaface_partial',
        '--gpus', '1',
        '--use_16bit',
        '--arch', 'ir_50',
        '--batch_size', '64',
        '--num_instances', '4',
        '--pk_sampler',  # This should enable PK sampling
        '--num_workers', '8',
        '--epochs', '385',
        '--lr_milestones', '185,285,337',
        '--lr', '0.05',
        '--head', 'adaface',
        '--m', '0.4',
        '--h', '0.333',
        '--low_res_augmentation_prob', '0.2',
        '--crop_augmentation_prob', '0.2',
        '--photometric_augmentation_prob', '0.2',
        '--use_wandb'
    ]
    
    args = get_args()
    
    print("Parsed arguments:")
    print(f"  pk_sampler: {args.pk_sampler}")
    print(f"  num_instances: {args.num_instances}")
    print(f"  batch_size: {args.batch_size}")
    
    # Check if pk_sampler is actually True
    if hasattr(args, 'pk_sampler'):
        print(f"✅ pk_sampler attribute exists: {args.pk_sampler}")
    else:
        print("❌ pk_sampler attribute does NOT exist!")
    
    # Check if num_instances is set
    if hasattr(args, 'num_instances'):
        print(f"✅ num_instances attribute exists: {args.num_instances}")
    else:
        print("❌ num_instances attribute does NOT exist!")
    
    print("\n" + "="*60)
    print("DEBUG COMPLETED")
    print("="*60)

if __name__ == "__main__":
    debug_pk_sampler() 