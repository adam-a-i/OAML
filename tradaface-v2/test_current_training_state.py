#!/usr/bin/env python3
"""
Test script to analyze current training state and suggest improvements.
"""

import torch
import torch.nn as nn
import numpy as np
import config
import data
import train_val
from utils import dotdict

def test_current_state():
    print("="*80)
    print("ANALYZING CURRENT TRAINING STATE")
    print("="*80)
    
    # Get configuration
    args = config.get_args()
    hparams = dotdict(vars(args))
    
    print(f"Current configuration:")
    print(f"  - Learning rate: {hparams.lr}")
    print(f"  - Batch size: {hparams.batch_size}")
    print(f"  - TransMatcher loss weight: 0.1 (hardcoded)")
    print(f"  - Gradient clipping: max_norm=1.0")
    print(f"  - Precision: {16 if hparams.use_16bit else 32}")
    
    # Initialize model
    print("\nInitializing model...")
    trainer = train_val.Trainer(**hparams)
    
    # Initialize data
    print("Initializing data...")
    data_module = data.DataModule(**hparams)
    data_module.setup()
    
    # Get a batch
    print("Getting a sample batch...")
    batch = next(iter(data_module.train_dataloader()))
    img, label = batch
    
    print(f"Batch info:")
    print(f"  - Image shape: {img.shape}")
    print(f"  - Label shape: {label.shape}")
    print(f"  - Unique labels: {torch.unique(label).tolist()}")
    print(f"  - Image stats: min={img.min().item():.6f}, max={img.max().item():.6f}, mean={img.mean().item():.6f}")
    
    # Forward pass
    print("\nRunning forward pass...")
    trainer.eval()
    with torch.no_grad():
        embedding, norm, feature_maps, transmatcher = trainer.model(img, return_feature_map=True)
        
        print(f"Model outputs:")
        print(f"  - Embedding shape: {embedding.shape}")
        print(f"  - Norm shape: {norm.shape}")
        print(f"  - Feature maps shape: {feature_maps.shape}")
        print(f"  - Embedding stats: min={embedding.min().item():.6f}, max={embedding.max().item():.6f}, mean={embedding.mean().item():.6f}")
        print(f"  - Norm stats: min={norm.min().item():.6f}, max={norm.max().item():.6f}, mean={norm.mean().item():.6f}")
        print(f"  - Feature maps stats: min={feature_maps.min().item():.6f}, max={feature_maps.max().item():.6f}, mean={feature_maps.mean().item():.6f}")
    
    # Test loss computation
    print("\nTesting loss computation...")
    trainer.train()
    
    # AdaFace loss
    cos_thetas = trainer.head(embedding, norm, label)
    if isinstance(cos_thetas, tuple):
        cos_thetas, bad_grad = cos_thetas
        label[bad_grad.squeeze(-1)] = -100
    
    adaface_loss = trainer.cross_entropy_loss(cos_thetas, label)
    print(f"AdaFace loss: {adaface_loss.item():.6f}")
    print(f"Cos thetas stats: min={cos_thetas.min().item():.6f}, max={cos_thetas.max().item():.6f}, mean={cos_thetas.mean().item():.6f}")
    
    # TransMatcher loss
    if trainer.pairwise_matching_loss is not None:
        trainer.pairwise_matching_loss.matcher = transmatcher
        transmatcher_loss, transmatcher_acc = trainer.pairwise_matching_loss(feature_maps, label)
        print(f"TransMatcher loss: {transmatcher_loss.mean().item():.6f}")
        print(f"TransMatcher accuracy: {transmatcher_acc.mean().item():.6f}")
        
        # Check pairwise loss details
        print(f"TransMatcher loss shape: {transmatcher_loss.shape}")
        print(f"TransMatcher loss min/max: {transmatcher_loss.min().item():.6f}/{transmatcher_loss.max().item():.6f}")
    
    # Combined loss
    total_loss = adaface_loss + 0.1 * transmatcher_loss.mean()
    print(f"Total loss: {total_loss.item():.6f}")
    
    # Check gradients
    print("\nTesting gradient computation...")
    total_loss.backward()
    
    # Check for NaN gradients
    has_nan_grad = False
    for name, param in trainer.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"✗ ERROR: NaN gradient in {name}")
            has_nan_grad = True
    
    if not has_nan_grad:
        print("✓ No NaN gradients detected")
    
    # Check gradient norms
    total_norm = 0
    for name, param in trainer.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            if param_norm.item() > 10:
                print(f"⚠️  High gradient norm in {name}: {param_norm.item():.6f}")
    
    total_norm = total_norm ** (1. / 2)
    print(f"Total gradient norm: {total_norm:.6f}")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if adaface_loss.item() > 10:
        print("⚠️  AdaFace loss is high. Consider:")
        print("   - Increasing learning rate")
        print("   - Checking model initialization")
        print("   - Verifying data preprocessing")
    
    if transmatcher_loss.mean().item() > 5:
        print("⚠️  TransMatcher loss is high. Consider:")
        print("   - Reducing TransMatcher loss weight (currently 0.1)")
        print("   - Checking TransMatcher initialization")
        print("   - Verifying feature map normalization")
    
    if total_norm > 10:
        print("⚠️  Gradient norm is high. Consider:")
        print("   - Reducing learning rate")
        print("   - Increasing gradient clipping (currently 1.0)")
        print("   - Adding batch normalization")
    
    print("\nSuggested immediate actions:")
    print("1. Try increasing learning rate to 5e-5 or 1e-4")
    print("2. Reduce TransMatcher loss weight to 0.05 or 0.01")
    print("3. Increase gradient clipping to 5.0")
    print("4. Add learning rate warmup for first few epochs")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    test_current_state() 