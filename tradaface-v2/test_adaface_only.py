#!/usr/bin/env python3
"""
Test script to verify AdaFace alone works without NaN gradients.
This will help us isolate whether the NaN issues are coming from TransMatcher or AdaFace.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import net
import head
import utils
import config

def test_adaface_only():
    print("="*60)
    print("TESTING ADAFACE ALONE (NO TRANSMATCHER)")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get configuration
    args = config.get_args()
    
    # Build model (AdaFace only)
    print("Building AdaFace model...")
    model = net.build_model(model_name=args.arch)
    model = model.to(device)
    model.train()
    
    # Build head
    class_num = utils.get_num_class(args)
    print(f"Building head with {class_num} classes...")
    head_module = head.build_head(
        head_type=args.head,
        embedding_size=512,
        class_num=class_num,
        m=args.m,
        h=args.h,
        t_alpha=args.t_alpha,
        s=args.s,
    )
    head_module = head_module.to(device)
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer
    optimizer = optim.SGD([
        {'params': model.parameters(), 'lr': 0.0001, 'weight_decay': 5e-4},
        {'params': head_module.parameters(), 'lr': 0.0001, 'weight_decay': 5e-4}
    ])
    
    # Create dummy data
    batch_size = 4
    print(f"Creating dummy data with batch size {batch_size}...")
    
    # Create random images (normalized)
    images = torch.randn(batch_size, 3, 112, 112, device=device)
    images = images * 0.5 + 0.5  # Normalize to [0, 1]
    
    # Create random labels
    labels = torch.randint(0, class_num, (batch_size,), device=device)
    
    print(f"Input shapes: images={images.shape}, labels={labels.shape}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    try:
        with torch.no_grad():
            embeddings, norms = model(images)
            print(f"✓ Model forward pass successful")
            print(f"  - Embeddings shape: {embeddings.shape}")
            print(f"  - Norms shape: {norms.shape}")
            print(f"  - Embeddings stats: min={embeddings.min().item():.6f}, max={embeddings.max().item():.6f}, mean={embeddings.mean().item():.6f}")
            print(f"  - Norms stats: min={norms.min().item():.6f}, max={norms.max().item():.6f}, mean={norms.mean().item():.6f}")
            
            # Check for NaNs
            if torch.isnan(embeddings).any():
                print("✗ ERROR: Embeddings contain NaNs!")
                return False
            if torch.isnan(norms).any():
                print("✗ ERROR: Norms contain NaNs!")
                return False
            print("✓ No NaNs in model outputs")
            
            # Test head
            cos_thetas = head_module(embeddings, norms, labels)
            print(f"✓ Head forward pass successful")
            print(f"  - Cos thetas shape: {cos_thetas.shape}")
            print(f"  - Cos thetas stats: min={cos_thetas.min().item():.6f}, max={cos_thetas.max().item():.6f}, mean={cos_thetas.mean().item():.6f}")
            
            if torch.isnan(cos_thetas).any():
                print("✗ ERROR: Cos thetas contain NaNs!")
                return False
            print("✓ No NaNs in head outputs")
            
    except Exception as e:
        print(f"✗ ERROR in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test backward pass
    print("\nTesting backward pass...")
    try:
        # Forward pass with gradients
        embeddings, norms = model(images)
        
        # Check for NaNs after forward pass
        if torch.isnan(embeddings).any():
            print("✗ ERROR: Embeddings contain NaNs after forward pass!")
            return False
        
        # Head forward pass
        cos_thetas = head_module(embeddings, norms, labels)
        
        # Check for NaNs after head
        if torch.isnan(cos_thetas).any():
            print("✗ ERROR: Cos thetas contain NaNs after head!")
            return False
        
        # Compute loss
        loss = criterion(cos_thetas, labels)
        print(f"✓ Loss computation successful: {loss.item():.6f}")
        
        if torch.isnan(loss):
            print("✗ ERROR: Loss is NaN!")
            return False
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(head_module.parameters(), max_norm=1.0)
        
        print("✓ Backward pass successful")
        
        # Check for NaN gradients
        has_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"✗ ERROR: NaN gradients in model parameter {name}")
                has_nan_grad = True
        
        for name, param in head_module.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"✗ ERROR: NaN gradients in head parameter {name}")
                has_nan_grad = True
        
        if has_nan_grad:
            return False
        
        print("✓ No NaN gradients detected")
        
        # Optimizer step
        optimizer.step()
        print("✓ Optimizer step successful")
        
    except Exception as e:
        print(f"✗ ERROR in backward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test multiple iterations
    print("\nTesting multiple iterations...")
    try:
        for i in range(5):
            # Forward pass
            embeddings, norms = model(images)
            cos_thetas = head_module(embeddings, norms, labels)
            loss = criterion(cos_thetas, labels)
            
            # Check for NaNs
            if torch.isnan(loss):
                print(f"✗ ERROR: Loss is NaN at iteration {i}")
                return False
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Check for NaN gradients
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"✗ ERROR: NaN gradients in model parameter {name} at iteration {i}")
                    return False
            
            for name, param in head_module.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"✗ ERROR: NaN gradients in head parameter {name} at iteration {i}")
                    return False
            
            optimizer.step()
            
            if i % 2 == 0:
                print(f"  ✓ Iteration {i}: loss={loss.item():.6f}")
        
        print("✓ Multiple iterations successful")
        
    except Exception as e:
        print(f"✗ ERROR in multiple iterations: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("✓ ADAFACE ALONE TEST PASSED - NO NaN GRADIENTS DETECTED")
    print("="*60)
    return True

if __name__ == "__main__":
    success = test_adaface_only()
    if success:
        print("\nCONCLUSION: AdaFace alone works correctly without NaN gradients.")
        print("The NaN gradient issues are likely coming from the TransMatcher integration.")
    else:
        print("\nCONCLUSION: AdaFace alone has NaN gradient issues.")
        print("The problem is in the base AdaFace model, not TransMatcher.")
    
    sys.exit(0 if success else 1)