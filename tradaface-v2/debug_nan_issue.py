#!/usr/bin/env python3
"""
Debug script to identify the source of NaN gradients in TransMatcher training.
"""

import torch
import torch.nn as nn
import numpy as np
from net import AdaFaceWithTransMatcher
import head
import utils

def debug_nan_issue():
    print("DEBUGGING NaN GRADIENT ISSUE")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create a small test batch
    batch_size = 4
    img = torch.randn(batch_size, 3, 112, 112, device=device)
    label = torch.randint(0, 1000, (batch_size,), device=device)
    
    print(f"Input shape: {img.shape}")
    print(f"Label shape: {label.shape}")
    print(f"Input stats: min={img.min().item():.6f}, max={img.max().item():.6f}, mean={img.mean().item():.6f}")
    
    # Check for NaNs in input
    if torch.isnan(img).any():
        print("ERROR: Input contains NaNs!")
        return
    
    # Create model
    from net import build_model
    backbone = build_model(model_name='ir_50')
    
    transmatcher_params = {
        'seq_len': 49,
        'd_model': 512,
        'num_decoder_layers': 3,
        'dim_feedforward': 2048,
    }
    
    model = AdaFaceWithTransMatcher(backbone, transmatcher_params).to(device)
    
    # Create head
    head_model = head.build_head(
        head_type='adaface',
        embedding_size=512,
        class_num=1000,
        m=0.4,
        h=0.333,
        t_alpha=1.0,
        s=64.0
    ).to(device)
    
    print("Model created successfully")
    
    # Test forward pass
    print("\nTesting forward pass...")
    try:
        with torch.no_grad():
            embeddings, norms, feature_maps, transmatcher = model(img)
        
        print(f"Forward pass successful!")
        print(f"  - Embeddings shape: {embeddings.shape}")
        print(f"  - Norms shape: {norms.shape}")
        print(f"  - Feature maps shape: {feature_maps.shape}")
        
        # Check for NaNs in outputs
        if torch.isnan(embeddings).any():
            print("ERROR: Embeddings contain NaNs!")
            return
        
        if torch.isnan(norms).any():
            print("ERROR: Norms contain NaNs!")
            return
        
        if torch.isnan(feature_maps).any():
            print("ERROR: Feature maps contain NaNs!")
            return
        
        print("  - All outputs are valid (no NaNs)")
        
    except Exception as e:
        print(f"ERROR in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test with gradients
    print("\nTesting with gradients...")
    try:
        embeddings, norms, feature_maps, transmatcher = model(img)
        
        # Normalize embeddings
        emb_norms = torch.norm(embeddings, 2, 1, True).clamp(min=1e-6)
        embeddings = torch.div(embeddings, emb_norms)
        
        print(f"Embedding normalization successful!")
        print(f"  - Normalized embeddings stats: min={embeddings.min().item():.6f}, max={embeddings.max().item():.6f}, mean={embeddings.mean().item():.6f}")
        
        if torch.isnan(embeddings).any():
            print("ERROR: Normalized embeddings contain NaNs!")
            return
        
        # Test AdaFace head
        logits = head_model(embeddings, emb_norms, label)
        if isinstance(logits, tuple):
            logits, bad_grad = logits
            label[bad_grad.squeeze(-1)] = -100
        
        print(f"AdaFace head successful!")
        print(f"  - Logits shape: {logits.shape}")
        print(f"  - Logits stats: min={logits.min().item():.6f}, max={logits.max().item():.6f}, mean={logits.mean().item():.6f}")
        
        if torch.isnan(logits).any():
            print("ERROR: Logits contain NaNs!")
            return
        
        # Test loss computation
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, label)
        
        print(f"Loss computation successful!")
        print(f"  - Loss value: {loss.item():.6f}")
        
        if torch.isnan(loss):
            print("ERROR: Loss is NaN!")
            return
        
        # Test backward pass
        loss.backward()
        
        print("Backward pass successful!")
        
        # Check for NaN gradients
        has_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"ERROR: NaN gradient in {name}")
                has_nan_grad = True
        
        if has_nan_grad:
            print("ERROR: NaN gradients detected!")
        else:
            print("SUCCESS: No NaN gradients detected!")
        
    except Exception as e:
        print(f"ERROR in gradient test: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    debug_nan_issue() 