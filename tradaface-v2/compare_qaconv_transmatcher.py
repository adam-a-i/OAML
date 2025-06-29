#!/usr/bin/env python3
"""
Comparison script to test QAConv vs TransMatcher side by side
This will help identify exactly what's different between the two implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import traceback
from collections import defaultdict
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import qaconv
import transmatcher
import net
import head
import pairwise_matching_loss
from torch.nn import CrossEntropyLoss

class ComparisonTest:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Test parameters
        self.batch_size = 32
        self.num_classes = 100  # Small number for testing
        self.feature_channels = 512
        self.feature_height = 7
        self.feature_width = 7
        self.seq_len = self.feature_height * self.feature_width  # 49
        
        # Initialize models
        self.setup_models()
        
    def setup_models(self):
        """Initialize both QAConv and TransMatcher models"""
        print("\n" + "="*60)
        print("SETTING UP MODELS")
        print("="*60)
        
        # QAConv setup
        print("Setting up QAConv...")
        self.qaconv = qaconv.QAConv(
            num_features=self.feature_channels,
            height=self.feature_height,
            width=self.feature_width,
            num_classes=self.num_classes,
            k_nearest=20
        ).to(self.device)
        
        # TransMatcher setup
        print("Setting up TransMatcher...")
        self.transmatcher = transmatcher.TransMatcher(
            seq_len=self.seq_len,
            d_model=self.feature_channels,
            num_decoder_layers=3,
            dim_feedforward=2048
        ).to(self.device)
        
        # Backbone for feature extraction
        print("Setting up backbone...")
        self.backbone = net.build_model(model_name='ir_50').to(self.device)
        
        # Head for AdaFace
        print("Setting up AdaFace head...")
        self.head = head.build_head(
            head_type='adaface',
            embedding_size=512,
            class_num=self.num_classes,
            m=0.4,
            h=0.333,
            t_alpha=1.0,
            s=64.0
        ).to(self.device)
        
        # Loss functions
        self.ce_loss = CrossEntropyLoss()
        self.qaconv_loss = pairwise_matching_loss.PairwiseMatchingLoss(self.qaconv)
        self.transmatcher_loss = pairwise_matching_loss.PairwiseMatchingLoss(self.transmatcher)
        
        print("All models initialized successfully!")
        
    def generate_test_data(self, num_batches=5):
        """Generate test data with controlled batch composition"""
        print(f"\nGenerating {num_batches} test batches...")
        
        batches = []
        for batch_idx in range(num_batches):
            # Create a batch with controlled identity distribution
            # Use N=8 identities, K=4 samples per identity (batch_size=32)
            num_identities = 8
            samples_per_identity = 4
            
            # Generate labels: [0,0,0,0, 1,1,1,1, 2,2,2,2, ...]
            labels = torch.tensor([
                i // samples_per_identity for i in range(self.batch_size)
            ], dtype=torch.long, device=self.device)
            
            # Generate random images (simulating 112x112 RGB images)
            images = torch.randn(self.batch_size, 3, 112, 112, device=self.device)
            
            # Normalize images
            images = (images - images.mean()) / (images.std() + 1e-8)
            
            batches.append((images, labels))
            
        print(f"Generated {len(batches)} batches")
        print(f"Sample batch labels: {batches[0][1]}")
        print(f"Unique identities in first batch: {len(torch.unique(batches[0][1]))}")
        
        return batches
    
    def test_feature_extraction(self, images):
        """Test feature extraction through backbone"""
        print("\nTesting feature extraction...")
        
        try:
            # Extract features using backbone
            with torch.no_grad():
                # The backbone now returns more than 2 values due to TransMatcher integration
                backbone_output = self.backbone(images)
                
                print(f"DEBUG: Backbone output type: {type(backbone_output)}")
                if isinstance(backbone_output, tuple):
                    print(f"DEBUG: Backbone output length: {len(backbone_output)}")
                    for i, item in enumerate(backbone_output):
                        print(f"DEBUG: Item {i} type: {type(item)}")
                        if hasattr(item, 'shape'):
                            print(f"DEBUG: Item {i} shape: {item.shape}")
                
                # Handle different return formats
                if isinstance(backbone_output, tuple):
                    if len(backbone_output) == 2:
                        # Standard format: (embeddings, norms)
                        embeddings, norms = backbone_output
                        feature_maps = None
                    elif len(backbone_output) >= 3:
                        # TransMatcher format: (embeddings, norms, feature_maps, transmatcher)
                        embeddings, norms = backbone_output[:2]
                        third_item = backbone_output[2]
                        
                        # Check if third item is feature maps or TransMatcher
                        if hasattr(third_item, 'shape') and len(third_item.shape) == 4:
                            # It's feature maps
                            feature_maps = third_item
                        else:
                            # It's likely TransMatcher object, try to get feature maps differently
                            print("DEBUG: Third item is not feature maps, trying alternative extraction...")
                            try:
                                # Try to get feature maps from the backbone's internal layers
                                feature_maps = self.backbone.input_layer(images)
                                for layer in self.backbone.body:
                                    feature_maps = layer(feature_maps)
                                print("DEBUG: Successfully extracted feature maps from backbone layers")
                            except Exception as e:
                                print(f"DEBUG: Failed to extract feature maps: {e}")
                                feature_maps = None
                    else:
                        raise ValueError(f"Unexpected backbone output length: {len(backbone_output)}")
                else:
                    # Single output
                    embeddings = backbone_output
                    norms = None
                    feature_maps = None
                
                # If we still don't have feature maps, try to get them explicitly
                if feature_maps is None:
                    try:
                        print("DEBUG: Trying explicit feature map extraction...")
                        feature_maps = self.backbone(images, return_feature_map=True)[2]
                        print("DEBUG: Successfully extracted feature maps with return_feature_map=True")
                    except Exception as e:
                        print(f"DEBUG: Could not extract feature maps: {e}")
                        return embeddings, norms, None
            
            print(f"✓ Feature extraction successful")
            print(f"  - Embeddings shape: {embeddings.shape}")
            if norms is not None:
                print(f"  - Norms shape: {norms.shape}")
            if feature_maps is not None:
                print(f"  - Feature maps shape: {feature_maps.shape}")
                print(f"  - Feature maps stats: min={feature_maps.min().item():.6f}, max={feature_maps.max().item():.6f}, mean={feature_maps.mean().item():.6f}")
            
            return embeddings, norms, feature_maps
            
        except Exception as e:
            print(f"✗ Feature extraction failed: {e}")
            print(traceback.format_exc())
            return None, None, None
    
    def test_qaconv_forward(self, feature_maps, labels):
        """Test QAConv forward pass"""
        print("\nTesting QAConv forward pass...")
        
        try:
            # Normalize feature maps for QAConv
            fm_flat = feature_maps.view(feature_maps.size(0), -1)
            fm_norms = torch.norm(fm_flat, p=2, dim=1, keepdim=True).clamp(min=1e-8)
            fm_normalized = (fm_flat / fm_norms).view_as(feature_maps)
            
            # QAConv forward pass
            start_time = time.time()
            qaconv_scores = self.qaconv(fm_normalized, labels=labels)
            qaconv_time = time.time() - start_time
            
            print(f"✓ QAConv forward pass successful")
            print(f"  - Scores shape: {qaconv_scores.shape}")
            print(f"  - Scores stats: min={qaconv_scores.min().item():.6f}, max={qaconv_scores.max().item():.6f}, mean={qaconv_scores.mean().item():.6f}")
            print(f"  - Forward time: {qaconv_time:.4f}s")
            
            return qaconv_scores, qaconv_time
            
        except Exception as e:
            print(f"✗ QAConv forward pass failed: {e}")
            print(traceback.format_exc())
            return None, 0
    
    def test_transmatcher_forward(self, feature_maps):
        """Test TransMatcher forward pass"""
        print("\nTesting TransMatcher forward pass...")
        
        try:
            # Normalize feature maps for TransMatcher
            fm_flat = feature_maps.view(feature_maps.size(0), -1)
            fm_norms = torch.norm(fm_flat, p=2, dim=1, keepdim=True).clamp(min=1e-8)
            fm_normalized = (fm_flat / fm_norms).view_as(feature_maps)
            
            # TransMatcher forward pass
            start_time = time.time()
            transmatcher_scores = self.transmatcher(fm_normalized)
            transmatcher_time = time.time() - start_time
            
            print(f"✓ TransMatcher forward pass successful")
            print(f"  - Scores shape: {transmatcher_scores.shape}")
            print(f"  - Scores stats: min={transmatcher_scores.min().item():.6f}, max={transmatcher_scores.max().item():.6f}, mean={transmatcher_scores.mean().item():.6f}")
            print(f"  - Forward time: {transmatcher_time:.4f}s")
            
            return transmatcher_scores, transmatcher_time
            
        except Exception as e:
            print(f"✗ TransMatcher forward pass failed: {e}")
            print(traceback.format_exc())
            return None, 0
    
    def test_loss_computation(self, feature_maps, labels):
        """Test loss computation for both methods"""
        print("\nTesting loss computation...")
        
        results = {}
        
        # Test QAConv loss
        try:
            print("Testing QAConv loss...")
            qaconv_loss = self.qaconv_loss(feature_maps, labels)
            print(f"✓ QAConv loss: {qaconv_loss.item():.6f}")
            results['qaconv_loss'] = qaconv_loss.item()
        except Exception as e:
            print(f"✗ QAConv loss failed: {e}")
            results['qaconv_loss'] = None
        
        # Test TransMatcher loss
        try:
            print("Testing TransMatcher loss...")
            transmatcher_loss = self.transmatcher_loss(feature_maps, labels)
            print(f"✓ TransMatcher loss: {transmatcher_loss.item():.6f}")
            results['transmatcher_loss'] = transmatcher_loss.item()
        except Exception as e:
            print(f"✗ TransMatcher loss failed: {e}")
            results['transmatcher_loss'] = None
        
        return results
    
    def test_adaface_head(self, embeddings, norms, labels):
        """Test AdaFace head"""
        print("\nTesting AdaFace head...")
        
        try:
            # Normalize embeddings
            emb_norms = torch.norm(embeddings, 2, 1, True).clamp(min=1e-6)
            embeddings_normalized = torch.div(embeddings, emb_norms)
            
            # AdaFace head forward pass
            cos_thetas = self.head(embeddings_normalized, norms, labels)
            if isinstance(cos_thetas, tuple):
                cos_thetas, bad_grad = cos_thetas
                labels[bad_grad.squeeze(-1)] = -100
            
            # Compute AdaFace loss
            adaface_loss = self.ce_loss(cos_thetas, labels)
            
            print(f"✓ AdaFace head successful")
            print(f"  - Cos thetas shape: {cos_thetas.shape}")
            print(f"  - Cos thetas stats: min={cos_thetas.min().item():.6f}, max={cos_thetas.max().item():.6f}, mean={cos_thetas.mean().item():.6f}")
            print(f"  - AdaFace loss: {adaface_loss.item():.6f}")
            
            return cos_thetas, adaface_loss.item()
            
        except Exception as e:
            print(f"✗ AdaFace head failed: {e}")
            print(traceback.format_exc())
            return None, None
    
    def run_comparison(self):
        """Run the full comparison test"""
        print("\n" + "="*60)
        print("STARTING QAConv vs TransMatcher COMPARISON")
        print("="*60)
        
        # Generate test data
        test_batches = self.generate_test_data(num_batches=3)
        
        # Results storage
        all_results = {
            'qaconv': {'forward_times': [], 'losses': [], 'scores_stats': []},
            'transmatcher': {'forward_times': [], 'losses': [], 'scores_stats': []},
            'adaface': {'losses': []}
        }
        
        for batch_idx, (images, labels) in enumerate(test_batches):
            print(f"\n{'='*40}")
            print(f"BATCH {batch_idx + 1}/{len(test_batches)}")
            print(f"{'='*40}")
            print(f"Labels: {labels.tolist()}")
            print(f"Unique identities: {len(torch.unique(labels))}")
            
            # Test feature extraction
            embeddings, norms, feature_maps = self.test_feature_extraction(images)
            if feature_maps is None:
                print("Skipping batch due to feature extraction failure")
                continue
            
            # Test QAConv
            qaconv_scores, qaconv_time = self.test_qaconv_forward(feature_maps, labels)
            if qaconv_scores is not None:
                all_results['qaconv']['forward_times'].append(qaconv_time)
                all_results['qaconv']['scores_stats'].append({
                    'min': qaconv_scores.min().item(),
                    'max': qaconv_scores.max().item(),
                    'mean': qaconv_scores.mean().item()
                })
            
            # Test TransMatcher
            transmatcher_scores, transmatcher_time = self.test_transmatcher_forward(feature_maps)
            if transmatcher_scores is not None:
                all_results['transmatcher']['forward_times'].append(transmatcher_time)
                all_results['transmatcher']['scores_stats'].append({
                    'min': transmatcher_scores.min().item(),
                    'max': transmatcher_scores.max().item(),
                    'mean': transmatcher_scores.mean().item()
                })
            
            # Test losses
            loss_results = self.test_loss_computation(feature_maps, labels)
            if loss_results['qaconv_loss'] is not None:
                all_results['qaconv']['losses'].append(loss_results['qaconv_loss'])
            if loss_results['transmatcher_loss'] is not None:
                all_results['transmatcher']['losses'].append(loss_results['transmatcher_loss'])
            
            # Test AdaFace head
            cos_thetas, adaface_loss = self.test_adaface_head(embeddings, norms, labels)
            if adaface_loss is not None:
                all_results['adaface']['losses'].append(adaface_loss)
        
        # Print summary
        self.print_summary(all_results)
    
    def print_summary(self, results):
        """Print comparison summary"""
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        
        # QAConv summary
        if results['qaconv']['forward_times']:
            avg_time = np.mean(results['qaconv']['forward_times'])
            avg_loss = np.mean(results['qaconv']['losses']) if results['qaconv']['losses'] else None
            print(f"QAConv:")
            print(f"  - Average forward time: {avg_time:.4f}s")
            print(f"  - Average loss: {avg_loss:.6f}" if avg_loss else "  - Average loss: N/A")
        
        # TransMatcher summary
        if results['transmatcher']['forward_times']:
            avg_time = np.mean(results['transmatcher']['forward_times'])
            avg_loss = np.mean(results['transmatcher']['losses']) if results['transmatcher']['losses'] else None
            print(f"TransMatcher:")
            print(f"  - Average forward time: {avg_time:.4f}s")
            print(f"  - Average loss: {avg_loss:.6f}" if avg_loss else "  - Average loss: N/A")
        
        # AdaFace summary
        if results['adaface']['losses']:
            avg_loss = np.mean(results['adaface']['losses'])
            print(f"AdaFace:")
            print(f"  - Average loss: {avg_loss:.6f}")
        
        # Key differences
        print(f"\nKEY DIFFERENCES:")
        if results['qaconv']['forward_times'] and results['transmatcher']['forward_times']:
            qa_time = np.mean(results['qaconv']['forward_times'])
            tm_time = np.mean(results['transmatcher']['forward_times'])
            print(f"  - TransMatcher is {tm_time/qa_time:.2f}x slower than QAConv")
        
        if results['qaconv']['losses'] and results['transmatcher']['losses']:
            qa_loss = np.mean(results['qaconv']['losses'])
            tm_loss = np.mean(results['transmatcher']['losses'])
            print(f"  - QAConv loss: {qa_loss:.6f}, TransMatcher loss: {tm_loss:.6f}")
            print(f"  - Loss ratio (TM/QA): {tm_loss/qa_loss:.2f}")

def main():
    """Main function"""
    print("QAConv vs TransMatcher Comparison Test")
    print("This will help identify differences between the two implementations")
    
    try:
        test = ComparisonTest()
        test.run_comparison()
        print("\n✓ Comparison completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Comparison failed: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 