"""
Diagnostic script to identify why AdaFace is not learning properly.
This will check gradient flow, parameter updates, and identify any issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from net import build_model
from head import build_head
import utils


def diagnose_gradient_flow():
    """Check if gradients are flowing correctly to backbone from AdaFace loss."""
    print("=" * 80)
    print("DIAGNOSTIC 1: Gradient Flow Analysis")
    print("=" * 80)

    # Build a small model for testing
    model = build_model('ir_18')  # Use smaller model for faster testing
    model.train()

    # Create dummy data
    batch_size = 4
    images = torch.randn(batch_size, 3, 112, 112)
    labels = torch.tensor([0, 1, 2, 3])

    # Store initial backbone weights for comparison
    initial_weights = {}
    for name, param in model.named_parameters():
        if 'body' in name or 'input_layer' in name:
            initial_weights[name] = param.clone().detach()

    print(f"\nTracking {len(initial_weights)} backbone parameters")

    # Forward pass - mimicking training_step
    x = model.input_layer(images)
    for layer in model.body:
        x = layer(x)

    # CRITICAL CHECK: Is x requires_grad=True?
    print(f"\nAfter backbone forward:")
    print(f"  x.requires_grad = {x.requires_grad}")
    print(f"  x.shape = {x.shape}")

    # Normalize for QAConv (like in training_step)
    x_norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1, keepdim=True).view(x.size(0), 1, 1, 1)
    x_norm = torch.clamp(x_norm, min=1e-8)
    feature_maps = x / x_norm

    # DETACH feature_maps for QAConv
    feature_maps_detached = feature_maps.clone().detach().requires_grad_(True)
    print(f"\nAfter detach for QAConv:")
    print(f"  feature_maps_detached.requires_grad = {feature_maps_detached.requires_grad}")
    print(f"  feature_maps_detached.grad_fn = {feature_maps_detached.grad_fn}")

    # Get AdaFace embeddings (should use non-detached x)
    embeddings = model.output_layer(x)
    print(f"\nAfter output_layer:")
    print(f"  embeddings.requires_grad = {embeddings.requires_grad}")
    print(f"  embeddings.grad_fn = {embeddings.grad_fn}")

    # L2 normalize
    embeddings_norm = torch.norm(embeddings, 2, 1, True)
    embeddings = embeddings / embeddings_norm.clamp(min=1e-6)

    # Create a simple loss (simulating AdaFace loss)
    # In real training, this goes through the head
    adaface_loss = embeddings.sum()  # Simple loss for gradient checking

    print(f"\nAdaFace loss: {adaface_loss.item():.4f}")
    print(f"  adaface_loss.requires_grad = {adaface_loss.requires_grad}")
    print(f"  adaface_loss.grad_fn = {adaface_loss.grad_fn}")

    # Backward pass
    adaface_loss.backward()

    # Check gradients on backbone
    print("\n" + "-" * 40)
    print("Gradient check on backbone parameters:")
    print("-" * 40)

    has_gradient = 0
    no_gradient = 0
    grad_magnitudes = []

    for name, param in model.named_parameters():
        if 'body' in name or 'input_layer' in name:
            if param.grad is not None:
                grad_mag = param.grad.abs().mean().item()
                grad_magnitudes.append(grad_mag)
                has_gradient += 1
                if has_gradient <= 5:  # Print first 5
                    print(f"  {name}: grad_mean = {grad_mag:.6e}")
            else:
                no_gradient += 1
                if no_gradient <= 5:
                    print(f"  {name}: NO GRADIENT!")

    print(f"\nSummary:")
    print(f"  Parameters with gradient: {has_gradient}")
    print(f"  Parameters without gradient: {no_gradient}")
    if grad_magnitudes:
        print(f"  Mean gradient magnitude: {sum(grad_magnitudes)/len(grad_magnitudes):.6e}")
        print(f"  Max gradient magnitude: {max(grad_magnitudes):.6e}")
        print(f"  Min gradient magnitude: {min(grad_magnitudes):.6e}")

    return has_gradient > 0


def diagnose_loss_weights():
    """Check if loss weights are causing the issue."""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 2: Loss Weight Analysis")
    print("=" * 80)

    # These are the current weights
    qaconv_loss_weight = 0.9
    adaface_loss_weight = 0.1
    occlusion_loss_weight = 0.1

    print(f"\nCurrent loss weights:")
    print(f"  QAConv weight: {qaconv_loss_weight}")
    print(f"  AdaFace weight: {adaface_loss_weight}")
    print(f"  Occlusion weight: {occlusion_loss_weight}")

    print(f"\nGradient contribution analysis:")
    print(f"  QAConv: DETACHED - 0% gradient to backbone")
    print(f"  AdaFace: {adaface_loss_weight * 100:.1f}% gradient to backbone")
    print(f"  Occlusion: DETACHED (no_grad) - 0% gradient to backbone")

    print(f"\n  TOTAL gradient to backbone: {adaface_loss_weight * 100:.1f}%")
    print(f"\n  WARNING: Backbone only receives {adaface_loss_weight * 100:.1f}% of normal gradient!")

    return adaface_loss_weight


def diagnose_training_step_simulation():
    """Simulate the exact training_step to check for issues."""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 3: Training Step Simulation")
    print("=" * 80)

    model = build_model('ir_18')
    model.train()

    # Simulate optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    batch_size = 4
    images = torch.randn(batch_size, 3, 112, 112)

    # Store initial weights
    initial_body_weight = model.body[0][0].weight.clone().detach()

    print(f"\nInitial body[0][0].weight mean: {initial_body_weight.mean().item():.6f}")

    # Forward pass exactly like training_step
    x = model.input_layer(images)
    for layer in model.body:
        x = layer(x)

    # Normalize for QAConv
    x_norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1, keepdim=True).view(x.size(0), 1, 1, 1)
    x_norm = torch.clamp(x_norm, min=1e-8)
    feature_maps = x / x_norm

    # DETACH for QAConv (this is the key line!)
    feature_maps = feature_maps.clone().detach().requires_grad_(True)

    # AdaFace path - uses original x (not detached)
    embeddings = model.output_layer(x)
    embeddings, norms = utils.l2_norm(embeddings, axis=1)

    # Simulate AdaFace loss
    adaface_loss = embeddings.pow(2).sum()

    # Simulate QAConv loss (from detached feature_maps)
    qaconv_loss = feature_maps.pow(2).sum()

    # Combined loss with weights
    adaface_weight = 0.1
    qaconv_weight = 0.9
    total_loss = adaface_weight * adaface_loss + qaconv_weight * qaconv_loss

    print(f"\nLosses:")
    print(f"  AdaFace loss: {adaface_loss.item():.4f}")
    print(f"  QAConv loss: {qaconv_loss.item():.4f}")
    print(f"  Total loss: {total_loss.item():.4f}")

    # Backward and step
    optimizer.zero_grad()
    total_loss.backward()

    # Check gradient on backbone
    body_grad = model.body[0][0].weight.grad
    if body_grad is not None:
        print(f"\nBody[0][0].weight gradient mean: {body_grad.abs().mean().item():.6e}")
    else:
        print(f"\nBody[0][0].weight gradient: NONE!")

    optimizer.step()

    # Check if weights changed
    new_body_weight = model.body[0][0].weight.clone().detach()
    weight_change = (new_body_weight - initial_body_weight).abs().mean().item()

    print(f"\nAfter optimizer.step():")
    print(f"  Weight change mean: {weight_change:.6e}")

    if weight_change < 1e-10:
        print(f"  WARNING: Weights did not change!")
    else:
        print(f"  OK: Weights updated")

    return weight_change > 1e-10


def diagnose_committed_vs_current():
    """Compare the training flow between committed and current code."""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 4: Code Difference Analysis")
    print("=" * 80)

    import subprocess

    # Get the diff for training_step
    result = subprocess.run(
        ['git', 'diff', 'HEAD', '--', 'train_val.py'],
        capture_output=True, text=True, cwd='/Users/adamahmed/73/OAML'
    )

    diff_output = result.stdout

    # Look for key changes
    print("\nKey changes in train_val.py:")

    changes = {
        'niqab_dataloader': 'niqab' in diff_output.lower(),
        'occlusion_loss': 'occlusion_loss' in diff_output,
        'model.eval()': 'model.eval()' in diff_output,
        'torch.no_grad()': 'no_grad' in diff_output,
        'feature_maps detach changed': False,  # Need to check manually
    }

    for change, present in changes.items():
        status = "ADDED" if present else "NOT CHANGED"
        print(f"  {change}: {status}")

    # Check if feature_maps detach was changed
    if 'feature_maps = feature_maps.clone().detach()' in diff_output:
        print(f"\n  WARNING: feature_maps detach line was modified!")

    return changes


def diagnose_output_layer_input():
    """Check what's being passed to output_layer."""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 5: Output Layer Input Analysis")
    print("=" * 80)

    model = build_model('ir_18')
    model.train()

    batch_size = 4
    images = torch.randn(batch_size, 3, 112, 112)

    # Forward through backbone
    x = model.input_layer(images)
    for layer in model.body:
        x = layer(x)

    print(f"\nAfter backbone:")
    print(f"  x.shape = {x.shape}")
    print(f"  x.requires_grad = {x.requires_grad}")
    print(f"  x.grad_fn = {x.grad_fn}")

    # Normalize for QAConv
    x_norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1, keepdim=True).view(x.size(0), 1, 1, 1)
    x_norm = torch.clamp(x_norm, min=1e-8)
    feature_maps = x / x_norm

    print(f"\nAfter normalization (feature_maps):")
    print(f"  feature_maps.shape = {feature_maps.shape}")
    print(f"  feature_maps.requires_grad = {feature_maps.requires_grad}")
    print(f"  feature_maps.grad_fn = {feature_maps.grad_fn}")

    # DETACH for QAConv
    feature_maps_detached = feature_maps.clone().detach().requires_grad_(True)

    print(f"\nAfter detach (feature_maps_detached):")
    print(f"  feature_maps_detached.requires_grad = {feature_maps_detached.requires_grad}")
    print(f"  feature_maps_detached.grad_fn = {feature_maps_detached.grad_fn}")

    # CRITICAL: What goes into output_layer?
    # In current code: embeddings = self.model.output_layer(x)
    # x is the ORIGINAL backbone output, NOT the detached feature_maps

    embeddings = model.output_layer(x)

    print(f"\nAfter output_layer(x):")
    print(f"  embeddings.shape = {embeddings.shape}")
    print(f"  embeddings.requires_grad = {embeddings.requires_grad}")
    print(f"  embeddings.grad_fn = {embeddings.grad_fn}")

    # Check gradient flow
    loss = embeddings.sum()
    loss.backward()

    # Check if backbone got gradients
    body_grad = model.body[0][0].weight.grad
    input_grad = model.input_layer[0].weight.grad

    print(f"\nGradient check:")
    print(f"  body[0][0] gradient exists: {body_grad is not None}")
    print(f"  input_layer[0] gradient exists: {input_grad is not None}")

    if body_grad is not None:
        print(f"  body[0][0] gradient magnitude: {body_grad.abs().mean().item():.6e}")

    return embeddings.requires_grad


def diagnose_exact_training_step():
    """Simulate the EXACT training_step to check gradient flow."""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 6: Exact Training Step Simulation")
    print("=" * 80)

    import sys
    sys.path.insert(0, '/Users/adamahmed/73/OAML')

    from net import build_model
    from head import build_head
    import utils

    model = build_model('ir_18')
    model.train()

    # Create head (like in Trainer.__init__)
    head = build_head('adaface', embedding_size=512, class_num=100, m=0.4, h=0.333, s=64.0, t_alpha=0.01)

    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    batch_size = 8
    images = torch.randn(batch_size, 3, 112, 112)
    labels = torch.randint(0, 100, (batch_size,))
    device = images.device

    print(f"\nInput: images {images.shape}, labels {labels.shape}")

    # Store initial weights
    initial_input_layer_weight = model.input_layer[0].weight.clone().detach()
    initial_body_weight = model.body[0][0].weight.clone().detach()

    # ============ EXACT TRAINING_STEP CODE ============
    # get features from model up to before output layer
    x = model.input_layer(images)

    for i, layer in enumerate(model.body):
        x = layer(x)

    print(f"\nAfter backbone:")
    print(f"  x.shape = {x.shape}")
    print(f"  x.requires_grad = {x.requires_grad}")
    print(f"  x.grad_fn = {x.grad_fn}")

    # normalize feature maps for qaconv
    x_norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1, keepdim=True).view(x.size(0), 1, 1, 1)
    x_norm = torch.clamp(x_norm, min=1e-8)
    feature_maps = x / x_norm

    # DETACH for QAConv
    feature_maps = feature_maps.clone().detach().requires_grad_(True)

    print(f"\nAfter feature_maps detach:")
    print(f"  feature_maps.requires_grad = {feature_maps.requires_grad}")
    print(f"  feature_maps.grad_fn = {feature_maps.grad_fn}")
    print(f"  x.requires_grad = {x.requires_grad}")  # x should still have grad_fn!
    print(f"  x.grad_fn = {x.grad_fn}")

    # get adaface embeddings through output layer
    embeddings = model.output_layer(x)

    print(f"\nAfter output_layer(x):")
    print(f"  embeddings.shape = {embeddings.shape}")
    print(f"  embeddings.requires_grad = {embeddings.requires_grad}")
    print(f"  embeddings.grad_fn = {embeddings.grad_fn}")

    embeddings, norms = utils.l2_norm(embeddings, axis=1)

    # get adaface loss
    cos_thetas = head(embeddings, norms, labels)
    if isinstance(cos_thetas, tuple):
        cos_thetas, bad_grad = cos_thetas
        labels[bad_grad.squeeze(-1)] = -100

    adaface_loss = cross_entropy_loss(cos_thetas, labels)

    print(f"\nAdaFace loss:")
    print(f"  adaface_loss = {adaface_loss.item():.4f}")
    print(f"  adaface_loss.requires_grad = {adaface_loss.requires_grad}")
    print(f"  adaface_loss.grad_fn = {adaface_loss.grad_fn}")

    # Apply weight
    weighted_loss = 0.1 * adaface_loss

    print(f"\nWeighted loss (0.1 * adaface_loss):")
    print(f"  weighted_loss = {weighted_loss.item():.4f}")

    # Backward
    weighted_loss.backward()

    # Check gradients
    print("\n" + "-" * 40)
    print("Gradient check after backward:")
    print("-" * 40)

    input_layer_grad = model.input_layer[0].weight.grad
    body_grad = model.body[0][0].weight.grad

    if input_layer_grad is not None:
        print(f"  input_layer[0] gradient magnitude: {input_layer_grad.abs().mean().item():.6e}")
    else:
        print(f"  input_layer[0] gradient: NONE!")

    if body_grad is not None:
        print(f"  body[0][0] gradient magnitude: {body_grad.abs().mean().item():.6e}")
    else:
        print(f"  body[0][0] gradient: NONE!")

    # Simulate optimizer step
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    optimizer.step()

    # Check weight changes
    input_layer_change = (model.input_layer[0].weight - initial_input_layer_weight).abs().mean().item()
    body_change = (model.body[0][0].weight - initial_body_weight).abs().mean().item()

    print(f"\nWeight changes after optimizer.step():")
    print(f"  input_layer[0] change: {input_layer_change:.6e}")
    print(f"  body[0][0] change: {body_change:.6e}")

    if input_layer_change < 1e-10 and body_change < 1e-10:
        print(f"\n  CRITICAL: Weights did NOT change! Backbone is not learning!")
        return False
    else:
        print(f"\n  OK: Weights are changing")
        return True


def diagnose_with_qaconv():
    """Test with QAConv loss to see the full picture."""
    print("\n" + "=" * 80)
    print("DIAGNOSTIC 7: Full Training with QAConv (detached)")
    print("=" * 80)

    from net import build_model
    from head import build_head
    import utils

    model = build_model('ir_18')
    model.train()
    head = build_head('adaface', embedding_size=512, class_num=100, m=0.4, h=0.333, s=64.0, t_alpha=0.01)
    cross_entropy_loss = torch.nn.CrossEntropyLoss()

    batch_size = 8
    images = torch.randn(batch_size, 3, 112, 112)
    labels = torch.randint(0, 100, (batch_size,))

    # Store initial weights
    initial_body_weight = model.body[0][0].weight.clone().detach()

    # Forward
    x = model.input_layer(images)
    for layer in model.body:
        x = layer(x)

    # QAConv path (detached)
    x_norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1, keepdim=True).view(x.size(0), 1, 1, 1)
    x_norm = torch.clamp(x_norm, min=1e-8)
    feature_maps = x / x_norm
    feature_maps = feature_maps.clone().detach().requires_grad_(True)

    # Simulate QAConv loss (detached)
    qaconv_loss = feature_maps.pow(2).sum() * 0.001  # Simulated loss

    # AdaFace path
    embeddings = model.output_layer(x)
    embeddings, norms = utils.l2_norm(embeddings, axis=1)
    cos_thetas = head(embeddings, norms, labels)
    if isinstance(cos_thetas, tuple):
        cos_thetas, _ = cos_thetas
    adaface_loss = cross_entropy_loss(cos_thetas, labels)

    # Combined loss with weights
    total_loss = 0.1 * adaface_loss + 0.9 * qaconv_loss

    print(f"\nLosses:")
    print(f"  AdaFace loss: {adaface_loss.item():.4f}")
    print(f"  QAConv loss (detached): {qaconv_loss.item():.4f}")
    print(f"  Total loss: {total_loss.item():.4f}")

    # Backward
    total_loss.backward()

    # Check which parameters got gradients
    adaface_head_grad = head.kernel.grad
    body_grad = model.body[0][0].weight.grad

    print(f"\nGradient check:")
    print(f"  AdaFace head kernel gradient: {adaface_head_grad.abs().mean().item():.6e}" if adaface_head_grad is not None else "  AdaFace head kernel gradient: NONE!")
    print(f"  Backbone body[0][0] gradient: {body_grad.abs().mean().item():.6e}" if body_grad is not None else "  Backbone body[0][0] gradient: NONE!")

    # The KEY question: Does QAConv loss contribute gradient to backbone?
    # Since feature_maps is detached, QAConv loss should NOT contribute gradient to backbone
    # Only AdaFace loss (0.1 weight) should contribute

    # Optimizer step
    optimizer = torch.optim.SGD(list(model.parameters()) + [head.kernel], lr=0.1, momentum=0.9)
    optimizer.step()

    body_change = (model.body[0][0].weight - initial_body_weight).abs().mean().item()
    print(f"\n  Backbone weight change: {body_change:.6e}")

    # Analysis
    print("\n" + "-" * 40)
    print("ANALYSIS:")
    print("-" * 40)
    print(f"  QAConv is DETACHED - its loss does NOT train backbone")
    print(f"  Only AdaFace loss (weight={0.1}) trains backbone")
    print(f"  Backbone receives only 10% of gradient signal!")
    print(f"  This may cause slow learning of backbone features")

    return body_change > 1e-10


def main():
    print("=" * 80)
    print("ADAFACE TRAINING DIAGNOSTIC TOOL")
    print("=" * 80)

    results = {}

    # Run all diagnostics
    results['gradient_flow'] = diagnose_gradient_flow()
    results['loss_weights'] = diagnose_loss_weights()
    results['training_simulation'] = diagnose_training_step_simulation()
    results['code_changes'] = diagnose_committed_vs_current()
    results['output_layer_input'] = diagnose_output_layer_input()
    results['exact_training_step'] = diagnose_exact_training_step()
    results['with_qaconv'] = diagnose_with_qaconv()

    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)

    print(f"\n1. Gradient flow to backbone: {'OK' if results['gradient_flow'] else 'FAILED'}")
    print(f"2. AdaFace loss weight: {results['loss_weights']} (backbone gets {results['loss_weights']*100}% gradient)")
    print(f"3. Training step simulation: {'OK' if results['training_simulation'] else 'FAILED'}")
    print(f"4. Output layer receives gradients: {'OK' if results['output_layer_input'] else 'FAILED'}")
    print(f"5. Exact training step: {'OK' if results['exact_training_step'] else 'FAILED'}")
    print(f"6. Full training with QAConv: {'OK' if results['with_qaconv'] else 'FAILED'}")

    print("\n" + "-" * 40)
    print("POTENTIAL ISSUES IDENTIFIED:")
    print("-" * 40)

    if results['loss_weights'] < 0.5:
        print(f"\n  ISSUE: AdaFace loss weight is only {results['loss_weights']}")
        print(f"         With QAConv detached, backbone only gets {results['loss_weights']*100}% gradient")
        print(f"         This may cause SLOW backbone learning!")
        print(f"\n  RECOMMENDATION OPTIONS:")
        print(f"    1. Increase adaface_loss_weight to 0.5 or higher")
        print(f"    2. Remove the feature_maps detach if QAConv should also train backbone")
        print(f"    3. Keep current setup but train for more epochs")

    if not results['gradient_flow']:
        print(f"\n  CRITICAL: No gradients flowing to backbone!")
        print(f"            Check if output_layer is receiving the correct input")

    if not results['exact_training_step']:
        print(f"\n  CRITICAL: Exact training step simulation FAILED!")
        print(f"            There may be an issue with the computation graph")

    print("\n" + "=" * 80)
    print("FINAL ASSESSMENT")
    print("=" * 80)

    all_ok = (results['gradient_flow'] and results['training_simulation'] and
              results['output_layer_input'] and results['exact_training_step'] and results['with_qaconv'])

    if all_ok:
        print(f"\n  All gradient checks PASSED!")
        print(f"  The backbone IS receiving gradients from AdaFace loss.")
        print(f"\n  However, with 0.9 QAConv (detached) and 0.1 AdaFace weights,")
        print(f"  the backbone only gets 10% of the gradient signal.")
        print(f"\n  This configuration may require MORE EPOCHS to reach 80-90% accuracy.")
        print(f"  Consider one of these options:")
        print(f"    - Increase adaface_loss_weight (e.g., 0.5 AdaFace, 0.5 QAConv)")
        print(f"    - Remove feature_maps detach to let QAConv also train backbone")
        print(f"    - Train for significantly more epochs with current settings")
    else:
        print(f"\n  PROBLEMS DETECTED - see issues above")

    return results


if __name__ == '__main__':
    main()
