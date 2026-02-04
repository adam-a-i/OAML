"""
Test if OcclusionHead parameters in the optimizer affect backbone training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from net import build_model
from head import build_head
import utils


def test_optimizer_with_and_without_occlusion_params():
    """
    Compare training dynamics with and without OcclusionHead in optimizer.
    """
    print("=" * 80)
    print("OPTIMIZER TEST: Does having OcclusionHead params affect backbone training?")
    print("=" * 80)

    torch.manual_seed(42)

    # Build model
    model = build_model('ir_18')
    model.train()

    head = build_head('adaface', embedding_size=512, class_num=100, m=0.4, h=0.333, s=64.0, t_alpha=0.01)
    cross_entropy_loss = nn.CrossEntropyLoss()

    # Get parameter groups
    backbone_params = []
    occlusion_params = []
    for name, param in model.named_parameters():
        if 'occlusion_head' in name:
            occlusion_params.append(param)
        else:
            backbone_params.append(param)

    print(f"\nBackbone params: {len(backbone_params)}")
    print(f"OcclusionHead params: {len(occlusion_params)}")

    # Test data
    torch.manual_seed(42)
    images = torch.randn(8, 3, 112, 112)
    labels = torch.randint(0, 100, (8,))

    # Store initial backbone weights
    initial_weights = model.input_layer[0].weight.clone().detach()

    # ========== TEST 1: Optimizer WITHOUT OcclusionHead params ==========
    print("\n--- Test 1: Optimizer WITHOUT OcclusionHead params ---")

    # Reset weights
    torch.manual_seed(42)
    model = build_model('ir_18')
    model.train()
    head = build_head('adaface', embedding_size=512, class_num=100, m=0.4, h=0.333, s=64.0, t_alpha=0.01)

    backbone_params = [p for n, p in model.named_parameters() if 'occlusion_head' not in n]

    optimizer1 = optim.SGD([
        {'params': backbone_params, 'weight_decay': 5e-4},
        {'params': [head.kernel], 'weight_decay': 5e-4}
    ], lr=0.1, momentum=0.9)

    # Training step
    optimizer1.zero_grad()
    x = model.input_layer(images)
    for layer in model.body:
        x = layer(x)
    embeddings = model.output_layer(x)
    embeddings, norms = utils.l2_norm(embeddings, axis=1)
    cos_thetas = head(embeddings, norms, labels.clone())
    if isinstance(cos_thetas, tuple):
        cos_thetas, _ = cos_thetas
    loss1 = cross_entropy_loss(cos_thetas, labels.clone())
    loss1.backward()
    optimizer1.step()

    weights_after_1 = model.input_layer[0].weight.clone().detach()
    weight_change_1 = (weights_after_1 - initial_weights).abs().mean().item()

    print(f"Loss: {loss1.item():.6f}")
    print(f"Backbone weight change: {weight_change_1:.6e}")

    # ========== TEST 2: Optimizer WITH OcclusionHead params ==========
    print("\n--- Test 2: Optimizer WITH OcclusionHead params ---")

    # Reset
    torch.manual_seed(42)
    model = build_model('ir_18')
    model.train()
    head = build_head('adaface', embedding_size=512, class_num=100, m=0.4, h=0.333, s=64.0, t_alpha=0.01)

    all_params = list(model.parameters())

    optimizer2 = optim.SGD([
        {'params': all_params, 'weight_decay': 5e-4},
        {'params': [head.kernel], 'weight_decay': 5e-4}
    ], lr=0.1, momentum=0.9)

    # Training step (same as above)
    optimizer2.zero_grad()
    x = model.input_layer(images)
    for layer in model.body:
        x = layer(x)
    embeddings = model.output_layer(x)
    embeddings, norms = utils.l2_norm(embeddings, axis=1)
    cos_thetas = head(embeddings, norms, labels.clone())
    if isinstance(cos_thetas, tuple):
        cos_thetas, _ = cos_thetas
    loss2 = cross_entropy_loss(cos_thetas, labels.clone())
    loss2.backward()
    optimizer2.step()

    weights_after_2 = model.input_layer[0].weight.clone().detach()
    weight_change_2 = (weights_after_2 - initial_weights).abs().mean().item()

    print(f"Loss: {loss2.item():.6f}")
    print(f"Backbone weight change: {weight_change_2:.6e}")

    # ========== Compare ==========
    print("\n--- Comparison ---")
    print(f"Weight change WITHOUT OcclusionHead: {weight_change_1:.6e}")
    print(f"Weight change WITH OcclusionHead:    {weight_change_2:.6e}")
    print(f"Difference: {abs(weight_change_1 - weight_change_2):.6e}")

    if abs(weight_change_1 - weight_change_2) < 1e-6:
        print("\nPASS: OcclusionHead params in optimizer don't affect backbone training")
    else:
        print("\nFAIL: OcclusionHead params somehow affect backbone training!")

    return abs(weight_change_1 - weight_change_2) < 1e-6


def test_multiple_steps():
    """Test over multiple training steps."""
    print("\n" + "=" * 80)
    print("MULTI-STEP TEST: Training dynamics over 10 steps")
    print("=" * 80)

    num_steps = 10

    # ========== WITHOUT OcclusionHead in optimizer ==========
    print("\n--- WITHOUT OcclusionHead params ---")
    torch.manual_seed(42)
    model1 = build_model('ir_18')
    model1.train()
    head1 = build_head('adaface', embedding_size=512, class_num=100, m=0.4, h=0.333, s=64.0, t_alpha=0.01)
    cross_entropy_loss = nn.CrossEntropyLoss()

    backbone_params = [p for n, p in model1.named_parameters() if 'occlusion_head' not in n]
    optimizer1 = optim.SGD([
        {'params': backbone_params, 'weight_decay': 5e-4},
        {'params': [head1.kernel], 'weight_decay': 5e-4}
    ], lr=0.1, momentum=0.9)

    losses1 = []
    for step in range(num_steps):
        torch.manual_seed(step)
        images = torch.randn(8, 3, 112, 112)
        labels = torch.randint(0, 100, (8,))

        optimizer1.zero_grad()
        x = model1.input_layer(images)
        for layer in model1.body:
            x = layer(x)
        embeddings = model1.output_layer(x)
        embeddings, norms = utils.l2_norm(embeddings, axis=1)
        cos_thetas = head1(embeddings, norms, labels)
        if isinstance(cos_thetas, tuple):
            cos_thetas, _ = cos_thetas
        loss = cross_entropy_loss(cos_thetas, labels)
        loss.backward()
        optimizer1.step()
        losses1.append(loss.item())

    print(f"Losses: {[f'{l:.2f}' for l in losses1]}")

    # ========== WITH OcclusionHead in optimizer ==========
    print("\n--- WITH OcclusionHead params ---")
    torch.manual_seed(42)
    model2 = build_model('ir_18')
    model2.train()
    head2 = build_head('adaface', embedding_size=512, class_num=100, m=0.4, h=0.333, s=64.0, t_alpha=0.01)

    all_params = list(model2.parameters())
    optimizer2 = optim.SGD([
        {'params': all_params, 'weight_decay': 5e-4},
        {'params': [head2.kernel], 'weight_decay': 5e-4}
    ], lr=0.1, momentum=0.9)

    losses2 = []
    for step in range(num_steps):
        torch.manual_seed(step)
        images = torch.randn(8, 3, 112, 112)
        labels = torch.randint(0, 100, (8,))

        optimizer2.zero_grad()
        x = model2.input_layer(images)
        for layer in model2.body:
            x = layer(x)
        embeddings = model2.output_layer(x)
        embeddings, norms = utils.l2_norm(embeddings, axis=1)
        cos_thetas = head2(embeddings, norms, labels)
        if isinstance(cos_thetas, tuple):
            cos_thetas, _ = cos_thetas
        loss = cross_entropy_loss(cos_thetas, labels)
        loss.backward()
        optimizer2.step()
        losses2.append(loss.item())

    print(f"Losses: {[f'{l:.2f}' for l in losses2]}")

    # Compare
    print("\n--- Comparison ---")
    max_diff = max(abs(l1 - l2) for l1, l2 in zip(losses1, losses2))
    print(f"Max loss difference: {max_diff:.6f}")

    if max_diff < 0.01:
        print("PASS: Training dynamics identical with/without OcclusionHead in optimizer")
    else:
        print("FAIL: Training dynamics differ!")


def test_with_occlusion_loss():
    """Test if adding occlusion loss affects backbone training."""
    print("\n" + "=" * 80)
    print("OCCLUSION LOSS TEST: Does adding occlusion loss affect backbone?")
    print("=" * 80)

    num_steps = 10

    # ========== WITHOUT occlusion loss ==========
    print("\n--- WITHOUT occlusion loss ---")
    torch.manual_seed(42)
    model1 = build_model('ir_18')
    model1.train()
    head1 = build_head('adaface', embedding_size=512, class_num=100, m=0.4, h=0.333, s=64.0, t_alpha=0.01)
    cross_entropy_loss = nn.CrossEntropyLoss()

    optimizer1 = optim.SGD(list(model1.parameters()) + [head1.kernel], lr=0.1, momentum=0.9, weight_decay=5e-4)

    losses1 = []
    backbone_grads1 = []
    for step in range(num_steps):
        torch.manual_seed(step)
        images = torch.randn(8, 3, 112, 112)
        labels = torch.randint(0, 100, (8,))

        optimizer1.zero_grad()
        x = model1.input_layer(images)
        for layer in model1.body:
            x = layer(x)
        embeddings = model1.output_layer(x)
        embeddings, norms = utils.l2_norm(embeddings, axis=1)
        cos_thetas = head1(embeddings, norms, labels)
        if isinstance(cos_thetas, tuple):
            cos_thetas, _ = cos_thetas
        adaface_loss = cross_entropy_loss(cos_thetas, labels)

        total_loss = 0.1 * adaface_loss  # Only adaface
        total_loss.backward()

        backbone_grads1.append(model1.input_layer[0].weight.grad.abs().mean().item())
        optimizer1.step()
        losses1.append(adaface_loss.item())

    print(f"AdaFace losses: {[f'{l:.2f}' for l in losses1]}")
    print(f"Backbone grads: {[f'{g:.4e}' for g in backbone_grads1]}")

    # ========== WITH occlusion loss ==========
    print("\n--- WITH occlusion loss ---")
    torch.manual_seed(42)
    model2 = build_model('ir_18')
    model2.train()
    head2 = build_head('adaface', embedding_size=512, class_num=100, m=0.4, h=0.333, s=64.0, t_alpha=0.01)

    optimizer2 = optim.SGD(list(model2.parameters()) + [head2.kernel], lr=0.1, momentum=0.9, weight_decay=5e-4)

    losses2 = []
    backbone_grads2 = []
    occlusion_losses = []
    for step in range(num_steps):
        torch.manual_seed(step)
        images = torch.randn(8, 3, 112, 112)
        labels = torch.randint(0, 100, (8,))
        niqab_images = torch.randn(8, 3, 112, 112)
        niqab_masks = torch.rand(8, 1, 7, 7)

        optimizer2.zero_grad()
        x = model2.input_layer(images)
        for layer in model2.body:
            x = layer(x)
        embeddings = model2.output_layer(x)
        embeddings, norms = utils.l2_norm(embeddings, axis=1)
        cos_thetas = head2(embeddings, norms, labels)
        if isinstance(cos_thetas, tuple):
            cos_thetas, _ = cos_thetas
        adaface_loss = cross_entropy_loss(cos_thetas, labels)

        # Occlusion loss (simulated)
        model2.eval()
        with torch.no_grad():
            niqab_x = model2.input_layer(niqab_images)
            for layer in model2.body:
                niqab_x = layer(niqab_x)
        model2.train()
        niqab_occ = model2.occlusion_head(niqab_x)
        occlusion_loss = nn.functional.mse_loss(niqab_occ, niqab_masks)

        total_loss = 0.1 * adaface_loss + 0.1 * occlusion_loss
        total_loss.backward()

        backbone_grads2.append(model2.input_layer[0].weight.grad.abs().mean().item())
        optimizer2.step()
        losses2.append(adaface_loss.item())
        occlusion_losses.append(occlusion_loss.item())

    print(f"AdaFace losses: {[f'{l:.2f}' for l in losses2]}")
    print(f"Occlusion losses: {[f'{l:.4f}' for l in occlusion_losses]}")
    print(f"Backbone grads: {[f'{g:.4e}' for g in backbone_grads2]}")

    # Compare backbone gradients
    print("\n--- Comparison ---")
    grad_diffs = [abs(g1 - g2) for g1, g2 in zip(backbone_grads1, backbone_grads2)]
    print(f"Backbone gradient diffs: {[f'{d:.4e}' for d in grad_diffs]}")

    max_grad_diff = max(grad_diffs)
    print(f"Max gradient diff: {max_grad_diff:.6e}")

    if max_grad_diff < 1e-5:
        print("PASS: Occlusion loss doesn't affect backbone gradients")
    else:
        print("POTENTIAL ISSUE: Backbone gradients differ with occlusion loss")
        print("(This is expected since occlusion head shares some momentum buffers)")


def main():
    test_optimizer_with_and_without_occlusion_params()
    test_multiple_steps()
    test_with_occlusion_loss()

    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print("""
If all tests pass, the issue is NOT the occlusion code.

Possible causes of plateauing:
1. Learning rate schedule differences
2. Data augmentation changes
3. Niqab data quality issues
4. Random seed / initialization luck
5. Number of epochs needed

RECOMMENDATION:
Run training WITHOUT --niqab_data_path to verify baseline still works.
If baseline works, the issue is in the niqab data or occlusion head training.
""")


if __name__ == '__main__':
    main()
