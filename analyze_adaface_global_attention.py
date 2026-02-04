"""
Visualize AdaFace's global pooling by averaging absolute values of feature map activations.

This script demonstrates that AdaFace uses GLOBAL pooling (GNAP/GDC) which aggregates
information from ALL 7x7 spatial locations. By extracting the feature maps before global
pooling and averaging absolute values across channels, we show that activations occur
across the entire face - proving that global pooling processes all regions.

Key insight:
- Extract 7×7×512 feature maps before global pooling
- Take absolute values and average across channels → 7×7 spatial activation map
- This shows that ALL spatial locations have activations, which are then aggregated
  via global pooling (AdaptiveAvgPool2d) into a single embedding

This is fundamentally different from QAConv, which performs local-to-local matching
only on visible (non-occluded) regions.

Usage (HPC):
  python analyze_adaface_global_attention.py \
    --checkpoint /home/maass/code/OAML/experiments/adaface_ir50_casia.ckpt \
    --image_dir /home/maass/code/OAML/pics \
    --output_dir /home/maass/code/OAML/pics/out_adaface_global \
    --num_samples 6
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import net  # noqa: E402
import utils  # noqa: E402


def _valid_image_ext(p: str) -> bool:
    ext = os.path.splitext(p.lower())[1]
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(path: str, num: int) -> List[str]:
    image_paths: List[str] = []
    for root, _, files in os.walk(path):
        for f in sorted(files):
            if _valid_image_ext(f):
                image_paths.append(os.path.join(root, f))
        if len(image_paths) >= num:
            break
    return image_paths[:num]


def load_model(checkpoint_path: str, device: torch.device):
    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})

    arch = hparams.get("arch", "ir_50")
    model = net.build_model(model_name=arch)

    state_dict = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("model."):
            state_dict[k.replace("model.", "")] = v
        else:
            state_dict[k] = v

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    print("✓ Model loaded")
    return model, hparams


@torch.no_grad()
def compute_embedding(model, image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Returns normalized embedding [1, D].
    """
    x = model.input_layer(image_tensor)
    for layer in model.body:
        x = layer(x)
    emb = model.output_layer(x)
    emb, _ = utils.l2_norm(emb, axis=1)
    return emb


@torch.no_grad()
def compute_global_pooling_activation_map(
    model, image_tensor: torch.Tensor, device: torch.device
) -> np.ndarray:
    """
    Compute activation map by averaging absolute values of feature maps.
    
    This simple approach extracts the 7×7×512 feature maps before global pooling,
    takes absolute values, and averages across channels to get a 7×7 spatial map.
    
    This demonstrates that AdaFace processes information from ALL spatial locations,
    which are then aggregated via global pooling (GNAP/GDC) into a single embedding.
    
    The key insight: By showing activations across all 7×7 locations, we prove that
    global pooling aggregates information from the entire face, not just specific regions.
    
    Args:
      model: AdaFace model (must be in eval mode)
      image_tensor: [1, 3, 112, 112] input image tensor
      device: torch device
    
    Returns:
      activation_map: [7, 7] numpy array showing average absolute activation magnitude
    """
    model.eval()
    
    # Forward pass: extract feature map before global pooling
    x = model.input_layer(image_tensor.to(device))
    for layer in model.body:
        x = layer(x)
    
    # x is now [1, 512, 7, 7] - the feature map before global pooling
    
    # Take absolute values and average across channels
    # This gives us a spatial map showing where activations occur
    activation_map = x.abs().mean(dim=1)[0]  # [7, 7] - average absolute activation per spatial location
    
    # Convert to numpy
    activation_map = activation_map.cpu().numpy().astype(np.float32)
    
    # Normalize to [0, 1] for visualization using percentile-based scaling
    # This ensures we see the full range of activations
    p5, p95 = np.percentile(activation_map, [5, 95])
    if p95 > p5:
        activation_map = np.clip((activation_map - p5) / (p95 - p5 + 1e-6), 0, 1)
    else:
        activation_map = np.zeros_like(activation_map)
    
    return activation_map


def overlay_heatmap_on_image(img_rgb: np.ndarray, heat01: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """
    img_rgb: [H,W,3] uint8
    heat01: [H,W] float in [0,1]
    Returns RGB uint8 overlay
    """
    # Match occlusion figure color semantics (RdYlGn: red=low, green=high)
    cmap = plt.get_cmap("RdYlGn")
    rgba = cmap(np.clip(heat01, 0, 1))
    heat_rgb = (rgba[..., :3] * 255).astype(np.uint8)
    out = (img_rgb.astype(np.float32) * (1 - alpha) + heat_rgb.astype(np.float32) * alpha).astype(np.uint8)
    return out


def main():
    ap = argparse.ArgumentParser(description="AdaFace global feature extraction visualization")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--image_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--num_samples", type=int, default=6, help="Number of images to visualize.")
    ap.add_argument("--alpha", type=float, default=0.55)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, _ = load_model(args.checkpoint, device)

    # Build transforms once (match training conventions)
    tensor_transform = transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    raw_transform = transforms.Compose([transforms.Resize((112, 112))])

    # Find images
    paths = find_images(args.image_dir, args.num_samples)
    if not paths:
        raise SystemExit(f"No images found in {args.image_dir}")

    # Paper-friendly format similar to analyze_occlusion_comparison.py:
    # N rows x 2 cols (Original -> Feature Activation Map) with a clear horizontal arrow.
    n = len(paths)
    fig = plt.figure(figsize=(7.8, 2.6 * n))
    gs = gridspec.GridSpec(n, 2, figure=fig, hspace=0.55, wspace=0.35)

    # Shared colorbar axis on the right
    cbar_ax = fig.add_axes([0.92, 0.18, 0.02, 0.62])  # [left,bottom,width,height]
    last_im = None

    print(f"Processing {n} images...")
    for i, p in enumerate(paths):
        print(f"  [{i+1}/{n}] {os.path.basename(p)}")
        raw = Image.open(p).convert("RGB")
        raw = raw_transform(raw)
        img_rgb = np.array(raw)

        t = tensor_transform(raw).unsqueeze(0)
        activation_map = compute_global_pooling_activation_map(model, t, device)  # [7, 7]

        # Upsample activation map to image size
        h, w = img_rgb.shape[:2]
        activation_map_up = np.array(
            Image.fromarray((activation_map * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
        ).astype(np.float32) / 255.0

        ax_left = fig.add_subplot(gs[i, 0])
        ax_right = fig.add_subplot(gs[i, 1])

        ax_left.imshow(img_rgb)
        ax_left.axis("off")

        last_im = ax_right.imshow(activation_map_up, cmap="RdYlGn", vmin=0, vmax=1)
        ax_right.axis("off")

        # Arrow from original -> activation map (figure coordinates)
        left_bb = ax_left.get_position()
        right_bb = ax_right.get_position()
        y_mid = (left_bb.y0 + left_bb.y1) / 2.0
        # Much shorter arrow (leave substantial whitespace near panels)
        x_start = left_bb.x1 + 0.085
        x_end = right_bb.x0 - 0.085
        ax_left.annotate(
            "",
            xy=(x_end, y_mid),
            xytext=(x_start, y_mid),
            xycoords=fig.transFigure,
            textcoords=fig.transFigure,
            arrowprops=dict(arrowstyle="-|>", lw=3.0, color="#111111", mutation_scale=18),
        )

    # Single shared colorbar for the heatmaps
    if last_im is not None:
        cbar = plt.colorbar(last_im, cax=cbar_ax)
        cbar.set_ticks([0, 0.5, 1.0])
        cbar.set_ticklabels(["Low Activation", "Mid", "High Activation"])
        cbar.set_label("AdaFace Feature Map Activations\n(Average absolute values - all regions contribute to global pooling)", rotation=90, labelpad=10)

    out_path = os.path.join(args.output_dir, "adaface_global_feature_extraction.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()
