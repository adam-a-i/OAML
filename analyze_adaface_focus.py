"""
AdaFace focus visualization (paper figure).

This script estimates "where AdaFace focuses" by computing a Grad-CAM-style heatmap
on the backbone's last spatial feature map (e.g., 7x7x512 for 112x112 input),
using a simple AdaFace objective: embedding L2 norm.

Outputs a clean grid figure:
  Original | Overlay | Heatmap

Usage (HPC):
  python analyze_adaface_focus.py \
    --checkpoint /home/maass/code/OAML/experiments/adaface_ir50_casia.ckpt \
    --image_dir /home/maass/code/OAML/pics \
    --output_dir /home/maass/code/OAML/pics/out_adafocus \
    --num_samples 6
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

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


def compute_adaface_focus_map(model, image_tensor: torch.Tensor, ref_emb: torch.Tensor) -> np.ndarray:
    """
    Returns:
      cam_7x7: numpy [h, w] in [0,1] (spatial focus on last feature map)
    """
    model.eval()

    # Forward to get last spatial feature maps.
    # Important: make the feature map a *leaf* tensor so .grad is populated.
    with torch.no_grad():
        x = model.input_layer(image_tensor)  # [1, C, H, W]
        for layer in model.body:
            x = layer(x)

    # Leaf feature map for Grad-CAM
    x = x.detach().requires_grad_(True)

    # AdaFace embedding is produced by output_layer on final spatial map
    emb = model.output_layer(x)  # [1, D]
    emb, _ = utils.l2_norm(emb, axis=1)  # normalize

    # Scalar objective: cosine similarity to a reference embedding.
    # This produces meaningful gradients indicating which regions increase identity similarity.
    ref_emb = ref_emb.to(device=emb.device, dtype=emb.dtype)
    score = (emb * ref_emb).sum()

    # Backprop for CAM
    model.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad.zero_()
    score.backward(retain_graph=False)

    grads = x.grad  # [1, C, h, w]
    if grads is None:
        raise RuntimeError("No gradients for feature map; cannot compute CAM.")

    # Grad-CAM weights: global-average grads over spatial dims
    weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
    cam = (weights * x).sum(dim=1, keepdim=False)  # [1, h, w]
    cam = F.relu(cam)
    cam = cam.squeeze(0)

    # Normalize to [0,1]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)
    return cam.detach().cpu().numpy().astype(np.float32)


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
    ap = argparse.ArgumentParser(description="AdaFace focus visualization (Grad-CAM style)")
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--image_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--num_samples", type=int, default=6, help="Number of images to visualize (excluding reference).")
    ap.add_argument(
        "--ref_image",
        type=str,
        default=None,
        help="Reference image path for similarity-based focus. Default: first image found in --image_dir.",
    )
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

    # Grab extra so we can exclude the reference image from the visualization set.
    paths_all = find_images(args.image_dir, args.num_samples + 1)
    if not paths_all:
        raise SystemExit(f"No images found in {args.image_dir}")

    # Choose reference image (identity target) - NOT shown in the final figure.
    ref_path = args.ref_image or paths_all[0]
    print(f"Using reference image for similarity: {ref_path}")

    # Compute reference embedding
    ref_raw = Image.open(ref_path).convert("RGB")
    ref_raw = raw_transform(ref_raw)
    ref_t = tensor_transform(ref_raw).unsqueeze(0).to(device)
    ref_emb = compute_embedding(model, ref_t).detach()

    # Exclude reference image from displayed set
    paths = [p for p in paths_all if os.path.abspath(p) != os.path.abspath(ref_path)]
    paths = paths[: args.num_samples]
    if not paths:
        raise SystemExit("After excluding reference image, no images remain to visualize.")

    # Paper-friendly format similar to analyze_occlusion_comparison.py:
    # N rows x 2 cols (Original -> Focus Map) with a clear horizontal arrow.
    n = len(paths)
    fig = plt.figure(figsize=(7.8, 2.6 * n))
    gs = gridspec.GridSpec(n, 2, figure=fig, hspace=0.55, wspace=0.35)

    # Shared colorbar axis on the right
    cbar_ax = fig.add_axes([0.92, 0.18, 0.02, 0.62])  # [left,bottom,width,height]
    last_im = None

    for i, p in enumerate(paths):
        raw = Image.open(p).convert("RGB")
        raw = raw_transform(raw)
        img_rgb = np.array(raw)

        t = tensor_transform(raw).unsqueeze(0).to(device)
        cam = compute_adaface_focus_map(model, t, ref_emb)  # similarity-based CAM

        # Upsample CAM to image size (like analyze_occlusion_maps.py does for occlusion maps)
        h, w = img_rgb.shape[:2]
        cam_up = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
        ).astype(np.float32) / 255.0

        ax_left = fig.add_subplot(gs[i, 0])
        ax_right = fig.add_subplot(gs[i, 1])

        ax_left.imshow(img_rgb)
        ax_left.axis("off")

        last_im = ax_right.imshow(cam_up, cmap="RdYlGn", vmin=0, vmax=1)
        ax_right.axis("off")

        # Arrow from original -> focus map (figure coordinates)
        left_bb = ax_left.get_position()
        right_bb = ax_right.get_position()
        y_mid = (left_bb.y0 + left_bb.y1) / 2.0
        x_start = left_bb.x1 + 0.01
        x_end = right_bb.x0 - 0.01
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
        cbar.set_ticklabels(["Low Focus", "Mid", "High Focus"])
        cbar.set_label("AdaFace Focus (Grad-CAM intensity)", rotation=90, labelpad=10)

    out_path = os.path.join(args.output_dir, "adaface_focus.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()

