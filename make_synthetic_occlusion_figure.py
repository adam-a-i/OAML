#!/usr/bin/env python3
"""
Generate segmentation-looking visibility/occlusion maps from user-drawn scribbles,
and export a paper-ready composite figure (original / overlay / map).

Expected inputs (in --input_dir):
  - Face images:          <id>.jpg / <id>.png / <id>.jpeg
  - Scribble masks:       <id>_scribble.png

Scribble convention:
  - White/bright pixels in *_scribble.png indicate "VISIBLE" regions (non-occluded).
  - Everything else is treated as unknown and will be expanded + smoothed into a soft visibility map.

Outputs (in --output_dir):
  - Per-sample: <id>_orig.png, <id>_overlay.png, <id>_map.png
  - Combined grid figure: <figure_name>.png
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

try:
    import matplotlib.cm as _cm  # type: ignore
except Exception:
    _cm = None


@dataclass
class SamplePaths:
    stem: str
    image_path: str
    scribble_path: str


def _is_image_file(name: str) -> bool:
    ext = os.path.splitext(name.lower())[1]
    return ext in {".jpg", ".jpeg", ".png", ".webp"}


def discover_samples(input_dir: str) -> List[SamplePaths]:
    files = sorted(os.listdir(input_dir))
    images = [f for f in files if _is_image_file(f) and not f.endswith("_scribble.png")]
    samples: List[SamplePaths] = []
    for img in images:
        stem = os.path.splitext(img)[0]
        scrib = f"{stem}_scribble.png"
        scrib_path = os.path.join(input_dir, scrib)
        if os.path.exists(scrib_path):
            samples.append(
                SamplePaths(
                    stem=stem,
                    image_path=os.path.join(input_dir, img),
                    scribble_path=scrib_path,
                )
            )
    return samples


def discover_samples_with_suffix(input_dir: str, scribble_suffix: str) -> List[SamplePaths]:
    """
    Like discover_samples(), but allows customizing the scribble suffix.
    Example: scribble_suffix=\"_scribble\" expects <id>_scribble.png
    """
    files = sorted(os.listdir(input_dir))
    images = [f for f in files if _is_image_file(f) and not f.endswith(f"{scribble_suffix}.png")]
    samples: List[SamplePaths] = []
    for img in images:
        stem = os.path.splitext(img)[0]
        scrib = f"{stem}{scribble_suffix}.png"
        scrib_path = os.path.join(input_dir, scrib)
        if os.path.exists(scrib_path):
            samples.append(
                SamplePaths(
                    stem=stem,
                    image_path=os.path.join(input_dir, img),
                    scribble_path=scrib_path,
                )
            )
    return samples


def debug_expected_pairs(input_dir: str, scribble_suffix: str) -> str:
    files = sorted(os.listdir(input_dir))
    images = [f for f in files if _is_image_file(f) and not f.endswith(f"{scribble_suffix}.png")]
    scribbles = [f for f in files if f.lower().endswith(f"{scribble_suffix}.png")]
    lines = []
    lines.append(f"[DEBUG] input_dir: {input_dir}")
    lines.append(f"[DEBUG] scribble_suffix: {scribble_suffix!r} (expects <stem>{scribble_suffix}.png)")
    lines.append(f"[DEBUG] found images ({len(images)}): {images}")
    lines.append(f"[DEBUG] found scribbles ({len(scribbles)}): {scribbles}")
    missing = []
    for img in images:
        stem = os.path.splitext(img)[0]
        expected = f"{stem}{scribble_suffix}.png"
        if expected not in files:
            missing.append(expected)
    if missing:
        lines.append(f"[DEBUG] missing scribbles for these images ({len(missing)}): {missing}")
    return "\n".join(lines)


def read_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def read_scribble_visible_mask(path: str, target_hw: Tuple[int, int]) -> np.ndarray:
    """
    Returns float32 mask in [0,1], where 1 means visible.
    """
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Failed to read scribble mask: {path}")
    h, w = target_hw
    if (m.shape[0], m.shape[1]) != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    # Normalize; treat bright pixels as visible scribbles
    m = (m.astype(np.float32) / 255.0)
    visible = (m >= 0.5).astype(np.float32)
    return visible


def generate_soft_visibility_map(
    visible_scribble: np.ndarray,
    rng: np.random.Generator,
    *,
    dilate_radius: int = 6,
    blur_sigma: float = 6.0,
    noise_strength: float = 0.12,
    edge_soften: float = 0.5,
    spread_sigma: float = 18.0,
) -> np.ndarray:
    """
    Turn sparse visible scribbles into a smooth visibility map in [0,1].
    """
    h, w = visible_scribble.shape[:2]

    # 1) Slightly thicken scribbles so they become stable "seeds"
    k = max(1, int(dilate_radius))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    seed = cv2.dilate(visible_scribble, kernel, iterations=1).astype(np.uint8)

    # 2) Distance-decay from the scribble seed (prevents "whole face becomes visible" failure mode)
    # dist=0 at scribble, grows outward; visibility decays smoothly with distance.
    inv = (1 - seed).astype(np.uint8)  # 1 where NOT scribbled
    dist = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3).astype(np.float32)
    sigma = max(1.0, float(spread_sigma))
    soft = np.exp(-0.5 * (dist / sigma) ** 2).astype(np.float32)
    # Ensure scribble pixels are strongly visible
    soft = np.maximum(soft, seed.astype(np.float32))

    # 3) Optional smoothing (keeps maps "occlusion-head-like", but won't inflate visibility too far)
    if blur_sigma > 0:
        soft = cv2.GaussianBlur(soft, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    # 4) Add low-frequency noise (blurred noise) for "segmentation-like" texture
    if noise_strength > 0:
        noise = rng.normal(loc=0.0, scale=1.0, size=(h, w)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=blur_sigma * 0.6 + 1.0, sigmaY=blur_sigma * 0.6 + 1.0)
        noise = noise / (np.std(noise) + 1e-6)
        # Apply noise mostly near boundaries (soft*(1-soft) peaks around 0.5)
        boundary_mask = (soft * (1.0 - soft)) * 4.0  # ~[0,1]
        soft = soft + (noise_strength * boundary_mask) * noise

    # 5) Gentle edge softening / contrast shaping
    # Map to [0,1] then apply a sigmoid-ish curve
    soft = np.clip(soft, 0.0, 1.0)
    if edge_soften > 0:
        # pull mid-values slightly toward 0/1 depending on edge_soften
        # edge_soften in [0,1]: 0 no-op, 1 stronger shaping
        gamma = 1.0 + 2.0 * edge_soften
        soft = np.power(soft, 1.0 / gamma)

    return np.clip(soft, 0.0, 1.0).astype(np.float32)


def _contrast_stretch_01(x: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """
    Robustly stretch values to [0,1] using percentiles. Helps get a red->green spectrum like matplotlib.
    """
    x = x.astype(np.float32)
    a = float(np.percentile(x, lo))
    b = float(np.percentile(x, hi))
    if b <= a + 1e-6:
        return np.clip(x, 0.0, 1.0)
    y = (x - a) / (b - a)
    return np.clip(y, 0.0, 1.0)


def _add_display_texture(x01: np.ndarray, rng: np.random.Generator, strength: float = 0.08) -> np.ndarray:
    """
    Add subtle display-only texture/noise so maps look closer to real predicted maps.
    """
    if strength <= 0:
        return x01
    h, w = x01.shape
    # Low-frequency only (real occlusion maps look smooth; avoid speckle)
    low = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
    low = cv2.GaussianBlur(low, (0, 0), sigmaX=max(1.0, min(h, w) / 40.0), sigmaY=max(1.0, min(h, w) / 40.0))
    low = low / (np.std(low) + 1e-6)
    y = x01 + strength * low
    return np.clip(y, 0.0, 1.0)


def make_overlay_red_occluded(rgb: np.ndarray, visibility: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """
    Overlay red where occluded (1-visibility).
    """
    occ = 1.0 - visibility
    occ = np.clip(occ, 0.0, 1.0)
    red = np.zeros_like(rgb, dtype=np.float32)
    red[..., 0] = 255.0  # R
    base = rgb.astype(np.float32)
    out = base * (1.0 - alpha * occ[..., None]) + red * (alpha * occ[..., None])
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def make_map_green_visible(visibility: np.ndarray) -> np.ndarray:
    """
    Create a green-ish heatmap where green=visible, red=occluded (similar to your examples).
    """
    # First: stretch gently so we get a red->yellow->green spectrum without saturating everything to neon green.
    v = np.clip(visibility.astype(np.float32), 0.0, 1.0)
    v = _contrast_stretch_01(v, lo=5.0, hi=95.0)
    # Keep highs from hitting 1.0 too often (reduces "too green" look)
    v = 0.08 + 0.84 * v  # map into [0.08, 0.92]

    # Prefer matplotlib's RdYlGn if available (closest match to analyze_occlusion_comparison.py)
    if _cm is not None:
        rgba = _cm.get_cmap("RdYlGn")(v)  # H,W,4 in 0..1
        rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
        return rgb

    # Fallback: simple red->yellow->green (no blue channel)
    t = v[..., None]  # H,W,1
    rgb = np.zeros((v.shape[0], v.shape[1], 3), dtype=np.float32)
    lo_mask = (t <= 0.5).astype(np.float32)
    hi_mask = 1.0 - lo_mask
    a = np.clip(t / 0.5, 0.0, 1.0)
    rgb[..., 0:1] += lo_mask * 255.0
    rgb[..., 1:2] += lo_mask * (255.0 * a)
    b = np.clip((t - 0.5) / 0.5, 0.0, 1.0)
    rgb[..., 0:1] += hi_mask * (255.0 * (1.0 - b))
    rgb[..., 1:2] += hi_mask * 255.0
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def save_rgb(path: str, rgb: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def make_grid_figure(
    rows: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, float]],
    out_path: str,
    *,
    title: str = "Occlusion Head Predictions: Occluded vs Clean Faces (Synthetic)",
    subtitle_left: str = "Original",
    subtitle_mid: str = "Overlay (Red=Occluded)",
    subtitle_right: str = "Map (Green=Visible)",
    cell_pad: int = 18,
    header_h: int = 90,
) -> None:
    """
    Create a single PNG with rows, each row = (orig, overlay, map).
    Also draws simple arrows between panels.
    """
    if len(rows) == 0:
        raise ValueError("No rows to render.")

    # Images may have different sizes; normalize to a common size for the grid.
    max_h = max(r[1].shape[0] for r in rows)
    max_w = max(r[1].shape[1] for r in rows)
    col_w = int(max_w)
    col_h = int(max_h)
    n = len(rows)

    canvas_h = header_h + n * col_h + (n + 1) * cell_pad
    canvas_w = 3 * col_w + 4 * cell_pad
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # Title
    cv2.putText(
        canvas,
        title,
        (cell_pad, int(header_h * 0.55)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    # Column subtitles
    y_sub = header_h - 20
    subtitles = [subtitle_left, subtitle_mid, subtitle_right]
    for c, sub in enumerate(subtitles):
        x = cell_pad + c * (col_w + cell_pad)
        cv2.putText(canvas, sub, (x, y_sub), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)

    for i, (stem, orig, overlay, vis_map, mu) in enumerate(rows):
        # Resize each panel to the common grid size (keep it simple for paper figures)
        if (orig.shape[0], orig.shape[1]) != (col_h, col_w):
            orig = cv2.resize(orig, (col_w, col_h), interpolation=cv2.INTER_AREA)
        if (overlay.shape[0], overlay.shape[1]) != (col_h, col_w):
            overlay = cv2.resize(overlay, (col_w, col_h), interpolation=cv2.INTER_AREA)
        if (vis_map.shape[0], vis_map.shape[1]) != (col_h, col_w):
            # Use linear/area to avoid blocky heatmaps
            vis_map = cv2.resize(vis_map, (col_w, col_h), interpolation=cv2.INTER_LINEAR)

        y0 = header_h + cell_pad + i * (col_h + cell_pad)
        x0 = cell_pad
        x1 = cell_pad * 2 + col_w
        x2 = cell_pad * 3 + col_w * 2

        canvas[y0 : y0 + col_h, x0 : x0 + col_w] = orig
        canvas[y0 : y0 + col_h, x1 : x1 + col_w] = overlay
        canvas[y0 : y0 + col_h, x2 : x2 + col_w] = vis_map

        # Row label + mean visibility
        label = f"{stem}   (μ={mu:.3f})"
        cv2.putText(canvas, label, (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2, cv2.LINE_AA)

        # Draw arrows: orig -> overlay -> map
        y_mid = y0 + col_h // 2
        arrow_color = (40, 40, 40)
        cv2.arrowedLine(canvas, (x0 + col_w + 6, y_mid), (x1 - 6, y_mid), arrow_color, 2, tipLength=0.02)
        cv2.arrowedLine(canvas, (x1 + col_w + 6, y_mid), (x2 - 6, y_mid), arrow_color, 2, tipLength=0.02)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, bgr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default=None, help="Folder containing images + scribble masks")
    ap.add_argument("--output_dir", default=None, help="Where to write outputs")
    ap.add_argument(
        "--scribble_suffix",
        type=str,
        default="_scribble",
        help="Scribble filename suffix before .png (default: _scribble).",
    )
    ap.add_argument("--image_path", default=None, help="Single image path (overrides --input_dir discovery)")
    ap.add_argument("--scribble_path", default=None, help="Single scribble mask path (white=visible)")
    ap.add_argument("--max_samples", type=int, default=6, help="Max samples to include in the grid figure")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dilate_radius", type=int, default=18)
    ap.add_argument("--blur_sigma", type=float, default=10.0)
    ap.add_argument("--noise_strength", type=float, default=0.12)
    ap.add_argument(
        "--spread_sigma",
        type=float,
        default=18.0,
        help="How far visibility spreads from scribbles (pixels). Smaller = tighter around scribbles.",
    )
    ap.add_argument(
        "--display_noise_strength",
        type=float,
        default=0.03,
        help="Subtle display-only texture for the map (default: 0.03). Set 0 to disable.",
    )
    ap.add_argument("--edge_soften", type=float, default=0.5)
    ap.add_argument("--overlay_alpha", type=float, default=0.55)
    ap.add_argument("--figure_name", type=str, default="synthetic_occlusion_figure")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Mode A: single image + scribble provided explicitly
    if args.image_path is not None or args.scribble_path is not None:
        if not args.image_path or not args.scribble_path:
            raise SystemExit("--image_path and --scribble_path must be provided together.")
        samples = [
            SamplePaths(
                stem=os.path.splitext(os.path.basename(args.image_path))[0],
                image_path=args.image_path,
                scribble_path=args.scribble_path,
            )
        ]
        if args.output_dir is None:
            raise SystemExit("--output_dir is required when using --image_path/--scribble_path.")
    else:
        # Mode B: discover pairs in a directory
        if args.input_dir is None or args.output_dir is None:
            raise SystemExit("Provide either (--input_dir and --output_dir) or (--image_path and --scribble_path and --output_dir).")
        samples = discover_samples_with_suffix(args.input_dir, args.scribble_suffix)
        if len(samples) == 0:
            msg = (
                f"No samples found in {args.input_dir}. Expected <id>.(jpg/png) and <id>{args.scribble_suffix}.png pairs.\n"
                + debug_expected_pairs(args.input_dir, args.scribble_suffix)
                + "\n"
                + "Fix: create matching scribble files (white=visible) OR run single-file mode with "
                + "--image_path ... --scribble_path ... --output_dir ..."
            )
            raise SystemExit(msg)

    rows = []
    for s in samples[: args.max_samples]:
        rgb = read_rgb(s.image_path)
        h, w = rgb.shape[:2]
        visible_scrib = read_scribble_visible_mask(s.scribble_path, (h, w))
        vis = generate_soft_visibility_map(
            visible_scrib,
            rng,
            dilate_radius=args.dilate_radius,
            blur_sigma=args.blur_sigma,
            noise_strength=args.noise_strength,
            edge_soften=args.edge_soften,
            spread_sigma=args.spread_sigma,
        )
        # Add display-only texture and smoothing for a more realistic RdYlGn look (like analyze_occlusion_comparison.py)
        vis_for_display = _contrast_stretch_01(vis, lo=5.0, hi=95.0)
        vis_for_display = _add_display_texture(
            vis_for_display,
            rng,
            strength=max(0.0, float(args.display_noise_strength)),
        )
        # Extra blur to remove blockiness in the final visualization
        vis_for_display = cv2.GaussianBlur(vis_for_display, (0, 0), sigmaX=2.2, sigmaY=2.2)
        overlay = make_overlay_red_occluded(rgb, vis, alpha=args.overlay_alpha)
        vis_map = make_map_green_visible(vis_for_display)
        mu = float(vis.mean())

        # Save per-sample outputs
        save_rgb(os.path.join(args.output_dir, f"{s.stem}_orig.png"), rgb)
        save_rgb(os.path.join(args.output_dir, f"{s.stem}_overlay.png"), overlay)
        save_rgb(os.path.join(args.output_dir, f"{s.stem}_map.png"), vis_map)

        rows.append((s.stem, rgb, overlay, vis_map, mu))

    fig_path = os.path.join(args.output_dir, f"{args.figure_name}.png")
    make_grid_figure(rows, fig_path)
    print(f"[OK] Wrote {len(rows)} samples + figure to: {args.output_dir}")
    print(f"[OK] Figure: {fig_path}")


if __name__ == "__main__":
    main()

