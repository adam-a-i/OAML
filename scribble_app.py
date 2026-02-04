#!/usr/bin/env python3
"""
Tiny Flask app to draw "visible" scribbles (white) over face images and save *_scribble.png masks.

Usage (on your HPC):
  pip install flask
  python scribble_app.py --dir /home/maass/code/OAML/pics --port 7860

Then open in your browser (via SSH tunnel / port-forward):
  http://127.0.0.1:7860

Scribble convention:
  - White pixels = VISIBLE (non-occluded)
  - Black/transparent = unknown
"""

from __future__ import annotations

import argparse
import base64
import io
import os
from dataclasses import dataclass
from typing import List, Optional

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, url_for

import numpy as np
from PIL import Image


ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class Item:
    stem: str
    filename: str
    # Backward/forward compatibility: some templates expect scribble_filename,
    # newer ones use scribble_relpath (e.g., masks/<name>_scribble.png).
    scribble_filename: str
    scribble_relpath: str
    has_scribble: bool


def list_items(img_dir: str, scribble_suffix: str, masks_subdir: str) -> List[Item]:
    files = sorted(os.listdir(img_dir))
    items: List[Item] = []
    masks_dir = os.path.join(img_dir, masks_subdir)
    for f in files:
        ext = os.path.splitext(f.lower())[1]
        if ext not in ALLOWED_EXTS:
            continue
        stem = os.path.splitext(f)[0]
        scrib = f"{stem}{scribble_suffix}.png"
        scrib_rel = os.path.join(masks_subdir, scrib)
        items.append(
            Item(
                stem=stem,
                filename=f,
                scribble_filename=scrib,
                scribble_relpath=scrib_rel,
                has_scribble=os.path.exists(os.path.join(img_dir, scrib_rel)),
            )
        )
    return items


def load_image_size(path: str) -> tuple[int, int]:
    with Image.open(path) as im:
        im = im.convert("RGB")
        return im.size  # (w,h)


def decode_data_url_png(data_url: str) -> Image.Image:
    # expected: data:image/png;base64,....
    if "," not in data_url:
        raise ValueError("Invalid data URL")
    header, b64 = data_url.split(",", 1)
    if "base64" not in header:
        raise ValueError("Expected base64 data URL")
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGBA")


def ensure_rgba_canvas_size(mask_rgba: Image.Image, w: int, h: int) -> Image.Image:
    if mask_rgba.size != (w, h):
        mask_rgba = mask_rgba.resize((w, h), resample=Image.NEAREST)
    return mask_rgba


def rgba_to_visible_scribble(mask_rgba: Image.Image) -> Image.Image:
    """
    Convert RGBA canvas to grayscale scribble:
      - non-transparent pixels -> 255
      - transparent pixels -> 0
    """
    arr = np.array(mask_rgba)  # H,W,4
    alpha = arr[..., 3]
    out = (alpha > 0).astype(np.uint8) * 255
    return Image.fromarray(out, mode="L")


def create_app(img_dir: str, scribble_suffix: str, masks_subdir: str) -> Flask:
    app = Flask(__name__)
    app.config["IMG_DIR"] = img_dir
    app.config["SCRIBBLE_SUFFIX"] = scribble_suffix
    app.config["MASKS_SUBDIR"] = masks_subdir

    @app.get("/")
    def index():
        items = list_items(app.config["IMG_DIR"], app.config["SCRIBBLE_SUFFIX"], app.config["MASKS_SUBDIR"])
        if not items:
            return render_template("index.html", items=[], current=None)
        # default: first item
        return redirect(url_for("annotate", idx=0))

    @app.get("/annotate/<int:idx>")
    def annotate(idx: int):
        items = list_items(app.config["IMG_DIR"], app.config["SCRIBBLE_SUFFIX"], app.config["MASKS_SUBDIR"])
        if not items:
            return render_template("index.html", items=[], current=None)
        idx = max(0, min(idx, len(items) - 1))
        current = items[idx]
        w, h = load_image_size(os.path.join(app.config["IMG_DIR"], current.filename))
        return render_template(
            "index.html",
            items=items,
            current=current,
            idx=idx,
            w=w,
            h=h,
            scribble_suffix=app.config["SCRIBBLE_SUFFIX"],
            masks_subdir=app.config["MASKS_SUBDIR"],
        )

    @app.get("/img/<path:filename>")
    def img(filename: str):
        return send_from_directory(app.config["IMG_DIR"], filename)

    @app.get("/mask/<path:relpath>")
    def mask(relpath: str):
        # serve existing scribble mask if present; else 404 and client will start blank
        return send_from_directory(app.config["IMG_DIR"], relpath)

    @app.post("/save")
    def save():
        payload = request.get_json(force=True)
        image_filename = payload.get("image_filename")
        scribble_relpath = payload.get("scribble_relpath")
        scribble_filename = payload.get("scribble_filename")
        data_url = payload.get("mask_data_url")
        if not image_filename or not data_url or (not scribble_relpath and not scribble_filename):
            return jsonify({"ok": False, "error": "Missing fields"}), 400

        img_path = os.path.join(app.config["IMG_DIR"], image_filename)
        if not os.path.exists(img_path):
            return jsonify({"ok": False, "error": f"Image not found: {image_filename}"}), 404

        w, h = load_image_size(img_path)
        mask_rgba = decode_data_url_png(data_url)
        mask_rgba = ensure_rgba_canvas_size(mask_rgba, w, h)
        scrib = rgba_to_visible_scribble(mask_rgba)

        # Prefer relpath (e.g., masks/<file>), fall back to legacy filename (next to image dir).
        target = scribble_relpath or scribble_filename
        out_path = os.path.join(app.config["IMG_DIR"], target)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        scrib.save(out_path)
        return jsonify({"ok": True, "saved": out_path, "w": w, "h": h})

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory with images to annotate")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--scribble_suffix", default="_scribble", help="Suffix for saved masks (default: _scribble)")
    ap.add_argument("--masks_subdir", default="masks", help="Subdirectory inside --dir to store masks (default: masks)")
    args = ap.parse_args()

    img_dir = os.path.abspath(args.dir)
    if not os.path.isdir(img_dir):
        raise SystemExit(f"Not a directory: {img_dir}")

    app = create_app(img_dir, args.scribble_suffix, args.masks_subdir)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

