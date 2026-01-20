# src/main.py
"""
Command-line entrypoint for the Diffusion Engine.

This module provides a small, production-style CLI that:
- Loads the diffusion pipeline
- Runs CFG sampling
- Saves the resulting image to disk
"""

from __future__ import annotations

import argparse
import os
import warnings

# Suppress CUDA autocast warning on non-CUDA machines (must be set before torch imports)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"torch\.amp\.autocast_mode",
)

from src.config import DEVICE, ensure_dirs
from src.model import load_pipeline
from src.sampler import generate_image, SampleConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="diffusion-engine",
        description="Generate an image using a diffusion sampling engine (CFG).",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt describing the image to generate.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Optional negative prompt to discourage unwanted attributes.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps (higher = slower, potentially better quality).",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=8,
        help="Classifier-free guidance scale (higher = closer to prompt, can reduce diversity).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Output image height in pixels (typically 512 for SD v1).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Output image width in pixels (typically 512 for SD v1).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/generated.png",
        help="Output image path (PNG).",
    )

    return parser.parse_args()


def main() -> int:
    ensure_dirs()

    args = parse_args()

    print(f"[INFO] Device: {DEVICE.type}")
    print(f"[INFO] Prompt: {args.prompt!r}")
    if args.negative_prompt:
        print(f"[INFO] Negative prompt: {args.negative_prompt!r}")
    print(f"[INFO] Steps: {args.steps} | Guidance: {args.guidance} | Seed: {args.seed}")
    print(f"[INFO] Size: {args.width}x{args.height}")
    print("[INFO] Loading models...")

    pipe = load_pipeline()

    cfg = SampleConfig(
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        height=args.height,
        width=args.width,
    )

    print("[INFO] Sampling...")
    image = generate_image(
        pipe=pipe,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        cfg=cfg,
        device=DEVICE,
    )

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    image.save(out_path)

    print(f"[OK] Saved image -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())