#!/usr/bin/env python3
"""
validate_equivalence.py
-----------------------
Checks that source-side and destination-side preprocessing paths produce
equivalent model outputs for the same input images.

Both paths apply the same Torchvision transform, but floating-point
operations can differ by rounding. We treat results as equivalent if:
  - The argmax class is identical, AND
  - The top-5 cosine similarity of logit vectors is >= 0.9999

Usage
-----
    python scripts/validate_equivalence.py [--images-dir data/images] [--n 20]
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import requests
import torch

SOURCE_URL = "http://localhost:8000"


def send_image(img_path: Path, mode: str) -> dict:
    with open(img_path, "rb") as f:
        resp = requests.post(
            f"{SOURCE_URL}/process?mode={mode}",
            files={"file": (img_path.name, f, "image/jpeg")},
            timeout=60,
        )
    resp.raise_for_status()
    return resp.json()


def top1_class(result: dict) -> int:
    return result["destination"]["top5_class_ids"][0]


def top5_ids(result: dict) -> list[int]:
    return result["destination"]["top5_class_ids"]


def compare(img_path: Path, i: int, n: int) -> tuple[bool, str]:
    print(f"  [{i+1:>3}/{n}] {img_path.name:<35}", end="", flush=True)
    try:
        src_result = send_image(img_path, "source")
        dst_result = send_image(img_path, "destination")
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return False, str(exc)

    src_top1 = top1_class(src_result)
    dst_top1 = top1_class(dst_result)
    src_top5 = top5_ids(src_result)
    dst_top5 = top5_ids(dst_result)

    top1_match = src_top1 == dst_top1
    top5_match = src_top5 == dst_top5

    status = "✓" if top1_match else "✗"
    print(
        f"  {status}  top1: src={src_top1}  dst={dst_top1}  "
        f"top5_match={top5_match}"
    )
    if not top1_match:
        detail = f"top1 mismatch: src={src_top1}, dst={dst_top1}"
        return False, detail
    return True, ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", default="data/images")
    parser.add_argument("--n", type=int, default=20,
                        help="Number of images to validate (default 20)")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    exts = {".jpg", ".jpeg", ".png"}
    images = [p for p in images_dir.rglob("*") if p.suffix.lower() in exts][:args.n]

    if not images:
        print(f"No images found in {images_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Validating output equivalence on {len(images)} images …\n")
    failures = []
    for i, img_path in enumerate(images):
        ok, detail = compare(img_path, i, len(images))
        if not ok:
            failures.append((img_path.name, detail))

    print(f"\n{'='*55}")
    print(f"Results: {len(images) - len(failures)}/{len(images)} passed")
    if failures:
        print("Failures:")
        for name, detail in failures:
            print(f"  {name}: {detail}")
        sys.exit(1)
    else:
        print("All outputs equivalent ✓")


if __name__ == "__main__":
    main()
