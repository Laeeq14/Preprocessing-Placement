#!/usr/bin/env python3
"""
download_coco_sample.py
-----------------------
Downloads a small sample of COCO val2017 images for use as test data.

Usage
-----
    python scripts/download_coco_sample.py [--n 100] [--out-dir data/images]

Strategy
--------
1. Fetch the real val2017 image list from the COCO annotations JSON
   (cached locally after the first download).
2. Randomly sample --n filenames from the full 5,000-image val set.
3. Download each image from the COCO CDN.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ANNOTATIONS_ZIP_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
ANNOTATION_FILE_IN_ZIP = "annotations/instances_val2017.json"
IMAGE_BASE_URL = "http://images.cocodataset.org/val2017/"
CACHE_DIR = Path("data") / ".cache"


def fetch_bytes(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def get_val_filenames() -> list[str]:
    """Return all val2017 image filenames from the COCO annotation zip.
    Downloads and caches the zip on first run (~25 MB)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = CACHE_DIR / "annotations_trainval2017.zip"

    if not zip_path.exists():
        print("Fetching COCO annotation list (~25 MB, one-time download) …")
        try:
            data = fetch_bytes(ANNOTATIONS_ZIP_URL)
        except Exception as exc:
            print(f"ERROR: Could not fetch annotations: {exc}", file=sys.stderr)
            sys.exit(1)
        with open(zip_path, "wb") as f:
            f.write(data)
        print("  Cached to", zip_path)
    else:
        print(f"Using cached annotation zip: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(ANNOTATION_FILE_IN_ZIP) as af:
            ann = json.load(af)

    return [img["file_name"] for img in ann["images"]]


def download_file(url: str, dest: Path, label: str = "") -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        return True
    except Exception as exc:
        print(f"  WARN: Could not download {label}: {exc}", file=sys.stderr)
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100,
                        help="Number of images to download (default 100)")
    parser.add_argument("--out-dir", default="data/images",
                        help="Output directory (default data/images)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default 42)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_filenames = get_val_filenames()
    random.seed(args.seed)
    random.shuffle(all_filenames)
    to_download = all_filenames[: args.n]

    print(f"\nDownloading {len(to_download)} COCO val2017 images to {out_dir} …\n")

    success = 0
    for i, filename in enumerate(to_download):
        dest = out_dir / filename
        if dest.exists():
            print(f"  [{i+1:>3}] {filename}  (already exists)")
            success += 1
            continue

        url = IMAGE_BASE_URL + filename
        ok = download_file(url, dest, filename)
        status = "✓" if ok else "✗"
        print(f"  [{i+1:>3}] {filename}  {status}")
        if ok:
            success += 1
        time.sleep(0.05)  # polite rate limit

    print(f"\nDownloaded {success}/{len(to_download)} images to {out_dir}")
    if success == 0:
        print("ERROR: No images downloaded.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
