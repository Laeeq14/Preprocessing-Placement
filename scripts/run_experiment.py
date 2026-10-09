#!/usr/bin/env python3
"""
run_experiment.py
-----------------
Orchestrates preprocessing placement experiments.

Usage
-----
    python scripts/run_experiment.py [OPTIONS]

Options
-------
  --mode        adaptive | source | destination | all   (default: all)
  --bw          Bandwidth in Kbps to apply  (default: 1000)
  --delay       Added latency in ms         (default: 0)
  --n           Images per run              (default: 50)
  --images-dir  Directory of test images    (default: data/images)
  --output      CSV output file             (default: results/results.csv)
  --no-shape    Skip network shaping (useful for local testing without NET_ADMIN)

The script:
  1. (Optionally) applies bandwidth shaping via docker exec into source
  2. Resets the policy state at source
  3. Sends N images to source /process?mode=<mode>
  4. Records per-image metrics to CSV
  5. Appends a summary row
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

SOURCE_URL = os.getenv("SOURCE_URL", "http://localhost:8000")
SOURCE_CONTAINER = os.getenv("SOURCE_CONTAINER", "source")

FIELDNAMES = [
    "run_id",
    "mode",
    "bw_kbps",
    "delay_ms",
    "image_name",
    "image_seq",
    "decision",
    "raw_size_bytes",
    "bytes_transferred",
    "src_preprocess_time_s",
    "transfer_time_s",
    "total_time_s",
    "dst_preprocess_time_s",
    "dst_infer_time_s",
    "error",
]


def wait_for_services(timeout: int = 60) -> None:
    print("Waiting for services to be healthy …", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{SOURCE_URL}/health", timeout=3)
            if r.ok:
                print(" OK")
                return
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(2)
    print()
    raise RuntimeError("Services did not become healthy within timeout.")


def apply_bandwidth(bw_kbps: int, delay_ms: int, no_shape: bool) -> None:
    if no_shape:
        print(f"[shape] Skipping (--no-shape). BW={bw_kbps}Kbps delay={delay_ms}ms")
        return
    cmd = [
        "docker", "exec", SOURCE_CONTAINER,
        "bash", "/scripts/shape_network.sh",
        "set", str(bw_kbps), str(delay_ms), "0",
    ]
    print(f"[shape] Applying {bw_kbps} Kbps, {delay_ms} ms …")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[shape] WARNING: {result.stderr.strip()}", file=sys.stderr)
    else:
        print(f"[shape] {result.stdout.strip()}")


def clear_bandwidth(no_shape: bool) -> None:
    if no_shape:
        return
    subprocess.run(
        ["docker", "exec", SOURCE_CONTAINER,
         "bash", "/scripts/shape_network.sh", "clear"],
        capture_output=True,
    )


def reset_policy() -> None:
    try:
        requests.post(f"{SOURCE_URL}/reset", timeout=5)
        print("[policy] Reset.")
    except Exception as e:
        print(f"[policy] Reset failed: {e}", file=sys.stderr)


def collect_images(images_dir: Path, n: int) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_images = [p for p in images_dir.rglob("*") if p.suffix.lower() in exts]
    if not all_images:
        raise FileNotFoundError(f"No images found in {images_dir}")
    random.shuffle(all_images)
    # cycle if fewer images than n
    images = []
    while len(images) < n:
        images.extend(all_images)
    return images[:n]


def run_single(
    mode: str,
    bw_kbps: int,
    delay_ms: int,
    n: int,
    images: list[Path],
    writer: csv.DictWriter,
    run_id: str,
) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"  RUN: mode={mode}  bw={bw_kbps}Kbps  delay={delay_ms}ms  n={n}")
    print(f"{'='*60}")

    reset_policy()
    rows = []

    for seq, img_path in enumerate(images):
        row = {
            "run_id": run_id,
            "mode": mode,
            "bw_kbps": bw_kbps,
            "delay_ms": delay_ms,
            "image_name": img_path.name,
            "image_seq": seq,
            "decision": "",
            "raw_size_bytes": "",
            "bytes_transferred": "",
            "src_preprocess_time_s": "",
            "transfer_time_s": "",
            "total_time_s": "",
            "dst_preprocess_time_s": "",
            "dst_infer_time_s": "",
            "error": "",
        }
        try:
            with open(img_path, "rb") as f:
                resp = requests.post(
                    f"{SOURCE_URL}/process?mode={mode}",
                    files={"file": (img_path.name, f, "image/jpeg")},
                    timeout=120,
                )
            if resp.status_code == 200:
                data = resp.json()
                dst = data.get("destination", {})
                row.update({
                    "decision": data.get("decision", ""),
                    "raw_size_bytes": data.get("raw_size_bytes", ""),
                    "bytes_transferred": data.get("bytes_transferred", ""),
                    "src_preprocess_time_s": data.get("src_preprocess_time_s", ""),
                    "transfer_time_s": data.get("transfer_time_s", ""),
                    "total_time_s": data.get("total_time_s", ""),
                    "dst_preprocess_time_s": dst.get("preprocess_time_s", ""),
                    "dst_infer_time_s": dst.get("infer_time_s", ""),
                })
                status = f"✓  {data['total_time_s']:.3f}s  [{data['decision']}]"
            else:
                row["error"] = f"HTTP {resp.status_code}"
                status = f"✗  HTTP {resp.status_code}"
        except Exception as exc:
            row["error"] = str(exc)
            status = f"✗  {exc}"

        print(f"  [{seq+1:>3}/{n}] {img_path.name:<30} {status}")
        writer.writerow(row)
        rows.append(row)

    return rows


def summarize(rows: list[dict], mode: str, bw_kbps: int) -> None:
    valid = [r for r in rows if not r["error"]]
    if not valid:
        print("  No valid rows to summarize.")
        return
    latencies = [float(r["total_time_s"]) for r in valid]
    transferred = [int(r["bytes_transferred"]) for r in valid if r["bytes_transferred"]]
    decisions = [r["decision"] for r in valid]
    src_count = decisions.count("source")
    dst_count = decisions.count("destination")

    print(f"\n  Summary ({mode}, {bw_kbps}Kbps):")
    print(f"    Completed   : {len(valid)}/{len(rows)}")
    print(f"    Latency     : mean={sum(latencies)/len(latencies)*1000:.1f}ms  "
          f"min={min(latencies)*1000:.1f}ms  max={max(latencies)*1000:.1f}ms")
    if transferred:
        print(f"    Transferred : total={sum(transferred)/1024:.1f}KB  "
              f"mean={sum(transferred)/len(transferred)/1024:.1f}KB/item")
    print(f"    Decisions   : source={src_count}  destination={dst_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocessing placement experiment runner")
    parser.add_argument("--mode", default="all",
                        help="adaptive | source | destination | all")
    parser.add_argument("--bw", type=int, default=1000,
                        help="Bandwidth in Kbps (default 1000)")
    parser.add_argument("--delay", type=int, default=0,
                        help="Added latency in ms (default 0)")
    parser.add_argument("--n", type=int, default=50,
                        help="Images per run (default 50)")
    parser.add_argument("--images-dir", default="data/images",
                        help="Directory of test images")
    parser.add_argument("--output", default="results/results.csv",
                        help="Output CSV file")
    parser.add_argument("--no-shape", action="store_true",
                        help="Skip tc network shaping")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for image shuffling")
    args = parser.parse_args()

    random.seed(args.seed)

    images_dir = Path(args.images_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wait_for_services()

    images = collect_images(images_dir, args.n)
    print(f"\nUsing {len(images)} images from {images_dir}")

    modes = ["source", "destination", "adaptive"] if args.mode == "all" else [args.mode]
    # Randomize mode order to reduce order bias
    random.shuffle(modes)

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    file_exists = output_path.exists()

    apply_bandwidth(args.bw, args.delay, args.no_shape)

    try:
        with open(output_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
            if not file_exists:
                writer.writeheader()

            for mode in modes:
                rows = run_single(
                    mode=mode,
                    bw_kbps=args.bw,
                    delay_ms=args.delay,
                    n=args.n,
                    images=images,
                    writer=writer,
                    run_id=run_id,
                )
                summarize(rows, mode, args.bw)
    finally:
        clear_bandwidth(args.no_shape)

    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
