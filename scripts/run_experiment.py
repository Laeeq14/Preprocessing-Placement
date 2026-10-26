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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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
    "wire_time_s",
    "total_time_s",
    "dst_preprocess_time_s",
    "dst_infer_time_s",
    "pred_cost_s",
    "bw_est_kbps",
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
        # Fatal: do not run experiments with broken shaping
        raise RuntimeError(
            f"[shape] FAILED to apply shaping: {result.stderr.strip()}\n"
            "Ensure the source container is running and ./scripts is mounted at /scripts."
        )
    print(f"[shape] {result.stdout.strip()}")


def validate_shaper(bw_kbps: int, no_shape: bool, probe_image: str) -> None:
    """Send 3 probe requests and verify wire_time_s is in the right ballpark.

    At a given BW, wire_time_s should be roughly: bytes / (bw_kbps * 125).
    We check it is at least 20% of the theoretical value.  If not, shaping
    is probably not working (or the image passed through the burst bucket
    unshaped).
    """
    if no_shape:
        return
    print(f"[validate] Checking shaper at {bw_kbps} Kbps with {probe_image} …", flush=True)
    wire_times = []
    img_path = Path(probe_image)
    if not img_path.exists():
        print("[validate] Probe image not found, skipping validation.")
        return
    for _ in range(3):
        try:
            with open(img_path, "rb") as f:
                resp = requests.post(
                    f"{SOURCE_URL}/process?mode=destination",
                    files={"file": (img_path.name, f, "image/jpeg")},
                    timeout=120,
                )
            if resp.ok:
                wt = resp.json().get("wire_time_s", 0.0)
                wire_times.append(wt)
        except Exception as exc:
            print(f"[validate] probe failed: {exc}")

    if not wire_times:
        print("[validate] WARNING: no probe responses, cannot validate shaper.")
        return

    # Expected wire time for the probe image
    img_bytes = img_path.stat().st_size
    expected_s = (img_bytes * 8) / (bw_kbps * 1000)
    median_wt  = sorted(wire_times)[len(wire_times) // 2]
    ratio = median_wt / expected_s if expected_s > 0 else 0

    print(f"[validate] wire_time median={median_wt*1000:.0f}ms  "
          f"expected={expected_s*1000:.0f}ms  ratio={ratio:.2f}")
    if ratio < 0.20:
        raise RuntimeError(
            f"[validate] Shaper does not appear to be working: "
            f"wire_time={median_wt*1000:.0f}ms but expected >={expected_s*1000*0.20:.0f}ms "
            f"at {bw_kbps} Kbps. Check that /scripts is mounted and tc shows a tbf qdisc."
        )
    print("[validate] Shaper OK [OK]")


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

def send_image(
    mode: str,
    bw_kbps: int,
    delay_ms: int,
    img_path: Path,
    seq: int,
    run_id: str,
) -> dict:
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
        "wire_time_s": "",
        "total_time_s": "",
        "dst_preprocess_time_s": "",
        "dst_infer_time_s": "",
        "pred_cost_s": "",
        "bw_est_kbps": "",
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
            pol = data.get("policy", {})
            chosen_decision = data.get("decision", mode)
            pred_cost = pol.get("src_cost_s", "") if chosen_decision == "source" else pol.get("dst_cost_s", "")
            row.update({
                "decision": data.get("decision", ""),
                "raw_size_bytes": data.get("raw_size_bytes", ""),
                "bytes_transferred": data.get("bytes_transferred", ""),
                "src_preprocess_time_s": data.get("src_preprocess_time_s", ""),
                "transfer_time_s": data.get("transfer_time_s", ""),
                "wire_time_s": data.get("wire_time_s", ""),
                "total_time_s": data.get("total_time_s", ""),
                "dst_preprocess_time_s": dst.get("preprocess_time_s", ""),
                "dst_infer_time_s": dst.get("infer_time_s", ""),
                "pred_cost_s": pred_cost,
                "bw_est_kbps": pol.get("bw_kbps", ""),
            })
            status = f"OK  total={data['total_time_s']:.3f}s  wire={data.get('wire_time_s', 0):.3f}s  pred={pred_cost:.3f}s  [{data['decision']}]" if isinstance(pred_cost, float) else f"OK  total={data['total_time_s']:.3f}s  wire={data.get('wire_time_s', 0):.3f}s  [{data['decision']}]"
        else:
            row["error"] = f"HTTP {resp.status_code}"
            status = f"ERR  HTTP {resp.status_code}"
    except Exception as exc:
        row["error"] = str(exc)
        status = f"ERR  {exc}"

    print(f"  [{mode:<11}] [{seq+1:>3}] {img_path.name:<25} {status}")
    return row


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
        row = send_image(mode, bw_kbps, delay_ms, img_path, seq, run_id)
        writer.writerow(row)
        rows.append(row)

    return rows


def bootstrap_ci(data: list[float], n_boot: int = 2000, ci: float = 0.95) -> tuple[float, float]:
    """Compute 95% bootstrap confidence interval for the sample median."""
    if len(data) < 2:
        return (data[0], data[0]) if data else (0.0, 0.0)
    arr = np.array(data)
    boot_samples = np.random.choice(arr, size=(n_boot, len(arr)), replace=True)
    boot_medians = np.median(boot_samples, axis=1)
    alpha = (1.0 - ci) / 2.0
    low = np.percentile(boot_medians, alpha * 100)
    high = np.percentile(boot_medians, (1.0 - alpha) * 100)
    return float(low), float(high)


def summarize(rows: list[dict], mode: str, bw_kbps: int) -> None:
    valid = [r for r in rows if not r["error"]]
    if not valid:
        print("  No valid rows to summarize.")
        return
    latencies = sorted(float(r["total_time_s"]) for r in valid)
    wire_times = sorted(float(r["wire_time_s"]) for r in valid if r.get("wire_time_s"))
    transferred = [int(r["bytes_transferred"]) for r in valid if r["bytes_transferred"]]
    decisions = [r["decision"] for r in valid]
    src_count = decisions.count("source")
    dst_count = decisions.count("destination")

    n = len(latencies)
    med_lat = latencies[n // 2]
    p95_lat = latencies[int(0.95 * (n - 1))]
    q25 = latencies[int(0.25 * (n - 1))]
    q75 = latencies[int(0.75 * (n - 1))]
    iqr_lat = q75 - q25
    ci_low, ci_high = bootstrap_ci(latencies)

    print(f"\n  Summary ({mode}, {bw_kbps}Kbps, N={n}):")
    print(f"    Latency ms  : median={med_lat*1000:.1f} (95% CI: [{ci_low*1000:.1f}, {ci_high*1000:.1f}])  "
          f"p95={p95_lat*1000:.1f}  iqr={iqr_lat*1000:.1f}  mean={sum(latencies)/n*1000:.1f}")
    if wire_times:
        wn = len(wire_times)
        med_wire = wire_times[wn // 2]
        p95_wire = wire_times[int(0.95 * (wn - 1))]
        ci_wlow, ci_whigh = bootstrap_ci(wire_times)
        print(f"    Wire ms     : median={med_wire*1000:.1f} (95% CI: [{ci_wlow*1000:.1f}, {ci_whigh*1000:.1f}])  "
              f"p95={p95_wire*1000:.1f}  mean={sum(wire_times)/wn*1000:.1f}")
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
    parser.add_argument("--interleave", action="store_true", default=False,
                        help="Interleave modes per image to eliminate CPU inference noise")
    args = parser.parse_args()

    random.seed(args.seed)

    images_dir = Path(args.images_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wait_for_services()

    images = collect_images(images_dir, args.n)
    print(f"\nUsing {len(images)} images from {images_dir}")

    modes = ["source", "destination", "adaptive"] if args.mode == "all" else [args.mode]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    file_exists = output_path.exists()

    apply_bandwidth(args.bw, args.delay, args.no_shape)
    validate_shaper(args.bw, args.no_shape, str(images[0]) if images else "")

    try:
        with open(output_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
            if not file_exists:
                writer.writeheader()

            if args.interleave and len(modes) > 1:
                print(f"\n{'='*60}")
                print(f"  INTERLEAVED RUN: modes={modes}  bw={args.bw}Kbps  n={args.n}")
                print(f"{'='*60}")
                reset_policy()
                mode_rows = {m: [] for m in modes}
                for seq, img_path in enumerate(images):
                    img_modes = list(modes)
                    random.shuffle(img_modes)
                    for m in img_modes:
                        row = send_image(m, args.bw, args.delay, img_path, seq, run_id)
                        writer.writerow(row)
                        csvfile.flush()
                        mode_rows[m].append(row)
                for m in modes:
                    summarize(mode_rows[m], m, args.bw)
            else:
                random.shuffle(modes)
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
