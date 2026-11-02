"""
phase_b_sweep.py — Phase B end-to-end bandwidth sweep bracketing BW*.

Runs three modes at each of six bandwidths:
  source_thumbnail : source encodes ~20 KB thumbnail, destination decodes + preprocesses
  destination      : source sends full raw JPEG (~1.4 MB), destination preprocesses
  adaptive         : existing EMA policy (uses Phase A paths — for reference only)

Bandwidth sweep (relative to BW* = 6,393 Kbps at --cpus=0.1):
  0.25×  1,598 Kbps  → source_thumbnail should win (thumbnail saves wire time)
  0.50×  3,197 Kbps  → source_thumbnail should win
  1.00×  6,393 Kbps  → crossover (expect near-tie)
  2.00× 12,786 Kbps  → destination should win
  4.00× 25,572 Kbps  → destination should win
  8.00× 51,144 Kbps  → destination should win

Usage
-----
  py scripts/phase_b_sweep.py --image data/phase_b/medium.jpg --n 10

Output
------
  results/phase_b_sweep.csv   — raw per-trial rows
  Console table               — mean latency by mode × bandwidth

Requirements
------------
  - Containers running: docker compose up -d  (source at :8000, destination at :8001)
  - Source container CPU quota must be 10000 100000 (--cpus=0.1 via docker-compose.yml)
  - shape_network.sh must exist at /scripts/shape_network.sh inside source container
"""

import argparse
import csv
import os
import statistics
import subprocess
import sys
import time

import requests

SOURCE_URL      = "http://localhost:8000"
PROCESS_EP      = "/process"
RESET_EP        = "/reset"
BW_STAR_KBPS    = 6_393   # from compute_bw_star.py, --cpus=0.1 throttle

BANDWIDTHS = [
    (int(BW_STAR_KBPS * 0.25), "0.25×BW*"),
    (int(BW_STAR_KBPS * 0.50), "0.50×BW*"),
    (int(BW_STAR_KBPS * 1.00), "1.00×BW*"),
    (int(BW_STAR_KBPS * 2.00), "2.00×BW*"),
    (int(BW_STAR_KBPS * 4.00), "4.00×BW*"),
    (int(BW_STAR_KBPS * 8.00), "8.00×BW*"),
]

MODES = ["source_thumbnail", "destination"]

N_WARMUP = 2   # discarded warmup requests per (bw, mode) combination


def set_bandwidth(bw_kbps: int):
    """Apply tc-tbf shaping via shape_network.sh inside the source container."""
    result = subprocess.run(
        ["docker", "exec", "source", "bash", "/scripts/shape_network.sh",
         "set", str(bw_kbps), "0", "0"],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"shape_network.sh failed:\n  {result.stderr.strip()}"
        )


def clear_bandwidth():
    """Remove all tc shaping."""
    subprocess.run(
        ["docker", "exec", "source", "bash", "/scripts/shape_network.sh", "clear"],
        capture_output=True, text=True, timeout=10,
    )


def reset_policy():
    """Reset source EMA state."""
    requests.post(f"{SOURCE_URL}{RESET_EP}", timeout=10).raise_for_status()


def send(mode: str, image_bytes: bytes) -> float:
    """Send one image through the source /process endpoint, return latency (s)."""
    t0 = time.perf_counter()
    r = requests.post(
        f"{SOURCE_URL}{PROCESS_EP}",
        files={"file": ("image.jpg", image_bytes, "image/jpeg")},
        params={"mode": mode},
        timeout=180,
    )
    r.raise_for_status()
    return time.perf_counter() - t0


def run_sweep(image_bytes: bytes, n_trials: int, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = []
    summary = {}   # (bw_kbps, mode) → list of latencies

    for bw_kbps, bw_label in BANDWIDTHS:
        print(f"\n{'='*60}")
        print(f"  Bandwidth: {bw_kbps:,} Kbps  ({bw_label})")
        set_bandwidth(bw_kbps)

        for mode in MODES:
            reset_policy()
            print(f"\n  Mode: {mode}")

            # Warmup
            for _ in range(N_WARMUP):
                send(mode, image_bytes)

            latencies = []
            for i in range(n_trials):
                t = send(mode, image_bytes)
                latencies.append(t)
                rows.append({
                    "bw_kbps":   bw_kbps,
                    "bw_label":  bw_label,
                    "mode":      mode,
                    "trial":     i + 1,
                    "latency_s": t,
                })
                print(f"    trial {i+1:2d}: {t*1000:7.1f} ms")

            med  = statistics.median(latencies) * 1000
            mean = statistics.mean(latencies) * 1000
            cv   = statistics.stdev(latencies) / statistics.mean(latencies)
            summary[(bw_kbps, mode)] = latencies
            print(f"    → median={med:.1f} ms  mean={mean:.1f} ms  CV={cv:.3f}")

    clear_bandwidth()

    # Write CSV
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {out_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"PHASE B SWEEP SUMMARY — Mean latency (ms)")
    print(f"BW*={BW_STAR_KBPS:,} Kbps  (--cpus=0.1 throttle)")
    print(f"{'BW (Kbps)':>12}  {'Label':>10}  {'src_thumbnail':>15}  {'destination':>12}  {'Winner':>12}")
    print("-" * 67)

    for bw_kbps, bw_label in BANDWIDTHS:
        t_src  = statistics.mean(summary[(bw_kbps, "source_thumbnail")]) * 1000
        t_dst  = statistics.mean(summary[(bw_kbps, "destination")]) * 1000
        winner = "src_thumbnail" if t_src < t_dst else "destination"
        print(f"{bw_kbps:>12,}  {bw_label:>10}  {t_src:>15.1f}  {t_dst:>12.1f}  {winner:>12}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Phase B bandwidth sweep")
    parser.add_argument("--image", required=True, help="Path to test JPEG (720p+)")
    parser.add_argument("--n",     type=int, default=10, help="Trials per (bw, mode)")
    parser.add_argument("--out",   default="results/phase_b_sweep.csv")
    args = parser.parse_args()

    print(f"Loading image: {args.image}")
    with open(args.image, "rb") as f:
        image_bytes = f.read()
    print(f"  {len(image_bytes):,} bytes")

    # Quick health check
    try:
        requests.get(f"{SOURCE_URL}/health", timeout=5).raise_for_status()
    except Exception as e:
        print(f"ERROR: Source container not reachable at {SOURCE_URL}: {e}")
        sys.exit(1)

    # Verify source CPU quota matches Phase B measurement
    result = subprocess.run(
        ["docker", "exec", "source", "cat", "/sys/fs/cgroup/cpu.max"],
        capture_output=True, text=True, timeout=5,
    )
    quota = result.stdout.strip()
    print(f"  Source cgroup cpu.max: {quota}")
    if quota != "10000 100000":
        print(f"  WARNING: Expected '10000 100000' (--cpus=0.1). Got '{quota}'.")
        print(f"           BW* was measured at --cpus=0.1. Results may not match predictions.")

    print(f"\nRunning {len(BANDWIDTHS)} bandwidths × {len(MODES)} modes × {args.n} trials")
    print(f"  Total requests: {len(BANDWIDTHS) * len(MODES) * (N_WARMUP + args.n)}"
          f"  ({N_WARMUP} warmup + {args.n} recorded per combination)\n")

    run_sweep(image_bytes, args.n, args.out)


if __name__ == "__main__":
    main()

