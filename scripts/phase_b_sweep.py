"""
phase_b_sweep.py — Phase B end-to-end bandwidth sweep bracketing BW*.

Runs three modes at each of six bandwidths:
  source_thumbnail : source encodes ~20 KB thumbnail, destination decodes + preprocesses
  destination      : source sends full raw JPEG (~1.4 MB), destination preprocesses
  adaptive         : dynamic placement policy deciding between thumbnail and destination

Bandwidth sweep (relative to BW* = 15,271 Kbps at --cpus=0.1):
  0.10×  1,598 Kbps  → source_thumbnail wins (thumbnail saves wire time)
  0.21×  3,196 Kbps  → source_thumbnail wins
  0.42×  6,393 Kbps  → source_thumbnail wins
  0.84× 12,786 Kbps  → source_thumbnail wins
  1.67× 25,572 Kbps  → destination wins
  3.35× 51,144 Kbps  → destination wins

Usage
-----
  py scripts/phase_b_sweep.py --image data/phase_b/medium.jpg --n 10

Output
------
  results/phase_b_sweep.csv   — raw per-trial rows
  Console table               — mean latency by mode × bandwidth
"""

import argparse
import csv
import os
import statistics
import subprocess
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import requests

SOURCE_URL      = "http://localhost:8000"
PROCESS_EP      = "/process"
RESET_EP        = "/reset"
BW_STAR_KBPS    = 15_271   # from compute_bw_star.py (1.41 MB raw vs 20.5 KB thumb at --cpus=0.1)

BANDWIDTHS = [
    (1598,  "0.10xBW*"),
    (3196,  "0.21xBW*"),
    (6393,  "0.42xBW*"),
    (12786, "0.84xBW*"),
    (25572, "1.67xBW*"),
    (51144, "3.35xBW*"),
]

DEFAULT_MODES = ["source_thumbnail", "destination", "adaptive"]

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


def send(mode: str, image_bytes: bytes) -> tuple[float, dict]:
    """Send one image through the source /process endpoint, return (latency_s, resp_json)."""
    t0 = time.perf_counter()
    r = requests.post(
        f"{SOURCE_URL}{PROCESS_EP}",
        files={"file": ("image.jpg", image_bytes, "image/jpeg")},
        params={"mode": mode},
        timeout=180,
    )
    r.raise_for_status()
    latency = time.perf_counter() - t0
    return latency, r.json()


def run_sweep(image_bytes: bytes, n_trials: int, out_path: str, modes: list[str]):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = []
    summary_lat = {}       # (bw_kbps, mode) → list of latencies
    summary_decisions = {} # (bw_kbps, mode) → list of decision strings

    for bw_kbps, bw_label in BANDWIDTHS:
        print(f"\n{'='*70}")
        print(f"  Bandwidth: {bw_kbps:,} Kbps  ({bw_label})")
        print(f"{'='*70}")
        set_bandwidth(bw_kbps)

        for mode in modes:
            reset_policy()
            print(f"\n  Mode: {mode}")

            # Warmup
            for _ in range(N_WARMUP):
                send(mode, image_bytes)

            latencies = []
            decisions = []
            for i in range(n_trials):
                t, data = send(mode, image_bytes)
                latencies.append(t)
                decision = data.get("decision", mode)
                decisions.append(decision)

                pol = data.get("policy", {})
                rows.append({
                    "bw_kbps":         bw_kbps,
                    "bw_label":        bw_label,
                    "mode_requested":  mode,
                    "decision":        decision,
                    "trial":           i + 1,
                    "latency_s":       t,
                    "wire_time_s":     data.get("wire_time_s", 0.0),
                    "bytes_sent":      data.get("bytes_transferred", 0),
                    "src_prep_s":      data.get("src_preprocess_time_s", 0.0),
                    "pred_decision":   pol.get("pred_decision", ""),
                    "bw_est_kbps":     pol.get("bw_kbps", 0.0),
                    "src_cost_s":      pol.get("src_cost_s", 0.0),
                    "dst_cost_s":      pol.get("dst_cost_s", 0.0),
                })
                dec_tag = f" [{decision}]" if mode == "adaptive" else ""
                print(f"    trial {i+1:2d}: {t*1000:7.1f} ms{dec_tag}")

            med  = statistics.median(latencies) * 1000
            mean = statistics.mean(latencies) * 1000
            cv   = statistics.stdev(latencies) / statistics.mean(latencies) if len(latencies) > 1 else 0.0
            summary_lat[(bw_kbps, mode)] = latencies
            summary_decisions[(bw_kbps, mode)] = decisions
            print(f"    -> median={med:.1f} ms  mean={mean:.1f} ms  CV={cv:.3f}")

    clear_bandwidth()

    # Write CSV
    if rows:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults written to {out_path}")

    # Print summary tables (both Median and Mean)
    print(f"\n{'='*82}")
    print(f"PHASE B SWEEP SUMMARY — Median latency (ms) [Robust to CFS jitter]")
    print(f"BW* = {BW_STAR_KBPS:,} Kbps (--cpus=0.1 throttle)")
    print(f"{'BW (Kbps)':>10}  {'Label':>9}  {'src_thumb':>12}  {'destination':>12}  {'adaptive':>12}  {'Winner':>14}  {'Adapt Pick':>12}")
    print("-" * 82)

    for bw_kbps, bw_label in BANDWIDTHS:
        m_src  = statistics.median(summary_lat[(bw_kbps, "source_thumbnail")]) * 1000 if (bw_kbps, "source_thumbnail") in summary_lat else float("nan")
        m_dst  = statistics.median(summary_lat[(bw_kbps, "destination")]) * 1000 if (bw_kbps, "destination") in summary_lat else float("nan")
        m_adp  = statistics.median(summary_lat[(bw_kbps, "adaptive")]) * 1000 if (bw_kbps, "adaptive") in summary_lat else float("nan")

        winner = "src_thumbnail" if m_src <= m_dst else "destination"

        adapt_picks = summary_decisions.get((bw_kbps, "adaptive"), [])
        if adapt_picks:
            thumb_cnt = sum(1 for d in adapt_picks if d == "source_thumbnail")
            pick_str = f"{thumb_cnt}/{len(adapt_picks)} thumb" if thumb_cnt not in (0, len(adapt_picks)) else ("thumbnail" if thumb_cnt == len(adapt_picks) else "destination")
        else:
            pick_str = "N/A"

        print(f"{bw_kbps:>10,}  {bw_label:>9}  {m_src:>12.1f}  {m_dst:>12.1f}  {m_adp:>12.1f}  {winner:>14}  {pick_str:>12}")

    print(f"\n{'='*82}")
    print(f"PHASE B SWEEP SUMMARY — Mean latency (ms)")
    print(f"BW* = {BW_STAR_KBPS:,} Kbps (--cpus=0.1 throttle)")
    print(f"{'BW (Kbps)':>10}  {'Label':>9}  {'src_thumb':>12}  {'destination':>12}  {'adaptive':>12}  {'Winner':>14}  {'Adapt Pick':>12}")
    print("-" * 82)

    for bw_kbps, bw_label in BANDWIDTHS:
        t_src  = statistics.mean(summary_lat[(bw_kbps, "source_thumbnail")]) * 1000 if (bw_kbps, "source_thumbnail") in summary_lat else float("nan")
        t_dst  = statistics.mean(summary_lat[(bw_kbps, "destination")]) * 1000 if (bw_kbps, "destination") in summary_lat else float("nan")
        t_adp  = statistics.mean(summary_lat[(bw_kbps, "adaptive")]) * 1000 if (bw_kbps, "adaptive") in summary_lat else float("nan")

        winner = "src_thumbnail" if t_src <= t_dst else "destination"

        adapt_picks = summary_decisions.get((bw_kbps, "adaptive"), [])
        if adapt_picks:
            thumb_cnt = sum(1 for d in adapt_picks if d == "source_thumbnail")
            pick_str = f"{thumb_cnt}/{len(adapt_picks)} thumb" if thumb_cnt not in (0, len(adapt_picks)) else ("thumbnail" if thumb_cnt == len(adapt_picks) else "destination")
        else:
            pick_str = "N/A"

        print(f"{bw_kbps:>10,}  {bw_label:>9}  {t_src:>12.1f}  {t_dst:>12.1f}  {t_adp:>12.1f}  {winner:>14}  {pick_str:>12}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Phase B bandwidth sweep with adaptive mode")
    parser.add_argument("--image", required=True, help="Path to test JPEG (720p+)")
    parser.add_argument("--n",     type=int, default=10, help="Trials per (bw, mode)")
    parser.add_argument("--out",   default="results/phase_b_sweep.csv")
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES, help="Modes to run")
    args = parser.parse_args()

    print(f"Loading image: {args.image}")
    with open(args.image, "rb") as f:
        image_bytes = f.read()
    print(f"  {len(image_bytes):,} bytes")

    # Quick health check
    try:
        r = requests.get(f"{SOURCE_URL}/health", timeout=5)
        r.raise_for_status()
        print(f"  Source service ready: {r.json()}")
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

    print(f"\nRunning {len(BANDWIDTHS)} bandwidths × {len(args.modes)} modes × {args.n} trials")
    print(f"  Total requests: {len(BANDWIDTHS) * len(args.modes) * (N_WARMUP + args.n)}"
          f"  ({N_WARMUP} warmup + {args.n} recorded per combination)\n")

    run_sweep(image_bytes, args.n, args.out, args.modes)


if __name__ == "__main__":
    main()
