"""
phase_b_transition.py — Test dynamic bandwidth transitions across BW*.

Validates adaptive switching behavior when link bandwidth changes dynamically
during a run without resetting the policy EMA.

Transition Sequence:
  Phase 1: 3,196 Kbps (0.21× BW*) — optimal: source_thumbnail
  Phase 2: 25,572 Kbps (1.67× BW*) — optimal: destination
  Phase 3: 3,196 Kbps (0.21× BW*) — optimal: source_thumbnail

Usage:
  py scripts/phase_b_transition.py --image data/phase_b/medium.jpg --n-per-step 10
"""

import argparse
import csv
import os
import subprocess
import sys
import time

import requests

SOURCE_URL = "http://localhost:8000"
PROCESS_EP = "/process"
RESET_EP   = "/reset"

STEPS = [
    (3196,  "Low BW (0.21× BW*) — Expect source_thumbnail"),
    (25572, "High BW (1.67× BW*) — Expect destination"),
    (3196,  "Low BW (0.21× BW*) — Expect return to source_thumbnail"),
]


def set_bandwidth(bw_kbps: int):
    result = subprocess.run(
        ["docker", "exec", "source", "bash", "/scripts/shape_network.sh",
         "set", str(bw_kbps), "0", "0"],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(f"shape_network.sh failed: {result.stderr.strip()}")


def clear_bandwidth():
    subprocess.run(
        ["docker", "exec", "source", "bash", "/scripts/shape_network.sh", "clear"],
        capture_output=True, text=True, timeout=10,
    )


def reset_policy():
    requests.post(f"{SOURCE_URL}{RESET_EP}", timeout=10).raise_for_status()


def send(image_bytes: bytes) -> tuple[float, dict]:
    t0 = time.perf_counter()
    r = requests.post(
        f"{SOURCE_URL}{PROCESS_EP}",
        files={"file": ("image.jpg", image_bytes, "image/jpeg")},
        params={"mode": "adaptive"},
        timeout=180,
    )
    r.raise_for_status()
    return time.perf_counter() - t0, r.json()


def run_transition(image_bytes: bytes, n_per_step: int, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    reset_policy()

    rows = []
    global_seq = 0
    switches_detected = 0
    last_decision = None

    print(f"\n{'='*75}")
    print(f"  PHASE B ADAPTIVE DYNAMIC TRANSITION EXPERIMENT")
    print(f"{'='*75}")

    for step_idx, (bw_kbps, step_label) in enumerate(STEPS, start=1):
        print(f"\n--- Step {step_idx}: Shaping to {bw_kbps:,} Kbps ({step_label}) ---")
        set_bandwidth(bw_kbps)

        for i in range(n_per_step):
            global_seq += 1
            latency, data = send(image_bytes)
            decision = data.get("decision", "unknown")
            pol = data.get("policy", {})
            bw_est = pol.get("bw_kbps", 0.0)

            switched = (last_decision is not None and decision != last_decision)
            if switched:
                switches_detected += 1
                switch_tag = f" *** SWITCH ({last_decision} -> {decision}) ***"
            else:
                switch_tag = ""
            last_decision = decision

            row = {
                "global_seq": global_seq,
                "step": step_idx,
                "target_bw_kbps": bw_kbps,
                "latency_s": latency,
                "decision": decision,
                "bw_est_kbps": bw_est,
                "wire_time_s": data.get("wire_time_s", 0.0),
                "bytes_transferred": data.get("bytes_transferred", 0),
                "src_cost_s": pol.get("src_cost_s", 0.0),
                "dst_cost_s": pol.get("dst_cost_s", 0.0),
                "switched": switched,
            }
            rows.append(row)

            print(
                f"  Req {global_seq:>2d} | Step {step_idx} [{i+1:>2d}/{n_per_step}] | "
                f"BW target={bw_kbps:>5d}k | est={bw_est:>5.0f}k | "
                f"decision={decision:<16s} | lat={latency*1000:>6.1f}ms{switch_tag}"
            )

    clear_bandwidth()

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*75}")
    print(f"Transition run completed. Total switches detected: {switches_detected}")
    print(f"Results written to: {out_path}")
    print(f"{'='*75}\n")


def main():
    parser = argparse.ArgumentParser(description="Phase B adaptive transition test")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--n-per-step", type=int, default=8, help="Requests per transition step")
    parser.add_argument("--out", default="results/phase_b_transition.csv", help="Output CSV")
    args = parser.parse_args()

    with open(args.image, "rb") as f:
        image_bytes = f.read()

    run_transition(image_bytes, args.n_per_step, args.out)


if __name__ == "__main__":
    main()

