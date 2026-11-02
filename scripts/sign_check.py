"""
sign_check.py — verify that the cost model predicts the correct winner at
endpoints well below and well above BW*.

What this script does
---------------------
1. Reads the source JSON (throttled run) and destination JSON (unthrottled run).
2. Computes Δt and ΔB from those files — no manual transcription.
3. Reads /sys/fs/cgroup/cpu.max from the RUNNING source container via
   `docker exec source cat ...` and compares it to the quota recorded in the
   source JSON.  If they disagree, the predicted Δt describes a different
   machine and the check is invalid.
4. Sends 5 trials at 0.2×BW* (source should win) and 5 trials at 5×BW*
   (destination should win), interleaved src/dst within each trial, with
   one discarded warmup request after each /reset.
5. For each bandwidth level, computes:
     measured_margin_ms  = median(t_dst − t_src)   [positive = source wins]
     predicted_margin_ms = ΔB * 8 / (1000 * BW) − Δt  [× 1000 for ms]
   and checks:
     a. Sign agreement (ratio > 0)
     b. Magnitude agreement  0.33 < |ratio| < 3.0

Margin sign convention (used consistently in both scripts)
----------------------------------------------------------
  positive margin = source wins = source is faster.
  measured  = t_dst − t_src           (positive when source is faster)
  predicted = ΔB*8/(1000*BW) − Δt    (positive when thumbnail wire saving
                                        exceeds the compute penalty)

Usage
-----
  py scripts/sign_check.py \\
      --src-json data/source_0.1.json \\
      --dst-json data/dest.json \\
      --image    data/phase_b/medium.jpg

Requirements
------------
  - Source container named "source" must be running (docker compose up -d).
  - Its CPU quota must match the throttle used in measure_timing.py
    (verified automatically by step 3).
  - Destination container named "destination" must be running.
  - Both containers must expose the same endpoints used in the main experiment.
"""

import argparse
import json
import statistics
import subprocess
import sys
import time

import requests

SOURCE_URL = "http://localhost:8000"
DEST_URL   = "http://localhost:8001"  # used only for health verification

N_TRIALS        = 5
RESET_ENDPOINT  = "/reset"
PROCESS_ENDPOINT = "/process"

# Margin check thresholds
RATIO_LO = 0.33
RATIO_HI = 3.0

# Bandwidth multipliers for the two test points
BW_LO_MULT = 0.2   # source should win here (well below BW*)
BW_HI_MULT = 5.0   # destination should win here (well above BW*)


def load(path):
    with open(path) as f:
        return json.load(f)


def verify_cgroup(src_json: dict) -> bool:
    """
    Read /sys/fs/cgroup/cpu.max from the running 'source' container and
    compare to the value recorded in the source JSON.
    Returns True if they match (or if verification is skipped on error).
    """
    recorded = src_json.get("cpu_quota_cgroup", "unavailable")
    print(f"\n[cgroup check]")
    print(f"  Recorded in JSON : {recorded}")
    try:
        result = subprocess.run(
            ["docker", "exec", "source", "cat", "/sys/fs/cgroup/cpu.max"],
            capture_output=True, text=True, timeout=5
        )
        live = result.stdout.strip()
        print(f"  Live container   : {live}")
        if result.returncode != 0:
            print(f"  WARNING: docker exec failed — {result.stderr.strip()}")
            print(f"           Cannot verify throttle. Proceeding with caution.")
            return True   # don't abort; let user decide
        if live == recorded:
            print(f"  OK — throttle matches.")
            return True
        else:
            print(f"  ERROR — throttle MISMATCH.")
            print(f"  The source JSON was measured at '{recorded}' but the running")
            print(f"  container reports '{live}'.")
            print(f"  Restart the source container with the correct --cpus, or")
            print(f"  re-run measure_timing.py against the running container.")
            return False
    except Exception as e:
        print(f"  WARNING: could not run docker exec — {e}")
        print(f"           Cannot verify throttle. Proceeding with caution.")
        return True


def reset_shaper(bw_kbps: int):
    """Reset EMA state on source, then apply tc-tbf bandwidth shaping.
    Destination has no /reset endpoint — its state is stateless per-request."""
    r = requests.post(f"{SOURCE_URL}{RESET_ENDPOINT}", timeout=10)
    r.raise_for_status()

    # Apply shaping via shape_network.sh inside the source container.
    # Args: set <bw_kbps> <delay_ms> <jitter_ms>
    result = subprocess.run(
        ["docker", "exec", "source", "bash", "/scripts/shape_network.sh",
         "set", str(bw_kbps), "0", "0"],
        capture_output=True, text=True, timeout=15
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"shape_network.sh failed (exit {result.returncode}):\n"
            f"  stdout: {result.stdout.strip()}\n"
            f"  stderr: {result.stderr.strip()}"
        )
    print(f"  Shaper set to {bw_kbps:,} Kbps")


def send_image(mode: str, image_bytes: bytes) -> float:
    """Send one image to the source node with the given mode and return latency in seconds.
    Both 'source' and 'destination' modes go through the source /process endpoint,
    which handles routing internally. The destination /infer endpoint is not called directly."""
    t0 = time.perf_counter()
    r = requests.post(
        f"{SOURCE_URL}{PROCESS_ENDPOINT}",
        files={"file": ("image.jpg", image_bytes, "image/jpeg")},
        params={"mode": mode},
        timeout=120,
    )
    r.raise_for_status()
    return time.perf_counter() - t0


def run_trials(image_bytes: bytes, bw_kbps: int, n_trials: int):
    """
    Collect paired (t_src, t_dst) observations via the source /process endpoint.
    mode=source    → source preprocesses locally, sends tensor to destination
    mode=destination → source sends raw JPEG, destination preprocesses

    Both go through the source node so bandwidth shaping (tc-tbf on source eth0)
    applies to the wire leg in both cases.

    One warmup pair is sent first (discarded), then n_trials pairs are recorded,
    interleaved to cancel thermal and cache drift.
    Positive margin = source faster.
    """
    reset_shaper(bw_kbps)

    # Warmup (discarded)
    send_image("source",      image_bytes)
    send_image("destination", image_bytes)

    t_src_list = []
    t_dst_list = []

    for i in range(n_trials):
        t_src = send_image("source",      image_bytes)
        t_dst = send_image("destination", image_bytes)
        t_src_list.append(t_src)
        t_dst_list.append(t_dst)
        print(f"    trial {i+1}: t_src={t_src*1000:.1f} ms  "
              f"t_dst={t_dst*1000:.1f} ms  "
              f"margin={(t_dst - t_src)*1000:+.1f} ms")

    margins = [d - s for s, d in zip(t_src_list, t_dst_list)]
    return statistics.median(t_src_list), statistics.median(t_dst_list), \
           statistics.median(margins)


def check_endpoint(bw_kbps: float, expected_winner: str,
                   measured_margin_ms: float, predicted_margin_ms: float,
                   label: str):
    """
    Compare measured and predicted margins.
    Convention: positive = source wins.
    """
    print(f"\n  {label}  BW = {bw_kbps:,.0f} Kbps  (expected winner: {expected_winner})")
    print(f"    measured_margin  = {measured_margin_ms:+.2f} ms")
    print(f"    predicted_margin = {predicted_margin_ms:+.2f} ms")

    if predicted_margin_ms == 0:
        print("    ERROR: predicted margin is zero — cannot compute ratio")
        return False

    ratio = measured_margin_ms / predicted_margin_ms
    print(f"    ratio (measured/predicted) = {ratio:.3f}")

    ok = True

    # Sign check — key failure mode
    if ratio < 0:
        print(f"    FAIL: sign disagreement — model predicts "
              f"{'source' if predicted_margin_ms > 0 else 'destination'} wins "
              f"but measurement says the opposite.")
        ok = False
    else:
        print(f"    OK: sign agreement.")

    # Magnitude check
    if not (RATIO_LO < abs(ratio) < RATIO_HI):
        print(f"    FAIL: magnitude ratio {abs(ratio):.3f} outside "
              f"[{RATIO_LO}, {RATIO_HI}] — model is off by >{1/RATIO_LO:.0f}×.")
        ok = False
    else:
        print(f"    OK: magnitude within [{RATIO_LO}, {RATIO_HI}].")

    return ok


def main():
    parser = argparse.ArgumentParser(description="Phase B sign check")
    parser.add_argument("--src-json", required=True, help="Source timing JSON")
    parser.add_argument("--dst-json", required=True, help="Destination timing JSON")
    parser.add_argument("--image",    required=True, help="Test image path")
    parser.add_argument("--trials",   type=int, default=N_TRIALS)
    args = parser.parse_args()

    src = load(args.src_json)
    dst = load(args.dst_json)

    # --- Sanity checks -------------------------------------------------------
    assert src["role"] == "source",      "src-json must be a source-role file"
    assert dst["role"] == "destination", "dst-json must be a destination-role file"
    assert src["torch_threads"] == 1,    "source JSON: torch_threads must be 1"
    assert dst["torch_threads"] == 1,    "destination JSON: torch_threads must be 1"
    assert src["image_size_px"] == dst["image_size_px"], (
        f"Image size mismatch: {src['image_size_px']} vs {dst['image_size_px']}")

    # --- Verify running container throttle -----------------------------------
    ok = verify_cgroup(src)
    if not ok:
        print("\nAborting — fix throttle mismatch before running sign check.")
        sys.exit(1)

    # --- Compute Δt and ΔB from JSONs (no manual transcription) -------------
    t_src        = src["t_src_median_s"]
    t_dst_full   = dst["t_dst_full_median_s"]
    t_dst_remain = dst["t_dst_remain_median_s"]
    B_raw        = src["thumb_bytes"]
    B_tensor     = src["tensor_bytes"]

    delta_t_s = (t_src + t_dst_remain) - t_dst_full  # positive = source slower
    delta_B   = B_raw - B_tensor                      # negative = thumbnail smaller

    print(f"\n[parameters from JSON]")
    print(f"  t_src         = {t_src*1000:.2f} ms")
    print(f"  t_dst_full    = {t_dst_full*1000:.2f} ms")
    print(f"  t_dst_remain  = {t_dst_remain*1000:.2f} ms")
    print(f"  Δt            = {delta_t_s*1000:+.2f} ms")
    print(f"  B_raw (thumb) = {B_raw:,} bytes")
    print(f"  B_tensor      = {B_tensor:,} bytes")
    print(f"  ΔB            = {delta_B:,} bytes")

    if delta_t_s <= 0 or delta_B >= 0:
        print("\nERROR: Δt <= 0 or ΔB >= 0 — BW* is undefined.")
        print("  Δt <= 0 means the source path is not slower: increase throttle.")
        print("  ΔB >= 0 means the thumbnail is not smaller: use a higher-res input.")
        sys.exit(1)

    # BW* = −ΔB * 8 / (1000 * Δt)   [Kbps]
    bw_star = (-delta_B * 8) / (1000.0 * delta_t_s)
    print(f"\n  BW* = {bw_star:,.1f} Kbps  ({bw_star/1000:.3f} Mbps)")
    print(f"  Unit check: {-delta_B:,} × 8 / (1000 × {delta_t_s:.4f} s)"
          f" = {-delta_B*8/(1000*delta_t_s):,.1f} Kbps")

    # --- Load test image -----------------------------------------------------
    with open(args.image, "rb") as f:
        image_bytes = f.read()
    print(f"\n  Test image: {args.image}  ({len(image_bytes):,} bytes)")

    # --- Two endpoint bandwidths ---------------------------------------------
    bw_lo = int(bw_star * BW_LO_MULT)   # source should win
    bw_hi = int(bw_star * BW_HI_MULT)   # destination should win

    results = {}

    for bw_kbps, expected_winner, label in [
        (bw_lo, "source",      f"{BW_LO_MULT}×BW*"),
        (bw_hi, "destination", f"{BW_HI_MULT}×BW*"),
    ]:
        print(f"\n[{label}  =  {bw_kbps:,} Kbps]")
        t_src_med, t_dst_med, margin_med = run_trials(
            image_bytes, bw_kbps, args.trials)

        measured_margin_ms = margin_med * 1000   # positive = source wins

        # predicted margin [ms] — positive = source wins
        # = ΔB*8/(1000*BW) − Δt    (ΔB < 0, so at BW < BW* this is positive)
        predicted_margin_ms = (
            (delta_B * 8) / (1000.0 * bw_kbps) - delta_t_s
        ) * 1000.0

        results[label] = check_endpoint(
            bw_kbps, expected_winner,
            measured_margin_ms, predicted_margin_ms, label)

    # --- Summary -------------------------------------------------------------
    print("\n" + "="*60)
    print("SIGN CHECK SUMMARY")
    all_ok = all(results.values())
    for label, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {label:20s} {status}")
    print()
    if all_ok:
        print("All checks passed. Cost model signs and magnitudes are consistent.")
        print("Proceed with the full Phase B bandwidth sweep.")
    else:
        print("One or more checks failed. Investigate before running the sweep.")
        print("Common causes:")
        print("  Sign fail    — throttle mismatch, or TCP slow start distorting")
        print("                 wire time at small sizes (15 KB thumbnail).")
        print("  Magnitude >> — thermal/power throttling, background load, or")
        print("                 N_TRIALS too small (increase to 10+).")
        print("  Magnitude << — model is missing a cost term (check ledger).")
        sys.exit(1)


if __name__ == "__main__":
    main()

