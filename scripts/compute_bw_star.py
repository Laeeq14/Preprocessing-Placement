"""
compute_bw_star.py — offline BW* computation from per-role JSON files.

Usage
-----
  py scripts/compute_bw_star.py \\
      data/source_0.1.json \\
      data/source_0.2.json \\
      data/source_0.5.json \\
      data/dest.json

For each source JSON file, BW* is computed against the single destination
JSON.  All timing inputs are medians (robust to CFS stall outliers).

BW* derivation
--------------
  Convention: positive margin = source faster = source wins.

  Source-path total cost (throttled source + unthrottled dest remainder):
    t_src_total = t_src  +  t_dst_remain

  Destination-path cost (unthrottled):
    t_dst_full

  Dt = t_src_total − t_dst_full      [positive = source is slower overall]

  Wire-byte difference:
    B_raw    = thumb_bytes            [thumbnail sent over the wire]
    B_tensor = 3 * 224 * 224 * 4     [float32 tensor sent if dest preprocesses]
    DB       = B_raw − B_tensor      [negative = thumbnail is smaller]

  At bandwidth BW [Kbps], the wire-time difference is:
    Dwire = DB * 8 / (1000 * BW)    [seconds; negative = thumbnail saves time]

  Source wins when its total latency is lower:
    t_src_total + B_raw * 8 / (1000 * BW)  <  t_dst_full + B_tensor * 8 / (1000 * BW)
    Dwire < −Dt                             (source saves enough wire time)

  Rearranging (DB < 0, so BW* is positive):
    BW* = DB * 8 / (1000 * (−Dt))
        = −DB * 8 / (1000 * Dt)         [Kbps]

  Unit chain (explicit):
    DB  [bytes] * 8  [bits/byte]
    ─────────────────────────────── = bits/s → / 1000 → Kbps
    1000 [bits/Kbps] * Dt [s]

Sweep note
----------
  The sweep below runs at 1/4×, 1/2×, 1×, 2×, 4×, and 8× BW* (six points).
  That is two octaves below and three octaves above BW*.
  Adjust in sign_check.py / run_experiment.py to match.

CV warning thresholds
---------------------
  CV > 0.10 : notice     — results may be noisy
  CV > 0.20 : warning    — re-run recommended
"""

import json
import sys


CV_NOTICE  = 0.10
CV_WARN    = 0.20


def load(path):
    with open(path) as f:
        return json.load(f)


def check_cv(label, val, threshold_notice=CV_NOTICE, threshold_warn=CV_WARN):
    if val > threshold_warn:
        print(f"  !! CV WARNING  {label}: CV={val:.3f} > {threshold_warn:.2f} — re-run")
    elif val > threshold_notice:
        print(f"  !  CV notice   {label}: CV={val:.3f} > {threshold_notice:.2f}")


def compute_bw_star(src: dict, dst: dict) -> dict:
    """
    Compute BW* from one source JSON and one destination JSON.
    Returns a dict of key results.
    """
    # --- Sanity checks -------------------------------------------------------
    assert src["role"] == "source",      f"Expected role=source, got {src['role']}"
    assert dst["role"] == "destination", f"Expected role=destination, got {dst['role']}"
    assert src["torch_threads"] == 1,    "torch_threads must be 1 in source run"
    assert dst["torch_threads"] == 1,    "torch_threads must be 1 in destination run"
    assert src["image_size_px"] == dst["image_size_px"], (
        f"Image size mismatch: source={src['image_size_px']} dst={dst['image_size_px']}")
    # tensor_bytes field kept in JSON for reference; BW* now uses image_bytes as B_dst_wire

    # --- CV warnings ---------------------------------------------------------
    check_cv("t_src",        src["t_src_cv"])
    check_cv("t_dst_full",   dst["t_dst_full_cv"])
    check_cv("t_dst_remain", dst["t_dst_remain_cv"])

    # --- Timing (medians) ----------------------------------------------------
    t_src        = src["t_src_median_s"]
    t_dst_full   = dst["t_dst_full_median_s"]
    t_dst_remain = dst["t_dst_remain_median_s"]

    # --- Bytes ---------------------------------------------------------------
    B_raw    = src["thumb_bytes"]    # thumbnail  — source_thumbnail path wire size
    B_tensor = src["image_bytes"]    # full raw JPEG — destination path wire size
    # NOTE: "B_tensor" name kept for compat; in Phase B this is the raw JPEG, not the tensor.
    # An earlier version used src["tensor_bytes"] (602 KB) here, which is the Phase A
    # float32 tensor size and underestimates the destination-path wire payload by ~2.3x.

    # --- BW* -----------------------------------------------------------------
    delta_t  = (t_src + t_dst_remain) - t_dst_full   # seconds; + = source slower
    delta_B  = B_raw - B_tensor                       # bytes;   − = thumbnail smaller

    if delta_t <= 0:
        bw_star = None
        note = ("Dt <= 0: source path is not slower than destination path. "
                "Throttle is not strong enough, or transforms are asymmetric "
                "in the wrong direction. No crossover exists at this throttle.")
    elif delta_B >= 0:
        bw_star = None
        note = ("DB >= 0: thumbnail is NOT smaller than the tensor. "
                "Source path always loses on both compute and wire. "
                "Use a higher-res input or lower JPEG quality.")
    else:
        # BW* [Kbps] = −DB * 8 / (1000 * Dt)
        # Unit chain: bytes * 8 bits/byte / (1000 bits/Kbps * seconds) = Kbps
        bw_star = (-delta_B * 8) / (1000.0 * delta_t)
        note = "OK"

    return {
        "cpu_quota_cgroup": src["cpu_quota_cgroup"],
        "image_size_px":    src["image_size_px"],
        "B_raw_bytes":      B_raw,
        "B_tensor_bytes":   B_tensor,
        "delta_B_bytes":    delta_B,
        "t_src_s":          t_src,
        "t_dst_full_s":     t_dst_full,
        "t_dst_remain_s":   t_dst_remain,
        "delta_t_s":        delta_t,
        "bw_star_kbps":     bw_star,
        "note":             note,
    }


def print_result(src_path: str, r: dict):
    print(f"\n{'='*60}")
    print(f"  Source JSON  : {src_path}")
    print(f"  Throttle     : {r['cpu_quota_cgroup']}")
    print(f"  Image size   : {r['image_size_px']}")
    print(f"  B_raw        : {r['B_raw_bytes']:>10,} bytes  (thumbnail)")
    print(f"  B_dst_wire   : {r['B_tensor_bytes']:>10,} bytes  (full raw JPEG, destination path)")
    print(f"  DB           : {r['delta_B_bytes']:>+10,} bytes")
    print(f"  t_src        : {r['t_src_s']*1000:8.2f} ms  (throttled, median)")
    print(f"  t_dst_full   : {r['t_dst_full_s']*1000:8.2f} ms  (unthrottled, median)")
    print(f"  t_dst_remain : {r['t_dst_remain_s']*1000:8.2f} ms  (unthrottled, median)")
    print(f"  Dt           : {r['delta_t_s']*1000:+8.2f} ms  "
          f"(+= source path slower)")

    if r["bw_star_kbps"] is None:
        print(f"  BW*          : UNDEFINED — {r['note']}")
    else:
        bw = r["bw_star_kbps"]
        print(f"  BW*          : {bw:,.0f} Kbps  ({bw/1000:.2f} Mbps)")
        print(f"  Note         : {r['note']}")

        # Suggested bandwidth sweep (two octaves below, three above)
        print("\n  Suggested sign-check / sweep bandwidths:")
        for mult, label in [(0.25, "0.25×"), (0.50, "0.5×"),
                            (1.00, "1.0×  (BW*)"),
                            (2.00, "2×"),  (4.00, "4×"), (8.00, "8×")]:
            print(f"    {label:18s} {bw * mult:>10,.0f} Kbps")

    # Unit chain (explicit)
    if r["bw_star_kbps"] is not None:
        dB = r["delta_B_bytes"]
        dt = r["delta_t_s"]
        bw = r["bw_star_kbps"]
        unit = -dB*8/(1000*dt)
        if abs(bw - unit) < 1:
            print(f"\n  Unit check: {-dB:,} bytes x 8 / (1000 x {dt:.4f} s) = {unit:,.1f} Kbps  OK")
        else:
            print(f"\n  Unit check: MISMATCH - investigate (got {unit:,.1f}, expected {bw:,.1f})")
    print()


def main():
    if len(sys.argv) < 3:
        print("Usage: compute_bw_star.py <source1.json> [source2.json ...] <dest.json>")
        sys.exit(1)

    *src_paths, dst_path = sys.argv[1:]
    dst = load(dst_path)

    print(f"Destination JSON : {dst_path}")
    print(f"  t_dst_full   median={dst['t_dst_full_median_s']*1000:.2f} ms  "
          f"CV={dst['t_dst_full_cv']:.3f}")
    print(f"  t_dst_remain median={dst['t_dst_remain_median_s']*1000:.2f} ms  "
          f"CV={dst['t_dst_remain_cv']:.3f}")

    for sp in src_paths:
        src = load(sp)
        r = compute_bw_star(src, dst)
        print_result(sp, r)


if __name__ == "__main__":
    main()

