"""
measure_timing.py — per-role timing measurement for Phase B BW* estimation.

Usage
-----
# Source role — run inside a throttled container
docker run --cpus=0.1 --rm \\
  -v ${PWD}/data:/data \\
  -v ${PWD}/shared:/app/shared \\
  -v ${PWD}/scripts:/app/scripts \\
  preprocess_placement-source \\
  python /app/scripts/measure_timing.py \\
    --role source \\
    --image /data/phase_b/medium.jpg \\
    --out /data/source_0.1.json

# Destination role — run UNTHROTTLED (no --cpus flag)
docker run --rm \\
  -v ${PWD}/data:/data \\
  -v ${PWD}/shared:/app/shared \\
  -v ${PWD}/scripts:/app/scripts \\
  preprocess_placement-destination \\
  python /app/scripts/measure_timing.py \\
    --role destination \\
    --image /data/phase_b/medium.jpg \\
    --out /data/dest.json

Output JSON fields (source role)
---------------------------------
  role              : "source"
  image_path        : str
  image_size_px     : [W, H]
  image_bytes       : int          # raw file size
  thumb_bytes       : int          # thumbnail wire size
  tensor_bytes      : int          # fixed = 3*224*224*4
  n_trials          : int
  torch_threads     : int
  cpu_quota_cgroup  : str          # contents of /sys/fs/cgroup/cpu.max
  t_src_median_s    : float        # median of source-path trials
  t_src_mean_s      : float
  t_src_cv          : float        # coefficient of variation
  trials_src_s      : [float]

Output JSON fields (destination role)
--------------------------------------
  role              : "destination"
  image_path, image_size_px, image_bytes, thumb_bytes, tensor_bytes,
  n_trials, torch_threads, cpu_quota_cgroup  (same as above)
  t_dst_full_median_s   : float
  t_dst_full_mean_s     : float
  t_dst_full_cv         : float
  t_dst_remain_median_s : float
  t_dst_remain_mean_s   : float
  t_dst_remain_cv       : float
  trials_dst_full_s     : [float]
  trials_dst_remain_s   : [float]

Notes
-----
- Run source and destination containers SEQUENTIALLY — they share host cores.
- torch.set_num_threads(1) is called here AND must be set in the service nodes.
  Changing from 2→1 makes Phase A and Phase B latency numbers not directly
  comparable; document this in the write-up.
- Reads image bytes into memory once; disk I/O does not leak into timed region.
- Decode is INSIDE the timed loop so it scales with the input you're sweeping.
- Reports mean and median; use median for BW* (robust to CFS stall periods).
- CV > 0.10 triggers a warning; re-run if CV > 0.20.
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch

# Ensure shared/ is importable whether running as /app/scripts/... or locally
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
import phase_b_transforms as T  # noqa: E402

CV_WARN  = 0.10
CV_ABORT = 0.20
N_WARMUP = 3
N_TRIALS = 20


def read_cgroup_cpu():
    """Return the contents of /sys/fs/cgroup/cpu.max, or 'unavailable'."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            return f.read().strip()
    except OSError:
        return "unavailable"


def timed(fn, *args):
    """Return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - t0


def measure_source(image_bytes: bytes, n_trials: int):
    """
    Time the full source-side pipeline for each trial.
    Decode is inside the loop so it scales with input resolution.
    """
    # Warmup (not recorded)
    for _ in range(N_WARMUP):
        T.run_source_path(image_bytes)

    trials = []
    for _ in range(n_trials):
        _, elapsed = timed(T.run_source_path, image_bytes)
        trials.append(elapsed)

    # Measure thumb size from one representative run
    thumb = T.run_source_path(image_bytes)
    return trials, len(thumb)


def measure_destination(image_bytes: bytes, n_trials: int):
    """
    Time t_dst_full and t_dst_remain on alternating trials to reduce drift.
    t_dst_remain uses a freshly produced thumbnail for each trial.
    """
    # Pre-produce thumbnail (its decode cost is captured in t_dst_remain below)
    thumb = T.run_source_path(image_bytes)

    # Warmup
    for _ in range(N_WARMUP):
        T.run_dst_full(image_bytes)
        T.run_dst_remain(thumb)

    trials_full   = []
    trials_remain = []

    for _ in range(n_trials):
        # Interleave to cancel slow thermal/cache drift
        _, t_full   = timed(T.run_dst_full,   image_bytes)
        _, t_remain = timed(T.run_dst_remain,  thumb)
        trials_full.append(t_full)
        trials_remain.append(t_remain)

    return trials_full, trials_remain, len(thumb)


def cv(vals):
    m = statistics.mean(vals)
    return statistics.stdev(vals) / m if m > 0 else float("inf")


def check_cv(label, coefficient):
    if coefficient > CV_ABORT:
        print(f"  WARNING: {label} CV={coefficient:.3f} > {CV_ABORT} — "
              "results unreliable, re-run with fewer background processes.")
    elif coefficient > CV_WARN:
        print(f"  NOTICE:  {label} CV={coefficient:.3f} > {CV_WARN} — "
              "consider re-running.")


def main():
    parser = argparse.ArgumentParser(description="Phase B timing measurement")
    parser.add_argument("--role",   required=True, choices=["source", "destination"])
    parser.add_argument("--image",  required=True, help="Path to input JPEG")
    parser.add_argument("--out",    required=True, help="Output JSON path")
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    args = parser.parse_args()

    # Pin threads — must match service nodes
    torch.set_num_threads(1)
    print(f"[measure_timing] torch.get_num_threads() = {torch.get_num_threads()}")

    # Read image into memory once — disk I/O must not leak into timed region
    with open(args.image, "rb") as f:
        image_bytes = f.read()

    # Verify image can be decoded and get dimensions
    import io
    from PIL import Image
    img_check = Image.open(io.BytesIO(image_bytes))
    w, h = img_check.size
    print(f"[measure_timing] Image: {args.image}  {w}×{h}  {len(image_bytes):,} bytes")

    cgroup = read_cgroup_cpu()
    print(f"[measure_timing] /sys/fs/cgroup/cpu.max = {cgroup}")

    result = {
        "role":             args.role,
        "image_path":       args.image,
        "image_size_px":    [w, h],
        "image_bytes":      len(image_bytes),
        "tensor_bytes":     T.TENSOR_BYTES,
        "n_trials":         args.trials,
        "torch_threads":    torch.get_num_threads(),
        "cpu_quota_cgroup": cgroup,
    }

    if args.role == "source":
        trials, thumb_bytes = measure_source(image_bytes, args.trials)
        med  = statistics.median(trials)
        mean = statistics.mean(trials)
        c    = cv(trials)
        check_cv("t_src", c)
        result.update({
            "thumb_bytes":       thumb_bytes,
            "t_src_median_s":    med,
            "t_src_mean_s":      mean,
            "t_src_cv":          c,
            "trials_src_s":      trials,
        })
        print(f"  t_src  median={med*1000:.1f} ms  mean={mean*1000:.1f} ms  CV={c:.3f}")
        print(f"  thumb_bytes = {thumb_bytes:,}")

    else:  # destination
        trials_full, trials_remain, thumb_bytes = measure_destination(
            image_bytes, args.trials)

        med_full    = statistics.median(trials_full)
        mean_full   = statistics.mean(trials_full)
        cv_full     = cv(trials_full)
        med_remain  = statistics.median(trials_remain)
        mean_remain = statistics.mean(trials_remain)
        cv_remain   = cv(trials_remain)

        check_cv("t_dst_full",   cv_full)
        check_cv("t_dst_remain", cv_remain)

        result.update({
            "thumb_bytes":            thumb_bytes,
            "t_dst_full_median_s":    med_full,
            "t_dst_full_mean_s":      mean_full,
            "t_dst_full_cv":          cv_full,
            "t_dst_remain_median_s":  med_remain,
            "t_dst_remain_mean_s":    mean_remain,
            "t_dst_remain_cv":        cv_remain,
            "trials_dst_full_s":      trials_full,
            "trials_dst_remain_s":    trials_remain,
        })
        print(f"  t_dst_full   median={med_full*1000:.1f} ms  "
              f"mean={mean_full*1000:.1f} ms  CV={cv_full:.3f}")
        print(f"  t_dst_remain median={med_remain*1000:.1f} ms  "
              f"mean={mean_remain*1000:.1f} ms  CV={cv_remain:.3f}")
        print(f"  thumb_bytes = {thumb_bytes:,}")

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[measure_timing] Wrote {args.out}")


if __name__ == "__main__":
    main()

