#!/usr/bin/env python3
"""
analyze_results.py
------------------
Reads results/results.csv and produces:
  1. Per-mode summary statistics (latency, throughput, bytes)
  2. Adaptive vs. fixed policy comparison
  3. Crossover detection across bandwidth conditions
  4. Plots (if matplotlib is available)

Usage
-----
    python scripts/analyze_results.py [--input results/results.csv] [--plots]
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if not r.get("error")]


def group_by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    return dict(groups)


def latency_stats(rows: list[dict]) -> dict:
    lats = [float(r["total_time_s"]) * 1000 for r in rows if r.get("total_time_s")]
    if not lats:
        return {}
    return {
        "n": len(lats),
        "mean_ms": round(statistics.mean(lats), 2),
        "median_ms": round(statistics.median(lats), 2),
        "stdev_ms": round(statistics.stdev(lats), 2) if len(lats) > 1 else 0.0,
        "min_ms": round(min(lats), 2),
        "max_ms": round(max(lats), 2),
        "throughput_rps": round(1000 / statistics.mean(lats), 2) if lats else 0,
    }


def bytes_stats(rows: list[dict]) -> dict:
    sizes = [int(r["bytes_transferred"]) for r in rows if r.get("bytes_transferred")]
    if not sizes:
        return {}
    return {
        "total_kb": round(sum(sizes) / 1024, 1),
        "mean_kb": round(statistics.mean(sizes) / 1024, 2),
    }


def decision_stats(rows: list[dict]) -> dict:
    decisions = [r.get("decision", "") for r in rows]
    n = len(decisions)
    if n == 0:
        return {}
    src = decisions.count("source")
    dst = decisions.count("destination")
    return {
        "source_pct": round(src / n * 100, 1),
        "destination_pct": round(dst / n * 100, 1),
        "switches": sum(
            1 for i in range(1, len(decisions))
            if decisions[i] != decisions[i - 1]
        ),
    }


def print_table(headers: list[str], rows: list[list]) -> None:
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    sep = "  ".join("-" * w for w in widths)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/results.csv")
    parser.add_argument("--plots", action="store_true",
                        help="Generate matplotlib plots (requires matplotlib)")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"ERROR: {path} not found.", file=sys.stderr)
        sys.exit(1)

    rows = load_csv(path)
    print(f"Loaded {len(rows)} valid rows from {path}\n")

    # -----------------------------------------------------------------------
    # 1. Per-mode latency summary
    # -----------------------------------------------------------------------
    print("=" * 65)
    print("LATENCY SUMMARY BY MODE")
    print("=" * 65)
    by_mode = group_by(rows, "mode")
    table_rows = []
    for mode in ["source", "destination", "adaptive"]:
        if mode not in by_mode:
            continue
        s = latency_stats(by_mode[mode])
        b = bytes_stats(by_mode[mode])
        table_rows.append([
            mode,
            s.get("n", 0),
            s.get("mean_ms", "–"),
            s.get("median_ms", "–"),
            s.get("stdev_ms", "–"),
            s.get("throughput_rps", "–"),
            b.get("mean_kb", "–"),
        ])
    print_table(
        ["mode", "n", "mean_ms", "median_ms", "stdev_ms", "rps", "mean_KB"],
        table_rows,
    )

    # -----------------------------------------------------------------------
    # 2. Decision breakdown for adaptive
    # -----------------------------------------------------------------------
    if "adaptive" in by_mode:
        print("\n" + "=" * 65)
        print("ADAPTIVE POLICY DECISIONS")
        print("=" * 65)
        ds = decision_stats(by_mode["adaptive"])
        print(f"  Source-side    : {ds.get('source_pct', 0):.1f}%")
        print(f"  Dest-side      : {ds.get('destination_pct', 0):.1f}%")
        print(f"  Switches       : {ds.get('switches', 0)}")

    # -----------------------------------------------------------------------
    # 3. Per-bandwidth breakdown
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("LATENCY BY BANDWIDTH × MODE")
    print("=" * 65)
    bw_mode_groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        bw_mode_groups[(r["bw_kbps"], r["mode"])].append(r)

    bw_table = []
    for (bw, mode), grp in sorted(bw_mode_groups.items(), key=lambda x: (int(x[0][0] or 0), x[0][1])):
        s = latency_stats(grp)
        bw_table.append([bw, mode, s.get("n", 0), s.get("mean_ms", "–"), s.get("stdev_ms", "–")])
    print_table(["bw_kbps", "mode", "n", "mean_ms", "stdev_ms"], bw_table)

    # -----------------------------------------------------------------------
    # 4. Optional plots
    # -----------------------------------------------------------------------
    if args.plots:
        try:
            import matplotlib.pyplot as plt

            out_dir = Path("results/plots")
            out_dir.mkdir(parents=True, exist_ok=True)

            # Latency CDF per mode
            fig, ax = plt.subplots(figsize=(8, 5))
            for mode, color in [("source", "steelblue"), ("destination", "tomato"), ("adaptive", "seagreen")]:
                if mode not in by_mode:
                    continue
                lats = sorted([float(r["total_time_s"]) * 1000 for r in by_mode[mode] if r.get("total_time_s")])
                cdf = [(i + 1) / len(lats) for i in range(len(lats))]
                ax.plot(lats, cdf, label=mode, color=color, linewidth=2)
            ax.set_xlabel("End-to-end latency (ms)")
            ax.set_ylabel("CDF")
            ax.set_title("Latency CDF by Placement Mode")
            ax.legend()
            ax.grid(True, alpha=0.3)
            cdf_path = out_dir / "latency_cdf.png"
            fig.savefig(cdf_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"\nPlot saved: {cdf_path}")

            # Mean latency vs bandwidth
            bws = sorted(set(r["bw_kbps"] for r in rows), key=lambda x: int(x or 0))
            fig, ax = plt.subplots(figsize=(8, 5))
            for mode, color in [("source", "steelblue"), ("destination", "tomato"), ("adaptive", "seagreen")]:
                means = []
                for bw in bws:
                    grp = [r for r in rows if r["bw_kbps"] == bw and r["mode"] == mode]
                    if grp:
                        lats = [float(r["total_time_s"]) * 1000 for r in grp if r.get("total_time_s")]
                        means.append(statistics.mean(lats) if lats else None)
                    else:
                        means.append(None)
                valid = [(int(bw), m) for bw, m in zip(bws, means) if m is not None]
                if valid:
                    xs, ys = zip(*valid)
                    ax.plot(xs, ys, marker="o", label=mode, color=color, linewidth=2)
            ax.set_xlabel("Bandwidth (Kbps)")
            ax.set_ylabel("Mean latency (ms)")
            ax.set_title("Mean Latency vs Bandwidth")
            ax.legend()
            ax.grid(True, alpha=0.3)
            bw_path = out_dir / "latency_vs_bandwidth.png"
            fig.savefig(bw_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Plot saved: {bw_path}")

        except ImportError:
            print("\nmatplotlib not installed. Skipping plots.")

    print("\nDone.")


if __name__ == "__main__":
    main()
