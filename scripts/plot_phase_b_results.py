#!/usr/bin/env python3
"""
plot_phase_b_results.py
-----------------------
Generates high-resolution presentation plots for Phase B:
1. phase_b_latency_vs_bandwidth.png: Latency vs. Bandwidth (src_thumbnail vs. destination vs. adaptive)
   highlighting the theoretical crossover BW* = 15.3 Mbps.
2. phase_b_transition_trace.png: Dynamic adaptation trace across BW* showing latency,
   estimated bandwidth, and decision switching points.
"""

import os
from pathlib import Path

# Fix matplotlib temp dir warning on Windows
os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "/tmp"), "mpl_config")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
SWEEP_CSV = RESULTS_DIR / "phase_b_sweep.csv"
TRANSITION_CSV = RESULTS_DIR / "phase_b_transition.csv"

BW_STAR_KBPS = 15_271


def plot_sweep():
    if not SWEEP_CSV.exists():
        print(f"Sweep CSV not found at {SWEEP_CSV}, skipping sweep plot.")
        return

    df = pd.read_csv(SWEEP_CSV)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    # Styles
    src_color = "#d95f02"   # Orange/Red for thumbnail
    dst_color = "#1f78b4"   # Blue for destination
    adp_color = "#2ca02c"   # Green for adaptive

    # Group by bandwidth and mode requested
    grouped = df.groupby(["bw_kbps", "mode_requested"])

    bws = sorted(df["bw_kbps"].unique())
    bws_mbps = [bw / 1000.0 for bw in bws]

    # Extract medians and 25-75th percentiles (or IQR) for robustness
    src_meds = []
    dst_meds = []
    adp_meds = []

    for bw in bws:
        if (bw, "source_thumbnail") in grouped.groups:
            src_meds.append(grouped.get_group((bw, "source_thumbnail"))["latency_s"].median() * 1000)
        else:
            src_meds.append(np.nan)

        if (bw, "destination") in grouped.groups:
            dst_meds.append(grouped.get_group((bw, "destination"))["latency_s"].median() * 1000)
        else:
            dst_meds.append(np.nan)

        if (bw, "adaptive") in grouped.groups:
            adp_meds.append(grouped.get_group((bw, "adaptive"))["latency_s"].median() * 1000)
        else:
            adp_meds.append(np.nan)

    # Plot baseline curves
    ax.plot(bws_mbps, src_meds, "o-", color=src_color, linewidth=2.2, markersize=8, label="Fixed: source_thumbnail (~20.5 KB)", zorder=4)
    ax.plot(bws_mbps, dst_meds, "s-", color=dst_color, linewidth=2.2, markersize=8, label="Fixed: destination (~1.41 MB)", zorder=4)
    ax.plot(bws_mbps, adp_meds, "^--", color=adp_color, linewidth=2.5, markersize=9, label="Adaptive Placement Policy", zorder=5)

    # Vertical line at theoretical crossover BW*
    bw_star_mbps = BW_STAR_KBPS / 1000.0
    ax.axvline(x=bw_star_mbps, color="#666666", linestyle=":", linewidth=2, label=f"Theoretical $BW^* = {bw_star_mbps:.1f}$ Mbps", zorder=2)

    # Shaded regions
    ax.axvspan(min(bws_mbps) * 0.8, bw_star_mbps, color=src_color, alpha=0.07, label="Thumbnail Wins ($BW < BW^*$)")
    ax.axvspan(bw_star_mbps, max(bws_mbps) * 1.2, color=dst_color, alpha=0.07, label="Destination Wins ($BW > BW^*$)")

    ax.set_xscale("log")
    ax.set_xlabel("Bandwidth (Mbps, log scale)", fontsize=12, labelpad=8)
    ax.set_ylabel("End-to-End Latency (ms, Median)", fontsize=12, labelpad=8)
    ax.set_title("Phase B: Adaptive Preprocessing Placement vs. Bandwidth\n(Compressive Workload, Source CPU = 0.1 core)", fontsize=13, fontweight="bold", pad=12)

    ax.set_xticks(bws_mbps)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}" if x < 10 else f"{int(x)}"))
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=10, loc="upper right")

    out_file = PLOTS_DIR / "phase_b_latency_vs_bandwidth.png"
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"Saved: {out_file}")


def plot_transition():
    if not TRANSITION_CSV.exists():
        print(f"Transition CSV not found at {TRANSITION_CSV}, skipping transition plot.")
        return

    df = pd.read_csv(TRANSITION_CSV)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, dpi=300)

    seqs = df["global_seq"]
    lat_ms = df["latency_s"] * 1000.0
    bw_target_mbps = df["target_bw_kbps"] / 1000.0
    bw_est_mbps = df["bw_est_kbps"] / 1000.0
    decisions = df["decision"]

    # Colors for decisions
    colors = ["#d95f02" if d == "source_thumbnail" else "#1f78b4" for d in decisions]

    # Top plot: Latency
    ax1.plot(seqs, lat_ms, color="#888888", linestyle="-", alpha=0.5, zorder=2)
    scatter = ax1.scatter(seqs, lat_ms, c=colors, s=70, zorder=3)
    ax1.set_ylabel("Latency (ms)", fontsize=11)
    ax1.set_title("Phase B: Dynamic Bandwidth Transition & Adaptive Switching", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Highlight switch points
    switch_points = df[df["switched"] == True]
    for _, row in switch_points.iterrows():
        ax1.annotate(
            f"SWITCH\n({row['decision']})",
            xy=(row["global_seq"], row["latency_s"] * 1000),
            xytext=(row["global_seq"], row["latency_s"] * 1000 + 1200),
            arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
            fontsize=9, fontweight="bold", ha="center"
        )

    # Custom legend for top plot
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d95f02', markersize=9, label='Decision: source_thumbnail'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f78b4', markersize=9, label='Decision: destination'),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", frameon=True)

    # Bottom plot: Bandwidth
    ax2.step(seqs, bw_target_mbps, where="mid", color="#333333", linestyle="--", linewidth=1.8, label="Target Shaped Bandwidth")
    ax2.plot(seqs, bw_est_mbps, marker="x", color="#2ca02c", linewidth=2, label="Policy EMA Bandwidth Estimate")
    ax2.axhline(y=BW_STAR_KBPS / 1000.0, color="#d62728", linestyle=":", linewidth=1.8, label=f"BW* Crossover ({BW_STAR_KBPS/1000.0:.1f} Mbps)")

    ax2.set_xlabel("Request Sequence", fontsize=11)
    ax2.set_ylabel("Bandwidth (Mbps)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", frameon=True)

    out_file = PLOTS_DIR / "phase_b_transition_trace.png"
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"Saved: {out_file}")


def main():
    plot_sweep()
    plot_transition()


if __name__ == "__main__":
    main()

