#!/usr/bin/env python3
"""
plot_phase_a_results.py
-----------------------
Generates presentation-quality plots for Phase A (Expansive Workload),
mirroring the structure and aesthetics of Phase B:
1. phase_a_latency_vs_bandwidth.png: Latency vs. Bandwidth (source vs. destination vs. adaptive)
   highlighting that destination wins at all bandwidths (no crossover due to 3.75x payload inflation).
2. phase_a_transition_trace.png: Dynamic adaptation trace across the 5 bandwidth levels showing
   end-to-end latency, target shaped bandwidth, policy EMA bandwidth tracking, and 100% destination decisions.
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
SWEEP_CSV = RESULTS_DIR / "validation_sweep_20.csv"


def plot_sweep():
    if not SWEEP_CSV.exists():
        print(f"Sweep CSV not found at {SWEEP_CSV}, skipping sweep plot.")
        return

    df = pd.read_csv(SWEEP_CSV)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    # Styles matching Phase B
    src_color = "#d95f02"   # Orange/Red for source tensor
    dst_color = "#1f78b4"   # Blue for destination raw JPEG
    adp_color = "#2ca02c"   # Green for adaptive

    # Group by bandwidth and mode
    grouped = df.groupby(["bw_kbps", "mode"])

    bws = sorted(df["bw_kbps"].unique())
    bws_mbps = [bw / 1000.0 for bw in bws]

    # Extract medians and IQRs (25th and 75th percentiles)
    src_meds = []
    dst_meds = []
    adp_meds = []

    for bw in bws:
        if (bw, "source") in grouped.groups:
            src_meds.append(grouped.get_group((bw, "source"))["total_time_s"].median() * 1000)
        else:
            src_meds.append(np.nan)

        if (bw, "destination") in grouped.groups:
            dst_meds.append(grouped.get_group((bw, "destination"))["total_time_s"].median() * 1000)
        else:
            dst_meds.append(np.nan)

        if (bw, "adaptive") in grouped.groups:
            adp_meds.append(grouped.get_group((bw, "adaptive"))["total_time_s"].median() * 1000)
        else:
            adp_meds.append(np.nan)

    # Plot baseline curves
    ax.plot(bws_mbps, src_meds, "o-", color=src_color, linewidth=2.2, markersize=8,
            label="Fixed: source (~589.2 KB float32 tensor)", zorder=4)
    ax.plot(bws_mbps, dst_meds, "s-", color=dst_color, linewidth=2.2, markersize=8,
            label="Fixed: destination (~157.0 KB raw JPEG)", zorder=4)
    ax.plot(bws_mbps, adp_meds, "^--", color=adp_color, linewidth=2.5, markersize=9,
            label="Adaptive Placement Policy", zorder=5)

    # In Phase A, Delta B < 0 and Delta t > 0 => No crossover exists (BW* undefined).
    # Destination wins at every bandwidth!
    ax.axvspan(min(bws_mbps) * 0.8, max(bws_mbps) * 1.25, color=dst_color, alpha=0.07,
               label=r"Destination Wins Everywhere ($\Delta B < 0$, No Crossover)", zorder=1)

    # Add text annotation explaining the null result / geometry
    ax.annotate(
        "No Crossover Possible:\nDestination payload (157 KB) < Source tensor (589 KB)\nExpansion ratio = 3.75× -> Destination dominates at all BW",
        xy=(bws_mbps[2], dst_meds[2]),
        xytext=(0.4, 7500),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#1f78b4", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="#1f78b4", lw=1.5),
        fontsize=9, fontweight="semibold", color="#08306b"
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Bandwidth (Mbps, log scale)", fontsize=12, labelpad=8)
    ax.set_ylabel("End-to-End Latency (ms, log scale)", fontsize=12, labelpad=8)
    ax.set_title("Phase A: Preprocessing Placement vs. Bandwidth\n(Expansive Workload, COCO ResNet-18 Preprocessing)",
                 fontsize=13, fontweight="bold", pad=12)

    ax.set_xticks(bws_mbps)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}" if x < 1 else f"{x:.1f}" if x < 10 else f"{int(x)}"))
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=10, loc="upper right")

    out_file = PLOTS_DIR / "phase_a_latency_vs_bandwidth.png"
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"Saved: {out_file}")

    # Also update latency_vs_bandwidth.png in standard location
    out_file_std = PLOTS_DIR / "latency_vs_bandwidth.png"
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    ax.plot(bws_mbps, src_meds, "o-", color=src_color, linewidth=2.2, markersize=8,
            label="Fixed: source (~589.2 KB float32 tensor)", zorder=4)
    ax.plot(bws_mbps, dst_meds, "s-", color=dst_color, linewidth=2.2, markersize=8,
            label="Fixed: destination (~157.0 KB raw JPEG)", zorder=4)
    ax.plot(bws_mbps, adp_meds, "^--", color=adp_color, linewidth=2.5, markersize=9,
            label="Adaptive Placement Policy", zorder=5)
    ax.axvspan(min(bws_mbps) * 0.8, max(bws_mbps) * 1.25, color=dst_color, alpha=0.07,
               label=r"Destination Wins Everywhere ($\Delta B < 0$, No Crossover)", zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Bandwidth (Mbps, log scale)", fontsize=12, labelpad=8)
    ax.set_ylabel("End-to-End Latency (ms, log scale)", fontsize=12, labelpad=8)
    ax.set_title("Phase A: Preprocessing Placement vs. Bandwidth\n(Expansive Workload, COCO ResNet-18 Preprocessing)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(bws_mbps)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2f}" if x < 1 else f"{x:.1f}" if x < 10 else f"{int(x)}"))
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_file_std)
    plt.close()
    print(f"Saved: {out_file_std}")


def plot_transition():
    if not SWEEP_CSV.exists():
        print(f"Sweep CSV not found at {SWEEP_CSV}, skipping transition plot.")
        return

    df = pd.read_csv(SWEEP_CSV)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Filter adaptive requests across the stepped bandwidth run
    adp_df = df[df["mode"] == "adaptive"].copy().reset_index(drop=True)
    if adp_df.empty:
        print("No adaptive mode rows found in sweep CSV.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, dpi=300)

    seqs = np.arange(1, len(adp_df) + 1)
    lat_ms = adp_df["total_time_s"] * 1000.0
    bw_target_mbps = adp_df["bw_kbps"] / 1000.0
    bw_est_mbps = adp_df["bw_est_kbps"] / 1000.0
    decisions = adp_df["decision"]

    # Colors for decisions (consistent with Phase B)
    colors = ["#1f78b4" if d == "destination" else "#d95f02" for d in decisions]

    # Top plot: Latency
    ax1.plot(seqs, lat_ms, color="#888888", linestyle="-", alpha=0.5, zorder=2)
    scatter = ax1.scatter(seqs, lat_ms, c=colors, s=55, zorder=3)
    ax1.set_ylabel("Latency (ms)", fontsize=11)
    ax1.set_title("Phase A: Dynamic Bandwidth Stepping & Policy Tracking\n(5 Bandwidths × 20 Images, Interleaved Validated Sweep)",
                  fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Annotate 100% destination decision adherence
    ax1.annotate(
        "100% Destination Decisions (0 Switches)\nAdaptive Policy avoids 3.75× tensor inflation",
        xy=(50, lat_ms.median()),
        xytext=(50, lat_ms.max() * 0.7),
        ha="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#e6f2ff", edgecolor="#1f78b4", alpha=0.9),
        fontsize=9, fontweight="bold", color="#08306b"
    )

    # Custom legend for top plot
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f78b4', markersize=9, label='Decision: destination (100/100)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d95f02', markersize=9, label='Decision: source (0/100)'),
    ]
    ax1.legend(handles=legend_elements, loc="upper left", frameon=True)

    # Bottom plot: Bandwidth
    ax2.step(seqs, bw_target_mbps, where="mid", color="#333333", linestyle="--", linewidth=1.8, label="Target Shaped Bandwidth")
    ax2.plot(seqs, bw_est_mbps, marker="x", color="#2ca02c", markersize=4, linewidth=1.8, label="Policy EMA Bandwidth Estimate")

    # Add vertical step boundaries and labels
    step_transitions = [20.5, 40.5, 60.5, 80.5]
    step_bws = ["5.0 Mbps", "2.0 Mbps", "1.0 Mbps", "0.5 Mbps", "0.256 Mbps"]
    step_centers = [10, 30, 50, 70, 90]

    for x in step_transitions:
        ax1.axvline(x=x, color="#999999", linestyle=":", alpha=0.6)
        ax2.axvline(x=x, color="#999999", linestyle=":", alpha=0.6)

    for center, lbl in zip(step_centers, step_bws):
        ax2.text(center, 5.3, lbl, ha="center", va="bottom", fontsize=8, fontweight="bold", color="#444444")

    ax2.set_xlabel("Request Sequence (COCO images)", fontsize=11)
    ax2.set_ylabel("Bandwidth (Mbps)", fontsize=11)
    ax2.set_ylim(-0.2, 9.0)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", frameon=True)

    out_file = PLOTS_DIR / "phase_a_transition_trace.png"
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"Saved: {out_file}")


def main():
    plot_sweep()
    plot_transition()


if __name__ == "__main__":
    main()
