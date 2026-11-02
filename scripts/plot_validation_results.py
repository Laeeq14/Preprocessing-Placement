#!/usr/bin/env python3
"""
plot_validation_results.py
--------------------------
Generates:
1. Predicted vs. Measured Wire Time plot (log y-axis, 5 bandwidths, both payloads).
2. Summary statistics table: median latency, p95, and 95% bootstrap CI per mode per bandwidth.
"""
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = REPO_ROOT / "results" / "validation_sweep_20.csv"
PLOTS_DIR = REPO_ROOT / "results" / "plots"

def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)

    bws = [256, 500, 1000, 2000, 5000]
    
    # -------------------------------------------------------------
    # 1. Predicted vs. Measured Wire Time Plot
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 6), dpi=300)
    ax = plt.subplot(1, 1, 1)

    # Styles
    dst_color = "#1f77b4"  # Blue
    src_color = "#d62728"  # Red

    # Group by bandwidth and mode for Destination and Source
    dst_data = df[df["mode"] == "destination"].groupby("bw_kbps")
    src_data = df[df["mode"] == "source"].groupby("bw_kbps")

    dst_meas_mean = [dst_data.get_group(bw)["wire_time_s"].mean() for bw in bws]
    dst_meas_std = [dst_data.get_group(bw)["wire_time_s"].std() for bw in bws]
    dst_pred_mean = [dst_data.get_group(bw)["pred_cost_s"].mean() for bw in bws]

    src_meas_mean = [src_data.get_group(bw)["wire_time_s"].mean() for bw in bws]
    src_meas_std = [src_data.get_group(bw)["wire_time_s"].std() for bw in bws]
    src_pred_mean = [src_data.get_group(bw)["pred_cost_s"].mean() for bw in bws]

    # Plot Measured
    ax.errorbar(
        bws, dst_meas_mean, yerr=dst_meas_std, fmt="o", color=dst_color,
        capsize=4, elinewidth=1.5, markersize=8, label="Measured Raw JPEG (~157 KiB)", zorder=5
    )
    ax.errorbar(
        bws, src_meas_mean, yerr=src_meas_std, fmt="s", color=src_color,
        capsize=4, elinewidth=1.5, markersize=8, label="Measured Float32 Tensor (~589 KiB)", zorder=5
    )

    # Plot Predicted (dashed curves)
    ax.plot(
        bws, dst_pred_mean, linestyle="--", color=dst_color, linewidth=2,
        label="Predicted Raw JPEG ($S_{\\text{raw}} / \\text{BW}$)", zorder=4
    )
    ax.plot(
        bws, src_pred_mean, linestyle="--", color=src_color, linewidth=2,
        label="Predicted Tensor ($S_{\\text{tensor}} / \\text{BW}$)", zorder=4
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(bws)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax.set_xlabel("Configured Bandwidth (Kbps)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Wire Transmission Time (seconds, log scale)", fontsize=12, fontweight="bold")
    ax.set_title("Cost Model Validation: Predicted vs. Measured Wire Time\n(Expansive Workload, 5 Bandwidths, 20 Images Each)", fontsize=13, fontweight="bold", pad=12)

    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(frameon=True, fontsize=10, loc="upper right")

    plot_path = PLOTS_DIR / "predicted_vs_measured_wire_time.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to: {plot_path}")

    # -------------------------------------------------------------
    # 2. Table of Median, p95, and Bootstrap CI
    # -------------------------------------------------------------
    rng = np.random.default_rng(42)
    B = 10000

    rows = []
    for bw in [5000, 2000, 1000, 500, 256]:
        for mode in ["destination", "source", "adaptive"]:
            vals = df[(df["bw_kbps"] == bw) & (df["mode"] == mode)]["total_time_s"].values * 1000
            med = float(np.median(vals))
            p95 = float(np.percentile(vals, 95))
            
            boot = rng.choice(vals, size=(B, len(vals)), replace=True)
            boot_meds = np.median(boot, axis=1)
            ci_low = float(np.percentile(boot_meds, 2.5))
            ci_high = float(np.percentile(boot_meds, 97.5))
            
            rows.append({
                "Bandwidth (Kbps)": bw,
                "Mode": mode.capitalize(),
                "Median (ms)": f"{med:.1f}",
                "p95 (ms)": f"{p95:.1f}",
                "95% Bootstrap CI (ms)": f"[{ci_low:.1f}, {ci_high:.1f}]"
            })

    table_df = pd.DataFrame(rows)
    print("\n" + table_df.to_markdown(index=False))

if __name__ == "__main__":
    main()
