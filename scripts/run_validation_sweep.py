#!/usr/bin/env python3
"""
run_validation_sweep.py
-----------------------
Runs a 20-image interleaved sweep across bandwidths to validate
the placement cost model against measured wire and total latencies.
"""
import subprocess
import sys
from pathlib import Path

BANDWIDTHS = [5000, 2000, 1000, 500, 256]
N_IMAGES = 20
OUTPUT_CSV = "results/validation_sweep_20.csv"

def main():
    out_file = Path(OUTPUT_CSV)
    if out_file.exists():
        out_file.unlink()

    print(f"=== Starting 20-Image Cost-Model Validation Sweep ===")
    print(f"Bandwidths: {BANDWIDTHS} Kbps")
    print(f"Images per condition: {N_IMAGES} (Interleaved across modes)")
    print(f"Output: {OUTPUT_CSV}\n")

    for bw in BANDWIDTHS:
        print(f"\n>>>> Executing Bandwidth {bw} Kbps <<<<")
        cmd = [
            sys.executable,
            "scripts/run_experiment.py",
            "--mode", "all",
            "--bw", str(bw),
            "--n", str(N_IMAGES),
            "--interleave",
            "--output", OUTPUT_CSV,
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"ERROR: Experiment runner exited with code {res.returncode}")
            sys.exit(res.returncode)

    print("\n=== Validation Sweep Complete ===")

if __name__ == "__main__":
    main()
