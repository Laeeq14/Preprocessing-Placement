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

REPO_ROOT = Path(__file__).resolve().parent.parent
BANDWIDTHS = [5000, 2000, 1000, 500, 256]
N_IMAGES = 20
OUTPUT_CSV = REPO_ROOT / "results" / "validation_sweep_20.csv"
RUN_EXP_SCRIPT = REPO_ROOT / "scripts" / "run_experiment.py"

def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()

    print(f"=== Starting 20-Image Cost-Model Validation Sweep ===")
    print(f"Bandwidths: {BANDWIDTHS} Kbps")
    print(f"Images per condition: {N_IMAGES} (Interleaved across modes)")
    print(f"Output: {OUTPUT_CSV}\n")

    for bw in BANDWIDTHS:
        print(f"\n>>>> Executing Bandwidth {bw} Kbps <<<<")
        cmd = [
            sys.executable,
            str(RUN_EXP_SCRIPT),
            "--mode", "all",
            "--bw", str(bw),
            "--n", str(N_IMAGES),
            "--interleave",
            "--output", str(OUTPUT_CSV),
        ]
        res = subprocess.run(cmd, cwd=REPO_ROOT)
        if res.returncode != 0:
            print(f"ERROR: Experiment runner exited with code {res.returncode}")
            sys.exit(res.returncode)

    print("\n=== Validation Sweep Complete ===")

if __name__ == "__main__":
    main()
