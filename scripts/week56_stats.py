#!/usr/bin/env python3
"""Compute Week 5 and Week 6 summary statistics from results.csv."""
import csv, statistics
from collections import defaultdict, Counter
from pathlib import Path

rows = [r for r in csv.DictReader(open('results/results.csv')) if not r.get('error')]

# -- Week 5: all rows (the CSV has no 'seed' column; old filter always returned empty) --
w5 = rows
print(f"Week 5 rows: {len(w5)}")

bw_mode = defaultdict(list)
for r in w5:
    if r.get('total_time_s'):
        bw_mode[(int(r['bw_kbps']), r['mode'])].append(float(r['total_time_s'])*1000)

header = "BW(Kbps)  Mode           N   Mean(ms)  Stdev(ms)  Min(ms)  Max(ms)"
print("\nWeek 5 – Pooled latency (seeds 1-3):")
print(header)
print("-" * len(header))
for (bw, mode) in sorted(bw_mode.keys()):
    lats = bw_mode[(bw, mode)]
    mn = statistics.mean(lats)
    sd = statistics.stdev(lats) if len(lats) > 1 else 0.0
    print(f"{bw:>8}  {mode:<13} {len(lats):>4}  {mn:>8.1f}  {sd:>9.1f}  {min(lats):>7.1f}  {max(lats):>7.1f}")

# ── Week 6: the last 4 adaptive run_ids ──────────────────────────────────────
adaptive_all = [r for r in rows if r['mode'] == 'adaptive']
run_ids_sorted = sorted(set(r['run_id'] for r in adaptive_all), reverse=True)
w6_runs = set(run_ids_sorted[:4])
w6_rows = [r for r in adaptive_all if r['run_id'] in w6_runs]

print(f"\nWeek 6 rows: {len(w6_rows)}")
print("Week 6 – Transition runs (adaptive only):")
print("RunID                  BW(Kbps)  N   Mean(ms)  Decisions")
print("-" * 65)
for rid in sorted(w6_runs):
    grp = [r for r in w6_rows if r['run_id'] == rid]
    lats = [float(r['total_time_s'])*1000 for r in grp if r.get('total_time_s')]
    decs = Counter(r.get('decision') for r in grp)
    bw = grp[0]['bw_kbps'] if grp else '?'
    n = len(grp)
    mean_lat = statistics.mean(lats) if lats else 0
    dec_str = f"src={decs['source']} dst={decs['destination']}"
    print(f"{rid}  {bw:>8}  {n:>3}  {mean_lat:>8.1f}  {dec_str}")

print("\nDone.")
