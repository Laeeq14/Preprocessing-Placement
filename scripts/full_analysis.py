#!/usr/bin/env python3
"""Full cross-week analysis for the research summary."""
import csv, statistics
from collections import defaultdict, Counter

rows = [r for r in csv.DictReader(open('results/results.csv')) if not r.get('error')]
print(f'Total rows: {len(rows)}')

# All adaptive decisions
adaptive = [r for r in rows if r['mode'] == 'adaptive']
decs = Counter(r.get('decision') for r in adaptive)
print(f'Adaptive total: {len(adaptive)} images, decisions: {dict(decs)}')

# Per-bw winner
print()
print('Per-BW winner (lowest mean latency, all data):')
bw_mode = defaultdict(list)
for r in rows:
    if r.get('total_time_s'):
        bw_mode[(int(r['bw_kbps']), r['mode'])].append(float(r['total_time_s']) * 1000)

bws = sorted(set(int(r['bw_kbps']) for r in rows))
for bw in bws:
    res = {}
    for mode in ('source', 'destination', 'adaptive'):
        lats = bw_mode.get((bw, mode), [])
        if lats:
            res[mode] = statistics.mean(lats)
    best = min(res, key=lambda m: res[m])
    src  = f"{res.get('source', 0):>7.1f}"
    dst  = f"{res.get('destination', 0):>7.1f}"
    adp  = f"{res.get('adaptive', 0):>7.1f}"
    print(f"  {bw:>5} Kbps  src={src}ms  dst={dst}ms  adp={adp}ms  winner={best}")

# Transfer sizes
print()
raw_sizes = [int(r['raw_size_bytes']) for r in rows if r.get('raw_size_bytes')]
xfer_src  = [int(r['bytes_transferred']) for r in rows if r['mode'] == 'source' and r.get('bytes_transferred')]
xfer_dst  = [int(r['bytes_transferred']) for r in rows if r['mode'] == 'destination' and r.get('bytes_transferred')]
print(f"Raw JPEG mean:         {statistics.mean(raw_sizes)/1024:.1f} KB")
print(f"Source xfer (tensor):  {statistics.mean(xfer_src)/1024:.1f} KB")
print(f"Dest xfer (raw JPEG):  {statistics.mean(xfer_dst)/1024:.1f} KB")
print(f"Ratio raw/tensor:      {statistics.mean(xfer_dst)/statistics.mean(xfer_src):.3f}")

# Preprocessing times
src_prep  = [float(r['src_preprocess_time_s']) * 1000 for r in rows if r.get('src_preprocess_time_s')]
dst_prep  = [float(r['dst_preprocess_time_s']) * 1000 for r in rows if r.get('dst_preprocess_time_s') and r['dst_preprocess_time_s']]
dst_infer = [float(r['dst_infer_time_s']) * 1000 for r in rows if r.get('dst_infer_time_s') and r['dst_infer_time_s']]
if src_prep:
    print(f"Src preprocess:  mean={statistics.mean(src_prep):.2f}ms  stdev={statistics.stdev(src_prep):.2f}ms")
if dst_prep:
    print(f"Dst preprocess:  mean={statistics.mean(dst_prep):.2f}ms  stdev={statistics.stdev(dst_prep):.2f}ms")
if dst_infer:
    print(f"Dst inference:   mean={statistics.mean(dst_infer):.2f}ms  stdev={statistics.stdev(dst_infer):.2f}ms")

# Variability: coefficient of variation per bw x mode
print()
print('Coefficient of Variation (stdev/mean) per BW x Mode:')
for bw in bws:
    for mode in ('source', 'destination', 'adaptive'):
        lats = bw_mode.get((bw, mode), [])
        if len(lats) > 1:
            cv = statistics.stdev(lats) / statistics.mean(lats)
            print(f"  {bw:>5} Kbps  {mode:<13}  n={len(lats):>4}  cv={cv:.2f}")
