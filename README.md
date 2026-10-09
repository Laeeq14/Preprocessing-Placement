# Adaptive Preprocessing Placement

> **EE 5239 Research Project** — Investigating whether dynamically selecting where image preprocessing occurs improves end-to-end latency/throughput compared with fixed source-side or destination-side preprocessing under changing network conditions.

---

## Project Structure

```
Preprocess_Placement/
├── docker/
│   ├── source/
│   │   ├── Dockerfile          # Source container image
│   │   └── source_node.py      # FastAPI source service (port 8000)
│   └── destination/
│       ├── Dockerfile          # Destination container image
│       └── destination_node.py # FastAPI destination service (port 8001)
├── shared/
│   └── placement_policy.py     # Adaptive placement policy (EMA cost model)
├── scripts/
│   ├── download_coco_sample.py # Download COCO val2017 test images
│   ├── shape_network.sh        # tc-netem/tbf bandwidth shaping (runs in container)
│   ├── run_experiment.py       # Experiment orchestrator
│   ├── validate_equivalence.py # Verify source≡destination output
│   └── analyze_results.py      # Summary statistics + plots
├── data/
│   ├── images/                 # Test images (COCO val2017 JPEGs)
│   └── .cache/                 # Cached COCO annotation zip (auto-created)
├── results/                    # Experiment CSVs and plots written here
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Containerization** | Docker + Docker Compose | Runs source & destination as isolated services on a virtual bridge network |
| **Network Emulation** | `tc-netem` (Linux kernel) | Adds artificial latency and jitter to the source container's outgoing traffic |
| **Bandwidth Limiting** | `tc-tbf` Token Bucket Filter | Caps throughput to a configured Kbps value |
| **Container Permission** | `NET_ADMIN` capability | Grants the source container permission to modify its own network interface |
| **Source Service** | FastAPI + Uvicorn (Python) | Receives images, makes placement decisions, forwards to destination |
| **Destination Service** | FastAPI + Uvicorn (Python) | Optionally preprocesses and runs ResNet-18 CPU inference |
| **Preprocessing** | Torchvision + Pillow | Resize(256) → CenterCrop(224) → ToTensor → Normalize (ImageNet stats) |
| **Inference** | ResNet-18 (PyTorch, CPU) | Fixed downstream workload used for all three placement modes |
| **Placement Policy** | Python EMA cost model | Estimates and compares source-side vs destination-side cost per image |
| **Orchestration** | Python `subprocess` | Host-side script drives `docker exec` to apply/change/clear network shaping |

### How Network Conditions Are Controlled

All network conditions are **simulated entirely in software** — no physical network changes needed.

Docker Compose creates a private virtual bridge network (`pipeline_net`) between the two containers. The source container is granted `NET_ADMIN` capability, which allows Linux's built-in `tc` (traffic control) tool to be run **inside the container** to shape its own outgoing traffic on `eth0`:

```bash
# Applied inside the source container
# Example: 500 Kbps cap + 20ms added delay
tc qdisc add dev eth0 root handle 1: netem delay 20ms 0ms
tc qdisc add dev eth0 parent 1:1 handle 10: tbf rate 500000bps burst 32kb latency 50ms
```

Every byte the source sends toward the destination passes through these emulated conditions. The experiment runner applies settings automatically via `docker exec` before each run and clears them after. This is the same approach used in networking research (e.g. Netflix, Google) for controlled experiments.

---

## Setup (One-Time)

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and **running**
- Python 3.10+ on the host (use `py` on Windows if `python` is not found)
- A free [Docker Hub account](https://hub.docker.com) — needed to pull the base image

### Step 1 — Log in to Docker Hub

```powershell
docker login
```

Follow the browser prompt if the web-based flow appears. Only needed once per machine.

### Step 2 — Build and Start the Containers

```powershell
docker compose up --build -d
```

**What this does:**
- Builds two Docker images from the `docker/source` and `docker/destination` Dockerfiles
- Source image installs: FastAPI, Uvicorn, PyTorch (CPU), Torchvision, Pillow, `iproute2` (for `tc`)
- Destination image installs the same Python stack; ResNet-18 weights are downloaded on first start
- Creates the `pipeline_net` bridge network (subnet `172.28.0.0/24`) linking the two containers
- Starts both containers in the background (`-d`)

First build takes ~5 minutes. Subsequent builds reuse cached layers and take seconds.

### Step 3 — Check Health

```powershell
Invoke-WebRequest -UseBasicParsing http://localhost:8000/health
Invoke-WebRequest -UseBasicParsing http://localhost:8001/health
```

Both should return `{"status":"ok","service":"source"}` and `{"status":"ok","service":"destination"}`.

> **Note:** On Windows, `curl` is an alias for `Invoke-WebRequest`. Use the `-UseBasicParsing` flag to avoid the HTML execution security prompt, or just press **Y** when it appears.

### Step 4 — Download Test Images

```powershell
py scripts/download_coco_sample.py --n 100 --out-dir data/images
```

**What this does:**
- On first run, downloads the COCO val2017 annotation zip (~25 MB) and caches it to `data/.cache/`
- Reads the real list of ~5,000 val2017 image filenames from the annotation JSON inside the zip
- Randomly samples 100 filenames (seed=42) and downloads them from the COCO CDN
- Already-downloaded images are skipped on future runs

> **Why not just guess IDs?** The original script guessed sequential numeric IDs — most don't exist in COCO val2017 (it's a sparse set), causing ~70% 404 errors. The current script reads the official annotation list, guaranteeing all downloads succeed.

### Step 5 — Validate Output Equivalence

```powershell
py scripts/validate_equivalence.py --images-dir data/images --n 20
```

**What this does:**
- Sends 20 images through both preprocessing paths (source-side and destination-side)
- Checks that both paths produce identical top-1 and top-5 ResNet-18 class predictions
- Ensures the experiment compares *placement strategy*, not *different processing quality*

**Expected output:** `20/20 passed — All outputs equivalent ✓`

Only needs to be run once after initial setup.

---

## Running Experiments

### First Baseline Run

```powershell
py scripts/run_experiment.py --mode all --bw 1000 --n 50
```

**What this does:**
- Calls `docker exec source bash /scripts/shape_network.sh set 1000 0 0` to cap bandwidth to 1 Mbps
- Resets the adaptive policy's EMA state before each mode (via `POST /reset`)
- Sends 50 images through each of `source`, `destination`, `adaptive` in random order
- Records per-image timing breakdown, bytes transferred, and placement decisions to `results/results.csv`
- Clears the `tc` shaping when done (even if the script is interrupted)

### Experiment CLI Options

| Flag | Default | Description |
|---|---|---|
| `--mode` | `all` | `source` / `destination` / `adaptive` / `all` |
| `--bw` | `1000` | Bandwidth cap in Kbps |
| `--delay` | `0` | Added one-way latency in ms |
| `--n` | `50` | Images per mode per run |
| `--no-shape` | off | Skip `tc` shaping (quick testing without NET_ADMIN) |
| `--seed` | `42` | Random seed for image shuffle order |

### Bandwidth Sweep

```powershell
py scripts/run_experiment.py --mode all --bw 5000 --n 50
py scripts/run_experiment.py --mode all --bw 2000 --n 50
py scripts/run_experiment.py --mode all --bw 1000 --n 50
py scripts/run_experiment.py --mode all --bw 500  --n 50
py scripts/run_experiment.py --mode all --bw 256  --n 50
```

### Bandwidth Transition Experiments (Week 6)

```powershell
# High → Low: does adaptive switch away from destination?
py scripts/run_experiment.py --mode adaptive --bw 5000 --n 25
py scripts/run_experiment.py --mode adaptive --bw 256  --n 25

# Low → High: does adaptive switch back to destination?
py scripts/run_experiment.py --mode adaptive --bw 256  --n 25
py scripts/run_experiment.py --mode adaptive --bw 5000 --n 25
```

### Manual Network Shaping

```powershell
# 500 Kbps, 20ms delay, ±5ms jitter
docker exec source bash /scripts/shape_network.sh set 500 20 5

# Remove all shaping
docker exec source bash /scripts/shape_network.sh clear

# Inspect current tc qdiscs
docker exec source bash /scripts/shape_network.sh show
```

### Analyze Results and Generate Plots

```powershell
py scripts/analyze_results.py --input results/results.csv --plots
```

Plots are saved to `results/plots/`.

### Stop Everything

```powershell
docker compose down        # stop and remove containers, keep model cache volume
docker compose down -v     # also delete model weight cache (forces re-download next time)
```

---

## Placement Policy

Defined in [`shared/placement_policy.py`](shared/placement_policy.py), mounted read-only into the source container at `/shared`.

### Cost Model

| Path | Estimated Cost |
|------|---------------|
| Source-side | `t_preprocess_src + (preprocessed_bytes / bandwidth)` |
| Dest-side | `(raw_bytes / bandwidth) + t_preprocess_dst` |

All three quantities — `bandwidth`, `t_preprocess_src`, `t_preprocess_dst` — are updated via **Exponential Moving Average (EMA)** after each image. This lets the policy react as network conditions change mid-run.

A **switching margin** (default 5%) prevents oscillation: the alternative path must be at least 5% cheaper than the current one before the policy switches.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SWITCHING_MARGIN` | `0.05` | Fraction cheaper the alternative must be before switching |
| `EMA_ALPHA` | `0.3` | EMA smoothing factor (higher = reacts faster to changes) |

### Policy Introspection

```powershell
# Check current EMA estimates and decision count live
Invoke-WebRequest -UseBasicParsing http://localhost:8000/stats
```

---

## Results CSV Schema

| Column | Description |
|---|---|
| `run_id` | UTC timestamp of the experiment run |
| `mode` | `source` / `destination` / `adaptive` |
| `bw_kbps` | Applied bandwidth limit |
| `delay_ms` | Applied added latency |
| `decision` | Actual placement used (`source` or `destination`) |
| `raw_size_bytes` | Original JPEG file size |
| `bytes_transferred` | Bytes actually sent over the network |
| `src_preprocess_time_s` | Local preprocessing duration |
| `transfer_time_s` | Round-trip HTTP transfer duration |
| `total_time_s` | End-to-end wall-clock time |
| `dst_preprocess_time_s` | Destination preprocessing time (raw mode only) |
| `dst_infer_time_s` | ResNet-18 inference time at destination |

---

## Progress and Findings

### ✅ Week 1 — Runnable Baseline System

**What we built:**
- Two containerized FastAPI services (source port 8000, destination port 8001)
- Fixed source-side and destination-side preprocessing paths using identical Torchvision transforms
- ResNet-18 CPU inference at the destination as the fixed downstream workload
- Docker Compose network (`pipeline_net`) with `NET_ADMIN` capability for traffic shaping
- COCO val2017 image downloader (fixed from sequential ID guessing to using the real annotation list)
- Output equivalence validator confirming both paths produce identical predictions

**Status:** ✅ Complete — 100/100 COCO images downloaded. 20/20 equivalence checks pass. Both containers healthy.

---

### ✅ Week 2 — Feasibility Validation

**What we ran:**
```
py scripts/validate_equivalence.py --images-dir data/images --n 20
py scripts/run_experiment.py --mode all --bw 1000 --n 50
```

**Results at 1000 Kbps, 0ms delay:**

| Mode | Mean Latency | Transferred | Decisions |
|---|---|---|---|
| `destination` (fixed) | **26.5 ms** | 160 KB/item | destination: 50 |
| `source` (fixed) | 56.1 ms | 589 KB/item | source: 50 |
| `adaptive` | 110.2 ms | 160 KB/item | destination: 50, source: 0 |

**Key findings:**

1. **Preprocessing inflates data, not shrinks it.** A raw JPEG averages ~160 KB. After Torchvision preprocessing (Resize → CenterCrop → ToTensor → Normalize), it becomes a float32 tensor of ~589 KB. At 1 Mbps, sending the small JPEG is far cheaper than sending the tensor — so destination-side preprocessing dominates.

2. **Adaptive correctly chose destination all 50 times.** The EMA policy identified the right strategy. However, it showed higher mean latency (110 ms vs 26 ms for fixed-destination) because in adaptive mode the source always runs a *trial preprocessing* first to measure local compute time before making a decision — adding overhead on every image.

3. **The crossover must exist at lower bandwidth.** At very low bandwidth (e.g. 256 Kbps), even the smaller 160 KB JPEG takes ~5 seconds to transfer, which may make source-side preprocessing competitive despite the larger tensor. This will be confirmed in Week 5.

**Status:** ✅ Complete — feasibility confirmed, policy is functioning correctly.

---

### 🔲 Week 3 — Cost Profiling and Policy Tuning

**Goal:** Profile preprocessing and transfer costs in detail. Validate that the EMA estimates match observed latencies. Tune `EMA_ALPHA` and `SWITCHING_MARGIN` if needed.

```powershell
# Inspect live policy state
Invoke-WebRequest -UseBasicParsing http://localhost:8000/stats

# Run adaptive at low bandwidth to stress the cost model
py scripts/run_experiment.py --mode adaptive --bw 256 --n 50
py scripts/run_experiment.py --mode adaptive --bw 256 --delay 50 --n 50
```

---

### 🔲 Week 4 — Pilot Workloads and Frozen Protocol

**Goal:** Run the full bandwidth sweep, identify the crossover point between source-side and destination-side, and finalize the experimental conditions before the main data collection phase.

```powershell
py scripts/run_experiment.py --mode all --bw 5000 --n 50
py scripts/run_experiment.py --mode all --bw 2000 --n 50
py scripts/run_experiment.py --mode all --bw 500  --n 50
py scripts/run_experiment.py --mode all --bw 256  --n 50

py scripts/analyze_results.py --input results/results.csv --plots
```

---

### 🔲 Week 5 — Stable Bandwidth Experiments

**Goal:** Run repeated experiments under stable bandwidth conditions. Randomize image and mode order to reduce order bias. Report mean, min, max, and variability per condition.

```powershell
# 3 repeated runs per bandwidth with different image orderings
py scripts/run_experiment.py --mode all --bw 256  --n 50 --seed 1
py scripts/run_experiment.py --mode all --bw 256  --n 50 --seed 2
py scripts/run_experiment.py --mode all --bw 256  --n 50 --seed 3

py scripts/run_experiment.py --mode all --bw 1000 --n 50 --seed 1
py scripts/run_experiment.py --mode all --bw 1000 --n 50 --seed 2
py scripts/run_experiment.py --mode all --bw 1000 --n 50 --seed 3

# Repeat for 500, 2000, 5000 Kbps
```

---

### 🔲 Week 6 — Changing Bandwidth Experiments

**Goal:** Test how quickly the adaptive policy responds to bandwidth transitions (high→low and low→high). Measure the number of images processed before the policy switches strategy.

```powershell
# High → Low transition
py scripts/run_experiment.py --mode adaptive --bw 5000 --n 25
py scripts/run_experiment.py --mode adaptive --bw 256  --n 25

# Low → High transition
py scripts/run_experiment.py --mode adaptive --bw 256  --n 25
py scripts/run_experiment.py --mode adaptive --bw 5000 --n 25

py scripts/analyze_results.py --input results/results.csv --plots
```

---

### 🔲 Week 7 — Analysis and Unexpected Results

**Goal:** Generate final comparison plots and tables. Investigate cases where adaptive underperforms. Determine when adaptive placement provides a meaningful advantage vs. when a fixed strategy is sufficient.

---

### 🔲 Week 8 — Final Report and Presentation (Due December 15, 2026)

**Deliverables:**
- Final written report with full results, analysis, and conclusions
- This README as the reproducibility guide
- Live demonstration of the running pipeline
- Presentation slides
