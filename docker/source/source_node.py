"""
source_node.py
--------------
Source-side FastAPI service.

Endpoints
---------
POST /process
    Accepts a JPEG/PNG image upload.
    Query param ?mode=adaptive|source|destination controls placement.
    Returns timing breakdown and inference result from destination.

GET /health     Liveness probe.
GET /stats      Current policy statistics.
POST /reset     Reset policy EMA state (between experiment phases).
"""

from __future__ import annotations

import io
import logging
import os
import time

import requests
import torch
import torchvision.transforms as T
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

# Shared policy (mounted via Docker volume at /shared)
import sys
sys.path.insert(0, "/shared")
from placement_policy import PlacementPolicy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEST_URL = os.getenv("DEST_URL", "http://destination:8001")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [SOURCE] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Preprocessing transform (must match destination exactly)
# ---------------------------------------------------------------------------
TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# App and policy
# ---------------------------------------------------------------------------
app = FastAPI(title="Source Node", version="1.0.0")
policy = PlacementPolicy(
    switching_margin=float(os.getenv("SWITCHING_MARGIN", "0.05")),
    ema_alpha=float(os.getenv("EMA_ALPHA", "0.3")),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def preprocess(img: Image.Image) -> torch.Tensor:
    return TRANSFORM(img)


def tensor_to_bytes(t: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(t, buf)
    return buf.getvalue()


def _post_to_dest(path: str, data: bytes, headers: dict) -> dict:
    """Send payload to destination and return parsed JSON."""
    t0 = time.perf_counter()
    resp = requests.post(
        f"{DEST_URL}{path}",
        data=data,
        headers=headers,
        timeout=60,
    )
    elapsed = time.perf_counter() - t0
    if resp.status_code != 200:
        logger.error("Destination returned %d: %s", resp.status_code, resp.text[:200])
        raise HTTPException(status_code=502, detail="Destination error")
    return resp.json(), elapsed


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/process")
async def process_image(
    file: UploadFile = File(...),
    mode: str = Query("adaptive", description="adaptive | source | destination"),
):
    """
    Process one image through the pipeline.

    mode=adaptive  : policy decides where to preprocess
    mode=source    : always preprocess at source (baseline A)
    mode=destination : always preprocess at destination (baseline B)
    """
    t_start = time.perf_counter()

    # --- Load image ---
    raw_bytes = await file.read()
    try:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    raw_size = len(raw_bytes)
    src_preprocess_time = 0.0
    decision = mode

    # --- Adaptive decision ---
    if mode == "adaptive":
        # Do a trial preprocess to measure local time and estimate payload size
        t_p = time.perf_counter()
        tensor = preprocess(img)
        src_preprocess_time = time.perf_counter() - t_p
        policy.update_src_preprocess_time(src_preprocess_time)

        preprocessed_bytes = tensor_to_bytes(tensor)
        preprocessed_size = len(preprocessed_bytes)

        estimate = policy.decide(
            raw_size_bytes=raw_size,
            preprocessed_size_bytes=preprocessed_size,
        )
        decision = estimate.decision
        policy_meta = {
            "src_cost_s": estimate.source_cost_s,
            "dst_cost_s": estimate.dest_cost_s,
            "bw_kbps": estimate.bandwidth_bps / 1024,
        }
    else:
        preprocessed_bytes = None
        policy_meta = {}

    # --- Execute chosen path ---
    if decision == "source":
        # Preprocess locally if not already done
        if preprocessed_bytes is None:
            t_p = time.perf_counter()
            tensor = preprocess(img)
            src_preprocess_time = time.perf_counter() - t_p
            preprocessed_bytes = tensor_to_bytes(tensor)

        payload = preprocessed_bytes
        t_tx_start = time.perf_counter()
        dest_result, transfer_time = _post_to_dest(
            "/infer",
            payload,
            {"X-Mode": "preprocessed", "Content-Type": "application/octet-stream"},
        )
        bytes_transferred = len(payload)

    else:  # destination
        payload = raw_bytes
        t_tx_start = time.perf_counter()
        dest_result, transfer_time = _post_to_dest(
            "/infer",
            payload,
            {"X-Mode": "raw", "Content-Type": "application/octet-stream"},
        )
        bytes_transferred = raw_size

        # Update dst preprocess time from destination's report
        if "preprocess_time_s" in dest_result:
            policy.update_dst_preprocess_time(dest_result["preprocess_time_s"])

    # Update bandwidth estimate
    policy.update_bandwidth(bytes_transferred, transfer_time)

    total_time = time.perf_counter() - t_start

    return JSONResponse({
        "decision": decision,
        "mode_requested": mode,
        "raw_size_bytes": raw_size,
        "bytes_transferred": bytes_transferred,
        "src_preprocess_time_s": src_preprocess_time,
        "transfer_time_s": transfer_time,
        "total_time_s": total_time,
        "policy": policy_meta,
        "destination": dest_result,
    })


@app.get("/health")
def health():
    return {"status": "ok", "service": "source"}


@app.get("/stats")
def stats():
    return policy.stats()


@app.post("/reset")
def reset_policy():
    policy.reset()
    return {"status": "reset"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level=LOG_LEVEL.lower())
