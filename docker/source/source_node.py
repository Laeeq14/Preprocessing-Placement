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

# Pin PyTorch thread count to reduce thread contention
torch.set_num_threads(1)  # Phase B: pinned to 1; note Phase A used 2 (not directly comparable)

# Shared policy (mounted via Docker volume at /shared)
import sys
sys.path.insert(0, "/shared")
from placement_policy import PlacementPolicy
from phase_b_transforms import run_source_path as thumbnail_encode  # Phase B thumbnail pipeline

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

# Actual byte size of a torch.save'd 1×3×224×224 float32 tensor.
# This is a fixed constant independent of the input image; pre-compute it once
# to avoid the raw_size * 4 approximation that was wrong for two reasons:
#   (a) raw_size varies per image; the tensor shape does not.
#   (b) torch.save adds a pickle header (~100 B), making the true size ~589 KiB.
_DUMMY = torch.zeros(1, 3, 224, 224)
_buf = io.BytesIO()
torch.save(_DUMMY, _buf)
TENSOR_BYTES: int = _buf.tell()    # bytes; used by cost model for source-path payload
del _DUMMY, _buf

# ---------------------------------------------------------------------------
# App and policy
# ---------------------------------------------------------------------------
app = FastAPI(title="Source Node", version="1.0.0")
policy = PlacementPolicy(
    switching_margin=float(os.getenv("SWITCHING_MARGIN", "0.05")),
    ema_alpha=float(os.getenv("EMA_ALPHA", "0.3")),
)
_policy_warmed = False
_WARMUP_IMAGES = int(os.getenv("WARMUP_IMAGES", "5"))  # local preprocess runs at startup


def _warm_policy_once() -> None:
    """
    Seed the EMA with measurements from the destination /profile endpoint
    (dst preprocess + inference averages) and a few local trial preprocesses
    on a small synthetic image.  Runs at most once per process lifetime.
    """
    global _policy_warmed
    if _policy_warmed:
        return
    _policy_warmed = True

    # Seed dst preprocess + inference times from destination profile
    try:
        resp = requests.get(f"{DEST_URL}/profile", timeout=5)
        if resp.ok:
            data = resp.json()
            prep_ms = (data.get("preprocess") or {}).get("mean_ms") or None
            infer_ms = (data.get("inference") or {}).get("mean_ms") or None
            if prep_ms:
                policy.update_dst_preprocess_time(prep_ms / 1000.0)
                logger.info("[warmup] dst preprocess from /profile: %.2f ms", prep_ms)
            if infer_ms:
                logger.info("[warmup] dst inference from /profile: %.2f ms", infer_ms)
    except Exception as exc:
        logger.warning("[warmup] Could not reach destination /profile: %s", exc)

    # Seed src preprocess time from local trials on a synthetic JPEG image.
    # Must include JPEG decode in the timer because production now times the full
    # path: Image.open(JPEG) + convert('RGB') + TRANSFORM.
    import numpy as np
    from PIL import Image as PILImage
    _syn_raw = PILImage.fromarray(
        np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    )
    _syn_buf = io.BytesIO()
    _syn_raw.save(_syn_buf, format="JPEG", quality=85)
    _syn_bytes = _syn_buf.getvalue()
    times = []
    for _ in range(_WARMUP_IMAGES):
        t0 = time.perf_counter()
        _img = PILImage.open(io.BytesIO(_syn_bytes)).convert("RGB")
        TRANSFORM(_img)
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    policy.update_src_preprocess_time(avg)
    logger.info("[warmup] src preprocess (decode+transform) from %d trials: %.2f ms", _WARMUP_IMAGES, avg * 1000)



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
    src_decode_time = 0.0
    decision = mode

    # Warm the EMA once from /profile + local timing trials (no per-image probe)
    _warm_policy_once()
    # Use the actual torch.save tensor byte count (fixed shape, invariant to input).
    # The old raw_size * 4 approximation was image-dependent and ~17% off on average.
    preprocessed_size = TENSOR_BYTES

    # Compute cost model predictions for both paths
    cost_src, cost_dst, bw_est = policy.estimate(raw_size, preprocessed_size)

    # --- Adaptive decision / Mode assignment ---
    if mode == "adaptive":
        estimate = policy.decide(
            raw_size_bytes=raw_size,
            preprocessed_size_bytes=preprocessed_size,
        )
        decision = estimate.decision
        policy_meta = {
            "src_cost_s": estimate.source_cost_s,
            "dst_cost_s": estimate.dest_cost_s,
            # bandwidth_bps is in bytes/s; divide by 125 to convert to Kbps
            # (1 Kbps = 1000 bits/s = 125 bytes/s).
            "bw_kbps": estimate.bandwidth_bps / 125,
            "pred_decision": estimate.decision,
        }
    else:
        decision = mode
        policy_meta = {
            "src_cost_s": cost_src,
            "dst_cost_s": cost_dst,
            "bw_kbps": bw_est / 125,
            "pred_decision": "source" if cost_src <= cost_dst else "destination",
        }

    # --- Execute chosen path ---
    if decision == "source_thumbnail":
        # Phase B: decode full image, resize, re-encode as JPEG Q85 thumbnail (~20 KB),
        # then send the thumbnail to destination with X-Mode: raw.
        # Destination decodes the thumbnail and applies its standard TRANSFORM
        # (Resize256 → CenterCrop224 → ToTensor → Normalize) — the resize is a near-no-op
        # because the thumbnail is already sized to ~256px on the short side.
        t_p = time.perf_counter()
        thumb_bytes = thumbnail_encode(raw_bytes)  # decode + resize256 + JPEG-Q85 encode
        src_preprocess_time = time.perf_counter() - t_p
        src_decode_time = src_preprocess_time  # full time attributed to source

        payload = thumb_bytes
        t_tx_start = time.perf_counter()
        dest_result, transfer_time = _post_to_dest(
            "/infer",
            payload,
            {"X-Mode": "raw", "Content-Type": "application/octet-stream"},
        )
        bytes_transferred = len(payload)

        if "preprocess_time_s" in dest_result:
            policy.update_dst_preprocess_time(dest_result["preprocess_time_s"])

    elif decision == "source":
        # Preprocess locally — time the FULL path: decode + transform.
        # Previously the timer started after Image.open, missing ~4 ms of decode.
        t_p = time.perf_counter()
        img_decoded = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        src_decode_time = time.perf_counter() - t_p
        tensor = preprocess(img_decoded)
        src_preprocess_time = time.perf_counter() - t_p   # decode + transform
        policy.update_src_preprocess_time(src_preprocess_time)
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

    total_time = time.perf_counter() - t_start

    # Derive wire-only transfer time: subtract destination compute from round-trip
    dst_preprocess = dest_result.get("preprocess_time_s", 0.0) or 0.0
    dst_infer      = dest_result.get("infer_time_s", 0.0) or 0.0
    raw_wire_time  = transfer_time - dst_preprocess - dst_infer
    if raw_wire_time <= 0:
        logger.warning(
            "Non-positive raw wire time: %.4fs (transfer=%.4f, dst_prep=%.4f, dst_infer=%.4f)",
            raw_wire_time, transfer_time, dst_preprocess, dst_infer,
        )
    # Report the raw derived wire time in the response (may be near-zero on loopback).
    wire_time = max(0.0, raw_wire_time)

    # Update bandwidth estimate only when the wire time is plausible (> 1 ms).
    # Clamping to a small floor (the old 0.0005 s) would inject fake ~GB/s readings
    # into the EMA whenever destination compute exceeds measured round-trip time.
    if raw_wire_time > 0.001:
        policy.update_bandwidth(bytes_transferred, raw_wire_time)
    else:
        logger.debug(
            "Skipping BW update: wire_time=%.4fs too small to be reliable",
            raw_wire_time,
        )

    return JSONResponse({
        "decision": decision,
        "mode_requested": mode,
        "raw_size_bytes": raw_size,
        "bytes_transferred": bytes_transferred,
        # Full source preprocess time: decode + resize + crop + ToTensor + Normalize
        "src_preprocess_time_s": src_preprocess_time,
        # Decode-only component (Image.open + convert); zero when path = destination
        "src_decode_time_s": src_decode_time,
        "transfer_time_s": transfer_time,
        "wire_time_s": wire_time,
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
