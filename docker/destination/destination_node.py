"""
destination_node.py
-------------------
Destination-side FastAPI service.

Accepts either:
  - X-Mode: preprocessed  → body is a torch-saved tensor, runs inference directly
  - X-Mode: raw           → body is JPEG/PNG bytes, preprocesses then runs inference

Model: ResNet-18 (ImageNet weights, CPU-only).

Endpoints
---------
POST /infer       Main inference endpoint (called by source node)
GET  /health      Liveness probe
GET  /profile     Return averaged preprocessing/inference timing (for policy warm-up)
"""

from __future__ import annotations

import io
import logging
import os
import statistics
import time
from collections import deque

import torch
import torchvision.models as models
import torchvision.transforms as T
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [DEST ] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Preprocessing transform (must match source exactly)
# ---------------------------------------------------------------------------
TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# App and model
# ---------------------------------------------------------------------------
app = FastAPI(title="Destination Node", version="1.0.0")

logger.info("Loading ResNet-18 (ImageNet1K_V1) on CPU …")
_weights = models.ResNet18_Weights.IMAGENET1K_V1
model = models.resnet18(weights=_weights)
model.eval()
logger.info("Model ready.")

# Rolling timing windows (for /profile)
_preprocess_times: deque = deque(maxlen=50)
_infer_times: deque = deque(maxlen=50)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/infer")
async def infer(request: Request):
    """
    Run inference on one image.

    Headers
    -------
    X-Mode : str
        "preprocessed" — body is a torch-saved float32 tensor (1×3×224×224 after unsqueeze)
        "raw"          — body is raw JPEG/PNG bytes; node will preprocess here
    """
    mode = request.headers.get("X-Mode", "preprocessed").lower()
    body = await request.body()

    if not body:
        raise HTTPException(status_code=400, detail="Empty body")

    preprocess_time_s = 0.0

    if mode == "preprocessed":
        try:
            buf = io.BytesIO(body)
            tensor = torch.load(buf, weights_only=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not deserialize tensor: {exc}")
    elif mode == "raw":
        try:
            img = Image.open(io.BytesIO(body)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")
        t_p = time.perf_counter()
        tensor = TRANSFORM(img)
        preprocess_time_s = time.perf_counter() - t_p
        _preprocess_times.append(preprocess_time_s)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown X-Mode: {mode}")

    # Ensure shape is (1, C, H, W)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    # --- Inference ---
    t_infer = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
    infer_time_s = time.perf_counter() - t_infer
    _infer_times.append(infer_time_s)

    probs = torch.softmax(logits, dim=1)
    top5_vals, top5_idx = torch.topk(probs, 5)

    return JSONResponse({
        "mode": mode,
        "preprocess_time_s": preprocess_time_s,
        "infer_time_s": infer_time_s,
        "top5_class_ids": top5_idx[0].tolist(),
        "top5_probs": [round(p, 6) for p in top5_vals[0].tolist()],
    })


@app.get("/health")
def health():
    return {"status": "ok", "service": "destination"}


@app.get("/profile")
def profile():
    """Return averaged preprocessing/inference timing for policy warm-up."""
    def _stats(data):
        if not data:
            return {"mean_ms": None, "std_ms": None, "n": 0}
        vals = list(data)
        return {
            "mean_ms": round(statistics.mean(vals) * 1000, 2),
            "std_ms": round(statistics.stdev(vals) * 1000, 2) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }

    return {
        "preprocess": _stats(_preprocess_times),
        "inference": _stats(_infer_times),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level=LOG_LEVEL.lower())
