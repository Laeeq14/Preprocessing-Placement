"""
phase_b_transforms.py — canonical Phase B transform split.

Import this module in BOTH measure_timing.py and the service nodes so
the measured pipeline and the live pipeline cannot drift apart.

Split definition
----------------
Source path (runs on source node, throttled):
  1. decode_large   : PIL.Image.open + convert('RGB')  on the full-res input
  2. resize_256     : Resize to 256 px on the short side (torchvision)
  3. encode_thumb   : re-encode as JPEG Q85 → bytes  (~15 KB)
  Wire: thumbnail bytes (B_raw  ≈ 15 KB)

Destination, source-path remainder (unthrottled, on destination node):
  4. decode_thumb   : PIL.Image.open on the thumbnail bytes
  5. center_crop    : CenterCrop(224)
  6. to_tensor      : ToTensor
  7. normalize      : Normalize(ImageNet mean/std)
  Wire output: float32 tensor [3,224,224]  (≈ 588 KB)

Destination, destination path (unthrottled, on destination node):
  1. decode_large   : PIL.Image.open + convert('RGB')  on the full-res input
  2. resize_256     : Resize(256)
  3. center_crop    : CenterCrop(224)
  4. to_tensor      : ToTensor
  5. normalize      : Normalize(ImageNet mean/std)

Cost-model ledger
-----------------
  t_src        = time for steps 1-3 on throttled source
  t_dst_remain = time for steps 4-7 on unthrottled destination
  t_dst_full   = time for steps 1-5 on unthrottled destination

  Δt  = t_src + t_dst_remain − t_dst_full   (positive = source path is slower)
  ΔB  = B_raw − B_tensor                    (positive = thumbnail is smaller than tensor)
        where B_raw ≈ thumbnail bytes, B_tensor = 3*224*224*4 bytes

  BW* [Kbps] = ΔB * 8 / (1000 * Δt)
  Source wins when BW < BW* (thumbnail saves enough wire time to pay for extra compute).
"""

import io
import torch
import torchvision.transforms.functional as TF
from PIL import Image

# ImageNet statistics
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

THUMBNAIL_QUALITY = 85   # JPEG Q used for the thumbnail wire format
RESIZE_SIZE       = 256  # short-side resize
CROP_SIZE         = 224  # final crop / tensor size

# Fixed tensor size for ΔB calculation
TENSOR_BYTES = 3 * CROP_SIZE * CROP_SIZE * 4  # float32


def src_decode(image_bytes: bytes) -> Image.Image:
    """Step 1: decode full-resolution JPEG from raw bytes."""
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def src_resize(img: Image.Image) -> Image.Image:
    """Step 2: resize so the short side is RESIZE_SIZE px."""
    w, h = img.size
    if w <= h:
        new_w = RESIZE_SIZE
        new_h = int(h * RESIZE_SIZE / w)
    else:
        new_h = RESIZE_SIZE
        new_w = int(w * RESIZE_SIZE / h)
    return img.resize((new_w, new_h), Image.BILINEAR)


def src_encode_thumb(img: Image.Image) -> bytes:
    """Step 3: encode resized image as JPEG Q85 thumbnail bytes."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=THUMBNAIL_QUALITY)
    return buf.getvalue()


def dst_decode_thumb(thumb_bytes: bytes) -> Image.Image:
    """Step 4 (source-path remainder): decode thumbnail bytes."""
    return Image.open(io.BytesIO(thumb_bytes)).convert("RGB")


def dst_crop_tensor_normalize(img: Image.Image) -> torch.Tensor:
    """Steps 5-7 (source-path remainder) or steps 3-5 (dest path).
    CenterCrop → ToTensor → Normalize."""
    img = TF.center_crop(img, CROP_SIZE)
    t   = TF.to_tensor(img)                     # [3, H, W] float32 in [0,1]
    t   = TF.normalize(t, _MEAN, _STD)
    return t


# ── Convenience bundles used by measure_timing.py ──────────────────────────

def run_source_path(image_bytes: bytes):
    """Full source-side pipeline. Returns (thumb_bytes, tensor_shape)."""
    img        = src_decode(image_bytes)
    img        = src_resize(img)
    thumb      = src_encode_thumb(img)
    return thumb


def run_dst_remain(thumb_bytes: bytes) -> torch.Tensor:
    """Destination-node work for the source path (steps 4-7)."""
    img = dst_decode_thumb(thumb_bytes)
    return dst_crop_tensor_normalize(img)


def run_dst_full(image_bytes: bytes) -> torch.Tensor:
    """Full destination-side pipeline (steps 1-5)."""
    img = src_decode(image_bytes)            # reuse decode helper
    img = src_resize(img)                    # reuse resize helper
    return dst_crop_tensor_normalize(img)

