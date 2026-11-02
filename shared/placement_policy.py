"""
placement_policy.py
-------------------
Adaptive preprocessing placement policy.

Decision logic (Phase A — Expansive workload):
  cost_source = t_preprocess_src + (preprocessed_bytes / bandwidth)
  cost_dest   = (raw_bytes / bandwidth) + t_preprocess_dst
  Decides: "source" (tensor) vs "destination" (raw)

Decision logic (Phase B — Compressive workload / Thumbnail mode):
  cost_source = t_preprocess_src + (thumb_bytes / bandwidth) + t_dst_remain
  cost_dest   = (raw_bytes / bandwidth) + t_dst_full
  Decides: "source_thumbnail" (JPEG Q85 thumbnail) vs "destination" (full raw JPEG)

The policy tracks bandwidth and per-side processing times using an
Exponential Moving Average (EMA) over recent observations.
A switching margin prevents rapid oscillation when estimates are close.
"""

from __future__ import annotations

import os
import threading
import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CostEstimate:
    source_cost_s: float       # estimated end-to-end cost for source-side path
    dest_cost_s: float         # estimated end-to-end cost for dest-side path
    bandwidth_bps: float       # bandwidth used in estimate (bytes/s)
    src_preprocess_s: float    # src processing time used
    dst_preprocess_s: float    # dst processing time used (t_dst_full in Phase B)
    decision: str              # "source_thumbnail" (or "source") vs "destination"


class PlacementPolicy:
    """
    Lightweight cost-based placement policy.

    Parameters
    ----------
    pipeline : str
        "phase_b" (thumbnail compressive pipeline, default) or
        "phase_a" (tensor expansive pipeline).
    switching_margin : float
        Fraction by which the alternative path must beat the current one
        before switching (e.g. 0.05 -> 5 %). Prevents oscillation.
    ema_alpha : float
        Smoothing factor for all EMA estimates (higher = faster adaptation).
    bandwidth_window : int
        Maximum number of past transfer samples kept (ring buffer).
    src_preprocess_default_s : float
        Fallback source preprocessing time before any measurements.
    dst_preprocess_default_s : float
        Fallback destination preprocessing time before any measurements.
    """

    def __init__(
        self,
        pipeline: Optional[str] = None,
        switching_margin: float = 0.05,
        ema_alpha: float = 0.3,
        bandwidth_window: int = 10,
        src_preprocess_default_s: Optional[float] = None,
        dst_preprocess_default_s: Optional[float] = None,
        dst_remain_default_s: float = 0.0008,
        thumb_bytes_default: int = 20_484,
    ) -> None:
        self.pipeline = pipeline or os.getenv("PIPELINE", "phase_b").lower()
        self.switching_margin = switching_margin
        self.ema_alpha = ema_alpha

        # Default fallbacks depend on pipeline mode
        if self.pipeline == "phase_b":
            self.src_preprocess_default_s = (
                src_preprocess_default_s if src_preprocess_default_s is not None else 0.765
            )
            self.dst_preprocess_default_s = (
                dst_preprocess_default_s if dst_preprocess_default_s is not None else 0.067
            )
        else:
            self.src_preprocess_default_s = (
                src_preprocess_default_s if src_preprocess_default_s is not None else 0.05
            )
            self.dst_preprocess_default_s = (
                dst_preprocess_default_s if dst_preprocess_default_s is not None else 0.03
            )

        self.dst_remain_default_s = dst_remain_default_s
        self.thumb_bytes_default = thumb_bytes_default

        self._lock = threading.Lock()

        # Bandwidth EMA (bytes/s)
        self._bandwidth_ema: Optional[float] = None
        self._transfer_history: deque = deque(maxlen=bandwidth_window)

        # Processing time EMAs (seconds)
        self._src_preprocess_ema: Optional[float] = None
        self._dst_preprocess_ema: Optional[float] = None  # Phase A dst or Phase B full
        self._dst_remain_ema: Optional[float] = None      # Phase B thumbnail remainder
        self._thumb_bytes_ema: Optional[float] = None     # Phase B thumbnail byte size

        # Hysteresis: track last decision to apply switching margin
        self._source_label = "source_thumbnail" if self.pipeline == "phase_b" else "source"
        self._last_decision: str = self._source_label

        # Statistics
        self.decisions_made: int = 0
        self.switches: int = 0

    # ------------------------------------------------------------------
    # Update methods (call after each observation)
    # ------------------------------------------------------------------

    def update_bandwidth(self, bytes_transferred: int, elapsed_s: float) -> None:
        """Record a completed transfer to refine bandwidth estimate."""
        if elapsed_s <= 0 or bytes_transferred <= 0:
            return
        speed_bps = bytes_transferred / elapsed_s
        with self._lock:
            self._transfer_history.append(speed_bps)
            self._bandwidth_ema = self._ema_update(self._bandwidth_ema, speed_bps)
        logger.debug("BW update: %.1f KB/s (EMA)", speed_bps / 1024)

    def update_src_preprocess_time(self, elapsed_s: float) -> None:
        """Record a measured source-side preprocessing duration."""
        if elapsed_s <= 0:
            return
        with self._lock:
            self._src_preprocess_ema = self._ema_update(self._src_preprocess_ema, elapsed_s)

    def update_dst_full_time(self, elapsed_s: float) -> None:
        """Record measured destination-side full preprocessing duration (Phase B t_dst_full)."""
        if elapsed_s <= 0:
            return
        with self._lock:
            self._dst_preprocess_ema = self._ema_update(self._dst_preprocess_ema, elapsed_s)

    def update_dst_remain_time(self, elapsed_s: float) -> None:
        """Record measured destination-side thumbnail remainder duration (Phase B t_dst_remain)."""
        if elapsed_s <= 0:
            return
        with self._lock:
            self._dst_remain_ema = self._ema_update(self._dst_remain_ema, elapsed_s)

    def update_dst_preprocess_time(self, elapsed_s: float) -> None:
        """Record measured destination preprocessing duration (backwards compatible)."""
        self.update_dst_full_time(elapsed_s)

    def update_thumb_bytes(self, size_bytes: int) -> None:
        """Record measured thumbnail payload size."""
        if size_bytes <= 0:
            return
        with self._lock:
            self._thumb_bytes_ema = self._ema_update(self._thumb_bytes_ema, float(size_bytes))

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def estimate(
        self,
        raw_size_bytes: int,
        preprocessed_size_bytes: Optional[int] = None,
    ) -> tuple[float, float, float]:
        """Compute estimated costs (source, dest, bw_bps) without modifying decision state."""
        with self._lock:
            bw = self._bandwidth_ema or 1_000_000
            src_t = self._src_preprocess_ema or self.src_preprocess_default_s
            dst_t = self._dst_preprocess_ema or self.dst_preprocess_default_s

            if self.pipeline == "phase_b":
                dst_remain_t = self._dst_remain_ema or self.dst_remain_default_s
                thumb_bytes = (
                    preprocessed_size_bytes
                    or (int(self._thumb_bytes_ema) if self._thumb_bytes_ema else None)
                    or self.thumb_bytes_default
                )
                cost_source = src_t + (thumb_bytes / bw) + dst_remain_t
                cost_dest = (raw_size_bytes / bw) + dst_t
            else:
                prep_bytes = preprocessed_size_bytes or 589_200
                cost_source = src_t + (prep_bytes / bw)
                cost_dest = (raw_size_bytes / bw) + dst_t

        return cost_source, cost_dest, bw

    def decide(
        self,
        raw_size_bytes: int,
        preprocessed_size_bytes: Optional[int] = None,
    ) -> CostEstimate:
        """
        Choose preprocessing placement for the current item.

        Parameters
        ----------
        raw_size_bytes : int
            Size of the original (unprocessed) image payload.
        preprocessed_size_bytes : int, optional
            Size of the preprocessed payload (thumbnail bytes in Phase B, tensor bytes in Phase A).

        Returns
        -------
        CostEstimate with .decision == "source_thumbnail" (or "source") or "destination".
        """
        cost_source, cost_dest, bw = self.estimate(raw_size_bytes, preprocessed_size_bytes)
        source_label = "source_thumbnail" if self.pipeline == "phase_b" else "source"

        with self._lock:
            src_t = self._src_preprocess_ema or self.src_preprocess_default_s
            dst_t = self._dst_preprocess_ema or self.dst_preprocess_default_s
            last = self._last_decision

        # Apply switching margin to the *current* path to dampen oscillation.
        # Discount the current path so the alternative must beat it by at least
        # switching_margin before a switch occurs.
        effective_cost_source = cost_source
        effective_cost_dest = cost_dest
        if last == source_label:
            effective_cost_source *= (1.0 - self.switching_margin)
        else:
            effective_cost_dest *= (1.0 - self.switching_margin)

        decision = source_label if effective_cost_source <= effective_cost_dest else "destination"

        with self._lock:
            if decision != self._last_decision:
                self.switches += 1
            self._last_decision = decision
            self.decisions_made += 1

        estimate = CostEstimate(
            source_cost_s=cost_source,
            dest_cost_s=cost_dest,
            bandwidth_bps=bw,
            src_preprocess_s=src_t,
            dst_preprocess_s=dst_t,
            decision=decision,
        )
        logger.info(
            "Decision [%s]: %-16s | src=%.4fs  dst=%.4fs | bw=%.1f KB/s (effective src=%.4fs dst=%.4fs)",
            self.pipeline,
            decision.upper(),
            cost_source,
            cost_dest,
            bw / 1024,
            effective_cost_source,
            effective_cost_dest,
        )
        return estimate

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        with self._lock:
            return {
                "pipeline": self.pipeline,
                "decisions_made": self.decisions_made,
                "switches": self.switches,
                # bandwidth_ema is stored in bytes/s; ÷125 → Kbps (1 Kbps = 125 B/s)
                "bandwidth_ema_kbps": (self._bandwidth_ema or 0) / 125,
                "src_preprocess_ema_ms": (self._src_preprocess_ema or 0) * 1000,
                "dst_preprocess_ema_ms": (self._dst_preprocess_ema or 0) * 1000,
                "dst_remain_ema_ms": (self._dst_remain_ema or 0) * 1000,
                "thumb_bytes_ema": self._thumb_bytes_ema or self.thumb_bytes_default,
                "last_decision": self._last_decision,
            }

    def reset(self) -> None:
        """Reset all EMA state (useful between experiment phases)."""
        with self._lock:
            self._bandwidth_ema = None
            self._src_preprocess_ema = None
            self._dst_preprocess_ema = None
            self._dst_remain_ema = None
            self._thumb_bytes_ema = None
            self._last_decision = "source_thumbnail" if self.pipeline == "phase_b" else "source"
            self.decisions_made = 0
            self.switches = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ema_update(self, current: Optional[float], new_value: float) -> float:
        if current is None:
            return new_value
        return self.ema_alpha * new_value + (1.0 - self.ema_alpha) * current
