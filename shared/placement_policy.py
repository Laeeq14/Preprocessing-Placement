"""
placement_policy.py
-------------------
Adaptive preprocessing placement policy.

Decision logic:
  cost_source = t_preprocess_src + (preprocessed_bytes / bandwidth)
  cost_dest   = (raw_bytes / bandwidth) + t_preprocess_dst

The policy tracks bandwidth and per-side processing times using an
Exponential Moving Average (EMA) over recent observations.
A switching margin prevents rapid oscillation when estimates are close.
"""

from __future__ import annotations

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
    dst_preprocess_s: float    # dst processing time used
    decision: str              # "source" or "destination"


class PlacementPolicy:
    """
    Lightweight cost-based placement policy.

    Parameters
    ----------
    switching_margin : float
        Fraction by which the alternative path must beat the current one
        before switching (e.g. 0.05 -> 5 %).  Prevents oscillation.
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
        switching_margin: float = 0.05,
        ema_alpha: float = 0.3,
        bandwidth_window: int = 10,
        src_preprocess_default_s: float = 0.05,
        dst_preprocess_default_s: float = 0.03,
    ) -> None:
        self.switching_margin = switching_margin
        self.ema_alpha = ema_alpha
        self.src_preprocess_default_s = src_preprocess_default_s
        self.dst_preprocess_default_s = dst_preprocess_default_s

        self._lock = threading.Lock()

        # Bandwidth EMA (bytes/s)
        self._bandwidth_ema: Optional[float] = None
        self._transfer_history: deque = deque(maxlen=bandwidth_window)

        # Processing time EMAs (seconds)
        self._src_preprocess_ema: Optional[float] = None
        self._dst_preprocess_ema: Optional[float] = None

        # Hysteresis: track last decision to apply switching margin
        self._last_decision: str = "source"

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

    def update_dst_preprocess_time(self, elapsed_s: float) -> None:
        """Record a measured destination-side preprocessing duration."""
        if elapsed_s <= 0:
            return
        with self._lock:
            self._dst_preprocess_ema = self._ema_update(self._dst_preprocess_ema, elapsed_s)

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def decide(
        self,
        raw_size_bytes: int,
        preprocessed_size_bytes: int,
    ) -> CostEstimate:
        """
        Choose preprocessing placement for the current item.

        Parameters
        ----------
        raw_size_bytes : int
            Size of the original (unprocessed) image payload.
        preprocessed_size_bytes : int
            Size of the preprocessed tensor payload.

        Returns
        -------
        CostEstimate with .decision == "source" or "destination".
        """
        with self._lock:
            bw = self._bandwidth_ema or 1_000_000  # 1 MB/s fallback
            src_t = self._src_preprocess_ema or self.src_preprocess_default_s
            dst_t = self._dst_preprocess_ema or self.dst_preprocess_default_s
            last = self._last_decision

        cost_source = src_t + (preprocessed_size_bytes / bw)
        cost_dest = (raw_size_bytes / bw) + dst_t

        # Apply switching margin to the *current* path to dampen oscillation
        effective_cost_source = cost_source
        effective_cost_dest = cost_dest
        if last == "source":
            effective_cost_dest *= (1.0 - self.switching_margin)
        else:
            effective_cost_source *= (1.0 - self.switching_margin)

        decision = "source" if effective_cost_source <= effective_cost_dest else "destination"

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
            "Decision: %-12s | src=%.4fs  dst=%.4fs | bw=%.1f KB/s",
            decision.upper(),
            cost_source,
            cost_dest,
            bw / 1024,
        )
        return estimate

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        with self._lock:
            return {
                "decisions_made": self.decisions_made,
                "switches": self.switches,
                "bandwidth_ema_kbps": (self._bandwidth_ema or 0) / 1024,
                "src_preprocess_ema_ms": (self._src_preprocess_ema or 0) * 1000,
                "dst_preprocess_ema_ms": (self._dst_preprocess_ema or 0) * 1000,
                "last_decision": self._last_decision,
            }

    def reset(self) -> None:
        """Reset all EMA state (useful between experiment phases)."""
        with self._lock:
            self._bandwidth_ema = None
            self._src_preprocess_ema = None
            self._dst_preprocess_ema = None
            self._last_decision = "source"
            self.decisions_made = 0
            self.switches = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ema_update(self, current: Optional[float], new_value: float) -> float:
        if current is None:
            return new_value
        return self.ema_alpha * new_value + (1.0 - self.ema_alpha) * current
