"""
Adaptive Threshold Updater — Layer 4
======================================
Implements the online calibration mechanism (Pseudocode 4 in the paper).

Every `update_interval` frames, if the controller is in a calibrated state
and the local environment CS drifts from the base thresholds by more than
a hysteresis gap γ, the thresholds are shifted toward the local CS mean
by at most δ_θ per calibration step.

This allows the system to self-adapt to deployment environments that differ
from the training distribution (e.g. Arctic clear water vs tropical turbid
coastal water), without requiring manual re-tuning.
"""

from dataclasses import dataclass
from collections import deque
import numpy as np


@dataclass
class ThresholdConfig:
    # Base (data-driven) thresholds from training set distribution
    tau_down_base: float = 0.40     # 25th percentile of CS on train split
    tau_up_base:   float = 0.55     # 65th percentile of CS on train split

    # Online adaptation limits
    tau_down_min:  float = 0.25     # never go below this
    tau_down_max:  float = 0.50
    tau_up_min:    float = 0.45
    tau_up_max:    float = 0.70

    # Adaptation step and gate
    delta_theta:   float = 0.15     # max shift per calibration event
    gamma:         float = 0.02     # hysteresis gap that triggers shift

    # Calibration schedule
    update_interval: int = 100      # frames between calibration checks
    env_window:      int = 100      # frames used to estimate local CS mean


class AdaptiveThresholdUpdater:
    """
    Online threshold adaptation (Layer 4 of the controller).

    Usage
    -----
    atu = AdaptiveThresholdUpdater()
    ...
    for fi, cs in enumerate(cs_stream):
        tau_down, tau_up = atu.update(cs, fi)
    """

    def __init__(self, cfg: ThresholdConfig | None = None) -> None:
        self.cfg       = cfg or ThresholdConfig()
        self.tau_down  = self.cfg.tau_down_base
        self.tau_up    = self.cfg.tau_up_base
        self._cs_buf   = deque(maxlen=self.cfg.env_window)
        self._calibrated = False
        self._n_updates  = 0
        self._history: list[dict] = []

    # ── Main API ────────────────────────────────────────────────────────────────

    def update(self, cs_smooth: float, frame_idx: int) -> tuple[float, float]:
        """
        Ingest one CS value.  Returns (tau_down, tau_up) — possibly updated.

        Parameters
        ----------
        cs_smooth  : current smoothed Complexity Score
        frame_idx  : current frame index (0-based)

        Returns
        -------
        tau_down, tau_up : current thresholds
        """
        self._cs_buf.append(cs_smooth)

        # Mark calibrated after first full window
        if not self._calibrated and len(self._cs_buf) >= self.cfg.env_window:
            self._calibrated = True

        # Trigger calibration every `update_interval` frames
        if (
            self._calibrated
            and (frame_idx + 1) % self.cfg.update_interval == 0
        ):
            self._calibrate(frame_idx)

        return self.tau_down, self.tau_up

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def n_updates(self) -> int:
        return self._n_updates

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    def reset(self) -> None:
        """Reset to base thresholds (e.g. for a new mission segment)."""
        self.tau_down = self.cfg.tau_down_base
        self.tau_up   = self.cfg.tau_up_base
        self._cs_buf.clear()
        self._calibrated = False
        self._n_updates  = 0

    # ── Internal ────────────────────────────────────────────────────────────────

    def _calibrate(self, frame_idx: int) -> None:
        """
        Shift thresholds toward the local environment CS mean when the
        drift exceeds the hysteresis gap γ.
        """
        cfg    = self.cfg
        cs_env = float(np.mean(self._cs_buf))

        old_down, old_up = self.tau_down, self.tau_up
        changed = False

        # ── Downgrade threshold calibration ─────────────────────────────────
        if abs(cs_env - self.tau_down) > cfg.gamma:
            shift = np.clip(
                cfg.delta_theta * np.sign(cs_env - self.tau_down),
                -cfg.delta_theta, cfg.delta_theta,
            )
            new_down = float(np.clip(
                self.tau_down + shift * 0.5,   # half-step per update
                cfg.tau_down_min, cfg.tau_down_max,
            ))
            if abs(new_down - self.tau_down) > 1e-4:
                self.tau_down = new_down
                changed = True

        # ── Upgrade threshold calibration ────────────────────────────────────
        if abs(cs_env - self.tau_up) > cfg.gamma:
            shift = np.clip(
                cfg.delta_theta * np.sign(cs_env - self.tau_up),
                -cfg.delta_theta, cfg.delta_theta,
            )
            new_up = float(np.clip(
                self.tau_up + shift * 0.5,
                cfg.tau_up_min, cfg.tau_up_max,
            ))
            if abs(new_up - self.tau_up) > 1e-4:
                self.tau_up = new_up
                changed = True

        # Always maintain the hysteresis gap
        if self.tau_up - self.tau_down < 0.10:
            mid = (self.tau_up + self.tau_down) / 2.0
            self.tau_down = float(np.clip(mid - 0.07, cfg.tau_down_min, cfg.tau_down_max))
            self.tau_up   = float(np.clip(mid + 0.07, cfg.tau_up_min,   cfg.tau_up_max))

        if changed:
            self._n_updates += 1
            self._history.append({
                "frame":    frame_idx,
                "cs_env":   round(cs_env, 4),
                "tau_down": round(self.tau_down, 4),
                "tau_up":   round(self.tau_up,   4),
                "delta":    round(self.tau_down - old_down, 5),
            })
