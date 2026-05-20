"""
DeepClean v4 — Full Adaptive Dual-Mode Controller
===================================================
This module assembles all four layers of the controller into a single
coherent class that can be dropped into any inference loop.

Architecture
------------
  Input frame
      │
      ▼
  ParameterExtractor  → 20 normalised params (P1–P20)
      │
      ▼
  ComplexityScoreComputer  → CS_raw, CS̃(t)  [with 13 dynamic weight rules]
      │
      ▼
  AdaptiveThresholdUpdater → τ_down(t), τ_up(t)   [Layer 4, every 100 frames]
      │
      ▼
  3-Layer Switching Controller
    ├── Layer 1: Asymmetric hysteresis          (every frame)
    ├── Layer 2: Time-based feature validation  (in hysteresis zone, ≥50 frames)
    └── Layer 3: Statistical stability          (in hysteresis zone, ≥20 frames)
      │
      ▼
  Energy safety override  (B_t ≤ B_crit → force lightweight)
      │
      ▼
  Model selection: YOLOv8n  or  YOLOv8x (or configured alternatives)

Usage
-----
    from controller.deepclean_v4 import DeepCleanController

    ctrl = DeepCleanController()
    for frame in video_frames:
        result = ctrl.step(frame, detections, model_metrics, mission_state)
        use(result.model_name)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from controller.parameter_extractor    import ParameterExtractor
from controller.complexity_score       import ComplexityScoreComputer
from controller.adaptive_threshold     import AdaptiveThresholdUpdater, ThresholdConfig


# ─── Layer configuration ──────────────────────────────────────────────────────

@dataclass
class ControllerConfig:
    # Model names
    model_lightweight: str = "YOLOv8n"
    model_heavyweight: str = "YOLOv8x"

    # Layer 1
    tau_down:          float = 0.40
    tau_up:            float = 0.55
    min_frames_switch: int   = 30

    # Layer 2
    l2_dwell_frames:       int   = 50
    l2_window:             int   = 20
    l2_consistency_thresh: float = 0.80
    l2_features_needed:    int   = 4
    # Feature thresholds for Layer 2 validation
    l2_feature_thresholds: dict  = field(default_factory=lambda: {
        "p1": 0.50,   # turbidity
        "p3": 0.55,   # texture richness
        "p5": 0.60,   # motion blur
        "p6": 0.55,   # camera stability
        "p8": 0.50,   # object density
    })

    # Layer 3
    l3_window:          int   = 20
    l3_cs_std_thresh:   float = 0.03
    l3_param_std_thresh:float = 0.05
    l3_features_needed: int   = 4

    # EMA smoothing
    smoothing_alpha:   float = 0.30

    # Energy safety
    b_crit:            float = 0.20   # critical battery → force lightweight

    # Layer 4 (adaptive thresholds)
    use_adaptive_thresholds: bool = True

    # Active layers
    active_layers: tuple = (1, 2, 3)


# ─── Step result ─────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    frame_idx:     int
    model_name:    str
    cs_raw:        float
    cs_smooth:     float
    tau_down:      float
    tau_up:        float
    switched:      bool
    trigger_layer: int | None    # which layer caused the switch (None = no switch)
    forced:        bool          # True if energy safety override triggered
    params:        dict[str, float] = field(default_factory=dict)
    weights:       dict[str, float] = field(default_factory=dict)
    controller_ms: float = 0.0   # controller overhead (excluding YOLO inference)


# ─── Main controller ──────────────────────────────────────────────────────────

class DeepCleanController:
    """
    Full 3-Layer Intelligent Switching Controller with optional Layer 4
    adaptive threshold updates.

    Parameters
    ----------
    cfg : ControllerConfig
        Tunable parameters.  Defaults reproduce the paper's configuration.
    active_layers : tuple[int] | None
        Override which layers are active (default: (1, 2, 3)).
        Useful for ablation studies.
    """

    def __init__(
        self,
        cfg:           ControllerConfig | None  = None,
        active_layers: tuple | None             = None,
    ) -> None:
        self.cfg    = cfg or ControllerConfig()
        self.layers = set(
            active_layers
            if active_layers is not None
            else self.cfg.active_layers
        )

        # Sub-components
        self._extractor  = ParameterExtractor()
        self._cs_computer= ComplexityScoreComputer(alpha=self.cfg.smoothing_alpha)
        self._threshold  = AdaptiveThresholdUpdater(
            ThresholdConfig(
                tau_down_base = self.cfg.tau_down,
                tau_up_base   = self.cfg.tau_up,
            )
        ) if self.cfg.use_adaptive_thresholds else None

        # State
        self._current_model:    str          = self.cfg.model_lightweight
        self._frame_idx:        int          = 0
        self._frames_since_sw:  int          = 0
        self._hysteresis_cnt:   int          = 0

        # Buffers for Layers 2 & 3
        self._param_buf = deque(maxlen=max(
            self.cfg.l2_window, self.cfg.l3_window
        ))
        self._cs_buf    = deque(maxlen=self.cfg.l3_window)

        # History
        self.cs_raw_hist:     list[float] = []
        self.cs_smooth_hist:  list[float] = []
        self.model_hist:      list[str]   = []
        self.switch_events:   list[dict]  = []
        self.switch_count:    int         = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def step(
        self,
        frame,
        detections     = None,
        model_metrics: dict | None = None,
        mission_state: dict | None = None,
    ) -> StepResult:
        """
        Process one frame and return the model selection decision.

        Parameters
        ----------
        frame          : BGR numpy array (H × W × 3)
        detections     : ultralytics Results object (or None for vision-only mode)
        model_metrics  : dict — latency_s, dropout_rate, energy_ratio
        mission_state  : dict — battery, mission_phase, trash_priority, …

        Returns
        -------
        StepResult with model_name, cs values, switch flag, etc.
        """
        t0 = time.perf_counter()
        fi = self._frame_idx
        mission_state = mission_state or {}

        # ── 1. Extract parameters ──────────────────────────────────────────────
        params = self._extractor.extract(
            frame, detections, model_metrics, mission_state
        )

        # ── 2. Compute CS ──────────────────────────────────────────────────────
        cs_raw, cs_smooth, weights = self._cs_computer.update(params)

        # ── 3. Adaptive thresholds (Layer 4) ──────────────────────────────────
        if self._threshold and self.cfg.use_adaptive_thresholds:
            tau_down, tau_up = self._threshold.update(cs_smooth, fi)
        else:
            tau_down = self.cfg.tau_down
            tau_up   = self.cfg.tau_up

        # Update buffers
        self._cs_buf.append(cs_smooth)
        self._param_buf.append(params)
        self._frames_since_sw += 1

        switched      = False
        trigger_layer = None
        forced        = False

        # ── 4. Energy safety override ──────────────────────────────────────────
        battery = mission_state.get("battery", 1.0)
        if battery <= self.cfg.b_crit and self._current_model != self.cfg.model_lightweight:
            self._current_model = self.cfg.model_lightweight
            switched = True
            forced   = True
            self._on_switch(fi, forced=True)

        # ── 5. Rate limiter ────────────────────────────────────────────────────
        rate_ok = self._frames_since_sw >= self.cfg.min_frames_switch

        if not forced and rate_ok:
            # ── Layer 1: Asymmetric hysteresis ─────────────────────────────────
            if 1 in self.layers:
                l1 = self._layer1(cs_smooth, tau_down, tau_up)
                if l1 == "up":
                    self._current_model = self.cfg.model_heavyweight
                    switched = True; trigger_layer = 1
                    self._hysteresis_cnt = 0
                elif l1 == "down":
                    self._current_model = self.cfg.model_lightweight
                    switched = True; trigger_layer = 1
                    self._hysteresis_cnt = 0

            # Track hysteresis-zone dwell time
            in_hysteresis = (
                tau_down <= cs_smooth <= tau_up
                and self._current_model == self.cfg.model_heavyweight
            )
            self._hysteresis_cnt = (self._hysteresis_cnt + 1) if in_hysteresis else 0

            if not switched and in_hysteresis:
                # ── Layer 3: Statistical stability (faster, checked first) ─────
                if 3 in self.layers:
                    if self._layer3():
                        self._current_model = self.cfg.model_lightweight
                        switched = True; trigger_layer = 3
                        self._hysteresis_cnt = 0

                # ── Layer 2: Feature validation (conservative) ─────────────────
                if not switched and 2 in self.layers:
                    if self._layer2(params):
                        self._current_model = self.cfg.model_lightweight
                        switched = True; trigger_layer = 2
                        self._hysteresis_cnt = 0

        if switched and not forced:
            self._on_switch(fi, layer=trigger_layer)

        # ── 6. Record history ──────────────────────────────────────────────────
        self.cs_raw_hist.append(cs_raw)
        self.cs_smooth_hist.append(cs_smooth)
        self.model_hist.append(self._current_model)
        self._frame_idx += 1

        ctrl_ms = (time.perf_counter() - t0) * 1000

        return StepResult(
            frame_idx     = fi,
            model_name    = self._current_model,
            cs_raw        = cs_raw,
            cs_smooth     = cs_smooth,
            tau_down      = tau_down,
            tau_up        = tau_up,
            switched      = switched,
            trigger_layer = trigger_layer,
            forced        = forced,
            params        = params,
            weights       = weights,
            controller_ms = ctrl_ms,
        )

    @property
    def current_model(self) -> str:
        return self._current_model

    def summary(self) -> dict:
        """Return a summary dict of the entire recorded session."""
        n = len(self.model_hist)
        if n == 0:
            return {}
        n_light = sum(1 for m in self.model_hist if m == self.cfg.model_lightweight)
        return {
            "total_frames":  n,
            "switch_count":  self.switch_count,
            "switch_rate_per_1k": round(self.switch_count / n * 1000, 2),
            "frac_lightweight": round(n_light / n, 4),
            "frac_heavyweight": round(1 - n_light / n, 4),
            "mean_cs_smooth":  round(float(np.mean(self.cs_smooth_hist)), 4),
            "std_cs_smooth":   round(float(np.std(self.cs_smooth_hist)),  4),
            "active_layers":   sorted(self.layers),
        }

    # ── Layer implementations ──────────────────────────────────────────────────

    def _layer1(self, cs: float, tau_down: float, tau_up: float) -> str | None:
        """Layer 1: Asymmetric hysteresis."""
        if cs < tau_down and self._current_model == self.cfg.model_heavyweight:
            return "down"
        if cs > tau_up   and self._current_model == self.cfg.model_lightweight:
            return "up"
        return None

    def _layer2(self, params: dict) -> bool:
        """
        Layer 2: Time-based feature validation.
        Triggers downgrade if ≥L2_FEATURES_NEEDED of the monitored scene
        parameters have been below their thresholds for ≥80% of the last
        L2_WINDOW frames AND the model has been in hysteresis ≥ dwell_frames.
        """
        if self._hysteresis_cnt < self.cfg.l2_dwell_frames:
            return False
        if len(self._param_buf) < self.cfg.l2_window:
            return False

        buf = list(self._param_buf)[-self.cfg.l2_window:]
        consistent_count = 0
        for pk, thr in self.cfg.l2_feature_thresholds.items():
            frac_below = sum(1 for b in buf if b.get(pk, 1.0) < thr) / self.cfg.l2_window
            if frac_below >= self.cfg.l2_consistency_thresh:
                consistent_count += 1

        return consistent_count >= self.cfg.l2_features_needed

    def _layer3(self) -> bool:
        """
        Layer 3: Statistical stability detection.
        Triggers downgrade if both CS and key scene parameters have low variance
        over the last L3_WINDOW frames.
        """
        if len(self._cs_buf) < self.cfg.l3_window:
            return False

        cs_arr = np.array(list(self._cs_buf))
        if float(np.std(cs_arr)) >= self.cfg.l3_cs_std_thresh:
            return False

        param_keys = list(self.cfg.l2_feature_thresholds.keys())
        buf = list(self._param_buf)[-self.cfg.l3_window:]
        stable_params = 0
        for pk in param_keys:
            arr = np.array([b.get(pk, 0.5) for b in buf])
            if len(arr) >= self.cfg.l3_window and float(np.std(arr)) < self.cfg.l3_param_std_thresh:
                stable_params += 1

        return stable_params >= self.cfg.l3_features_needed

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _on_switch(self, frame_idx: int, layer: int | None = None, forced: bool = False) -> None:
        self._frames_since_sw = 0
        self.switch_count    += 1
        self.switch_events.append({
            "frame":   frame_idx,
            "model":   self._current_model,
            "layer":   layer,
            "forced":  forced,
        })
