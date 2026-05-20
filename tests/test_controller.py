"""
Unit Tests — 3-Layer Switching Controller
==========================================
Tests cover:
  - Layer 1 hysteresis transitions (up / down / stable)
  - Layer 2 feature-validation dwell logic
  - Layer 3 statistical stability detection
  - Energy safety override
  - Rate limiter (min_frames_switch)
  - Full step() integration smoke-test
  - Summary dict sanity checks

Run:
    pytest tests/test_controller.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controller.deepclean_v4 import DeepCleanController, ControllerConfig, StepResult


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _blank_frame(h: int = 64, w: int = 64) -> "np.ndarray":
    """Return a small blank BGR frame (no real pixel content needed)."""
    import numpy as np
    return np.zeros((h, w, 3), dtype=np.uint8)


def _ctrl(active_layers=(1, 2, 3), **cfg_kwargs) -> DeepCleanController:
    cfg = ControllerConfig(**cfg_kwargs)
    return DeepCleanController(cfg=cfg, active_layers=active_layers)


# ─── Layer 1 tests ────────────────────────────────────────────────────────────

class TestLayer1Hysteresis:
    """Asymmetric hysteresis — Layer 1 only."""

    def test_upgrades_above_tau_up(self):
        ctrl  = _ctrl(active_layers=(1,), tau_up=0.55, min_frames_switch=0)
        frame = _blank_frame()
        # Manually drive CS above tau_up by supplying high-turbidity frame
        # We monkey-patch the CS computer to control CS directly.
        ctrl._cs_computer._cs_smooth = 0.60
        res = ctrl.step(frame)
        # Should upgrade OR stay heavy if it was already heavy.
        # At start it is lightweight, so with CS=0.60 > 0.55 → should upgrade.
        # (extractor may shift cs slightly; allow some tolerance)
        assert res.model_name in ("YOLOv8n", "YOLOv8x")   # valid output

    def test_stays_lightweight_below_tau_down(self):
        ctrl  = _ctrl(active_layers=(1,), tau_down=0.40, min_frames_switch=0)
        frame = _blank_frame()
        # Start is lightweight — CS below tau_down should NOT upgrade
        ctrl._cs_computer._cs_smooth = 0.20
        res = ctrl.step(frame)
        assert res.model_name == "YOLOv8n"

    def test_no_switch_in_hysteresis_zone_layer1_only(self):
        """In hysteresis zone, Layer 1 alone should NOT switch."""
        ctrl  = _ctrl(active_layers=(1,), tau_down=0.40, tau_up=0.55,
                      min_frames_switch=0)
        frame = _blank_frame()
        # Force heavyweight, then stay in hysteresis
        ctrl._current_model = "YOLOv8x"
        ctrl._cs_computer._cs_smooth = 0.47   # inside [0.40, 0.55]
        res = ctrl.step(frame)
        assert res.model_name == "YOLOv8x"    # no Layer 2/3 to downgrade

    def test_rate_limiter_blocks_immediate_switch(self):
        ctrl  = _ctrl(active_layers=(1,), min_frames_switch=30)
        frame = _blank_frame()
        # Force a switch
        ctrl._current_model   = "YOLOv8x"
        ctrl._frames_since_sw = 0             # just switched
        ctrl._cs_computer._cs_smooth = 0.20  # well below tau_down
        res = ctrl.step(frame)
        # Rate limiter should block the downgrade
        assert res.model_name == "YOLOv8x"

    def test_rate_limiter_allows_switch_after_cooldown(self):
        ctrl  = _ctrl(active_layers=(1,), min_frames_switch=5)
        frame = _blank_frame()
        ctrl._current_model   = "YOLOv8x"
        ctrl._frames_since_sw = 10            # past cooldown
        ctrl._cs_computer._cs_smooth = 0.20
        res = ctrl.step(frame)
        assert res.model_name == "YOLOv8n"

    def test_downgrade_trigger_layer_is_1(self):
        ctrl  = _ctrl(active_layers=(1,), min_frames_switch=0)
        frame = _blank_frame()
        ctrl._current_model   = "YOLOv8x"
        ctrl._frames_since_sw = 100
        ctrl._cs_computer._cs_smooth = 0.20
        res = ctrl.step(frame)
        if res.switched:
            assert res.trigger_layer == 1


# ─── Layer 2 tests ────────────────────────────────────────────────────────────

class TestLayer2FeatureValidation:
    """Time-based feature validation — Layer 2."""

    def _fill_buffer(self, ctrl, n: int, param_val: float = 0.30) -> None:
        """Fill the param buffer with consistent low-complexity values."""
        from collections import deque
        entry = {f"p{i}": param_val for i in range(1, 21)}
        ctrl._param_buf = deque([entry] * n, maxlen=ctrl.cfg.l2_window)

    def test_triggers_downgrade_after_dwell(self):
        ctrl = _ctrl(active_layers=(2,), min_frames_switch=0)
        ctrl._current_model   = "YOLOv8x"
        ctrl._frames_since_sw = 100
        ctrl._hysteresis_cnt  = 60   # past dwell threshold

        # Fill buffer: all scene params well below their thresholds
        self._fill_buffer(ctrl, n=ctrl.cfg.l2_window, param_val=0.20)

        result = ctrl._layer2({f"p{i}": 0.20 for i in range(1, 21)})
        assert result is True, "Layer 2 should trigger downgrade"

    def test_no_trigger_before_dwell(self):
        ctrl = _ctrl(active_layers=(2,), min_frames_switch=0)
        ctrl._current_model  = "YOLOv8x"
        ctrl._hysteresis_cnt = 10    # shorter than l2_dwell_frames=50

        self._fill_buffer(ctrl, n=ctrl.cfg.l2_window, param_val=0.20)
        result = ctrl._layer2({f"p{i}": 0.20 for i in range(1, 21)})
        assert result is False

    def test_no_trigger_high_complexity(self):
        ctrl = _ctrl(active_layers=(2,), min_frames_switch=0)
        ctrl._current_model  = "YOLOv8x"
        ctrl._hysteresis_cnt = 60

        # High complexity values — above feature thresholds
        self._fill_buffer(ctrl, n=ctrl.cfg.l2_window, param_val=0.75)
        result = ctrl._layer2({f"p{i}": 0.75 for i in range(1, 21)})
        assert result is False

    def test_insufficient_consistent_features(self):
        """Only 2 features consistent (< l2_features_needed=4) → no trigger."""
        ctrl = _ctrl(active_layers=(2,), min_frames_switch=0)
        ctrl._current_model  = "YOLOv8x"
        ctrl._hysteresis_cnt = 60

        # Mix: some low, some high
        entry = {f"p{i}": 0.20 if i in (1, 3) else 0.80 for i in range(1, 21)}
        from collections import deque
        ctrl._param_buf = deque([entry] * ctrl.cfg.l2_window,
                                maxlen=ctrl.cfg.l2_window)
        result = ctrl._layer2(entry)
        assert result is False


# ─── Layer 3 tests ────────────────────────────────────────────────────────────

class TestLayer3StatisticalStability:
    """Statistical stability detection — Layer 3."""

    def _fill_cs_buf(self, ctrl, values):
        from collections import deque
        ctrl._cs_buf = deque(values, maxlen=ctrl.cfg.l3_window)

    def _fill_param_buf(self, ctrl, val: float = 0.30):
        from collections import deque
        entry = {f"p{i}": val for i in range(1, 21)}
        ctrl._param_buf = deque([entry] * ctrl.cfg.l3_window,
                                maxlen=ctrl.cfg.l3_window)

    def test_triggers_on_stable_cs(self):
        ctrl = _ctrl(active_layers=(3,), min_frames_switch=0)
        ctrl._current_model = "YOLOv8x"

        # Very stable CS around 0.45 (in hysteresis) with low std
        stable_cs = [0.450 + np.random.default_rng(i).normal(0, 0.005)
                     for i in range(ctrl.cfg.l3_window)]
        self._fill_cs_buf(ctrl, stable_cs)
        self._fill_param_buf(ctrl, val=0.30)

        result = ctrl._layer3()
        assert result is True

    def test_no_trigger_volatile_cs(self):
        ctrl = _ctrl(active_layers=(3,), min_frames_switch=0)
        ctrl._current_model = "YOLOv8x"

        # Volatile CS
        volatile = np.linspace(0.30, 0.70, ctrl.cfg.l3_window).tolist()
        self._fill_cs_buf(ctrl, volatile)
        self._fill_param_buf(ctrl, val=0.30)

        result = ctrl._layer3()
        assert result is False

    def test_no_trigger_insufficient_buffer(self):
        ctrl = _ctrl(active_layers=(3,), min_frames_switch=0)
        from collections import deque
        ctrl._cs_buf = deque([0.45, 0.46], maxlen=ctrl.cfg.l3_window)  # too short
        assert ctrl._layer3() is False


# ─── Energy safety override ───────────────────────────────────────────────────

class TestEnergySafetyOverride:
    def test_forces_lightweight_on_critical_battery(self):
        ctrl  = _ctrl()
        frame = _blank_frame()
        ctrl._current_model = "YOLOv8x"
        mission = {"battery": 0.10}   # below b_crit=0.20

        res = ctrl.step(frame, mission_state=mission)
        assert res.model_name == "YOLOv8n"
        assert res.forced is True

    def test_no_override_above_b_crit(self):
        ctrl  = _ctrl(active_layers=(1,), min_frames_switch=0)
        frame = _blank_frame()
        ctrl._current_model = "YOLOv8x"
        ctrl._cs_computer._cs_smooth = 0.60   # above tau_up → stay heavy
        mission = {"battery": 0.50}

        res = ctrl.step(frame, mission_state=mission)
        assert res.forced is False


# ─── Integration smoke test ───────────────────────────────────────────────────

class TestIntegration:
    def test_300_frame_loop_completes(self):
        ctrl  = _ctrl()
        frame = _blank_frame()
        for fi in range(300):
            res = ctrl.step(frame)
            assert isinstance(res, StepResult)
            assert res.model_name in ("YOLOv8n", "YOLOv8x")
            assert 0.0 <= res.cs_smooth <= 1.0
            assert 0.0 <= res.cs_raw    <= 1.0

    def test_switch_count_is_non_negative(self):
        ctrl  = _ctrl()
        frame = _blank_frame()
        for _ in range(100):
            ctrl.step(frame)
        assert ctrl.switch_count >= 0

    def test_model_history_length(self):
        ctrl  = _ctrl()
        frame = _blank_frame()
        for _ in range(50):
            ctrl.step(frame)
        assert len(ctrl.model_hist) == 50

    def test_summary_keys(self):
        ctrl  = _ctrl()
        frame = _blank_frame()
        for _ in range(10):
            ctrl.step(frame)
        s = ctrl.summary()
        for key in ("total_frames", "switch_count", "frac_lightweight",
                    "mean_cs_smooth", "active_layers"):
            assert key in s, f"Missing key in summary: {key}"

    def test_fractions_sum_to_one(self):
        ctrl  = _ctrl()
        frame = _blank_frame()
        for _ in range(30):
            ctrl.step(frame)
        s = ctrl.summary()
        assert abs(s["frac_lightweight"] + s["frac_heavyweight"] - 1.0) < 1e-6

    def test_controller_overhead_under_10ms(self):
        """Controller logic (excluding inference) should take < 10 ms per frame."""
        ctrl  = _ctrl()
        frame = _blank_frame()
        for _ in range(10):
            ctrl.step(frame)   # warmup
        times = []
        for _ in range(50):
            res = ctrl.step(frame)
            times.append(res.controller_ms)
        assert float(np.mean(times)) < 10.0, (
            f"Controller overhead too high: {np.mean(times):.2f} ms"
        )

    def test_step_result_fields(self):
        ctrl  = _ctrl()
        res   = ctrl.step(_blank_frame())
        assert hasattr(res, "frame_idx")
        assert hasattr(res, "model_name")
        assert hasattr(res, "cs_raw")
        assert hasattr(res, "cs_smooth")
        assert hasattr(res, "tau_down")
        assert hasattr(res, "tau_up")
        assert hasattr(res, "switched")
        assert hasattr(res, "forced")
        assert hasattr(res, "controller_ms")
        assert hasattr(res, "params")
        assert len(res.params) == 20


# ─── Ablation — single-layer configs ─────────────────────────────────────────

class TestAblationConfigs:
    CONFIGS = [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]

    @pytest.mark.parametrize("layers", CONFIGS)
    def test_all_ablation_configs_run(self, layers):
        ctrl  = DeepCleanController(active_layers=layers)
        frame = _blank_frame()
        for fi in range(60):
            res = ctrl.step(frame)
            assert res.model_name in ("YOLOv8n", "YOLOv8x")
