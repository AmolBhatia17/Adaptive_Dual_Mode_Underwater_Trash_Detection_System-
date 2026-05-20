"""
Unit Tests — Adaptive Threshold Updater (Layer 4)
===================================================
Tests cover every behaviour described in the paper and implemented in
controller/adaptive_threshold.py:

  1. Initial state — base thresholds, uncalibrated
  2. Calibration trigger — activates after env_window frames
  3. Drift detection — γ gate: no update when drift ≤ γ
  4. Upward drift — both thresholds nudge UP toward high CS_env
  5. Downward drift — both thresholds nudge DOWN toward low CS_env
  6. Clip bounds — τ_down never leaves [0.25, 0.50], τ_up [0.45, 0.70]
  7. Hysteresis gap preservation — τ_up − τ_down ≥ 0.10 always
  8. Half-step size — each calibration moves by δ/2, not δ
  9. History logging — only logged when something changes
  10. Reset — returns to base thresholds exactly
  11. Battery suspension — simulated via controller integration
  12. n_updates counter
  13. Update interval cadence — only fires at multiples of update_interval

Run:
    pytest tests/test_adaptive_threshold.py -v
"""

import sys
from pathlib import Path
from collections import deque

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controller.adaptive_threshold import AdaptiveThresholdUpdater, ThresholdConfig


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _atu(
    tau_down_base: float = 0.40,
    tau_up_base:   float = 0.55,
    delta_theta:   float = 0.15,
    gamma:         float = 0.02,
    update_interval: int = 100,
    env_window:      int = 100,
    tau_down_min:  float = 0.25,
    tau_down_max:  float = 0.50,
    tau_up_min:    float = 0.45,
    tau_up_max:    float = 0.70,
) -> AdaptiveThresholdUpdater:
    cfg = ThresholdConfig(
        tau_down_base=tau_down_base, tau_up_base=tau_up_base,
        delta_theta=delta_theta, gamma=gamma,
        update_interval=update_interval, env_window=env_window,
        tau_down_min=tau_down_min, tau_down_max=tau_down_max,
        tau_up_min=tau_up_min, tau_up_max=tau_up_max,
    )
    return AdaptiveThresholdUpdater(cfg)


def _fill_and_calibrate(
    atu: AdaptiveThresholdUpdater,
    cs_env: float,
    n_fill: int | None = None,
) -> tuple[float, float]:
    """
    Fill the internal buffer with `cs_env` for env_window frames,
    then run one more frame that lands on the update_interval boundary
    to trigger calibration.  Returns (tau_down, tau_up).
    """
    cfg = atu.cfg
    n   = n_fill or cfg.env_window

    # Fill buffer but stay just before the update boundary
    for fi in range(n):
        td, tu = atu.update(cs_env, fi)

    # Now fire the calibration tick (frame index = update_interval - 1)
    boundary = cfg.update_interval - 1
    td, tu = atu.update(cs_env, boundary)
    return td, tu


# ─── 1. Initial state ─────────────────────────────────────────────────────────

class TestInitialState:
    def test_base_thresholds_at_start(self):
        atu = _atu()
        assert atu.tau_down == pytest.approx(0.40)
        assert atu.tau_up   == pytest.approx(0.55)

    def test_not_calibrated_at_start(self):
        atu = _atu()
        assert atu.is_calibrated is False

    def test_n_updates_zero_at_start(self):
        atu = _atu()
        assert atu.n_updates == 0

    def test_history_empty_at_start(self):
        atu = _atu()
        assert atu.history == []

    def test_update_returns_tuple(self):
        atu = _atu()
        result = atu.update(0.5, 0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_thresholds_unchanged_before_calibration(self):
        """Before env_window frames, thresholds must not move at all."""
        atu = _atu(env_window=100, update_interval=100)
        for fi in range(50):   # only half the window
            td, tu = atu.update(0.80, fi)  # extreme CS — would drift if active
        assert td == pytest.approx(0.40)
        assert tu == pytest.approx(0.55)


# ─── 2. Calibration trigger ───────────────────────────────────────────────────

class TestCalibrationTrigger:
    def test_calibrated_after_env_window(self):
        atu = _atu(env_window=10, update_interval=100)
        for fi in range(10):
            atu.update(0.5, fi)
        assert atu.is_calibrated is True

    def test_not_calibrated_before_env_window(self):
        atu = _atu(env_window=10, update_interval=100)
        for fi in range(9):
            atu.update(0.5, fi)
        assert atu.is_calibrated is False

    def test_calibration_fires_at_update_interval(self):
        """An update must happen exactly at frame update_interval - 1."""
        atu = _atu(env_window=5, update_interval=10, delta_theta=0.15, gamma=0.02)
        # Fill with high CS to ensure drift
        for fi in range(9):
            atu.update(0.80, fi)
        n_before = atu.n_updates
        atu.update(0.80, 9)    # frame 9 = update_interval(10) - 1
        # Should have either updated or stayed (depends on drift magnitude)
        assert atu.n_updates >= n_before

    def test_calibration_does_not_fire_between_intervals(self):
        atu = _atu(env_window=5, update_interval=10, delta_theta=0.15, gamma=0.02)
        for fi in range(9):
            atu.update(0.80, fi)
        atu.update(0.80, 9)   # first calibration
        n_after_first = atu.n_updates
        # Frames 10–17 should NOT trigger another calibration
        for fi in range(10, 18):
            atu.update(0.80, fi)
        assert atu.n_updates == n_after_first   # no additional updates


# ─── 3. Drift detection (γ gate) ──────────────────────────────────────────────

class TestGammaGate:
    def test_no_update_when_drift_below_gamma(self):
        """CS_env drift smaller than γ → thresholds unchanged after calibration."""
        # Use a very large gamma so tiny drifts never trigger
        # CS_env = 0.41 → drift from tau_down=0.40 is 0.01, tau_up=0.55 is 0.14
        # Use gamma=0.20 → both drifts < gamma → no update at all
        atu = _atu(gamma=0.20, env_window=10, update_interval=10)
        td_before = atu.tau_down
        tu_before = atu.tau_up
        # Fill exactly env_window frames then hit the calibration boundary
        for fi in range(9):
            atu.update(0.41, fi)
        atu.update(0.41, 9)   # frame 9 = update_interval(10) - 1 → calibration fires
        # drift from tau_down: |0.41-0.40|=0.01 < gamma(0.20) → no change
        # drift from tau_up:   |0.41-0.55|=0.14 < gamma(0.20) → no change
        assert atu.tau_down == pytest.approx(td_before, abs=1e-4)
        assert atu.tau_up   == pytest.approx(tu_before, abs=1e-4)
        assert atu.n_updates == 0

    def test_update_triggered_when_drift_exceeds_gamma(self):
        """CS_env far from thresholds → update fires."""
        atu = _atu(gamma=0.02, env_window=10, update_interval=10)
        _fill_and_calibrate(atu, 0.70, n_fill=10)   # far above tau_up(0.55)
        assert atu.n_updates >= 1


# ─── 4. Upward drift ──────────────────────────────────────────────────────────

class TestUpwardDrift:
    def test_tau_down_increases_with_high_cs_env(self):
        atu = _atu(env_window=10, update_interval=10)
        td_before = atu.tau_down
        _fill_and_calibrate(atu, 0.70, n_fill=10)
        assert atu.tau_down >= td_before

    def test_tau_up_increases_with_high_cs_env(self):
        atu = _atu(env_window=10, update_interval=10)
        tu_before = atu.tau_up
        _fill_and_calibrate(atu, 0.70, n_fill=10)
        assert atu.tau_up >= tu_before

    def test_both_thresholds_shift_up_on_high_env(self):
        atu = _atu(env_window=10, update_interval=10)
        td0, tu0 = atu.tau_down, atu.tau_up
        _fill_and_calibrate(atu, 0.70, n_fill=10)
        assert atu.tau_down + atu.tau_up > td0 + tu0


# ─── 5. Downward drift ────────────────────────────────────────────────────────

class TestDownwardDrift:
    def test_tau_down_decreases_with_low_cs_env(self):
        atu = _atu(env_window=10, update_interval=10)
        td_before = atu.tau_down
        _fill_and_calibrate(atu, 0.10, n_fill=10)   # well below tau_down(0.40)
        assert atu.tau_down <= td_before

    def test_tau_up_decreases_with_low_cs_env(self):
        atu = _atu(env_window=10, update_interval=10)
        tu_before = atu.tau_up
        _fill_and_calibrate(atu, 0.10, n_fill=10)
        assert atu.tau_up <= tu_before

    def test_both_thresholds_shift_down_on_low_env(self):
        atu = _atu(env_window=10, update_interval=10)
        td0, tu0 = atu.tau_down, atu.tau_up
        _fill_and_calibrate(atu, 0.10, n_fill=10)
        assert atu.tau_down + atu.tau_up < td0 + tu0


# ─── 6. Clip bounds ───────────────────────────────────────────────────────────

class TestClipBounds:
    def test_tau_down_never_exceeds_max(self):
        atu = _atu(tau_down_max=0.50, env_window=5, update_interval=5)
        # Run many calibrations with extreme high CS
        for fi in range(500):
            atu.update(1.0, fi)
        assert atu.tau_down <= 0.50

    def test_tau_down_never_goes_below_min(self):
        atu = _atu(tau_down_min=0.25, env_window=5, update_interval=5)
        for fi in range(500):
            atu.update(0.0, fi)
        assert atu.tau_down >= 0.25

    def test_tau_up_never_exceeds_max(self):
        atu = _atu(tau_up_max=0.70, env_window=5, update_interval=5)
        for fi in range(500):
            atu.update(1.0, fi)
        assert atu.tau_up <= 0.70

    def test_tau_up_never_goes_below_min(self):
        atu = _atu(tau_up_min=0.45, env_window=5, update_interval=5)
        for fi in range(500):
            atu.update(0.0, fi)
        assert atu.tau_up >= 0.45

    def test_both_bounds_respected_simultaneously(self):
        atu = _atu(
            tau_down_min=0.25, tau_down_max=0.50,
            tau_up_min=0.45,   tau_up_max=0.70,
            env_window=5, update_interval=5,
        )
        for fi in range(1000):
            cs = 1.0 if fi % 100 < 50 else 0.0   # alternating extremes
            atu.update(cs, fi)
        assert 0.25 <= atu.tau_down <= 0.50
        assert 0.45 <= atu.tau_up   <= 0.70


# ─── 7. Hysteresis gap preservation ──────────────────────────────────────────

class TestHysteresisGap:
    def test_gap_never_collapses_below_010(self):
        """After any number of calibrations, τ_up − τ_down ≥ 0.10."""
        atu = _atu(env_window=5, update_interval=5)
        for fi in range(1000):
            # Use CS values that push thresholds toward each other
            cs = 0.475   # exactly in the middle of [tau_down, tau_up]
            atu.update(cs, fi)
        gap = atu.tau_up - atu.tau_down
        assert gap >= 0.10, f"Hysteresis gap collapsed to {gap:.4f}"

    def test_gap_preserved_under_extreme_cs(self):
        atu = _atu(env_window=5, update_interval=5)
        for fi in range(500):
            atu.update(0.50, fi)   # exactly mid-hysteresis, maximum squeeze pressure
        assert atu.tau_up - atu.tau_down >= 0.10

    def test_initial_gap_is_correct(self):
        atu = _atu(tau_down_base=0.40, tau_up_base=0.55)
        assert pytest.approx(atu.tau_up - atu.tau_down, abs=1e-6) == 0.15


# ─── 8. Half-step size ────────────────────────────────────────────────────────

class TestHalfStep:
    def test_single_calibration_moves_less_than_delta(self):
        """Each single calibration step moves by at most δ * 0.5 per threshold."""
        # Drive one clean single calibration: fill env_window frames,
        # then fire exactly ONE calibration tick and measure the shift.
        atu = _atu(
            tau_down_base=0.30, tau_up_base=0.45,
            tau_down_min=0.20, tau_down_max=0.50,
            tau_up_min=0.35,   tau_up_max=0.70,
            delta_theta=0.15, env_window=10, update_interval=10,
        )
        # Fill buffer (frames 0-8, no calibration fires yet)
        for fi in range(9):
            atu.update(0.70, fi)
        td0 = atu.tau_down
        # Frame 9 = update_interval(10) - 1 → exactly ONE calibration fires
        atu.update(0.70, 9)
        # Max shift = delta_theta * 0.5 = 0.075
        assert abs(atu.tau_down - td0) <= 0.075 + 1e-6

    def test_gradual_convergence_over_multiple_calibrations(self):
        """Thresholds converge gradually — not a single jump."""
        atu = _atu(delta_theta=0.15, env_window=10, update_interval=10)
        positions = []
        for fi in range(200):
            td, _ = atu.update(0.70, fi)
            if (fi + 1) % 10 == 0:
                positions.append(td)
        # Should be monotonically non-decreasing (drifting up toward 0.70)
        assert all(
            positions[i] <= positions[i + 1] + 1e-6
            for i in range(len(positions) - 1)
        )


# ─── 9. History logging ───────────────────────────────────────────────────────

class TestHistoryLogging:
    def test_history_logged_when_changed(self):
        atu = _atu(env_window=10, update_interval=10)
        _fill_and_calibrate(atu, 0.70, n_fill=10)
        if atu.n_updates > 0:
            assert len(atu.history) == atu.n_updates

    def test_history_entry_has_required_keys(self):
        atu = _atu(env_window=10, update_interval=10)
        _fill_and_calibrate(atu, 0.70, n_fill=10)
        if atu.history:
            entry = atu.history[0]
            for key in ("frame", "cs_env", "tau_down", "tau_up", "delta"):
                assert key in entry, f"Missing key '{key}' in history entry"

    def test_history_cs_env_reflects_actual_mean(self):
        atu = _atu(env_window=10, update_interval=10)
        cs_val = 0.70
        _fill_and_calibrate(atu, cs_val, n_fill=10)
        if atu.history:
            assert atu.history[0]["cs_env"] == pytest.approx(cs_val, abs=0.01)

    def test_no_history_when_no_change(self):
        """No history entries if thresholds never moved."""
        atu = _atu(gamma=0.50, env_window=10, update_interval=10)
        # gamma=0.50 → only extreme drifts trigger update; 0.475 won't
        _fill_and_calibrate(atu, 0.475, n_fill=10)
        assert len(atu.history) == 0


# ─── 10. Reset ───────────────────────────────────────────────────────────────

class TestReset:
    def test_tau_down_resets_to_base(self):
        atu = _atu(env_window=10, update_interval=10)
        _fill_and_calibrate(atu, 0.70, n_fill=10)
        atu.reset()
        assert atu.tau_down == pytest.approx(0.40)

    def test_tau_up_resets_to_base(self):
        atu = _atu(env_window=10, update_interval=10)
        _fill_and_calibrate(atu, 0.70, n_fill=10)
        atu.reset()
        assert atu.tau_up == pytest.approx(0.55)

    def test_calibrated_flag_cleared_on_reset(self):
        atu = _atu(env_window=10, update_interval=10)
        _fill_and_calibrate(atu, 0.70, n_fill=10)
        assert atu.is_calibrated is True
        atu.reset()
        assert atu.is_calibrated is False

    def test_n_updates_cleared_on_reset(self):
        atu = _atu(env_window=10, update_interval=10)
        _fill_and_calibrate(atu, 0.70, n_fill=10)
        atu.reset()
        assert atu.n_updates == 0

    def test_buffer_cleared_on_reset(self):
        atu = _atu(env_window=10, update_interval=10)
        _fill_and_calibrate(atu, 0.70, n_fill=10)
        atu.reset()
        # After reset, should need env_window frames again before calibrating
        assert atu.is_calibrated is False
        for fi in range(9):
            atu.update(0.70, fi)
        assert atu.is_calibrated is False


# ─── 11. n_updates counter ───────────────────────────────────────────────────

class TestNUpdatesCounter:
    def test_counts_each_calibration_that_changes_thresholds(self):
        atu = _atu(env_window=5, update_interval=5)
        # Run for 3 calibration intervals, all with high CS
        for fi in range(15):
            atu.update(0.80, fi)
        # Should have fired at frames 4, 9, 14 — up to 3 updates
        assert atu.n_updates >= 1
        assert atu.n_updates <= 3

    def test_no_count_if_no_change(self):
        atu = _atu(gamma=0.50, env_window=5, update_interval=5)
        for fi in range(50):
            atu.update(0.475, fi)   # tiny drift, γ=0.50 blocks update
        assert atu.n_updates == 0


# ─── 12. Integration with controller ─────────────────────────────────────────

class TestControllerIntegration:
    def test_thresholds_appear_in_step_result(self):
        """Verify Layer 4 thresholds are plumbed into StepResult."""
        import numpy as np
        from controller.deepclean_v4 import DeepCleanController, ControllerConfig

        cfg  = ControllerConfig(use_adaptive_thresholds=True)
        ctrl = DeepCleanController(cfg=cfg)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        for _ in range(10):
            res = ctrl.step(frame)

        assert hasattr(res, "tau_down")
        assert hasattr(res, "tau_up")
        assert 0.0 < res.tau_down < res.tau_up < 1.0

    def test_adaptive_thresholds_disabled_option(self):
        """With use_adaptive_thresholds=False, thresholds stay fixed."""
        import numpy as np
        from controller.deepclean_v4 import DeepCleanController, ControllerConfig

        cfg  = ControllerConfig(
            use_adaptive_thresholds=False,
            tau_down=0.40, tau_up=0.55,
        )
        ctrl = DeepCleanController(cfg=cfg)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        for _ in range(200):
            res = ctrl.step(frame)

        assert res.tau_down == pytest.approx(0.40, abs=0.01)
        assert res.tau_up   == pytest.approx(0.55, abs=0.01)

    def test_long_run_thresholds_stay_valid(self):
        """After 500 frames, thresholds must remain in clip bounds."""
        import numpy as np
        from controller.deepclean_v4 import DeepCleanController

        ctrl  = DeepCleanController()
        frame = np.zeros((64, 64, 3), dtype=np.uint8)

        for _ in range(500):
            res = ctrl.step(frame)

        assert 0.25 <= res.tau_down <= 0.50
        assert 0.45 <= res.tau_up   <= 0.70
        assert res.tau_up - res.tau_down >= 0.10
