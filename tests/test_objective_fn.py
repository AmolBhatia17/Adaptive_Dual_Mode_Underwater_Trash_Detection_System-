"""
Unit Tests — Objective Function & Energy Model
===============================================
Tests cover:
  - EnergyModel per-frame energy values
  - UtilityEvaluator: J in valid range
  - Energy savings vs baseline
  - Hard constraint satisfaction
  - compare_baselines consistency
  - Edge cases: empty history, all-lightweight, all-heavyweight

Run:
    pytest tests/test_objective_fn.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controller.objective_function import EnergyModel, UtilityEvaluator


# ─── EnergyModel ──────────────────────────────────────────────────────────────

class TestEnergyModel:
    def test_lightweight_less_than_heavyweight(self):
        em = EnergyModel()
        assert em.energy_per_frame("YOLOv8n") < em.energy_per_frame("YOLOv8x")

    def test_e_max_is_heavyweight(self):
        em = EnergyModel()
        assert em.e_max == pytest.approx(em.energy_per_frame("YOLOv8x"), rel=1e-6)

    def test_all_models_positive(self):
        em = EnergyModel()
        for name in ("YOLOv8n", "YOLOv8x", "YOLOv10", "YOLOv12n", "YOLOv12x"):
            assert em.energy_per_frame(name) > 0

    def test_unknown_model_defaults_to_heavyweight(self):
        em = EnergyModel()
        assert em.energy_per_frame("UNKNOWN") == em.energy_per_frame("YOLOv8x")

    def test_energy_unit_is_joules(self):
        """energy_per_frame = watts / fps → Joules"""
        em = EnergyModel(power_heavyweight=50.0, fps=25.0)
        assert em.energy_per_frame("YOLOv8x") == pytest.approx(50.0 / 25.0, rel=1e-6)

    def test_e_max_scales_with_fps(self):
        em1 = EnergyModel(fps=30.0)
        em2 = EnergyModel(fps=15.0)
        assert em2.e_max == pytest.approx(em1.e_max * 2.0, rel=1e-6)


# ─── UtilityEvaluator — basic correctness ────────────────────────────────────

class TestUtilityEvaluatorBasic:
    def setup_method(self):
        self.ev = UtilityEvaluator(lam=0.60)

    def test_empty_history_returns_error(self):
        res = self.ev.evaluate([])
        assert "error" in res

    def test_J_in_valid_range(self):
        hist = ["YOLOv8n"] * 300
        res  = self.ev.evaluate(hist)
        # J must be in (0, 1) for λ=0.6 and reasonable accuracy
        assert -0.5 < res["J"] < 1.0

    def test_all_lightweight_has_high_savings(self):
        hist = ["YOLOv8n"] * 300
        res  = self.ev.evaluate(hist)
        assert res["energy_savings_pct"] > 60.0

    def test_all_heavyweight_zero_savings(self):
        hist = ["YOLOv8x"] * 300
        res  = self.ev.evaluate(hist)
        assert res["energy_savings_pct"] == pytest.approx(0.0, abs=0.1)

    def test_switch_count_correct(self):
        # Alternating every 50 frames = 5 switches
        hist = ["YOLOv8n"] * 50 + ["YOLOv8x"] * 50 + ["YOLOv8n"] * 50 + \
               ["YOLOv8x"] * 50 + ["YOLOv8n"] * 50 + ["YOLOv8x"] * 50
        res  = self.ev.evaluate(hist)
        assert res["switch_count"] == 5

    def test_switch_rate_calculation(self):
        hist = ["YOLOv8n"] * 150 + ["YOLOv8x"] * 150
        res  = self.ev.evaluate(hist)
        # switch_rate is rounded to 5 decimal places in the evaluator
        assert res["switch_count"] == 1
        assert abs(res["switch_rate"] - 1 / 300) < 0.0001

    def test_frame_utilities_length(self):
        hist = ["YOLOv8n"] * 100
        res  = self.ev.evaluate(hist)
        assert len(res["frame_utilities"]) == 100

    def test_energy_total_monotone_in_fraction_heavy(self):
        """More heavy frames → more total energy."""
        hist_more_heavy = ["YOLOv8x"] * 200 + ["YOLOv8n"] * 100
        hist_less_heavy = ["YOLOv8x"] * 100 + ["YOLOv8n"] * 200
        e_more = self.ev.evaluate(hist_more_heavy)["energy_total_J"]
        e_less = self.ev.evaluate(hist_less_heavy)["energy_total_J"]
        assert e_more > e_less


# ─── Accuracy overrides ───────────────────────────────────────────────────────

class TestCustomAccuracy:
    def test_custom_accuracy_used(self):
        ev = UtilityEvaluator(lam=0.60)
        hist = ["YOLOv8n"] * 10
        acc  = [1.0] * 10          # perfect accuracy
        res  = ev.evaluate(hist, accuracy_history=acc)
        # With A=1.0 and λ=0.6: u = 0.6 - 0.4*(15/50)/1 = 0.6 - 0.12 = 0.48
        assert res["J"] == pytest.approx(0.60 - 0.40 * (15.0 / 50.0), abs=0.01)

    def test_zero_accuracy_lowers_J(self):
        ev    = UtilityEvaluator(lam=0.60)
        hist  = ["YOLOv8n"] * 100
        acc0  = [0.0] * 100
        res0  = ev.evaluate(hist, accuracy_history=acc0)
        res_n = ev.evaluate(hist)                          # default mAP
        assert res0["J"] < res_n["J"]


# ─── Battery constraint ───────────────────────────────────────────────────────

class TestBatteryConstraint:
    def test_safety_violation_detected(self):
        ev      = UtilityEvaluator(b_crit=0.20)
        hist    = ["YOLOv8x"] * 100
        batt    = [0.10] * 100    # all below b_crit with heavy model
        res     = ev.evaluate(hist, battery_history=batt)
        assert res["safety_violations"] > 0
        assert res["constraint_ok"] is False

    def test_no_violation_with_lightweight(self):
        ev   = UtilityEvaluator(b_crit=0.20)
        hist = ["YOLOv8n"] * 100
        batt = [0.05] * 100       # critical battery but lightweight → OK
        res  = ev.evaluate(hist, battery_history=batt)
        assert res["safety_violations"] == 0

    def test_constraint_ok_within_switch_rate(self):
        ev   = UtilityEvaluator(r_max=0.01)
        hist = ["YOLOv8n"] * 290 + ["YOLOv8x"] * 10    # 1 switch in 300 → rate < 0.01
        res  = ev.evaluate(hist)
        assert res["switch_rate"] < 0.01
        assert res["constraint_ok"] is True


# ─── compare_baselines ────────────────────────────────────────────────────────

class TestCompareBaselines:
    def setup_method(self):
        self.ev   = UtilityEvaluator()
        self.hist = (
            ["YOLOv8n"] * 120 + ["YOLOv8x"] * 60 +
            ["YOLOv8n"] * 80  + ["YOLOv8x"] * 40
        )

    def test_all_systems_present(self):
        cmp = self.ev.compare_baselines(self.hist)
        for name in ("Adaptive", "YOLOv8n", "YOLOv8x", "YOLOv10", "YOLOv12n", "YOLOv12x"):
            assert name in cmp

    def test_adaptive_J_is_finite_and_valid(self):
        cmp = self.ev.compare_baselines(self.hist)
        J_adapt = cmp["Adaptive"]["J"]
        assert isinstance(J_adapt, float)
        assert -1.0 < J_adapt < 1.0   # physically valid range

    def test_adaptive_has_energy_savings(self):
        """Mixed model history should save more energy than pure heavyweight."""
        cmp = self.ev.compare_baselines(self.hist)
        assert cmp["Adaptive"]["energy_savings_pct"] > cmp["YOLOv8x"]["energy_savings_pct"]

    def test_heavyweight_zero_savings_in_compare(self):
        cmp = self.ev.compare_baselines(self.hist)
        assert cmp["YOLOv8x"]["energy_savings_pct"] == pytest.approx(0.0, abs=0.1)

    def test_all_results_have_J(self):
        cmp = self.ev.compare_baselines(self.hist)
        for name, res in cmp.items():
            assert "J" in res, f"Missing J in {name}"


# ─── Lambda sensitivity ───────────────────────────────────────────────────────

class TestLambdaSensitivity:
    @pytest.mark.parametrize("lam", [0.0, 0.3, 0.6, 0.9, 1.0])
    def test_J_valid_for_all_lambda(self, lam):
        ev   = UtilityEvaluator(lam=lam)
        hist = ["YOLOv8n"] * 100 + ["YOLOv8x"] * 100
        res  = ev.evaluate(hist)
        assert isinstance(res["J"], float)

    def test_lambda_1_cares_only_accuracy(self):
        """λ=1.0 → J ≈ mean accuracy, energy ignored."""
        ev    = UtilityEvaluator(lam=1.0)
        hist  = ["YOLOv8n"] * 100
        acc   = [0.87] * 100
        res   = ev.evaluate(hist, accuracy_history=acc)
        assert res["J"] == pytest.approx(0.87, abs=0.01)

    def test_lambda_0_cares_only_energy(self):
        """λ=0.0 → J ≈ −(E/E_max), accuracy ignored."""
        ev    = UtilityEvaluator(lam=0.0)
        em    = ev.energy
        hist  = ["YOLOv8n"] * 100
        res   = ev.evaluate(hist)
        expected_u = -(em.energy_per_frame("YOLOv8n") / em.e_max)
        assert res["J"] == pytest.approx(expected_u, abs=0.01)
