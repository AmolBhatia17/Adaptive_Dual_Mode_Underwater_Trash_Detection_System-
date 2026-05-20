"""
Unit Tests — Complexity Score Computation
==========================================
Tests cover:
  - BASE_WEIGHTS sum to 1.0
  - CS output is always in [0, 1]
  - EMA smoothing convergence
  - All 13 dynamic adaptation rules fire correctly
  - Category breakdown sums match CS_raw
  - Top-N contributors ordering

Run:
    pytest tests/test_cs_computation.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controller.complexity_score import (
    ComplexityScoreComputer,
    BASE_WEIGHTS,
    ADAPTATION_RULES,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _params(val: float = 0.5) -> dict[str, float]:
    """Return a dict of 20 parameters all set to `val`."""
    return {f"p{i}": val for i in range(1, 21)}


def _params_dict(**overrides) -> dict[str, float]:
    p = _params(0.4)
    p.update(overrides)
    return p


# ─── Base weight tests ────────────────────────────────────────────────────────

class TestBaseWeights:
    def test_sum_to_one(self):
        s = sum(BASE_WEIGHTS.values())
        assert abs(s - 1.0) < 1e-6, f"Weights sum to {s}, expected 1.0"

    def test_all_positive(self):
        for k, v in BASE_WEIGHTS.items():
            assert v > 0, f"{k} has non-positive weight {v}"

    def test_count(self):
        assert len(BASE_WEIGHTS) == 20, "Expected exactly 20 weights"

    def test_scene_group_total(self):
        scene = sum(BASE_WEIGHTS[f"w{i}"] for i in range(1, 9))
        # Scene group (P1–P8) carries the largest category share
        assert scene > 0.0 and scene <= 1.0
        assert abs(scene - round(scene, 2)) < 1e-6  # cleanly rounded

    def test_model_group_total(self):
        model = sum(BASE_WEIGHTS[f"w{i}"] for i in range(9, 15))
        assert model > 0.0 and model <= 1.0

    def test_mission_group_total(self):
        mission = sum(BASE_WEIGHTS[f"w{i}"] for i in range(15, 21))
        assert mission > 0.0 and mission <= 1.0

    def test_category_ordering(self):
        """Scene ≥ Model ≥ Mission by design (can relax if weights change)."""
        scene   = sum(BASE_WEIGHTS[f"w{i}"] for i in range(1,  9))
        model   = sum(BASE_WEIGHTS[f"w{i}"] for i in range(9,  15))
        mission = sum(BASE_WEIGHTS[f"w{i}"] for i in range(15, 21))
        assert scene >= model >= mission


# ─── CS output range ──────────────────────────────────────────────────────────

class TestCSRange:
    @pytest.mark.parametrize("val", [0.0, 0.1, 0.5, 0.9, 1.0])
    def test_cs_raw_in_unit_interval(self, val):
        comp = ComplexityScoreComputer()
        cs_raw, cs_smooth, _ = comp.update(_params(val))
        assert 0.0 <= cs_raw   <= 1.0
        assert 0.0 <= cs_smooth <= 1.0

    def test_all_zeros(self):
        comp = ComplexityScoreComputer()
        cs_raw, cs_smooth, _ = comp.update(_params(0.0))
        assert cs_raw   == pytest.approx(0.0, abs=1e-6)

    def test_all_ones(self):
        comp = ComplexityScoreComputer()
        cs_raw, cs_smooth, _ = comp.update(_params(1.0))
        assert cs_raw   == pytest.approx(1.0, abs=1e-6)

    def test_monotone_in_params(self):
        """Higher param values → higher CS_raw."""
        comp_lo = ComplexityScoreComputer(use_dynamic_weights=False)
        comp_hi = ComplexityScoreComputer(use_dynamic_weights=False)
        cs_lo, _, _ = comp_lo.update(_params(0.2))
        cs_hi, _, _ = comp_hi.update(_params(0.8))
        assert cs_hi > cs_lo


# ─── EMA smoothing ────────────────────────────────────────────────────────────

class TestEMASmoothing:
    def test_smooth_converges_to_raw(self):
        """After many identical frames, EMA smooth should stabilise (low variance)."""
        comp = ComplexityScoreComputer(alpha=0.30)
        p    = _params(0.60)
        for _ in range(300):
            comp.update(p)
        smooths = comp.history_smooth[-20:]
        # Smooth should have very low variance once converged
        assert float(np.std(smooths)) < 0.005, \
            f"EMA not converged — std={np.std(smooths):.4f}"

    def test_smooth_lags_raw(self):
        """After a sudden jump, smooth should lag behind raw."""
        comp = ComplexityScoreComputer(alpha=0.30)
        for _ in range(50):
            comp.update(_params(0.20))
        raw, smooth, _ = comp.update(_params(0.90))
        assert smooth < raw, "EMA should lag behind a sudden jump"

    def test_history_grows(self):
        comp = ComplexityScoreComputer()
        for i in range(10):
            comp.update(_params(0.5))
        assert len(comp.history_raw)    == 10
        assert len(comp.history_smooth) == 10

    def test_reset_clears_ema(self):
        comp = ComplexityScoreComputer(alpha=0.30)
        for _ in range(50):
            comp.update(_params(0.90))
        comp.reset()
        assert abs(comp.cs_smooth - 0.5) < 1e-6   # reset to 0.5 initial

    def test_alpha_0_frozen(self):
        """With α=0, smooth never changes after first frame."""
        comp = ComplexityScoreComputer(alpha=0.0)
        _, s0, _ = comp.update(_params(0.3))
        _, s1, _ = comp.update(_params(0.9))
        assert s0 == s1   # frozen


# ─── Dynamic weight adaptation ────────────────────────────────────────────────

class TestDynamicWeights:
    def test_static_weights_when_disabled(self):
        comp = ComplexityScoreComputer(use_dynamic_weights=False)
        w    = comp._compute_adaptive_weights(_params(0.99))
        for k, v in BASE_WEIGHTS.items():
            assert w[k] == pytest.approx(v, rel=1e-6)

    def test_adapted_weights_still_sum_to_one(self):
        comp = ComplexityScoreComputer(use_dynamic_weights=True)
        for p_val in [0.1, 0.5, 0.9]:
            w = comp._compute_adaptive_weights(_params(p_val))
            assert abs(sum(w.values()) - 1.0) < 1e-6

    @pytest.mark.parametrize("rule_idx,trigger_params", [
        (0,  {"p1": 0.70}),   # R1: high turbidity
        (1,  {"p5": 0.70}),   # R2: severe blur
        (2,  {"p10": 0.60}),  # R3: high confidence variance
        (3,  {"p15": 0.80}),  # R4: critical battery
        (4,  {"p8": 0.60}),   # R5: dense scene
        (5,  {"p12": 0.60}),  # R6: high dropout
        (6,  {"p6": 0.70}),   # R7: camera shake
        (7,  {"p16": 0.90}),  # R8: return phase
        (8,  {"p17": 0.80}),  # R9: high priority
        (9,  {"p11": 0.80}),  # R10: high latency
        (10, {"p7": 0.60}),   # R11: colour cast
        (11, {"p13": 0.60}),  # R12: false positives
        (12, {"p19": 0.90}),  # R13: mission near end
    ])
    def test_rule_fires_and_amplifies(self, rule_idx, trigger_params):
        rule = ADAPTATION_RULES[rule_idx]
        comp = ComplexityScoreComputer(use_dynamic_weights=True)
        p    = _params_dict(**trigger_params)

        assert rule.condition(p), f"Rule {rule_idx} condition did not fire"

        w_base    = dict(BASE_WEIGHTS)
        w_adapted = comp._compute_adaptive_weights(p)

        # The targeted weight should be amplified relative to re-normalised base
        key = rule.target_weight
        # Re-normalise base manually with only this rule
        w_test = dict(BASE_WEIGHTS)
        w_test[key] *= rule.factor
        total = sum(w_test.values())
        w_test = {k: v / total for k, v in w_test.items()}

        assert w_adapted[key] >= w_test[key] * 0.95   # within 5% tolerance


# ─── Category breakdown ───────────────────────────────────────────────────────

class TestCategoryBreakdown:
    def test_breakdown_keys(self):
        comp = ComplexityScoreComputer()
        bd   = comp.category_breakdown(_params(0.5))
        assert set(bd.keys()) == {"scene", "model", "mission"}

    def test_breakdown_all_positive(self):
        comp = ComplexityScoreComputer()
        bd   = comp.category_breakdown(_params(0.5))
        for k, v in bd.items():
            assert v >= 0.0, f"Negative category contribution: {k}={v}"

    def test_breakdown_sums_approximate_cs(self):
        """Sum of category breakdowns ≈ CS_raw (no dynamic weights)."""
        comp = ComplexityScoreComputer(use_dynamic_weights=False)
        p    = _params(0.5)
        cs_raw, _, _ = comp.update(p)
        bd   = comp.category_breakdown(p)
        assert abs(sum(bd.values()) - cs_raw) < 1e-6


# ─── Top contributors ─────────────────────────────────────────────────────────

class TestTopContributors:
    def test_returns_n_items(self):
        comp = ComplexityScoreComputer()
        p    = _params(0.5)
        comp.update(p)
        tops = comp.top_contributors(p, n=5)
        assert len(tops) == 5

    def test_sorted_descending(self):
        comp = ComplexityScoreComputer()
        p    = _params(0.5)
        comp.update(p)
        tops = comp.top_contributors(p, n=20)
        vals = [v for _, v in tops]
        assert vals == sorted(vals, reverse=True)

    def test_contributions_non_negative(self):
        comp = ComplexityScoreComputer()
        p    = _params(0.5)
        comp.update(p)
        for _, v in comp.top_contributors(p, n=20):
            assert v >= 0.0

    def test_param_with_highest_value_in_top(self):
        """P1 with high value and w1=0.08 should appear in top contributors."""
        comp = ComplexityScoreComputer(use_dynamic_weights=False)
        p    = _params(0.0)
        p["p1"] = 1.0   # only P1 is non-zero
        comp.update(p)
        tops = comp.top_contributors(p, n=3)
        names = [n for n, _ in tops]
        assert "P1" in names
