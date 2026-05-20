"""
Complexity Score (CS) Computation
===================================
Implements:
  1. Expert base weights  w₁ … w₂₀  (summing to 1.0)
  2. 13 dynamic weight adaptation rules
  3. Raw CS computation:   CS_raw(t) = Σ wᵢ · pᵢ(t)
  4. EMA temporal smoothing: CS̃(t) = α · CS_raw + (1−α) · CS̃(t−1)

All weight categories and their empirical justification are described in
Section 3.1.1 of the paper.
"""

import numpy as np
from dataclasses import dataclass, field


# ─── Base weight table ────────────────────────────────────────────────────────
# Weights satisfy Σwᵢ = 1.0
# Grouped:  Scene Attributes (p1–p8)  = 40%
#           Model Feedback   (p9–p14) = 35%
#           Mission Context  (p15–p20)= 25%

BASE_WEIGHTS: dict[str, float] = {
    # ── Scene Attributes (40% total) ─────────────────────────────────────────
    "w1":  0.08,   # P1  turbidity            (highest scene weight)
    "w2":  0.06,   # P2  lighting variation
    "w3":  0.07,   # P3  texture richness
    "w4":  0.07,   # P4  occlusion level
    "w5":  0.05,   # P5  motion blur
    "w6":  0.04,   # P6  camera stability
    "w7":  0.06,   # P7  colour cast
    "w8":  0.05,   # P8  object density / object density

    # ── Model Feedback (35% total) ────────────────────────────────────────────
    "w9":  0.08,   # P9  low confidence        (highest model weight)
    "w10": 0.05,   # P10 confidence variance
    "w11": 0.04,   # P11 inference latency
    "w12": 0.06,   # P12 detection dropout rate
    "w13": 0.04,   # P13 false positive ratio
    "w14": 0.05,   # P14 bounding-box instability → changed from energy to BBox

    # ── Mission Context (25% total) ───────────────────────────────────────────
    "w15": 0.06,   # P15 battery (inverted: low battery → high param)
    "w16": 0.05,   # P16 mission phase
    "w17": 0.03,   # P17 trash priority
    "w18": 0.02,   # P18 distance to coverage zone
    "w19": 0.02,   # P19 time remaining (inverted)
    "w20": 0.02,   # P20 bandwidth (inverted)
}

# Sanity check
assert abs(sum(BASE_WEIGHTS.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"


# ─── Dynamic weight adaptation rules ──────────────────────────────────────────
# Each rule is (condition_fn, param_key, amplification_factor)
# condition_fn receives the current parameter dict and returns True/False.

@dataclass
class AdaptationRule:
    description:   str
    condition:     object          # callable: (params: dict) -> bool
    target_weight: str             # which weight to amplify
    factor:        float           # amplification factor (e.g. 1.20)

ADAPTATION_RULES: list[AdaptationRule] = [
    # R1: High turbidity → amplify turbidity weight
    AdaptationRule(
        "High turbidity (P1 > 0.6) → amplify w1",
        lambda p: p["p1"] > 0.60,
        "w1", 1.20,
    ),
    # R2: Severe motion blur → amplify blur weight
    AdaptationRule(
        "Severe motion blur (P5 > 0.6) → amplify w5",
        lambda p: p["p5"] > 0.60,
        "w5", 1.15,
    ),
    # R3: High confidence variance → amplify model uncertainty weight
    AdaptationRule(
        "High confidence variance (P10 > 0.5) → amplify w10",
        lambda p: p["p10"] > 0.50,
        "w10", 1.20,
    ),
    # R4: Very low battery → amplify battery weight
    AdaptationRule(
        "Critical battery (P15 > 0.7) → amplify w15",
        lambda p: p["p15"] > 0.70,
        "w15", 1.50,
    ),
    # R5: High object density → amplify density weight
    AdaptationRule(
        "Dense scene (P8 > 0.5) → amplify w8",
        lambda p: p["p8"] > 0.50,
        "w8", 1.15,
    ),
    # R6: High dropout → amplify dropout weight
    AdaptationRule(
        "High detection dropout (P12 > 0.5) → amplify w12",
        lambda p: p["p12"] > 0.50,
        "w12", 1.25,
    ),
    # R7: High camera instability → amplify stability weight
    AdaptationRule(
        "High camera shake (P6 > 0.6) → amplify w6",
        lambda p: p["p6"] > 0.60,
        "w6", 1.20,
    ),
    # R8: Return mission phase → amplify mission phase weight
    AdaptationRule(
        "Late return phase (P16 > 0.8) → amplify w16",
        lambda p: p["p16"] > 0.80,
        "w16", 1.30,
    ),
    # R9: High trash priority → amplify trash priority weight
    AdaptationRule(
        "High-priority trash detected (P17 > 0.7) → amplify w17",
        lambda p: p["p17"] > 0.70,
        "w17", 1.40,
    ),
    # R10: High inference latency → amplify latency weight
    AdaptationRule(
        "High inference latency (P11 > 0.7) → amplify w11",
        lambda p: p["p11"] > 0.70,
        "w11", 1.20,
    ),
    # R11: Strong colour cast (turbid blue shift) → amplify colour weight
    AdaptationRule(
        "Strong colour cast (P7 > 0.5) → amplify w7",
        lambda p: p["p7"] > 0.50,
        "w7", 1.15,
    ),
    # R12: High false positive ratio → amplify FP weight
    AdaptationRule(
        "High false-positive ratio (P13 > 0.5) → amplify w13",
        lambda p: p["p13"] > 0.50,
        "w13", 1.20,
    ),
    # R13: Low remaining mission time → amplify time weight
    AdaptationRule(
        "Mission near completion (P19 > 0.8) → amplify w19",
        lambda p: p["p19"] > 0.80,
        "w19", 1.30,
    ),
]


# ─── ComplexityScoreComputer ──────────────────────────────────────────────────

class ComplexityScoreComputer:
    """
    Computes the Complexity Score from 20 normalised parameters.

    Parameters
    ----------
    alpha : float
        EMA smoothing coefficient (α = 0.3 in paper).
    use_dynamic_weights : bool
        Apply the 13 adaptive weight rules (default: True).
    """

    def __init__(self, alpha: float = 0.30, use_dynamic_weights: bool = True) -> None:
        self.alpha               = alpha
        self.use_dynamic_weights = use_dynamic_weights
        self._cs_smooth: float   = 0.5     # initialise at mid-range
        self._cs_raw_hist:    list[float] = []
        self._cs_smooth_hist: list[float] = []

    # ── Main entry ─────────────────────────────────────────────────────────────

    def update(self, params: dict[str, float]) -> tuple[float, float, dict]:
        """
        Compute CS for one frame.

        Returns
        -------
        cs_raw    : float  — unsmoothed score
        cs_smooth : float  — EMA-smoothed score  (CS̃)
        weights   : dict   — effective weights used this frame
        """
        weights = self._compute_adaptive_weights(params)
        cs_raw  = float(np.clip(
            sum(weights[f"w{i}"] * params[f"p{i}"] for i in range(1, 21)),
            0.0, 1.0,
        ))
        self._cs_smooth = (
            self.alpha * cs_raw + (1.0 - self.alpha) * self._cs_smooth
        )
        self._cs_raw_hist.append(cs_raw)
        self._cs_smooth_hist.append(self._cs_smooth)
        return cs_raw, self._cs_smooth, weights

    @property
    def cs_smooth(self) -> float:
        return self._cs_smooth

    @property
    def history_raw(self) -> list[float]:
        return list(self._cs_raw_hist)

    @property
    def history_smooth(self) -> list[float]:
        return list(self._cs_smooth_hist)

    def reset(self) -> None:
        """Reset EMA state (call between independent sequences)."""
        self._cs_smooth = 0.5

    # ── Adaptive weight computation ────────────────────────────────────────────

    def _compute_adaptive_weights(self, params: dict[str, float]) -> dict[str, float]:
        """
        Apply the 13 conditional amplification rules and re-normalise so
        weights still sum to 1.0.
        """
        weights = dict(BASE_WEIGHTS)  # copy

        if not self.use_dynamic_weights:
            return weights

        for rule in ADAPTATION_RULES:
            if rule.condition(params):
                weights[rule.target_weight] *= rule.factor

        # Re-normalise
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def top_contributors(
        self, params: dict[str, float], n: int = 5
    ) -> list[tuple[str, float]]:
        """
        Return the top-N (parameter_name, weighted_contribution) pairs
        for the current frame, using the effective adaptive weights.
        """
        weights = self._compute_adaptive_weights(params)
        contribs = [
            (f"P{i}", weights[f"w{i}"] * params[f"p{i}"])
            for i in range(1, 21)
        ]
        return sorted(contribs, key=lambda x: x[1], reverse=True)[:n]

    def category_breakdown(self, params: dict[str, float]) -> dict[str, float]:
        """
        Return scene / model / mission category totals for diagnostics.
        """
        weights = self._compute_adaptive_weights(params)
        return {
            "scene":   sum(weights[f"w{i}"] * params[f"p{i}"] for i in range(1,  9)),
            "model":   sum(weights[f"w{i}"] * params[f"p{i}"] for i in range(9,  15)),
            "mission": sum(weights[f"w{i}"] * params[f"p{i}"] for i in range(15, 21)),
        }
