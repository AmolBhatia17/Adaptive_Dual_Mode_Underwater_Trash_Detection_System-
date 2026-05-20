"""
Mission Utility Objective Function
=====================================
Implements the formal optimisation objective defined in Section 3.0 of the paper:

    J = (1/T) Σ_{t=1}^{T} [ λ · A(Mₜ, t) − (1−λ) · E(Mₜ)/E_max ]

subject to:
    Mₜ ∈ {M_L, M_H}
    switch rate  ≤ r_max
    Bₜ > B_crit  for all t

Parameters
----------
    λ       = 0.6    accuracy-energy trade-off weight
    E_max   = 50 W   peak per-frame power (YOLOv8x)
    B_crit  = 0.20   critical battery fraction (triggers forced lightweight)
    r_max   = 0.01   maximum switch fraction per total frames

This module also provides an `EnergyModel` for estimating per-frame
energy consumption and a `UtilityEvaluator` that computes J offline
over a recorded model-selection history.
"""

from dataclasses import dataclass
from typing import Sequence
import numpy as np


# ─── Energy model ─────────────────────────────────────────────────────────────

@dataclass
class EnergyModel:
    """Per-frame average power draw for each model (Watts)."""
    power_lightweight:  float = 15.0    # YOLOv8n
    power_heavyweight:  float = 50.0    # YOLOv8x / YOLOv12x
    power_yolov10:      float = 35.0    # YOLOv10m (intermediate)
    power_yolov12n:     float = 16.0    # YOLOv12n (near-lightweight)
    fps:                float = 30.0    # inference frame rate

    def energy_per_frame(self, model_name: str) -> float:
        """Return Joules consumed per frame for `model_name`."""
        power_map = {
            "YOLOv8n":  self.power_lightweight,
            "YOLOv8x":  self.power_heavyweight,
            "YOLOv10":  self.power_yolov10,
            "YOLOv12n": self.power_yolov12n,
            "YOLOv12x": self.power_heavyweight,
        }
        watts = power_map.get(model_name, self.power_heavyweight)
        return watts / self.fps   # W / (frames/s) = J/frame

    @property
    def e_max(self) -> float:
        """Peak energy per frame (normalization denominator)."""
        return self.power_heavyweight / self.fps


# ─── Objective function evaluator ─────────────────────────────────────────────

class UtilityEvaluator:
    """
    Evaluate mission utility J over a recorded inference sequence.

    Parameters
    ----------
    lam     : float  — accuracy weight λ (default 0.6)
    energy  : EnergyModel
    b_crit  : float  — critical battery fraction (default 0.20)
    r_max   : float  — max allowable switch rate (default 0.01)
    """

    # Empirical per-model mAP@0.5 on the merged test set
    MAP_TABLE: dict[str, float] = {
        "YOLOv8n":  0.870,
        "YOLOv8x":  0.930,
        "YOLOv10":  0.925,
        "YOLOv12n": 0.875,
        "YOLOv12x": 0.940,
        "Adaptive":  0.910,   # measured adaptive system
    }

    def __init__(
        self,
        lam:    float       = 0.60,
        energy: EnergyModel | None = None,
        b_crit: float       = 0.20,
        r_max:  float       = 0.01,
    ) -> None:
        self.lam    = lam
        self.energy = energy or EnergyModel()
        self.b_crit = b_crit
        self.r_max  = r_max

    # ── Main evaluation ────────────────────────────────────────────────────────

    def evaluate(
        self,
        model_history:    Sequence[str],
        accuracy_history: Sequence[float] | None = None,
        battery_history:  Sequence[float] | None = None,
    ) -> dict:
        """
        Compute J and associated diagnostics over a complete mission.

        Parameters
        ----------
        model_history    : sequence of model names per frame
        accuracy_history : per-frame accuracy proxy in [0,1]; if None, uses MAP_TABLE
        battery_history  : per-frame battery fraction in [0,1]; if None, assumes 1.0

        Returns
        -------
        dict with keys:
            J                  — mission utility scalar
            constraint_ok      — True if all hard constraints satisfied
            energy_savings_pct — % energy saved vs always-heavy baseline
            switch_rate        — switches per total frame
            safety_violations  — # frames where Bₜ ≤ B_crit with heavy model
            frame_utilities    — per-frame utility values
        """
        T = len(model_history)
        if T == 0:
            return {"J": 0.0, "constraint_ok": False, "error": "empty history"}

        e_max = self.energy.e_max

        # Per-frame utility terms
        frame_utilities: list[float] = []
        energy_total    = 0.0
        switch_count    = 0
        safety_violations = 0

        for t, model in enumerate(model_history):
            # Accuracy
            if accuracy_history is not None:
                A = float(accuracy_history[t])
            else:
                A = self.MAP_TABLE.get(model, 0.90)

            # Energy
            E = self.energy.energy_per_frame(model)
            energy_total += E

            # Battery constraint check
            if battery_history is not None:
                B_t = float(battery_history[t])
                if B_t <= self.b_crit and model != "YOLOv8n":
                    safety_violations += 1

            # Utility
            u = self.lam * A - (1.0 - self.lam) * (E / e_max)
            frame_utilities.append(u)

            # Count switches
            if t > 0 and model_history[t] != model_history[t - 1]:
                switch_count += 1

        J = float(np.mean(frame_utilities))

        # Always-heavyweight baseline energy
        energy_baseline = T * self.energy.energy_per_frame("YOLOv8x")
        energy_savings_pct = max(
            0.0,
            (energy_baseline - energy_total) / energy_baseline * 100.0,
        )

        switch_rate = switch_count / T

        # Constraint satisfaction
        constraint_ok = (
            switch_rate     <= self.r_max
            and safety_violations == 0
        )

        return {
            "J":                   round(J, 5),
            "constraint_ok":       constraint_ok,
            "energy_savings_pct":  round(energy_savings_pct, 2),
            "switch_rate":         round(switch_rate, 5),
            "switch_count":        switch_count,
            "safety_violations":   safety_violations,
            "energy_total_J":      round(energy_total, 3),
            "frame_utilities":     frame_utilities,
            "lambda":              self.lam,
        }

    def compare_baselines(self, model_history: Sequence[str]) -> dict[str, dict]:
        """
        Compare the adaptive system against fixed-model baselines.

        Returns a dict keyed by system name, each value being the evaluate() dict.
        """
        T = len(model_history)

        results: dict[str, dict] = {}

        # Adaptive (actual recorded history)
        results["Adaptive"] = self.evaluate(model_history)

        # Fixed baselines
        for model in ("YOLOv8n", "YOLOv8x", "YOLOv10", "YOLOv12n", "YOLOv12x"):
            fixed_hist = [model] * T
            results[model] = self.evaluate(fixed_hist)

        return results

    def print_report(self, comparison: dict[str, dict]) -> None:
        """Print a formatted comparison table."""
        header = f"{'System':<14} {'J':>7} {'mAP(proxy)':>11} {'Energy Sav%':>12} {'SwitchRate':>11} {'OK':>4}"
        print("=" * len(header))
        print(header)
        print("─" * len(header))

        for name, res in comparison.items():
            J        = res.get("J", 0)
            savings  = res.get("energy_savings_pct", 0)
            sw_rate  = res.get("switch_rate", 0)
            ok       = "✓" if res.get("constraint_ok", False) else "✗"
            map_val  = self.MAP_TABLE.get(name, "—")
            print(
                f"{name:<14} {J:>7.4f} {str(map_val):>11} "
                f"{savings:>11.1f}% {sw_rate:>11.5f} {ok:>4}"
            )

        print("=" * len(header))
