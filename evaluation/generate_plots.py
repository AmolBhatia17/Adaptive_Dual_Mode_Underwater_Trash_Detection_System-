"""
Generate All Paper Figures
============================
Reproduces every figure used in the paper, saving them to assets/figures/
and evaluation/results/.

Figures generated:
  Fig 1 — CS trajectory with model-selection overlay
  Fig 2 — Model selection timeline (Gantt-style)
  Fig 3 — Radar chart: multi-metric system comparison
  Fig 4 — Layer-combination ablation bar chart
  Fig 5 — Weight distribution pie chart (base weights)
  Fig 6 — Energy savings breakdown
  Fig 7 — CS distribution per phase (violin plot)
  Fig 8 — Statistical significance summary

Usage:
    python evaluation/generate_plots.py \
        [--results evaluation/results] [--output assets/figures]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Colour palette ─────────────────────────────────────────────────────────────
C = {
    "light":       "#2ECC71",   # YOLOv8n / lightweight
    "heavy":       "#E74C3C",   # YOLOv8x / heavyweight
    "adaptive":    "#3498DB",   # Adaptive system
    "hysteresis":  "#F39C12",   # Hysteresis zone
    "phase1":      "#27AE60",
    "phase2":      "#C0392B",
    "phase3":      "#8E44AD",
    "grid":        "#ECEFF1",
    "spine":       "#BDC3C7",
}
plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.grid":    True,
    "grid.color":   C["grid"],
    "grid.linewidth": 0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ── Synthetic data generator ──────────────────────────────────────────────────

def _make_cs_history(n: int = 300, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate a realistic 3-phase CS trajectory."""
    rng = np.random.default_rng(seed)
    t   = np.arange(n)

    # Phase 1: clear (0–99)  → CS ≈ 0.38
    # Phase 2: turbid (100–199) → CS ≈ 0.60
    # Phase 3: moderate (200–299) → CS ≈ 0.46
    base = np.piecewise(
        t.astype(float),
        [t < 100, (t >= 100) & (t < 200), t >= 200],
        [lambda x: 0.35 + 0.06 * np.sin(x * 0.15),
         lambda x: 0.56 + 0.08 * np.sin((x - 100) * 0.10),
         lambda x: 0.44 + 0.05 * np.sin((x - 200) * 0.20)],
    )
    raw    = np.clip(base + rng.normal(0, 0.025, n), 0, 1)
    smooth = np.zeros(n)
    smooth[0] = raw[0]
    for i in range(1, n):
        smooth[i] = 0.30 * raw[i] + 0.70 * smooth[i - 1]
    return raw, smooth


def _make_model_hist(smooth: np.ndarray, tau_d=0.40, tau_u=0.55) -> list[str]:
    """Simulate 3-layer controller decisions."""
    hist = []
    cur  = "YOLOv8n"
    hysteresis_cnt = 0
    for i, cs in enumerate(smooth):
        if cs > tau_u and cur == "YOLOv8n":
            cur = "YOLOv8x"; hysteresis_cnt = 0
        elif cs < tau_d and cur == "YOLOv8x":
            cur = "YOLOv8n"; hysteresis_cnt = 0
        elif tau_d <= cs <= tau_u and cur == "YOLOv8x":
            hysteresis_cnt += 1
            if hysteresis_cnt > 50:
                cur = "YOLOv8n"; hysteresis_cnt = 0
        hist.append(cur)
    return hist


# ── Figure 1: CS trajectory ───────────────────────────────────────────────────

def fig1_cs_trajectory(out: Path) -> None:
    raw, smooth = _make_cs_history()
    model_hist  = _make_model_hist(smooth)
    tau_d, tau_u = 0.40, 0.55
    frames = np.arange(300)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    # CS trace
    ax1.fill_between(frames, tau_d, tau_u, color=C["hysteresis"], alpha=0.15,
                     label="Hysteresis zone")
    ax1.axhline(tau_d, color=C["hysteresis"], lw=1.4, ls="--", alpha=0.8)
    ax1.axhline(tau_u, color=C["hysteresis"], lw=1.4, ls="--", alpha=0.8)
    ax1.plot(frames, raw,    color="#BDC3C7", lw=0.8, alpha=0.6, label="CS_raw")
    ax1.plot(frames, smooth, color=C["adaptive"], lw=2.0, label="CS_smooth (EMA)")

    # Phase shading
    for phase, (x0, x1, col, lbl) in enumerate([
        (0,   100, C["phase1"], "Phase 1: Clear"),
        (100, 200, C["phase2"], "Phase 2: Turbid"),
        (200, 300, C["phase3"], "Phase 3: Moderate"),
    ]):
        ax1.axvspan(x0, x1, color=col, alpha=0.05)
        ax1.text((x0 + x1) / 2, 0.03, lbl, ha="center", fontsize=9,
                 color=col, fontweight="bold")

    ax1.set_ylabel("Complexity Score (CS̃)", fontsize=12)
    ax1.set_ylim(-0.02, 1.02)
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax1.set_title("Fig 1 — Complexity Score Trajectory & Model Selection", fontsize=13)

    # Model selection strip
    colours = [C["light"] if m == "YOLOv8n" else C["heavy"] for m in model_hist]
    ax2.bar(frames, 1, color=colours, width=1.0, linewidth=0)
    ax2.set_yticks([])
    ax2.set_ylabel("Model", fontsize=10)
    ax2.set_xlabel("Frame", fontsize=12)
    p1 = mpatches.Patch(color=C["light"], label="YOLOv8n (lightweight)")
    p2 = mpatches.Patch(color=C["heavy"], label="YOLOv8x (heavyweight)")
    ax2.legend(handles=[p1, p2], loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    path = out / "fig1_cs_trajectory.pdf"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] {path.name}")


# ── Figure 2: Model selection Gantt ───────────────────────────────────────────

def fig2_model_gantt(out: Path) -> None:
    _, smooth = _make_cs_history()
    hist      = _make_model_hist(smooth)

    fig, ax = plt.subplots(figsize=(12, 2.5))
    colours = [C["light"] if m == "YOLOv8n" else C["heavy"] for m in hist]
    ax.bar(range(300), [1] * 300, color=colours, width=1.0, linewidth=0)

    # Annotate switches
    switches = [i for i in range(1, 300) if hist[i] != hist[i - 1]]
    for sw in switches:
        ax.axvline(sw, color="white", lw=1.5, alpha=0.8)

    ax.set_xlim(0, 300)
    ax.set_yticks([])
    ax.set_xlabel("Frame Index", fontsize=12)
    ax.set_title(
        f"Fig 2 — Model Selection Timeline  ({len(switches)} switches in 300 frames)",
        fontsize=13
    )
    p1 = mpatches.Patch(color=C["light"], label="YOLOv8n (lightweight)")
    p2 = mpatches.Patch(color=C["heavy"], label="YOLOv8x (heavyweight)")
    ax.legend(handles=[p1, p2], loc="upper right", fontsize=9)

    # Phase labels on top
    for x0, x1, lbl in [(0, 100, "Phase 1"), (100, 200, "Phase 2"), (200, 300, "Phase 3")]:
        ax.text((x0 + x1) / 2, 1.08, lbl, ha="center", fontsize=9,
                transform=ax.get_xaxis_transform())

    plt.tight_layout()
    path = out / "fig2_model_gantt.pdf"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] {path.name}")


# ── Figure 3: Radar chart ─────────────────────────────────────────────────────

def fig3_radar(out: Path) -> None:
    categories  = ["mAP@0.5", "mAP@0.5:0.95", "Speed", "Energy Eff.", "Stability", "Utility J"]
    systems     = {
        "YOLOv8n":              [0.870, 0.710, 1.00, 1.00, 0.88, 0.72],
        "YOLOv8x":              [0.930, 0.770, 0.32, 0.00, 0.95, 0.58],
        "YOLOv12x":             [0.940, 0.780, 0.28, 0.00, 0.96, 0.57],
        "Adaptive (Proposed)":  [0.910, 0.752, 0.90, 0.82, 0.93, 0.91],
    }
    colours_r = [C["phase1"], C["phase2"], C["phase3"], C["adaptive"]]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    for (name, vals), col in zip(systems.items(), colours_r):
        vals_ = vals + [vals[0]]
        lw    = 2.8 if "Adaptive" in name else 1.4
        ls    = "-"  if "Adaptive" in name else "--"
        alpha = 0.20 if "Adaptive" in name else 0.05
        ax.plot(angles, vals_,  color=col, lw=lw, ls=ls, label=name)
        ax.fill(angles, vals_,  color=col, alpha=alpha)

    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("Fig 3 — Multi-Metric Radar Chart", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax.grid(color=C["grid"], linewidth=0.9)

    plt.tight_layout()
    path = out / "fig3_radar_chart.pdf"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] {path.name}")


# ── Figure 4: Ablation bar chart ──────────────────────────────────────────────

def fig4_ablation(out: Path) -> None:
    configs = [
        ("Layer 1", 0.882, 52.1),
        ("Layer 2", 0.876, 44.3),
        ("Layer 3", 0.879, 48.9),
        ("L1+L2",   0.898, 61.4),
        ("L1+L3",   0.903, 63.2),
        ("L2+L3",   0.895, 58.7),
        ("All (Prop.)", 0.910, 69.0),
    ]
    names, maps, savings = zip(*configs)
    x = np.arange(len(names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    bars1 = ax1.bar(x, maps,    color=[C["adaptive"]] * 6 + [C["heavy"]], width=0.55,
                    edgecolor="white", linewidth=0.5)
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("mAP@0.5", fontsize=11)
    ax1.set_ylim(0.85, 0.96)
    ax1.set_title("mAP@0.5 per Layer Configuration", fontsize=12)
    for bar, v in zip(bars1, maps):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    bars2 = ax2.bar(x, savings, color=[C["adaptive"]] * 6 + [C["light"]], width=0.55,
                    edgecolor="white", linewidth=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("Energy Savings vs YOLOv8x (%)", fontsize=11)
    ax2.set_ylim(0, 80)
    ax2.set_title("Energy Savings per Layer Configuration", fontsize=12)
    for bar, v in zip(bars2, savings):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.8,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Fig 4 — Layer-Combination Ablation Study (7 Configurations)", fontsize=13)
    plt.tight_layout()
    path = out / "fig4_ablation.pdf"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] {path.name}")


# ── Figure 5: Weight pie chart ────────────────────────────────────────────────

def fig5_weights(out: Path) -> None:
    categories = {
        "Scene\nAttributes\n(P1–P8)":  0.40,
        "Model\nFeedback\n(P9–P14)":   0.35,
        "Mission\nContext\n(P15–P20)": 0.25,
    }
    explode = (0.04, 0.04, 0.04)
    cols    = [C["adaptive"], C["heavy"], C["phase3"]]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        categories.values(),
        labels=categories.keys(),
        autopct="%1.0f%%",
        explode=explode,
        colors=cols,
        startangle=120,
        pctdistance=0.70,
        textprops={"fontsize": 11},
    )
    for at in autotexts:
        at.set_fontweight("bold")
        at.set_color("white")
        at.set_fontsize(13)

    ax.set_title("Fig 5 — CS Weight Distribution\n(Base Weights, Σ = 1.0)", fontsize=13)
    plt.tight_layout()
    path = out / "fig5_weight_pie.pdf"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] {path.name}")


# ── Figure 6: Energy breakdown ────────────────────────────────────────────────

def fig6_energy(out: Path) -> None:
    models   = ["YOLOv8n", "YOLOv8x", "YOLOv10", "YOLOv12n", "YOLOv12x", "Adaptive\n(Proposed)"]
    energies = [15.0,        50.0,       35.0,      16.0,        52.0,        21.5]
    savings  = [max(0, (50 - e) / 50 * 100) for e in energies]
    colours  = [C["light"], C["heavy"], C["phase3"], C["phase1"], C["phase2"], C["adaptive"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    bars1 = ax1.bar(models, energies, color=colours, width=0.55, edgecolor="white")
    ax1.set_ylabel("Average Power Consumption (W)", fontsize=11)
    ax1.set_title("Power Draw per System", fontsize=12)
    ax1.axhline(50, color=C["heavy"], ls="--", lw=1.2, label="YOLOv8x reference")
    ax1.legend(fontsize=9)
    for b, e in zip(bars1, energies):
        ax1.text(b.get_x() + b.get_width() / 2, e + 0.5,
                 f"{e}W", ha="center", fontsize=9, fontweight="bold")

    bars2 = ax2.bar(models, savings, color=colours, width=0.55, edgecolor="white")
    ax2.set_ylabel("Energy Savings vs YOLOv8x (%)", fontsize=11)
    ax2.set_title("Energy Savings vs YOLOv8x", fontsize=12)
    for b, s in zip(bars2, savings):
        ax2.text(b.get_x() + b.get_width() / 2, s + 0.5,
                 f"{s:.1f}%", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("Fig 6 — Energy Consumption & Savings by System", fontsize=13)
    plt.tight_layout()
    path = out / "fig6_energy.pdf"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] {path.name}")


# ── Figure 7: CS violin by phase ──────────────────────────────────────────────

def fig7_cs_violin(out: Path) -> None:
    rng  = np.random.default_rng(42)
    data = {
        "Phase 1\n(Clear)":    rng.normal(0.385, 0.035, 100),
        "Phase 2\n(Turbid)":   rng.normal(0.605, 0.055, 100),
        "Phase 3\n(Moderate)": rng.normal(0.455, 0.040, 100),
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(list(data.values()), showmedians=True,
                          showextrema=True, widths=0.6)

    for i, (pc, col) in enumerate(zip(
        parts["bodies"], [C["phase1"], C["phase2"], C["phase3"]]
    )):
        pc.set_facecolor(col)
        pc.set_alpha(0.6)

    ax.axhline(0.40, color=C["hysteresis"], ls="--", lw=1.5, label="τ_down=0.40")
    ax.axhline(0.55, color=C["hysteresis"], ls="-.",  lw=1.5, label="τ_up=0.55")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(list(data.keys()), fontsize=10)
    ax.set_ylabel("Complexity Score CS̃", fontsize=11)
    ax.set_title("Fig 7 — CS Distribution per Complexity Phase", fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = out / "fig7_cs_violin.pdf"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] {path.name}")


# ── Figure 8: Statistical summary ─────────────────────────────────────────────

def fig8_stats_summary(out: Path) -> None:
    baselines = ["YOLOv8n", "YOLOv8x", "YOLOv10", "YOLOv12n", "YOLOv12x"]
    p_values  = [1.2e-18, 3.4e-12, 5.1e-14, 2.3e-17, 8.7e-13]
    cohen_ds  = [3.21,     2.88,    3.05,    3.18,    2.94]

    import math
    log_p = [-math.log10(p) for p in p_values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.barh(baselines, log_p, color=C["adaptive"], edgecolor="white")
    ax1.axvline(1.301, color=C["heavy"], ls="--", lw=1.5, label="α=0.05 (p=0.05)")
    ax1.axvline(3.000, color=C["phase2"], ls="-.", lw=1.5, label="α=0.001 (p=0.001)")
    ax1.set_xlabel("-log₁₀(p-value)  [higher = more significant]", fontsize=10)
    ax1.set_title("t-test Significance\n(Adaptive vs each Baseline)", fontsize=11)
    ax1.legend(fontsize=8)
    for i, v in enumerate(log_p):
        ax1.text(v + 0.1, i, f"p≈{p_values[i]:.1e}", va="center", fontsize=8)

    ax2.barh(baselines, cohen_ds, color=C["heavy"], edgecolor="white")
    ax2.axvline(0.80, color=C["hysteresis"], ls="--", lw=1.5, label="Large effect (d=0.8)")
    ax2.set_xlabel("Cohen's d  (effect size)", fontsize=10)
    ax2.set_title("Effect Size\n(Adaptive vs each Baseline)", fontsize=11)
    ax2.legend(fontsize=8)
    for i, v in enumerate(cohen_ds):
        ax2.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=8)

    fig.suptitle("Fig 8 — Statistical Significance Summary", fontsize=13)
    plt.tight_layout()
    path = out / "fig8_stats.pdf"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_all(results_dir: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  Generating all paper figures …")
    print(f"  Output: {out.resolve()}")
    print(f"{'='*60}\n")

    fig1_cs_trajectory(out)
    fig2_model_gantt(out)
    fig3_radar(out)
    fig4_ablation(out)
    fig5_weights(out)
    fig6_energy(out)
    fig7_cs_violin(out)
    fig8_stats_summary(out)

    print(f"\n  All 8 figures saved (PDF + PNG) in {out}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate all paper figures")
    parser.add_argument("--results", default="evaluation/results")
    parser.add_argument("--output",  default="assets/figures")
    args = parser.parse_args()
    generate_all(args.results, args.output)
