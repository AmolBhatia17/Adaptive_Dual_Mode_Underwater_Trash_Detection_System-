# Paper — Adaptive Dual-Mode Underwater Trash Detection System

**Authors:** Amol Bhatia¹·*, Krisvarish V.²·⁺, Harsh Yadav², Avishkar Jaiswal², Shivani Gupta¹, Saurav Gupta¹  
¹ School of Computer Science and Engineering  
² School of Electronics Engineering

---

## Abstract

Underwater trash detection faces compounding challenges from light scattering, colour distortion, motion blur, and resource-constrained deployment on autonomous underwater vehicles. This paper introduces an **Adaptive Dual-Mode Detection Framework** that dynamically selects between a lightweight detector (YOLOv8n, 3.2 M parameters) and a heavyweight detector (YOLOv8x, 66.2 M parameters) by solving a formally defined mission utility optimisation problem that balances detection accuracy against energy consumption.

The core contribution is a **Complexity Score (CS)** that aggregates 20 normalised parameters — covering scene attributes (40% weight), model-feedback signals (35%), and mission constraints (25%) — into a single scalar optimisation signal. A **3-Layer Intelligent Switching Controller** uses asymmetric hysteresis thresholds, time-based feature validation, and statistical stability detection to minimise unnecessary model transitions while maintaining near-heavyweight detection quality.

Experiments on the Underwater Trash Detection Dataset and validation on the TrashCAN 1.0 benchmark demonstrate that the proposed system retains **97.8% of YOLOv8x accuracy** (mAP@0.5 = 0.91) while achieving **69% energy savings** relative to continuous heavyweight operation, at a controller overhead of **less than 1 ms per frame**. Statistical significance is confirmed by independent t-tests (p < 0.001) and Wilcoxon signed-rank tests with large effect sizes (Cohen's d ≈ 3.0).

---

## Objective Function

At each frame t, the system selects M_t ∈ {M_L, M_H} to maximise:

$$J = \frac{1}{T} \sum_{t=1}^{T} \left[ \lambda \cdot A(M_t, t) - (1-\lambda) \cdot \frac{E(M_t)}{E_{\max}} \right]$$

subject to:
- M_t ∈ {M_L, M_H}
- |{t : M_t ≠ M_{t−1}}| / T ≤ r_max  (switch rate constraint)
- B_t > B_crit for all t  (energy safety constraint)

**Parameters:** λ = 0.6, E_max = 50 W, B_crit = 0.20, r_max = 0.01

---

## Table 1 — Base Parameter Weights

| # | Parameter | Category | Base Weight | Physical Meaning |
|---|---|---|---|---|
| P1 | Turbidity | Scene | 8% | Blue-channel attenuation |
| P2 | Lighting Variation | Scene | 6% | Brightness CV in L* channel |
| P3 | Texture Richness | Scene | 7% | Laplacian variance |
| P4 | Occlusion Level | Scene | 7% | Inter-object IoU |
| P5 | Motion Blur | Scene | 5% | Inverse sharpness |
| P6 | Camera Stability | Scene | 4% | Optical flow magnitude |
| P7 | Colour Cast | Scene | 6% | Max channel-mean difference |
| P8 | Object Density | Scene | 5% | Objects per frame area |
| P9 | Low Confidence | Model | 8% | 1 − mean detection confidence |
| P10 | Confidence Variance | Model | 5% | Normalised conf. variance |
| P11 | Inference Latency | Model | 4% | Normalised wall-clock time |
| P12 | Detection Dropout | Model | 6% | Fraction of skipped frames |
| P13 | False Positive Ratio | Model | 4% | Low-conf. detections fraction |
| P14 | BBox Instability | Model | 5% | 1 − inter-frame IoU |
| P15 | Battery (inv.) | Mission | 6% | 1 − battery level |
| P16 | Mission Phase | Mission | 5% | 0=explore, 1=return |
| P17 | Trash Priority | Mission | 3% | Priority weighting |
| P18 | Distance to Zone | Mission | 2% | Coverage zone proximity |
| P19 | Time Remaining (inv.) | Mission | 2% | 1 − time remaining |
| P20 | Bandwidth (inv.) | Mission | 2% | 1 − bandwidth available |

**Σ = 1.0** · Dynamic amplification by 13 context-sensitive rules; re-normalised after each update.

---

## Table 2 — Model Parameter and Efficiency Comparison

| Model | mAP@0.5 | Params (M) | FLOPs (B) | Inference (ms) | Energy | Efficiency Score |
|---|---|---|---|---|---|---|
| YOLOv8n | 0.870 | 3.2 | 8.7 | 18 | Low | 0.82 |
| YOLOv8x | 0.930 | 66.2 | 257.8 | 56 | Very High | 0.65 |
| YOLOv10 | 0.925 | ~16.0 | ~63.0 | 48 | High | 0.70 |
| YOLOv12x | 0.940 | 71.6 | 199.0 | 60 | Very High | 0.60 |
| **Adaptive (Proposed)** | **0.910** | **3.2↔66.2** | **Dynamic** | **24.2** | **Optimised** | **0.90** |

*mAP@0.5 for adaptive system = 0.91 on merged test set (97.8% of YOLOv8x accuracy)*

---

## Table 3 — Switch Frequency and Confidence Across Ablation Variants

| Configuration | Switches/1000 fr | Avg. Confidence | Confidence Var. |
|---|---|---|---|
| **Full System (Proposed)** | **4** | **0.416** | **0.021** |
| w/o Stability Layer (L3) | 12 | 0.409 | 0.037 |
| w/o Temporal Validation (L2) | 11 | 0.402 | 0.041 |
| w/o Asymmetric Thresholds (L1) | 9 | 0.410 | 0.034 |
| w/o Dynamic Weights | 7 | 0.404 | 0.029 |

---

## Table 4 — Energy and Detection Across Ablation Variants

| Configuration | mAP@0.5 | Energy Savings (%) | Efficiency Score |
|---|---|---|---|
| **Full System (Proposed)** | **0.910** | **69.0** | **0.90** |
| w/o Stability Layer (L3) | 0.900 | 63.5 | 0.84 |
| w/o Temporal Validation (L2) | 0.890 | 65.2 | 0.81 |
| w/o Asymmetric Thresholds (L1) | 0.900 | 61.5 | 0.83 |
| w/o Dynamic Weights | 0.890 | 60.0 | 0.79 |
| Static YOLOv8n | 0.870 | 70.2 | 0.82 |
| Static YOLOv8x | 0.930 | 0.0 | 0.65 |

*Bootstrapped 95% CI: ±0.012 on mAP, ±0.015 on mean confidence (10,000 iterations)*

---

## Table 5 — Statistical Significance (Adaptive vs Baselines)

| Metric | YOLOv8n | YOLOv8x | Adaptive | t-stat | p-value | Wilcoxon p | Cohen's d |
|---|---|---|---|---|---|---|---|
| mAP@0.5 | 0.870 | 0.930 | 0.910 | 29.91 | < 0.001 | < 0.001 | 2.99 (large) |
| mAP@[0.5:0.95] | 0.710 | 0.770 | 0.740 | — | — | — | — |
| Latency (ms/fr) | 18.0 | 56.0 | 24.2 | −51.88 | < 0.001 | < 0.001 | −5.19 (large) |
| Energy Savings (%) | — | 0.0 | 69.0 | — | — | — | — |
| Avg Confidence | 0.416 | 0.461 | 0.416 | — | — | — | — |
| Switches/1k fr | 0 | 0 | 4 | — | — | — | — |
| Efficiency Score | 0.82 | 0.65 | 0.90 | — | — | — | — |

*Independent two-sample t-tests (two-tailed, n=200 per group, bootstrapped 10,000 iterations).  
All p-values pass Bonferroni correction: α_corrected = 0.001/6 = 0.000167.*

---

## Table 6 — Layer-Combination Ablation (All 7 Configurations)

| Configuration | mAP@0.5 | Energy Sav. (%) | Switches/1k fr | Avg Conf | Conf Var | Eff Score |
|---|---|---|---|---|---|---|
| Layer 1 only (Hysteresis) | 0.891 | 34.1 | 3.3 | 0.439 | 0.024 | 0.770 |
| Layer 2 only (Temporal) | 0.874 | 70.0 | 0.0 | 0.416 | 0.024 | 0.964 |
| Layer 3 only (Stability) | 0.876 | 70.0 | 0.0 | 0.416 | 0.024 | 0.965 |
| Layers 1+2 | 0.891 | 46.0 | 6.7 | 0.431 | 0.023 | 0.838 |
| Layers 1+3 | 0.890 | 53.2 | 6.7 | 0.427 | 0.023 | 0.878 |
| Layers 2+3 | 0.880 | 70.0 | 0.0 | 0.416 | 0.023 | 0.968 |
| **All Layers — PROPOSED** | **0.910** | **69.0** | **4.0** | **0.416** | **0.021** | **0.900** |

**Analysis:** Layer 1 alone gives reactivity but insufficient downgrade selectivity (high switches, moderate savings). Layers 2 and 3 alone never upgrade to YOLOv8x (pure downgrade mechanisms) — maximum savings but lower mAP on complex scenes. The full 3-layer system achieves the best mAP (0.910) and lowest confidence variance (0.021) with stable switch rate of 4/1000 frames — demonstrating all three layers contribute complementary, non-redundant control logic.

---

## Layer 2 Feature Validation Thresholds

| Parameter | Physical Meaning | Threshold |
|---|---|---|
| P1 | Water turbidity | < 0.50 |
| P3 | Background complexity | < 0.55 |
| P5 | Motion magnitude | < 0.60 |
| P6 | Camera stability | < 0.55 |
| P8 | Edge sharpness (blur proxy) | < 0.50 |

Downgrade triggered when ≥ 4/5 parameters consistent below threshold for ≥ 80% of last W=20 frames, after T_min = 50 frames in hysteresis.

---

## Controller Threshold Configuration

| Parameter | Value | Rationale |
|---|---|---|
| τ_down | 0.40 | 25th percentile of CS on training set |
| τ_up | 0.55 | 65th percentile of CS on training set |
| Hysteresis gap Δ | 0.15 | Reduced from conventional 0.20 for faster response |
| Rate limiter | 30 frames | Prevents oscillatory switching |
| EMA α | 0.30 | Balances responsiveness vs. noise suppression |
| B_crit | 0.20 | Hard override battery threshold |
| Layer 2 dwell | 50 frames | ≈ 1.67 s at 30 FPS |
| Layer 3 window W | 20 frames | σ_CS stability window |
| Layer 3 σ threshold | 0.03 | ±0.06 CS variation = stable |
| Adaptive update interval | 100 frames | Threshold calibration cadence |

---

## Key Quantitative Results

| Metric | Value |
|---|---|
| mAP@0.5 (Adaptive) | **0.910** (97.8% of YOLOv8x) |
| mAP@[0.5:0.95] | 0.740 |
| Precision | 0.930 |
| Recall | 0.900 |
| F1-Score | 0.910 |
| Avg. Confidence | 0.416 |
| Avg. Inference Latency | **24.2 ms/frame** |
| Controller Overhead | **< 1 ms/frame** |
| Energy Savings vs YOLOv8x | **69.0%** |
| Frames in Lightweight Mode | **81%** |
| Frames in Heavyweight Mode | 19% |
| Switches per 1000 frames | **4.0** |
| Efficiency Score E_score | **0.90** |

---

## Reproducibility

All code, training configs, evaluation scripts, and the synthetic video generator are in this repository.
To reproduce the main result without a GPU:

```bash
python generate_synthetic_video.py         # 300-frame synthetic underwater video
python run_adaptive_test.py                # Full pipeline + all result plots
python evaluation/run_statistical_tests.py # Table 5
python evaluation/run_ablation.py          # Table 6
python evaluation/benchmark_models.py --simulate  # Table 2
python evaluation/generate_plots.py        # All 8 paper figures
pytest tests/ -v                           # 102 unit tests
```

---

## Citation

```bibtex
@article{bhatia2025deepclean,
  title   = {Adaptive Dual-Mode Underwater Trash Detection System},
  author  = {Bhatia, Amol and V., Krisvarish and Yadav, Harsh and
             Jaiswal, Avishkar and Gupta, Shivani and Gupta, Saurav},
  journal = {arXiv preprint},
  year    = {2025}
}
```

---

## References (abbreviated)

1. Er et al. "Deep learning-based underwater marine object detection: A review." *Sensors* 2023.
2. Dogra et al. "A comprehensive survey on underwater object detection." *Neural Computing and Applications* 2024.
6. Redmon et al. "You only look once: Unified, real-time object detection." *CVPR* 2016.
8. Terven & Cordova-Esparza. "A comprehensive review of YOLO architectures." *MLKE* 2024.
9. Hong et al. "TrashCAN: A semantically-segmented dataset for marine debris." *arXiv* 2020.
18. Gao et al. "Dual dynamic inference." *IEEE J. Sel. Topics Signal Process.* 2023.
26. Recht. "A tour of reinforcement learning: The view from continuous control." *ARCRAS* 2019.
27. Wang & Liao. "YOLOv9: Learning What You Want to Learn." *arXiv* 2024.
28. Tian et al. "YOLOv12: Attention-Centric Real-Time Object Detectors." *arXiv* 2025.
