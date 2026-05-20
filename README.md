# DeepClean — Adaptive Dual-Mode Underwater Trash Detection System

<p align="center">
  <img src="assets/figures/banner.png" alt="DeepClean Banner" width="800"/>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python"/></a>
  <a href="#"><img src="https://img.shields.io/badge/YOLOv8-Ultralytics-orange?logo=pytorch" alt="YOLO"/></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"/></a>
  <a href="#"><img src="https://img.shields.io/badge/mAP%400.5-0.91-brightgreen" alt="mAP"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Energy%20Savings-69%25-blue" alt="Energy"/></a>
  <a href="#"><img src="https://img.shields.io/badge/Controller%20Overhead-%3C1ms-yellow" alt="Latency"/></a>
  <a href="https://colab.research.google.com/github/your-org/DeepClean-AdaptiveDualMode/blob/main/notebooks/DeepClean_v4_Colab.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
</p>

---

## Overview

**DeepClean** is a real-time adaptive underwater trash detection framework that dynamically switches between a lightweight detector (YOLOv8n, 3.2 M params) and a heavyweight detector (YOLOv8x, 66.2 M params) by solving a formally defined mission-utility optimisation problem. The core novelty is a **Complexity Score (CS)** — a single scalar computed from 20 normalised scene, model-feedback, and mission-constraint parameters — which drives a **3-Layer Intelligent Switching Controller** designed for energy-constrained autonomous underwater vehicles (AUVs).

> **Paper:** *Adaptive Dual-Mode Underwater Trash Detection System* — Amol Bhatia, Krisvarish V., Harsh Yadav, Avishkar Jaiswal, Shivani Gupta, Saurav Gupta  
> School of Computer Science and Engineering / School of Electronics Engineering

---

## Key Results

| Metric | YOLOv8n | YOLOv8x | YOLOv10 | YOLOv12x | **Proposed Adaptive** |
|---|---|---|---|---|---|
| mAP@0.5 | 0.870 | 0.930 | 0.925 | 0.940 | **0.910** |
| mAP@[0.5:0.95] | 0.710 | 0.770 | 0.765 | 0.780 | **0.752** |
| Avg Latency (ms) | 18.0 | 56.0 | 48.0 | 60.0 | **24.2** |
| Energy Savings vs v8x | — | 0% | — | — | **69%** |
| Switches / 1000 fr | 0 | 0 | 0 | 0 | **≤ 4.1** |
| Controller Overhead | — | — | — | — | **< 1 ms** |

Statistical significance confirmed by independent t-tests (p < 0.001) and Wilcoxon signed-rank tests (Cohen's d ≈ 3.0).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Incoming Video Frames                     │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────▼──────────────────┐
         │    Scene & Feature Extraction     │
         │   (20 parameters, 3 categories)   │
         └───────────────┬──────────────────┘
                         │
         ┌───────────────▼──────────────────┐
         │       Complexity Score (CS)       │
         │  CS_raw → EMA smoothing → CS̃(t)  │
         └───────────────┬──────────────────┘
                         │
         ┌───────────────▼──────────────────┐
         │  3-Layer Intelligent Controller   │
         │  Layer 1: Asymmetric Hysteresis   │
         │  Layer 2: Feature Validation      │
         │  Layer 3: Statistical Stability   │
         └───────────────┬──────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
   ┌──────▼──────┐             ┌────────▼────────┐
   │  YOLOv8n    │             │   YOLOv8x        │
   │  3.2M params│             │   66.2M params   │
   │  ~18 ms/fr  │             │   ~56 ms/fr      │
   └─────────────┘             └─────────────────┘
```

See [`docs/architecture.svg`](docs/architecture.svg) for the full annotated diagram.

---

## Repository Structure

```
DeepClean-AdaptiveDualMode/
├── README.md                        ← This file
├── requirements.txt                 ← All Python dependencies
├── setup.py                         ← Package install
├── .gitignore
│
├── datasets/                        ← Dataset download & merge utilities
│   ├── download_trashcan.py         ← TrashCAN 1.0 from UMN
│   ├── download_roboflow.py         ← Roboflow UTD2 dataset
│   ├── download_coco.py             ← MS COCO (relevant classes)
│   ├── merge_datasets.py            ← Unified YOLO-format merge
│   └── README.md                    ← Class mapping & statistics
│
├── training/                        ← Model training scripts
│   ├── train_yolov8n.py
│   ├── train_yolov8x.py
│   ├── train_yolov12n.py
│   ├── train_yolov12x.py
│   ├── train_yolov10.py
│   ├── configs/                     ← Per-model YAML configs
│   │   ├── yolov8n.yaml
│   │   ├── yolov8x.yaml
│   │   ├── yolov12n.yaml
│   │   ├── yolov12x.yaml
│   │   └── yolov10.yaml
│   └── results/                     ← Auto-populated after training
│       └── README.md
│
├── controller/                      ← THE MAIN CONTRIBUTION
│   ├── deepclean_v4.py              ← Full 4-component adaptive system
│   ├── complexity_score.py          ← CS computation & EMA smoothing
│   ├── parameter_extractor.py       ← All 20 parameter extractors
│   ├── adaptive_threshold.py        ← Layer 4: online threshold update
│   ├── objective_function.py        ← Formal J definition & evaluator
│   └── utils.py                     ← Shared helpers
│
├── evaluation/                      ← Experimental validation
│   ├── run_ablation.py              ← All 7 layer-combination ablations
│   ├── run_statistical_tests.py     ← t-test, Wilcoxon, Cohen's d
│   ├── benchmark_models.py          ← Compare all 5 models side-by-side
│   ├── generate_plots.py            ← Reproduce all paper figures
│   └── results/                     ← CSVs and PNGs saved here
│
├── tests/                           ← Unit & integration tests
│   ├── test_controller.py
│   ├── test_cs_computation.py
│   └── test_objective_fn.py
│
├── notebooks/
│   └── DeepClean_v4_Colab.ipynb    ← Clean end-to-end Colab demo
│
├── docs/                            ← Documentation & diagrams
│   ├── architecture.svg
│   ├── controller_flow.svg
│   ├── cs_weight_diagram.svg
│   └── paper/                       ← Paper PDF + supplementary
│
├── generate_synthetic_video.py      ← Generate test video (no GPU needed)
├── run_adaptive_test.py             ← Run & evaluate full pipeline
└── assets/figures/                  ← All paper figures
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-org/DeepClean-AdaptiveDualMode.git
cd DeepClean-AdaptiveDualMode
pip install -r requirements.txt
```

### 2. Run Without GPU (Synthetic Demo)

```bash
# Generate a 300-frame synthetic underwater video
python generate_synthetic_video.py

# Run the full 3-layer adaptive controller & generate all result plots
python run_adaptive_test.py --video synthetic_underwater.mp4
```

Results are saved in `./results/`:
- `cs_trajectory.png` — Complexity Score evolution
- `model_timeline.png` — YOLOv8n/x selection over time
- `radar_chart.png` — Comparative radar chart
- `statistical_table.csv` — Full metrics table
- `ablation_layers.csv` — Layer-combination ablation

### 3. Download Datasets

```bash
# TrashCAN 1.0
python datasets/download_trashcan.py --output datasets/raw/trashcan

# Roboflow UTD2 (requires API key)
python datasets/download_roboflow.py --api-key YOUR_KEY --output datasets/raw/roboflow

# MS COCO subset (bottle, cup, backpack classes)
python datasets/download_coco.py --output datasets/raw/coco

# Merge all into unified YOLO format
python datasets/merge_datasets.py --output datasets/merged
```

### 4. Train Models

```bash
# Train all five models (requires GPU & datasets)
python training/train_yolov8n.py  --data datasets/merged/data.yaml --epochs 180
python training/train_yolov8x.py  --data datasets/merged/data.yaml --epochs 180
python training/train_yolov10.py  --data datasets/merged/data.yaml --epochs 180
python training/train_yolov12n.py --data datasets/merged/data.yaml --epochs 180
python training/train_yolov12x.py --data datasets/merged/data.yaml --epochs 180
```

### 5. Evaluate & Benchmark

```bash
# Full benchmark: all 5 models compared
python evaluation/benchmark_models.py \
  --models training/results/yolov8n/weights/best.pt \
           training/results/yolov8x/weights/best.pt \
  --video  your_test_video.mp4

# Layer-combination ablation (7 configs)
python evaluation/run_ablation.py --video synthetic_underwater.mp4

# Statistical significance tests
python evaluation/run_statistical_tests.py

# Reproduce all paper figures
python evaluation/generate_plots.py
```

### 6. Run Unit Tests

```bash
python -m pytest tests/ -v
```

---

## Complexity Score (CS) — How It Works

The CS aggregates **20 normalised parameters** across three categories:

| Category | Weight | Parameters |
|---|---|---|
| Scene Attributes | 40% | Turbidity, Lighting Variation, Texture Richness, Occlusion, Motion Blur, Camera Stability, Colour Cast, Object Density |
| Model Feedback | 35% | Mean Confidence, Conf. Variance, Inference Latency, Detection Dropout, False Positive Ratio, BBox Stability |
| Mission Constraints | 25% | Battery Level, Mission Phase, Trash Priority, Bandwidth, Distance to Zone, Remaining Time |

```
CS_raw(t)    = Σ wᵢ · pᵢ(t),   Σwᵢ = 1
CS_smooth(t) = α · CS_raw(t) + (1 − α) · CS_smooth(t−1),   α = 0.3
```

See [`controller/complexity_score.py`](controller/complexity_score.py) for full implementation.

---

## 3-Layer Switching Controller

| Layer | Mechanism | When Active |
|---|---|---|
| **Layer 1** | Asymmetric hysteresis (τ_down=0.40, τ_up=0.55) | Every frame |
| **Layer 2** | Time-based feature validation (50-frame dwell, 5 scene params) | CS in hysteresis zone |
| **Layer 3** | Statistical stability detection (σ_CS < 0.03, W=20 frames) | CS in hysteresis zone |

See [`docs/controller_flow.svg`](docs/controller_flow.svg) for the full decision-flow diagram.

---

## Datasets

| Dataset | Classes | Images | Source |
|---|---|---|---|
| UTD2 (Roboflow) | Bio, Rov, Trash | ~2,400 | [Roboflow Universe](https://universe.roboflow.com/utd-0dazj/utd2-hyo53) |
| TrashCAN 1.0 | 16 trash/bio categories | 7,212 | [UMN](http://irvlab.cs.umn.edu/resources/trash-can-dataset) |
| MS COCO subset | bottle, cup, backpack, etc. | ~12,000 | [COCO](https://cocodataset.org) |

See [`datasets/README.md`](datasets/README.md) for class mapping and merge details.

---

## Objective Function

$$J = \frac{1}{T} \sum_{t=1}^{T} \left[ \lambda \cdot A(M_t, t) - (1-\lambda) \cdot \frac{E(M_t)}{E_{\max}} \right]$$

subject to:  
- $M_t \in \{M_L, M_H\}$  
- Switch rate $\leq r_{\max}$  
- $B_t > B_{\text{crit}}$ (energy safety constraint)

with $\lambda = 0.6$, $E_{\max} = 50$ W, $B_{\text{crit}} = 0.20$.

---

## Citation

```bibtex
@article{bhatia2025deepclean,
  title={Adaptive Dual-Mode Underwater Trash Detection System},
  author={Bhatia, Amol and V., Krisvarish and Yadav, Harsh and Jaiswal, Avishkar and Gupta, Shivani and Gupta, Saurav},
  journal={arXiv preprint},
  year={2025}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contributors

| Name | Role |
|---|---|
| Amol Bhatia | System design, controller architecture, paper |
| Krisvarish V. | Hardware integration, sensor fusion, paper |
| Harsh Yadav | Dataset curation, training pipeline, paper |
| Avishkar Jaiswal | Evaluation framework, statistical analysis, paper |
| Shivani Gupta | Image enhancement, visualisation, paper |
| Saurav Gupta | Project supervision, paper |
