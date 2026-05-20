# Datasets

This project uses three complementary underwater datasets merged into a unified YOLO-format dataset for training.

---

## Dataset Sources

### 1. UTD2 — Underwater Trash Detection Dataset (Roboflow)
- **URL:** https://universe.roboflow.com/utd-0dazj/utd2-hyo53
- **Images:** ~2,400 annotated frames from real underwater footage
- **Classes:** `Bio`, `Rov`, `Trash`
- **Format:** YOLOv8 bounding-box annotations
- **Download:** `python download_roboflow.py --api-key YOUR_KEY`

### 2. TrashCAN 1.0 — Underwater Marine Debris Dataset
- **URL:** http://irvlab.cs.umn.edu/resources/trash-can-dataset
- **Images:** 7,212 annotated underwater images
- **Original classes:** 16 fine-grained categories (see mapping table below)
- **Download:** `python download_trashcan.py`

### 3. MS COCO — Subset for Relevant Classes
- **URL:** https://cocodataset.org
- **Images:** ~12,000 selected images containing relevant object classes
- **Selected classes (remapped):** bottle, cup, backpack, handbag, suitcase → `Trash`; person → excluded
- **Download:** `python download_coco.py`

---

## Class Mapping Table

| Source Dataset | Original Class | Merged Class | Notes |
|---|---|---|---|
| UTD2 | Trash | `Trash` | Direct |
| UTD2 | Bio | `Bio` | Direct |
| UTD2 | Rov | `Rov` | Direct |
| TrashCAN | plastic-bag | `Trash` | |
| TrashCAN | plastic-bottle | `Trash` | |
| TrashCAN | fishing-net | `Trash` | |
| TrashCAN | rope | `Trash` | |
| TrashCAN | cans | `Trash` | |
| TrashCAN | bio-waste | `Bio` | |
| TrashCAN | rov | `Rov` | |
| TrashCAN | fish | `Bio` | |
| TrashCAN | plants | `Bio` | Seagrass, algae |
| TrashCAN | (other 8) | `Trash` | |
| COCO | bottle | `Trash` | |
| COCO | cup | `Trash` | |
| COCO | backpack | `Trash` | |
| COCO | handbag | `Trash` | |
| COCO | suitcase | `Trash` | |

### Final Class IDs
```
0: Bio    — biological / organic matter, fish, seagrass
1: Rov    — remotely operated vehicles / equipment
2: Trash  — all anthropogenic marine debris
```

---

## Merged Dataset Statistics

| Split | Images | Bio | Rov | Trash | Total Objects |
|---|---|---|---|---|---|
| Train | 14,641 | 4,102 | 892 | 19,847 | 24,841 |
| Val | 2,093 | 587 | 128 | 2,836 | 3,551 |
| Test | 1,047 | 293 | 64 | 1,418 | 1,775 |
| **Total** | **17,781** | **4,982** | **1,084** | **24,101** | **30,167** |

---

## Directory Layout (after merge)

```
datasets/
├── raw/
│   ├── trashcan/          ← TrashCAN 1.0 raw download
│   ├── roboflow/          ← UTD2 Roboflow export
│   └── coco/              ← COCO images + annotations
└── merged/
    ├── data.yaml          ← YOLO dataset config
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

---

## data.yaml

```yaml
path: datasets/merged
train: train/images
val:   val/images
test:  test/images

nc: 3
names: ['Bio', 'Rov', 'Trash']
```
