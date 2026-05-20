"""
Controller Utilities
=====================
Shared helper functions used across the DeepClean controller modules:
  - Frame annotation / overlay rendering
  - Metric history management
  - CSV / JSON export helpers
  - Timing context manager
"""

import csv
import json
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np


# ─── Timing ───────────────────────────────────────────────────────────────────

@contextmanager
def timer(label: str = ""):
    """Context manager that prints elapsed time in ms."""
    start = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - start) * 1000
    if label:
        print(f"[{label}] {elapsed_ms:.2f} ms")


class InferenceTimer:
    """Rolling-average inference timer."""

    def __init__(self, window: int = 30) -> None:
        self._times: list[float] = []
        self._window = window

    def record(self, elapsed_s: float) -> None:
        self._times.append(elapsed_s)
        if len(self._times) > self._window:
            self._times.pop(0)

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self._times)) * 1000 if self._times else 0.0

    @property
    def latest_s(self) -> float:
        return self._times[-1] if self._times else 0.0


# ─── Frame annotation ─────────────────────────────────────────────────────────

# Colour palette
COLOUR = {
    "lightweight": (39, 174, 96),     # green
    "heavyweight": (231, 76, 60),     # red
    "hysteresis":  (241, 196, 15),    # yellow
    "white":       (255, 255, 255),
    "grey":        (180, 180, 180),
    "dark":        (30, 30, 30),
}


def draw_detection_boxes(
    frame: np.ndarray,
    detections,
    class_names: list[str],
    conf_threshold: float = 0.25,
) -> np.ndarray:
    """
    Draw YOLO detection bounding boxes with labels onto `frame`.

    Parameters
    ----------
    frame          : BGR frame
    detections     : ultralytics Results object
    class_names    : list of class names indexed by class id
    conf_threshold : minimum confidence to draw

    Returns
    -------
    Annotated frame (in-place modification).
    """
    if detections is None or len(detections.boxes) == 0:
        return frame

    for box in detections.boxes:
        conf = float(box.conf[0].cpu())
        if conf < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
        cls_id = int(box.cls[0].cpu())
        label  = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)

        colour = COLOUR["lightweight"] if conf >= 0.5 else COLOUR["hysteresis"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 2, y1), colour, -1)
        cv2.putText(
            frame, text, (x1 + 1, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1,
        )
    return frame


def draw_hud(
    frame: np.ndarray,
    model_name:   str,
    cs_smooth:    float,
    cs_raw:       float,
    tau_down:     float,
    tau_up:       float,
    frame_idx:    int,
    switch_count: int,
    params:       dict | None = None,
    top_n:        int = 3,
) -> np.ndarray:
    """
    Render a HUD overlay panel at the bottom of the frame with system state.
    """
    h, w = frame.shape[:2]
    panel_h = 110
    panel = np.full((panel_h, w, 3), 30, dtype=np.uint8)

    # Model name
    is_heavy = "v8x" in model_name.lower() or "v12x" in model_name.lower()
    m_colour = COLOUR["heavyweight"] if is_heavy else COLOUR["lightweight"]
    cv2.putText(panel, f"Model: {model_name}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, m_colour, 2)

    # CS value & bar
    cs_colour = (
        COLOUR["lightweight"] if cs_smooth < tau_down
        else COLOUR["heavyweight"] if cs_smooth > tau_up
        else COLOUR["hysteresis"]
    )
    cv2.putText(panel, f"CS: {cs_smooth:.3f}  (raw: {cs_raw:.3f})",
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cs_colour, 1)

    bar_x0, bar_y0, bar_w, bar_h2 = 10, 64, 300, 12
    cv2.rectangle(panel, (bar_x0, bar_y0), (bar_x0 + bar_w, bar_y0 + bar_h2),
                  (70, 70, 70), -1)
    filled = int(np.clip(cs_smooth, 0, 1) * bar_w)
    cv2.rectangle(panel, (bar_x0, bar_y0),
                  (bar_x0 + filled, bar_y0 + bar_h2), cs_colour, -1)
    # Threshold markers
    for tau, color in [(tau_down, (255, 100, 100)), (tau_up, (100, 255, 100))]:
        tx = bar_x0 + int(tau * bar_w)
        cv2.line(panel, (tx, bar_y0 - 2), (tx, bar_y0 + bar_h2 + 2), color, 2)

    # Frame / switch counter
    cv2.putText(panel, f"Frame {frame_idx}  Switches: {switch_count}",
                (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOUR["grey"], 1)

    # Top contributing parameters
    if params:
        from controller.complexity_score import ComplexityScoreComputer, BASE_WEIGHTS
        contribs = sorted(
            [(f"P{i}", BASE_WEIGHTS[f"w{i}"] * params.get(f"p{i}", 0))
             for i in range(1, 21)],
            key=lambda x: x[1], reverse=True,
        )[:top_n]
        x_off = 340
        cv2.putText(panel, "Top contributors:", (x_off, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOUR["white"], 1)
        for idx, (pname, val) in enumerate(contribs):
            cv2.putText(panel, f"{pname}: {val:.3f}", (x_off, 40 + idx * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOUR["grey"], 1)

    return np.vstack([frame, panel])


# ─── Export helpers ───────────────────────────────────────────────────────────

def save_csv(rows: list[list], path: str | Path) -> None:
    """Write rows (list of lists) to a CSV file, creating parent dirs."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        csv.writer(fh).writerows(rows)


def save_json(obj, path: str | Path, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=indent)


def load_json(path: str | Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


# ─── Video utilities ──────────────────────────────────────────────────────────

def open_video(path: str | Path):
    """Open a cv2.VideoCapture and return (cap, fps, W, H, total_frames)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps    = cap.get(cv2.CAP_PROP_FPS)  or 30.0
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, W, H, total


def make_writer(path: str | Path, fps: float, width: int, height: int):
    """Return a cv2.VideoWriter for MP4 output."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))
