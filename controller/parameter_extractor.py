"""
Parameter Extractor
====================
Extracts all 20 normalised parameters (P1–P20) from a video frame and
detector outputs, grouped into three categories:

  Scene Attributes  (P1–P8)   — visual difficulty indicators
  Model Feedback    (P9–P14)  — real-time detector performance signals
  Mission Context   (P15–P20) — operational constraints

All values are normalised to [0, 1].  Higher values always indicate
higher complexity / difficulty, consistent with the Complexity Score
formulation in the paper.
"""

import cv2
import numpy as np
from collections import deque
from typing import Any

try:
    from skimage.measure import shannon_entropy
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


class ParameterExtractor:
    """
    Extract all 20 parameters for the Complexity Score.

    Parameters
    ----------
    flow_history_len : int
        Frames kept in optical-flow history for smoothing (default 5).
    bbox_history_len : int
        Frames kept for bounding-box stability (inter-frame IoU) tracking.
    """

    # ── Class constants ────────────────────────────────────────────────────────
    # Laplacian variance thresholds for blur / texture normalisation
    _LAP_BLUR_MAX    = 1_000.0    # above this → sharp (blur=0)
    _LAP_TEX_MAX     = 2_000.0    # above this → max texture (texture=1)
    _FLOW_MAX        = 50.0       # pixels/frame → max instability
    _CV_MAX          = 2.0        # lighting coefficient-of-variation max

    def __init__(
        self,
        flow_history_len: int = 5,
        bbox_history_len: int = 10,
    ) -> None:
        self.prev_gray:   np.ndarray | None = None
        self.prev_boxes:  list | None       = None
        self._flow_hist   = deque(maxlen=flow_history_len)
        self._latency_hist= deque(maxlen=30)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def extract(
        self,
        frame:         np.ndarray,
        detections:    Any   = None,   # ultralytics Results object or None
        model_metrics: dict  | None = None,
        mission_state: dict  | None = None,
    ) -> dict[str, float]:
        """
        Extract all 20 parameters.

        Parameters
        ----------
        frame         : BGR frame (H × W × 3)
        detections    : ultralytics Results object from model(frame) — or None
        model_metrics : dict with keys:
                          latency_s       (float) — inference wall-clock time
                          dropout_rate    (float) — fraction of skipped frames
                          energy_ratio    (float) — current power / max power
        mission_state : dict with keys:
                          battery         (float) [0,1]
                          mission_phase   (float) [0,1]  0=explore 1=return
                          trash_priority  (float) [0,1]
                          bandwidth       (float) [0,1]
                          distance_zone   (float) [0,1]
                          time_remaining  (float) [0,1]

        Returns
        -------
        dict  p1 … p20, all in [0, 1]
        """
        model_metrics  = model_metrics  or {}
        mission_state  = mission_state  or {}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        b, g, r = cv2.split(frame)

        params: dict[str, float] = {}

        # ── Scene Attributes ──────────────────────────────────────────────────
        params["p1"]  = self._turbidity(gray, b, r)
        params["p2"]  = self._lighting_variation(frame)
        params["p3"]  = self._texture_richness(gray)
        params["p4"]  = self._occlusion_level(detections)
        params["p5"]  = self._motion_blur(gray)
        params["p6"]  = self._camera_stability(gray)
        params["p7"]  = self._colour_cast(r, g, b)
        params["p8"]  = self._object_density(detections, frame.shape)

        # ── Model Feedback ────────────────────────────────────────────────────
        params["p9"]  = self._low_confidence(detections)
        params["p10"] = self._confidence_variance(detections)
        params["p11"] = self._inference_latency(model_metrics)
        params["p12"] = float(np.clip(model_metrics.get("dropout_rate", 0.0), 0, 1))
        params["p13"] = self._false_positive_ratio(detections)
        params["p14"] = self._bbox_instability(detections)

        # ── Mission Constraints ───────────────────────────────────────────────
        # Battery: low battery → higher "complexity" to force lightweight mode
        params["p15"] = float(np.clip(1.0 - mission_state.get("battery", 0.6), 0, 1))
        params["p16"] = float(np.clip(mission_state.get("mission_phase",   0.5), 0, 1))
        params["p17"] = float(np.clip(mission_state.get("trash_priority",  0.5), 0, 1))
        params["p18"] = float(np.clip(mission_state.get("distance_zone",   0.5), 0, 1))
        params["p19"] = float(np.clip(1.0 - mission_state.get("time_remaining", 0.5), 0, 1))
        params["p20"] = float(np.clip(1.0 - mission_state.get("bandwidth",      0.5), 0, 1))

        # Update stateful fields for next frame
        self.prev_gray  = gray
        self.prev_boxes = self._get_boxes(detections)

        return params

    # ──────────────────────────────────────────────────────────────────────────
    # Scene parameter extractors
    # ──────────────────────────────────────────────────────────────────────────

    def _turbidity(self, gray: np.ndarray, b: np.ndarray, r: np.ndarray) -> float:
        """
        P1 — Turbidity.
        Combines darkness (inverse brightness) with blue-channel dominance,
        since underwater turbid scenes are characteristically dark and blue-shifted.
        """
        darkness    = 1.0 - float(np.mean(gray)) / 255.0
        blue_dominance = max(
            0.0,
            (float(np.mean(b)) - float(np.mean(r))) / 255.0,
        )
        return float(np.clip(darkness + 0.3 * blue_dominance, 0.0, 1.0))

    def _lighting_variation(self, frame: np.ndarray) -> float:
        """P2 — Lighting variation: coefficient of variation in L* channel."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        L   = lab[:, :, 0].astype(np.float64)
        mean = float(np.mean(L))
        cv   = float(np.std(L)) / (mean + 1e-6)
        return float(np.clip(cv / self._CV_MAX, 0.0, 1.0))

    def _texture_richness(self, gray: np.ndarray) -> float:
        """P3 — Texture richness: Laplacian variance normalised by empirical max."""
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return float(np.clip(lap.var() / self._LAP_TEX_MAX, 0.0, 1.0))

    def _occlusion_level(self, detections: Any) -> float:
        """P4 — Occlusion: edge density as a proxy for cluttered scenes."""
        # When no detections object: fall back to pure edge density
        if detections is None:
            return 0.0
        boxes = self._get_boxes(detections)
        if len(boxes) < 2:
            return 0.0
        # Pairwise IoU between detected bounding boxes
        ious = []
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                ious.append(self._iou(boxes[i], boxes[j]))
        return float(np.clip(np.mean(ious) if ious else 0.0, 0.0, 1.0))

    def _motion_blur(self, gray: np.ndarray) -> float:
        """P5 — Motion blur: inverse Laplacian variance (blur → low variance)."""
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(np.clip(lap.var() / self._LAP_BLUR_MAX, 0.0, 1.0))
        return 1.0 - sharpness

    def _camera_stability(self, gray: np.ndarray) -> float:
        """P6 — Camera stability: mean optical-flow magnitude."""
        if self.prev_gray is None:
            return 0.0
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        mag = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
        self._flow_hist.append(mag)
        # Smooth over recent frames
        smoothed = float(np.mean(self._flow_hist))
        return float(np.clip(smoothed / self._FLOW_MAX, 0.0, 1.0))

    def _colour_cast(
        self, r: np.ndarray, g: np.ndarray, b: np.ndarray
    ) -> float:
        """P7 — Colour cast: max pairwise difference of channel means."""
        rm = float(np.mean(r)) / 255.0
        gm = float(np.mean(g)) / 255.0
        bm = float(np.mean(b)) / 255.0
        return float(max(abs(rm - gm), abs(gm - bm), abs(bm - rm)))

    def _object_density(self, detections: Any, shape: tuple) -> float:
        """P8 — Object density: detected objects per unit frame area."""
        boxes = self._get_boxes(detections)
        if not boxes:
            return 0.0
        frame_area = shape[0] * shape[1]
        density = len(boxes) / frame_area
        return float(np.clip(density / 1e-2, 0.0, 1.0))  # 1% coverage = 1.0

    # ──────────────────────────────────────────────────────────────────────────
    # Model feedback extractors
    # ──────────────────────────────────────────────────────────────────────────

    def _low_confidence(self, detections: Any) -> float:
        """P9 — 1 − mean detection confidence (low conf = high complexity)."""
        confs = self._get_confidences(detections)
        if not confs:
            return 1.0          # no detections → maximum uncertainty
        return float(np.clip(1.0 - np.mean(confs), 0.0, 1.0))

    def _confidence_variance(self, detections: Any) -> float:
        """P10 — Normalised variance of confidence scores."""
        confs = self._get_confidences(detections)
        if len(confs) < 2:
            return 0.0
        return float(np.clip(np.var(confs) / 0.25, 0.0, 1.0))  # max var ≈ 0.25

    def _inference_latency(self, model_metrics: dict) -> float:
        """P11 — Normalised inference latency (relative to 60 ms budget)."""
        lat = model_metrics.get("latency_s", None)
        if lat is None:
            return 0.5
        self._latency_hist.append(lat)
        smoothed = float(np.mean(self._latency_hist))
        return float(np.clip(smoothed / 0.060, 0.0, 1.0))   # 60 ms → 1.0

    def _false_positive_ratio(self, detections: Any) -> float:
        """P13 — Fraction of detections with confidence < 0.5."""
        confs = self._get_confidences(detections)
        if not confs:
            return 0.0
        return float(np.sum(np.array(confs) < 0.5) / len(confs))

    def _bbox_instability(self, detections: Any) -> float:
        """
        P14 — Bounding-box instability: 1 − mean inter-frame IoU with
        the nearest matching box from the previous frame.
        """
        curr_boxes = self._get_boxes(detections)
        prev_boxes = self.prev_boxes or []
        if not curr_boxes or not prev_boxes:
            return 0.5
        ious = []
        for cb in curr_boxes:
            best_iou = max((self._iou(cb, pb) for pb in prev_boxes), default=0.0)
            ious.append(best_iou)
        return float(np.clip(1.0 - np.mean(ious), 0.0, 1.0))

    # ──────────────────────────────────────────────────────────────────────────
    # Geometry helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_boxes(detections: Any) -> list[tuple[float, float, float, float]]:
        """Return list of (x1,y1,x2,y2) from an ultralytics Results object."""
        if detections is None:
            return []
        try:
            return [tuple(b.tolist()) for b in detections.boxes.xyxy.cpu()]
        except Exception:
            return []

    @staticmethod
    def _get_confidences(detections: Any) -> list[float]:
        """Return list of confidence scores from an ultralytics Results object."""
        if detections is None:
            return []
        try:
            return detections.boxes.conf.cpu().tolist()
        except Exception:
            return []

    @staticmethod
    def _iou(
        b1: tuple[float, float, float, float],
        b2: tuple[float, float, float, float],
    ) -> float:
        """Axis-aligned intersection-over-union."""
        xi1 = max(b1[0], b2[0]); yi1 = max(b1[1], b2[1])
        xi2 = min(b1[2], b2[2]); yi2 = min(b1[3], b2[3])
        inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
        a1    = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2    = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = a1 + a2 - inter
        return inter / (union + 1e-6)
