"""
Synthetic Underwater Video Generator
=====================================
Generates a 300-frame synthetic underwater video with:
  - Three complexity phases (clear → turbid → medium)
  - Simulated trash/debris objects (rectangles with underwater coloring)
  - Simulated ROV artifacts, motion blur, and lighting variation
  - Underwater color cast (blue-green tint)

Output: synthetic_underwater.mp4  (640×480, 30 FPS)

Run:  python generate_synthetic_video.py
"""

import cv2
import numpy as np
import random
import math
import os

# ─── Configuration ────────────────────────────────────────────────────────────
OUTPUT_PATH  = "synthetic_underwater.mp4"
WIDTH, HEIGHT = 640, 480
FPS           = 30
TOTAL_FRAMES  = 300          # 10 seconds
SEED          = 42
random.seed(SEED)
np.random.seed(SEED)

# ─── Object definitions ───────────────────────────────────────────────────────
TRASH_TYPES = [
    {"name": "Trash",  "color_bgr": (30,  80,  160), "size": (60, 40)},
    {"name": "Trash",  "color_bgr": (20, 120,  200), "size": (80, 30)},
    {"name": "Bio",    "color_bgr": (20, 120,  60),  "size": (50, 50)},
    {"name": "Rov",    "color_bgr": (180, 60, 20),   "size": (90, 60)},
    {"name": "Trash",  "color_bgr": (60,  60, 150),  "size": (45, 35)},
]

class FloatingObject:
    def __init__(self, obj_type, frame_w, frame_h):
        self.obj   = obj_type
        self.w     = frame_w
        self.h     = frame_h
        self.x     = float(random.randint(50, frame_w - 100))
        self.y     = float(random.randint(50, frame_h - 100))
        self.vx    = random.uniform(-0.8, 0.8)
        self.vy    = random.uniform(-0.5, 0.5)
        self.life  = random.randint(60, 180)   # frames
        self.age   = 0
        self.angle = random.uniform(0, 360)
        self.spin  = random.uniform(-1.5, 1.5)

    def update(self):
        self.x    += self.vx + random.gauss(0, 0.2)
        self.y    += self.vy + random.gauss(0, 0.1)
        self.angle = (self.angle + self.spin) % 360
        self.age  += 1
        # soft bounce
        if self.x < 20 or self.x > self.w - 20:
            self.vx *= -1
        if self.y < 20 or self.y > self.h - 20:
            self.vy *= -1
        return self.age < self.life

    def draw(self, frame, phase_turbidity):
        """Draw rotated rectangle with depth-based opacity."""
        cx, cy  = int(self.x), int(self.y)
        ow, oh  = self.obj["size"]
        alpha   = max(0.25, 1.0 - phase_turbidity * 0.55)
        fade    = min(1.0, (self.life - self.age) / 20.0) * min(1.0, self.age / 10.0)
        alpha  *= fade

        pts = cv2.boxPoints(((cx, cy), (ow, oh), self.angle))
        pts = pts.astype(np.int32)

        overlay = frame.copy()
        col = self.obj["color_bgr"]
        cv2.fillPoly(overlay, [pts], col)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # White label
        label_alpha = max(0, alpha - 0.15)
        if label_alpha > 0.1:
            ov2 = frame.copy()
            cv2.putText(ov2, self.obj["name"], (cx - 15, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
            cv2.addWeighted(ov2, label_alpha, frame, 1 - label_alpha, 0, frame)

# ─── Background generators ────────────────────────────────────────────────────
def make_underwater_bg(frame_idx, turbidity, width, height):
    """Generate a procedural underwater background."""
    bg = np.zeros((height, width, 3), dtype=np.uint8)

    # Base gradient: dark at top, slightly lighter below
    for row in range(height):
        depth_factor = row / height
        base_b = int(60 + depth_factor * 20 + turbidity * 40)
        base_g = int(40 + depth_factor * 15 + turbidity * 20)
        base_r = int(10 + depth_factor *  5 + turbidity * 10)
        bg[row, :] = (
            min(255, base_b),
            min(255, base_g),
            min(255, base_r)
        )

    # Caustic light ripples (only visible in clear water)
    if turbidity < 0.5:
        t = frame_idx * 0.05
        ripple_strength = int((0.5 - turbidity) * 30)
        for _ in range(8):
            cx = int((math.sin(t * 0.7 + _ * 1.3) * 0.4 + 0.5) * width)
            cy = int((math.cos(t * 0.5 + _ * 0.9) * 0.3 + 0.35) * height)
            r  = random.randint(30, 80)
            ov = bg.copy()
            cv2.circle(ov, (cx, cy), r, (ripple_strength, ripple_strength//2, 0), -1)
            cv2.addWeighted(ov, 0.08, bg, 0.92, 0, bg)

    # Sand/rock bottom
    bottom_y = int(height * 0.75)
    sand_col  = (max(0, 80 - int(turbidity * 30)),
                 max(0, 65 - int(turbidity * 20)),
                 max(0, 30 - int(turbidity * 10)))
    cv2.rectangle(bg, (0, bottom_y), (width, height), sand_col, -1)

    # Rock shapes
    for i in range(5):
        rx = int((i / 5.0 + 0.1) * width)
        ry = bottom_y + random.randint(-10, 10) if i != 2 else bottom_y
        cv2.ellipse(bg, (rx, ry), (random.randint(25, 60), random.randint(15, 30)),
                    0, 0, 180, (45, 40, 25), -1)

    return bg


def add_turbidity_effect(frame, turbidity):
    """Apply haze / particle scattering proportional to turbidity."""
    if turbidity < 0.05:
        return frame
    # Blur (scatter)
    ksize = int(turbidity * 12) * 2 + 1
    blurred = cv2.GaussianBlur(frame, (ksize, ksize), 0)
    cv2.addWeighted(blurred, turbidity * 0.5, frame, 1 - turbidity * 0.5, 0, frame)

    # Suspended particles
    num_particles = int(turbidity * 200)
    for _ in range(num_particles):
        px = random.randint(0, frame.shape[1] - 1)
        py = random.randint(0, frame.shape[0] - 1)
        brightness = random.randint(80, 160)
        frame[py, px] = (brightness, brightness + 10, brightness - 10)

    return frame


def add_motion_blur(frame, strength):
    if strength < 0.05:
        return frame
    k = int(strength * 12) * 2 + 1
    kernel = np.zeros((k, k))
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(frame, -1, kernel)


def add_lighting_variation(frame, frame_idx, amplitude):
    if amplitude < 0.01:
        return frame
    flicker = 1.0 + amplitude * math.sin(frame_idx * 0.3) * 0.15
    return np.clip((frame.astype(np.float32) * flicker), 0, 255).astype(np.uint8)


# ─── Phase schedule ───────────────────────────────────────────────────────────
def get_phase_params(frame_idx, total):
    """
    Phase 1 (0-33%):    Clear water,  low turbidity  → CS in [0.35–0.45]
    Phase 2 (33–66%):   Turbid water, high complexity → CS in [0.52–0.65]
    Phase 3 (66–100%):  Moderate,     medium          → CS in [0.42–0.52]
    """
    progress = frame_idx / total
    if progress < 0.33:
        t = progress / 0.33
        return {
            "turbidity":      0.10 + t * 0.05,
            "motion_blur":    0.05 + t * 0.05,
            "lighting_amp":   0.10,
            "num_objects":    2,
            "label":          "Phase 1: Clear",
        }
    elif progress < 0.66:
        t = (progress - 0.33) / 0.33
        return {
            "turbidity":      0.45 + t * 0.25,
            "motion_blur":    0.15 + t * 0.20,
            "lighting_amp":   0.35 + t * 0.15,
            "num_objects":    4 + int(t * 2),
            "label":          "Phase 2: Turbid/Complex",
        }
    else:
        t = (progress - 0.66) / 0.34
        return {
            "turbidity":      0.35 - t * 0.10,
            "motion_blur":    0.15 - t * 0.05,
            "lighting_amp":   0.25 - t * 0.10,
            "num_objects":    3,
            "label":          "Phase 3: Moderate",
        }


# ─── Main generation loop ─────────────────────────────────────────────────────
def generate():
    os.makedirs(os.path.dirname(OUTPUT_PATH) if os.path.dirname(OUTPUT_PATH) else ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))

    objects: list[FloatingObject] = []

    print(f"Generating {TOTAL_FRAMES} frames → {OUTPUT_PATH}")
    for fi in range(TOTAL_FRAMES):
        params = get_phase_params(fi, TOTAL_FRAMES)

        # Background
        frame = make_underwater_bg(fi, params["turbidity"], WIDTH, HEIGHT)

        # Maintain object pool
        while len(objects) < params["num_objects"]:
            t = random.choice(TRASH_TYPES)
            objects.append(FloatingObject(t, WIDTH, HEIGHT))

        objects = [o for o in objects if o.update()]

        # Draw objects
        for o in objects:
            o.draw(frame, params["turbidity"])

        # Environmental effects
        frame = add_turbidity_effect(frame, params["turbidity"])
        frame = add_motion_blur(frame, params["motion_blur"])
        frame = add_lighting_variation(frame, fi, params["lighting_amp"])

        # Phase label overlay
        cv2.putText(frame, params["label"], (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, f"Frame {fi+1}/{TOTAL_FRAMES}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
        turbidity_text = f"Turbidity: {params['turbidity']:.2f}"
        cv2.putText(frame, turbidity_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 200, 160), 1)

        writer.write(frame)

        if (fi + 1) % 50 == 0:
            print(f"  [{fi+1}/{TOTAL_FRAMES}] turbidity={params['turbidity']:.2f}, "
                  f"objects={len(objects)}, blur={params['motion_blur']:.2f}")

    writer.release()
    file_size = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"\nDone! {OUTPUT_PATH} ({file_size:.1f} KB)")
    return OUTPUT_PATH


if __name__ == "__main__":
    generate()
