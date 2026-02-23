"""
Minimal UI renderer — bounding box + stabilized plate label only.
No side panel, no zoom grid, no debug overlays.
"""

import cv2
import numpy as np


def _track_color(pid: int):
    """Deterministic BGR color per persistent ID."""
    np.random.seed(pid * 7 + 13)
    hue = int(np.random.randint(0, 180))
    hsv = np.uint8([[[hue, 220, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def draw_frame(frame: np.ndarray, tracks_info: list, fps: float, config: dict) -> np.ndarray:
    """
    Draw minimal ALPR overlay:
      - Coloured bounding box per active track
      - Track ID label above box
      - Stabilized plate number below box (only when confirmed stable)
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    for t in tracks_info:
        if not t.get("is_active", True):
            continue

        pid   = t["persistent_id"]
        x1, y1, x2, y2 = t["x1"], t["y1"], t["x2"], t["y2"]
        color = _track_color(pid)
        plate = t.get("stable_plate", "")   # only confirmed stable plate

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Track ID label above box
        id_label = f"P{pid}"
        (lw, lh), _ = cv2.getTextSize(id_label, font, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 4, y1), color, -1)
        cv2.putText(frame, id_label, (x1 + 2, y1 - 4),
                    font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Stabilized plate text below box — only when confirmed
        if plate:
            (pw, ph), _ = cv2.getTextSize(plate, font, 0.75, 2)
            cv2.rectangle(frame, (x1, y2 + 2), (x1 + pw + 6, y2 + ph + 10),
                          (20, 20, 20), -1)
            cv2.putText(frame, plate, (x1 + 3, y2 + ph + 6),
                        font, 0.75, color, 2, cv2.LINE_AA)

    return frame
