import csv
import json
import os
from datetime import datetime


class PlateLogger:
    def __init__(self, config: dict):
        log_cfg = config.get("logging", {})
        self.enable_csv  = log_cfg.get("enable_csv", True)
        self.enable_json = log_cfg.get("enable_json", True)
        self.log_dir     = log_cfg.get("log_directory", "logs")

        os.makedirs(self.log_dir, exist_ok=True)

        # Track which IDs have already been logged (avoid per-frame spam)
        self._logged_ids: set = set()

        # File paths
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path  = os.path.join(self.log_dir, f"plates_{ts}.csv")
        self._json_path = os.path.join(self.log_dir, f"plates_{ts}.jsonl")

        if self.enable_csv:
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "frame", "track_id",
                    "plate", "raw_ocr", "confidence",
                    "stability", "x1", "y1", "x2", "y2"
                ])

    def log_stable_event(self, frame_num: int, track_id: int,
                         stable_plate: str, raw_ocr: str,
                         confidence: float, stability: float,
                         bbox: tuple):
        """
        Logs a plate event the first time a track becomes stable.
        bbox = (x1, y1, x2, y2)
        """
        if track_id in self._logged_ids:
            return

        self._logged_ids.add(track_id)
        self._write(frame_num, track_id, stable_plate,
                    raw_ocr, confidence, stability, bbox)

    def log_track_end(self, frame_num: int, track_id: int,
                      stable_plate: str, raw_ocr: str,
                      confidence: float, stability: float,
                      bbox: tuple):
        """
        Logs a final event when a track disappears (even if not fully stable).
        Always writes — used as a track-end record.
        """
        self._write(frame_num, track_id, stable_plate,
                    raw_ocr, confidence, stability, bbox)

    def _write(self, frame_num, track_id, stable_plate,
               raw_ocr, confidence, stability, bbox):
        ts = datetime.now().isoformat()
        x1, y1, x2, y2 = bbox

        if self.enable_csv:
            with open(self._csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    ts, frame_num, track_id,
                    stable_plate, raw_ocr,
                    f"{confidence:.3f}", f"{stability:.3f}",
                    x1, y1, x2, y2
                ])

        if self.enable_json:
            event = {
                "timestamp": ts,
                "frame": frame_num,
                "track_id": track_id,
                "plate": stable_plate,
                "raw_ocr": raw_ocr,
                "confidence": round(confidence, 3),
                "stability": round(stability, 3),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            }
            with open(self._json_path, "a") as f:
                f.write(json.dumps(event) + "\n")
