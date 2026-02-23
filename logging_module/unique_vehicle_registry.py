"""
Unique Vehicle Registry
========================
Tracks confirmed plate numbers and logs each one exactly once to a CSV file.
Duplicate plates (same vehicle reappearing) are silently ignored.
"""

import csv
import os
from datetime import datetime


class UniqueVehicleRegistry:
    def __init__(self, config: dict):
        log_cfg  = config.get("logging", {})
        log_dir  = log_cfg.get("log_directory", "logs")
        os.makedirs(log_dir, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path = os.path.join(log_dir, f"unique_vehicles_{ts}.csv")
        self._seen: set = set()   # confirmed plate strings already logged

        with open(self._csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "plate_number", "first_seen_frame"])

        print(f"Vehicle registry : {os.path.abspath(self._csv_path)}")

    def register(self, plate: str, frame_num: int) -> bool:
        """
        Register a confirmed plate.
        Returns True if this is a new unique vehicle, False if duplicate.
        """
        if not plate or plate in self._seen:
            return False

        self._seen.add(plate)
        ts = datetime.now().isoformat()

        with open(self._csv_path, "a", newline="") as f:
            csv.writer(f).writerow([ts, plate, frame_num])

        print(f"[Registry] New vehicle: {plate} (frame {frame_num})")
        return True

    @property
    def count(self) -> int:
        return len(self._seen)
