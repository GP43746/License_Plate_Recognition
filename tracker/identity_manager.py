"""
Persistent Identity Manager
============================
Maps transient SORT track IDs to long-lived persistent_plate_ids.

Key behaviours:
  - New SORT track → check recent_lost via IOU; if match ≥ thresh, reuse old ID
  - EMA bbox smoothing to remove visual jitter
  - Prunes lost-track buffer after lost_buffer_frames
"""

import numpy as np


def _iou(a, b):
    """Compute IOU between two boxes [x1,y1,x2,y2]."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


class IdentityManager:
    def __init__(self, config: dict):
        p = config.get("persistence", {})
        self.lost_buffer_frames  = p.get("lost_buffer_frames", 30)
        self.reassign_iou_thresh = p.get("reassign_iou_thresh", 0.65)
        self.alpha               = p.get("ema_alpha", 0.7)   # EMA weight for new bbox

        self._next_pid: int = 1                         # monotonic persistent ID counter
        self._sort_to_pid: dict  = {}                   # sort_id  → persistent_id
        self._pid_to_sort: dict  = {}                   # persistent_id → sort_id (active)
        self._smoothed_bbox: dict = {}                  # persistent_id → smoothed [x1,y1,x2,y2]

        # persistent_id → {bbox, last_seen_frame, last_stable_plate}
        self.recent_lost: dict = {}

    # ------------------------------------------------------------------
    def update(self, sort_tracks: list, frame_num: int):
        """
        Parameters
        ----------
        sort_tracks : list of [x1, y1, x2, y2, sort_id]
        frame_num   : current frame number

        Returns
        -------
        list of (persistent_id, smoothed_bbox [x1,y1,x2,y2])
        """
        current_sort_ids = {int(t[4]) for t in sort_tracks}

        # Detect dropped SORT IDs → move their persistent IDs to recent_lost
        dropped_sort_ids = set(self._sort_to_pid.keys()) - current_sort_ids
        for sid in dropped_sort_ids:
            pid = self._sort_to_pid.pop(sid)
            self._pid_to_sort.pop(pid, None)
            bbox = self._smoothed_bbox.get(pid, [0, 0, 0, 0])
            self.recent_lost[pid] = {
                "bbox":              bbox,
                "last_seen_frame":   frame_num,
                "last_stable_plate": "",     # pipeline fills this in
            }

        # Prune stale lost entries
        stale = [pid for pid, v in self.recent_lost.items()
                 if frame_num - v["last_seen_frame"] > self.lost_buffer_frames]
        for pid in stale:
            self.recent_lost.pop(pid, None)
            self._smoothed_bbox.pop(pid, None)

        results = []
        for track in sort_tracks:
            x1, y1, x2, y2, sort_id = int(track[0]), int(track[1]), int(track[2]), int(track[3]), int(track[4])
            raw_bbox = [x1, y1, x2, y2]

            if sort_id in self._sort_to_pid:
                # Known active track
                pid = self._sort_to_pid[sort_id]
            else:
                # Try to reassign from recent_lost via IOU
                pid = self._try_reassign(raw_bbox)
                if pid is None:
                    pid = self._next_pid
                    self._next_pid += 1
                else:
                    self.recent_lost.pop(pid, None)

                self._sort_to_pid[sort_id] = pid
                self._pid_to_sort[pid]     = sort_id

            # EMA smoothing
            if pid in self._smoothed_bbox:
                prev = self._smoothed_bbox[pid]
                smoothed = [
                    int(self.alpha * raw_bbox[i] + (1 - self.alpha) * prev[i])
                    for i in range(4)
                ]
            else:
                smoothed = raw_bbox[:]

            self._smoothed_bbox[pid] = smoothed
            results.append((pid, smoothed))

        return results

    # ------------------------------------------------------------------
    def notify_stable_plate(self, persistent_id: int, plate: str):
        """Pipeline calls this so recent_lost stores the last known plate."""
        if persistent_id in self.recent_lost:
            self.recent_lost[persistent_id]["last_stable_plate"] = plate

    def get_last_known_plate(self, persistent_id: int) -> str:
        return self.recent_lost.get(persistent_id, {}).get("last_stable_plate", "")

    def is_recently_lost(self, persistent_id: int) -> bool:
        return persistent_id in self.recent_lost

    def active_persistent_ids(self) -> set:
        return set(self._pid_to_sort.keys())

    # ------------------------------------------------------------------
    def _try_reassign(self, bbox) -> int | None:
        best_pid   = None
        best_iou   = self.reassign_iou_thresh  # must exceed threshold
        for pid, info in self.recent_lost.items():
            iou = _iou(bbox, info["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_pid = pid
        return best_pid
