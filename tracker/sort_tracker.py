import numpy as np
from tracker.sort import Sort


class PlateTracker:
    def __init__(self, config: dict = None):
        if config is None:
            config = {}
        trk = config.get("tracker", {})
        self.tracker = Sort(
            max_age=trk.get("max_age", 30),
            min_hits=trk.get("min_hits", 1),
            iou_threshold=trk.get("iou_threshold", 0.2)
        )

    def update(self, detection):
        if len(detection) > 0:
            dets = np.array(detection)
        else:
            dets = np.empty((0, 5))

        tracks = self.tracker.update(dets)

        results = []
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            results.append([int(x1), int(y1), int(x2), int(y2), int(track_id)])

        return results
