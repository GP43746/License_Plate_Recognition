from collections import defaultdict, deque, Counter


class PlateStabilizer:
    def __init__(self, history_size=15, vote_threshold=0.6, min_samples=5):
        self.history_size = history_size
        self.vote_threshold = vote_threshold
        self.min_samples = min_samples

        # Per-track history: track_id -> deque of plate strings
        self.history = defaultdict(lambda: deque(maxlen=self.history_size))

    def update(self, track_id, plate_text):
        if plate_text:
            self.history[track_id].append(plate_text)

    def get_stable_plate(self, track_id):
        """
        Returns (stable_plate: str, stability_ratio: float).
        stable_plate is "" if not yet stable.
        Early stabilization fires when:
          - total samples >= min_samples AND
          - top candidate frequency / total samples >= vote_threshold
        Full stabilization fires when history is full.
        """
        if track_id not in self.history:
            return "", 0.0

        records = self.history[track_id]
        n = len(records)

        if n < self.min_samples:
            return "", 0.0

        counts = Counter(records)
        plate, freq = counts.most_common(1)[0]
        ratio = freq / n

        if ratio >= self.vote_threshold:
            return plate, round(ratio, 3)

        return "", round(ratio, 3)

    def remove_track(self, track_id):
        """Call when SORT drops a track to free memory."""
        if track_id in self.history:
            del self.history[track_id]

    def get_last_known(self, track_id):
        """Return the last OCR'd plate text for a disappearing track."""
        records = self.history.get(track_id)
        if records:
            counts = Counter(records)
            return counts.most_common(1)[0][0]
        return ""