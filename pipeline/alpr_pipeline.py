import cv2

from detector.yolo_detector import detector
from tracker.sort_tracker import PlateTracker
from tracker.identity_manager import IdentityManager
from tracker.track_fsm import TrackFSM, TrackState
from preprocessing.plate_preprocess import plate_preprocess
from ocr1.ocr_engine import OCREngine
from validation.regex_validator import PlateValidator
from stabilization.plate_stabilizer import PlateStabilizer
from ui.renderer import draw_frame
from logging_module.unique_vehicle_registry import UniqueVehicleRegistry


class ALPRPipeline:
    def __init__(self, config, schema_path):
        self.config = config
        stab_cfg = config.get("stabilization", {})

        self.detector = detector(
            model_path=config["yolo"]["model_path"],
            device=config["device"],
            half=config["half_precision"]
        )
        self.tracker    = PlateTracker(config)
        self.identity   = IdentityManager(config)
        self.fsm        = TrackFSM()
        self.ocr        = OCREngine(use_gpu=config["ocr"]["use_gpu"])
        self.validator  = PlateValidator(schema_path)
        self.stabilizer = PlateStabilizer(
            history_size=stab_cfg.get("history_size", 15),
            vote_threshold=stab_cfg.get("vote_threshold", 0.6),
            min_samples=stab_cfg.get("min_samples", 5)
        )
        self.registry = UniqueVehicleRegistry(config)

        self._pid_state: dict = {}
        self._prev_active_pids: set = set()
        self._frame_num: int = 0

    def process_frame(self, frame, fps: float = 0.0):
        self._frame_num += 1
        fn = self._frame_num

        detections  = self.detector.detect(frame, confidence=self.config["yolo"]["conf_threshold"])
        sort_tracks = self.tracker.update(detections)
        id_pairs    = self.identity.update(sort_tracks, fn)
        current_pids = {pid for pid, _ in id_pairs}

        for pid, bbox in id_pairs:
            x1, y1, x2, y2 = bbox
            try:
                crop      = frame[y1:y2, x1:x2]
                processed = plate_preprocess(crop)
                raw, conf = self.ocr.read(processed)
                valid     = self.validator.validate(raw)
                self.stabilizer.update(pid, valid)
                stable, ratio = self.stabilizer.get_stable_plate(pid)

                self._pid_state[pid] = {
                    "stable_plate":    stable,
                    "stability_ratio": ratio,
                    "bbox":            bbox,
                }

                self.fsm.transition(pid, stable, ratio, is_active=True)

                if stable:
                    self.identity.notify_stable_plate(pid, stable)
                    self.registry.register(stable, fn)

            except Exception as e:
                print(f"[Warning] PID {pid}: {e}")

        # Handle dropped tracks
        for pid in self._prev_active_pids - current_pids:
            self.fsm.transition(pid, "", 0.0, is_active=False)
            self.stabilizer.remove_track(pid)

        # Prune expired LOST tracks
        for pid in [p for p in self._pid_state
                    if p not in current_pids
                    and not self.identity.is_recently_lost(p)]:
            self._pid_state.pop(pid, None)
            self.fsm.remove(pid)

        self._prev_active_pids = current_pids

        # Build minimal tracks_info — only active tracks, only stable plates shown
        tracks_info = []
        for pid, bbox in id_pairs:
            s = self._pid_state.get(pid, {})
            x1, y1, x2, y2 = bbox
            tracks_info.append({
                "persistent_id": pid,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "stable_plate":  s.get("stable_plate", ""),
                "is_active":     True,
            })

        return draw_frame(frame, tracks_info, fps, self.config)
