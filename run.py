import cv2
import yaml
import sys
import os
from collections import deque
import time

from pipeline.alpr_pipeline import ALPRPipeline


def main():
    # Load configuration
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Auto-create log directory
    log_dir = config.get("logging", {}).get("log_directory", "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Initialize pipeline
    pipeline = ALPRPipeline(config, "config/plate_schema.yaml")

    # Video source: CLI arg overrides config
    if len(sys.argv) > 1:
        source = sys.argv[1]
    else:
        source = config.get("video", {}).get("source", "")

    if not source:
        print("Error: No video source specified.")
        print("  Usage : python -m run path/to/video.mp4")
        print("  Or set video.source in config/config.yaml")
        sys.exit(1)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Unable to open video source: {source}")
        sys.exit(1)

    raw_fps     = cap.get(cv2.CAP_PROP_FPS) or 30
    source_name = os.path.splitext(os.path.basename(str(source)))[0]
    output_path = os.path.join(os.path.dirname(str(source)), f"{source_name}_output.mp4")
    print(f"Source       : {source}")
    print(f"Output video : {output_path}")
    print(f"Logs folder  : {os.path.abspath(log_dir)}")

    writer: cv2.VideoWriter | None = None
    fps_deque: deque = deque(maxlen=30)
    display_fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            composite = pipeline.process_frame(frame, fps=display_fps)

            if writer is None:
                ch, cw = composite.shape[:2]
                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    raw_fps,
                    (cw, ch)
                )

            writer.write(composite)
            cv2.imshow("ALPR System", composite)

            fps_deque.append(time.perf_counter() - t0)
            if len(fps_deque) > 1:
                display_fps = len(fps_deque) / sum(fps_deque)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User quit.")
                break

    except Exception as e:
        print(f"[Fatal] {e}")

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("Done. Output saved.")


if __name__ == "__main__":
    main()