# License Plate Recognition — ALPR System

A production-grade Automatic License Plate Recognition (ALPR) pipeline for video streams. Detects, tracks, stabilizes, and logs vehicle number plates with a clean minimal UI.

## Features
- **YOLOv8** license plate detection
- **EasyOCR** for text recognition
- **SORT** tracker with persistent identity management (IOU-based ID reassignment, EMA smoothing)
- **Track state machine** (NEW → STABILIZING → STABLE → LOST) with flicker-free display
- **Unique vehicle registry** — logs each confirmed plate once to CSV
- Annotated video output saved alongside source file

## Project Structure
```
├── config/
│   ├── config.yaml         # All tunable parameters
│   └── plate_schema.yaml   # Plate regex + swap-correction map
├── detector/               # YOLOv8 wrapper
├── tracker/                # SORT + persistent identity manager + FSM
├── ocr1/                   # EasyOCR engine
├── preprocessing/          # Plate image preprocessing
├── stabilization/          # Voting-based plate stabilizer
├── validation/             # Regex validator + character swap correction
├── pipeline/               # Main ALPR pipeline
├── ui/                     # Minimal renderer (bbox + plate label)
├── logging_module/         # Unique vehicle registry (CSV)
├── models/                 # Place your best.pt here (not committed)
├── logs/                   # Auto-created at runtime
└── run.py                  # Entry point
```

## Setup

```bash
# Create and activate virtual environment
python -m venv ocr
ocr\Scripts\activate      # Windows
# source ocr/bin/activate  # Linux/macOS

# Install dependencies
pip install ultralytics easyocr opencv-python pyyaml numpy scipy filterpy
```

> **Model:** Download or train your YOLOv8 license plate model and place it at `models/best.pt`.

## Usage

```bash
# Pass video path as CLI argument
python -m run "path/to/your_video.mp4"

# Or set video.source in config/config.yaml and run:
python -m run
```

Output video is saved as `<source_name>_output.mp4` in the same directory as the input.  
Unique vehicle CSV is saved to `logs/unique_vehicles_<timestamp>.csv`.

## Configuration (`config/config.yaml`)

| Key | Description |
|-----|-------------|
| `yolo.conf_threshold` | Detection confidence threshold |
| `tracker.max_age` | Frames to keep a track alive without detection |
| `persistence.lost_buffer_frames` | Frames to attempt ID reassignment after track drop |
| `persistence.reassign_iou_thresh` | IOU threshold for re-linking a reappearing vehicle |
| `stabilization.min_samples` | Min valid OCR hits before displaying a plate |
| `stabilization.vote_threshold` | Fraction of agreeing readings required for stability |

## Requirements
- Python 3.10+
- CUDA GPU recommended (CPU works but slower)
