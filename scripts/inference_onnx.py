#!/usr/bin/env python3
"""
ONNX Production Inference — VISTRIKE Video Analysis (batch).

Production / Docker / hosted entry point for ONNX Runtime inference. Uses the
same pipeline as PyTorch (tracking, events, side mapping, gap grouping, etc.)
via scripts/utils/batch_video_analyzer.py with backend='onnx'.

For PyTorch testing and QA with .pt checkpoints, use scripts/10_inference.py.

================================================================================
CONTEXT:
================================================================================

- Runs from project root (or cd scripts) so configs/action_types.yaml and
  data/attributes resolve like other inference scripts.
- Requires unified_model.onnx + unified_model.meta.json under models/unified/.
  Action models are optional per-type: models/actions/<type>/<type>_model.onnx
  and matching .meta.json (from 12_export_onnx.py).
- onnxruntime-gpu recommended for NVIDIA production; onnxruntime for CPU.
- Startup logs active ORT execution providers (CUDA vs CPU).

================================================================================
USAGE:
================================================================================

# Basic (JSON + annotated video)
python3 scripts/inference_onnx.py --video path/to/video.mp4

# JSON only (faster)
python3 scripts/inference_onnx.py --video video.mp4 --save_video false

# CSV output
python3 scripts/inference_onnx.py --video video.mp4 --output_format csv

# Custom output folder
python3 scripts/inference_onnx.py --video video.mp4 --output results/fight_001/

# Detection / attribute / action thresholds
python3 scripts/inference_onnx.py --video video.mp4 --confidence 0.7
python3 scripts/inference_onnx.py --video video.mp4 --attr_confidence 0.5
python3 scripts/inference_onnx.py --video video.mp4 --action_confidence 0.6

# Event spacing and grouping
python3 scripts/inference_onnx.py --video video.mp4 --min_separation 3
python3 scripts/inference_onnx.py --video video.mp4 --use_gap_grouping

# Side-assignment (left/right to red/blue)
python3 scripts/inference_onnx.py --video video.mp4 --assign_single_fighter
python3 scripts/inference_onnx.py --video video.mp4 --side_confidence_min 0.5
python3 scripts/inference_onnx.py --video video.mp4 --stable_side_frames 5

# Annotated frames
python3 scripts/inference_onnx.py --video video.mp4 --save_frames

# Device (ORT: auto prefers CUDA if available, else CPU; MPS not used here)
python3 scripts/inference_onnx.py --video video.mp4 --device cuda
python3 scripts/inference_onnx.py --video video.mp4 --device cpu

# YOLO person detector (strict XOR with unified; no fallback)
python3 scripts/inference_onnx.py --video video.mp4 --yolo_run true
# Unified (default): omit the flag, or be explicit with --yolo_run false
python3 scripts/inference_onnx.py --video video.mp4 --yolo_run false
# Requires models/yolo/weights/best.onnx AND configs/yolo_detector.yaml when true.
# Unified ONNX is NOT loaded in this mode; missing files exit non-zero.
# Requires the `ultralytics` package at runtime.

================================================================================
ALL OPTIONS:
================================================================================

Required:
  --video PATH             Path to input video file

Optional:
  --output PATH            Output folder (default: results/{video_name}/)
  --output_format NAME     json | csv (default: json)
  --save_video BOOL        Save annotated video: true | false (default: true)
  --save_frames            Save annotated frames as images
  --models PATH            Models directory (default: models/)

Confidence:
  --confidence FLOAT       Detection confidence threshold (default: 0.5)
  --attr_confidence FLOAT  Attribute confidence; below -> uncertain (default: 0.0)
  --action_confidence FLOAT Action threshold for events (default: 0.6)
  --min_separation N       Min frames between events from same fighter (default: 3)

Event grouping:
  --use_gap_grouping       Gap-based grouping for non-defense; peak detection off for those - works better

Side-assignment:
  --assign_single_fighter  When only one fighter tracked, map actions to that fighter
  --side_confidence_min FLOAT  Min side confidence to keep assignment (default: 0.0)
  --stable_side_frames N   Median box position over last K frames for L/R (default: 0)

Device:
  --device NAME            auto | cuda | cpu (default: auto; ORT providers)

Detector:
  --yolo_run BOOL          true | false (also 1/0, yes/no, on/off). Default: false.
                           true  = Ultralytics YOLO (models/yolo/weights/best.onnx
                                   + configs/yolo_detector.yaml). Strict XOR with
                                   unified — unified ONNX is NOT loaded.
                                   Requires the `ultralytics` package.
                           false = Unified ONNX detector (default; also the
                                   behavior when the flag is omitted).
                           Exits non-zero if required files for the selected mode
                           are missing (no cross-mode fallback).

================================================================================
REQUIRED ARTIFACTS:
================================================================================

Unified mode (default):
  models/unified/unified_model.onnx
  models/unified/unified_model.meta.json

YOLO mode (--yolo_run true; strict XOR with unified):
  models/yolo/weights/best.onnx         # exported via Ultralytics (not 12_export_onnx.py)
  configs/yolo_detector.yaml            # class order + imgsz + default conf/iou

Optional (per action type from configs/action_types.yaml, both modes):
  models/actions/<type>/<type>_model.onnx
  models/actions/<type>/<type>_model.meta.json

Generate unified + actions with: python3 scripts/12_export_onnx.py --model all
Export YOLO separately:
  yolo export model=models/yolo/weights/best.pt format=onnx imgsz=640

================================================================================
OUTPUT:
================================================================================

results/{video_name}/  (or --output)
├── analysis.json
├── summary.json
├── {video_stem}_annotated.mp4   (if --save_video true)
└── frames/                      (if --save_frames)

================================================================================
CONFIG:
================================================================================

Same as 10_inference: configs/action_types.yaml drives action types, colors,
fighter keys, and event confidence keys. Ship the same YAML in prod as in QA.

================================================================================
"""

import argparse
import sys
from pathlib import Path


def _parse_bool(value: str) -> bool:
    """Parse a CLI boolean argument (accepts true/false, 1/0, yes/no, on/off).

    Used by ``--yolo_run`` so the flag takes an explicit value instead of
    behaving as a bare switch, e.g. ``--yolo_run true`` / ``--yolo_run false``.
    """
    v = str(value).strip().lower()
    if v in ('1', 'true', 't', 'yes', 'y', 'on'):
        return True
    if v in ('0', 'false', 'f', 'no', 'n', 'off'):
        return False
    raise argparse.ArgumentTypeError(
        f"expected a boolean value (true/false, 1/0, yes/no, on/off); got {value!r}"
    )


def _check_onnx_artifacts(models_dir: Path, yolo_run: bool = False):
    """Fail fast if required ONNX artifacts are missing.

    Strict XOR between the unified detector and YOLO:

    - ``yolo_run=False`` (default): require unified ``.onnx`` + ``.meta.json``.
    - ``yolo_run=True``: require ``models/yolo/weights/best.onnx`` and
      ``configs/yolo_detector.yaml``; unified ONNX is NOT required.

    Action ONNX models are optional in either mode — a warning is printed if
    none are present so callers can still run detector-only smoke tests.
    """
    if yolo_run:
        yolo_weights = models_dir / 'yolo' / 'weights' / 'best.onnx'
        yolo_cfg = Path('configs/yolo_detector.yaml')
        missing = [p for p in (yolo_weights, yolo_cfg) if not p.exists()]
        if missing:
            print("ERROR: --yolo_run true requires the following files:")
            for p in missing:
                print(f"  - {p} (missing)")
            print("Export YOLO to ONNX (see main_usage_guides/09_MODEL_TRAINING.md) "
                  "or use --yolo_run false (the default) to run the unified ONNX detector.")
            sys.exit(1)
        detector_summary = f"yolo OK ({yolo_weights})"
    else:
        unified_onnx = models_dir / 'unified' / 'unified_model.onnx'
        unified_meta = models_dir / 'unified' / 'unified_model.meta.json'

        if not unified_onnx.exists():
            print(f"ERROR: Unified ONNX model not found: {unified_onnx}")
            print("Run 12_export_onnx.py to generate ONNX exports first.")
            sys.exit(1)
        if not unified_meta.exists():
            print(f"ERROR: Unified metadata not found: {unified_meta}")
            print("Run 12_export_onnx.py to generate .meta.json sidecars.")
            sys.exit(1)
        detector_summary = "unified OK"

    actions_dir = models_dir / 'actions'
    found_actions = 0
    if actions_dir.exists():
        for action_dir in sorted(actions_dir.iterdir()):
            if action_dir.is_dir():
                onnx_file = action_dir / f'{action_dir.name}_model.onnx'
                meta_file = action_dir / f'{action_dir.name}_model.meta.json'
                if onnx_file.exists():
                    if not meta_file.exists():
                        print(f"WARNING: ONNX found but metadata missing: {meta_file}")
                    found_actions += 1

    if found_actions == 0:
        print("WARNING: No action ONNX models found. Action detection will be unavailable.")

    print(f"Artifacts check: {detector_summary}, {found_actions} action model(s) found.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ONNX production inference for boxing video analysis"
    )

    parser.add_argument('--video', type=Path, required=True,
                        help='Path to input video file')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output folder (default: results/{video_name}/)')
    parser.add_argument('--output_format', type=str, default='json',
                        choices=['json', 'csv'],
                        help='Output format')
    parser.add_argument('--save_video', type=str, default='true',
                        help='Save annotated video (true/false)')
    parser.add_argument('--save_frames', action='store_true',
                        help='Save annotated frames as images')
    parser.add_argument('--models', type=Path, default=Path('models/'),
                        help='Path to models directory')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Detection confidence threshold')
    parser.add_argument('--attr_confidence', type=float, default=0.0,
                        help='Attribute confidence threshold')
    parser.add_argument('--action_confidence', type=float, default=0.6,
                        help='Action confidence threshold')
    parser.add_argument('--min_separation', type=int, default=3,
                        help='Minimum frames between events from same fighter')
    parser.add_argument('--assign_single_fighter', action='store_true', default=False,
                        help='Assign action to single visible fighter')
    parser.add_argument('--side_confidence_min', type=float, default=0.0,
                        help='Min side confidence to keep fighter assignment')
    parser.add_argument('--stable_side_frames', type=int, default=0,
                        help='Median position over last K frames for left/right')
    parser.add_argument('--use_gap_grouping', action='store_true', default=False,
                        help='Use gap-based event grouping')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cuda/cpu)')

    parser.add_argument('--yolo_run', type=_parse_bool, default=False,
                        metavar='BOOL',
                        help='Use the Ultralytics YOLO three-class detector '
                             '(models/yolo/weights/best.onnx + configs/yolo_detector.yaml) '
                             'instead of the unified ONNX model. Accepts true/false '
                             '(also 1/0, yes/no, on/off). Omitted or "false" = unified; '
                             '"true" = YOLO. Strict XOR — unified ONNX is not loaded '
                             'in YOLO mode; missing files exit non-zero. Requires the '
                             '`ultralytics` package when true. Default: false.')

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.video.exists():
        print(f"ERROR: Video not found: {args.video}")
        sys.exit(1)

    _check_onnx_artifacts(args.models, yolo_run=args.yolo_run)

    # Add scripts/ to sys.path so relative imports in BVA work
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from utils.batch_video_analyzer import run_main
    run_main(args, backend='onnx')


if __name__ == '__main__':
    main()
