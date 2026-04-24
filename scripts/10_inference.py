#!/usr/bin/env python3
"""
Full Video Analysis Pipeline - Inference with trained models (PyTorch).

This script is the PyTorch testing/QA entry point. The implementation lives in
scripts/utils/batch_video_analyzer.py; this file is a thin CLI wrapper that
delegates to run_main(args, backend='pytorch').

All public symbols (BoxingAnalyzer, compute_summary, save_json, etc.) are
re-exported so existing callers (e.g. 13_evaluate_test.py) continue to work.

Analyzes boxing videos using trained unified and action models.
Outputs JSON/CSV analysis and optionally annotated video.

NOTE: SSL workaround included for macOS certificate issues.

================================================================================
USAGE:
================================================================================

# Basic analysis (JSON output + annotated video)
python3 scripts/10_inference.py --video path/to/video.mp4

# JSON only (fastest)
python3 scripts/10_inference.py --video video.mp4 --save_video false

# CSV output
python3 scripts/10_inference.py --video video.mp4 --output_format csv

# Custom output folder
python3 scripts/10_inference.py --video video.mp4 --output results/fight_001/

# Higher detection confidence threshold
python3 scripts/10_inference.py --video video.mp4 --confidence 0.7

# Filter low-confidence attribute predictions (mark as 'uncertain')
python3 scripts/10_inference.py --video video.mp4 --attr_confidence 0.5

# Action confidence threshold (controls event detection sensitivity)
python3 scripts/10_inference.py --video video.mp4 --action_confidence 0.6

# Minimum frames between events from same fighter (default 3)
python3 scripts/10_inference.py --video video.mp4 --min_separation 3

# Gap-based event grouping (disables peak detection for non-defense actions)
python3 scripts/10_inference.py --video video.mp4 --use_gap_grouping

# Side-assignment options (one fighter visible, overlap/clinch, position swaps)
python3 scripts/10_inference.py --video video.mp4 --assign_single_fighter
python3 scripts/10_inference.py --video video.mp4 --side_confidence_min 0.5
python3 scripts/10_inference.py --video video.mp4 --stable_side_frames 5

# Save annotated frames as images
python3 scripts/10_inference.py --video video.mp4 --save_frames

# CPU inference
python3 scripts/10_inference.py --video video.mp4 --device cpu

# YOLO person detector (strict XOR with unified; no fallback)
python3 scripts/10_inference.py --video video.mp4 --yolo_run true
# Unified (default): omit the flag, or be explicit with --yolo_run false
python3 scripts/10_inference.py --video video.mp4 --yolo_run false
# Requires models/yolo/weights/best.pt AND configs/yolo_detector.yaml when true.
# Unified weights are NOT loaded in this mode; missing files exit non-zero.

================================================================================
ALL OPTIONS:
================================================================================

Required:
  --video PATH             # Path to input video file

Optional:
  --output PATH            # Output folder (default: results/{video_name}/)
  --output_format NAME     # json | csv (default: json)
  --save_video BOOL        # Save annotated video: true | false (default: true)
  --save_frames            # Save annotated frames as images
  --models PATH            # Path to models directory (default: models/)
                           # Action clips are resized per checkpoint img_size (112 if key missing)

Confidence:
  --confidence FLOAT       # Detection confidence threshold (default: 0.5)
  --attr_confidence FLOAT  # Attribute confidence; below -> 'uncertain' (default: 0.0)
  --action_confidence FLOAT # Action threshold for event detection (default: 0.6)
  --min_separation N       # Min frames between events from same fighter (default: 3)

Event grouping:
  --use_gap_grouping       # Use gap-based grouping for non-defense actions (peak detection off). Default: off.
                           # Works better (turns off peak detection when on)

Side-assignment (fighter identity from left/right):
  --assign_single_fighter  # When only 1 fighter tracked, assign action to that fighter
  --side_confidence_min FLOAT # Min side confidence to keep assignment (default: 0.0 = off)
  --stable_side_frames N   # Median position over last N frames for left/right (default: 0)

Device:
  --device NAME            # auto (CUDA then CPU; not MPS) | cuda | mps | cpu

Detector:
  --yolo_run BOOL          # true | false (also 1/0, yes/no, on/off). Default: false.
                           # true  = use Ultralytics YOLO (models/yolo/weights/best.pt
                           #         + configs/yolo_detector.yaml). Strict XOR with
                           #         unified — unified is NOT loaded.
                           # false = use the unified detector (default).
                           # Omitting the flag is equivalent to --yolo_run false.
                           # Exits non-zero if required files for the selected mode
                           # are missing (no cross-mode fallback).

================================================================================
SIDE-ASSIGNMENT OPTIONS (fighter identity from left/right mapping):
================================================================================

--assign_single_fighter  (default: False)
  When only 1 fighter is tracked, assign action to that fighter.
  Without this flag, 1-fighter frames get fighter='unknown'.

--side_confidence_min FLOAT  (default: 0.0 = off)
  Min side confidence (e.g. attacker_side_confidence) to keep fighter assignment.
  Below this threshold -> fighter set to 'unknown'. Use for overlap/clinch.

--stable_side_frames INT  (default: 0 = off)
  Use median center_x over last K frames per track for left/right mapping.
  Smooths brief position swaps. 0 = use current frame only.

================================================================================
CONFIG:
================================================================================

Action types, colors, fighter mapping, and event confidence keys are loaded
from configs/action_types.yaml. Event confidence keys come from
confidence_attributes (when set) or key_attributes (fallback), plus
fighter_field. No code change needed to add a new type — add it in YAML,
run 06 and train; inference uses it when models/actions/<type>/best.pt exists.

================================================================================
REQUIRED MODELS:
================================================================================

models/
├── unified/best.pt          # Detection + attributes (Faster R-CNN) — default mode
│   OR
├── unified_mps/best.pt      # Detection + attributes (SSD, for Apple Silicon)
│
├── yolo/weights/best.pt     # Three-class YOLO detector (ONLY when --yolo_run true)
│                            # Requires configs/yolo_detector.yaml at repo root.
│                            # No last.pt fallback for YOLO.
│
└── actions/                  # Action recognition (temporal) — always required
    ├── punch/best.pt
    ├── defense/best.pt
    ├── footwork/best.pt
    └── clinch/best.pt

NOTE: Person detector selection is strict XOR. With --yolo_run false (default,
      also when omitted) the script auto-detects a unified checkpoint
      (best.pt → last.pt under unified/ then unified_mps/) and fails fast if
      none exists. With --yolo_run true it loads YOLO only; missing YOLO
      weights or configs/yolo_detector.yaml exit non-zero (no automatic
      fallback to unified).

INFERENCE BEHAVIOUR (Story 15):
- Frame buffer size for action models is taken from the loaded checkpoints:
  max of each action model's saved window_size (default 16 if none loaded).
  Models trained with --window 24 receive 24-frame clips at inference.
- Event counting: default is peak detection. Use --use_gap_grouping to switch
  non-defense actions to gap-based grouping (peak detection disabled for those).
  Defense remains punch-anchored in both modes. See main_usage_guides/05_VIDEO_INFERENCE.md.

================================================================================
OUTPUT:
================================================================================

results/{video_name}/
├── analysis.json            # Frame-by-frame analysis
├── summary.json             # Fight statistics
├── annotated.mp4            # Video with overlays (if --save_video)
├── frames/                  # Annotated frames (if --save_frames)
│   └── frame_000000.jpg, ...
└── csv/                     # CSV files (if --output_format csv)
    ├── persons.csv
    ├── punches.csv
    └── ...

================================================================================
CONFIDENCE THRESHOLDS:
================================================================================

Models output confidence scores (softmax probabilities) for each prediction.
Use these thresholds to filter uncertain predictions:

--confidence FLOAT       Detection confidence (default: 0.5)
                         Lower = more boxes (including false positives)
                         Higher = fewer boxes (only high-confidence)

--attr_confidence FLOAT  Attribute confidence (default: 0.0)
                         Predictions below this threshold are marked 'uncertain'
                         0.0 = show all predictions
                         0.5 = hide low-confidence attributes

Output includes confidence per attribute:
  {"corner": "blue", "corner_confidence": 0.87, "guard": "uncertain", "guard_confidence": 0.32}

Recommended settings:
  - Development:    --confidence 0.3 --attr_confidence 0.0
  - General use:    --confidence 0.5 --attr_confidence 0.0
  - High quality:   --confidence 0.6 --attr_confidence 0.5
  - Strict:         --confidence 0.8 --attr_confidence 0.7

================================================================================
ACTION COUNTING (punch / defense / footwork / clinch):
================================================================================

Event detection uses a single source of truth (compute_all_events) shared by
both the annotated video (RECENT EVENTS panel) and summary.json. summary.json
is the SINGLE SOURCE OF TRUTH for action counts and discrete events: UIs and
reports must use summary.json (actions + events list) and must NOT recompute
events from analysis.json. analysis.json is a by-product (raw per-frame data);
use it only for per-frame display.

Algorithm:
  - Punch, footwork, clinch: peak detection (local max in 10-frame window)
    - confidence >= --action_confidence (default 0.6)
    - At least --min_separation frames since last event from same fighter
  - Defense: PUNCH-ANCHORED (only counts within ±3 frames of a punch event)
    - No standalone defense without a nearby punch

--action_confidence FLOAT  (default: 0.6)
  Minimum confidence to consider a frame as an event candidate.
  Higher = fewer events (stricter), lower = more events.

--min_separation INT  (default: 3)
  Minimum frames between events from the SAME fighter.
  Red and blue can have events in the same frame (simultaneous punches).
  Lower = allows faster combos, higher = merges rapid actions.

================================================================================
TRACKING (IoU + appearance, box smoothing, carry-forward):
================================================================================

Detections are matched to tracks by combined score (0.7*IoU + 0.3*appearance).
Appearance uses L2-normalized ROI features; red/blue are stable IDs, not glove colors.
Box positions are smoothed (70% new, 30% old). Unmatched fighter tracks get a ghost
box using velocity prediction for up to 60 frames. Embeddings are internal only (not
saved to JSON).

Referee detection: When 2 fighter tracks exist, a third unmatched detection
is classified as referee (role='referee', id=-1, corner='unknown'). Role confidence
< 0.5 prevents creating a fighter track; low appearance similarity to both fighters
reinforces referee classification. See main_usage_guides/05_VIDEO_INFERENCE.md
(Referee Detection) and 02_TECHNICAL_OVERVIEW.md.

================================================================================
ANNOTATED VIDEO - EVENT LOG:
================================================================================

The annotated video shows a persistent "RECENT EVENTS" panel (top-left) with the last
5 discrete actions (punches, defenses, footwork, clinches). Events are logged at the
frame where action confidence is highest (the "peak" frame), so the display does not
flicker. Whether that frame aligns with "fully extended arm" / "punch landed" depends
on how the action model was trained. See main_usage_guides/05_VIDEO_INFERENCE.md
(Event Log & Peak Frame Selection) for details.
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

# ---------------------------------------------------------------------------
# Re-export all public symbols from batch_video_analyzer so existing callers
# (e.g. 13_evaluate_test.py) that do `mod10.BoxingAnalyzer(...)` keep working.
# ---------------------------------------------------------------------------
from utils.batch_video_analyzer import (  # noqa: F401
    BoxingAnalyzer,
    compute_all_events,
    compute_summary,
    get_device,
    load_action_types_config,
    print_summary,
    run_main,
    save_csv,
    save_json,
    ACTION_TYPES,
    ACTION_TYPES_CONFIG_PATH,
    COLORS,
    CONFIDENCE_KEYS_MAP,
    FIGHTER_FIELDS_MAP,
    FIGHTER_KEYS_MAP,
    IMAGENET_MEAN,
    IMAGENET_STD,
    KEY_ATTRIBUTES_MAP,
    MultiHeadActionModel,
    UnifiedMPSModel,
    AttributeHead,
    SSDDetectionHead,
)


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze boxing video with trained models"
    )

    # Input
    parser.add_argument('--video', type=Path, required=True,
                        help='Path to input video file')

    # Output
    parser.add_argument('--output', type=Path, default=None,
                        help='Output folder (default: results/{video_name}/)')
    parser.add_argument('--output_format', type=str, default='json',
                        choices=['json', 'csv'],
                        help='Output format')
    parser.add_argument('--save_video', type=str, default='true',
                        help='Save annotated video (true/false)')
    parser.add_argument('--save_frames', action='store_true',
                        help='Save annotated frames as images')

    # Models
    parser.add_argument('--models', type=Path, default=Path('models/'),
                        help='Path to models directory')

    # Inference
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Detection confidence threshold')
    parser.add_argument('--attr_confidence', type=float, default=0.0,
                        help='Attribute confidence threshold (0=show all, 0.5=hide low confidence)')
    parser.add_argument('--action_confidence', type=float, default=0.6,
                        help='Action confidence threshold for event detection (default 0.6)')
    parser.add_argument('--min_separation', type=int, default=3,
                        help='Minimum frames between events from same fighter (default 3)')

    # Side-based assignment options
    parser.add_argument('--assign_single_fighter', action='store_true', default=False,
                        help='When only 1 fighter tracked, assign action to that fighter (default: unknown)')
    parser.add_argument('--side_confidence_min', type=float, default=0.0,
                        help='Min side confidence to keep fighter assignment; below this -> unknown (default 0.0 = off)')
    parser.add_argument('--stable_side_frames', type=int, default=0,
                        help='Use median position over last K frames for left/right mapping (0 = use current frame only)')

    # Event grouping
    parser.add_argument('--use_gap_grouping', action='store_true', default=False,
                        help='Use gap-based event grouping for all non-defense actions. '
                             'When enabled, peak detection is disabled for those actions. '
                             'Gap threshold is auto-derived from the action model temporal '
                             'window (~0.75 * window_size). Default: off (peak detection).')

    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto = CUDA if available else CPU; use mps only with --device mps')

    # Detector selection (strict XOR with unified; no fallback, no last.pt for YOLO)
    parser.add_argument('--yolo_run', type=_parse_bool, default=False,
                        metavar='BOOL',
                        help='Use the Ultralytics YOLO three-class detector '
                             '(models/yolo/weights/best.pt + configs/yolo_detector.yaml) '
                             'instead of the unified model. Accepts true/false '
                             '(also 1/0, yes/no, on/off). Omitted or "false" = '
                             'unified; "true" = YOLO. Exits if required files are '
                             'missing; no fallback to unified. Default: false.')

    return parser.parse_args()


def _check_pytorch_artifacts(args: argparse.Namespace) -> None:
    """Fail-fast preflight for the PyTorch batch pipeline.

    Strict XOR: when ``--yolo_run`` is set we require the YOLO .pt weight and
    its YAML contract; otherwise we require a resolvable unified checkpoint
    under ``models/unified/`` or ``models/unified_mps/`` (best.pt → last.pt).
    Missing required artifacts exit non-zero *before* any video analysis.
    """
    models_dir = Path(args.models)

    if args.yolo_run:
        yolo_weights = models_dir / 'yolo' / 'weights' / 'best.pt'
        yolo_cfg = Path('configs/yolo_detector.yaml')
        missing = [p for p in (yolo_weights, yolo_cfg) if not p.exists()]
        if missing:
            print("ERROR: --yolo_run true requires the following files:")
            for p in missing:
                print(f"  - {p} (missing)")
            print("Train / export YOLO (see main_usage_guides/09_MODEL_TRAINING.md) "
                  "or use --yolo_run false (the default) to run the unified detector.")
            sys.exit(1)
        return

    unified_candidates = [
        models_dir / 'unified' / 'best.pt',
        models_dir / 'unified' / 'last.pt',
        models_dir / 'unified_mps' / 'best.pt',
        models_dir / 'unified_mps' / 'last.pt',
    ]
    if not any(p.exists() for p in unified_candidates):
        print("ERROR: No unified PyTorch checkpoint found. Looked for:")
        for p in unified_candidates:
            print(f"  - {p}")
        print("Train a unified model (see main_usage_guides/09_MODEL_TRAINING.md) "
              "or pass --yolo_run true to use the YOLO detector.")
        sys.exit(1)


def main():
    args = parse_args()
    _check_pytorch_artifacts(args)
    run_main(args, backend='pytorch')


if __name__ == '__main__':
    main()
