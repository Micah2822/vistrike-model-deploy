#!/usr/bin/env python3
"""
ONNX vs PyTorch Parity Validation — Release Gate.

Runs the same golden video through scripts/10_inference.py (PyTorch, .pt) and
scripts/inference_onnx.py (ONNX Runtime, .onnx + .meta.json), then compares
summary.json at a summary level (event counts per action type, detection
stats). Use before shipping hosted ONNX builds; not a bitwise box match.

Unified detection can differ slightly (NMS, fixed export resolution); action
models should be closer. Adjust --event_count_tolerance to your release policy.

================================================================================
CONTEXT:
================================================================================

- Invokes both CLIs as subprocesses from the project repo (run from project root).
- Requires full dev stack for PyTorch run + exported ONNX + meta for ONNX run.
- Compares summary['actions'][*].count, detection aggregates, and event list length.
- Exit 0 = all per-action-type event count diffs within tolerance; 1 = failure.

================================================================================
USAGE:
================================================================================

# Single golden clip (default tolerance)
python3 scripts/validate_onnx_parity.py --video path/to/golden_clip.mp4

# Stricter or looser event-count tolerance
python3 scripts/validate_onnx_parity.py --video clip.mp4 --event_count_tolerance 1
python3 scripts/validate_onnx_parity.py --video clip.mp4 --event_count_tolerance 5

# GPU for both runs
python3 scripts/validate_onnx_parity.py --video clip.mp4 --device cuda

# Match inference confidence used in QA
python3 scripts/validate_onnx_parity.py --video clip.mp4 --confidence 0.6

================================================================================
ALL OPTIONS:
================================================================================

Required:
  --video PATH             Golden test clip (short MP4 recommended)

Optional:
  --models PATH            Models directory for both runs (default: models/).
                           Forwarded to 10_inference.py and inference_onnx.py as --models

Confidence (forwarded to both inference subprocesses):
  --confidence FLOAT       Detection confidence (default: 0.5)

Device (forwarded to both subprocesses):
  --device NAME            auto | cuda | cpu | mps (default: auto)

Parity:
  --event_count_tolerance INT  Max allowed |PT count - ONNX count| per action
                               type before FAIL (default: 2)

================================================================================
PREREQUISITES:
================================================================================

PyTorch path:
  models/unified/best.pt (or last.pt), models/actions/<type>/best.pt as needed

ONNX path:
  models/unified/unified_model.onnx + unified_model.meta.json
  models/actions/<type>/<type>_model.onnx + .meta.json per type

Export ONNX + meta: python3 scripts/12_export_onnx.py --model all

================================================================================
EXIT CODES:
================================================================================

  0  Parity OK within tolerance (or comparison logic passes)
  1  Video missing, either subprocess failed, or event counts out of tolerance

================================================================================
OUTPUT:
================================================================================

Temporary directories only; prints comparison tables to stdout. Does not write
results beside the video. For full artifacts, run 10_inference.py or
inference_onnx.py directly.

Wall time: each run uses the same timer — time.perf_counter() around subprocess.run
only, with cwd=repo root and the same flags except the script (PT vs ONNX). Not
kernel-only; it is full end-to-end for that CLI (startup, load, video, inference,
write JSON). Compared fairly because both are measured the same way.

================================================================================
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PT_SCRIPT = SCRIPT_DIR / '10_inference.py'
ONNX_SCRIPT = SCRIPT_DIR / 'inference_onnx.py'


def _preflight(models: Path) -> bool:
    """Check that required model files exist before launching subprocesses."""
    ok = True
    unified = models / 'unified'

    pt_weights = unified / 'best.pt'
    pt_alt = unified / 'last.pt'
    if not pt_weights.exists() and not pt_alt.exists():
        print(f"  WARN: No PyTorch weights found ({pt_weights} or {pt_alt})")
        ok = False

    onnx_model = unified / 'unified_model.onnx'
    onnx_meta = unified / 'unified_model.meta.json'
    if not onnx_model.exists():
        print(f"  WARN: ONNX model missing ({onnx_model})")
        ok = False
    if not onnx_meta.exists():
        print(f"  WARN: ONNX metadata missing ({onnx_meta})")
        ok = False

    return ok


def run_inference(
    script: Path,
    video: Path,
    output_dir: Path,
    device: str,
    confidence: float,
    models: Path,
):
    """Run an inference script and return (summary_dict, wall_seconds) or (None, wall_seconds).

    Wall time is measured identically for PyTorch and ONNX: elapsed
    time.perf_counter() seconds for subprocess.run(...) only (same cwd, same argv
    shape except script path).
    """
    cmd = [
        'python3', '-u', str(script),
        '--video', str(video),
        '--output', str(output_dir),
        '--models', str(models),
        '--confidence', str(confidence),
        '--device', device,
        '--save_video', 'false',
        '--output_format', 'json',
    ]
    print(f"  Running: {' '.join(cmd)}")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    wall_s = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode}) [{wall_s:.1f}s]:")
        print(result.stdout[-2000:] if result.stdout else '')
        print(result.stderr[-2000:] if result.stderr else '')
        return None, wall_s

    summary_path = output_dir / 'summary.json'
    if not summary_path.exists():
        print(f"  ERROR: summary.json not found at {summary_path}")
        return None, wall_s

    with open(summary_path) as f:
        return json.load(f), wall_s


def compare_summaries(
    pt_summary: dict,
    onnx_summary: dict,
    event_count_tolerance: int = 2,
    pt_wall: float = 0.0,
    onnx_wall: float = 0.0,
) -> bool:
    """Compare two summary dicts and report differences.

    Returns True if within tolerance.
    """
    ok = True

    # -- Timing (same formula for both: frames / wall_s) --
    pt_frames = pt_summary.get('total_frames', 0)
    onnx_frames = onnx_summary.get('total_frames', 0)
    pt_fps = pt_frames / pt_wall if pt_wall > 0 else 0.0
    onnx_fps = onnx_frames / onnx_wall if onnx_wall > 0 else 0.0

    print("\n--- Wall time (equal measurement: perf_counter around each subprocess only) ---")
    print("  (full CLI run each: startup, load, decode, inference, write JSON — PT vs ONNX)")
    print(f"  {'':15s}  {'Wall_s':>9s}  {'Frames':>7s}  {'Frames/s':>9s}")
    print(f"  {'PyTorch':15s}  {pt_wall:9.1f}  {pt_frames:7d}  {pt_fps:9.2f}")
    print(f"  {'ONNX':15s}  {onnx_wall:9.1f}  {onnx_frames:7d}  {onnx_fps:9.2f}")
    if pt_fps > 0 and onnx_fps > 0:
        print(f"  Throughput ratio (ONNX_frames/s / PT_frames/s): {onnx_fps / pt_fps:.2f}x")

    # -- Event counts --
    print("\n--- Event Count Comparison ---")
    pt_actions = pt_summary.get('actions', {})
    onnx_actions = onnx_summary.get('actions', {})

    all_types = sorted(set(pt_actions.keys()) | set(onnx_actions.keys()))
    for action_type in all_types:
        pt_count = pt_actions.get(action_type, {}).get('count', 0)
        onnx_count = onnx_actions.get(action_type, {}).get('count', 0)
        diff = abs(pt_count - onnx_count)
        status = 'OK' if diff <= event_count_tolerance else 'FAIL'
        if status == 'FAIL':
            ok = False
        print(f"  {action_type:15s}  PT={pt_count:3d}  ONNX={onnx_count:3d}  diff={diff:2d}  [{status}]")

    # -- Detection stats --
    print("\n--- Detection Stats ---")
    pt_det = pt_summary.get('detection', {})
    onnx_det = onnx_summary.get('detection', {})
    for key in ['frames_with_detections', 'total_person_detections']:
        pt_val = pt_det.get(key, 0)
        onnx_val = onnx_det.get(key, 0)

        if pt_val == 0 and onnx_val > 0:
            status = 'FAIL'
            detail = "PT=0 but ONNX non-zero -- check PT unified model load"
            ok = False
        elif onnx_val == 0 and pt_val > 0:
            status = 'FAIL'
            detail = "ONNX=0 but PT non-zero -- check ONNX export"
            ok = False
        else:
            pct = abs(pt_val - onnx_val) / max(pt_val, 1) * 100
            status = 'OK' if pct < 20 else 'WARN'
            detail = f"{pct:.1f}% diff"

        print(f"  {key:35s}  PT={pt_val:6d}  ONNX={onnx_val:6d}  ({detail})  [{status}]")

    # -- Event list length --
    print("\n--- Event List Length ---")
    pt_events = len(pt_summary.get('events', []))
    onnx_events = len(onnx_summary.get('events', []))
    diff = abs(pt_events - onnx_events)
    status = 'OK' if diff <= event_count_tolerance * max(len(all_types), 1) else 'WARN'
    print(f"  Total events: PT={pt_events}  ONNX={onnx_events}  diff={diff}  [{status}]")

    return ok


def main():
    parser = argparse.ArgumentParser(description='Validate ONNX vs PyTorch parity')
    parser.add_argument('--video', type=Path, required=True, help='Golden clip to test')
    parser.add_argument('--models', type=Path, default=Path('models/'), help='Models directory')
    parser.add_argument('--device', type=str, default='auto', help='Device for both runs')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--event_count_tolerance', type=int, default=2,
                        help='Max allowed difference in event count per action type')
    args = parser.parse_args()

    if not args.video.exists():
        print(f"ERROR: Video not found: {args.video}")
        sys.exit(1)

    models = args.models.resolve()

    print(f"\n{'='*60}")
    print("VISTRIKE ONNX Parity Validation")
    print(f"{'='*60}")
    print(f"Video:      {args.video}")
    print(f"Device:     {args.device}")
    print(f"Models:     {models}")
    print(f"Tolerance:  {args.event_count_tolerance} events/type")

    print("\n--- Preflight ---")
    if not _preflight(models):
        print("  Preflight warnings detected (see above). Continuing anyway...")
    else:
        print("  All expected model files found.")

    with tempfile.TemporaryDirectory(prefix='vistrike_parity_') as tmpdir:
        pt_dir = Path(tmpdir) / 'pytorch'
        onnx_dir = Path(tmpdir) / 'onnx'
        pt_dir.mkdir()
        onnx_dir.mkdir()

        print(f"\n[1/2] Running PyTorch inference...")
        pt_summary, pt_wall = run_inference(
            PT_SCRIPT, args.video, pt_dir, args.device, args.confidence, models,
        )
        if pt_summary is None:
            print("ABORT: PyTorch run failed.")
            sys.exit(1)
        print(f"  Done ({pt_wall:.1f}s)")

        print(f"\n[2/2] Running ONNX inference...")
        onnx_summary, onnx_wall = run_inference(
            ONNX_SCRIPT, args.video, onnx_dir, args.device, args.confidence, models,
        )
        if onnx_summary is None:
            print("ABORT: ONNX run failed.")
            sys.exit(1)
        print(f"  Done ({onnx_wall:.1f}s)")

        parity_ok = compare_summaries(
            pt_summary, onnx_summary,
            event_count_tolerance=args.event_count_tolerance,
            pt_wall=pt_wall,
            onnx_wall=onnx_wall,
        )

        print(f"\n{'='*60}")
        if parity_ok:
            print("RESULT: PARITY OK - within tolerance")
            print(f"{'='*60}\n")
        else:
            print("RESULT: PARITY FAILED - event counts or detection stats differ beyond tolerance")
            print(f"{'='*60}\n")
            sys.exit(1)


if __name__ == '__main__':
    main()
