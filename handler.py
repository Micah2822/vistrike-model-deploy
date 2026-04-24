"""
RunPod Serverless handler for VISTRIKE video inference (ONNX Runtime).

Receives a job with a video URL + params, runs ONNX inference via
inference_onnx pipeline, uploads results to Supabase Storage, returns
summary + artifact URLs.

Progress payload contract:
  Every runpod.serverless.progress_update payload always includes
  score_range, detection_threshold, and boxes_above_threshold.
  Unknown values are sent as the explicit string "unknown".

Inference defaults:
  All inference knobs (device, thresholds, grouping, models_dir, yolo_run,
  etc.) live in configs/inference.yaml. Edit that file (and rebuild the
  image, or mount a replacement) to change defaults. Per-job values in
  job["input"] override the YAML for these keys only:
    confidence, attr_confidence, action_confidence, save_video, min_separation
  All other YAML keys are config-only and require a worker restart to apply.

Detector selection:
  configs/inference.yaml → `yolo_run: false` (default) loads the unified
  ONNX detector (models/unified/unified_model.onnx + .meta.json).
  `yolo_run: true` loads the Ultralytics YOLO three-class detector
  (models/yolo/weights/best.onnx + configs/yolo_detector.yaml) instead;
  the unified session is NOT created. Strict XOR — missing files for the
  selected mode fail fast at worker startup.

Environment variables (set in RunPod dashboard → endpoint → secrets):
  SUPABASE_URL              – e.g. https://xxxx.supabase.co
  SUPABASE_SERVICE_ROLE_KEY – service-role key (NOT the anon key)
  SUPABASE_BUCKET           – storage bucket name, e.g. "vistrike-results"
  INFERENCE_CONFIG_PATH     – optional path to the inference YAML. Defaults
                            to /app/configs/inference.yaml.
"""

import io
import os
import sys
import json
import time
import tempfile
import urllib.request
from pathlib import Path

import requests
import runpod
import yaml

PROJECT_ROOT = Path("/app")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
os.chdir(str(PROJECT_ROOT))

from utils.batch_video_analyzer import BoxingAnalyzer, compute_summary, save_json
from inference_onnx import _check_onnx_artifacts

# ---------------------------------------------------------------------------
# Inference configuration (YAML)
# ---------------------------------------------------------------------------

DEFAULT_INFERENCE_CONFIG_PATH = PROJECT_ROOT / "configs" / "inference.yaml"

# Keys accepted from job["input"] as per-run overrides. Everything else in the
# YAML is config-only and requires a worker restart to change.
_INPUT_OVERRIDE_KEYS = (
    "confidence",
    "attr_confidence",
    "action_confidence",
    "save_video",
    "min_separation",
)


def _load_inference_config() -> dict:
    """Read configs/inference.yaml (or INFERENCE_CONFIG_PATH) once at import."""
    override = os.environ.get("INFERENCE_CONFIG_PATH", "").strip()
    path = Path(override) if override else DEFAULT_INFERENCE_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Inference config not found at {path}. Set INFERENCE_CONFIG_PATH "
            f"or ensure configs/inference.yaml is present in the image."
        )
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Invalid YAML in inference config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(
            f"Inference config at {path} must be a YAML mapping, got {type(data).__name__}"
        )
    print(f"Loaded inference config from {path}")
    return data


INFERENCE_DEFAULTS = _load_inference_config()


def _coerce_bool(value, fallback: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    if value is None:
        return fallback
    return bool(value)


def _merge_inference(job_input: dict) -> dict:
    """Overlay whitelisted job["input"] keys on top of INFERENCE_DEFAULTS."""
    effective = dict(INFERENCE_DEFAULTS)
    for key in _INPUT_OVERRIDE_KEYS:
        if key not in job_input or job_input[key] is None:
            continue
        raw = job_input[key]
        if key in ("confidence", "attr_confidence", "action_confidence"):
            effective[key] = float(raw)
        elif key == "min_separation":
            effective[key] = int(raw)
        elif key == "save_video":
            effective[key] = _coerce_bool(raw, bool(effective.get(key, True)))
        else:
            effective[key] = raw
    return effective

# ---------------------------------------------------------------------------
# Supabase Storage helpers
# ---------------------------------------------------------------------------

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "vistrike-results")


def _supabase_headers(content_type: str = "application/octet-stream") -> dict:
    return {
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "apikey": SUPABASE_KEY,
        "Content-Type": content_type,
    }


def supabase_upload(filepath: Path, object_path: str) -> str:
    """Upload a file to Supabase Storage. Returns the public HTTPS URL."""
    content_type = "application/json"
    if filepath.suffix == ".mp4":
        content_type = "video/mp4"

    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{object_path}"
    with open(filepath, "rb") as f:
        resp = requests.post(
            url,
            headers=_supabase_headers(content_type),
            data=f,
            timeout=300,
        )

    if resp.status_code == 400 and "Duplicate" in resp.text:
        resp = requests.put(
            url,
            headers=_supabase_headers(content_type),
            data=open(filepath, "rb").read(),
            timeout=300,
        )

    resp.raise_for_status()
    public_url = (
        f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{object_path}"
    )
    return public_url


def _supabase_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

MODEL = None

YOLO_RUN = _coerce_bool(INFERENCE_DEFAULTS.get("yolo_run"), False)


def _detector_weights_present(models_root: Path, yolo_run: bool) -> bool:
    """True if the artifacts required by the selected detector live under models_root.

    - ``yolo_run=False``: look for a unified model artifact (.pt or ONNX).
    - ``yolo_run=True``: look for the exported YOLO ONNX weight
      (``<models_root>/yolo/weights/best.onnx``). The ONNX runtime is the
      only backend this handler uses, so we do not probe for ``best.pt``.
    """
    if yolo_run:
        return (models_root / "yolo" / "weights" / "best.onnx").exists()

    u = models_root / "unified"
    m = models_root / "unified_mps"
    for base in (u, m):
        if (base / "best.pt").exists() or (base / "last.pt").exists():
            return True
    if (u / "unified_model.onnx").exists():
        return True
    return False


def _diagnose_paths(yolo_run: bool = False):
    """Print diagnostic info about model directories so volume issues are obvious."""
    vol_root = Path("/runpod-volume")
    vol_models = vol_root / "models"
    app_models = PROJECT_ROOT / "models"
    subdir = "yolo/weights" if yolo_run else "unified"

    print("--- Model path diagnostics ---")
    print(f"  detector mode: {'yolo' if yolo_run else 'unified'}")
    print(f"  /runpod-volume exists: {vol_root.exists()}")
    if vol_root.exists():
        try:
            contents = list(vol_root.iterdir())
            print(f"  /runpod-volume contents: {[p.name for p in contents]}")
        except PermissionError:
            print("  /runpod-volume: permission denied listing contents")

    print(f"  /runpod-volume/models exists: {vol_models.exists()}")
    if vol_models.exists():
        try:
            contents = list(vol_models.iterdir())
            print(f"  /runpod-volume/models contents: {[p.name for p in contents]}")
            inspect = vol_models / subdir
            if inspect.exists():
                print(f"  /runpod-volume/models/{subdir} contents: "
                      f"{[p.name for p in inspect.iterdir()]}")
        except PermissionError:
            print("  /runpod-volume/models: permission denied listing contents")

    print(f"  /app/models exists: {app_models.exists()}")
    if app_models.exists():
        try:
            contents = list(app_models.iterdir())
            print(f"  /app/models contents: {[p.name for p in contents]}")
        except Exception:
            pass
    print("--- End diagnostics ---")


def _resolve_models_dir(yolo_run: bool) -> str:
    """Pick weights directory: YAML models_dir, else baked /app/models, else volume.

    The probe depends on ``yolo_run``: unified artifacts for the unified
    detector, YOLO ONNX weights for YOLO mode. This prevents a YOLO-only
    deploy from silently pointing at a stale unified-models directory (or
    vice versa).
    """
    override = str(INFERENCE_DEFAULTS.get("models_dir") or "").strip()
    if override:
        print(f"models_dir override from inference config: {override}")
        return override

    app_models = PROJECT_ROOT / "models"
    vol_models = Path("/runpod-volume/models")
    label = "YOLO weights" if yolo_run else "unified checkpoint"

    if _detector_weights_present(app_models, yolo_run):
        print(f"Found {label} in /app/models (baked into image)")
        return str(app_models)
    if _detector_weights_present(vol_models, yolo_run):
        print(f"Found {label} in /runpod-volume/models (network volume)")
        return str(vol_models)

    _diagnose_paths(yolo_run)
    print(
        f"WARNING: No {label} found in /app/models or /runpod-volume/models. "
        "Is the network volume attached to this endpoint? "
        "(Serverless → endpoint → Edit → Advanced → Network Volumes)"
    )
    return str(app_models)


def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    cfg = INFERENCE_DEFAULTS
    device = "cpu" if str(cfg.get("device", "cuda")).lower() == "cpu" else "cuda"
    models_dir = _resolve_models_dir(YOLO_RUN)
    print(f"Using models_dir={models_dir} (yolo_run={YOLO_RUN})")
    _check_onnx_artifacts(Path(models_dir), yolo_run=YOLO_RUN)
    MODEL = BoxingAnalyzer(
        models_dir=models_dir,
        device=device,
        confidence=float(cfg.get("confidence", 0.5)),
        attr_confidence=float(cfg.get("attr_confidence", 0.0)),
        action_confidence=float(cfg.get("action_confidence", 0.6)),
        min_event_separation=int(cfg.get("min_separation", 3)),
        assign_single_fighter=_coerce_bool(cfg.get("assign_single_fighter"), False),
        side_confidence_min=float(cfg.get("side_confidence_min", 0.0)),
        stable_side_frames=int(cfg.get("stable_side_frames", 0)),
        use_gap_grouping=_coerce_bool(cfg.get("use_gap_grouping"), True),
        backend="onnx",
        yolo_run=YOLO_RUN,
    )
    return MODEL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def download_video(url: str, dest: Path):
    print(f"Downloading video from {url}")
    urllib.request.urlretrieve(url, str(dest))
    size_mb = dest.stat().st_size / 1024 / 1024
    print(f"Downloaded {size_mb:.1f} MB")


class _TeeStream:
    """Write to both the original stream and a StringIO buffer."""

    def __init__(self, original, buffer):
        self._original = original
        self._buffer = buffer

    def write(self, data):
        self._original.write(data)
        try:
            self._buffer.write(data)
        except Exception:
            pass
        return len(data)

    def flush(self):
        self._original.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)


def _progress(job, message: str, percent: int, status: str = "processing",
              log_buffer=None):
    """Send a progress update the Vercel proxy can forward to the browser."""
    payload = {
        "message": message,
        "percent": percent,
        "status": status,
        "score_range": "unknown",
        "detection_threshold": "unknown",
        "boxes_above_threshold": "unknown",
    }
    if log_buffer is not None:
        payload["logs"] = log_buffer.getvalue()[-50_000:]
    runpod.serverless.progress_update(job, payload)


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def handler(job):
    job_input = job.get("input") or {}

    # --- Required input -------------------------------------------------------
    video_url = job_input.get("video_url")
    if not video_url:
        return {"error": "Missing video_url in input"}

    # --- Merge YAML defaults with per-job overrides ---------------------------
    effective = _merge_inference(job_input)
    confidence = float(effective["confidence"])
    attr_confidence = float(effective["attr_confidence"])
    action_confidence = float(effective["action_confidence"])
    min_separation = int(effective["min_separation"])
    save_video = _coerce_bool(effective.get("save_video"), True)

    # --- Tee stdout/stderr into an in-memory buffer for live log streaming ----
    log_buffer = io.StringIO()
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout = _TeeStream(_orig_stdout, log_buffer)
        sys.stderr = _TeeStream(_orig_stderr, log_buffer)
    except Exception:
        pass  # fall back to normal streams if tee setup fails

    try:
        _progress(job, "Downloading video…", 0, "downloading", log_buffer)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            video_path = tmpdir / "input_video.mp4"
            output_dir = tmpdir / "output"
            output_dir.mkdir()

            download_video(video_url, video_path)

            _progress(job, "Loading model…", 5, "loading_model", log_buffer)
            analyzer = load_model()
            analyzer.confidence = confidence
            analyzer.attr_confidence = attr_confidence
            analyzer.action_confidence = action_confidence
            analyzer.min_event_separation = min_separation

            _progress(job, "Analyzing video…", 10, "analyzing", log_buffer)

            _last_progress_time = 0

            def on_frame_progress(current_frame, total_frames, **kwargs):
                nonlocal _last_progress_time
                now = time.time()
                is_last = current_frame >= total_frames - 1
                if not is_last and (now - _last_progress_time) < 0.5 and current_frame % 5 != 0:
                    return
                _last_progress_time = now
                pct = 10 + int((current_frame / max(total_frames, 1)) * 70)
                runpod.serverless.progress_update(job, {
                    "message": f"Analyzing frame {current_frame}/{total_frames}…",
                    "percent": pct,
                    "status": "analyzing",
                    "current_frame": current_frame,
                    "total_frames": total_frames,
                    "fps": kwargs.get("fps", 0),
                    "boxes_detected": kwargs.get("boxes_detected", 0),
                    "avg_confidence": kwargs.get("avg_confidence", 0),
                    "video_resolution": kwargs.get("video_resolution", ""),
                    "video_fps": kwargs.get("video_fps", 0),
                    "score_range": kwargs.get("score_range", "unknown"),
                    "detection_threshold": kwargs.get("detection_threshold", "unknown"),
                    "boxes_above_threshold": kwargs.get("boxes_above_threshold", "unknown"),
                    "logs": log_buffer.getvalue()[-50_000:],
                })

            start = time.time()
            results = analyzer.analyze_video(
                str(video_path), progress_callback=on_frame_progress
            )
            elapsed = time.time() - start

            _progress(job, "Computing summary…", 80, "computing_summary", log_buffer)

            summary = compute_summary(
                results,
                action_confidence=action_confidence,
                min_separation=min_separation,
                gap_threshold=analyzer.get_gap_threshold(),
            )

            analysis_path = output_dir / "analysis.json"
            summary_path = output_dir / "summary.json"
            save_json(results, analysis_path)
            save_json(summary, summary_path)

            annotated_path = None
            if save_video:
                _progress(job, "Creating annotated video…", 85, "creating_video",
                          log_buffer)
                annotated_path = output_dir / f"{video_path.stem}_annotated.mp4"
                analyzer.create_annotated_video(
                    str(video_path), results, str(annotated_path)
                )

            # --- Upload to Supabase --------------------------------------------
            job_id = job["id"]
            summary_url = None
            analysis_url = None
            video_url_out = None

            print(
                "Supabase config check:",
                {
                    "SUPABASE_URL_set": bool(SUPABASE_URL),
                    "SUPABASE_KEY_set": bool(SUPABASE_KEY),
                    "SUPABASE_KEY_len": len(SUPABASE_KEY or ""),
                    "SUPABASE_BUCKET": SUPABASE_BUCKET,
                },
            )
            if _supabase_configured():
                _progress(job, "Uploading results…", 92, "uploading_results",
                          log_buffer)
                prefix = f"results/{job_id}"
                print("Entering Supabase upload block", {"job_id": job_id, "prefix": prefix})
                summary_url = supabase_upload(summary_path,
                                              f"{prefix}/summary.json")
                analysis_url = supabase_upload(analysis_path,
                                               f"{prefix}/analysis.json")
                if save_video and annotated_path and annotated_path.exists():
                    _progress(job, "Uploading annotated video…", 96,
                              "uploading_video", log_buffer)
                    video_url_out = supabase_upload(
                        annotated_path, f"{prefix}/annotated.mp4"
                    )
            else:
                print("WARNING: Supabase not configured — skipping upload")

            print("FINAL_RETURN: sending completed payload")
            return {
                "status": "completed",
                "elapsed_seconds": round(elapsed, 1),
                "total_frames": results.get("total_frames", 0),
                "summary": summary,
                "summary_url": summary_url,
                "analysis_url": analysis_url,
                "video_url": video_url_out,
            }
    finally:
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr


# ---------------------------------------------------------------------------
# Entrypoint — load model at container start (warm worker), then serve jobs
# ---------------------------------------------------------------------------
load_model()
runpod.serverless.start({"handler": handler})
