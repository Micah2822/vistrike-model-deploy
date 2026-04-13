"""
RunPod Serverless handler for VISTRIKE video inference (ONNX Runtime).

Receives a job with a video URL + params, runs ONNX inference via
inference_onnx pipeline, uploads results to Supabase Storage, returns
summary + artifact URLs.

Progress payload contract:
  Every runpod.serverless.progress_update payload always includes
  score_range, detection_threshold, and boxes_above_threshold.
  Unknown values are sent as the explicit string "unknown".

Environment variables (set in RunPod dashboard → endpoint → secrets):
  SUPABASE_URL              – e.g. https://xxxx.supabase.co
  SUPABASE_SERVICE_ROLE_KEY – service-role key (NOT the anon key)
  SUPABASE_BUCKET           – storage bucket name, e.g. "vistrike-results"
  DEVICE                    – "cuda" (default) or "cpu"
  DEFAULT_CONFIDENCE        – detection threshold (default 0.5)
  DEFAULT_ATTR_CONFIDENCE   – attribute threshold (default 0.0)
  DEFAULT_ACTION_CONFIDENCE – action event threshold (default 0.6)
  USE_GAP_GROUPING          – "true" to use gap-based event grouping (default "true").
                            Gap grouping replaces peak detection for non-defense
                            action types and generally produces more accurate counts.
  MODELS_DIR                – optional override for weights directory. If unset,
                            uses /app/models when it contains a unified checkpoint,
                            else /runpod-volume/models (RunPod network volume).
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

PROJECT_ROOT = Path("/app")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
os.chdir(str(PROJECT_ROOT))

from utils.batch_video_analyzer import BoxingAnalyzer, compute_summary, save_json
from inference_onnx import _check_onnx_artifacts

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


def _unified_checkpoint_present(models_root: Path) -> bool:
    """True if a unified model artifact is present under models_root (.pt or ONNX)."""
    u = models_root / "unified"
    m = models_root / "unified_mps"
    for base in (u, m):
        if (base / "best.pt").exists() or (base / "last.pt").exists():
            return True
    if (u / "unified_model.onnx").exists():
        return True
    return False


def _diagnose_paths():
    """Print diagnostic info about model directories so volume issues are obvious."""
    vol_root = Path("/runpod-volume")
    vol_models = vol_root / "models"
    app_models = PROJECT_ROOT / "models"

    print("--- Model path diagnostics ---")
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
            unified = vol_models / "unified"
            if unified.exists():
                u_contents = list(unified.iterdir())
                print(f"  /runpod-volume/models/unified contents: {[p.name for p in u_contents]}")
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


def _resolve_models_dir() -> str:
    """Pick weights directory: MODELS_DIR env, else baked /app/models, else volume."""
    override = os.environ.get("MODELS_DIR", "").strip()
    if override:
        print(f"MODELS_DIR override set: {override}")
        return override

    app_models = PROJECT_ROOT / "models"
    vol_models = Path("/runpod-volume/models")

    if _unified_checkpoint_present(app_models):
        print("Found unified checkpoint in /app/models (baked into image)")
        return str(app_models)
    if _unified_checkpoint_present(vol_models):
        print("Found unified checkpoint in /runpod-volume/models (network volume)")
        return str(vol_models)

    _diagnose_paths()
    print(
        "WARNING: No unified checkpoint found in /app/models or /runpod-volume/models. "
        "Is the network volume attached to this endpoint? "
        "(Serverless → endpoint → Edit → Advanced → Network Volumes)"
    )
    return str(app_models)


def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    device = "cuda" if os.environ.get("DEVICE", "cuda") != "cpu" else "cpu"
    models_dir = _resolve_models_dir()
    print(f"Using models_dir={models_dir}")
    _check_onnx_artifacts(Path(models_dir))
    MODEL = BoxingAnalyzer(
        models_dir=models_dir,
        device=device,
        confidence=float(os.environ.get("DEFAULT_CONFIDENCE", "0.5")),
        attr_confidence=float(os.environ.get("DEFAULT_ATTR_CONFIDENCE", "0.0")),
        action_confidence=float(os.environ.get("DEFAULT_ACTION_CONFIDENCE", "0.6")),
        use_gap_grouping=os.environ.get("USE_GAP_GROUPING", "true").lower() in ("true", "1", "yes"),
        backend="onnx",
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

    # --- Optional inputs (cloud GPU — ignore device from client) --------------
    confidence = float(job_input.get("confidence", 0.5))
    attr_confidence = float(job_input.get("attr_confidence", 0.0))
    action_confidence = float(job_input.get("action_confidence", 0.6))
    save_video = job_input.get("save_video", True)
    if isinstance(save_video, str):
        save_video = save_video.lower() in ("true", "1", "yes")

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
                min_separation=3,
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
