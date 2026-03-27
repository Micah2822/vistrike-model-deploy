"""
RunPod Serverless handler for VISTRIKE video inference (ONNX Runtime).

Receives a job with a video URL + params, runs ONNX inference via
inference_onnx pipeline, uploads results to Supabase Storage, returns
summary + artifact URLs.

Environment variables (set in RunPod dashboard → endpoint → secrets):
  SUPABASE_URL              – e.g. https://xxxx.supabase.co
  SUPABASE_SERVICE_ROLE_KEY – service-role key (NOT the anon key)
  SUPABASE_BUCKET           – storage bucket name, e.g. "vistrike-results"
  DEVICE                    – "cuda" (default) or "cpu"
  DEFAULT_CONFIDENCE        – detection threshold (default 0.5)
  DEFAULT_ATTR_CONFIDENCE   – attribute threshold (default 0.0)
  DEFAULT_ACTION_CONFIDENCE – action event threshold (default 0.6)
  MODELS_DIR                – optional override for weights directory. If unset,
                            uses /app/models when it contains a unified checkpoint,
                            else /runpod-volume/models (RunPod network volume).
"""

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
            data=open(filepath, "rb"),
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


def _progress(job, message: str, percent: int, status: str = "processing"):
    """Send a progress update the Vercel proxy can forward to the browser."""
    runpod.serverless.progress_update(
        job, {"message": message, "percent": percent, "status": status}
    )


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def handler(job):
    job_input = job["input"]

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

    _progress(job, "Downloading video…", 0, "downloading")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        video_path = tmpdir / "input_video.mp4"
        output_dir = tmpdir / "output"
        output_dir.mkdir()

        download_video(video_url, video_path)

        _progress(job, "Loading model…", 5, "loading_model")
        analyzer = load_model()
        analyzer.confidence = confidence
        analyzer.attr_confidence = attr_confidence
        analyzer.action_confidence = action_confidence

        _progress(job, "Analyzing video…", 10, "analyzing")

        start = time.time()
        results = analyzer.analyze_video(str(video_path))
        elapsed = time.time() - start

        _progress(job, "Computing summary…", 80, "computing_summary")

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
            _progress(job, "Creating annotated video…", 85, "creating_video")
            annotated_path = output_dir / f"{video_path.stem}_annotated.mp4"
            analyzer.create_annotated_video(
                str(video_path), results, str(annotated_path)
            )

        # --- Upload to Supabase ------------------------------------------------
        job_id = job["id"]
        summary_url = None
        analysis_url = None
        video_url_out = None

        if _supabase_configured():
            _progress(job, "Uploading results…", 92, "uploading_results")
            prefix = f"results/{job_id}"
            summary_url = supabase_upload(summary_path, f"{prefix}/summary.json")
            analysis_url = supabase_upload(analysis_path, f"{prefix}/analysis.json")
            if save_video and annotated_path and annotated_path.exists():
                _progress(job, "Uploading annotated video…", 96, "uploading_video")
                video_url_out = supabase_upload(
                    annotated_path, f"{prefix}/annotated.mp4"
                )
        else:
            print("WARNING: Supabase not configured — skipping upload")

        _progress(job, "Done", 100, "completed")

        return {
            "status": "completed",
            "elapsed_seconds": round(elapsed, 1),
            "total_frames": results.get("total_frames", 0),
            "summary": summary,
            "summary_url": summary_url,
            "analysis_url": analysis_url,
            "video_url": video_url_out,
        }


# ---------------------------------------------------------------------------
# Entrypoint — load model at container start (warm worker), then serve jobs
# ---------------------------------------------------------------------------
load_model()
runpod.serverless.start({"handler": handler})
