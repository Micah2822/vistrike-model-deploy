"""
RunPod Serverless handler for VISTRIKE video inference.

Receives a job with a video URL + params, runs inference, uploads results
to S3-compatible storage, returns URLs.

RunPod sends jobs as: {"id": "...", "input": {...}, "s3Config": {...}}
"""

import os
import sys
import json
import time
import tempfile
import urllib.request
from pathlib import Path

import runpod

PROJECT_ROOT = Path("/app")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
os.chdir(str(PROJECT_ROOT))

from utils.batch_video_analyzer import BoxingAnalyzer, compute_summary, save_json

MODEL = None

def load_model():
    global MODEL
    if MODEL is not None:
        return MODEL
    device = "cuda" if os.environ.get("DEVICE", "cuda") != "cpu" else "cpu"
    MODEL = BoxingAnalyzer(
        models_dir=str(PROJECT_ROOT / "models"),
        device=device,
        confidence=float(os.environ.get("DEFAULT_CONFIDENCE", "0.5")),
        attr_confidence=float(os.environ.get("DEFAULT_ATTR_CONFIDENCE", "0.0")),
        action_confidence=float(os.environ.get("DEFAULT_ACTION_CONFIDENCE", "0.6")),
        backend="pytorch",
    )
    return MODEL


def download_video(url: str, dest: Path):
    print(f"Downloading video from {url}")
    urllib.request.urlretrieve(url, str(dest))
    size_mb = dest.stat().st_size / 1024 / 1024
    print(f"Downloaded {size_mb:.1f} MB")


def upload_file(filepath: Path, s3_config: dict, key: str):
    """Upload a file to S3-compatible storage. Returns the public/presigned URL."""
    import boto3
    s3 = boto3.client(
        "s3",
        endpoint_url=s3_config.get("endpointUrl"),
        aws_access_key_id=s3_config["accessId"],
        aws_secret_access_key=s3_config["accessSecret"],
    )
    bucket = s3_config["bucketName"]
    content_type = "application/json"
    if key.endswith(".mp4"):
        content_type = "video/mp4"
    s3.upload_file(
        str(filepath), bucket, key,
        ExtraArgs={"ContentType": content_type},
    )
    url = f"{s3_config['endpointUrl']}/{bucket}/{key}"
    return url


def handler(job):
    job_input = job["input"]
    s3_config = job.get("s3Config")

    video_url = job_input.get("video_url")
    if not video_url:
        return {"error": "Missing video_url in input"}

    confidence = float(job_input.get("confidence", 0.5))
    attr_confidence = float(job_input.get("attr_confidence", 0.0))
    action_confidence = float(job_input.get("action_confidence", 0.6))
    save_video = job_input.get("save_video", True)
    if isinstance(save_video, str):
        save_video = save_video.lower() in ("true", "1", "yes")

    runpod.serverless.progress_update(job, {"status": "downloading", "percent": 0})

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        video_path = tmpdir / "input_video.mp4"
        output_dir = tmpdir / "output"
        output_dir.mkdir()

        download_video(video_url, video_path)

        runpod.serverless.progress_update(job, {"status": "loading_model", "percent": 5})
        analyzer = load_model()
        analyzer.confidence = confidence
        analyzer.attr_confidence = attr_confidence
        analyzer.action_confidence = action_confidence

        runpod.serverless.progress_update(job, {"status": "analyzing", "percent": 10})

        start = time.time()
        results = analyzer.analyze_video(str(video_path))
        elapsed = time.time() - start

        runpod.serverless.progress_update(job, {"status": "computing_summary", "percent": 80})

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

        video_url_out = None
        if save_video:
            runpod.serverless.progress_update(job, {"status": "creating_video", "percent": 85})
            annotated_path = output_dir / f"{video_path.stem}_annotated.mp4"
            analyzer.create_annotated_video(str(video_path), results, str(annotated_path))

        runpod.serverless.progress_update(job, {"status": "uploading_results", "percent": 95})

        job_id = job["id"]

        if s3_config:
            prefix = f"results/{job_id}"
            summary_url = upload_file(summary_path, s3_config, f"{prefix}/summary.json")
            analysis_url = upload_file(analysis_path, s3_config, f"{prefix}/analysis.json")
            if save_video and annotated_path.exists():
                video_url_out = upload_file(annotated_path, s3_config, f"{prefix}/annotated.mp4")

            return {
                "status": "completed",
                "elapsed_seconds": round(elapsed, 1),
                "total_frames": results.get("total_frames", 0),
                "summary_url": summary_url,
                "analysis_url": analysis_url,
                "video_url": video_url_out,
                "summary": summary,
            }
        else:
            return {
                "status": "completed",
                "elapsed_seconds": round(elapsed, 1),
                "total_frames": results.get("total_frames", 0),
                "summary": summary,
            }


load_model()
runpod.serverless.start({"handler": handler})
