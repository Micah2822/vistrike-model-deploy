# VISTRIKE — RunPod Serverless GPU Worker

Self-contained Docker image that runs boxing-video inference on RunPod
Serverless. Scales to zero — you pay only while a job runs.

Results (summary JSON, frame-by-frame analysis JSON, optional annotated MP4)
are uploaded to **Supabase Storage** so the Vercel front-end can display
charts and dashboards.

---

## Prerequisites

| Requirement | Why |
|-------------|-----|
| **Docker** (Desktop or CLI) | Build + push the worker image |
| **Docker Hub / GHCR account** | Host the image so RunPod can pull it |
| **RunPod account** | Create the serverless endpoint |
| **Supabase project** | Storage bucket for inference artifacts |
| **Model weights** in `models/` | Baked into the image, downloaded at startup, or on a RunPod network volume |

> If your model weights live in a monorepo, copy or symlink `models/unified/best.pt`
> (and any action sub-models) into this folder before building.

### Supabase (one-time, before RunPod)

1. **Create a project** at [supabase.com](https://supabase.com/dashboard) (free tier is fine).
2. **Storage → New bucket** — name it something memorable (e.g. `vistrike-results`). That exact name is what you set as `SUPABASE_BUCKET` on RunPod.
3. **Public vs private** — The worker returns **public** object URLs (`/storage/v1/object/public/...`). For the dashboard to load JSON and MP4 in the browser without extra auth, turn **Public bucket** on for that bucket (Storage → bucket → **Public**). If you keep the bucket private, you must serve files via your own API with signed URLs; the URLs returned by the worker alone will not work in `<video>` / `fetch` without auth.
4. **Project URL + service role** — **Project Settings → API**: copy **Project URL** → `SUPABASE_URL` (looks like `https://<project-ref>.supabase.co`). Copy **service_role** `secret` → `SUPABASE_SERVICE_ROLE_KEY`. Use **service_role** only on RunPod (server-side); never put it in the browser or Vercel client bundles. The **anon** key is for the front-end if you use Supabase client-side for uploads.
5. **Layout** — One bucket is enough: your upload flow writes user clips under `uploads/…`; the GPU worker writes `results/<job_id>/summary.json`, `analysis.json`, and optional `annotated.mp4` (see path table below).

---

## Folder tree

```
vistrike-model-deploy/
├── handler.py              ← RunPod serverless entry point
├── Dockerfile
├── requirements.txt
├── configs/
│   └── action_types.yaml   ← action-type definitions (punch, defense, …)
├── scripts/
│   ├── 10_inference.py
│   ├── 11_live_analysis.py
│   ├── inference_onnx.py
│   ├── validate_onnx_parity.py
│   └── utils/
│       ├── __init__.py
│       ├── batch_video_analyzer.py
│       ├── onnx_export_wrappers.py
│       ├── onnx_model_metadata.py
│       └── ort_video_backend.py
├── models/                 ← gitignored; see "Getting models in" below
│   ├── unified/best.pt
│   └── actions/…
├── README.md               ← this file
└── RUNPOD.md               ← RunPod console setup, env vars, API ref
```

### What is gitignored vs. copied at build time

| Path | In git? | In Docker image? | Notes |
|------|---------|-------------------|-------|
| `handler.py` | Yes | Yes | Entrypoint |
| `scripts/` | Yes | Yes | Inference pipeline |
| `configs/` | Yes | Yes | Action type YAML |
| `models/` | **No** (gitignored) | Optional — see below | Too large for git |
| `testing-ui/` | **No** (deleted) | **No** | Obsolete Flask UI; not part of worker |
| `__pycache__/`, `*.pyc` | No | No | Excluded by `.dockerignore` |

---

## Quick start (numbered)

### 1. Clone / copy this folder

```bash
git clone <your-remote> vistrike-model-deploy
cd vistrike-model-deploy
```

If scripts or configs come from a monorepo, sync them:

```bash
cp -r /path/to/monorepo/scripts ./scripts
cp -r /path/to/monorepo/configs ./configs
```

### 2. Place model weights

```bash
mkdir -p models/unified
cp /path/to/best.pt models/unified/best.pt
# Optional action sub-models:
mkdir -p models/actions/punch models/actions/defense
cp /path/to/punch_model/* models/actions/punch/
cp /path/to/defense_model/* models/actions/defense/
```

Or uncomment `COPY models/ /app/models/` in the Dockerfile to bake them in.

### 3. Build the Docker image

```bash
docker build -t yourdockerhub/vistrike-worker:latest .
```

### 4. (Optional) Test locally

```bash
# Requires GPU + models present in models/
docker run --rm --gpus all \
  -e DEVICE=cuda \
  -v $(pwd)/models:/app/models \
  yourdockerhub/vistrike-worker:latest \
  python -u /app/handler.py --test_input '{
    "input": {
      "video_url": "https://example.com/test.mp4",
      "confidence": 0.5,
      "save_video": false
    }
  }'
```

### 5. Push to registry

```bash
docker push yourdockerhub/vistrike-worker:latest
```

### 6. Create RunPod serverless endpoint

See **[RUNPOD.md](./RUNPOD.md)** for step-by-step console instructions,
every environment variable, and a test curl command.

### 7. Tag a release (optional)

```bash
docker tag yourdockerhub/vistrike-worker:latest \
           yourdockerhub/vistrike-worker:v1.0.0
docker push yourdockerhub/vistrike-worker:v1.0.0
```

---

## Getting models into the worker

| Strategy | Pros | Cons |
|----------|------|------|
| **Bake into image** (`COPY models/`) | Simple, reproducible | Image is huge; rebuild on weight changes |
| **Download at startup** (add download step before `load_model()`) | Small image | First cold start is slower |
| **RunPod network volume** | Shared across workers, fast attach | Requires volume setup in RunPod console |

---

## Job input / output reference

### Input (`job["input"]`)

```json
{
  "video_url": "https://<project-ref>.supabase.co/storage/v1/object/public/<bucket>/uploads/video.mp4",
  "confidence": 0.5,
  "attr_confidence": 0.0,
  "action_confidence": 0.6,
  "save_video": true
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `video_url` | string | **(required)** | HTTPS URL to the source video |
| `confidence` | float | `0.5` | Detection confidence threshold |
| `attr_confidence` | float | `0.0` | Attribute confidence threshold |
| `action_confidence` | float | `0.6` | Action event confidence threshold |
| `save_video` | bool | `true` | Generate annotated MP4 |

> `device` is ignored — the worker always uses the GPU it was launched on.

### Output (returned in RunPod `COMPLETED` response `.output`)

```json
{
  "status": "completed",
  "elapsed_seconds": 42.3,
  "total_frames": 1800,
  "summary": { "…inline summary dict…" },
  "summary_url": "https://xxxx.supabase.co/storage/v1/object/public/vistrike-results/results/job-abc/summary.json",
  "analysis_url": "https://xxxx.supabase.co/storage/v1/object/public/vistrike-results/results/job-abc/analysis.json",
  "video_url": "https://xxxx.supabase.co/storage/v1/object/public/vistrike-results/results/job-abc/annotated.mp4"
}
```

- `summary` is always present (inline dict).
- `summary_url`, `analysis_url`, `video_url` are HTTPS URLs if Supabase is
  configured; `null` otherwise.
- `video_url` is `null` when `save_video` is `false`.

### Supabase Storage path convention

```
<bucket>/
├── uploads/                    ← user videos (presigned PUT from browser)
│   └── <filename>.mp4
└── results/
    └── <job_id>/
        ├── summary.json
        ├── analysis.json
        └── annotated.mp4       ← only when save_video=true
```

### Progress updates (during processing)

The handler calls `runpod.serverless.progress_update()` at each stage.
The Vercel proxy can read these from the RunPod `/status/{jobId}` response
and forward to the browser progress UI.

```json
{ "message": "Analyzing video…", "percent": 10, "status": "analyzing" }
```

| `status` value | `percent` | Meaning |
|----------------|-----------|---------|
| `downloading` | 0 | Fetching source video |
| `loading_model` | 5 | Loading PyTorch weights |
| `analyzing` | 10 | Running frame-by-frame inference |
| `computing_summary` | 80 | Aggregating events into summary |
| `creating_video` | 85 | Rendering annotated MP4 |
| `uploading_results` | 92 | Pushing JSON to Supabase |
| `uploading_video` | 96 | Pushing annotated MP4 to Supabase |
| `completed` | 100 | Job finished |
