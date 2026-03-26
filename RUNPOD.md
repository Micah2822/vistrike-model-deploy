# RunPod Serverless setup (VISTRIKE inference)

Serverless = **pay only while inference runs**, scale to zero when idle, no VM babysitting.

---

## Architecture

```
Browser (Vercel)                   Vercel API routes               RunPod Serverless
───────────────                    ─────────────────               ─────────────────
1. User picks video                                                
2. Gets presigned URL  ──────────▶ /api/upload-url                 
3. Uploads video to R2/S3                                          
4. Calls "start job"   ──────────▶ /api/analyze ──────────────────▶ POST /run
                                   (passes video_url + params)      → handler.py downloads video
                                                                    → runs BoxingAnalyzer
                                                                    → uploads results to R2/S3
                                                                    → returns summary + URLs
5. Polls progress       ──────────▶ /api/progress/:jobId ─────────▶ GET /status/:jobId
6. Gets results          ──────────▶ /api/results (reads from R2)  
```

**No iframe.** The analysis UI (`index.html`, `app.js`, `style.css`) is served from **Vercel** and calls Vercel API routes. The old `testing-ui/app.py` Flask server is **replaced** by `handler.py`.

---

## What you need

| Component | What | Where |
|-----------|------|-------|
| **RunPod Serverless endpoint** | Runs `handler.py` in a Docker container with GPU | RunPod |
| **Object storage** (R2 / S3) | Holds uploaded videos + inference results | Cloudflare R2, AWS S3, etc. |
| **Vercel API routes** | Proxy between browser and RunPod API; generates presigned upload URLs; hides API keys | Vercel (other repo) |
| **Static analysis UI** | `index.html`, `app.js`, `style.css` — served from Vercel, calls `/api/*` routes | Vercel (other repo) |

---

## RunPod console: create serverless endpoint

### Step 1 — Build and push Docker image

```bash
cd vistrike-runpod-deploy

# If models are small enough to bake in, uncomment COPY models/ in Dockerfile.
# Otherwise, download at handler startup or use RunPod network volume.

docker build -t yourdockerhub/vistrike-worker:latest .
docker push yourdockerhub/vistrike-worker:latest
```

Or use **GitHub integration** (RunPod can build from a repo automatically).

### Step 2 — Create endpoint in RunPod console

Go to **[Serverless → Endpoints → New Endpoint](https://www.console.runpod.io/serverless)**.

| Setting | Value |
|---------|-------|
| **Endpoint name** | `vistrike-inference` (or whatever you want) |
| **Worker image** | `yourdockerhub/vistrike-worker:latest` (or GitHub source) |
| **GPU** | Pick based on your model size (see GPU sizing below) |
| **Min workers** | `0` (scale to zero = no cost when idle) |
| **Max workers** | `1` to start (increase for concurrency) |
| **Idle timeout** | e.g. `60s` — worker stays warm this long after a job |
| **Execution timeout** | e.g. `600000` ms (10 min) — max time per job |
| **Container disk** | 20 GB+ (needs room for temp video + output) |
| **Network volume** (optional) | Mount with your `models/` if you don't bake them in |

### Step 3 — Environment variables

Set in the endpoint **Environment** section:

| Variable | Purpose |
|----------|---------|
| `DEVICE` | `cuda` (default) or `cpu` |
| `DEFAULT_CONFIDENCE` | Detection threshold (default `0.5`) |
| `DEFAULT_ATTR_CONFIDENCE` | Attribute threshold (default `0.0`) |
| `DEFAULT_ACTION_CONFIDENCE` | Action event threshold (default `0.6`) |

### Step 4 — Note your endpoint ID and API key

- **Endpoint ID**: shown on the endpoint page (e.g. `abc123xyz`)
- **API key**: RunPod account → Settings → API Keys

You'll give both to the Vercel repo as env vars.

---

## GPU sizing

| VRAM | Cost | Fits |
|------|------|------|
| **~8 GB** | Cheapest | Tight; may OOM on high-res or large models |
| **12–16 GB** | Mid | **Good default** for most boxing clips |
| **24 GB+** | Higher | Headroom for bigger models or higher res |

Test with your **longest / highest-res clip** to find the right tier.

---

## Getting models into the worker

1. **Bake into Docker image** — Uncomment `COPY models/ /app/models/` in Dockerfile. Simple but image is huge; rebuild on weight changes.
2. **Download at startup** — Add a download step in `handler.py` before `load_model()` (e.g. from S3/R2/HF). First cold start is slower.
3. **RunPod network volume** — Attach a volume with `models/` pre-loaded; set mount path in endpoint config so `/app/models` resolves.

---

## How the handler works

`handler.py` receives a job:

```json
{
  "id": "job-abc",
  "input": {
    "video_url": "https://your-bucket/uploads/video.mp4",
    "confidence": 0.5,
    "save_video": true
  },
  "s3Config": {
    "accessId": "...",
    "accessSecret": "...",
    "bucketName": "vistrike-results",
    "endpointUrl": "https://your-r2-endpoint"
  }
}
```

1. Downloads video from `video_url`
2. Runs `BoxingAnalyzer.analyze_video()` + `compute_summary()`
3. If `s3Config` provided: uploads `summary.json`, `analysis.json`, annotated video to storage
4. Returns summary inline + result URLs
5. Sends progress updates via `runpod.serverless.progress_update()`

---

## RunPod API (what Vercel calls)

Base: `https://api.runpod.ai/v2/{ENDPOINT_ID}`  
Auth: `Authorization: Bearer {RUNPOD_API_KEY}`

| Action | Method | Path | Body |
|--------|--------|------|------|
| Start job | POST | `/run` | `{"input": {...}, "s3Config": {...}}` |
| Poll status | GET | `/status/{job_id}` | — |
| Cancel | POST | `/cancel/{job_id}` | — |
| Health | GET | `/health` | — |

**`/run`** returns `{"id": "job-abc", "status": "IN_QUEUE"}`.  
**`/status`** returns `{"status": "IN_PROGRESS", "output": {...}}` or `{"status": "COMPLETED", "output": {...}}`.

---

## Object storage (R2 / S3)

You need a bucket for:
- **Uploaded videos** (user uploads here via presigned URL from Vercel)
- **Results** (handler uploads here after inference)

Create a bucket, get access keys, and set them in:
- RunPod handler (passed as `s3Config` in the job, or as env vars)
- Vercel API routes (to generate presigned upload URLs)

---

## Testing locally

```bash
cd vistrike-runpod-deploy
python handler.py --test_input '{"input": {"video_url": "file:///path/to/test.mp4", "confidence": 0.5, "save_video": false}}'
```

---

## Files in this folder

| File | Purpose |
|------|---------|
| `handler.py` | RunPod serverless handler — replaces `testing-ui/app.py` |
| `Dockerfile` | Builds the worker image |
| `requirements.txt` | Python deps for the worker |
| `README.md` | Full deployment guide (files, architecture, Vercel prompt) |
| `RUNPOD.md` | This file — RunPod serverless config |
