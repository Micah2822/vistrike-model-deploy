# VISTRIKE Serverless Inference (RunPod + Vercel)

**No always-on VM.** RunPod Serverless scales to zero — you pay only while inference runs.

RunPod console config, GPU sizing, env vars, endpoint settings: **[RUNPOD.md](./RUNPOD.md)**

---

## Architecture

```
┌─────────────────────────────┐         ┌──────────────────────┐         ┌──────────────────────┐
│  VERCEL                     │         │  OBJECT STORAGE      │         │  RUNPOD SERVERLESS   │
│                             │         │  (R2 / S3)           │         │                      │
│  React app + analysis UI    │         │                      │         │  handler.py          │
│  Vercel API routes:         │         │  uploads/video.mp4   │         │  scripts/            │
│    /api/upload-url ─────────│────▶    │  results/job-id/     │    ◀────│  models/             │
│    /api/analyze ────────────│─────────│──────────────────────│────▶    │  configs/            │
│    /api/progress/:id ───────│─────────│──────────────────────│────▶    │                      │
│    /api/results ────────────│────▶    │                      │         │                      │
└─────────────────────────────┘         └──────────────────────┘         └──────────────────────┘
```

**Flow:**
1. User picks a video in the browser (Vercel)
2. Vercel API route generates a **presigned upload URL** → browser uploads video to R2/S3
3. Vercel API route calls **RunPod `/run`** with `video_url` + params + `s3Config`
4. RunPod worker (`handler.py`) downloads video, runs inference, uploads results to R2/S3, returns summary
5. Browser polls **Vercel `/api/progress/:jobId`** → Vercel polls **RunPod `/status/:jobId`**
6. When done, browser fetches results from R2/S3

**No iframe. No Flask. No always-on server.**

---

## What's in this folder

| File | What it does |
|------|-------------|
| **`handler.py`** | RunPod serverless worker — downloads video, runs `BoxingAnalyzer`, uploads results to storage, returns summary + URLs |
| **`Dockerfile`** | Builds the Docker image for the worker (PyTorch + CUDA + scripts) |
| **`requirements.txt`** | Python deps for the worker image |
| **`RUNPOD.md`** | RunPod console setup — endpoint creation, GPU sizing, env vars, model loading, API reference |

---

## What goes where

### RunPod (this folder → Docker image)

| In the image | Source |
|--------------|--------|
| `handler.py` | This folder |
| `scripts/10_inference.py` | `scripts/10_inference.py` |
| `scripts/inference_onnx.py` | `scripts/inference_onnx.py` |
| `scripts/utils/` (full folder) | `scripts/utils/` |
| `configs/action_types.yaml` | `configs/action_types.yaml` |
| `models/unified/best.pt` | Your weights — bake in, download at startup, or use network volume ([RUNPOD.md](./RUNPOD.md)) |

### Vercel (the other repo)

| What | Purpose |
|------|---------|
| React app (Home, About, etc.) | Main site |
| `Upload.jsx` + `Upload.css` | Upload page — **rewritten: no iframe**, calls Vercel API routes |
| `testing-ui/static/*` (`index.html`, `app.js`, `style.css`) | Analysis UI — served from Vercel as static files, `API_BASE` points to Vercel API routes |
| **Vercel API routes** (new) | `/api/upload-url`, `/api/analyze`, `/api/progress/:id`, `/api/results` — proxy to RunPod + R2 |

### Object storage (R2 / S3)

One bucket with two prefixes:
- `uploads/` — user videos (presigned PUT from browser)
- `results/{job_id}/` — `summary.json`, `analysis.json`, `annotated.mp4` (written by handler)

---

## Build and deploy the worker

```bash
cd vistrike-runpod-deploy
docker build -t yourdockerhub/vistrike-worker:latest .
docker push yourdockerhub/vistrike-worker:latest
```

Then create a serverless endpoint in RunPod console pointing at this image. Full steps in **[RUNPOD.md](./RUNPOD.md)**.

---

## Vercel env vars (set in dashboard)

| Variable | Value |
|----------|-------|
| `RUNPOD_ENDPOINT_ID` | Your RunPod serverless endpoint ID |
| `RUNPOD_API_KEY` | Your RunPod API key |
| `R2_ENDPOINT_URL` | e.g. `https://xxxx.r2.cloudflarestorage.com` |
| `R2_ACCESS_KEY_ID` | R2 access key |
| `R2_SECRET_ACCESS_KEY` | R2 secret |
| `R2_BUCKET_NAME` | e.g. `vistrike-data` |

---

## Prompt to give Claude on the Vercel repo

```
We're switching from an iframe + Flask backend to RunPod Serverless.
All static files from testing-ui/ are already copied into this repo.
Upload.jsx and Upload.css are already in src/pages/.

Here's the new architecture:
- NO iframe. NO Flask server. NO VITE_BACKEND_URL.
- The analysis UI (testing-ui/static/index.html, app.js, style.css) 
  is served from THIS repo as static assets under public/analysis/ 
  (or similar).
- Vercel API routes proxy between the browser and RunPod Serverless + R2.
- Object storage (Cloudflare R2) holds uploaded videos and inference results.

I need you to implement:

1. VERCEL API ROUTES (src/api/ or app/api/ depending on framework):

   a) POST /api/upload-url
      - Generates a presigned PUT URL for R2 so the browser can upload 
        the video directly to storage.
      - Input: { filename, contentType }
      - Output: { uploadUrl, videoKey }
      - Uses env vars: R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, 
        R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME

   b) POST /api/analyze
      - Calls RunPod serverless: POST https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/run
      - Auth: Authorization: Bearer {RUNPOD_API_KEY}
      - Body: {
          "input": {
            "video_url": "<R2 public/presigned URL for the uploaded video>",
            "confidence": <from request>,
            "attr_confidence": <from request>,
            "save_video": <from request>
          },
          "s3Config": {
            "accessId": R2_ACCESS_KEY_ID,
            "accessSecret": R2_SECRET_ACCESS_KEY,
            "bucketName": R2_BUCKET_NAME,
            "endpointUrl": R2_ENDPOINT_URL
          }
        }
      - Output: { jobId } (from RunPod response.id)
      - Uses env vars: RUNPOD_ENDPOINT_ID, RUNPOD_API_KEY, R2_*

   c) GET /api/progress/:jobId
      - Calls RunPod: GET https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/{jobId}
      - Auth: Authorization: Bearer {RUNPOD_API_KEY}
      - Returns the RunPod status response to the browser.
        status will be IN_QUEUE, IN_PROGRESS, COMPLETED, or FAILED.
        When IN_PROGRESS, output may contain progress_update data
        (percent, status message).
        When COMPLETED, output contains summary + result URLs.

   d) GET /api/results/:key
      - Fetches a file from R2 and streams it to the browser.
      - Used for analysis.json, summary.json, annotated video.
      - Or: generate a presigned GET URL and redirect.

2. UPDATE testing-ui/static/app.js (now at public/analysis/app.js 
   or wherever you placed it):
   - Change API_BASE from '' to '' (or wherever Vercel API routes live 
     — should be same origin so '' works).
   - Replace the analyzeVideo() function flow:
     OLD: FormData upload to /api/analyze, poll /api/progress, 
          fetch /api/results.
     NEW:
       a) Call /api/upload-url to get presigned URL
       b) PUT the video file directly to that URL (show upload progress)
       c) Call /api/analyze with the video key + params → get jobId
       d) Poll /api/progress/:jobId until COMPLETED
       e) Read results from the COMPLETED response output (summary 
          inline, result URLs for video/analysis.json)
   - The progress response shape changes:
     OLD: { status, current_frame, total_frames, ... }
     NEW: RunPod status: { status: "IN_PROGRESS", 
          output: { status: "analyzing", percent: 50 } }
     Map appropriately in the polling UI.
   - Results: summary comes inline in COMPLETED output. Annotated 
     video URL comes from output.video_url. analysis.json from 
     output.analysis_url.
   - Remove checkServerStatus() or adapt it to call /api/analyze 
     health endpoint or just show "ready".

3. UPDATE Upload.jsx:
   - Remove the iframe entirely.
   - Instead, either:
     a) Embed the analysis page directly (import the HTML/JS), or
     b) Route to a page that loads the static analysis UI 
        (simplest: put index.html content in a React component, 
        or load public/analysis/index.html in the page).
   - Remove VITE_BACKEND_URL references.
   - Remove the "backend not running" warning (there's no backend 
     to check — RunPod scales on demand).

4. ENV VARS needed in Vercel dashboard:
   RUNPOD_ENDPOINT_ID, RUNPOD_API_KEY,
   R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME

The RunPod serverless handler (handler.py) is already deployed 
separately — you don't need to touch it. It expects:
- input.video_url (URL to download the video from)
- input.confidence, input.attr_confidence, input.save_video
- s3Config for uploading results
- Returns: { status, summary, summary_url, analysis_url, video_url }
- Progress updates during processing (percent + status string)
```

---

## Checklist

- [ ] Docker image built and pushed
- [ ] RunPod serverless endpoint created with correct GPU, env vars, image
- [ ] R2/S3 bucket created with access keys
- [ ] Models in the image (baked, downloaded, or on volume)
- [ ] Vercel API routes implemented (upload-url, analyze, progress, results)
- [ ] `app.js` updated for new flow (presigned upload → RunPod job → poll → results)
- [ ] `Upload.jsx` updated (no iframe)
- [ ] Vercel env vars set: `RUNPOD_ENDPOINT_ID`, `RUNPOD_API_KEY`, `R2_*`
- [ ] Test end-to-end: upload video → inference → view results
