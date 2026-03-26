# VISTRIKE RunPod Deployment Guide

This folder contains everything needed to deploy the VISTRIKE inference backend on RunPod (or any GPU cloud). The React frontend (on Vercel) loads this backend inside an iframe via `VITE_BACKEND_URL`.

## Architecture

```
┌──────────────────────────────┐       iframe src        ┌──────────────────────────────────┐
│  VERCEL (marketing repo)     │  ───────────────────▶   │  RUNPOD (GPU cloud)              │
│                              │                         │                                  │
│  React app:                  │                         │  testing-ui/                     │
│   - Home, About, Privacy     │   VITE_BACKEND_URL      │    app.py  (Flask, port 5001)    │
│   - Upload.jsx  ─────────────│──  points here ──────▶  │    static/ (index.html, app.js,  │
│   - Upload.css               │                         │            style.css)             │
│                              │                         │                                  │
│  NO models, NO scripts,      │                         │  scripts/                        │
│  NO testing-ui, NO Python    │                         │    10_inference.py, utils/ …     │
│                              │                         │  configs/                        │
│                              │                         │                                  │
│                              │                         │  models/                         │
│                              │                         │    unified/best.pt (etc.)        │
│                              │                         │                                  │
│                              │                         │  results/ (auto-created)         │
│                              │                         │  uploads/ (auto-created)         │
└──────────────────────────────┘                         └──────────────────────────────────┘
```

## How it works

1. User visits `/upload` on the Vercel site
2. `Upload.jsx` calls `GET {VITE_BACKEND_URL}/api/status` to check if backend is alive
3. If alive, it renders `<iframe src={VITE_BACKEND_URL}>` which loads the full analysis UI from Flask
4. The analysis UI (`static/index.html` + `static/app.js`) handles video upload, progress polling, and results display — all served by the Flask backend
5. When a user uploads a video, `app.py` runs `scripts/10_inference.py` via `subprocess.Popen` using the **Device** they picked in the UI (`cpu`, `cuda`, or `auto`)
6. Results (annotated video, `summary.json`, `analysis.json`) are written to `results/` and served via `/api/results/`

## Directory layout on RunPod

```
/workspace/vistrike/
├── testing-ui/
│   ├── app.py              # Flask server (serves UI + API)
│   └── static/
│       ├── index.html      # Analysis UI shell
│       ├── app.js          # Analysis UI logic (upload, polling, playback, dashboard)
│       └── style.css       # Analysis UI styles
├── scripts/
│   ├── 10_inference.py     # CLI wrapper (imports utils)
│   ├── inference_onnx.py   # ONNX entry (imports utils)
│   └── utils/              # REQUIRED — pipeline implementation (batch_video_analyzer, ORT backend, …)
│       ├── __init__.py
│       ├── batch_video_analyzer.py
│       ├── ort_video_backend.py
│       ├── onnx_model_metadata.py
│       └── onnx_export_wrappers.py
├── configs/
│   └── action_types.yaml   # Action types, colors, event keys (repo root relative to scripts/)
├── models/
│   └── unified/
│       └── best.pt         # Trained model weights
├── results/                # Auto-created; inference outputs go here
└── uploads/                # Auto-created inside testing-ui/; cleaned after processing
```

## Files to copy from the main repo

| Destination on RunPod | Source in VISTRIKE-AI-Official |
|---|---|
| `testing-ui/app.py` | `Vistrike-Main-UI/testing-ui/app.py` |
| `testing-ui/static/index.html` | `Vistrike-Main-UI/testing-ui/static/index.html` |
| `testing-ui/static/app.js` | `Vistrike-Main-UI/testing-ui/static/app.js` |
| `testing-ui/static/style.css` | `Vistrike-Main-UI/testing-ui/static/style.css` |
| `scripts/10_inference.py` | `scripts/10_inference.py` |
| `scripts/inference_onnx.py` | `scripts/inference_onnx.py` |
| `scripts/utils/` (entire package) | `scripts/utils/*.py` — **required**; `10_inference.py` does `from utils.batch_video_analyzer import …` with `cwd=PROJECT_ROOT`, so Python resolves `utils` as `scripts/utils` |
| `configs/action_types.yaml` | `configs/action_types.yaml` — used by `batch_video_analyzer` (has in-code fallback if missing, but you want the real file in production) |
| `models/unified/best.pt` | Your trained weights — too large for GitHub; get them onto the pod using the next section |

*(Optional: `data/attributes/` only for some old checkpoints — skip unless you know you need it.)*

## Getting models from your machine onto RunPod

1. **SCP over SSH** — RunPod shows **SSH** details on the pod page (host, port, key). SSH in once and run `mkdir -p /workspace/vistrike/models/unified` if needed. From your laptop:

   ```bash
   scp -i /path/to/key -P PORT ./best.pt root@POD_HOST:/workspace/vistrike/models/unified/best.pt
   ```

   Replace host, port, key path, and destination with your layout. Use `scp -r ./models root@POD_HOST:/workspace/vistrike/` to copy a whole `models` folder.

2. **Upload to cloud, then download on the pod** — Put `best.pt` in S3, R2, etc. (or use a presigned URL). On the pod:

   ```bash
   mkdir -p /workspace/vistrike/models/unified
   curl -L -o /workspace/vistrike/models/unified/best.pt "https://YOUR_DOWNLOAD_URL"
   ```

3. **RunPod Network Volume** — Attach a volume to the pod, copy weights onto it once (via SCP from your machine to the pod path that mounts the volume), so new pods can reuse the same volume without re-uploading.

## Required changes to `app.py`

### 1. Fix `PROJECT_ROOT`

The original line assumes the monorepo layout (`VISTRIKE-AI-Official/Vistrike-Main-UI/testing-ui/app.py` → 3 parents up). On RunPod, if the layout is `/workspace/vistrike/testing-ui/app.py`, change it to 1 parent up:

```python
# BEFORE (monorepo layout)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# AFTER (RunPod layout: /workspace/vistrike/testing-ui/app.py)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
```

This makes `PROJECT_ROOT` = `/workspace/vistrike/`, so it correctly finds `scripts/`, `models/`, and `results/`.

### 2. Lock down CORS for production

```python
# BEFORE
CORS(app)

# AFTER — only allow your Vercel frontend
CORS(app, origins=["https://your-site.vercel.app"])
```

### 3. Disable debug mode

```python
# BEFORE
app.run(host='0.0.0.0', port=port, debug=True)

# AFTER
app.run(host='0.0.0.0', port=port, debug=False)
```

## API endpoints (what the frontend expects)

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the analysis UI (`static/index.html`) — loaded inside the iframe |
| `GET` | `/api/status` | Health check. Returns `{ status, inference_script, has_unified_model, ... }` |
| `POST` | `/api/analyze` | Upload video + start inference. Multipart form: `video`, `confidence`, `attr_confidence`, `device`, `save_video`, `backend` |
| `GET` | `/api/progress/<output_dir>` | Poll inference progress (JSON: `status`, `current_frame`, `total_frames`, etc.) |
| `GET` | `/api/results/<path:filename>` | Serve result files (`summary.json`, `analysis.json`, annotated video) |

## Python dependencies

On the pod:

```bash
pip install flask flask-cors werkzeug opencv-python-headless
```

Also install everything `scripts/10_inference.py` needs from the main repo’s `requirements.txt`.

**GPU pods:** install a **CUDA-enabled** PyTorch build (CPU-only wheels will never use the GPU). Use the command from [pytorch.org](https://pytorch.org) that matches your CUDA version (check with `nvidia-smi` on the pod). Example shape:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

(Pick the `cu1xx` URL that matches what `nvidia-smi` reports.)

## RunPod pod setup (GPU, ports, checks)

**1. Pick a GPU template** — Create the pod from a template that includes an **NVIDIA** GPU and a recent CUDA driver. CPU-only pods cannot run CUDA inference.

**2. Confirm the GPU is visible**

```bash
nvidia-smi
```

You should see your GPU and driver version. If this fails, fix the template / image before debugging Python.

**3. Confirm PyTorch sees CUDA**

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

Expect `True` and a CUDA version string. If `False`, reinstall `torch` with a CUDA wheel (see above).

**4. Web UI device = CUDA** — In `static/index.html`, the **Device** dropdown defaults to **CPU**. Users must choose **CUDA (GPU)** before **Analyze** (or use **Auto** if that resolves to CUDA on your stack). For a pod-only deployment you can change the default `<option>` to `cuda` in your fork so you do not rely on users switching it.

**5. Expose the Flask port** — In RunPod, map the container port your app listens on (e.g. `5001`, or whatever you set `PORT` to) to the public **HTTP** port / proxy URL you put in `VITE_BACKEND_URL`.

**6. Disk** — Leave enough volume space for `models/`, uploaded videos, and `results/` (annotated videos are large).

**7. Timeouts** — Inference can run many minutes; use RunPod / proxy settings that do not cut off long requests or idle streaming for huge uploads.

## Running

```bash
cd /workspace/vistrike/testing-ui
python3 app.py
```

Flask starts on port 5001 by default. Override with the `PORT` env var:

```bash
PORT=8080 python3 app.py
```

## Vercel side setup

On the Vercel repo, the **only** thing needed:

1. `Upload.jsx` + `Upload.css` exist in `src/pages/`
2. Route wired in `App.jsx`: `<Route path="/upload" element={<Upload />} />`
3. Nav link in `Header.jsx`: `<Link to="/upload">`
4. **Environment variable** in Vercel dashboard:

```
VITE_BACKEND_URL = https://your-runpod-public-url.com
```

Redeploy after setting this. No other code changes needed — `Upload.jsx` already reads `VITE_BACKEND_URL`.

## Prompt to give Claude on the Vercel repo

After copying `Upload.jsx` and `Upload.css` into the other repo, paste this to Claude:

```
I've already copied these files into this repo:
- src/pages/Upload.jsx (iframe wrapper that loads VITE_BACKEND_URL)
- src/pages/Upload.css

I need you to wire them up:

1. In src/App.jsx: import Upload from './pages/Upload' and add
   <Route path="/upload" element={<Upload />} />

2. In the Header/nav component: add a Link to="/upload" labeled
   "Upload" (use class "nav-link nav-link-primary" if we have that
   pattern, otherwise match existing nav link style).

3. If Home.jsx has any CTA buttons, add a "Upload Footage" button
   that links to /upload matching the existing button style.

4. Upload.jsx reads VITE_BACKEND_URL (already in the file). I will
   set this env var in Vercel to point at our RunPod backend. No
   code change needed for that.

5. The Upload page shows a "Backend Not Running" warning when it
   can't reach the backend -- update the warning message text to say
   something like "The analysis server is currently offline. Please
   try again later." instead of the current dev instructions about
   running a local Flask server. Remove the <ol> with terminal
   instructions since end users won't be running anything locally.

That's all. The Upload page is just an iframe -- all analysis UI,
video upload, progress tracking, and results display are served by
the remote backend inside that iframe. No testing-ui/, no scripts/,
no models/ in this repo.
```

## Production checklist

- [ ] RunPod **GPU** template (not CPU-only); `nvidia-smi` works on the pod
- [ ] PyTorch **CUDA** build installed; `torch.cuda.is_available()` is `True`
- [ ] Analysis UI uses **CUDA (GPU)** or **Auto** (not left on default **CPU** unless intentional)
- [ ] Enough disk for models + `results/`
- [ ] `PROJECT_ROOT` in `app.py` points to the correct root (where `scripts/` and `models/` live)
- [ ] CORS origin set to your Vercel domain
- [ ] `debug=False` in Flask
- [ ] RunPod port is exposed with HTTPS
- [ ] Upload size limit passes through any proxy (500MB)
- [ ] Request timeout is long enough for inference (can take several minutes)
- [ ] `GET /api/status` returns 200 so Upload page shows "connected"
- [ ] `VITE_BACKEND_URL` set in Vercel dashboard and site redeployed
- [ ] Model weights present at `models/unified/best.pt`
- [ ] Inference scripts present at `scripts/10_inference.py` and `scripts/inference_onnx.py`
