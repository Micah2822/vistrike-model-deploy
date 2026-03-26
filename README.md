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
│                              │                         │  configs/, data/attributes/      │
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
5. When a user uploads a video, `app.py` runs `scripts/10_inference.py` via `subprocess.Popen` on the GPU
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
├── data/
│   └── attributes/         # label_map.json per attribute (cwd = PROJECT_ROOT when Flask runs inference)
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
| `data/attributes/` | `data/attributes/` — label maps for attribute heads; relative to **process cwd** (`PROJECT_ROOT`). If missing, you get a warning and some attribute labeling may degrade |
| `models/unified/best.pt` | Your trained model weights |

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

Install these on the RunPod instance:

```bash
pip install flask flask-cors werkzeug opencv-python-headless
```

Plus whatever `scripts/10_inference.py` needs (typically `torch`, `ultralytics`, `numpy`, etc.). Check the main repo's `requirements.txt` for the full list.

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

- [ ] RunPod instance has GPU access and enough disk for models + video results
- [ ] `PROJECT_ROOT` in `app.py` points to the correct root (where `scripts/` and `models/` live)
- [ ] CORS origin set to your Vercel domain
- [ ] `debug=False` in Flask
- [ ] RunPod port is exposed with HTTPS
- [ ] Upload size limit passes through any proxy (500MB)
- [ ] Request timeout is long enough for inference (can take several minutes)
- [ ] `GET /api/status` returns 200 so Upload page shows "connected"
- [ ] `VITE_BACKEND_URL` set in Vercel dashboard and site redeployed
- [ ] Model weights present at `models/unified/best.pt` (or `unified_mps/best.pt`)
- [ ] Inference scripts present at `scripts/10_inference.py` and `scripts/inference_onnx.py`
