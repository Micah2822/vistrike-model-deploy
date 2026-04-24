# RunPod Serverless Setup — VISTRIKE Inference Worker

Step-by-step guide to deploy the VISTRIKE GPU worker on
[RunPod Serverless](https://docs.runpod.io/serverless/overview).

---

## 1. Build and push the Docker image

```bash
cd vistrike-model-deploy

# (Optional) bake models into the image:
#   uncomment "COPY models/ /app/models/" in Dockerfile

docker build -t yourdockerhub/vistrike-worker:latest .
docker push yourdockerhub/vistrike-worker:latest
```

Or use **RunPod GitHub integration** to build directly from a repo.

---

## 2. Create a serverless endpoint (current RunPod wizard)

Open **Serverless** in the console and start **Deploy a New Endpoint** (or **Create a new deployment**). You should see a multi-step flow.

Official field reference: [Endpoint settings](https://docs.runpod.io/serverless/endpoints/endpoint-configurations).

### Screen A — Choose how to deploy

Pick **Custom deployment** (not "Start from a template" or "Run code locally").

Then choose **one** of:

| Path | When to use |
|------|-------------|
| **Deploy from Github** | RunPod builds the image from your repo. Repo must contain the `Dockerfile` at the path you set. Models usually **not** in Git — use a network volume (see **Getting models into the worker**). |
| **Deploy from docker registry** | You already ran `docker build` + `docker push`. Paste the full image name (e.g. Docker Hub `YOUR_USER/vistrike-worker:latest`). |

**If GitHub:** set **Branch** (e.g. `main`), **Dockerfile Path** (often `Dockerfile` or `./Dockerfile`), **Build Context** (usually repo root: `/` or blank). Wait until the UI shows the Dockerfile was found.

**"Could not find runpod.serverless.start()…"** — RunPod scans your **default GitHub branch** for this call. If you see this warning: make sure `handler.py` (which contains `runpod.serverless.start` at the bottom) is committed and pushed to that branch. Wait a few minutes for GitHub indexing. You can continue if the scanner is wrong and your branch is correct.

### Screen B — Configure Endpoint

| Field | What to do |
|-------|------------|
| **Endpoint name** | e.g. `vistrike-inference` (any label you like). |
| **Endpoint type** | **Queue-based.** This project uses a Python handler + `runpod.serverless.start()` and the async **`/run`** API. Do **not** pick load-balancing. |
| **Worker type** | Use RunPod's default or **Enhanced** depending on budget. |
| **GPU configuration** | Pick a GPU with enough VRAM — see **GPU sizing** below (~16 GB recommended). |
| **Model** (Hugging Face link / model name) | **Leave empty.** VISTRIKE loads weights from **`/app/models`** or **`/runpod-volume/models`** (see **Getting models into the worker**), not from Hugging Face. |
| **Container start command** | **Leave blank** so the image uses its Dockerfile **CMD** (`python -u /app/handler.py`). |
| **Container disk** | **20** GB or more (video download + annotated output). |
| **Expose HTTP ports** / **Expose TCP ports** | **Leave empty** for this serverless handler. |
| **Environment variables** | Add the vars from **Step 3** below. Use **Secrets** (lock icon) for `SUPABASE_SERVICE_ROLE_KEY`. |

### Screen C — Deploy

Confirm **Deploy Serverless Endpoint**. Pricing shows at the bottom.

**Scaling (Active workers, idle timeout, execution timeout):** These may **not** appear on the initial deploy screen. After the endpoint exists, open it → **Edit** / **Settings** and look for:

| Setting | Recommended |
|---------|-------------|
| **Active workers** (old name: "min workers") | `0` — scale to zero, no idle GPU cost |
| **Max workers** | `1` (increase for concurrency) |
| **Idle timeout** | `60` seconds |
| **Execution timeout** | `600` seconds (10 min) |

**Network volume:** If weights are not baked into the image, attach a volume under **Advanced → Network Volumes** (see **Getting models into the worker**).

---

## 3. Set environment variables / secrets

In the endpoint **Environment** section, add every variable below.

### Supabase (artifact upload — required for dashboard parity)

**Set up Storage first (Supabase Dashboard):**

1. **Storage → New bucket** — pick a name (e.g. `vistrike-results`). That string must match `SUPABASE_BUCKET` below.
2. **Public bucket** — If you want the worker’s returned `*_url` values to work in the browser (charts, `<video>`, `fetch`) without your proxy adding auth, enable **Public** on that bucket. Private buckets need signed URLs or a server-side proxy; the handler currently emits **public** URLs only.
3. **Project Settings → API** — Copy **Project URL** and the **service_role** key (long JWT). The **anon** key is not sufficient for server uploads from RunPod; use **service_role** in endpoint secrets only.
4. **Prefixes** — No extra dashboard step: create folders implicitly by path. Front-end uploads go to `uploads/…`; the worker writes under `results/<job_id>/…` in the same bucket.

| Variable | Purpose | Example |
|----------|---------|---------|
| `SUPABASE_URL` | Your Supabase project URL | `https://abcdefghij.supabase.co` |
| `SUPABASE_SERVICE_ROLE_KEY` | Service-role key (NOT the anon key) — grants write access to Storage | `eyJhbGciOiJIUz…` (long JWT) |
| `SUPABASE_BUCKET` | Storage bucket name (must match the bucket you created) | `vistrike-results` |

### Inference defaults

All inference knobs (device, thresholds, gap grouping, side-assignment, `models_dir`) live in **[`configs/inference.yaml`](./configs/inference.yaml)**, which is baked into the image via `COPY configs/`. Edit the file and rebuild the image to change defaults. Per-job values in `job["input"]` override the YAML for these keys only: `confidence`, `attr_confidence`, `action_confidence`, `save_video`, `min_separation`. All other YAML keys are config-only — changing them requires a new image or a mounted replacement file plus a worker restart.

Key fields in `configs/inference.yaml`:

| YAML key | Purpose | Default |
|----------|---------|---------|
| `device` | `cuda` or `cpu` | `cuda` |
| `models_dir` | Fixed weights directory. Empty/null uses auto-discovery (`/app/models` → `/runpod-volume/models`). | `""` |
| `confidence` | Detection threshold (per-job override allowed) | `0.5` |
| `attr_confidence` | Attribute threshold (per-job override allowed) | `0.0` |
| `action_confidence` | Action event threshold (per-job override allowed) | `0.6` |
| `use_gap_grouping` | Gap-based event grouping for non-defense actions. More accurate than peak detection; set `false` to revert. | `true` |
| `min_separation` | Minimum frames between events from the same fighter (per-job override allowed) | `3` |
| `assign_single_fighter`, `side_confidence_min`, `stable_side_frames` | Side-based fighter assignment | see YAML |
| `save_video` | Whether to render and upload an annotated video by default (per-job override allowed) | `true` |

Optional env var:

| Variable | Purpose | Default |
|----------|---------|---------|
| `INFERENCE_CONFIG_PATH` | Absolute path to the inference YAML. Use this to point at a mounted file instead of the baked-in copy. | `/app/configs/inference.yaml` |

---

## 4. Note your endpoint ID and API key

- **Endpoint ID** — shown on the endpoint page (e.g. `abc123xyz`).
- **API key** — RunPod → Account → Settings → API Keys.

Give both to the Vercel front-end repo as `RUNPOD_ENDPOINT_ID` and
`RUNPOD_API_KEY`.

---

## 5. Submit a test job

```bash
ENDPOINT_ID="your-endpoint-id"
RUNPOD_API_KEY="your-api-key"

curl -s -X POST \
  "https://api.runpod.ai/v2/${ENDPOINT_ID}/run" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "video_url": "https://<project-ref>.supabase.co/storage/v1/object/public/<bucket>/uploads/test.mp4",
      "confidence": 0.5,
      "attr_confidence": 0.0,
      "action_confidence": 0.6,
      "save_video": true
    }
  }'
```

Response:

```json
{ "id": "job-abc123", "status": "IN_QUEUE" }
```

### Poll for status

```bash
JOB_ID="job-abc123"

curl -s \
  "https://api.runpod.ai/v2/${ENDPOINT_ID}/status/${JOB_ID}" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}"
```

While running:

```json
{
  "id": "job-abc123",
  "status": "IN_PROGRESS",
  "output": { "message": "Analyzing video…", "percent": 45, "status": "analyzing" }
}
```

When finished:

```json
{
  "id": "job-abc123",
  "status": "COMPLETED",
  "output": {
    "status": "completed",
    "elapsed_seconds": 42.3,
    "total_frames": 1800,
    "summary": { "…" },
    "summary_url": "https://xxxx.supabase.co/storage/v1/object/public/vistrike-results/results/job-abc123/summary.json",
    "analysis_url": "https://xxxx.supabase.co/storage/v1/object/public/vistrike-results/results/job-abc123/analysis.json",
    "video_url": "https://xxxx.supabase.co/storage/v1/object/public/vistrike-results/results/job-abc123/annotated.mp4"
  }
}
```

---

## 6. RunPod API reference (what Vercel calls)

Base URL: `https://api.runpod.ai/v2/{ENDPOINT_ID}`
Auth header: `Authorization: Bearer {RUNPOD_API_KEY}`

| Action | Method | Path | Body |
|--------|--------|------|------|
| Submit job | `POST` | `/run` | `{"input": {…}}` |
| Submit + wait (sync, ≤30 s) | `POST` | `/runsync` | `{"input": {…}}` |
| Poll status | `GET` | `/status/{job_id}` | — |
| Cancel job | `POST` | `/cancel/{job_id}` | — |
| Health check | `GET` | `/health` | — |

---

## GPU sizing

| VRAM | Tier | Notes |
|------|------|-------|
| ~8 GB | Budget | Tight — may OOM on high-res or multiple action models |
| **12–16 GB** | **Recommended** | Good default for standard boxing clips |
| 24 GB+ | Premium | Headroom for bigger models or 1080p+ input |

Test with your **longest / highest-resolution clip** to find the right tier.

---

## Getting models into the worker

The handler picks a weights directory automatically: the `models_dir` value in **[`configs/inference.yaml`](./configs/inference.yaml)** if set; otherwise **`/app/models`** when it contains **`unified/best.pt`** (or **`last.pt`** / **`unified_mps/…`**); otherwise **`/runpod-volume/models`** when that path has a unified checkpoint (RunPod attaches network volumes at **`/runpod-volume`**). Upload with S3 using prefix **`models/…`** so files land under **`/runpod-volume/models`**.

| Strategy | How |
|----------|-----|
| **Bake into image** | Uncomment `COPY models/ /app/models/` in `Dockerfile`. Rebuild on weight changes. |
| **Network volume** | Create volume → upload **`models/unified/best.pt`** via [S3 API](https://docs.runpod.io/storage/s3-api) (`aws s3 sync ./models s3://VOLUME_ID/models …`) → attach volume on the serverless endpoint (**Advanced → Network Volumes**). Leave `models_dir` empty in the YAML and ensure the image has no conflicting weights under **`/app/models`**. |
| **Download at startup** | Add a download step before `load_model()` into **`/app/models`** or set `models_dir` in the YAML to an explicit path. Not implemented in-repo. |
| **`models_dir` override** | Set `models_dir` in `configs/inference.yaml` if you store weights somewhere else entirely. |

### Network volume: upload from your machine

Paths like **`/app`** and **`/runpod-volume`** are **inside the Docker container on RunPod**, not names of your GitHub repo or local folder. Your repo can be called anything; on disk you still use a **`models/`** folder with **`unified/best.pt`** (and optional **`models/actions/…`**) when uploading.

**1. Create a network volume**

- Open **[RunPod → Storage](https://www.console.runpod.io/user/storage)** → **New Network Volume**.
- Choose **size**, **name**, and **data center**. The data center must support the **S3-compatible API** (see [RunPod S3 API — datacenters](https://docs.runpod.io/storage/s3-api#datacenter-availability)); each region has its own endpoint URL (e.g. `EU-RO-1` → `https://s3api-eu-ro-1.runpod.io/`).
- After creation, copy the volume **ID** — this string is the S3 **bucket** name for the CLI.

**2. Create an S3 API key (separate from your normal RunPod API key)**

- **Settings** → **S3 API Keys** → **Create**.
- Save the **access key** (e.g. `user_…`) and **secret** (e.g. `rps_…`). The secret is shown **once**.

**3. Install AWS CLI and configure credentials**

Install [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html), then:

```bash
aws configure
```

- **AWS Access Key ID:** your RunPod S3 access key (`user_…`).
- **AWS Secret Access Key:** your RunPod S3 secret (`rps_…`).
- **Default region:** your volume’s **data center ID** exactly (e.g. `EU-RO-1`). It must match the volume and the S3 endpoint host.
- **Output format:** `json` or leave blank.

**4. Prepare a local `models` tree**

From your laptop, you need at least:

```text
models/unified/best.pt
```

Optional: `models/unified/last.pt`, `models/unified_mps/…`, `models/actions/<type>/best.pt`, etc. — same layout as local inference.

**5. Upload with `aws s3 sync`**

Run this from the directory that **contains** the `models` folder (not inside `models`). Replace:

- `YOUR_VOLUME_ID` — volume ID from step 1.
- `EU-RO-1` — your data center ID.
- `https://s3api-eu-ro-1.runpod.io/` — the endpoint for **your** data center from the [S3 API table](https://docs.runpod.io/storage/s3-api#datacenter-availability).

```bash
aws s3 sync ./models "s3://YOUR_VOLUME_ID/models" \
  --region EU-RO-1 \
  --endpoint-url "https://s3api-eu-ro-1.runpod.io/"

# should be:

aws s3 sync ./models "s3://28td86o173/models" \
  --region EU-RO-1 \
  --endpoint-url "https://s3api-eu-ro-1.runpod.io/"
```

This writes keys under `models/…` on the volume. On a serverless worker that appears as **`/runpod-volume/models/…`**. The handler will auto-select that path when **`/app/models`** has no unified checkpoint (see intro above).

**6. Verify**

```bash
aws s3 ls "s3://YOUR_VOLUME_ID/models" \
  --region EU-RO-1 \
  --endpoint-url "https://s3api-eu-ro-1.runpod.io/"
```

**7. Large uploads / flaky transfers**

```bash
export AWS_RETRY_MODE=standard
export AWS_MAX_ATTEMPTS=10
```

Then re-run `sync`. For very large files RunPod also documents multipart / timeout tweaks in the [S3 API guide](https://docs.runpod.io/storage/s3-api).

**8. Attach the volume to your serverless endpoint**

- **Serverless** → your endpoint → **Manage** → **Edit Endpoint**.
- **Advanced** → **Network Volumes** → select this volume → **Save**.

Attaching ties workers to that volume’s **data center**, which can limit GPU availability vs “any region.”

**9. Redeploy / save**

Ensure the worker image does **not** ship a stale **`unified/best.pt`** under **`/app/models`** if you want the volume weights to win (keep `COPY models/` commented out in `Dockerfile` for image-only code).

---

## Troubleshooting

### Cold start takes 30–60+ seconds

Normal. The PyTorch base image is large. The first job after scale-to-zero
must pull the image, start the container, and run `load_model()`.
Set **Idle timeout** higher (e.g. 120 s) to keep the worker warm between jobs.

### OOM (Out of Memory) — worker killed

- Move to a GPU with more VRAM.
- Reduce input resolution before upload.
- Set `save_video: false` to skip the annotated-video render (saves ~1–2 GB).

### Job times out (FAILED / TIMEOUT)

- Increase **Execution timeout** in endpoint settings (default 600 000 ms = 10 min).
- Check that the source `video_url` is publicly accessible from RunPod's network.

### Download timeout / SSL error

- Ensure the `video_url` is a direct-download HTTPS link (not a web page).
- Supabase public bucket URLs work without auth headers.
- If using a presigned URL, ensure it has not expired.

### Supabase upload fails (403 / 401)

- Verify `SUPABASE_SERVICE_ROLE_KEY` is the **service-role** key from **Project Settings → API**, not the anon key.
- Verify `SUPABASE_BUCKET` matches the bucket name exactly (case-sensitive).
- If uploads succeed but browser cannot load `summary_url` / `video_url`, the bucket may be private — either mark the bucket **Public** (Storage → bucket settings) or serve those objects through your API with [signed URLs](https://supabase.com/docs/guides/storage/serving/downloads#create-signed-urls).
- Rare: project paused on free tier — resume in Supabase Dashboard.

### "Models not found" on startup

- Check worker logs for **`Using models_dir=…`**.
- **Baked in:** `models/` on build machine and `COPY models/ /app/models/` uncommented.
- **Volume:** S3 sync uses prefix **`s3://VOLUME_ID/models`** so **`unified/best.pt`** exists at **`/runpod-volume/models/unified/best.pt`**, and **`/app/models`** must not contain a stale **`unified/best.pt`** (otherwise baked path wins). Leave `models_dir` empty in `configs/inference.yaml` unless overriding.
- Required: **`unified/best.pt`** or **`unified/last.pt`** (or **`unified_mps/…`**).

---

## Full job input / output JSON

### Input

```json
{
  "input": {
    "video_url": "https://<project-ref>.supabase.co/storage/v1/object/public/<bucket>/uploads/sparring_clip.mp4",
    "confidence": 0.5,
    "attr_confidence": 0.0,
    "action_confidence": 0.6,
    "save_video": true
  }
}
```

| Field | Type | Default | Required |
|-------|------|---------|----------|
| `video_url` | string | — | **Yes** |
| `confidence` | float | `0.5` | No |
| `attr_confidence` | float | `0.0` | No |
| `action_confidence` | float | `0.6` | No |
| `save_video` | bool | `true` | No |

### Output (inside `COMPLETED` → `output`)

```json
{
  "status": "completed",
  "elapsed_seconds": 42.3,
  "total_frames": 1800,
  "summary": {
    "total_punches": 127,
    "by_type": { "jab": 45, "cross": 38, "hook": 44 },
    "by_result": { "landed": 72, "blocked": 31, "missed": 24 },
    "red_corner": { "punches_thrown": 68, "punches_landed": 41 },
    "blue_corner": { "punches_thrown": 59, "punches_landed": 31 }
  },
  "summary_url": "https://xxxx.supabase.co/storage/v1/object/public/vistrike-results/results/job-abc/summary.json",
  "analysis_url": "https://xxxx.supabase.co/storage/v1/object/public/vistrike-results/results/job-abc/analysis.json",
  "video_url": "https://xxxx.supabase.co/storage/v1/object/public/vistrike-results/results/job-abc/annotated.mp4"
}
```

### Supabase Storage paths

```
<SUPABASE_BUCKET>/
├── uploads/                          ← user videos (presigned PUT from Vercel)
│   └── sparring_clip.mp4
└── results/
    └── <job_id>/
        ├── summary.json              ← high-level stats
        ├── analysis.json             ← per-frame detections + events
        └── annotated.mp4             ← rendered video (when save_video=true)
```

---

## Links

- [RunPod Serverless docs](https://docs.runpod.io/serverless/overview)
- [RunPod API reference](https://docs.runpod.io/serverless/endpoints/job-operations)
- [Supabase Storage docs](https://supabase.com/docs/guides/storage)
