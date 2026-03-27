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

## 2. Create a serverless endpoint

Go to **[Serverless → Endpoints → + New Endpoint](https://www.runpod.io/console/serverless)**.

| Setting | Recommended value |
|---------|-------------------|
| **Endpoint name** | `vistrike-inference` |
| **Worker image** | `yourdockerhub/vistrike-worker:latest` |
| **GPU type** | 16 GB VRAM (e.g. A4000, A5000) — see GPU sizing below |
| **Min workers** | `0` (scale to zero = no cost when idle) |
| **Max workers** | `1` (increase for concurrency) |
| **Idle timeout** | `60` seconds |
| **Execution timeout** | `600000` ms (10 min) |
| **Container disk** | `20` GB+ (temp video + output) |
| **Network volume** | (optional) mount at `/app/models` if models are not baked in |

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

| Variable | Purpose | Default |
|----------|---------|---------|
| `DEVICE` | `cuda` or `cpu` | `cuda` |
| `DEFAULT_CONFIDENCE` | Detection threshold (overridden per-job if client sends `confidence`) | `0.5` |
| `DEFAULT_ATTR_CONFIDENCE` | Attribute threshold | `0.0` |
| `DEFAULT_ACTION_CONFIDENCE` | Action event threshold | `0.6` |

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

| Strategy | How |
|----------|-----|
| **Bake into image** | Uncomment `COPY models/ /app/models/` in `Dockerfile`. Rebuild on weight changes. |
| **Download at startup** | Add a download step in `handler.py` before `load_model()`. Slower first cold start. |
| **Network volume** | Create a RunPod volume, upload weights, mount at `/app/models` in endpoint config. |

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

- If baking in: make sure `models/` is present before `docker build` and
  `COPY models/` is uncommented in the Dockerfile.
- If using a volume: verify the mount path resolves to `/app/models`.
- Required minimum: `models/unified/best.pt`.

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
