# RunPod configuration guide (VISTRIKE backend)

Companion to **[README.md](./README.md)** (files, `app.py`, Vercel). RunPod docs: [Pods overview](https://docs.runpod.io/pods/overview), [Templates](https://docs.runpod.io/pods/templates/overview), [Expose ports](https://docs.runpod.io/pods/configuration/expose-ports).

---

## 1. Pod vs Serverless — what to pick

| | **GPU Pod** | **Serverless (endpoint / worker)** |
|---|-------------|-------------------------------------|
| **What it is** | A long-running VM with a GPU; you SSH in, install deps, run `python app.py`. | Managed workers that run **handler code** per request/job; scales up/down; you ship a Docker image + handler API. |
| **Fits VISTRIKE today?** | **Yes.** Flask stays up, writes to `results/`, runs `subprocess` inference, serves static UI + REST. | **Not without a rewrite.** You’d replace the Flask process model with a worker that accepts jobs (e.g. upload to object storage → process → return), no always-on disk session as-is. |
| **Pros** | Full control, Jupyter/SSH, same as local dev, long jobs OK after upload. | Pay per active inference second, autoscale, no babysitting a VM. |
| **Cons** | You pay while the pod runs (even idle). You manage updates, security, CORS. | Cold starts, different architecture, not drop-in for current `testing-ui/app.py`. |
| **Recommendation** | **Use a Pod** for this repo’s backend until you intentionally redesign for serverless. | Use later if you build a dedicated inference worker + queue/storage. |

**CPU-only Pod:** Cheapest for smoke tests. Install CPU `torch`, use **Device → CPU** in the UI. Too slow for real users.

---

## 2. Template vs custom image vs “GitHub”

| Source | What you do | Pros | Cons |
|--------|-------------|------|------|
| **Community / official template** (e.g. PyTorch + CUDA) | Pick template in console → deploy Pod. | Fastest path; drivers + CUDA usually correct. | Less reproducible unless you snapshot; you still `pip install` Flask + your deps after start. |
| **My template** (saved in console) | Save a configured Pod as your own template. | One-click same disk size, ports, image, env next time. | You maintain it when RunPod or CUDA updates. |
| **Custom container** (Docker Hub / GHCR / ECR) | Point Pod at **your** image that already has Python, CUDA, Flask, etc. | Reproducible, CI can build it. | You build and push images; larger upfront work. |
| **Hub / repo “deploy as Pod”** | Some RunPod Hub flows deploy a repo as a Pod ([docs](https://docs.runpod.io/hub/overview#deploy-as-a-pod)). | Good if the repo is meant to run as a Pod. | Must match their layout; verify it’s not serverless-only. |

**Practical default for VISTRIKE:** start from a **PyTorch + NVIDIA CUDA** community template, then follow **README.md** to lay out `PROJECT_ROOT`, `pip install`, and run Flask. Move to a **custom Dockerfile** when you want repeatable deploys without manual steps.

---

## 3. GPU memory (VRAM) — rough guide

Exact need depends on your checkpoint size, resolution, and batch settings. Use this as a budget guide:

| VRAM (per GPU) | Cost | Pros | Cons |
|----------------|------|------|------|
| **~8 GB** (e.g. RTX 3060 class) | Lower | Cheap experiments. | Higher **OOM** risk on long 1080p+ clips or heavy unified + action stack; may need smaller input or CPU offload tricks. |
| **12–16 GB** | Mid | **Good default** for many single-video boxing runs if models are not huge. | Still watch resolution and concurrent jobs (you should run **one** analysis at a time per pod unless you scale). |
| **24 GB+** | Higher | Headroom for larger models, higher res, fewer surprises. | $$$; often overkill if 12–16 GB works in testing. |

**How to decide:** run a **max-length / max-resolution** clip locally on a machine with known VRAM; match or exceed that on RunPod. If you see CUDA OOM in logs, step up VRAM or reduce load (resolution, batch, or model).

---

## 4. Container / Pod fields (what to set in the console)

| Setting | What to choose |
|---------|----------------|
| **GPU type** | NVIDIA with enough VRAM (see §3). |
| **Cloud** | **Secure Cloud** — more stable / enterprise-ish. **Community Cloud** — often cheaper; IPs/ports can change more on restart. |
| **Container disk** | Enough for: OS + conda/pip + **PyTorch** + **all model weights** + **uploads** + **results** (annotated MP4s are large). **50 GB+** is a sane starting point if models are multi-GB; increase if you keep many jobs. |
| **Network volume** (optional) | Attach if `models/` (or full `PROJECT_ROOT`) must **survive pod delete**. Mount path depends on template — confirm where `/workspace` (or equivalent) points. |
| **Expose HTTP ports** | Internal port Flask uses (**`5001`** unless you change `PORT`). Public URL: `https://<POD_ID>-<PORT>.proxy.runpod.net` → **`VITE_BACKEND_URL`**. |
| **Expose TCP ports** | e.g. **22** for SSH/SCP to copy weights. Use **Connect** in UI for mapped host:port. |
| **Start command** (if asked) | Often empty if you start manually over SSH; or a script that `cd testing-ui && python3 app.py` after deps exist. |

**Binding:** Flask must listen on **`0.0.0.0`**, not only `127.0.0.1` (already true in `app.py`).

**HTTP proxy caveat:** Cloudflare ~**100 s** per long request. **`POST /api/analyze`** stays open until the **upload** finishes — huge/slow uploads can **524**. Progress/results **GET**s are short and usually fine.

---

## 5. Environment variables

Set in the Pod / template **Environment** section where the console allows it.

| Variable | Purpose |
|----------|---------|
| **`PORT`** | Port Flask listens on; must match **Expose HTTP ports** (default in code: `5001`). |
| **`CUDA_VISIBLE_DEVICES`** | e.g. `0` to pin a single GPU if multiple are visible. |
| **`PYTHONUNBUFFERED=1`** | Cleaner live logs (optional). |
| **`RUNPOD_TCP_PORT_*`** | RunPod may inject mapped ports for advanced TCP setups ([symmetrical ports](https://docs.runpod.io/pods/configuration/expose-ports)); only if you use that pattern. |
| **Secrets for bootstrap** | e.g. **`HF_TOKEN`** (Hugging Face), **`AWS_*`** — only if your **startup script** downloads models; not required if you SCP weights in. |

App-specific: **`VITE_BACKEND_URL`** is set on **Vercel**, not on RunPod.

---

## 6. After the pod starts

```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```

In the analysis UI: **Device → CUDA (GPU)** (the HTML default is **CPU**).

---

## 7. Links

- [Expose ports / proxy URL / timeouts](https://docs.runpod.io/pods/configuration/expose-ports)  
- [Pod templates](https://docs.runpod.io/pods/templates/overview)  
- [Pod pricing](https://docs.runpod.io/pods/pricing)  
- [Serverless](https://docs.runpod.io/serverless/overview) — read before choosing it over a Pod for production.
