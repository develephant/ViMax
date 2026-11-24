# Run ViMax on vast.ai

Run ViMax entirely on self-hosted GPUs: **ComfyUI** for keyframes (FLUX.2
[klein]) and video (MiniMax-H3), plus a **vLLM** instance for the LLM. No hosted
image/video APIs required.

> **Status: verified on a live instance (2026-11-24)** with an RTX PRO 6000 WS
> (96 GB) media box and an RTX A6000 (48 GB) LLM box: FLUX.2 klein 4B text-to-image
> and reference editing, MiniMax-H3 (full and turbo) image-to-video with voice,
> Qwen3-VL-32B tool calling + vision through the Portal, and a full ViMax
> Idea2Video monologue (1 scene, 2 shots) planned and rendered end to end.
> See "Measured on live hardware" below.

## Concepts in 60 seconds

- **Template**: a saved bundle of image + env + ports + disk settings. You
  create a template once, then rent instances from it.
- **Instance**: a rented GPU machine running your template's Docker image.
- **Instance Portal**: the base image's built-in web dashboard. It proxies
  internal ports to `http://<host>:<mapped-port>` and protects them with
  your instance's **OPEN_BUTTON_TOKEN** (find it on the instance card in the
  console). External port `4173` is the convention used below.
- **Billing**: while an instance is `running` you pay GPU + disk. `stopped`
  means no GPU charges but **disk is still billed**. `destroy` deletes
  everything and stops all billing. Storage charges start at creation.

## Architecture: two instances

```
+------------------------------+         +-----------------------------+
|  Instance A: "media"         |         |  Instance B: "llm"          |
|  image: vastai/comfy         |         |  image: vastai/vllm         |
|                              |  HTTPS  |                             |
|  ViMax web :14173  ----------|-------->|  vLLM OpenAI API :8000      |
|    (portal maps :4173)       |  token  |  Qwen3-VL-32B-Instruct-FP8  |
|      |  http://127.0.0.1     |         |                             |
|      v                       |         |                             |
|  ComfyUI :18188              |         |                             |
|    FLUX.2 klein 4B, H3       |         |                             |
+------------------------------+         +-----------------------------+
        | You browse: http://<media-host>:<mapped-4173>  (Portal token)
        | ViMax Settings -> LLM base_url: http://<llm-host>:<mapped-8000>/v1
```

The ViMax web UI, agent runtime, and ComfyUI all live on the media instance.
Only the LLM is remote (or use any OpenAI-compatible API you already have).

## Step 1 — media instance (ComfyUI + ViMax)

### Console (click-path)

1. Console → **Templates** → create template:
   - Image: `vastai/comfy:@vastai-automatic-tag`
   - Env / ports: `-p 4173:4173` plus the variables below
   - Disk: **200 GB**
2. Set environment variables on the template/instance:

   ```
   PROVISIONING_SCRIPT=https://raw.githubusercontent.com/develephant/ViMax/main/deploy/vastai/provision.sh
   PORTAL_CONFIG=<existing entries>|localhost:4173:14173:/:ViMax
   VIMAX_MODELS=flux2-klein-4b,minimax-h3
   HF_TOKEN=hf_...            # optional, only needed for gated repos
   ```

   `PORTAL_CONFIG` entries are `host:external_port:internal_port:path:name`
   joined by `|` — keep whatever the template already has and append
   `localhost:4173:14173:/:ViMax`. The provisioning script also tries to add
   this entry to `/etc/portal.yaml` itself, but setting it here is the
   reliable path.
3. **Create** → search offers (see GPU table) → rent.
4. Wait for `actual_status == "running"`, then give provisioning time
   (model downloads are tens of GB). Watch with `vastai logs <id>`.
5. Open the **Instance Portal** button on the instance card → the **ViMax**
   tile opens the UI.

### CLI equivalent

```bash
vastai create template \
  --name "vimax-media" \
  --image "vastai/comfy" \
  --image_tag "@vastai-automatic-tag" \
  --env '-e PROVISIONING_SCRIPT=https://raw.githubusercontent.com/develephant/ViMax/main/deploy/vastai/provision.sh -e VIMAX_MODELS=flux2-klein-4b,minimax-h3 -e PORTAL_CONFIG="localhost:1111:11111:/:Instance Portal|localhost:8188:18188:/:ComfyUI|localhost:8080:8080:/:Jupyter|localhost:4173:14173:/:ViMax" -p 1111:1111 -p 8188:8188 -p 8080:8080 -p 4173:4173' \
  --disk_space 200 \
  --ssh --direct
# note the hash_id, then:
vastai create instance <OFFER_ID> --template_hash <HASH_ID> --disk 200
```

(PORTAL_CONFIG quoting/escaping through the CLI is fiddly — the console
template editor is the less painful path. Unverified on a live instance.)

## Step 2 — LLM instance (vLLM)

Any OpenAI-compatible endpoint works. For a self-hosted one:

```bash
vastai create template \
  --name "vimax-llm" \
  --image "vastai/vllm" \
  --image_tag "@vastai-automatic-tag" \
  --env '-e VLLM_MODEL=Qwen/Qwen3-VL-32B-Instruct-FP8 -e VLLM_ARGS="--max-num-seqs 8 --max-model-len 32768 --gpu-memory-utilization 0.95 --enable-auto-tool-choice --tool-call-parser hermes --download-dir /workspace/models --host 127.0.0.1 --port 18000" -e PORTAL_CONFIG="localhost:1111:11111:/:Instance Portal|localhost:8000:18000:/docs:vLLM API|localhost:8080:18080:/:Jupyter" -p 1111:1111 -p 8000:8000 -p 8080:8080' \
  --disk_space 120 \
  --ssh --direct
```

Cheaper alternative: `VLLM_MODEL=Qwen/Qwen3-VL-8B-Instruct` on a 24–32 GB card.

Notes on `VLLM_ARGS` (all learned the hard way on a live box):

- Keep `--host 127.0.0.1 --port 18000`: the Portal (Caddy) proxies external
  port 8000 to internal 18000. Without them vLLM listens on the wrong port and
  the Portal returns 502.
- On a 48 GB A6000, Qwen3-VL-32B-FP8 takes ~34 GB for weights, leaving room
  for roughly 36k tokens of KV cache. `--max-model-len 65536` and `40960` both
  fail at startup; `32768` with `--gpu-memory-utilization 0.95` works.
- `--tool-call-parser hermes` produces native `tool_calls` for Qwen3-VL.
- To change args on a running box: edit `VLLM_ARGS` in `/etc/environment`,
  then `supervisorctl restart vllm`.

### Finding offers

```bash
vastai search offers 'gpu_name=RTX_PRO_6000 num_gpus=1 verified=true rentable=true gpu_ram>=80' -o 'dph_total+'
vastai search offers 'gpu_name=RTX_A6000 num_gpus=1 verified=true rentable=true' -o 'dph_total+'
```

### GPU / price guide (~USD/hr, 2026-11 — check live prices)

| Role | GPU | VRAM | ~$/hr |
|---|---|---|---|
| Media (recommended) | RTX PRO 6000 (Blackwell) | 96 GB | 1.00–1.55 |
| Media (fallback — NVFP4 text encoder untested on Ampere) | A100 80GB | 80 GB | 1.02–1.20 |
| LLM | RTX A6000 | 48 GB | 0.40–0.52 |
| LLM cheap (8B) | RTX 5090 / 4090 | 32/24 GB | 0.40–0.80 |

Avoid GB10 machines (ARM). MiniMax-H3 (32B NVFP4 encoder + int8 diffusion)
peaked at ~49 GB VRAM on the 96 GB card; the 4B klein model adds ~16 GB.

## Step 3 — connect ViMax Settings

In the ViMax web UI Settings (or `configs/agent.local.yaml`):

```yaml
llm:
  model_provider: openai
  model: Qwen/Qwen3-VL-32B-Instruct-FP8
  base_url: http://<llm-host>:<mapped-8000>/v1
  api_key: <LLM instance OPEN_BUTTON_TOKEN>
image:
  model: comfyui/flux2-klein-4b
  base_url: http://127.0.0.1:18188
video:
  model: comfyui/minimax-h3
  base_url: http://127.0.0.1:18188
```

`comfyui/<preset>` selects a preset from `configs/comfyui/presets.yaml`;
`base_url` is the ComfyUI root (the vastai/comfy image serves it on port
18188 internally). `api_key` is optional — if set it is sent as
`Authorization: Bearer ...`.

`<mapped-8000>` is the host port vast assigns to 8000 (instance card, or
`vastai show instance <id>` → `ports`). The Portal answers **plain HTTP** on
these mapped ports and requires the instance's `OPEN_BUTTON_TOKEN` as
`Authorization: Bearer <token>` (401 without it), which is exactly what
ViMax's OpenAI client sends as `api_key`. Traffic, including that token, is
unencrypted between the two instances; use the SSH tunnel below if that
matters to you.

The agent's context window defaults to 32768 tokens on the media box
(`VIMAX_CONTEXT_WINDOW_TOKENS`, set by the `vimax-web` supervisor script) to
match the vLLM `--max-model-len`. If you raise one, raise the other.

### SSH tunnel fallback (no portal needed)

```bash
# 4173 -> ViMax web, 8188 -> ComfyUI (optional, for direct poking)
ssh -p <ssh-port> root@<media-host> -L 4173:localhost:14173 -L 8188:localhost:18188
```

Then browse `http://127.0.0.1:4173` and set base_urls to the local forwards.

## Presets & LoRAs

Shipped presets (`configs/comfyui/presets.yaml`):

| preset | kind | models |
|---|---|---|
| `flux2-klein-4b` | image | klein 4B distilled + Qwen3-4B encoder (Apache-2.0) |
| `flux2-klein-9b` | image | klein 9B fp8 + Qwen3-8B encoder (non-commercial) |
| `minimax-h3` | video | MiniMax-H3 i2v (with audio) |
| `minimax-h3-turbo` | video | + lightx2v 8-step turbo LoRA |

To add a LoRA: drop the `.safetensors` in `ComfyUI/models/loras`, copy a
klein preset in `presets.yaml`, switch `workflow:` to
`flux2-klein-lora.json` and add `lora_name`/`lora_strength` to `values`
(see the commented example). Train LoRAs against **FLUX.2 klein 4B Base**
(`flux-2-klein-base-4b.safetensors`), e.g. with ai-toolkit.

## Tips: monologues, length, env

- **Dialogue / monologue**: put the spoken lines in the shot's *audio*
  description — MiniMax-H3 generates native stereo audio including voices.
- **Clip length**: H3 frame counts snap to `17k + 5`. Set
  `VIMAX_COMFYUI_VIDEO_SECONDS` (default 5) — e.g. 5s @24fps → 124 frames.
- **Env var reference**:
  - `VIMAX_COMFYUI_TIMEOUT_SECONDS` (default 1800) — total queue+run wait
  - `VIMAX_COMFYUI_POLL_SECONDS` (default 3) — history poll interval
  - `VIMAX_COMFYUI_VIDEO_SECONDS` (default 5) — requested clip length
  - `VIMAX_CONTEXT_WINDOW_TOKENS` (default 32768 on vast) — agent context budget; keep ≤ vLLM `--max-model-len`
  - `VIMAX_MODELS` (provision) — `flux2-klein-4b,flux2-klein-9b,minimax-h3,h3-turbo`
  - `VIMAX_REPO`, `VIMAX_REF`, `VIMAX_DIR`, `VIMAX_WEB_PORT_INTERNAL` (provision)

## Measured on live hardware (2026-11-24)

| Item | Result |
|---|---|
| Media provisioning (models + ViMax build) | ~10 min on a ~110 MB/s host; ~60 GB of models, ~80 GB disk used of 200 |
| FLUX.2 klein 4B, 1600×896 | 6.4 s first run (incl. model load), ~3 s after; ~16 GB VRAM |
| klein edit with 1 reference | 3.4 s, same face/beard/clothes carried over |
| MiniMax-H3 turbo, 5 s clip @ 1344×768 + stereo audio | ~90 s per clip |
| MiniMax-H3 full (20 steps), 5 s clip | ~207 s per clip; ~49 GB VRAM peak |
| vLLM Qwen3-VL-32B-FP8 on A6000 | ~34 GB weights, 32k context; tool call 2 s, image description 4 s |
| ViMax Idea2Video, 1 scene / 2 shots (turbo video) | ~3 min planning + ~5.5 min render; 10.4 s final MP4 with voice |

## Licences

- **MiniMax-H3**: local commercial use requires a licence from Comfy
  (see the Comfy-Org/MiniMax-H3 model card).
- **FLUX.2 klein 4B**: Apache-2.0.
- **FLUX.2 klein 9B**: non-commercial.

## Control panel

`deploy/vastai/launcher.py` is a self-contained uv script (PEP 723 — `uv run`
fetches Gradio + the vastai CLI binding itself) that manages the two-instance
stack from your local machine. It reads `VASTAI_TOKEN` from the environment
or the repo-root `.env`.

```bash
uv run deploy/vastai/launcher.py            # panel on http://127.0.0.1:7860
uv run deploy/vastai/launcher.py --dry-run  # read-only status + offer preview
```

- **Launch stack**: reuses the labeled `vimax-media` / `vimax-llm` instances
  if healthy (starting them first if they are stopped — the host may not have
  the GPU free, in which case use Retry), otherwise rents new ones from each
  role's saved template,
  waits for readiness (portal health-check on each box), then writes the LLM
  settings into ViMax (`PUT /api/config` — the llm section only).
- **Retry media / Retry LLM**: destroys that role's labeled instances and
  re-rents on a *different* host (previous hosts, plus any in `BAD_HOSTS` at
  the top of `launcher.py`, are excluded). There is no confirmation. Use when an
  instance is stuck in `exited`/`offline`/`unknown` or error loops. After
  either retry it re-wires the LLM config (a fresh media box has none).
- **Links**: after a successful run the panel shows ViMax, ComfyUI and vLLM
  URLs with `?token=` — these contain the instance access token, keep them
  private.
- **Cost warning**: launching/retrying rents real GPUs (~$1.00–1.55/hr media,
  ~$0.40–0.55/hr LLM). Stopped instances still bill disk — destroy to stop
  all charges.

## What provisioning does

`provision.sh` (runs on first boot, idempotent): installs uv + Node 22,
clones the ViMax repo, `uv sync --frozen`, `npm ci && npm run build` in
`web/`, downloads the selected models into `ComfyUI/models/…`, writes
`configs/agent.local.yaml` (mode 600, no secrets) if absent, registers a
`vimax-web` supervisor program serving the UI on `127.0.0.1:14173`, and
adds the `localhost:4173` portal entry.
