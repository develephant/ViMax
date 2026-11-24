# Run ViMax on vast.ai

Run ViMax entirely on self-hosted GPUs: **ComfyUI** for keyframes (FLUX.2
[klein]) and video (MiniMax-H3), plus a **vLLM** instance for the LLM. No hosted
image/video APIs required.

> **Status — verified on a live instance: pending.**
> The pieces below that have NOT yet been verified on a running vast.ai
> instance: the shipped ComfyUI workflow JSONs (converted by hand from the
> official Comfy-Org UI templates — always verify on a live ComfyUI before
> relying on them), VRAM/disk sizing numbers, whether the vast.ai direct-port
> TLS for the vLLM endpoint is a trusted cert or self-signed, the
> `--tool-call-parser hermes` flag for Qwen3-VL, and MiniMax-H3 turbo LoRA
> step count.

## Concepts in 60 seconds

- **Template**: a saved bundle of image + env + ports + disk settings. You
  create a template once, then rent instances from it.
- **Instance**: a rented GPU machine running your template's Docker image.
- **Instance Portal**: the base image's built-in web dashboard. It proxies
  internal ports to `https://<host>:<external-port>` and protects them with
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
        | You browse: https://<media-host>:4173  (OPEN_BUTTON_TOKEN)
        | ViMax Settings -> LLM base_url: https://<llm-host>:<port>/v1
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
  --env '-e VLLM_MODEL=Qwen/Qwen3-VL-32B-Instruct-FP8 -e VLLM_ARGS="--max-model-len 65536 --enable-auto-tool-choice --tool-call-parser hermes" -p 8000:8000' \
  --disk_space 120 \
  --ssh --direct
```

Cheaper alternative: `VLLM_MODEL=Qwen/Qwen3-VL-8B-Instruct` on a 24–32 GB card.

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
comfortably wants ~80 GB; the 4B klein image model is much lighter.

## Step 3 — connect ViMax Settings

In the ViMax web UI Settings (or `configs/agent.local.yaml`):

```yaml
llm:
  model_provider: openai
  model: Qwen/Qwen3-VL-32B-Instruct-FP8
  base_url: https://<llm-host>:<mapped-8000>/v1
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

The vLLM endpoint behind the portal may serve a self-signed certificate —
TLS verification handling is **pending verification**. If your client stack
rejects the cert, use the SSH tunnel below instead.

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
  - `VIMAX_MODELS` (provision) — `flux2-klein-4b,flux2-klein-9b,minimax-h3,h3-turbo`
  - `VIMAX_REPO`, `VIMAX_REF`, `VIMAX_DIR`, `VIMAX_WEB_PORT_INTERNAL` (provision)

## Licences

- **MiniMax-H3**: local commercial use requires a licence from Comfy
  (see the Comfy-Org/MiniMax-H3 model card).
- **FLUX.2 klein 4B**: Apache-2.0.
- **FLUX.2 klein 9B**: non-commercial.

## What provisioning does

`provision.sh` (runs on first boot, idempotent): installs uv + Node 22,
clones the ViMax repo, `uv sync --frozen`, `npm ci && npm run build` in
`web/`, downloads the selected models into `ComfyUI/models/…`, writes
`configs/agent.local.yaml` (mode 600, no secrets) if absent, registers a
`vimax-web` supervisor program serving the UI on `127.0.0.1:14173`, and
adds the `localhost:4173` portal entry.
