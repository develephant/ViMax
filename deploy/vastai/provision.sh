#!/bin/bash
# ViMax provisioning script for the vastai/comfy base image on vast.ai.
# Runs once at first boot (PROVISIONING_SCRIPT). Idempotent: safe to re-run.
set -euo pipefail

log() { echo "[vimax-provision] $*"; }

VIMAX_REPO="${VIMAX_REPO:-https://github.com/develephant/ViMax.git}"
VIMAX_REF="${VIMAX_REF:-main}"
VIMAX_DIR="${VIMAX_DIR:-${WORKSPACE:-/workspace}/ViMax}"
VIMAX_MODELS="${VIMAX_MODELS:-flux2-klein-4b,minimax-h3}"
VIMAX_WEB_PORT_INTERNAL="${VIMAX_WEB_PORT_INTERNAL:-14173}"
export VIMAX_WEB_PORT_INTERNAL

COMFYUI_DIR="${COMFYUI_DIR:-${WORKSPACE:-/workspace}/ComfyUI}"
COMFYUI_BASE_URL="${COMFYUI_BASE_URL:-http://127.0.0.1:18188}"
VENV="/venv/main"
export PATH="${VENV}/bin:${HOME}/.local/bin:${PATH}"

MODEL_DIRS="${COMFYUI_DIR}/models"

# ---------------------------------------------------------------------------
# 1. Tooling: uv + hf CLI
# ---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    log "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi
UV_BIN="$(command -v uv)"

if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
    log "installing huggingface_hub CLI"
    "${UV_BIN}" tool install huggingface_hub || "${VENV}/bin/pip" install -U "huggingface_hub[cli]"
fi
if command -v hf >/dev/null 2>&1; then
    HF_CLI="hf"
else
    HF_CLI="huggingface-cli"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN
fi

# ---------------------------------------------------------------------------
# 2. Node.js >= 20
# ---------------------------------------------------------------------------
node_ok() {
    command -v node >/dev/null 2>&1 && [ "$(node -v | sed 's/v\([0-9]*\).*/\1/')" -ge 20 ]
}
if ! node_ok; then
    log "installing Node.js 22"
    NODE_VER="v22.14.0"
    curl -fsSL "https://nodejs.org/dist/${NODE_VER}/node-${NODE_VER}-linux-x64.tar.xz" -o /tmp/node.tar.xz
    mkdir -p /opt/node
    tar -xJf /tmp/node.tar.xz -C /opt/node --strip-components=1
    ln -sf /opt/node/bin/node /usr/local/bin/node
    ln -sf /opt/node/bin/npm /usr/local/bin/npm
    ln -sf /opt/node/bin/npx /usr/local/bin/npx
    rm -f /tmp/node.tar.xz
fi
log "node $(node -v), npm $(npm -v)"

# ---------------------------------------------------------------------------
# 3. ViMax repo + deps
# ---------------------------------------------------------------------------
if [[ -d "${VIMAX_DIR}/.git" ]]; then
    log "updating ${VIMAX_DIR} to ${VIMAX_REF}"
    git -C "${VIMAX_DIR}" fetch origin
    git -C "${VIMAX_DIR}" checkout "${VIMAX_REF}"
    git -C "${VIMAX_DIR}" pull --ff-only origin "${VIMAX_REF}" || true
else
    log "cloning ${VIMAX_REPO} (${VIMAX_REF}) -> ${VIMAX_DIR}"
    mkdir -p "$(dirname "${VIMAX_DIR}")"
    git clone --branch "${VIMAX_REF}" --depth 1 "${VIMAX_REPO}" "${VIMAX_DIR}"
fi

log "uv sync"
(cd "${VIMAX_DIR}" && "${UV_BIN}" sync --frozen)

log "web build"
(cd "${VIMAX_DIR}/web" && npm ci && npm run build)

# ---------------------------------------------------------------------------
# 4. Model downloads
# ---------------------------------------------------------------------------
# Each entry: <token>|<subdir>|<hf repo>|<hf file>
MODEL_SPECS=(
    "flux2-klein-4b|diffusion_models|Comfy-Org/Flux-2-Klein-4B|split_files/diffusion_models/flux-2-klein-4b.safetensors"
    "flux2-klein-4b|text_encoders|Comfy-Org/Flux-2-Klein-4B|split_files/text_encoders/qwen_3_4b.safetensors"
    "flux2-klein-4b|vae|Comfy-Org/Flux-2-Klein-4B|split_files/vae/flux2-vae.safetensors"
    "flux2-klein-9b|diffusion_models|black-forest-labs/FLUX.2-klein-9b-fp8|flux-2-klein-9b-fp8.safetensors"
    "flux2-klein-9b|text_encoders|Comfy-Org/Flux-2-Klein-9B|split_files/text_encoders/qwen_3_8b.safetensors"
    "flux2-klein-9b|vae|Comfy-Org/Flux-2-Klein-9B|split_files/vae/flux2-vae.safetensors"
    "minimax-h3|diffusion_models|Comfy-Org/MiniMax-H3|diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    "minimax-h3|text_encoders|Comfy-Org/MiniMax-H3|text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    "minimax-h3|vae|Comfy-Org/MiniMax-H3|vae/minimax_h3_video_vae_int8_convrot.safetensors"
    "minimax-h3|vae|Comfy-Org/MiniMax-H3|vae/minimax_h3_audio_vae_fp32.safetensors"
    "h3-turbo|loras|Comfy-Org/MiniMax-H3|loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
)

want_model() {
    local token="$1"
    local list=",${VIMAX_MODELS},"
    [[ "${list}" == *",${token},"* ]]
}

download_file() {
    local repo="$1" file="$2" destdir="$3"
    local dest
    dest="${destdir}/$(basename "${file}")"
    mkdir -p "${destdir}"
    if [[ -f "${dest}" ]]; then
        log "skip existing ${dest}"
        return 0
    fi
    local tmp
    tmp="$(mktemp -d)"
    log "downloading ${repo}/${file}"
    "${HF_CLI}" download "${repo}" "${file}" --local-dir "${tmp}"
    mv "${tmp}/${file}" "${dest}"
    rm -rf "${tmp}"
}

for spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r token subdir repo file <<< "${spec}"
    if want_model "${token}"; then
        download_file "${repo}" "${file}" "${MODEL_DIRS}/${subdir}"
    fi
done

# ---------------------------------------------------------------------------
# 5. configs/agent.local.yaml (only if absent; contains no secrets)
# ---------------------------------------------------------------------------
AGENT_CONFIG="${VIMAX_DIR}/configs/agent.local.yaml"
if [[ ! -f "${AGENT_CONFIG}" ]]; then
    log "writing ${AGENT_CONFIG}"
    umask 077
    cat > "${AGENT_CONFIG}" <<EOF
llm:
  model_provider: openai
  model: ""
  base_url: ""
  api_key: ""
image:
  model: comfyui/flux2-klein-4b
  base_url: ${COMFYUI_BASE_URL}
  api_key: ""
video:
  model: comfyui/minimax-h3
  base_url: ${COMFYUI_BASE_URL}
  api_key: ""
EOF
    chmod 600 "${AGENT_CONFIG}"
fi

# ---------------------------------------------------------------------------
# 6. Supervisor program for the ViMax web UI
# ---------------------------------------------------------------------------
cat > /opt/supervisor-scripts/vimax-web.sh <<EOF
#!/bin/bash

utils=/opt/supervisor-scripts/utils
. "\${utils}/logging.sh"
. "\${utils}/cleanup_generic.sh"
. "\${utils}/environment.sh"

# Wait for provisioning to complete (same pattern as comfyui.sh)
while [ -f "/.provisioning" ]; do
    echo "\$PROC_NAME startup paused until instance provisioning has completed (/.provisioning present)"
    sleep 5
done

export VIMAX_WEB_HOST=127.0.0.1
export VIMAX_WEB_PORT=${VIMAX_WEB_PORT_INTERNAL}
export VIMAX_UV_CMD=${UV_BIN}
# Match the vLLM --max-model-len (see deploy/vastai/README.md); override via template env.
export VIMAX_CONTEXT_WINDOW_TOKENS="\${VIMAX_CONTEXT_WINDOW_TOKENS:-32768}"

cd "${VIMAX_DIR}"
exec npm --prefix "${VIMAX_DIR}/web" run start 2>&1
EOF
chmod +x /opt/supervisor-scripts/vimax-web.sh

cat > /etc/supervisor/conf.d/vimax-web.conf <<'EOF'
[program:vimax-web]
environment=PROC_NAME="%(program_name)s"
command=/opt/supervisor-scripts/vimax-web.sh
autostart=true
autorestart=true
exitcodes=0
startsecs=10
stopasgroup=true
killasgroup=true
stopsignal=TERM
stopwaitsecs=10
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_events_enabled=true
stdout_logfile_maxbytes=0
stdout_logfile_backups=0
EOF

# ---------------------------------------------------------------------------
# 7. Portal entry: localhost:4173 -> 127.0.0.1:VIMAX_WEB_PORT_INTERNAL
# ---------------------------------------------------------------------------
# The portal config is generated from the PORTAL_CONFIG env var; ensure the
# ViMax entry is also present in /etc/portal.yaml (idempotent).
python3 - <<'PYEOF' || log "WARNING: could not update /etc/portal.yaml; set PORTAL_CONFIG to include localhost:4173:${VIMAX_WEB_PORT_INTERNAL}:/:ViMax"
import os

path = "/etc/portal.yaml"
internal = int(os.environ.get("VIMAX_WEB_PORT_INTERNAL", "14173"))

import yaml

data = {}
if os.path.exists(path):
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
apps = data.setdefault("applications", {})
if "ViMax" not in apps:
    apps["ViMax"] = {
        "hostname": "localhost",
        "external_port": 4173,
        "internal_port": internal,
        "open_path": "/",
        "name": "ViMax",
    }
    with open(path, "w", encoding="utf-8") as handle:
        yaml.dump(data, handle, default_flow_style=False, sort_keys=False)
PYEOF

if command -v supervisorctl >/dev/null 2>&1; then
    supervisorctl reread || true
    supervisorctl update || true
fi

log "done. ViMax web will listen on 127.0.0.1:${VIMAX_WEB_PORT_INTERNAL} (portal: localhost:4173)"
