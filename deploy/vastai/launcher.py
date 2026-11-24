# /// script
# requires-python = ">=3.12"
# dependencies = ["gradio>=6.32.0", "vastai>=1.11.0", "requests>=2.34.2"]
# ///
"""Local control panel for the ViMax vast.ai stack (media + LLM instances).

Run:  uv run deploy/vastai/launcher.py          # web UI on 127.0.0.1
      uv run deploy/vastai/launcher.py --dry-run # read-only status check
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

ROLES = {
    "media": {
        "label": "vimax-media",
        "template": "vimax-media",
        "disk": 200,
        "query": (
            "gpu_name in [RTX_PRO_6000_WS,RTX_PRO_6000_S,RTX_PRO_6000_Max-Q] "
            "num_gpus=1 verified=true rentable=true direct_port_count>=8 "
            "disk_space>=200 reliability>0.98 inet_down>=500 geolocation=US dph_total<=2.0"
        ),
        "ready_port": "4173/tcp",
        "ready_path": "/api/health",
    },
    "llm": {
        "label": "vimax-llm",
        "template": "vimax-llm",
        "disk": 120,
        "query": (
            "gpu_name=RTX_A6000 num_gpus=1 verified=true rentable=true "
            "direct_port_count>=8 disk_space>=120 reliability>0.98 inet_down>=500 "
            "geolocation=US dph_total<=0.8"
        ),
        "ready_port": "8000/tcp",
        "ready_path": "/v1/models",
    },
}

FAILED_STATES = {"exited", "offline", "unknown"}
# Hosts that failed for us before (e.g. 367287: Docker daemon EOF on container start).
BAD_HOSTS = {367287}
POLL_SECONDS = 20
READY_TIMEOUT = 40 * 60

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def load_api_key() -> str:
    key = os.environ.get("VASTAI_TOKEN", "").strip()
    if key:
        return key
    if ENV_PATH.is_file():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            name, value = line.split("=", 1)
            if name.strip() == "VASTAI_TOKEN":
                return value.strip().strip('"').strip("'")
    return ""


def parse_cli_json(stdout: str):
    match = re.search(r"[\{\[]", stdout)
    if not match:
        raise RuntimeError(f"vastai CLI returned no JSON: {stdout[:300]}")
    return json.JSONDecoder().raw_decode(stdout[match.start():])[0]


def scrub(text: str, key: str) -> str:
    return text.replace(key, "<key>") if key else text


def vast(key: str, *args):
    cli = shutil.which("vastai")
    if not cli:
        raise RuntimeError("vastai CLI not on PATH")
    proc = subprocess.run(
        [cli, *args, "--raw", "--api-key", key],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"vastai {' '.join(args)} failed: {scrub(proc.stderr or proc.stdout, key)[:500]}")
    return parse_cli_json(proc.stdout)


def user_id(key: str) -> int:
    return vast(key, "show", "user")["id"]


def template_hash(key: str, name: str) -> str:
    entries = [t for t in vast(key, "search", "templates", f"name={name}")
               if t.get("creator_id") == user_id(key)]
    if not entries:
        raise RuntimeError(f"no template named {name!r} owned by this account")
    return max(entries, key=lambda t: t.get("created_at") or 0)["hash_id"]


def labeled_instances(key: str, label: str) -> list[dict]:
    return [i for i in vast(key, "show", "instances") if i.get("label") == label]


def host_port(inst: dict, port: str) -> int | None:
    try:
        return int(inst["ports"][port][0]["HostPort"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def probe(inst: dict, port_name: str, path: str) -> bool:
    ip = inst.get("public_ipaddr")
    hp = host_port(inst, port_name)
    if not ip or not hp:
        return False
    try:
        r = requests.get(
            f"http://{ip}:{hp}{path}",
            headers={"Authorization": f"Bearer {inst.get('jupyter_token', '')}"},
            timeout=10,
        )
        return r.status_code == 200
    except requests.RequestException:
        return False


def search_offers(key: str, cfg: dict) -> list[dict]:
    offers = vast(key, "search", "offers", cfg["query"], "-o", "dph_total",
                  "--limit", "10", "--storage", str(cfg["disk"]))
    return offers if isinstance(offers, list) else []


def ensure_role(key: str, role: str, log, exclude_hosts=None, skip_ids=()) -> dict:
    cfg, exclude_hosts = ROLES[role], BAD_HOSTS | set(exclude_hosts or ())
    for inst in labeled_instances(key, cfg["label"]):
        if inst["id"] in skip_ids:
            continue
        if inst.get("actual_status") not in FAILED_STATES:
            if inst.get("actual_status") == "stopped":
                log(f"{role}: starting stopped instance {inst['id']}")
                vast(key, "start", "instance", str(inst["id"]))
            log(f"{role}: reusing instance {inst['id']} ({inst.get('gpu_name')})")
            return inst
        log(f"{role}: instance {inst['id']} is {inst.get('actual_status')}, ignoring")
    thash = template_hash(key, cfg["template"])
    offers = [o for o in search_offers(key, cfg)
              if (o.get("host_id") or o.get("machine_id")) not in exclude_hosts]
    if not offers:
        raise RuntimeError(f"{role}: no offers match query")
    last_err = None
    for offer in offers[:3]:
        log(f"{role}: renting offer {offer.get('id')} "
            f"{offer.get('gpu_name')} ${offer.get('dph_total', 0):.3f}/hr "
            f"{offer.get('geolocation', '?')}")
        try:
            out = vast(key, "create", "instance", str(offer["id"]),
                       "--template_hash", thash, "--disk", str(cfg["disk"]),
                       "--label", cfg["label"], "--cancel-unavail")
            inst_id = out.get("new_contract")
            log(f"{role}: created instance {inst_id}")
            return {"id": inst_id}
        except Exception as exc:  # try next offer
            last_err = exc
            log(f"{role}: offer {offer.get('id')} failed: {exc}")
    raise RuntimeError(f"{role}: could not rent any offer: {last_err}")


def wait_ready(key: str, role: str, inst: dict, log) -> dict:
    cfg = ROLES[role]
    inst_id = inst["id"]
    deadline, last_sig, err_polls = time.time() + READY_TIMEOUT, None, 0
    while time.time() < deadline:
        cur = vast(key, "show", "instance", str(inst_id))
        status = cur.get("actual_status")
        msg = (cur.get("status_msg") or "")[:100]
        ready = status == "running" and probe(cur, cfg["ready_port"], cfg["ready_path"])
        sig = (status, msg, ready)
        if sig != last_sig:
            log(f"{role}: status={status} msg={msg!r} ready={ready}")
            last_sig = sig
        if ready:
            return cur
        err_polls = err_polls + 1 if "error" in msg.lower() else 0
        if status in FAILED_STATES or err_polls >= 3:
            log(f"{role} looks stuck — press Retry {role.capitalize()}")
            raise RuntimeError(f"{role} stuck in status {status}: {msg}")
        time.sleep(POLL_SECONDS)
    raise RuntimeError(f"{role}: timed out waiting for readiness")


def wire(key: str, media: dict, llm: dict, log) -> None:
    ip, hp = llm.get("public_ipaddr"), host_port(llm, "8000/tcp")
    token = llm.get("jupyter_token", "")
    models = requests.get(f"http://{ip}:{hp}/v1/models",
                          headers={"Authorization": f"Bearer {token}"}, timeout=15).json()
    model_id = models["data"][0]["id"]
    m_ip, m_hp = media.get("public_ipaddr"), host_port(media, "4173/tcp")
    payload = {"sections": {"llm": {
        "model_provider": "openai", "model": model_id,
        "base_url": f"http://{ip}:{hp}/v1", "api_key": token,
    }}}
    requests.put(f"http://{m_ip}:{m_hp}/api/config",
                 headers={"Authorization": f"Bearer {media.get('jupyter_token', '')}"},
                 json=payload, timeout=15).raise_for_status()
    log(f"wire: ViMax LLM set to {model_id} @ http://{ip}:{hp}/v1 (token hidden)")


def links_md(media: dict | None, llm: dict | None) -> str:
    media, llm = media or {}, llm or {}
    m_ip, l_ip = media.get("public_ipaddr"), llm.get("public_ipaddr")
    m_tok, l_tok = media.get("jupyter_token", ""), llm.get("jupyter_token", "")
    out = ["### Links (contain the access token — keep them private)", ""]
    if m_ip:
        out.append(f"- [ViMax](http://{m_ip}:{host_port(media, '4173/tcp')}/?token={m_tok})")
        out.append(f"- [ComfyUI](http://{m_ip}:{host_port(media, '8188/tcp')}/?token={m_tok})")
    if l_ip:
        out.append(f"- [vLLM API docs](http://{l_ip}:{host_port(llm, '8000/tcp')}/docs?token={l_tok})")
    return "\n".join(out)


def launch_stack(key: str, log):
    media = ensure_role(key, "media", log)
    llm = ensure_role(key, "llm", log)
    media = wait_ready(key, "media", media, log)
    llm = wait_ready(key, "llm", llm, log)
    wire(key, media, llm, log)
    log("stack is up")
    return links_md(media, llm)


def running_role(key: str, role: str) -> dict | None:
    return next((i for i in labeled_instances(key, ROLES[role]["label"])
                 if i.get("actual_status") == "running"), None)


def retry_role(key: str, role: str, log):
    cfg = ROLES[role]
    excluded, destroyed = set(), set()
    for inst in labeled_instances(key, cfg["label"]):
        hid = inst.get("host_id") or inst.get("machine_id")
        if hid:
            excluded.add(hid)
        log(f"{role}: destroying instance {inst['id']}")
        vast(key, "destroy", "instance", str(inst["id"]), "-y")
        destroyed.add(inst["id"])
    inst = wait_ready(key, role, ensure_role(key, role, log, excluded, destroyed), log)
    media = inst if role == "media" else running_role(key, "media")
    llm = inst if role == "llm" else running_role(key, "llm")
    if media and llm:
        wire(key, media, llm, log)  # a fresh media box has no LLM config
    log(f"{role} recovered")
    return links_md(media, llm)


def stream(fn, key: str):
    """Wrap a (key, log)->links fn as a Gradio generator streaming the log."""
    logs: list[str] = []
    done: dict = {}

    def target():
        try:
            done["links"] = fn(key, lambda m: logs.append(m))
        except Exception as exc:
            done["error"] = str(exc)
    import threading
    t = threading.Thread(target=target, daemon=True)
    t.start()
    while t.is_alive():
        yield "\n".join(logs), ""
        time.sleep(1)
    yield "\n".join(logs + ([f"ERROR: {done['error']}"] if "error" in done else ["done"])), done.get("links", "")


def dry_run(key: str) -> None:
    me = user_id(key)
    print(f"account user_id={me}")
    for role, cfg in ROLES.items():
        thash = template_hash(key, cfg["template"])
        print(f"\n[{role}] template {cfg['template']} -> {thash}")
        insts = labeled_instances(key, cfg["label"])
        for inst in insts:
            ready = probe(inst, cfg["ready_port"], cfg["ready_path"])
            print(f"  instance {inst['id']} status={inst.get('actual_status')} "
                  f"gpu={inst.get('gpu_name')} ready={ready} "
                  f"ip={inst.get('public_ipaddr')} hostport={host_port(inst, cfg['ready_port'])}")
        if not insts:
            print("  no labeled instance")
        offers = [o for o in search_offers(key, cfg) if o.get("host_id") not in BAD_HOSTS]
        if offers:
            o = offers[0]
            print(f"  would rent offer {o.get('id')} {o.get('gpu_name')} "
                  f"${o.get('dph_total', 0):.3f}/hr {o.get('geolocation', '?')} host={o.get('host_id')}")
        else:
            print("  no offers match query")


def build_ui(key: str):
    import gradio as gr
    with gr.Blocks(title="ViMax vast.ai control panel") as demo:
        gr.Markdown(
            "## ViMax vast.ai control panel\n"
            "Launches/checks the media (ComfyUI + ViMax) and LLM (vLLM) instances, then wires "
            "ViMax's LLM settings automatically. Retry destroys the labeled instance and re-rents "
            "on a different host. Links below contain the access token — keep them private."
        )
        launch = gr.Button("Launch stack", variant="primary", size="lg")
        with gr.Row():
            retry_m = gr.Button("Retry media", variant="secondary")
            retry_l = gr.Button("Retry LLM", variant="secondary")
        links = gr.Markdown("")
        log = gr.Textbox(label="Log", lines=18, max_lines=40, autoscroll=True, interactive=False)
        kw = {"concurrency_id": "vast", "concurrency_limit": 1}
        launch.click(lambda: (yield from stream(launch_stack, key)), outputs=[log, links], **kw)
        retry_m.click(
            lambda: (yield from stream(lambda k, lg: retry_role(k, "media", lg), key)),
            outputs=[log, links], **kw)
        retry_l.click(
            lambda: (yield from stream(lambda k, lg: retry_role(k, "llm", lg), key)),
            outputs=[log, links], **kw)
    return demo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    key = load_api_key()
    if not key:
        print("VASTAI_TOKEN not set (env or .env)", file=sys.stderr)
        return 2
    if args.dry_run:
        dry_run(key)
        return 0
    build_ui(key).queue().launch(server_name="127.0.0.1", server_port=7860)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
