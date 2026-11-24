from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import re
import uuid
from pathlib import Path
from typing import Any, Iterable

import aiohttp
import yaml


class ComfyUIError(RuntimeError):
    def __init__(self, message: str, details: Any = None) -> None:
        self.details = details
        if details is not None:
            message = f"{message}: {details}"
        super().__init__(message)


DEFAULT_PRESETS_PATH = Path(__file__).resolve().parents[1] / "configs" / "comfyui" / "presets.yaml"
PLACEHOLDER_RE = re.compile(r"^\{\{(\w+)\}\}$")


def _env_float(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _poll_seconds() -> float:
    return max(0.5, _env_float("VIMAX_COMFYUI_POLL_SECONDS", 3.0))


def _timeout_seconds() -> float:
    return _env_float("VIMAX_COMFYUI_TIMEOUT_SECONDS", 1800.0)


class ComfyUIClient:
    def __init__(self, base_url: str, api_key: str = "") -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.client_id = uuid.uuid4().hex

    def _headers(self) -> dict[str, str]:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    async def upload_image(self, path: str | Path) -> str:
        """POST /upload/image; returns the server-side image name (subfolder/name)."""
        form = aiohttp.FormData()
        form.add_field("overwrite", "true")
        with open(path, "rb") as handle:
            form.add_field("image", handle.read(), filename=Path(path).name)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/upload/image",
                data=form,
                headers=self._headers(),
            ) as response:
                body = await response.json(content_type=None)
                if response.status >= 400:
                    raise ComfyUIError(f"ComfyUI image upload failed with HTTP {response.status}", body)
        name = body.get("name")
        if not name:
            raise ComfyUIError("ComfyUI image upload response missing name", body)
        subfolder = body.get("subfolder")
        return f"{subfolder}/{name}" if subfolder else name

    async def queue(self, workflow: dict) -> str:
        """POST /prompt; returns the prompt_id."""
        payload = {"prompt": workflow, "client_id": self.client_id}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/prompt",
                json=payload,
                headers={**self._headers(), "Content-Type": "application/json"},
            ) as response:
                text = await response.text()
                try:
                    body = json.loads(text)
                except json.JSONDecodeError:
                    body = {"message": text}
                if response.status >= 400:
                    raise ComfyUIError(f"ComfyUI prompt rejected with HTTP {response.status}", body)
        node_errors = body.get("node_errors") or {}
        if node_errors:
            raise ComfyUIError("ComfyUI workflow has node_errors", node_errors)
        prompt_id = body.get("prompt_id")
        if not prompt_id:
            raise ComfyUIError("ComfyUI /prompt response missing prompt_id", body)
        return prompt_id

    async def wait(self, prompt_id: str) -> dict:
        """Poll GET /history/{prompt_id} until the run finishes; returns the history entry."""
        deadline = asyncio.get_running_loop().time() + _timeout_seconds()
        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(_poll_seconds())
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/history/{prompt_id}",
                    headers=self._headers(),
                ) as response:
                    if response.status >= 400:
                        raise ComfyUIError(f"ComfyUI history poll failed with HTTP {response.status}")
                    body = await response.json(content_type=None)
            entry = body.get(prompt_id)
            if not entry:
                continue
            status = entry.get("status") or {}
            status_str = status.get("status_str")
            if status_str == "error":
                raise ComfyUIError(
                    "ComfyUI execution failed",
                    _execution_error_detail(status) or status,
                )
            if status.get("completed") or status_str == "success":
                return entry
            # Entries without a finished/completed flag but with outputs are done.
            if entry.get("outputs") and status_str != "error":
                return entry
        raise ComfyUIError(f"ComfyUI execution timed out after {_timeout_seconds():g}s for prompt {prompt_id}")

    async def download(self, file_info: dict) -> bytes:
        """GET /view?filename=&subfolder=&type=."""
        params = {
            "filename": file_info.get("filename", ""),
            "subfolder": file_info.get("subfolder", ""),
            "type": file_info.get("type", "output"),
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/view",
                params=params,
                headers=self._headers(),
            ) as response:
                if response.status >= 400:
                    raise ComfyUIError(f"ComfyUI download failed with HTTP {response.status}", file_info)
                return await response.read()


def _execution_error_detail(status: dict) -> Any:
    for message in status.get("messages") or []:
        if isinstance(message, (list, tuple)) and len(message) >= 2 and message[0] == "execution_error":
            return message[1]
        if isinstance(message, dict) and message.get("type") == "execution_error":
            return message
    return None


def load_presets(path: str | Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    presets = payload.get("presets")
    if not isinstance(presets, dict):
        raise RuntimeError(f"ComfyUI presets file has no 'presets' mapping: {path}")
    return presets


def get_preset(presets: dict, name: str) -> dict:
    preset = presets.get(name)
    if preset is None:
        available = ", ".join(sorted(presets)) or "(none)"
        raise RuntimeError(f"Unknown ComfyUI preset '{name}'. Available presets: {available}")
    return preset


def load_workflow(directory: str | Path, filename: str) -> dict:
    path = Path(directory) / filename
    return json.loads(path.read_text(encoding="utf-8"))


def _node_has_unfilled_placeholder(node: dict) -> bool:
    for value in (node.get("inputs") or {}).values():
        if isinstance(value, str) and PLACEHOLDER_RE.match(value):
            return True
    return False


def _is_link(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[1], int)


def fill_placeholders(
    workflow: dict,
    values: dict,
    optional_inputs: Iterable[str] = (),
) -> dict:
    """Substitute {{name}} placeholders, delete unfilled nodes, and cascade.

    - A string input exactly "{{name}}" with name in values is replaced by the
      value, keeping its type.
    - Any node still containing an unfilled placeholder is deleted.
    - Then repeatedly: a remaining node whose input links to a deleted node
      drops that input if its name is in optional_inputs, otherwise the node
      is deleted too — until stable.
    """
    result = copy.deepcopy(workflow)
    optional = set(optional_inputs)

    for node in result.values():
        inputs = node.get("inputs") or {}
        for key, value in list(inputs.items()):
            if isinstance(value, str):
                match = PLACEHOLDER_RE.match(value)
                if match and match.group(1) in values:
                    inputs[key] = values[match.group(1)]

    deleted = {node_id for node_id, node in result.items() if _node_has_unfilled_placeholder(node)}
    for node_id in deleted:
        del result[node_id]

    changed = True
    while changed:
        changed = False
        for node_id, node in list(result.items()):
            inputs = node.get("inputs") or {}
            for input_name, value in list(inputs.items()):
                if _is_link(value) and str(value[0]) in deleted:
                    if input_name in optional:
                        del inputs[input_name]
                        continue
                    deleted.add(node_id)
                    del result[node_id]
                    changed = True
                    break
    return result


def _fresh_ids(existing: Iterable[str], prefix: str, count: int) -> list[str]:
    used = {str(node_id) for node_id in existing}
    ids = []
    index = 0
    while len(ids) < count:
        candidate = f"{prefix}_{index}"
        index += 1
        if candidate not in used:
            used.add(candidate)
            ids.append(candidate)
    return ids


def attach_reference_chain(workflow: dict, image_names: list[str], chain_cfg: dict) -> dict:
    """Add LoadImage -> ImageScaleToTotalPixels -> VAEEncode per reference image
    and wrap each target's conditioning input through ReferenceLatent nodes."""
    result = copy.deepcopy(workflow)
    if not image_names:
        return result

    megapixels = chain_cfg.get("megapixels", 1.0)
    vae_link = list(chain_cfg["vae"])
    load_ids = _fresh_ids(result, "vimax_ref_load", len(image_names))
    scale_ids = _fresh_ids([*result, *load_ids], "vimax_ref_scale", len(image_names))
    encode_ids = _fresh_ids([*result, *load_ids, *scale_ids], "vimax_ref_encode", len(image_names))

    for i, name in enumerate(image_names):
        result[load_ids[i]] = {
            "class_type": "LoadImage",
            "inputs": {"image": name},
        }
        result[scale_ids[i]] = {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {
                "image": [load_ids[i], 0],
                "upscale_method": "lanczos",
                "megapixels": megapixels,
            },
        }
        result[encode_ids[i]] = {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": [scale_ids[i], 0],
                "vae": vae_link,
            },
        }

    ref_ids = _fresh_ids(result, "vimax_ref_latent", len(image_names) * max(1, len(chain_cfg.get("targets", []))))
    ref_index = 0
    for target in chain_cfg.get("targets", []):
        node_id, input_name = target[0], target[1]
        node = result.get(str(node_id))
        if node is None:
            raise ComfyUIError(f"reference_chain target node '{node_id}' not in workflow")
        inputs = node.setdefault("inputs", {})
        if input_name not in inputs:
            raise ComfyUIError(f"reference_chain target input '{input_name}' missing on node '{node_id}'")
        conditioning = inputs[input_name]
        for i in range(len(image_names)):
            ref_id = ref_ids[ref_index]
            ref_index += 1
            result[ref_id] = {
                "class_type": "ReferenceLatent",
                "inputs": {
                    "conditioning": conditioning,
                    "latent": [encode_ids[i], 0],
                },
            }
            conditioning = [ref_id, 0]
        inputs[input_name] = conditioning
    return result


def find_output_files(history_entry: dict, extensions: Iterable[str]) -> list[dict]:
    """Collect output file dicts (filename matching extensions) across all
    outputs[*] lists; entries with type == "output" are preferred."""
    exts = {ext.lower().lstrip(".") for ext in extensions}
    matched: list[dict] = []
    outputs = history_entry.get("outputs") or {}
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        for file_list in node_output.values():
            if not isinstance(file_list, list):
                continue
            for item in file_list:
                if not isinstance(item, dict):
                    continue
                filename = item.get("filename") or ""
                if filename.rsplit(".", 1)[-1].lower() in exts:
                    matched.append(item)
    preferred = [item for item in matched if item.get("type") == "output"]
    return preferred or matched


def snap_h3_length(seconds: float, fps: int = 24) -> int:
    """MiniMax-H3 requires a frame count of 17k + 5."""
    k = max(1, round((seconds * fps - 5) / 17))
    return 17 * k + 5


def parse_size(
    size: Any,
    aspect_ratio: Any,
    default: tuple[int, int] = (1600, 900),
) -> tuple[int, int]:
    """Resolve a pixel size from "WxH"/"W*H" or an aspect ratio at ~1 MP.
    Each dimension is rounded to the nearest multiple of 16, minimum 64."""
    parsed = _parse_wxh(size)
    if parsed is None:
        ratio = _parse_wxh(aspect_ratio)
        if ratio:
            scale = math.sqrt(1_000_000 / (ratio[0] * ratio[1]))
            parsed = (ratio[0] * scale, ratio[1] * scale)
        else:
            parsed = default
    width = max(64, int(round(parsed[0] / 16)) * 16)
    height = max(64, int(round(parsed[1] / 16)) * 16)
    return width, height


def _parse_wxh(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    separator = "x" if "x" in normalized else "*" if "*" in normalized else ":" if ":" in normalized else ""
    if not separator:
        return None
    left, right = normalized.split(separator, 1)
    try:
        width = float(left.strip())
        height = float(right.strip())
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height
