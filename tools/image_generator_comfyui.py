from __future__ import annotations

import asyncio
import random
from io import BytesIO
from pathlib import Path
from typing import Any, List

import aiohttp
from PIL import Image
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from interfaces.image_output import ImageOutput
from tools.comfyui_client import (
    DEFAULT_PRESETS_PATH,
    ComfyUIClient,
    ComfyUIError,
    attach_reference_chain,
    fill_placeholders,
    find_output_files,
    get_preset,
    load_presets,
    load_workflow,
    parse_size,
)
from tools.image_orientation import ensure_not_portrait, landscape_guard_requested
from utils.rate_limiter import RateLimiter
from utils.retry import after_func

IMAGE_EXTENSIONS = ("png", "jpg", "jpeg", "webp")


def _is_retryable_comfyui_error(exc: BaseException) -> bool:
    if isinstance(exc, ComfyUIError):
        return False
    if isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError)):
        return True
    return isinstance(exc, ValueError) and "portrait-oriented" in str(exc)


class ImageGeneratorComfyUI:
    """Generate images through a self-hosted ComfyUI instance.

    ``model`` is ``comfyui/<preset>`` where the preset names an entry in
    configs/comfyui/presets.yaml.
    """

    def __init__(
        self,
        base_url: str,
        model: str = "comfyui/flux2-klein-4b",
        api_key: str = "",
        presets_path: str | Path | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.presets_path = Path(presets_path) if presets_path else DEFAULT_PRESETS_PATH
        self.preset_name = model.split("/", 1)[1] if "/" in model else model
        self.preset = get_preset(load_presets(self.presets_path), self.preset_name)
        if self.preset.get("kind") != "image":
            raise RuntimeError(f"ComfyUI preset '{self.preset_name}' is kind '{self.preset.get('kind')}', expected 'image'")
        self.max_reference_images = int(self.preset.get("max_reference_images", 4))
        self.rate_limiter = rate_limiter

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_is_retryable_comfyui_error),
        after=after_func,
        reraise=True,
    )
    async def generate_single_image(
        self,
        prompt: str,
        reference_image_paths: List[str] | None = None,
        aspect_ratio: str | None = "16:9",
        **kwargs: Any,
    ) -> ImageOutput:
        progress = kwargs.get("progress")
        references = list(reference_image_paths or [])
        if len(references) > self.max_reference_images:
            _emit_progress(
                progress,
                "image_reference_clamped",
                f"ComfyUI preset '{self.preset_name}' supports at most {self.max_reference_images} reference images; dropping {len(references) - self.max_reference_images}",
                {"model": self.model, "reference_count": len(references)},
            )
            references = references[: self.max_reference_images]
        if self.rate_limiter is not None:
            await self.rate_limiter.acquire()

        enforce_landscape = landscape_guard_requested(
            size=kwargs.get("size"),
            aspect_ratio=aspect_ratio,
            enforce_landscape=kwargs.get("enforce_landscape", True),
            allow_portrait=kwargs.get("allow_portrait", False),
        )
        request_prompt = _prompt_with_landscape_requirement(prompt, aspect_ratio) if enforce_landscape else prompt
        width, height = parse_size(kwargs.get("size"), aspect_ratio)

        client = ComfyUIClient(self.base_url, api_key=self.api_key)
        uploaded_names = [await client.upload_image(path) for path in references]

        workflow_dir = self.presets_path.parent
        workflow = load_workflow(workflow_dir, self.preset["workflow"])
        values = dict(self.preset.get("values") or {})
        values.update(
            prompt=request_prompt,
            seed=random.randint(0, 2**31 - 1),
            width=width,
            height=height,
        )
        workflow = fill_placeholders(workflow, values, self.preset.get("optional_inputs") or ())
        if uploaded_names:
            workflow = attach_reference_chain(workflow, uploaded_names, self.preset["reference_chain"])

        _emit_progress(
            progress,
            "image_generation",
            f"Queueing ComfyUI image generation with preset {self.preset_name}",
            {"model": self.model, "reference_count": len(references), "width": width, "height": height},
        )
        prompt_id = await client.queue(workflow)
        history_entry = await client.wait(prompt_id)

        files = find_output_files(history_entry, IMAGE_EXTENSIONS)
        if not files:
            raise ComfyUIError("ComfyUI run produced no image output", list((history_entry.get("outputs") or {}).keys()))
        raw = await client.download(files[0])
        with Image.open(BytesIO(raw)) as opened:
            opened.load()
            image = opened.copy()
        ext = files[0].get("filename", "x.png").rsplit(".", 1)[-1].lower()
        if ext == "jpeg":
            ext = "jpg"
        if enforce_landscape:
            ensure_not_portrait(image)
        _emit_progress(
            progress,
            "image_completed",
            "ComfyUI image generation completed",
            {"model": self.model, "width": image.width, "height": image.height},
        )
        return ImageOutput(fmt="pil", ext=ext, data=image)


def _prompt_with_landscape_requirement(prompt: str, aspect_ratio: str | None) -> str:
    ratio = aspect_ratio or "16:9"
    return f"{prompt}\n\nComposition requirement: create a landscape image with an approximate {ratio} aspect ratio; the width must be greater than the height."


def _emit_progress(progress: Any, stage: str, message: str, metadata: dict | None = None) -> None:
    if progress is not None:
        progress(stage, message, metadata or {})
