from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, List

from interfaces.video_output import VideoOutput
from tools.comfyui_client import (
    DEFAULT_PRESETS_PATH,
    ComfyUIClient,
    ComfyUIError,
    fill_placeholders,
    find_output_files,
    get_preset,
    load_presets,
    load_workflow,
    snap_h3_length,
)

VIDEO_EXTENSIONS = ("mp4", "webm")


def _video_seconds() -> float:
    try:
        return max(1.0, float(os.environ.get("VIMAX_COMFYUI_VIDEO_SECONDS", "5")))
    except ValueError:
        return 5.0


class VideoGeneratorComfyUI:
    """Generate videos through a self-hosted ComfyUI instance.

    ``model`` is ``comfyui/<preset>`` where the preset names an entry in
    configs/comfyui/presets.yaml.
    """

    def __init__(
        self,
        base_url: str,
        model: str = "comfyui/minimax-h3",
        api_key: str = "",
        presets_path: str | Path | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.presets_path = Path(presets_path) if presets_path else DEFAULT_PRESETS_PATH
        self.preset_name = model.split("/", 1)[1] if "/" in model else model
        self.preset = get_preset(load_presets(self.presets_path), self.preset_name)
        if self.preset.get("kind") != "video":
            raise RuntimeError(f"ComfyUI preset '{self.preset_name}' is kind '{self.preset.get('kind')}', expected 'video'")

    async def generate_single_video(
        self,
        prompt: str = "",
        reference_image_paths: List[str] = [],
        aspect_ratio: str = "16:9",
        **kwargs: Any,
    ) -> VideoOutput:
        progress = kwargs.get("progress")
        references = list(reference_image_paths or [])
        if len(references) > 2:
            raise ValueError("ComfyUI video generation supports at most first and last frame images")

        client = ComfyUIClient(self.base_url, api_key=self.api_key)
        uploaded_names = [await client.upload_image(path) for path in references]
        first_frame = uploaded_names[0] if uploaded_names else None
        last_frame = uploaded_names[1] if len(uploaded_names) > 1 else None

        seconds = _video_seconds()
        workflow_dir = self.presets_path.parent
        workflow = load_workflow(workflow_dir, self.preset["workflow"])
        values = dict(self.preset.get("values") or {})
        values.update(
            prompt=prompt,
            seed=random.randint(0, 2**31 - 1),
            length=snap_h3_length(seconds, int(values.get("fps", 24))),
        )
        if first_frame is not None:
            values["first_frame"] = first_frame
        if last_frame is not None:
            values["last_frame"] = last_frame
        workflow = fill_placeholders(workflow, values, self.preset.get("optional_inputs") or ())

        _emit_progress(
            progress,
            "video_create",
            f"Queueing ComfyUI video generation with preset {self.preset_name}",
            {"model": self.model, "seconds": seconds, "length": values["length"], "has_first_frame": first_frame is not None, "has_last_frame": last_frame is not None},
        )
        prompt_id = await client.queue(workflow)
        _emit_progress(progress, "video_status", "ComfyUI video generation queued", {"model": self.model, "prompt_id": prompt_id})
        history_entry = await client.wait(prompt_id)

        files = find_output_files(history_entry, VIDEO_EXTENSIONS)
        if not files:
            raise ComfyUIError("ComfyUI run produced no video output", list((history_entry.get("outputs") or {}).keys()))
        data = await client.download(files[0])
        ext = files[0].get("filename", "x.mp4").rsplit(".", 1)[-1].lower()
        _emit_progress(
            progress,
            "video_completed",
            "ComfyUI video generation completed and downloaded",
            {"model": self.model, "prompt_id": prompt_id, "bytes": len(data)},
        )
        return VideoOutput(fmt="bytes", ext=ext, data=data)


def _emit_progress(progress: Any, stage: str, message: str, metadata: dict | None = None) -> None:
    if progress is not None:
        progress(stage, message, metadata or {})
