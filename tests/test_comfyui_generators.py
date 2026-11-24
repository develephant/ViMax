import base64
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, patch

import yaml
from PIL import Image

from agent_runtime.config import is_comfyui_model
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
    snap_h3_length,
)
from tools.image_generator_comfyui import ImageGeneratorComfyUI
from tools.video_generator_comfyui import VideoGeneratorComfyUI


def _png_bytes(size=(16, 9)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, "blue").save(buffer, format="PNG")
    return buffer.getvalue()


def _png_file(path: Path, size=(16, 9)) -> str:
    Image.new("RGB", size, "red").save(path)
    return str(path)


class FillPlaceholdersTests(unittest.TestCase):
    def setUp(self):
        self.workflow = {
            "1": {"class_type": "Loader", "inputs": {"name": "{{model_name}}", "steps": "{{steps}}"}},
            "2": {"class_type": "Encode", "inputs": {"clip": ["1", 0], "text": "{{prompt}}"}},
            "3": {"class_type": "OptImage", "inputs": {"image": "{{first_frame}}"}},
            "4": {"class_type": "Main", "inputs": {"req": ["2", 0], "opt_in": ["3", 0], "opt_in2": ["5", 0]}},
            "5": {"class_type": "OptImage2", "inputs": {"image": "{{last_frame}}"}},
        }

    def test_values_keep_their_type(self):
        result = fill_placeholders(self.workflow, {"model_name": "x.safetensors", "steps": 4, "prompt": "hi", "first_frame": "a.png", "last_frame": "b.png"})
        self.assertEqual(result["1"]["inputs"]["steps"], 4)
        self.assertEqual(result["1"]["inputs"]["name"], "x.safetensors")

    def test_unfilled_nodes_deleted_and_required_link_cascades(self):
        result = fill_placeholders(self.workflow, {"model_name": "x", "steps": 4, "prompt": "hi"})
        # 3 and 5 unfilled -> deleted; node 4 links to them but its inputs are
        # not optional, so node 4 is cascade-deleted as well.
        self.assertNotIn("3", result)
        self.assertNotIn("5", result)
        self.assertNotIn("4", result)
        self.assertIn("1", result)
        self.assertIn("2", result)

    def test_optional_inputs_are_dropped_not_cascaded(self):
        result = fill_placeholders(
            self.workflow,
            {"model_name": "x", "steps": 4, "prompt": "hi", "first_frame": "a.png"},
            optional_inputs=("opt_in2",),
        )
        self.assertNotIn("5", result)
        self.assertIn("4", result)
        self.assertNotIn("opt_in2", result["4"]["inputs"])
        self.assertEqual(result["4"]["inputs"]["opt_in"], ["3", 0])

    def test_input_workflow_not_mutated(self):
        fill_placeholders(self.workflow, {"model_name": "x", "steps": 4, "prompt": "p", "first_frame": "a", "last_frame": "b"})
        self.assertEqual(self.workflow["1"]["inputs"]["steps"], "{{steps}}")


class AttachReferenceChainTests(unittest.TestCase):
    def setUp(self):
        self.workflow = {
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": "v.safetensors"}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "p"}},
            "6": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "cfg": 1.0}},
        }
        self.chain = {"vae": ["3", 0], "targets": [["6", "positive"]], "megapixels": 1.0}

    def test_no_images_returns_unchanged(self):
        result = attach_reference_chain(self.workflow, [], self.chain)
        self.assertEqual(result, self.workflow)

    def test_single_image_rewires_target(self):
        result = attach_reference_chain(self.workflow, ["ref.png"], self.chain)
        encodes = [nid for nid, n in result.items() if n["class_type"] == "VAEEncode"]
        self.assertEqual(len(encodes), 1)
        encode = result[encodes[0]]
        self.assertEqual(encode["inputs"]["vae"], ["3", 0])
        scale = result[encode["inputs"]["pixels"][0]]
        self.assertEqual(scale["class_type"], "ImageScaleToTotalPixels")
        self.assertEqual(scale["inputs"]["megapixels"], 1.0)
        load = result[scale["inputs"]["image"][0]]
        self.assertEqual(load["inputs"]["image"], "ref.png")
        positive = result["6"]["inputs"]["positive"]
        ref_node = result[positive[0]]
        self.assertEqual(ref_node["class_type"], "ReferenceLatent")
        self.assertEqual(ref_node["inputs"]["conditioning"], ["4", 0])
        self.assertEqual(ref_node["inputs"]["latent"], [encodes[0], 0])

    def test_four_images_chain_and_unique_ids(self):
        names = ["a.png", "b.png", "c.png", "d.png"]
        result = attach_reference_chain(self.workflow, names, self.chain)
        encodes = [nid for nid, n in result.items() if n["class_type"] == "VAEEncode"]
        refs = [nid for nid, n in result.items() if n["class_type"] == "ReferenceLatent"]
        self.assertEqual(len(encodes), 4)
        self.assertEqual(len(refs), 4)
        # chain: last ref node is the new positive input
        positive = result["6"]["inputs"]["positive"]
        self.assertEqual(positive[0], refs[-1] if refs[-1] == positive[0] else positive[0])
        seen = set()
        node = result[positive[0]]
        while node["class_type"] == "ReferenceLatent":
            seen.add(node["inputs"]["latent"][0])
            node = result[node["inputs"]["conditioning"][0]] if isinstance(node["inputs"]["conditioning"], list) else None
            if node is None:
                break
        self.assertEqual(seen, set(encodes))

    def test_missing_target_node_raises(self):
        with self.assertRaises(ComfyUIError):
            attach_reference_chain(self.workflow, ["a.png"], {"vae": ["3", 0], "targets": [["99", "positive"]]})


class PureHelperTests(unittest.TestCase):
    def test_parse_size(self):
        # explicit WxH is rounded to nearest multiple of 16
        self.assertEqual(parse_size("1920x1080", "16:9"), (1920, 1088))
        w, h = parse_size("1000*500", None)
        self.assertEqual((w % 16, h % 16), (0, 0))
        self.assertTrue((w, h) in {(992, 496), (1008, 496)})
        self.assertEqual(parse_size("30x30", None), (64, 64))
        w, h = parse_size(None, "16:9")
        self.assertGreater(w, h)
        self.assertEqual(w % 16, 0)
        self.assertEqual(h % 16, 0)
        self.assertEqual(parse_size(None, "bad"), (1600, 896))

    def test_snap_h3_length(self):
        # 5s @ 24fps -> k = round((120-5)/17) = round(6.76) = 7 -> 17*7+5 = 124
        self.assertEqual(snap_h3_length(5, 24), 124)
        self.assertEqual(snap_h3_length(1, 24), 22)
        self.assertEqual(snap_h3_length(10, 24), 17 * 14 + 5)

    def test_find_output_files_prefers_output_type(self):
        entry = {
            "outputs": {
                "9": {"images": [{"filename": "a.png", "type": "temp"}, {"filename": "b.png", "type": "output"}, {"filename": "c.txt", "type": "output"}]},
                "10": {"gifs": [{"filename": "d.png", "type": "output"}]},
            }
        }
        files = find_output_files(entry, ["png"])
        self.assertEqual({f["filename"] for f in files}, {"b.png", "d.png"})
        files = find_output_files(entry, ["png", "txt"])
        self.assertEqual(len(files), 3)

    def test_is_comfyui_model(self):
        self.assertTrue(is_comfyui_model("comfyui/flux2-klein-4b"))
        self.assertTrue(is_comfyui_model("ComfyUI/Minimax-H3"))
        self.assertFalse(is_comfyui_model("openai/gpt-image-2"))
        self.assertFalse(is_comfyui_model(""))


class FakeResponse:
    def __init__(self, status=200, body=None, raw=b""):
        self.status = status
        self._body = body
        self._raw = raw

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def json(self, content_type=None):
        return self._body

    async def text(self):
        return json.dumps(self._body)

    async def read(self):
        return self._raw


class FakeSession:
    def __init__(self, routes):
        self.routes = routes
        self.requests = []

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def _match(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs))
        for (m, suffix), response in self.routes.items():
            if m == method and url.endswith(suffix):
                return response
        raise AssertionError(f"no route for {method} {url}")

    def post(self, url, **kwargs):
        return self._match("POST", url, **kwargs)

    def get(self, url, **kwargs):
        return self._match("GET", url, **kwargs)


class ComfyUIClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_queue_raises_on_node_errors(self):
        session = FakeSession({("POST", "/prompt"): FakeResponse(200, {"prompt_id": "p1", "node_errors": {"6": {"errors": ["bad"]}}})})
        client = ComfyUIClient("http://localhost:18188")
        with patch("tools.comfyui_client.aiohttp.ClientSession", session):
            with self.assertRaises(ComfyUIError) as ctx:
                await client.queue({"1": {}})
        self.assertIn("node_errors", str(ctx.exception))

    async def test_queue_raises_on_http_error(self):
        session = FakeSession({("POST", "/prompt"): FakeResponse(400, {"error": "nope"})})
        client = ComfyUIClient("http://localhost:18188")
        with patch("tools.comfyui_client.aiohttp.ClientSession", session):
            with self.assertRaises(ComfyUIError):
                await client.queue({"1": {}})

    async def test_queue_returns_prompt_id(self):
        session = FakeSession({("POST", "/prompt"): FakeResponse(200, {"prompt_id": "abc", "node_errors": {}})})
        client = ComfyUIClient("http://localhost:18188")
        with patch("tools.comfyui_client.aiohttp.ClientSession", session):
            self.assertEqual(await client.queue({"1": {}}), "abc")

    async def test_wait_surfaces_execution_error(self):
        entry = {
            "status": {"status_str": "error", "completed": False, "messages": [["execution_error", {"node_id": "5", "exception_message": "boom"}]]},
            "outputs": {},
        }
        session = FakeSession({("GET", "/history/p1"): FakeResponse(200, {"p1": entry})})
        client = ComfyUIClient("http://localhost:18188")
        with patch("tools.comfyui_client.aiohttp.ClientSession", session), \
             patch.dict("os.environ", {"VIMAX_COMFYUI_POLL_SECONDS": "0.5"}):
            with self.assertRaises(ComfyUIError) as ctx:
                await client.wait("p1")
        self.assertIn("boom", str(ctx.exception))

    async def test_wait_returns_entry_on_success(self):
        entry = {"status": {"status_str": "success", "completed": True}, "outputs": {"9": {}}}
        session = FakeSession({("GET", "/history/p1"): FakeResponse(200, {"p1": entry})})
        client = ComfyUIClient("http://localhost:18188")
        with patch("tools.comfyui_client.aiohttp.ClientSession", session), \
             patch.dict("os.environ", {"VIMAX_COMFYUI_POLL_SECONDS": "0.5"}):
            result = await client.wait("p1")
        self.assertIs(result, entry)

    async def test_authorization_header_sent_when_key_set(self):
        session = FakeSession({("POST", "/prompt"): FakeResponse(200, {"prompt_id": "x"})})
        client = ComfyUIClient("http://localhost:18188", api_key="secret")
        with patch("tools.comfyui_client.aiohttp.ClientSession", session):
            await client.queue({"1": {}})
        self.assertEqual(session.requests[0][2]["headers"]["Authorization"], "Bearer secret")


class PresetsFileTests(unittest.TestCase):
    def test_shipped_presets_load_and_workflows_cover_placeholders(self):
        presets = load_presets(DEFAULT_PRESETS_PATH)
        self.assertIn("flux2-klein-4b", presets)
        self.assertIn("minimax-h3", presets)
        known_filled = {
            "prompt", "seed", "width", "height", "length",
            "first_frame", "last_frame",
        }
        for name, preset in presets.items():
            workflow_path = DEFAULT_PRESETS_PATH.parent / preset["workflow"]
            self.assertTrue(workflow_path.is_file(), f"{name}: missing {workflow_path}")
            workflow = load_workflow(DEFAULT_PRESETS_PATH.parent, preset["workflow"])
            placeholders = set()
            for node in workflow.values():
                for value in (node.get("inputs") or {}).values():
                    if isinstance(value, str) and value.startswith("{{") and value.endswith("}}"):
                        placeholders.add(value[2:-2])
            fillable = set((preset.get("values") or {}).keys()) | known_filled
            missing = placeholders - fillable
            self.assertFalse(missing, f"{name}: placeholders with no fill source: {missing}")

    def test_unknown_preset_lists_available(self):
        presets = load_presets(DEFAULT_PRESETS_PATH)
        with self.assertRaises(RuntimeError) as ctx:
            get_preset(presets, "does-not-exist")
        self.assertIn("flux2-klein-4b", str(ctx.exception))

    def test_generator_rejects_wrong_kind(self):
        with self.assertRaises(RuntimeError):
            ImageGeneratorComfyUI(base_url="http://x", model="comfyui/minimax-h3")
        with self.assertRaises(RuntimeError):
            VideoGeneratorComfyUI(base_url="http://x", model="comfyui/flux2-klein-4b")


class ImageGeneratorComfyUITests(unittest.IsolatedAsyncioTestCase):
    def _generator(self):
        return ImageGeneratorComfyUI(base_url="http://localhost:18188", model="comfyui/flux2-klein-4b")

    def _mock_client(self, test_case):
        upload = AsyncMock(side_effect=lambda path: f"up/{Path(path).name}")
        queue = AsyncMock(return_value="pid1")
        wait = AsyncMock(return_value={"outputs": {"13": {"images": [{"filename": "out.png", "type": "output"}]}}})
        download = AsyncMock(return_value=_png_bytes())
        test_case.enterContext(patch.object(ComfyUIClient, "upload_image", upload))
        test_case.enterContext(patch.object(ComfyUIClient, "queue", queue))
        test_case.enterContext(patch.object(ComfyUIClient, "wait", wait))
        test_case.enterContext(patch.object(ComfyUIClient, "download", download))
        return upload, queue, wait, download

    async def test_end_to_end(self):
        generator = self._generator()
        self.assertEqual(generator.preset_name, "flux2-klein-4b")
        self.assertEqual(generator.max_reference_images, 4)
        upload, queue, wait, download = self._mock_client(self)
        progress = []
        result = await generator.generate_single_image(
            "a beach", aspect_ratio="16:9",
            progress=lambda s, m, md: progress.append(s),
        )
        self.assertIsInstance(result, ImageOutput)
        self.assertEqual(result.ext, "png")
        self.assertEqual(result.data.size, (16, 9))
        self.assertEqual(upload.await_count, 0)
        workflow = queue.await_args.args[0]
        self.assertIn("landscape image", workflow["4"]["inputs"]["text"])
        self.assertIn("image_generation", progress)
        self.assertIn("image_completed", progress)

    async def test_reference_clamp_to_four(self):
        generator = self._generator()
        upload, queue, wait, download = self._mock_client(self)
        progress = []
        await generator.generate_single_image(
            "edit", reference_image_paths=[f"/tmp/r{i}.png" for i in range(6)],
            progress=lambda s, m, md: progress.append(s),
        )
        self.assertEqual(upload.await_count, 4)
        self.assertIn("image_reference_clamped", progress)
        workflow = queue.await_args.args[0]
        ref_nodes = [n for n in workflow.values() if n["class_type"] == "ReferenceLatent"]
        self.assertEqual(len(ref_nodes), 4)

    async def test_portrait_output_retries_and_raises(self):
        generator = self._generator()
        upload, queue, wait, download = self._mock_client(self)
        download.return_value = _png_bytes((9, 16))
        with self.assertRaises(ValueError):
            await generator.generate_single_image("a beach")


class VideoGeneratorComfyUITests(unittest.IsolatedAsyncioTestCase):
    def _generator(self):
        return VideoGeneratorComfyUI(base_url="http://localhost:18188", model="comfyui/minimax-h3")

    def _mock_client(self, test_case):
        upload = AsyncMock(side_effect=lambda path: f"up/{Path(path).name}")
        queue = AsyncMock(return_value="pid1")
        wait = AsyncMock(return_value={"outputs": {"16": {"videos": [{"filename": "out.mp4", "type": "output"}]}}})
        download = AsyncMock(return_value=b"mp4bytes")
        test_case.enterContext(patch.object(ComfyUIClient, "upload_image", upload))
        test_case.enterContext(patch.object(ComfyUIClient, "queue", queue))
        test_case.enterContext(patch.object(ComfyUIClient, "wait", wait))
        test_case.enterContext(patch.object(ComfyUIClient, "download", download))
        return upload, queue, wait, download

    async def test_text_to_video_no_frames(self):
        generator = self._generator()
        upload, queue, wait, download = self._mock_client(self)
        progress = []
        result = await generator.generate_single_video(prompt="a cat", progress=lambda s, m, md: progress.append(s))
        self.assertEqual(result.fmt, "bytes")
        self.assertEqual(result.data, b"mp4bytes")
        workflow = queue.await_args.args[0]
        # LoadImage nodes removed; first_frame/last_frame inputs dropped (optional).
        self.assertNotIn("first_frame", workflow["5"]["inputs"])
        self.assertNotIn("last_frame", workflow["5"]["inputs"])
        self.assertEqual(workflow["5"]["inputs"]["length"], snap_h3_length(5, 24))
        self.assertEqual(progress, ["video_create", "video_status", "video_completed"])

    async def test_first_frame_only(self):
        generator = self._generator()
        upload, queue, wait, download = self._mock_client(self)
        await generator.generate_single_video(prompt="p", reference_image_paths=["/tmp/f.png"])
        workflow = queue.await_args.args[0]
        self.assertEqual(upload.await_count, 1)
        load_nodes = [n for n in workflow.values() if n["class_type"] == "LoadImage"]
        self.assertEqual(len(load_nodes), 1)
        self.assertNotIn("last_frame", workflow["5"]["inputs"])

    async def test_first_and_last_frame(self):
        generator = self._generator()
        upload, queue, wait, download = self._mock_client(self)
        await generator.generate_single_video(prompt="p", reference_image_paths=["/tmp/f.png", "/tmp/l.png"])
        workflow = queue.await_args.args[0]
        load_nodes = [n for n in workflow.values() if n["class_type"] == "LoadImage"]
        self.assertEqual(len(load_nodes), 2)
        images = sorted(n["inputs"]["image"] for n in load_nodes)
        self.assertEqual(images, ["up/f.png", "up/l.png"])

    async def test_more_than_two_frames_raises(self):
        generator = self._generator()
        self._mock_client(self)
        with self.assertRaises(ValueError):
            await generator.generate_single_video(prompt="p", reference_image_paths=["a", "b", "c"])


class AdapterRoutingTests(unittest.TestCase):
    def test_image_adapter_comfyui_no_api_key(self):
        from agent_runtime.vimax_adapters import _build_image_generator
        with patch("agent_runtime.vimax_adapters.image_api_key", return_value=""), \
             patch("agent_runtime.vimax_adapters.image_model", return_value="comfyui/flux2-klein-4b"), \
             patch("agent_runtime.vimax_adapters.image_base_url", return_value="http://127.0.0.1:18188"):
            generator = _build_image_generator()
        self.assertIsInstance(generator, ImageGeneratorComfyUI)
        self.assertEqual(generator.api_key, "")

    def test_video_adapter_comfyui_no_api_key(self):
        from agent_runtime.vimax_adapters import _build_video_generator
        with patch("agent_runtime.vimax_adapters.video_api_key", return_value=""), \
             patch("agent_runtime.vimax_adapters.video_model", return_value="comfyui/minimax-h3"), \
             patch("agent_runtime.vimax_adapters.video_base_url", return_value="http://127.0.0.1:18188"):
            generator = _build_video_generator()
        self.assertIsInstance(generator, VideoGeneratorComfyUI)

    def test_image_adapter_non_comfyui_still_requires_key(self):
        from agent_runtime.vimax_adapters import _build_image_generator
        with patch("agent_runtime.vimax_adapters.image_api_key", return_value=""), \
             patch("agent_runtime.vimax_adapters.image_model", return_value="gemini-3.1-flash-image-preview"), \
             patch("agent_runtime.vimax_adapters.image_base_url", return_value="https://yunwu.ai"):
            with self.assertRaises(RuntimeError):
                _build_image_generator()


class ReferenceSelectorCapTests(unittest.IsolatedAsyncioTestCase):
    def _selector(self, response):
        from agents.reference_image_selector import ReferenceImageSelector

        class Chain:
            def __init__(self):
                self.messages = None

            async def ainvoke(self, messages):
                self.messages = messages
                return response

        class ChatModel:
            def __init__(self, chain):
                self.chain = chain

            def __or__(self, other):
                return self.chain

        chain = Chain()
        return ReferenceImageSelector(ChatModel(chain)), chain

    async def test_max_images_in_multimodal_prompt_and_truncation(self):
        from agents.reference_image_selector import RefImageIndicesAndTextPrompt

        with tempfile.TemporaryDirectory() as tmp:
            pairs = [(_png_file(Path(tmp) / f"r{i}.png"), f"desc {i}") for i in range(6)]
            response = RefImageIndicesAndTextPrompt(ref_image_indices=[0, 1, 2, 3, 4], text_prompt="p")
            selector, chain = self._selector(response)
            with self.assertLogs(level="WARNING"):
                result = await selector.select_reference_images_and_generate_prompt(
                    pairs, "a frame", max_images=3,
                )
            system_content = chain.messages[0].content
            self.assertIn("up to 3", system_content)
            self.assertIn("**3**", system_content)
            self.assertEqual(len(result["reference_image_path_and_text_pairs"]), 3)


class PipelineReferenceCapTests(unittest.IsolatedAsyncioTestCase):
    async def test_pipeline_passes_generator_cap(self):
        from pipelines.script2video_pipeline import Script2VideoPipeline
        import asyncio

        class Generator:
            max_reference_images = 4

            async def generate_single_image(self, **kwargs):
                return ImageOutput(fmt="pil", ext="png", data=Image.new("RGB", (16, 9)))

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Script2VideoPipeline(
                chat_model=object(), image_generator=Generator(), video_generator=object(),
                working_dir=str(tmp), num_image_candidates=1,
            )
            pipeline.reference_image_selector.select_reference_images_and_generate_prompt = AsyncMock(
                return_value={"reference_image_path_and_text_pairs": [], "text_prompt": "frame"},
            )
            pipeline.frame_events[0] = {"first_frame": asyncio.Event(), "last_frame": asyncio.Event()}
            (Path(tmp) / "shots" / "0").mkdir(parents=True)
            await pipeline.generate_frame_for_single_shot(0, "first_frame", ("reference.png", "reference"), "a scene", [], {})
        call = pipeline.reference_image_selector.select_reference_images_and_generate_prompt.await_args
        self.assertEqual(call.kwargs["max_images"], 4)

    async def test_pipeline_defaults_to_eight_without_cap(self):
        from pipelines.script2video_pipeline import Script2VideoPipeline
        import asyncio

        class Generator:
            async def generate_single_image(self, **kwargs):
                return ImageOutput(fmt="pil", ext="png", data=Image.new("RGB", (16, 9)))

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Script2VideoPipeline(
                chat_model=object(), image_generator=Generator(), video_generator=object(),
                working_dir=str(tmp), num_image_candidates=1,
            )
            pipeline.reference_image_selector.select_reference_images_and_generate_prompt = AsyncMock(
                return_value={"reference_image_path_and_text_pairs": [], "text_prompt": "frame"},
            )
            pipeline.frame_events[0] = {"first_frame": asyncio.Event(), "last_frame": asyncio.Event()}
            (Path(tmp) / "shots" / "0").mkdir(parents=True)
            await pipeline.generate_frame_for_single_shot(0, "first_frame", ("reference.png", "reference"), "a scene", [], {})
        call = pipeline.reference_image_selector.select_reference_images_and_generate_prompt.await_args
        self.assertEqual(call.kwargs["max_images"], 8)


if __name__ == "__main__":
    unittest.main()
