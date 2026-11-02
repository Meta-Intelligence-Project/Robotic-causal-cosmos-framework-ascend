# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest


def test_finalize_data_batch_does_not_mutate_reusable_video_list() -> None:
    torch = pytest.importorskip("torch")

    from cosmos_framework.inference.inference import _finalize_data_batch

    model = SimpleNamespace(input_video_key="video", input_image_key="images", input_caption_key="caption")
    original_video = torch.zeros(3, 2, 4, 4, dtype=torch.uint8)
    source_batch = {"video": [original_video], "caption": ["prompt"]}

    first_pass = _finalize_data_batch(source_batch, batch_size=1, model=model)
    first_pass["video"][0] = first_pass["video"][0].float() / 127.5 - 1.0
    first_pass["is_preprocessed"] = True
    second_pass = _finalize_data_batch(source_batch, batch_size=1, model=model)

    assert first_pass["video"] is not source_batch["video"]
    assert second_pass["video"] is not source_batch["video"]
    assert source_batch["video"][0] is original_video
    assert source_batch["video"][0].dtype == torch.uint8
    assert "is_preprocessed" not in source_batch


def _make_v2v_sample_args(**overrides: Any) -> SimpleNamespace:
    """v2v ``OmniSampleArgs`` stand-in for ``get_sample_data`` tests."""
    from cosmos_framework.inference.args import ModelMode, NegativeMetadataMode

    defaults = dict(
        action_path=None,
        aspect_ratio="16,9",
        autoregressive=False,
        camera_trajectory=None,
        condition_frame_indexes_vision=[0, 1],
        condition_video_keep=None,
        condition_vision_mode="video",
        duration_template=None,
        enable_sound=False,
        fps=24,
        inverse_duration_template=None,
        inverse_resolution_template=None,
        model_mode=ModelMode.VIDEO2VIDEO,
        native_prompt_upsampling=False,
        negative_metadata_mode=NegativeMetadataMode.NONE,
        negative_prompt=None,
        num_frames=125,
        num_outputs=1,
        prompt="prompt",
        resolution_template=None,
        transfer_hints={},
        vision_path="conditioning.mp4",
        vision_size=(32, 16),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.mark.parametrize(
    ("condition_video_keep", "expected_loader_keep"),
    [
        ("last", "last"),
        ("first", "first"),
        (None, "first"),  # default
    ],
)
def test_video_conditioning_plumbs_keep_and_pixel_frame_count(
    monkeypatch: pytest.MonkeyPatch,
    condition_video_keep: str | None,
    expected_loader_keep: str,
) -> None:
    """v2v: tokenizer derives pixel-frame count from latent count, ``keep`` passes through to the loader."""
    torch = pytest.importorskip("torch")

    from cosmos_framework.inference import inference

    class Tokenizer:
        calls: list[int]

        def __init__(self) -> None:
            self.calls = []

        def get_pixel_num_frames(self, num_latent_frames: int) -> int:
            self.calls.append(num_latent_frames)
            return 5

    tokenizer = Tokenizer()
    model = SimpleNamespace(
        input_image_key="image",
        input_video_key="video",
        input_caption_key="caption",
        tokenizer_vision_gen=tokenizer,
    )
    sample_args = _make_v2v_sample_args(condition_video_keep=condition_video_keep)
    conditioning_frames = torch.zeros(3, 5, 16, 32)
    sequence_plan = ["sequence-plan"]
    load_conditioning_video_mock = Mock(return_value=conditioning_frames)
    build_conditioned_video_batch_mock = Mock(
        return_value={
            "video": [torch.zeros(1, 3, 125, 16, 32)],
            "sequence_plan": sequence_plan,
        }
    )
    monkeypatch.setattr(inference, "load_conditioning_video", load_conditioning_video_mock)
    monkeypatch.setattr(inference, "build_conditioned_video_batch", build_conditioned_video_batch_mock)

    out = inference.get_sample_data(sample_args, model, device="cpu")

    assert tokenizer.calls == [2]  # max([0, 1]) + 1
    load_conditioning_video_mock.assert_called_once_with(
        Path("conditioning.mp4"),
        target_h=16,
        target_w=32,
        max_frames=5,
        keep=expected_loader_keep,
    )
    build_conditioned_video_batch_mock.assert_called_once()
    build_args, build_kwargs = build_conditioned_video_batch_mock.call_args
    assert build_args == (conditioning_frames,)
    assert build_kwargs == {
        "condition_frames_vision": [0, 1],
        "w": 32,
        "h": 16,
        "num_frames": 125,
        "fps": 24,
        "batch_size": 1,
    }
    assert out["sequence_plan"] is sequence_plan


def test_json_prompt_metadata_for_single_frame_omits_temporal_fields() -> None:
    from cosmos_framework.inference.inference import _format_json_prompt_with_template

    prompt = _format_json_prompt_with_template(
        {"subjects": [], "duration": "8s", "fps": 24.0},
        fps=24,
        num_frames=1,
        aspect_ratio="1,1",
        h=1024,
        w=1024,
        include_temporal_metadata=False,
    )

    assert prompt == '{"subjects": [], "resolution": {"H": 1024, "W": 1024}, "aspect_ratio": "1,1"}'
    parsed = json.loads(prompt)
    assert parsed["resolution"] == {"H": 1024, "W": 1024}
    assert parsed["aspect_ratio"] == "1,1"
    assert "duration" not in parsed
    assert "fps" not in parsed


def test_json_prompt_metadata_for_video_keeps_temporal_fields() -> None:
    from cosmos_framework.inference.inference import _format_json_prompt_with_template

    prompt = _format_json_prompt_with_template(
        {"subjects": []},
        fps=24,
        num_frames=189,
        aspect_ratio="16,9",
        h=720,
        w=1280,
        include_temporal_metadata=True,
    )

    assert prompt == (
        '{"subjects": [], "duration": "7s", "fps": 24.0, "resolution": {"H": 720, "W": 1280}, "aspect_ratio": "16,9"}'
    )
    assert json.loads(prompt) == {
        "subjects": [],
        "duration": "7s",
        "fps": 24.0,
        "resolution": {"H": 720, "W": 1280},
        "aspect_ratio": "16,9",
    }


def _make_reasoner_sample_args(**overrides: Any) -> SimpleNamespace:
    """Reasoner ``OmniSampleArgs`` stand-in for ``get_sample_data`` tests."""
    from cosmos_framework.inference.args import ModelMode

    defaults = dict(
        model_mode=ModelMode.REASONER,
        prompt="Describe a robotic arm.",
        vision_path=None,
        video_fps=None,
        max_new_tokens=8,
        do_sample=False,
        temperature=1.0,
        top_k=None,
        top_p=None,
        num_outputs=1,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.mark.L0
def test_get_sample_data_reasoner_text_only() -> None:
    from cosmos_framework.inference import inference

    model = SimpleNamespace(input_caption_key="caption")
    sample_args = _make_reasoner_sample_args()

    out = inference.get_sample_data(sample_args, model, device="cpu")

    assert out == {
        "caption": ["Describe a robotic arm."],
        "reasoner_images": [None],
        "reasoner_videos": [None],
    }


@pytest.mark.L0
def test_get_sample_data_reasoner_with_image(tmp_path: Path) -> None:
    PIL = pytest.importorskip("PIL.Image")
    from cosmos_framework.inference import inference

    img_path = tmp_path / "arm.png"
    PIL.new("RGB", (8, 8), color="red").save(img_path)

    model = SimpleNamespace(input_caption_key="caption")
    sample_args = _make_reasoner_sample_args(vision_path=str(img_path))

    out = inference.get_sample_data(sample_args, model, device="cpu")

    assert list(out) == ["caption", "reasoner_images", "reasoner_videos"]
    assert out["caption"] == ["Describe a robotic arm."]
    assert out["reasoner_videos"] == [None]
    assert len(out["reasoner_images"]) == 1
    assert out["reasoner_images"][0].size == (8, 8)
    assert out["reasoner_images"][0].mode == "RGB"


@pytest.mark.L0
def test_get_sample_data_reasoner_with_video(monkeypatch: pytest.MonkeyPatch) -> None:
    """A video ``vision_path`` routes through ``_decode_reasoner_video`` into ``reasoner_videos``.

    The decoder is monkeypatched (real decode needs torchvision + an actual clip);
    this asserts the routing/contract, not the decode itself."""
    from cosmos_framework.inference import inference

    decoded = {"frames": ["F0", "F1"], "fps": 2.0}
    monkeypatch.setattr(inference, "_decode_reasoner_video", lambda path, fps: decoded)
    model = SimpleNamespace(input_caption_key="caption")
    sample_args = _make_reasoner_sample_args(vision_path="/tmp/clip.mp4", video_fps=2.0)

    out = inference.get_sample_data(sample_args, model, device="cpu")

    assert out["caption"] == ["Describe a robotic arm."]
    assert out["reasoner_videos"] == [decoded]
    assert out["reasoner_images"] == [None]


@pytest.mark.L0
def test_reasoner_defaults_json_round_trip() -> None:
    import json as _json

    from cosmos_framework.inference.args import PACKAGE_DIR, _load_modality_defaults

    defaults = _load_modality_defaults("reasoner")
    assert defaults["model_mode"] == "reasoner"
    assert defaults["max_new_tokens"] == 1024
    on_disk = _json.loads((PACKAGE_DIR / "defaults/reasoner/sample_args.json").read_text())
    assert defaults == on_disk


@pytest.mark.L0
def test_reasoner_overrides_round_trip() -> None:
    import pydantic

    from cosmos_framework.inference.args import ModelMode, ReasonerDataOverrides

    overrides = ReasonerDataOverrides(max_new_tokens=128, temperature=0.7, top_p=0.9)
    assert overrides.max_new_tokens == 128
    assert overrides.temperature == 0.7
    assert overrides.top_p == 0.9
    with pytest.raises(pydantic.ValidationError):
        ReasonerDataOverrides(top_p=1.5)
    with pytest.raises(pydantic.ValidationError):
        ReasonerDataOverrides(temperature=0)
    assert ModelMode.REASONER.is_reasoner
    assert not ModelMode.TEXT2VIDEO.is_reasoner
    assert not ModelMode.REASONER.is_action


@pytest.mark.L0
def test_generate_reasoner_batch_writes_outputs(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from cosmos_framework.inference import inference
    from cosmos_framework.inference.args import ModelMode

    out_dir = tmp_path / "hello"

    class _SA(SimpleNamespace):
        def model_dump(self, **_):
            return {"name": self.name, "model_mode": self.model_mode}

        def model_dump_json(self, **_):
            import json as _json

            return _json.dumps(self.model_dump())

    sample_args = _SA(
        name="hello",
        model_mode=ModelMode.REASONER,
        output_dir=out_dir,
        prompt="Describe a robotic arm.",
        max_new_tokens=8,
        do_sample=False,
        temperature=1.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        seed=None,
    )

    def _fake_generate_reasoner_text(prompts, *, images=None, **kwargs):
        assert prompts == ["Describe a robotic arm."]
        assert images is None
        return ["A six-axis arm with a parallel-jaw gripper."]

    model = SimpleNamespace(
        input_caption_key="caption",
        generate_reasoner_text=_fake_generate_reasoner_text,
    )

    pipe = inference.OmniInference.__new__(inference.OmniInference)
    pipe.model = model
    pipe.should_process_sample = lambda sa: True  # type: ignore[attr-defined]
    pipe._run_text_guardrail = lambda *_a, **_kw: None  # type: ignore[attr-defined]
    pipe._handle_sample_exception = lambda sa, e: (_ for _ in ()).throw(e)  # type: ignore[attr-defined]

    from contextlib import nullcontext

    pipe._get_timer = lambda *_a, **_kw: nullcontext()  # type: ignore[attr-defined]

    data_batch = {"caption": ["Describe a robotic arm."], "reasoner_images": [None]}
    results = pipe._generate_reasoner_batch([sample_args], data_batch)

    assert len(results) == 1
    so = results[0]
    assert so.outputs[0].content == {"reasoner_text": "A six-axis arm with a parallel-jaw gripper."}
    txt_file = out_dir / "reasoner_text.txt"
    assert txt_file.read_text() == "A six-axis arm with a parallel-jaw gripper."
    assert (out_dir / "sample_args.json").is_file()
    assert (out_dir / "sample_outputs.json").is_file()


@pytest.mark.L0
def test_generate_reasoner_batch_rejects_mixed_image_text_only(tmp_path: Path) -> None:
    PIL = pytest.importorskip("PIL.Image")
    from cosmos_framework.inference import inference
    from cosmos_framework.inference.args import ModelMode

    pipe = inference.OmniInference.__new__(inference.OmniInference)
    pipe.model = SimpleNamespace(input_caption_key="caption")
    pipe.should_process_sample = lambda sa: False  # type: ignore[attr-defined]

    sa1 = SimpleNamespace(model_mode=ModelMode.REASONER, output_dir=tmp_path / "a")
    sa2 = SimpleNamespace(model_mode=ModelMode.REASONER, output_dir=tmp_path / "b")

    data_batch = {
        "caption": ["p1", "p2"],
        "reasoner_images": [PIL.new("RGB", (8, 8)), None],
    }
    with pytest.raises(ValueError, match="mixes image-conditioned and text-only"):
        pipe._generate_reasoner_batch([sa1, sa2], data_batch)


@pytest.mark.L0
def test_reasoner_build_rejects_empty_prompt() -> None:
    from cosmos_framework.inference.args import ModelMode, OmniSampleOverrides, SampleMeta, VisionMode

    overrides = OmniSampleOverrides(prompt="   ")
    meta = SampleMeta(model_mode=ModelMode.REASONER, vision_mode=VisionMode.IMAGE, condition_vision_mode=None)
    with pytest.raises(ValueError, match="non-empty 'prompt'"):
        overrides._build_reasoner_data(model_config=None, sample_meta=meta)


@pytest.mark.L0
def test_reasoner_defaults_validate_against_overrides() -> None:
    """The defaults JSON must validate against ``OmniSampleOverrides`` so
    ``build_sample`` cannot silently drop a field after an upstream rename."""
    from cosmos_framework.inference.args import OmniSampleOverrides, _load_modality_defaults

    defaults = _load_modality_defaults("reasoner")
    filtered = {k: v for k, v in defaults.items() if k in OmniSampleOverrides.model_fields}
    assert set(defaults) - set(filtered) == set(), f"defaults has unknown fields: {set(defaults) - set(filtered)}"
    OmniSampleOverrides.model_validate(filtered)


@pytest.mark.parametrize("resolution,target_size", [("256", (256, 320)), ("480", (480, 640))])
@pytest.mark.parametrize("viewpoint", ["ego_view", "concat_view"])
def test_action_inference_matches_camera_preparation_and_training_metadata(
    monkeypatch, resolution, target_size, viewpoint
):
    import torch

    from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline
    from cosmos_framework.data.generator.action.video_view import VideoViewConfig
    from cosmos_framework.inference import action
    from cosmos_framework.inference.args import ModelMode
    from cosmos_framework.utils.generator.data_utils import get_vision_data_resolution

    view = VideoViewConfig(cameras={role: role for role in ("head", "left", "right")}, resolution=resolution)
    cameras = {
        "head": torch.full((2, 3, 96, 128), 17, dtype=torch.uint8),
        "left": torch.full((2, 3, 48, 64), 81, dtype=torch.uint8),
        "right": torch.full((2, 3, 48, 64), 191, dtype=torch.uint8),
    }
    source_video = cameras["head"].permute(1, 0, 2, 3)
    if viewpoint == "concat_view":
        source_video = torch.cat(
            [source_video, torch.cat([cameras["left"], cameras["right"]], dim=-1).permute(1, 0, 2, 3)], dim=-2
        )
    monkeypatch.setattr(action, "read_media_frames", lambda *args, **kwargs: (source_video, 30.0))
    batch = action.get_action_sample_data(
        SimpleNamespace(input_video_key="video"),
        batch_size=2,
        prompt="Pick up the cup.",
        vision_path=Path("observation.png"),
        model_mode=ModelMode.POLICY,
        action_path=None,
        domain_name="agibotworld",
        view_point=viewpoint,
        resolution=resolution,
        action_chunk_size=4,
        max_action_dim=55,
        fps=30,
        device="cpu",
    )
    prepared = view.compose(cameras, 2, viewpoint=viewpoint)
    prepared = torch.cat([prepared, prepared[:, -1:].repeat(1, 3, 1, 1)], dim=1)
    training = ActionTransformPipeline(max_action_dim=55, format_prompt_as_json=True)(
        {
            "video": prepared,
            "action": torch.zeros(4, 55),
            "mode": "policy",
            "ai_caption": "Pick up the cup.",
            "viewpoint": viewpoint,
            "additional_view_description": view.describe(viewpoint=viewpoint),
            "conditioning_fps": torch.tensor(30),
        }
    )
    h, w = target_size
    final_h = h if viewpoint == "ego_view" else h + h // 2
    for video in batch["video"]:
        assert video[0].shape == (3, 5, final_h, w)
        assert video[0].dtype == torch.uint8
        assert torch.equal(video[0], training["video"])
    assert batch["image_size"].tolist() == [[final_h, w, final_h, w]] * 2
    assert batch["image_size"][0].tolist() == training["image_size"].tolist()
    assert json.loads(batch["ai_caption"][0]) == training["ai_caption"]
    assert get_vision_data_resolution(tuple(batch["image_size"][0, :2].int().tolist())) == (
        get_vision_data_resolution(tuple(training["image_size"][:2].int().tolist()))
    )


@pytest.mark.parametrize("viewpoint,height,width", [("ego_view", 480, 640), ("concat_view", 720, 640)])
def test_action_prepared_media_passes_through_without_interpolation(monkeypatch, viewpoint, height, width):
    import torch

    from cosmos_framework.inference import action
    from cosmos_framework.inference.args import ModelMode

    frames = torch.randint(256, (3, 5, height, width), dtype=torch.uint8)
    monkeypatch.setattr(action, "read_media_frames", lambda *args, **kwargs: (frames, 30.0))

    def forbidden(*args, **kwargs):
        pytest.fail("Prepared action media must pass through without camera preparation")

    monkeypatch.setattr(action.VideoViewConfig, "compose", forbidden)
    batch = action.get_action_sample_data(
        SimpleNamespace(input_video_key="video"),
        batch_size=1,
        prompt="Pick up the cup.",
        vision_path=Path("prepared.png"),
        model_mode=ModelMode.POLICY,
        action_path=None,
        domain_name="agibotworld",
        view_point=viewpoint,
        resolution="480",
        action_video_prepared=True,
        action_chunk_size=4,
        max_action_dim=55,
        fps=30,
        device="cpu",
    )
    assert batch["video"][0][0] is frames
    assert batch["image_size"].tolist() == [[height, width, height, width]]


@pytest.mark.parametrize(
    "viewpoint,height,width,prepared,error",
    [
        ("concat_view", 721, 640, False, "concat input"),
        ("concat_view", 720, 641, False, "concat input"),
        ("ego_view", 240, 320, True, "Prepared action input"),
        ("concat_view", 360, 320, True, "Prepared action input"),
        ("unknown", 480, 640, False, "viewpoint"),
    ],
)
def test_action_media_rejects_ambiguous_layout_or_invalid_prepared_size(
    monkeypatch, viewpoint, height, width, prepared, error
):
    import torch

    from cosmos_framework.inference import action
    from cosmos_framework.inference.args import ModelMode

    monkeypatch.setattr(
        action, "read_media_frames", lambda *args, **kwargs: (torch.zeros(3, 1, height, width, dtype=torch.uint8), 30)
    )
    with pytest.raises(ValueError, match=error):
        action.get_action_sample_data(
            SimpleNamespace(input_video_key="video"),
            batch_size=1,
            prompt="Pick up the cup.",
            vision_path=Path("observation.png"),
            model_mode=ModelMode.POLICY,
            action_path=None,
            domain_name="agibotworld",
            view_point=viewpoint,
            resolution="480",
            action_video_prepared=prepared,
            action_chunk_size=4,
            max_action_dim=55,
            fps=30,
            device="cpu",
        )


@pytest.mark.parametrize("num_frames", [1, 5, 7])
def test_action_batch_only_aligns_time_and_preserves_spatial_pixels(num_frames):
    import torch

    from cosmos_framework.inference.action import build_action_batch
    from cosmos_framework.inference.args import ModelMode

    frames = torch.arange(num_frames, dtype=torch.uint8)[None, :, None, None].expand(3, num_frames, 17, 23)
    batch = build_action_batch(
        video=frames,
        action=torch.zeros(4, 55),
        raw_action_dim=29,
        prompt="Pick up the cup.",
        view_point="ego_view",
        domain_name="agibotworld",
        model_mode=ModelMode.POLICY,
        action_chunk_size=4,
        fps=30,
        input_video_key="video",
        device="cpu",
    )
    video = batch["video"][0][0]
    assert video.shape == (3, 5, 17, 23)
    assert video[0, :, 0, 0].tolist() == [min(index, num_frames - 1) for index in range(5)]
    assert batch["image_size"].tolist() == [[17, 23, 17, 23]]
