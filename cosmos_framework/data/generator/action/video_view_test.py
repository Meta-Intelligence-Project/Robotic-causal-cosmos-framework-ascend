# SPDX-License-Identifier: OpenMDW-1.1
"""Camera ordering, missing-camera policy and synchronized query checks."""

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.action.video_view import VideoViewConfig


@pytest.fixture
def small_resolution(monkeypatch):
    # Keep ordering/identity tests small; bucket tests below use the real action table.
    monkeypatch.setattr(
        "cosmos_framework.data.generator.action.video_view.ACTION_RES_SIZE_INFO",
        {"256": {"odd": (11, 8), "even": (10, 8)}},
    )


def test_layout_preserves_camera_order_and_odd_width(small_resolution):
    view = VideoViewConfig(cameras={"head": "head", "left": "left", "right": "right"})
    frames = {
        key: torch.full((33, 3, 8, 11), float(i)) for i, key in enumerate(view.camera_keys(viewpoint="concat_view"))
    }
    result = view.compose(frames, 33, viewpoint="concat_view")
    assert result.shape == (3, 33, 12, 11)
    assert (result[:, :, :8] == 0).all()
    assert (result[:, :, 8:, :5] == 1).all()
    assert (result[:, :, 8:, 5:] == 2).all()


def test_ego_keeps_pixels_and_frame_order(small_resolution):
    view = VideoViewConfig(cameras={"head": "head"})
    frames = torch.rand(65, 3, 8, 10)
    torch.testing.assert_close(view.compose({"head": frames}, 65, viewpoint="ego_view"), frames.permute(1, 0, 2, 3))


def test_missing_camera_or_frame_is_an_error():
    view = VideoViewConfig(cameras={"head": "head"})
    with pytest.raises(ValueError, match="Required video camera"):
        view.validate_features({}, viewpoint="ego_view")
    with pytest.raises(ValueError, match="shape"):
        view.compose({"head": torch.zeros(32, 3, 8, 8)}, 33, viewpoint="ego_view")


def test_only_selected_cameras_receive_same_timestamps(tmp_path, monkeypatch):
    from lerobot.datasets import video_utils

    view = VideoViewConfig(cameras={"head": "head", "left": "left", "right": "right"})
    meta = SimpleNamespace(root=tmp_path, get_video_file_path=lambda episode, key: f"{key}.mp4")
    calls = []

    def query(metadata, times, episode, *, tolerance_s, video_backend):
        assert metadata is meta and tolerance_s == 1e-4 and video_backend == "pyav"
        calls.append((times, episode))
        return {key: torch.zeros(len(ts), 3, 8, 8) for key, ts in times.items()}

    monkeypatch.setattr(video_utils, "query_videos", query)
    with pytest.raises(FileNotFoundError, match="Required camera head"):
        view.read(meta, 7, [0.5, 0.6], viewpoint="concat_view", tolerance_s=1e-4, video_backend="pyav")
    for key in view.camera_keys(viewpoint="concat_view"):
        (tmp_path / f"{key}.mp4").touch()
    view.read(meta, 7, [0.5, 0.6], viewpoint="concat_view", tolerance_s=1e-4, video_backend="pyav")
    assert calls == [({key: [0.5, 0.6] for key in view.camera_keys(viewpoint="concat_view")}, 7)]


def test_role_mapping_and_ego_camera_selection(small_resolution):
    view = VideoViewConfig(cameras={"head": "robot.front", "left": "robot.wrist"})
    assert view.camera_keys(viewpoint="ego_view") == ("robot.front",)
    frames = torch.rand(2, 3, 8, 10)
    torch.testing.assert_close(
        view.compose({"robot.front": frames}, 2, viewpoint="ego_view"), frames.permute(1, 0, 2, 3)
    )


def test_unsupported_layout_and_missing_role_are_rejected():
    view = VideoViewConfig(cameras={"head": "front"})
    with pytest.raises(ValueError, match="Unsupported viewpoint"):
        view.camera_keys(viewpoint="two_view")
    with pytest.raises(ValueError, match="left"):
        view.camera_keys(viewpoint="concat_view")


def test_same_camera_config_supports_different_sample_layouts(small_resolution):
    view = VideoViewConfig(cameras={"head": "front", "left": "wrist_l", "right": "wrist_r"})
    frames = {key: torch.rand(2, 3, 8, 10) for key in view.cameras.values()}
    assert view.compose(frames, 2, viewpoint="ego_view").shape == (3, 2, 8, 10)
    assert view.compose(frames, 2, viewpoint="concat_view").shape == (3, 2, 12, 10)
    assert "wrist" not in view.describe(viewpoint="ego_view")
    assert "wrist" in view.describe(viewpoint="concat_view")
    assert set(vars(view)) == {"cameras", "resolution"}


@pytest.mark.parametrize("resolution,target_size", [(256, (256, 320)), ("480", (480, 640))])
@pytest.mark.parametrize("viewpoint", ["ego_view", "concat_view"])
@pytest.mark.parametrize(
    "names,shape", [(["height", "width", "rgb"], [8, 11, 3]), (["rgb", "height", "width"], [3, 8, 11])]
)
def test_resized_layout_offsets_and_instance_isolation(
    tmp_path, monkeypatch, viewpoint, names, shape, resolution, target_size
):
    from lerobot.datasets import video_utils

    view = VideoViewConfig(cameras={"head": "head", "left": "left", "right": "right"}, resolution=resolution)
    offsets = {"head": 1000.0, "left": 2000.0, "right": 3000.0}
    meta = SimpleNamespace(
        root=tmp_path,
        features={"head": {"names": names, "shape": shape}},
        episodes=[{f"videos/{key}/from_timestamp": value for key, value in offsets.items()}],
        get_video_file_path=lambda episode, key: f"{key}.mp4",
    )
    for key in offsets:
        (tmp_path / f"{key}.mp4").touch()
    calls = []

    def decode(path, timestamps, tolerance, *, backend, resize_h, resize_w):
        key = path.stem
        calls.append((key, timestamps, tolerance, backend, resize_h, resize_w))
        return torch.full((len(timestamps), 3, resize_h, resize_w), list(offsets).index(key), dtype=torch.uint8)

    monkeypatch.setattr(video_utils, "decode_video_frames", decode)
    times = [0.0, 1 / 30, 2 / 30]
    legacy_calls = []

    def query(metadata, timestamps, episode, *, tolerance_s, video_backend):
        assert metadata is meta and tolerance_s == 1e-4 and video_backend == "pyav"
        legacy_calls.append((timestamps, episode))
        return {key: torch.full((3, 3, 8, 11), 0.5) for key in timestamps}

    monkeypatch.setattr(video_utils, "query_videos", query)
    before = view.read(meta, 0, times, viewpoint=viewpoint, tolerance_s=1e-4, video_backend="pyav")
    with monkeypatch.context() as patch:

        def forbidden(*args, **kwargs):
            pytest.fail("Direct PyAV resize must not interpolate the decoded tensors again")

        patch.setattr("cosmos_framework.data.generator.action.video_view.F.interpolate", forbidden)
        result = view.read(meta, 0, times, viewpoint=viewpoint, tolerance_s=1e-4, video_backend="pyav_resize")
    after = view.read(meta, 0, times, viewpoint=viewpoint, tolerance_s=1e-4, video_backend="pyav")
    assert torch.equal(before, after) and len(legacy_calls) == 2
    assert result.dtype == torch.uint8
    h, w = target_size
    assert result.shape == (3, 3, h if viewpoint == "ego_view" else h + h // 2, w)
    assert before.shape == result.shape
    assert (result[:, :, :h] == 0).all()
    if viewpoint == "concat_view":
        assert (result[:, :, h:, : w // 2] == 1).all()
        assert (result[:, :, h:, w // 2 :] == 2).all()
    for key, shifted, tolerance, backend, height, width in calls:
        assert shifted == [offsets[key] + t for t in times]
        assert tolerance == 1e-4 and backend == "pyav_resize"
        assert (height, width) == {"head": (h, w), "left": (h // 2, w // 2), "right": (h // 2, w - w // 2)}[key]
    assert [call[0] for call in calls] == list(view.camera_roles(viewpoint))


@pytest.mark.parametrize("height,width", [(None, None), (8, None), (0, 8), (8, -1), (True, 8)])
def test_resize_backend_requires_explicit_positive_dimensions(height, width):
    from lerobot.datasets.video_utils import decode_video_frames

    with pytest.raises(ValueError, match="positive integer"):
        decode_video_frames("unused.mp4", [0.0], 1e-4, "pyav_resize", height, width)


@pytest.mark.parametrize("backend", [None, "pyav", "video_reader", "torchcodec", "pyav_resize"])
@pytest.mark.parametrize("resized", [False, True])
def test_decoder_dispatch_preserves_legacy_routes(monkeypatch, backend, resized):
    from lerobot.datasets import video_utils

    calls = []
    for name in ("torchvision", "torchcodec", "pyav_resized"):

        def record(*args, _name=name, **kwargs):
            calls.append((_name, args, kwargs))
            return _name

        monkeypatch.setattr(video_utils, f"decode_video_frames_{name}", record)
    monkeypatch.setattr(video_utils, "get_safe_default_codec", lambda: "pyav")
    h, w = (8, 10) if resized else (None, None)
    if backend == "pyav_resize" and not resized:
        with pytest.raises(ValueError):
            video_utils.decode_video_frames("clip", [1.0], 1e-4, backend, h, w)
        assert not calls
        return
    expected = "torchcodec" if backend == "torchcodec" else "pyav_resized" if resized else "torchvision"
    assert video_utils.decode_video_frames("clip", [1.0], 1e-4, backend, h, w) == expected
    assert len(calls) == 1
    assert calls[0][1][:3] == ("clip", [1.0], 1e-4)


def test_resize_backend_real_video_preserves_frame_selection(tmp_path):
    import av
    import numpy as np

    from lerobot.datasets.video_utils import FrameTimestampError, decode_video_frames

    path = tmp_path / "frames.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = 16, 16, "yuv420p"
        for index in range(12):
            frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), index * 20, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    times = [7 / 30, 2 / 30, 7 / 30, 11 / 30]
    original = decode_video_frames(path, times, 1e-4, "pyav")
    resized = decode_video_frames(path, times, 1e-4, "pyav_resize", 16, 16)
    assert torch.equal((original * 255).to(torch.uint8), resized)
    # Existing video-generation resize route must keep its pixels and uint8 contract.
    existing = decode_video_frames(path, times, 1e-4, "pyav", 8, 8)
    assert torch.equal(existing, decode_video_frames(path, times, 1e-4, "pyav_resize", 8, 8))
    for backend in ("pyav", "pyav_resize"):
        with pytest.raises(FrameTimestampError):
            decode_video_frames(
                path,
                [0.015],
                1e-4,
                backend,
                8 if backend == "pyav_resize" else None,
                8 if backend == "pyav_resize" else None,
            )


@pytest.mark.parametrize("backend", ["pyav", "pyav_resize"])
def test_new_layout_only_changes_declaration(tmp_path, monkeypatch, backend, small_resolution):
    from cosmos_framework.data.generator.action import video_view as module
    from lerobot.datasets import video_utils

    # Wrist row first: only a layout declaration changes, not either backend.
    layout = module._VideoLayout(
        rows=(("left", "right"), ("head",)),
        resize_targets=module._concat_resize_targets,
    )
    monkeypatch.setattr(module, "_get_layout", lambda viewpoint: layout)
    view = VideoViewConfig(cameras={"head": "front", "left": "wrist", "right": "wrist"})
    for key in ("front", "wrist"):
        (tmp_path / f"{key}.mp4").touch()
    meta = SimpleNamespace(
        root=tmp_path,
        features={"front": {"names": ["height", "width", "rgb"], "shape": [8, 11, 3]}},
        episodes=[{"videos/front/from_timestamp": 0, "videos/wrist/from_timestamp": 1}],
        get_video_file_path=lambda episode, key: f"{key}.mp4",
    )
    decode_calls, query_calls = [], []

    def query(metadata, timestamps, episode, *, tolerance_s, video_backend):
        assert metadata is meta and tolerance_s == 1e-4 and video_backend == "pyav"
        query_calls.append(timestamps)
        # Metadata is irrelevant to the legacy path; actual head dimensions win.
        return {key: torch.full((2, 3, 8, 11), 0.25 if key == "front" else 0.75) for key in timestamps}

    def decode(path, timestamps, tolerance, *, backend, resize_h, resize_w):
        decode_calls.append((path.stem, resize_h, resize_w))
        return torch.full((2, 3, resize_h, resize_w), 64 if path.stem == "front" else 192, dtype=torch.uint8)

    monkeypatch.setattr(video_utils, "decode_video_frames", decode)
    monkeypatch.setattr(video_utils, "query_videos", query)
    if backend == "pyav":
        meta.features = {}  # Legacy must not start depending on head metadata.
    else:

        def forbidden(*args, **kwargs):
            pytest.fail("PyAV preparation and composition must not call legacy resize")

        monkeypatch.setattr(module.F, "interpolate", forbidden)
    result = view.read(meta, 0, [0, 1 / 30], viewpoint="wrists_above_head", tolerance_s=1e-4, video_backend=backend)
    assert result.shape == (3, 2, 12, 11)
    assert (result[:, :, :4] == (0.75 if backend == "pyav" else 192)).all()
    assert (result[:, :, 4:] == (0.25 if backend == "pyav" else 64)).all()
    if backend == "pyav":
        assert len(query_calls) == 1 and set(query_calls[0]) == {"front", "wrist"}
    else:
        # Same camera field, different sizes: do not reuse one resized tensor for both wrists.
        assert decode_calls == [("wrist", 4, 5), ("wrist", 4, 6), ("front", 8, 11)]


@pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
def test_pure_composition_never_resizes_or_casts(monkeypatch, dtype):
    from cosmos_framework.data.generator.action import video_view as module

    def forbidden(*args, **kwargs):
        pytest.fail("Pure composition must not resize")

    monkeypatch.setattr(module.F, "interpolate", forbidden)
    frames = {
        "head": torch.full((2, 3, 8, 11), 2, dtype=dtype),
        "left": torch.full((2, 3, 4, 5), 3, dtype=dtype),
        "right": torch.full((2, 3, 4, 6), 4, dtype=dtype),
    }
    output = module._compose_prepared(frames, 2, module._get_layout("concat_view"))
    assert output.dtype == dtype
    assert torch.equal(output[:, :, :8], frames["head"].permute(1, 0, 2, 3))
    assert torch.equal(output[:, :, 8:, :5], frames["left"].permute(1, 0, 2, 3))
    assert torch.equal(output[:, :, 8:, 5:], frames["right"].permute(1, 0, 2, 3))


def test_prepared_rgb_cameras_keep_pixels_without_resize(monkeypatch, small_resolution):
    def forbidden(*args, **kwargs):
        pytest.fail("RGB cameras at their targets must not be interpolated")

    monkeypatch.setattr("cosmos_framework.data.generator.action.video_view.F.interpolate", forbidden)
    view = VideoViewConfig(cameras={role: role for role in ("head", "left", "right")})
    frames = {"head": torch.rand(2, 3, 8, 11), "left": torch.rand(2, 3, 4, 5), "right": torch.rand(2, 3, 4, 6)}
    result = view.compose(frames, 2, viewpoint="concat_view")
    assert torch.equal(result[:, :, :8], frames["head"].permute(1, 0, 2, 3))
    assert torch.equal(result[:, :, 8:, :5], frames["left"].permute(1, 0, 2, 3))
    assert torch.equal(result[:, :, 8:, 5:], frames["right"].permute(1, 0, 2, 3))


@pytest.mark.parametrize("error", ["missing", "count", "channels", "height", "width", "zero", "dtype"])
def test_prepared_layout_rejects_invalid_inputs(error):
    from cosmos_framework.data.generator.action import video_view as module

    frames = {"head": torch.zeros(2, 3, 8, 11), "left": torch.zeros(2, 3, 4, 5), "right": torch.zeros(2, 3, 4, 6)}
    if error == "missing":
        del frames["left"]
    elif error == "dtype":
        frames["left"] = frames["left"].to(torch.uint8)
    else:
        shape = {
            "count": (1, 3, 4, 5),
            "channels": (2, 1, 4, 5),
            "height": (2, 3, 3, 5),
            "width": (2, 3, 4, 4),
            "zero": (2, 3, 0, 5),
        }[error]
        frames["left"] = torch.zeros(shape)
    with pytest.raises(TypeError if error == "dtype" else ValueError):
        module._compose_prepared(frames, 2, module._get_layout("concat_view"))


@pytest.mark.parametrize("dtype", [torch.float32, torch.uint8])
@pytest.mark.parametrize(
    "resolution,source_size,target_size",
    [
        ("256", (240, 240), (256, 256)),
        (256, (240, 320), (256, 320)),
        ("256", (320, 240), (320, 256)),
        ("256", (480, 600), (256, 320)),
        ("256", (600, 480), (320, 256)),
        ("256", (360, 640), (256, 448)),
        ("256", (640, 360), (448, 256)),
        (480, (240, 240), (512, 512)),
        ("480", (480, 640), (480, 640)),
        ("480", (640, 480), (640, 480)),
        ("480", (360, 640), (352, 640)),
        ("480", (640, 360), (640, 352)),
    ],
)
def test_action_resolution_resizes_cameras_before_composition(dtype, resolution, source_size, target_size):
    view = VideoViewConfig(cameras={"head": "front", "left": "wrist_l", "right": "wrist_r"}, resolution=resolution)
    frames = {
        "front": torch.full((1, 3, *source_size), 10, dtype=dtype),
        "wrist_l": torch.full((1, 3, 32, 48), 20, dtype=dtype),
        "wrist_r": torch.full((1, 3, 64, 32), 30, dtype=dtype),
    }
    h, w = target_size
    ego = view.compose({"front": frames["front"]}, 1, viewpoint="ego_view")
    concat = view.compose(frames, 1, viewpoint="concat_view")
    assert ego.shape == (3, 1, h, w) and concat.shape == (3, 1, h + h // 2, w)
    assert ego.dtype == concat.dtype == dtype
    torch.testing.assert_close(concat[:, :, :h], ego)
    torch.testing.assert_close(concat[:, :, h:, : w // 2], torch.full((3, 1, h // 2, w // 2), 20, dtype=dtype))
    torch.testing.assert_close(concat[:, :, h:, w // 2 :], torch.full((3, 1, h // 2, w // 2), 30, dtype=dtype))
    assert frames["front"].shape[-2:] == source_size


def test_action_resolution_does_not_resize_cameras_already_at_target(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Cameras already at their targets must retain their pixels")

    monkeypatch.setattr("cosmos_framework.data.generator.action.video_view.F.interpolate", forbidden)
    view = VideoViewConfig(cameras={role: role for role in ("head", "left", "right")}, resolution="256")
    frames = {
        "head": torch.randint(256, (1, 3, 256, 320), dtype=torch.uint8),
        "left": torch.randint(256, (1, 3, 128, 160), dtype=torch.uint8),
        "right": torch.randint(256, (1, 3, 128, 160), dtype=torch.uint8),
    }
    result = view.compose(frames, 1, viewpoint="concat_view")
    assert torch.equal(result[:, :, :256], frames["head"].permute(1, 0, 2, 3))
    assert torch.equal(result[:, :, 256:, :160], frames["left"].permute(1, 0, 2, 3))
    assert torch.equal(result[:, :, 256:, 160:], frames["right"].permute(1, 0, 2, 3))


@pytest.mark.parametrize("resolution", [None, 720, "unknown"])
def test_action_resolution_rejects_unsupported_tiers(resolution):
    with pytest.raises(ValueError, match="resolution"):
        VideoViewConfig(cameras={"head": "front"}, resolution=resolution)
