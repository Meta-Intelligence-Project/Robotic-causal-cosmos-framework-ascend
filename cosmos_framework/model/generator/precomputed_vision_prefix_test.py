# SPDX-License-Identifier: OpenMDW-1.1
"""Precomputed observation latents bypass the VAE and retain ordinary crops."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_framework.data.generator.action.history_window import build_history_window
from cosmos_framework.inference.causal_action.window_streaming import ObservationWindowEncoder
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.model.generator.tokenizers.wan2pt2_stream_encode_test import _tokenizer
from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import Wan2pt2VAEInterface


@pytest.fixture(autouse=True, scope="module")
def _small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class _Tokenizer:
    is_causal = True
    spatial_compression_factor = 16
    latent_ch = 2

    @staticmethod
    def get_latent_num_frames(frames):
        return 1 + (frames - 1) // 4


class _Model(OmniMoTModel):
    def __init__(self):
        nn.Module.__init__(self)
        self.tokenizer_vision_gen = _Tokenizer()
        self.tokenizer_sound_gen = None
        self.input_video_key, self.input_image_key = "video", "images"
        self.tensor_kwargs_fp32 = self.tensor_kwargs = dict(device="cpu", dtype=torch.float32)
        self.config = SimpleNamespace(
            diffusion_expert_config=SimpleNamespace(vision_temporal_position_mode="latent_index")
        )
        self.log_enc_time_every_n = 100

    def encode(self, video):
        raise AssertionError("The precomputed-prefix path must not call the VAE")


def _raw():
    return torch.zeros(1, 3, 17, 32, 48)


def _prefix():
    return torch.linspace(-2, 3, 36).reshape(2, 3, 2, 3)


@pytest.mark.parametrize("batched", [False, True])
def test_scatter_preserves_normalized_prefix_and_zero_fills_only_the_future(batched):
    raw = _raw() if batched else _raw().squeeze(0)
    prefix = _prefix().requires_grad_(True)
    with torch.no_grad():
        result = _Model()._scatter_precomputed_vision_prefixes([raw], [prefix], [[0, 1, 2]])[0]
    expected_shape = (1, 2, 5, 2, 3) if batched else (2, 5, 2, 3)
    assert result.shape == expected_shape and result.dtype == raw.dtype and result.device == raw.device
    result = result.squeeze(0) if batched else result
    torch.testing.assert_close(result[:, :3], prefix, rtol=0, atol=0)
    assert not result[:, 3:].any() and not result.requires_grad


def test_get_data_bypasses_vae_and_crops_after_scattering_without_renormalizing_prefix():
    model = _Model()
    prefix = _prefix()
    batch = dict(
        video=[torch.zeros(3, 17, 32, 48, dtype=torch.uint8)],
        causal_action_precomputed_vision_prefix=[prefix],
        image_size=[torch.tensor([32, 48, 16, 32])],
        conditioning_fps=torch.tensor([7.5]),
    )
    with torch.no_grad():
        clean = model.get_data_and_condition(batch, vision_condition_indexes=[[0, 1, 2]])
    latent = clean.x0_tokens_vision[0]
    assert clean.raw_state_vision[0].shape == (1, 3, 17, 32, 48)
    assert (clean.raw_state_vision[0] == -1).all()
    assert latent.shape == (1, 2, 5, 1, 2) and latent.is_contiguous()
    torch.testing.assert_close(latent[0, :, :3], prefix[:, :, :1, :2], rtol=0, atol=0)
    assert not latent[:, :, 3:].any()
    assert clean.batch_size == 1 and not clean.is_image_batch
    assert clean.temporal_positions_vision is None


@pytest.mark.parametrize(
    "prefix",
    [
        torch.zeros(3, 3, 2, 3),
        torch.zeros(2, 2, 2, 3),
        torch.zeros(2, 3, 1, 3),
        torch.zeros(2, 3, 2, 2),
        torch.zeros(1, 2, 3, 2, 3),
        torch.zeros(2, 3, 2, 3, dtype=torch.int64),
        torch.full((2, 3, 2, 3), float("nan")),
        torch.empty(2, 3, 2, 3, device="meta"),
    ],
)
def test_invalid_prefix_shape_dtype_values_or_device_is_rejected(prefix):
    with torch.no_grad(), pytest.raises(ValueError):
        _Model()._scatter_precomputed_vision_prefixes([_raw()], [prefix], [[0, 1, 2]])


@pytest.mark.parametrize("indexes", [[], [0, 2], [1, 2, 3], [0, 1, 1], [-1, 0]])
def test_conditioning_must_be_an_exact_prefix_from_frame_zero(indexes):
    with torch.no_grad(), pytest.raises(ValueError, match="contiguous"):
        _Model()._scatter_precomputed_vision_prefixes([_raw()], [_prefix()], [indexes])


def test_prefix_cannot_extend_past_the_video_latent_horizon():
    with torch.no_grad(), pytest.raises(ValueError, match="shape"):
        _Model()._scatter_precomputed_vision_prefixes([_raw()], [torch.zeros(2, 6, 2, 3)], [list(range(6))])


@pytest.mark.parametrize("mode", ["grad", "training_without_conditioning", "multi_item", "multiview", "noncausal"])
def test_training_and_unsupported_layouts_cannot_use_precomputed_prefixes(mode):
    model = _Model()
    indexes = None if mode == "training_without_conditioning" else [[0, 1, 2]]
    if mode == "noncausal":
        model.tokenizer_vision_gen.is_causal = False
    kwargs = dict(
        num_vision_items_per_sample=[1] if mode == "multi_item" else None,
        num_views_per_vision_item=[2] if mode == "multiview" else None,
    )
    with torch.set_grad_enabled(mode == "grad"), pytest.raises(ValueError):
        model._scatter_precomputed_vision_prefixes([_raw()], [_prefix()], indexes, **kwargs)


@pytest.mark.parametrize("prefixes,indexes", [([], [[0, 1, 2]]), ([_prefix()], []), (_prefix(), [[0, 1, 2]])])
def test_prefix_batch_counts_must_match_video_items(prefixes, indexes):
    with torch.no_grad(), pytest.raises(ValueError, match="per sample"):
        _Model()._scatter_precomputed_vision_prefixes([_raw()], prefixes, indexes)


class _TinyTokenizer(Wan2pt2VAEInterface):
    @property
    def latent_ch(self):
        return self.model.model.z_dim


def _fixed_model(dtype=torch.bfloat16):
    tokenizer = _TinyTokenizer.__new__(_TinyTokenizer)
    tokenizer.model = _tokenizer(dtype, temporal_window=24).model
    tokenizer._spatial_compression_factor = 16
    tokenizer._temporal_compression_factor = 4
    tokenizer._causal = True
    model = _Model()
    model.tokenizer_vision_gen = tokenizer
    return model


def _fixed_plan():
    return SimpleNamespace(causal_action_metadata=SimpleNamespace(include_visual_prefix=True))


def _track_sessions(monkeypatch, tokenizer):
    sessions = []
    original = tokenizer.stream_encode_start

    def start(frames):
        state = original(frames)
        sessions.append(state)
        return state

    monkeypatch.setattr(tokenizer, "stream_encode_start", start)
    return sessions


@pytest.mark.parametrize("history_blocks", [1, 32])
def test_real_window_manager_matches_canonical_inference_with_same_bf16_weights(monkeypatch, history_blocks):
    model = _fixed_model()
    sessions = _track_sessions(monkeypatch, model.tokenizer_vision_gen)
    manager = ObservationWindowEncoder(model, history_blocks=history_blocks, history_padding="repeat_first")
    anchor = 17
    images = {
        tick: torch.randint(256, (3, 32, 32), generator=torch.Generator().manual_seed(21 + tick), dtype=torch.uint8)
        for tick in range(anchor + 1)
    }
    manager.add(0, images[0])
    manager.prepare(anchor)
    for tick in range(1, anchor + 1):
        manager.add(tick, images[tick])
    prepared = manager.count
    streamed_prefix, timing = manager.finish(anchor)
    window = build_history_window(anchor, history_blocks=history_blocks, history_padding="repeat_first")
    pixels = torch.stack([images[tick] for tick in window.observation_ticks], dim=1)
    pixels = torch.cat((pixels, torch.zeros(3, 8, 32, 32, dtype=torch.uint8)), dim=1)
    raw = pixels.unsqueeze(0).float() / 127.5 - 1
    observed_latents = 1 + history_blocks * 2
    canonical = model._encode_fixed_window_vision([raw], [_fixed_plan()], [list(range(observed_latents))])[0]
    torch.testing.assert_close(canonical[0, :, :observed_latents], streamed_prefix, rtol=0, atol=0)
    assert canonical.shape == (1, 2, observed_latents + 2, 2, 2)
    assert not canonical[:, :, observed_latents:].any()
    assert timing["sampled_frames_encoded_before_predict"] == prepared == 1 + history_blocks * 8
    assert manager.state is None and all(session.closed for session in sessions)
    assert all(not session.feat_cache and not session.latent_chunks for session in sessions)


def test_fixed_window_training_encodes_future_but_never_changes_observed_latents(monkeypatch):
    model = _fixed_model()
    sessions = _track_sessions(monkeypatch, model.tokenizer_vision_gen)
    pixels = torch.rand((1, 3, 17, 32, 32), generator=torch.Generator().manual_seed(81)) * 2 - 1
    changed = pixels.clone()
    changed[:, :, 9:] = -changed[:, :, 9:]
    training = model._encode_fixed_window_vision([pixels], [_fixed_plan()], None)[0]
    changed_training = model._encode_fixed_window_vision([changed], [_fixed_plan()], None)[0]
    inference = model._encode_fixed_window_vision([pixels], [_fixed_plan()], [[0, 1, 2]])[0]
    torch.testing.assert_close(training[:, :, :3], changed_training[:, :, :3], rtol=0, atol=0)
    torch.testing.assert_close(training[:, :, :3], inference[:, :, :3], rtol=0, atol=0)
    assert not torch.equal(training[:, :, 3:], changed_training[:, :, 3:])
    assert training[:, :, 3:].abs().sum() > 0 and not inference[:, :, 3:].any()
    assert [session.frame_count for session in sessions] == [17, 17, 9]
    assert all(session.closed for session in sessions)


@pytest.mark.parametrize("inference", [False, True])
def test_get_data_fixed_window_uses_canonical_sessions_and_closes_them(monkeypatch, inference):
    model = _fixed_model(torch.float32)
    sessions = _track_sessions(monkeypatch, model.tokenizer_vision_gen)
    pixels = torch.randint(256, (3, 17, 32, 32), generator=torch.Generator().manual_seed(31), dtype=torch.uint8)
    batch = dict(video=[pixels], sequence_plan=[_fixed_plan()], image_size=[torch.tensor([32, 32, 16, 32])])
    indexes = [[0, 1, 2]] if inference else None
    with torch.set_grad_enabled(not inference):
        clean = model.get_data_and_condition(batch, vision_condition_indexes=indexes)
    assert len(sessions) == 1 and sessions[0].closed
    assert sessions[0].frame_count == (9 if inference else 17)
    assert not sessions[0].feat_cache and not sessions[0].latent_chunks
    latent = clean.x0_tokens_vision[0]
    assert latent.shape == (1, 2, 5, 1, 2) and latent.dtype == torch.float32
    assert not latent.requires_grad and latent[:, :, :3].abs().sum() > 0
    if inference:
        assert not latent[:, :, 3:].any()
    else:
        assert latent[:, :, 3:].abs().sum() > 0
