# SPDX-License-Identifier: OpenMDW-1.1
"""Explicit streaming windows through the real Wan 2.2 encoder and caches."""

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16 import (
    Wan2pt2VAEInterface,
    WanVAE,
    WanVAE_,
)


@pytest.fixture(autouse=True, scope="module")
def _small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _tokenizer(dtype=torch.float32, *, temporal_window=4):
    torch.manual_seed(73)
    network = WanVAE_(dim=4, dec_dim=4, z_dim=2, num_res_blocks=1, temporal_window=temporal_window)
    network.eval().requires_grad_(False).to(dtype=dtype)
    wrapper = WanVAE.__new__(WanVAE)
    wrapper.model = network
    wrapper.dtype = dtype
    wrapper.device = torch.device("cpu")
    wrapper.scale = (
        torch.tensor([-0.2, 0.3], dtype=dtype),
        torch.tensor([1.1, 0.7], dtype=dtype),
    )
    tokenizer = Wan2pt2VAEInterface.__new__(Wan2pt2VAEInterface)
    tokenizer.model = wrapper
    return tokenizer


def _video(frames, *, seed=19):
    return torch.rand((1, 3, frames, 16, 16), generator=torch.Generator().manual_seed(seed)) * 2 - 1


@pytest.mark.parametrize("frames", [1, 5, 9, 17, 25])
@pytest.mark.parametrize("vae_dtype", [torch.float32, torch.bfloat16])
def test_stream_matches_offline_with_same_four_frame_chunks(frames, vae_dtype):
    tokenizer = _tokenizer(vae_dtype)
    video = _video(frames)
    expected = tokenizer.encode(video)
    state = tokenizer.stream_encode_start(video[:, :, :1])
    assert state.latents.shape == (1, 2, 1, 1, 1)
    assert all(cache is None or not cache.requires_grad for cache in state.feat_cache)
    if frames > 1:
        new = tokenizer.stream_encode_step(state, video[:, :, 1:])
        assert new.shape[2] == (frames - 1) // 4
        torch.testing.assert_close(new, expected[:, :, 1:], rtol=0, atol=0)
    assert state.frame_count == frames
    actual = tokenizer.stream_encode_finish(state)
    assert actual.dtype == video.dtype
    assert not actual.requires_grad
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert state.closed and not state.feat_cache and not state.latent_chunks


def test_interleaved_sessions_and_ordinary_encode_do_not_share_caches():
    tokenizer = _tokenizer()
    a, b = _video(17), _video(9, seed=29)
    expected_a, expected_b = tokenizer.encode(a), tokenizer.encode(b)
    sa = tokenizer.stream_encode_start(a[:, :, :1])
    sb = tokenizer.stream_encode_start(b[:, :, :1])
    tokenizer.stream_encode_step(sa, a[:, :, 1:5])
    prefix = sa.latents.clone()
    tokenizer.stream_encode_step(sb, b[:, :, 1:])
    torch.testing.assert_close(tokenizer.encode(b), expected_b, rtol=0, atol=0)
    torch.testing.assert_close(sa.latents, prefix, rtol=0, atol=0)
    tokenizer.stream_encode_step(sa, a[:, :, 5:])
    torch.testing.assert_close(tokenizer.stream_encode_finish(sa), expected_a, rtol=0, atol=0)
    torch.testing.assert_close(tokenizer.stream_encode_finish(sb), expected_b, rtol=0, atol=0)


def test_future_frames_cannot_change_an_already_encoded_prefix():
    tokenizer = _tokenizer()
    video = _video(17)
    changed = video.clone()
    changed[:, :, 9:] = -changed[:, :, 9:]
    outputs = []
    for source in (video, changed):
        state = tokenizer.stream_encode_start(source[:, :, :1])
        tokenizer.stream_encode_step(state, source[:, :, 1:9])
        prefix = state.latents.clone()
        tokenizer.stream_encode_step(state, source[:, :, 9:])
        result = tokenizer.stream_encode_finish(state)
        torch.testing.assert_close(result[:, :, :3], prefix, rtol=0, atol=0)
        outputs.append(result)
    torch.testing.assert_close(outputs[0][:, :, :3], outputs[1][:, :, :3], rtol=0, atol=0)
    assert not torch.equal(outputs[0][:, :, 3:], outputs[1][:, :, 3:])


def test_stream_always_uses_one_then_four_real_frames_without_padding(monkeypatch):
    tokenizer = _tokenizer(temporal_window=24)
    video = _video(17).transpose(3, 4)
    reference = _tokenizer()
    expected = reference.encode(video)
    calls = []
    original = tokenizer.model.model._encode_chunk_impl

    def observe(chunk, cache, scale):
        calls.append((chunk.shape[2], chunk.is_contiguous()))
        return original(chunk, cache, scale)

    monkeypatch.setattr(tokenizer.model.model, "_encode_chunk_impl", observe)
    state = tokenizer.stream_encode_start(video[:, :, :1])
    tokenizer.stream_encode_step(state, video[:, :, 1:9])
    tokenizer.stream_encode_step(state, video[:, :, 9:])
    actual = tokenizer.stream_encode_finish(state)
    assert calls == [(1, True), (4, True), (4, True), (4, True), (4, True)]
    assert tokenizer.model.model.temporal_window == 24
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_matching_aot_chunk_dispatch_and_missing_shape_eager_fallback(monkeypatch):
    tokenizer = _tokenizer()
    video = _video(9)
    expected = tokenizer.encode(video)
    network = tokenizer.model.model
    calls = []

    def compiled_prime(chunk, cache):
        calls.append((chunk.shape[2], cache[0]))
        return network._encode_chunk_impl(chunk, cache, tokenizer.model.scale)

    monkeypatch.setattr(network, "_aot_chunk_fns", {(1, 8, 8, 0): compiled_prime}, raising=False)
    state = tokenizer.stream_encode_start(video[:, :, :1])
    tokenizer.stream_encode_step(state, video[:, :, 1:])
    torch.testing.assert_close(tokenizer.stream_encode_finish(state), expected, rtol=0, atol=0)
    assert calls == [(1, None)]


@pytest.mark.parametrize("exact_duration", [False, True])
def test_ordinary_encode_retains_configured_chunks_padding_and_trimming(monkeypatch, exact_duration):
    tokenizer = _tokenizer(temporal_window=24)
    network = tokenizer.model.model
    if exact_duration:
        network._encode_exact_durations = {9}
    original = network._encode_chunk_impl
    calls = []

    def observe(chunk, cache, scale):
        calls.append(chunk.shape[2])
        if chunk.shape[2] == 24:
            assert not chunk[:, :, 8:].any()
        return original(chunk, cache, scale)

    monkeypatch.setattr(network, "_encode_chunk_impl", observe)
    latent = tokenizer.encode(_video(9))
    assert calls == [1, 8 if exact_duration else 24]
    assert latent.shape == (1, 2, 3, 1, 1)


@pytest.mark.parametrize(
    "invalid",
    [torch.zeros(1, 3, 5, 16, 16), torch.zeros(1, 3, 1, 17, 16), torch.zeros(1, 3, 1, 16, 16, dtype=torch.uint8)],
)
def test_start_rejects_invalid_prime_input(invalid):
    with pytest.raises(ValueError):
        _tokenizer().stream_encode_start(invalid)


def test_session_input_contract_finish_abort_and_reset():
    tokenizer = _tokenizer()
    video = _video(9)
    state = tokenizer.stream_encode_start(video[:, :, :1])
    for invalid in (video[:, :, 1:3], video[:, :, 1:5].double(), video[:, :, 1:5].repeat(2, 1, 1, 1, 1)):
        with pytest.raises(ValueError):
            tokenizer.stream_encode_step(state, invalid)
    assert not state.closed and state.frame_count == 1
    with pytest.raises(ValueError, match="another tokenizer"):
        _tokenizer().stream_encode_step(state, video[:, :, 1:5])
    tokenizer.stream_encode_abort(state)
    tokenizer.stream_encode_abort(state)
    assert state.closed and not state.feat_cache and not state.latent_chunks
    with pytest.raises(ValueError, match="closed"):
        tokenizer.stream_encode_step(state, video[:, :, 1:5])
    with pytest.raises(ValueError, match="closed"):
        tokenizer.stream_encode_finish(state)
    fresh = tokenizer.stream_encode_reset(state, video[:, :, :1])
    tokenizer.stream_encode_step(fresh, video[:, :, 1:])
    torch.testing.assert_close(tokenizer.stream_encode_finish(fresh), tokenizer.encode(video), rtol=0, atol=0)


def test_encoder_failure_aborts_without_retaining_cache_or_partial_latents(monkeypatch):
    tokenizer = _tokenizer()
    video = _video(9)
    state = tokenizer.stream_encode_start(video[:, :, :1])
    original = tokenizer.model.model._run_encode_chunk
    calls = 0

    def fail_second_chunk(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("encoder failure")
        return original(*args)

    monkeypatch.setattr(tokenizer.model.model, "_run_encode_chunk", fail_second_chunk)
    with pytest.raises(RuntimeError, match="encoder failure"):
        tokenizer.stream_encode_step(state, video[:, :, 1:])
    assert state.closed and not state.feat_cache and not state.latent_chunks


def test_model_changes_during_session_are_rejected():
    tokenizer = _tokenizer()
    video = _video(5)
    state = tokenizer.stream_encode_start(video[:, :, :1])
    tokenizer.model = SimpleNamespace(model=tokenizer.model.model, dtype=torch.float64)
    with pytest.raises(ValueError, match="model or dtype changed"):
        tokenizer.stream_encode_step(state, video[:, :, 1:])
    tokenizer.stream_encode_abort(state)
