# SPDX-License-Identifier: OpenMDW-1.1
"""Full windows from long episodes; one minimally padded window from each short episode."""

import json
import pickle
import warnings
from itertools import islice
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.action.block_state import BlockStatistics, Quantiles, build_block_sample
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionSFTDataset
from cosmos_framework.data.generator.action.datasets.causal_action_dataset_group import CausalActionDatasetGroup
from cosmos_framework.data.generator.action.datasets.causal_action_mixture import CausalActionMixture
from cosmos_framework.data.generator.action.datasets.causal_action_sft_dataset import CausalActionSFTDataset
from cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot import split_episode_ids
from cosmos_framework.data.generator.action.datasets.segment_lerobot_dataset_test import (
    full_range_packer,
    make_reader,
)
from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline


@pytest.mark.parametrize("offset", [-2, 0, 1, 3])
def test_candidate_indexes_cover_legal_starts_and_episode_boundaries(monkeypatch, offset):
    lengths = (0, 1, 2, 34, 96, 97, 98, 131)
    reader, queries = make_reader(monkeypatch, lengths=lengths, offset=offset, maximum=96, window_mode="frame_window")
    expected, blocks = [], []
    for episode in split_episode_ids(len(lengths), 0, 0.0, "full"):
        count, begin = lengths[episode], sum(lengths[:episode])
        candidates = []
        for start in range(max(0, count - 1)):
            if start + offset < 0 or start + offset >= count:
                continue
            actions = min(96, count - start - 1, count - start - offset)
            if actions > 0:
                candidates.append((0, episode, begin + start, actions))
        full_windows = [candidate for candidate in candidates if candidate[-1] == 96]
        candidates = full_windows or candidates[:1]
        if candidates:
            blocks.append((len(expected), len(candidates)))
            expected.extend(candidates)
    assert len(reader) == len(expected)
    assert [reader._resolve_index(i) for i in range(len(reader))] == expected
    assert reader._resolve_index(-1) == expected[-1]
    assert reader._resolve_index(-len(reader)) == expected[0]
    assert reader.get_shuffle_blocks() == blocks
    assert reader.skipped_window_episodes == len(lengths) - len(blocks)
    assert reader._segments == reader._episode_records == reader._range_bounds == []
    assert len(reader._frame_ranges) == len(blocks)
    assert not queries  # Index construction does not read trajectory rows.
    for index in (-len(reader) - 1, len(reader)):
        with pytest.raises(IndexError):
            reader[index]


@pytest.mark.parametrize("backend", ["hf", "parquet"])
@pytest.mark.parametrize("early", [False, True])
@pytest.mark.parametrize("offset", [-2, 0, 1, 3])
@pytest.mark.parametrize("available", [1, 31, 32, 33, 95, 96, 110])
def test_full_windows_and_short_episodes_read_real_rows_and_only_pad_one_block(
    monkeypatch, tmp_path, backend, early, offset, available
):
    start = max(0, -offset) + (5 if available >= 96 else 0)
    frames = start + available + max(1, offset)
    reader, queries = make_reader(
        monkeypatch,
        lengths=(frames,),
        root=tmp_path,
        table_backend=backend,
        offset=offset,
        maximum=96,
        window_mode="frame_window",
        timestamp_start=1000.0,
    )
    reader.early_video_sampling = early
    requested = []

    def video(meta, episode, times, *, viewpoint):
        requested.extend(times)
        return torch.zeros(3, len(times), 2, 2)

    monkeypatch.setattr(reader, "_read_video", video)
    raw = reader[start - max(0, -offset)]
    real = min(available, 96)
    out = ((real + 31) // 32) * 32
    assert raw["real_action_steps"] == real
    assert raw["action_target"].shape == (out, 55)
    assert raw["state_trajectory"].shape == (out + 1, 55)
    assert "full_range" not in raw
    torch.testing.assert_close(raw["state_trajectory"][: real + 1, 0], torch.arange(start, start + real + 1).float())
    torch.testing.assert_close(
        raw["action_target"][:real, 0], torch.arange(start + offset, start + offset + real).float() + 1000
    )
    torch.testing.assert_close(raw["state_trajectory"][real:, 0], torch.full((out - real + 1,), float(start + real)))
    torch.testing.assert_close(
        raw["action_target"][real - 1 :, 0], torch.full((out - real + 1,), float(start + offset + real - 1 + 1000))
    )
    assert raw["state_timestamps"].dtype == torch.float64
    torch.testing.assert_close(raw["state_timestamps"], 1000 + torch.arange(start, start + out + 1).double() / 30)
    indexes = torch.arange(0, out + 1, 4 if early else 1)
    expected_video_times = 1000 + (start + indexes.clamp(max=real)).double() / 30
    torch.testing.assert_close(torch.tensor(requested, dtype=torch.float64), expected_video_times, atol=1e-12, rtol=0)
    sample, metadata = build_block_sample(raw, template=reader.template, planner=reader.planner, history_blocks=3)
    assert metadata.real_action_steps == real
    assert len(metadata.states) == out // 32
    sampled_real = torch.arange(0, out + 1, 4) <= real
    expected_latents = torch.cat((sampled_real[:1], sampled_real[1:].reshape(-1, 4).any(dim=1)))
    assert metadata.num_valid_video_latents == int(expected_latents.sum())
    assert sample["video"].shape[1] == out // 4 + 1
    if backend == "hf":
        assert {"task_index": [start]} in queries


@pytest.mark.parametrize("backend", ["hf", "parquet"])
@pytest.mark.parametrize("from_state", [False, True])
@pytest.mark.parametrize("offset", [-2, 0, 1, 3])
@pytest.mark.parametrize("frames", [40, 137])
def test_frame_window_target_offset_is_applied_once(monkeypatch, tmp_path, backend, from_state, offset, frames):
    reader, _ = make_reader(
        monkeypatch,
        lengths=(frames,),
        root=tmp_path,
        table_backend=backend,
        from_state=from_state,
        offset=offset,
        maximum=96,
        window_mode="frame_window",
    )
    index = 3 if frames == 137 else 0
    _, _, start, real = reader._resolve_index(index)
    raw = reader[index]
    expected = torch.arange(start + offset, start + offset + real).float() + (0 if from_state else 1000)
    torch.testing.assert_close(raw["action_target"][:real, 0], expected)


@pytest.mark.parametrize("backend", ["hf", "parquet"])
@pytest.mark.parametrize("use_subtask", [False, True])
def test_caption_uses_original_start_annotation_and_start_row_task(monkeypatch, tmp_path, backend, use_subtask):
    bounds = [(2, 10, "first"), (10, 15, "second"), (20, 25, "third")]
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/episodes.jsonl").write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {
                    "episode_index": 0,
                    "action_config": [
                        {"start_frame": lo, "end_frame": hi, "action_text": text} for lo, hi, text in bounds
                    ],
                },
                {"episode_index": 1, "action_config": None},
            ]
        )
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reader, _ = make_reader(
            monkeypatch,
            lengths=(140, 8, 7),
            root=tmp_path,
            table_backend=backend,
            window_mode="frame_window",
            maximum=96,
            use_subtask=use_subtask,
            task_indexes=[(i // 8) % 2 for i in range(155)],
        )
    assert len([w for w in caught if "falling back to task text" in str(w.message)]) == (2 if use_subtask else 0)
    assert len(reader) == 44 + 1 + 1
    episode0_indexes = {}
    for idx in range(len(reader)):
        _, episode, start, real = reader._resolve_index(idx)
        local = start if episode == 0 else start - (140 if episode == 1 else 148)
        if episode == 0:
            episode0_indexes[local] = idx
        expected = None
        if use_subtask and episode == 0:
            expected = next((text for lo, hi, text in bounds if lo <= local < hi), None)
        if expected is None:
            expected = "second task" if (start // 8) % 2 else "test task"
        raw = reader[idx]
        assert raw["ai_caption"] == expected
        assert raw["real_action_steps"] == real
    # A window starting in an unannotated gap retains task text even when it crosses annotations.
    assert reader[episode0_indexes[0]]["ai_caption"] == "test task"
    assert reader[episode0_indexes[0]]["real_action_steps"] == 96
    if use_subtask:
        assert reader[episode0_indexes[2]]["ai_caption"] == "first"
        assert reader[episode0_indexes[10]]["ai_caption"] == "second"
        assert [(r.start, r.stop) for r in reader._subtask_ranges[0]] == [(2, 10), (10, 15), (20, 25)]
    else:
        assert not reader._subtask_ranges


def test_index_storage_grows_by_episodes_not_candidate_frames(monkeypatch):
    sizes = []
    for frames in (100, 10000):
        reader, queries = make_reader(monkeypatch, lengths=(frames, 50), maximum=96, window_mode="frame_window")
        assert len(reader) == frames - 95  # frames-96 full windows, plus one short-episode fallback.
        assert len(reader._frame_ranges) == len(reader._episode_cum_ends) == 2
        assert not reader._segments and not reader._episode_records and not queries
        sizes.append(len(pickle.dumps((reader._frame_ranges, reader._episode_cum_ends))))
    assert sizes[1] - sizes[0] < 16


def test_split_keeps_only_selected_episode_indexes_and_annotations(monkeypatch, tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/episodes.jsonl").write_text(
        "\n".join(
            json.dumps(
                dict(episode_index=i, action_config=[dict(start_frame=2, end_frame=5, action_text=f"episode {i}")])
            )
            for i in range(4)
        )
    )
    seen = []
    for split in ("train", "val"):
        reader, _ = make_reader(
            monkeypatch,
            lengths=(40, 50, 100, 130),
            root=tmp_path,
            maximum=96,
            window_mode="frame_window",
            use_subtask=True,
            split=split,
            split_val_ratio=0.5,
        )
        selected = split_episode_ids(4, 0, 0.5, split)
        assert [r[1] for r in reader._frame_ranges] == selected
        assert set(reader._subtask_ranges) == set(selected)
        assert len(reader) == sum((1, 1, 4, 34)[i] for i in selected)
        seen.append(set(selected))
    assert not seen[0] & seen[1] and seen[0] | seen[1] == set(range(4))


@pytest.mark.parametrize("mode,maximum", [("unknown", 96), ("frame_window", -1)])
def test_reader_rejects_invalid_window_modes(monkeypatch, mode, maximum):
    with pytest.raises(ValueError, match="window_mode|positive max_action_steps"):
        make_reader(monkeypatch, window_mode=mode, maximum=maximum)


def test_real_readers_keep_global_candidate_numbering_in_group_and_uniform_mixture(monkeypatch, tmp_path):
    from cosmos_framework.data.generator.action.datasets import causal_action_mixture as mixture_module

    # Parquet readers retain their own on-disk tables even after metadata construction for other roots.
    readers = [
        make_reader(
            monkeypatch,
            lengths=(frames,),
            root=tmp_path / str(i),
            table_backend="parquet",
            maximum=96,
            window_mode="frame_window",
        )[0]
        for i, frames in enumerate((99, 102, 3))
    ]
    groups = [CausalActionDatasetGroup(readers[:2]), CausalActionDatasetGroup(readers[2:])]
    assert groups[0].get_shuffle_blocks() == [(0, 3), (3, 6)]
    assert groups[1].get_shuffle_blocks() == [(0, 1)]
    draws = [0, 2, 3, 8, 9, 3]
    selected = iter(draws)
    monkeypatch.setattr(
        mixture_module.random, "Random", lambda seed: SimpleNamespace(randrange=lambda n: next(selected))
    )
    mixture = CausalActionMixture(groups, weights=[99, 1], sampling_strategy="sample_uniform")
    assert mixture._source_cum_ends == [9, 10]
    assert len(mixture) == 10
    samples = list(islice(mixture, len(draws)))
    expected = [(0, 0), (0, 2), (1, 0), (1, 5), (2, 0), (1, 0)]
    for sample, (root, start) in zip(samples, expected, strict=True):
        assert sample["source"] == str(tmp_path / str(root))
        assert sample["state_trajectory"][0, 0] == start


def frame_window_adapter(reader, *, mode="policy"):
    raw = reader[0]
    statistics = BlockStatistics(
        Quantiles(torch.full((55,), -10000.0), torch.full((55,), 10000.0), raw["state_mask"]),
        Quantiles(torch.full((55,), -10000.0), torch.full((55,), 10000.0), raw["action_mask"]),
    )
    transform = ActionTransformPipeline(
        max_action_dim=55,
        video_temporal_downsample=4,
        append_viewpoint_info=False,
        append_duration_fps_timestamps=False,
        append_resolution_info=False,
    )
    return CausalActionSFTDataset(
        ActionSFTDataset(reader, transform),
        mode=mode,
        statistics=statistics,
        history_blocks_min=3,
        history_blocks_max=3,
    )


@pytest.mark.parametrize("mode", ["policy", "forward_dynamics", "inverse_dynamics"])
def test_transform_collate_and_lookahead_preserve_actual_window_lengths(monkeypatch, tmp_path, mode):
    reader, _ = make_reader(
        monkeypatch,
        lengths=(11, 34, 111),
        root=tmp_path,
        table_backend="parquet",
        maximum=96,
        window_mode="frame_window",
    )
    monkeypatch.setattr(reader, "_read_video", lambda meta, episode, times, **kw: torch.zeros(3, len(times), 64, 64))
    adapter = frame_window_adapter(reader, mode=mode)
    by_episode = {
        record[1]: first for (first, _), record in zip(reader.get_shuffle_blocks(), reader._frame_ranges, strict=True)
    }
    samples = [adapter[by_episode[i]] for i in (0, 1, 2)]  # Short episodes 10/33, plus a full 96-action window.
    assert [len(s["action"]) for s in samples] == [32, 64, 96]
    assert [s["video"].shape[1] for s in samples] == [9, 17, 25]
    assert [s["sequence_plan"].causal_action_metadata.real_action_steps for s in samples] == [10, 33, 96]
    assert [s["sequence_plan"].causal_action_metadata.history_blocks for s in samples] == [3, 3, 3]
    assert all(s["sequence_plan"].causal_action_metadata.full_range is None for s in samples)
    loader = full_range_packer([*samples, samples[0]], None)
    costs = [
        loader._compute_num_tokens_per_sample(
            {"video": [s["video"]], "action": [s["action"]], "sequence_plan": [s["sequence_plan"]]}
        )
        for s in samples
    ]
    assert costs[0] < costs[1] < costs[2]
    loader.max_sequence_length = sum(costs[:2]) + 1
    batches = list(loader)
    # The two short windows fit together; the long lookahead sample is retained for the next batch.
    assert [[len(a[0]) for a in b["action"]] for b in batches] == [[32, 64], [96], [32]]
    assert batches[1]["action"][0][0] is samples[2]["action"]


@pytest.mark.parametrize("mode", ["policy", "forward_dynamics", "inverse_dynamics"])
def test_mixed_frame_windows_keep_all_past_history_and_sample_isolation(monkeypatch, tmp_path, mode):
    from cosmos_framework.data.generator.sequence_packing import pack_input_sequence
    from cosmos_framework.data.generator.sequence_packing.causal_action import dense_action_mask, expand_action_sequence
    from cosmos_framework.model.generator.omni_mot_causal_action_model import OmniMoTCausalActionModel
    from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean

    reader, _ = make_reader(
        monkeypatch,
        lengths=(11, 34, 111),
        root=tmp_path,
        table_backend="parquet",
        maximum=96,
        window_mode="frame_window",
    )
    monkeypatch.setattr(reader, "_read_video", lambda meta, episode, times, **kw: torch.zeros(3, len(times), 64, 64))
    adapter = frame_window_adapter(reader, mode=mode)
    by_episode = {
        record[1]: first for (first, _), record in zip(reader.get_shuffle_blocks(), reader._frame_ranges, strict=True)
    }
    samples = [adapter[by_episode[i]] for i in (0, 1, 2)]
    plans = [s["sequence_plan"] for s in samples]
    frames = [3, 5, 7]
    model = OmniMoTCausalActionModel.__new__(OmniMoTCausalActionModel)
    geometry = model.prepare_teacher_forcing_geometry_from_plans(frames, plans)
    videos = [torch.zeros(1, 2, t, 1, 1) for t in frames]
    clean = GenerationDataClean(
        batch_size=3,
        is_image_batch=False,
        x0_tokens_vision=videos,
        x0_tokens_action=[s["action"] for s in samples],
        fps_vision=torch.stack([s["conditioning_fps"] for s in samples]),
        fps_action=torch.stack([s["conditioning_fps_action"] for s in samples]),
    )
    packed = pack_input_sequence(
        plans,
        [[11, 12], [21, 22], [31, 32]],
        clean,
        torch.tensor([0.3, 0.5, 0.7]),
        dict(bos_token_id=1, eos_token_id=2, start_of_generation=3, end_of_generation=4),
        enable_fps_modulation=True,
        action_dim=55,
    )
    packed.causal_action_metadata = [p.causal_action_metadata for p in plans]
    expanded = expand_action_sequence(packed, videos, geometry)
    layout = expanded.teacher_forcing.layout
    assert geometry.block_sizes == (2, 2, 2)
    assert geometry.history_blocks == (3, 3, 3)
    assert [len(m.states) for m in expanded.causal_action_metadata] == [1, 2, 3]
    visible = dense_action_mask(layout)
    queries = layout.gen_query_indexes
    same_sample = layout.sample_ids[queries, None] == layout.sample_ids[None, :]
    assert not visible[~same_sample].any()
    past_visual = (
        same_sample
        & (layout.stream_ids[None, :] == 0)
        & (layout.block_ids[None, :] < layout.block_ids[queries, None])
        & (layout.stream_ids[queries, None] != 3)
    )
    assert past_visual.any() and visible[past_visual].all()
    future = layout.block_ids[None, :] > layout.block_ids[queries, None]
    assert not visible[future.expand_as(visible)].any()
