# SPDX-License-Identifier: OpenMDW-1.1
"""Reader boundary and alignment checks without video/model dependencies."""

import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from datasets import Dataset

from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate55
from cosmos_framework.data.generator.action.causal_block_geometry import CausalBlockGeometry
from cosmos_framework.data.generator.action.datasets import segment_lerobot_dataset as module
from cosmos_framework.data.generator.action.sample_contract import ActionReadOptions
from cosmos_framework.data.generator.action.segment_planner import SegmentPlanner


def make_reader(
    monkeypatch,
    *,
    lengths=(100, 50),
    offset=0,
    from_state=False,
    root="synthetic",
    use_subtask=False,
    split="full",
    maximum=64,
    overlap=16,
    table_backend="hf",
    timestamp_start=0.0,
    window_mode="segmented",
    task_indexes=None,
    split_val_ratio=0.0,
):
    template = ActionStateTemplate55()
    size = sum(lengths)
    state = torch.arange(size, dtype=torch.float32)[:, None].expand(-1, template.width).clone()
    mask = torch.zeros(template.width, dtype=torch.bool)
    mask[template.fields["left_arm_joint"]] = True
    columns = {
        "observation.state": state,
        "action": state + 1000,
        "mask_state": mask.expand(size, -1),
        "mask_action": mask.expand(size, -1),
        "timestamp": timestamp_start + torch.cat([torch.arange(n, dtype=torch.float64) / 30 for n in lengths]),
        "task_index": torch.zeros(size, dtype=torch.long) if task_indexes is None else torch.tensor(task_indexes),
    }
    episodes = []
    start = 0
    for n in lengths:
        episodes.append({"dataset_from_index": start, "dataset_to_index": start + n})
        start += n
    meta = SimpleNamespace(
        fps=30,
        features={key: {"shape": list(value.shape[1:])} for key, value in columns.items()},
        episodes=episodes,
        total_episodes=len(lengths),
        root=root,
        tasks=pd.DataFrame({"task_index": [0, 1]}, index=["test task", "second task"]),
    )
    if table_backend == "parquet":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        for episode in episodes:
            episode.update({"data/chunk_index": 0, "data/file_index": 0})
        meta.episodes = Dataset.from_list(episodes)
        meta.get_data_file_path = lambda episode_id: Path("data.parquet")
        for key, value in columns.items():
            meta.features[key] = {
                "dtype": "bool" if value.dtype == torch.bool else str(value.dtype).removeprefix("torch."),
                "shape": tuple(value.shape[1:]) or (1,),
            }
        pq.write_table(pa.table({key: value.tolist() for key, value in columns.items()}), root / "data.parquet")
    monkeypatch.setattr(module, "LeRobotDatasetMetadata", lambda **kwargs: meta)
    queries = []

    def query(indices):
        queries.append(indices)
        return {key: columns[key][rows] for key, rows in indices.items()}

    ds = SimpleNamespace(
        meta=meta,
        _query_hf_dataset=query,
        _ensure_hf_dataset_loaded=lambda: None,
        hf_dataset=Dataset.from_dict({"timestamp": columns["timestamp"].tolist()}),
    )
    monkeypatch.setattr(module.SegmentLeRobotDataset, "_get_dataset", lambda self, index: ds)
    reader = module.SegmentLeRobotDataset(
        root=root,
        table_backend=table_backend,
        template=template,
        planner=SegmentPlanner(
            max_action_steps=maximum,
            overlap_action_steps=0 if window_mode == "frame_window" else overlap,
            geometry=CausalBlockGeometry(temporal_compression_factor=4),
        ),
        read_options=ActionReadOptions(action_from_state=from_state, action_time_offset_steps=offset),
        use_subtask=use_subtask,
        window_mode=window_mode,
        split=split,
        split_val_ratio=split_val_ratio,
    )
    return reader, queries


def test_annotation_lookup_uses_original_half_open_ranges():
    reader = object.__new__(module.SegmentLeRobotDataset)
    annotations = [
        {"start_frame": lo, "end_frame": hi, "action_text": caption, "unused": {"large": "metadata"}}
        for lo, hi, caption in [(2, 10, "  first caption  "), (10, 15, "second"), (20, 25, "third")]
    ]
    ranges = reader._parse_subtask_annotations(annotations, 30)
    assert ranges == [
        module.ActionTrainingRange(2, 10, "  first caption  ", "0"),
        module.ActionTrainingRange(10, 15, "second", "1"),
        module.ActionTrainingRange(20, 25, "third", "2"),
    ]
    reader.read_options = ActionReadOptions(action_time_offset_steps=1)
    reader.planner = SegmentPlanner(
        max_action_steps=64, overlap_action_steps=16, geometry=CausalBlockGeometry(temporal_compression_factor=4)
    )
    # 切片可以向前、向后借帧，但不能扩大 caption 所属范围。
    assert reader._subtask_read_bounds(ranges[2], 100) == (20, 53)
    assert reader._subtask_read_bounds(ranges[2], 30) == (0, 30)
    expected_matches = {
        -1: None,
        0: None,
        1: None,
        2: ranges[0],
        9: ranges[0],
        10: ranges[1],
        14: ranges[1],
        15: None,
        19: None,
        20: ranges[2],
        24: ranges[2],
        25: None,
        29: None,
        30: None,
    }
    for frame, expected in expected_matches.items():
        assert reader._find_subtask_annotation(ranges, frame) is expected
        assert reader._find_subtask_annotation([], frame) is None
    # 读取范围计算不修改原始标注。
    assert [(r.start, r.stop) for r in ranges] == [(2, 10), (10, 15), (20, 25)]


def test_metadata_does_not_turn_invalid_annotations_into_an_episode_annotation(tmp_path):
    reader = object.__new__(module.SegmentLeRobotDataset)
    reader.root, reader.use_subtask = tmp_path, True
    reader.window_mode = "segmented"
    (tmp_path / "meta").mkdir()
    valid = {"start_frame": 2, "end_frame": 10, "action_text": "valid"}
    invalid = {"start_frame": 9, "end_frame": 15, "action_text": "overlap"}
    (tmp_path / "meta/episodes.jsonl").write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {"episode_index": 0, "action_config": [valid]},
                {"episode_index": 1, "action_config": [valid, invalid]},
            ]
        )
    )
    meta = SimpleNamespace(total_episodes=3, episodes=[{"dataset_from_index": 0, "dataset_to_index": 30}] * 3)
    with pytest.warns(UserWarning, match="Episode 1.*falling back to episode mode"):
        _, reader._subtask_ranges = reader._load_episode_metadata(tmp_path, meta)
    assert reader._subtask_ranges == {0: [module.ActionTrainingRange(2, 10, "valid", "0")], 1: []}
    reader.unannotated_observation_frames = 0
    assert list(reader._iter_training_ranges(1, 30)) == [module.ActionTrainingRange(0, 30)]
    with pytest.warns(UserWarning, match="Episode 2.*missing metadata"):
        assert list(reader._iter_training_ranges(2, 30)) == [module.ActionTrainingRange(0, 30)]
    assert reader.unannotated_observation_frames == 0  # 沿用旧 episode 回退计数。


@pytest.mark.parametrize("table_backend", ["hf", "parquet"])
@pytest.mark.parametrize("maximum", [64, -1])
@pytest.mark.parametrize("use_subtask", [False, True])
def test_annotation_refactor_preserves_sample_indexes_rows_and_captions(
    monkeypatch, tmp_path, table_backend, maximum, use_subtask
):
    captions = ["  lift cup  ", "place", "release"]
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "action_config": [
                    {"start_frame": lo, "end_frame": hi, "action_text": caption}
                    for (lo, hi), caption in zip([(1, 34), (45, 49), (70, 130)], captions)
                ],
            }
        )
    )
    if use_subtask:
        with pytest.warns(UserWarning, match="Episode 1.*missing metadata"):
            reader, _ = make_reader(
                monkeypatch,
                lengths=(131, 100),
                root=tmp_path,
                offset=1,
                use_subtask=True,
                maximum=maximum,
                table_backend=table_backend,
            )
        if maximum == -1:
            expected = [(0, 0, 1, 33), (0, 0, 45, 4), (0, 0, 70, 60), (0, 1, 131, 99)]
            bounds = [(1, 34, 1, 35), (45, 49, 45, 50), (70, 130, 70, 131)]
            blocks = [(0, 1), (1, 1), (2, 1), (3, 1)]
        else:
            expected = [(0, 0, 1, 64), (0, 0, 45, 32), (0, 0, 66, 64), (0, 1, 131, 64), (0, 1, 166, 64)]
            bounds = [(1, 34, 1, 66), (45, 49, 45, 78), (70, 130, 66, 131)]
            blocks = [(0, 1), (1, 1), (2, 1), (3, 2)]
        expected_captions = captions + ["test task"] * (len(expected) - 3)
        assert reader._range_bounds == bounds
        assert reader.unannotated_observation_frames == 34
    else:
        reader, _ = make_reader(
            monkeypatch, lengths=(131, 100), root=tmp_path, offset=1, maximum=maximum, table_backend=table_backend
        )
        if maximum == -1:
            expected = [(0, 0, 0, 130), (0, 1, 131, 99)]
            blocks = [(0, 1), (1, 1)]
        else:
            expected = [(0, 0, 0, 64), (0, 0, 48, 64), (0, 0, 66, 64), (0, 1, 131, 64), (0, 1, 166, 64)]
            blocks = [(0, 3), (3, 2)]
        expected_captions = ["test task"] * len(expected)
        assert reader._range_bounds == []
        assert reader.unannotated_observation_frames == 0
    assert reader._segments == expected
    assert len(reader) == len(expected)
    assert reader.get_shuffle_blocks() == blocks
    assert reader._subtask_ranges == {}  # 原始标注只在建索引时使用，随后释放。
    for index, (_, _, start, actions) in enumerate(expected):
        sample = reader[index]
        torch.testing.assert_close(
            sample["state_trajectory"][: actions + 1, 0], torch.arange(start, start + actions + 1).float()
        )
        torch.testing.assert_close(
            sample["action_target"][:actions, 0], torch.arange(start + 1, start + actions + 1).float() + 1000
        )
        assert sample["ai_caption"] == expected_captions[index]
        expected_caption = expected_captions[index] if use_subtask and index < 3 else None
        assert reader._caption_for_index(index) == expected_caption
    assert reader[-1]["ai_caption"] == expected_captions[-1]


@pytest.mark.parametrize("offset", [-2, 0, 1, 3])
@pytest.mark.parametrize("from_state", [False, True])
def test_subtask_extension_covers_boundaries_and_keeps_targets_aligned(monkeypatch, tmp_path, offset, from_state):
    annotations = [[(0, 33), (40, 177), (177, 181)], [(0, 65)]]
    meta = tmp_path / "meta"
    meta.mkdir()
    episodes = [
        {
            "episode_index": ep,
            "action_config": [
                {"start_frame": lo, "end_frame": hi, "action_text": f"{ep}/{i}"} for i, (lo, hi) in enumerate(ranges)
            ],
        }
        for ep, ranges in enumerate(annotations)
    ]
    (meta / "episodes.jsonl").write_text("\n".join(json.dumps(ep) for ep in episodes))
    reader, _ = make_reader(
        monkeypatch, lengths=(181, 65), offset=offset, from_state=from_state, root=tmp_path, use_subtask=True
    )
    for (begin_idx, size), (lo, hi, first, stop) in zip(reader.get_shuffle_blocks(), reader._range_bounds):
        covered = set()
        for idx in range(begin_idx, begin_idx + size):
            _, ep, start, actions = reader._segments[idx]
            begin, end = ((0, 181), (181, 246))[ep]
            covered.update(range(start - begin, start - begin + actions))
            assert begin <= start and start + actions < end
            assert begin <= start + offset and start + actions - 1 + offset < end
            sample = reader[idx]
            expected = torch.arange(start + offset, start + offset + actions).float()
            torch.testing.assert_close(sample["action_target"][:, 0], expected + (0 if from_state else 1000))
            torch.testing.assert_close(
                sample["state_trajectory"][:, 0], torch.arange(start, start + actions + 1).float()
            )
            assert sample["ai_caption"] == reader._caption_for_index(begin_idx)
        legal_end = min(end - begin - 1, end - begin - offset)
        assert set(range(max(lo, -offset, 0), min(hi, legal_end))) <= covered
        assert covered == set(range(first, stop - 1))
    assert reader.planner.discarded_action_steps == reader.planner.skipped_ranges == 0


@pytest.mark.parametrize(
    "bounds,length,expected",
    [
        ((0, 181), 602, (0, 193)),
        ((181, 421), 602, (181, 438)),
        ((421, 602), 602, (409, 602)),
        ((0, 33), 100, (0, 65)),  # 已整除也不能漏掉动作起点 32。
        ((90, 99), 100, (67, 100)),
        ((10, 11), 100, (10, 43)),
        ((0, 181), 181, (0, 181)),  # 整个 episode 不够向上补齐，交给 planner 保尾。
    ],
)
def test_subtask_borrows_forward_before_backward(bounds, length, expected):
    reader = object.__new__(module.SegmentLeRobotDataset)
    reader.read_options = ActionReadOptions(action_time_offset_steps=1)
    reader.planner = SegmentPlanner(
        max_action_steps=896, overlap_action_steps=16, geometry=CausalBlockGeometry(temporal_compression_factor=4)
    )
    assert reader._subtask_read_bounds(module.ActionTrainingRange(*bounds), length) == expected


@pytest.mark.parametrize("offset", [-2, -1, 0, 1, 2, 3])
@pytest.mark.parametrize("from_state", [False, True])
def test_targets_use_one_offset_and_stay_in_episode(monkeypatch, offset, from_state):
    reader, queries = make_reader(monkeypatch, lengths=(210, 170), offset=offset, from_state=from_state)
    for idx in range(len(reader)):
        _, ep, start, count = reader._segments[idx]
        sample = reader[idx]
        expected = torch.arange(start, start + count) + offset
        expected = expected + (0 if from_state else 1000)
        torch.testing.assert_close(sample["action_target"][:, 0], expected.float())
        assert sample["state_trajectory"].shape[0] == count + 1
        assert sample["conditioning_fps"] == 30
        begin, end = ((0, 210), (210, 380))[ep]
        assert start >= begin and start + count < end
        assert min(expected).item() - (0 if from_state else 1000) >= begin
        assert max(expected).item() - (0 if from_state else 1000) < end
    # 多次取样依然只查询一次数据集级 mask。
    assert sum("mask_state" in query for query in queries) == 1


def test_variable_lengths_and_shuffle_indexes(monkeypatch):
    reader, _ = make_reader(monkeypatch)
    assert sorted(reader[i]["action_target"].shape[0] for i in range(len(reader))) == [32, 64, 64]
    covered = [i for start, count in reader.get_shuffle_blocks() for i in range(start, start + count)]
    assert covered == list(range(len(reader)))
    assert reader[-1]["ai_caption"] == "test task"
    with pytest.raises(IndexError):
        reader[len(reader)]


def test_next_state_needs_only_endpoint_observation(monkeypatch):
    reader, _ = make_reader(monkeypatch, lengths=(33,), from_state=True, offset=1)
    assert len(reader) == 1
    assert reader[0]["action_target"][-1, 0] == 32


def test_short_episode_warns_and_is_skipped(monkeypatch):
    with pytest.warns(UserWarning, match="at least 33 required"):
        reader, _ = make_reader(monkeypatch, lengths=(32,))
    assert len(reader) == 0
    assert reader.planner.skipped_ranges == 1


def test_timestamp_precision_and_video_hook(monkeypatch):
    reader, _ = make_reader(monkeypatch, lengths=(65,))
    ds = reader._get_dataset(0)
    # 1000 秒附近 float32 相邻帧差分已不足以满足采样率校验。
    timestamps = (1000 + torch.arange(65, dtype=torch.float64) / 30).tolist()
    ds.hf_dataset = Dataset.from_dict({"timestamp": timestamps})
    received = []
    monkeypatch.setattr(reader, "_read_video", lambda ds, episode, times, *, viewpoint: received.append(times))
    sample = reader[0]
    assert sample["state_timestamps"].tolist() == timestamps
    assert received == [timestamps]


def test_result_uses_parent_video_format_and_keeps_absolute_targets(monkeypatch):
    reader, _ = make_reader(monkeypatch, lengths=(65,), from_state=True, offset=1)
    frames = torch.full((3, 65, 2, 2), 0.5)
    monkeypatch.setattr(reader, "_read_video", lambda ds, episode, times, *, viewpoint: frames)
    sample = reader[0]
    assert isinstance(sample, dict)
    assert sample["action"] is sample["action_target"]
    assert sample["video"].dtype == torch.uint8
    assert sample["video"].shape == frames.shape
    assert (sample["video"] == 127).all()
    torch.testing.assert_close(sample["action_target"], sample["state_trajectory"][1:])
    torch.testing.assert_close(sample["action_mask"], sample["state_mask"])
    assert sample["mode"] is None and sample["domain_id"] is None


def test_source_fps_uses_dense_array_with_missing_field_fallback(monkeypatch, tmp_path):
    path = tmp_path / "meta" / "episodes.jsonl"
    path.parent.mkdir()
    # 故意打乱 JSONL 顺序，必须按 episode_index 而非行号关联。
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"episode_index": 2},
                {"episode_index": 1, "source_fps": 29.97},
                {"episode_index": 0, "source_fps": 25},
            ]
        )
    )
    reader, _ = make_reader(monkeypatch, root=tmp_path, lengths=(100, 100, 100))
    assert reader._episode_fps.dtype == np.float32
    assert reader._episode_fps.nbytes == 3 * 4
    assert np.isnan(reader._episode_fps[2])
    assert not reader._episode_fps.flags.writeable
    path.unlink()  # getitem 只查数组，不再打开 JSONL。
    for index, (_, episode, _, _) in enumerate(reader._segments):
        sample = reader[index]
        assert sample["source_fps"].item() == pytest.approx({0: 25, 1: 29.97, 2: 30}[episode])
        assert sample["conditioning_fps"].item() == sample["source_fps"].item()
        assert sample["storage_fps"].item() == 30
    other, _ = make_reader(monkeypatch)  # 不同根目录独立读取。
    assert other[0]["source_fps"].item() == 30


def test_reader_rejects_nonuniform_storage_timestamps(monkeypatch):
    reader, _ = make_reader(monkeypatch, lengths=(65,))
    ds = reader._get_dataset(0)
    times = [i / 30 for i in range(65)]
    times[12] += 0.01
    ds.hf_dataset = Dataset.from_dict({"timestamp": times})
    with pytest.raises(ValueError, match="timestamp intervals must match meta.fps"):
        reader[0]


@pytest.mark.parametrize("fps", [0, -1, None, True, float("nan"), float("inf")])
def test_invalid_source_fps_does_not_fall_back(monkeypatch, tmp_path, fps):
    path = tmp_path / "meta" / "episodes.jsonl"
    path.parent.mkdir()
    path.write_text(json.dumps({"episode_index": 0, "source_fps": fps}))
    with pytest.raises(ValueError, match="Invalid source_fps"):
        make_reader(monkeypatch, root=tmp_path, lengths=(65,))


def test_resized_video_conversion_is_opt_in(monkeypatch):
    reader, _ = make_reader(monkeypatch, lengths=(33,))
    frames = torch.full((33, 3, 2, 2), 128, dtype=torch.uint8)
    with pytest.raises(TypeError, match="floating-point"):
        reader._convert_video(frames)
    reader._pyav_resize = True
    assert torch.equal(reader._convert_video(frames), frames.permute(1, 0, 2, 3))
    with pytest.raises(TypeError, match="uint8"):
        reader._convert_video(frames.float())
    with pytest.raises(ValueError, match="shape"):
        reader._convert_video(frames[0])
    assert reader._convert_video(None) is None
    reader._skip_video_loading = True
    assert reader._convert_video(frames) is None


@pytest.mark.parametrize(
    "bad_config",
    [
        None,
        [],
        {},
        "bad",
        [None],
        [{"start_frame": 0, "end_frame": 32}],
        [{"start_frame": True, "end_frame": 32, "action_text": "bad"}],
        [{"start_frame": -1, "end_frame": 32, "action_text": "bad"}],
        [{"start_frame": 20, "end_frame": 20, "action_text": "bad"}],
        [{"start_frame": 0, "end_frame": 101, "action_text": "bad"}],
        [{"start_frame": 0, "end_frame": 32, "action_text": " "}],
        [
            {"start_frame": 0, "end_frame": 40, "action_text": "valid first"},
            {"start_frame": 39, "end_frame": 65, "action_text": "overlap"},
        ],
    ],
)
@pytest.mark.parametrize("offset", [-2, 0, 1, 3])
def test_invalid_subtask_falls_back_only_affected_episode(monkeypatch, tmp_path, bad_config, offset):
    meta = tmp_path / "meta"
    meta.mkdir()
    first = {"episode_index": 0}
    if bad_config is not None:
        first["action_config"] = bad_config
    second = {
        "episode_index": 1,
        "action_config": [{"start_frame": 10, "end_frame": 42, "action_text": "valid subtask"}],
    }
    (meta / "episodes.jsonl").write_text("\n".join(json.dumps(x) for x in [first, second]))
    baseline, _ = make_reader(monkeypatch, lengths=(100, 100), root=tmp_path, offset=offset)
    with pytest.warns(UserWarning, match="Episode 0.*falling back to episode mode"):
        reader, _ = make_reader(monkeypatch, lengths=(100, 100), root=tmp_path, offset=offset, use_subtask=True)
    assert [s for s in reader._segments if s[1] == 0] == [s for s in baseline._segments if s[1] == 0]
    for index, segment in enumerate(reader._segments):
        sample = reader[index]
        if segment[1] == 0:
            assert sample["ai_caption"] == "test task"
            expected = baseline[index]
            for key in ["action_target", "state_trajectory", "state_timestamps", "action_timestamps"]:
                torch.testing.assert_close(sample[key], expected[key])
        else:
            assert sample["ai_caption"] == "valid subtask"
    assert len(reader._range_bounds) == 1


@pytest.mark.parametrize("metadata", [None, "", '{"episode_index": 0, "action_config": null}\n'])
def test_missing_subtasks_match_episode_mode(monkeypatch, tmp_path, metadata):
    if metadata is not None:
        (tmp_path / "meta").mkdir()
        (tmp_path / "meta/episodes.jsonl").write_text(metadata)
    baseline, _ = make_reader(monkeypatch, lengths=(50, 50), root=tmp_path)
    with pytest.warns(UserWarning, match="falling back to episode mode") as warnings:
        reader, _ = make_reader(monkeypatch, lengths=(50, 50), root=tmp_path, use_subtask=True)
    assert len(warnings) == 2
    assert reader._segments == baseline._segments
    assert reader._range_captions == baseline._range_captions
    assert reader.planner.discarded_action_steps == baseline.planner.discarded_action_steps


@pytest.mark.parametrize(
    "last_config",
    [
        [{"start_frame": 20, "end_frame": 52, "action_text": "last caption"}],
        None,
        [{"start_frame": 0, "end_frame": 200, "action_text": "invalid"}],
    ],
)
def test_duplicate_episode_logs_error_and_uses_last_record(monkeypatch, tmp_path, last_config, caplog):
    (tmp_path / "meta").mkdir()
    first = {"episode_index": 0, "action_config": [{"start_frame": 0, "end_frame": 32, "action_text": "first caption"}]}
    last = {"episode_index": 0, "action_config": last_config}
    path = tmp_path / "meta/episodes.jsonl"
    path.write_text(json.dumps(last))
    if last_config is not None and last_config[0]["end_frame"] <= 100:
        expected, _ = make_reader(monkeypatch, lengths=(100,), root=tmp_path, use_subtask=True)
    else:
        expected, _ = make_reader(monkeypatch, lengths=(100,), root=tmp_path)
    path.write_text("\n".join(json.dumps(row) for row in [first, last]))
    reader, _ = make_reader(monkeypatch, lengths=(100,), root=tmp_path, use_subtask=True)
    assert any(r.levelname == "ERROR" and "Duplicate episode 0" in r.message for r in caplog.records)
    assert reader._segments == expected._segments
    assert reader._range_captions == expected._range_captions
    for i in range(len(reader)):
        assert reader[i]["ai_caption"] == expected[i]["ai_caption"]


@pytest.mark.parametrize("table_backend", ["hf", "parquet"])
@pytest.mark.parametrize("frames,offset", [(2, 0), (17, 0), (33, 1), (34, 0), (98, -2), (98, 3), (961, 1)])
@pytest.mark.parametrize("from_state", [False, True])
def test_full_episode_retains_real_rows_and_repeats_tail(
    monkeypatch, tmp_path, table_backend, frames, offset, from_state
):
    from cosmos_framework.data.generator.action.block_state import build_block_sample

    reader, _ = make_reader(
        monkeypatch,
        lengths=(frames,),
        offset=offset,
        from_state=from_state,
        root=tmp_path,
        maximum=-1,
        overlap=10000,
        table_backend=table_backend,
        timestamp_start=1000.0,
    )
    first, stop = max(0, -offset), min(frames, frames + 1 - offset)
    real_actions = stop - first - 1
    padded_actions = ((real_actions + 31) // 32) * 32
    assert reader._segments == [(0, 0, first, real_actions)]
    assert reader.get_shuffle_blocks() == [(0, 1)]
    assert reader.chunk_length is None
    raw = reader[0]
    assert raw["full_range"] == (first, stop)
    assert raw["state_trajectory"].shape == (padded_actions + 1, reader.template.width)
    expected_states = torch.arange(first, stop).float()
    expected_targets = torch.arange(first + offset, first + offset + real_actions).float()
    if not from_state:
        expected_targets += 1000
    torch.testing.assert_close(raw["state_trajectory"][: real_actions + 1, 0], expected_states)
    torch.testing.assert_close(raw["action_target"][:real_actions, 0], expected_targets)
    assert (raw["state_trajectory"][real_actions:, 0] == expected_states[-1]).all()
    assert (raw["action_target"][real_actions - 1 :, 0] == expected_targets[-1]).all()
    assert raw["state_timestamps"].dtype == raw["action_timestamps"].dtype == torch.float64
    torch.testing.assert_close(
        raw["state_timestamps"],
        1000 + torch.arange(first, first + padded_actions + 1, dtype=torch.float64) / 30,
        rtol=0,
        atol=1e-12,
    )
    torch.testing.assert_close(raw["action_timestamps"], raw["state_timestamps"][:-1], rtol=0, atol=0)
    encoded, metadata = build_block_sample(raw, template=reader.template, planner=reader.planner, history_blocks=2)
    assert metadata.full_range == raw["full_range"]
    assert metadata.action_mask.shape == (padded_actions, reader.template.width)
    assert metadata.action_mask[:, 0].all()  # 通道有效性不承担 padding loss 屏蔽。
    assert metadata.real_action_steps == real_actions
    assert type(metadata.num_valid_video_latents) is int
    decoded = reader.template.decode_action_delta(
        encoded["action"], metadata.anchors.repeat_interleave(32, dim=0), raw["action_mask"]
    )
    torch.testing.assert_close(decoded[:, 0], raw["action_target"][:, 0])


@pytest.mark.parametrize("output_actions", [3, 8])
def test_tail_padding_accepts_target_length_independently_of_block_geometry(output_actions):
    reader = object.__new__(module.SegmentLeRobotDataset)
    reader._fps = 30
    state = torch.arange(8).reshape(4, 2).float()
    target = torch.arange(6).reshape(3, 2).float() + 100
    times = 1000 + torch.arange(4, dtype=torch.float64) / 30
    originals = [tensor.clone() for tensor in (state, target, times)]
    padded = reader._pad_to_action_steps(state, target, times, real_actions=3, output_actions=output_actions)
    expected_rows = torch.arange(output_actions + 1).clamp(max=3)
    expected_actions = torch.arange(output_actions).clamp(max=2)
    torch.testing.assert_close(padded[0], state[expected_rows], rtol=0, atol=0)
    torch.testing.assert_close(padded[1], target[expected_actions], rtol=0, atol=0)
    torch.testing.assert_close(
        padded[2], 1000 + torch.arange(output_actions + 1, dtype=torch.float64) / 30, rtol=0, atol=1e-12
    )
    for original, tensor, result in zip(originals, (state, target, times), padded):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)
        if output_actions == 3:
            assert result is tensor  # 不补齐时不分配副本。


@pytest.mark.parametrize("real_actions,output_actions", [(0, 32), (3, 2)])
def test_tail_padding_rejects_empty_data_or_truncation(real_actions, output_actions):
    reader = object.__new__(module.SegmentLeRobotDataset)
    with pytest.raises(ValueError, match="1 <= real_actions <= output_actions"):
        reader._pad_to_action_steps(None, None, None, real_actions=real_actions, output_actions=output_actions)


@pytest.mark.parametrize("table_backend", ["hf", "parquet"])
@pytest.mark.parametrize("offset", [-2, 3])
@pytest.mark.parametrize("early", [False, True])
def test_positive_segments_keep_actual_length_and_video_timeline(monkeypatch, tmp_path, table_backend, offset, early):
    reader, _ = make_reader(
        monkeypatch, lengths=(50,), root=tmp_path, table_backend=table_backend, offset=offset, timestamp_start=1000.0
    )
    reader.early_video_sampling = early
    received = []
    monkeypatch.setattr(reader, "_read_video", lambda meta, episode, times, *, viewpoint: received.append(times))
    first = max(0, -offset)
    assert reader._segments == [(0, 0, first, 32)]
    raw = reader[0]
    assert "full_range" not in raw
    assert raw["state_trajectory"].shape[0] == 33
    assert raw["action_target"].shape[0] == 32  # 仍截至完整 block，不补至 maximum=64。
    torch.testing.assert_close(raw["state_trajectory"][:, 0], torch.arange(first, first + 33).float())
    torch.testing.assert_close(
        raw["action_target"][:, 0], torch.arange(first + offset, first + offset + 32).float() + 1000
    )
    expected_times = 1000 + torch.arange(first, first + 33, dtype=torch.float64) / 30
    torch.testing.assert_close(raw["state_timestamps"], expected_times, rtol=0, atol=0)
    torch.testing.assert_close(raw["action_timestamps"], expected_times[:-1], rtol=0, atol=0)
    assert received == [expected_times[:: 4 if early else 1].tolist()]


@pytest.mark.parametrize("table_backend", ["hf", "parquet"])
@pytest.mark.parametrize("offset", [-2, 0, 1, 3])
def test_full_subtask_uses_one_exact_legal_range_and_keeps_caption(monkeypatch, tmp_path, table_backend, offset):
    annotations = [(1, 34), (45, 49), (70, 130)]
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/episodes.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 0,
                "action_config": [
                    {"start_frame": lo, "end_frame": hi, "action_text": str(i)}
                    for i, (lo, hi) in enumerate(annotations)
                ],
            }
        )
    )
    reader, _ = make_reader(
        monkeypatch,
        lengths=(131,),
        offset=offset,
        root=tmp_path,
        use_subtask=True,
        maximum=-1,
        table_backend=table_backend,
    )
    assert len(reader) == len(annotations)
    for i, (lo, hi) in enumerate(annotations):
        first = max(lo, -offset)
        stop = min(hi + 1, 131, 132 - offset)
        assert reader._segments[i] == (0, 0, first, stop - first - 1)
        assert reader._range_bounds[i] == (lo, hi, first, stop)
        raw = reader[i]
        assert raw["full_range"] == (first, stop)
        assert raw["ai_caption"] == str(i)


def test_full_invalid_subtask_still_falls_back_to_complete_episode(monkeypatch, tmp_path):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/episodes.jsonl").write_text(json.dumps({"episode_index": 0, "action_config": None}))
    with pytest.warns(UserWarning, match="falling back to episode mode"):
        reader, _ = make_reader(monkeypatch, lengths=(98,), root=tmp_path, use_subtask=True, maximum=-1)
    assert reader._segments == [(0, 0, 0, 97)]
    assert reader[0]["ai_caption"] == "test task"


@pytest.mark.parametrize("frames", [0, 1])
def test_full_episode_without_an_action_fails_instead_of_skipping(monkeypatch, tmp_path, frames):
    with pytest.raises(ValueError, match="Full range.*at least one action"):
        make_reader(monkeypatch, lengths=(frames,), root=tmp_path, maximum=-1)


@pytest.mark.parametrize("early", [False, True])
@pytest.mark.parametrize("table_backend", ["hf", "parquet"])
@pytest.mark.parametrize("frames,offset", [(33, 1), (34, 0), (34, -2), (34, 3), (107, 1)])
@pytest.mark.parametrize("window_mode", ["segmented", "frame_window"])
def test_tail_video_repeats_actual_last_frame_without_reading_past_episode(
    monkeypatch, tmp_path, early, table_backend, frames, offset, window_mode
):
    import av

    from cosmos_framework.data.generator.action.video_view import VideoViewConfig

    monkeypatch.setattr(
        "cosmos_framework.data.generator.action.video_view.ACTION_RES_SIZE_INFO", {"256": {"square": (16, 16)}}
    )
    reader, _ = make_reader(
        monkeypatch,
        lengths=(frames,),
        root=tmp_path,
        maximum=-1 if window_mode == "segmented" else 96,
        table_backend=table_backend,
        offset=offset,
        window_mode=window_mode,
    )
    meta = reader._direct_meta if table_backend == "parquet" else reader._get_dataset(0).meta
    meta.features["camera"] = {"dtype": "video", "shape": [16, 16, 3], "names": ["height", "width", "rgb"]}
    episode = dict(meta.episodes[0], **{"videos/camera/from_timestamp": 0.0})
    meta.episodes = Dataset.from_list([episode])
    meta.get_video_file_path = lambda episode_id, key: Path("camera.mp4")
    with av.open(str(tmp_path / "camera.mp4"), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = 16, 16, "yuv420p"
        for i in range(frames):
            frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), (3 * i) % 256, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    reader.video_view = VideoViewConfig(cameras={"head": "camera"})
    reader._viewpoint = "ego_view"
    reader.video_backend, reader._pyav_resize = "pyav_resize", True
    reader.early_video_sampling = early
    original = reader.video_view.read(
        meta, 0, [i / 30 for i in range(frames)], viewpoint="ego_view", tolerance_s=1e-4, video_backend="pyav_resize"
    )
    received = []
    read_video = reader._read_video

    def capture_video(meta, episode_id, times, *, viewpoint):
        received.extend(times)
        return read_video(meta, episode_id, times, viewpoint=viewpoint)

    monkeypatch.setattr(reader, "_read_video", capture_video)
    index = 5 if window_mode == "frame_window" and len(reader) > 1 else 0
    raw = reader[index]
    first, stop = max(0, -offset) + index, min(frames, frames + 1 - offset)
    real_actions = min(96, stop - first - 1) if window_mode == "frame_window" else stop - first - 1
    stop = first + real_actions + 1
    padded_actions = ((real_actions + 31) // 32) * 32
    indexes = torch.arange(0, padded_actions + 1, 4 if early else 1)
    expected_rows = (first + indexes).clamp(max=stop - 1)
    torch.testing.assert_close(raw["video"], original[:, expected_rows], rtol=0, atol=0)
    torch.testing.assert_close(
        torch.tensor(received, dtype=torch.float64), expected_rows.double() / 30, rtol=0, atol=1e-12
    )
    if window_mode == "segmented":
        assert raw["full_range"] == (first, stop)
    else:
        assert "full_range" not in raw
    assert raw["real_action_steps"] == real_actions
    assert "video_frame_validity" not in raw
    if early:
        assert torch.equal(raw["video_observation_indexes"], indexes)


def full_sft_sample(monkeypatch, tmp_path, mode="policy"):
    from cosmos_framework.data.generator.action.block_state import BlockStatistics, Quantiles
    from cosmos_framework.data.generator.action.datasets.action_sft_dataset import ActionSFTDataset
    from cosmos_framework.data.generator.action.datasets.causal_action_sft_dataset import CausalActionSFTDataset
    from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline

    reader, _ = make_reader(monkeypatch, lengths=(34,), root=tmp_path, maximum=-1, table_backend="parquet")
    monkeypatch.setattr(
        module.SegmentLeRobotDataset,
        "_read_video",
        lambda self, meta, episode_id, times, **kwargs: torch.zeros(3, len(times), 64, 64),
    )
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
    adapter = CausalActionSFTDataset(ActionSFTDataset(reader, transform), mode=mode, statistics=statistics)
    return adapter[0]


@pytest.mark.parametrize("mode", ["policy", "forward_dynamics", "inverse_dynamics"])
def test_full_sft_adapter_carries_valid_lengths_and_original_range(monkeypatch, tmp_path, mode):
    from cosmos_framework.data.generator.joint_dataloader import custom_collate_fn

    result = full_sft_sample(monkeypatch, tmp_path, mode)
    metadata = result["sequence_plan"].causal_action_metadata
    assert result["action"].shape == (64, 55)
    assert result["video"].shape == (3, 17, 64, 64)
    assert metadata.full_range == (0, 34)
    assert metadata.action_frame_ids.shape == (64,)
    assert metadata.action_mask[:, 0].all()
    assert metadata.real_action_steps == 33
    assert metadata.num_valid_video_latents == 3
    assert "full_range" not in result
    assert "real_action_steps" not in result and "video_frame_validity" not in result
    batch = custom_collate_fn([result])
    assert batch["sequence_plan"][0].causal_action_metadata.full_range == (0, 34)


def full_range_packer(samples, budget):
    from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, custom_collate_fn

    # Exercise the real packing loop and collate without distributed loader initialization.
    loader = object.__new__(PackingDataLoader)
    loader.dataloader_list = [SimpleNamespace(dataset_name="test")]
    loader.dataset_name_list = ["test"]
    loader.dataloaders = [iter(custom_collate_fn([sample]) for sample in samples)]
    loader.buffers = [deque()]
    loader.lookahead_limits = [1]
    loader.max_samples_per_batch = None
    loader.max_sequence_length = budget
    loader.tokenizer_spatial_compression_factor = 16
    loader.tokenizer_temporal_compression_factor = 4
    loader.patch_spatial = 2
    loader.uniae_chunk_frames = None
    loader.sound_latent_fps = 0
    loader.data_len = len(samples)
    loader.global_id = 0
    return loader


def test_full_range_packing_keeps_lookahead_sample_without_rereading(monkeypatch, tmp_path):
    result = full_sft_sample(monkeypatch, tmp_path)
    loader = full_range_packer([result] * 3, None)
    tokens = loader._compute_num_tokens_per_sample(
        {"video": [result["video"]], "action": [result["action"]], "sequence_plan": [result["sequence_plan"]]}
    )
    loader.max_sequence_length = 2 * tokens
    batches = list(loader)
    assert len(batches) == 3  # Existing packing uses a strict '< budget' boundary.
    for batch in batches:
        assert len(batch["action"]) == 1
        assert batch["action"][0][0] is result["action"]
        assert batch["sequence_plan"][0].causal_action_metadata.full_range == (0, 34)


@pytest.mark.parametrize("budget_offset", [-1, 0])
def test_full_range_over_packing_budget_is_an_error(monkeypatch, tmp_path, budget_offset):
    result = full_sft_sample(monkeypatch, tmp_path)
    loader = full_range_packer([result], None)
    tokens = loader._compute_num_tokens_per_sample(
        {"video": [result["video"]], "action": [result["action"]], "sequence_plan": [result["sequence_plan"]]}
    )
    loader.max_sequence_length = tokens + budget_offset
    with pytest.raises(ValueError, match="Full segment source=.*observations=\\(0, 34\\).*tokens.*packing"):
        next(iter(loader))
