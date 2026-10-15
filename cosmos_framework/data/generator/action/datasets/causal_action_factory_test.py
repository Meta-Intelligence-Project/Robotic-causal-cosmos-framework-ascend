# SPDX-License-Identifier: OpenMDW-1.1
"""The configured window mode reaches real readers before adapters and mixtures."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.action.block_state import BlockStatistics, Quantiles
from cosmos_framework.data.generator.action.datasets import causal_action_factory as module
from cosmos_framework.data.generator.action.datasets.segment_lerobot_dataset_test import make_reader
from cosmos_framework.data.generator.action.sample_contract import ActionReadOptions
from cosmos_framework.data.generator.action.video_view import VideoViewConfig


class SerialPool:
    def __init__(self, *args):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def imap(self, function, roots):
        return map(function, roots)


def test_factory_carries_frame_window_to_real_multiroot_readers(monkeypatch, tmp_path):
    roots = [tmp_path / name for name in ("a0", "a1", "b0")]
    lengths = dict(zip(roots, (35, 37, 3), strict=True))
    manifest = tmp_path / "sources.json"
    manifest.write_text(
        json.dumps(
            {
                "sources": [
                    dict(
                        root=name, reader="test", viewpoint="ego_view", statistics_path="shared.json", use_subtask=False
                    )
                    for name in ("a", "b")
                ]
            }
        )
    )
    monkeypatch.setattr(module, "Pool", SerialPool)
    monkeypatch.setattr(module, "discover_dataset_roots", lambda paths: roots[:2] if paths == ["a"] else roots[2:])
    monkeypatch.setitem(
        module.READERS,
        "test",
        SimpleNamespace(
            default_read_options=ActionReadOptions(), default_video_view=VideoViewConfig(cameras={"head": "camera"})
        ),
    )
    built, statistics_reads = [], []

    def build(root, **kwargs):
        assert kwargs["window_mode"] == "frame_window"
        assert kwargs["planner"].overlap_action_steps == 0
        reader, _ = make_reader(
            monkeypatch,
            lengths=(lengths[root],),
            root=root,
            table_backend=kwargs["table_backend"],
            maximum=kwargs["planner"].max_action_steps,
            window_mode=kwargs["window_mode"],
        )
        reader._viewpoint = kwargs["viewpoint"]
        reader.early_video_sampling = kwargs["early_video_sampling"]
        monkeypatch.setattr(
            reader, "_read_video", lambda meta, episode, times, **kw: torch.zeros(3, len(times), 64, 64)
        )
        built.append(reader)
        return reader

    def load_statistics(path):
        statistics_reads.append(path)
        raw = built[0][0]
        return BlockStatistics(
            Quantiles(torch.full((55,), -10000.0), torch.full((55,), 10000.0), raw["state_mask"]),
            Quantiles(torch.full((55,), -10000.0), torch.full((55,), 10000.0), raw["action_mask"]),
        )

    monkeypatch.setattr(module, "_build_reader", build)
    monkeypatch.setattr(module.BlockStatistics, "load", load_statistics)
    mixture = module.get_causal_action_dataset(
        sources_file=manifest,
        template="cosmos_framework.data.generator.action.action_state_template.ActionStateTemplate55",
        window_mode="frame_window",
        max_action_steps=32,
        overlap_action_steps=161,  # Irrelevant legacy overlap must not prevent a short frame window.
        mode="policy",
        sampling_strategy="sample_uniform",
    )
    assert len(mixture) == 9
    assert mixture._source_cum_ends == [8, 9]
    assert [len(group) for group in mixture.datasets] == [8, 1]
    assert [len(reader) for reader in built] == [3, 5, 1]
    assert statistics_reads == [Path("shared.json").resolve()]
    for group in mixture.datasets:
        for dataset in group.datasets:
            sample = dataset[-1]
            assert sample["action"].shape == (32, 55)
            assert sample["video"].shape[1] == 9
            expected_real = min(32, lengths[dataset._dataset.root] - 1)
            assert sample["sequence_plan"].causal_action_metadata.real_action_steps == expected_real
            assert sample["sequence_plan"].causal_action_metadata.full_range is None


@pytest.mark.parametrize(
    "mode,maximum,overlap,message",
    [
        ("unknown", 96, 16, "window_mode"),
        ("frame_window", -1, 16, "positive max_action_steps"),
        ("frame_window", 0, 16, "positive max_action_steps"),
        ("frame_window", 33, 161, "num_frames"),
        ("segmented", 32, 161, "overlap_action_steps"),
    ],
)
def test_factory_rejects_invalid_windows_before_reading_sources(mode, maximum, overlap, message):
    with pytest.raises(ValueError, match=message):
        module.get_causal_action_dataset(
            sources_file="missing.json",
            template="cosmos_framework.data.generator.action.action_state_template.ActionStateTemplate55",
            window_mode=mode,
            max_action_steps=maximum,
            overlap_action_steps=overlap,
        )


@pytest.mark.parametrize("frames", [0, 1])
def test_factory_reports_an_empty_frame_window_root(monkeypatch, tmp_path, frames):
    root = tmp_path / "empty"
    manifest = tmp_path / "sources.json"
    manifest.write_text(json.dumps({"sources": [dict(root=str(root), reader="agibot", viewpoint="ego_view")]}))
    monkeypatch.setattr(module, "Pool", SerialPool)
    monkeypatch.setattr(module, "discover_dataset_roots", lambda paths: [root])
    reader, _ = make_reader(
        monkeypatch, lengths=(frames,), root=root, maximum=96, window_mode="frame_window", table_backend="parquet"
    )
    monkeypatch.setattr(module, "_build_reader", lambda root, **kwargs: reader)
    assert len(reader) == 0 and reader.skipped_window_episodes == 1
    with pytest.raises(ValueError, match="No complete training segments"):
        module.get_causal_action_dataset(
            sources_file=manifest,
            template="cosmos_framework.data.generator.action.action_state_template.ActionStateTemplate55",
            window_mode="frame_window",
        )
