# SPDX-License-Identifier: OpenMDW-1.1
"""Task scheduling, distributed source sharding and causal packing budgets."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset

from cosmos_framework.data.generator.causal_multitask_dataloader import CausalMultitaskDataLoader
from cosmos_framework.data.generator.sequence_packing import SequencePlan


class TinySource(Dataset):
    def __init__(self, has_action):
        self.has_action = has_action

    def __len__(self):
        return 8

    def __getitem__(self, index):
        plan = SequencePlan(has_text=True, has_vision=True, has_action=self.has_action)
        result = dict(video=torch.zeros(3, 9, 32, 32), text_token_ids=torch.tensor([1, 2]), sequence_plan=plan)
        if self.has_action:
            plan.causal_action_metadata = SimpleNamespace(states=torch.zeros(1, 55))
            result["action"] = torch.zeros(32, 55)
        return result


def make_loader(robot_ratio=1, video_ratio=1, **kwargs):
    return CausalMultitaskDataLoader(
        dataloader=dict(
            num_workers=0,
            batch_size=1,
            persistent_workers=True,
            prefetch_factor=2,
            datasets=dict(
                robots=dict(dataset=TinySource(True), ratio=robot_ratio),
                video=dict(dataset=TinySource(False), ratio=video_ratio),
            ),
        ),
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
        max_sequence_length=None,
        max_samples_per_batch=1,
        seed=None,
        prewarm=False,
        **kwargs,
    )


def test_tasks_alternate_and_resume_at_the_same_schedule_position():
    loader = make_loader()
    iterator = iter(loader)
    batches = [next(iterator) for _ in range(6)]
    assert [b["dataset_name"][0] for b in batches] == ["robots", "video"] * 3
    assert [b["sequence_plan"][0].has_action for b in batches] == [True, False] * 3
    assert all("action" not in batch for batch in batches[1::2])
    resumed = make_loader()
    resumed.set_start_iteration(3)
    assert next(iter(resumed))["dataset_name"] == batches[3]["dataset_name"]


def test_each_source_is_sharded_over_the_full_training_world(monkeypatch):
    monkeypatch.setattr("cosmos_framework.data.generator.causal_multitask_dataloader.dist.is_initialized", lambda: True)
    monkeypatch.setattr("cosmos_framework.data.generator.causal_multitask_dataloader.dist.get_rank", lambda: 1)
    monkeypatch.setattr("cosmos_framework.data.generator.causal_multitask_dataloader.dist.get_world_size", lambda: 2)
    loader = make_loader()
    for data in loader.dataloader_list:
        assert data.dataset.shard_rank == 1
        assert data.dataset.shard_world_size == 2


@pytest.mark.parametrize("ratios,task", [((1, 0), "robots"), ((0, 1), "video")])
def test_zero_weight_disables_a_task(ratios, task):
    assert next(iter(make_loader(*ratios)))["dataset_name"] == [task]


def test_invalid_sampling_ratios_are_rejected():
    with pytest.raises(ValueError, match="positive"):
        make_loader(0, 0)
    with pytest.raises(ValueError, match="nonnegative"):
        make_loader(-1, 1)


def test_packing_counts_clean_video_history_and_state_tokens():
    loader = make_loader()
    video = dict(
        video=[torch.zeros(3, 9, 32, 32)],
        text_token_ids=[torch.tensor([1, 2])],
        sequence_plan=SequencePlan(has_text=True, has_vision=True),
    )
    # 3 text tokens + 2 modality markers + 3 noisy and 3 clean visual tokens.
    assert loader._compute_num_tokens_per_sample(video) == 11
    robot = {**video, "action": [torch.zeros(32, 55)]}
    robot["sequence_plan"] = SequencePlan(
        has_text=True,
        has_vision=True,
        has_action=True,
        causal_action_metadata=SimpleNamespace(states=torch.zeros(1, 55)),
    )
    assert loader._compute_num_tokens_per_sample(robot) == 44
