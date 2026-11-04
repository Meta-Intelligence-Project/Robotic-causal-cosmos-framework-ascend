# SPDX-License-Identifier: OpenMDW-1.1
"""Synchronize video-only and robot-action steps while packing each task independently."""

import torch.distributed as dist
from torch.utils.data import DataLoader

from cosmos_framework.data.generator.joint_dataloader import IterativeJointDataLoader
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import instantiate


class CausalMultitaskDataLoader(IterativeJointDataLoader):
    def __init__(self, dataloader, lookahead_limit=1, **kwargs):
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        loader_kwargs = {k: v for k, v in dataloader.items() if k != "datasets"}
        if loader_kwargs.get("num_workers", 0) == 0:
            loader_kwargs.pop("prefetch_factor", None)
            loader_kwargs.pop("persistent_workers", None)
        dataloaders = {}
        for i, (name, source) in enumerate(dataloader["datasets"].items()):
            if source["ratio"] < 0:
                raise ValueError("Causal task sampling ratios must be nonnegative")
            if source["ratio"] == 0:
                continue
            dataset = instantiate(source["dataset"])
            dataset.shard_rank, dataset.shard_world_size, dataset.shard_id = rank, world_size, i
            dataloaders[name] = dict(dataloader=L(DataLoader)(dataset=dataset, **loader_kwargs), ratio=source["ratio"])
        if not dataloaders:
            raise ValueError("At least one causal task sampling ratio must be positive")
        super().__init__(dataloaders=dataloaders, default_lookahead_limit=lookahead_limit, **kwargs)

    def _compute_num_tokens_per_sample(self, data_batch):
        return super()._compute_num_tokens_per_sample(data_batch, include_clean_vision=True)
