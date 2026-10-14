# SPDX-License-Identifier: OpenMDW-1.1
"""Mix source datasets before max_tokens packing on every training rank."""

import logging
import math
import random
from bisect import bisect_right

import torch
from torch.utils.data import IterableDataset, get_worker_info

logger = logging.getLogger(__name__)


class CausalActionMixture(IterableDataset):
    """Mix transformed, map-style causal sources by traversal or uniform sampling.

    Place this as one dataset in RankPartitionedDataLoader so a packed batch
    can contain different robots/valid slots even on a single rank. Every
    source retains its own adapter and training statistics. Segments are independent
    training units; sharding does not require one whole episode per worker.
    """

    def __init__(self, datasets, weights=None, seed=42, *, sampling_strategy="shuffle_cycle"):
        if sampling_strategy not in ("shuffle_cycle", "sample_uniform"):
            raise ValueError("sampling_strategy must be shuffle_cycle or sample_uniform")
        self.sampling_strategy = sampling_strategy
        self.datasets = list(datasets)
        self.weights = [1.0] * len(self.datasets) if weights is None else list(weights)
        if not self.datasets or len(self.weights) != len(self.datasets):
            raise ValueError("Mixture requires one weight per source")
        if sampling_strategy == "shuffle_cycle" and any(not math.isfinite(w) or w <= 0 for w in self.weights):
            raise ValueError("Source weights must be finite and positive")
        if any(isinstance(d, IterableDataset) for d in self.datasets):
            raise ValueError("Set iterable_shuffle=False for sources; mixture owns sharding")
        templates = {(d.template.template_id, d.template.width) for d in self.datasets}
        if len(templates) != 1:
            raise ValueError("Mixed sources must use the same action template")
        self.seed = seed
        self.shard_rank, self.shard_world_size = 0, 1
        if sampling_strategy == "sample_uniform":
            self._source_cum_ends = []
            total = 0
            for i, dataset in enumerate(self.datasets):
                size = len(dataset)
                if not size:
                    raise ValueError(f"Source {i} has no valid segments")
                total += size
                self._source_cum_ends.append(total)
            logger.info(
                "Action sampling strategy=sample_uniform; source_segments=%s; source weights ignored",
                [len(dataset) for dataset in self.datasets],
            )
        else:
            logger.info("Action sampling strategy=shuffle_cycle; source_weights=%s", self.weights)

    def __len__(self):
        # Nominal epoch size for PackingDataLoader accounting; iteration cycles.
        return sum(len(dataset) for dataset in self.datasets)

    def __iter__(self):
        worker = get_worker_info()
        wid, nw = (worker.id, worker.num_workers) if worker else (0, 1)
        rng = random.Random(self.seed + self.shard_rank * nw + wid)
        if self.sampling_strategy == "sample_uniform":
            # All consumers draw from the full segment space, with replacement.
            while True:
                index = rng.randrange(self._source_cum_ends[-1])
                source = bisect_right(self._source_cum_ends, index)
                index -= self._source_cum_ends[source - 1] if source else 0
                try:
                    yield self.datasets[source][index]
                except (MemoryError, torch.OutOfMemoryError):
                    raise
                except Exception:
                    logger.error(
                        "Skipping unreadable or unprocessable action sample: rank=%s worker=%s source=%s index=%s",
                        self.shard_rank,
                        wid,
                        source,
                        index,
                        exc_info=True,
                    )
                    continue
        else:
            streams = []
            shard = self.shard_rank * nw + wid
            total_shards = self.shard_world_size * nw
            for i, dataset in enumerate(self.datasets):
                if len(dataset) < total_shards:
                    raise ValueError("Each source needs at least one segment per rank/worker; reduce workers or ranks")
                streams.append(self._source_stream(dataset, self.seed + i, shard, total_shards))
            while True:
                index = rng.choices(range(len(streams)), weights=self.weights, k=1)[0]
                yield next(streams[index])

    @staticmethod
    def _source_stream(dataset, seed, shard, total_shards):
        # 各 rank/worker 共用 episode 顺序，再按片段序号分片；短数据源也不复制样本。
        blocks = dataset.get_shuffle_blocks()
        epoch = 0
        while True:
            order = torch.randperm(len(blocks), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
            position = 0
            for block in order:
                start, length = blocks[block]
                first = start + (shard - position) % total_shards
                for index in range(first, start + length, total_shards):
                    try:
                        yield dataset[index]
                    except (MemoryError, torch.OutOfMemoryError):
                        raise
                    except Exception:
                        logger.error(
                            "Skipping unreadable or unprocessable action sample: shard=%s/%s epoch=%s index=%s",
                            shard,
                            total_shards,
                            epoch,
                            index,
                            exc_info=True,
                        )
                        continue
                position += length
            epoch += 1
