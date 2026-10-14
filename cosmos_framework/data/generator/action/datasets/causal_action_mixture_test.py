# SPDX-License-Identifier: OpenMDW-1.1
import pickle
from collections import Counter
from itertools import islice
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from cosmos_framework.data.generator.action.datasets import causal_action_mixture as module
from cosmos_framework.data.generator.action.datasets.causal_action_dataset_group import CausalActionDatasetGroup
from cosmos_framework.data.generator.action.datasets.causal_action_mixture import CausalActionMixture


class Source:
    template = SimpleNamespace(template_id="test", width=3)

    def __init__(self, lengths, name=None):
        self.name = name
        self.blocks = []
        n = 0
        for length in lengths:
            self.blocks.append((n, length))
            n += length
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return (self.name, index) if self.name is not None else index

    def get_shuffle_blocks(self):
        return self.blocks


class WorkerSource(Source):
    def __getitem__(self, index):
        worker = torch.utils.data.get_worker_info()
        return worker.id if worker else -1, index


@pytest.mark.parametrize("lengths", [[17], [3, 7, 11], [1, 1, 1, 7]])
@pytest.mark.parametrize("shards", [1, 4, 8])
def test_segment_sharding_complete_and_disjoint(lengths, shards):
    dataset = Source(lengths)
    all_indexes = []
    for shard in range(shards):
        count = len(range(shard, len(dataset), shards))
        stream = CausalActionMixture._source_stream(dataset, 42, shard, shards)
        indexes = list(islice(stream, count))
        assert len(indexes) == len(set(indexes))
        all_indexes.extend(indexes)
    assert sorted(all_indexes) == list(range(len(dataset)))


def test_small_source_fails_instead_of_repeating():
    mixture = CausalActionMixture([Source([2])])
    mixture.shard_world_size = 4
    with pytest.raises(ValueError, match="one segment"):
        next(iter(mixture))


def test_source_weights_and_reproducible_sampling():
    class NamedSource(Source):
        def __init__(self, name):
            super().__init__([16])
            self.name = name

        def __getitem__(self, index):
            return self.name, index

    datasets = [NamedSource("agibot"), NamedSource("egosuite")]
    first = list(islice(iter(CausalActionMixture(datasets, [1, 3], seed=42)), 4000))
    second = list(islice(iter(CausalActionMixture(datasets, [1, 3], seed=42)), 4000))
    assert first == second
    assert 0.72 < sum(name == "egosuite" for name, _ in first) / len(first) < 0.78


def test_incompatible_templates_fail_before_iteration():
    other = Source([16])
    other.template = SimpleNamespace(template_id="different", width=3)
    with pytest.raises(ValueError, match="same action template"):
        CausalActionMixture([Source([16]), other])


@pytest.mark.parametrize("error", [OSError("read failed"), ValueError("transform failed")])
def test_bad_sample_logs_and_continues(error, caplog):
    class BrokenSource(Source):
        def __getitem__(self, index):
            if index == 0:
                raise error
            return index

    stream = CausalActionMixture._source_stream(BrokenSource([3]), 42, 0, 1)
    assert list(islice(stream, 4)) == [1, 2, 1, 2]
    assert "index=0" in caplog.text
    assert any(r.levelname == "ERROR" and r.exc_info for r in caplog.records)


def test_failed_epoch_retries_same_source(caplog):
    class RecoveringSource(Source):
        calls = 0

        def __getitem__(self, index):
            self.calls += 1
            if self.calls <= len(self):
                raise OSError("temporarily unavailable")
            return index

    source = RecoveringSource([3])
    stream = iter(CausalActionMixture([source]))
    assert list(islice(stream, 3)) == [0, 1, 2]
    assert source.calls == 6
    assert sum(r.levelname == "ERROR" for r in caplog.records) == 3


@pytest.mark.parametrize("error", [MemoryError("oom"), torch.OutOfMemoryError("oom"), KeyboardInterrupt()])
@pytest.mark.parametrize("sampling_strategy", ["shuffle_cycle", "sample_uniform"])
def test_resource_errors_and_interrupts_are_not_skipped(error, sampling_strategy):
    class BrokenSource(Source):
        def __getitem__(self, index):
            raise error

    with pytest.raises(type(error)):
        next(iter(CausalActionMixture([BrokenSource([2])], sampling_strategy=sampling_strategy)))


def test_uniform_global_boundaries_and_repeated_draws(monkeypatch, caplog):
    draws = iter([0, 2, 3, 7, 2, 0])
    monkeypatch.setattr(module.random, "Random", lambda seed: SimpleNamespace(randrange=lambda stop: next(draws)))
    caplog.set_level("INFO", logger=module.__name__)
    mixture = CausalActionMixture(
        [Source([3], "a"), Source([5], "b")], weights=[0, 1000], sampling_strategy="sample_uniform"
    )
    assert list(islice(mixture, 6)) == [("a", 0), ("a", 2), ("b", 0), ("b", 4), ("a", 2), ("a", 0)]
    assert "source weights ignored" in caplog.text


def test_uniform_multiroot_probabilities_and_consumer_random_streams(monkeypatch):
    def forbidden(*args):
        pytest.fail("Uniform sampling must not build block traversal state")

    monkeypatch.setattr(Source, "get_shuffle_blocks", forbidden)
    datasets = [
        CausalActionDatasetGroup([Source([2], "a/0"), Source([7], "a/1")]),
        CausalActionDatasetGroup([Source([1], "b/0")]),
    ]
    sequences = []
    for rank in range(2):
        for wid in range(2):
            monkeypatch.setattr(module, "get_worker_info", lambda: SimpleNamespace(id=wid, num_workers=2))
            mixture = CausalActionMixture(datasets, weights=[1, 99], sampling_strategy="sample_uniform")
            mixture.shard_rank, mixture.shard_world_size = rank, 2
            samples = list(islice(mixture, 6000))
            assert samples == list(islice(mixture, 6000))
            sequences.append(samples[:60])
            counts = Counter(samples)
            assert len(counts) == 10
            assert all(0.08 < count / len(samples) < 0.12 for count in counts.values())
            assert 0.88 < sum(name.startswith("a/") for name, _ in samples) / len(samples) < 0.92
    assert len({tuple(sequence) for sequence in sequences}) == 4


@pytest.mark.parametrize("num_workers", [0, 2])
def test_uniform_real_loader_reads_small_source_from_full_pool(num_workers):
    mixture = CausalActionMixture([WorkerSource([2])], sampling_strategy="sample_uniform")
    mixture.shard_rank, mixture.shard_world_size = 1, 4
    loader = DataLoader(mixture, batch_size=None, num_workers=num_workers)
    stream = iter(loader)
    try:
        samples = list(islice(stream, 80))
        workers = set(range(num_workers)) if num_workers else {-1}
        assert {worker for worker, _ in samples} == workers
        for worker in workers:
            assert {index for wid, index in samples if wid == worker} == {0, 1}
    finally:
        if num_workers:
            stream._shutdown_workers()


def test_uniform_large_segment_count_only_stores_source_boundaries():
    mixture = CausalActionMixture([Source([10**12]), Source([3])], sampling_strategy="sample_uniform")
    assert mixture._source_cum_ends == [10**12, 10**12 + 3]
    assert len(mixture) == 10**12 + 3
    assert len(pickle.dumps(mixture)) < 4096
    assert all(0 <= index < 10**12 for index in islice(mixture, 50))


@pytest.mark.parametrize("sizes", [[], [0], [3, 0]])
def test_uniform_empty_sources_fail_at_initialization(sizes):
    with pytest.raises(ValueError, match="one weight per source|no valid segments"):
        CausalActionMixture([Source([size]) for size in sizes], sampling_strategy="sample_uniform")


def test_unknown_sampling_strategy_fails():
    with pytest.raises(ValueError, match="sampling_strategy"):
        CausalActionMixture([Source([3])], sampling_strategy="unknown")


@pytest.mark.parametrize("error", [OSError("read failed"), ValueError("transform failed")])
def test_uniform_bad_sample_logs_and_continues(monkeypatch, error, caplog):
    class BrokenSource(Source):
        def __getitem__(self, index):
            if index == 0:
                raise error
            return index

    draws = iter([0, 1, 0, 2])
    monkeypatch.setattr(module.random, "Random", lambda seed: SimpleNamespace(randrange=lambda stop: next(draws)))
    mixture = CausalActionMixture([BrokenSource([3])], sampling_strategy="sample_uniform")
    assert list(islice(mixture, 2)) == [1, 2]
    assert sum(record.levelname == "ERROR" for record in caplog.records) == 2
    assert "source=0 index=0" in caplog.text
