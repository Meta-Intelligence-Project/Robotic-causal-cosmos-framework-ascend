# SPDX-License-Identifier: OpenMDW-1.1
"""Read local LeRobot tables without an HF disk cache or a cross-sample LRU."""

import pyarrow
import pyarrow.parquet as pq
from datasets.table import table_cast

from lerobot.datasets.utils import get_hf_features_from_features


def build_file_starts(meta) -> dict[tuple[int, int], int]:
    """一个文件可能包含多个完整 episode；从已有元数据推导文件全局起点。"""
    episodes = meta.episodes
    file_starts = {}
    for chunk, file, start in zip(
        episodes["data/chunk_index"],
        episodes["data/file_index"],
        episodes["dataset_from_index"],
        strict=True,
    ):
        key = (int(chunk), int(file))
        file_starts[key] = min(file_starts.get(key, int(start)), int(start))
    return file_starts


def read_window(meta, episode_id: int, start: int, stop: int, columns: list[str], *, file_starts) -> dict:
    """返回独立 Python 数据，避免窗口切片引用整文件的 Arrow buffers。

    区间为 episode 内的全局行号 [start, stop)；字段只读取一次。使用与 HF 相同的 schema
    转换检查实际维度和类型，忽略未声明的附加列，保留 timestamp 原始精度。
    """
    episode = meta.episodes[episode_id]
    if not int(episode["dataset_from_index"]) <= start < stop <= int(episode["dataset_to_index"]):
        raise IndexError(f"Invalid row range [{start}, {stop}) for episode {episode_id} in {meta.root}")
    columns = list(dict.fromkeys(columns))
    features = get_hf_features_from_features({key: meta.features[key] for key in columns})
    if set(features) != set(columns):
        raise ValueError("Only non-video columns can be read from parquet")
    file_start = file_starts[(int(episode["data/chunk_index"]), int(episode["data/file_index"]))]
    path = meta.root / meta.get_data_file_path(episode_id)
    with pq.ParquetFile(path) as parquet:
        table = parquet.read(columns=columns)
    # 先截取样本再做类型转换；只物化样本，避免固定长度转换额外复制整文件。
    table = table.slice(start - file_start, stop - start)
    if table.num_rows != stop - start:
        raise ValueError(f"{path}: missing rows for episode {episode_id}, range [{start}, {stop})")
    table = table_cast(table, features.arrow_schema)
    result = table.to_pydict()
    del table
    # 样本已转为独立 Python 数据；在当前读取进程归还空闲 Arrow 内存，降低 RSS 驻留。
    # 每个窗口调用一次，不降低解压期间的瞬时峰值。
    pyarrow.default_memory_pool().release_unused()
    return result
