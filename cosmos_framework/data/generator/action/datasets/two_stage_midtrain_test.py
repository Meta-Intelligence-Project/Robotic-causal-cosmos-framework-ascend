# SPDX-License-Identifier: OpenMDW-1.1
"""Two recipe contracts through real reading, workers, packing and checkpoint loading."""

import json
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomllib
import torch
import torch.distributed.checkpoint as dcp
from torch.nn import functional as F

from cosmos_framework.checkpoint.dcp import CustomLoadPlanner, ModelWrapper
from cosmos_framework.data.generator.action.datasets import causal_action_factory as factory
from cosmos_framework.data.generator.action.datasets.causal_action_factory_test import SerialPool
from cosmos_framework.data.generator.action.datasets.frame_window_test import frame_window_adapter
from cosmos_framework.data.generator.action.datasets.segment_lerobot_dataset_test import full_range_packer, make_reader
from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, custom_collate_fn
from cosmos_framework.data.generator.sequence_packing import pack_input_sequence
from cosmos_framework.data.generator.sequence_packing.causal_action import (
    dense_action_mask,
    expand_action_sequence,
)
from cosmos_framework.model.generator.mot.causal_action_template_test import SmallActionNetwork
from cosmos_framework.model.generator.omni_mot_causal_action_model import OmniMoTCausalActionModel
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean

RECIPES = Path(__file__).parents[5] / "examples/action_pretrain"


def recipe(stage):
    return tomllib.loads((RECIPES / f"action_midtrain_stage{stage}_edge_causal_tnd.toml").read_text())


@pytest.mark.parametrize("stage", [1, 2])
def test_recipe_worst_case_text_and_camera_layout_fit_packing(stage):
    from cosmos_framework.data.generator.utils import ACTION_RES_SIZE_INFO

    config = recipe(stage)
    actions = config["action"]["max_action_steps"]
    counts = []
    for width, height in ACTION_RES_SIZE_INFO["480"].values():
        for view in ("ego_view", "concat_view"):
            output_height = height if view == "ego_view" else height + height // 2
            sample = {
                "video": torch.empty(1).expand(3, actions // 4 + 1, output_height, width),
                "action": torch.empty(1).expand(actions, 55),
                "text_token_ids": torch.empty(4096, dtype=torch.long),
                "sequence_plan": SimpleNamespace(causal_action_metadata=SimpleNamespace(states=range(actions // 32))),
            }
            counter = full_range_packer([], None)
            tokens = counter._compute_num_tokens_per_sample({k: [v] for k, v in sample.items()})
            counts.append(tokens)
    assert max(counts) == (14384 if stage == 1 else 109907)
    # Stage 2's maximum window can exceed its requested budget for large layouts.
    # Auditing the actual sources is required before adding longer subtask/episode data.
    assert (max(counts) < config["dataloader_train"]["max_sequence_length"]) == (stage == 1)


@pytest.mark.parametrize("stage", [1, 2])
def test_recipe_factory_reaches_reader_and_keeps_source_contract(stage, monkeypatch, tmp_path):
    from omegaconf import OmegaConf

    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml

    manifest = tmp_path / "sources.json"
    manifest.write_text(
        json.dumps(
            {
                "sources": [
                    dict(
                        root="fixture",
                        reader="agibot",
                        viewpoint="ego_view",
                        statistics_path=str(tmp_path / "stats.json"),
                        read_options=dict(action_from_state=True, action_time_offset_steps=1),
                    )
                ]
            }
        )
    )
    for name in ("WAN_VAE_PATH", "BASE_CHECKPOINT_PATH", "STAGE1_CHECKPOINT_PATH"):
        monkeypatch.setenv(name, "/tmp/fixture")
    monkeypatch.setenv("ACTION_SOURCES_FILE", str(manifest))
    config = load_experiment_from_toml(str(RECIPES / f"action_midtrain_stage{stage}_edge_causal_tnd.toml"))
    node = config.dataloader_train.dataloader.datasets.robots.dataset
    kwargs = OmegaConf.to_container(node, resolve=True)
    kwargs.pop("_target_")
    kwargs.update(tokenizer_config=None)
    built = []

    def build(root, **options):
        assert options["read_options"].action_from_state is True
        assert options["read_options"].action_time_offset_steps == 1
        reader, _ = make_reader(
            monkeypatch,
            root=tmp_path,
            lengths=(175,),
            table_backend="parquet",
            maximum=options["planner"].max_action_steps,
            overlap=options["planner"].overlap_action_steps,
            window_mode=options["window_mode"],
            offset=1,
            from_state=True,
        )
        monkeypatch.setattr(reader, "_read_video", lambda *args, **kw: torch.zeros(3, len(args[2]), 64, 64))
        frame_window_adapter(reader, mode="policy").statistics.save(tmp_path / "stats.json")
        built.append(reader)
        return reader

    monkeypatch.setattr(factory, "Pool", SerialPool)
    monkeypatch.setattr(factory, "discover_dataset_roots", lambda paths: [tmp_path])
    monkeypatch.setattr(factory, "_build_reader", build)
    dataset = factory.get_causal_action_dataset(**kwargs)
    sample = dataset.datasets[0][0]
    metadata = sample["sequence_plan"].causal_action_metadata
    assert built[0].window_mode == ("frame_window" if stage == 1 else "segmented")
    assert metadata.real_action_steps == 160
    assert len(sample["action"]) == 160
    if stage == 1:
        assert metadata.history_blocks == 5
    assert metadata.full_range is None
    assert dataset.sampling_strategy == "sample_uniform"


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("stage", [1, 2])
def test_two_stage_workers_keep_supplying_packed_samples(stage, workers, monkeypatch, tmp_path):
    from cosmos_framework.data.generator.action.datasets.causal_action_dataset_group import CausalActionDatasetGroup
    from cosmos_framework.data.generator.action.datasets.causal_action_mixture import CausalActionMixture

    torch.set_num_threads(1)
    if workers and hasattr(torch, "npu") and not torch.npu.is_available():
        # torch_npu wraps the CPU iterator with device binding; restore PyTorch's
        # original constructor only in this hardware-free CPU test.
        from torch_npu.utils._module import origin_mpdl_iter_init

        monkeypatch.setattr(
            torch.utils.data.dataloader._MultiProcessingDataLoaderIter, "__init__", origin_mpdl_iter_init
        )
    reader, _ = make_reader(
        monkeypatch,
        root=tmp_path,
        lengths=(11, 34, 171),
        table_backend="parquet",
        maximum=recipe(stage)["action"]["max_action_steps"],
        overlap=recipe(stage)["action"]["overlap_action_steps"],
        window_mode="frame_window" if stage == 1 else "segmented",
    )
    monkeypatch.setattr(reader, "_read_video", lambda meta, episode, times, **kw: torch.zeros(3, len(times), 64, 64))
    adapter = frame_window_adapter(reader, mode="policy")
    cfg = recipe(stage)
    adapter.full_history = cfg["action"].get("full_history", False)
    if not adapter.full_history:
        adapter.history_range = tuple(cfg["model"][f"teacher_forcing_history_blocks_{b}"] for b in ("min", "max"))
    mixture = CausalActionMixture([CausalActionDatasetGroup([adapter])], sampling_strategy="sample_uniform")
    inner = torch.utils.data.DataLoader(
        mixture,
        batch_size=1,
        num_workers=workers,
        collate_fn=custom_collate_fn,
        timeout=30 if workers else 0,
    )
    packed = PackingDataLoader(
        inner,
        tokenizer_spatial_compression_factor=16,
        tokenizer_temporal_compression_factor=4,
        patch_spatial=2,
        max_sequence_length=1000,
        lookahead_limit=1,
    )
    batches = list(islice(packed, 4))
    assert len(batches) == 4
    for batch in batches:
        for action, plan in zip(batch["action"], batch["sequence_plan"], strict=True):
            metadata = plan.causal_action_metadata
            assert len(action[0]) % 32 == 0
            assert len(action[0]) <= 160
            if stage == 1:
                assert metadata.history_blocks == len(action[0]) // 32
            assert metadata.full_range is None
            assert 1 <= metadata.real_action_steps <= len(action[0])


def test_stage2_recipe_uses_configured_overlap_and_right_aligned_tail(monkeypatch, tmp_path):
    config = recipe(2)["action"]
    reader, _ = make_reader(
        monkeypatch,
        root=tmp_path,
        lengths=(5000,),
        table_backend="parquet",
        maximum=config["max_action_steps"],
        overlap=config["overlap_action_steps"],
        window_mode=config["window_mode"],
    )
    assert [reader._resolve_index(i)[2:] for i in range(len(reader))] == [
        (0, 1792),
        (1471, 1792),
        (2942, 1792),
        (3207, 1792),
    ]


@pytest.mark.parametrize("maximum", [96, 160, 224])
@pytest.mark.parametrize("real", [10, 33, None])
def test_full_history_tracks_actual_blocks_when_window_changes(maximum, real, monkeypatch, tmp_path):
    reader, _ = make_reader(
        monkeypatch,
        root=tmp_path,
        lengths=((real or maximum) + 1,),
        table_backend="parquet",
        maximum=maximum,
        window_mode="frame_window",
    )
    monkeypatch.setattr(reader, "_read_video", lambda meta, episode, times, **kw: torch.zeros(3, len(times), 64, 64))
    adapter = frame_window_adapter(reader, mode="policy")
    adapter.full_history = True
    # A deliberately small numeric history limit must not cap full-history mode.
    adapter.history_range = (1, 1)
    sample = adapter[0]
    metadata = sample["sequence_plan"].causal_action_metadata
    assert metadata.history_blocks == len(sample["action"]) // 32
    adapter.full_history = False
    assert adapter[0]["sequence_plan"].causal_action_metadata.history_blocks == 1


@pytest.mark.parametrize("mode", ["policy", "forward_dynamics", "inverse_dynamics"])
def test_two_stage_forward_backward_and_adapter_checkpoint(mode, monkeypatch, tmp_path):
    torch.manual_seed(7)
    net = SmallActionNetwork(55)
    net.vision_head = torch.nn.Linear(net.hidden_size, 2)
    saved = None
    for stage in (1, 2):
        if stage == 2:
            net = SmallActionNetwork(55)
            net.vision_head = torch.nn.Linear(net.hidden_size, 2)
            wrapper = ModelWrapper(net)
            state = wrapper.state_dict()
            dcp.load(
                state,
                checkpoint_id=str(tmp_path / "checkpoint"),
                no_dist=True,
                planner=CustomLoadPlanner(keys_to_skip_loading=recipe(2)["checkpoint"]["keys_to_skip_loading"]),
            )
            wrapper.load_state_dict(state)
            for key, expected in saved.items():
                torch.testing.assert_close(net.state_dict()[key], expected, rtol=0, atol=0)
        reader, _ = make_reader(
            monkeypatch,
            root=tmp_path / f"stage{stage}",
            lengths=(171,),
            table_backend="parquet",
            maximum=recipe(stage)["action"]["max_action_steps"],
            overlap=recipe(stage)["action"]["overlap_action_steps"],
            window_mode="frame_window" if stage == 1 else "segmented",
        )
        monkeypatch.setattr(
            reader, "_read_video", lambda meta, episode, times, **kw: torch.zeros(3, len(times), 64, 64)
        )
        adapter = frame_window_adapter(reader, mode=mode)
        cfg = recipe(stage)
        adapter.full_history = cfg["action"].get("full_history", False)
        if not adapter.full_history:
            adapter.history_range = tuple(cfg["model"][f"teacher_forcing_history_blocks_{b}"] for b in ("min", "max"))
        sample = adapter[0]
        steps = len(sample["action"])
        frames = 1 + steps // 16
        video = torch.zeros(1, 2, frames, 1, 1)
        plan = sample["sequence_plan"]
        clean = GenerationDataClean(
            batch_size=1,
            is_image_batch=False,
            x0_tokens_vision=[video],
            x0_tokens_action=[sample["action"]],
            fps_vision=sample["conditioning_fps"][None],
            fps_action=sample["conditioning_fps_action"][None],
        )
        packed = pack_input_sequence(
            [plan],
            [[11, 12]],
            clean,
            torch.tensor([0.5]),
            dict(bos_token_id=1, eos_token_id=2, start_of_generation=3, end_of_generation=4),
            enable_fps_modulation=True,
            action_dim=55,
        )
        packed.causal_action_metadata = [plan.causal_action_metadata]
        model = OmniMoTCausalActionModel.__new__(OmniMoTCausalActionModel)
        geometry = model.prepare_teacher_forcing_geometry_from_plans([frames], [plan])
        packed = expand_action_sequence(packed, [video], geometry)
        layout = packed.teacher_forcing.layout
        visible = dense_action_mask(layout)
        queries = layout.gen_query_indexes
        if stage == 1:
            assert geometry.history_blocks == (5,)
            past_visual = (
                (layout.stream_ids[None, :] == 0)
                & (layout.block_ids[None, :] < layout.block_ids[queries, None])
                & (layout.stream_ids[queries, None] != 3)
            )
            assert past_visual.any() and visible[past_visual].all()
            future = layout.block_ids[None, :] > layout.block_ids[queries, None]
            assert not visible[future.expand_as(visible)].any()
        encoded = torch.zeros(packed.sequence_length, net.hidden_size)
        net._encode_action(packed, encoded, torch.float32)
        hidden = encoded.clone()
        hidden[queries] = F.scaled_dot_product_attention(
            encoded[queries][None, None], encoded[None, None], encoded[None, None], attn_mask=visible
        )[0, 0]
        result = {}
        net._decode_action(packed, hidden, result)
        result["preds_vision"] = [net.vision_head(hidden[packed.vision.sequence_indexes]).T.reshape(2, frames, 1, 1)]
        # Exercise the real action model loss and backward on the small interfaces.
        torch.nn.Module.__init__(model)
        model.tensor_kwargs_fp32 = dict(device="cpu", dtype=torch.float32)
        model.config = SimpleNamespace(
            vision_gen=True,
            action_gen=True,
            sound_gen=False,
            causal_action_log_loss_by_mode=False,
            rectified_flow_training_config=SimpleNamespace(
                normalize_loss_by_active=False, image_loss_scale=None, loss_scale=1.0, action_loss_weight=1.0
            ),
        )
        model.rectified_flow_action = SimpleNamespace(train_time_weight=lambda ts, kwargs: torch.ones_like(ts))
        model.rectified_flow_video = model.rectified_flow_action
        loss, _ = model._compute_losses(
            result,
            packed,
            SimpleNamespace(
                vt_target_action=[torch.ones_like(result["preds_action"][0])],
                vt_target_vision=[torch.ones_like(result["preds_vision"][0])],
            ),
            torch.full((1, frames), 0.5),
            False,
        )
        assert torch.isfinite(loss)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
        loss.backward()
        assert net.llm2action.weight.grad is not None
        assert net.state2llm.weight.grad.abs().sum() > 0
        if mode != "inverse_dynamics":
            assert net.vision_head.weight.grad.abs().sum() > 0
        optimizer.step()
        if stage == 1:
            saved = {key: value.detach().clone() for key, value in net.state_dict().items()}
            dcp.save(ModelWrapper(net).state_dict(), checkpoint_id=str(tmp_path / "checkpoint"), no_dist=True)
