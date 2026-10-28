# SPDX-License-Identifier: OpenMDW-1.1
"""CPU coverage for RTC launch opt-in, compatibility and CLI boundary wiring."""

import builtins
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from cosmos_framework.inference.causal_action import cli
from cosmos_framework.inference.causal_action.rtc import RTCPrefix


def _required(output="/unused-output"):
    return [
        "--checkpoint",
        "/checkpoint",
        "--sources-file",
        "/sources.json",
        "--processor",
        "/processor",
        "--vae",
        "/vae.pth",
        "--output",
        str(output),
    ]


def _prefix(path="/remaining.json"):
    return ["--rtc-prefix", str(path), "--rtc-delay-steps", "2"]


@pytest.mark.parametrize("flags", [[], ["--no-enable_rtc"], ["--no-enable-rtc"]])
def test_default_and_explicit_off_preserve_ordinary_defaults(flags):
    args = cli._parse_args([*_required(), *flags])
    assert args.enable_rtc is False
    assert args.rtc_prefix is None
    assert (args.guidance, args.steps, args.cache_mode, args.kv_cache) == (1.0, 10, "full", True)


@pytest.mark.parametrize("flag", ["--enable_rtc", "--enable-rtc", None])
def test_enable_aliases_and_legacy_prefix_pair_opt_in(flag):
    args = cli._parse_args([*_required(), *([] if flag is None else [flag]), *_prefix()])
    assert args.enable_rtc is True
    assert args.rtc_prefix == Path("/remaining.json")
    assert args.rtc_delay_steps == 2


def test_enable_flag_preserves_explicit_sampling_and_reference_cache_options():
    args = cli._parse_args(
        [*_required(), "--enable_rtc", *_prefix(), "--steps", "5", "--guidance", "3", "--cache-mode", "plans"]
    )
    assert (args.steps, args.guidance, args.cache_mode) == (5, 3.0, "plans")


@pytest.mark.parametrize(
    "flags,message",
    [
        (["--enable_rtc"], "requires --rtc-prefix and --rtc-delay-steps"),
        (["--enable-rtc"], "requires --rtc-prefix and --rtc-delay-steps"),
        (["--rtc-prefix", "/unused.json"], "must be supplied together"),
        (["--rtc-delay-steps", "2"], "must be supplied together"),
        (["--no-enable_rtc", *_prefix()], "explicitly disabled"),
        (["--no-enable-rtc", *_prefix()], "explicitly disabled"),
        (["--enable_rtc", *_prefix(), "--mode", "forward_dynamics"], "requires policy mode"),
        (["--enable-rtc", *_prefix(), "--mode", "inverse_dynamics"], "requires policy mode"),
        ([*_prefix(), "--mode", "inverse_dynamics"], "requires policy mode"),
        (["--enable_rtc", *_prefix(), "--preview-ground-truth-states"], "requires policy mode"),
    ],
)
def test_invalid_rtc_launch_fails_before_hardware_or_model_import(monkeypatch, capsys, flags, message):
    original_import = builtins.__import__

    def import_guard(name, *args, **kwargs):
        if name in ("torch", "torch_npu.contrib", "cosmos_framework.utils.generator.model_loader"):
            pytest.fail(f"Invalid CLI invocation reached hardware/model import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_guard)
    monkeypatch.setattr(sys, "argv", ["causal-action", *_required(), *flags])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("flag", [None, "--enable_rtc", "--enable-rtc"])
def test_launcher_print_only_preserves_defaults_and_forwards_rtc_alias(tmp_path, flag):
    script = Path(__file__).resolve().parents[3] / "examples/action_pretrain/inference_causal_action.sh"
    environment = os.environ.copy()
    for key in ("NUM_STEPS", "GUIDANCE", "CACHE_MODE", "HISTORY_BLOCKS"):
        environment.pop(key, None)
    environment.update(
        PRINT_ONLY="1",
        CHECKPOINT_ROOT="/checkpoint",
        ACTION_SOURCES_FILE="/sources.json",
        COSMOS3_EDGE_PROCESSOR_PATH="/processor",
        WAN_VAE_PATH="/vae.pth",
        OUTPUT_ROOT=str(tmp_path / "output"),
    )
    flags = [] if flag is None else [flag, *_prefix(tmp_path / "remaining prefix.json")]
    result = subprocess.run(["bash", str(script), *flags], env=environment, check=True, capture_output=True, text=True)
    command = shlex.split(result.stdout.split("Command: ", 1)[1])
    assert command[0] == "torchrun"
    arguments = command[command.index("cosmos_framework.inference.causal_action.cli") + 1 :]
    args = cli._parse_args(arguments)
    assert args.enable_rtc is (flag is not None)
    assert (args.steps, args.guidance, args.cache_mode, args.kv_cache) == (10, 1.0, "full", True)
    if flag is not None:
        assert flag in arguments
        assert args.rtc_prefix == tmp_path / "remaining prefix.json"


def _fake_module(monkeypatch, name, **members):
    module = ModuleType(name)
    module.__dict__.update(members)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.mark.parametrize("requested,reported", [(False, False), (True, True), (True, False)])
def test_main_wires_real_prefix_and_exports_actual_rtc_metadata(monkeypatch, tmp_path, requested, reported):
    """Run the whole CLI boundary with a fake loader, retaining real tensor IO/prefix validation."""
    captured = {}
    previous = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10
    prefix_path = tmp_path / "remaining.json"
    prefix_path.write_text(json.dumps({"previous_actions": previous.tolist()}))
    output = tmp_path / "output"
    arguments = [*_required(output), *(["--enable_rtc", *_prefix(prefix_path)] if requested else [])]
    monkeypatch.setattr(sys, "argv", ["causal-action", *arguments])
    monkeypatch.setenv("COSMOS_DEVICE", "cpu")
    monkeypatch.setenv("WAN_VAE_PATH", "/preexisting-vae.pth")
    # The test runs on CPU and must not import or install Ascend compatibility hooks.
    _fake_module(monkeypatch, "torch_npu", __path__=[])
    _fake_module(monkeypatch, "torch_npu.contrib", transfer_to_npu=object())
    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    class Dataset:
        def set_mode(self, mode):
            captured["dataset_mode"] = mode

        def __getitem__(self, index):
            return {
                "caption": "recorded observation",
                "sequence_plan": [SimpleNamespace(condition_frame_indexes_vision=[0])],
            }

    class Model:
        net = torch.nn.Linear(2, 2)
        input_caption_key = "caption"
        config = SimpleNamespace(vlm_config=SimpleNamespace(tokenizer="test-tokenizer"))

        def generate_samples_from_batch(self, batch, **options):
            assert not self.net.training and all(not p.requires_grad for p in self.net.parameters())
            assert not torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
            captured.update(batch=batch, options=options)
            return {
                "vision": [torch.ones(1, 1)],
                "action": [torch.ones(32, 3)],
                "rtc": {"enabled": reported} if requested else None,
                "inference_cache": {"full_cache": reported, "gradient_safe_prefix": reported},
                "preview_semantics": "test prediction",
            }

    def load_model(*args, **kwargs):
        captured["load_options"] = kwargs
        return Model(), None

    def get_dataset(**options):
        captured["dataset_options"] = options
        return SimpleNamespace(datasets=[Dataset()])

    _fake_module(
        monkeypatch,
        "cosmos_framework.data.generator.action.datasets.causal_action_factory",
        get_causal_action_dataset=get_dataset,
        latent_block_size=lambda actions, stride: 2,
    )
    _fake_module(
        monkeypatch, "cosmos_framework.data.generator.joint_dataloader", custom_collate_fn=lambda rows: rows[0]
    )
    _fake_module(
        monkeypatch,
        "cosmos_framework.inference.causal_action.outputs",
        real_actions=lambda result, batch: result["action"],
    )
    _fake_module(monkeypatch, "cosmos_framework.utils.generator.model_loader", load_model_from_checkpoint=load_model)
    from cosmos_framework import utils

    monkeypatch.setattr(utils, "distributed", SimpleNamespace(init=lambda: None), raising=False)
    cli.main()
    batch = captured["batch"]
    assert batch["causal_action_cache_mode"] == "full"
    assert batch["causal_action_current_block"] == 0
    assert batch["sequence_plan"][0].condition_frame_indexes_vision == [0]
    if requested:
        prefix = batch["causal_action_rtc"]
        assert isinstance(prefix, RTCPrefix)
        torch.testing.assert_close(prefix.previous_actions, previous, rtol=0, atol=0)
        assert (prefix.delay_steps, prefix.beta) == (2, 5.0)
    else:
        assert "causal_action_rtc" not in batch
    assert captured["options"] == {
        "num_steps": 10,
        "guidance": 1.0,
        "seed": [42],
        "causal_block_size": 2,
        "causal_history_blocks": 8,
        "causal_use_kv_cache": True,
    }
    metrics = json.loads((output / "metrics.json").read_text())
    sample = torch.load(output / "sample.pt", map_location="cpu", weights_only=True)
    assert metrics["rtc_enabled"] is sample["rtc_enabled"] is reported
    assert metrics["kv_cache"] is (not requested or reported)
    assert metrics["kv_cache_requested"] is True
