# SPDX-License-Identifier: OpenMDW-1.1
"""Physical RTC exports preserve the generated mask and committed commands."""

from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate55
from cosmos_framework.data.generator.action.block_state import BlockStateMetadata, BlockStatistics, Quantiles
from cosmos_framework.inference.causal_action.outputs import real_actions


def _current_block_result():
    template = ActionStateTemplate55()
    valid = torch.zeros(template.width, dtype=torch.bool)
    valid[[0, 17, 18, 19, 20, 28]] = True
    action_mask = torch.zeros(4, template.width, dtype=torch.bool)
    action_mask[2:] = valid
    anchors = torch.zeros(2, template.width)
    anchors[:, 20] = 1.0
    anchors[:, 0] = torch.tensor([1.0, 10.0])
    bounds = Quantiles(-torch.ones(template.width), torch.ones(template.width), valid.clone())
    metadata = BlockStateMetadata(
        block_size=2,
        history_blocks=1,
        action_frame_ids=torch.tensor([1, 2, 3, 4]),
        action_mask=action_mask,
        states=anchors.clone(),
        state_mask=valid.expand(2, -1).clone(),
        state_action_indexes=torch.tensor([0, 2]),
        anchors=anchors,
        state_latent_indexes=torch.tensor([0, 2]),
        statistics=BlockStatistics(bounds, bounds),
        template=template,
    )
    normalized = torch.zeros(4, template.width)
    normalized[2:, 0] = torch.tensor([0.25, 0.5])
    normalized[2:, 20] = 1.0
    normalized[2:, 28] = torch.tensor([0.6, 0.7])
    result = {
        "action": [normalized],
        "generated_action_mask": [torch.tensor([False, False, True, True])],
    }
    batch = {"sequence_plan": [SimpleNamespace(causal_action_metadata=metadata)]}
    return result, batch, valid


def test_current_actions_decode_when_historical_action_mask_is_empty():
    result, batch, valid = _current_block_result()
    physical = real_actions(result, batch)[0]
    assert physical.shape == (2, 55)
    torch.testing.assert_close(physical[:, 0], torch.tensor([10.25, 10.5]))
    torch.testing.assert_close(physical[:, 28], torch.tensor([0.6, 0.7]))
    torch.testing.assert_close(physical[:, 17:21], torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(2, -1))
    assert not physical[:, ~valid].any()


def test_committed_physical_prefix_is_exact_after_history_filtering_and_decode():
    result, batch, valid = _current_block_result()
    uncommitted = real_actions(result, batch)[0]
    committed = torch.zeros(1, 55)
    committed[0, 0] = 123.456789  # Not recoverable by clipping to the training quantiles.
    committed[0, 28] = 0.31415927
    committed[0, 17:21] = torch.tensor([0.1, 0.2, 0.3, 0.92736185])
    original = committed.clone()
    result["rtc_committed_actions"] = [committed]
    physical = real_actions(result, batch)[0]
    assert torch.equal(physical[:1, valid], committed[:, valid])
    assert torch.equal(physical[1:], uncommitted[1:])
    assert not physical[:, ~valid].any()
    assert torch.equal(committed, original)


def test_generated_actions_with_inconsistent_valid_channels_are_rejected():
    result, batch, _ = _current_block_result()
    batch["sequence_plan"][0].causal_action_metadata.action_mask[-1, 1] = True
    with pytest.raises(ValueError, match="mask|valid"):
        real_actions(result, batch)


def test_no_generated_actions_exports_an_empty_physical_tensor():
    result, batch, _ = _current_block_result()
    result["generated_action_mask"][0].fill_(False)
    physical = real_actions(result, batch)[0]
    assert physical.shape == (0, 55)


@pytest.mark.parametrize("shape", [(55,), (1, 54), (3, 55)])
def test_committed_prefix_must_fit_generated_physical_output(shape):
    result, batch, _ = _current_block_result()
    result["rtc_committed_actions"] = [torch.zeros(shape)]
    with pytest.raises(ValueError, match="committed prefix"):
        real_actions(result, batch)
