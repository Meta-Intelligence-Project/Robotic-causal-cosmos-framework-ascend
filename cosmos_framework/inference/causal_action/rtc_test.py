# SPDX-License-Identifier: OpenMDW-1.1
"""RTC equations, joint input-VJP and physical action-prefix rebasing checks."""

import math
from dataclasses import replace

import pytest
import torch

from cosmos_framework.data.generator.action.action_state_template import ActionStateTemplate55
from cosmos_framework.data.generator.action.block_state import BlockStateMetadata, BlockStatistics, Quantiles
from cosmos_framework.inference.causal_action.rtc import RTCPrefix, guided_velocity, prefix_weights, prepare_prefix


def _metadata():
    template = ActionStateTemplate55()
    mask = torch.zeros(55, dtype=torch.bool)
    mask[[0, 14, 15, 16, 17, 18, 19, 20, 28]] = True
    state_mask = mask.clone()
    state_mask[28] = False
    anchors = torch.zeros(2, 55)
    anchors[:, 20] = 1
    anchors[:, 0] = torch.tensor([8.0, 10.0])
    anchors[1, 14] = 0.5
    anchors[1, 17:21] = torch.tensor([0.0, 0.0, 2**-0.5, 2**-0.5])
    anchors[:, 28] = float("nan")  # Absolute gripper targets need no gripper anchor.
    low, high = -torch.ones(55), torch.ones(55)
    low[14:17], high[14:17] = -0.01, 0.01  # metres, not millimetres
    low[28], high[28] = 0.0, 1.0
    statistics = BlockStatistics(
        Quantiles(-torch.ones(55), torch.ones(55), state_mask),
        Quantiles(low, high, mask),
    )
    return BlockStateMetadata(
        block_size=2,
        history_blocks=32,
        action_frame_ids=torch.tensor([1, 1, 2, 2, 3, 3, 4, 4]),
        action_mask=mask.expand(8, -1).clone(),
        states=torch.zeros(2, 55),
        state_mask=state_mask.expand(2, -1).clone(),
        state_action_indexes=torch.tensor([0, 4]),
        anchors=anchors,
        state_latent_indexes=torch.tensor([0, 2]),
        template=template,
        statistics=statistics,
    )


def _prefix(count=2, *, delay=1, beta=5.0):
    actions = torch.full((count, 55), float("nan"))
    actions[:, [0, 14, 15, 16, 17, 18, 19, 20, 28]] = 0
    actions[:, 0] = 12.5
    actions[:, 14] = 0.52
    actions[:, 19] = 1  # Absolute 180-degree z rotation.
    actions[:, 28] = 0.8
    return RTCPrefix(actions, delay, beta)


def test_soft_weights_match_equation_five_and_boundaries():
    expected = [1.0, 1.0]
    expected += [c * math.expm1(c) / math.expm1(1.0) for c in (0.75, 0.5, 0.25)]
    expected += [0.0] * 3
    torch.testing.assert_close(prefix_weights(2, 5, 8), torch.tensor(expected))
    assert torch.equal(prefix_weights(0, 0, 8), torch.zeros(8))
    assert torch.equal(prefix_weights(8, 8, 8), torch.ones(8))
    assert torch.equal(prefix_weights(3, 3, 8), torch.tensor([1.0] * 3 + [0.0] * 5))
    all_soft = prefix_weights(0, 8, 8)
    assert 0 < all_soft[-1] < all_soft[0] < 1
    assert (all_soft[1:] < all_soft[:-1]).all()


@pytest.mark.parametrize("arguments", [(-1, 2, 4), (3, 2, 4), (0, 5, 4), (0, 0, 0), (True, 2, 4), (1.0, 2, 4)])
def test_soft_weights_reject_invalid_lengths(arguments):
    with pytest.raises(ValueError):
        prefix_weights(*arguments)


def test_prefix_rebases_rotation_joint_and_metres_at_current_anchor_without_clipping():
    metadata = _metadata()
    prefix = _prefix()
    prepared = prepare_prefix(prefix, metadata, 1, device="cpu")
    assert prepared.action_indexes.tolist() == [4, 5, 6, 7]
    assert prepared.normalized_targets.shape == (4, 55)
    # 12.5 - new anchor 10.0, not old anchor 8.0; training would clip 2.5 to 1.5.
    torch.testing.assert_close(prepared.normalized_targets[:2, 0], torch.full((2,), 2.5))
    # 0.52m - 0.50m = 0.02m -> 2.0 in the declared +/-0.01m delta range.
    torch.testing.assert_close(prepared.normalized_targets[:2, 14], torch.full((2,), 2.0))
    expected_q = torch.tensor([0.0, 0.0, 2**-0.5, 2**-0.5]).expand(2, -1)
    torch.testing.assert_close(prepared.normalized_targets[:2, 17:21], expected_q)
    torch.testing.assert_close(prepared.normalized_targets[:2, 28], torch.full((2,), 0.6))
    action_mask = metadata.action_mask[0]
    assert torch.isfinite(prepared.absolute_prefix).all()
    assert not prepared.absolute_prefix[:, ~action_mask].any()
    assert not prepared.normalized_targets[:, ~action_mask].any()
    assert not prepared.weights[:, ~action_mask].any()
    assert not prepared.weights[2:].any()
    assert not prepared.normalized_targets[2:].any()
    delta = metadata.statistics.action.denormalize(prepared.normalized_targets[:2], action_mask)
    decoded = metadata.template.decode_action_delta(delta, metadata.anchors[1].nan_to_num(), action_mask)
    torch.testing.assert_close(decoded, prepared.absolute_prefix)
    assert prefix.previous_actions[:, ~action_mask].isnan().all()  # Caller storage is not mutated.


def test_empty_prefix_is_unconstrained_and_output_is_detached():
    prepared = prepare_prefix(_prefix(0, delay=0), _metadata(), 1, device="cpu")
    assert prepared.absolute_prefix.shape == (0, 55)
    assert not prepared.weights.any()
    assert not prepared.normalized_targets.any()
    assert not prepared.normalized_targets.requires_grad


def test_physical_prefix_keeps_original_valid_quaternion_values_exactly():
    metadata, prefix = _metadata(), _prefix()
    prefix.previous_actions[:, 17:21] = torch.tensor([0.0, 0.0, 2**-0.5, 2**-0.5]) * (1 + 1e-6)
    mask = metadata.action_mask[0]
    original = prefix.previous_actions.clone()
    sanitized = metadata.template.sanitize(original, mask)
    assert not torch.equal(original[:, 17:21], sanitized[:, 17:21])
    prepared = prepare_prefix(prefix, metadata, 1, device="cpu")
    assert torch.equal(prepared.absolute_prefix[:, mask], original[:, mask])
    assert not prepared.absolute_prefix[:, ~mask].any()
    # The normalized guidance target still uses the template's unit rotations.
    delta = metadata.statistics.action.denormalize(prepared.normalized_targets[:2], mask)
    decoded = metadata.template.decode_action_delta(delta, metadata.anchors[1].nan_to_num(), mask)
    torch.testing.assert_close(decoded, sanitized)


def test_constant_statistics_are_unconstrained_only_when_target_is_representable():
    metadata = _metadata()
    metadata.statistics.action.low[0] = metadata.statistics.action.high[0] = 2.5
    result = prepare_prefix(_prefix(), metadata, 1, device="cpu")
    assert not result.weights[:, 0].any()
    assert not result.normalized_targets[:, 0].any()
    metadata.statistics.action.high[0] = metadata.statistics.action.low[0] = 1.5
    with pytest.raises(ValueError, match="constant"):
        prepare_prefix(_prefix(), metadata, 1, device="cpu")


@pytest.mark.parametrize(
    "make_prefix",
    [
        lambda: RTCPrefix(torch.zeros(2, 55), -1),
        lambda: RTCPrefix(torch.zeros(2, 55), 3),
        lambda: RTCPrefix(torch.zeros(2, 55), True),
        lambda: RTCPrefix(torch.zeros(2, 55), 1.0),
        lambda: RTCPrefix(torch.zeros(2, 55), 1, float("nan")),
        lambda: RTCPrefix(torch.zeros(2, 55), 1, -1),
        lambda: RTCPrefix(torch.zeros(2, 55), 1, True),
        lambda: RTCPrefix(torch.zeros(55), 1),
        lambda: RTCPrefix(torch.zeros(2, 55, dtype=torch.long), 1),
    ],
)
def test_prefix_constructor_rejects_invalid_inputs(make_prefix):
    with pytest.raises(ValueError):
        make_prefix()


@pytest.mark.parametrize("current", [-1, True, 1.0, 2])
def test_prepare_rejects_invalid_current_block(current):
    with pytest.raises(ValueError):
        prepare_prefix(_prefix(), _metadata(), current, device="cpu")


def test_prepare_rejects_excess_prefix_wrong_width_and_missing_metadata():
    with pytest.raises(ValueError, match="horizon"):
        prepare_prefix(_prefix(5), _metadata(), 1, device="cpu")
    with pytest.raises(ValueError, match="width"):
        prepare_prefix(RTCPrefix(torch.zeros(2, 54), 1), _metadata(), 1, device="cpu")
    with pytest.raises(ValueError, match="statistics"):
        prepare_prefix(_prefix(), replace(_metadata(), statistics=None), 1, device="cpu")


@pytest.mark.parametrize("field", ["action", "anchor", "coverage", "bounds", "state_mask", "action_mask", "frame_ids"])
def test_prepare_rejects_invalid_valid_values_and_contracts(field):
    metadata, prefix = _metadata(), _prefix()
    if field == "action":
        prefix.previous_actions[0, 0] = float("inf")
    elif field == "anchor":
        metadata.anchors[1, 0] = float("nan")
    elif field == "coverage":
        metadata.statistics.action.valid = metadata.statistics.action.valid.clone()
        metadata.statistics.action.valid[0] = False
    elif field == "bounds":
        metadata.statistics.action.high[0] = float("inf")
    elif field == "state_mask":
        metadata.state_mask[1, 0] = False
    elif field == "action_mask":
        metadata.action_mask = metadata.action_mask.float()
    else:
        metadata.action_frame_ids = metadata.action_frame_ids.float()
    with pytest.raises(ValueError):
        prepare_prefix(prefix, metadata, 1, device="cpu")


def _nonlinear_velocity(x):
    return torch.stack((x[0].square() + 0.5 * x[1], x[0].sin() + 2 * x[1], 0.1 + x[1] * x[2]))


@pytest.mark.parametrize("sigma,beta", [(0.5, 5.0), (0.2, 0.3), (1.0, 5.0)])
def test_joint_nonlinear_input_vjp_matches_analytic_jacobian(sigma, beta):
    x = torch.tensor([0.2, -0.3, 0.4], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([0.0, 0.8, -0.5], dtype=x.dtype)
    weights = torch.tensor([0.0, 1.0, 0.4], dtype=x.dtype)  # First coordinate is video.
    v = _nonlinear_velocity(x.detach())
    jacobian = torch.tensor([[0.4, 0.5, 0.0], [math.cos(0.2), 2.0, 0.0], [0.0, 0.4, -0.3]], dtype=x.dtype)
    clean = x.detach() - sigma * v
    error = (target - clean) * weights
    vjp = (torch.eye(3, dtype=x.dtype) - sigma * jacobian).T @ error
    tau = 1 - sigma
    gain = beta if sigma == 1 else min(beta, (tau**2 + sigma**2) / (tau * sigma))
    with torch.no_grad():
        actual = guided_velocity(x, _nonlinear_velocity, sigma, target, weights, beta)
    torch.testing.assert_close(actual, v - gain * vjp)
    assert actual[0] != v[0]  # Action-only error must retain joint video-input derivatives.
    assert not actual.requires_grad
    assert x.grad is None


def test_guidance_has_correct_direction_when_cosmos_sigma_decreases():
    sample, target, weights = torch.tensor([0.5]), torch.tensor([2.0]), torch.ones(1)
    velocity = guided_velocity(sample, torch.zeros_like, 1.0, target, weights, 5.0)
    assert velocity.item() == -7.5
    next_sample = sample - 0.1 * velocity
    assert (next_sample - target).abs().item() < (sample - target).abs().item()


@pytest.mark.parametrize("sigma,beta,weights", [(0.0, 5.0, [1.0]), (0.5, 0.0, [1.0]), (0.5, 5.0, [0.0])])
def test_zero_sigma_beta_or_weights_uses_plain_velocity_without_grad(sigma, beta, weights):
    def velocity(x):
        assert not torch.is_grad_enabled()
        return x.square()

    result = guided_velocity(torch.tensor([2.0]), velocity, sigma, torch.zeros(1), torch.tensor(weights), beta)
    assert result.item() == 4.0
    assert not result.requires_grad


def test_guidance_never_accumulates_or_changes_parameter_gradients():
    parameter = torch.nn.Parameter(torch.tensor(0.7))
    x = torch.tensor([0.4, -0.2])
    before = parameter.detach().clone()
    for existing_grad in (None, torch.tensor(3.0)):
        parameter.grad = existing_grad
        output = guided_velocity(x, lambda value: parameter * value.square(), 0.3, torch.ones(2), torch.ones(2), 5.0)
        assert torch.isfinite(output).all()
        assert parameter.grad is existing_grad
        torch.testing.assert_close(parameter.detach(), before)


def test_guidance_rejects_inference_mode_explicitly():
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
        guided_velocity(torch.ones(1), torch.square, 0.5, torch.zeros(1), torch.ones(1), 5.0)


@pytest.mark.parametrize("sigma", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_guidance_rejects_invalid_sigma(sigma):
    with pytest.raises(ValueError):
        guided_velocity(torch.ones(1), torch.square, sigma, torch.zeros(1), torch.ones(1), 5.0)


@pytest.mark.parametrize("beta", [-1.0, float("nan"), float("inf"), True])
def test_guidance_rejects_invalid_beta(beta):
    with pytest.raises(ValueError):
        guided_velocity(torch.ones(1), torch.square, 0.5, torch.zeros(1), torch.ones(1), beta)


@pytest.mark.parametrize("bad", ["shape", "sample_nan", "target_nan", "weight_nan", "weight_negative", "weight_large"])
def test_guidance_rejects_invalid_flat_vectors(bad):
    sample, target, weights = torch.ones(2), torch.zeros(2), torch.ones(2)
    if bad == "shape":
        target = torch.zeros(1)
    elif bad == "sample_nan":
        sample[0] = float("nan")
    elif bad == "target_nan":
        target[0] = float("nan")
    elif bad == "weight_nan":
        weights[0] = float("nan")
    elif bad == "weight_negative":
        weights[0] = -0.1
    else:
        weights[0] = 1.1
    with pytest.raises(ValueError):
        guided_velocity(sample, torch.square, 0.5, target, weights, 5.0)


@pytest.mark.parametrize("velocity", [lambda x: None, lambda x: x[:0], lambda x: x * float("nan")])
def test_guidance_rejects_invalid_network_velocity(velocity):
    with pytest.raises(ValueError):
        guided_velocity(torch.ones(2), velocity, 0.5, torch.zeros(2), torch.ones(2), 5.0)
