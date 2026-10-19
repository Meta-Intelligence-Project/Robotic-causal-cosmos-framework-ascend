# SPDX-License-Identifier: OpenMDW-1.1
"""Recover absolute physical targets using each action's original block anchor."""

import torch

from cosmos_framework.data.generator.action.block_state import prediction_block_ids


def real_actions(result, batch):
    outputs = []
    for i, action in enumerate(result["action"]):
        metadata = batch["sequence_plan"][i].causal_action_metadata
        if metadata is None or metadata.statistics is None or metadata.template is None:
            raise ValueError("Action export requires its template, block anchors and delta statistics")
        mask = metadata.action_mask.to(action.device)
        keep = mask.any(-1)
        if "generated_action_mask" in result:
            keep &= result["generated_action_mask"][i].to(keep.device)
        blocks = prediction_block_ids(metadata.action_frame_ids.to(action.device)[keep], metadata.block_size)
        delta = metadata.statistics.action.denormalize(action[keep], mask[keep])
        outputs.append(
            metadata.template.decode_action_delta(
                delta,
                metadata.anchors.to(action.device)[blocks],
                mask[0],
            )
        )
    return outputs


def normalized_action_metrics(result, batch):
    """Score one offline sample in training's normalized block-relative action space.

    Targets come directly from the training transform, including its clipping.
    Predictions stay in sampler space without extra clipping or re-normalization.
    Only generated, valid channels count; clean action conditions are excluded.
    """
    (prediction,) = result["action"]
    (target,) = batch["action"]
    (plan,) = batch["sequence_plan"]
    metadata = plan.causal_action_metadata
    if metadata is None or metadata.template is None or metadata.statistics is None:
        raise ValueError("Action evaluation requires the training template, statistics and block metadata")
    prediction = prediction.detach().float().cpu()
    target = target.detach().float().cpu()
    mask = metadata.action_mask.detach().bool().cpu().clone()
    if prediction.ndim != 2 or prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Predicted actions, normalized targets and action masks must have matching [T,D] shapes")
    if "generated_action_mask" in result:
        generated = result["generated_action_mask"][0].detach().bool().cpu()
        if generated.shape != (len(target),):
            raise ValueError("generated_action_mask must have shape [T]")
        mask &= generated[:, None]
    mask[plan.condition_frame_indexes_action] = False
    if not mask.any():
        raise ValueError("Action evaluation requires at least one generated, valid action channel")
    if not torch.isfinite(prediction[mask]).all() or not torch.isfinite(target[mask]).all():
        raise ValueError("Action evaluation received nonfinite values in valid channels")
    error = prediction.double() - target.double()

    def summarize(errors, valid):
        values = errors[valid]
        mse = values.square().mean().item()
        return dict(valid_elements=values.numel(), mse=mse, rmse=mse**0.5, mae=values.abs().mean().item())

    fields = {}
    for name, slots in metadata.template.fields.items():
        if mask[:, slots].any():
            fields[name] = summarize(error[:, slots], mask[:, slots])
    block_ids = prediction_block_ids(metadata.action_frame_ids.cpu(), metadata.block_size)
    blocks = []
    for block in block_ids[mask.any(-1)].unique().tolist():
        rows = block_ids == block
        blocks.append(dict(block=block, num_steps=int(mask[rows].any(-1).sum()), **summarize(error[rows], mask[rows])))
    return dict(
        semantics="Open-loop normalized action reconstruction error; not diffusion flow-matching training loss",
        space="Training normalized block-relative action; gripper/hand targets remain absolute",
        formula="sum((prediction - normalized_ground_truth)^2 over generated valid channels) / valid_elements",
        prediction_clipped=False,
        source=metadata.source,
        num_steps=int(mask.any(-1).sum()),
        num_valid_dimensions=int(mask.any(0).sum()),
        **summarize(error, mask),
        per_field=fields,
        per_block=blocks,
    )


def gather_source_slots(actions, mask):
    """Execution adapters select explicit slots, never a contiguous raw-width prefix."""
    if mask.ndim != 1 or mask.shape[0] != actions.shape[-1]:
        raise ValueError("Execution slot mask must match action width")
    return actions[..., mask.to(actions.device)]
