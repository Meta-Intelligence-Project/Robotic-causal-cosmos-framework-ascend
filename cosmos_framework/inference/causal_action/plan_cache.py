# SPDX-License-Identifier: OpenMDW-1.1
"""Immutable attention indices shared by one request's denoising steps and layers.

Each CFG template owns its own plan. Attach it after moving the template to the
inference device, and rebuild when replacing its layout. No model-global state
or cache survives the request.
"""

from dataclasses import dataclass

import torch

from cosmos_framework.data.generator.sequence_packing.causal_action import CausalActionLayout
from cosmos_framework.model.generator.mot.causal_action_cache import target_only_plan
from cosmos_framework.model.generator.mot.teacher_forcing_tnd import TNDPlan, build_tnd_plan


@dataclass(frozen=True)
class InferenceAttentionPlan:
    layout: CausalActionLayout
    max_kv_tokens: int
    full_plan: TNDPlan
    target_only: tuple[torch.Tensor, TNDPlan]


def prepare_attention_plan(template, *, max_kv_tokens: int = 131072) -> InferenceAttentionPlan:
    """Prepare one CFG branch's indices, without caching any input-dependent values."""
    teacher_forcing = template.teacher_forcing
    if teacher_forcing is None or not isinstance(teacher_forcing.layout, CausalActionLayout):
        raise ValueError("Action inference plan caching requires a causal action layout")
    layout = teacher_forcing.layout
    existing = getattr(teacher_forcing, "inference_attention_plan", None)
    if existing is not None and existing.layout is layout and existing.max_kv_tokens == max_kv_tokens:
        return existing
    full_plan = build_tnd_plan(layout, device=layout.gen_query_indexes.device, max_kv_tokens=max_kv_tokens)
    plan = InferenceAttentionPlan(layout, max_kv_tokens, full_plan, target_only_plan(layout, full_plan))
    teacher_forcing.inference_attention_plan = plan
    return plan
