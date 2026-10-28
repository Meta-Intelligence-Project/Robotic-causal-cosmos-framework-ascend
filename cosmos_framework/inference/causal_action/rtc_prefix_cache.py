# SPDX-License-Identifier: OpenMDW-1.1
"""Request-local fixed K/V reuse with a fresh current-token autograd graph.

A no-grad full prefill establishes fixed conditions, including every layer's
normalized text keys. Even the first public result is then recomputed with a
compact, differentiable current-block forward. Only inputs of current noisy
video/action are differentiated; fixed conditions and model weights must stay
unchanged. This is separate from the existing inference-only prefix cache.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from numbers import Integral

import torch

from cosmos_framework.data.generator.sequence_packing.runtime import (
    from_mode_splits,
    get_all_seq,
    get_gen_seq,
    get_und_seq,
    init_sequence_pack,
)
from cosmos_framework.inference.causal_action.prefix_cache import current_block_plan
from cosmos_framework.model.generator.mot.teacher_forcing_tnd import TNDChunk, TNDPlan, teacher_forcing_tnd_attention


def _fixed_dependency_closure(layout, full_plan, current_plan, current_packed):
    """Prove fixed current-KV ancestors never read any changing stream-1 token."""
    roles = layout.stream_ids.detach().cpu().tolist()
    gen = layout.gen_query_indexes.detach().cpu().tolist()
    if full_plan.num_keys != len(roles) or full_plan.num_queries != len(gen):
        raise ValueError("RTC prefix plan dimensions differ from the causal layout")
    current = set(current_packed.detach().cpu().tolist())
    direct = {key for chunk in current_plan.chunks for key in chunk.kv_indexes.detach().cpu().tolist()}
    if any(roles[key] == 1 and key not in current for key in direct):
        raise ValueError("Current queries cannot depend on historical noisy actions")
    groups, query_group = [], {}
    for chunk in full_plan.chunks:
        queries, keys = chunk.query_indexes.detach().cpu().tolist(), chunk.kv_indexes.detach().cpu().tolist()
        q0 = k0 = 0
        for q1, k1 in zip(chunk.query_ends, chunk.kv_ends, strict=True):
            group = len(groups)
            groups.append(tuple(keys[k0:k1]))
            for query in queries[q0:q1]:
                packed = gen[query]
                if packed in query_group:
                    raise ValueError("RTC prefix plan repeats a fixed query")
                query_group[packed] = group
            q0, k0 = q1, k1
    pending = list(direct - current)
    fixed, processed_groups = set(), set()
    while pending:
        key = pending.pop()
        if key in fixed:
            continue
        if not 0 <= key < len(roles) or roles[key] not in (-1, 0, 2, 3):
            raise ValueError("Fixed prefix dependency must not contain a noisy stream-1 token")
        fixed.add(key)
        if roles[key] == -1:
            continue  # UND attention is causal text-only in the supported MoT path.
        if key not in query_group:
            raise ValueError("A fixed K/V row has no causal query dependency group")
        group = query_group[key]
        if group not in processed_groups:
            processed_groups.add(group)
            ancestors = groups[group]
            if any(roles[ancestor] == 1 for ancestor in ancestors):
                raise ValueError("Fixed prefix K/V depends on noisy stream-1 inputs")
            pending.extend(ancestor for ancestor in ancestors if ancestor not in fixed)
    return torch.tensor(sorted(fixed), device=current_packed.device, dtype=torch.long)


def _compact_key_plan(plan, fixed_packed, current_packed):
    """Keep group key ordering while dropping unrelated rows from K/V storage."""
    visible = torch.cat((fixed_packed, current_packed))
    mapping = torch.full((plan.num_keys,), -1, device=visible.device, dtype=torch.long)
    mapping = mapping.index_copy(0, visible, torch.arange(len(visible), device=visible.device))
    chunks = []
    for chunk in plan.chunks:
        indexes = mapping.index_select(0, chunk.kv_indexes)
        if (indexes < 0).any():
            raise ValueError("RTC compact attention is missing a current query's K/V dependency")
        chunks.append(TNDChunk(chunk.query_indexes, indexes, chunk.query_ends, chunk.kv_ends))
    return TNDPlan(tuple(chunks), plan.num_queries, len(visible))


@dataclass
class _RTCEntry:
    layout: object
    gen_indexes: torch.Tensor
    packed_indexes: torch.Tensor
    fixed_indexes: torch.Tensor
    plan: TNDPlan
    compact_metadata: dict
    fixed_inputs: torch.Tensor
    fixed_positions: torch.Tensor
    layers: dict = field(default_factory=dict)


@contextmanager
def action_rtc_prefix_cache(net):
    """Cache one request's fixed prefix; only current-token input VJPs survive.

    Set controller['key']=(current_block, cfg_branch) before each network call.
    Clear entries whenever conditions change. All parameters must be frozen and
    the network must be in eval mode. No cache/patch survives context exit, and
    a model must not be used concurrently while the context is active.
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError("RTC prefix input gradients require no_grad/enable_grad, not inference_mode")
    parameters = tuple(net.parameters())
    if net.training or any(parameter.requires_grad for parameter in parameters):
        raise ValueError("RTC prefix caching requires an eval network with frozen model parameters")
    versions = tuple(parameter._version for parameter in parameters)
    language_model = net.language_model
    if getattr(language_model.model.rotary_emb, "rope_type", "default") != "default":
        raise ValueError("RTC prefix caching supports fixed default RoPE only")
    if getattr(language_model.model.config, "num_experts", 0):
        raise ValueError("RTC prefix caching supports dense language models only")
    if getattr(language_model, "_rtc_prefix_cache_active", False):
        raise RuntimeError("RTC prefix cache contexts cannot be nested on one model")
    layers = language_model.model.layers
    original_forward = language_model.forward
    original_dispatch = [(layer.self_attn, layer.self_attn.dispatch_attention_fn) for layer in layers]
    controller = {
        "key": None,
        "entries": {},
        "stats": {
            "prefill_calls": 0,
            "replay_calls": 0,
            "compact_calls": 0,
            "grad_compact_calls": 0,
            "active_tokens": {},
            "fixed_tokens": {},
        },
    }
    phase = {}

    def wrap_dispatch(original, layer_index):
        def dispatch(
            query,
            key,
            value,
            attention_mask,
            natten_metadata=None,
            memory_value=None,
            packed_key_states_normalized=None,
        ):
            entry = phase["entry"]
            normalized = packed_key_states_normalized if packed_key_states_normalized is not None else key
            if phase["prefill"]:
                result = original(
                    query,
                    key,
                    value,
                    attention_mask,
                    natten_metadata=natten_metadata,
                    memory_value=memory_value,
                    packed_key_states_normalized=packed_key_states_normalized,
                )
                # Clone only fixed ancestors; never retain full/current tensors or graphs.
                entry.layers[layer_index] = (
                    get_all_seq(normalized).index_select(0, entry.fixed_indexes).detach().clone(),
                    get_all_seq(value).index_select(0, entry.fixed_indexes).detach().clone(),
                )
                return result
            fixed_key, fixed_value = entry.layers[layer_index]
            keys = torch.cat((fixed_key, get_gen_seq(normalized)), dim=0)
            values = torch.cat((fixed_value, get_gen_seq(value)), dim=0)
            result = teacher_forcing_tnd_attention(get_gen_seq(query), keys, values, entry.plan).flatten(-2, -1)
            return from_mode_splits(result.new_empty((0, result.shape[-1])), result, query), None

        return dispatch

    def forward(pack, attention_mask, position_ids, natten_metadata_list=None, memory=None):
        if torch.is_inference_mode_enabled():
            raise RuntimeError("RTC prefix input gradients cannot run in inference_mode")
        if net.training or any(parameter.requires_grad for parameter in parameters):
            raise ValueError("RTC prefix caching supports frozen eval input gradients, not parameter training")
        if tuple(parameter._version for parameter in parameters) != versions:
            raise ValueError("Model weights changed inside an RTC prefix request")
        if memory is not None or natten_metadata_list is not None or pack.get("is_sharded", False):
            raise ValueError("RTC prefix caching requires unsharded grouped_tnd without external memory")
        if attention_mask.dense_mode != "grouped_tnd" or not getattr(attention_mask.layout, "includes_action", False):
            raise ValueError("RTC prefix caching requires a causal-action grouped_tnd layout")
        cache_key = controller["key"]
        if (
            not isinstance(cache_key, tuple)
            or len(cache_key) != 2
            or isinstance(cache_key[0], bool)
            or not isinstance(cache_key[0], Integral)
            or cache_key[0] < 0
        ):
            raise ValueError("Set controller['key']=(current_block, cfg_branch) before RTC inference")
        try:
            hash(cache_key)
        except TypeError as exc:
            raise ValueError("RTC CFG branch keys must be hashable") from exc
        layout = attention_mask.layout
        entry = controller["entries"].get(cache_key)
        if entry is None:
            target, packed, current_plan = current_block_plan(layout, attention_mask.tnd_plan, int(cache_key[0]))
            fixed = _fixed_dependency_closure(layout, attention_mask.tnd_plan, current_plan, packed)
            plan = _compact_key_plan(current_plan, fixed, packed)
            n = target.numel()
            meta = init_sequence_pack([n], [0, n], ["causal", "full"], target.device)
            meta.update(max_num_tokens=n, is_sharded=False)
            with torch.no_grad():
                inputs = get_all_seq(pack).index_select(0, fixed).detach().clone()
                positions = position_ids.index_select(-1, fixed).detach().clone()
            entry = _RTCEntry(layout, target, packed, fixed, plan, meta, inputs, positions)
            phase.update(prefill=True, entry=entry)
            with torch.no_grad():
                original_forward(
                    pack,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    natten_metadata_list=None,
                    memory=None,
                )
            if len(entry.layers) != len(layers):
                raise RuntimeError("RTC prefix prefill did not populate every transformer layer")
            controller["entries"][cache_key] = entry
            controller["stats"]["prefill_calls"] += 1
            controller["stats"]["active_tokens"][str(cache_key)] = n
            controller["stats"]["fixed_tokens"][str(cache_key)] = len(fixed)
        else:
            if layout is not entry.layout:
                raise ValueError("RTC prefix layout changed within a request; clear entries first")
            with torch.no_grad():
                fixed_inputs = get_all_seq(pack).index_select(0, entry.fixed_indexes)
                positions = position_ids.index_select(-1, entry.fixed_indexes)
                if (
                    fixed_inputs.dtype != entry.fixed_inputs.dtype
                    or fixed_inputs.device != entry.fixed_inputs.device
                    or not torch.equal(fixed_inputs, entry.fixed_inputs)
                    or not torch.equal(positions, entry.fixed_positions)
                ):
                    raise ValueError(
                        "RTC fixed prefix inputs/positions changed; clear entries before reusing the cache"
                    )
        phase.update(prefill=False, entry=entry)
        active = get_gen_seq(pack).index_select(0, entry.gen_indexes)
        compact = from_mode_splits(active.new_empty((0, active.shape[-1])), active, entry.compact_metadata)
        output, metadata = original_forward(
            compact,
            attention_mask=attention_mask,
            position_ids=position_ids.index_select(-1, entry.packed_indexes),
            natten_metadata_list=None,
            memory=None,
        )
        # Build a new full head layout every call; no current graph enters the cache.
        full_gen = (
            get_gen_seq(pack).new_zeros(get_gen_seq(pack).shape).index_copy(0, entry.gen_indexes, get_gen_seq(output))
        )
        full_und = get_und_seq(pack).new_zeros(get_und_seq(pack).shape)
        restored = from_mode_splits(full_und, full_gen, pack)
        restored.pop("all_seq", None)
        stats = controller["stats"]
        stats["replay_calls"] += 1
        stats["compact_calls"] += 1
        stats["grad_compact_calls"] += int(torch.is_grad_enabled())
        return restored, metadata

    language_model._rtc_prefix_cache_active = True
    try:
        language_model.forward = forward
        for index, (attention, original) in enumerate(original_dispatch):
            attention.dispatch_attention_fn = wrap_dispatch(original, index)
        yield controller
    finally:
        language_model.forward = original_forward
        for attention, original in original_dispatch:
            attention.dispatch_attention_fn = original
        controller["entries"].clear()
        phase.clear()
        delattr(language_model, "_rtc_prefix_cache_active")
