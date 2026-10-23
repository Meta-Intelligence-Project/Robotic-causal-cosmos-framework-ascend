# SPDX-License-Identifier: OpenMDW-1.1
"""Request-local fixed-prefix K/V prefill and compact current-block replay.

Only current-block stream-1 predictions are an output contract of this cache.
Historical stream-1 actions have changing timestep embeddings, but cannot be
read by the current block. They are dead branches, not cached constant outputs.
Training and input-VJP callers continue to use the original network path.
Compacting matrix shapes can change BF16 rounding on accelerators; mathematical
equivalence does not imply bitwise equality or validated downstream task quality.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from cosmos_framework.data.generator.sequence_packing.runtime import (
    from_mode_splits,
    get_all_seq,
    get_gen_seq,
    init_sequence_pack,
)
from cosmos_framework.model.generator.mot.teacher_forcing_tnd import (
    TNDChunk,
    TNDPlan,
    teacher_forcing_tnd_attention,
)


@dataclass
class PrefixEntry:
    layout: object
    gen_indexes: torch.Tensor
    packed_indexes: torch.Tensor
    plan: TNDPlan
    compact_metadata: dict
    layers: dict
    output: object = None


def current_block_plan(layout, plan, block_index):
    """Select current noisy queries; preserve each group's exact original KV set."""
    device = layout.gen_query_indexes.device
    roles, blocks, gen = (
        x.detach().cpu().tolist() for x in (layout.stream_ids, layout.block_ids, layout.gen_query_indexes)
    )
    selected = [i for i, packed in enumerate(gen) if roles[packed] == 1 and blocks[packed] == block_index]
    if not selected:
        raise ValueError("Prefix replay needs at least one current-block noisy token")
    mapping = {old: new for new, old in enumerate(selected)}
    chunks = []
    for chunk in plan.chunks:
        queries, keys = chunk.query_indexes.cpu().tolist(), chunk.kv_indexes.cpu().tolist()
        qs, ks, qe, ke = [], [], [], []
        q0 = k0 = 0
        for q1, k1 in zip(chunk.query_ends, chunk.kv_ends, strict=True):
            chosen = [mapping[q] for q in queries[q0:q1] if q in mapping]
            if chosen:
                group_keys = keys[k0:k1]
                if any(roles[k] == 1 and blocks[k] != block_index for k in group_keys):
                    raise ValueError("Current queries must not depend on historical noisy actions")
                qs.extend(chosen)
                ks.extend(group_keys)
                qe.append(len(qs))
                ke.append(len(ks))
            q0, k0 = q1, k1
        if qs:
            chunks.append(
                TNDChunk(
                    torch.tensor(qs, device=device, dtype=torch.long),
                    torch.tensor(ks, device=device, dtype=torch.long),
                    tuple(qe),
                    tuple(ke),
                )
            )
    target = torch.tensor(selected, device=device, dtype=torch.long)
    packed = layout.gen_query_indexes[target]
    return target, packed, TNDPlan(tuple(chunks), len(selected), plan.num_keys)


@contextmanager
def action_prefix_cache(net):
    """Temporarily install compact inference on one model, isolated by block/CFG key.

    The model must not be used concurrently during this context, like the legacy
    action_attention_cache. No values or patches survive context exit, including
    exception exits. Set controller['key']=(block_index, cfg_branch) per call.
    Observations, states, text, positions, and model weights must remain fixed
    for a key; clear controller['entries'] before changing any of them.
    """
    if net.training:
        raise ValueError("Prefix caching requires an eval-mode network")
    language_model = net.language_model
    if getattr(language_model.model.rotary_emb, "rope_type", "default") != "default":
        raise ValueError("Prefix caching currently supports fixed default RoPE only")
    if getattr(language_model.model.config, "num_experts", 0):
        raise ValueError("Prefix caching currently supports dense language models only")
    layers = language_model.model.layers
    controller = {"key": None, "entries": {}, "stats": {"prefill_calls": 0, "replay_calls": 0, "active_tokens": {}}}
    original_forward = language_model.forward
    original_dispatch = [(layer.self_attn, layer.self_attn.dispatch_attention_fn) for layer in layers]
    phase = {"prefill": True, "entry": None}

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
                # Gen queries read the UND-normalized K branch, not raw text K.
                normalized = packed_key_states_normalized if packed_key_states_normalized is not None else key
                entry.layers[layer_index] = (get_all_seq(normalized).detach(), get_all_seq(value).detach())
                return result
            keys, values = entry.layers[layer_index]
            current_key = packed_key_states_normalized if packed_key_states_normalized is not None else key
            keys.index_copy_(0, entry.packed_indexes, get_gen_seq(current_key))
            values.index_copy_(0, entry.packed_indexes, get_gen_seq(value))
            result = teacher_forcing_tnd_attention(get_gen_seq(query), keys, values, entry.plan).flatten(-2, -1)
            return from_mode_splits(result.new_empty((0, result.shape[-1])), result, query), None

        return dispatch

    def forward(pack, attention_mask, position_ids, natten_metadata_list=None, memory=None):
        if torch.is_grad_enabled():
            raise RuntimeError("Prefix caching is inference-only; disable it for input-VJP calls")
        if memory is not None or natten_metadata_list is not None or pack.get("is_sharded", False):
            raise ValueError("Prefix caching requires unsharded grouped_tnd attention without external memory")
        if attention_mask.dense_mode != "grouped_tnd" or not getattr(attention_mask.layout, "includes_action", False):
            raise ValueError("Prefix caching requires a causal action grouped_tnd layout")
        cache_key = controller["key"]
        if not isinstance(cache_key, tuple) or len(cache_key) != 2:
            raise ValueError("Set the current block and CFG branch before prefix inference")
        entry = controller["entries"].get(cache_key)
        if entry is None:
            target, packed, plan = current_block_plan(attention_mask.layout, attention_mask.tnd_plan, cache_key[0])
            n = target.numel()
            meta = init_sequence_pack([n], [0, n], ["causal", "full"], target.device)
            meta.update(max_num_tokens=n, is_sharded=False)
            entry = PrefixEntry(attention_mask.layout, target, packed, plan, meta, {})
            controller["entries"][cache_key] = entry
            phase.update(prefill=True, entry=entry)
            result = original_forward(
                pack, attention_mask=attention_mask, position_ids=position_ids, natten_metadata_list=None, memory=None
            )
            entry.output = result
            controller["stats"]["prefill_calls"] += 1
            controller["stats"]["active_tokens"][str(cache_key)] = n
            return result
        if attention_mask.layout is not entry.layout:
            raise ValueError("Prefix cache layout changed within a request; rebuild the cache")
        phase.update(prefill=False, entry=entry)
        active = get_gen_seq(pack).index_select(0, entry.gen_indexes)
        compact = from_mode_splits(active.new_empty((0, active.shape[-1])), active, entry.compact_metadata)
        positions = position_ids.index_select(-1, entry.packed_indexes)
        output, _ = original_forward(
            compact, attention_mask=attention_mask, position_ids=positions, natten_metadata_list=None, memory=None
        )
        # Restore the full head layout. Inactive predictions are discarded by the
        # sampler; only fixed streams and current noisy predictions are reused.
        full_output, metadata = entry.output
        get_gen_seq(full_output).index_copy_(0, entry.gen_indexes, get_gen_seq(output))
        controller["stats"]["replay_calls"] += 1
        return full_output, metadata

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
