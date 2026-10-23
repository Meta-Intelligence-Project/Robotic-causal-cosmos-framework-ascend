# SPDX-License-Identifier: OpenMDW-1.1
"""CPU checks for request-local attention plans, independent of checkpoints."""

import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import torch

from cosmos_framework.data.generator.sequence_packing.causal_action import (
    CausalActionData,
    CausalActionGeometry,
    build_action_layout,
    dense_action_mask,
)
from cosmos_framework.data.generator.sequence_packing.runtime import from_mode_splits, get_causal_seq, get_full_only_seq
from cosmos_framework.inference.causal_action import plan_cache
from cosmos_framework.model.generator.mot import attention, causal_action_cache
from cosmos_framework.model.generator.mot.teacher_forcing_tnd import teacher_forcing_tnd_attention


def _template(*, text_length=2, history=2, current_condition=False):
    metadata = SimpleNamespace(
        action_frame_ids=torch.tensor([1, 2, 3, 4]),
        block_size=2,
        states=torch.zeros(2, 55),
        state_action_indexes=torch.tensor([0, 2]),
    )
    layout = build_action_layout(
        und_counts=[text_length],
        vision_shapes=[(5, 1, 1)],
        action_lengths=[4],
        vision_conditions=[torch.tensor([True, True, True, current_condition, False])],
        action_conditions=[torch.tensor([True, True, False, False])],
        geometry=CausalActionGeometry((2,), (history,)),
        metadata=[metadata],
    )
    return SimpleNamespace(teacher_forcing=CausalActionData(layout, []))


def _pack(template, plan=None, values=None):
    layout = template.teacher_forcing.layout
    values = torch.zeros(sum(layout.sample_lens), 2, 4) if values is None else values
    return attention.build_packed_sequence(
        "teacher_forcing",
        packed_sequence=values,
        attn_modes=list(layout.attn_modes),
        split_lens=list(layout.split_lens),
        sample_lens=list(layout.sample_lens),
        packed_und_token_indexes=torch.where(layout.stream_ids == -1)[0],
        packed_gen_token_indexes=layout.gen_query_indexes,
        num_heads=2,
        head_dim=4,
        num_layers=3,
        teacher_forcing_layout=layout,
        teacher_forcing_dense_mode="grouped_tnd",
        teacher_forcing_tnd_plan=None if plan is None else plan.full_plan,
        teacher_forcing_target_only_plan=None if plan is None else plan.target_only,
    )[:2]


class CausalActionPlanCacheTest(unittest.TestCase):
    def test_build_once_for_all_steps_and_layers(self):
        template = _template()
        with (
            patch.object(plan_cache, "build_tnd_plan", wraps=plan_cache.build_tnd_plan) as build,
            patch.object(plan_cache, "target_only_plan", wraps=plan_cache.target_only_plan) as target,
            patch.object(attention, "build_tnd_plan", wraps=attention.build_tnd_plan) as fallback,
        ):
            plan = plan_cache.prepare_attention_plan(template)
            self.assertIs(plan_cache.prepare_attention_plan(template), plan)
            for _ in range(5):
                _, meta = _pack(template, plan)
                self.assertIs(meta.tnd_plan, plan.full_plan)
                self.assertIs(meta.target_only_tnd_plan, plan.target_only)
            self.assertEqual(build.call_count, 1)
            self.assertEqual(target.call_count, 1)
            self.assertEqual(fallback.call_count, 0)

    def test_cfg_requests_and_layout_changes_are_isolated(self):
        cond, uncond, another_request = _template(), _template(text_length=3), _template()
        plans = [plan_cache.prepare_attention_plan(template) for template in (cond, uncond, another_request)]
        self.assertEqual(len({id(plan.full_plan) for plan in plans}), 3)
        self.assertNotEqual(plans[0].full_plan.num_keys, plans[1].full_plan.num_keys)
        self.assertNotEqual(
            plans[0].full_plan.chunks[0].kv_indexes.data_ptr(), plans[2].full_plan.chunks[0].kv_indexes.data_ptr()
        )
        cond.teacher_forcing.layout = _template(history=1, current_condition=True).teacher_forcing.layout
        replacement = plan_cache.prepare_attention_plan(cond)
        self.assertIsNot(replacement, plans[0])
        self.assertIs(replacement.layout, cond.teacher_forcing.layout)
        new_limit = plan_cache.prepare_attention_plan(cond, max_kv_tokens=16)
        self.assertIsNot(new_limit, replacement)
        self.assertEqual(new_limit.max_kv_tokens, 16)

    def test_uncached_path_still_builds_each_call(self):
        template = _template()
        with patch.object(attention, "build_tnd_plan", wraps=attention.build_tnd_plan) as build:
            _, first = _pack(template)
            _, second = _pack(template)
        self.assertEqual(build.call_count, 2)
        self.assertIsNone(first.target_only_tnd_plan)
        self.assertIsNot(first.tnd_plan, second.tnd_plan)

    def test_cached_output_matches_original_and_dense_visibility(self):
        torch.manual_seed(42)
        for template in (_template(), _template(text_length=3, history=1, current_condition=True)):
            with self.subTest(layout=template.teacher_forcing.layout.geometry):
                plan = plan_cache.prepare_attention_plan(template, max_kv_tokens=16)
                _, original = _pack(template)
                _, cached = _pack(template, plan)
                q = torch.randn(plan.full_plan.num_queries, 2, 4)
                k = torch.randn(plan.full_plan.num_keys, 1, 4)
                v = torch.randn_like(k)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    first = teacher_forcing_tnd_attention(q, k, v, original.tnd_plan)
                    second = teacher_forcing_tnd_attention(q, k, v, cached.tnd_plan)
                    target, target_plan = plan.target_only
                    target_output = teacher_forcing_tnd_attention(q[target], k, v, target_plan)
                expected = torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(0, 1),
                    k.repeat_interleave(2, dim=1).transpose(0, 1),
                    v.repeat_interleave(2, dim=1).transpose(0, 1),
                    attn_mask=dense_action_mask(plan.layout)[None],
                    dropout_p=0.0,
                ).transpose(0, 1)
                torch.testing.assert_close(second, first, rtol=0, atol=0)
                torch.testing.assert_close(second, expected, rtol=1e-6, atol=2e-7)
                torch.testing.assert_close(target_output, expected[target], rtol=1e-6, atol=2e-7)

    def test_attention_cache_reuses_target_plan_across_layers(self):
        template = _template()
        plan = plan_cache.prepare_attention_plan(template)
        packed, meta = _pack(template, plan)

        def original(query, key, value, attention_mask, **kwargs):
            return from_mode_splits(
                get_causal_seq(query)[0].flatten(-2), get_full_only_seq(query)[0].flatten(-2), query
            ), None

        layers = [SimpleNamespace(self_attn=SimpleNamespace(dispatch_attention_fn=original)) for _ in range(3)]
        net = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=layers)))
        with (
            torch.no_grad(),
            patch.object(causal_action_cache, "target_only_plan", wraps=causal_action_cache.target_only_plan) as target,
            causal_action_cache.action_attention_cache(net, True) as cache,
        ):
            cache["key"] = (0, "cond")
            for layer in layers:
                layer.self_attn.dispatch_attention_fn(packed, packed, packed, meta)
            self.assertEqual(target.call_count, 0)
            self.assertEqual(len(cache["entries"]), 3)
        self.assertTrue(all(layer.self_attn.dispatch_attention_fn is original for layer in layers))
        self.assertEqual(cache["entries"], {})

    def test_rejects_missing_layout_and_mismatched_plan(self):
        with self.assertRaisesRegex(ValueError, "requires a causal action layout"):
            plan_cache.prepare_attention_plan(SimpleNamespace(teacher_forcing=None))
        plan = plan_cache.prepare_attention_plan(_template())
        with self.assertRaisesRegex(ValueError, "does not match"):
            _pack(_template(text_length=3), plan)


if __name__ == "__main__":
    unittest.main()
