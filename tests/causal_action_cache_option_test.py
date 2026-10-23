# SPDX-License-Identifier: OpenMDW-1.1
"""The public inference API must preserve the caller's action-cache setting."""

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from cosmos_framework.model.generator.omni_mot_causal_action_model import OmniMoTCausalActionModel
from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel


class CausalActionCacheOptionTest(unittest.TestCase):
    def test_action_guidance_default_and_explicit_argument_forwarding(self):
        self.assertEqual(
            inspect.signature(OmniMoTModel.generate_samples_from_batch).parameters["guidance"].default, 1.5
        )
        self.assertEqual(
            inspect.signature(OmniMoTModel.generate_samples_from_batch).parameters["num_steps"].default, 35
        )
        model = OmniMoTCausalActionModel.__new__(OmniMoTCausalActionModel)
        batch, net, sampler = {}, object(), object()
        interval = [0.1, 0.9]
        expected = {"action": [torch.zeros(32, 55)]}
        cases = (
            ((batch,), {}, (batch, None, None, 1.0), {"num_steps": 10}),
            ((batch,), {"guidance": 3.0, "num_steps": 5}, (batch, None, None, 3.0), {"num_steps": 5}),
            ((batch,), {"num_steps": None}, (batch, None, None, 1.0), {"num_steps": None}),
            (
                (batch, net, sampler, 2.0, interval),
                {"seed": [42]},
                (batch, net, sampler, 2.0, interval),
                {"seed": [42], "num_steps": 10},
            ),
            (
                (batch, net, sampler, 2.0, interval, None, [42], None, False, 7, 6.0),
                {},
                (batch, net, sampler, 2.0, interval, None, [42], None, False, 7, 6.0),
                {},
            ),
        )
        for args, kwargs, expected_args, expected_kwargs in cases:
            with self.subTest(args=args, kwargs=kwargs):
                with patch.object(
                    OmniMoTModel, "generate_samples_from_batch", autospec=True, return_value=expected
                ) as generate:
                    self.assertIs(model.generate_samples_from_batch(*args, **kwargs), expected)
                    generate.assert_called_once_with(model, *expected_args, **expected_kwargs)

    def test_public_api_forwards_disabled_and_default_cache(self):
        model = SimpleNamespace(
            parallel_dims=None,
            _maybe_apply_prompt_upsampling=lambda batch, **kwargs: batch,
            _prepare_inference_data=lambda batch, seed, negative: (
                [],
                None,
                [],
                [],
                [torch.zeros(1)],
                [],
                [],
                True,
            ),
        )
        model._generate_causal_inference_from_prepared = lambda **kwargs: (
            OmniMoTCausalActionModel._generate_causal_inference_from_prepared(model, **kwargs)
        )
        expected = {"action": [torch.zeros(32, 55)]}
        with patch(
            "cosmos_framework.inference.causal_action.sampling.sample_prepared", return_value=expected
        ) as sample:
            for options, enabled in (({"causal_use_kv_cache": False}, False), ({}, True)):
                with self.subTest(options=options):
                    result = OmniMoTModel.generate_samples_from_batch(model, {}, seed=[42], **options)
                    self.assertIs(result, expected)
                    self.assertIs(sample.call_args.kwargs["causal_use_kv_cache"], enabled)


if __name__ == "__main__":
    unittest.main()
