# SPDX-License-Identifier: OpenMDW-1.1
"""Selection fast paths preserve grouped attention and first-order gradients."""

import sys
from types import SimpleNamespace

import pytest
import torch

from cosmos_framework.model.generator.mot import teacher_forcing_tnd as tnd


@pytest.mark.parametrize(
    ("indexes", "expected"),
    [
        ([], None),
        ([0, 1, 2], (0, 3)),
        ([2, 3], (2, 4)),
        ([0, 2], None),
        ([1, 0], None),
        ([1, 1], None),
        ([-1, 0], None),
    ],
)
def test_only_contiguous_nonnegative_indexes_get_spans(indexes, expected):
    chunk = tnd.TNDChunk.from_lists(indexes, indexes, [len(indexes)], [len(indexes)], device="cpu")
    assert chunk.query_span == expected
    assert chunk.kv_span == expected


@pytest.mark.parametrize("indexes", [[0, 1, 2, 3], [1, 2], [0, 2], [2, 1], [1, 1]])
@pytest.mark.parametrize("contiguous", [False, True])
def test_selection_reuses_only_contiguous_views(indexes, contiguous):
    tensor = torch.randn(4, 2, 3) if contiguous else torch.randn(4, 3, 2).transpose(1, 2)
    chunk = tnd.TNDChunk.from_lists(indexes, indexes, [len(indexes)], [len(indexes)], device="cpu")
    actual = tnd._select_tokens(tensor, chunk.query_indexes, chunk.query_span)
    torch.testing.assert_close(actual, tensor.index_select(0, chunk.query_indexes), rtol=0, atol=0)
    assert actual.is_contiguous()
    shares_storage = actual.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
    assert shares_storage == (contiguous and chunk.query_span is not None)
    if contiguous and indexes == [0, 1, 2, 3]:
        assert actual is tensor


def test_out_of_bounds_span_preserves_index_select_error():
    tensor = torch.randn(4, 2, 3)
    chunk = tnd.TNDChunk.from_lists([3, 4], [3, 4], [2], [2], device="cpu")
    with pytest.raises(IndexError):
        tnd._select_tokens(tensor, chunk.query_indexes, chunk.query_span)


def _sdpa(query, key, value, *, scale, actual_seq_qlen, actual_seq_kvlen):
    parts = []
    q0 = k0 = 0
    repeats = query.shape[1] // key.shape[1]
    for q1, k1 in zip(actual_seq_qlen, actual_seq_kvlen, strict=True):
        parts.append(
            torch.nn.functional.scaled_dot_product_attention(
                query[q0:q1].transpose(0, 1),
                key[k0:k1].repeat_interleave(repeats, dim=1).transpose(0, 1),
                value[k0:k1].repeat_interleave(repeats, dim=1).transpose(0, 1),
                scale=scale,
            ).transpose(0, 1)
        )
        q0, k0 = q1, k1
    return torch.cat(parts)


@pytest.fixture
def native_attention_stub(monkeypatch):
    """Exercise the NPU autograd wrapper on CPU; hardware checks run separately."""
    calls = {"inputs": [], "outputs": []}

    def forward(query, key, value, **kwargs):
        calls["inputs"].append((query, key, value))
        result = _sdpa(
            query,
            key,
            value,
            scale=kwargs["scale"],
            actual_seq_qlen=kwargs["actual_seq_qlen"],
            actual_seq_kvlen=kwargs["actual_seq_kvlen"],
        )
        calls["outputs"].append(result)
        empty = result.new_empty(0)
        return result, empty, empty, empty, 0, 0, 0

    def backward(query, key, value, grad_output, head_num, layout, **kwargs):
        del head_num, layout
        with torch.enable_grad():
            inputs = [tensor.detach().requires_grad_() for tensor in (query, key, value)]
            output = _sdpa(
                *inputs,
                scale=kwargs["scale_value"],
                actual_seq_qlen=kwargs["actual_seq_qlen"],
                actual_seq_kvlen=kwargs["actual_seq_kvlen"],
            )
            return torch.autograd.grad(output, inputs, grad_output)

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_fusion_attention=forward))
    monkeypatch.setattr(torch.ops.npu, "npu_fusion_attention_grad", SimpleNamespace(default=backward), raising=False)
    return calls


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize(
    "groups",
    [
        [([0, 1, 2, 3], [0, 1, 2, 3, 4], [4], [5])],
        [([0, 1], [1, 2, 3, 4], [2], [4]), ([2, 3], [0, 1, 2], [2], [3])],
        [([0, 1, 2, 3], [0, 1, 2, 0, 2, 4], [1, 4], [3, 6])],
        [([0, 1, 2, 3], [4, 1, 3], [4], [3])],
    ],
)
def test_native_wrapper_matches_gathers_and_accumulates_shared_kv_gradients(
    native_attention_stub, groups, contiguous, dtype
):
    torch.manual_seed(12)
    shape = (4, 2, 4) if contiguous else (4, 4, 2)
    query = torch.randn(shape, dtype=dtype)
    key = torch.randn((5, 1, 4) if contiguous else (5, 4, 1), dtype=dtype)
    value = torch.randn_like(key)
    if not contiguous:
        query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
    grad_output = torch.randn(4, 2, 4, dtype=dtype)
    inputs = [tensor.detach().requires_grad_() for tensor in (query, key, value)]
    fast_chunks = tuple(tnd.TNDChunk.from_lists(*group, device="cpu") for group in groups)
    slow_chunks = tuple(
        tnd.TNDChunk(chunk.query_indexes, chunk.kv_indexes, chunk.query_ends, chunk.kv_ends) for chunk in fast_chunks
    )
    fast = tnd._NPUTeacherForcing.apply(*inputs, tnd.TNDPlan(fast_chunks, 4, 5), 0.5)
    fast_gradients = torch.autograd.grad(fast, inputs, grad_output)
    slow = tnd._NPUTeacherForcing.apply(*inputs, tnd.TNDPlan(slow_chunks, 4, 5), 0.5)
    slow_gradients = torch.autograd.grad(slow, inputs, grad_output)
    torch.testing.assert_close(fast, slow, rtol=0, atol=0)
    for actual, expected in zip(fast_gradients, slow_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_single_chunk_returns_native_output_without_concat(native_attention_stub):
    query, key, value = torch.randn(4, 2, 4), torch.randn(5, 1, 4), torch.randn(5, 1, 4)
    chunk = tnd.TNDChunk.from_lists(list(range(4)), list(range(5)), [4], [5], device="cpu")
    output = tnd._NPUTeacherForcing.apply(query, key, value, tnd.TNDPlan((chunk,), 4, 5), 0.5)
    assert output.data_ptr() == native_attention_stub["outputs"][0].data_ptr()
    assert all(
        actual is original
        for actual, original in zip(native_attention_stub["inputs"][0], (query, key, value), strict=True)
    )
