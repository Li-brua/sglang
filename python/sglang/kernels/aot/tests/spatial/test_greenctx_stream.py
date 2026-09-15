import sys

import pytest
import torch
from sgl_kernel import create_greenctx_stream_by_value, get_sm_available
from sgl_kernel.spatial import (
    create_asymmetric_overlapped_greenctx_stream_by_value,
    create_overlapped_greenctx_stream_by_value,
)


def test_green_ctx():
    A = torch.randn(5120, 5120).cuda()
    B = torch.randn(5120, 5120).cuda()
    C = torch.matmul(A, B)
    sm_counts = get_sm_available(0)
    stream_group = create_greenctx_stream_by_value(sm_counts // 2, sm_counts // 2, 0)
    with torch.cuda.stream(stream_group[0]):
        for _ in range(100):
            result_0 = torch.matmul(A, B)
    with torch.cuda.stream(stream_group[1]):
        for _ in range(100):
            result_1 = torch.matmul(A, B)
    torch.cuda.synchronize()
    assert torch.allclose(result_0, C)
    assert torch.allclose(result_1, C)


def test_overlapped_green_ctx_with_protected_sm_floors():
    total_sm = get_sm_available(0)
    if total_sm <= 24:
        pytest.skip("not enough SMs for two reservations and a shared partition")

    prefill_stream, decode_stream, counts = create_overlapped_greenctx_stream_by_value(
        8, 0
    )
    prefill_sm, decode_sm, prefill_only, decode_only, shared = counts
    assert prefill_only >= 8 and decode_only >= 8 and shared > 0
    assert prefill_only + decode_only + shared == total_sm
    assert prefill_sm == prefill_only + shared
    assert decode_sm == decode_only + shared

    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    expected = a @ b
    with torch.cuda.stream(prefill_stream):
        prefill_result = a @ b
    with torch.cuda.stream(decode_stream):
        decode_result = a @ b
    torch.cuda.synchronize()
    assert torch.allclose(prefill_result, expected)
    assert torch.allclose(decode_result, expected)


def test_asymmetric_overlapped_green_ctx_with_protected_sm_floors():
    total_sm = get_sm_available(0)
    if total_sm < 40:
        pytest.skip("not enough SMs for asymmetric reservations and sharing")

    _, _, counts = create_asymmetric_overlapped_greenctx_stream_by_value(8, 24, 0)
    prefill_sm, decode_sm, prefill_only, decode_only, shared = counts
    assert (prefill_only, decode_only) == (8, 24)
    assert shared == total_sm - 32
    assert prefill_sm == total_sm - 24
    assert decode_sm == total_sm - 8


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
