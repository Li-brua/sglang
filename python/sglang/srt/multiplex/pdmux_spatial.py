"""Small cached JIT fallback for protected PDMux green-context streams."""

from __future__ import annotations

import torch
from torch.cuda.streams import ExternalStream

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.jit.utils.compile.toolchain import cuda_home

@cache_once
def _jit_module():
    cuda_root = cuda_home()
    return load_jit(
        "pdmux_protected_greenctx",
        "v2",
        cpp_files=["pdmux/greenctx_stream.cpp"],
        extra_include_paths=[f"{cuda_root}/include"],
        extra_ldflags=[f"-L{cuda_root}/lib64", "-lcuda"],
        header_only=False,
    )


def create_overlapped_greenctx_stream_by_value(
    prefill_reserved_sm: int, decode_reserved_sm: int, device_id: int
) -> tuple[ExternalStream, ExternalStream, tuple[int, int, int, int, int]]:
    """Create protected overlay streams without rebuilding the AOT kernel wheel."""
    result = _jit_module().create_overlapped_greenctx_stream_by_value(
        prefill_reserved_sm, decode_reserved_sm, device_id
    )
    device = torch.device(f"cuda:{device_id}")
    prefill_stream = ExternalStream(stream_ptr=result[0], device=device)
    decode_stream = ExternalStream(stream_ptr=result[1], device=device)
    return prefill_stream, decode_stream, tuple(result[2:])
