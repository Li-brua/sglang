import torch
from torch.cuda.streams import ExternalStream

try:
    from . import spatial_ops  # triggers TORCH extension registration
except Exception as _e:
    _spatial_import_error = _e
else:
    _spatial_import_error = None

_IMPORT_ERROR = ImportError(
    "Failed to load sgl_kernel.spatial_ops extension. Ensure CUDA Driver >= 12.4"
)


def create_greenctx_stream_by_value(
    SM_a: int, SM_b: int, device_id: int = None
) -> tuple[ExternalStream, ExternalStream]:
    """
    Create two streams for greenctx.
    Args:
        sm_A (int): The SM of stream A.
        sm_B (int): The weight of stream B.
        device_id (int): The device id.
    Returns:
        tuple[ExternalStream, ExternalStream]: The two streams.
    """
    if _spatial_import_error is not None:
        raise _IMPORT_ERROR from _spatial_import_error
    if device_id is None:
        device_id = torch.cuda.current_device()

    res = torch.ops.sgl_kernel.create_greenctx_stream_by_value(SM_a, SM_b, device_id)

    stream_a = ExternalStream(
        stream_ptr=res[0], device=torch.device(f"cuda:{device_id}")
    )
    stream_b = ExternalStream(
        stream_ptr=res[1], device=torch.device(f"cuda:{device_id}")
    )

    return stream_a, stream_b


def create_overlapped_greenctx_stream_by_value(
    reserved_sm: int, device_id: int = None
) -> tuple[ExternalStream, ExternalStream, tuple[int, int, int, int, int]]:
    """Create priority decode/prefill streams with exclusive SM floors and a shared remainder.

    The counts are the driver's actual (prefill, decode, prefill-only,
    decode-only, shared) SM allocations, which may exceed the requested floor.
    """
    if _spatial_import_error is not None:
        raise _IMPORT_ERROR from _spatial_import_error
    if device_id is None:
        device_id = torch.cuda.current_device()

    try:
        op = torch.ops.sgl_kernel.create_overlapped_greenctx_stream_by_value
    except AttributeError as exc:
        raise ImportError(
            "The loaded sgl_kernel.spatial_ops binary is older than its Python "
            "package and lacks protected PDMux overlay support. Rebuild and "
            "reinstall python/sglang/kernels/aot, then restart the server."
        ) from exc
    res = op(reserved_sm, device_id)
    stream_a = ExternalStream(
        stream_ptr=res[0], device=torch.device(f"cuda:{device_id}")
    )
    stream_b = ExternalStream(
        stream_ptr=res[1], device=torch.device(f"cuda:{device_id}")
    )
    return stream_a, stream_b, tuple(res[2:])


def create_asymmetric_overlapped_greenctx_stream_by_value(
    prefill_reserved_sm: int,
    decode_reserved_sm: int,
    device_id: int = None,
) -> tuple[ExternalStream, ExternalStream, tuple[int, int, int, int, int]]:
    """Create overlay streams with independent prefill and decode SM floors."""
    if _spatial_import_error is not None:
        raise _IMPORT_ERROR from _spatial_import_error
    if device_id is None:
        device_id = torch.cuda.current_device()

    try:
        op = torch.ops.sgl_kernel.create_asymmetric_overlapped_greenctx_stream_by_value
    except AttributeError as exc:
        raise ImportError(
            "The loaded sgl_kernel.spatial_ops binary lacks asymmetric protected "
            "PDMux overlay support. Rebuild the kernel or use the JIT fallback."
        ) from exc
    res = op(prefill_reserved_sm, decode_reserved_sm, device_id)
    stream_a = ExternalStream(
        stream_ptr=res[0], device=torch.device(f"cuda:{device_id}")
    )
    stream_b = ExternalStream(
        stream_ptr=res[1], device=torch.device(f"cuda:{device_id}")
    )
    return stream_a, stream_b, tuple(res[2:])


def get_sm_available(device_id: int = None) -> int:
    """
    Get the SMs available on the device.
    Args:
        device_id (int): The device id.
    Returns:
        int: The SMs available.
    """
    if _spatial_import_error is not None:
        raise _IMPORT_ERROR from _spatial_import_error
    if device_id is None:
        device_id = torch.cuda.current_device()

    device_props = torch.cuda.get_device_properties(device_id)

    # Get the number of Streaming Multiprocessors (SMs)
    sm_count = device_props.multi_processor_count

    return sm_count
