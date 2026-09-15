import importlib
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import yaml

logger = logging.getLogger(__name__)

STREAM_GROUPS = []
SM_COUNTS = []
SM_GROUP_NUM = 8  # Default number of SM groups
CURRENT_STREAM_IDX = 0
CURRENT_STREAM_GROUP = None


@dataclass
class PDMuxConfig:
    sm_group_num: int = 8
    manual_divisions: List[List[int]] = field(
        default_factory=list
    )  # [prefill_sm, decode_sm, decode_bs_threshold]
    split_forward_token_budget: int = 65536
    decode_bs_divisor: int = 36
    # Legacy overlap: prefill is capped while decode has a full-device stream.
    # With overlap_prefill_reserved_sm, both lanes use overlapping green
    # contexts instead. Protected mode derives both lane sizes from the device
    # and reservations, so only the threshold column in manual_divisions is used.
    overlap_decode_full_sm: bool = False
    # Opt-in overlay with independently protected SM floors. If the decode
    # floor is omitted, it inherits the prefill floor for backward compatibility.
    # CUDA may round a symmetric floor upward; the actual layout is logged.
    overlap_prefill_reserved_sm: int = 0
    overlap_decode_reserved_sm: Optional[int] = None
    # Layer-prefill only: after a request finishes one incomplete token chunk,
    # stash it and put it at the tail of the prefill queue. This turns the
    # privileged chunk continuation into a one-chunk round-robin quantum.
    layer_prefill_chunk_round_robin: bool = False


def is_pdmux_standard_prefill() -> bool:
    """True when PD-Multiplexing submits prefills as one standard EXTEND.

    Model and layer code reads this to pick the lane-safe variant of a path
    that would otherwise place work on a stream the prefill green context does
    not own (a model-internal helper stream) or share one communicator /
    scratch buffer between the two lanes. It is deliberately a resolved-config
    read rather than a constructor argument: the callers are leaf modules that
    already read `get_disagg()` for the sibling `enable_pdmux` decisions.
    """
    from sglang.srt.runtime_context import get_disagg

    disagg = get_disagg()
    return disagg.enable_pdmux and disagg.pdmux_prefill_mode == "standard"


def decode_lane_attn_backend(model_runner):
    """The backend for decode-lane work that a caller plans before the forward.

    TARGET_VERIFY is classified as an extend mode but runs on the decode lane.
    On the standard lane it must plan into the per-stream decode backend the
    eager runner will resolve for it, because the prefill instance may be
    serving an in-flight prefill on the other stream and both write their
    metadata in place. Every other configuration -- no PDMux, or layer_split --
    keeps the runner's default, which is what those paths always used.
    """
    if is_pdmux_standard_prefill():
        return model_runner.decode_attn_backend
    return model_runner.attn_backend


def load_pdmux_config(config_path: str) -> PDMuxConfig:
    """Load pdmux configuration from YAML file into a dataclass."""
    if not config_path:
        return PDMuxConfig()

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    if "sm_group_num" not in raw:
        raise ValueError("Missing required field: sm_group_num")

    if raw["sm_group_num"] < 3:
        raise ValueError("sm_group_num must be >= 3")

    manual_divisions = raw.get("manual_divisions", [])

    expected = raw["sm_group_num"] - 2
    if manual_divisions and len(manual_divisions) != expected:
        raise ValueError(
            f"manual_divisions must have {expected} entries, "
            f"but got {len(manual_divisions)}"
        )

    overlap_decode_full_sm = raw.get("overlap_decode_full_sm", False)
    if overlap_decode_full_sm and not manual_divisions:
        raise ValueError(
            "overlap_decode_full_sm requires explicit manual_divisions: the "
            "automatic divide_sm split enforces prefill >= 50% for mutually "
            "exclusive partitions, which does not describe an overlapped one."
        )
    overlap_prefill_reserved_sm = raw.get("overlap_prefill_reserved_sm", 0)
    if (
        isinstance(overlap_prefill_reserved_sm, bool)
        or not isinstance(overlap_prefill_reserved_sm, int)
        or overlap_prefill_reserved_sm < 0
    ):
        raise ValueError("overlap_prefill_reserved_sm must be a non-negative integer")
    if overlap_prefill_reserved_sm and not overlap_decode_full_sm:
        raise ValueError(
            "overlap_prefill_reserved_sm requires overlap_decode_full_sm: true"
        )
    overlap_decode_reserved_sm = raw.get("overlap_decode_reserved_sm")
    if overlap_decode_reserved_sm is not None and (
        isinstance(overlap_decode_reserved_sm, bool)
        or not isinstance(overlap_decode_reserved_sm, int)
        or overlap_decode_reserved_sm <= 0
    ):
        raise ValueError("overlap_decode_reserved_sm must be a positive integer")
    if overlap_decode_reserved_sm is not None and not overlap_prefill_reserved_sm:
        raise ValueError(
            "overlap_decode_reserved_sm requires a positive overlap_prefill_reserved_sm"
        )
    layer_prefill_chunk_round_robin = raw.get(
        "layer_prefill_chunk_round_robin", False
    )
    if not isinstance(layer_prefill_chunk_round_robin, bool):
        raise ValueError("layer_prefill_chunk_round_robin must be a boolean")

    return PDMuxConfig(
        sm_group_num=raw["sm_group_num"],
        manual_divisions=manual_divisions,
        split_forward_token_budget=raw.get("split_forward_token_budget", 65536),
        decode_bs_divisor=raw.get("decode_bs_divisor", 36),
        overlap_decode_full_sm=overlap_decode_full_sm,
        overlap_prefill_reserved_sm=overlap_prefill_reserved_sm,
        overlap_decode_reserved_sm=overlap_decode_reserved_sm,
        layer_prefill_chunk_round_robin=layer_prefill_chunk_round_robin,
    )


def get_arch_constraints(compute_capability):
    major, minor = compute_capability
    # green context constraints for different architectures
    if major == 6:
        return 1, 1  # min_per_part, multiple
    elif major == 7:
        return 2, 2
    elif major == 8:
        return 4, 2
    elif major == 9 and minor >= 0:
        return 8, 8
    else:
        raise ValueError(f"Unsupported compute capability: {major}.{minor}")


def divide_sm(total_sms, compute_capability, groups):
    """
    :param total_sms: total sm count on a single GPU
    :param compute_capability: (major, minor)
    :return: SM partition group(prefill sm, decode sm)
    """
    min_per_part, multiple = get_arch_constraints(compute_capability)
    possible_values = [
        x
        for x in range(min_per_part, total_sms - min_per_part + 1, multiple)
        if x >= total_sms - x and total_sms - x >= 16
    ]
    if not possible_values:
        raise ValueError(
            f"No valid partitions found for total SMs {total_sms} "
            f"with constraints (min per part: {min_per_part}, multiple: {multiple})"
        )

    if len(possible_values) >= groups:
        step = max(1, len(possible_values) // groups)
        selected_values = possible_values[::step][:groups]
    else:
        selected_values = possible_values

    divisions = []
    for part1 in selected_values:
        part2 = total_sms - part1
        divisions.append((part1, part2))

    divisions.reverse()  # Reverse to have larger prefill SM first

    return divisions


def initialize_stream_groups(gpu_id: int, config: PDMuxConfig):
    from sgl_kernel import spatial

    global \
        STREAM_GROUPS, \
        SM_COUNTS, \
        SM_GROUP_NUM, \
        CURRENT_STREAM_IDX, \
        CURRENT_STREAM_GROUP
    # for pd_multiplexing, Init stream_groups
    device = torch.cuda.current_device()
    total_sm_count = spatial.get_sm_available(gpu_id)
    # (prefill_sm_count, decode_sm_count)
    if config.manual_divisions:
        divisions = [
            (prefill_sm, decode_sm)
            for prefill_sm, decode_sm, _ in config.manual_divisions
        ]
    else:
        divisions = divide_sm(
            total_sm_count,
            torch.cuda.get_device_capability(device),
            config.sm_group_num - 2,
        )

    if config.overlap_decode_full_sm:
        decode_reserved_sm = (
            config.overlap_decode_reserved_sm
            if config.overlap_decode_reserved_sm is not None
            else config.overlap_prefill_reserved_sm
        )
        if config.overlap_prefill_reserved_sm:
            if (
                config.overlap_prefill_reserved_sm + decode_reserved_sm
                >= total_sm_count
            ):
                raise ValueError(
                    "The prefill and decode SM reservations must leave at least one "
                    f"shared SM; got {config.overlap_prefill_reserved_sm} + "
                    f"{decode_reserved_sm} on a {total_sm_count}-SM device."
                )
            # Reservations determine this topology. Ignore the first two
            # manual_divisions columns so one config works across GPU sizes.
            # Driver-reported values replace these provisional counts below.
            divisions = [
                (
                    total_sm_count - decode_reserved_sm,
                    total_sm_count - config.overlap_prefill_reserved_sm,
                )
                for _ in divisions
            ]
        else:
            for prefill_sm, _ in divisions:
                if not 0 < prefill_sm < total_sm_count:
                    raise ValueError(
                        f"overlap_decode_full_sm needs a prefill_sm strictly inside "
                        f"(0, {total_sm_count}); got {prefill_sm}. A full-device cap "
                        f"leaves decode no SMs of its own."
                    )
            # The legacy decode mask covers the device.
            divisions = [(prefill_sm, total_sm_count) for prefill_sm, _ in divisions]
    else:
        for prefill_sm, decode_sm in divisions:
            if prefill_sm + decode_sm > total_sm_count:
                raise ValueError(
                    f"manual_divisions entry ({prefill_sm}, {decode_sm}) needs "
                    f"{prefill_sm + decode_sm} SMs but the device has "
                    f"{total_sm_count}. Mutually exclusive partitions must sum to "
                    f"at most the device size; a decode_sm of {total_sm_count} "
                    f"describes the overlapped layout, which needs "
                    f"overlap_decode_full_sm: true."
                )

    SM_COUNTS = []
    SM_COUNTS.append((total_sm_count, 0))  # Normal stream for prefill
    SM_COUNTS.extend(divisions)  # Add the divided SM counts
    SM_COUNTS.append((0, total_sm_count))  # Normal stream for decode
    STREAM_GROUPS = []
    STREAM_GROUPS.append(
        (torch.cuda.Stream(gpu_id), torch.cuda.Stream(gpu_id))
    )  # Normal stream for prefill
    for group_idx, (prefill_sm, decode_sm) in enumerate(divisions, start=1):
        if config.overlap_decode_full_sm:
            if config.overlap_prefill_reserved_sm:
                prefill_reserved_sm = config.overlap_prefill_reserved_sm
                decode_reserved_sm = (
                    config.overlap_decode_reserved_sm
                    if config.overlap_decode_reserved_sm is not None
                    else prefill_reserved_sm
                )
                create_asymmetric_overlay = getattr(
                    spatial,
                    "create_asymmetric_overlapped_greenctx_stream_by_value",
                    None,
                )
                create_symmetric_overlay = getattr(
                    spatial, "create_overlapped_greenctx_stream_by_value", None
                )
                try:
                    if prefill_reserved_sm == decode_reserved_sm:
                        if create_symmetric_overlay is None:
                            raise ImportError(
                                "operator is absent from sgl_kernel.spatial"
                            )
                        prefill_stream, decode_stream, actual = (
                            create_symmetric_overlay(prefill_reserved_sm, gpu_id)
                        )
                    else:
                        if create_asymmetric_overlay is None:
                            raise ImportError(
                                "asymmetric operator is absent from sgl_kernel.spatial"
                            )
                        prefill_stream, decode_stream, actual = (
                            create_asymmetric_overlay(
                                prefill_reserved_sm, decode_reserved_sm, gpu_id
                            )
                        )
                except ImportError:
                    logger.info(
                        "The installed sglang-kernel lacks protected PDMux "
                        "overlay support; loading the small cached JIT fallback."
                    )
                    create_protected_overlay = importlib.import_module(
                        "sglang.srt.multiplex.pdmux_spatial"
                    ).create_overlapped_greenctx_stream_by_value
                    prefill_stream, decode_stream, actual = create_protected_overlay(
                        prefill_reserved_sm, decode_reserved_sm, gpu_id
                    )
                (
                    actual_prefill,
                    actual_decode,
                    prefill_only,
                    decode_only,
                    shared,
                ) = actual
                SM_COUNTS[group_idx] = (actual_prefill, actual_decode)
                logger.info(
                    "PDMux protected overlay SMs: prefill_only=%s, shared=%s, "
                    "decode_only=%s (requested prefill=%s, decode=%s)",
                    prefill_only,
                    shared,
                    decode_only,
                    prefill_reserved_sm,
                    decode_reserved_sm,
                )
            else:
                # Legacy overlay: decode's high-priority plain stream can reach
                # all SMs, including those used by the capped prefill stream.
                prefill_stream, _unused_decode_stream = (
                    spatial.create_greenctx_stream_by_value(
                        prefill_sm, total_sm_count - prefill_sm, gpu_id
                    )
                )
                decode_stream = torch.cuda.Stream(gpu_id, priority=-1)
            STREAM_GROUPS.append((prefill_stream, decode_stream))
        else:
            STREAM_GROUPS.append(
                spatial.create_greenctx_stream_by_value(prefill_sm, decode_sm, gpu_id)
            )
    STREAM_GROUPS.append(
        (torch.cuda.Stream(gpu_id), torch.cuda.Stream(gpu_id))
    )  # Normal stream for decode

    CURRENT_STREAM_IDX = 0
    CURRENT_STREAM_GROUP = STREAM_GROUPS[CURRENT_STREAM_IDX]


def set_current_stream_idx(idx: int):
    global CURRENT_STREAM_IDX, CURRENT_STREAM_GROUP
    if idx < 0 or idx >= len(STREAM_GROUPS):
        raise ValueError(f"Invalid stream index: {idx}")
    CURRENT_STREAM_IDX = idx
    CURRENT_STREAM_GROUP = STREAM_GROUPS[CURRENT_STREAM_IDX]


def get_stream_groups() -> list[tuple[torch.cuda.Stream, torch.cuda.Stream]]:
    """Get the stream groups."""
    return STREAM_GROUPS


def get_sm_counts() -> list[tuple[int, int]]:
    """Get the SM counts."""
    return SM_COUNTS


def get_current_stream_idx() -> int:
    """Get the current stream index."""
    return CURRENT_STREAM_IDX
