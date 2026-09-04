from dataclasses import dataclass, field
from typing import List

import torch
import yaml

STREAM_GROUPS = []
SM_COUNTS = []
GREEN_CONTEXT_STREAMS = []
SM_GROUP_NUM = 8  # Default number of SM groups
CURRENT_STREAM_IDX = 0
CURRENT_STREAM_GROUP = None
_RESERVED_GREEN_STREAMS = []

# CUDA uses lower numeric values for higher stream priorities.  This is a
# scheduling hint for pending work; it does not preempt already-running CTAs.
_OVERLAY_DECODE_STREAM_PRIORITY = -1


@dataclass
class PDMuxConfig:
    sm_group_num: int = 8
    manual_divisions: List[List[int]] = field(
        default_factory=list
    )  # [prefill_sm, decode_sm, decode_bs_threshold]
    # ``decode_sm`` is ignored for overlap_decode_full_sm.
    overlap_decode_full_sm: bool = False
    split_forward_token_budget: int = 65536
    decode_bs_divisor: int = 36


def load_pdmux_config(
    config_path: str, default_sm_group_num: int = SM_GROUP_NUM
) -> PDMuxConfig:
    """Load pdmux configuration from YAML file into a dataclass."""
    if not config_path:
        return PDMuxConfig(sm_group_num=default_sm_group_num)

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    if "sm_group_num" not in raw:
        raise ValueError("Missing required field: sm_group_num")

    if raw["sm_group_num"] < 3:
        raise ValueError("sm_group_num must be >= 3")

    manual_divisions = raw.get("manual_divisions", [])
    overlap_decode_full_sm = raw.get("overlap_decode_full_sm", False)

    expected = raw["sm_group_num"] - 2
    if manual_divisions and len(manual_divisions) != expected:
        raise ValueError(
            f"manual_divisions must have {expected} entries, "
            f"but got {len(manual_divisions)}"
        )
    if overlap_decode_full_sm and not manual_divisions:
        raise ValueError("overlap_decode_full_sm requires manual_divisions")

    previous_threshold = None
    for i, division in enumerate(manual_divisions):
        if len(division) != 3:
            raise ValueError(
                "manual_divisions entries must be "
                "[prefill_sm, decode_sm, decode_bs_threshold]"
            )
        prefill_sm, decode_sm, threshold = division
        if prefill_sm <= 0:
            raise ValueError(f"manual_divisions[{i}] prefill_sm must be positive")
        if overlap_decode_full_sm:
            # The decode column is intentionally ignored in this mode, but a
            # negative value is still a configuration error.  Zero is useful
            # as an explicit placeholder in hand-written YAML.
            if decode_sm < 0:
                raise ValueError(
                    f"manual_divisions[{i}] decode_sm must be non-negative "
                    "when overlap_decode_full_sm is enabled"
                )
        elif decode_sm <= 0:
            raise ValueError(f"manual_divisions[{i}] decode_sm must be positive")
        if threshold < 0:
            raise ValueError(
                f"manual_divisions[{i}] decode_bs_threshold must be non-negative"
            )
        if previous_threshold is not None and threshold <= previous_threshold:
            raise ValueError(
                "manual_divisions decode_bs_threshold values must be "
                "strictly increasing"
            )
        previous_threshold = threshold

    split_forward_token_budget = raw.get("split_forward_token_budget", 65536)
    decode_bs_divisor = raw.get("decode_bs_divisor", 36)
    if split_forward_token_budget <= 0:
        raise ValueError("split_forward_token_budget must be positive")
    if decode_bs_divisor <= 0:
        raise ValueError("decode_bs_divisor must be positive")

    return PDMuxConfig(
        sm_group_num=raw["sm_group_num"],
        manual_divisions=manual_divisions,
        overlap_decode_full_sm=overlap_decode_full_sm,
        split_forward_token_budget=split_forward_token_budget,
        decode_bs_divisor=decode_bs_divisor,
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
    elif major >= 9:
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
    # Keep both sides valid for Green Context, but do not force prefill to own
    # at least half the device.  Large decode batches can be the dominant
    # workload, and denying them the larger partition strands otherwise idle
    # SMs on the prefill side.  Decode retains the existing 16-SM floor used by
    # the decode kernels.
    possible_values = [
        x
        for x in range(min_per_part, total_sms - min_per_part + 1, multiple)
        if total_sms - x >= 16
    ]
    if not possible_values:
        raise ValueError(
            f"No valid partitions found for total SMs {total_sms} "
            f"with constraints (min per part: {min_per_part}, multiple: {multiple})"
        )

    if groups == 1:
        # There is no range to sample with a single shared group. Keep the
        # historical large-prefill choice; larger decode partitions are still
        # available as soon as the caller requests multiple groups.
        selected_values = [possible_values[-1]]
    elif len(possible_values) > groups:
        # Include both endpoints and sample deterministically across the full
        # range.  The old ``x >= total_sms - x`` filter happened to make the
        # largest decode partition unreachable; sampling the complete range
        # preserves a large-prefill first group while adding decode-majority
        # groups at the tail.
        selected_indices = [
            round(i * (len(possible_values) - 1) / (groups - 1))
            for i in range(groups)
        ]
        selected_values = [possible_values[index] for index in selected_indices]
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

    global CURRENT_STREAM_GROUP, CURRENT_STREAM_IDX, SM_GROUP_NUM
    global GREEN_CONTEXT_STREAMS, SM_COUNTS, STREAM_GROUPS
    global _RESERVED_GREEN_STREAMS
    # for pd_multiplexing, Init stream_groups
    device = torch.cuda.current_device()
    total_sm_count = spatial.get_sm_available(gpu_id)
    # (prefill_sm_count, decode_sm_count)
    if config.manual_divisions:
        requested_divisions = [
            (prefill_sm, decode_sm)
            for prefill_sm, decode_sm, _ in config.manual_divisions
        ]
        if config.overlap_decode_full_sm:
            for prefill_sm, _ in requested_divisions:
                if prefill_sm >= total_sm_count:
                    raise ValueError(
                        "overlap_decode_full_sm requires every prefill_sm to be "
                        f"between 1 and {total_sm_count - 1}, got {prefill_sm}"
                    )
            divisions = [
                (prefill_sm, total_sm_count)
                for prefill_sm, _ in requested_divisions
            ]
        else:
            for prefill_sm, decode_sm in requested_divisions:
                if prefill_sm + decode_sm != total_sm_count:
                    raise ValueError(
                        "exclusive PDMux manual divisions must assign every SM: "
                        f"prefill_sm ({prefill_sm}) + decode_sm ({decode_sm}) "
                        f"must equal the device SM count ({total_sm_count}). "
                        "Use overlap_decode_full_sm for a capped prefill with "
                        "full-device decode."
                    )
            divisions = requested_divisions
    else:
        divisions = divide_sm(
            total_sm_count,
            torch.cuda.get_device_capability(device),
            config.sm_group_num - 2,
        )

    SM_COUNTS = []
    SM_COUNTS.append((total_sm_count, 0))  # Normal stream for prefill
    SM_COUNTS.extend(divisions)  # Add the divided SM counts
    SM_COUNTS.append((0, total_sm_count))  # Normal stream for decode
    STREAM_GROUPS = []
    GREEN_CONTEXT_STREAMS = []
    _RESERVED_GREEN_STREAMS = []
    STREAM_GROUPS.append(
        (torch.cuda.Stream(gpu_id), torch.cuda.Stream(gpu_id))
    )  # Normal stream for prefill
    for prefill_sm, decode_sm in divisions:
        if config.overlap_decode_full_sm:
            prefill_stream, reserved_stream = (
                spatial.create_greenctx_stream_by_value(
                    prefill_sm, total_sm_count - prefill_sm, gpu_id
                )
            )
            # Keep the unused half alive because the CUDA extension owns the
            # paired Green Contexts through the returned streams.  Decode uses
            # a high-priority primary-context stream so its implicit device-wide
            # SM mask overlaps the capped prefill mask.
            _RESERVED_GREEN_STREAMS.append(reserved_stream)
            GREEN_CONTEXT_STREAMS.append(prefill_stream)
            decode_stream = torch.cuda.Stream(
                gpu_id, priority=_OVERLAY_DECODE_STREAM_PRIORITY
            )
            STREAM_GROUPS.append((prefill_stream, decode_stream))
        else:
            stream_group = spatial.create_greenctx_stream_by_value(
                prefill_sm, decode_sm, gpu_id
            )
            GREEN_CONTEXT_STREAMS.extend(stream_group)
            STREAM_GROUPS.append(stream_group)
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


def is_green_context_stream(stream_ptr: int) -> bool:
    return any(stream.cuda_stream == stream_ptr for stream in GREEN_CONTEXT_STREAMS)


def get_current_stream_idx() -> int:
    """Get the current stream index."""
    return CURRENT_STREAM_IDX
