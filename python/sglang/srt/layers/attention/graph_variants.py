from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional, Protocol

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


logger = logging.getLogger(__name__)

DSA_DENSE = "dense"
DSA_SPARSE = "sparse"
DSV41_CANDIDATE_FILTERED = "candidate_filtered"


class AttentionGraphVariants(Protocol):
    # Capture order is significant when variants share a graph memory pool.
    capture_labels: tuple[str, ...]

    def select(self, forward_batch: ForwardBatch) -> str:
        """Select one of capture_labels for the batch."""
        ...


@dataclass(frozen=True)
class DsaGraphVariants:
    index_topk: int
    # Dense comes first: the sparse capture peak subsumes its shared-pool storage.
    capture_labels: ClassVar[tuple[str, ...]] = (DSA_DENSE, DSA_SPARSE)

    def select(self, forward_batch: ForwardBatch) -> str:
        seq_lens_cpu = forward_batch.seq_lens_cpu
        if seq_lens_cpu is not None and seq_lens_cpu.numel() > 0:
            # Plain decode maintains this host mirror without a D2H sync.
            max_kv_len = int(seq_lens_cpu.max().item())
        elif forward_batch.seq_lens is not None and forward_batch.seq_lens.numel() > 0:
            # Fallback: a single scalar reduction d2h (cheap, per-step).
            max_kv_len = int(forward_batch.seq_lens.max().item())
        else:
            # No length info: be safe and use the correct-for-all sparse graph.
            return DSA_SPARSE
        return DSA_DENSE if max_kv_len <= self.index_topk else DSA_SPARSE


@dataclass(frozen=True)
class Dsv41CandidateGraphVariants:
    """Select exact short-history candidate-indexer graph variants."""

    graph_limits: tuple[tuple[str, int], ...]
    capture_labels: tuple[str, ...]

    def select(self, forward_batch: ForwardBatch) -> str:
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        if (
            seq_lens_cpu is None
            or seq_lens_cpu.numel() == 0
            or seq_lens_cpu.device.type != "cpu"
        ):
            return DSV41_CANDIDATE_FILTERED
        max_seq_len = int(seq_lens_cpu.max().item())
        for variant, limit in self.graph_limits:
            if max_seq_len <= limit:
                return variant
        return DSV41_CANDIDATE_FILTERED


def _create_dsv41_candidate_variants(text_config) -> Optional[AttentionGraphVariants]:
    if (
        getattr(text_config, "model_type", None) != "deepseek_v41"
        or getattr(text_config, "candidate_source_layer_id", -1) < 0
    ):
        return None

    span = (
        getattr(text_config, "candidate_topk_blocks", 0)
        * getattr(text_config, "candidate_block_size", 0)
    )
    if span <= 0:
        return None

    ratios = set(getattr(text_config, "compress_ratios", ())) & {1, 2}
    topk = getattr(text_config, "index_topk", 0)
    candidates = []
    if topk > 0 and ratios:
        candidates.append(("candidate_all", topk * min(ratios)))
        if ratios == {1, 2}:
            candidates.append(("candidate_c2_all", topk * 2))
    candidates.append(("candidate_unfiltered", span))

    limits = []
    for variant, limit in candidates:
        limits.append((variant, min(limit, span)))
        if limit >= span:
            break
    graph_limits = tuple(limits)
    capture_labels = tuple(variant for variant, _ in graph_limits) + (
        DSV41_CANDIDATE_FILTERED,
    )
    logger.info(
        "Candidate indexer graph limits: %s; use full filtering above %s.",
        graph_limits,
        span,
    )
    return Dsv41CandidateGraphVariants(graph_limits, capture_labels)


def create_attention_graph_variants(
    hf_config, *, enable_dsv41_candidates: bool = False
) -> Optional[AttentionGraphVariants]:
    from sglang.srt.configs.model_config import get_dsa_index_topk, is_deepseek_dsa
    from sglang.srt.utils import is_hip

    if is_hip() and is_deepseek_dsa(hf_config):
        index_topk = get_dsa_index_topk(hf_config)
        logger.info(
            "[dense-decode] DSA dual-graph enabled: capturing "
            "dense (k-only) + sparse (full indexer) decode graphs; "
            "dispatch on max_kv_len vs index_topk=%d.",
            index_topk,
        )
        return DsaGraphVariants(index_topk)
    if enable_dsv41_candidates:
        text_config = getattr(hf_config, "text_config", hf_config)
        return _create_dsv41_candidate_variants(text_config)
    return None
