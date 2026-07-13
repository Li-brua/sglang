from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Optional, Sequence

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


@dataclass
class SloAwarePrefillDecision:
    chunked_prefill_size: Optional[int]
    max_prefill_requests: Optional[int]
    allow_prefill: bool
    optimize_ttft: bool
    ttft_pressure: float
    tpot_pressure: float


class SloAwarePrefillController:
    """A lightweight SOLA-inspired controller for prefill admission.

    This first version controls the amount of prefill work admitted in each
    scheduler iteration. It keeps decode untouched and only changes the local
    `PrefillAdder` inputs when explicitly enabled.
    """

    def __init__(
        self,
        *,
        ttft_slo_ms: float,
        tpot_slo_ms: float,
        base_chunked_prefill_size: Optional[int],
        max_prefill_tokens: int,
        page_size: int,
        tile_size: int,
        min_chunk_size: Optional[int],
        prefill_priority_boost: bool,
    ) -> None:
        self.ttft_slo_s = ttft_slo_ms / 1000.0
        self.tpot_slo_s = tpot_slo_ms / 1000.0
        self.base_chunked_prefill_size = base_chunked_prefill_size
        self.max_prefill_tokens = max_prefill_tokens
        self.page_size = page_size
        self.tile_size = max(tile_size, page_size, 1)
        self.min_chunk_size = min_chunk_size or self.tile_size
        self.min_chunk_size = max(self.min_chunk_size, page_size, 1)
        self.prefill_priority_boost = prefill_priority_boost

    def make_decision(
        self,
        *,
        waiting_queue: Sequence["Req"],
        running_batch: "ScheduleBatch",
        chunked_req: Optional["Req"],
        default_chunked_prefill_size: Optional[int],
        default_prefill_max_requests: Optional[int],
    ) -> SloAwarePrefillDecision:
        now = time.perf_counter()
        ttft_pressure = self._ttft_pressure(now, waiting_queue, chunked_req)
        tpot_pressure = self._tpot_pressure(now, running_batch.reqs)
        optimize_ttft = ttft_pressure >= tpot_pressure

        chunked_prefill_size = None
        base_chunk = default_chunked_prefill_size or self.base_chunked_prefill_size
        if base_chunk is not None:
            base_chunk = max(1, min(base_chunk, self.max_prefill_tokens))
            chunked_prefill_size = self._scale_chunk(
                base_chunk, ttft_pressure, tpot_pressure
            )
        allow_prefill = True
        max_prefill_requests = default_prefill_max_requests

        if tpot_pressure >= 1.0 and ttft_pressure < tpot_pressure:
            if ttft_pressure < 0.5:
                allow_prefill = False
            max_prefill_requests = 1

        if chunked_req is not None:
            allow_prefill = True

        return SloAwarePrefillDecision(
            chunked_prefill_size=chunked_prefill_size,
            max_prefill_requests=max_prefill_requests,
            allow_prefill=allow_prefill,
            optimize_ttft=optimize_ttft,
            ttft_pressure=ttft_pressure,
            tpot_pressure=tpot_pressure,
        )

    def prioritize_waiting_queue(self, waiting_queue: list["Req"]) -> None:
        if not self.prefill_priority_boost or len(waiting_queue) <= 1:
            return
        now = time.perf_counter()
        waiting_queue.sort(key=lambda req: self._prefill_sort_key(now, req))

    def _scale_chunk(
        self, base_chunk: int, ttft_pressure: float, tpot_pressure: float
    ) -> int:
        if tpot_pressure >= 1.0 and ttft_pressure < tpot_pressure:
            scale = 0.25 if ttft_pressure < 0.5 else 0.5
        elif tpot_pressure >= 0.85 and ttft_pressure < tpot_pressure:
            scale = 0.5
        elif ttft_pressure >= 1.0 and ttft_pressure >= tpot_pressure:
            scale = 1.0
        elif ttft_pressure > tpot_pressure:
            scale = 0.75
        else:
            scale = 0.5

        chunk = int(base_chunk * scale)
        chunk = max(self.min_chunk_size, chunk)
        chunk = min(base_chunk, chunk, self.max_prefill_tokens)
        return self._floor_to_unit(chunk)

    def _floor_to_unit(self, value: int) -> int:
        unit = math.lcm(max(self.page_size, 1), max(self.tile_size, 1))
        if value < unit:
            return max(self.page_size, min(value, self.max_prefill_tokens))
        return max(unit, value // unit * unit)

    def _ttft_pressure(
        self, now: float, waiting_queue: Sequence["Req"], chunked_req: Optional["Req"]
    ) -> float:
        if self.ttft_slo_s <= 0:
            return 0.0
        max_wait = 0.0
        for req in self._prefill_candidates(waiting_queue, chunked_req):
            entry = req.time_stats.wait_queue_entry_time or req.time_stats.scheduler_recv_time
            if entry > 0.0:
                max_wait = max(max_wait, now - entry)
        return max_wait / self.ttft_slo_s

    def _tpot_pressure(self, now: float, running_reqs: Iterable["Req"]) -> float:
        if self.tpot_slo_s <= 0:
            return 0.0
        max_tpot = 0.0
        for req in running_reqs:
            if req.finished() or req.is_retracted or len(req.output_ids) <= 1:
                continue
            start = req.time_stats.prefill_finished_time
            last = req.time_stats.last_decode_finish_time or now
            if start <= 0.0 or last <= start:
                continue
            decode_tokens = max(len(req.output_ids) - 1, 1)
            max_tpot = max(max_tpot, (last - start) / decode_tokens)
        return max_tpot / self.tpot_slo_s

    def _prefill_candidates(
        self, waiting_queue: Sequence["Req"], chunked_req: Optional["Req"]
    ) -> Iterable["Req"]:
        if chunked_req is not None:
            yield chunked_req
        for req in waiting_queue:
            if len(req.output_ids) == 0:
                yield req

    def _prefill_sort_key(self, now: float, req: "Req") -> tuple[int, float, float]:
        is_prefill = len(req.output_ids) == 0
        entry = req.time_stats.wait_queue_entry_time or req.time_stats.scheduler_recv_time
        wait = now - entry if entry > 0.0 else 0.0
        remaining_input = max(req.seqlen - req.num_matched_prefix_tokens, 0)
        predicted_ttft = wait + remaining_input / max(self.max_prefill_tokens, 1)
        return (0 if is_prefill else 1, -predicted_ttft, entry)
