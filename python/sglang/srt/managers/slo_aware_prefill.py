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
    objective: str
    ttft_pressure: float
    tpot_pressure: float
    smoothed_ttft_pressure: float
    smoothed_tpot_pressure: float
    has_decode_work: bool
    yield_prefill_to_decode: bool


class SloAwarePrefillController:
    """A lightweight SOLA-inspired controller for prefill admission.

    This controller approximates SOLA's state-aware scheduling in SGLang's
    existing scheduler by changing prefill phase priority and workload size.
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
        self.pressure_alpha = 0.25
        self.objective_margin = 0.10
        self._has_pressure_sample = False
        self._ttft_pressure_ema = 0.0
        self._tpot_pressure_ema = 0.0
        self._last_objective = "ttft"

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
        has_decode_work = self._has_decode_work(running_batch.reqs)
        tpot_pressure = self._tpot_pressure(now, running_batch.reqs)
        smoothed_ttft_pressure, smoothed_tpot_pressure = self._update_pressure(
            ttft_pressure, tpot_pressure
        )
        objective = self._choose_objective(
            smoothed_ttft_pressure, smoothed_tpot_pressure, has_decode_work
        )
        optimize_ttft = objective == "ttft"

        chunked_prefill_size = None
        base_chunk = default_chunked_prefill_size or self.base_chunked_prefill_size
        if base_chunk is not None:
            base_chunk = max(1, min(base_chunk, self.max_prefill_tokens))
            chunked_prefill_size = self._scale_chunk(
                base_chunk, ttft_pressure, tpot_pressure, objective, has_decode_work
            )

        allow_prefill = True
        yield_prefill_to_decode = False
        max_prefill_requests = self._max_prefill_requests(
            default_prefill_max_requests, objective, ttft_pressure, tpot_pressure
        )

        if objective == "tpot" and has_decode_work:
            if ttft_pressure < 1.0:
                allow_prefill = False
                yield_prefill_to_decode = True
            else:
                max_prefill_requests = 1

        if chunked_req is not None:
            allow_prefill = True

        self._last_objective = objective
        return SloAwarePrefillDecision(
            chunked_prefill_size=chunked_prefill_size,
            max_prefill_requests=max_prefill_requests,
            allow_prefill=allow_prefill,
            optimize_ttft=optimize_ttft,
            objective=objective,
            ttft_pressure=ttft_pressure,
            tpot_pressure=tpot_pressure,
            smoothed_ttft_pressure=smoothed_ttft_pressure,
            smoothed_tpot_pressure=smoothed_tpot_pressure,
            has_decode_work=has_decode_work,
            yield_prefill_to_decode=yield_prefill_to_decode,
        )

    def prioritize_waiting_queue(self, waiting_queue: list["Req"]) -> None:
        if not self.prefill_priority_boost or len(waiting_queue) <= 1:
            return
        now = time.perf_counter()
        waiting_queue.sort(key=lambda req: self._request_sort_key(now, req))

    def _update_pressure(self, ttft_pressure: float, tpot_pressure: float) -> tuple[float, float]:
        if not self._has_pressure_sample:
            self._ttft_pressure_ema = ttft_pressure
            self._tpot_pressure_ema = tpot_pressure
            self._has_pressure_sample = True
        else:
            beta = 1.0 - self.pressure_alpha
            self._ttft_pressure_ema = beta * self._ttft_pressure_ema + self.pressure_alpha * ttft_pressure
            self._tpot_pressure_ema = beta * self._tpot_pressure_ema + self.pressure_alpha * tpot_pressure
        return self._ttft_pressure_ema, self._tpot_pressure_ema

    def _choose_objective(
        self, ttft_pressure: float, tpot_pressure: float, has_decode_work: bool
    ) -> str:
        if not has_decode_work:
            return "ttft"
        if ttft_pressure >= 1.0 and tpot_pressure >= 1.0:
            return "ttft" if ttft_pressure >= tpot_pressure else "tpot"
        if ttft_pressure >= 1.0:
            return "ttft"
        if tpot_pressure >= 1.0:
            return "tpot"
        if ttft_pressure > tpot_pressure + self.objective_margin:
            return "ttft"
        if tpot_pressure > ttft_pressure + self.objective_margin:
            return "tpot"
        return self._last_objective

    def _max_prefill_requests(
        self,
        default_prefill_max_requests: Optional[int],
        objective: str,
        ttft_pressure: float,
        tpot_pressure: float,
    ) -> Optional[int]:
        if objective == "ttft":
            if tpot_pressure >= 1.5 and ttft_pressure < tpot_pressure:
                return 1
            return default_prefill_max_requests
        if default_prefill_max_requests is None:
            return 1
        return min(default_prefill_max_requests, 1)

    def _scale_chunk(
        self,
        base_chunk: int,
        ttft_pressure: float,
        tpot_pressure: float,
        objective: str,
        has_decode_work: bool,
    ) -> int:
        if not has_decode_work:
            scale = 1.0 if ttft_pressure >= 1.0 else 0.75
        elif objective == "tpot":
            scale = 0.0 if ttft_pressure < 1.0 else 0.25
        elif tpot_pressure >= 1.5 and ttft_pressure < tpot_pressure:
            scale = 0.25
        elif tpot_pressure >= 1.0:
            scale = 0.5
        elif tpot_pressure >= 0.85:
            scale = 0.75
        elif ttft_pressure >= 1.0:
            scale = 1.0
        else:
            scale = 0.75

        chunk = int(base_chunk * scale)
        chunk = max(self.min_chunk_size, chunk)
        chunk = min(base_chunk, chunk, self.max_prefill_tokens)
        return self._floor_to_unit(chunk)

    def _floor_to_unit(self, value: int) -> int:
        unit = math.lcm(max(self.page_size, 1), max(self.tile_size, 1))
        if value < unit:
            return max(self.page_size, min(value, self.max_prefill_tokens))
        return max(unit, value // unit * unit)

    def _has_decode_work(self, running_reqs: Iterable["Req"]) -> bool:
        return any(
            not req.finished()
            and not req.is_retracted
            and len(req.output_ids) > 0
            for req in running_reqs
        )

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
            if req.finished() or req.is_retracted or len(req.output_ids) == 0:
                continue
            start = req.time_stats.prefill_finished_time
            last = req.time_stats.last_decode_finish_time or now
            if start > 0.0:
                decode_anchor = req.time_stats.last_decode_finish_time or start
                if now > decode_anchor:
                    max_tpot = max(max_tpot, now - decode_anchor)
                if last > start and len(req.output_ids) > 1:
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

    def _request_sort_key(self, now: float, req: "Req") -> tuple[int, float, float]:
        is_prefill = len(req.output_ids) == 0
        entry = req.time_stats.wait_queue_entry_time or req.time_stats.scheduler_recv_time
        wait = now - entry if entry > 0.0 else 0.0
        if self._last_objective == "tpot":
            if not is_prefill:
                tpot = self._request_tpot_pressure(now, req)
                return (0, -tpot, entry)
            return (1, -wait, entry)
        remaining_input = max(req.seqlen - req.num_matched_prefix_tokens, 0)
        predicted_ttft = wait + remaining_input / max(self.max_prefill_tokens, 1)
        return (0 if is_prefill else 1, -predicted_ttft, entry)

    def _request_tpot_pressure(self, now: float, req: "Req") -> float:
        start = req.time_stats.prefill_finished_time
        if start <= 0.0 or self.tpot_slo_s <= 0.0:
            return 0.0
        anchor = req.time_stats.last_decode_finish_time or start
        return max(now - anchor, 0.0) / self.tpot_slo_s
