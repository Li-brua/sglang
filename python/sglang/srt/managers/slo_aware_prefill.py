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
    prefill_cost_per_token_s: float
    decode_cost_s: float
    ttft_stat: str
    tpot_stat: str


@dataclass(frozen=True)
class SloAwarePrefillPressureState:
    ttft_pressure: float
    tpot_pressure: float
    has_decode_work: bool
    prefill_cost_per_token_s: float = 0.0
    decode_cost_s: float = 0.0


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
        ttft_stat: str = "max",
        tpot_stat: str = "max",
        initial_prefill_cost_ms_per_1k: Optional[float] = None,
        initial_decode_cost_ms: Optional[float] = None,
        disable_online_cost_model: bool = False,
        yield_guard_ratio: float = 0.05,
        enable_dp_attention: bool = False,
        dp_size: int = 1,
    ) -> None:
        self.ttft_slo_s = ttft_slo_ms / 1000.0
        self.tpot_slo_s = tpot_slo_ms / 1000.0
        self.base_chunked_prefill_size = base_chunked_prefill_size
        self.max_prefill_tokens = max_prefill_tokens
        self.page_size = page_size
        self.tile_size = max(tile_size, page_size, 1)
        self.min_chunk_size = self._resolve_min_chunk_size(
            min_chunk_size=min_chunk_size,
            enable_dp_attention=enable_dp_attention,
            dp_size=dp_size,
        )
        self.prefill_priority_boost = prefill_priority_boost
        self.ttft_stat = self._normalize_pressure_stat(ttft_stat)
        self.tpot_stat = self._normalize_pressure_stat(tpot_stat)
        self.pressure_alpha = 0.25
        self.cost_alpha = 0.20
        self.objective_margin = 0.10
        self.yield_guard_ratio = max(yield_guard_ratio, 0.0)
        self.hard_prefill_ttft_pressure = 1.50
        default_cost_tokens = max(
            self.base_chunked_prefill_size or self.max_prefill_tokens,
            self.min_chunk_size,
            1,
        )
        self.default_prefill_cost_per_token_s = max(
            self.tpot_slo_s / default_cost_tokens, 1e-6
        )
        self.default_decode_cost_s = max(min(self.tpot_slo_s, 0.05), 1e-4)
        if initial_prefill_cost_ms_per_1k is not None:
            self.default_prefill_cost_per_token_s = max(
                initial_prefill_cost_ms_per_1k / 1_000_000.0, 1e-9
            )
        if initial_decode_cost_ms is not None:
            self.default_decode_cost_s = max(initial_decode_cost_ms / 1000.0, 1e-9)
        self._prefill_cost_points_s: list[tuple[int, float]] = []
        self._decode_cost_points_s: list[tuple[int, float]] = []
        self._prefill_cost_per_token_s = self.default_prefill_cost_per_token_s
        self._decode_cost_s = self.default_decode_cost_s
        self.disable_online_cost_model = disable_online_cost_model
        self._active_prefill_cost_per_token_s = self._prefill_cost_per_token_s
        self._active_decode_cost_s = self._decode_cost_s
        self._has_prefill_cost_sample = False
        self._has_decode_cost_sample = False
        self._has_pressure_sample = False
        self._ttft_pressure_ema = 0.0
        self._tpot_pressure_ema = 0.0
        self._last_objective = "ttft"

    def _resolve_min_chunk_size(
        self,
        *,
        min_chunk_size: Optional[int],
        enable_dp_attention: bool,
        dp_size: int,
    ) -> int:
        min_chunk = min_chunk_size or self.tile_size
        if enable_dp_attention and min_chunk_size is not None and dp_size > 1:
            min_chunk = math.ceil(min_chunk / dp_size)
        return max(min_chunk, self.page_size, 1)

    def make_decision(
        self,
        *,
        waiting_queue: Sequence["Req"],
        running_batch: "ScheduleBatch",
        chunked_req: Optional["Req"],
        default_chunked_prefill_size: Optional[int],
        default_prefill_max_requests: Optional[int],
    ) -> SloAwarePrefillDecision:
        pressure_state = self.compute_pressure_state(
            waiting_queue=waiting_queue,
            running_batch=running_batch,
            chunked_req=chunked_req,
        )
        return self.make_decision_from_pressure_state(
            pressure_state=pressure_state,
            chunked_req=chunked_req,
            default_chunked_prefill_size=default_chunked_prefill_size,
            default_prefill_max_requests=default_prefill_max_requests,
        )

    def compute_pressure_state(
        self,
        *,
        waiting_queue: Sequence["Req"],
        running_batch: "ScheduleBatch",
        chunked_req: Optional["Req"],
    ) -> SloAwarePrefillPressureState:
        now = time.perf_counter()
        decode_reqs = self._decode_reqs(running_batch.reqs)
        return SloAwarePrefillPressureState(
            ttft_pressure=self._ttft_pressure(now, waiting_queue, chunked_req),
            tpot_pressure=self._tpot_pressure(now, decode_reqs),
            has_decode_work=len(decode_reqs) > 0,
            prefill_cost_per_token_s=self._prefill_cost_per_token_s,
            decode_cost_s=self._decode_cost_for_batch(len(decode_reqs)),
        )

    def make_decision_from_pressure_state(
        self,
        *,
        pressure_state: SloAwarePrefillPressureState,
        chunked_req: Optional["Req"],
        default_chunked_prefill_size: Optional[int],
        default_prefill_max_requests: Optional[int],
    ) -> SloAwarePrefillDecision:
        ttft_pressure = pressure_state.ttft_pressure
        tpot_pressure = pressure_state.tpot_pressure
        has_decode_work = pressure_state.has_decode_work
        self._active_prefill_cost_per_token_s = (
            pressure_state.prefill_cost_per_token_s or self._prefill_cost_per_token_s
        )
        self._active_decode_cost_s = pressure_state.decode_cost_s or self._decode_cost_s
        smoothed_ttft_pressure, smoothed_tpot_pressure = self._update_pressure(
            ttft_pressure, tpot_pressure
        )
        objective = self._choose_objective(
            ttft_pressure, tpot_pressure, has_decode_work
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
            max_prefill_requests = 1
            if self._can_yield_to_decode(ttft_pressure):
                allow_prefill = False
                yield_prefill_to_decode = True

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
            prefill_cost_per_token_s=self._active_prefill_cost_per_token_s,
            decode_cost_s=self._active_decode_cost_s,
            ttft_stat=self.ttft_stat,
            tpot_stat=self.tpot_stat,
        )

    def _normalize_pressure_stat(self, stat: str) -> str:
        if stat not in ("max", "mean", "p90"):
            raise ValueError(f"Unsupported SLO pressure stat: {stat}")
        return stat

    def set_startup_cost_profile(
        self,
        *,
        prefill_cost_ms: Sequence[tuple[int, float]],
        decode_cost_ms: Sequence[tuple[int, float]],
    ) -> None:
        prefill_points = [
            (int(tokens), float(cost_ms) / 1000.0)
            for tokens, cost_ms in prefill_cost_ms
            if tokens > 0 and cost_ms > 0.0
        ]
        decode_points = [
            (int(batch_size), float(cost_ms) / 1000.0)
            for batch_size, cost_ms in decode_cost_ms
            if batch_size > 0 and cost_ms > 0.0
        ]
        self._prefill_cost_points_s = self._monotonic_cost_points(prefill_points)
        self._decode_cost_points_s = self._monotonic_cost_points(decode_points)
        if self._prefill_cost_points_s:
            tokens, cost_s = self._prefill_cost_points_s[-1]
            self._prefill_cost_per_token_s = max(cost_s / max(tokens, 1), 1e-9)
        if self._decode_cost_points_s:
            self._decode_cost_s = self._decode_cost_for_batch(1)

    def observe_batch_cost(
        self, *, prefill_tokens: int, decode_tokens: int, elapsed_s: float
    ) -> None:
        if self.disable_online_cost_model or elapsed_s <= 0.0:
            return
        if prefill_tokens > 0:
            effective_elapsed = elapsed_s
            if decode_tokens > 0 and self._has_decode_cost_sample:
                effective_elapsed = max(elapsed_s - self._decode_cost_s, elapsed_s * 0.25)
            sample = effective_elapsed / max(prefill_tokens, 1)
            self._prefill_cost_per_token_s = self._ema_cost(
                self._prefill_cost_per_token_s, sample, self._has_prefill_cost_sample
            )
            self._has_prefill_cost_sample = True
        elif decode_tokens > 0:
            self._decode_cost_s = self._ema_cost(
                self._decode_cost_s, elapsed_s, self._has_decode_cost_sample
            )
            self._has_decode_cost_sample = True

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

        ttft_pressure = self._quantize_pressure(ttft_pressure)
        tpot_pressure = self._quantize_pressure(tpot_pressure)
        margin = self.objective_margin

        if tpot_pressure >= 1.0 and ttft_pressure < 1.0:
            return "tpot"
        if ttft_pressure >= 1.0 and tpot_pressure < 1.0:
            return "ttft"
        if tpot_pressure > ttft_pressure + margin:
            return "tpot"
        return "ttft"

    def _quantize_pressure(self, pressure: float) -> float:
        return round(pressure, 2)

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
            return self._floor_to_unit(base_chunk)

        if objective == "ttft":
            chunk = self._ttft_objective_token_budget(
                base_chunk, ttft_pressure, tpot_pressure
            )
        else:
            chunk = self._tpot_objective_token_budget(
                base_chunk, ttft_pressure, tpot_pressure
            )

        chunk = max(self.min_chunk_size, chunk)
        chunk = min(base_chunk, chunk, self.max_prefill_tokens)
        return self._floor_to_unit(chunk)

    def _ttft_objective_token_budget(
        self, base_chunk: int, ttft_pressure: float, tpot_pressure: float
    ) -> int:
        if self.tpot_slo_s <= 0.0:
            return base_chunk
        if ttft_pressure >= self.hard_prefill_ttft_pressure:
            return base_chunk
        if tpot_pressure >= 1.0 and ttft_pressure < tpot_pressure:
            return self.min_chunk_size

        if ttft_pressure >= 1.0 and tpot_pressure <= 0.5:
            return base_chunk

        tpot_slack_s = max((1.0 - tpot_pressure) * self.tpot_slo_s, 0.0)
        token_budget = self._prefill_tokens_for_latency(tpot_slack_s)
        if ttft_pressure >= 1.0:
            token_budget = max(token_budget, base_chunk // 2)
        return max(self.min_chunk_size, min(base_chunk, token_budget))

    def _tpot_objective_token_budget(
        self, base_chunk: int, ttft_pressure: float, tpot_pressure: float
    ) -> int:
        if self._has_ttft_slack_for_decode_yield(ttft_pressure):
            return self.min_chunk_size

        tpot_prefill_budget_s = self._tpot_prefill_budget_s(tpot_pressure)
        if tpot_prefill_budget_s > 0.0:
            return self._prefill_tokens_for_latency(tpot_prefill_budget_s)

        return self._ttft_recovery_prefill_budget(base_chunk, ttft_pressure)

    def _tpot_prefill_budget_s(self, tpot_pressure: float) -> float:
        tpot_slack_s = max((1.0 - tpot_pressure) * self.tpot_slo_s, 0.0)
        min_prefill_cost_s = self._estimate_prefill_cost(self.min_chunk_size)
        guard_s = self._prefill_budget_guard_s(min_prefill_cost_s)
        return tpot_slack_s - self._active_decode_cost_s - guard_s

    def _prefill_budget_guard_s(self, min_prefill_cost_s: float) -> float:
        return max(
            0.5 * self._active_decode_cost_s,
            0.2 * min_prefill_cost_s,
            self.yield_guard_ratio * self.tpot_slo_s,
        )

    def _ttft_recovery_prefill_budget(
        self, base_chunk: int, ttft_pressure: float
    ) -> int:
        if ttft_pressure >= self.hard_prefill_ttft_pressure:
            return base_chunk
        if ttft_pressure < 1.0:
            return self.min_chunk_size

        min_prefill_cost_s = self._estimate_prefill_cost(self.min_chunk_size)
        ttft_debt_s = (ttft_pressure - 1.0) * self.ttft_slo_s
        recovery_budget_s = ttft_debt_s + min_prefill_cost_s
        return self._prefill_tokens_for_latency(recovery_budget_s)

    def _can_yield_to_decode(self, ttft_pressure: float) -> bool:
        return self._has_ttft_slack_for_decode_yield(ttft_pressure)

    def _has_ttft_slack_for_decode_yield(self, ttft_pressure: float) -> bool:
        if ttft_pressure >= 1.0:
            return False
        ttft_slack_s = max((1.0 - ttft_pressure) * self.ttft_slo_s, 0.0)
        min_prefill_cost_s = self._estimate_prefill_cost(self.min_chunk_size)
        guard_s = self._yield_guard_s(min_prefill_cost_s)
        return (
            ttft_slack_s
            > self._active_decode_cost_s + min_prefill_cost_s + guard_s
        )

    def _yield_guard_s(self, min_prefill_cost_s: float) -> float:
        return max(
            2.0 * self._active_decode_cost_s,
            0.2 * min_prefill_cost_s,
            self.yield_guard_ratio * self.ttft_slo_s,
        )

    def _prefill_tokens_for_latency(self, latency_s: float) -> int:
        if latency_s <= 0.0:
            return self.min_chunk_size
        if self._prefill_cost_points_s:
            return max(
                self.min_chunk_size,
                self._tokens_for_profiled_prefill_cost(latency_s),
            )
        return int(latency_s / max(self._active_prefill_cost_per_token_s, 1e-9))

    def _estimate_prefill_cost(self, tokens: int) -> float:
        tokens = max(tokens, 0)
        if tokens == 0:
            return 0.0
        if self._prefill_cost_points_s:
            return self._profiled_cost(self._prefill_cost_points_s, tokens)
        return tokens * max(self._active_prefill_cost_per_token_s, 1e-9)

    def _decode_cost_for_batch(self, batch_size: int) -> float:
        if batch_size <= 0 or not self._decode_cost_points_s:
            return self._decode_cost_s
        return self._profiled_cost(self._decode_cost_points_s, batch_size)

    def _profiled_cost(self, points: list[tuple[int, float]], size: int) -> float:
        if size <= 0:
            return 0.0
        if len(points) == 1:
            point_size, point_cost_s = points[0]
            return point_cost_s * size / max(point_size, 1)
        if size <= points[0][0]:
            return points[0][1] * size / max(points[0][0], 1)
        for left, right in zip(points, points[1:]):
            left_size, left_cost_s = left
            right_size, right_cost_s = right
            if size <= right_size:
                ratio = (size - left_size) / max(right_size - left_size, 1)
                return left_cost_s + ratio * (right_cost_s - left_cost_s)
        prev_size, prev_cost_s = points[-2]
        last_size, last_cost_s = points[-1]
        slope = (last_cost_s - prev_cost_s) / max(last_size - prev_size, 1)
        slope = max(slope, last_cost_s / max(last_size, 1), 1e-9)
        return last_cost_s + (size - last_size) * slope

    def _monotonic_cost_points(
        self, points: list[tuple[int, float]]
    ) -> list[tuple[int, float]]:
        points = sorted(points)
        ret = []
        max_cost = 0.0
        for size, cost_s in points:
            if ret and size == ret[-1][0]:
                max_cost = max(ret[-1][1], cost_s, max_cost)
                ret[-1] = (size, max_cost)
                continue
            max_cost = max(cost_s, max_cost)
            ret.append((size, max_cost))
        return ret

    def _tokens_for_profiled_prefill_cost(self, latency_s: float) -> int:
        points = self._prefill_cost_points_s
        if latency_s <= points[0][1]:
            tokens = points[0][0] * latency_s / max(points[0][1], 1e-9)
            return int(tokens)
        for left, right in zip(points, points[1:]):
            left_size, left_cost_s = left
            right_size, right_cost_s = right
            if latency_s <= right_cost_s:
                ratio = (latency_s - left_cost_s) / max(
                    right_cost_s - left_cost_s, 1e-9
                )
                return int(left_size + ratio * (right_size - left_size))
        prev_size, prev_cost_s = points[-2] if len(points) > 1 else (0, 0.0)
        last_size, last_cost_s = points[-1]
        slope = (last_cost_s - prev_cost_s) / max(last_size - prev_size, 1)
        slope = max(slope, last_cost_s / max(last_size, 1), 1e-9)
        return int(last_size + (latency_s - last_cost_s) / slope)

    def _ema_cost(self, old: float, sample: float, has_sample: bool) -> float:
        sample = max(sample, 1e-9)
        if not has_sample:
            return sample
        beta = 1.0 - self.cost_alpha
        return beta * old + self.cost_alpha * sample

    def _floor_to_unit(self, value: int) -> int:
        unit = math.lcm(max(self.page_size, 1), max(self.tile_size, 1))
        if value < unit:
            return max(self.page_size, min(value, self.max_prefill_tokens))
        return max(unit, value // unit * unit)

    def _has_decode_work(self, running_reqs: Iterable["Req"]) -> bool:
        return len(self._decode_reqs(running_reqs)) > 0

    def _decode_reqs(self, running_reqs: Iterable["Req"]) -> list["Req"]:
        return [
            req
            for req in running_reqs
            if not req.finished()
            and not req.is_retracted
            and len(req.output_ids) > 0
        ]

    def _ttft_pressure(
        self, now: float, waiting_queue: Sequence["Req"], chunked_req: Optional["Req"]
    ) -> float:
        if self.ttft_slo_s <= 0:
            return 0.0
        pressures = []
        for req in self._prefill_candidates(waiting_queue, chunked_req):
            entry = req.time_stats.wait_queue_entry_time or req.time_stats.scheduler_recv_time
            if entry > 0.0:
                pressures.append((now - entry) / self.ttft_slo_s)
        return self._aggregate_pressure(pressures, self.ttft_stat)

    def _tpot_pressure(self, now: float, running_reqs: Iterable["Req"]) -> float:
        if self.tpot_slo_s <= 0:
            return 0.0
        pressures = []
        for req in running_reqs:
            if req.finished() or req.is_retracted or len(req.output_ids) == 0:
                continue
            start = req.time_stats.prefill_finished_time
            last = req.time_stats.last_decode_finish_time or now
            if start > 0.0:
                tpot_s = 0.0
                decode_anchor = req.time_stats.last_decode_finish_time or start
                if now > decode_anchor:
                    tpot_s = max(tpot_s, now - decode_anchor)
                if last > start and len(req.output_ids) > 1:
                    decode_tokens = max(len(req.output_ids) - 1, 1)
                    tpot_s = max(tpot_s, (last - start) / decode_tokens)
                pressures.append(tpot_s / self.tpot_slo_s)
        return self._aggregate_pressure(pressures, self.tpot_stat)

    def _aggregate_pressure(self, pressures: Sequence[float], stat: str) -> float:
        if len(pressures) == 0:
            return 0.0
        if stat == "mean":
            return sum(pressures) / len(pressures)
        if stat == "p90":
            sorted_pressures = sorted(pressures)
            index = max(math.ceil(0.90 * len(sorted_pressures)) - 1, 0)
            return sorted_pressures[min(index, len(sorted_pressures) - 1)]
        return max(pressures)

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
                predicted_tpot = self._request_predicted_tpot(now, req)
                return (0, -predicted_tpot, entry)
            return (1, -wait, entry)
        remaining_input = max(req.seqlen - req.num_matched_prefix_tokens, 0)
        predicted_ttft = wait + self._estimate_prefill_cost(remaining_input)
        return (0 if is_prefill else 1, -predicted_ttft, entry)

    def _request_predicted_tpot(self, now: float, req: "Req") -> float:
        output_len = max(len(req.output_ids), 1)
        current_tpot_s = self._request_tpot_seconds(now, req)
        remaining_output_len = self._predicted_remaining_output_len(req)
        total_output_len = output_len + remaining_output_len
        if total_output_len <= 0:
            return current_tpot_s
        return (
            current_tpot_s * output_len
            + self._active_decode_cost_s * remaining_output_len
        ) / total_output_len

    def _request_tpot_seconds(self, now: float, req: "Req") -> float:
        start = req.time_stats.prefill_finished_time
        if start <= 0.0:
            return 0.0
        anchor = req.time_stats.last_decode_finish_time or start
        gap = max(now - anchor, 0.0)
        if req.time_stats.last_decode_finish_time > start and len(req.output_ids) > 1:
            avg = (req.time_stats.last_decode_finish_time - start) / max(
                len(req.output_ids) - 1, 1
            )
            return max(gap, avg)
        return gap

    def _predicted_remaining_output_len(self, req: "Req") -> int:
        max_new_tokens = getattr(getattr(req, "sampling_params", None), "max_new_tokens", 0)
        if max_new_tokens is None or max_new_tokens <= 0:
            return max(1, len(req.output_ids))
        return max(max_new_tokens - len(req.output_ids), 1)
