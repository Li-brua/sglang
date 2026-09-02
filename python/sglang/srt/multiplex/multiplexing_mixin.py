"""
Mixin class providing multiplexing scheduling logic
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist
from torch.cuda.streams import ExternalStream

from sglang.srt.distributed.parallel_state import set_pdmux_status
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.multiplex.pdmux_context import (
    get_current_stream_idx,
    get_sm_counts,
    get_stream_groups,
    initialize_stream_groups,
    load_pdmux_config,
    set_current_stream_idx,
)
from sglang.srt.runtime_context import get_disagg

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerMultiplexMixin:
    def init_pdmux(self: Scheduler):
        # The current split prefill batch
        self.split_prefill_batch: Optional[ScheduleBatch] = None

        # for pd_multiplexing, Init stream_groups, exclude normal stream for prefill only and decode only
        self.pdmux_config = load_pdmux_config(get_disagg().pdmux_config_path)
        initialize_stream_groups(self.ps.gpu_id, self.pdmux_config)
        self.stream_groups = get_stream_groups()
        self.sm_counts = get_sm_counts()
        self.real_sm_group_num = len(self.stream_groups)
        logger.info(
            f"PD-Multiplexing enabled with {self.real_sm_group_num} stream groups, sm_counts (prefill_sm, decode_sm): {self.sm_counts}"
        )

    # TODO(jason-fxz): This is a temporary demo
    def adjust_stream_groups(
        self: Scheduler, running_batch: ScheduleBatch
    ) -> tuple[int, tuple[ExternalStream, ExternalStream]]:
        if not running_batch.is_empty() and self.split_prefill_batch:
            decode_bs = running_batch.batch_size()
            manual_divisions = self.pdmux_config.manual_divisions
            if manual_divisions:
                # A decode batch under every configured threshold still has to
                # land on a shared group: index 0 is the prefill-only stream,
                # which would starve decode entirely.
                stream_idx = 1
                for i in range(len(manual_divisions)):
                    _, _, threshold = manual_divisions[i]
                    if decode_bs >= threshold:
                        stream_idx = i + 1
            else:
                stream_idx = max(
                    1,
                    min(
                        self.real_sm_group_num - 2,
                        decode_bs
                        * (self.real_sm_group_num - 2)
                        // self.pdmux_config.decode_bs_divisor,
                    ),
                )
            set_current_stream_idx(stream_idx)
        elif not running_batch.is_empty():
            set_current_stream_idx(self.real_sm_group_num - 1)
        else:
            set_current_stream_idx(0)

        stream_idx = get_current_stream_idx()

        self.tp_worker.model_runner.update_decode_attn_backend(stream_idx)
        return stream_idx, self.stream_groups[stream_idx]

    def update_split_prefill_batch(
        self: Scheduler, sm_count: int, running_batch: ScheduleBatch
    ) -> tuple[bool, ScheduleBatch]:
        if self.split_prefill_batch:
            return False, running_batch

        # No split forward is in flight here, which matches the normal loop's
        # "top of the scheduling step" safe point for tearing down an aborted
        # chunked request before its next chunk is formed.
        self.process_pending_chunked_abort()

        # add new request
        prefill_plan = self.get_new_batch_prefill(running_batch)
        batch = prefill_plan.batch_to_run
        running_batch = prefill_plan.running_batch
        if batch and not batch.is_empty():
            batch.forward_mode = (
                ForwardMode.SPLIT_PREFILL
            )  # Set forward mode for split prefill
            self.split_prefill_batch = batch
            return True, running_batch
        return False, running_batch

    def _get_split_forward_count(self: Scheduler) -> int:
        remaining_layers = (
            self.model_config.num_hidden_layers - self.split_prefill_batch.split_index
        )

        # Splitting only benefits decode work that can run between prefill
        # intervals. Without decode work, finish prefill in one model call to
        # avoid repeating the full scheduler/model-runner setup per layer.
        if self.running_batch is None or self.running_batch.is_empty():
            return remaining_layers

        if self.split_prefill_batch.extend_num_tokens <= 0:
            return remaining_layers

        forward_count = max(
            1,
            self.pdmux_config.split_forward_token_budget
            // self.split_prefill_batch.extend_num_tokens,
        )
        return min(forward_count, remaining_layers)

    def init_pdmux_prefill_plan_limit(
        self: Scheduler, attn_backend: AttentionBackend
    ) -> None:
        """Resolve the backend's hard prefill-plan token limit for admission.

        Without chunked prefill, PDMux cannot split an oversized request, so
        request validation and admission are clamped to the limit. With chunked
        prefill, every forward's aggregate extend tokens are capped by the
        chunk budget instead, which must itself fit one plan.
        """
        self.pdmux_max_prefill_plan_tokens = (
            attn_backend.max_prefill_plan_tokens if self.enable_pdmux else None
        )
        if self.pdmux_max_prefill_plan_tokens is None:
            return
        logger.info(
            "PDMux prefill planner hard limit: %s tokens",
            self.pdmux_max_prefill_plan_tokens,
        )
        if self.chunked_prefill_size is not None:
            SchedulerMultiplexMixin._validate_pdmux_chunked_prefill_size(self)
        else:
            self.max_req_input_len = SchedulerMultiplexMixin._get_max_req_input_len(
                self, self.max_req_input_len
            )

    def _validate_pdmux_chunked_prefill_size(self: Scheduler) -> None:
        # Dynamic chunking could raise a batch's chunk size past the static
        # value, but it is PP-only and PDMux asserts pp_size == 1, so the
        # configured size is the true per-forward bound.
        hard_limit = self.pdmux_max_prefill_plan_tokens
        aligned_limit = hard_limit - hard_limit % self.page_size
        if self.chunked_prefill_size > aligned_limit:
            raise ValueError(
                f"--chunked-prefill-size ({self.chunked_prefill_size}) exceeds "
                f"the attention backend's prefill plan limit of {aligned_limit} "
                f"tokens (page-aligned). Set --chunked-prefill-size to at most "
                f"{aligned_limit}."
            )

    def _get_pdmux_prefill_token_limit(
        self: Scheduler, max_prefill_tokens: int
    ) -> Optional[int]:
        hard_limit = self.pdmux_max_prefill_plan_tokens
        if not (self.enable_pdmux and hard_limit is not None):
            return None
        if self.chunked_prefill_size is not None:
            # The chunk budget caps every forward's aggregate extend tokens
            # below the planner limit (validated at init), so neither request
            # validation nor admission needs the hard clamp.
            return None

        # PrefillAdder accounts input tokens in page-aligned units. Align the
        # backend's raw-token limit down so every accepted request can consume
        # the admission budget instead of remaining in the waiting queue.
        limit = min(max_prefill_tokens, hard_limit)
        return limit - limit % self.page_size

    def _get_prefill_admission_config(
        self: Scheduler, max_prefill_tokens: int
    ) -> tuple[int, bool]:
        effective_limit = SchedulerMultiplexMixin._get_pdmux_prefill_token_limit(
            self, max_prefill_tokens
        )
        if effective_limit is None:
            return max_prefill_tokens, False
        return effective_limit, True

    def _get_max_req_input_len(self: Scheduler, max_req_input_len: int) -> int:
        effective_limit = SchedulerMultiplexMixin._get_pdmux_prefill_token_limit(
            self, self.max_prefill_tokens
        )
        if effective_limit is None:
            return max_req_input_len
        # Request validation rejects lengths >= max_req_input_len.
        return min(max_req_input_len, effective_limit + 1)

    def _merge_finished_prefill_batch(
        self: Scheduler,
        prefill_result,
        prefill_stream,
        decode_stream,
        running_batch: ScheduleBatch,
    ) -> ScheduleBatch:
        batch = self.split_prefill_batch
        self.process_batch_result(batch, prefill_result)

        # Mirror get_next_batch_to_run's chunked bookkeeping: a request that
        # only finished a middle chunk must stay out of the decode batch, and
        # its chunk KV must be stashed so the next chunk extends the cached
        # prefix instead of recomputing it.
        chunked_req_to_exclude = set()
        if self.chunked_req is not None:
            chunked_req_to_exclude.add(self.chunked_req)
            # Stash only when this chunk produced new KV beyond what is
            # already cached. A parked chunk (add_chunked_req hybrid-SWA
            # early-return) has nothing new to cache.
            if self.chunked_req.extend_range.end > len(self.chunked_req.prefix_indices):
                # The stash rewrites the request's req_to_token row, which the
                # sparse-prefill scaffolding cache snapshots at segment 0, so
                # it is only legal once every split segment has run.
                assert batch.split_prefill_finished
                self.stash_chunked_request(self.chunked_req)
        if batch.chunked_req is not None:
            chunked_req_to_exclude.add(batch.chunked_req)

        # Mirror get_next_batch_to_run: filter unconditionally, not only when
        # a chunked request is excluded -- the filter also drops requests that
        # FINISHED during prefill, whose KV and req slots process_batch_result
        # already released. Merging them into the decode batch leaves a freed
        # req_pool_idx registered as a live owner until the next filter.
        last_bs = batch.batch_size()
        batch.filter_batch(chunked_req_to_exclude=list(chunked_req_to_exclude))
        if batch.batch_size() < last_bs:
            running_batch.batch_is_full = False

        if not batch.is_empty():
            if running_batch and not running_batch.is_empty():
                running_batch.merge_batch(batch)
            else:
                running_batch = batch

        self.running_batch = running_batch
        self.split_prefill_batch = None

        # merge_batch and the chunk stash enqueue tensor work (concatenations,
        # radix-cache inserts, page frees) on the prefill stream. The next loop
        # prepares decode before the stream-group synchronization, so publish
        # the dependency even when nothing was merged — decode may reallocate
        # pages the stash just freed.
        merge_done = prefill_stream.record_event()
        decode_stream.wait_event(merge_done)
        return running_batch

    # Pump the HiCache event drain every Nth in-flight iteration instead of
    # every iteration: each pump pays a TP-wide gloo all-reduce plus ack
    # bookkeeping (~10ms/iteration measured on a busy 8-rank host), while the
    # acks it retires are latency-insensitive background accounting -- the
    # transfers themselves are ordered by CUDA events, not by the pump. The
    # tick advances under a rank-consistent condition, so every rank pumps on
    # the same iterations and the pump's collectives stay aligned.
    HICACHE_PUMP_INTERVAL = 16

    @torch.inference_mode()
    def event_loop_pdmux(self: Scheduler):
        """A scheduler loop for pd multiplexing."""
        decode_done = False
        prefill_done = False
        wait_prefill_kernel_done = False
        adjust_stream_group = False
        self._hicache_pump_tick = 0
        stream_idx = get_current_stream_idx()
        stream_group = self.stream_groups[stream_idx]
        prefill_stream = stream_group[0]
        decode_stream = stream_group[1]
        torch.cuda.empty_cache()

        logger.debug("Starting event loop for pd multiplexing...")

        while True:
            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                recv_reqs = self.request_receiver.recv_requests()
                self.process_input_requests(recv_reqs)
                running_batch = self.running_batch

            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                sm_count = self.sm_counts[stream_idx][0]
                formation_done = None
                # Batch formation is the only other caller of the HiCache pump,
                # and it is skipped for every iteration a split prefill occupies
                # -- which is most of them under the long prefills PDMux exists
                # to overlap. Pump exactly on those iterations: skipping starves
                # HiCache of ack processing and host-lock release for the whole
                # prefill, while doubling up desyncs the pump's collective
                # all-reduces across TP ranks, which deadlocks rather than
                # degrades. The pump's cache actions can free device KV
                # segments and zero full-to-SWA mapping rows on this stream, so
                # it needs the same dependency publication as batch formation
                # -- decode allocates from that free list and the decode graph
                # re-reads the mapping on replay.
                had_inflight_split = self.split_prefill_batch is not None
                if wait_prefill_kernel_done or had_inflight_split:
                    self._hicache_pump_tick += 1
                    if self._hicache_pump_tick % self.HICACHE_PUMP_INTERVAL == 0:
                        # Publish a dependency only when the pump reports it
                        # may have enqueued device work (write_back frees,
                        # storage-queue actions): an event recorded here lands
                        # after the in-flight split segments and serializes
                        # decode behind the whole prefill's completion, so a
                        # host-only ack drain must not pay it.
                        if self.check_hicache_events_if_enabled():
                            formation_done = prefill_stream.record_event()
                if not wait_prefill_kernel_done:
                    created, running_batch = self.update_split_prefill_batch(
                        sm_count, running_batch=running_batch
                    )
                    self.running_batch = running_batch
                    adjust_stream_group = created or adjust_stream_group
                    if not had_inflight_split:
                        # Batch formation enqueued radix-cache and allocator
                        # work (prefix-match concatenations, evictions, KV
                        # allocation) on the prefill stream, rebinding the
                        # free-page list that decode-side allocation slices.
                        # Publish the dependency before decode prepares its
                        # next step. Record ONLY when formation actually ran
                        # (or the pump above enqueued device work): an event
                        # recorded on an idle iteration lands after the
                        # in-flight split segments and serializes every decode
                        # step behind the whole prefill's completion -- a
                        # ~50% TPOT regression under prefill-heavy load, for
                        # no ordering benefit.
                        formation_done = prefill_stream.record_event()

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                if formation_done is not None:
                    decode_stream.wait_event(formation_done)
                running_batch = self.update_running_batch(running_batch)
                self.running_batch = running_batch
                adjust_stream_group = adjust_stream_group or (
                    stream_idx > 0 and running_batch.is_empty()
                )
                if running_batch.is_empty() and self.split_prefill_batch is None:
                    self.on_idle()

            if adjust_stream_group:
                prefill_stream.synchronize()
                decode_stream.synchronize()
                stream_idx, stream_group = self.adjust_stream_groups(
                    running_batch=running_batch
                )
                prefill_stream = stream_group[0]
                decode_stream = stream_group[1]
                adjust_stream_group = False
                logger.debug(
                    f"Adjusting stream groups: {stream_idx}, prefill sm: {self.sm_counts[stream_idx][0]}, decode sm: {self.sm_counts[stream_idx][1]}"
                )

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                # process decode batch
                if running_batch and not running_batch.is_empty():
                    decode_result = self.run_batch(running_batch)
                    decode_done = True
                else:
                    decode_done = False
            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                if (
                    self.split_prefill_batch
                    and not self.split_prefill_batch.is_empty()
                    and not wait_prefill_kernel_done
                ):
                    prefill_done = True
                    forward_count = self._get_split_forward_count()
                    next_split_index = min(
                        self.split_prefill_batch.split_index + forward_count,
                        self.model_config.num_hidden_layers,
                    )
                    forward_count = (
                        next_split_index - self.split_prefill_batch.split_index
                    )

                    self.split_prefill_batch.split_forward_count = forward_count
                    prefill_result = self.run_batch(self.split_prefill_batch)
                    if next_split_index == self.model_config.num_hidden_layers:
                        self.split_prefill_batch.split_prefill_finished = True
                        prefill_exe_done = prefill_stream.record_event()
                    self.split_prefill_batch.split_index = next_split_index

                elif wait_prefill_kernel_done:
                    prefill_done = True
                else:
                    prefill_done = False

            with torch.cuda.stream(decode_stream):
                set_pdmux_status(False)
                decode_stream.synchronize()
                if decode_done:
                    self.process_batch_result(running_batch, decode_result)

            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                if prefill_done and self.split_prefill_batch.split_prefill_finished:
                    wait_prefill_kernel_done = True
                    prefill_exe_done_flag = prefill_exe_done.query()
                    flags = (
                        torch.ones(1, device="cpu", dtype=torch.int32)
                        if prefill_exe_done_flag
                        else torch.zeros(1, device="cpu", dtype=torch.int32)
                    )

                    self.tp_cpu_group.allreduce(flags, dist.ReduceOp.SUM).wait()
                    if flags.item() == self.ps.tp_size:
                        running_batch = self._merge_finished_prefill_batch(
                            prefill_result,
                            prefill_stream,
                            decode_stream,
                            running_batch,
                        )
                        wait_prefill_kernel_done = False
                        adjust_stream_group = True
