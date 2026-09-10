"""
Mixin class providing multiplexing scheduling logic
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.distributed as dist
from torch.cuda.streams import ExternalStream

from sglang.srt.distributed.parallel_state import set_pdmux_status
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
        # The current token-chunk prefill batch. PDMux overlaps this ordinary
        # EXTEND forward with decode; it does not split the model by layer.
        self.prefill_batch: Optional[ScheduleBatch] = None
        # Identity marker for the work item currently being submitted through
        # run_batch (see `_submit_pdmux_prefill`). Set only around that call so a
        # batch that becomes the decode batch is never mistaken for a prefill.
        self._pdmux_prefill_batch: Optional[ScheduleBatch] = None
        # Pipelined completion state for the in-flight prefill: the result and
        # its exe-done event while the work item is tracked, plus the one vote
        # outstanding (issued one iteration, consumed the next). See
        # `_advance_pdmux_prefill`.
        self._pdmux_prefill_result = None
        self._pdmux_prefill_exe_done = None
        self._pdmux_prefill_done_vote = None
        self._pdmux_prefill_done_flags = None

        # for pd_multiplexing, Init stream_groups, exclude normal stream for prefill only and decode only
        self.pdmux_config = load_pdmux_config(
            get_disagg().pdmux_config_path,
            default_sm_group_num=get_disagg().sm_group_num,
        )
        initialize_stream_groups(self.ps.gpu_id, self.pdmux_config)
        self.stream_groups = get_stream_groups()
        self.sm_counts = get_sm_counts()
        self.real_sm_group_num = len(self.stream_groups)
        logger.info(
            f"PD-Multiplexing enabled with {self.real_sm_group_num} stream groups, sm_counts (prefill_sm, decode_sm): {self.sm_counts}"
        )

    def pdmux_inflight_prefill_batches(self: Scheduler) -> List[ScheduleBatch]:
        """The prefill batch that lives outside running_batch / last_batch.

        Between formation and merge the loop holds the prefill batch in its own
        field, so callers enumerating in-flight work would otherwise miss it:
        `abort_request` would not see those requests until they reach the decode
        batch, and `is_fully_idle` would call the scheduler idle while a prefill
        forward is still running -- which lets flush_cache reset the tree cache
        and the request / KV pools underneath it. Empty unless PDMux is on, since
        `prefill_batch` is only created by init_pdmux.
        """
        if not self.enable_pdmux or self.prefill_batch is None:
            return []
        return [self.prefill_batch]

    def _submit_pdmux_prefill(self: Scheduler, batch: ScheduleBatch):
        """Submit the PDMux prefill forward.

        `self._pdmux_prefill_batch` marks the single run_batch call that submits
        the prefill, so run_batch can pin the forward's tensor lifetime onto the
        result: the prefill stream runs on for many decode iterations before the
        merge, and run_batch rebinds the batch's input_ids / seq_lens / spec_info
        as it returns. A snapshot taken inside run_batch after resolve_forward_inputs
        is what keeps the tensors the forward read alive across that rebind; it
        rides on the result's extra_keep_alive_refs, so the loop holding the
        result until the merge vote pins it for exactly that span.
        """
        assert self._pdmux_prefill_batch is None, "a PDMux prefill is already tracked"
        self._pdmux_prefill_batch = batch
        try:
            return self.run_batch(batch)
        finally:
            self._pdmux_prefill_batch = None

    def _issue_pdmux_done_vote(self: Scheduler) -> None:
        """Sample this iteration's completion flag and start the allreduce."""
        flags = torch.zeros(1, device="cpu", dtype=torch.int32)
        if self._pdmux_prefill_exe_done.query():
            flags[0] = 1
        self._pdmux_prefill_done_flags = flags
        self._pdmux_prefill_done_vote = self.tp_cpu_group.allreduce(
            flags, dist.ReduceOp.SUM
        )

    def _advance_pdmux_prefill(
        self: Scheduler,
        *,
        running_batch: ScheduleBatch,
        prefill_stream,
        decode_stream,
    ) -> tuple[ScheduleBatch, bool]:
        """One completion-pipeline step for the in-flight prefill.

        Consumes the vote issued last iteration, then issues this iteration's
        fresh vote, so the collective started here is waited on at the top of the
        next step -- it gets a full iteration of host work (decode submit and its
        result processing) to land in the background. HEAD issued and waited the
        allreduce back to back, paying the rendezvous latency inline on every
        iteration a long prefill spans.

        Every rank consumes the same collectives on the same iterations and votes
        under the same `wait_prefill_kernel_done` condition, so a rank whose event
        is ready an iteration early just votes 1 until stragglers catch up -- the
        reduced sum is the only decision input and no rank can finalize on a
        different iteration. A new flag tensor each vote: the previous one may
        still be owned by the Work waited on above. Returns (running_batch, merged).
        """
        if self._pdmux_prefill_done_vote is not None:
            # Consume last iteration's vote.
            self._pdmux_prefill_done_vote.wait()
            self._pdmux_prefill_done_vote = None
            if self._pdmux_prefill_done_flags.item() == self.ps.tp_size:
                running_batch = self._merge_finished_prefill_batch(
                    self._pdmux_prefill_result,
                    prefill_stream,
                    decode_stream,
                    running_batch,
                )
                self._pdmux_prefill_result = None
                self._pdmux_prefill_exe_done = None
                return running_batch, True

        self._issue_pdmux_done_vote()
        return running_batch, False

    # TODO(jason-fxz): This is a temporary demo
    def _select_stream_idx(self: Scheduler, running_batch: ScheduleBatch) -> int:
        """Select the SM layout for the current decode/prefill workload.

        This helper is intentionally side-effect free.  The scheduler uses it
        both when installing a layout and when deciding whether a completed
        decode request moved the workload across a layout boundary.

        ``decode_bs_divisor`` is the batch size for the middle (roughly
        balanced) partition.  The range is intentionally allowed to continue
        into decode-majority partitions for larger batches.
        """
        if not running_batch.is_empty() and self.prefill_batch:
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
                return stream_idx

            return max(
                1,
                min(
                    self.real_sm_group_num - 2,
                    decode_bs
                    * (self.real_sm_group_num - 2)
                    // (2 * self.pdmux_config.decode_bs_divisor),
                ),
            )
        if not running_batch.is_empty():
            return self.real_sm_group_num - 1
        return 0

    def adjust_stream_groups(
        self: Scheduler, running_batch: ScheduleBatch
    ) -> tuple[int, tuple[ExternalStream, ExternalStream]]:
        selector = getattr(self, "_select_stream_idx", None)
        stream_idx = (
            selector(running_batch)
            if selector is not None
            else SchedulerMultiplexMixin._select_stream_idx(self, running_batch)
        )
        set_current_stream_idx(stream_idx)

        stream_idx = get_current_stream_idx()

        # Speculative decoding has target and draft runners. Let the active
        # worker switch all of them so eager draft attention does not keep
        # using the backend bound to the previous Green Context stream.
        model_worker = getattr(self, "model_worker", None)
        if model_worker is not None and hasattr(
            model_worker, "update_pdmux_decode_attn_backend"
        ):
            model_worker.update_pdmux_decode_attn_backend(stream_idx)
        else:
            self.tp_worker.model_runner.update_decode_attn_backend(stream_idx)
        return stream_idx, self.stream_groups[stream_idx]

    def update_prefill_batch(
        self: Scheduler, sm_count: int, running_batch: ScheduleBatch
    ) -> tuple[bool, ScheduleBatch]:
        if self.prefill_batch is not None:
            return False, running_batch

        # No token chunk forward is in flight here, which matches the normal
        # loop's safe point for tearing down an aborted request before its next
        # chunk is formed.
        self.process_pending_chunked_abort()

        # add new request
        prefill_plan = self.get_new_batch_prefill(running_batch)
        batch = prefill_plan.batch_to_run
        running_batch = prefill_plan.running_batch
        # PDMux forms batches outside Scheduler.get_next_batch_to_run(), so it
        # must perform the same DP/MLP synchronization here. Call it even when
        # there is no local prefill: peer DP ranks may have work, in which case
        # the adapter produces an idle batch and keeps the lane's collectives
        # aligned. Without this, DeepSeek V4 reaches its first DP gather with no
        # global token-count metadata.
        batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(batch)
        # An idle batch has no requests by construction, but still must run so
        # this DP rank participates in a peer rank's prefill collectives.
        if batch is not None:
            self.prefill_batch = batch
            return True, running_batch
        return False, running_batch

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
        batch = self.prefill_batch
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
        self.prefill_batch = None

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
                # and it is skipped for every iteration a token-chunk prefill occupies
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
                had_inflight_prefill = self.prefill_batch is not None
                if wait_prefill_kernel_done or had_inflight_prefill:
                    self._hicache_pump_tick += 1
                    if self._hicache_pump_tick % self.HICACHE_PUMP_INTERVAL == 0:
                        # Publish a dependency only when the pump reports it
                        # may have enqueued device work (write_back frees,
                        # storage-queue actions): an event recorded here lands
                        # after the in-flight prefill and serializes
                        # decode behind the whole prefill's completion, so a
                        # host-only ack drain must not pay it.
                        if self.check_hicache_events_if_enabled():
                            formation_done = prefill_stream.record_event()
                if not wait_prefill_kernel_done:
                    created, running_batch = self.update_prefill_batch(
                        sm_count, running_batch=running_batch
                    )
                    self.running_batch = running_batch
                    adjust_stream_group = created or adjust_stream_group
                    if not had_inflight_prefill:
                        # Batch formation enqueued radix-cache and allocator
                        # work (prefix-match concatenations, evictions, KV
                        # allocation) on the prefill stream, rebinding the
                        # free-page list that decode-side allocation slices.
                        # Publish the dependency before decode prepares its
                        # next step. Record ONLY when formation actually ran
                        # (or the pump above enqueued device work): an event
                        # recorded on an idle iteration lands after the
                        # in-flight prefill and serializes every decode
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
                if running_batch.is_empty() and self.prefill_batch is None:
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
                # Mirror the regular scheduler's per-step DP/MLP sync. Passing
                # None on a locally idle rank is intentional: the adapter may
                # synthesize an idle batch so it can join peer decode
                # collectives without replacing this rank's running batch.
                decode_batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(
                    running_batch if not running_batch.is_empty() else None
                )
                if decode_batch is not None:
                    decode_result = self.run_batch(decode_batch)
                    decode_done = True
                else:
                    decode_done = False
            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                if self.prefill_batch is not None and not wait_prefill_kernel_done:
                    # ``get_new_batch_prefill`` already applies the configured
                    # token chunk limit. Run that ordinary EXTEND batch through
                    # all model layers in one forward; PDMux only overlaps the
                    # resulting prefill work with decode on separate streams.
                    wait_prefill_kernel_done = True
                    self._pdmux_prefill_result = self._submit_pdmux_prefill(
                        self.prefill_batch
                    )
                    self._pdmux_prefill_exe_done = prefill_stream.record_event()

            if decode_done:
                decode_stream.synchronize()
                with torch.cuda.stream(decode_stream):
                    set_pdmux_status(False)
                    self.process_batch_result(decode_batch, decode_result)

            with torch.cuda.stream(prefill_stream):
                set_pdmux_status(True)
                if wait_prefill_kernel_done:
                    # Runs every iteration a prefill is tracked, including the
                    # submission iteration (issues the first vote). One vote per
                    # iteration, consumed at the top of the next one.
                    running_batch, merged = self._advance_pdmux_prefill(
                        running_batch=running_batch,
                        prefill_stream=prefill_stream,
                        decode_stream=decode_stream,
                    )
                    if merged:
                        wait_prefill_kernel_done = False
                        adjust_stream_group = True
