import ast
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import sglang.srt.distributed.parallel_state as parallel_state
import sglang.srt.multiplex.pdmux_context as pdmux_context
from sglang.srt.distributed.parallel_state import (
    is_pdmux_enabled,
    is_pdmux_prefill_enabled,
    set_pdmux_status,
)
from sglang.srt.multiplex.multiplexing_mixin import SchedulerMultiplexMixin
from sglang.srt.multiplex.pdmux_context import load_pdmux_config
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SCHEDULER_PATH = (
    Path(__file__).resolve().parents[4] / "python/sglang/srt/managers/scheduler.py"
)
TP_WORKER_PATH = SCHEDULER_PATH.parent / "tp_worker.py"
DP_ATTN_PATH = SCHEDULER_PATH.parent / "scheduler_components" / "dp_attn.py"


def _init_call_order(class_name, targets):
    """Where each target call lands in `__init__`'s execution.

    Resolves one level of `self.init_*()` indirection, so a target that moves
    into (or out of) a helper is still placed at the point `__init__` runs it.
    `__init__` is straight-line code, so source order is execution order.
    """
    tree = ast.parse(SCHEDULER_PATH.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    methods = {
        node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)
    }

    def self_calls(node):
        return sorted(
            (call.lineno, call.func.attr)
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "self"
        )

    positions = {}
    for step, (_lineno, name) in enumerate(self_calls(methods["__init__"])):
        if name in targets:
            positions.setdefault(name, step)
        elif name in methods:
            for _, nested in self_calls(methods[name]):
                if nested in targets:
                    positions.setdefault(nested, step)
    return positions


class _Batch:
    def __init__(self, empty):
        self._empty = empty

    def is_empty(self):
        return self._empty


class _ChunkedReq:
    """Hashable stand-in for Req (the merge path stores it in a set)."""

    def __init__(self, *, extend_end, prefix_len):
        self.extend_range = SimpleNamespace(end=extend_end)
        self.prefix_indices = [0] * prefix_len


def _make_chunked_req(*, extend_end, prefix_len):
    return _ChunkedReq(extend_end=extend_end, prefix_len=prefix_len)


class TestPDMuxScheduler(unittest.TestCase):
    def tearDown(self):
        set_pdmux_status(False)

    def _make_scheduler(
        self,
        *,
        decode_empty,
        split_index=0,
        extend_num_tokens=128000,
        token_budget=65536,
        overlap_decode_full_sm=False,
    ):
        return SimpleNamespace(
            model_config=SimpleNamespace(num_hidden_layers=61),
            running_batch=_Batch(decode_empty),
            pdmux_config=SimpleNamespace(
                split_forward_token_budget=token_budget,
                overlap_decode_full_sm=overlap_decode_full_sm,
            ),
            prefill_batch=SimpleNamespace(
                split_index=split_index,
                extend_num_tokens=extend_num_tokens,
            ),
        )

    def test_layerwise_prefill_runs_remaining_layers_without_decode_work(self):
        scheduler = self._make_scheduler(decode_empty=True, split_index=7)

        count = SchedulerMultiplexMixin._get_split_forward_count(scheduler)

        self.assertEqual(count, 54)

    def test_layerwise_prefill_uses_token_layer_budget_with_decode_work(self):
        scheduler = self._make_scheduler(decode_empty=False)

        count = SchedulerMultiplexMixin._get_split_forward_count(scheduler)

        self.assertEqual(count, 1)

    def test_layerwise_prefill_count_is_clamped_to_remaining_layers(self):
        scheduler = self._make_scheduler(
            decode_empty=False,
            split_index=59,
            extend_num_tokens=8192,
        )

        count = SchedulerMultiplexMixin._get_split_forward_count(scheduler)

        self.assertEqual(count, 2)

    def test_layerwise_prefill_expands_short_exclusive_segment(self):
        scheduler = self._make_scheduler(
            decode_empty=False,
            extend_num_tokens=1024,
            token_budget=1024,
        )
        scheduler.running_batch.batch_size = lambda: 32

        count = SchedulerMultiplexMixin._get_split_forward_count(
            scheduler,
            prefill_sm_count=96,
            decode_sm_count=32,
        )

        self.assertEqual(count, 6)

    def test_layerwise_prefill_keeps_budget_with_full_sm_decode(self):
        scheduler = self._make_scheduler(
            decode_empty=False,
            extend_num_tokens=1024,
            token_budget=1024,
            overlap_decode_full_sm=True,
        )
        scheduler.running_batch.batch_size = lambda: 32

        count = SchedulerMultiplexMixin._get_split_forward_count(
            scheduler,
            prefill_sm_count=32,
            decode_sm_count=128,
        )

        self.assertEqual(count, 1)

    def test_layerwise_prefill_uses_rank_stable_dp_token_counts(self):
        scheduler = self._make_scheduler(
            decode_empty=True,
            extend_num_tokens=1,
            token_budget=65536,
        )
        scheduler.prefill_batch.global_num_tokens = [8192] * 8
        decode_batch = SimpleNamespace(global_num_tokens=[1] * 8)

        count = SchedulerMultiplexMixin._get_split_forward_count(
            scheduler,
            decode_batch=decode_batch,
        )

        self.assertEqual(count, 1)

    def test_intermediate_layer_segment_does_not_merge_token_chunk(self):
        running_batch = object()
        merge_finished = Mock()
        scheduler = SimpleNamespace(
            enable_pdmux_layerwise_prefill=True,
            prefill_batch=SimpleNamespace(split_prefill_finished=False),
            _pdmux_prefill_done_vote=SimpleNamespace(wait=Mock()),
            _pdmux_prefill_done_flags=SimpleNamespace(item=lambda: 1),
            _pdmux_prefill_result=object(),
            _pdmux_prefill_exe_done=object(),
            ps=SimpleNamespace(tp_size=1),
            _merge_finished_prefill_batch=merge_finished,
        )

        returned, segment_done = SchedulerMultiplexMixin._advance_pdmux_prefill(
            scheduler,
            running_batch=running_batch,
            prefill_stream=object(),
            decode_stream=object(),
        )

        self.assertIs(returned, running_batch)
        self.assertTrue(segment_done)
        self.assertIsNone(scheduler._pdmux_prefill_result)
        self.assertIsNone(scheduler._pdmux_prefill_exe_done)
        merge_finished.assert_not_called()

    def test_finished_layerwise_prefill_releases_persistent_forward_batch(self):
        split_forward_batch = object()
        batch = SimpleNamespace(
            split_prefill_finished=True,
            split_forward_batch=split_forward_batch,
            split_index=61,
            split_forward_count=3,
            chunked_req=None,
            batch_size=Mock(side_effect=[1, 1]),
            filter_batch=Mock(),
            is_empty=Mock(return_value=True),
        )
        merge_event = object()
        prefill_stream = SimpleNamespace(record_event=Mock(return_value=merge_event))
        decode_stream = SimpleNamespace(wait_event=Mock())
        running_batch = SimpleNamespace(batch_is_full=True)
        scheduler = SimpleNamespace(
            enable_pdmux_layerwise_prefill=True,
            prefill_batch=batch,
            process_batch_result=Mock(),
            chunked_req=None,
            running_batch=running_batch,
        )

        returned = SchedulerMultiplexMixin._merge_finished_prefill_batch(
            scheduler,
            prefill_result=object(),
            prefill_stream=prefill_stream,
            decode_stream=decode_stream,
            running_batch=running_batch,
        )

        self.assertIs(returned, running_batch)
        self.assertIsNone(batch.split_forward_batch)
        self.assertEqual(batch.split_index, 0)
        self.assertEqual(batch.split_forward_count, 1)
        self.assertFalse(batch.split_prefill_finished)
        self.assertIsNone(scheduler.prefill_batch)
        decode_stream.wait_event.assert_called_once_with(merge_event)

    def test_dsv4_prefill_admission_uses_planner_hard_limit(self):
        scheduler = SimpleNamespace(
            enable_pdmux=True,
            pdmux_max_prefill_plan_tokens=(1 << 16) - 1,
            page_size=16,
            chunked_prefill_size=None,
        )

        budget, enforce = SchedulerMultiplexMixin._get_prefill_admission_config(
            scheduler, 131072
        )

        self.assertEqual(budget, 65520)
        self.assertTrue(enforce)

    def test_non_dsv4_prefill_admission_preserves_soft_budget(self):
        scheduler = SimpleNamespace(
            enable_pdmux=True,
            pdmux_max_prefill_plan_tokens=None,
            page_size=16,
            chunked_prefill_size=None,
        )

        budget, enforce = SchedulerMultiplexMixin._get_prefill_admission_config(
            scheduler, 131072
        )

        self.assertEqual(budget, 131072)
        self.assertFalse(enforce)

    def test_chunked_prefill_admission_preserves_soft_budget(self):
        """Chunked prefill enforces the planner limit per chunk, so the hard
        admission clamp must deactivate — keeping it would re-reject the long
        requests chunking exists to serve."""
        scheduler = SimpleNamespace(
            enable_pdmux=True,
            pdmux_max_prefill_plan_tokens=(1 << 16) - 1,
            page_size=16,
            chunked_prefill_size=16384,
        )

        budget, enforce = SchedulerMultiplexMixin._get_prefill_admission_config(
            scheduler, 131072
        )

        self.assertEqual(budget, 131072)
        self.assertFalse(enforce)

    def test_dsv4_request_length_stays_within_planner_limit(self):
        scheduler = SimpleNamespace(
            enable_pdmux=True,
            pdmux_max_prefill_plan_tokens=(1 << 16) - 1,
            max_prefill_tokens=131072,
            page_size=16,
            chunked_prefill_size=None,
        )

        max_input_len = SchedulerMultiplexMixin._get_max_req_input_len(
            scheduler, 1048576
        )

        self.assertEqual(max_input_len, 65521)

    def test_dsv4_request_limit_matches_smaller_prefill_budget(self):
        scheduler = SimpleNamespace(
            enable_pdmux=True,
            pdmux_max_prefill_plan_tokens=(1 << 16) - 1,
            max_prefill_tokens=32767,
            page_size=16,
            chunked_prefill_size=None,
        )

        budget, enforce = SchedulerMultiplexMixin._get_prefill_admission_config(
            scheduler, scheduler.max_prefill_tokens
        )
        max_input_len = SchedulerMultiplexMixin._get_max_req_input_len(
            scheduler, 1048576
        )

        self.assertEqual(budget, 32752)
        self.assertTrue(enforce)
        self.assertEqual(max_input_len, budget + 1)

    def test_init_rejects_chunked_prefill_size_over_plan_limit(self):
        """65520 is the page-aligned uint16 compressor-plan cap; a larger
        chunk budget would overflow a single prefill plan at runtime."""
        scheduler = SimpleNamespace(
            enable_pdmux=True,
            page_size=16,
            chunked_prefill_size=65536,
            max_prefill_tokens=131072,
            max_req_input_len=1048576,
        )
        attn_backend = SimpleNamespace(max_prefill_plan_tokens=(1 << 16) - 1)

        with self.assertRaisesRegex(ValueError, "65520"):
            SchedulerMultiplexMixin.init_pdmux_prefill_plan_limit(
                scheduler, attn_backend=attn_backend
            )

    def test_init_accepts_chunked_prefill_size_at_plan_limit(self):
        """With a valid chunk budget the per-request length clamp must stay
        off: chunking is what serves requests beyond the planner limit."""
        scheduler = SimpleNamespace(
            enable_pdmux=True,
            page_size=16,
            chunked_prefill_size=65520,
            max_prefill_tokens=131072,
            max_req_input_len=1048576,
        )
        attn_backend = SimpleNamespace(max_prefill_plan_tokens=(1 << 16) - 1)

        SchedulerMultiplexMixin.init_pdmux_prefill_plan_limit(
            scheduler, attn_backend=attn_backend
        )

        self.assertEqual(scheduler.pdmux_max_prefill_plan_tokens, (1 << 16) - 1)
        self.assertEqual(scheduler.max_req_input_len, 1048576)

    def test_init_tightens_request_length_without_chunked_prefill(self):
        """Without chunking PDMux cannot admit an oversized request, so init
        must clamp request validation to the planner limit."""
        scheduler = SimpleNamespace(
            enable_pdmux=True,
            page_size=16,
            chunked_prefill_size=None,
            max_prefill_tokens=131072,
            max_req_input_len=1048576,
        )
        attn_backend = SimpleNamespace(max_prefill_plan_tokens=(1 << 16) - 1)

        SchedulerMultiplexMixin.init_pdmux_prefill_plan_limit(
            scheduler, attn_backend=attn_backend
        )

        self.assertEqual(scheduler.max_req_input_len, 65521)

    @staticmethod
    @contextmanager
    def _stubbed_stream_idx():
        """Stand in for the module-level stream-index state.

        The real setter validates against `STREAM_GROUPS`, which only
        `initialize_stream_groups` fills and which needs a GPU.
        """
        state = {"idx": 0}
        with (
            patch(
                "sglang.srt.multiplex.multiplexing_mixin.set_current_stream_idx",
                lambda idx: state.update(idx=idx),
            ),
            patch(
                "sglang.srt.multiplex.multiplexing_mixin.get_current_stream_idx",
                lambda: state["idx"],
            ),
        ):
            yield

    def _make_stream_group_scheduler(self, *, manual_divisions, group_num):
        model_runner = SimpleNamespace(update_decode_attn_backend=Mock())
        return SimpleNamespace(
            prefill_batch=object(),
            pdmux_config=SimpleNamespace(
                manual_divisions=manual_divisions, decode_bs_divisor=36
            ),
            real_sm_group_num=group_num,
            tp_worker=SimpleNamespace(model_runner=model_runner),
            stream_groups=[(f"p{i}", f"d{i}") for i in range(group_num)],
        )

    def test_manual_division_below_every_threshold_uses_first_shared_group(self):
        """A decode batch under every configured threshold still needs a group.

        The selection loop only assigns a stream index on a threshold it meets,
        so a batch below all of them left the index unbound -- an
        UnboundLocalError raised from the scheduler loop. A single-division
        config (`--sm-group-num 3`) whose threshold is above 1 hits this for
        every small decode batch that overlaps a token chunk.
        """
        scheduler = self._make_stream_group_scheduler(
            manual_divisions=[[128, 0, 8]], group_num=3
        )
        running_batch = SimpleNamespace(is_empty=lambda: False, batch_size=lambda: 1)

        with self._stubbed_stream_idx():
            stream_idx, stream_group = SchedulerMultiplexMixin.adjust_stream_groups(
                scheduler, running_batch
            )

        self.assertEqual(stream_idx, 1)
        self.assertEqual(stream_group, ("p1", "d1"))

    def test_manual_division_picks_the_highest_met_threshold(self):
        scheduler = self._make_stream_group_scheduler(
            manual_divisions=[[32, 0, 1], [64, 0, 8]], group_num=4
        )
        running_batch = SimpleNamespace(is_empty=lambda: False, batch_size=lambda: 12)

        with self._stubbed_stream_idx():
            stream_idx, _ = SchedulerMultiplexMixin.adjust_stream_groups(
                scheduler, running_batch
            )

        self.assertEqual(stream_idx, 2)

    def test_stream_switch_updates_speculative_target_and_draft_backends(self):
        scheduler = self._make_stream_group_scheduler(
            manual_divisions=[[32, 0, 1]], group_num=3
        )
        scheduler.model_worker = SimpleNamespace(
            update_pdmux_decode_attn_backend=Mock()
        )
        target_update = scheduler.tp_worker.model_runner.update_decode_attn_backend
        running_batch = SimpleNamespace(is_empty=lambda: False, batch_size=lambda: 1)

        with self._stubbed_stream_idx():
            SchedulerMultiplexMixin.adjust_stream_groups(scheduler, running_batch)

        scheduler.model_worker.update_pdmux_decode_attn_backend.assert_called_once_with(
            1
        )
        target_update.assert_not_called()

    def test_stream_selection_shrinks_with_decode_batch_during_token_prefill(self):
        """A non-empty decode batch must move to the smaller layout as it drains.

        With six shared groups and the default divisor, a 132-SM device maps
        batch sizes 36 and 1 to groups 3 and 1 respectively.  Group 3 is the
        roughly balanced 72/60 layout; larger batches continue into the
        decode-majority groups. The event loop uses this side-effect-free
        selector after filtering completed requests so it can trigger an event
        handoff before the next forward.
        """
        scheduler = self._make_stream_group_scheduler(manual_divisions=[], group_num=8)
        scheduler.pdmux_config.decode_bs_divisor = 36

        large_batch = SimpleNamespace(is_empty=lambda: False, batch_size=lambda: 36)
        small_batch = SimpleNamespace(is_empty=lambda: False, batch_size=lambda: 1)

        self.assertEqual(
            SchedulerMultiplexMixin._select_stream_idx(scheduler, large_batch), 3
        )
        self.assertEqual(
            SchedulerMultiplexMixin._select_stream_idx(scheduler, small_batch), 1
        )

    def test_stream_selection_does_not_resize_within_same_decode_partition(self):
        """Batch churn below a layout boundary should not pay another handoff."""
        scheduler = self._make_stream_group_scheduler(manual_divisions=[], group_num=8)
        scheduler.pdmux_config.decode_bs_divisor = 36

        batch_12 = SimpleNamespace(is_empty=lambda: False, batch_size=lambda: 12)
        batch_13 = SimpleNamespace(is_empty=lambda: False, batch_size=lambda: 13)

        self.assertEqual(
            SchedulerMultiplexMixin._select_stream_idx(scheduler, batch_12), 1
        )
        self.assertEqual(
            SchedulerMultiplexMixin._select_stream_idx(scheduler, batch_13), 1
        )

    def test_stream_selection_shrinks_with_manual_decode_thresholds(self):
        scheduler = self._make_stream_group_scheduler(
            manual_divisions=[[32, 0, 8], [64, 0, 36]], group_num=4
        )

        batch_36 = SimpleNamespace(is_empty=lambda: False, batch_size=lambda: 36)
        batch_1 = SimpleNamespace(is_empty=lambda: False, batch_size=lambda: 1)

        self.assertEqual(
            SchedulerMultiplexMixin._select_stream_idx(scheduler, batch_36), 2
        )
        self.assertEqual(
            SchedulerMultiplexMixin._select_stream_idx(scheduler, batch_1), 1
        )

    def test_auto_divisions_include_decode_majority_partitions(self):
        """Automatic layouts must expose a partition with decode > half SMs."""
        divisions = pdmux_context.divide_sm(132, (9, 0), groups=6)

        self.assertEqual(len(divisions), 6)
        self.assertTrue(
            any(prefill_sm < decode_sm for prefill_sm, decode_sm in divisions)
        )
        self.assertTrue(
            all(prefill_sm + decode_sm == 132 for prefill_sm, decode_sm in divisions)
        )

    def test_auto_divisions_keep_large_prefill_first_and_decode_large_last(self):
        divisions = pdmux_context.divide_sm(132, (9, 0), groups=6)

        self.assertGreater(divisions[0][0], divisions[-1][0])
        self.assertGreater(divisions[-1][1], divisions[-1][0])

    def test_pdmux_prefill_forward_installs_hicache_consumer_first(self):
        """Every PDMux token chunk installs the HiCache consumer index first.

        `set_hicache_consumer` selects the transfer event set the KV pool waits
        on before reading loaded-back pages. The ordinary generation forward
        is the only PDMux prefill entry point and must install it before the
        model reads the cache.
        """
        tree = ast.parse(TP_WORKER_PATH.read_text(encoding="utf-8"))
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "forward_batch_generation"
        )
        calls = [
            call.func.attr
            for call in ast.walk(method)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
        ]

        self.assertIn("set_hicache_consumer", calls)
        self.assertLess(calls.index("set_hicache_consumer"), calls.index("forward"))

    def test_plan_limit_resolves_after_chunked_prefill_size(self):
        """Scheduler init must resolve the chunk size before the plan limit.

        `init_pdmux_prefill_plan_limit` branches on `self.chunked_prefill_size`,
        which only `init_chunked_prefill` sets. Resolving the limit first raises
        AttributeError during startup for every attention backend that declares
        a plan limit -- which is exactly the PDMux configurations that need the
        limit, so the crash is not hypothetical.
        """
        targets = ("init_chunked_prefill", "init_pdmux_prefill_plan_limit")
        positions = _init_call_order("Scheduler", targets)

        for target in targets:
            self.assertIn(target, positions, f"{target} is not reached from __init__")
        self.assertLess(
            positions["init_chunked_prefill"],
            positions["init_pdmux_prefill_plan_limit"],
        )

    def test_pdmux_initialization_uses_parallel_state_gpu_id(self):
        config = object()
        scheduler = SimpleNamespace(
            ps=SimpleNamespace(gpu_id=3),
        )

        with (
            patch(
                "sglang.srt.multiplex.multiplexing_mixin.load_pdmux_config",
                return_value=config,
            ) as load_pdmux_config,
            patch(
                "sglang.srt.multiplex.multiplexing_mixin.get_disagg",
                return_value=SimpleNamespace(
                    pdmux_config_path="pdmux.yaml", sm_group_num=8
                ),
            ),
            patch(
                "sglang.srt.multiplex.multiplexing_mixin.initialize_stream_groups"
            ) as initialize_stream_groups,
            patch(
                "sglang.srt.multiplex.multiplexing_mixin.get_stream_groups",
                return_value=[object(), object(), object()],
            ),
            patch(
                "sglang.srt.multiplex.multiplexing_mixin.get_sm_counts",
                return_value=[(1, 0), (1, 1), (0, 1)],
            ),
        ):
            SchedulerMultiplexMixin.init_pdmux(scheduler)

        load_pdmux_config.assert_called_once_with("pdmux.yaml", default_sm_group_num=8)
        initialize_stream_groups.assert_called_once_with(3, config)
        self.assertEqual(scheduler.real_sm_group_num, 3)

    def test_pdmux_prefill_status_is_observable(self):
        self.assertFalse(is_pdmux_prefill_enabled())

        set_pdmux_status(True)
        self.assertTrue(is_pdmux_prefill_enabled())

        set_pdmux_status(False)
        self.assertFalse(is_pdmux_prefill_enabled())

    def test_dp_attn_adapter_uses_active_pdmux_tp_group(self):
        tree = ast.parse(DP_ATTN_PATH.read_text(encoding="utf-8"))
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "SchedulerDPAttnAdapter"
        )
        method = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "prepare_mlp_sync_batch"
        )
        prepare_call = next(
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "prepare_mlp_sync_batch_raw"
        )
        tp_group = next(
            keyword.value
            for keyword in prepare_call.keywords
            if keyword.arg == "tp_group"
        )

        self.assertIsInstance(tp_group, ast.Call)
        self.assertIsInstance(tp_group.func, ast.Name)
        self.assertEqual(tp_group.func.id, "get_tp_group")

    def test_pdmux_cli_sm_group_num_is_used_without_yaml(self):
        config = load_pdmux_config(None, default_sm_group_num=3)

        self.assertEqual(config.sm_group_num, 3)
        self.assertEqual(config.split_forward_token_budget, 65536)

    def test_pdmux_rejects_nonpositive_split_forward_token_budget(self):
        config_text = """\
sm_group_num: 3
split_forward_token_budget: 0
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as config_file:
            config_file.write(config_text)
            config_file.flush()
            with self.assertRaisesRegex(ValueError, "split_forward_token_budget"):
                load_pdmux_config(config_file.name)

    def test_pdmux_overlap_config_uses_full_device_decode_stream(self):
        config_text = """\
sm_group_num: 3
overlap_decode_full_sm: true
manual_divisions:
  - [32, 128, 0]
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as config_file:
            config_file.write(config_text)
            config_file.flush()
            config = load_pdmux_config(config_file.name)

        prefill_green = SimpleNamespace(cuda_stream=101)
        reserved_green = SimpleNamespace(cuda_stream=102)
        spatial = SimpleNamespace(
            get_sm_available=Mock(return_value=128),
            create_greenctx_stream_by_value=Mock(
                return_value=(prefill_green, reserved_green)
            ),
        )
        full_streams = iter(
            [
                "prefill-only",
                "unused-0",
                "decode-full",
                "unused-1",
                "decode-only",
            ]
        )
        with (
            patch.dict(sys.modules, {"sgl_kernel": SimpleNamespace(spatial=spatial)}),
            patch.object(pdmux_context.torch.cuda, "current_device", return_value=0),
            patch.object(
                pdmux_context.torch.cuda,
                "Stream",
                side_effect=lambda _gpu_id, priority=0: next(full_streams),
            ),
            patch.object(pdmux_context, "STREAM_GROUPS", []),
            patch.object(pdmux_context, "SM_COUNTS", []),
            patch.object(pdmux_context, "GREEN_CONTEXT_STREAMS", []),
            patch.object(pdmux_context, "_RESERVED_GREEN_STREAMS", []),
        ):
            pdmux_context.initialize_stream_groups(0, config)

            spatial.create_greenctx_stream_by_value.assert_called_once_with(32, 96, 0)
            self.assertEqual(
                pdmux_context.STREAM_GROUPS[1], (prefill_green, "decode-full")
            )
            self.assertEqual(pdmux_context.SM_COUNTS[1], (32, 128))
            self.assertEqual(pdmux_context._RESERVED_GREEN_STREAMS, [reserved_green])
            self.assertTrue(pdmux_context.is_green_context_stream(101))
            self.assertFalse(pdmux_context.is_green_context_stream(102))
            self.assertEqual(
                pdmux_context.torch.cuda.Stream.call_args_list,
                [
                    unittest.mock.call(0),
                    unittest.mock.call(0),
                    unittest.mock.call(0, priority=-1),
                    unittest.mock.call(0),
                    unittest.mock.call(0),
                ],
            )

    def test_overlap_config_rejects_negative_decode_placeholder(self):
        config_text = """\
sm_group_num: 3
overlap_decode_full_sm: true
manual_divisions:
  - [32, -1, 1]
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as config_file:
            config_file.write(config_text)
            config_file.flush()
            with self.assertRaisesRegex(ValueError, "non-negative"):
                load_pdmux_config(config_file.name)

    def test_exclusive_manual_division_cannot_leave_sms_unassigned(self):
        config = pdmux_context.PDMuxConfig(
            sm_group_num=3,
            manual_divisions=[[32, 32, 1]],
        )
        spatial = SimpleNamespace(get_sm_available=Mock(return_value=128))
        with (
            patch.dict(sys.modules, {"sgl_kernel": SimpleNamespace(spatial=spatial)}),
            patch.object(pdmux_context.torch.cuda, "current_device", return_value=0),
        ):
            with self.assertRaisesRegex(ValueError, "must assign every SM"):
                pdmux_context.initialize_stream_groups(0, config)

    def test_blackwell_uses_eight_sm_green_context_granularity(self):
        self.assertEqual(pdmux_context.get_arch_constraints((10, 0)), (8, 8))

    def test_pdmux_process_status_does_not_follow_prefill_phase(self):
        with patch.object(parallel_state, "_PDMUX_PREFILL_TP_GROUP", object()):
            set_pdmux_status(False)

            self.assertTrue(is_pdmux_enabled())
            self.assertFalse(is_pdmux_prefill_enabled())

    def _make_merge_streams(self, operations):
        prefill_stream = Mock()
        merge_done = object()
        prefill_stream.record_event.side_effect = lambda: (
            operations.append(("record", None)) or merge_done
        )
        decode_stream = Mock()
        decode_stream.wait_event.side_effect = lambda event: operations.append(
            ("wait", event)
        )
        return prefill_stream, decode_stream, merge_done

    def test_finished_prefill_merge_publishes_decode_dependency(self):
        operations = []
        split_batch = Mock()
        split_batch.chunked_req = None
        split_batch.is_empty.return_value = False
        # The unconditional filter drops nothing here: same size before/after.
        split_batch.batch_size.side_effect = [2, 2]
        running_batch = Mock()
        running_batch.is_empty.return_value = False
        running_batch.batch_is_full = True
        running_batch.merge_batch.side_effect = lambda batch: operations.append(
            ("merge", batch)
        )
        prefill_stream, decode_stream, merge_done = self._make_merge_streams(operations)
        scheduler = SimpleNamespace(
            running_batch=running_batch,
            prefill_batch=split_batch,
            chunked_req=None,
            process_batch_result=Mock(),
        )
        prefill_result = object()

        merged_batch = SchedulerMultiplexMixin._merge_finished_prefill_batch(
            scheduler,
            prefill_result,
            prefill_stream,
            decode_stream,
            running_batch,
        )

        scheduler.process_batch_result.assert_called_once_with(
            split_batch, prefill_result
        )
        split_batch.filter_batch.assert_called_once_with(chunked_req_to_exclude=[])
        self.assertTrue(running_batch.batch_is_full)
        self.assertEqual(
            operations,
            [("merge", split_batch), ("record", None), ("wait", merge_done)],
        )
        self.assertIs(merged_batch, running_batch)
        self.assertIs(scheduler.running_batch, running_batch)
        self.assertIsNone(scheduler.prefill_batch)

    def test_merge_excludes_and_stashes_unfinished_chunked_request(self):
        """A request that only finished a middle chunk must be stashed and
        kept out of the decode batch; merging it would start decoding with a
        partial prefill."""
        operations = []
        chunked_req = _make_chunked_req(extend_end=32, prefix_len=16)
        split_batch = Mock()
        split_batch.chunked_req = chunked_req
        split_batch.batch_size.side_effect = [2, 1]
        split_batch.is_empty.return_value = False
        split_batch.filter_batch.side_effect = lambda **kwargs: operations.append(
            ("filter", kwargs)
        )
        running_batch = Mock()
        running_batch.is_empty.return_value = False
        running_batch.batch_is_full = True
        running_batch.merge_batch.side_effect = lambda batch: operations.append(
            ("merge", batch)
        )
        prefill_stream, decode_stream, merge_done = self._make_merge_streams(operations)
        scheduler = SimpleNamespace(
            running_batch=running_batch,
            prefill_batch=split_batch,
            chunked_req=chunked_req,
            process_batch_result=Mock(),
            stash_chunked_request=Mock(
                side_effect=lambda req: operations.append(("stash", req))
            ),
        )

        merged_batch = SchedulerMultiplexMixin._merge_finished_prefill_batch(
            scheduler,
            object(),
            prefill_stream,
            decode_stream,
            running_batch,
        )

        scheduler.stash_chunked_request.assert_called_once_with(chunked_req)
        (filter_op,) = [op for op in operations if op[0] == "filter"]
        self.assertEqual(filter_op[1]["chunked_req_to_exclude"], [chunked_req])
        self.assertFalse(running_batch.batch_is_full)
        self.assertEqual(
            [op[0] for op in operations],
            ["stash", "filter", "merge", "record", "wait"],
        )
        self.assertIs(merged_batch, running_batch)
        self.assertIsNone(scheduler.prefill_batch)

    def test_merge_of_pure_middle_chunk_keeps_decode_batch(self):
        """A batch holding only a middle chunk merges nothing into decode, but
        the dependency event must still be published: the stash frees
        deduplicated KV pages on the prefill stream that decode may
        reallocate right after."""
        operations = []
        chunked_req = _make_chunked_req(extend_end=32, prefix_len=16)
        split_batch = Mock()
        split_batch.chunked_req = chunked_req
        split_batch.batch_size.side_effect = [1, 0]
        split_batch.is_empty.return_value = True
        running_batch = Mock()
        running_batch.batch_is_full = True
        prefill_stream, decode_stream, merge_done = self._make_merge_streams(operations)
        scheduler = SimpleNamespace(
            running_batch=running_batch,
            prefill_batch=split_batch,
            chunked_req=chunked_req,
            process_batch_result=Mock(),
            stash_chunked_request=Mock(),
        )

        merged_batch = SchedulerMultiplexMixin._merge_finished_prefill_batch(
            scheduler,
            object(),
            prefill_stream,
            decode_stream,
            running_batch,
        )

        running_batch.merge_batch.assert_not_called()
        self.assertIs(merged_batch, running_batch)
        self.assertIs(scheduler.running_batch, running_batch)
        self.assertFalse(running_batch.batch_is_full)
        self.assertEqual(
            [op[0] for op in operations],
            ["record", "wait"],
        )

    def test_merge_skips_stash_for_parked_chunk(self):
        """A parked chunk (no new KV beyond the cached prefix) must be
        excluded from the merge without being stashed — stashing it would be
        a no-op insert that still pays radix-cache work."""
        chunked_req = _make_chunked_req(extend_end=16, prefix_len=16)
        split_batch = Mock()
        split_batch.chunked_req = chunked_req
        split_batch.batch_size.side_effect = [1, 0]
        split_batch.is_empty.return_value = True
        running_batch = Mock()
        prefill_stream, decode_stream, _ = self._make_merge_streams([])
        scheduler = SimpleNamespace(
            running_batch=running_batch,
            prefill_batch=split_batch,
            chunked_req=chunked_req,
            process_batch_result=Mock(),
            stash_chunked_request=Mock(),
        )

        SchedulerMultiplexMixin._merge_finished_prefill_batch(
            scheduler,
            object(),
            prefill_stream,
            decode_stream,
            running_batch,
        )

        scheduler.stash_chunked_request.assert_not_called()
        split_batch.filter_batch.assert_called_once_with(
            chunked_req_to_exclude=[chunked_req]
        )

    def test_update_prefill_batch_processes_pending_chunked_abort(self):
        """PDMux never calls get_next_batch_to_run, so the mixin must drain
        pending chunked aborts itself; without this an aborted chunked request
        leaks its KV forever."""
        running_batch = _Batch(empty=True)
        scheduler = SimpleNamespace(
            prefill_batch=None,
            process_pending_chunked_abort=Mock(),
            get_new_batch_prefill=Mock(
                return_value=SimpleNamespace(
                    batch_to_run=None, running_batch=running_batch
                )
            ),
            dp_attn_adapter=SimpleNamespace(
                maybe_prepare_mlp_sync_batch=Mock(return_value=None)
            ),
        )

        created, returned = SchedulerMultiplexMixin.update_prefill_batch(
            scheduler, 1, running_batch
        )

        scheduler.process_pending_chunked_abort.assert_called_once_with()
        scheduler.dp_attn_adapter.maybe_prepare_mlp_sync_batch.assert_called_once_with(
            None
        )
        self.assertFalse(created)
        self.assertIs(returned, running_batch)

    def test_update_prefill_batch_prepares_dp_attention_metadata(self):
        """PDMux prefill formation must initialize the DP/MLP sync fields.

        The regular scheduler prepares these fields in
        ``get_next_batch_to_run``; PDMux stores the batch directly and must do
        the equivalent work before launching it on the prefill stream.
        """
        running_batch = _Batch(empty=True)
        batch = _Batch(empty=False)
        prepared_batch = _Batch(empty=False)
        adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=Mock(return_value=prepared_batch)
        )
        scheduler = SimpleNamespace(
            prefill_batch=None,
            process_pending_chunked_abort=Mock(),
            get_new_batch_prefill=Mock(
                return_value=SimpleNamespace(
                    batch_to_run=batch, running_batch=running_batch
                )
            ),
            dp_attn_adapter=adapter,
        )

        created, returned = SchedulerMultiplexMixin.update_prefill_batch(
            scheduler, 1, running_batch
        )

        self.assertTrue(created)
        self.assertIs(returned, running_batch)
        self.assertIs(scheduler.prefill_batch, prepared_batch)
        adapter.maybe_prepare_mlp_sync_batch.assert_called_once_with(batch)

    def test_update_prefill_batch_accepts_peer_dp_idle_batch(self):
        """A rank with no local request must still run a peer's DP prefill step."""
        running_batch = _Batch(empty=True)
        idle_batch = _Batch(empty=True)
        adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=Mock(return_value=idle_batch)
        )
        scheduler = SimpleNamespace(
            prefill_batch=None,
            process_pending_chunked_abort=Mock(),
            get_new_batch_prefill=Mock(
                return_value=SimpleNamespace(
                    batch_to_run=None, running_batch=running_batch
                )
            ),
            dp_attn_adapter=adapter,
        )

        created, returned = SchedulerMultiplexMixin.update_prefill_batch(
            scheduler, 1, running_batch
        )

        self.assertTrue(created)
        self.assertIs(returned, running_batch)
        self.assertIs(scheduler.prefill_batch, idle_batch)
        adapter.maybe_prepare_mlp_sync_batch.assert_called_once_with(None)

    def test_event_loop_prepares_decode_dp_attention_metadata(self):
        """PDMux decode bypasses get_next_batch_to_run and needs its own sync."""

        class StopLoop(Exception):
            pass

        @contextmanager
        def use_stream(_stream):
            yield

        running_batch = _Batch(empty=True)
        prefill_stream = Mock()
        decode_stream = Mock()
        adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=Mock(side_effect=StopLoop)
        )
        scheduler = SimpleNamespace(
            running_batch=running_batch,
            prefill_batch=None,
            stream_groups=[(prefill_stream, decode_stream)],
            sm_counts=[(1, 1)],
            request_receiver=SimpleNamespace(recv_requests=Mock(return_value=[])),
            process_input_requests=Mock(),
            update_prefill_batch=Mock(return_value=(False, running_batch)),
            update_running_batch=Mock(return_value=running_batch),
            on_idle=Mock(),
            dp_attn_adapter=adapter,
        )

        with (
            self._stubbed_stream_idx(),
            patch(
                "sglang.srt.multiplex.multiplexing_mixin.torch.cuda.stream",
                side_effect=use_stream,
            ),
            patch("sglang.srt.multiplex.multiplexing_mixin.torch.cuda.empty_cache"),
            self.assertRaises(StopLoop),
        ):
            SchedulerMultiplexMixin.event_loop_pdmux(scheduler)

        adapter.maybe_prepare_mlp_sync_batch.assert_called_once_with(None)

    def test_update_prefill_batch_defers_abort_while_chunk_in_flight(self):
        """Tearing down a chunked request while its token forward is running
        is unsafe; the abort must wait for the between-chunks safe point."""
        running_batch = _Batch(empty=True)
        scheduler = SimpleNamespace(
            prefill_batch=Mock(),
            process_pending_chunked_abort=Mock(),
        )

        created, returned = SchedulerMultiplexMixin.update_prefill_batch(
            scheduler, 1, running_batch
        )

        scheduler.process_pending_chunked_abort.assert_not_called()
        self.assertFalse(created)
        self.assertIs(returned, running_batch)


if __name__ == "__main__":
    unittest.main()
