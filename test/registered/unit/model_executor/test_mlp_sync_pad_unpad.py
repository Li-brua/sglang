"""Unit tests for the DP-attention MLP-sync pad/unpad round-trip.

``prepare_mlp_sync_batch`` pads per-request tensors (positions / seq_lens /
req_pool_indices) by appending dummy rows after the real ones so all DP ranks
agree on tensor shapes. ``post_forward_mlp_sync_batch`` must slice them back so
post-forward consumers — seeded sampling (which asserts positions rows ==
sampling rows), ngram token-table updates — never see the padding.

Pure dataclass logic — CPU only.
"""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.distributed import parallel_state
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.layers import dp_attention
from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.managers.scheduler_components.dp_attn import MLPSyncBatchInfo
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.runtime_context import get_context, get_flags, get_forward, get_parallel
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _mock_model_runner(seq_len_fill_value: int = 1) -> MagicMock:
    runner = MagicMock()
    runner.attn_backend.get_cuda_graph_seq_len_fill_value.return_value = (
        seq_len_fill_value
    )
    return runner


def _logits_output(num_rows: int) -> SimpleNamespace:
    return SimpleNamespace(
        next_token_logits=torch.randn(num_rows, 16), hidden_states=None
    )


class TestMlpSyncPadUnpad(CustomTestCase):
    def test_idle_rank_does_not_index_dummy_last_token(self):
        # MLP-sync turns an idle rank into a dummy zero-token EXTEND batch.
        empty = torch.empty(0, dtype=torch.int64)
        batch = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            input_ids=empty,
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([0]),
            out_cache_loc=empty,
            seq_lens_sum=0,
            positions=empty,
            extend_seq_lens=torch.tensor([0]),
            extend_seq_lens_cpu=[0],
            _original_forward_mode=ForwardMode.IDLE,
            _original_batch_size=0,
        )
        hidden = torch.empty(0, 4)
        pruned, *_ = LogitsProcessor._get_pruned_states(
            None, hidden, None, None, LogitsMetadata.from_forward_batch(batch)
        )
        self.assertEqual(pruned.shape, (0, 4))
        # Attention and MLP execution still use the padded mode.
        self.assertEqual(batch.forward_mode, ForwardMode.EXTEND)

    def test_init_mlp_sync_metadata_scales_speculative_request_width(self):
        spec_info = SimpleNamespace(
            num_tokens_per_req=4,
            num_tokens_for_logprob_per_req=2,
        )
        fb = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=2,
            input_ids=torch.arange(8),
            req_pool_indices=torch.tensor([0, 1]),
            seq_lens=torch.tensor([5, 6]),
            out_cache_loc=torch.arange(8),
            seq_lens_sum=11,
            positions=torch.arange(8),
            spec_info=spec_info,
        )
        batch = SimpleNamespace(
            global_num_tokens=[2, 0, 3],
            global_num_tokens_for_logprob=[2, 0, 3],
            can_run_decode_cuda_graph=True,
            can_run_dp_draft_cuda_graph=False,
            dp_spec_prefill_coordination_applied=False,
        )

        fb.init_mlp_sync_metadata(batch, torch.device("cpu"))

        self.assertEqual(fb.original_global_num_tokens_cpu, [2, 0, 3])
        self.assertEqual(fb.global_num_tokens_cpu, [8, 0, 12])
        self.assertEqual(fb.global_num_tokens_for_logprob_cpu, [4, 0, 6])
        torch.testing.assert_close(fb.global_num_tokens_gpu, torch.tensor([8, 0, 12]))
        torch.testing.assert_close(
            fb.global_num_tokens_for_logprob_gpu, torch.tensor([4, 0, 6])
        )
        self.assertTrue(fb.can_run_decode_cuda_graph)
        self.assertFalse(fb.can_run_dp_draft_cuda_graph)

    def test_draft_graph_gate_has_an_independent_dp_vote(self):
        sync_info = MLPSyncBatchInfo(
            num_dp_ranks=1,
            tp_size=1,
            cp_size=1,
            num_tokens=1,
            num_tokens_for_logprob=1,
            can_run_decode_cuda_graph=True,
            can_run_draft_cuda_graph=False,
            can_run_prefill_cuda_graph=False,
            is_extend_in_batch=False,
            local_can_run_tbo=True,
            local_forward_mode=ForwardMode.DECODE.value,
            prefill_cuda_graph_max_prefix_len=128,
        )

        local = sync_info._get_local_tensor(device="cpu")
        fallback = sync_info._get_fallback_tensor(device="cpu")

        self.assertEqual(local[2].item(), 1)
        self.assertEqual(local[7].item(), 128)
        self.assertEqual(local[8].item(), 0)
        # Idle/inactive ranks stay permissive; an active incompatible rank wins
        # through the all-gathered min reduction.
        self.assertEqual(fallback[7].item(), 0)
        self.assertEqual(fallback[8].item(), 1)

    def test_draft_only_gate_does_not_disable_draft_extend_graph(self):
        draft_runner = object.__new__(EAGLEDraftCudaGraphRunner)
        draft_extend_runner = object.__new__(EAGLEDraftExtendCudaGraphRunner)
        for runner in (draft_runner, draft_extend_runner):
            runner.require_mlp_tp_gather = False
            runner.require_mlp_sync = True
            runner.disable_padding = False
            runner.captured_req_width = 1
            runner.max_bs = 8

        forward_batch = SimpleNamespace(
            spec_info=SimpleNamespace(num_tokens_per_req=1),
            batch_size=1,
            seq_lens=torch.ones(1),
            can_run_decode_cuda_graph=True,
            can_run_dp_draft_cuda_graph=False,
        )

        self.assertFalse(draft_runner.can_run_graph(forward_batch))
        self.assertTrue(draft_extend_runner.can_run_graph(forward_batch))

    def test_draft_input_without_hidden_states_runs_padding_hook(self):
        pad_batch = MagicMock()
        spec_info = SimpleNamespace(
            is_draft_input=lambda: True,
            hidden_states=None,
            pad_batch=pad_batch,
        )
        fb = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=1,
            input_ids=torch.tensor([11]),
            req_pool_indices=torch.tensor([5]),
            seq_lens=torch.tensor([7]),
            out_cache_loc=torch.tensor([0]),
            seq_lens_sum=7,
            positions=torch.tensor([6]),
            seq_lens_cpu=torch.tensor([7]),
            lora_ids=[None],
            spec_info=spec_info,
        )

        fb._pad_inputs_to_size(_mock_model_runner(), num_tokens=2, bs=1)

        self.assertIsNone(spec_info.hidden_states)
        pad_batch.assert_called_once()
        pad_tensor_to_size, batch_size = pad_batch.call_args.args
        self.assertEqual(batch_size, 1)
        torch.testing.assert_close(
            pad_tensor_to_size(torch.tensor([1]), 2), torch.tensor([1, 0])
        )

    def test_dp_cuda_graph_batch_size_uses_raw_request_counts(self):
        fb = SimpleNamespace(original_global_num_tokens_cpu=[3, 11, 7])
        self.assertEqual(DecodeCudaGraphRunner._max_dp_batch_size(fb), 11)

        fb.original_global_num_tokens_cpu = None
        with self.assertRaisesRegex(RuntimeError, "raw per-rank request counts"):
            DecodeCudaGraphRunner._max_dp_batch_size(fb)

    def test_decode_post_forward_unpads_per_request_tensors(self):
        fb = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=3,
            input_ids=torch.tensor([11, 12, 13]),
            req_pool_indices=torch.tensor([5, 6, 7]),
            seq_lens=torch.tensor([7, 8, 9]),
            out_cache_loc=torch.tensor([0, 1, 2]),
            seq_lens_sum=24,
            positions=torch.tensor([6, 7, 8]),
            seq_lens_cpu=torch.tensor([7, 8, 9]),
            lora_ids=[None, None, None],
        )
        # Mirror the decode arm of prepare_mlp_sync_batch: record the original
        # batch size, adopt the synced (padded) one, then pad the inputs.
        padded = 5
        fb._original_batch_size = fb.batch_size
        fb.batch_size = padded
        fb._pad_inputs_to_size(_mock_model_runner(), num_tokens=padded, bs=padded)

        # Padding appends dummy rows after the real ones.
        self.assertEqual(fb.positions.shape[0], padded)
        self.assertEqual(fb.seq_lens.shape[0], padded)
        self.assertEqual(fb.req_pool_indices.shape[0], padded)
        torch.testing.assert_close(fb.positions[:3], torch.tensor([6, 7, 8]))

        logits_output = _logits_output(padded)
        fb.post_forward_mlp_sync_batch(logits_output)

        self.assertEqual(fb.batch_size, 3)
        torch.testing.assert_close(fb.positions, torch.tensor([6, 7, 8]))
        torch.testing.assert_close(fb.seq_lens, torch.tensor([7, 8, 9]))
        torch.testing.assert_close(fb.req_pool_indices, torch.tensor([5, 6, 7]))
        torch.testing.assert_close(fb.seq_lens_cpu, torch.tensor([7, 8, 9]))
        self.assertEqual(logits_output.next_token_logits.shape[0], 3)
        # Seeded sampling asserts positions rows == sampled (real) rows.
        self.assertEqual(fb.positions.shape[0], fb.batch_size)

    def test_extend_post_forward_unpads_positions(self):
        fb = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=2,
            input_ids=torch.arange(7),
            req_pool_indices=torch.tensor([1, 2]),
            seq_lens=torch.tensor([3, 4]),
            out_cache_loc=torch.arange(7),
            seq_lens_sum=7,
            positions=torch.tensor([0, 1, 2, 0, 1, 2, 3]),
            seq_lens_cpu=torch.tensor([3, 4]),
            lora_ids=[None, None],
        )
        # Extend keeps batch_size; only token-level tensors get padded.
        fb._original_batch_size = fb.batch_size
        fb._pad_inputs_to_size(_mock_model_runner(), num_tokens=10, bs=2)

        self.assertEqual(fb.positions.shape[0], 10)

        logits_output = _logits_output(10)
        fb.post_forward_mlp_sync_batch(logits_output)

        torch.testing.assert_close(fb.positions, torch.tensor([0, 1, 2, 0, 1, 2, 3]))
        torch.testing.assert_close(fb.seq_lens, torch.tensor([3, 4]))
        # sample() derives prefill sampling positions from seq_lens - 1, so the
        # row count must match the real request count.
        self.assertEqual((fb.seq_lens - 1).shape[0], fb.batch_size)

    def test_split_prefill_intermediate_unpads_without_logits(self):
        fb = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=2,
            input_ids=torch.arange(7),
            req_pool_indices=torch.tensor([1, 2]),
            seq_lens=torch.tensor([3, 4]),
            out_cache_loc=torch.arange(7),
            seq_lens_sum=7,
            positions=torch.tensor([0, 1, 2, 0, 1, 2, 3]),
            seq_lens_cpu=torch.tensor([3, 4]),
            lora_ids=[None, None],
        )
        fb._original_batch_size = fb.batch_size
        fb._pad_inputs_to_size(_mock_model_runner(), num_tokens=10, bs=2)

        fb.post_forward_mlp_sync_batch(None)

        self.assertEqual(fb.batch_size, 2)
        torch.testing.assert_close(fb.positions, torch.tensor([0, 1, 2, 0, 1, 2, 3]))
        torch.testing.assert_close(fb.seq_lens, torch.tensor([3, 4]))
        torch.testing.assert_close(fb.req_pool_indices, torch.tensor([1, 2]))

    def test_draft_extend_dummy_request_pads_cpu_and_gpu_lens(self):
        spec_info = MagicMock()
        spec_info.num_tokens_per_req = 4
        spec_info.is_draft_input.return_value = False
        fb = ForwardBatch(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2,
            batch_size=1,
            input_ids=torch.empty(0, dtype=torch.int64),
            req_pool_indices=torch.empty(0, dtype=torch.int64),
            seq_lens=torch.empty(0, dtype=torch.int64),
            seq_lens_sum=0,
            out_cache_loc=torch.empty(0, dtype=torch.int64),
            positions=torch.empty(0, dtype=torch.int64),
            seq_lens_cpu=torch.empty(0, dtype=torch.int64),
            extend_seq_lens=torch.empty(0, dtype=torch.int32),
            extend_prefix_lens=torch.empty(0, dtype=torch.int64),
            extend_seq_lens_cpu=[],
            extend_prefix_lens_cpu=[],
            extend_logprob_start_lens_cpu=[],
            spec_info=spec_info,
        )

        fb._pad_inputs_to_size(_mock_model_runner(), num_tokens=4, bs=1)

        torch.testing.assert_close(
            fb.extend_seq_lens, torch.tensor([4], dtype=torch.int32)
        )
        torch.testing.assert_close(fb.extend_prefix_lens, torch.tensor([0]))
        self.assertEqual(fb.extend_seq_lens_cpu, [4])
        self.assertEqual(fb.extend_prefix_lens_cpu, [0])
        self.assertEqual(fb.extend_logprob_start_lens_cpu, [0])


class TestDraftScopeMlpSync(CustomTestCase):
    SLOT = 2
    HIDDEN = 8

    def setUp(self):
        override = get_context().override_server_args(
            tp_size=4,
            attn_dp_size=4,
            cuda_graph_config=CudaGraphConfig(prefill=PhaseConfig(bs=[])),
        )
        override.install()
        self.addCleanup(override.restore)

    def _sync_in_draft_scope(self, global_num_tokens):
        num_tokens = global_num_tokens[self.SLOT]
        fb = ForwardBatch(
            forward_mode=ForwardMode.EXTEND,
            batch_size=1,
            input_ids=torch.arange(num_tokens),
            req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([num_tokens]),
            out_cache_loc=torch.arange(num_tokens),
            seq_lens_sum=num_tokens,
            positions=torch.arange(num_tokens),
            seq_lens_cpu=torch.tensor([num_tokens]),
            mm_input_embeds=torch.ones(num_tokens, self.HIDDEN),
            is_extend_in_batch=True,
            global_num_tokens_cpu=list(global_num_tokens),
            global_num_tokens_for_logprob_cpu=list(global_num_tokens),
            global_num_tokens_gpu=torch.zeros(
                len(global_num_tokens), dtype=torch.int32
            ),
        )
        runner = _mock_model_runner()
        runner.attn_backend.get_cpu_graph_seq_len_fill_value.return_value = 1
        runner.is_draft_worker = True
        runner.attn_tp_sequence_sharded.return_value = False
        draft_group = GroupCoordinator.__new__(GroupCoordinator)
        draft_group.world_size = 1
        draft_group.rank_in_group = 0
        with (
            get_flags().dp.override(enabled=True),
            get_parallel().override(
                tp_rank=self.SLOT, attn_tp_rank=0, attn_dp_rank=self.SLOT
            ),
            patch.object(parallel_state, "_TP", draft_group),
            # CPU runners have no driver to pin the synced token counts with.
            patch("sglang.srt.model_executor.forward_batch_info._is_cpu", True),
            parallel_state.patch_tensor_parallel_group(
                draft_group, owns_attention=True
            ),
        ):
            fb.prepare_mlp_sync_batch(runner)
        return fb

    def _assert_token_rows(self, fb, rows):
        self.assertEqual(fb.input_ids.shape[0], rows)
        self.assertEqual(fb.positions.shape[0], rows)
        self.assertEqual(tuple(fb.mm_input_embeds.shape), (rows, self.HIDDEN))

    def test_extend_keeps_its_own_rank_count(self):
        """A draft extend pads to its own DP rank's token count, not rank 0's."""
        fb = self._sync_in_draft_scope([9, 5, 6, 4])
        self.assertEqual(fb.global_num_tokens_cpu, [9, 5, 6, 4])
        self._assert_token_rows(fb, rows=6)

    def test_max_len_extend_pads_mm_input_embeds(self):
        """A MAX_LEN-padded extend pads mm_input_embeds along with input_ids."""
        with get_flags().dp.override(max_len_with_idle=True):
            fb = self._sync_in_draft_scope([9, 0, 6, 4])
        self.assertEqual(fb.global_num_tokens_cpu, [9, 9, 9, 9])
        self._assert_token_rows(fb, rows=9)
        torch.testing.assert_close(fb.mm_input_embeds[6:], torch.zeros(3, self.HIDDEN))


class TestPDMuxMlpSync(CustomTestCase):
    def test_split_prefill_retains_padding_across_decode_steps(self):
        for dp_size in (8, 2):
            with self.subTest(dp_size=dp_size):
                self._check_split_prefill_reuse(SpeculativeAlgorithm.NONE, dp_size)

    def test_dspark_split_prefill_retains_padding_across_decode_steps(self):
        for dp_size in (8, 2):
            with self.subTest(dp_size=dp_size):
                self._check_split_prefill_reuse(SpeculativeAlgorithm.DSPARK, dp_size)

    def _dspark_worker(self, runner, batch, num_tokens):
        """Exercise the real worker finalizer with CPU target-layer outputs."""
        worker = object.__new__(DSparkWorkerV2)
        worker.model_runner = runner
        runner.prefill_attention_backend_str = "dsv4"
        worker.device = "cpu"
        worker.verify_num_draft_tokens = 2
        worker._target_hidden_projection_enabled = False
        worker._verify_planner = SimpleNamespace(note_non_decode_step=MagicMock())
        worker._observers = SimpleNamespace(note_prefill_step=MagicMock())
        worker._tp_sync = MagicMock()
        worker._kv_injector = MagicMock()
        schedule_batch = SimpleNamespace(
            split_index=0,
            split_forward_batch=batch,
            forward_mode=batch.forward_mode,
            seq_lens=batch.seq_lens.clone(),
            req_pool_indices=batch.req_pool_indices.clone(),
            extend_lens=[num_tokens] if num_tokens else [],
            prefix_lens=[0] if num_tokens else [],
            out_cache_loc=batch.out_cache_loc.clone(),
        )
        target_outputs = []

        def target(current, *, capture_hidden_mode):
            self.assertIs(capture_hidden_mode, CaptureHiddenMode.FULL)
            output = runner._forward_raw(
                current.split_forward_batch, None, split_forward_count=2
            )
            output.next_token_ids = torch.zeros(
                current.seq_lens.numel(), dtype=torch.int64
            )
            target_outputs.append(output)
            return output

        worker._target_worker = SimpleNamespace(
            model_runner=runner, forward_batch_split_prefill=target
        )
        return worker, schedule_batch, target_outputs

    def _check_split_prefill_reuse(self, algorithm, dp_size):
        """Resuming a prefill must restore its DP geometry without copying inputs.

        Decode overwrites the process-wide DP sizes between slices. Retaining
        the prepared batch must also preserve the original unpadding bounds,
        including the fabricated request on an idle MAX_LEN participant.
        """
        override = get_context().override_server_args(
            tp_size=8,
            attn_dp_size=dp_size,
            ep_size=1,
            enable_pdmux=True,
            cuda_graph_config=CudaGraphConfig(prefill=PhaseConfig(bs=[])),
        )
        override.install()
        self.addCleanup(override.restore)
        old_dp_metadata = (
            dp_attention.get_global_dp_buffer_len(),
            dp_attention.get_local_dp_buffer_len(),
            dp_attention._DpGatheredBufferWrapper._dp_max_padding,
            dp_attention.get_dp_global_num_tokens(),
            dp_attention._DpGatheredBufferWrapper._global_num_tokens_gpu,
        )
        self.addCleanup(dp_attention.set_dp_buffer_len, *old_dp_metadata)
        old_extend = get_forward().is_extend_in_batch
        self.addCleanup(dp_attention.set_is_extend_in_batch, old_extend)

        cases = (
            (
                ([9, 5, 6, 4, 3, 2, 1, 7], 2, False),
                ([9, 0, 6, 4, 0, 0, 0, 0], 2, True),
                ([9, 0, 6, 4, 0, 0, 0, 0], 1, True),
                ([9, 0, 6, 4, 0, 0, 0, 0], 1, False),
                ([9, 0, 0, 0, 0, 0, 0, 0], 0, False),
                ([9, 0, 0, 0, 0, 0, 0, 0], 7, False),
            )
            if dp_size == 8
            else (
                ([9, 5], 1, False),
                ([9, 0], 0, True),
                ([9, 0], 1, True),
                ([9, 0], 1, False),
                ([9, 0], 0, False),
            )
        )
        attn_tp_size = 8 // dp_size
        for counts, slot, max_len_idle in cases:
            with (
                self.subTest(counts=counts, slot=slot, max_len_idle=max_len_idle),
                get_flags().dp.override(enabled=True, max_len_with_idle=max_len_idle),
                get_parallel().override(
                    tp_rank=slot * attn_tp_size, attn_tp_rank=0, attn_dp_rank=slot
                ),
                patch("sglang.srt.model_executor.forward_batch_info._is_cpu", True),
            ):
                num_tokens = counts[slot]
                active = num_tokens > 0
                batch = ForwardBatch(
                    forward_mode=ForwardMode.SPLIT_PREFILL
                    if active
                    else ForwardMode.IDLE,
                    batch_size=int(active),
                    input_ids=torch.arange(num_tokens),
                    req_pool_indices=torch.zeros(int(active), dtype=torch.long),
                    seq_lens=torch.tensor(
                        [num_tokens] if active else [], dtype=torch.long
                    ),
                    orig_seq_lens=torch.tensor(
                        [num_tokens] if active else [], dtype=torch.long
                    ),
                    out_cache_loc=torch.arange(num_tokens),
                    seq_lens_sum=num_tokens,
                    positions=torch.arange(num_tokens),
                    is_extend_in_batch=True,
                    global_num_tokens_cpu=list(counts),
                    global_num_tokens_gpu=torch.tensor(counts),
                    global_num_tokens_for_logprob_cpu=list(counts),
                    global_num_token_non_padded=torch.tensor(num_tokens),
                    global_num_token_non_padded_cpu=num_tokens,
                )
                runner = object.__new__(ModelRunner)
                runner.device = "cpu"
                runner.is_draft_worker = False
                runner.spec_algorithm = algorithm
                runner.lora_manager = None
                runner.hisparse_coordinator = None
                runner.req_to_token_pool = None
                runner.decode_cuda_graph_runner = None
                runner.device_timer = None
                runner.pp_group = SimpleNamespace(is_last_rank=True)
                runner.model_config = SimpleNamespace(
                    num_hidden_layers=5,
                    linear_attn_registry_result=None,
                    is_draft_model=False,
                    hf_config=SimpleNamespace(
                        model_type="deepseek_v4",
                        architectures=None,
                        get_text_config=lambda: SimpleNamespace(),
                    ),
                )
                runner.attn_backend = _mock_model_runner().attn_backend
                runner.attn_backend.get_cpu_graph_seq_len_fill_value.return_value = 1
                runner.attn_tp_sequence_sharded = lambda rows: False
                prepared_inputs = {}

                def forward(input_ids, positions, forward_batch, interval):
                    aligned_counts = [
                        (n + attn_tp_size - 1) // attn_tp_size * attn_tp_size
                        for n in counts
                    ]
                    expected_counts = (
                        [max(aligned_counts)] * dp_size
                        if max_len_idle and 0 in counts
                        else aligned_counts
                    )
                    self.assertEqual(
                        dp_attention.get_dp_global_num_tokens(), expected_counts
                    )
                    self.assertEqual(
                        dp_attention.get_global_dp_buffer_len(), sum(expected_counts)
                    )
                    self.assertEqual(
                        dp_attention.get_local_dp_buffer_len(), expected_counts[slot]
                    )
                    self.assertTrue(get_forward().is_extend_in_batch)
                    self.assertEqual(
                        dp_attention.is_dp_max_padding(),
                        max_len_idle and 0 in counts,
                    )
                    self.assertIs(
                        dp_attention._DpGatheredBufferWrapper._global_num_tokens_gpu,
                        forward_batch.global_num_tokens_unpadded_gpu,
                    )
                    torch.testing.assert_close(
                        forward_batch.global_num_tokens_unpadded_gpu,
                        torch.tensor(counts),
                    )
                    self.assertEqual(
                        forward_batch.global_num_token_non_padded_cpu, num_tokens
                    )
                    self.assertEqual(
                        forward_batch.num_token_non_padded.item(), num_tokens
                    )
                    for name in ("input_ids", "positions", "seq_lens", "out_cache_loc"):
                        value = getattr(forward_batch, name)
                        if name in prepared_inputs:
                            self.assertIs(value, prepared_inputs[name])
                        else:
                            prepared_inputs[name] = value
                    if forward_batch.hidden_states is None:
                        forward_batch.hidden_states = input_ids.float().unsqueeze(1)
                    forward_batch.hidden_states = (
                        forward_batch.hidden_states + interval[1]
                    )
                    if interval[1] == 5:
                        return SimpleNamespace(
                            next_token_logits=forward_batch.hidden_states.clone(),
                            hidden_states=forward_batch.hidden_states.clone(),
                            hidden_states_token_indices=None,
                        )
                    return None

                runner.model = SimpleNamespace(forward_split_prefill=forward)
                if algorithm.is_dspark():
                    worker, schedule_batch, target_outputs = self._dspark_worker(
                        runner, batch, num_tokens
                    )
                with (
                    patch(
                        "sglang.srt.speculative.dspark_components.dspark_worker_v2."
                        "pdmux_prefill_handoff",
                        side_effect=nullcontext,
                    ) as handoff,
                    patch(
                        "sglang.srt.speculative.dspark_components.dspark_worker_v2."
                        "compute_position",
                        return_value=(torch.arange(num_tokens), None),
                    ),
                    patch(
                        "sglang.srt.speculative.dspark_components.dspark_worker_v2."
                        "is_unified_kv_triton",
                        return_value=False,
                    ),
                    patch.object(
                        runner,
                        "_prepare_eager_forward_batch",
                        wraps=runner._prepare_eager_forward_batch,
                    ) as prepare,
                    patch.object(
                        batch,
                        "post_forward_mlp_sync_batch",
                        wraps=batch.post_forward_mlp_sync_batch,
                    ) as unpad,
                ):
                    for i in range(3):
                        # Decode replaces both host sizes and the device-count
                        # tensor, without changing the retained prefill batch.
                        dp_attention.set_dp_buffer_len(
                            8, 1, True, [1] * 8, torch.ones(8)
                        )
                        dp_attention.set_is_extend_in_batch(False)
                        if algorithm.is_dspark():
                            schedule_batch.split_index = batch.split_index
                            result = worker.forward_batch_split_prefill(schedule_batch)
                            raw_logits = target_outputs[-1].logits_output
                        else:
                            result = runner._forward_raw(
                                batch, None, split_forward_count=2
                            )
                            raw_logits = result.logits_output
                        if i < 2:
                            self.assertIsNone(result.logits_output)
                            self.assertIs(batch.positions, prepared_inputs["positions"])
                            unpad.assert_not_called()
                    prepare.assert_called_once_with(batch)
                    unpad.assert_called_once_with(raw_logits)
                    if algorithm.is_dspark():
                        handoff.assert_called_once_with()
                        worker._verify_planner.note_non_decode_step.assert_called_once()
                        worker._observers.note_prefill_step.assert_called_once()
                        if active:
                            injected = worker._kv_injector.inject_target_hidden.call_args.kwargs
                            self.assertEqual(
                                injected["target_hidden"].shape[0], num_tokens
                            )
                            self.assertEqual(injected["cache_loc"].shape[0], num_tokens)
                            self.assertEqual(injected["positions"].shape[0], num_tokens)
                            torch.testing.assert_close(
                                injected["target_hidden"],
                                (torch.arange(num_tokens).float() + 11).unsqueeze(1),
                            )
                            self.assertIsNone(result.logits_output.hidden_states)
                            self.assertIs(
                                result.next_draft_input.new_seq_lens,
                                schedule_batch.seq_lens,
                            )
                        else:
                            worker._kv_injector.inject_target_hidden.assert_not_called()
                            self.assertIsNone(result.logits_output)
                self.assertEqual(batch.batch_size, int(active))
                self.assertEqual(batch.positions.shape[0], num_tokens)
                self.assertEqual(
                    batch.forward_mode,
                    ForwardMode.SPLIT_PREFILL if active else ForwardMode.IDLE,
                )
                torch.testing.assert_close(
                    raw_logits.next_token_logits,
                    (torch.arange(num_tokens).float() + 11).unsqueeze(1),
                )

    def test_split_prefill_special_workers_keep_per_slice_preparation(self):
        self._check_special_worker_preparation(SpeculativeAlgorithm.NONE)

    def test_dspark_special_workers_keep_per_slice_preparation(self):
        self._check_special_worker_preparation(SpeculativeAlgorithm.DSPARK)

    def _check_special_worker_preparation(self, algorithm):
        """Speculative and request-specific preparation must not be bypassed."""
        override = get_context().override_server_args(enable_pdmux=True)
        override.install()
        self.addCleanup(override.restore)
        for guard in (
            "ordinary",
            "draft",
            "spec_algorithm",
            "spec_info",
            "lora",
            "hisparse",
        ):
            with self.subTest(guard=guard):
                runner = object.__new__(ModelRunner)
                runner.device = "cpu"
                runner.is_draft_worker = guard == "draft"
                runner.spec_algorithm = (
                    SpeculativeAlgorithm.EAGLE
                    if guard == "spec_algorithm"
                    else algorithm
                )
                runner.lora_manager = object() if guard == "lora" else None
                runner.hisparse_coordinator = object() if guard == "hisparse" else None
                runner.decode_cuda_graph_runner = None
                runner.attn_backend = None
                runner.pp_group = SimpleNamespace(is_last_rank=True)
                runner._prepare_eager_forward_batch = MagicMock()
                runner._maybe_execute_deferred_mamba_cow_and_clear = MagicMock()
                runner.forward_split_prefill = MagicMock(return_value=None)
                runner.prefill_cuda_graph_runner = None
                runner.eager_runner = SimpleNamespace(
                    execute=MagicMock(return_value=None)
                )
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.EXTEND
                    if guard == "ordinary"
                    else ForwardMode.SPLIT_PREFILL,
                    global_num_tokens_cpu=[1] * 8,
                    spec_info=object() if guard == "spec_info" else None,
                    split_index=2,
                    post_forward_mlp_sync_batch=MagicMock(),
                )
                with patch(
                    "sglang.srt.model_executor.model_runner.get_global_dwdp_manager",
                    return_value=None,
                ):
                    for _ in range(2):
                        runner._forward_raw(
                            batch,
                            None,
                            split_forward_count=None if guard == "ordinary" else 2,
                        )
                if guard == "ordinary":
                    self.assertEqual(runner.eager_runner.execute.call_count, 2)
                    runner.forward_split_prefill.assert_not_called()
                self.assertEqual(runner._prepare_eager_forward_batch.call_count, 2)
                self.assertEqual(batch.post_forward_mlp_sync_batch.call_count, 2)


if __name__ == "__main__":
    unittest.main()
