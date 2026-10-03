"""CPU regressions for the DSpark/PDMux prefill-to-decode handoff."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.arg_groups.validation_hook import check_pdmux_speculative_compat
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

WORKER_MODULE = "sglang.srt.speculative.dspark_components.dspark_worker_v2"
EAGER_MODULE = "sglang.srt.model_executor.runner.eager_runner"


class TestDSparkPDMux(unittest.TestCase):
    def setUp(self):
        self.worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        self.worker.model_runner = SimpleNamespace(
            model_config=SimpleNamespace(num_hidden_layers=3),
            prefill_attention_backend_str="torch_native",
            attn_backend=Mock(),
            decode_attn_backend=Mock(),
        )
        self.worker._target_worker = Mock()
        self.worker._target_worker.model_runner = self.worker.model_runner
        self.worker.model_runner.model = Mock()
        self.worker._target_hidden_projection_enabled = False
        self.worker._tp_sync = Mock()
        self.worker._prefill_tp_sync = Mock()
        self.worker._verify_planner = Mock()
        self.worker._observers = Mock()
        self.worker._kv_injector = Mock()
        self.worker.device = "cpu"
        self.worker.verify_num_draft_tokens = 4
        self.batch = SimpleNamespace(
            split_index=0,
            split_forward_batch=SimpleNamespace(split_index=1),
            forward_mode=ForwardMode.SPLIT_PREFILL,
            seq_lens=torch.tensor([7, 13]),
            prefix_lens=[5, 10],
            extend_lens=[2, 3],
            out_cache_loc=torch.tensor([20, 21, 30, 31, 32]),
            req_pool_indices=torch.tensor([1, 2]),
        )
        self.hidden = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        self.result = SimpleNamespace(
            logits_output=SimpleNamespace(
                hidden_states=self.hidden, hidden_states_token_indices=None
            ),
            next_token_ids=torch.tensor([8, 9]),
            new_seq_lens=None,
            next_draft_input=None,
        )
        self.worker.target_worker.forward_batch_split_prefill.return_value = self.result
        self.unified_patch = patch(
            f"{WORKER_MODULE}.is_unified_kv_triton", return_value=False
        )
        self.unified_patch.start()
        self.addCleanup(self.unified_patch.stop)

    def test_intermediate_slices_do_not_inject_or_publish_draft_state(self):
        for start, end in ((0, 1), (1, 2)):
            self.batch.split_index = start
            self.batch.split_forward_batch.split_index = end
            result = self.worker.forward_batch_split_prefill(self.batch)
            self.assertIs(result, self.result)
            self.assertIsNone(result.next_draft_input)
        self.worker._kv_injector.inject_target_hidden.assert_not_called()
        self.worker._prefill_tp_sync.sync.assert_not_called()
        self.worker._verify_planner.note_non_decode_step.assert_called_once()
        self.worker._observers.note_prefill_step.assert_called_once()
        self.worker.target_worker.forward_batch_split_prefill.assert_called_with(
            self.batch, capture_hidden_mode=CaptureHiddenMode.FULL
        )

    def test_final_slice_injects_chunk_positions_and_carries_bonus_tokens(self):
        self.batch.split_forward_batch.split_index = 3
        result = self.worker.forward_batch_split_prefill(self.batch)
        injected = self.worker._kv_injector.inject_target_hidden.call_args.kwargs
        self.assertIs(injected["target_hidden"], self.hidden)
        torch.testing.assert_close(injected["cache_loc"], self.batch.out_cache_loc)
        torch.testing.assert_close(
            injected["positions"], torch.tensor([5, 6, 10, 11, 12])
        )
        self.assertIsInstance(result.next_draft_input, DFlashDraftInputV2)
        torch.testing.assert_close(
            result.next_draft_input.bonus_tokens, self.result.next_token_ids
        )
        torch.testing.assert_close(
            result.next_draft_input.new_seq_lens, self.batch.seq_lens
        )
        self.assertIs(result.new_seq_lens, self.batch.seq_lens)
        self.assertIsNone(result.logits_output.hidden_states)
        self.worker._prefill_tp_sync.sync.assert_called_once()
        self.worker._tp_sync.sync.assert_not_called()

    def test_hidden_token_subset_uses_matching_cache_slots_and_positions(self):
        self.batch.split_forward_batch.split_index = 3
        indices = torch.tensor([1, 4])
        self.result.logits_output.hidden_states = self.hidden[indices]
        self.result.logits_output.hidden_states_token_indices = indices
        self.worker.forward_batch_split_prefill(self.batch)
        injected = self.worker._kv_injector.inject_target_hidden.call_args.kwargs
        torch.testing.assert_close(injected["cache_loc"], torch.tensor([21, 32]))
        torch.testing.assert_close(injected["positions"], torch.tensor([6, 12]))
        self.assertIsNone(self.result.logits_output.hidden_states_token_indices)

    def test_idle_rank_executes_all_slices_and_only_publishes_at_completion(self):
        self.batch.forward_mode = ForwardMode.IDLE
        self.assertIs(self.worker.forward_batch_split_prefill(self.batch), self.result)
        self.batch.split_index = 1
        self.batch.split_forward_batch.split_index = 3
        result = self.worker.forward_batch_split_prefill(self.batch)
        self.assertEqual(result.next_draft_input.bonus_tokens.numel(), 0)
        self.assertEqual(result.new_seq_lens.numel(), 0)
        self.assertEqual(
            self.worker.target_worker.forward_batch_split_prefill.call_count, 2
        )
        self.worker._kv_injector.inject_target_hidden.assert_not_called()
        self.worker._prefill_tp_sync.sync.assert_not_called()

    def test_missing_features_fail_before_draft_kv_injection(self):
        self.batch.split_forward_batch.split_index = 3
        self.result.logits_output.hidden_states = None
        with self.assertRaisesRegex(RuntimeError, "target aux hidden capture"):
            self.worker.forward_batch_split_prefill(self.batch)
        self.worker._kv_injector.inject_target_hidden.assert_not_called()

    def test_unsplit_prefill_uses_the_same_handoff(self):
        self.batch.forward_mode = ForwardMode.EXTEND
        self.worker.target_worker.forward_batch_generation.return_value = self.result
        publish = Mock()
        result = self.worker._forward_prefill(self.batch, publish)
        self.assertIsInstance(result.next_draft_input, DFlashDraftInputV2)
        publish.assert_called_once_with(self.batch.seq_lens)
        self.worker._kv_injector.inject_target_hidden.assert_called_once()

    def test_mamba_commit_uses_the_backend_that_ran_verify(self):
        self.worker._need_mamba_verify_commit = True
        self.batch.mamba_track_indices = None
        for enabled in (True, False):
            with (
                self.subTest(pdmux=enabled),
                patch(
                    f"{WORKER_MODULE}.get_disagg",
                    return_value=SimpleNamespace(enable_pdmux=enabled),
                ),
                patch(
                    f"{WORKER_MODULE}.get_spec",
                    return_value=SimpleNamespace(speculative_eagle_topk=1),
                ),
            ):
                self.worker._commit_target_mamba_states_after_verify(
                    batch=self.batch,
                    seq_lens_pre_verify=self.batch.seq_lens,
                    seq_lens_post_verify=self.batch.seq_lens + 2,
                    commit_lens=torch.tensor([1, 2]),
                )
        for backend in (
            self.worker.model_runner.attn_backend,
            self.worker.model_runner.decode_attn_backend,
        ):
            backend.update_mamba_state_after_mtp_verify.assert_called_once()
            torch.testing.assert_close(
                backend.update_mamba_state_after_mtp_verify.call_args.kwargs[
                    "last_correct_step_indices"
                ],
                torch.tensor([0, 1]),
            )

    def test_target_worker_keeps_full_capture_on_the_persistent_forward_batch(self):
        forward_batch = SimpleNamespace(capture_hidden_mode=CaptureHiddenMode.FULL)
        runner = Mock()
        runner.forward.return_value = SimpleNamespace(
            logits_output=None, can_run_graph=False, expert_distribution_metrics=None
        )
        target = SimpleNamespace(
            model_runner=runner,
            set_hicache_consumer=Mock(),
            _maybe_finalize_elastic_cuda_graph_scale=Mock(),
        )
        self.batch.hicache_consumer_index = 5
        self.batch.split_forward_count = 1
        with patch(
            "sglang.srt.managers.tp_worker.ForwardBatch.init_new",
            return_value=forward_batch,
        ) as init:
            TpModelWorker.forward_batch_split_prefill(
                target, self.batch, capture_hidden_mode=CaptureHiddenMode.FULL
            )
            self.batch.split_index = 1
            TpModelWorker.forward_batch_split_prefill(
                target, self.batch, capture_hidden_mode=CaptureHiddenMode.FULL
            )
        init.assert_called_once_with(
            self.batch,
            runner,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            return_hidden_states_before_norm=False,
        )
        self.assertIs(self.batch.split_forward_batch, forward_batch)
        self.assertEqual(runner.forward.call_count, 2)
        self.assertEqual(target.set_hicache_consumer.call_count, 2)


class TestPDMuxVerifyBackend(unittest.TestCase):
    def test_idle_split_prefill_cannot_replay_a_decode_graph(self):
        runner = SimpleNamespace(
            device="cuda",
            attn_backend=Mock(),
            decode_cuda_graph_runner=Mock(),
            hisparse_coordinator=None,
            _prepare_eager_forward_batch=Mock(),
            _maybe_execute_deferred_mamba_cow_and_clear=Mock(),
            forward_split_prefill=Mock(return_value="split result"),
            pp_group=SimpleNamespace(is_last_rank=True),
        )
        runner.decode_cuda_graph_runner.can_run_graph.return_value = True
        batch = SimpleNamespace(
            forward_mode=ForwardMode.IDLE, global_num_tokens_cpu=None
        )
        with patch(
            "sglang.srt.model_executor.model_runner.get_global_dwdp_manager",
            return_value=None,
        ):
            result = ModelRunner._forward_raw(
                runner, batch, None, split_forward_count=1
            )
        self.assertEqual(result.logits_output, "split result")
        self.assertFalse(result.can_run_graph)
        runner.decode_cuda_graph_runner.can_run_graph.assert_not_called()
        runner.decode_cuda_graph_runner.execute.assert_not_called()
        runner.forward_split_prefill.assert_called_once_with(
            batch, reinit_attn_backend=False, forward_count=1
        )

    def test_eager_verify_and_idle_do_not_overwrite_split_prefill_metadata(self):
        prefill, decode = Mock(), Mock()
        observed = []
        runner = SimpleNamespace(
            attn_backend=prefill,
            decode_attn_backend=decode,
            device_timer=None,
            attn_dcp_size=1,
            device="cpu",
            model=SimpleNamespace(
                forward=lambda *args, **kwargs: observed.append(get_attn_backend())
            ),
            prefill_cuda_graph_runner=None,
            _extend_forward_kwargs=lambda *args: {},
            _pp_kwargs=lambda *args: {},
        )
        eager = EagerRunner.__new__(EagerRunner)
        eager.model_runner = runner
        eager.enable_pdmux = True
        with (
            patch(f"{EAGER_MODULE}.is_cp_active", return_value=False),
            patch(f"{EAGER_MODULE}.maybe_publish_prefill_shared_read_done"),
            patch(f"{EAGER_MODULE}.device_timer_ctx", return_value=nullcontext()),
        ):
            for mode in (ForwardMode.TARGET_VERIFY, ForwardMode.IDLE):
                batch = SimpleNamespace(
                    forward_mode=mode,
                    batch_size=1,
                    input_ids=torch.tensor([1]),
                    positions=torch.tensor([7]),
                    needs_forward_metadata_init=lambda: True,
                )
                eager.execute(batch)
        self.assertEqual(observed, [decode, decode])
        self.assertEqual(decode.init_forward_metadata.call_count, 2)
        prefill.init_forward_metadata.assert_not_called()

    def test_supported_speculative_combinations_are_checked_at_startup(self):
        for enabled, algorithm in ((True, None), (True, "DSPARK"), (False, "EAGLE")):
            check_pdmux_speculative_compat(
                SimpleNamespace(enable_pdmux=enabled, speculative_algorithm=algorithm)
            )
        for algorithm in ("EAGLE", "EAGLE3", "DFLASH", "NGRAM"):
            with (
                self.subTest(algorithm=algorithm),
                self.assertRaisesRegex(ValueError, "only DSPARK"),
            ):
                check_pdmux_speculative_compat(
                    SimpleNamespace(enable_pdmux=True, speculative_algorithm=algorithm)
                )


if __name__ == "__main__":
    unittest.main()
