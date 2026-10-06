import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSparkPDMux(unittest.TestCase):
    def test_scheduler_pp_proxy_argument_is_accepted_for_decode(self):
        worker = object.__new__(DSparkWorkerV2)
        worker._hosts_draft = True
        worker.enable_dp_spec_prefill_coordination = False
        worker._forward_decode = Mock(return_value="decoded")
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: False),
            is_extend_in_batch=False,
        )
        proxy = object()

        result = worker.forward_batch_generation(batch, pp_proxy_tensors=proxy)

        self.assertEqual(result, "decoded")
        worker._forward_decode.assert_called_once_with(batch, None, None)

    def test_scheduler_pp_proxy_argument_reaches_target_prefill(self):
        worker = object.__new__(DSparkWorkerV2)
        worker._hosts_draft = True
        worker.enable_dp_spec_prefill_coordination = False
        worker._verify_planner = SimpleNamespace(note_non_decode_step=Mock())
        worker._observers = SimpleNamespace(note_prefill_step=Mock())
        worker._forward_prefill = Mock(return_value="prefilled")
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: True),
            is_extend_in_batch=True,
        )
        proxy = object()

        result = worker.forward_batch_generation(batch, pp_proxy_tensors=proxy)

        self.assertEqual(result, "prefilled")
        worker._forward_prefill.assert_called_once_with(
            batch, None, pp_proxy_tensors=proxy
        )

    def test_layer_split_rejects_target_without_aux_capture_support(self):
        target_worker = SimpleNamespace(
            device="cuda",
            model_runner=SimpleNamespace(model=object()),
        )

        with (
            patch(
                "sglang.srt.speculative.dspark_components.dspark_worker_v2.get_disagg",
                return_value=SimpleNamespace(
                    enable_pdmux=True, pdmux_prefill_mode="layer_split"
                ),
            ),
            patch(
                "sglang.srt.speculative.dspark_components.dspark_worker_v2."
                "get_schedule",
                return_value=SimpleNamespace(page_size=256),
            ),
            patch(
                "sglang.srt.speculative.dspark_components.dspark_worker_v2.get_parallel",
                return_value=SimpleNamespace(
                    pp_group=SimpleNamespace(is_last_rank=True)
                ),
            ),
            self.assertRaisesRegex(NotImplementedError, "auxiliary hidden states"),
        ):
            DSparkWorkerV2(
                server_args=SimpleNamespace(),
                gpu_id=0,
                nccl_port=0,
                target_worker=target_worker,
            )

    def make_worker(self, idle=False):
        worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
        worker.model_runner = SimpleNamespace(
            model_config=SimpleNamespace(num_hidden_layers=2)
        )
        worker._verify_planner = SimpleNamespace(note_non_decode_step=Mock())
        worker._observers = SimpleNamespace(note_prefill_step=Mock())
        worker._finalize_prefill = Mock(return_value="injected")
        worker._decode_idle_result = Mock(return_value="idle")
        batch = SimpleNamespace(
            split_index=0,
            split_forward_batch=SimpleNamespace(split_index=0),
            forward_mode=ForwardMode.IDLE if idle else ForwardMode.SPLIT_PREFILL,
        )
        intermediate = SimpleNamespace(logits_output=None)
        # Real final IDLE may have no logits. Completion is a layer boundary.
        final = SimpleNamespace(logits_output=None if idle else object())

        def target(current, **kwargs):
            current.split_forward_batch.split_index += 1
            return (
                intermediate if current.split_forward_batch.split_index == 1 else final
            )

        worker._target_worker = SimpleNamespace(
            forward_batch_split_prefill=Mock(side_effect=target)
        )
        return worker, batch, intermediate, final

    def run_split(self, idle=False):
        worker, batch, intermediate, final = self.make_worker(idle)
        prefill, decode = Mock(), Mock()
        with (
            patch(
                "sglang.srt.multiplex.pdmux_context.get_current_stream_idx",
                return_value=0,
            ),
            patch(
                "sglang.srt.multiplex.pdmux_context.get_stream_groups",
                return_value=[(prefill, decode)],
            ),
        ):
            self.assertIs(worker.forward_batch_split_prefill(batch), intermediate)
            worker._finalize_prefill.assert_not_called()
            prefill.wait_stream.assert_not_called()
            decode.wait_event.assert_not_called()
            batch.split_index = 1
            self.assertEqual(
                worker.forward_batch_split_prefill(batch),
                "idle" if idle else "injected",
            )
        worker._verify_planner.note_non_decode_step.assert_called_once_with()
        worker._observers.note_prefill_step.assert_called_once_with()
        prefill.wait_stream.assert_called_once_with(decode)
        prefill.record_event.assert_called_once_with()
        decode.wait_event.assert_called_once_with(prefill.record_event.return_value)
        self.assertEqual(
            worker.target_worker.forward_batch_split_prefill.call_args.kwargs,
            {"capture_hidden_mode": CaptureHiddenMode.FULL},
        )
        if idle:
            worker._decode_idle_result.assert_called_once_with(on_publish=None)
            worker._finalize_prefill.assert_not_called()
        else:
            worker._finalize_prefill.assert_called_once_with(
                batch, final, on_publish=None
            )

    def test_final_slice_injects_and_fences_next_decode(self):
        self.run_split()

    def test_final_idle_without_logits_skips_injection(self):
        self.run_split(idle=True)

    def test_stream_switch_updates_target_and_draft_runners(self):
        worker = object.__new__(DSparkWorkerV2)
        worker._hosts_draft = True
        worker.enable_dp_spec_prefill_coordination = False
        worker.model_runner = SimpleNamespace(update_decode_attn_backend=Mock())
        worker.draft_model_runner = SimpleNamespace(update_decode_attn_backend=Mock())

        worker.update_pdmux_decode_attn_backend(2)

        worker.model_runner.update_decode_attn_backend.assert_called_once_with(2)
        worker.draft_model_runner.update_decode_attn_backend.assert_called_once_with(2)


if __name__ == "__main__":
    unittest.main()
