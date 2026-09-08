import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSparkPDMux(unittest.TestCase):
    def test_pdmux_uses_the_regular_token_chunk_prefill_path(self):
        batch_output = SimpleNamespace(logits_output=object())
        target_worker = SimpleNamespace(
            forward_batch_generation=Mock(return_value=batch_output)
        )
        worker = object.__new__(DSparkWorkerV2)
        worker._target_worker = target_worker
        worker._finalize_prefill = Mock(return_value="finalized")
        batch = SimpleNamespace()

        result = worker._forward_prefill(batch, on_publish=None)

        self.assertEqual(result, "finalized")
        target_worker.forward_batch_generation.assert_called_once_with(
            batch, capture_hidden_mode=unittest.mock.ANY
        )
        worker._finalize_prefill.assert_called_once_with(
            batch, batch_output, on_publish=None
        )

    def test_stream_switch_updates_target_and_draft_runners(self):
        worker = object.__new__(DSparkWorkerV2)
        worker.model_runner = SimpleNamespace(update_decode_attn_backend=Mock())
        worker.draft_model_runner = SimpleNamespace(update_decode_attn_backend=Mock())

        worker.update_pdmux_decode_attn_backend(2)

        worker.model_runner.update_decode_attn_backend.assert_called_once_with(2)
        worker.draft_model_runner.update_decode_attn_backend.assert_called_once_with(2)


if __name__ == "__main__":
    unittest.main()
