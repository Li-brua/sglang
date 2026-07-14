import time
import unittest
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MODULE_PATH = _REPO_ROOT / "python/sglang/srt/managers/slo_aware_prefill.py"
_SPEC = importlib.util.spec_from_file_location("slo_aware_prefill", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
SloAwarePrefillController = _MODULE.SloAwarePrefillController
SloAwarePrefillPressureState = _MODULE.SloAwarePrefillPressureState


class FakeReq:
    def __init__(
        self,
        *,
        wait_s=0.0,
        prefill_finished_s=0.0,
        last_decode_finish_s=0.0,
        output_len=0,
        seqlen=128,
        matched=0,
    ):
        now = time.perf_counter()
        self.time_stats = SimpleNamespace(
            wait_queue_entry_time=now - wait_s if wait_s else 0.0,
            scheduler_recv_time=now - wait_s if wait_s else 0.0,
            prefill_finished_time=(
                now - prefill_finished_s if prefill_finished_s else 0.0
            ),
            last_decode_finish_time=(
                now - last_decode_finish_s if last_decode_finish_s else 0.0
            ),
        )
        self.output_ids = [0] * output_len
        self.seqlen = seqlen
        self.num_matched_prefix_tokens = matched
        self.is_retracted = False

    def finished(self):
        return False


class TestSloAwarePrefillController(unittest.TestCase):
    def create_controller(self):
        return SloAwarePrefillController(
            ttft_slo_ms=1000,
            tpot_slo_ms=100,
            base_chunked_prefill_size=1024,
            max_prefill_tokens=4096,
            page_size=1,
            tile_size=128,
            min_chunk_size=None,
            prefill_priority_boost=True,
        )

    def test_decode_pressure_limits_prefill_before_ttft_slo(self):
        controller = self.create_controller()
        running = SimpleNamespace(
            reqs=[
                FakeReq(
                    prefill_finished_s=1.0,
                    last_decode_finish_s=0.0,
                    output_len=5,
                )
            ]
        )

        decision = controller.make_decision(
            waiting_queue=[FakeReq(wait_s=0.6)],
            running_batch=running,
            chunked_req=None,
            default_chunked_prefill_size=1024,
            default_prefill_max_requests=None,
        )

        self.assertTrue(decision.allow_prefill)
        self.assertEqual(decision.chunked_prefill_size, 256)
        self.assertEqual(decision.max_prefill_requests, 1)
        self.assertTrue(decision.has_decode_work)
        self.assertFalse(decision.yield_prefill_to_decode)

    def test_no_decode_uses_full_prefill_chunk(self):
        controller = self.create_controller()
        running = SimpleNamespace(reqs=[])

        decision = controller.make_decision(
            waiting_queue=[FakeReq(wait_s=0.1)],
            running_batch=running,
            chunked_req=None,
            default_chunked_prefill_size=1024,
            default_prefill_max_requests=None,
        )

        self.assertEqual(decision.objective, "ttft")
        self.assertTrue(decision.allow_prefill)
        self.assertEqual(decision.chunked_prefill_size, 1024)
        self.assertFalse(decision.yield_prefill_to_decode)

    def test_balanced_slack_allows_prefill(self):
        controller = self.create_controller()
        running = SimpleNamespace(
            reqs=[
                FakeReq(
                    prefill_finished_s=0.01,
                    last_decode_finish_s=0.0,
                    output_len=2,
                )
            ]
        )

        decision = controller.make_decision(
            waiting_queue=[FakeReq(wait_s=0.1)],
            running_batch=running,
            chunked_req=None,
            default_chunked_prefill_size=1024,
            default_prefill_max_requests=None,
        )

        self.assertLess(decision.tpot_pressure, 1.0)
        self.assertTrue(decision.allow_prefill)
        self.assertFalse(decision.yield_prefill_to_decode)
        self.assertEqual(decision.objective, "ttft")

    def test_synced_pressure_keeps_pre_slo_tpot_adaptive(self):
        controller = self.create_controller()

        decision = controller.make_decision_from_pressure_state(
            pressure_state=SloAwarePrefillPressureState(
                ttft_pressure=0.10,
                tpot_pressure=0.30,
                has_decode_work=True,
            ),
            chunked_req=None,
            default_chunked_prefill_size=1024,
            default_prefill_max_requests=None,
        )

        self.assertEqual(decision.objective, "tpot")
        self.assertFalse(decision.allow_prefill)
        self.assertTrue(decision.yield_prefill_to_decode)
        self.assertEqual(decision.chunked_prefill_size, 128)

    def test_online_cost_model_expands_chunk_with_fast_prefill(self):
        controller = self.create_controller()
        controller.observe_batch_cost(prefill_tokens=2048, decode_tokens=0, elapsed_s=0.02)

        decision = controller.make_decision_from_pressure_state(
            pressure_state=SloAwarePrefillPressureState(
                ttft_pressure=0.8,
                tpot_pressure=0.2,
                has_decode_work=True,
                prefill_cost_per_token_s=controller._prefill_cost_per_token_s,
                decode_cost_s=controller._decode_cost_s,
            ),
            chunked_req=None,
            default_chunked_prefill_size=1024,
            default_prefill_max_requests=None,
        )

        self.assertEqual(decision.objective, "ttft")
        self.assertEqual(decision.chunked_prefill_size, 1024)

    def test_online_cost_model_constrains_chunk_with_high_tpot_pressure(self):
        controller = self.create_controller()
        controller.observe_batch_cost(prefill_tokens=128, decode_tokens=0, elapsed_s=0.2)

        decision = controller.make_decision_from_pressure_state(
            pressure_state=SloAwarePrefillPressureState(
                ttft_pressure=0.7,
                tpot_pressure=0.75,
                has_decode_work=True,
                prefill_cost_per_token_s=controller._prefill_cost_per_token_s,
                decode_cost_s=controller._decode_cost_s,
            ),
            chunked_req=None,
            default_chunked_prefill_size=1024,
            default_prefill_max_requests=None,
        )

        self.assertEqual(decision.objective, "ttft")
        self.assertEqual(decision.chunked_prefill_size, 128)

    def test_decode_pressure_allows_limited_prefill_after_ttft_slo(self):
        controller = self.create_controller()
        running = SimpleNamespace(
            reqs=[
                FakeReq(
                    prefill_finished_s=1.0,
                    last_decode_finish_s=0.0,
                    output_len=5,
                )
            ]
        )

        decision = controller.make_decision(
            waiting_queue=[FakeReq(wait_s=1.2)],
            running_batch=running,
            chunked_req=None,
            default_chunked_prefill_size=1024,
            default_prefill_max_requests=None,
        )

        self.assertTrue(decision.allow_prefill)
        self.assertEqual(decision.chunked_prefill_size, 512)
        self.assertEqual(decision.max_prefill_requests, 1)
        self.assertTrue(decision.has_decode_work)
        self.assertFalse(decision.yield_prefill_to_decode)
        self.assertEqual(decision.objective, "tpot")

    def test_high_ttft_low_tpot_uses_full_prefill_capacity(self):
        controller = self.create_controller()
        running = SimpleNamespace(
            reqs=[
                FakeReq(
                    prefill_finished_s=0.01,
                    last_decode_finish_s=0.0,
                    output_len=5,
                )
            ]
        )

        decision = controller.make_decision(
            waiting_queue=[FakeReq(wait_s=2.0)],
            running_batch=running,
            chunked_req=None,
            default_chunked_prefill_size=1024,
            default_prefill_max_requests=None,
        )

        self.assertTrue(decision.allow_prefill)
        self.assertEqual(decision.objective, "ttft")
        self.assertEqual(decision.chunked_prefill_size, 1024)
        self.assertIsNone(decision.max_prefill_requests)
        self.assertFalse(decision.yield_prefill_to_decode)

    def test_ambiguous_low_pressure_defaults_to_ttft_without_sticky_tpot(self):
        controller = self.create_controller()
        high_tpot_running = SimpleNamespace(
            reqs=[
                FakeReq(
                    prefill_finished_s=1.0,
                    last_decode_finish_s=0.0,
                    output_len=5,
                )
            ]
        )
        controller.make_decision(
            waiting_queue=[FakeReq(wait_s=0.1)],
            running_batch=high_tpot_running,
            chunked_req=None,
            default_chunked_prefill_size=1024,
            default_prefill_max_requests=None,
        )

        balanced_running = SimpleNamespace(
            reqs=[
                FakeReq(
                    prefill_finished_s=0.018,
                    last_decode_finish_s=0.0,
                    output_len=5,
                )
            ]
        )
        decision = controller.make_decision(
            waiting_queue=[FakeReq(wait_s=0.286)],
            running_batch=balanced_running,
            chunked_req=None,
            default_chunked_prefill_size=1024,
            default_prefill_max_requests=None,
        )

        self.assertEqual(decision.objective, "ttft")
        self.assertTrue(decision.allow_prefill)
        self.assertFalse(decision.yield_prefill_to_decode)

    def test_high_tpot_low_ttft_can_delay_prefill(self):
        controller = self.create_controller()
        running = SimpleNamespace(
            reqs=[
                FakeReq(
                    prefill_finished_s=1.0,
                    last_decode_finish_s=0.0,
                    output_len=4,
                )
            ]
        )

        decision = controller.make_decision(
            waiting_queue=[FakeReq(wait_s=0.1)],
            running_batch=running,
            chunked_req=None,
            default_chunked_prefill_size=1024,
            default_prefill_max_requests=None,
        )

        self.assertFalse(decision.allow_prefill)

    def test_chunked_request_is_never_blocked(self):
        controller = self.create_controller()
        running = SimpleNamespace(
            reqs=[
                FakeReq(
                    prefill_finished_s=1.0,
                    last_decode_finish_s=0.0,
                    output_len=4,
                )
            ]
        )

        decision = controller.make_decision(
            waiting_queue=[FakeReq(wait_s=0.1)],
            running_batch=running,
            chunked_req=FakeReq(wait_s=0.1),
            default_chunked_prefill_size=1024,
            default_prefill_max_requests=None,
        )

        self.assertTrue(decision.allow_prefill)
        self.assertTrue(decision.yield_prefill_to_decode)

    def test_prefill_urgency_sort(self):
        controller = self.create_controller()
        older = FakeReq(wait_s=0.8, seqlen=256)
        newer = FakeReq(wait_s=0.1, seqlen=512)
        decoded = FakeReq(wait_s=2.0, output_len=2)
        queue = [newer, decoded, older]

        controller.prioritize_waiting_queue(queue)

        self.assertIs(queue[0], older)
        self.assertIs(queue[1], newer)
        self.assertIs(queue[2], decoded)


if __name__ == "__main__":
    unittest.main()
