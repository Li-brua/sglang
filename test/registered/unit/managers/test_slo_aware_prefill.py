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

    def test_dp_attention_normalizes_min_chunk_to_local_budget(self):
        controller = SloAwarePrefillController(
            ttft_slo_ms=1000,
            tpot_slo_ms=100,
            base_chunked_prefill_size=4096,
            max_prefill_tokens=4096,
            page_size=1,
            tile_size=128,
            min_chunk_size=4096,
            prefill_priority_boost=True,
            enable_dp_attention=True,
            dp_size=8,
        )

        self.assertEqual(controller.min_chunk_size, 512)

    def test_dp_attention_keeps_default_min_chunk_local(self):
        controller = SloAwarePrefillController(
            ttft_slo_ms=1000,
            tpot_slo_ms=100,
            base_chunked_prefill_size=4096,
            max_prefill_tokens=4096,
            page_size=1,
            tile_size=128,
            min_chunk_size=None,
            prefill_priority_boost=True,
            enable_dp_attention=True,
            dp_size=8,
        )

        self.assertEqual(controller.min_chunk_size, 128)

    def test_mean_ttft_pressure_uses_average_wait(self):
        controller = SloAwarePrefillController(
            ttft_slo_ms=1000,
            tpot_slo_ms=100,
            base_chunked_prefill_size=1024,
            max_prefill_tokens=4096,
            page_size=1,
            tile_size=128,
            min_chunk_size=None,
            prefill_priority_boost=True,
            ttft_stat="mean",
        )
        running = SimpleNamespace(reqs=[])

        decision = controller.make_decision(
            waiting_queue=[FakeReq(wait_s=0.2), FakeReq(wait_s=0.8)],
            running_batch=running,
            chunked_req=None,
            default_chunked_prefill_size=1024,
            default_prefill_max_requests=None,
        )

        self.assertAlmostEqual(decision.ttft_pressure, 0.5, delta=0.05)
        self.assertEqual(decision.ttft_stat, "mean")

    def test_p90_tpot_pressure_ignores_smallest_tail_sample(self):
        controller = SloAwarePrefillController(
            ttft_slo_ms=1000,
            tpot_slo_ms=100,
            base_chunked_prefill_size=1024,
            max_prefill_tokens=4096,
            page_size=1,
            tile_size=128,
            min_chunk_size=None,
            prefill_priority_boost=True,
            tpot_stat="p90",
        )
        running = SimpleNamespace(
            reqs=[
                FakeReq(prefill_finished_s=0.03, last_decode_finish_s=0.02, output_len=2),
                FakeReq(prefill_finished_s=0.05, last_decode_finish_s=0.04, output_len=2),
                FakeReq(prefill_finished_s=0.07, last_decode_finish_s=0.06, output_len=2),
            ]
        )

        decision = controller.make_decision(
            waiting_queue=[FakeReq(wait_s=0.1)],
            running_batch=running,
            chunked_req=None,
            default_chunked_prefill_size=1024,
            default_prefill_max_requests=None,
        )

        self.assertGreater(decision.tpot_pressure, 0.35)
        self.assertLess(decision.tpot_pressure, 0.70)
        self.assertEqual(decision.tpot_stat, "p90")


    def test_initial_costs_can_be_fixed_by_disabling_online_updates(self):
        controller = SloAwarePrefillController(
            ttft_slo_ms=1000,
            tpot_slo_ms=100,
            base_chunked_prefill_size=1024,
            max_prefill_tokens=4096,
            page_size=1,
            tile_size=128,
            min_chunk_size=None,
            prefill_priority_boost=True,
            initial_prefill_cost_ms_per_1k=20.0,
            initial_decode_cost_ms=7.0,
            disable_online_cost_model=True,
        )

        self.assertAlmostEqual(controller._prefill_cost_per_token_s, 20e-6)
        self.assertAlmostEqual(controller._decode_cost_s, 0.007)
        controller.observe_batch_cost(prefill_tokens=1, decode_tokens=0, elapsed_s=10.0)
        controller.observe_batch_cost(prefill_tokens=0, decode_tokens=1, elapsed_s=10.0)
        self.assertAlmostEqual(controller._prefill_cost_per_token_s, 20e-6)
        self.assertAlmostEqual(controller._decode_cost_s, 0.007)

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
            waiting_queue=[FakeReq(wait_s=0.9)],
            running_batch=running,
            chunked_req=None,
            default_chunked_prefill_size=1024,
            default_prefill_max_requests=None,
        )

        self.assertTrue(decision.allow_prefill)
        self.assertEqual(decision.chunked_prefill_size, 128)
        self.assertEqual(decision.max_prefill_requests, 1)
        self.assertTrue(decision.has_decode_work)
        self.assertFalse(decision.yield_prefill_to_decode)

    def test_ttft_pressure_includes_remaining_prefill_cost(self):
        controller = self.create_controller()
        controller.set_startup_cost_profile(
            prefill_cost_ms=[(128, 100.0), (1024, 800.0)],
            decode_cost_ms=[(1, 10.0)],
        )
        running = SimpleNamespace(reqs=[])

        decision = controller.make_decision(
            waiting_queue=[FakeReq(wait_s=0.2, seqlen=1024)],
            running_batch=running,
            chunked_req=None,
            default_chunked_prefill_size=1024,
            default_prefill_max_requests=None,
        )

        self.assertGreater(decision.ttft_pressure, 0.95)
        self.assertAlmostEqual(
            decision.ttft_remaining_prefill_cost_s,
            0.8,
            delta=0.05,
        )

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

    def test_tpot_objective_uses_profiled_prefill_budget_when_yield_is_unsafe(self):
        controller = SloAwarePrefillController(
            ttft_slo_ms=1000,
            tpot_slo_ms=1000,
            base_chunked_prefill_size=1024,
            max_prefill_tokens=4096,
            page_size=1,
            tile_size=128,
            min_chunk_size=None,
            prefill_priority_boost=True,
        )
        controller.set_startup_cost_profile(
            prefill_cost_ms=[(128, 10.0), (1024, 100.0)],
            decode_cost_ms=[(1, 170.0)],
        )

        decision = controller.make_decision_from_pressure_state(
            pressure_state=SloAwarePrefillPressureState(
                ttft_pressure=0.5,
                tpot_pressure=0.7,
                has_decode_work=True,
                prefill_cost_per_token_s=controller._prefill_cost_per_token_s,
                decode_cost_s=controller._decode_cost_for_batch(1),
            ),
            chunked_req=None,
            default_chunked_prefill_size=1024,
            default_prefill_max_requests=None,
        )

        self.assertEqual(decision.objective, "tpot")
        self.assertTrue(decision.allow_prefill)
        self.assertFalse(decision.yield_prefill_to_decode)
        self.assertEqual(decision.chunked_prefill_size, 384)
        self.assertEqual(decision.max_prefill_requests, 1)

    def test_startup_cost_profile_drives_decode_yield(self):
        controller = self.create_controller()
        controller.yield_guard_ratio = 0.0
        controller.set_startup_cost_profile(
            prefill_cost_ms=[(128, 10.0), (1024, 30.0)],
            decode_cost_ms=[(1, 5.0), (4, 20.0)],
        )

        decision = controller.make_decision_from_pressure_state(
            pressure_state=SloAwarePrefillPressureState(
                ttft_pressure=0.6,
                tpot_pressure=1.2,
                has_decode_work=True,
                prefill_cost_per_token_s=controller._prefill_cost_per_token_s,
                decode_cost_s=controller._decode_cost_for_batch(4),
            ),
            chunked_req=None,
            default_chunked_prefill_size=1024,
            default_prefill_max_requests=None,
        )

        self.assertEqual(decision.objective, "tpot")
        self.assertEqual(decision.chunked_prefill_size, 128)
        self.assertFalse(decision.allow_prefill)
        self.assertTrue(decision.yield_prefill_to_decode)
        self.assertAlmostEqual(decision.decode_cost_s, 0.02)

    def test_contextual_decode_cost_interpolates_context_buckets(self):
        controller = self.create_controller()
        controller.set_startup_cost_profile(
            prefill_cost_ms=[(128, 10.0)],
            decode_cost_by_context_ms=[
                (128, 1, 10.0),
                (128, 4, 20.0),
                (16384, 1, 100.0),
                (16384, 4, 120.0),
            ],
        )

        midpoint = (128 + 16384) // 2

        self.assertAlmostEqual(controller._decode_cost_for_batch(1, 128), 0.010)
        self.assertAlmostEqual(controller._decode_cost_for_batch(1, 16384), 0.100)
        self.assertAlmostEqual(
            controller._decode_cost_for_batch(1, midpoint),
            0.055,
            delta=1e-6,
        )
        self.assertAlmostEqual(controller._decode_cost_for_batch(4, 16384), 0.120)

    def test_compute_pressure_uses_runtime_decode_context_len(self):
        controller = self.create_controller()
        controller.set_startup_cost_profile(
            prefill_cost_ms=[(128, 10.0)],
            decode_cost_by_context_ms=[
                (128, 1, 10.0),
                (32768, 1, 90.0),
            ],
        )
        running = SimpleNamespace(
            reqs=[FakeReq(prefill_finished_s=0.01, output_len=1, seqlen=32768)]
        )

        pressure_state = controller.compute_pressure_state(
            waiting_queue=[],
            running_batch=running,
            chunked_req=None,
        )

        self.assertEqual(pressure_state.decode_context_len, 32768)
        self.assertAlmostEqual(pressure_state.decode_cost_s, 0.090)

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

    def test_high_ttft_restores_full_chunk_in_ttft_objective(self):
        controller = self.create_controller()

        decision = controller.make_decision_from_pressure_state(
            pressure_state=SloAwarePrefillPressureState(
                ttft_pressure=1.6,
                tpot_pressure=1.2,
                has_decode_work=True,
            ),
            chunked_req=None,
            default_chunked_prefill_size=1024,
            default_prefill_max_requests=None,
        )

        self.assertEqual(decision.objective, "ttft")
        self.assertEqual(decision.chunked_prefill_size, 1024)

    def test_high_ttft_restores_full_chunk_even_in_tpot_objective(self):
        controller = self.create_controller()

        decision = controller.make_decision_from_pressure_state(
            pressure_state=SloAwarePrefillPressureState(
                ttft_pressure=1.6,
                tpot_pressure=1.8,
                has_decode_work=True,
            ),
            chunked_req=None,
            default_chunked_prefill_size=1024,
            default_prefill_max_requests=None,
        )

        self.assertEqual(decision.objective, "tpot")
        self.assertEqual(decision.chunked_prefill_size, 1024)
        self.assertEqual(decision.max_prefill_requests, 1)

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
        self.assertEqual(decision.chunked_prefill_size, 1024)
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
