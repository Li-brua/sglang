import unittest
from contextlib import nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_v4 import (
    MM_PAD_SHIFT_VALUE,
    DeepseekV4ForCausalLM,
    DeepseekV4Model,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeLayer:
    def __init__(self, layer_id):
        self.layer_id = layer_id
        self.calls = []
        self.hc_post_calls = 0

    def __call__(
        self,
        *,
        hidden_states,
        prev_residual,
        prev_post,
        prev_comb,
        **kwargs,
    ):
        self.calls.append((prev_residual, prev_post, prev_comb))
        value = self.layer_id + 1
        return (
            hidden_states + value,
            torch.tensor(value),
            torch.tensor(value + 10),
            torch.tensor(value + 20),
        )

    def hc_post(self, hidden_states, residual, post, comb):
        self.hc_post_calls += 1
        return hidden_states + residual + post + comb


class TestDeepseekV4SplitPrefill(unittest.TestCase):
    def _make_model(self):
        layers = [_FakeLayer(0), _FakeLayer(1)]
        model = SimpleNamespace(
            vision=None,
            embed_tokens=lambda input_ids: input_ids.float().unsqueeze(-1),
            hc_mult=2,
            layers=layers,
            end_layer=len(layers),
            use_fused_mhc_post_pre=True,
            hc_pre_from_prev_sublayer=False,
            hc_head=lambda hidden, *args: hidden.sum(dim=1),
            hc_head_fn=None,
            hc_head_scale=None,
            hc_head_base=None,
            norm=lambda hidden: hidden,
            dspark_layers_to_capture=None,
        )
        return model, layers

    def _run_split(self, model, forward_batch, split_interval):
        with (
            patch(
                "sglang.srt.models.deepseek_v4.get_parallel",
                return_value=SimpleNamespace(attn_dp_size=1),
            ),
            patch(
                "sglang.srt.models.deepseek_v4.check_cuda_graph_backend",
                return_value=True,
            ),
        ):
            return DeepseekV4Model.forward_split_prefill(
                model,
                torch.tensor([1, 2]),
                torch.tensor([0, 1]),
                forward_batch,
                split_interval,
            )

    def test_pre_mix_split_matches_normal_forward_with_engram_and_dspark(self):
        class PreMixLayer:
            def __init__(self, index):
                self.index = index
                self.engram = None

            def forward_hc_pre_from_prev(self, *, hidden_states, prev_pre, **kwargs):
                return (
                    hidden_states + self.index + 1 + (prev_pre or 0),
                    self.index + 1,
                )

        def make_model():
            model, _ = self._make_model()
            model.hc_pre_from_prev_sublayer = True
            model.use_fused_mhc_post_pre = False
            model.layers = [PreMixLayer(0), PreMixLayer(1)]
            model.layers[1].engram = _Engram()
            model.engram_hasher = Mock(return_value=torch.zeros(2, 1))
            model.late_layer_start = None
            model.start_layer = 0
            model.config = SimpleNamespace(model_type="deepseek_v4")
            model.pp_group = SimpleNamespace(
                world_size=1, is_first_rank=True, is_last_rank=True
            )
            model.dspark_layers_to_capture = [0, 1]
            model._can_run_tbo = lambda batch: False
            model._forward_layers_hc_pre_from_prev = MethodType(
                DeepseekV4Model._forward_layers_hc_pre_from_prev, model
            )
            return model

        class _Engram:
            layer_hash_index = 0

            def __call__(self, hidden, hashes, batch, **kwargs):
                return hidden + 100

        normal_model, split_model = make_model(), make_model()
        normal_batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        split_batch = SimpleNamespace(forward_mode=ForwardMode.SPLIT_PREFILL)
        ids, positions = torch.tensor([1, 2]), torch.tensor([0, 1])
        with (
            patch(
                "sglang.srt.models.deepseek_v4.get_parallel",
                return_value=SimpleNamespace(attn_dp_size=1),
            ),
            patch("sglang.srt.models.deepseek_v4.is_cp_active", return_value=False),
            patch(
                "sglang.srt.models.deepseek_v4.get_attn_backend", return_value=Mock()
            ),
            patch(
                "sglang.srt.models.deepseek_v4.check_cuda_graph_backend",
                return_value=True,
            ),
            patch(
                "sglang.kernels.ops.layernorm.mhc.hc_combine",
                side_effect=lambda x, pre, mult, dtype: (
                    x.reshape(-1, mult, 1).sum(1) + pre
                ),
            ),
        ):
            expected, expected_aux = DeepseekV4Model.forward(
                normal_model, ids, positions, normal_batch, None
            )
            self.assertIsNone(
                DeepseekV4Model.forward_split_prefill(
                    split_model, ids, positions, split_batch, (0, 1)
                )
            )
            self.assertEqual(split_batch.model_specific_states["prev_pre"], 1)
            actual, actual_aux = DeepseekV4Model.forward_split_prefill(
                split_model, ids, positions, split_batch, (1, 2)
            )
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want)
        for got, want in zip(actual_aux, expected_aux):
            torch.testing.assert_close(got, want)
        split_model.engram_hasher.assert_called_once()

    def test_split_uses_input_embeddings(self):
        model, _ = self._make_model()
        batch = SimpleNamespace()
        embeddings = torch.tensor([[10.0], [20.0]])
        with patch(
            "sglang.srt.models.deepseek_v4.get_parallel",
            return_value=SimpleNamespace(attn_dp_size=1),
        ):
            DeepseekV4Model.forward_split_prefill(
                model,
                torch.tensor([1, 2]),
                torch.tensor([0, 1]),
                batch,
                (0, 0),
                input_embeds=embeddings,
            )
        torch.testing.assert_close(
            batch.hidden_states, embeddings.unsqueeze(1).repeat(1, 2, 1)
        )

    def test_split_execution_preserves_cross_layer_mhc_state(self):
        model, layers = self._make_model()
        forward_batch = SimpleNamespace(
            hidden_states=None,
            model_specific_states=None,
            freqs_cis_c4=object(),
            freqs_cis_c128=object(),
        )

        self.assertIsNone(self._run_split(model, forward_batch, (0, 1)))
        self.assertFalse(hasattr(forward_batch, "freqs_cis_c4"))
        self.assertFalse(hasattr(forward_batch, "freqs_cis_c128"))
        self.assertEqual(layers[0].hc_post_calls, 0)

        result = self._run_split(model, forward_batch, (1, 2))

        self.assertIsNotNone(result)
        for actual, expected in zip(layers[1].calls[0], (1, 11, 21)):
            self.assertEqual(actual.item(), expected)
        self.assertEqual(layers[0].hc_post_calls, 0)
        self.assertEqual(layers[1].hc_post_calls, 1)

    def test_split_execution_matches_one_shot_execution(self):
        split_model, _ = self._make_model()
        split_batch = SimpleNamespace(hidden_states=None, model_specific_states=None)
        self._run_split(split_model, split_batch, (0, 1))
        split_result = self._run_split(split_model, split_batch, (1, 2))

        one_shot_model, _ = self._make_model()
        one_shot_batch = SimpleNamespace(hidden_states=None, model_specific_states=None)
        one_shot_result = self._run_split(one_shot_model, one_shot_batch, (0, 2))

        torch.testing.assert_close(split_result[0], one_shot_result[0])
        torch.testing.assert_close(split_result[1], one_shot_result[1])

    def test_split_execution_accumulates_dspark_captures(self):
        split_model, _ = self._make_model()
        split_model.dspark_layers_to_capture = [0, 1]
        split_batch = SimpleNamespace(hidden_states=None, model_specific_states=None)

        self.assertIsNone(self._run_split(split_model, split_batch, (0, 1)))
        split_result, split_aux = self._run_split(split_model, split_batch, (1, 2))

        one_shot_model, _ = self._make_model()
        one_shot_model.dspark_layers_to_capture = [0, 1]
        one_shot_batch = SimpleNamespace(hidden_states=None, model_specific_states=None)
        one_shot_result, one_shot_aux = self._run_split(
            one_shot_model, one_shot_batch, (0, 2)
        )

        torch.testing.assert_close(split_result[0], one_shot_result[0])
        self.assertEqual(len(split_aux), 2)
        for actual, expected in zip(split_aux, one_shot_aux):
            torch.testing.assert_close(actual, expected)

    def test_causal_lm_prepares_image_embeddings_once_and_reuses_remapped_ids(self):
        embeddings = torch.ones(2, 1)
        batch = SimpleNamespace(mm_inputs=[object()], model_specific_states=None)

        def split_forward(ids, positions, current, interval, input_embeds):
            if interval[0] == 0:
                current.model_specific_states = {"input_ids": ids}
                self.assertIs(input_embeds, embeddings)
                return None
            return torch.ones(2, 1), torch.ones(2, 1)

        model = SimpleNamespace(
            vision=object(),
            config=SimpleNamespace(image_token_id=7),
            _prepare_mm_embeddings=Mock(return_value=embeddings),
            model=SimpleNamespace(
                forward_split_prefill=Mock(side_effect=split_forward)
            ),
            logits_processor=Mock(return_value="logits"),
            lm_head=object(),
            capture_aux_hidden_states=False,
        )
        ids = torch.tensor([1, MM_PAD_SHIFT_VALUE + 3])
        attn_context = SimpleNamespace(
            maybe_input_scattered=lambda batch: nullcontext()
        )
        with patch(
            "sglang.srt.models.deepseek_v4.get_attn_tp_context",
            return_value=attn_context,
        ):
            for interval in ((0, 1), (1, 2)):
                DeepseekV4ForCausalLM.forward_split_prefill(
                    model, ids, torch.tensor([0, 1]), batch, interval
                )
        model._prepare_mm_embeddings.assert_called_once_with(ids, batch)
        for call in model.model.forward_split_prefill.call_args_list:
            torch.testing.assert_close(call.args[0], torch.tensor([1, 7]))
        torch.testing.assert_close(
            model.logits_processor.call_args.args[0], torch.tensor([1, 7])
        )

    def test_causal_lm_processes_only_final_split_logits(self):
        prepare_cp = Mock()
        model_forward = Mock(
            side_effect=[None, (torch.tensor([1.0]), torch.tensor([2.0]))]
        )
        logits_processor = Mock(return_value="logits")
        model = SimpleNamespace(
            vision=None,
            _prepare_dsa_prefill_cp=prepare_cp,
            model=SimpleNamespace(forward_split_prefill=model_forward),
            logits_processor=logits_processor,
            lm_head=object(),
            capture_aux_hidden_states=False,
        )
        attn_context = SimpleNamespace(
            maybe_input_scattered=lambda forward_batch: nullcontext()
        )
        args = (
            torch.tensor([1]),
            torch.tensor([0]),
            SimpleNamespace(),
        )

        with patch(
            "sglang.srt.models.deepseek_v4.get_attn_tp_context",
            return_value=attn_context,
        ):
            self.assertIsNone(
                DeepseekV4ForCausalLM.forward_split_prefill(
                    model, *args, split_interval=(0, 1)
                )
            )
            result = DeepseekV4ForCausalLM.forward_split_prefill(
                model, *args, split_interval=(1, 2)
            )

        self.assertEqual(result, "logits")
        self.assertEqual(model_forward.call_count, 2)
        logits_processor.assert_called_once()

    def test_causal_lm_passes_split_dspark_captures_to_logits_processor(self):
        aux = [torch.tensor([[3.0]])]
        model = SimpleNamespace(
            vision=None,
            _prepare_dsa_prefill_cp=Mock(),
            model=SimpleNamespace(
                forward_split_prefill=Mock(
                    return_value=((torch.tensor([1.0]), torch.tensor([2.0])), aux)
                )
            ),
            logits_processor=Mock(return_value="logits"),
            lm_head=object(),
            capture_aux_hidden_states=True,
        )
        attn_context = SimpleNamespace(
            maybe_input_scattered=lambda forward_batch: nullcontext()
        )
        input_ids = torch.tensor([1])
        forward_batch = SimpleNamespace()

        with patch(
            "sglang.srt.models.deepseek_v4.get_attn_tp_context",
            return_value=attn_context,
        ):
            result = DeepseekV4ForCausalLM.forward_split_prefill(
                model,
                input_ids,
                torch.tensor([0]),
                forward_batch,
                split_interval=(0, 2),
            )

        self.assertEqual(result, "logits")
        call = model.logits_processor.call_args
        self.assertIs(call.args[4], aux)
        self.assertIsNone(call.kwargs["hidden_states_before_norm"])


if __name__ == "__main__":
    unittest.main()
