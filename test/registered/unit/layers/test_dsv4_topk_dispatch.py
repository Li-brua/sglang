import unittest
from unittest.mock import Mock, patch

import torch

from sglang.kernels.ops.attention.dsv4.topk import topk_transform_paged_v2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDeepseekV4TopKDispatch(unittest.TestCase):
    def test_cluster_control_is_forwarded_to_jit_kernel(self):
        module = Mock()
        scores = torch.empty((1, 4), dtype=torch.float32)
        seq_lens = torch.tensor([4], dtype=torch.int32)
        page_table = torch.empty((1, 1), dtype=torch.int32)
        out = torch.empty((1, 1), dtype=torch.int32)
        metadata = torch.empty((2, 2), dtype=torch.int32)

        raw_indices = torch.empty_like(out)
        for pdmux_enabled, requested_cluster in ((False, False), (True, True)):
            with self.subTest(pdmux_enabled=pdmux_enabled):
                module.reset_mock()
                with (
                    patch(
                        "sglang.kernels.ops.attention.dsv4.topk._jit_topk_v2_module",
                        return_value=module,
                    ),
                    patch(
                        "sglang.kernels.ops.attention.dsv4.topk.is_pdmux_enabled",
                        return_value=pdmux_enabled,
                    ),
                ):
                    topk_transform_paged_v2(
                        scores,
                        seq_lens,
                        page_table,
                        out,
                        256,
                        metadata,
                        out_raw_indices=raw_indices,
                        enable_cluster=requested_cluster,
                    )
                module.topk_transform_paged.assert_called_once_with(
                    scores,
                    seq_lens,
                    page_table,
                    out,
                    256,
                    metadata,
                    raw_indices,
                    False,
                )


if __name__ == "__main__":
    unittest.main()
