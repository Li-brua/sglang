"""Full-device decode may overlap a capped PDMux prefill Green Context."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

from sglang.srt.multiplex import pdmux_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def load_config(body):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "pdmux.yaml"
        path.write_text(body)
        return pdmux_context.load_pdmux_config(str(path))


class TestPDMuxOverlapStreams(unittest.TestCase):
    def setUp(self):
        spatial = Mock()
        spatial.get_sm_available.return_value = 132
        spatial.create_greenctx_stream_by_value.return_value = (
            "prefill-green",
            "unused-green",
        )
        kernel = ModuleType("sgl_kernel")
        kernel.spatial = spatial
        self.spatial = spatial
        self.patches = [
            patch.dict(sys.modules, {"sgl_kernel": kernel}),
            patch.object(pdmux_context.torch.cuda, "current_device", return_value=0),
            patch.object(
                pdmux_context.torch.cuda,
                "Stream",
                side_effect=lambda device, priority=0: f"cuda-{device}-{priority}",
            ),
            patch.object(pdmux_context, "STREAM_GROUPS", []),
            patch.object(pdmux_context, "SM_COUNTS", []),
            patch.object(pdmux_context, "_RESERVED_GREEN_STREAMS", []),
            patch.object(pdmux_context, "CURRENT_STREAM_GROUP", None),
        ]
        for active in self.patches:
            active.start()
            self.addCleanup(active.stop)

    def test_full_device_decode_uses_primary_stream(self):
        config = load_config(
            """sm_group_num: 3
manual_divisions:
  - [104, 132, 0]
split_forward_token_budget: 65536
overlap_decode_full_sm: true
"""
        )

        pdmux_context.initialize_stream_groups(0, config)

        self.spatial.create_greenctx_stream_by_value.assert_called_once_with(
            104, 28, 0
        )
        self.assertEqual(pdmux_context.get_sm_counts()[1], (104, 132))
        self.assertEqual(
            pdmux_context.get_stream_groups()[1], ("prefill-green", "cuda-0--1")
        )
        self.assertEqual(pdmux_context._RESERVED_GREEN_STREAMS, ["unused-green"])

    def test_oversubscribed_exclusive_division_is_rejected_early(self):
        config = load_config(
            """sm_group_num: 3
manual_divisions:
  - [104, 132, 0]
"""
        )

        with self.assertRaisesRegex(ValueError, "must equal the device SM count"):
            pdmux_context.initialize_stream_groups(0, config)
        self.spatial.create_greenctx_stream_by_value.assert_not_called()

    def test_overlap_requires_manual_division(self):
        with self.assertRaisesRegex(ValueError, "requires manual_divisions"):
            load_config("sm_group_num: 3\noverlap_decode_full_sm: true\n")


if __name__ == "__main__":
    unittest.main()
