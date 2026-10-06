"""Token synchronization must not initialize NCCL for a singleton group."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers import sampler as sampler_module
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSamplerTokenSync(unittest.TestCase):
    def _sync(self, *, world_size, grammars, force_sync=False):
        sampler = sampler_module.Sampler.__new__(sampler_module.Sampler)
        sampler.tp_sync_group = object()
        token_ids = torch.tensor([7, 11])
        original_token_ids = token_ids.clone()
        sampling_info = SimpleNamespace(grammars=grammars)
        with (
            patch.object(sampler_module, "SYNC_TOKEN_IDS_ACROSS_TP", force_sync),
            patch.object(
                sampler_module.dist, "get_world_size", return_value=world_size
            ) as get_world_size,
            patch.object(sampler_module.dist, "all_reduce") as all_reduce,
        ):
            sampler._sync_token_ids_across_tp(token_ids, sampling_info)

        torch.testing.assert_close(token_ids, original_token_ids)
        if grammars or force_sync:
            if world_size > 1:
                all_reduce.assert_called_once_with(
                    token_ids,
                    op=sampler_module.dist.ReduceOp.MIN,
                    group=sampler.tp_sync_group,
                )
            else:
                all_reduce.assert_not_called()
        else:
            get_world_size.assert_not_called()
            all_reduce.assert_not_called()

    def test_grammar_singleton_skips_collective(self):
        self._sync(world_size=1, grammars=[object()])

    def test_forced_singleton_skips_collective(self):
        self._sync(world_size=1, grammars=None, force_sync=True)

    def test_grammar_multi_rank_preserves_collective(self):
        self._sync(world_size=2, grammars=[object()])

    def test_forced_multi_rank_preserves_collective(self):
        self._sync(world_size=2, grammars=None, force_sync=True)

    def test_unsynchronized_sampling_skips_group_lookup(self):
        self._sync(world_size=8, grammars=None)


if __name__ == "__main__":
    unittest.main()
