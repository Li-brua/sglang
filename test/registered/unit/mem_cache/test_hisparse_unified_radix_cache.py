import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hisparse_unified_radix_cache import HiSparseUnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHiSparseUnifiedRadixCache(unittest.TestCase):
    def test_direct_backup_uses_controller_normalized_indices(self):
        cache = HiSparseUnifiedRadixCache.__new__(HiSparseUnifiedRadixCache)
        cache.page_size = 2
        cache.host_pool_group = MagicMock()

        normalized_host_indices = torch.tensor([10, 11], dtype=torch.int64)
        normalized_device_indices = torch.tensor([20, 21], dtype=torch.int64)
        normalized_pool_transfer = PoolTransfer(
            name=PoolName.INDEXER,
            host_indices=torch.tensor([30, 31], dtype=torch.int64),
            device_indices=torch.tensor([40, 41], dtype=torch.int64),
        )

        controller = MagicMock()
        controller.io_backend = "direct"
        controller.move_hybrid_indices.return_value = (
            normalized_host_indices,
            normalized_device_indices,
            [normalized_pool_transfer],
        )
        cache.cache_controller = controller

        host_indices = torch.tensor([0, 1], dtype=torch.int64)
        device_indices = torch.tensor([2, 3], dtype=torch.int64)
        pool_transfer = PoolTransfer(
            name=PoolName.INDEXER,
            host_indices=torch.tensor([4, 5], dtype=torch.int64),
            device_indices=torch.tensor([6, 7], dtype=torch.int64),
        )

        cache.backup_from_device_all_layer(
            mem_pool_device=MagicMock(),
            host_indices=host_indices,
            device_indices=device_indices,
            pool_transfers=[pool_transfer],
        )

        controller.move_hybrid_indices.assert_called_once()
        operation = controller.move_hybrid_indices.call_args.args[0]
        self.assertTrue(torch.equal(operation.host_indices, host_indices))
        self.assertTrue(torch.equal(operation.device_indices, device_indices))
        self.assertEqual(operation.pool_transfers[0].name, PoolName.INDEXER)
        self.assertTrue(
            torch.equal(
                operation.pool_transfers[0].host_indices,
                pool_transfer.host_indices,
            )
        )
        self.assertTrue(
            torch.equal(
                operation.pool_transfers[0].device_indices,
                pool_transfer.device_indices,
            )
        )

        cache.host_pool_group.backup_from_device_all_layer.assert_called_once()
        args, kwargs = cache.host_pool_group.backup_from_device_all_layer.call_args
        self.assertIs(args[1], normalized_host_indices)
        self.assertIs(args[2], normalized_device_indices)
        self.assertEqual(kwargs["io_backend"], "direct")
        self.assertEqual(kwargs["pool_transfers"], [normalized_pool_transfer])


if __name__ == "__main__":
    unittest.main()
