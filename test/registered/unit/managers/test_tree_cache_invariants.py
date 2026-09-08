import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_checker(*, is_hybrid_swa=True, is_hybrid_ssm=False):
    tree_cache = SimpleNamespace(
        is_tree_cache=Mock(return_value=True),
        supports_swa=Mock(return_value=True),
        supports_mamba=Mock(return_value=True),
        sanity_check=Mock(),
    )
    checker = SchedulerInvariantChecker(
        is_hybrid_swa=is_hybrid_swa,
        is_hybrid_ssm=is_hybrid_ssm,
        disaggregation_mode=None,
        page_size=1,
        full_tokens_per_layer=None,
        swa_tokens_per_layer=None,
        max_total_num_tokens=0,
        tree_cache=tree_cache,
        token_to_kv_pool_allocator=None,
        req_to_token_pool=None,
        pool_stats_observer=None,
        get_last_batch=lambda: None,
        get_running_batch=lambda: None,
    )
    return checker, tree_cache


class TestTreeCacheInvariants(CustomTestCase):
    def test_full_tree_check_is_disabled_by_default(self):
        checker, tree_cache = _make_checker()

        with envs.SGLANG_CHECK_TREE_CACHE_INVARIANTS.override(False):
            checker._check_tree_cache()

        tree_cache.is_tree_cache.assert_not_called()
        tree_cache.sanity_check.assert_not_called()

    def test_full_tree_check_can_be_enabled_for_debugging(self):
        checker, tree_cache = _make_checker()

        with envs.SGLANG_CHECK_TREE_CACHE_INVARIANTS.override(True):
            checker._check_tree_cache()

        tree_cache.sanity_check.assert_called_once_with()

    def test_enabled_check_still_skips_non_hybrid_caches(self):
        checker, tree_cache = _make_checker(
            is_hybrid_swa=False, is_hybrid_ssm=False
        )

        with envs.SGLANG_CHECK_TREE_CACHE_INVARIANTS.override(True):
            checker._check_tree_cache()

        tree_cache.sanity_check.assert_not_called()


if __name__ == "__main__":
    unittest.main()
