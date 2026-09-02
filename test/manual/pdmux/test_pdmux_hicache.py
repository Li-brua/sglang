"""PDMux + HiCache end-to-end verification. Needs one CUDA GPU with green
context support (compute capability >= 8.0).

    python -m pytest test/manual/pdmux/test_pdmux_hicache.py -v

Covers both PDMux SM layouts, since they share `event_loop_pdmux` and differ
only in how the prefill and decode streams are masked:

  - exclusive green-context partitions (the default divide)
  - overlapped masks (`overlap_decode_full_sm`: decode runs on the full device)

The server is deliberately configured to make the HiCache paths run rather than
sit idle: a small device KV pool forces eviction to host, `write_through` backs
every finished request up, and a small `split_forward_token_budget` makes each
prefill split per layer so it spans many scheduler iterations.

The scheduler-loop property HiCache depends on -- that transfer acks are
drained exactly once per iteration, including the iterations a split prefill
occupies -- is pinned deterministically in
test/registered/unit/multiplex/test_pdmux_hicache_events.py. These tests check
the end-to-end consequences: correct output after a host reload, and acks that
keep draining under long overlapped prefills.
"""

from __future__ import annotations

import os
import re
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

LOAD_BACK_TOKENS = "sglang:load_back_tokens_total"
BACKUP_TOKENS = "sglang:hicache_backup_tokens_total"

# Long enough that a prefill splits per layer at the budget below, so a single
# request occupies many scheduler iterations alongside decode.
PROMPT_LEN = 2048
SPLIT_FORWARD_TOKEN_BUDGET = 512

# Device KV pool, in tokens. Small on purpose: several of these prompts do not
# fit at once, which is what pushes evicted pages to host.
MAX_TOTAL_TOKENS = 8192

# Bounds the eviction phase's runtime when the pool ends up larger than asked.
MAX_EVICTION_ROUNDS = 32

# Green-context partitions must be a multiple of 8 SMs to satisfy both the
# Ampere (min 4, multiple 2) and Hopper (min 8, multiple 8) constraints.
SM_GRANULARITY = 8


def sm_multiple(value: int) -> int:
    return max(SM_GRANULARITY, (value // SM_GRANULARITY) * SM_GRANULARITY)


def _write_pdmux_config(body: str) -> str:
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", prefix="pdmux_hicache_", delete=False
    )
    with handle:
        handle.write(body)
    return handle.name


class PDMuxHiCacheMixin:
    """Server lifecycle plus the checks both SM layouts must pass.

    Plain mixin rather than a `CustomTestCase` base so the shared cases are not
    collected on their own -- they need a concrete layout's config to run.
    Subclasses override the class attributes to retarget a different model;
    `test_dsv4_pdmux_hicache_tp8.py` reuses the cases against DeepSeek V4.
    """

    sm_group_num = 4
    model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    max_total_tokens = MAX_TOTAL_TOKENS
    prompt_len = PROMPT_LEN
    # Extra launch args and env, appended after the shared ones.
    extra_server_args: list = []
    server_env: dict = {}

    @classmethod
    def pdmux_config_body(cls) -> str:
        raise NotImplementedError

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.config_path = _write_pdmux_config(cls.pdmux_config_body())
        popen_kwargs = (
            {"env": {**os.environ, **cls.server_env}} if cls.server_env else {}
        )
        cls.process = popen_launch_server(
            cls.model_path,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-pdmux",
                "--disable-overlap-schedule",
                "--sm-group-num",
                str(cls.sm_group_num),
                "--pdmux-config-path",
                cls.config_path,
                "--enable-hierarchical-cache",
                "--hicache-write-policy",
                "write_through",
                # DSV4 rejects --hicache-size outright (hybrid_pool_assembler
                # raises); --hicache-ratio is the portable knob.
                "--hicache-ratio",
                "2",
                "--max-total-tokens",
                str(cls.max_total_tokens),
                "--mem-fraction-static",
                "0.7",
                *cls.extra_server_args,
            ],
            **popen_kwargs,
        )
        # Size the eviction workload from the pool the server actually built:
        # page rounding and hybrid-SWA splits move it away from the requested
        # --max-total-tokens, and a hardcoded round count silently stops
        # evicting when it drifts.
        info = requests.get(cls.base_url + "/get_server_info", timeout=120).json()
        cls.pool_tokens = info["max_total_num_tokens"]
        cls.eviction_rounds = min(
            MAX_EVICTION_ROUNDS, cls.pool_tokens // cls.prompt_len + 2
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        os.unlink(cls.config_path)

    # --- helpers -----------------------------------------------------------

    def _generate(self, input_ids, max_new_tokens=8):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0,
                },
            },
            timeout=600,
        )
        response.raise_for_status()
        return response.json()

    def _counter(self, name) -> float:
        """Sum a Prometheus counter across its per-pool label sets."""
        metrics = requests.get(self.base_url + "/metrics", timeout=60).text
        return sum(
            float(match.group(1))
            for match in re.finditer(
                rf"^{re.escape(name)}\{{[^}}]*\}}\s+([0-9.eE+-]+)$",
                metrics,
                re.MULTILINE,
            )
        )

    def _flush(self):
        requests.post(self.base_url + "/flush_cache", timeout=120).raise_for_status()

    def _prompt(self, seed) -> list:
        base = 1000 + seed * self.prompt_len
        return list(range(base, base + self.prompt_len))

    # --- cases -------------------------------------------------------------

    def test_evicted_prefix_reloads_from_host_with_identical_output(self):
        """Greedy output must survive a round trip through the host pool.

        The prompt is prefilled, evicted to host by unrelated traffic, then
        requested again -- which loads its KV back layer by layer while the
        model is already reading those layers. A layer read before its
        host-to-device transfer landed would show up as output diverging from
        the cold reference, not as a loud failure.
        """
        self._flush()
        prompt = self._prompt(0)
        cold = self._generate(prompt)

        # Push distinct traffic through the device pool until the prompt's
        # pages are evicted; write_through already backed them up.
        for seed in range(1, self.eviction_rounds + 1):
            self._generate(self._prompt(seed), max_new_tokens=1)

        before = self._counter(LOAD_BACK_TOKENS)
        warm = self._generate(prompt)
        after = self._counter(LOAD_BACK_TOKENS)

        self.assertGreater(after, before, "prompt was not served from the host cache")
        self.assertEqual(warm["output_ids"], cold["output_ids"])

    def test_backups_keep_draining_under_overlapped_long_prefills(self):
        """Transfer acks must keep retiring while split prefills are in flight.

        HiCache write acks are drained by the scheduler loop, and PDMux spends
        most of its iterations inside a split prefill. Long prefills running
        concurrently with decode must not freeze the backup counter -- if they
        did, host pages would stay pinned for the whole prefill and the pool
        would eventually stall.
        """
        self._flush()
        before = self._counter(BACKUP_TOKENS)

        # Seeds well clear of the eviction phase's range, so these prompts are
        # genuinely cold rather than prefix hits from the other case.
        prompts = [self._prompt(seed) for seed in range(200, 206)]
        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            results = list(
                pool.map(lambda p: self._generate(p, max_new_tokens=32), prompts)
            )

        after = self._counter(BACKUP_TOKENS)

        for result in results:
            self.assertEqual(result["meta_info"]["completion_tokens"], 32)
        self.assertGreater(
            after, before, "no device-to-host backup acks retired during the run"
        )
        requests.get(self.base_url + "/health", timeout=60).raise_for_status()


class TestPDMuxExclusivePartitionsHiCache(PDMuxHiCacheMixin, CustomTestCase):
    """Default layout: prefill and decode own disjoint green-context SM sets."""

    @classmethod
    def pdmux_config_body(cls) -> str:
        return (
            f"sm_group_num: {cls.sm_group_num}\n"
            f"split_forward_token_budget: {SPLIT_FORWARD_TOKEN_BUDGET}\n"
        )


class TestPDMuxOverlappedMasksHiCache(PDMuxHiCacheMixin, CustomTestCase):
    """Overlapped layout: prefill stays capped, decode reaches every SM."""

    @classmethod
    def pdmux_config_body(cls) -> str:
        import torch

        total_sm = torch.cuda.get_device_properties(0).multi_processor_count
        # overlap_decode_full_sm requires manual_divisions, and each prefill cap
        # must sit strictly inside (0, total_sm). The decode column is ignored:
        # the loader rewrites it to the full device. The first threshold is 1 so
        # every non-empty decode batch selects a group.
        divisions = [
            (sm_multiple(total_sm // 4), 1),
            (sm_multiple(total_sm // 2), 8),
        ]
        entries = "".join(
            f"  - [{prefill_sm}, 0, {threshold}]\n"
            for prefill_sm, threshold in divisions
        )
        return (
            f"sm_group_num: {cls.sm_group_num}\n"
            f"split_forward_token_budget: {SPLIT_FORWARD_TOKEN_BUDGET}\n"
            "overlap_decode_full_sm: true\n"
            f"manual_divisions:\n{entries}"
        )


if __name__ == "__main__":
    unittest.main()
