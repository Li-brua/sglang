#!/usr/bin/env python3
"""Probe SLO-aware prefill cost inputs against a running SGLang server.

This script sends streaming OpenAI-compatible chat requests, measures TTFT and
TPOT, and prints warm-start flags for --enable-slo-aware-prefill.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass
class RequestSample:
    ok: bool
    latency_s: float
    ttft_s: float
    tpot_s: float
    prompt_tokens: int
    completion_tokens: int
    chunks: int
    error: Optional[str] = None

    @property
    def prefill_cost_ms_per_1k(self) -> float:
        if self.prompt_tokens <= 0 or self.ttft_s <= 0:
            return 0.0
        return self.ttft_s * 1000.0 * 1000.0 / self.prompt_tokens


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = max(int((pct / 100.0) * len(values) + 0.999999) - 1, 0)
    return values[min(index, len(values) - 1)]


def summarize(values: Iterable[float]) -> dict[str, float]:
    vals = [v for v in values if v > 0]
    if not vals:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "p90": percentile(vals, 90),
        "p95": percentile(vals, 95),
        "p99": percentile(vals, 99),
        "max": max(vals),
    }


def pick_stat(summary: dict[str, float], stat: str) -> float:
    return summary[stat]


def build_prompt(input_len: int, prompt_file: Optional[str], prompt_unit: str) -> str:
    if prompt_file:
        return Path(prompt_file).read_text()
    return prompt_unit * max(input_len, 1)


def read_streaming_chat(
    *,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_s: float,
) -> RequestSample:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first_token_time: Optional[float] = None
    usage: dict[str, Any] = {}
    chunks = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if not data or data == "[DONE]":
                    continue
                obj = json.loads(data)
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices", []) or []:
                    delta = choice.get("delta") or {}
                    text = (
                        delta.get("content")
                        or delta.get("reasoning_content")
                        or choice.get("text")
                    )
                    if text:
                        chunks += 1
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
        end = time.perf_counter()
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        end = time.perf_counter()
        return RequestSample(
            ok=False,
            latency_s=end - start,
            ttft_s=0.0,
            tpot_s=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            chunks=chunks,
            error=str(exc),
        )

    ttft_s = (first_token_time or end) - start
    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion_tokens = int(
        usage.get("completion_tokens") or usage.get("output_tokens") or chunks
    )
    if prompt_tokens <= 0:
        prompt_tokens = 0
    decode_tokens = max(completion_tokens - 1, 0)
    tpot_s = (end - (first_token_time or start)) / decode_tokens if decode_tokens else 0.0
    return RequestSample(
        ok=True,
        latency_s=end - start,
        ttft_s=ttft_s,
        tpot_s=tpot_s,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        chunks=chunks,
    )


def run_phase(
    *,
    name: str,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    concurrency: int,
    num_requests: int,
    warmup: int,
    timeout_s: float,
) -> tuple[list[RequestSample], float]:
    for _ in range(warmup):
        read_streaming_chat(
            url=url,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )

    print(
        f"[probe] phase={name} concurrency={concurrency} "
        f"requests={num_requests} max_tokens={max_tokens}"
    )
    start = time.perf_counter()
    samples: list[RequestSample] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                read_streaming_chat,
                url=url,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
            )
            for _ in range(num_requests)
        ]
        for future in concurrent.futures.as_completed(futures):
            samples.append(future.result())
    duration_s = time.perf_counter() - start
    failed = [s for s in samples if not s.ok]
    if failed:
        print(f"[probe] phase={name} failed={len(failed)} first_error={failed[0].error}")
    return samples, duration_s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30000/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-len", type=int, default=8192)
    parser.add_argument("--prompt-file")
    parser.add_argument("--prompt-unit", default="hello ")
    parser.add_argument("--prefill-output-len", type=int, default=1)
    parser.add_argument("--decode-output-len", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument(
        "--cost-stat",
        choices=["mean", "median", "p90", "p95", "p99", "max"],
        default="p90",
    )
    parser.add_argument("--ttft-slo-ms", type=float, default=10000.0)
    parser.add_argument("--ttft-stat", choices=["mean", "p90", "max"], default="p90")
    parser.add_argument("--tpot-slo-ms", type=float)
    parser.add_argument("--target-decode-throughput", type=float)
    parser.add_argument("--tpot-stat", choices=["mean", "p90", "max"], default="mean")
    parser.add_argument("--output-json")
    args = parser.parse_args()

    if args.tpot_slo_ms is None:
        if args.target_decode_throughput:
            args.tpot_slo_ms = 1000.0 / args.target_decode_throughput
        else:
            args.tpot_slo_ms = 60.0

    prompt = build_prompt(args.input_len, args.prompt_file, args.prompt_unit)
    prefill_samples, prefill_duration_s = run_phase(
        name="prefill",
        url=args.url,
        model=args.model,
        prompt=prompt,
        max_tokens=args.prefill_output_len,
        concurrency=1,
        num_requests=max(args.concurrency, min(args.num_requests, args.concurrency * 2)),
        warmup=args.warmup,
        timeout_s=args.timeout_s,
    )
    decode_samples, decode_duration_s = run_phase(
        name="decode",
        url=args.url,
        model=args.model,
        prompt=prompt,
        max_tokens=args.decode_output_len,
        concurrency=args.concurrency,
        num_requests=args.num_requests,
        warmup=args.warmup,
        timeout_s=args.timeout_s,
    )

    ok_prefill = [s for s in prefill_samples if s.ok]
    ok_decode = [s for s in decode_samples if s.ok]
    prefill_cost = summarize([s.prefill_cost_ms_per_1k for s in ok_prefill])
    ttft = summarize([s.ttft_s * 1000.0 for s in ok_decode])
    tpot = summarize([s.tpot_s * 1000.0 for s in ok_decode])
    total_completion = sum(s.completion_tokens for s in ok_decode)
    output_throughput = total_completion / decode_duration_s if decode_duration_s > 0 else 0.0
    prompt_tokens = summarize([float(s.prompt_tokens) for s in ok_decode])

    recommended_prefill_cost = pick_stat(prefill_cost, args.cost_stat)
    recommended_decode_cost = pick_stat(tpot, args.cost_stat)
    result = {
        "config": vars(args),
        "prefill_phase_duration_s": prefill_duration_s,
        "decode_phase_duration_s": decode_duration_s,
        "successful_prefill_requests": len(ok_prefill),
        "successful_decode_requests": len(ok_decode),
        "prompt_tokens": prompt_tokens,
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "prefill_cost_ms_per_1k": prefill_cost,
        "output_throughput_tok_s": output_throughput,
        "recommended": {
            "slo_prefill_ttft_stat": args.ttft_stat,
            "slo_prefill_ttft_slo_ms": args.ttft_slo_ms,
            "slo_prefill_tpot_stat": args.tpot_stat,
            "slo_prefill_tpot_slo_ms": args.tpot_slo_ms,
            "slo_prefill_initial_prefill_cost_ms_per_1k": recommended_prefill_cost,
            "slo_prefill_initial_decode_cost_ms": recommended_decode_cost,
        },
        "samples": [asdict(s) for s in ok_decode],
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    print("\nSuggested SGLang flags:")
    print(f"  --slo-prefill-ttft-stat {args.ttft_stat} \\")
    print(f"  --slo-prefill-ttft-slo-ms {args.ttft_slo_ms:.3f} \\")
    print(f"  --slo-prefill-tpot-stat {args.tpot_stat} \\")
    print(f"  --slo-prefill-tpot-slo-ms {args.tpot_slo_ms:.3f} \\")
    print(
        "  --slo-prefill-initial-prefill-cost-ms-per-1k "
        f"{recommended_prefill_cost:.6f} \\")
    print(f"  --slo-prefill-initial-decode-cost-ms {recommended_decode_cost:.6f}")
    print(
        "\nIf online EMA drifts badly on your workload, add:\n"
        "  --disable-slo-prefill-online-cost-model"
    )

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
