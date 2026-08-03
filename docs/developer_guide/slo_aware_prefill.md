# SLO-Aware Prefill Scheduling 设计文档

本文档描述当前分支中的 **SLO-aware prefill scheduling**。该功能在每个 scheduler iteration 根据 TTFT/TPOT pressure 决定下一轮 forward 跑 prefill batch 还是 decode batch。

默认情况下，prefill 仍使用当前 effective `chunked_prefill_size`。`slo_prefill_min_chunk_size` 只是可选 fallback：仅在不能安全让 decode 插队、但 objective 又偏向 TPOT 时用于缩小本轮 prefill chunk；不传时等于 effective `chunked_prefill_size`，因此不会引入额外 chunk 档位。

- `chunked_prefill_size`：主路径 prefill chunk，也是推荐配置入口。
- `slo_prefill_min_chunk_size`：可选保守 fallback；通常不需要传。

## 启用参数

主要参数定义在 `python/sglang/srt/server_args.py`：

```bash
--enable-slo-aware-prefill
--slo-prefill-ttft-slo-ms <float>
--slo-prefill-tpot-slo-ms <float>
--slo-prefill-ttft-stat <max|mean|p90>
--slo-prefill-tpot-stat <max|mean|p90>
--slo-prefill-yield-guard-ratio <float>
--slo-prefill-cache-hit-io-cost-ratio <float>
--disable-slo-prefill-online-cost-model
--disable-slo-prefill-startup-profiling
--slo-prefill-profile-decode-context-len <int>
--slo-prefill-profile-decode-context-lens <int...>
--slo-prefill-profile-decode-batch-sizes <int...>
```

示例：

```bash
sglang serve \
  --model-path /path/to/model \
  --tp 4 \
  --chunked-prefill-size 32768 \
  --enable-slo-aware-prefill \
  --slo-prefill-ttft-slo-ms 15000 \
  --slo-prefill-ttft-stat p90 \
  --slo-prefill-tpot-slo-ms 60 \
  --slo-prefill-tpot-stat mean
```

参数语义：

- `--chunked-prefill-size` 是主路径 prefill chunk，SLO controller 会直接使用 scheduler 当前 effective 值。
- DP attention 开启时，SGLang 会先把 `--chunked-prefill-size` 除以 `dp_size` 得到本地有效值；SLO controller 直接使用该本地值，不会再把 `--slo-prefill-min-chunk-size` 除以 `dp_size`。
- `--slo-prefill-ttft-stat` / `--slo-prefill-tpot-stat` 控制 pressure 聚合口径，可选 `max`、`mean`、`p90`。
- `--slo-prefill-yield-guard-ratio` 是 TTFT slack 安全垫，默认 `0.05`。
- `--slo-prefill-cache-hit-io-cost-ratio` 是 cache hit token 的 HiCache IO 成本倍率，默认 `0.0`，即默认不额外计入 cache-hit IO 成本，避免对 TTFT 过度保守；如需显式估计 HiCache host/storage hit IO，可传入一个较小正数。
- `--disable-slo-prefill-online-cost-model` 默认开启，即默认不做运行时 online cost model 更新；启动 profiling 或初始值仍可用于静态成本估计。如需允许后续 online 更新，可显式传入 `--no-disable-slo-prefill-online-cost-model`。
- 启动 cost profiling 默认开启；失败或不支持时回退到初始值或默认成本估计。
- `--slo-prefill-min-chunk-size` 是可选 fallback chunk，不建议作为标准化配置默认传入；未传入时默认等于 effective `chunked_prefill_size`，因此通常不会改变调度行为。

## 调度流程

每轮 prefill admission 前，scheduler 计算并同步一个 `SloAwarePrefillPressureState`：

```text
(ttft_pressure, tpot_pressure, has_decode_work,
 prefill_cost, decode_cost, decode_context_len,
 ttft_future_prefill_cost, ttft_future_miss_tokens,
 ttft_future_hit_tokens, ttft_future_io_cost,
 ttft_cache_hit_rate)
```

然后按下面流程决策：

```text
objective = choose_objective(ttft_pressure, tpot_pressure, has_decode_work)

if has_decode_work and can_yield_to_decode(ttft_pressure):
    run decode
elif objective == tpot:
    run prefill with chunk = effective_min_chunk_size
else:
    run prefill with chunk = chunked_prefill_size
```

如果存在正在进行的 chunked prefill request，controller 不会阻塞它继续 prefill；scheduler 仍会优先处理 `yield_to_decode` 返回路径，避免不同 forward path 冲突。

## Pressure 计算

### TTFT Pressure

waiting queue 中尚未 prefill 的请求、以及正在 chunked prefill 的请求都会参与 TTFT pressure 统计。对每个请求，先估计剩余未完成 prompt token 中的 miss / hit token：

```text
cache_hit_rate = request_known_hit_rate
future_miss_tokens = clamp(total_prompt_tokens * (1 - cache_hit_rate) - already_computed_miss_tokens, 0, remaining_tokens)
future_hit_tokens = remaining_tokens - future_miss_tokens
```

随后 controller 按当前 `chunked_prefill_size` 和 `prefill_max_requests` 模拟后续 prefill backlog：waiting/chunked 请求会被打包成一个或多个估计 batch；chunk budget 只按 miss token 消耗，cache-hit token 默认不占用 prefill chunk，也不会把请求拆成更多轮。

```text
batch_compute_cost = Cp(batch_future_miss_tokens, batch_size)
batch_io_cost = cache_hit_io_cost_ratio * Cp(batch_future_hit_tokens, batch_size) if HiCache is enabled and cache_hit_io_cost_ratio > 0 else 0
estimated_remaining_cost = sum(prior/current estimated batch costs until this request finishes prefill)
per_req_ttft_pressure = (prefill_wait_time + estimated_remaining_cost) / ttft_slo
ttft_pressure = aggregate(per_req_ttft_pressure, stat=max|mean|p90)
```

`future_miss_tokens` 表示预计还需要实际 prefill 计算的 token；`future_hit_tokens` 表示预计从 prefix cache / HiCache 命中的 token。普通 prefix-cache device hit 不计入 IO 近似成本；默认 `cache_hit_io_cost_ratio=0` 时，cache hit token 也不额外计入 future cost。只有 HiCache 开启且显式配置正数 ratio 时，cache hit token 才按 `cache_hit_io_cost_ratio * Cp(hit_tokens, batch_size)` 计入 IO 近似成本。

### TPOT Pressure

running decode 请求参与 TPOT pressure 统计：

```text
decode_gap = now - last_decode_finish_time
historical_avg_tpot = (last_decode_finish_time - prefill_finished_time) / decoded_tokens
per_req_tpot_pressure = max(decode_gap, historical_avg_tpot) / tpot_slo
tpot_pressure = aggregate(per_req_tpot_pressure, stat=max|mean|p90)
```

`decode_gap` 捕捉当前 decode 被 prefill 阻塞的实时风险；`historical_avg_tpot` 保留已观察到的慢 decode 信息。

### Pressure 聚合口径

`--slo-prefill-ttft-stat` 和 `--slo-prefill-tpot-stat` 都是在每个 scheduler iteration 内，对当前参与统计的请求集合做即时聚合，不是历史滑动窗口：

- `max`：取当前请求集合里的最大 pressure，最保守，也是默认值。
- `mean`：对当前请求集合里的 per-request pressure 做等权平均。
- `p90`：对当前请求集合排序后按 nearest-rank 取第 90 分位，即 `ceil(0.90 * n) - 1` 对应的元素。

## Objective 与 Decode Yield

`objective` 是 prefill chunk 选择目标，用于在本轮不能安全让 decode 插队时判断应该用 full `chunked_prefill_size` 保护 TTFT，还是用可选 fallback chunk 降低对 TPOT 的影响：

```text
if no active decode:
    objective = ttft
elif tpot_pressure >= 1 and ttft_pressure < 1:
    objective = tpot
elif ttft_pressure >= 1 and tpot_pressure < 1:
    objective = ttft
elif tpot_pressure > ttft_pressure + margin:
    objective = tpot
else:
    objective = ttft
```

`margin=0.10`，比较前 pressure 会 round 到两位小数，避免 TP ranks 因微小时间差选择不同 objective。

decode yield 不再依赖 `objective=tpot`。调度会先做 TTFT slack 检查：只要当前存在 decode work，且 TTFT slack 足够覆盖本轮 decode 插队成本，就可以让 decode 先跑；因此即使 `objective=ttft`，在 TTFT 距离 SLO 还有足够空间时也会消费 slack 保护 TPOT / output throughput。只有不能安全 yield decode 时，`objective` 才决定下一轮 prefill 使用 full `chunked_prefill_size` 还是 optional fallback chunk。默认不传 `--slo-prefill-min-chunk-size` 时，fallback chunk 等于 effective `chunked_prefill_size`。

```text
can_yield_to_decode =
    ttft_pressure < 1
    and (1 - ttft_pressure) * ttft_slo
        > Cd(batch, kv_len_bucket) + guard
```

```text
guard = yield_guard_ratio * ttft_slo
```

`ttft_pressure` 已经包含 waiting/chunked 请求完成 prefill 前的 future prefill cost，因此 decode yield 判断只额外检查本轮 decode 插队新增的 `Cd` 和固定比例 `guard`，不再重复叠加 `Cp(min_chunk, 1)`。`guard` 只保留基于 TTFT SLO 的固定比例安全垫，避免再引入和 `Cd` / `Cp(min_chunk, 1)` 相关的额外经验项。

- `can_yield_to_decode=True`：本轮跑 decode，消费一部分 TTFT slack 保护 TPOT / output throughput。
- `can_yield_to_decode=False && objective=tpot`：本轮仍跑 prefill，使用 effective min chunk；默认等于 `chunked_prefill_size`，只有显式配置 `--slo-prefill-min-chunk-size` 时才会缩小。
- `can_yield_to_decode=False && objective=ttft`：本轮跑 prefill，并使用 full `chunked_prefill_size`。

## Cost Profiling

SLO controller 使用两类成本估计：

```text
Cp(tokens, batch_size)     # prefill batch token 数 + request batch size -> prefill forward latency
Cd(context_len, batch_size)  # decode KV length bucket + batch size -> decode forward latency
```

启动 profiling 成功后写入 controller：

```text
controller.set_startup_cost_profile(
  prefill_cost_by_batch_ms=[(prefill_tokens, batch_size, latency_ms), ...],
  decode_cost_by_context_ms=[(context_len, batch_size, latency_ms), ...],
)
```

当前 Cp 会在启动时读取 effective `chunked_prefill_size`，按每请求 1024-token 步长和不同 request batch size 自动构造 synthetic prefill batch 采样；例如 `chunked_prefill_size=8192` 会采样 `1x1024 ... 1x8192`、`2x1024 ... 2x4096`、`4x1024 ... 4x2048`、`8x1024` 等形状。不额外暴露 prefill profile 参数；运行时按 `tokens + batch_size` 插值/外推。Cd 按 `context_len + batch_size` 二维建模，每个 Cd 点先做 1 次 warmup，再采样 10 次 decode forward 取均值。

startup profiling 是 best effort：非 generation、disaggregation、pipeline parallelism 等场景会跳过；speculative decoding 会使用对应的 spec decode profiling 路径；任一采样失败只丢弃该样本并回退初始值或默认成本估计。

## TP / DP / HiCache 兼容性

- TP/DP attention 场景同步的是 pressure/cost 输入，而不是 Python decision 对象；同步后各 rank 基于相同输入本地计算相同 decision。
- DP attention 场景会覆盖 attention TP / CP 相关 group，避免不同 rank 进入不同 forward path。
- SLO controller 不重排 waiting queue，保留原有 schedule policy / prefix-cache locality 行为。
- startup profiling 使用 synthetic request，并跳过 radix/cache insert，避免污染线上 prefix/HiCache 状态。

## 日志

启动成功会看到：

```text
SLO-aware prefill enabled: ... startup_profiling=True, online_cost_model=False ...
SLO prefill startup cost profile: Cp(ms)=[(tokens, [(batch_size, latency_ms), ...]), ...], Cd_mean(ms)=[...], Cd_warmup=1, Cd_samples=10
```

运行时关键日志：

```text
SLO prefill decision: objective=..., allow=..., yield_to_decode=..., has_decode=..., chunk=..., prefill_max_requests=..., ttft_pressure=..., tpot_pressure=..., prefill_cost_ms_per_1k=..., decode_cost_ms=..., decode_context_len=..., ttft_future_cost_ms=..., ttft_future_miss_tokens=..., ttft_future_hit_tokens=..., ttft_future_io_cost_ms=..., cache_hit_io_cost_ratio=..., cache_hit_io_cost_enabled=..., ttft_cache_hit_rate=..., ttft_slack_ms=..., yield_rhs_ms=..., yield_guard_ms=..., min_prefill_cost_ms=..., waiting=..., running=...
```

其中：

- `objective` 表示当前保护目标。
- `yield_to_decode=True` 表示本轮可让 decode 插队。
- `chunk` 表示本轮 prefill 被允许时使用的 chunk。
- `decode_cost_ms` / `decode_context_len` 表示当前 decode batch 的 Cd 估计输入。
- `ttft_future_*` 表示 TTFT pressure 中计入的未来 prefill 计算与 cache-hit IO 估计。
- `ttft_slack_ms` 与 `yield_rhs_ms` 是 decode yield 判断的左右两边。

## 验证

单测：

```bash
python3 test/registered/unit/managers/test_slo_aware_prefill.py
```

基础编译检查：

```bash
python3 -m py_compile \
  python/sglang/srt/managers/slo_aware_prefill.py \
  python/sglang/srt/managers/scheduler.py \
  python/sglang/srt/server_args.py
```
