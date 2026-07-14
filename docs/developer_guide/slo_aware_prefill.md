# SLO-Aware Prefill Scheduling 设计文档

本文档描述当前分支中的 **SLO-aware prefill scheduling**。该特性保留 SGLang 原有连续 batching / chunked prefill 框架，在每个调度 iteration 根据 TTFT 与 TPOT 压力动态决定：本轮优先 prefill 还是 decode、prefill chunk 应该多大、是否需要让 decode 插队。

实现是 SOLA-inspired 的状态机近似，而不是完整论文 solver。核心目标是：在满足 decode/TPOT 约束的前提下尽量消化 prefill；当 TTFT 已经逼近 SLO 时，及时恢复 prefill 能力，避免长 prompt 饥饿。

## 启用参数

主要参数定义在 `python/sglang/srt/server_args.py`：

```bash
--enable-slo-aware-prefill
--slo-prefill-ttft-slo-ms <float>
--slo-prefill-tpot-slo-ms <float>
--slo-prefill-ttft-stat <max|mean|p90>
--slo-prefill-tpot-stat <max|mean|p90>
--slo-prefill-min-chunk-size <int>
--slo-prefill-tile-size <int>
--slo-prefill-yield-guard-ratio <float>
--disable-slo-prefill-online-cost-model
--disable-slo-prefill-startup-profiling
--slo-prefill-profile-prefill-step-size <int>
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
  --slo-prefill-tpot-stat mean \
  --slo-prefill-min-chunk-size 4096 \
  --slo-prefill-tile-size 128
```

说明：

- `--chunked-prefill-size` 仍作为当前动态 chunk 的静态上限；controller 在 `[min_chunk, chunked_prefill_size]` 内缩放。
- DP attention 开启时，SGLang 会把 `--chunked-prefill-size` 除以 `dp_size` 转成本地上限；SLO controller 也会把显式传入的 `--slo-prefill-min-chunk-size` 除以 `dp_size`，保持二者同一语义。例如全局 `chunked_prefill_size=32768`、`min_chunk_size=4096`、`dp_size=8` 时，本地范围是 `[512, 4096]`。
- 如果没有显式传入 `--slo-prefill-min-chunk-size`，默认最小 chunk 仍使用本地 `--slo-prefill-tile-size`，不会再除以 `dp_size`。
- `--slo-prefill-ttft-stat` / `--slo-prefill-tpot-stat` 控制 pressure 口径，可用 `p90` 或 `mean` 对齐压测 SLO。
- `--slo-prefill-yield-guard-ratio` 是 TTFT slack 安全垫，默认 `0.05`，表示至少保留 `5% * TTFT_SLO` 的额外余量。
- 启动 cost profiling 默认开启；如遇到不支持场景或 profile 失败，会自动回退到初始值 + 在线 EMA。
- 长上下文 decode 场景建议显式传入多个 Cd context bucket，例如 `--slo-prefill-profile-decode-context-lens 4096 8192 16384 32768`；如果不传该列表，则保持兼容，只使用 `--slo-prefill-profile-decode-context-len` 的单个长度。

## 状态机

每轮 prefill admission 前，scheduler 会计算并同步一个 `SloAwarePrefillPressureState`：

```text
(ttft_pressure, tpot_pressure, has_decode_work, prefill_cost, decode_cost, decode_context_len, ttft_remaining_prefill_cost)
```

### TTFT Pressure

```text
remaining_prefill_cost = Cp(remaining_prompt_tokens)
per_req_ttft_pressure = (prefill_wait_time + remaining_prefill_cost) / ttft_slo
ttft_pressure = aggregate(per_req_ttft_pressure, stat=max|mean|p90)
```

waiting queue 中尚未 prefill 的请求以及正在 chunked prefill 的请求都会参与统计。这里不是只看已经等待多久，而是把预计还要花在 prefill 上的时间也算进去；否则长 prompt 会等到接近 TTFT SLO 时才切回 prefill，最终 p90 TTFT 很容易超标。

### TPOT Pressure

```text
decode_gap = now - last_decode_finish_time
historical_avg_tpot = (last_decode_finish_time - prefill_finished_time) / decoded_tokens
per_req_tpot_pressure = max(decode_gap, historical_avg_tpot) / tpot_slo
tpot_pressure = aggregate(per_req_tpot_pressure, stat=max|mean|p90)
```

`decode_gap` 用于捕捉当前 decode 被 prefill 阻塞的实时风险；`historical_avg_tpot` 用于保留已观察到的慢 decode 信息。

### Objective 切换

状态机只在两个目标间切换：

```text
objective=ttft  # 优先 prefill，保护首 token 延迟
objective=tpot  # 优先 decode，保护输出节奏
```

规则：

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

## 动态 Yield 公式

旧实现使用固定阈值：

```text
ttft_pressure < hard_yield_ttft_pressure
```

这对不同硬件不够自适应。当前实现改为 cost-based slack 判断：

```text
can_yield_to_decode =
    (1 - ttft_pressure) * ttft_slo
    > Cd(batch, kv_len_bucket) + Cp(min_chunk) + guard
```

含义：

- `(1 - ttft_pressure) * ttft_slo`：当前 TTFT 余量；这里的 `ttft_pressure` 已包含预计剩余 prefill 成本。
- `Cd(batch, kv_len_bucket)`：按当前 decode batch size 和最大运行中 decode 序列长度估计的一次 decode forward 开销。
- `Cp(min_chunk)`：至少推进一个最小 prefill chunk 的开销。
- `guard`：安全垫，吸收采样误差、queue 抖动和同步开销。

当前 guard：

```text
guard = max(
  2 * Cd(batch, kv_len_bucket),
  0.2 * Cp(min_chunk),
  yield_guard_ratio * ttft_slo,
)
```

当 `objective=tpot` 且公式成立时，scheduler 返回 `None`，让本轮转去 decode。否则说明 TTFT slack 不足以支撑 “decode forward + 最小 prefill forward”，本轮必须先跑 prefill；此时 chunk 大小不再用固定比例，而是进入 TPOT 约束下的 token budget 求解。

## 启动 Cost Profiling

SLO controller 需要两个成本表：

```text
Cp(tokens)                  # prefill chunk tokens -> prefill forward latency
Cd(context_len, batch_size)  # decode KV length bucket + batch size -> decode forward latency
```

这些表会在服务启动阶段自动在线估计，不需要手动脚本。

### Cp 表

启动时按 `--slo-prefill-profile-prefill-step-size` 间隔采样：

```text
2048, 4096, 6144, ..., chunked_prefill_size
```

如果 `chunked_prefill_size=32668` 且 step 为 `2048`，最后会额外包含 `32668` 这个上限点。每个点构造一个 synthetic prefill request，走真实 `ScheduleBatch.prepare_for_extend()` 和模型 forward，因此能覆盖当前模型、attention backend、KV allocator、page size、SWA 等实际开销。

### Cd 表

Cd 现在按 `context_len + batch_size` 二维建模。`context_len` 表示 synthetic request 预填充后的 KV 长度桶，用来避免只在短上下文采样导致长上下文 decode 开销被低估。

decode batch size 默认来自已配置/捕获的 decode CUDA graph batch sizes：

```text
server_args.cuda_graph_config.decode.bs
```

如果该字段为空，则按 `ServerArgs._generate_decode_cuda_graph_batch_sizes(max_bs)` 生成。也可以通过 `--slo-prefill-profile-decode-batch-sizes` 显式指定。

`context_len` 默认使用兼容参数 `--slo-prefill-profile-decode-context-len` 的单个值；如果设置了 `--slo-prefill-profile-decode-context-lens`，则按该列表采样多个 KV 长度桶。

每个 Cd 点会构造 `batch_size` 个 synthetic request，先填充到对应 `context_len`，再执行一次真实 decode forward 并记录耗时。超过 KV/token pool 或 req slot 能力的 batch size 会在采样前过滤，避免启动阶段为了 profile 申请超过本 rank 可承载的 synthetic requests。

### 使用方式

启动 profile 成功后会写入 controller：

```text
controller.set_startup_cost_profile(
  prefill_cost_ms=[(tokens, latency_ms), ...],
  decode_cost_by_context_ms=[(context_len, batch_size, latency_ms), ...],
)
```

运行时：

- `Cp(x)` 使用表内线性插值，超出范围时按末端斜率外推。
- `Cd(context_len, batch_size)` 先按当前 decode 请求最大 `seqlen` 选择/插值 context bucket，再在该 bucket 内按 batch size 插值；超过最大 context bucket 时按末端非负斜率外推。
- 在线 EMA 仍会继续更新低维 fallback cost；启动表失败或缺失时使用 fallback。

### Best Effort 与跳过条件

启动 profiling 不应阻塞服务可用性，因此是 best effort：

- 非 generation 模型跳过。
- disaggregation prefill/decode 模式跳过。
- pipeline parallelism 暂跳过。
- speculative decoding 暂跳过，回退在线 EMA。
- 任一点采样失败只丢弃该点并打印 warning。
- 每个 synthetic request 在采样后直接释放 KV、Mamba sidecar 和 req slot，不走 prefix cache insert/finished path，避免启动 profiling 污染或耗尽线上 pool。
- 所有点失败时回退默认/在线成本模型。

## Workload 控制

### `objective=ttft`

TTFT 优先时尽量扩大 prefill：

```text
if no decode work:
    chunk = base_chunk
elif TPOT slack 足够:
    chunk = floor_to_tile(TPOT_slack / Cp_per_token)
else:
    chunk = min_chunk
```

当 TTFT pressure 已经很高且 TPOT 仍有余量时，会恢复大 chunk，防止 prefill throughput 崩掉。

### `objective=tpot`

TPOT 优先时先判断能否让 decode 插队：

```text
prefill_max_requests = 1
if can_yield_to_decode:
    run decode first
```

如果不能 yield，则本轮需要先跑 prefill。此时 chunk 使用 `Cp` 表反解，而不是固定的 `base_chunk * 0.25 / 0.5`：

```text
tpot_slack = (1 - tpot_pressure) * tpot_slo
prefill_budget_time = tpot_slack - Cd(batch, kv_len_bucket) - prefill_guard
chunk = Cp^-1(prefill_budget_time)
chunk = clamp_and_floor_to_tile(chunk, min_chunk, base_chunk)
```

其中：

```text
prefill_guard = max(
  0.5 * Cd(batch, kv_len_bucket),
  0.2 * Cp(min_chunk),
  yield_guard_ratio * tpot_slo,
)
```

这表示：如果 TPOT 仍有剩余 slack，就把这部分 slack 转换为下一轮允许的 prefill token 数 `p`。例如 `Cp^-1(45ms)=384 tokens`，则本轮 prefill chunk 就会被压到 `384` 附近并按 tile 对齐。

如果 `prefill_budget_time <= 0`，则说明在当前 TPOT 约束下已经没有可行 prefill 预算，但 `can_yield_to_decode=False` 又说明 TTFT 也不能继续等待，此时进入无解/冲突分支：

```text
if ttft_pressure < 1:
    chunk = min_chunk
elif ttft_pressure < hard_prefill_ttft_pressure:
    chunk = Cp^-1((ttft_pressure - 1) * ttft_slo + Cp(min_chunk))
else:
    chunk = base_chunk
```

也就是说，TTFT 尚未违约时只跑最小 chunk，尽快把机会还给 decode；TTFT 已经违约时按 TTFT debt 放大 chunk；TTFT 严重违约时恢复完整 chunk，避免 prefill starvation。

## TP / DP / HiCache 兼容性

### TP 一致性

调度前同步的是 pressure/cost 输入，而不是 Python decision 对象：

```text
local PressureState
  -> all_reduce(MAX)
  -> global PressureState
  -> all ranks locally compute identical decision
```

普通 TP 在 `tp_group` 内同步；DP attention 场景会额外覆盖 attention TP / CP 相关 group，避免不同 rank 进入不同 forward path。

### DP Attention

启动 profiling 的 synthetic batch 会复用 `SchedulerDPAttnAdapter` 的 MLP sync 逻辑；当需要 MLP TP/DP gather 时，采样 forward 会和普通调度一样填充 `global_num_tokens`、`can_run_dp_cuda_graph` 等字段。

### HiCache

SLO waiting queue 排序后会做稳定的 HiCache prefetch-ready 分组：已完成 prefetch 或无需 prefetch 的请求排在前面，仍在 storage prefetch 的请求稳定后移。启动 profiling 使用 synthetic request，并跳过 radix/cache insert，避免污染线上 prefix/HiCache 状态。

## 日志

启动成功会看到：

```text
SLO-aware prefill enabled: ... startup_profiling=True ...
SLO prefill startup cost profile: Cp(ms)=[...], Cd(ms)=[...]
```

运行时关键日志：

```text
SLO prefill decision: objective=..., allow=..., yield_to_decode=..., has_decode=..., chunk=..., prefill_max_requests=..., ttft_pressure=..., tpot_pressure=..., prefill_cost_ms_per_1k=..., decode_cost_ms=..., decode_context_len=..., ttft_remaining_prefill_cost_ms=..., ttft_slack_ms=..., yield_rhs_ms=..., yield_guard_ms=..., min_prefill_cost_ms=..., waiting=..., running=...
```

其中：

- `yield_to_decode=True` 表示本轮可让 decode 插队。
- `allow=True, yield_to_decode=True` 可能同时出现，通常表示存在 chunked request，但 scheduler 会优先按 `yield_to_decode` 让 decode 先跑。
- `prefill_cost_ms_per_1k` 是当前低维 fallback cost；真实 chunk 预算优先使用启动 Cp 表。
- `decode_cost_ms` 是按当前 decode batch size 和 `decode_context_len` 估计/同步后的 Cd。
- `decode_context_len` 是当前 running decode 请求的最大 `seqlen`，DP/TP 同步后取最大值。
- `ttft_remaining_prefill_cost_ms` 是按 Cp 表估计的剩余 prefill 时间，已经计入 `ttft_pressure`。
- `ttft_slack_ms` 与 `yield_rhs_ms` 分别是 yield 公式左右两边，便于判断为什么本轮让 decode 插队或继续 prefill。

## 已知限制

- 当前仍是状态机 + closed-form budget，不是完整 SOLA constrained solver。
- 启动 profiling 暂不覆盖 speculative decoding / PP / disaggregation。
- `chunked_prefill_size` 仍是静态上限；后续可改为基于当前 KV/token pool 峰值预测动态求 `memory_cap_chunk`。
- `Cp/Cd` 表已覆盖 prefill token 数、decode batch size 和 decode KV 长度桶；尚未区分 prefix cache hit、MoE routing、batch 内序列长度分布等特征。
- Percentile SLO 目前通过 pressure 聚合口径近似，没有实现完整 percentile-level relaxation。

## 验证

单测：

```bash
python3 test/registered/unit/managers/test_slo_aware_prefill.py
```

建议基础校验：

```bash
python3 -m py_compile \
  python/sglang/srt/managers/slo_aware_prefill.py \
  python/sglang/srt/managers/scheduler.py \
  python/sglang/srt/server_args.py
```
