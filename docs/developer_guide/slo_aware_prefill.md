# SLO-Aware Prefill Scheduling 设计文档

本文档描述当前分支中新增的 **SLO-aware prefill scheduling** 特性。该特性受 SOLA 论文启发，目标是在 SGLang 的现有 scheduler 架构内，通过动态控制 prefill admission、chunk size、prefill 请求数量和阶段优先级，在 TTFT 与 TPOT 之间做显式 tradeoff。

当前实现是 **SOLA-style incremental implementation**，不是完整 SOLA 论文实现。它优先解决单机 TP 场景下 prefill 与 decode 互相干扰的问题，并为后续补齐 cost model、peak memory prediction、request-level constrained optimization 预留结构。

## 背景

LLM serving 中每个请求主要包含两个阶段：

- **Prefill**：处理 prompt，生成首 token。主要影响 TTFT。
- **Decode**：逐 token 生成输出。主要影响 TPOT。

SGLang 默认调度策略在高负载下可能出现两类偏置：

- Prefill 过多：TTFT 较好，但 decode 被长 prefill chunk 阻塞，TPOT 恶化。
- Decode 过多：TPOT 较好，但 prefill 请求长期排队，TTFT 恶化。

SOLA 的核心思想是：每个 iteration 都根据当前系统状态决定调度目标和 workload，而不是固定 prefill-first 或 decode-first。当前实现将该思想落到 SGLang 的 prefill scheduling path 上。

## 用户参数

新增参数定义在 `python/sglang/srt/server_args.py`：

```bash
--enable-slo-aware-prefill
--slo-prefill-ttft-slo-ms <float>
--slo-prefill-tpot-slo-ms <float>
--slo-prefill-tile-size <int>
--slo-prefill-min-chunk-size <int>
--disable-slo-prefill-priority-boost
```

常用启动示例：

```bash
sglang serve \
  --model-path /path/to/model \
  --tp 4 \
  --enable-metrics \
  --chunked-prefill-size 8192 \
  --enable-slo-aware-prefill \
  --slo-prefill-ttft-slo-ms 2000 \
  --slo-prefill-tpot-slo-ms 50 \
  --slo-prefill-min-chunk-size 512 \
  --slo-prefill-tile-size 128
```

注意：

- `--enable-slo-aware-prefill` 只开启控制器。
- 当前实现中，`--chunked-prefill-size` 仍然需要显式设置，否则不会自动启用 chunked prefill。
- 当前实现把 `--chunked-prefill-size` 视为静态 `base_chunk` / 安全上限，SLO controller 只会在 `[slo_min_chunk, base_chunk]` 内动态缩放。
- 更符合 SOLA 的后续设计是：开启 SLO-aware 后不再把用户传入的 `--chunked-prefill-size` 当最终上限，而是由 peak memory prediction 计算当前 iteration 的 `memory_cap_chunk`，再用 `min(user_cap, memory_cap_chunk)` 或直接 `memory_cap_chunk` 作为动态上限。
- `--slo-prefill-ttft-slo-ms` 和 `--slo-prefill-tpot-slo-ms` 是必填目标值。
- `--slo-prefill-min-chunk-size` 决定 TPOT 保护模式下的最小 prefill chunk。

## 总体架构

主要改动文件：

- `python/sglang/srt/managers/slo_aware_prefill.py`
  - 定义 `SloAwarePrefillController`。
  - 根据 TTFT/TPOT pressure 生成 `SloAwarePrefillDecision`。
  - 暴露 `SloAwarePrefillPressureState`，支持 scheduler 先同步 pressure 再做决策。
- `python/sglang/srt/managers/scheduler.py`
  - 在 scheduler 初始化时创建 controller。
  - 在 `_get_new_batch_prefill_raw` 中调用 controller。
  - 根据 controller 决策调整 `chunked_prefill_size`、`prefill_max_requests`、是否 yield 给 decode。
- `python/sglang/srt/server_args.py`
  - 增加 CLI 参数和参数校验。
- `test/registered/unit/managers/test_slo_aware_prefill.py`
  - 覆盖 TPOT 保护、TTFT 优先、高并发边界、TP rank 确定性等行为。

调度链路如下：

```text
ServerArgs
  -> Scheduler.init_schedule_policy()
      -> SloAwarePrefillController(...)
  -> Scheduler._get_new_batch_prefill_raw()
      -> controller.compute_pressure_state(...)
      -> TP group all_reduce(MAX) syncs pressure state
      -> controller.make_decision_from_pressure_state(...)
      -> possibly return None to run decode first
      -> else adjust PrefillAdder inputs
          - chunked_prefill_size
          - prefill_max_requests
          - waiting_queue order
```

## 核心状态

### TTFT Pressure

TTFT pressure 表示当前 prefill 请求距离 TTFT SLO 的压力。

当前近似实现：

```text
ttft_pressure = max_prefill_wait_time / ttft_slo
```

其中 `max_prefill_wait_time` 来自：

- waiting queue 中所有尚未 prefill 的请求；
- 当前正在 chunked prefill 中的 `chunked_req`。

如果：

```text
--slo-prefill-ttft-slo-ms 2000
```

且日志中：

```text
ttft_pressure=0.5
```

则表示当前最急 prefill 请求已经等待约 `1000ms`。

### TPOT Pressure

TPOT pressure 表示当前 decode 请求距离 TPOT SLO 的压力。

当前近似实现综合两部分：

```text
decode_gap = now - last_decode_finish_time
historical_avg_tpot = (last_decode_finish_time - prefill_finished_time) / decoded_tokens
tpot_pressure = max(decode_gap, historical_avg_tpot) / tpot_slo
```

这样做的原因是：

- 只看 historical average TPOT 会反应太慢；
- 短输出请求只有几个 decode token，很可能请求结束前 controller 还没感知到 TPOT 变差；
- `decode_gap` 能捕捉当前 decode 被 prefill 阻塞的实时风险。

如果：

```text
--slo-prefill-tpot-slo-ms 50
```

且日志中：

```text
tpot_pressure=0.4
```

则表示当前最坏 decode gap 或平均 TPOT 约为 `20ms`。

### EMA Pressure

controller 还维护：

```text
ttft_ema
tpot_ema
```

EMA 当前只用于日志观察，不再直接决定调度目标。

早期实现曾使用 EMA + 上一次 objective 做 hysteresis，但 TP 多进程下不同 rank 的 `perf_counter()` 和 pressure 可能存在极小差异，导致不同 TP rank 在同一 iteration 选择不同 objective，进而造成 distributed forward hang。当前版本已经移除该粘滞逻辑，改为确定性目标选择。

## Objective 选择

当前 controller 有两个 objective：

```text
objective=ttft
objective=tpot
```

含义：

- `ttft`：优先 prefill，减少 waiting queue 中请求的 TTFT。
- `tpot`：优先 decode，保护 running 请求的 TPOT。

选择规则：

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

其中：

```text
margin = 0.10
```

并且 pressure 会 round 到两位小数后再比较，避免 TP ranks 因浮点微小差异产生不同决策。

这种规则对应 SOLA 的高层思想：

- TPOT 违反而 TTFT 有余量：优化 TPOT，约束 TTFT。
- TTFT 违反而 TPOT 有余量：优化 TTFT，约束 TPOT。
- 两者都未违反：优化更接近 SLO 边界的一侧。
- 模糊区域默认回到 TTFT，避免 prefill 长期饥饿和 TP rank 分歧。

## Workload 控制

### Online Cost Model

controller 维护轻量在线成本模型，对应 SOLA 论文中的 `Cp` / `Cd`：

```text
Cp ~= prefill_cost_per_token * prefill_tokens
Cd ~= decode_cost_per_iteration
```

当前实现使用 EMA 更新：

- 纯 prefill batch：用 prefill 日志间隔和 `#new-token` 更新 `prefill_cost_per_token`。
- decode batch：用 decode 日志间隔除以 `decode_log_interval` 更新 `decode_cost_per_iteration`。
- mixed prefill batch 暂不用于拟合 `Cp`，避免被中间 decode 间隔污染。
- TP group 同步 pressure 时也同步 cost 标量，确保各 TP rank 使用同一组预测输入。

### Dynamic Chunk Size

controller 根据 SOLA Eq. 1 / Eq. 2 的约束思想动态缩放 `chunked_prefill_size`。

当前实现中：

```text
base_chunk = explicit --chunked-prefill-size
chunk_upper_bound = base_chunk
```

这保证不会超过用户显式配置，但也意味着如果用户传入的 `--chunked-prefill-size` 偏小，SLO controller 无法利用剩余 KV/显存空间扩大 prefill workload；如果用户传入值偏大，则仍主要依赖 SGLang 原有 admission/allocator 来兜底。

更接近 SOLA 的设计应该把 `base_chunk` 替换为每轮动态上限：

```text
memory_cap_chunk = predict_max_chunk_from_peak_kv_memory(current_state)
chunk_upper_bound = memory_cap_chunk
# 或兼容模式：chunk_upper_bound = min(user_chunked_prefill_size, memory_cap_chunk)
```

其中 `memory_cap_chunk` 应在考虑 running decode、waiting prefill、page size、KV cache watermark、SWA/full attention token pool 后，预测本轮不触发 allocator failure / retraction / preemption 的最大 chunk。

简化规则：

```text
no active decode:
    chunk = chunk_upper_bound

objective=ttft:
    tpot_slack = (1 - tpot_pressure) * tpot_slo
    chunk = floor_to_tile(tpot_slack / prefill_cost_per_token)
    if TTFT has already violated and TPOT has large slack:
        chunk = chunk_upper_bound

objective=tpot:
    if TTFT pressure >= hard_prefill_ttft_pressure:
        chunk = chunk_upper_bound
    elif TTFT has enough slack for one decode iteration + min prefill chunk:
        yield_to_decode=True
    else:
        chunk = the minimum prefill budget needed to keep TTFT moving
```

最终 chunk 会被约束到：

```text
min_chunk_size <= chunk <= chunk_upper_bound <= max_prefill_tokens
```

当前 `chunk_upper_bound == base_chunk`；引入 peak memory prediction 后，`chunk_upper_bound` 应该变成当前 iteration 的内存安全上限。

并按 `lcm(page_size, tile_size)` 向下对齐。

### Prefill Request Count

在 `objective=tpot` 时，controller 会限制：

```text
prefill_max_requests = 1
```

这表示即使 TPOT 更紧，也保留一个受限 prefill 通道，避免长 prompt 在高并发下被 decode 长时间饿死。当 TTFT pressure 达到 `hard_prefill_ttft_pressure`（当前为 1.5）后，即使 objective 仍为 `tpot`，chunk 也会恢复到 `chunk_upper_bound`，防止高并发下 prefill throughput 崩掉。

在 `objective=ttft` 时，通常保留用户原始 `prefill_max_requests`，使高并发下 TTFT 能恢复。

如果 TPOT pressure 非常高，并且 TTFT pressure 仍然低于 TPOT pressure，则即使在 `objective=ttft` 下也会临时限制 prefill 请求数。

### Yield To Decode

当满足：

```text
objective=tpot
has_decode_work=True
ttft_pressure < hard_yield_ttft_pressure  # 当前为 0.50
```

controller 设置：

```text
yield_to_decode=True
```

scheduler 收到该决策后返回 `None`，从而让 `get_next_batch_to_run()` 走 decode path。

`hard_yield_ttft_pressure` 是 TTFT slack 保护阈值，不是 TPOT objective 切换阈值。TPOT 低于 SLO 但相对更紧时仍可切到 `objective=tpot`，只是不会在 TTFT 已经接近 SLO 时完全停止 prefill。

这解决了两个关键问题：已有 `chunked_req` 不应该连续占据所有 iteration，否则 decode token 会被多个 prefill chunk 阻塞；同时长 prompt 也不能在高并发下被 decode 长时间饿死。

## `allow_prefill` 与 `yield_to_decode`

日志中可能出现：

```text
allow=True, yield_to_decode=True
```

这不是矛盾。

原因是：

- 如果当前有 `chunked_req`，controller 会将 `allow_prefill=True`，避免把 chunked request 当成普通 prefill drop。
- 但 scheduler 中 `yield_to_decode` 优先级更高。

实际执行逻辑是：

```text
if yield_to_decode and chunked_req exists:
    return None  # run decode first
elif not allow_prefill:
    return None
else:
    run prefill
```

因此 `allow=True, yield_to_decode=True` 的含义是：

```text
当前有 chunked prefill，但本 iteration 仍然让 decode 先跑。
```

## Waiting Queue 排序

controller 支持 request-level priority boost。

当 `objective=ttft`：

- prefill 请求优先；
- 按预测 TTFT 降序排序；
- 预测值近似为：

```text
wait_time + remaining_input_tokens / max_prefill_tokens
```

当 `objective=tpot`：

- waiting queue 中若存在 decode/retracted 请求，则 decode 优先；
- decode 请求按当前 request-level TPOT pressure 排序。

如果需要关闭该排序：

```bash
--disable-slo-prefill-priority-boost
```

## TP Rank 一致性

TP=4 场景中，所有 TP ranks 必须在同一 iteration 做出一致的调度决策。

曾经出现的问题：

```text
TP0 objective=tpot, yield_to_decode=True
TP1 objective=tpot, yield_to_decode=True
TP2 objective=tpot, yield_to_decode=True
TP3 objective=ttft, yield_to_decode=False
```

这种分歧会让不同 rank 进入不同 forward path，一部分 rank 跑 decode，另一部分 rank 跑 prefill，容易在 collective 或 CUDA graph 中 hang，且后台不一定报错。

当前修复不是给 TPOT 增加固定切换阈值，也不是广播完整 Python decision object，而是同步 objective 的输入状态：

```text
local PressureState = (ttft_pressure, tpot_pressure, has_decode_work)
global PressureState = TP all_reduce(MAX, local PressureState)
all ranks compute SloAwarePrefillDecision from global PressureState
```

这样做有几个好处：

- 保留自适应：即使 TPOT 尚未达到 SLO，只要它相对 TTFT 更紧，仍然可以切到 `objective=tpot`。
- 避免 rank 分歧：objective、chunk、yield、prefill request cap 都来自同一组 global pressure。
- 避免 Python object broadcast：同步的是 3 个 float/bool 标量，当前使用 CPU tensor `all_reduce(MAX)`，不触碰 GPU stream。
- 采用保守聚合：任一 TP rank 看到更高 TTFT/TPOT pressure，全组都按更高压力处理。

同时仍保留：

- objective 不依赖上一轮状态；
- pressure 比较前 round 到两位小数；
- 模糊区确定性选择 `ttft`；
- 单测覆盖 sticky `tpot` 和 pre-SLO TPOT 自适应切换问题。

## 日志与调试

启动建议：

```bash
PYTHONFAULTHANDLER=1 \
SGLANG_LOG_MS=1 \
SGLANG_LOG_FORWARD_ITERS=1 \
SGLANG_RECORD_STEP_TIME=1 \
SGLANG_LOG_SCHEDULER_STATUS_TARGET=stdout \
SGLANG_LOG_SCHEDULER_STATUS_INTERVAL=5 \
sglang serve \
  ... \
  --log-level debug \
  --decode-log-interval 1 \
  --enable-request-time-stats-logging
```

关键日志：

```text
SLO-aware prefill enabled: ...
SLO prefill decision: objective=..., allow=..., yield_to_decode=..., chunk=..., ttft_pressure=..., tpot_pressure=...
scheduler.status ...
Prefill batch ...
Decode batch ...
```

排查命令：

```bash
grep -E "SLO-aware prefill enabled|SLO prefill decision|scheduler.status|Prefill batch|Decode batch" /tmp/sglang-sola-debug.log | tail -n 300
```

如果压测停止但服务仍有 `/metrics`：

```bash
curl -s http://127.0.0.1:30000/v1/loads | jq
curl -s http://127.0.0.1:30000/metrics | grep -E "num_running_reqs|num_queue_reqs|token_usage"
```

## 典型行为

### 低并发 / 无 decode

```text
objective=ttft
chunk=chunk_upper_bound
```

目标是快速消化 prefill。

### 中并发 / TPOT 接近 SLO

```text
objective=tpot
prefill_max_requests=1
chunk=min_chunk / 0.25x / 0.5x / 0.75x by TTFT slack
yield_to_decode=True only when TTFT has enough slack
```

目标是让 decode 插队，降低 TPOT p90/p99。

### 高并发 / TTFT 爆炸但 TPOT 正常

```text
objective=ttft
chunk=chunk_upper_bound
prefill_max_requests=None
```

目标是避免过度保护 TPOT 造成 prefill starvation；如果 TTFT pressure 继续升高到 `hard_prefill_ttft_pressure`，即使 TPOT 更紧也会临时恢复大 chunk。

### 两者都超 SLO

当前使用 pressure 大小做近似切换：

- TPOT pressure 明显更大：`objective=tpot`。
- 否则：`objective=ttft`。

完整 SOLA 中应该进一步引入 percentile-level constraint relaxation。

## 与 SOLA 论文的对应关系

当前实现已经覆盖 SOLA 的部分设计点：

| SOLA 机制 | 当前实现状态 |
| --- | --- |
| Fine-grained iteration-level scheduling | 通过 prefill admission/yield 控制实现 |
| Phase-level prioritization | `objective=ttft/tpot` 控制 prefill/decode 优先级 |
| Workload size control | 动态 `chunked_prefill_size` 与 `prefill_max_requests` |
| Request-level prioritization | waiting queue urgency sort |
| State monitor | TTFT/TPOT pressure + EMA 日志 |
| Constraint conversion | 根据 TTFT/TPOT pressure 切换 objective |
| Cost model `Cp/Cd` | 已实现轻量在线 EMA 版本，用于 chunk/yield 约束 |
| Constrained workload `ki/ni` | 已实现近似 Eq. 1 / Eq. 2 的 token budget 与 prefill cap |
| Peak memory prediction | 尚未实现；下一步应将其作为 `chunk_upper_bound` 来源 |
| Output length prediction | 已实现 request-level fallback：基于 `max_new_tokens - generated` |
| Percentile-level relaxation | 尚未实现 |

## 已知限制

1. **不是完整 constrained optimization**
   - 当前不是求解 Eq. 1 / Eq. 2，而是启发式近似。

2. **在线 cost model 仍是低维近似**
   - 当前不是完整多项式拟合，只估计 prefill token 单价和 decode iteration 单价。
   - mixed batch 暂不用于拟合 `Cp`，避免异步调度间隔污染。

3. **output length prediction 仍是 fallback**
   - 当前使用 `max_new_tokens - generated` 估计 remaining output length。
   - 尚未按输入长度 / max output length 分桶统计真实输出长度分布。

4. **缺少 peak memory prediction**
   - 当前 `chunk_upper_bound` 仍来自用户显式 `--chunked-prefill-size`。
   - 高并发下 memory 边界仍主要依赖 SGLang 原有 admission 和 allocator。
   - 更合理的 SOLA-style 实现应根据当前 KV/token pool 剩余容量、watermark、page size 和 running decode 未来 token 需求预测最大安全 chunk。

5. **TP 多进程一致性**
   - 当前同步的是 TP group 内的 pressure 标量，而不是完整 Python decision object。
   - deterministic objective 规则仍然保留，用于降低 pressure 边界抖动。

## 后续完整 SOLA 路线

### Phase 1：可观测性完善

- 暴露 Prometheus metrics：
  - `sglang:slo_prefill_ttft_pressure`
  - `sglang:slo_prefill_tpot_pressure`
  - `sglang:slo_prefill_objective`
  - `sglang:slo_prefill_chunk_size`
  - `sglang:slo_prefill_yield_total`
- 增加 objective transition 日志。

### Phase 2：在线 Cost Model

当前已实现轻量在线 EMA 模型，后续可升级为论文中的多项式模型：

```text
Cp ~= a0 * sum(l_has * l_in) + b0 * sum(l_in^2) + c0 * sum(l_in) + d0
Cd ~= a1 * batch_size + b1 * sum(kv_len) + c1
```

输入来自 scheduler metrics 中已统计的 prefill/decode 迭代间隔。后续升级项：

- 引入 prefix/KV 长度特征；
- 区分 batch size、chunk size、MoE routing 等特征；
- 使用 device timer 替代 wall-clock log interval。

### Phase 3：Constrained Workload

当前已基于 cost model 近似 SOLA Eq. 1 / Eq. 2：

- `objective=ttft`：用 TPOT slack 计算最大可接受 prefill token budget `ki`。
- `objective=tpot`：用 TTFT slack 判断是否 yield decode，或保留最小 prefill budget 防止 TTFT 饿死。

后续可进一步做 request-level exact solver，而不是当前的 closed-form budget。

### Phase 4：Peak Memory Prediction

在加入 prefill 请求前预测未来 KV 峰值，避免高并发下触发 preemption 或 allocator 边界问题。建议落地为动态 chunk 上限：

```text
free_pages = token_pool_free_pages - reserved_decode_pages - safety_margin_pages
memory_cap_chunk = floor_to_tile(free_pages * page_size / schedulable_prefill_reqs)
chunk_upper_bound = memory_cap_chunk
```

设计取舍：

- **直接忽略用户 `--chunked-prefill-size`**：最符合自适应，但行为变化较大；用户给小值时也可能被自动放大。
- **兼容模式 `min(user_cap, memory_cap_chunk)`**：更安全，但如果用户 cap 太小，SLO controller 仍无法充分利用显存。
- **推荐后续默认**：SLO-aware 开启时使用 `memory_cap_chunk` 作为真正上限，并把用户参数降级为 fallback / debug cap，或者新增 `--slo-prefill-respect-user-chunk-cap` 控制兼容行为。

### Phase 5：TP Rank Pressure Sync

该项已经在当前实现中落地：

```text
all ranks compute local PressureState
all_reduce(MAX) produces one global PressureState
all ranks derive identical objective/chunk/yield decision locally
```

后续可以继续把 SLO pressure 与其它 scheduler control-plane 状态合并同步，或按固定采样周期降低同步频率。

## 当前验证

当前实现配套单测：

```bash
python3 test/registered/unit/managers/test_slo_aware_prefill.py
```

覆盖场景：

- TPOT pressure 下延迟 prefill；
- TTFT pressure 高时恢复 prefill capacity；
- chunked prefill yield 给 decode；
- ambiguous low-pressure 场景默认 TTFT；
- 避免 sticky `tpot` 导致 TP rank 分歧；
- 低于 SLO 但 TPOT 相对更紧时仍自适应切换到 `tpot`。
