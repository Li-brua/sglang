# PDMux attention-DP prefill preparation reuse

This branch supports DP preparation reuse for non-speculative and DSPARK target
prefills in `--pdmux-prefill-mode layer_split` (the default). The first layer slice
prepares and pads a token chunk. Resumed slices keep those tensors and restore
batch-owned DP buffer sizes, real device token counts and the extend flag after
intervening decode. The last slice unpads before sampling or DSPARK KV injection.
A new token chunk gets a new batch and prepares again.

Real counts and padded gather counts are stored separately. In TP8/DP2, alignment
uses attention TP4; empty MAX_LEN ranks retain dummy rows until the final slice.
An explicit IDLE split participates in model layers even when a decode graph is
available. DSPARK completion follows the target layer boundary, including an IDLE
rank with no logits. Its final draft KV injection is fenced against adjacent decode.

Draft, target verify, EAGLE/MTP, LoRA and HiSparse forwards keep their existing
preparation path. `--pdmux-prefill-mode standard` executes the whole token chunk
in one call, so it has no resumed layer slices to optimize. SM divisions and
`split_forward_token_budget` are unchanged.

## CPU validation

Run these tests with CPU PyTorch installed:

```bash
PYTHONPATH=python python3 -m pytest -q \
  test/registered/unit/model_executor/test_mlp_sync_pad_unpad.py \
  test/registered/unit/model_executor/test_pdmux_split_dispatch.py \
  test/registered/unit/spec/test_dspark_pdmux.py \
  test/registered/unit/multiplex/test_pdmux_overlap_streams.py
```

They cover TP8/DP8 and TP8/DP2, uneven counts, active and IDLE ranks, SUM_LEN and
MAX_LEN padding, intervening decode metadata replacement, one preparation per
chunk, final-only unpadding and real-row DSPARK injection. Model layers and CUDA
streams are mocked; GPU collectives, generated-text parity, DSPARK acceptance
rate and throughput still require the host validation below.

## GPU validation

Use the same model, DSPARK draft/checkpoint, HiCache, SM divisions and token
budget for the parent and this commit. Add `--pdmux-prefill-mode layer_split`
explicitly to the existing TP8/DP8 launch with `--enable-dp-attention`,
`--enable-dp-lm-head` and `--speculative-algorithm DSPARK`. Repeat with `--dp 2`
for attention TP4. Compare greedy output and DSPARK acceptance, exercise a busy
prefill rank with IDLE peers under sustained decode, and check for collective
hangs. Repeat without speculative decoding and with multiple token chunks.
Profile prepare/copy/unpad calls per chunk: preparation and unpadding should
occur once rather than once per layer slice. Measure TTFT, decode latency and
throughput with the same concurrent workload. The standard lane should retain
its existing behavior.
