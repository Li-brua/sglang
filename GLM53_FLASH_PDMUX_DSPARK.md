# GLM-5.3-Flash with PDMux and DSpark

This recipe applies to `feat/glm53-flash-pdmux` in this checkout. The separate
`feat/glm53-pdmux-overlap` PR stack does not include this DSpark adaptation.

PDMux previously forwarded split prefill directly to `TpModelWorker` through
`DSparkWorkerV2.__getattr__`. That bypassed DSpark's full-token hidden capture,
draft context KV injection, and `DFlashDraftInputV2` handoff. It could leave the
next decode step without valid draft state even though the server started.

The DSpark worker now keeps target features on the persistent prefill batch,
injects draft KV after the last layer, and hands the sampled bonus token and
sequence lengths to decode. Injection happens before result processing or
chunk stashing can change `out_cache_loc`. Intermediate slices and idle DP
ranks do not inject KV. The prefill token broadcast uses the prefill TP group.
An idle split-prefill participant always executes the requested layer interval,
even when its local forward mode would otherwise permit decode graph replay.

Eager target verify and idle decode use the selected decode attention backend,
matching CUDA graph replay. Mamba/KDA acceptance commits use that same backend.
This preserves the attention metadata of a prefill that spans multiple decode
steps. GLM's helper stream also follows the full-device decode lane for verify.

## Launch

Use a DSpark draft checkpoint trained for **this exact GLM-5.3-Flash target**.
Set `MODEL_PATH` and `DRAFT_MODEL_PATH` to the local checkpoints. The existing
DSpark loader resolves the block size and target capture layers from the draft
config; a draft for another target is unsuitable.

The following example uses eight CUDA GPUs, TP8/DP8, EP1, FP8 target weights,
BF16 KV, and full-device decode overlapping a prefill capped at 104 SMs. Adjust
the prefill SM allocation to the GPU; it must leave at least 16 SMs for the
Green Context split. Run from this repository root:

```bash
export MODEL_PATH=/path/to/GLM-5.3-Flash
export DRAFT_MODEL_PATH=/path/to/compatible-dspark-draft
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

cat > /tmp/glm53-flash-pdmux.yaml <<'YAML'
sm_group_num: 3
split_forward_token_budget: 65536
max_split_forward_layers: 2
manual_divisions:
  - [104, 0, 0]
overlap_decode_full_sm: true
YAML

python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --trust-remote-code \
  --tp 8 --dp 8 --enable-dp-attention --enable-dp-lm-head \
  --mm-enable-dp-encoder \
  --quantization fp8 --disable-shared-experts-fusion \
  --attention-backend dsa \
  --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
  --linear-attn-backend triton --kv-cache-dtype bfloat16 \
  --moe-runner-backend deep_gemm --expert-parallel-size 1 \
  --enable-pdmux --disable-overlap-schedule --chunked-prefill-size -1 \
  --sm-group-num 3 --pdmux-config-path /tmp/glm53-flash-pdmux.yaml \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path "$DRAFT_MODEL_PATH" \
  --speculative-draft-attention-backend flashinfer \
  --mem-fraction-static 0.75 \
  --tool-call-parser glm47 --reasoning-parser glm45 \
  --enable-metrics --enable-cache-report \
  --host 0.0.0.0 --port 1213
```

PDMux requires PP1, disabled scheduler overlap, no mixed prefill/decode batch,
and no PD disaggregation. PDMux supports `DSPARK` and ordinary decoding;
other speculative algorithms are rejected at startup. Layerwise prefill does
not use prefill CUDA graphs. Decode/verify CUDA graphs retain their existing
per-stream captures, with an isolated eager backend for unsupported shapes.

## Validation

CPU regressions cover full-token capture on the persistent forward batch,
no early injection, chunk-prefix positions, token-subset KV alignment, idle DP
participation, the ordinary prefill path, decode backend selection, Mamba commit
selection, idle split-prefill graph exclusion, and GLM auxiliary features
surviving interleaved verify forwards.

```bash
PYTHONPATH=python python3 -m unittest discover \
  -s test/registered/unit/spec -p test_dspark_pdmux.py -v
PYTHONPATH=python python3 -m unittest discover \
  -s test/registered/unit/models -p test_glm5_next_pdmux.py -v
```

Development on macOS ran the source methods and these focused assertions with
real CPU PyTorch, isolating imports of unavailable serving/CUDA dependencies.
Full serving-runtime imports, CUDA kernels, graph replay, eight-rank stress,
accuracy, and performance have not been validated in this environment.
Before deployment, compare against the same target and draft with PDMux
disabled, exercise simultaneous long prefills and decode (including idle DP
ranks), then repeat with `--disable-cuda-graph`, chunked prefill, and HiCache.
