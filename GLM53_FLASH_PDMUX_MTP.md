# GLM-5.3-Flash MTP with PDMux

This adaptation is on `feat/glm53-flash-pdmux`. The previous DSpark adaptation
was reverted; the separate `feat/glm53-pdmux-overlap` PR stack is unchanged.

GLM's checkpoint NextN/MTP block runs through single-layer `EAGLE` (`NEXTN`
is an alias). PDMux previously called a missing split-prefill method on
`EAGLEWorkerV2`, and eager verify reused the in-flight prefill attention backend.

The target now captures all token hidden states across prefill slices. After
the last slice, MTP draft extend rotates the original prompt tokens and seeds
the next speculative round. It uses a separate batch view, including on idle
DP ranks, so target inputs and split progress survive unchanged. The draft
uses the prefill TP communicator and waits for outstanding decode work before
reusing its planner and index-share buffers. Earlier target slices can overlap
decode as before.

Verify masks, eager verify/idle attention, and accepted Mamba states use the
selected decode backend. GLM target and NextN helper streams are permitted
only on full-device decode lanes. Draft decode/extend graphs and draft prefill
graphs remain disabled under PDMux because their captures do not follow its
stream partitions; MTP draft forwards run eager. Target decode/verify graphs
retain the existing per-stream capture path. Split-prefill idle ranks always
execute their requested layer interval rather than replaying a decode graph.
Speculative planner work runs on the selected compute stream under PDMux.

## Launch

Keep the existing GLM PDMux launch and SM allocation. Replace any DSpark flags
with these arguments, using the **same GLM-5.3-Flash checkpoint** for target and
draft so its NextN weights are loaded:

```text
--speculative-algorithm EAGLE
--speculative-draft-model-path /path/to/GLM-5.3-Flash
--speculative-num-steps 5
--speculative-eagle-topk 1
--speculative-num-draft-tokens 6
```

The target and draft retain the GLM DSA backend. Remove any DSpark-specific
`--speculative-draft-attention-backend flashinfer` override. As before, PDMux
requires `--disable-overlap-schedule`, PP1, no mixed chunk and no PD
disaggregation. Multi-layer EAGLE, adaptive speculative parameters, EAGLE3,
DSpark and other speculative algorithms are rejected with PDMux.

## Validation

Run the focused tests in a complete SGLang serving environment:

```bash
PYTHONPATH=python python3 -m unittest discover \
  -s test/registered/unit/spec -p test_eagle_pdmux.py -v
PYTHONPATH=python python3 -m unittest discover \
  -s test/registered/unit/models -p test_glm5_next_pdmux.py -v
```

Development on macOS passed 17 isolated source-method tests with real CPU
PyTorch, plus 19 existing isolated PDMux regressions. They cover final-slice
MTP handoff, original token recovery, idle DP participation, chunked prompt
rotation, multimodal embedding alignment, decode mask/backend/Mamba selection,
helper reset, graph exclusion and unchanged ordinary prefill behavior.

Full runtime imports remain blocked by missing serving dependencies (`orjson`
is the first missing module). CUDA kernels, target graph replay, eight-rank
stress, accuracy and throughput have not been validated here. On the GPU
host, compare with MTP enabled and PDMux disabled, then exercise concurrent
long prefills and decode, uneven DP ranks, chunked prefill and HiCache. Repeat
with target CUDA graphs disabled to isolate graph-specific issues.
