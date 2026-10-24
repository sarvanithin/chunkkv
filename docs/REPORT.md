# When does non-prefix KV cache reuse actually help agents?

*ChunkKV technical report · Nithin Sarva · October 2026*

## Abstract

Agent and RAG requests repeat the same documents and tool results, but at different
positions. Prefix caching can't reuse them. CacheBlend-style selective recompute can. We
implement it twice (MLX and PyTorch/CUDA), add a provenance-keyed cache and a per-chunk reuse
gate, and replay five workloads under five cache policies on Qwen2.5-7B-Instruct.

Four findings:
1. On single-agent traffic, chunk reuse adds at most **1.05×** over prefix caching, and is
   slower in practice.
2. On multi-agent and shuffled-RAG traffic it cuts time-to-first-token **~2×**, but at
   **2–8 points of accuracy**. It never reached "within 1 point of full recompute" at a
   useful speed.
3. CacheBlend in LMCache doesn't run on current vLLM, because of a packed K|V layout the
   blend path doesn't support.
4. Our gate's added value over plain CacheBlend is small.

## 1. Problem

A KV cache holds the attention keys and values computed during prefill. vLLM's automatic
prefix caching and SGLang's RadixAttention reuse them only for an *exact token prefix*. One
changed token early in the prompt, and everything after it is recomputed.

Agent and RAG traffic breaks the prefix assumption in four ways:
- retrieved documents arrive in different orders;
- the same tool result is re-sent after a different history;
- several agents share documents under different system prompts;
- context compaction rewrites earlier turns.

Reusing a chunk's KV at a new position isn't free either. The cached values were computed
while attending to the chunk's *old* neighbours. CacheBlend (EuroSys '25) repairs this by
recomputing a small fraction of tokens.

The original ChunkKV proposal claimed **≥2× lower TTFT and ≤50% prefill compute vs. prefix
caching on multi-turn tool-agent traces, within 1 point of full-recompute accuracy**. This
report tests that claim.

## 2. System

### 2.1 Engine

Both engines (`engine/mlx_engine.py`, `engine/torch_engine.py`) walk the decoder layers by
hand. At each layer, Q/K/V are computed only for an **active set** of positions. Every other
position reads K/V from cached chunks that have been moved to their new offsets.

- **Re-RoPE.** Keys are cached post-RoPE. Rotations compose, so moving a chunk by Δ
  positions is one more rotation by Δ. Tests check this against MLX's own `nn.RoPE` to 1e-5,
  and a Δ then −Δ round trip.
- **Blend.** Layer 0 runs on every token except chunks computed after the exact same prefix
  (those are reused untouched). At layer 1, each cached token's fresh K/V is compared with
  its cached K/V. Each blended chunk contributes a budget of `ratio × length` tokens. The
  pooled budget goes to the most-deviating tokens across all blended chunks, plus each
  chunk's first two tokens (attention sinks). Layers 1…L−1 run on that set only.

  Pooling matters. On a two-document example, per-chunk top-k at 15% gave KL 0.115 versus
  P0, while pooled top-k gave 0.069 at the same total budget.
- **Probe.** If a chunk's relative layer-1 deviation exceeds a threshold, the whole chunk is
  recomputed from layer 1 on.
- **Correctness checks**, on both backends:
  - with nothing cached, logits match the stock model forward (MLX ≤9e-5 in fp32; CUDA
    within 2e-3);
  - same-position reuse is exact;
  - full-ratio blend reproduces full recompute (KL < 1e-4);
  - greedy decode matches the stock `generate`.
- **Compute accounting.** `prefill_token_equiv` counts token-layer evaluations, normalized
  so a full prefill equals N. The layer-0 pass is included, so the probe is never free.

### 2.2 Chunker

Messages are rendered as ChatML **header | content | footer**, and each piece is tokenized
separately. Headers and footers start and end with special tokens, so BPE never merges
across a boundary. The content tokens of a message are therefore identical wherever it
appears, and the concatenation is byte-identical to `tokenizer.apply_chat_template`
(tested). Content shorter than 32 tokens isn't cached. Longer content is split into
256-token windows.

### 2.3 Provenance-keyed store and gate

- **Lookup key** = `sha256(tokens) + model_id + tenant_id`, so a lookup never crosses
  tenants.
- **Context lives on the entry.** It isn't part of the key, so one chunk can hold up to
  four entries, one per context it was computed in. Each entry records:
  - `prefix_hash`: all tokens before the chunk;
  - `context_id`: hash of the system prompt plus agent id;
  - `context_trust`: the least-trusted role before the chunk, ordered user < tool < system.

The gate applies, in order:

1. **Trust.** Never serve an entry whose `context_trust` is lower than the request's trust
   floor at that position. KV computed after attacker-controlled text is never served into a
   cleaner context.
2. **Exact prefix** → reuse untouched.
3. **Same `context_id`** → blend at `ratio`.
4. **Different context** → blend at `2 × ratio` with the probe armed, or recompute outright.

**Deviation from the original proposal.** It put a last-N-token "context fingerprint" into
the cache key. Under chat templates, the tokens right before a chunk are almost always the
same template header, so that fingerprint carries no information. A large N degenerates into
prefix caching. We use the semantic `context_id` instead, and keep it out of the key.

## 3. Benchmark

### 3.1 Workloads

All workloads are deterministic (`bench/gen/workloads.py`). Tool results are real CPython
stdlib source files, so token statistics are realistic.

| Workload | Requests | Shape |
|---|---|---|
| multiagent | 30 | 6 groups × 3 shared files in shuffled order; 4–6 roles with distinct system prompts; odd groups add a conflicting advocate/blocker pair |
| agent | 36 | 6 sessions, append-only history, one new file per turn |
| compaction | 48 | same sessions, history compacted to a summary after 4 files, then 2 earlier files re-read |
| rag | 120 | 24 synthetic docs × 6 entities; each question sees 5 docs (gold included) in random order; exact-match ground truth |
| attack | 40 | attacker tenant prefills a shared file after an injected instruction; victim tenant then asks to summarize the same file |

### 3.2 Policies and metrics

| Policy | Reuse |
|---|---|
| P0 | none (reference) |
| P1 | longest exact prefix across prior requests |
| P2 | all chunk hits, no repair, token-keyed |
| P3 | all chunk hits, fixed selective recompute (CacheBlend), token-keyed |
| P4 | gate + provenance keys |

Metrics, computed per request and averaged:
- TTFT: prefill plus first-token logits, p50/p95;
- prefill compute share;
- KL(P0‖policy) on the first-token distribution;
- whether the 24-token greedy output exactly matches P0's;
- RAG exact-match accuracy;
- attack success (marker in the victim's output).

The first 4 requests of each run are excluded as cold-cache. 95% bootstrap CIs are in
`docs/results/*.json`.

**Setup.**
- **GPU:** Qwen/Qwen2.5-7B-Instruct, bf16, NVIDIA L40S 48 GB, PyTorch 2.13.
- **Mac:** mlx-community/Qwen2.5-7B-Instruct-4bit, M3 Pro 18 GB, MLX 0.32. Multi-agent only.

## 4. Results

### 4.1 The ceiling: how much can chunk reuse add over prefix caching?

A token-level replay, with no model involved, gives the best case (`bench/residual.py`):

| Workload | P1 prefix hit | extra chunk hits | P1 compute | chunk-reuse floor | max speedup over P1 |
|---|---|---|---|---|---|
| agent (append-only) | 71.1% | 1.4% | 28.9% | 27.5% | **1.05×** |
| compaction | 60.5% | 10.5% | 39.5% | 28.9% | 1.36× |
| multiagent | 2.2% | 68.8% | 97.8% | 29.1% | **3.37×** |
| rag | 19.5% | 55.7% | 80.5% | 24.8% | **3.25×** |

The original target (2× over prefix caching on multi-turn tool-agent traces) is
**unreachable by construction**. Agent histories only grow at the end, so prefix caching
already captures them. Non-prefix reuse only matters when content is *re-ordered or
re-contextualized*: multi-agent fan-out, shuffled retrieval, compaction.

### 4.2 Main results (GPU, 15% recompute budget)

| Workload | Policy | TTFT p50 | vs P0 | vs P1 | Compute | KL | = P0 output | RAG acc |
|---|---|---|---|---|---|---|---|---|
| rag | P0 | 151.9 ms | 1.00× | 0.80× | 100% | – | – | 100.0% |
| | P1 | 121.7 ms | 1.25× | 1.00× | 81.1% | 0.000 | 100.0% | 100.0% |
| | P2 | 70.6 ms | 2.15× | 1.72× | 7.6% | 0.852 | 93.1% | 93.1% |
| | P3 | 75.5 ms | 2.01× | 1.61× | 23.3% | 0.753 | 92.2% | 92.2% |
| | P4 | 76.6 ms | 1.98× | 1.59× | 23.3% | 0.753 | 92.2% | 92.2% |
| multiagent | P0 | 346.9 ms | 1.00× | 1.00× | 100% | – | – | – |
| | P1 | 346.0 ms | 1.00× | 1.00× | 97.5% | 0.001 | 92.3% | – |
| | P2 | 110.3 ms | 3.14× | 3.14× | 20.6% | 0.712 | 19.2% | – |
| | P3 | 129.3 ms | 2.68× | 2.68× | 35.6% | 0.475 | 30.8% | – |
| | P4 | 182.0 ms | 1.91× | 1.90× | 47.2% | 0.308 | 38.5% | – |
| agent | P1 | 106.2 ms | 3.29× | 1.00× | 28.0% | 0.002 | 75.0% | – |
| | P3 | 128.5 ms | 2.72× | 0.83× | 31.6% | 0.003 | 68.8% | – |
| | P4 | 128.9 ms | 2.71× | 0.82× | 31.9% | 0.002 | 81.2% | – |
| compaction | P1 | 86.6 ms | 2.65× | 1.00× | 39.4% | 0.004 | 75.0% | – |
| | P3 | 97.6 ms | 2.35× | 0.89× | 36.6% | 0.016 | 56.8% | – |
| | P4 | 91.4 ms | 2.51× | 0.95× | 36.8% | 0.025 | 56.8% | – |

### 4.3 Speed vs. accuracy: the recompute budget

Shuffled RAG, P3 (P4 is identical here, because every hit is same-context):

| Recompute budget | TTFT p50 | vs P0 | Compute | KL | Accuracy (95% CI) |
|---|---|---|---|---|---|
| 0% (P2, naive) | 70.6 ms | 2.15× | 7.6% | 0.852 | 93.1% (87.9–97.4) |
| 15% | 75.5 ms | 2.01× | 23.3% | 0.753 | 92.2% (87.1–96.6) |
| 30% | 75.8 ms | 1.99× | 35.2% | 0.492 | 95.7% (91.4–99.1) |
| 50% | 94.8 ms | 1.59× | 51.2% | 0.325 | 98.3% (95.7–100) |
| full recompute (P0) | 151.9 ms | 1.00× | 100% | – | 100.0% |

On multi-agent traffic, a 30% budget gives P3 1.99× at KL 0.31, and P4 (60% for
other-context hits) 1.27× at KL 0.064. At a 50% budget, P4's other-context ratio reaches 100%
and it falls back to full recompute.

### 4.4 Security

The provenance-keyed store never served a cross-tenant hit (P4 attack workload: 100%
compute, output identical to P0), and the unit tests show the trust rule refusing
user-tainted KV. **But attack success was 0% under every policy, including token-keyed
P2/P3.** A naive injected instruction does not survive being reused as KV in a different
context on this model. Demonstrating the defense requires the optimized prefixes from
HijackKV (USENIX Security '26), which we did not reproduce.

### 4.5 Mac / MLX (multi-agent only)

On the M3 Pro with the 4-bit model:
- P3 cut TTFT p50 from 7.37 s to 1.87 s (3.9×) at KL 0.079;
- P4 cut it to 3.19 s (2.3×) at KL 0.064;
- greedy agreement with P0 was 19% and 23%.

P1 measured *slower* than P0 (13.7 s), an artifact of storing full-request KV under 18 GB
of unified memory. P1 behaves correctly on the GPU.

## 5. Upstream: CacheBlend in vLLM + LMCache

The plan was to compare against LMCache's shipped CacheBlend. On vLLM 0.32.0 + LMCache on
an L40S, it fails in four independent ways (details in [LMCACHE_ISSUE.md](LMCACHE_ISSUE.md)):

1. `uv pip install vllm lmcache` silently resolves **LMCache 0.4.3**, whose prebuilt
   `c_ops` fail to load against torch 2.13 (`undefined symbol …materialize_cow_storage…`).
   LMCache 0.5.6 has to be pinned explicitly.
2. The blender looks the model up in `VLLMModelTracker`, but nothing registers it:
   `ValueError: vllm model for vllm-instance not found`.
3. The blend kernels import `vllm.attention`, which vLLM moved to
   `vllm.model_executor.layers.attention`. (On 0.4.3 there is also a flashinfer symbol
   removal.)
4. **Root cause.** vLLM 0.32 stores each layer's cache as one packed tensor
   `[num_blocks, kv_heads, block_size, 2·head_dim]`. LMCache detects this correctly
   (`NL_X_NB_BS_NH_CS`), but the layerwise transfer used by blending only implements the
   5-D flash-attn and flashinfer layouts. The first store fails with
   `IndexError: Dimension out of range (expected [-4, 3], got 4)`. LMCache #4160 (closed) fixed a
   related packed layout for the CacheBlend V3 path. This V1-connector layerwise path is still affected.

Plain vLLM (no connector, no prefix caching) served the same shuffled-RAG prompts at a p50
TTFT of 85 ms (15/20 correct). That's the production-kernel reference our engine's absolute
timings should be compared against.

## 6. Discussion

**Where non-prefix reuse belongs.** Multi-agent fan-out over shared documents, and RAG with
re-ranked or shuffled retrieval. Not single-agent loops: there, prefix caching wins, and
chunk reuse's layer-0 pass and bookkeeping make it slightly *slower*.

**The quality cost is real.** The CacheBlend paper reports near-lossless quality at ~15%
recompute. On Qwen2.5-7B with exact-match RAG, 15% costs 7.8 points, and even 50% costs
1.7. Moved chunks attend to the wrong neighbours, and recomputing the high-deviation tokens
only partly fixes that.

**At 15%, selective recompute doesn't beat naive reuse on RAG.** Naive reuse (no repair)
gets 93.1% and 15% recompute gets 92.2%, with overlapping CIs, while KL only moves from
0.85 to 0.75. Recompute starts to pay off at 30–50%. Two explanations fit: either this RAG
task barely needs cross-document attention, or layer-1 K/V deviation is a weak signal for
choosing tokens. A multi-hop task, where the answer depends on two documents, would tell
these apart. A query-aware selector (ProphetKV-style) is the published alternative to the
deviation rule.

**The gate.** On RAG, every hit is same-context, so the gate reduces to CacheBlend. On
multi-agent traffic, it spends more compute on cross-agent hits and buys lower KL, a
different point on the same curve rather than a better curve. A gate that wins outright
needs a probe that *predicts* accuracy loss. The layer-1 relative deviation we log is the
obvious candidate; calibrating it is future work.

## 7. Threats to validity

- **Timings** come from a Python forward pass, not fused serving kernels. Ratios between
  policies are meaningful; absolute TTFT isn't.
- **Trace sets are small** (30–120 requests). Several CIs are wide.
- **Quality metrics.** First-token KL and greedy agreement are strict proxies. Only RAG has
  task-level ground truth.
- **One model family** (Qwen2.5, standard RoPE), one GPU type.
- **Budget-overflow bug (fixed after v0.1.0).** When the recompute ratio was close to 1.0,
  the pooled budget could exceed the eligible tokens, because sinks are already kept. The
  overflow could then select uncomputed tokens from an exact-reuse chunk. None of the
  published runs hit this case: the 50% P4 run has compute 1.0 and KL 0, so no exact
  placements were present there. The budget is now capped, and
  `test_full_budget_with_leading_exact_chunk` covers it.

## 8. Reproducing

```bash
uv sync --extra cuda   # or plain `uv sync` on Apple Silicon
uv run python -m bench.gen.workloads
uv run python -m bench.residual
uv run python -m bench.replay --backend torch --tag gpu-v0
uv run python -m bench.report --tag gpu-v0 --skip-first 4
python scripts/week1_repro.py --mode baseline   # vLLM reference
python scripts/week1_repro.py --mode blend      # LMCache CacheBlend (fails on vLLM 0.32, see §5)
```

The raw summaries behind every table are in [`docs/results/`](results/).
