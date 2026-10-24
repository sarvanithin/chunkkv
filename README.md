# ChunkKV

**Non-prefix KV cache reuse for RAG and agents, measured honestly.**

ChunkKV reuses the attention KV cache of document and tool-result chunks that reappear at
*different positions* across LLM requests. Prefix caching (vLLM APC, SGLang RadixAttention)
can't reuse them. It implements CacheBlend-style selective recompute, adds a provenance-keyed
cache and a per-chunk reuse gate, and ships a benchmark that replays agent and RAG traffic
under five cache policies.

It runs on **Apple Silicon (MLX)** and **NVIDIA GPUs (PyTorch/CUDA)** with the same algorithm.

> 📄 Full write-up: [docs/REPORT.md](docs/REPORT.md) · LMCache upstream finding: [docs/LMCACHE_ISSUE.md](docs/LMCACHE_ISSUE.md)

---

## TL;DR results

Qwen2.5-7B-Instruct, bf16, one NVIDIA L40S. Everything runs through the same engine, so the
policies are compared like for like.

| Workload | Best reuse speedup (TTFT p50) | Quality cost |
|---|---|---|
| **Shuffled multi-doc RAG** | **2.0× vs full recompute**, 1.6× vs prefix cache | accuracy 100% → 92.2% (15% recompute) · 95.7% (30%) · 98.3% (50%, 1.6×) |
| **Multi-agent shared context** | **2.7× vs prefix cache** (CacheBlend) / 1.9× (ChunkKV gate) | first-token KL 0.31–0.47; greedy output identical to full recompute only 31–39% of the time |
| Single-agent, append-only | **0.82× (slower)** vs prefix cache | — |
| Agent with context compaction | **0.95× (no gain)** vs prefix cache | — |

What the data says:

1. **Prefix caching already covers single-agent traffic.** The token-level ceiling for
   chunk reuse on top of a prefix cache is **1.05×** for append-only agent histories and
   1.36× with compaction ([reuse_ceiling.txt](docs/results/reuse_ceiling.txt)). The gains
   live in **multi-agent and shuffled-RAG** traffic, where the ceiling is ~3.3×.
2. **Reuse is not free on quality.** On this model, selective recompute does not reach
   "within 1 point of full recompute". The speed/accuracy curve is the useful artifact.
3. **CacheBlend doesn't run on current vLLM + LMCache.** On vLLM 0.32 + LMCache 0.5.6,
   blending fails four separate ways. The root cause is vLLM's packed K|V cache layout, which
   LMCache's layerwise/blend path doesn't support. See [docs/LMCACHE_ISSUE.md](docs/LMCACHE_ISSUE.md).
4. **The reuse gate is not a win yet.** On RAG it matches plain CacheBlend. On multi-agent
   traffic it trades speed for quality. Provenance keys (tenant isolation) work as designed,
   but a naive prompt-injection attack never transferred through reused KV in the first
   place, so the HijackKV defense is still unproven.

---

## How it works

```
 client ──► Chunker ──► Lookup (every segment) ──► Reuse gate ──► Engine (blend prefill) ──► decode
 (OpenAI    split by       hash(tokens) +              reuse / blend /     re-RoPE moved chunks,
 messages)  message,       model + tenant              recompute           recompute the most-
            256-tok        ─────────────►  KV store  ◄─── store new chunk KV  deviating tokens
            windows                      (LRU, per-chunk variants with context + trust metadata)
```

- **Chunker** (`proxy/chunker.py`) renders ChatML piece by piece: header, content, footer.
  Each content segment therefore tokenizes identically wherever it appears. The test suite
  checks the result is byte-identical to `apply_chat_template`.
- **Re-RoPE** (`scheduler/rerope.py`): cached keys are stored post-RoPE. Rotations compose,
  so a chunk moves from position *p* to *p+Δ* by rotating its keys by Δ.
- **Selective recompute** (`engine/mlx_engine.py`, `engine/torch_engine.py`): layer 0 runs
  on all non-exact tokens. At layer 1, the fresh K/V of cached tokens are compared with the
  cached K/V. The most-deviating tokens (pooled across chunks) plus each chunk's first
  tokens (attention sinks) are recomputed through the remaining layers. Everything else reads
  cached K/V. An optional deviation probe escalates a chunk to full recompute.
- **Provenance-keyed store** (`store/`): the lookup key is content + model + tenant. Each
  entry records the semantic context it was computed in (system prompt + agent id) and the
  least-trusted content that preceded it.
- **Reuse gate** (`gate/gate.py`) applies these rules in order:
  1. never serve KV computed after less-trusted content than the request has;
  2. exact same prefix → reuse as-is;
  3. same agent context → blend at a low budget;
  4. different context → blend at a higher budget, or recompute.
- **Regression monitor** (`regress/suite.py`) fails loudly if an upgrade silently disables
  non-prefix reuse.

## Policies benchmarked

| | Policy | What is reused |
|---|---|---|
| P0 | No cache | nothing (accuracy reference) |
| P1 | Prefix cache | longest exact token prefix (like vLLM APC) |
| P2 | Naive chunk reuse | every matching chunk, no repair, token-keyed |
| P3 | CacheBlend | matching chunks + fixed selective recompute, token-keyed |
| P4 | ChunkKV | matching chunks under the reuse gate + provenance keys |

Workloads (`bench/gen/workloads.py`, deterministic, real Python-stdlib source as tool results):
- multi-agent shared context (including conflicting advocate/blocker roles)
- single-agent append-only
- agent with context compaction
- synthetic multi-doc RAG with exact ground truth
- a two-tenant cache-poisoning attempt

---

## Quickstart

### Apple Silicon (MLX)

```bash
git clone https://github.com/sarvanithin/chunkkv && cd chunkkv
uv sync
uv run pytest -q                         # 21 tests; downloads Qwen2.5-0.5B (~1 GB) on first run
uv run python scripts/spike_blend.py     # see blend vs naive reuse on a moved chunk
```

### NVIDIA GPU (PyTorch/CUDA)

```bash
uv sync --extra cuda
uv run pytest -q tests/test_torch_engine.py
```

### Reproduce the benchmark

```bash
uv run python -m bench.gen.workloads                         # write traces/*/trace.jsonl
uv run python -m bench.residual                              # token-level reuse ceiling (no model)

# GPU (Qwen/Qwen2.5-7B-Instruct, bf16)
uv run python -m bench.replay --backend torch --tag gpu-v0
uv run python -m bench.replay --backend torch --tag gpu-r30 --ratio 0.30 --policies P0 P3 P4 --workloads rag multiagent

# Mac (mlx-community/Qwen2.5-7B-Instruct-4bit)
uv run python -m bench.replay --backend mlx --tag mlx-v0

uv run python -m bench.report --tag gpu-v0 --skip-first 4    # results table
uv run python -m chunkkv.regress.suite                       # reuse regression check
```

### Use the runtime directly

```python
from chunkkv.engine.mlx_engine import MLXEngine          # or engine.torch_engine.TorchEngine
from chunkkv.gate.gate import GateConfig, ReuseGate
from chunkkv.proxy.chunker import Chunker
from chunkkv.runtime import ChunkReuseRuntime
from chunkkv.store.kvstore import KVStore

engine = MLXEngine.load("mlx-community/Qwen2.5-7B-Instruct-4bit")
runtime = ChunkReuseRuntime(engine, "qwen2.5-7b", KVStore(), ReuseGate(GateConfig()))
chunker = Chunker(engine.tokenizer)

req = chunker.chunk(messages, tenant_id="acme", agent_id="reviewer")
out = runtime.prefill(req)                 # reuses any cached chunk, at any position
print(out.stats["prefill_token_equiv"], [d.reason for d in out.decisions])
print(engine.tokenizer.decode(engine.generate(out.result, max_tokens=64)))
```

---

## Repository layout

```
src/chunkkv/
  engine/      base.py (shared types) · mlx_engine.py · torch_engine.py
  proxy/       chunker.py — ChatML-exact segmentation
  scheduler/   rerope.py — move cached keys to new offsets
  store/       keys.py · kvstore.py — provenance-keyed LRU store
  gate/        gate.py — reuse / blend / recompute decisions
  runtime.py   ChunkReuseRuntime (P2–P4) · PrefixCacheRuntime (P1)
  regress/     suite.py — reuse regression monitor
bench/         gen/workloads.py · residual.py · policies.py · replay.py · report.py
scripts/       spike_blend.py · week1_repro.py (vLLM + LMCache CacheBlend repro)
docs/          REPORT.md · LMCACHE_ISSUE.md · results/*.json
tests/         rerope, engine (MLX + CUDA), chunker, runtime, regression
```

## Limitations

- Timings come from ChunkKV's own Python forward pass, not vLLM's kernels. Speedups are
  comparable *between policies*, but absolute TTFT isn't production-representative.
- One model family (Qwen2.5, standard RoPE). Rope scaling, MLA and hybrid attention are
  unsupported.
- The quality metric is greedy agreement with full recompute (plus exact-match accuracy on
  RAG). Agent-task success under live re-execution isn't measured.
- The trace sets are small (30–120 requests per workload). Confidence intervals are in
  [docs/results](docs/results).

## License

Apache-2.0. Tool-result traces embed excerpts of CPython standard-library source (PSF License).

## Citation

```bibtex
@software{sarva2026chunkkv,
  author = {Sarva, Nithin},
  title  = {ChunkKV: Non-prefix KV cache reuse for RAG and agents},
  year   = {2026},
  url    = {https://github.com/sarvanithin/chunkkv}
}
```
