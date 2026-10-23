"""Week-1 repro: does CacheBlend non-prefix reuse work on current vLLM + LMCache?

Run on a CUDA box:
    python scripts/week1_repro.py [--model Qwen/Qwen2.5-7B-Instruct] [--mode blend|baseline]

`blend` enables LMCache blending (env flags below); `baseline` runs vLLM with no
KV connector and no prefix caching. Each mode warms chunks with one prompt,
then sends the same chunks reordered, and prints per-request TTFT plus the
generated text so the two modes can be compared. LMCache's own log lines
report how many tokens hit.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--mode", default="blend", choices=["blend", "baseline"])
    ap.add_argument("--ratio", type=float, default=0.15)
    ap.add_argument("--n", type=int, default=20, help="reordered requests to measure")
    args = ap.parse_args()

    if args.mode == "blend":
        os.environ.update({
            "LMCACHE_CHUNK_SIZE": "256",
            "LMCACHE_ENABLE_BLENDING": "True",
            "LMCACHE_BLEND_SPECIAL_STR": " # # ",
            "LMCACHE_USE_LAYERWISE": "True",
            "LMCACHE_BLEND_CHECK_LAYERS": "1",
            "LMCACHE_BLEND_RECOMPUTE_RATIOS": str(args.ratio),
            "LMCACHE_LOCAL_CPU": "True",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": "20",
        })

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    if args.mode == "blend":
        # Upstream break (vLLM 0.32 + LMCache 0.4.3): nothing registers the vLLM model
        # with LMCache's VLLMModelTracker, so the blender fails at startup with
        # "vllm model for vllm-instance not found". Register it ourselves, in-process.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        # Second break: LMCache's blend kernels import `vllm.attention`, which vLLM
        # moved to `vllm.model_executor.layers.attention`.
        import sys
        import vllm.model_executor.layers.attention as _attn

        sys.modules.setdefault("vllm.attention", _attn)
        from lmcache.integration.vllm.utils import ENGINE_NAME
        from lmcache.v1.compute.models.utils import VLLMModelTracker
        from vllm.v1.worker import gpu_worker

        orig_init = gpu_worker.Worker.initialize_from_config

        def init_with_registration(self, kv_cache_config):
            VLLMModelTracker.register_model(ENGINE_NAME, self.model_runner.get_model())
            return orig_init(self, kv_cache_config)

        gpu_worker.Worker.initialize_from_config = init_with_registration

        # Third break (the KV-layout failure, cf. LMCache issue 4160): the layerwise
        # connector normalizes vLLM's packed 4-D KV tensors to the 5-D layout only
        # once, in _lazy_initialize_buffer, but initialize_kvcaches_ptr resets
        # self.kvcaches to the raw tensors on every call, so later stores index a
        # 4-D tensor as 5-D. Re-normalize on every call.
        from lmcache.utils import EngineType
        from lmcache.v1.gpu_connector import gpu_connectors as gc

        layerwise = gc.VLLMBufferLayerwiseGPUConnector
        orig_ptr = layerwise.initialize_kvcaches_ptr

        def ptr_normalized(self, **kwargs):
            orig_ptr(self, **kwargs)
            if "kvcaches" in kwargs and getattr(self, "gpu_buffer_allocator", None) is not None:
                _, self.kvcaches = gc.normalize_kv_and_discover_format(
                    self.kvcaches, EngineType.VLLM, layout_hints=self.layout_hints
                )

        layerwise.initialize_kvcaches_ptr = ptr_normalized

    kw = dict(model=args.model, gpu_memory_utilization=0.8, max_model_len=16384, enable_prefix_caching=False)
    if args.mode == "blend":
        kw["kv_transfer_config"] = KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_both")
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0, max_tokens=24)

    # Shuffled-RAG prompts built from the chunkkv rag trace (same docs recur in new orders).
    reqs = [json.loads(l) for l in open(Path(__file__).parent.parent / "traces/rag/trace.jsonl")]
    sep = tok.encode(" # # ", add_special_tokens=False)
    docs = sorted({m["content"] for r in reqs for m in r["messages"] if m["role"] == "tool"})
    sys_ids = tok.encode(reqs[0]["messages"][0]["content"], add_special_tokens=False)

    def prompt(doc_texts: list[str], question: str) -> list[int]:
        ids = list(sys_ids)
        for d in doc_texts:
            ids += sep + tok.encode(d, add_special_tokens=False)
        return ids + sep + tok.encode(question, add_special_tokens=False)

    def run(ids: list[int]):
        t = time.perf_counter()
        out = llm.generate([{"prompt_token_ids": ids}], SamplingParams(temperature=0, max_tokens=1), use_tqdm=False)
        ttft = time.perf_counter() - t
        full = llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)[0].outputs[0].text
        return ttft, full

    # Warm: store every doc once as its own chunk (blend mode stores per separator segment).
    for d in docs:
        llm.generate([{"prompt_token_ids": prompt([d], "ok")}], SamplingParams(temperature=0, max_tokens=1), use_tqdm=False)

    rng = random.Random(0)
    results = []
    for r in reqs[: args.n]:
        ctx = [m["content"] for m in r["messages"] if m["role"] == "tool"]
        rng.shuffle(ctx)
        q = r["messages"][-1]["content"]
        ids = prompt(ctx, q)
        ttft, text = run(ids)
        results.append({"id": r["id"], "tokens": len(ids), "ttft": ttft, "out": text,
                        "correct": r["truth"].lower() in text.lower()})
        print(f"{r['id']:8s} toks={len(ids):5d} ttft={ttft*1000:7.1f}ms correct={results[-1]['correct']} | {text.strip()[:50]!r}", flush=True)

    out = Path("runs/week1"); out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.mode}.json").write_text(json.dumps(results, indent=2))
    ttfts = sorted(x["ttft"] for x in results)
    print(f"\nmode={args.mode} p50 TTFT={ttfts[len(ttfts)//2]*1000:.1f}ms "
          f"acc={sum(x['correct'] for x in results)}/{len(results)}")


if __name__ == "__main__":
    main()
