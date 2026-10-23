"""Replay traces under each policy and record speed, compute and output quality.

    uv run python -m bench.replay --workloads multiagent rag --policies P0 P1 P2 P3 P4

P0 always runs first; its last-position distribution and output are the reference
for KL and output agreement of every other policy on the same request.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

from bench.policies import POLICIES, build
from chunkkv.engine.mlx_engine import MLXEngine
from chunkkv.proxy.chunker import Chunker


def load_trace(workload: str, limit: int | None) -> list[dict]:
    reqs = [json.loads(l) for l in open(Path("traces") / workload / "trace.jsonl")]
    return reqs[:limit] if limit else reqs


def run_policy(policy, engine, chunker, reqs, model_id, args, ref):
    rt = build(policy, engine, model_id, ratio=args.ratio, escalate=args.escalate, cross_context=args.cross_context)
    rows = []
    for r in reqs:
        req = chunker.chunk(r["messages"], r["tenant"], r.get("agent"))
        t0 = time.perf_counter()
        out = rt.prefill(req)
        logits = out.result.logits[-1].astype(mx.float32)
        mx.eval(logits)
        ttft = time.perf_counter() - t0
        gen = engine.generate(out.result, args.max_new)
        text = engine.tokenizer.decode(gen)
        logp = logits - mx.logsumexp(logits)
        row = {
            "id": r["id"], "policy": policy, "tokens": len(req.tokens), "ttft": ttft,
            "compute": out.stats["prefill_token_equiv"] / len(req.tokens),
            "cached_tokens": out.stats["cached_tokens"],
            "recomputed_cached": out.stats["recomputed_cached_tokens"],
            "out": text, "conflict": r.get("conflict", False), "role": r.get("role"),
            "decisions": [[d.action.value, d.reason] for d in out.decisions],
            "chunks": out.stats["chunks"],
        }
        if "truth" in r:
            row["correct"] = r["truth"].lower() in text.lower()
        if "attack_marker" in r:
            row["attacked"] = r["attack_marker"].lower() in text.lower()
        if policy == "P0":
            ref[r["id"]] = (logp, gen)
        else:
            p0_logp, p0_gen = ref[r["id"]]
            row["kl"] = mx.sum(mx.exp(p0_logp) * (p0_logp - logp)).item()
            row["match_p0"] = gen == p0_gen
            n = 0
            for a, b in zip(gen, p0_gen):
                if a != b:
                    break
                n += 1
            row["agree_prefix"] = n / max(1, len(p0_gen))
        rows.append(row)
        del out
    del rt
    mx.clear_cache()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--workloads", nargs="+", default=["multiagent", "rag", "compaction", "agent", "attack"])
    ap.add_argument("--policies", nargs="+", default=POLICIES)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--ratio", type=float, default=0.15)
    ap.add_argument("--escalate", type=float, default=None)
    ap.add_argument("--cross-context", default="blend", choices=["blend", "recompute"])
    ap.add_argument("--tag", default="dev")
    args = ap.parse_args()

    engine = MLXEngine.load(args.model)
    chunker = Chunker(engine.tokenizer)
    model_id = args.model.split("/")[-1]
    warm = chunker.chunk([{"role": "user", "content": "warm up " * 200}], "w")
    mx.eval(engine.prefill(warm.tokens).logits)

    out_dir = Path("runs") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    policies = ["P0"] + [p for p in args.policies if p != "P0"]
    for w in args.workloads:
        reqs = load_trace(w, args.limit)
        ref: dict = {}
        for p in policies:
            t = time.perf_counter()
            rows = run_policy(p, engine, chunker, reqs, model_id, args, ref)
            with open(out_dir / f"{w}.{p}.jsonl", "w") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            print(f"{w:11s} {p}  {len(rows)} reqs  {time.perf_counter() - t:6.1f}s", flush=True)


if __name__ == "__main__":
    main()
