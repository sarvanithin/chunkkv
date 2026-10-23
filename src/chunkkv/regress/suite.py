"""Reuse regression monitor.

Replays a fixed suite of requests whose reusable chunks recur in shuffled
order and asserts that non-prefix chunk reuse still happens. Run it after
every mlx / mlx-lm (or vLLM / LMCache) upgrade: if the hit rate collapses, the
run fails loudly instead of silently falling back to full recompute.

    uv run python -m chunkkv.regress.suite [--model REPO] [--min-hit 0.8]
"""

from __future__ import annotations

import argparse
import random
import sys

from chunkkv.engine.mlx_engine import MLXEngine
from chunkkv.gate.gate import GateConfig, ReuseGate
from chunkkv.proxy.chunker import Chunker
from chunkkv.runtime import ChunkReuseRuntime
from chunkkv.store.kvstore import KVStore
from chunkkv.types import GateAction

DOCS = [
    f"Record {i}: service {name} owns the {area} pipeline; on-call rotates weekly and alerts page after "
    f"{i + 2} failed health checks. Escalate to the {area} lead if the queue backlog exceeds {100 * (i + 1)} jobs. "
    * 2
    for i, (name, area) in enumerate(
        [("atlas", "billing"), ("borealis", "search"), ("cobalt", "ingest"), ("dune", "auth"), ("ember", "export")]
    )
]


def suite_requests(n: int = 12, k: int = 3, seed: int = 0) -> list[list[dict]]:
    rng = random.Random(seed)
    reqs = []
    for _ in range(n):
        docs = rng.sample(DOCS, k)
        msgs = [{"role": "system", "content": "You are an operations assistant."}]
        msgs += [{"role": "tool", "content": d} for d in docs]
        msgs.append({"role": "user", "content": "Who should be paged first?"})
        reqs.append(msgs)
    return reqs


def run(engine: MLXEngine, model_id: str = "regress") -> dict:
    rt = ChunkReuseRuntime(engine, model_id, KVStore(), ReuseGate(GateConfig()))
    chunker = Chunker(engine.tokenizer)
    seen: set[tuple[int, ...]] = set()
    expected = served = non_prefix = 0
    for msgs in suite_requests():
        req = chunker.chunk(msgs, "regress", "ops")
        out = rt.prefill(req)
        served_ids = {d.segment for d in out.decisions if d.action is not GateAction.RECOMPUTE}
        for seg in req.segments:
            if not seg.reusable:
                continue
            if tuple(seg.tokens) in seen:
                expected += len(seg.tokens)
                if seg.index in served_ids:
                    served += len(seg.tokens)
                    d = next(d for d in out.decisions if d.segment == seg.index)
                    non_prefix += len(seg.tokens) if d.action is GateAction.BLEND else 0
            seen.add(tuple(seg.tokens))
    return {"expected": expected, "hit_rate": served / max(1, expected), "non_prefix_hit_rate": non_prefix / max(1, expected)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    ap.add_argument("--min-hit", type=float, default=0.8)
    args = ap.parse_args()
    r = run(MLXEngine.load(args.model))
    print(f"recurring chunk tokens={r['expected']}  hit rate={r['hit_rate']:.1%}  non-prefix={r['non_prefix_hit_rate']:.1%}")
    if r["non_prefix_hit_rate"] < args.min_hit:
        print(f"FAIL: non-prefix hit rate below {args.min_hit:.0%} - chunk reuse is silently broken", file=sys.stderr)
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main()
