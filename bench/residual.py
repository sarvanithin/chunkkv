"""Token-level reuse ceiling per workload: what prefix caching (P1) covers vs chunk reuse.

No model needed: replays the chunked token streams and counts tokens that could be
served from cache under each scheme (same tenant only).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mlx_lm.utils import load_tokenizer
from huggingface_hub import snapshot_download

from chunkkv.proxy.chunker import Chunker


def analyse(path: Path, chunker: Chunker) -> dict:
    seqs: dict[str, list[list[int]]] = {}
    seen: dict[str, set[tuple[int, ...]]] = {}
    total = prefix = chunk_extra = 0
    for line in path.open():
        r = json.loads(line)
        req = chunker.chunk(r["messages"], r["tenant"], r.get("agent"))
        toks, t = req.tokens, r["tenant"]
        lcp = 0
        for s in seqs.get(t, []):
            n = 0
            for a, b in zip(s, toks):
                if a != b:
                    break
                n += 1
            lcp = max(lcp, n)
        lcp = min(lcp, len(toks) - 1)
        extra = sum(
            len(seg.tokens) for seg in req.segments
            if seg.reusable and seg.start >= lcp and tuple(seg.tokens) in seen.get(t, set())
        )
        total += len(toks); prefix += lcp; chunk_extra += extra
        seqs.setdefault(t, []).append(toks)
        seen.setdefault(t, set()).update(tuple(s.tokens) for s in req.segments if s.reusable)
    return {"tokens": total, "p1_share": prefix / total, "chunk_extra_share": chunk_extra / total,
            "p1_compute": 1 - prefix / total, "chunk_compute_floor": 1 - (prefix + chunk_extra) / total}


def main() -> None:
    repo = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen2.5-7B-Instruct-4bit"
    chunker = Chunker(load_tokenizer(Path(snapshot_download(repo))))
    print(f"{'workload':11s} {'tokens':>8s} {'P1 hit':>7s} {'+chunk':>7s} {'P1 compute':>10s} {'chunk floor':>11s} {'max speedup':>11s}")
    for p in sorted(Path("traces").glob("*/trace.jsonl")):
        a = analyse(p, chunker)
        print(f"{p.parent.name:11s} {a['tokens']:8d} {a['p1_share']:7.1%} {a['chunk_extra_share']:7.1%} "
              f"{a['p1_compute']:10.1%} {a['chunk_compute_floor']:11.1%} {a['p1_compute']/a['chunk_compute_floor']:10.2f}x")


if __name__ == "__main__":
    main()
