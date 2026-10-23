"""Aggregate replay runs into a results table.

    uv run python -m bench.report --tag v0
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from bench.policies import POLICIES


def boot_ci(x: np.ndarray, stat=np.mean, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    s = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def load(run: Path, workload: str, policy: str) -> list[dict]:
    p = run / f"{workload}.{policy}.jsonl"
    return [json.loads(l) for l in p.open()] if p.exists() else []


def summarize(rows: list[dict], skip_first: int) -> dict:
    rows = rows[skip_first:]
    ttft = np.array([r["ttft"] for r in rows])
    s = {
        "n": len(rows),
        "ttft_p50": float(np.median(ttft)), "ttft_p95": float(np.percentile(ttft, 95)),
        "compute": float(np.sum([r["compute"] * r["tokens"] for r in rows]) / np.sum([r["tokens"] for r in rows])),
    }
    if "kl" in rows[0]:
        s["kl"] = float(np.mean([r["kl"] for r in rows]))
        m = np.array([r["match_p0"] for r in rows], float)
        s["match_p0"], s["match_ci"] = float(m.mean()), boot_ci(m)
        conf = np.array([r["match_p0"] for r in rows if r.get("conflict")], float)
        if len(conf):
            s["match_p0_conflict"] = float(conf.mean())
    if "correct" in rows[0]:
        c = np.array([r["correct"] for r in rows], float)
        s["acc"], s["acc_ci"] = float(c.mean()), boot_ci(c)
    att = [r["attacked"] for r in rows if "attacked" in r]
    if att:
        s["attack_success"] = float(np.mean(att))
    s["decisions"] = dict(Counter(a for r in rows for a, _ in r["decisions"]))
    s["reasons"] = dict(Counter(why for r in rows for _, why in r["decisions"]))
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--skip-first", type=int, default=0, help="drop the first N requests (cold cache) per run")
    args = ap.parse_args()
    run = Path("runs") / args.tag
    workloads = sorted({p.name.split(".")[0] for p in run.glob("*.jsonl")})
    out = {}
    for w in workloads:
        print(f"\n== {w}")
        print(f"{'pol':4s} {'n':>4s} {'TTFT p50':>9s} {'p95':>7s} {'vs P1':>6s} {'compute':>8s} {'KL':>7s} "
              f"{'=P0 out':>8s} {'=P0 confl':>9s} {'acc':>6s} {'attack':>7s}  decisions")
        base = None
        for p in POLICIES:
            rows = load(run, w, p)
            if not rows:
                continue
            s = summarize(rows, args.skip_first)
            out[f"{w}.{p}"] = s
            if p == "P1":
                base = s["ttft_p50"]
            speed = f"{base / s['ttft_p50']:5.2f}x" if base else "     -"
            fmt = lambda k, f: (f.format(s[k]) if k in s else "-")
            print(f"{p:4s} {s['n']:4d} {s['ttft_p50']:8.3f}s {s['ttft_p95']:6.3f}s {speed:>6s} {s['compute']:7.1%} "
                  f"{fmt('kl', '{:7.4f}'):>7s} {fmt('match_p0', '{:7.1%}'):>8s} {fmt('match_p0_conflict', '{:8.1%}'):>9s} "
                  f"{fmt('acc', '{:5.1%}'):>6s} {fmt('attack_success', '{:6.1%}'):>7s}  {s['reasons']}")
    (run / "summary.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
