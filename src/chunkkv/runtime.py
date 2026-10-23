"""Request path: chunked request -> lookup every segment -> gate -> prefill -> store.

`ChunkReuseRuntime` serves policies P2-P4 (chunk reuse at any position).
`PrefixCacheRuntime` serves P1 (longest exact token prefix, like vLLM APC).
P0 is either runtime with caching disabled.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

from chunkkv.engine.base import BlendConfig, Placement, PrefillResult
from chunkkv.gate.gate import ReuseGate
from chunkkv.store.keys import LookupKey, token_hash
from chunkkv.store.kvstore import Entry, KVStore
from chunkkv.types import ChunkedRequest, GateAction, GateDecision


@dataclass
class Outcome:
    result: PrefillResult
    decisions: list[GateDecision] = field(default_factory=list)

    @property
    def stats(self) -> dict:
        return self.result.stats


class ChunkReuseRuntime:
    def __init__(
        self,
        engine,
        model_id: str,
        store: KVStore,
        gate: ReuseGate,
        provenance: bool = True,
        enabled: bool = True,
    ):
        self.engine = engine
        self.model_id = model_id
        self.store = store
        self.gate = gate
        self.provenance = provenance
        self.enabled = enabled

    def _key(self, req: ChunkedRequest, tokens: list[int]) -> LookupKey:
        return LookupKey.build(tokens, self.model_id, req.tenant_id if self.provenance else None)

    def prefill(
        self,
        req: ChunkedRequest,
        extra_tokens: list[int] | None = None,
        logits_from: int | None = None,
    ) -> Outcome:
        """Prefill the request (plus optional teacher-forced `extra_tokens`)."""
        tokens = req.tokens + (extra_tokens or [])
        placements: list[Placement] = []
        decisions: list[GateDecision] = []
        if self.enabled:
            for seg in req.segments:
                if not seg.reusable:
                    continue
                entries = self.store.get(self._key(req, seg.tokens))
                if not entries:
                    continue
                d, entry = self.gate.decide(req, seg, entries)
                decisions.append(d)
                if d.action is GateAction.RECOMPUTE or entry is None:
                    continue
                entry.hits += 1
                placements.append(Placement(entry.kv, seg.start, d.ratio, exact=d.action is GateAction.REUSE))

        cfg = self.gate.config
        naive = cfg.mode == "naive" or all(d.action is GateAction.REUSE for d in decisions)
        blend = BlendConfig(
            recompute_ratio=0.0,
            full_first_layer=not naive,
            escalate_threshold=cfg.escalate_threshold if cfg.mode == "gated" else None,
        )
        result = self.engine.prefill(tokens, placements, blend, logits_from)

        if self.enabled:
            for seg in req.segments:
                if seg.reusable:
                    entry = Entry(
                        kv=result.chunk(seg.start, seg.end),
                        context_id=req.context_id,
                        prefix_hash=token_hash(req.tokens[: seg.start]),
                        context_trust=req.context_trust_before(seg.start),
                    )
                    self.store.put(self._key(req, seg.tokens), entry)
        return Outcome(result, decisions)


class PrefixCacheRuntime:
    """Longest-exact-prefix reuse over previously prefilled requests."""

    def __init__(self, engine, budget_bytes: int = 4 << 30):
        self.engine = engine
        self.budget = budget_bytes
        self._seqs: OrderedDict[int, tuple[list[int], PrefillResult]] = OrderedDict()
        self._next = 0
        self.bytes = 0

    def _lcp(self, tokens: list[int]) -> tuple[int, PrefillResult | None]:
        best, best_res = 0, None
        for sid, (seq, res) in self._seqs.items():
            n = 0
            for a, b in zip(seq, tokens):
                if a != b:
                    break
                n += 1
            if n > best:
                best, best_res = n, res
        return best, best_res

    def prefill(self, req: ChunkedRequest, extra_tokens: list[int] | None = None, logits_from: int | None = None) -> Outcome:
        tokens = req.tokens + (extra_tokens or [])
        limit = (len(tokens) - 1) if logits_from is None else logits_from
        n, res = self._lcp(tokens)
        n = min(n, limit)
        placements = [Placement(res.chunk(0, n), 0, exact=True)] if res is not None and n > 0 else []
        result = self.engine.prefill(tokens, placements, BlendConfig(full_first_layer=False), logits_from)

        self._seqs[self._next] = (tokens, result)
        self._next += 1
        self.bytes += sum(k.nbytes + v.nbytes for k, v in zip(result.keys, result.values))
        while self.bytes > self.budget and len(self._seqs) > 1:
            _, (_, old) = self._seqs.popitem(last=False)
            self.bytes -= sum(k.nbytes + v.nbytes for k, v in zip(old.keys, old.values))
        return Outcome(result)
