"""Reuse gate: per segment, decide reuse / blend / recompute.

Rules, in order:
  1. Trust: never serve K/V computed after less-trusted content than the
     request has before this position (HijackKV defense within a tenant;
     tenants are already separated by the lookup key).
  2. Exact context (same preceding tokens): reuse as-is.
  3. Same semantic context (system prompt + agent id): blend at a low budget.
  4. Different context: blend at a higher budget with the deviation probe
     armed, or recompute outright when `cross_context="recompute"`.

`mode="always_blend"` and `mode="naive"` skip the rules and serve every hit;
they implement policies P3 and P2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from chunkkv.store.keys import token_hash
from chunkkv.store.kvstore import Entry
from chunkkv.types import ChunkedRequest, GateAction, GateDecision, Segment


@dataclass
class GateConfig:
    mode: Literal["gated", "always_blend", "naive"] = "gated"
    ratio_same_context: float = 0.10
    ratio_other_context: float = 0.20
    cross_context: Literal["blend", "recompute"] = "blend"
    escalate_threshold: float | None = 0.5  # relative layer-1 deviation; see BlendConfig


class ReuseGate:
    def __init__(self, config: GateConfig | None = None):
        self.config = config or GateConfig()
        self.log: list[GateDecision] = []

    def decide(self, req: ChunkedRequest, seg: Segment, entries: list[Entry]) -> tuple[GateDecision, Entry | None]:
        cfg = self.config
        prefix = token_hash(req.tokens[: seg.start])

        if cfg.mode != "gated":
            entry = next((e for e in entries if e.prefix_hash == prefix), entries[0])
            if entry.prefix_hash == prefix:
                d = GateDecision(seg.index, GateAction.REUSE, "exact_prefix", 0.0, entry.id)
            elif cfg.mode == "naive":
                d = GateDecision(seg.index, GateAction.BLEND, "naive", 0.0, entry.id)
            else:
                d = GateDecision(seg.index, GateAction.BLEND, "fixed", cfg.ratio_same_context, entry.id)
            return self._log(d), entry

        floor = req.context_trust_before(seg.start)
        allowed = [e for e in entries if e.context_trust >= floor]
        if not allowed:
            return self._log(GateDecision(seg.index, GateAction.RECOMPUTE, "trust")), None

        exact = next((e for e in allowed if e.prefix_hash == prefix), None)
        if exact:
            return self._log(GateDecision(seg.index, GateAction.REUSE, "exact_prefix", 0.0, exact.id)), exact

        same = next((e for e in allowed if e.context_id == req.context_id), None)
        if same:
            d = GateDecision(seg.index, GateAction.BLEND, "same_context", cfg.ratio_same_context, same.id)
            return self._log(d), same

        if cfg.cross_context == "recompute":
            return self._log(GateDecision(seg.index, GateAction.RECOMPUTE, "other_context")), None
        entry = allowed[0]
        d = GateDecision(seg.index, GateAction.BLEND, "other_context", cfg.ratio_other_context, entry.id)
        return self._log(d), entry

    def _log(self, d: GateDecision) -> GateDecision:
        self.log.append(d)
        return d
