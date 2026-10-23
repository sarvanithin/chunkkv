"""Cache policies P0-P4 as runtimes over one engine."""

from __future__ import annotations

from chunkkv.gate.gate import GateConfig, ReuseGate
from chunkkv.runtime import ChunkReuseRuntime, PrefixCacheRuntime
from chunkkv.store.kvstore import KVStore

POLICIES = ["P0", "P1", "P2", "P3", "P4"]
DESCRIPTIONS = {
    "P0": "no cache (full recompute)",
    "P1": "exact prefix cache",
    "P2": "naive chunk reuse, token-keyed",
    "P3": "CacheBlend-style fixed recompute, token-keyed",
    "P4": "ChunkKV: gate + provenance keys",
}


def build(policy: str, engine, model_id: str, budget_bytes: int = 3 << 30, ratio: float = 0.15,
          escalate: float | None = None, cross_context: str = "blend"):
    store = KVStore(budget_bytes)
    if policy == "P0":
        return ChunkReuseRuntime(engine, model_id, store, ReuseGate(), enabled=False)
    if policy == "P1":
        return PrefixCacheRuntime(engine, budget_bytes)
    if policy == "P2":
        return ChunkReuseRuntime(engine, model_id, store, ReuseGate(GateConfig(mode="naive")), provenance=False)
    if policy == "P3":
        gate = ReuseGate(GateConfig(mode="always_blend", ratio_same_context=ratio))
        return ChunkReuseRuntime(engine, model_id, store, gate, provenance=False)
    if policy == "P4":
        gate = ReuseGate(GateConfig(ratio_same_context=ratio, ratio_other_context=min(1.0, 2 * ratio),
                                    escalate_threshold=escalate, cross_context=cross_context))
        return ChunkReuseRuntime(engine, model_id, store, gate, provenance=True)
    raise ValueError(policy)
