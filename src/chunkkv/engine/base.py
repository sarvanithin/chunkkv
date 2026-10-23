"""Backend-neutral engine types shared by the MLX and PyTorch engines.

Arrays are whatever the backend uses (mx.array or torch.Tensor); K/V are
(1, n_kv_heads, L, head_dim) with RoPE applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ChunkKV:
    """Per-layer K/V for one chunk, as computed at positions start..start+len."""

    keys: list[Any]
    values: list[Any]
    start: int

    @property
    def length(self) -> int:
        return self.keys[0].shape[2]


@dataclass
class Placement:
    """A cached chunk placed at a new offset in the request."""

    chunk: ChunkKV
    start: int
    ratio: float | None = None  # per-chunk recompute budget; None = config default
    exact: bool = False  # computed after exactly the same prefix: never recomputed, not even at layer 0


@dataclass
class BlendConfig:
    recompute_ratio: float = 0.15  # fraction of cached tokens recomputed at layers >= 1
    n_sink: int = 2  # first tokens of each chunk always recomputed
    full_first_layer: bool = True  # False = naive reuse (no repair at all)
    # Deviation probe: if a chunk's relative layer-1 K/V deviation exceeds this,
    # recompute the whole chunk from layer 1 on. None disables the probe.
    escalate_threshold: float | None = None


@dataclass
class PrefillResult:
    logits: Any  # (n, vocab) for the positions in `logit_positions`
    logit_positions: Any
    keys: list[Any]  # full K per layer for the whole request (1, H, N, D)
    values: list[Any]
    stats: dict = field(default_factory=dict)
    copy: Callable[[Any], Any] = lambda a: a  # backend copy, so stored chunks don't pin request buffers

    def chunk(self, start: int, end: int) -> ChunkKV:
        return ChunkKV(
            keys=[self.copy(k[:, :, start:end, :]) for k in self.keys],
            values=[self.copy(v[:, :, start:end, :]) for v in self.values],
            start=start,
        )


def plan_positions(n: int, placements: list[Placement], n_sink: int):
    """Per-position flags as python lists: cached, exact, sink."""
    cached, exact, sink = [False] * n, [False] * n, [False] * n
    for p in placements:
        for i in range(p.start, p.start + p.chunk.length):
            cached[i] = True
            if p.exact:
                exact[i] = True
            elif i < p.start + n_sink:
                sink[i] = True
    return cached, exact, sink
