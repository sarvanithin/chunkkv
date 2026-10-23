"""Shared types for ChunkKV."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum


class TrustLevel(IntEnum):
    """Who controls a piece of content. Higher = more trusted."""

    USER = 0
    TOOL = 1
    SYSTEM = 2


class GateAction(str, Enum):
    REUSE = "reuse"  # serve cached K/V as-is (exact prefix context)
    BLEND = "blend"  # serve with selective recompute
    RECOMPUTE = "recompute"  # do not serve; prefill from scratch


@dataclass
class Segment:
    """A contiguous span of request tokens that may be cached and reused."""

    index: int
    role: str
    trust: TrustLevel
    start: int  # position in the request's token sequence
    tokens: list[int]
    reusable: bool = True

    @property
    def end(self) -> int:
        return self.start + len(self.tokens)


@dataclass
class ChunkedRequest:
    tokens: list[int]
    segments: list[Segment]
    tenant_id: str
    context_id: str  # semantic fingerprint (system prompt + agent/task id)

    def context_trust_before(self, pos: int) -> TrustLevel:
        """Least-trusted level among all content strictly before `pos`."""
        levels = [s.trust for s in self.segments if s.start < pos]
        return min(levels) if levels else TrustLevel.SYSTEM


@dataclass
class GateDecision:
    segment: int
    action: GateAction
    reason: str
    ratio: float = 0.0
    entry_id: str | None = None
    extra: dict = field(default_factory=dict)
