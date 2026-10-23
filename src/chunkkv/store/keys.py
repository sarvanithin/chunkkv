"""Cache keys.

The lookup key is content + model, plus tenant when provenance is on. Context
(where and after what the chunk was computed) is *not* part of the lookup key:
a chunk computed under another agent's context should still be found, so the
gate can decide whether it is safe to serve. Context lives on the entry.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


def token_hash(tokens: list[int]) -> str:
    h = hashlib.sha256()
    h.update(len(tokens).to_bytes(4, "little"))
    for t in tokens:
        h.update(t.to_bytes(4, "little"))
    return h.hexdigest()


@dataclass(frozen=True)
class LookupKey:
    content: str
    model_id: str
    tenant_id: str | None  # None when provenance keys are disabled (P2/P3)

    @classmethod
    def build(cls, tokens: list[int], model_id: str, tenant_id: str | None) -> "LookupKey":
        return cls(token_hash(tokens), model_id, tenant_id)
