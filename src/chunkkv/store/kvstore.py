"""In-memory chunk KV store with LRU eviction by byte budget.

One lookup key can hold several entries: the same chunk computed under
different contexts (different preceding tokens, agents or trust levels).
"""

from __future__ import annotations

import itertools
from collections import OrderedDict
from dataclasses import dataclass, field

from chunkkv.engine.base import ChunkKV
from chunkkv.store.keys import LookupKey
from chunkkv.types import TrustLevel

_ids = itertools.count()


@dataclass
class Entry:
    kv: ChunkKV
    context_id: str  # semantic fingerprint of the request it was computed in
    prefix_hash: str  # hash of every token before the chunk at compute time
    context_trust: TrustLevel  # least-trusted content before the chunk at compute time
    id: str = field(default_factory=lambda: f"e{next(_ids)}")
    hits: int = 0

    @property
    def nbytes(self) -> int:
        return sum(k.nbytes + v.nbytes for k, v in zip(self.kv.keys, self.kv.values))


class KVStore:
    def __init__(self, budget_bytes: int = 4 << 30, max_variants: int = 4):
        self.budget = budget_bytes
        self.max_variants = max_variants
        self._data: OrderedDict[LookupKey, list[Entry]] = OrderedDict()
        self.bytes = 0

    def get(self, key: LookupKey) -> list[Entry]:
        entries = self._data.get(key)
        if entries:
            self._data.move_to_end(key)
        return list(entries or [])

    def put(self, key: LookupKey, entry: Entry) -> None:
        entries = self._data.setdefault(key, [])
        # Same context already stored: keep the existing entry.
        if any(e.prefix_hash == entry.prefix_hash and e.context_trust == entry.context_trust for e in entries):
            self._data.move_to_end(key)
            return
        entries.append(entry)
        self.bytes += entry.nbytes
        while len(entries) > self.max_variants:
            self.bytes -= entries.pop(0).nbytes
        self._data.move_to_end(key)
        self._evict()

    def _evict(self) -> None:
        while self.bytes > self.budget and self._data:
            _, entries = self._data.popitem(last=False)
            self.bytes -= sum(e.nbytes for e in entries)

    def __len__(self) -> int:
        return sum(len(v) for v in self._data.values())
