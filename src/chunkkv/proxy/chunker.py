"""Split a chat request into reusable segments.

Each message is rendered in ChatML as three pieces, header / content / footer:

    <|im_start|>{role}\\n  |  {content}  |  <|im_end|>\\n

and each piece is tokenized on its own. The header and footer start and end
with special tokens, so BPE never merges across a piece boundary and the
content tokens of a message are the same wherever the message appears. The
content is the reusable segment; headers and footers are short and always
prefilled fresh. Long contents are split into fixed token windows so a long
tool result can still hit when only part of it repeats.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from chunkkv.types import ChunkedRequest, Segment, TrustLevel

ROLE_TRUST = {
    "system": TrustLevel.SYSTEM,
    "tool": TrustLevel.TOOL,
    "user": TrustLevel.USER,
    "assistant": TrustLevel.TOOL,
}


@dataclass
class ChunkerConfig:
    window: int = 256  # fallback window size for long contents
    min_tokens: int = 32  # shorter contents are not worth caching
    trust_overrides: dict[str, TrustLevel] | None = None


def context_id(messages: list[dict], agent_id: str | None) -> str:
    """Semantic context fingerprint: system prompt + agent/task id."""
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    h = hashlib.sha256()
    h.update(system.encode())
    h.update(b"\x00")
    h.update((agent_id or "").encode())
    return h.hexdigest()[:16]


class Chunker:
    def __init__(self, tokenizer, config: ChunkerConfig | None = None):
        self.tok = tokenizer
        self.config = config or ChunkerConfig()

    def _encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def chunk(
        self,
        messages: list[dict],
        tenant_id: str,
        agent_id: str | None = None,
        add_generation_prompt: bool = True,
    ) -> ChunkedRequest:
        cfg = self.config
        trust_map = {**ROLE_TRUST, **(cfg.trust_overrides or {})}
        tokens: list[int] = []
        segments: list[Segment] = []

        def add(toks: list[int], role: str, reusable: bool):
            segments.append(Segment(len(segments), role, trust_map[role], len(tokens), toks, reusable))
            tokens.extend(toks)

        for m in messages:
            role = m["role"]
            add(self._encode(f"<|im_start|>{role}\n"), role, False)
            content = self._encode(m["content"])
            if len(content) < cfg.min_tokens:
                add(content, role, False)
            else:
                for i in range(0, len(content), cfg.window):
                    piece = content[i : i + cfg.window]
                    add(piece, role, len(piece) >= cfg.min_tokens)
            add(self._encode("<|im_end|>\n"), role, False)
        if add_generation_prompt:
            add(self._encode("<|im_start|>assistant\n"), "assistant", False)

        return ChunkedRequest(tokens, segments, tenant_id, context_id(messages, agent_id))
