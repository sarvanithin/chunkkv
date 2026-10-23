"""MLX engine with non-prefix KV reuse and selective recompute.

The forward pass walks the decoder layers by hand. Each layer computes Q/K/V
only for an *active* set of token positions; every other position reads its
K/V from cached chunks that have been moved to their new offsets. This is the
CacheBlend scheme:

  - layer 0 runs on every token (cheap; it repairs nothing on its own but
    gives every token a fresh layer-1 input),
  - at layer 1 the fresh K/V of cached tokens are compared with the cached
    K/V and the most-deviating fraction (plus the first `n_sink` tokens of each
    chunk) joins the active set,
  - layers 1..L-1 run on that active set only.

With nothing cached, every token is active at every layer and the result is
an ordinary prefill. Only Qwen2-style blocks (q/k/v/o projections, RMSNorm,
SwiGLU MLP, standard RoPE) are supported in v0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx

from chunkkv.scheduler.rerope import RopeSpec, apply_rope, shift_keys


@dataclass
class ChunkKV:
    """Per-layer K/V for one chunk, as computed at positions start..start+len."""

    keys: list[mx.array]  # each (1, n_kv_heads, L, head_dim), RoPE applied
    values: list[mx.array]
    start: int

    @property
    def length(self) -> int:
        return self.keys[0].shape[2]


@dataclass
class Placement:
    """A cached chunk placed at a new offset in the request."""

    chunk: ChunkKV
    start: int


@dataclass
class BlendConfig:
    recompute_ratio: float = 0.15  # fraction of cached tokens recomputed at layers >= 1
    n_sink: int = 2  # first tokens of each chunk always recomputed
    full_first_layer: bool = True  # False = naive reuse (no repair at all)


@dataclass
class PrefillResult:
    logits: mx.array  # (n_active_last, vocab) for the positions in `logit_positions`
    logit_positions: mx.array
    keys: list[mx.array]  # full K per layer for the whole request (1, H, N, D)
    values: list[mx.array]
    stats: dict = field(default_factory=dict)

    def chunk(self, start: int, end: int) -> ChunkKV:
        return ChunkKV(
            keys=[k[:, :, start:end, :] for k in self.keys],
            values=[v[:, :, start:end, :] for v in self.values],
            start=start,
        )


class MLXEngine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.inner = model.model
        self.layers = self.inner.layers
        self.rope = RopeSpec.from_model(model)

    @classmethod
    def load(cls, repo: str, dtype=None) -> "MLXEngine":
        from mlx_lm import load

        model, tokenizer = load(repo)
        if dtype is not None:
            model.set_dtype(dtype)
        return cls(model, tokenizer)

    # -- helpers ---------------------------------------------------------

    def _qkv(self, attn, x: mx.array, positions: mx.array):
        B, L, _ = x.shape
        q = attn.q_proj(x).reshape(B, L, attn.n_heads, -1).transpose(0, 2, 1, 3)
        k = attn.k_proj(x).reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        v = attn.v_proj(x).reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
        return apply_rope(q, positions, self.rope), apply_rope(k, positions, self.rope), v

    def _attend(self, layer, h, q, K, V, positions, n_total):
        attn = layer.self_attn
        mask = positions[:, None] >= mx.arange(n_total)[None, :]
        out = mx.fast.scaled_dot_product_attention(q, K, V, scale=attn.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(h.shape[0], h.shape[1], -1)
        h = h + attn.o_proj(out)
        return h + layer.mlp(layer.post_attention_layernorm(h))

    def _logits(self, h: mx.array) -> mx.array:
        h = self.inner.norm(h)
        if self.model.args.tie_word_embeddings:
            return self.inner.embed_tokens.as_linear(h)
        return self.model.lm_head(h)

    # -- main entry ------------------------------------------------------

    def prefill(
        self,
        tokens: list[int],
        placements: list[Placement] | None = None,
        config: BlendConfig | None = None,
        logits_from: int | None = None,
    ) -> PrefillResult:
        """Prefill `tokens`, reusing K/V for `placements` where allowed.

        Returns logits for positions >= `logits_from` (default: last token only).
        """
        placements = placements or []
        config = config or BlendConfig()
        N = len(tokens)
        n_layers = len(self.layers)
        logits_from = N - 1 if logits_from is None else logits_from

        cached = mx.zeros((N,), dtype=mx.bool_)
        sink = mx.zeros((N,), dtype=mx.bool_)
        idx = mx.arange(N)
        for p in placements:
            end = p.start + p.chunk.length
            span = (idx >= p.start) & (idx < end)
            cached = cached | span
            sink = sink | (span & (idx < p.start + config.n_sink))
        cached_np = [bool(c) for c in cached.tolist()]
        fresh_idx = mx.array([i for i in range(N) if not cached_np[i]], dtype=mx.int32)
        all_idx = mx.arange(N, dtype=mx.int32)

        # Base K/V per layer: cached chunks moved to their new offsets.
        def base_kv(layer_i: int, like_k: mx.array, like_v: mx.array):
            K = mx.zeros((1, like_k.shape[1], N, like_k.shape[3]), dtype=like_k.dtype)
            V = mx.zeros((1, like_v.shape[1], N, like_v.shape[3]), dtype=like_v.dtype)
            for p in placements:
                end = p.start + p.chunk.length
                K[:, :, p.start : end, :] = shift_keys(
                    p.chunk.keys[layer_i].astype(like_k.dtype), p.start - p.chunk.start, self.rope
                )
                V[:, :, p.start : end, :] = p.chunk.values[layer_i].astype(like_v.dtype)
            return K, V

        emb = self.inner.embed_tokens(mx.array(tokens)[None])
        active = all_idx if (config.full_first_layer or not placements) else fresh_idx
        h = emb[:, active, :]
        keys_out, values_out = [], []
        selected_cached = 0

        for li, layer in enumerate(self.layers):
            attn = layer.self_attn
            x = layer.input_layernorm(h)
            q, k, v = self._qkv(attn, x, active)
            K, V = base_kv(li, k, v)

            if li == 1 and placements and config.full_first_layer:
                # Select the cached tokens whose fresh layer-1 K/V deviate most.
                dev = mx.sum((k - K[:, :, active, :]).astype(mx.float32) ** 2, axis=(0, 1, 3))
                dev = dev + mx.sum((v - V[:, :, active, :]).astype(mx.float32) ** 2, axis=(0, 1, 3))
                n_cached = sum(cached_np)
                budget = int(round(config.recompute_ratio * n_cached))
                keep = (~cached) | sink
                if budget > 0:
                    dev_c = mx.where(cached & ~sink, dev, -1.0)
                    top = mx.argsort(-dev_c)[:budget]
                    keep[top] = True
                keep_list = [i for i, kk in enumerate(keep.tolist()) if kk]
                selected_cached = sum(1 for i in keep_list if cached_np[i])
                sel = mx.array(keep_list, dtype=mx.int32)
                h, q, k, v, x = h[:, sel], q[:, :, sel], k[:, :, sel], v[:, :, sel], x[:, sel]
                active = sel

            K[:, :, active, :] = k
            V[:, :, active, :] = v
            h = self._attend(layer, h, q, K, V, active, N)
            keys_out.append(K)
            values_out.append(V)

        active_list = active.tolist()
        want = [j for j, pos in enumerate(active_list) if pos >= logits_from]
        if len(want) != N - logits_from:
            raise ValueError("logits requested for positions that were not recomputed")
        logits = self._logits(h[:, mx.array(want, dtype=mx.int32)])[0]

        n_cached = sum(cached_np)
        layer0 = N if (config.full_first_layer or not placements) else N - n_cached
        rest = (N - n_cached + selected_cached) * (n_layers - 1)
        stats = {
            "tokens": N,
            "cached_tokens": n_cached,
            "recomputed_cached_tokens": selected_cached,
            # token-layer units, normalised so a full prefill == N
            "prefill_token_equiv": (layer0 + rest) / n_layers,
        }
        return PrefillResult(logits, mx.arange(logits_from, N), keys_out, values_out, stats)
