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

import mlx.core as mx
import numpy as np

from chunkkv.engine.base import BlendConfig, ChunkKV, Placement, PrefillResult  # noqa: F401
from chunkkv.scheduler.rerope import RopeSpec, apply_rope, shift_keys


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

    def generate(self, result: "PrefillResult", max_tokens: int = 32) -> list[int]:
        """Greedy decode continuing from a prefill (uses its last-position logits and K/V)."""
        from mlx_lm.models.cache import KVCache

        n = result.keys[0].shape[2]
        cache = []
        for K, V in zip(result.keys, result.values):
            c = KVCache()
            c.state = (K, V, n)
            cache.append(c)
        eos = set(self.tokenizer.eos_token_ids)
        out: list[int] = []
        nxt = int(mx.argmax(result.logits[-1]).item())
        while len(out) < max_tokens and nxt not in eos:
            out.append(nxt)
            logits = self.model(mx.array([[nxt]]), cache=cache)
            nxt = int(mx.argmax(logits[0, -1]).item())
        return out

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
        exact = mx.zeros((N,), dtype=mx.bool_)
        sink = mx.zeros((N,), dtype=mx.bool_)
        idx = mx.arange(N)
        for p in placements:
            end = p.start + p.chunk.length
            span = (idx >= p.start) & (idx < end)
            cached = cached | span
            if p.exact:
                exact = exact | span
            else:
                sink = sink | (span & (idx < p.start + config.n_sink))
        cached_np = [bool(c) for c in cached.tolist()]
        exact_np = [bool(c) for c in exact.tolist()]
        fresh_idx = mx.array([i for i in range(N) if not cached_np[i]], dtype=mx.int32)
        layer0_idx = mx.array([i for i in range(N) if not exact_np[i]], dtype=mx.int32)

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
        active = layer0_idx if config.full_first_layer else fresh_idx
        h = emb[:, active, :]
        keys_out, values_out = [], []
        selected_cached = 0
        chunk_stats: list[dict] = []

        for li, layer in enumerate(self.layers):
            attn = layer.self_attn
            x = layer.input_layernorm(h)
            q, k, v = self._qkv(attn, x, active)
            K, V = base_kv(li, k, v)

            if li == 1 and placements and config.full_first_layer:
                # Each chunk contributes a token budget; escalated chunks are recomputed
                # whole, and the pooled budget goes to the most-deviating tokens of the rest.
                # diff/norm are indexed by absolute position; exact chunks stay at zero.
                d_act = mx.sum((k - K[:, :, active, :]).astype(mx.float32) ** 2, axis=(0, 1, 3))
                d_act = d_act + mx.sum((v - V[:, :, active, :]).astype(mx.float32) ** 2, axis=(0, 1, 3))
                n_act = mx.sum(k.astype(mx.float32) ** 2, axis=(0, 1, 3)) + mx.sum(v.astype(mx.float32) ** 2, axis=(0, 1, 3))
                diff = mx.zeros((N,), dtype=mx.float32)
                norm = mx.zeros((N,), dtype=mx.float32)
                diff[active] = d_act
                norm[active] = n_act
                keep = (~cached) | sink
                pool = mx.zeros((N,), dtype=mx.bool_)
                budget = 0
                for p in placements:
                    if p.exact:
                        continue
                    end = p.start + p.chunk.length
                    rel = (mx.sum(diff[p.start : end]) / mx.sum(norm[p.start : end])).item()
                    ratio = config.recompute_ratio if p.ratio is None else p.ratio
                    escalated = config.escalate_threshold is not None and rel > config.escalate_threshold
                    if escalated:
                        keep[p.start : end] = True
                    elif ratio > 0:
                        pool[p.start : end] = True
                        budget += int(round(ratio * p.chunk.length))
                    chunk_stats.append({"start": p.start, "length": p.chunk.length, "rel_dev": rel, "escalated": escalated})
                # Sinks are already kept, so cap the budget at what is left to select.
                budget = min(budget, mx.sum(pool & ~keep).item())
                if budget > 0:
                    top = mx.argsort(-mx.where(pool & ~keep, diff, -1.0))[:budget]
                    keep[top] = True
                keep_list = [i for i, kk in enumerate(keep.tolist()) if kk]
                selected_cached = sum(1 for i in keep_list if cached_np[i])
                local = {pos: j for j, pos in enumerate(active.tolist())}
                sel = mx.array([local[i] for i in keep_list], dtype=mx.int32)
                h, q, k, v, x = h[:, sel], q[:, :, sel], k[:, :, sel], v[:, :, sel], x[:, sel]
                active = mx.array(keep_list, dtype=mx.int32)

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
        layer0 = len(layer0_idx) if config.full_first_layer else N - n_cached
        rest = (N - n_cached + selected_cached) * (n_layers - 1)
        stats = {
            "tokens": N,
            "cached_tokens": n_cached,
            "recomputed_cached_tokens": selected_cached,
            # token-layer units, normalised so a full prefill == N
            "prefill_token_equiv": (layer0 + rest) / n_layers,
            "chunks": chunk_stats,
        }
        return PrefillResult(logits, mx.arange(logits_from, N), keys_out, values_out, stats, copy=mx.contiguous)

    # -- backend-neutral helpers used by the bench --------------------------

    def last_logprobs(self, result: PrefillResult) -> np.ndarray:
        logits = result.logits[-1].astype(mx.float32)
        logp = logits - mx.logsumexp(logits)
        return np.array(logp)

    def clear_cache(self) -> None:
        mx.clear_cache()
