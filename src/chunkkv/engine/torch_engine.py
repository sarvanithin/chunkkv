"""PyTorch/CUDA engine with non-prefix KV reuse and selective recompute.

Same algorithm as `mlx_engine` (see its docstring), over Hugging Face
Qwen2/Llama-style weights. Used on NVIDIA GPUs, where vLLM + LMCache blending
is currently broken (vLLM 0.32 packed KV layout is unsupported by LMCache's
layerwise/blend path).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from chunkkv.engine.base import BlendConfig, ChunkKV, Placement, PrefillResult, plan_positions  # noqa: F401


class TorchEngine:
    def __init__(self, model, tokenizer):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.inner = model.model
        self.layers = self.inner.layers
        cfg = model.config
        scaling = getattr(cfg, "rope_scaling", None) or {}
        if scaling.get("rope_type", scaling.get("type", "default")) != "default":
            raise NotImplementedError("rope_scaling is not supported in v0")
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        base = getattr(cfg, "rope_theta", None) or scaling.get("rope_theta") or cfg.rope_parameters["rope_theta"]
        self.device = next(model.parameters()).device
        self.inv_freq = base ** (-torch.arange(0, self.head_dim, 2, device=self.device, dtype=torch.float32) / self.head_dim)
        eos = model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos]) | {tokenizer.eos_token_id}

    @classmethod
    def load(cls, repo: str, dtype=torch.bfloat16) -> "TorchEngine":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype).to("cuda")
        return cls(model, AutoTokenizer.from_pretrained(repo))

    # -- helpers ---------------------------------------------------------

    def _rotate(self, x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        cos, sin = angles.cos(), angles.sin()
        half = self.head_dim // 2
        x32 = x.float()
        x1, x2 = x32[..., :half], x32[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)

    def _rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self._rotate(x, positions.float()[:, None] * self.inv_freq[None, :])

    def shift_keys(self, keys: torch.Tensor, delta: int) -> torch.Tensor:
        if delta == 0:
            return keys
        return self._rotate(keys, (delta * self.inv_freq)[None, :].expand(keys.shape[-2], -1))

    def _qkv(self, attn, x, positions):
        B, L, _ = x.shape
        q = attn.q_proj(x).view(B, L, self.n_heads, -1).transpose(1, 2)
        k = attn.k_proj(x).view(B, L, self.n_kv, -1).transpose(1, 2)
        v = attn.v_proj(x).view(B, L, self.n_kv, -1).transpose(1, 2)
        return self._rope(q, positions), self._rope(k, positions), v

    def _attend(self, layer, h, q, K, V, mask):
        attn = layer.self_attn
        out = F.scaled_dot_product_attention(q, K, V, attn_mask=mask, enable_gqa=True)
        out = out.transpose(1, 2).reshape(h.shape[0], h.shape[1], -1)
        h = h + attn.o_proj(out)
        return h + layer.mlp(layer.post_attention_layernorm(h))

    def _logits(self, h):
        return self.model.lm_head(self.inner.norm(h))

    # -- main entry ------------------------------------------------------

    @torch.inference_mode()
    def prefill(self, tokens, placements=None, config=None, logits_from=None) -> PrefillResult:
        placements = placements or []
        config = config or BlendConfig()
        N = len(tokens)
        n_layers = len(self.layers)
        logits_from = N - 1 if logits_from is None else logits_from
        dev = self.device

        cached_l, exact_l, sink_l = plan_positions(N, placements, config.n_sink)
        cached = torch.tensor(cached_l, device=dev)
        sink = torch.tensor(sink_l, device=dev)
        fresh_idx = torch.tensor([i for i in range(N) if not cached_l[i]], device=dev, dtype=torch.long)
        layer0_idx = torch.tensor([i for i in range(N) if not exact_l[i]], device=dev, dtype=torch.long)
        ar = torch.arange(N, device=dev)

        def base_kv(li, dtype):
            K = torch.zeros((1, self.n_kv, N, self.head_dim), device=dev, dtype=dtype)
            V = torch.zeros_like(K)
            for p in placements:
                end = p.start + p.chunk.length
                K[:, :, p.start:end] = self.shift_keys(p.chunk.keys[li].to(dtype), p.start - p.chunk.start)
                V[:, :, p.start:end] = p.chunk.values[li].to(dtype)
            return K, V

        emb = self.inner.embed_tokens(torch.tensor([tokens], device=dev))
        active = layer0_idx if config.full_first_layer else fresh_idx
        h = emb[:, active]
        keys_out, values_out = [], []
        selected_cached = 0
        chunk_stats: list[dict] = []

        for li, layer in enumerate(self.layers):
            x = layer.input_layernorm(h)
            q, k, v = self._qkv(layer.self_attn, x, active)
            K, V = base_kv(li, k.dtype)

            if li == 1 and placements and config.full_first_layer:
                d_act = ((k - K[:, :, active]).float() ** 2).sum(dim=(0, 1, 3))
                d_act += ((v - V[:, :, active]).float() ** 2).sum(dim=(0, 1, 3))
                n_act = (k.float() ** 2).sum(dim=(0, 1, 3)) + (v.float() ** 2).sum(dim=(0, 1, 3))
                diff = torch.zeros(N, device=dev)
                norm = torch.zeros(N, device=dev)
                diff[active] = d_act
                norm[active] = n_act
                keep = (~cached) | sink
                pool = torch.zeros(N, dtype=torch.bool, device=dev)
                budget = 0
                for p in placements:
                    if p.exact:
                        continue
                    end = p.start + p.chunk.length
                    rel = (diff[p.start:end].sum() / norm[p.start:end].sum()).item()
                    ratio = config.recompute_ratio if p.ratio is None else p.ratio
                    escalated = config.escalate_threshold is not None and rel > config.escalate_threshold
                    if escalated:
                        keep[p.start:end] = True
                    elif ratio > 0:
                        pool[p.start:end] = True
                        budget += int(round(ratio * p.chunk.length))
                    chunk_stats.append({"start": p.start, "length": p.chunk.length, "rel_dev": rel, "escalated": escalated})
                # Sinks are already kept, so cap the budget at what is left to select.
                budget = min(budget, int((pool & ~keep).sum().item()))
                if budget > 0:
                    top = torch.argsort(-torch.where(pool & ~keep, diff, torch.full_like(diff, -1.0)))[:budget]
                    keep[top] = True
                keep_abs = keep.nonzero().squeeze(1)
                selected_cached = int(cached[keep_abs].sum().item())
                local = torch.full((N,), -1, dtype=torch.long, device=dev)
                local[active] = torch.arange(len(active), device=dev)
                sel = local[keep_abs]
                assert bool((sel >= 0).all()), "selected a token that was never computed"
                h, q, k, v = h[:, sel], q[:, :, sel], k[:, :, sel], v[:, :, sel]
                active = keep_abs

            K[:, :, active] = k
            V[:, :, active] = v
            mask = active[:, None] >= ar[None, :]
            h = self._attend(layer, h, q, K, V, mask)
            keys_out.append(K)
            values_out.append(V)

        want = (active >= logits_from).nonzero().squeeze(1)
        if len(want) != N - logits_from:
            raise ValueError("logits requested for positions that were not recomputed")
        logits = self._logits(h[:, want])[0]

        n_cached = sum(cached_l)
        layer0 = len(layer0_idx) if config.full_first_layer else N - n_cached
        rest = (N - n_cached + selected_cached) * (n_layers - 1)
        stats = {
            "tokens": N,
            "cached_tokens": n_cached,
            "recomputed_cached_tokens": selected_cached,
            "prefill_token_equiv": (layer0 + rest) / n_layers,
            "chunks": chunk_stats,
        }
        return PrefillResult(logits, torch.arange(logits_from, N), keys_out, values_out, stats,
                             copy=lambda t: t.contiguous())

    @torch.inference_mode()
    def generate(self, result: PrefillResult, max_tokens: int = 32) -> list[int]:
        Ks, Vs = list(result.keys), list(result.values)
        pos = Ks[0].shape[2]
        out: list[int] = []
        nxt = int(result.logits[-1].argmax().item())
        while len(out) < max_tokens and nxt not in self.eos:
            out.append(nxt)
            h = self.inner.embed_tokens(torch.tensor([[nxt]], device=self.device))
            p = torch.tensor([pos], device=self.device)
            for li, layer in enumerate(self.layers):
                x = layer.input_layernorm(h)
                q, k, v = self._qkv(layer.self_attn, x, p)
                Ks[li] = torch.cat([Ks[li], k], dim=2)
                Vs[li] = torch.cat([Vs[li], v], dim=2)
                h = self._attend(layer, h, q, Ks[li], Vs[li], None)
            nxt = int(self._logits(h)[0, -1].argmax().item())
            pos += 1
        return out

    def last_logprobs(self, result: PrefillResult) -> np.ndarray:
        return torch.log_softmax(result.logits[-1].float(), -1).cpu().numpy()

    def sync(self) -> None:
        torch.cuda.synchronize()

    def clear_cache(self) -> None:
        torch.cuda.empty_cache()
