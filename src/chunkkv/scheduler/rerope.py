"""Rotary position helpers for moving cached keys to new offsets.

RoPE rotates each frequency pair by angle (position * inv_freq). Rotations
compose, so a key computed at position p can be moved to p + delta by rotating
it again by delta. Only the non-traditional (half-split) layout without
scaling is supported, which covers Qwen2/2.5 and Llama-style models.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class RopeSpec:
    dims: int
    base: float
    traditional: bool = False

    @classmethod
    def from_model(cls, model) -> "RopeSpec":
        args = model.args
        if getattr(args, "rope_scaling", None):
            raise NotImplementedError("rope_scaling is not supported in v0")
        head_dim = getattr(args, "head_dim", None) or args.hidden_size // args.num_attention_heads
        return cls(dims=head_dim, base=float(args.rope_theta), traditional=bool(args.rope_traditional))


def _inv_freq(spec: RopeSpec) -> mx.array:
    half = spec.dims // 2
    return mx.power(spec.base, -mx.arange(0, half, dtype=mx.float32) * 2.0 / spec.dims)


def _rotate(x: mx.array, angles: mx.array, spec: RopeSpec) -> mx.array:
    """Rotate x (..., L, D) by per-token angles (L, D/2)."""
    if spec.traditional:
        raise NotImplementedError("traditional RoPE layout is not supported in v0")
    cos, sin = mx.cos(angles), mx.sin(angles)
    half = spec.dims // 2
    x32 = x.astype(mx.float32)
    x1, x2 = x32[..., :half], x32[..., half : spec.dims]
    out = mx.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    if spec.dims < x.shape[-1]:
        out = mx.concatenate([out, x32[..., spec.dims :]], axis=-1)
    return out.astype(x.dtype)


def apply_rope(x: mx.array, positions: mx.array, spec: RopeSpec) -> mx.array:
    """Apply RoPE to x (B, H, L, D) at arbitrary (possibly non-contiguous) positions (L,)."""
    angles = positions.astype(mx.float32)[:, None] * _inv_freq(spec)[None, :]
    return _rotate(x, angles, spec)


def shift_keys(keys: mx.array, delta: int, spec: RopeSpec) -> mx.array:
    """Move already-rotated keys (B, H, L, D) by `delta` positions."""
    if delta == 0:
        return keys
    angles = mx.broadcast_to(delta * _inv_freq(spec)[None, :], (keys.shape[-2], spec.dims // 2))
    return _rotate(keys, angles, spec)
