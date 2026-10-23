import mlx.core as mx
import mlx.nn as nn

from chunkkv.scheduler.rerope import RopeSpec, apply_rope, shift_keys

SPEC = RopeSpec(dims=64, base=1e6)
ROPE = nn.RoPE(64, traditional=False, base=1e6)


def test_apply_rope_matches_mlx_rope():
    x = mx.random.normal((1, 2, 10, 64))
    assert mx.allclose(ROPE(x, offset=5), apply_rope(x, mx.arange(5, 15), SPEC), atol=1e-5)


def test_shift_moves_keys_to_new_offset():
    x = mx.random.normal((1, 2, 10, 64))
    assert mx.allclose(shift_keys(ROPE(x, offset=5), 37, SPEC), ROPE(x, offset=42), atol=1e-4)


def test_shift_round_trip():
    k = ROPE(mx.random.normal((1, 2, 10, 64)), offset=3)
    assert mx.allclose(shift_keys(shift_keys(k, 37, SPEC), -37, SPEC), k, atol=1e-5)


def test_zero_shift_is_identity():
    k = mx.random.normal((1, 2, 4, 64))
    assert shift_keys(k, 0, SPEC) is k
