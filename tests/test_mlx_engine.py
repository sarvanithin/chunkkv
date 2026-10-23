import mlx.core as mx
import pytest

from chunkkv.engine.mlx_engine import BlendConfig, MLXEngine, Placement

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-bf16"


@pytest.fixture(scope="module")
def eng():
    return MLXEngine.load(MODEL, dtype=mx.float32)


@pytest.fixture(scope="module")
def parts(eng):
    t = eng.tokenizer.encode
    return dict(
        sys=t("You answer questions using the documents."),
        a=t(" Document A: The Halden reactor in Norway was shut down in 2018; its operator was IFE."),
        b=t(" Document B: The Zeta bridge opened in 1997 and spans 1.2 km across the Rana fjord."),
        q=t(" Question: Who operated the Halden reactor? Answer:"),
    )


def _kl(p_logits, q_logits):
    p = mx.softmax(p_logits, -1)
    return mx.sum(p * (mx.log(p + 1e-12) - mx.log(mx.softmax(q_logits, -1) + 1e-12)), -1).mean().item()


def test_no_cache_matches_model_forward(eng, parts):
    toks = parts["sys"] + parts["a"] + parts["b"] + parts["q"]
    ref = eng.model(mx.array(toks)[None])[0]
    out = eng.prefill(toks, logits_from=0)
    assert mx.allclose(out.logits, ref, atol=1e-3)
    assert out.stats["prefill_token_equiv"] == len(toks)


def test_same_position_reuse_is_exact(eng, parts):
    toks = parts["sys"] + parts["a"] + parts["b"] + parts["q"]
    first = eng.prefill(toks)
    s = len(parts["sys"])
    chunk = first.chunk(s, s + len(parts["a"]))
    out = eng.prefill(toks, [Placement(chunk, s)], BlendConfig(full_first_layer=False))
    assert mx.allclose(out.logits, first.logits, atol=1e-3)
    assert out.stats["prefill_token_equiv"] == len(toks) - len(parts["a"])


def test_moved_chunk_blend_beats_naive(eng, parts):
    s = len(parts["sys"])
    first = eng.prefill(parts["sys"] + parts["a"] + parts["b"] + parts["q"])
    chunk_a = first.chunk(s, s + len(parts["a"]))
    chunk_b = first.chunk(s + len(parts["a"]), s + len(parts["a"]) + len(parts["b"]))
    toks2 = parts["sys"] + parts["b"] + parts["a"] + parts["q"]
    ref = eng.model(mx.array(toks2)[None])[0, -1:]
    pl = [Placement(chunk_b, s), Placement(chunk_a, s + len(parts["b"]))]

    full = eng.prefill(toks2, pl, BlendConfig(recompute_ratio=1.0))
    naive = eng.prefill(toks2, pl, BlendConfig(full_first_layer=False))
    blend = eng.prefill(toks2, pl, BlendConfig(recompute_ratio=0.3))

    assert _kl(ref, full.logits) < 1e-4
    assert _kl(ref, blend.logits) < _kl(ref, naive.logits)
    assert naive.stats["prefill_token_equiv"] < blend.stats["prefill_token_equiv"] < len(toks2)
