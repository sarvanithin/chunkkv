"""CUDA-only checks for the PyTorch engine (skipped where torch/CUDA is missing)."""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from chunkkv.engine.base import BlendConfig, Placement  # noqa: E402
from chunkkv.engine.torch_engine import TorchEngine  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def te():
    return TorchEngine.load(MODEL, dtype=torch.float32)


@pytest.fixture(scope="module")
def parts(te):
    t = lambda s: te.tokenizer.encode(s, add_special_tokens=False)  # noqa: E731
    return dict(
        sys=t("You answer questions using the documents."),
        a=t(" Document A: The Halden reactor in Norway was shut down in 2018; its operator was IFE."),
        b=t(" Document B: The Zeta bridge opened in 1997 and spans 1.2 km across the Rana fjord."),
        q=t(" Question: Who operated the Halden reactor? Answer:"),
    )


def _kl(p, q):
    p, q = torch.log_softmax(p.float(), -1), torch.log_softmax(q.float(), -1)
    return (p.exp() * (p - q)).sum(-1).mean().item()


def test_no_cache_matches_hf_forward(te, parts):
    toks = parts["sys"] + parts["a"] + parts["b"] + parts["q"]
    with torch.inference_mode():
        ref = te.model(torch.tensor([toks], device="cuda")).logits[0]
    out = te.prefill(toks, logits_from=0)
    assert torch.allclose(out.logits, ref, atol=2e-3)


def test_same_position_reuse_is_exact(te, parts):
    toks = parts["sys"] + parts["a"] + parts["b"] + parts["q"]
    first = te.prefill(toks)
    s = len(parts["sys"])
    out = te.prefill(toks, [Placement(first.chunk(s, s + len(parts["a"])), s, exact=True)], BlendConfig())
    assert torch.allclose(out.logits, first.logits, atol=2e-3)


def test_moved_chunk_blend_beats_naive(te, parts):
    s = len(parts["sys"])
    first = te.prefill(parts["sys"] + parts["a"] + parts["b"] + parts["q"])
    ca = first.chunk(s, s + len(parts["a"]))
    cb = first.chunk(s + len(parts["a"]), s + len(parts["a"]) + len(parts["b"]))
    toks2 = parts["sys"] + parts["b"] + parts["a"] + parts["q"]
    with torch.inference_mode():
        ref = te.model(torch.tensor([toks2], device="cuda")).logits[0, -1:]
    pl = [Placement(cb, s), Placement(ca, s + len(parts["b"]))]
    full = te.prefill(toks2, pl, BlendConfig(recompute_ratio=1.0))
    naive = te.prefill(toks2, pl, BlendConfig(full_first_layer=False))
    blend = te.prefill(toks2, pl, BlendConfig(recompute_ratio=0.3))
    assert _kl(ref, full.logits) < 1e-4
    assert _kl(ref, blend.logits) < _kl(ref, naive.logits)


def test_generate_matches_hf_greedy(te, parts):
    toks = parts["sys"] + parts["a"] + parts["q"]
    ours = te.generate(te.prefill(toks), max_tokens=12)
    with torch.inference_mode():
        ref = te.model.generate(torch.tensor([toks], device="cuda"), max_new_tokens=12, do_sample=False, repetition_penalty=1.0)[0, len(toks):]
    ref = [t for t in ref.tolist() if t not in te.eos]
    assert ours == ref[: len(ours)] and len(ours) > 0
