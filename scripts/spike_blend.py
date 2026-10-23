"""Phase-2 spike: is injected/blended KV correct, and how much does reuse cost in quality?"""
import sys
import mlx.core as mx
from chunkkv.engine.mlx_engine import MLXEngine, Placement, BlendConfig

repo = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Qwen2.5-0.5B-Instruct-bf16"
eng = MLXEngine.load(repo, dtype=mx.float32)
tok = eng.tokenizer.encode

docA = tok(" Document A: The Halden reactor in Norway was shut down in 2018 after a fuel leak; its operator was IFE.")
docB = tok(" Document B: The Zeta bridge opened in 1997 and spans 1.2 km across the Rana fjord, built by Aas AS.")
sys_ = tok("You answer questions using the documents.")
q = tok(" Question: Who operated the Halden reactor, and when did it close? Answer:")

def kl(p_logits, q_logits):
    p = mx.softmax(p_logits, -1); return mx.sum(p * (mx.log(p + 1e-12) - mx.log(mx.softmax(q_logits, -1) + 1e-12)), -1).mean().item()

# 1. No cache == model forward
toks = sys_ + docA + docB + q
ref = eng.model(mx.array(toks)[None])[0]
out = eng.prefill(toks, logits_from=0)
print("P0 vs model():", mx.abs(out.logits - ref).max().item())

# 2. Same position + same prefix, full reuse, no repair -> exact
first = eng.prefill(sys_ + docA + docB + q)
a0 = len(sys_); chunkA = first.chunk(a0, a0 + len(docA))
o = eng.prefill(toks, [Placement(chunkA, a0)], BlendConfig(full_first_layer=False))
print("same-pos reuse vs P0 last-logit:", mx.abs(o.logits - ref[-1:]).max().item())

# 3. Moved chunk: request 2 has docB before docA
toks2 = sys_ + docB + docA + q
ref2 = eng.model(mx.array(toks2)[None])[0, -1:]
startA = len(sys_) + len(docB)
chunkB = first.chunk(a0 + len(docA), a0 + len(docA) + len(docB))
pl = [Placement(chunkA, startA), Placement(chunkB, len(sys_))]
for name, cfg in [("full recompute", BlendConfig(recompute_ratio=1.0)),
                  ("naive", BlendConfig(full_first_layer=False)),
                  ("blend 0%", BlendConfig(recompute_ratio=0.0)),
                  ("blend 15%", BlendConfig(recompute_ratio=0.15)),
                  ("blend 30%", BlendConfig(recompute_ratio=0.30))]:
    o = eng.prefill(toks2, pl, cfg)
    print(f"{name:15s} KL={kl(ref2, o.logits):.4f} argmax_same={int(mx.argmax(o.logits,-1).item()==mx.argmax(ref2,-1).item())} compute={o.stats['prefill_token_equiv']:.1f}/{len(toks2)}")
