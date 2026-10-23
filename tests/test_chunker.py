from chunkkv.proxy.chunker import Chunker, ChunkerConfig
from chunkkv.types import TrustLevel

DOC = "The service retries failed jobs three times with exponential backoff. " * 6


def test_matches_chat_template(tok):
    msgs = [
        {"role": "system", "content": "You are a helpful agent."},
        {"role": "user", "content": DOC},
        {"role": "assistant", "content": "Done."},
    ]
    ref = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    assert Chunker(tok).chunk(msgs, "t1").tokens == ref


def test_same_content_same_tokens_at_any_position(tok):
    c = Chunker(tok)
    r1 = c.chunk([{"role": "system", "content": "A"}, {"role": "tool", "content": DOC}], "t1")
    r2 = c.chunk(
        [{"role": "system", "content": "B, longer"}, {"role": "user", "content": "hi"}, {"role": "tool", "content": DOC}],
        "t1",
    )
    s1 = [s for s in r1.segments if s.reusable]
    s2 = [s for s in r2.segments if s.reusable]
    assert s1[0].tokens == s2[0].tokens and s1[0].start != s2[0].start
    assert s1[0].trust is TrustLevel.TOOL


def test_long_content_is_windowed(tok):
    r = Chunker(tok, ChunkerConfig(window=32, min_tokens=8)).chunk([{"role": "tool", "content": DOC * 3}], "t1")
    assert sum(s.reusable for s in r.segments) >= 3


def test_context_trust_before(tok):
    r = Chunker(tok).chunk(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}, {"role": "tool", "content": DOC}], "t1"
    )
    tool = next(s for s in r.segments if s.reusable)
    assert r.context_trust_before(tool.start) is TrustLevel.USER
    assert r.context_trust_before(0) is TrustLevel.SYSTEM
