from chunkkv.regress.suite import run


def test_non_prefix_reuse_still_works(eng):
    r = run(eng)
    assert r["expected"] > 0
    assert r["non_prefix_hit_rate"] > 0.8
