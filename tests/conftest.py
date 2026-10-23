import pytest

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-bf16"


def _has_mlx() -> bool:
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


collect_ignore = [] if _has_mlx() else [
    "test_rerope.py", "test_mlx_engine.py", "test_chunker.py", "test_runtime.py", "test_regress.py",
]


@pytest.fixture(scope="session")
def eng():
    import mlx.core as mx

    from chunkkv.engine.mlx_engine import MLXEngine

    return MLXEngine.load(MODEL, dtype=mx.float32)


@pytest.fixture(scope="session")
def tok(eng):
    return eng.tokenizer
