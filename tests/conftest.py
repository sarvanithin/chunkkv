import mlx.core as mx
import pytest

from chunkkv.engine.mlx_engine import MLXEngine

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-bf16"


@pytest.fixture(scope="session")
def eng():
    return MLXEngine.load(MODEL, dtype=mx.float32)


@pytest.fixture(scope="session")
def tok(eng):
    return eng.tokenizer
