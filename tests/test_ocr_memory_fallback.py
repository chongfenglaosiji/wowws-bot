"""A host-memory spike must not permanently downgrade OCR to CPU.

The reporting machine logged "OCR CUDA 推理失败，后续改用 CPU: Unable to
allocate 48.0 MiB ... float64" during a map load. That was a NumPy
``MemoryError``, not a CUDA fault, yet every later read (including the match
clock) ran on CPU and the setup became about three times slower.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from core import ocr as ocr_module
from core.ocr import RapidOcrBackend


def test_memory_shortage_is_recognised():
    assert RapidOcrBackend._is_memory_shortage(MemoryError("boom"))
    assert RapidOcrBackend._is_memory_shortage(
        RuntimeError("Unable to allocate 48.0 MiB for an array")
    )
    assert not RapidOcrBackend._is_memory_shortage(RuntimeError("CUDA failure 801"))


def backend_with(engine_result=None, engine_error=None):
    backend = RapidOcrBackend(prefer_gpu=True)
    engine = Mock(side_effect=engine_error) if engine_error else Mock(return_value=engine_result)
    backend._engine = engine
    backend.execution_provider = "CUDAExecutionProvider"
    return backend, engine


def test_memory_error_keeps_the_gpu_provider():
    backend, _engine = backend_with(engine_error=MemoryError("Unable to allocate 48.0 MiB"))
    fallback = Mock()
    backend._fallback_to_cpu = fallback

    tokens = backend.recognize(np.zeros((40, 120, 3), np.uint8))

    assert tokens == []
    fallback.assert_not_called()
    assert backend.execution_provider == "CUDAExecutionProvider"


def test_memory_error_keeps_the_gpu_provider_for_glyphs():
    backend = RapidOcrBackend(prefer_gpu=True)
    engine = Mock()
    engine.text_rec.side_effect = MemoryError("Unable to allocate 48.0 MiB")
    backend._engine = engine
    backend.execution_provider = "CUDAExecutionProvider"
    fallback = Mock()
    backend._fallback_to_cpu = fallback

    assert backend.recognize_glyph(np.zeros((20, 20, 3), np.uint8)) == []
    fallback.assert_not_called()
    assert backend.execution_provider == "CUDAExecutionProvider"


def test_real_cuda_failure_still_falls_back():
    backend, _engine = backend_with(engine_error=RuntimeError("CUDA failure 801"))
    calls = []

    def fake_fallback(error):
        calls.append(error)
        backend._engine = Mock(
            return_value=SimpleNamespace(txts=[], scores=[], boxes=[])
        )

    backend._fallback_to_cpu = fake_fallback
    assert backend.recognize(np.zeros((40, 120, 3), np.uint8)) == []
    assert len(calls) == 1
