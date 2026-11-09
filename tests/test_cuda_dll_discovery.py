"""NVIDIA runtime DLL discovery for the CUDA OCR backend.

ONNX Runtime only preloads the legacy ``nvidia/<library>/bin`` wheel layout.
The CUDA 13 wheels install ``cublasLt64_13.dll`` / ``cudart64_13.dll`` under
``nvidia/cu13/bin/x86_64`` instead, so the provider failed to load on every
machine that had a CUDA 13 wheel set.  These tests pin the directory discovery
for both layouts.
"""

from pathlib import Path

import pytest

from core.ocr import RapidOcrBackend


def make_library(root: Path, *relative: str) -> Path:
    directory = root.joinpath(*relative)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def test_legacy_cuda12_layout_is_discovered(tmp_path):
    root = tmp_path / "nvidia"
    cublas = make_library(root, "cublas", "bin")
    cudart = make_library(root, "cuda_runtime", "bin")
    cudnn = make_library(root, "cudnn", "bin")

    directories = RapidOcrBackend._nvidia_dll_directories(root)

    assert cublas in directories
    assert cudart in directories
    assert cudnn in directories


def test_cuda13_layout_is_discovered(tmp_path):
    root = tmp_path / "nvidia"
    cu13 = make_library(root, "cu13", "bin", "x86_64")
    cudnn = make_library(root, "cudnn", "bin")

    directories = RapidOcrBackend._nvidia_dll_directories(root)

    assert cu13 in directories, "CUDA 13 wheels install into nvidia/cu13/bin/<arch>"
    assert cudnn in directories


def test_discovery_ignores_missing_and_deduplicates(tmp_path):
    root = tmp_path / "nvidia"
    make_library(root, "cublas", "bin")
    (root / "not-a-directory.txt").write_text("x", encoding="utf-8")

    directories = RapidOcrBackend._nvidia_dll_directories(root)

    assert directories
    assert len(directories) == len(set(directories))
    assert all(directory.is_dir() for directory in directories)


def test_discovery_of_absent_root_is_empty(tmp_path):
    assert RapidOcrBackend._nvidia_dll_directories(tmp_path / "missing") == []


def test_cuda_dll_directories_include_pip_wheels(tmp_path, monkeypatch):
    root = tmp_path / "site-packages" / "nvidia"
    cu13 = make_library(root, "cu13", "bin", "x86_64")
    cudnn = make_library(root, "cudnn", "bin")
    monkeypatch.setattr(
        RapidOcrBackend, "_nvidia_package_roots", classmethod(lambda cls: [root])
    )
    monkeypatch.delenv("CUDA_PATH", raising=False)

    directories = RapidOcrBackend._cuda_dll_directories()

    assert cu13 in directories
    assert cudnn in directories


def test_prepare_cuda_runtime_registers_every_directory(tmp_path, monkeypatch):
    directory = make_library(tmp_path / "nvidia", "cu13", "bin", "x86_64")
    monkeypatch.setattr(
        RapidOcrBackend,
        "_cuda_dll_directories",
        classmethod(lambda cls: [directory]),
    )
    registered = []
    monkeypatch.setattr(
        "core.ocr.os.add_dll_directory", lambda path: registered.append(path), raising=False
    )

    calls = []

    class FakeOrt:
        @staticmethod
        def preload_dlls(*, cuda, cudnn, directory=""):
            calls.append((cuda, cudnn, directory))

    backend = RapidOcrBackend(prefer_gpu=True)
    backend._prepare_cuda_runtime(FakeOrt)

    assert str(directory) in registered
    assert calls, "CUDA DLLs should be preloaded once directories are known"
    backend._dll_directory_handles.clear()


def test_prepare_cuda_runtime_survives_preload_failure(tmp_path, monkeypatch):
    directory = make_library(tmp_path / "nvidia", "cudnn", "bin")
    monkeypatch.setattr(
        RapidOcrBackend,
        "_cuda_dll_directories",
        classmethod(lambda cls: [directory]),
    )
    monkeypatch.setattr(
        "core.ocr.os.add_dll_directory", lambda path: object(), raising=False
    )

    class BrokenOrt:
        @staticmethod
        def preload_dlls(**kwargs):
            raise RuntimeError("CUDA runtime missing")

    backend = RapidOcrBackend(prefer_gpu=True)
    # A preload failure must never raise: the session probe decides the fallback.
    backend._prepare_cuda_runtime(BrokenOrt)
    backend._dll_directory_handles.clear()


def test_cuda_dll_directories_empty_off_windows(monkeypatch):
    monkeypatch.setattr("core.ocr.os.name", "posix")
    assert RapidOcrBackend._cuda_dll_directories() == []


def test_nvidia_package_roots_are_directories():
    for root in RapidOcrBackend._nvidia_package_roots():
        assert isinstance(root, Path)
        assert root.is_dir()
        assert root.name == "nvidia"


@pytest.mark.parametrize("layout", ["cublas/bin", "cu13/bin/x86_64"])
def test_every_bin_depth_is_scanned(tmp_path, layout):
    root = tmp_path / "nvidia"
    expected = make_library(root, *layout.split("/"))
    assert expected in RapidOcrBackend._nvidia_dll_directories(root)
