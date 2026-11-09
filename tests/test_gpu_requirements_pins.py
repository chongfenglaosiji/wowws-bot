"""The CUDA stack pins in requirements-gpu.txt and the repair decisions.

The OCR backend silently ran on CPU for every user whose driver could not run
the installed CUDA build, because the launcher treated "provider is listed" as
"CUDA works" and never compared the installed wheels with the pinned ones.
"""

import pytest

from tools import run_with_ocr

PINS = """
# comment line
onnxruntime-gpu[cuda,cudnn]==1.26.0
nvidia-cuda-runtime-cu12==12.2.140
nvidia-cublas-cu12==12.2.5.6
nvidia-cudnn-cu12==9.1.0.70
"""


def fake_environment(monkeypatch, tmp_path, installed):
    root = tmp_path / "root"
    root.mkdir()
    (root / "requirements-gpu.txt").write_text(PINS, encoding="utf-8")
    monkeypatch.setattr(run_with_ocr, "ROOT", root)
    monkeypatch.setattr(
        run_with_ocr,
        "installed_distributions",
        lambda: {run_with_ocr.normalize_distribution(name) for name in installed},
    )
    monkeypatch.setattr(
        run_with_ocr,
        "installed_version",
        lambda name: installed.get(run_with_ocr.normalize_distribution(name)),
    )


def matching_install():
    return {
        "onnxruntime-gpu": "1.26.0",
        "nvidia-cuda-runtime-cu12": "12.2.140",
        "nvidia-cublas-cu12": "12.2.5.6",
        "nvidia-cudnn-cu12": "9.1.0.70",
    }


def test_distribution_names_are_normalised():
    assert run_with_ocr.normalize_distribution("ONNXRuntime_GPU") == "onnxruntime-gpu"
    assert run_with_ocr.normalize_distribution("nvidia.cudnn_cu12") == "nvidia-cudnn-cu12"


def test_pins_are_parsed_from_the_requirements_file(monkeypatch, tmp_path):
    fake_environment(monkeypatch, tmp_path, {})
    pins = run_with_ocr.pinned_requirements()
    assert pins["onnxruntime-gpu"] == "1.26.0"
    assert pins["nvidia-cudnn-cu12"] == "9.1.0.70"
    assert "comment" not in pins


def test_cuda12_and_cuda13_wheels_share_a_library_stem():
    assert run_with_ocr.nvidia_stem("nvidia-cudnn-cu12") == "nvidia-cudnn"
    assert run_with_ocr.nvidia_stem("nvidia-cudnn-cu13") == "nvidia-cudnn"
    assert run_with_ocr.nvidia_stem("nvidia-cudnn") == "nvidia-cudnn"


def test_matching_stack_is_reported_as_matching(monkeypatch, tmp_path):
    fake_environment(monkeypatch, tmp_path, matching_install())
    assert run_with_ocr.gpu_stack_matches_requirements() is True
    assert run_with_ocr.conflicting_distributions() == []


def test_cpu_wheel_always_conflicts(monkeypatch, tmp_path):
    installed = matching_install()
    installed["onnxruntime"] = "1.29.0"
    fake_environment(monkeypatch, tmp_path, installed)
    assert run_with_ocr.conflicting_distributions() == ["onnxruntime"]
    assert run_with_ocr.gpu_stack_matches_requirements() is False


def test_cuda13_cudnn_is_replaced(monkeypatch, tmp_path):
    installed = matching_install()
    installed.pop("nvidia-cudnn-cu12")
    installed["nvidia-cudnn-cu13"] = "9.28.0.42"
    fake_environment(monkeypatch, tmp_path, installed)
    assert run_with_ocr.conflicting_distributions() == ["nvidia-cudnn-cu13"]


def test_version_drifted_pin_is_replaced(monkeypatch, tmp_path):
    installed = matching_install()
    installed["nvidia-cudnn-cu12"] = "9.28.0.42"
    fake_environment(monkeypatch, tmp_path, installed)
    assert run_with_ocr.conflicting_distributions() == ["nvidia-cudnn-cu12"]


def test_unrelated_nvidia_library_is_left_alone(monkeypatch, tmp_path):
    installed = matching_install()
    installed["nvidia-nvjitlink-cu12"] = "12.9.86"
    fake_environment(monkeypatch, tmp_path, installed)
    assert run_with_ocr.conflicting_distributions() == []
    assert run_with_ocr.gpu_stack_matches_requirements() is True


def test_missing_pin_blocks_matching(monkeypatch, tmp_path):
    installed = matching_install()
    installed.pop("nvidia-cublas-cu12")
    fake_environment(monkeypatch, tmp_path, installed)
    assert run_with_ocr.gpu_stack_matches_requirements() is False


def test_missing_requirements_file_is_not_matching(monkeypatch, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(run_with_ocr, "ROOT", root)
    assert run_with_ocr.pinned_requirements() == {}
    assert run_with_ocr.gpu_stack_matches_requirements() is False


def test_shipped_requirements_pin_a_cuda12_build():
    """Guard the compatibility rule: ORT 1.27+ is CUDA 13 and needs driver 580."""
    pins = run_with_ocr.pinned_requirements()
    version = pins.get("onnxruntime-gpu")
    assert version, "requirements-gpu.txt must pin onnxruntime-gpu exactly"
    major, minor = (int(part) for part in version.split(".")[:2])
    assert (major, minor) <= (1, 26), (
        "ONNX Runtime 1.27 and newer are CUDA 13 builds; they fail with "
        "CUDA failure 801 on drivers below R580"
    )
    assert "nvidia-cudnn-cu12" in pins, "cuDNN must be pinned, not resolved"
    assert all(
        not name.endswith("-cu13") for name in pins
    ), "CUDA 13 wheels require a 2024+ driver"


def test_shipped_requirements_pin_the_cuda12_runtime():
    pins = run_with_ocr.pinned_requirements()
    assert pins.get("nvidia-cuda-runtime-cu12") == "12.2.140"
    assert pins.get("nvidia-cublas-cu12") == "12.2.5.6"
    assert pins.get("nvidia-cudnn-cu12") == "9.1.0.70"


@pytest.mark.parametrize(
    "driver,expected",
    [("531.14", "531.14"), ("580.97", "580.97")],
)
def test_driver_version_is_read_for_diagnostics(monkeypatch, driver, expected):
    class Result:
        returncode = 0
        stdout = f"{driver}\n"
        stderr = ""

    monkeypatch.setattr(
        run_with_ocr.subprocess, "run", lambda *args, **kwargs: Result()
    )
    assert run_with_ocr.nvidia_driver_version() == expected


def test_driver_version_absent_returns_empty(monkeypatch):
    monkeypatch.setattr(
        run_with_ocr.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("missing")),
    )
    assert run_with_ocr.nvidia_driver_version() == ""
