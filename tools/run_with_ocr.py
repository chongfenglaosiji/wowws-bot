"""Prepare optional GPU wheels in a child process before importing the runner."""

from importlib import metadata
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def normalize_distribution(name: str) -> str:
    """Canonicalise a distribution name the way pip does (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name or "").strip().lower()


def installed_version(name: str) -> str | None:
    """Return the installed version of one distribution, if any."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def installed_distributions() -> set[str]:
    """Return every installed distribution name, normalised and lower case."""
    names = set()
    try:
        for distribution in metadata.distributions():
            name = distribution.metadata["Name"]
            if name:
                names.add(normalize_distribution(name))
    except Exception:
        return names
    return names


def pinned_requirements() -> dict[str, str]:
    """Return ``{normalised name: version}`` for the ``==`` pins in the file."""
    try:
        text = (ROOT / "requirements-gpu.txt").read_text(encoding="utf-8")
    except OSError:
        return {}
    pins: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.match(
            r"^([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?\s*==\s*([^\s;]+)$", line
        )
        if match:
            pins[normalize_distribution(match.group(1))] = match.group(2)
    return pins


def nvidia_stem(name: str) -> str:
    """Return ``nvidia-cudnn`` for both ``nvidia-cudnn-cu12`` and ``-cu13``.

    The CUDA 13 wheels drop the ``-cuNN`` suffix (``nvidia-cudnn-cu13`` versus
    ``nvidia-cudnn``), so comparing stems is the only reliable way to notice
    that two wheels would install into the same ``site-packages/nvidia``
    directory.
    """
    return re.sub(r"-cu\d+$", "", normalize_distribution(name))


def gpu_stack_matches_requirements() -> bool:
    """True when the installed OCR runtime equals the pinned requirements.

    Reinstalling is only useful while the installed stack differs from the
    pinned one. Once it matches, a failing probe is a driver problem and
    replacing the wheels again would re-download them on every task start.
    """
    pins = pinned_requirements()
    if not pins:
        return False
    if conflicting_distributions():
        return False
    for name, version in pins.items():
        if installed_version(name) != version:
            return False
    return True


def conflicting_distributions() -> list[str]:
    """Return distributions that must be uninstalled before installing the pin.

    Two kinds of leftovers break the CUDA stack:

    * the CPU ``onnxruntime`` wheel, which writes the same files as the GPU
      wheel in ``site-packages/onnxruntime``;
    * any ``nvidia-*`` wheel for the same library but a different version
      family (``nvidia-cudnn-cu13`` versus ``nvidia-cudnn-cu12``), because the
      CUDA 13 layout and the CUDA 12 layout overlap and cuDNN is even stored
      under the same ``nvidia/cudnn/bin`` path in both.

    Unrelated NVIDIA libraries are left alone; the pinned stack does not use
    them and uninstalling them would be an unnecessary risk.
    """
    pins = pinned_requirements()
    pinned_stems = {nvidia_stem(name) for name in pins if name.startswith("nvidia-")}
    conflicts = []
    for name in sorted(installed_distributions()):
        if name == "onnxruntime":
            conflicts.append(name)
            continue
        if not name.startswith("nvidia-"):
            continue
        if nvidia_stem(name) not in pinned_stems:
            continue
        if pins.get(name) != installed_version(name):
            conflicts.append(name)
    return conflicts


def run_component(command, *, timeout, **kwargs):
    """Keep component preparation responsive to the console's Stop button."""
    stop_file = os.environ.get("WOWS_STOP_FILE")
    if stop_file and Path(stop_file).exists():
        raise InterruptedError("OCR 组件准备已终止")
    process = subprocess.Popen(command, **kwargs)
    deadline = time.monotonic() + timeout
    try:
        while process.poll() is None:
            if stop_file and Path(stop_file).exists():
                raise InterruptedError("OCR 组件准备已终止")
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(command, timeout)
            time.sleep(.1)
        return subprocess.CompletedProcess(command, process.returncode)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def gpu_runtime_available() -> bool:
    """True when this ONNX Runtime build even exposes a CUDA provider.

    A missing provider means the GPU wheels are genuinely absent, while a
    present provider that fails inference is usually a transient condition
    (GPU busy with the game, driver hiccup) that must not trigger a
    multi-hundred-MB re-download.
    """
    try:
        import onnxruntime as ort
    except Exception:
        return False
    return "CUDAExecutionProvider" in ort.get_available_providers()


def nvidia_driver_version() -> str:
    """Read the installed NVIDIA driver version for diagnostics only."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    for line in (result.stdout or "").splitlines():
        value = line.strip()
        if re.fullmatch(r"\d+(\.\d+)+", value):
            return value
    return ""


def clean_conflicting_runtimes() -> bool:
    """Remove wheels that would collide with the pinned CUDA 12 stack."""
    present = conflicting_distributions()
    if not present:
        return True
    print(
        "[OCR] 正在替换冲突的 onnxruntime/CUDA 组件: " + ", ".join(present),
        flush=True,
    )
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = run_component(
            [sys.executable, "-m", "pip", "uninstall", "-y", *present],
            cwd=ROOT,
            timeout=180,
            creationflags=flags,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"[OCR] 卸载冲突组件失败: {error}", flush=True)
        return False
    return result.returncode == 0


def install_gpu_requirements(timeout: int = 1200) -> bool:
    """Install the pinned CUDA 12 OCR stack after removing conflicting wheels."""
    if sys.prefix == sys.base_prefix:
        print("[OCR] 不在项目虚拟环境，跳过自动安装，避免修改系统 Python", flush=True)
        return False
    cache = ROOT / "data" / "pip_cache"
    temp = ROOT / "data" / "install_tmp"
    cache.mkdir(parents=True, exist_ok=True)
    temp.mkdir(parents=True, exist_ok=True)
    install_env = dict(
        os.environ, PIP_CACHE_DIR=str(cache), TMP=str(temp), TEMP=str(temp)
    )
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if not clean_conflicting_runtimes():
        # A partly removed wheel still leaves pip able to finish the job, so
        # the install is attempted instead of giving up on a stale directory.
        print("[OCR] 冲突组件未完全卸载，继续尝试安装锁定组件", flush=True)
    print(
        "[OCR] 未检测到可用的 CUDA 运行组件，自动安装项目锁定的 CUDA 12 组件",
        flush=True,
    )
    try:
        result = run_component(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--prefer-binary",
                "-r",
                str(ROOT / "requirements-gpu.txt"),
            ],
            cwd=ROOT,
            timeout=timeout,
            creationflags=flags,
            env=install_env,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"[OCR] CUDA 组件安装失败: {error}", flush=True)
        return False
    return result.returncode == 0


def probe_cuda(flags) -> bool:
    probe = [sys.executable, str(ROOT / "tools/check_ocr_acceleration.py")]
    try:
        return run_component(probe, cwd=ROOT, timeout=180, creationflags=flags).returncode == 0
    except InterruptedError:
        # A user stop is not a CUDA failure: never report it as one, otherwise
        # it would look like a driver problem and could trigger a reinstall.
        raise
    except (OSError, subprocess.TimeoutExpired):
        return False


def stop_requested() -> bool:
    stop_file = os.environ.get("WOWS_STOP_FILE")
    return bool(stop_file and Path(stop_file).exists())


def driver_advice() -> str:
    driver = nvidia_driver_version()
    detail = f"（当前驱动 {driver}）" if driver else ""
    return (
        f"CUDA 组件已安装但无法推理{detail}；"
        "CUDA 12 组件需要 NVIDIA 驱动 525 以上，请更新显卡驱动，"
        "或设置 WOWS_OCR_DEVICE=cpu 强制使用 CPU"
    )


def prepare_gpu() -> tuple[bool, str]:
    """Make the CUDA self-test pass, repairing the installed stack when needed.

    Returns ``(ready, reason)`` where ``reason`` explains a CPU fallback in
    terms the user can act on.
    """
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    if stop_requested():
        raise InterruptedError("OCR 组件准备已终止")
    if probe_cuda(flags):
        return True, ""
    needs_replacement = not gpu_stack_matches_requirements()
    if gpu_runtime_available():
        # Components are installed: the probe failure was transient (GPU busy,
        # driver hiccup).  Retry once, then fall back to CPU for this run
        # instead of re-downloading the GPU stack.
        print("[OCR] CUDA 组件已安装但探测失败，等待 3 秒后重试…", flush=True)
        time.sleep(3)
        if probe_cuda(flags):
            return True, ""
        if not needs_replacement:
            return False, driver_advice()
        print(
            "[OCR] 当前 OCR 运行组件与锁定的 CUDA 12 组件不一致，正在替换后重试",
            flush=True,
        )
        if not install_gpu_requirements():
            return False, "CUDA 组件替换失败"
        if probe_cuda(flags):
            return True, ""
        return False, driver_advice()
    if not install_gpu_requirements():
        return False, "CUDA 组件安装失败"
    if probe_cuda(flags):
        return True, ""
    return False, driver_advice()


if __name__ == "__main__":
    try:
        if os.environ.get("WOWS_OCR_DEVICE", "nvidia") == "nvidia":
            ready, reason = prepare_gpu()
            if not ready:
                print(
                    f"[OCR] NVIDIA OCR 不可用，本次回退 CPU：{reason}",
                    flush=True,
                )
                os.environ["WOWS_OCR_DEVICE"] = "cpu"
    except InterruptedError as error:
        print(f"[OCR] {error}", flush=True)
        sys.exit(0)
    if stop_requested():
        sys.exit(0)
    sys.path.insert(0, str(ROOT))
    runpy.run_path(str(ROOT / "main.py"), run_name="__main__")
