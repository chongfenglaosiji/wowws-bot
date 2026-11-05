"""Focused Windows deployment checks; no pip downloads, panel or game launch."""

import base64
import json
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows bootstrap")
# These end-to-end checks use the interpreter that runs pytest as the expected
# Python 3.11 runtime. On another version the launcher correctly rejects it, so
# only exercise them where the supported runtime is present.
requires_python_311 = pytest.mark.skipif(
    sys.version_info[:2] != (3, 11),
    reason="需要 64 位 Python 3.11 运行时",
)
ROOT = Path(__file__).resolve().parents[1]


def ps_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def run_ps(code, *, timeout=30):
    script = (
        "[Console]::OutputEncoding = New-Object Text.UTF8Encoding($false); "
        "$ErrorActionPreference = 'Stop'; "
        f". {ps_literal(ROOT / 'tools/start_project.ps1')}; "
        "$env:WOWS_PYTHON = ''; " + code
    )
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-EncodedCommand", encoded],
        capture_output=True, encoding="utf-8", timeout=timeout,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip().splitlines()[-1]


@requires_python_311
def test_python_on_path_is_accepted_without_py_launcher():
    # Exercise PATH discovery and the real interpreter probe while hiding py.
    python = ps_literal(sys.executable)
    output = run_ps(
        "function Get-Command { param($Name, $CommandType, [switch]$All, $ErrorAction) "
        f"if ($Name -eq 'python.exe') {{ [pscustomobject]@{{Source={python}}} }} }}; "
        "Find-BasePython"
    )
    assert Path(output).resolve() == Path(sys.executable).resolve()


@pytest.mark.parametrize("version,bits", [([3, 13, 0], 64), ([3, 11, 9], 32)])
def test_other_version_and_32_bit_python_cannot_create_environment(version, bits):
    details = json.dumps({"executable": sys.executable, "version": version, "bits": bits})
    output = run_ps(
        "function Invoke-PythonCommand { [pscustomobject]@{ExitCode=0; Error=''; Output="
        + ps_literal(details) + "} }; "
        f"$result = Test-PythonInterpreter {ps_literal(sys.executable)}; "
        "if ($result) { throw 'wrong interpreter accepted' }; 'rejected'"
    )
    assert output == "rejected"


def test_store_placeholder_never_executes_or_opens_store(tmp_path):
    alias = tmp_path / "Microsoft" / "WindowsApps" / "python.exe"
    alias.parent.mkdir(parents=True)
    alias.touch()
    assert run_ps(
        "function Invoke-PythonCommand { throw 'Store alias executed' }; "
        f"$result = Test-PythonInterpreter {ps_literal(alias)}; "
        "if ($result) { throw 'placeholder accepted' }; 'skipped'"
    ) == "skipped"


@requires_python_311
def test_custom_interpreter_and_check_only_do_not_create_environment(tmp_path):
    output = run_ps(
        f"$script:ProjectRoot = {ps_literal(tmp_path)}; "
        f"$env:WOWS_PYTHON = {ps_literal(sys.executable)}; "
        "$CheckPythonOnly = $true; Start-Project; 'checked'"
    )
    assert output == "checked"
    assert list(tmp_path.iterdir()) == []


@requires_python_311
def test_broken_environment_is_backed_up_and_rebuilt_in_non_ascii_path(tmp_path):
    project = tmp_path / "部署 space's! [project]"
    broken = project / ".venv"
    broken.mkdir(parents=True)
    (broken / "keep_marker").write_text("preserved")
    output = run_ps(
        f"$script:ProjectRoot = {ps_literal(project)}; "
        f"$env:WOWS_PYTHON = {ps_literal(sys.executable)}; "
        "$first = Resolve-ProjectPython; $second = Resolve-ProjectPython; "
        "if ($first -ne $second) { throw 'environment not reused' }; $second",
        timeout=60,
    )
    assert Path(output) == project / ".venv" / "Scripts" / "python.exe"
    backups = list(project.glob(".venv_backup_*"))
    assert len(backups) == 1
    assert (backups[0] / "keep_marker").read_text() == "preserved"


def test_python_probe_passes_spaces_quotes_unicode_and_trailing_slashes():
    arguments = ["空 格", "a'b!c", 'quote"value', "trailing\\"]
    args = ",".join(ps_literal(value) for value in arguments)
    output = run_ps(
        f"$result = Invoke-PythonCommand {ps_literal(sys.executable)} "
        f"@('-I','-c','import json,sys; print(json.dumps(sys.argv[1:]))',{args}); "
        "if ($result.ExitCode -ne 0) { throw $result.Error }; $result.Output.Trim()"
    )
    assert json.loads(output) == arguments


def test_bad_explicit_override_reports_path_without_using_another_python(tmp_path):
    assert run_ps(
        f"$env:WOWS_PYTHON = {ps_literal(tmp_path / 'missing.exe')}; "
        "function Get-PythonCandidates { throw 'override silently ignored' }; "
        "try { Find-BasePython; throw 'missing interpreter accepted' } "
        "catch { if ($_.Exception.Message -notlike '*WOWS_PYTHON*') { throw }; 'explicit_error' }"
    ) == "explicit_error"


def test_dependency_cache_is_reused_and_invalidated_by_requirements(tmp_path):
    project = tmp_path / "project"
    (project / "data").mkdir(parents=True)
    (project / ".venv" / "Scripts").mkdir(parents=True)
    (project / ".venv" / "Lib" / "site-packages").mkdir(parents=True)
    python = project / ".venv" / "Scripts" / "python.exe"
    python.write_bytes(b"stub")
    requirements = project / "requirements.txt"
    requirements.write_text("numpy>=1.24\n", encoding="utf-8")
    output = run_ps(
        f"$script:ProjectRoot = {ps_literal(project)}; "
        f"$python = {ps_literal(python)}; "
        "$fingerprint = Get-DependencyFingerprint $python; "
        "if (Test-DependencyCache $fingerprint) { throw 'unexpected cache hit' }; "
        "Save-DependencyCache $fingerprint; "
        "if (-not (Test-DependencyCache $fingerprint)) { throw 'cache miss after save' }; "
        f"Add-Content -LiteralPath {ps_literal(requirements)} -Value 'opencv-python'; "
        "$changed = Get-DependencyFingerprint $python; "
        "if ($changed -eq $fingerprint) { throw 'fingerprint did not change' }; "
        "if (Test-DependencyCache $changed) { throw 'stale cache accepted' }; 'cache_ok'"
    )
    assert output == "cache_ok"


def test_log_tail_returns_the_newest_complete_lines(tmp_path):
    log = tmp_path / "startup.log"
    log.write_text(
        "\n".join(f"line {index}" for index in range(1, 201)) + "\n",
        encoding="utf-8",
    )
    output = run_ps(
        f"$tail = Read-LogTail {ps_literal(log)} 2; ($tail -split [char]10) -join '|'"
    )
    assert output == "line 199|line 200"


def test_stale_instance_recorder_is_discarded(tmp_path):
    project = tmp_path / "project"
    (project / "data").mkdir(parents=True)
    recorded = project / "data" / "control_server.json"
    recorded.write_text(
        json.dumps({"pid": 999999, "port": 1, "version": "stale"}),
        encoding="utf-8",
    )
    output = run_ps(
        f"$script:ProjectRoot = {ps_literal(project)}; "
        "$result = Get-RecordedPanelUrl; "
        "if ($result) { throw 'stale recorder accepted' }; "
        f"if (Test-Path -LiteralPath {ps_literal(recorded)}) "
        "{ throw 'stale recorder kept' }; 'stale_ok'"
    )
    assert output == "stale_ok"


def gpu_project(tmp_path):
    project = tmp_path / "project"
    (project / "data").mkdir(parents=True)
    (project / "requirements-gpu.txt").write_text(
        "onnxruntime-gpu[cuda,cudnn]==1.29.0\n", encoding="utf-8"
    )
    python = project / ".venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"stub")
    return project, python


def test_gpu_install_is_skipped_without_nvidia_gpu(tmp_path):
    project, python = gpu_project(tmp_path)
    output = run_ps(
        f"$script:ProjectRoot = {ps_literal(project)}; "
        "$script:StartLog = $null; $script:InstallLog = $null; "
        "function Get-NvidiaGpuInfo { return $null }; "
        "function Invoke-PipInstall { throw 'pip must not run without an NVIDIA GPU' }; "
        f"Install-ProjectGpuDependencies {ps_literal(python)}; 'gpu_skipped'"
    )
    assert output == "gpu_skipped"


def test_gpu_install_failure_is_cached_and_force_retries(tmp_path):
    project, python = gpu_project(tmp_path)
    output = run_ps(
        f"$script:ProjectRoot = {ps_literal(project)}; "
        "$script:StartLog = $null; $script:InstallLog = $null; "
        "function Get-NvidiaGpuInfo { [pscustomobject]@{ Name='RTX Test'; Driver='1.0'; Source='nvidia-smi' } }; "
        "function Test-CudaProvider { [pscustomobject]@{ Available=$false; Providers=''; Detail='' } }; "
        "function Test-CudaInference { return $null }; "
        "$script:pipCalls = 0; "
        "function Invoke-PipInstall { $script:pipCalls++; return $false }; "
        f"Install-ProjectGpuDependencies {ps_literal(python)}; "
        f"Install-ProjectGpuDependencies {ps_literal(python)}; "
        "if ($script:pipCalls -ne 1) { throw 'cached failure was retried' }; "
        "$env:WOWS_FORCE_GPU_INSTALL = '1'; "
        f"Install-ProjectGpuDependencies {ps_literal(python)}; "
        "if ($script:pipCalls -ne 2) { throw 'force retry ignored' }; "
        f"$cache = Get-Content -LiteralPath {ps_literal(project / 'data' / '.gpu-deps.json')} -Encoding UTF8 -Raw | ConvertFrom-Json; "
        "if ($cache.status -ne 'failed') { throw 'failure not cached' }; 'gpu_cache_ok'"
    )
    assert output == "gpu_cache_ok"


def test_gpu_install_ready_state_skips_reinstall(tmp_path):
    project, python = gpu_project(tmp_path)
    output = run_ps(
        f"$script:ProjectRoot = {ps_literal(project)}; "
        "$script:StartLog = $null; $script:InstallLog = $null; "
        "function Get-NvidiaGpuInfo { [pscustomobject]@{ Name='RTX Test'; Driver='1.0'; Source='nvidia-smi' } }; "
        "$script:pipCalls = 0; $script:probeCalls = 0; "
        "function Invoke-PipInstall { $script:pipCalls++; return $true }; "
        "function Test-CudaProvider { $script:probeCalls++; "
        "if ($script:probeCalls -le 1) { return [pscustomobject]@{ Available=$false; Providers=''; Detail='' } }; "
        "[pscustomobject]@{ Available=$true; Providers='CUDAExecutionProvider'; Detail='' } }; "
        "function Test-CudaInference { [pscustomobject]@{ Passed=$true; Detail='ok' } }; "
        f"Install-ProjectGpuDependencies {ps_literal(python)}; "
        f"$cache = Get-Content -LiteralPath {ps_literal(project / 'data' / '.gpu-deps.json')} -Encoding UTF8 -Raw | ConvertFrom-Json; "
        "if ($cache.status -ne 'ready') { throw 'ready state not cached' }; "
        f"Install-ProjectGpuDependencies {ps_literal(python)}; "
        "if ($script:pipCalls -ne 1) { throw 'verified install reinstalled' }; 'gpu_ready_ok'"
    )
    assert output == "gpu_ready_ok"


def test_gpu_install_requires_a_working_nvidia_driver(tmp_path):
    project, python = gpu_project(tmp_path)
    output = run_ps(
        f"$script:ProjectRoot = {ps_literal(project)}; "
        "$script:StartLog = $null; $script:InstallLog = $null; "
        "function Get-NvidiaGpuInfo { [pscustomobject]@{ Name='NVIDIA GPU'; Driver='1.0'; Source='wmi' } }; "
        "function Test-CudaProvider { [pscustomobject]@{ Available=$false; Providers='CPUExecutionProvider'; Detail='' } }; "
        "function Invoke-PipInstall { throw 'pip must not run without a working driver' }; "
        f"Install-ProjectGpuDependencies {ps_literal(python)}; 'driver_missing_ok'"
    )
    assert output == "driver_missing_ok"


def test_gpu_install_skips_pip_when_cuda_already_available(tmp_path):
    project, python = gpu_project(tmp_path)
    output = run_ps(
        f"$script:ProjectRoot = {ps_literal(project)}; "
        "$script:StartLog = $null; $script:InstallLog = $null; "
        "function Get-NvidiaGpuInfo { [pscustomobject]@{ Name='RTX Test'; Driver='1.0'; Source='nvidia-smi' } }; "
        "function Invoke-PipInstall { throw 'pip must not run when CUDA is already available' }; "
        "function Test-CudaProvider { [pscustomobject]@{ Available=$true; Providers='CUDAExecutionProvider'; Detail='' } }; "
        "function Test-CudaInference { [pscustomobject]@{ Passed=$true; Detail='ok' } }; "
        f"Install-ProjectGpuDependencies {ps_literal(python)}; "
        f"$cache = Get-Content -LiteralPath {ps_literal(project / 'data' / '.gpu-deps.json')} -Encoding UTF8 -Raw | ConvertFrom-Json; "
        "if ($cache.status -ne 'ready') { throw 'ready state not cached' }; 'gpu_present_ok'"
    )
    assert output == "gpu_present_ok"


def test_install_log_captures_install_output(tmp_path):
    project = tmp_path / "project"
    (project / "data").mkdir(parents=True)
    log = project / "data" / "install.log"
    output = run_ps(
        f"$script:ProjectRoot = {ps_literal(project)}; "
        "$script:StartLog = $null; Initialize-InstallLog; "
        "Write-InstallLog 'pip install sample'; "
        f"if (-not (Test-Path -LiteralPath {ps_literal(log)})) {{ throw 'missing install log' }}; "
        f"if (-not (Select-String -LiteralPath {ps_literal(log)} -Pattern 'pip install sample' -Quiet)) "
        "{ throw 'missing install line' }; 'log_ok'"
    )
    assert output == "log_ok"
