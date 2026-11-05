"""Startup and status-reporting behaviour of the local control console."""

import json
from unittest.mock import Mock, patch

import pytest

import control_server
from control_server import RunnerManager


def empty_control_store():
    store = Mock()
    store.get_resume_candidate.return_value = None
    return store


def test_tail_log_returns_only_the_last_lines(tmp_path):
    log = tmp_path / "runtime.log"
    log.write_text(
        "\n".join(f"line {index}" for index in range(5000)) + "\n",
        encoding="utf-8",
    )
    with patch("control_server.LOG_PATH", log):
        assert control_server.tail_log(3) == ["line 4997", "line 4998", "line 4999"]


def test_tail_log_handles_missing_and_empty_files(tmp_path):
    missing = tmp_path / "absent.log"
    empty = tmp_path / "empty.log"
    empty.write_text("", encoding="utf-8")
    with patch("control_server.LOG_PATH", missing):
        assert control_server.tail_log(5) == []
    with patch("control_server.LOG_PATH", empty):
        assert control_server.tail_log(5) == []


def test_tail_log_never_returns_a_partial_first_line(tmp_path):
    log = tmp_path / "runtime.log"
    lines = [f"{'x' * 200}{index}" for index in range(40)]
    log.write_text("\n".join(lines), encoding="utf-8")
    with patch("control_server.LOG_PATH", log):
        tail = control_server.tail_log(5, max_bytes=64)
    assert tail == lines[-5:]


def test_instance_file_tracks_the_owning_process(tmp_path):
    path = tmp_path / "control_server.json"
    with (
        patch("control_server.DATA_DIR", tmp_path),
        patch("control_server.INSTANCE_PATH", path),
    ):
        control_server._write_instance_file("127.0.0.1", 8899)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["pid"] == control_server.os.getpid()
        assert payload["port"] == 8899
        assert payload["version"]
        control_server._clear_instance_file()
        assert not path.exists()


def test_instance_file_of_another_process_is_preserved(tmp_path):
    path = tmp_path / "control_server.json"
    path.write_text(json.dumps({"pid": 1, "port": 8899}), encoding="utf-8")
    with patch("control_server.INSTANCE_PATH", path):
        control_server._clear_instance_file()
    assert path.exists()


def test_main_reports_a_port_conflict_without_a_traceback(capsys):
    with (
        patch("control_server.ensure_elevated_control_server", return_value=True),
        patch.dict("control_server.os.environ", {"WOWS_PANEL_PORT": "8899"}),
        patch("control_server.sys.argv", ["control_server.py", "--no-browser"]),
        patch(
            "control_server.ControlHttpServer",
            side_effect=OSError(10048, "address already in use"),
        ),
    ):
        assert control_server.main() == 2
    error = capsys.readouterr().err
    assert "无法监听" in error
    assert "WOWS_PANEL_PORT" in error


def test_main_rejects_an_invalid_panel_port(capsys):
    with (
        patch("control_server.ensure_elevated_control_server", return_value=True),
        patch.dict("control_server.os.environ", {"WOWS_PANEL_PORT": "not-a-port"}),
        patch("control_server.sys.argv", ["control_server.py"]),
    ):
        assert control_server.main() == 2
    assert "WOWS_PANEL_PORT" in capsys.readouterr().err


def test_main_records_the_live_instance_and_clears_it_on_exit(tmp_path, capsys):
    server = Mock()
    server.serve_forever.side_effect = KeyboardInterrupt
    instance = tmp_path / "control_server.json"
    with (
        patch("control_server.ensure_elevated_control_server", return_value=True),
        patch("control_server.ControlHttpServer", return_value=server),
        patch("control_server.DATA_DIR", tmp_path),
        patch("control_server.INSTANCE_PATH", instance),
        patch("control_server.updater.start_auto"),
        patch("control_server.updater.stop_auto"),
        patch.dict("control_server.os.environ", {"WOWS_PANEL_PORT": "8899"}),
        patch("control_server.sys.argv", ["control_server.py", "--no-browser"]),
    ):
        assert control_server.main() == 0
    server.server_close.assert_called_once()
    assert not instance.exists()
    assert "战舰控制台已启动" in capsys.readouterr().out


def test_invalid_start_does_not_overwrite_the_saved_custom_ship():
    manager = RunnerManager(empty_control_store())
    with (
        patch(
            "control_server.resolve_custom_ship_for_run",
            return_value=("冯·祖克霍夫", 11.0),
        ),
        patch("control_server.save_custom_ship") as save,
        patch("control_server.running_game_requires_elevation", return_value=False),
    ):
        with pytest.raises(ValueError):
            manager.start(
                {
                    "ship": "custom",
                    "mode": "not-a-mode",
                    "limit_type": "rounds",
                    "limit_value": 1,
                }
            )
    save.assert_not_called()
