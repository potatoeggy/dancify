from __future__ import annotations

import asyncio
import json
from typing import Any

from typer.testing import CliRunner

from dancify.domain import SessionState
from dancify.terminal import cli
from dancify.terminal.config import ClientConfig
from dancify.terminal.controller import DemoResult, RunResult
from dancify.terminal.dto import Routine, Score, Session
from dancify.terminal.tui import DancifyTerminalApp, ResultsScreen, SetupScreen

runner = CliRunner()


class FakeAPI:
    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    async def __aenter__(self) -> FakeAPI:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def health(self) -> bool:
        return True

    async def get_routine(self, _: str) -> Routine:
        return Routine("r", "Demo", "demo.mp4", 2.0, 30.0, 1)

    async def import_routine(self, _: dict[str, Any]) -> Routine:
        return Routine("r", "Imported", "demo.mp4", 2.0, 30.0, 1)

    async def create_session(self, *_: str) -> Session:
        return Session("s", "r", "p", SessionState.CREATED, None, 0, 0, 0, 0)

    async def abort(self, _: str) -> Session:
        return Session("s", "r", "p", SessionState.ABORTED, None, 0, 0, 0, 1)

    async def get_windows(self, _: str) -> tuple[object, ...]:
        return ()

    async def get_session(self, _: str) -> Session:
        return Session("s", "r", "p", SessionState.READY, None, 0, 0, 0, 0)


def test_cli_help_doctor_and_show_commands(monkeypatch: Any, tmp_path: Any) -> None:
    async def socket_ok(_: ClientConfig) -> bool:
        return True

    monkeypatch.setattr(cli, "DancifyAPI", FakeAPI)
    monkeypatch.setattr(cli, "probe_socket", socket_ok)
    help_result = runner.invoke(cli.app, ["--help"])
    assert help_result.exit_code == 0
    assert "discover" in help_result.stdout
    assert "omit for" in help_result.stdout
    assert "right-handed mode" in help_result.stdout
    assert runner.invoke(cli.app, ["discover", "--help"]).exit_code == 0
    configured = runner.invoke(
        cli.app,
        ["--dsu-host", "dsu.local", "--left-slot", "2", "--right-slot", "3", "doctor"],
    )
    assert configured.exit_code == 0
    doctor = runner.invoke(cli.app, ["doctor"])
    assert doctor.exit_code == 0 and '"ok": true' in doctor.stdout
    routine = runner.invoke(cli.app, ["routine", "show", "r"])
    assert routine.exit_code == 0 and '"title": "Demo"' in routine.stdout
    windows = runner.invoke(cli.app, ["routine", "windows", "r"])
    assert windows.exit_code == 0 and windows.stdout.strip() == "[]"
    session = runner.invoke(cli.app, ["session", "show", "s"])
    assert session.exit_code == 0 and '"state": "ready"' in session.stdout
    routine_file = tmp_path / "routine.json"
    routine_file.write_text(json.dumps({"title": "Imported"}))
    imported = runner.invoke(cli.app, ["routine", "import", str(routine_file)])
    assert imported.exit_code == 0 and '"title": "Imported"' in imported.stdout
    created = runner.invoke(cli.app, ["session", "create", "r", "--player", "p"])
    assert created.exit_code == 0 and '"state": "created"' in created.stdout
    aborted = runner.invoke(cli.app, ["session", "abort", "s"])
    assert aborted.exit_code == 0 and '"state": "aborted"' in aborted.stdout


def test_cli_left_slot_omission_and_explicit_compatibility(monkeypatch: Any) -> None:
    captured: list[dict[str, Any]] = []

    def from_env(**values: Any) -> ClientConfig:
        captured.append(values)
        return ClientConfig("http://test", dsu_left_slot=values["dsu_left_slot"])

    monkeypatch.setattr(cli.ClientConfig, "from_env", staticmethod(from_env))
    monkeypatch.setattr(cli, "DancifyAPI", FakeAPI)
    monkeypatch.setattr(cli, "probe_socket", lambda _config: asyncio.sleep(0, result=True))
    assert runner.invoke(cli.app, ["doctor"]).exit_code == 0
    assert captured[-1]["dsu_left_slot"] is None
    assert runner.invoke(cli.app, ["--left-slot", "0", "doctor"]).exit_code == 0
    assert captured[-1]["dsu_left_slot"] == 0


def result() -> DemoResult:
    session = Session("s", "r", "p", SessionState.COMPLETED, 0, 2, 1, 90, 5)
    score = Score(0, 0, 90, 90, True)
    return DemoResult(Routine("r", "Demo", "demo.mp4", 2, 30, 1), RunResult(session, 200, (score,)))


def test_tui_setup_validation_and_results() -> None:
    async def scenario() -> None:
        app = DancifyTerminalApp(ClientConfig("http://test"))
        async with app.run_test() as pilot:
            await pilot.pause()
            assert isinstance(app.screen, SetupScreen)
            duration = app.screen.query_one("#duration")
            duration.value = "0.1"
            await pilot.click("#run")
            await pilot.pause()
            assert "at least" in str(app.screen.query_one("#status").render())
            await app.push_screen(ResultsScreen(result()))
            await pilot.pause()
            assert "90.000" in str(app.screen.query_one("#score").render())
            await pilot.click("#back")
            await pilot.pause()
            assert isinstance(app.screen, SetupScreen)

    asyncio.run(scenario())
