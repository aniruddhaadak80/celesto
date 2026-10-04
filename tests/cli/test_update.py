# Copyright 2026 Celesto AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for celesto update."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from celesto.cli.update import (
    _check_for_stable_update,
    _is_uv_tool_install,
    _run_upgrade,
    _UpdateCheck,
    run_update,
)


class TestCheckForStableUpdate:
    def test_returns_none_when_already_latest(self) -> None:
        """An installed version equal to the newest release leaves no upgrade pending."""
        with (
            patch("celesto.cli.update._get_current_version", return_value="1.0.0"),
            patch("celesto.cli.update._fetch_latest_from_pypi", return_value="1.0.0"),
        ):
            result = _check_for_stable_update()
            assert result.current == "1.0.0"
            assert result.latest is None
            assert result.reachable is True

    def test_returns_latest_when_update_available(self) -> None:
        """A newer release on PyPI is returned so the caller can offer the upgrade."""
        with (
            patch("celesto.cli.update._get_current_version", return_value="0.9.0"),
            patch("celesto.cli.update._fetch_latest_from_pypi", return_value="1.0.0"),
        ):
            result = _check_for_stable_update()
            assert result.current == "0.9.0"
            assert result.latest == "1.0.0"
            assert result.reachable is True

    def test_returns_none_on_network_failure(self) -> None:
        """A PyPI request that failed must not read as "nothing is newer".

        ``latest is None`` is only that claim while PyPI answered, so the
        failure has to be reported through ``reachable`` as well.
        """
        with (
            patch("celesto.cli.update._get_current_version", return_value="1.0.0"),
            patch("celesto.cli.update._fetch_latest_from_pypi", return_value=None),
        ):
            result = _check_for_stable_update()
            assert result.current == "1.0.0"
            assert result.latest is None
            assert result.reachable is False

    def test_handles_missing_current_version(self) -> None:
        """A PyPI answer with no readable local version reports unknown-but-reachable."""
        with (
            patch("celesto.cli.update._get_current_version", return_value=None),
            patch("celesto.cli.update._fetch_latest_from_pypi", return_value="1.0.0"),
        ):
            result = _check_for_stable_update()
            assert result.current is None
            assert result.latest is None
            assert result.reachable is True


class TestIsUvToolInstall:
    def test_returns_false_when_uv_not_found(self) -> None:
        """With no uv on PATH this cannot be a uv tool install."""
        with patch("celesto.cli.update.shutil.which", return_value=None):
            assert _is_uv_tool_install() is False

    def test_returns_true_when_celesto_in_uv_tool_list(self) -> None:
        """A uv tool list naming celesto means the install is managed by uv."""
        mock_result = MagicMock()
        mock_result.stdout = "celesto v0.0.19\n"
        with (
            patch("celesto.cli.update.shutil.which", return_value="/usr/bin/uv"),
            patch("celesto.cli.update.subprocess.run", return_value=mock_result),
        ):
            assert _is_uv_tool_install() is True

    def test_returns_false_when_only_celesto_core_in_uv_tool_list(self) -> None:
        """celesto-core is a different distribution and must not match celesto."""
        mock_result = MagicMock()
        mock_result.stdout = "celesto-core v0.0.14\n"
        with (
            patch("celesto.cli.update.shutil.which", return_value="/usr/bin/uv"),
            patch("celesto.cli.update.subprocess.run", return_value=mock_result),
        ):
            assert _is_uv_tool_install() is False

    def test_returns_false_when_celesto_not_in_uv_tool_list(self) -> None:
        """A uv tool list without celesto means uv manages other tools only."""
        mock_result = MagicMock()
        mock_result.stdout = "other-tool v1.0\n"
        with (
            patch("celesto.cli.update.shutil.which", return_value="/usr/bin/uv"),
            patch("celesto.cli.update.subprocess.run", return_value=mock_result),
        ):
            assert _is_uv_tool_install() is False

    def test_returns_false_when_uv_tool_list_times_out(self) -> None:
        """A hung uv must not raise into the caller that is trying to report a failure."""
        with (
            patch("celesto.cli.update.shutil.which", return_value="/usr/bin/uv"),
            patch(
                "celesto.cli.update.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd=["uv", "tool", "list"], timeout=10),
            ),
        ):
            assert _is_uv_tool_install() is False

    def test_returns_false_when_uv_is_missing_at_launch(self) -> None:
        """A uv that disappears between which() and launch must not raise either."""
        with (
            patch("celesto.cli.update.shutil.which", return_value="/usr/bin/uv"),
            patch(
                "celesto.cli.update.subprocess.run",
                side_effect=OSError("No such file or directory"),
            ),
        ):
            assert _is_uv_tool_install() is False

    def test_uv_tool_list_runs_with_a_positive_timeout(self) -> None:
        """The uv query is bounded, so a stalled uv cannot block the caller forever."""
        mock_result = MagicMock()
        mock_result.stdout = "other-tool v1.0\n"
        with (
            patch("celesto.cli.update.shutil.which", return_value="/usr/bin/uv"),
            patch("celesto.cli.update.subprocess.run", return_value=mock_result) as run,
        ):
            _is_uv_tool_install()
        timeout = run.call_args.kwargs.get("timeout")
        assert isinstance(timeout, int | float)
        assert timeout > 0


class TestRunUpdate:
    def test_check_only_no_update(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A check with no newer release reports up to date and exits 0."""
        with (
            patch(
                "celesto.cli.update._check_for_stable_update",
                return_value=_UpdateCheck("1.0.0", None, True),
            ),
        ):
            rc = run_update(check=True)
        assert rc == 0
        out = capsys.readouterr().out
        assert "up to date" in out

    def test_check_only_unknown_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A readable PyPI answer with no local version still asks the user to retry."""
        with (
            patch(
                "celesto.cli.update._check_for_stable_update",
                return_value=_UpdateCheck(None, None, True),
            ),
            patch("celesto.cli.update._is_uv_tool_install", return_value=False),
        ):
            rc = run_update(check=True)
        assert rc == 1
        err = capsys.readouterr().err
        assert "Could not determine" in err
        assert "pip install --upgrade celesto" in err

    def test_check_only_update_available(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A newer release on PyPI is announced with the exact target version."""
        with (
            patch(
                "celesto.cli.update._check_for_stable_update",
                return_value=_UpdateCheck("0.9.0", "1.0.0", True),
            ),
        ):
            rc = run_update(check=True)
        assert rc == 0
        out = capsys.readouterr().out
        assert "1.0.0" in out

    def test_check_only_json_no_update(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The JSON check payload reports update_available False and exits 0."""
        with (
            patch(
                "celesto.cli.update._check_for_stable_update",
                return_value=_UpdateCheck("1.0.0", None, True),
            ),
        ):
            rc = run_update(check=True, json_output=True)
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["data"]["update_available"] is False

    def test_already_latest_skips_pip(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Nothing to upgrade means the package manager is never invoked."""
        with (
            patch(
                "celesto.cli.update._check_for_stable_update",
                return_value=_UpdateCheck("1.0.0", None, True),
            ),
            patch("celesto.cli.update._run_upgrade") as mock_pip,
        ):
            rc = run_update()
        assert rc == 0
        mock_pip.assert_not_called()

    def test_upgrade_calls_pip(self) -> None:
        """A pending upgrade delegates to the single _run_upgrade call site."""
        with (
            patch(
                "celesto.cli.update._check_for_stable_update",
                return_value=_UpdateCheck("0.9.0", "1.0.0", True),
            ),
            patch("celesto.cli.update._run_upgrade", return_value=(0, "")) as mock_pip,
            patch("celesto.cli.update._get_current_version", return_value="1.0.0"),
        ):
            rc = run_update()
        assert rc == 0
        mock_pip.assert_called_once_with(json_output=False)

    def test_pip_failure_returns_nonzero(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A failed upgrade propagates the package manager's exit code."""
        with (
            patch(
                "celesto.cli.update._check_for_stable_update",
                return_value=_UpdateCheck("0.9.0", "1.0.0", True),
            ),
            patch("celesto.cli.update._run_upgrade", return_value=(1, "error output")),
            patch("celesto.cli.update._get_current_version", return_value="0.9.0"),
        ):
            rc = run_update()
        assert rc == 1

    def test_upgrade_json_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The JSON upgrade payload reports the new, previous, and upgraded state."""
        with (
            patch(
                "celesto.cli.update._check_for_stable_update",
                return_value=_UpdateCheck("0.9.0", "1.0.0", True),
            ),
            patch(
                "celesto.cli.update._run_upgrade",
                return_value=(0, "Successfully installed celesto-1.0.0"),
            ),
            patch("celesto.cli.update._get_current_version", return_value="1.0.0"),
        ):
            rc = run_update(json_output=True)
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["data"]["upgraded"] is True
        assert payload["data"]["current"] == "1.0.0"
        assert payload["data"]["previous"] == "0.9.0"


@pytest.mark.parametrize("uv_tool", [True, False])
def test_upgrade_targets_celesto_distribution(uv_tool: bool) -> None:
    """The upgrade targets the celesto distribution through uv or pip as appropriate."""
    result = MagicMock(returncode=0, stdout="", stderr="")
    with (
        patch("celesto.cli.update._is_uv_tool_install", return_value=uv_tool),
        patch("celesto.cli.update.shutil.which", return_value="/usr/bin/uv"),
        patch("celesto.cli.update.subprocess.run", return_value=result) as run,
    ):
        assert _run_upgrade(json_output=True) == (0, "")
    command = run.call_args.args[0]
    assert command[-1] == "celesto"
    assert command[1:3] == (["tool", "upgrade"] if uv_tool else ["-m", "pip"])


class TestRetryCommand:
    def test_failure_retry_command_is_uv_when_installed_as_uv_tool(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A failed upgrade tells a uv tool user to retry with uv."""
        with (
            patch(
                "celesto.cli.update._check_for_stable_update",
                return_value=_UpdateCheck("0.9.0", "1.0.0", True),
            ),
            patch("celesto.cli.update._is_uv_tool_install", return_value=True),
            patch("celesto.cli.update._run_upgrade", return_value=(1, "error output")),
            patch("celesto.cli.update._get_current_version", return_value="0.9.0"),
        ):
            rc = run_update()
        assert rc == 1
        err = capsys.readouterr().err
        assert "uv tool upgrade celesto" in err
        assert "pip install" not in err

    def test_failure_retry_command_is_pip_when_not_installed_as_uv_tool(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A failed upgrade tells a pip user to retry with pip."""
        with (
            patch(
                "celesto.cli.update._check_for_stable_update",
                return_value=_UpdateCheck("0.9.0", "1.0.0", True),
            ),
            patch("celesto.cli.update._is_uv_tool_install", return_value=False),
            patch("celesto.cli.update._run_upgrade", return_value=(1, "error output")),
            patch("celesto.cli.update._get_current_version", return_value="0.9.0"),
        ):
            rc = run_update()
        assert rc == 1
        err = capsys.readouterr().err
        assert "pip install --upgrade celesto" in err

    def test_unknown_version_retry_command_is_uv_when_installed_as_uv_tool(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An unknown installed version on a uv tool install also retries with uv."""
        with (
            patch(
                "celesto.cli.update._check_for_stable_update",
                return_value=_UpdateCheck(None, None, True),
            ),
            patch("celesto.cli.update._is_uv_tool_install", return_value=True),
        ):
            rc = run_update(check=True)
        assert rc == 1
        err = capsys.readouterr().err
        assert "uv tool upgrade celesto" in err
        assert "pip install" not in err


class TestPyPIUnreachable:
    """A release check that never reached PyPI must not claim there is nothing to do.

    Without this, a failed request returned ``None`` for the latest version and
    ``celesto update`` told the user they were already on the newest release.
    """

    def test_update_reports_unreachable_instead_of_being_latest(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An upgrade attempt that never reached PyPI must not claim it is latest."""
        with (
            patch("celesto.cli.update._get_current_version", return_value="0.1.0"),
            patch("celesto.cli.update._fetch_latest_from_pypi", return_value=None),
            patch("celesto.cli.update._is_uv_tool_install", return_value=False),
            patch("celesto.cli.update._run_upgrade") as mock_upgrade,
        ):
            rc = run_update()
        assert rc != 0
        mock_upgrade.assert_not_called()
        captured = capsys.readouterr()
        assert "already the latest stable release" not in captured.out
        assert "pypi.org" in captured.err
        assert "pip install --upgrade celesto" in captured.err

    def test_check_reports_unreachable_instead_of_being_up_to_date(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A check that never reached PyPI must not claim the install is up to date."""
        with (
            patch("celesto.cli.update._get_current_version", return_value="0.1.0"),
            patch("celesto.cli.update._fetch_latest_from_pypi", return_value=None),
            patch("celesto.cli.update._is_uv_tool_install", return_value=False),
        ):
            rc = run_update(check=True)
        assert rc != 0
        captured = capsys.readouterr()
        assert "up to date" not in captured.out
        assert "pypi.org" in captured.err
        assert "pip install --upgrade celesto" in captured.err

    def test_check_json_reports_unreachable_with_recovery(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The JSON check error carries the pypi_unreachable code and a recovery command."""
        with (
            patch("celesto.cli.update._get_current_version", return_value="0.1.0"),
            patch("celesto.cli.update._fetch_latest_from_pypi", return_value=None),
            patch("celesto.cli.update._is_uv_tool_install", return_value=False),
        ):
            rc = run_update(check=True, json_output=True)
        assert rc != 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is False
        assert payload["error"]["code"] == "pypi_unreachable"
        assert payload["error"]["recovery"] == "pip install --upgrade celesto"
        assert payload["data"]["checked_pypi"] is False

    def test_already_latest_when_pypi_answers(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The fix must not disturb the ordinary "nothing newer" answer."""
        with (
            patch("celesto.cli.update._get_current_version", return_value="0.1.0"),
            patch("celesto.cli.update._fetch_latest_from_pypi", return_value="0.1.0"),
            patch("celesto.cli.update._run_upgrade") as mock_upgrade,
        ):
            rc = run_update()
        assert rc == 0
        mock_upgrade.assert_not_called()
        assert "already the latest stable release" in capsys.readouterr().out


class TestUnreachableBeatsUnknownVersion:
    """An unreachable PyPI is reported ahead of an unknown installed version.

    When the check could not reach PyPI *and* the installed version could not be
    read, both branches would have been candidates. The unreachable report is
    the one that names the real cause, so it wins — and the unknown-version
    wording stays in place for the case where PyPI did answer.
    """

    def test_check_json_reports_unreachable_when_version_unknown(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Unreachable PyPI plus an unknown version must report pypi_unreachable."""
        with (
            patch("celesto.cli.update._get_current_version", return_value=None),
            patch("celesto.cli.update._fetch_latest_from_pypi", return_value=None),
            patch("celesto.cli.update._is_uv_tool_install", return_value=False),
        ):
            rc = run_update(check=True, json_output=True)
        assert rc == 1
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["ok"] is False
        assert payload["error"]["code"] == "pypi_unreachable"
        assert payload["error"]["recovery"] == "pip install --upgrade celesto"
        assert payload["data"]["current"] is None
        assert payload["data"]["latest"] is None
        assert payload["data"]["checked_pypi"] is False
        assert "Could not determine" not in captured.out + captured.err

    def test_check_reports_unreachable_when_version_unknown(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The human-readable check names the unreachable PyPI, not the unknown version."""
        with (
            patch("celesto.cli.update._get_current_version", return_value=None),
            patch("celesto.cli.update._fetch_latest_from_pypi", return_value=None),
            patch("celesto.cli.update._is_uv_tool_install", return_value=False),
        ):
            rc = run_update(check=True)
        assert rc == 1
        captured = capsys.readouterr()
        assert "Could not determine the installed celesto version" not in captured.err
        assert "pypi.org" in captured.err
        assert "pip install --upgrade celesto" in captured.err
        assert "up to date" not in captured.out

    def test_check_reports_unknown_version_when_pypi_answers(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Reordering must not change the reachable-but-unknown outcome."""
        with (
            patch("celesto.cli.update._get_current_version", return_value=None),
            patch("celesto.cli.update._fetch_latest_from_pypi", return_value="1.0.0"),
            patch("celesto.cli.update._is_uv_tool_install", return_value=False),
        ):
            rc = run_update(check=True)
        assert rc == 1
        captured = capsys.readouterr()
        assert "Could not determine the installed celesto version" in captured.err
        assert "pip install --upgrade celesto" in captured.err
        assert "pypi.org" not in captured.err

    def test_check_json_keeps_unknown_version_payload_when_pypi_answers(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The JSON check for an unknown version on a reachable PyPI stays un-error'd."""
        with (
            patch("celesto.cli.update._get_current_version", return_value=None),
            patch("celesto.cli.update._fetch_latest_from_pypi", return_value="1.0.0"),
            patch("celesto.cli.update._is_uv_tool_install", return_value=False),
        ):
            rc = run_update(check=True, json_output=True)
        assert rc == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["data"]["current"] is None
        assert payload["data"]["latest"] is None
        assert payload["data"]["update_available"] is False
        assert payload["error"] is None
