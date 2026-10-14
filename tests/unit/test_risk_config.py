"""B9: config/risk.yaml holds spec §10's limits, loads only as committed, and is frozen (D13).

ARCHITECTURE §8.1 and §9.3: the engine refuses to start if risk.yaml differs from the committed
version. The git tests build a small repository in tmp_path, with git's per-repository
variables scrubbed, because the commit hook exports GIT_INDEX_FILE and friends.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pydantic
import pytest

from quant.core.errors import ConfigError
from quant.core.orders import Label
from quant.risk.limits import (
    RISK_YAML,
    RiskConfig,
    committed_bytes,
    load_risk_config,
    parse_risk_config,
)
from quant_check.repo import scrubbed_env
from tests.support.risk import RISK_YAML as COMMITTED_FILE


def git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        env=scrubbed_env(),
    )


def repo_with_risk_yaml(root: Path) -> Path:
    """A repository whose HEAD commits a copy of this repo's config/risk.yaml."""
    (root / "config").mkdir(parents=True)
    (root / RISK_YAML).write_bytes(COMMITTED_FILE.read_bytes())
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "risk")
    return root


@pytest.mark.spec("SPEC§10", "D13", "D7", "C3", "A3")
def test_the_committed_file_holds_the_spec_section_10_limits() -> None:
    data = COMMITTED_FILE.read_bytes()
    limits = load_risk_config(COMMITTED_FILE, committed=data)
    assert limits.max_gross_exposure == 1.0
    assert [limits.label_caps.cap(label) for label in Label] == [1.0, 0.5, 0.25, 0.0]
    assert limits.uncertain_cap == 0.0
    assert limits.overnight_cap == 0.5
    assert limits.single_order_cap == 0.5
    assert limits.daily_loss_fraction == 0.02
    assert limits.max_drawdown_fraction == 0.10
    assert limits.manual_approval_notional == 25_000.0
    assert limits.max_intents_per_day == 20
    assert limits.submission_halt_at == 60
    assert limits.collar_bp == 10.0
    assert limits.quote_max_age_s == 5.0
    assert limits.snapshot_max_age_s == 5.0


@pytest.mark.spec("SPEC§10", "C2", "C8", "ARCH§8.3", "ARCH§16")
def test_the_committed_file_holds_the_kill_settings() -> None:
    """B9b: the kill switch's settings that B9a left out (judgment call 8)."""
    limits = load_risk_config(COMMITTED_FILE, committed=COMMITTED_FILE.read_bytes())
    assert limits.monitor_price_max_age_s == 120.0
    kill = limits.kill
    assert (kill.requote_s, kill.widen_bp, kill.max_collar_bp) == (30.0, 10.0, 100.0)
    assert (kill.poll_s, kill.deadline_s) == (1.0, 300.0)
    assert (kill.order_errors, kill.order_error_window_s) == (5, 600.0)
    assert kill.heartbeat_lost_s == 300.0
    assert (kill.crash_loop_starts, kill.crash_loop_window_s) == (3, 600.0)
    assert kill.flatten_incomplete_s == 600.0
    assert kill.stop_missing_s == 300.0
    assert kill.stop_distance_bp == 100.0  # B9c: the stop a kill leaves when it can't finish


@pytest.mark.spec("ARCH§8.1", "ARCH§9.3", "SPEC§10")
@pytest.mark.parametrize(
    "edit",
    [
        lambda text: text.replace("0.02", "0.03"),
        lambda text: text + "\n",
        lambda text: text.replace("# Hard risk limits", "# Soft risk limits"),
    ],
    ids=["a limit", "a trailing newline", "a comment"],
)
def test_the_loader_refuses_a_file_that_differs_from_the_committed_one(
    tmp_path: Path, edit: object
) -> None:
    committed = COMMITTED_FILE.read_bytes()
    path = tmp_path / "risk.yaml"
    path.write_text(edit(committed.decode()))  # type: ignore[operator]
    assert path.read_bytes() != committed
    with pytest.raises(ConfigError):
        load_risk_config(path, committed=committed)


@pytest.mark.spec("ARCH§8.1", "ARCH§9.3")
def test_committed_bytes_reads_head_and_the_loader_refuses_local_edits(tmp_path: Path) -> None:
    repo = repo_with_risk_yaml(tmp_path / "repo")
    path = repo / RISK_YAML
    assert committed_bytes(repo) == path.read_bytes()
    assert load_risk_config(path, committed=committed_bytes(repo)).max_gross_exposure == 1.0

    path.write_text(path.read_text().replace("max_gross_exposure: 1.0", "max_gross_exposure: 0.9"))
    with pytest.raises(ConfigError):
        load_risk_config(path, committed=committed_bytes(repo))  # edited, not committed

    with pytest.raises(ConfigError):
        committed_bytes(repo, "config/gates.yaml")  # not tracked
    with pytest.raises(ConfigError):
        committed_bytes(tmp_path / "not-a-repo")
    with pytest.raises(ConfigError):
        load_risk_config(tmp_path / "missing.yaml", committed=b"")


@pytest.mark.spec("SPEC§10", "D13")
@pytest.mark.parametrize(
    "edit",
    [
        lambda text: text + "max_leverage: 2.0\n",
        lambda text: text.replace("overnight_cap: 0.5", ""),
        lambda text: text.replace("CRASH: 0.0", "CRASH: 0.1"),
        lambda text: text.replace("CALM_UP: 1.0", "CALM_UP: 1.5"),
        lambda text: text.replace("uncertain_cap: 0.0", "uncertain_cap: 0.2"),
        lambda text: text.replace("collar_bp: 10.0", 'collar_bp: "10"'),
        lambda text: text.replace("max_intents_per_day: 20", "max_intents_per_day: 20.5"),
        lambda text: text.replace("daily_loss_fraction: 0.02", "daily_loss_fraction: .nan"),
        lambda text: "- a list\n",
        lambda text: "key: [unclosed\n",
        lambda text: text.replace("  poll_s: 1.0", "  poll_s: 30.0"),
        lambda text: text.replace("  deadline_s: 300.0", "  deadline_s: 30.0"),
        lambda text: text.replace("  max_collar_bp: 100.0", "  max_collar_bp: 5.0"),
        lambda text: text.replace("  order_errors: 5", "  order_errors: 0"),
        lambda text: text.replace("  stop_missing_s: 300.0", "  stop_missing_s: 300.0\n  extra: 1"),
        lambda text: text.replace("monitor_price_max_age_s: 120.0", ""),
    ],
    ids=[
        "an unknown key",
        "a missing key",
        "CRASH above 0",
        "a label cap above gross",
        "uncertainty above 0",
        "a string",
        "a fractional count",
        "NaN",
        "not a mapping",
        "not YAML",
        "a poll as slow as a re-quote",
        "a deadline no longer than a re-quote",
        "a kill exit's widest collar inside the collar",
        "no order errors",
        "an unknown kill key",
        "no monitor price age",
    ],
)
def test_the_parser_refuses_malformed_or_inconsistent_limits(edit: object) -> None:
    text = edit(COMMITTED_FILE.read_text())  # type: ignore[operator]
    with pytest.raises(ConfigError):
        parse_risk_config(text.encode())


@pytest.mark.spec("SPEC§10", "ARCH§8.1")
def test_the_loaded_config_is_frozen() -> None:
    limits = parse_risk_config(COMMITTED_FILE.read_bytes())
    assert isinstance(limits, RiskConfig)
    with pytest.raises(pydantic.ValidationError):
        limits.max_gross_exposure = 2.0  # type: ignore[misc]
    with pytest.raises(pydantic.ValidationError):
        limits.label_caps.CRASH = 1.0  # type: ignore[misc]
    assert limits.label_caps.cap(Label.CRASH) == 0.0
