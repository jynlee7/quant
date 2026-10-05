"""pytest plugin loaded by every test run (pyproject addopts).

- Tiers (P4): `--tier fast|full|integration` selects tests by marker. The fast tier runs on every
  commit, deselects `full` and `integration` tests, and blocks network access. `all` is for
  collecting the trace matrix.
- Red marks (P2): every xfail must be a red mark, `xfail(strict=True, reason="red:BNN",
  raises=...)`, so a red test that passes, or that fails for any reason other than the one it
  names, breaks the build.
- Trace: every test names the spec ids it enforces with `@pytest.mark.spec(...)`, and every id
  must be defined in the docs.
- `--quant-report` and `--quant-trace` write JSON that quant-check turns into findings.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any

import pytest

from quant_check.trace import ids_in_docs

TIERS = ("fast", "full", "integration", "all")
RED_REASON = re.compile(r"^red:(B\d+)$")
RED_MARK_SHAPE = 'xfail(strict=True, reason="red:BNN", raises=<the failure it expects>)'
_XPASS_STRICT = "[XPASS(strict)]"


class OfflineTierError(RuntimeError):
    """A fast-tier test tried to reach the network (P4)."""


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("quant", "quant build gate")
    group.addoption(
        "--tier",
        choices=TIERS,
        default=os.environ.get("QUANT_TIER", "fast"),
        help="test tier (P4): fast on every commit, full before merges and nightly",
    )
    group.addoption("--quant-report", metavar="PATH", help="write outcomes as JSON for quant-check")
    group.addoption("--quant-trace", metavar="PATH", help="write each test's spec ids as JSON")


def pytest_configure(config: pytest.Config) -> None:
    for line in (
        "spec(*ids): the spec, architecture or plan ids the test enforces",
        "full: full tier only, run before each merge and nightly (P4)",
        "integration: needs the Alpaca test account (AD3); run weekly and on demand",
    ):
        config.addinivalue_line("markers", line)
    tier = str(config.getoption("--tier"))
    _load_hypothesis_profile(tier)
    if tier == "fast":
        _block_network()
    config.pluginmanager.register(_Gate(config, tier), "quant-gate")


def _load_hypothesis_profile(tier: str) -> None:
    from hypothesis import settings

    # The fast tier gates commits, so it replays the same examples every run.
    settings.register_profile(
        "fast", max_examples=50, derandomize=True, database=None, deadline=None, print_blob=True
    )
    settings.register_profile("full", max_examples=500, deadline=None, print_blob=True)
    settings.load_profile("full" if tier == "full" else "fast")


def _is_local(host: object) -> bool:
    if host is None or host in ("localhost", b"localhost"):
        return True
    try:
        ipaddress.ip_address(host.decode() if isinstance(host, bytes) else str(host))
    except ValueError:
        return False
    return True


def _block_network() -> None:
    """Refuse IP connections and name lookups for the rest of this process."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def refuse(what: object) -> OfflineTierError:
        return OfflineTierError(
            f"the fast tier is offline (P4): a test tried to reach {what!r}; "
            "mark it full or integration, or use a recorded response"
        )

    def connect(self: socket.socket, address: Any) -> None:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise refuse(address)
        real_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> int:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            raise refuse(address)
        return real_connect_ex(self, address)

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if not _is_local(host):
            raise refuse(host)
        return real_getaddrinfo(host, *args, **kwargs)

    # setattr, because the replacements take socket.socket where typeshed declares _socket.socket.
    setattr(socket.socket, "connect", connect)  # noqa: B010
    setattr(socket.socket, "connect_ex", connect_ex)  # noqa: B010
    socket.getaddrinfo = getaddrinfo


def _tier_of(item: pytest.Item) -> str:
    if item.get_closest_marker("integration"):
        return "integration"
    return "full" if item.get_closest_marker("full") else "fast"


def _selected(item: pytest.Item, tier: str) -> bool:
    own = _tier_of(item)
    return {
        "fast": own == "fast",
        "full": own in ("fast", "full"),
        "integration": own == "integration",
        "all": True,
    }[tier]


def _red_step(item: pytest.Item) -> str | None:
    for mark in item.iter_markers("xfail"):
        match = RED_REASON.match(str(mark.kwargs.get("reason", "")))
        if match:
            return match.group(1)
    return None


def _spec_ids(item: pytest.Item) -> list[str]:
    return sorted({str(arg) for mark in item.iter_markers("spec") for arg in mark.args})


def _problem(rule: str, nodeid: str, message: str) -> dict[str, str]:
    return {"rule": rule, "nodeid": nodeid, "message": message}


def _mark_problems(item: pytest.Item, known: frozenset[str] | None) -> Iterator[dict[str, str]]:
    ids = _spec_ids(item)
    if not ids:
        yield _problem("trace", item.nodeid, "names no spec id; add @pytest.mark.spec(...)")
    for cid in ids:
        if known is not None and cid not in known:
            yield _problem("trace", item.nodeid, f"names {cid}, which no doc defines")
    for mark in item.iter_markers("xfail"):
        reason = str(mark.kwargs.get("reason", ""))
        if (
            mark.args
            or mark.kwargs.get("strict") is not True
            or not RED_REASON.match(reason)
            or "raises" not in mark.kwargs
        ):
            yield _problem("P2", item.nodeid, f"every xfail must be a red mark: {RED_MARK_SHAPE}")


def _outcome(report: pytest.TestReport) -> str:
    if report.failed:
        if report.when != "call":
            return "error"
        if isinstance(report.longrepr, str) and report.longrepr.startswith(_XPASS_STRICT):
            return "xpass-strict"
        return "failed"
    if report.skipped:
        return "xfailed" if hasattr(report, "wasxfail") else "skipped"
    return "passed"


def _message(report: pytest.TestReport | pytest.CollectReport) -> str:
    longrepr = report.longrepr
    crash = getattr(longrepr, "reprcrash", None)
    if crash is not None:
        text = str(crash.message)
    elif isinstance(longrepr, tuple):
        text = str(longrepr[2])
    else:
        text = str(longrepr or "")
    text = " ".join(text.split())
    return text if len(text) <= 300 else text[:297] + "..."


class _Gate:
    """Per-run state: selection, mark checks, outcomes and the JSON outputs."""

    def __init__(self, config: pytest.Config, tier: str) -> None:
        self.config = config
        self.tier = tier
        self.problems: list[dict[str, str]] = []
        self.results: dict[str, dict[str, str | None]] = {}
        self.red: dict[str, str | None] = {}

    def pytest_collection_modifyitems(self, items: list[pytest.Item]) -> None:
        known = ids_in_docs(self.config.rootpath)
        trace: list[dict[str, Any]] = []
        keep: list[pytest.Item] = []
        drop: list[pytest.Item] = []
        for item in items:
            self.problems.extend(_mark_problems(item, known))
            self.red[item.nodeid] = _red_step(item)
            trace.append(
                {
                    "nodeid": item.nodeid,
                    "ids": _spec_ids(item),
                    "red": self.red[item.nodeid],
                    "tier": _tier_of(item),
                }
            )
            (keep if _selected(item, self.tier) else drop).append(item)

        trace_path = self.config.getoption("--quant-trace")
        if trace_path:
            Path(trace_path).write_text(json.dumps(trace, indent=1))
        if drop:
            self.config.hook.pytest_deselected(items=drop)
            items[:] = keep
        if self.problems:
            self.write_report(collected=len(items), exitstatus=int(pytest.ExitCode.USAGE_ERROR))
            listed = "\n".join(
                f"  {p['rule']} {p['nodeid']}: {p['message']}" for p in self.problems
            )
            raise pytest.UsageError(f"test marks break the build rules:\n{listed}")

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_runtest_makereport(
        self, item: pytest.Item
    ) -> Generator[None, pytest.TestReport, pytest.TestReport]:
        report = yield
        if (
            report.when == "call"
            and report.failed
            and isinstance(report.longrepr, str)
            and report.longrepr.startswith(_XPASS_STRICT)
        ):
            step = self.red.get(item.nodeid)
            report.longrepr = (
                f"{_XPASS_STRICT} red:{step}: P2, this red test passes. "
                "Remove its mark in the step's green commit."
            )
        return report

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.failed:
            self.results[report.nodeid or "collection"] = {
                "outcome": "error",
                "message": _message(report),
                "red": None,
            }

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when == "call" or report.failed or (report.when == "setup" and report.skipped):
            self.results[report.nodeid] = {
                "outcome": _outcome(report),
                "message": "" if report.passed else _message(report),
                "red": self.red.get(report.nodeid),
            }

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        self.write_report(collected=session.testscollected, exitstatus=int(exitstatus))

    def write_report(self, *, collected: int, exitstatus: int) -> None:
        path = self.config.getoption("--quant-report")
        if not path:
            return
        results = [{"nodeid": nodeid, **result} for nodeid, result in self.results.items()]
        Path(path).write_text(
            json.dumps(
                {
                    "tier": self.tier,
                    "exitstatus": exitstatus,
                    "collected": collected,
                    "problems": self.problems,
                    "results": results,
                },
                indent=1,
            )
        )
