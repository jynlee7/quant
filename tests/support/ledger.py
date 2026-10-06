"""Trial ledgers and run specs for the B10 tests."""

from __future__ import annotations

import contextlib
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from quant.core.canonical import sha256_hex
from quant.core.clock import SimClock
from quant.core.errors import PolicyError
from quant.data.policy import DataPolicy, Purpose
from quant.research.ledger import RunSpec, TrialLedger

LEDGER_START = datetime(2026, 11, 3, 23, 30, tzinfo=UTC)  # an evening of the nightly harness
DEV_FIRST = date(2019, 1, 1)
DEV_LAST = date(2024, 12, 31)


def run_spec(
    config: str = "r1",
    code: str = "c1",
    *,
    first: date = DEV_FIRST,
    last: date = DEV_LAST,
    label: str = "r1",
) -> RunSpec:
    """A run of configuration `config` on code `code`, named by their hashes."""
    return RunSpec(
        config_hash=sha256_hex({"config": config}),
        code_hash=sha256_hex({"code": code}),
        first_day=first,
        last_day=last,
        label=label,
    )


def records(name: str) -> str:
    """The hash of a run's decision records, named `name`."""
    return sha256_hex({"records": name})


def filled_ledger(path: Path, clock: SimClock) -> None:
    """A ledger file holding one of each kind of entry, closed: two runs (one rerun), their
    results, a refused read and a gate configuration: six entries."""
    ledger = TrialLedger.open(path, clock=clock)
    try:
        ledger.record_gate_config({"G1": {"sharpe_above": 1.5}}, reason="first gates")
        clock.advance(timedelta(minutes=1))
        first = ledger.open_run(run_spec())
        ledger.settle(first, records("a"))
        clock.advance(timedelta(minutes=1))
        rerun = ledger.open_run(run_spec())
        ledger.settle(rerun, records("a"))
        clock.advance(timedelta(minutes=1))
        with contextlib.suppress(PolicyError):  # a DEV score without a ticket: refused, recorded
            DataPolicy(ledger=ledger).check(Purpose.SCORE, date(2019, 1, 2), date(2019, 3, 29))
    finally:
        ledger.close()
