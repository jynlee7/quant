"""B10 review, part 2: gaps the first review of the ledger found in its writes, its CLI and the
policy's refusals (BUILD_LOG, B10).

1. When the ledger couldn't record a refusal (a broken chain, a locked file), the recording's
   LedgerError replaced the refusal's own error, so a refused lockbox read or a dead ticket
   escaped every `except PolicyError` (C5).
2. `verify_file` opened the WAL database read-only, which still created its -shm and -wal
   files, and failed outright in a directory it couldn't write (C7).
3. Every append rehashed and reparsed the whole chain under the write lock (C8).
4. Two processes appending to one file at once were never tested, and an append didn't notice
   a rewrite of the tail it had seen, which `verify` can't catch (C9).
5. A ticket's `issued_at` came from a second reading of the clock, not from its run entry (C10).
"""

from __future__ import annotations

import multiprocessing
import os
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import quant.research.ledger as ledger_module
from quant.cli.main import app
from quant.core.canonical import canonical_dumps, canonical_loads
from quant.core.clock import SimClock
from quant.core.epoch import from_us
from quant.core.errors import LedgerError, LockboxError, TicketError
from quant.data.policy import DataPolicy, Purpose, Refusal
from quant.research.ledger import TrialLedger, entry_hash, verify_file
from tests.support.ledger import LEDGER_START, filled_ledger, records, run_spec


@pytest.fixture
def clock() -> SimClock:
    return SimClock(LEDGER_START)


def _write(path: Path, *statements: str) -> None:
    """Run `statements` on the ledger file from another connection, behind the ledger's back."""
    with closing(sqlite3.connect(path, isolation_level=None)) as conn:
        for statement in statements:
            conn.execute(statement)


def _refusal(n: int) -> Refusal:
    return Refusal("score", date(2019, 1, 2), date(2019, 1, 3), "PolicyError", f"#{n}", None)


# 1. A refusal raises its own error, recorded or not (C5).


@pytest.mark.spec("A2", "K5", "ARCH§6.3")
@pytest.mark.xfail(strict=True, reason="red:B10", raises=LedgerError)
def test_a_dead_ticket_is_refused_as_a_ticket_when_nothing_can_be_recorded(
    tmp_path: Path, clock: SimClock
) -> None:
    path = tmp_path / "research.db"
    ledger = TrialLedger.open(path, clock=clock)
    ticket = ledger.open_run(run_spec())
    _write(path, "DELETE FROM entries WHERE seq = 1")  # the chain no longer verifies
    with pytest.raises(TicketError) as refused:
        DataPolicy(ledger=ledger).check(
            Purpose.SCORE, date(2019, 1, 2), date(2019, 3, 29), ticket=ticket
        )
    assert any("could not record" in note for note in refused.value.__notes__)


@pytest.mark.spec("A2", "C1", "ARCH§6.3")
@pytest.mark.xfail(strict=True, reason="red:B10", raises=LedgerError)
def test_a_lockbox_read_is_refused_as_one_when_nothing_can_be_recorded(
    tmp_path: Path, clock: SimClock
) -> None:
    path = tmp_path / "research.db"
    ledger = TrialLedger.open(path, clock=clock)
    ledger.record_refusal(_refusal(1))
    _write(path, "UPDATE entries SET at = at + 1 WHERE seq = 1")
    with pytest.raises(LockboxError):
        DataPolicy(ledger=ledger).check(Purpose.SCORE, date(2025, 1, 2), date(2025, 1, 3))


# 2. verify_file writes nothing and needs no write access (C7).


@pytest.mark.spec("A2", "ARCH§10")
@pytest.mark.xfail(strict=True, reason="red:B10", raises=AssertionError)
def test_verifying_a_closed_ledger_leaves_no_files_behind(tmp_path: Path) -> None:
    path = tmp_path / "research.db"
    filled_ledger(path, SimClock(LEDGER_START))
    assert os.listdir(tmp_path) == ["research.db"]
    assert verify_file(path).ok
    assert os.listdir(tmp_path) == ["research.db"]


@contextmanager
def _read_only(directory: Path) -> Iterator[None]:
    directory.chmod(0o555)
    try:
        yield
    finally:
        directory.chmod(0o755)


@pytest.mark.spec("A2", "ARCH§10")
@pytest.mark.xfail(strict=True, reason="red:B10", raises=LedgerError)
def test_a_ledger_in_a_read_only_directory_verifies(tmp_path: Path) -> None:
    path = tmp_path / "ro" / "research.db"
    path.parent.mkdir()
    filled_ledger(path, SimClock(LEDGER_START))
    with _read_only(path.parent):
        report = verify_file(path)
        result = CliRunner().invoke(app, ["ledger", "verify", "--db", str(path)])
    assert (report.ok, report.entries) == (True, 6)
    assert result.exit_code == 0
    assert result.output.startswith("6 entries;")


@pytest.mark.spec("A2", "ARCH§10")
def test_a_ledger_held_open_by_a_writer_verifies(tmp_path: Path, clock: SimClock) -> None:
    path = tmp_path / "research.db"
    ledger = TrialLedger.open(path, clock=clock)
    try:
        ledger.settle(ledger.open_run(run_spec()), records("a"))
        report = verify_file(path)
        assert (report.ok, report.entries) == (True, 2)
        ledger.record_refusal(_refusal(1))
        assert verify_file(path).entries == 3
    finally:
        ledger.close()


# 3. An append checks only what is new since the head it last verified (C8).


@pytest.mark.spec("A2", "ARCH§10")
@pytest.mark.xfail(strict=True, reason="red:B10", raises=AssertionError)
def test_an_append_rehashes_only_what_is_new(
    clock: SimClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = TrialLedger.memory(clock=clock)
    for n in range(100):
        ledger.record_refusal(_refusal(n))
    hashed: list[int] = []

    def counted(seq: int, *rest: Any) -> str:
        hashed.append(seq)
        return entry_hash(seq, *rest)

    monkeypatch.setattr(ledger_module, "entry_hash", counted)
    ledger.record_refusal(_refusal(100))
    ledger.open_run(run_spec())
    ledger.record_gate_config({"G1": {"sharpe_above": 1.5}}, reason="first gates")
    assert len(hashed) <= 6, f"three appends hashed {len(hashed)} entries"
    assert ledger.verify().ok
    assert ledger.verify().entries == 103


def _edit_last_body(path: Path) -> None:
    with closing(sqlite3.connect(path, isolation_level=None)) as conn:
        seq, text = conn.execute("SELECT seq, body FROM entries ORDER BY seq DESC").fetchone()
        body = canonical_loads(text)
        body["reason"] = "edited"
        conn.execute(
            "UPDATE entries SET body = ? WHERE seq = ?", (canonical_dumps(body).decode(), seq)
        )


def _rehash_last(path: Path) -> None:
    with closing(sqlite3.connect(path, isolation_level=None)) as conn:
        seq, kind, at, text, prev = conn.execute(
            "SELECT seq, kind, at, body, prev FROM entries ORDER BY seq DESC"
        ).fetchone()
        body = canonical_loads(text)
        body["reason"] = "edited"
        digest = entry_hash(seq, kind, from_us(at), body, prev)
        conn.execute(
            "UPDATE entries SET body = ?, hash = ? WHERE seq = ?",
            (canonical_dumps(body).decode(), digest, seq),
        )
        conn.execute("UPDATE head SET hash = ? WHERE id = 1", (digest,))


def _plant_bad_row(path: Path) -> None:
    with closing(sqlite3.connect(path, isolation_level=None)) as conn:
        seq, at, digest = conn.execute(
            "SELECT seq, at, hash FROM entries ORDER BY seq DESC"
        ).fetchone()
        conn.execute(
            "INSERT INTO entries (seq, kind, at, body, prev, hash) VALUES (?, 'refusal', ?, '{}', "
            "?, ?)",
            (seq + 1, at, digest, "f" * 64),
        )
        conn.execute("UPDATE head SET seq = ?, hash = ? WHERE id = 1", (seq + 1, "f" * 64))


# A rewrite that re-chains the tail (or empties the file) passes `verify`, which has no anchor
# outside research.db; a ledger object that has seen the head refuses to append over it.
UNANCHORED = pytest.mark.xfail(strict=True, reason="red:B10", raises=pytest.fail.Exception)
TAMPERS = [
    pytest.param(_edit_last_body, id="edit the last row"),
    pytest.param(_rehash_last, id="edit and rehash the last row and the head", marks=UNANCHORED),
    pytest.param(_plant_bad_row, id="plant a row with a bad hash"),
    pytest.param(
        lambda path: _write(path, "DELETE FROM entries WHERE seq = (SELECT MAX(seq) FROM entries)"),
        id="delete the last row",
    ),
    pytest.param(
        lambda path: _write(path, "UPDATE head SET hash = '" + "e" * 64 + "'"),
        id="rewrite the head",
    ),
    pytest.param(
        lambda path: _write(path, "DELETE FROM entries", "DELETE FROM head"),
        id="wipe both tables",
        marks=UNANCHORED,
    ),
]


@pytest.mark.spec("A2", "ARCH§10")
@pytest.mark.parametrize("tamper", TAMPERS)
def test_an_append_still_refuses_a_tampered_tail(
    tmp_path: Path, clock: SimClock, tamper: Callable[[Path], None]
) -> None:
    path = tmp_path / "research.db"
    ledger = TrialLedger.open(path, clock=clock)
    for n in range(3):
        ledger.record_refusal(_refusal(n))
    tamper(path)
    with pytest.raises(LedgerError):
        ledger.record_refusal(_refusal(3))
    with pytest.raises(LedgerError):
        ledger.open_run(run_spec())


@pytest.mark.spec("A2", "ARCH§10")
def test_a_failed_append_leaves_the_chain_as_it_was(tmp_path: Path, clock: SimClock) -> None:
    path = tmp_path / "research.db"
    ledger = TrialLedger.open(path, clock=clock)
    ledger.record_refusal(_refusal(0))
    # The entry is written, then moving the head fails, so the transaction rolls back.
    _write(path, "CREATE TRIGGER stuck BEFORE UPDATE ON head BEGIN SELECT RAISE(ABORT, 'no'); END")
    with pytest.raises(sqlite3.Error):
        ledger.record_refusal(_refusal(1))
    _write(path, "DROP TRIGGER stuck")
    ledger.record_refusal(_refusal(2))
    assert ledger.verify().ok
    assert [entry.body["reason"] for entry in ledger.entries()] == ["#0", "#2"]


@pytest.mark.spec("A2", "C11", "ARCH§10")
def test_two_ledger_objects_on_one_file_take_turns(tmp_path: Path, clock: SimClock) -> None:
    path = tmp_path / "research.db"
    first = TrialLedger.open(path, clock=clock)
    second = TrialLedger.open(path, clock=clock)
    for n in range(5):
        first.record_refusal(_refusal(n))
        second.settle(second.open_run(run_spec(config=f"b{n}")), records(f"b{n}"))
    assert first.verify().ok
    assert first.trial_count() == second.trial_count() == 5
    assert len(first.entries()) == 15


# 4. Two processes append to one file at once (C9).

PER_PROCESS = 10


def _append_runs(path: str, name: str, start: Any) -> None:
    """In a child process: open the ledger, wait for the other child, then open and settle
    PER_PROCESS runs as fast as possible."""
    ledger = TrialLedger.open(Path(path), clock=SimClock(LEDGER_START))
    try:
        start.wait(timeout=30)
        for n in range(PER_PROCESS):
            spec = run_spec(config=f"{name}{n}", label=f"{name}{n}")
            ledger.settle(ledger.open_run(spec), records(f"{name}{n}"))
    finally:
        ledger.close()


@pytest.mark.spec("A2", "C11", "ARCH§10")
def test_two_processes_appending_at_once_keep_one_valid_chain(
    tmp_path: Path, clock: SimClock
) -> None:
    path = tmp_path / "research.db"
    TrialLedger.open(path, clock=clock).close()
    context = multiprocessing.get_context("spawn")
    start = context.Barrier(2)
    children = [
        context.Process(target=_append_runs, args=(str(path), name, start)) for name in "ab"
    ]
    for child in children:
        child.start()
    for child in children:
        child.join(timeout=60)
    assert [child.exitcode for child in children] == [0, 0]
    report = verify_file(path)
    assert (report.ok, report.entries) == (True, 4 * PER_PROCESS)
    ledger = TrialLedger.open(path, clock=clock)
    try:
        runs = [entry.body for entry in ledger.entries() if entry.kind == "run"]
        assert sorted(run["run"] for run in runs) == sorted(
            f"R{n}" for n in range(1, 2 * PER_PROCESS + 1)
        )
        assert sorted(run["label"] for run in runs) == sorted(
            f"{name}{n}" for name in "ab" for n in range(PER_PROCESS)
        )
        assert ledger.trial_count() == 2 * PER_PROCESS
    finally:
        ledger.close()


# 5. A ticket is issued at its run entry's time (C10).


class TickingClock(SimClock):
    """A clock that moves a second forward every time it is read."""

    def now(self) -> datetime:
        moment = super().now()
        self.advance(timedelta(seconds=1))
        return moment


@pytest.mark.spec("K5", "ARCH§10")
@pytest.mark.xfail(strict=True, reason="red:B10", raises=AssertionError)
def test_a_ticket_is_issued_at_its_run_entrys_time() -> None:
    ledger = TrialLedger.memory(clock=TickingClock(LEDGER_START))
    ledger.record_refusal(_refusal(0))
    ticket = ledger.open_run(run_spec())
    run = ledger.entries()[-1]
    assert run.body["run"] == ticket.run_id
    assert ticket.issued_at == run.at
