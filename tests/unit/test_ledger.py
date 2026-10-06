"""B10: the trial ledger, its trials and its tickets (A2, C11, K5, P9).

A run is appended before it may read DEV, and its ticket is issued only after the append. Trials
are counted as C11 defines them: the same configuration on the same code reuses its trial, and a
run whose decision records differ from every earlier scored run is a new one, even when only the
code changed. No lockbox ticket can be issued at all in v1.
"""

from __future__ import annotations

import copy
import dataclasses
import sqlite3
from datetime import date, timedelta

import pytest

from quant.core.canonical import sha256_hex
from quant.core.clock import SimClock
from quant.core.errors import LedgerError, LockboxError, PolicyError, TicketError
from quant.data.policy import Segments
from quant.research.ledger import EntryKind, TrialLedger
from quant.store.sqlite import connect
from tests.support.ledger import LEDGER_START, records, run_spec


@pytest.fixture
def clock() -> SimClock:
    return SimClock(LEDGER_START)


# --- tickets come after their entries (A2, K5) -------------------------------------------------


@pytest.mark.spec("A2", "K5", "ARCH§4.4")
def test_a_ticket_is_issued_only_after_its_run_is_appended(clock: SimClock) -> None:
    conn = connect(":memory:")
    ledger = TrialLedger(conn, clock=clock)
    ticket = ledger.open_run(run_spec())
    last = ledger.entries()[-1]
    assert last.kind is EntryKind.RUN
    assert (
        (last.body["run"], last.body["trial"]) == (ticket.run_id, ticket.trial_id) == ("R1", "T1")
    )
    assert last.body["first_day"] == "2019-01-01"
    assert last.body["last_day"] == "2024-12-31"
    assert (ticket.first_day, ticket.last_day) == (date(2019, 1, 1), date(2024, 12, 31))
    assert ticket.config_hash == run_spec().config_hash
    assert ticket.issued_at == clock.now()
    assert ledger.verify_ticket(ticket) is ticket

    # An append that fails issues nothing, and leaves nothing behind.
    conn.execute(
        "CREATE TEMP TRIGGER refuse BEFORE INSERT ON main.entries"
        " BEGIN SELECT RAISE(ABORT, 'the disk is full'); END"
    )
    with pytest.raises(sqlite3.DatabaseError):
        ledger.open_run(run_spec("r2"))
    conn.execute("DROP TRIGGER refuse")
    assert len(ledger.entries()) == 1
    assert ledger.trial_count() == 1
    assert ledger.open_run(run_spec("r2")).run_id == "R2"


@pytest.mark.spec("K5", "ARCH§4.4")
def test_a_ticket_cannot_be_changed_or_copied(clock: SimClock) -> None:
    ticket = TrialLedger.memory(clock=clock).open_run(run_spec())
    with pytest.raises(AttributeError):
        ticket.last_day = date(2025, 6, 30)
    with pytest.raises(AttributeError):
        del ticket.trial_id
    for duplicate in (copy.copy, copy.deepcopy):
        with pytest.raises(TypeError):
            duplicate(ticket)
    assert "T1" in repr(ticket)
    assert ticket.last_day == date(2024, 12, 31)


# --- trial identity (C11) ----------------------------------------------------------------------


@pytest.mark.spec("C11", "A2", "SPEC§11")
def test_an_identical_rerun_reuses_its_trial(clock: SimClock) -> None:
    ledger = TrialLedger.memory(clock=clock)
    first = ledger.open_run(run_spec())
    settled = ledger.settle(first, records("a"))
    assert (settled.run_id, settled.trial_id, settled.repeat_of) == ("R1", "T1", None)

    rerun = ledger.open_run(run_spec())  # the same configuration on the same code
    assert (rerun.run_id, rerun.trial_id) == ("R2", "T1")
    assert ledger.trial_count() == 1  # before its records are known
    again = ledger.settle(rerun, records("a"))
    assert (again.trial_id, again.repeat_of) == ("T1", "R1")
    assert ledger.trial_count() == 1

    # A run that crashed before settling still owns its trial, and its rerun reuses it.
    crashed = ledger.open_run(run_spec("r2"))
    assert crashed.trial_id == "T2"
    retry = ledger.open_run(run_spec("r2"))
    assert retry.trial_id == "T2"
    assert ledger.trial_count() == 2
    assert ledger.settle(retry, records("b")).trial_id == "T2"
    assert ledger.trial_count() == 2


@pytest.mark.spec("C11", "A2", "SPEC§11")
def test_a_run_with_different_records_adds_a_trial(clock: SimClock) -> None:
    ledger = TrialLedger.memory(clock=clock)
    ledger.settle(ledger.open_run(run_spec("r1", "c1")), records("a"))

    other = ledger.open_run(run_spec("r2", "c1"))  # another configuration
    assert other.trial_id == "T2"
    assert ledger.trial_count() == 2  # counted from the moment it may read DEV
    assert ledger.settle(other, records("b")).trial_id == "T2"

    recoded = ledger.open_run(run_spec("r1", "c2"))  # the code changed, and so did a decision
    assert recoded.trial_id == "T3"
    settled = ledger.settle(recoded, records("c"))
    assert (settled.trial_id, settled.repeat_of) == ("T3", None)
    assert ledger.trial_count() == 3

    narrower = ledger.open_run(run_spec("r1", "c1", first=date(2020, 1, 1)))  # other days
    assert narrower.trial_id == "T4"
    assert ledger.trial_count() == 4


@pytest.mark.spec("C11", "SPEC§11")
def test_a_code_change_that_moves_no_decision_is_no_new_trial(clock: SimClock) -> None:
    ledger = TrialLedger.memory(clock=clock)
    ledger.settle(ledger.open_run(run_spec("r1", "c1")), records("a"))
    refactored = ledger.open_run(run_spec("r1", "c2"))
    assert refactored.trial_id == "T2"
    assert ledger.trial_count() == 2  # until its records are known, it may be new
    settled = ledger.settle(refactored, records("a"))
    assert (settled.trial_id, settled.repeat_of) == ("T1", "R1")
    assert ledger.trial_count() == 1
    assert ledger.open_run(run_spec("r9")).trial_id == "T3"  # numbers are never reused
    assert ledger.trial_count() == 2


@pytest.mark.spec("C11", "SPEC§11")
def test_a_rerun_whose_decisions_changed_is_a_new_trial(clock: SimClock) -> None:
    ledger = TrialLedger.memory(clock=clock)
    ledger.settle(ledger.open_run(run_spec()), records("a"))
    rerun = ledger.open_run(run_spec())
    assert rerun.trial_id == "T1"
    settled = ledger.settle(rerun, records("b"))  # changed data, or a determinism bug
    assert (settled.trial_id, settled.repeat_of) == ("T2", None)
    assert ledger.trial_count() == 2


@pytest.mark.spec("C11", "K5", "A2")
def test_a_run_settles_once_with_its_own_ticket_and_a_real_hash(clock: SimClock) -> None:
    ledger = TrialLedger.memory(clock=clock)
    ticket = ledger.open_run(run_spec())
    foreign = TrialLedger.memory(clock=clock).open_run(run_spec())
    with pytest.raises(TicketError):
        ledger.settle(foreign, records("a"))
    with pytest.raises(LedgerError):
        ledger.settle(ticket, "not a hash")
    ledger.settle(ticket, records("a"))
    with pytest.raises(LedgerError):
        ledger.settle(ticket, records("a"))
    with pytest.raises(LedgerError):
        ledger.open_run(dataclasses.replace(run_spec(), code_hash="c1"))  # not SHA-256
    assert [e.kind for e in ledger.entries()] == [EntryKind.RUN, EntryKind.RESULT]
    assert ledger.trial_count() == 1


# --- no lockbox ticket (P9) --------------------------------------------------------------------


@pytest.mark.spec("P9", "A2", "ARCH§6.3")
@pytest.mark.parametrize(
    ("first", "last"),
    [
        (date(2024, 12, 1), date(2025, 1, 2)),  # DEV running into the lockbox
        (date(2025, 1, 2), date(2025, 6, 30)),  # inside it
        (date(2026, 10, 1), date(2026, 10, 30)),  # across the seal into forward data
    ],
    ids=["into", "inside", "across-the-seal"],
)
def test_no_lockbox_ticket_can_be_issued(clock: SimClock, first: date, last: date) -> None:
    ledger = TrialLedger.memory(clock=clock, segments=Segments(lockbox_end=date(2026, 10, 20)))
    with pytest.raises(LockboxError):
        ledger.open_run(run_spec(first=first, last=last))
    (refusal,) = ledger.entries()
    assert refusal.kind is EntryKind.REFUSAL
    assert refusal.body["purpose"] == "ticket"
    assert (refusal.body["first_day"], refusal.body["last_day"]) == (
        first.isoformat(),
        last.isoformat(),
    )
    assert refusal.body["error"] == "LockboxError"
    assert refusal.body["ticket"] is None
    assert ledger.trial_count() == 0
    assert ledger.verify().ok


@pytest.mark.spec("A2", "C11", "ARCH§6.3")
def test_a_ticket_covers_dev_days_only(clock: SimClock) -> None:
    ledger = TrialLedger.memory(clock=clock, segments=Segments(lockbox_end=date(2026, 10, 20)))
    for first, last in (
        (date(2018, 6, 1), date(2019, 6, 30)),  # FIT needs no ticket
        (date(2026, 10, 21), date(2026, 11, 30)),  # forward data is never scored
        (date(2020, 6, 1), date(2020, 5, 1)),  # an empty span
    ):
        with pytest.raises(PolicyError) as refused:
            ledger.open_run(run_spec(first=first, last=last))
        assert not isinstance(refused.value, LockboxError)
    assert [e.kind for e in ledger.entries()] == [EntryKind.REFUSAL] * 3
    assert ledger.trial_count() == 0


# --- gate configuration (ARCHITECTURE §10) -----------------------------------------------------


@pytest.mark.spec("A2", "ARCH§10")
def test_gate_config_changes_are_recorded(clock: SimClock) -> None:
    ledger = TrialLedger.memory(clock=clock)
    assert ledger.gate_config_hash() is None
    gates = {"G1": {"sharpe_above": 1.5}, "G2": {"max_drawdown_below": 0.15}}
    first = ledger.record_gate_config(gates, reason="the approved gates")
    assert first is not None
    assert first.kind is EntryKind.GATE_CONFIG
    assert first.body == {
        "config": gates,
        "config_hash": sha256_hex(gates),
        "previous": None,
        "reason": "the approved gates",
    }
    clock.advance(timedelta(days=1))
    same = {"G2": {"max_drawdown_below": 0.15}, "G1": {"sharpe_above": 1.5}}  # key order only
    assert ledger.record_gate_config(same, reason="restart") is None
    stricter = {**gates, "G1": {"sharpe_above": 1.6}}
    change = ledger.record_gate_config(stricter, reason="G1 to 1.6")
    assert change is not None
    assert change.body["previous"] == sha256_hex(gates)
    assert change.at == clock.now()
    assert ledger.gate_config_hash() == sha256_hex(stricter)
    assert [e.seq for e in ledger.entries()] == [1, 2]
    assert ledger.trial_count() == 0
    assert ledger.verify().ok
