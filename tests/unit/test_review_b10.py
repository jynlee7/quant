"""B10 review: gaps the review of the ledger's diff found (BUILD_LOG, B10).

1. The spec fixes where DEV and the lockbox start (A2, C1), but `Segments` let a caller move
   them, so a ledger built with a later lockbox issued a ticket for 2025 (R1). The ledger and
   the policy bound to it also kept separate segments (C6).
2. `verify_ticket` compared a ticket's fields with `==`, which a date subclass controls, and
   `open_run` checked a RunSpec's days with the caller's own comparisons (R2).
3. A ticket outlived its run entry, and the counts read the chain without checking it (R3, C1).
4. A crashed run stayed a trial of its own after an identical rerun joined an earlier trial
   (R4).
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from quant.core.canonical import canonical_dumps, canonical_loads
from quant.core.clock import SimClock
from quant.core.errors import ConfigError, LedgerError, TicketError
from quant.data.policy import DataPolicy, Purpose, Segments
from quant.research.ledger import TrialLedger
from tests.support.ledger import LEDGER_START, records, run_spec

SEAL = date(2026, 10, 20)


@pytest.fixture
def clock() -> SimClock:
    return SimClock(LEDGER_START)


def _write(path: Path, *statements: str) -> None:
    """Run `statements` on the ledger file from another connection, behind the ledger's back."""
    with closing(sqlite3.connect(path, isolation_level=None)) as conn:
        for statement in statements:
            conn.execute(statement)


class Reach(date):
    """A day that says it is at or after any DEV day, but not the lockbox's first."""

    def __ge__(self, other: object) -> bool:
        return isinstance(other, date) and other < date(2025, 1, 1)

    __hash__ = date.__hash__


class Equal(date):
    """A day equal to every other."""

    def __eq__(self, other: object) -> bool:
        return True

    __hash__ = date.__hash__


# 1. One lockbox, where the spec puts it (R1, C6).


@pytest.mark.spec("A2", "C1", "ARCH§6.3")
@pytest.mark.parametrize(
    ("start", "day"),
    [
        ("fit_start", date(2017, 1, 1)),
        ("dev_start", date(2020, 1, 1)),  # 2019 would be FIT, scored without a ticket
        ("lockbox_start", date(2026, 1, 1)),  # 2025 would be DEV, ticketed and scored
    ],
)
def test_no_caller_moves_where_a_segment_starts(start: str, day: date) -> None:
    with pytest.raises(TypeError):
        Segments(**{start: day})


@pytest.mark.spec("A2", "C1", "ARCH§6.3")
def test_segments_cannot_be_subclassed_to_move_the_lockbox() -> None:
    with pytest.raises(TypeError):

        class Moved(Segments):  # type: ignore[misc]
            lockbox_start = date(2026, 1, 1)


@pytest.mark.spec("A2", "C1", "ARCH§6.3")
def test_a_policy_and_its_ledger_agree_on_the_segments(clock: SimClock) -> None:
    unsealed = TrialLedger.memory(clock=clock)
    sealed = TrialLedger.memory(clock=clock, segments=Segments(lockbox_end=SEAL))
    with pytest.raises(ConfigError):
        DataPolicy(Segments(lockbox_end=SEAL), ledger=unsealed)
    with pytest.raises(ConfigError):
        DataPolicy(Segments(), ledger=sealed)
    DataPolicy(Segments(), ledger=unsealed)
    DataPolicy(Segments(lockbox_end=SEAL), ledger=sealed)
    assert sealed.segments == Segments(lockbox_end=SEAL)


# 2. A ticket's days are exactly what the ledger recorded (R2).


@pytest.mark.spec("K5", "A2", "C11")
def test_a_day_that_equals_anything_does_not_pass_for_the_issued_one(clock: SimClock) -> None:
    ledger = TrialLedger.memory(clock=clock)
    ticket = ledger.open_run(run_spec(first=date(2019, 1, 1), last=date(2019, 1, 31)))
    object.__setattr__(ticket, "last_day", Equal(2024, 12, 31))
    with pytest.raises(TicketError):
        ledger.verify_ticket(ticket)
    with pytest.raises(TicketError):
        DataPolicy(ledger=ledger).check(
            Purpose.SCORE, date(2019, 1, 2), date(2024, 12, 30), ticket=ticket
        )


@pytest.mark.spec("K5", "A2", "C11")
@pytest.mark.parametrize("field", ["first_day", "last_day", "config_hash", "code_hash", "label"])
def test_a_run_spec_must_hold_plain_dates_and_strings(clock: SimClock, field: str) -> None:
    class Text(str):
        __slots__ = ()

    spec = run_spec(first=date(2019, 1, 1), last=date(2019, 1, 31))
    odd: object = Text(getattr(spec, field))
    if field == "last_day":
        odd = Reach(2019, 1, 31)  # recorded as January, yet it would cover all of DEV
    elif field == "first_day":
        odd = Reach(2019, 1, 1)
    ledger = TrialLedger.memory(clock=clock)
    with pytest.raises(LedgerError):
        ledger.open_run(replace(spec, **{field: odd}))
    assert ledger.entries() == []


# 3. Reads come from a checked chain, and a ticket dies with its run entry (R3, C1).


@pytest.mark.spec("A2", "K5", "ARCH§10")
@pytest.mark.parametrize(
    "tamper",
    [
        ("DELETE FROM entries WHERE seq = 1",),
        ("DELETE FROM entries", "DELETE FROM head"),
    ],
    ids=["its-row", "every-row-and-the-head"],
)
def test_a_ticket_dies_with_its_run_entry(
    tmp_path: Path, clock: SimClock, tamper: tuple[str, ...]
) -> None:
    path = tmp_path / "research.db"
    ledger = TrialLedger.open(path, clock=clock)
    ticket = ledger.open_run(run_spec())
    policy = DataPolicy(ledger=ledger)
    policy.check(Purpose.SCORE, date(2019, 1, 2), date(2019, 3, 29), ticket=ticket)
    _write(path, *tamper)
    with pytest.raises(TicketError):
        ledger.verify_ticket(ticket)
    with pytest.raises(TicketError):  # recorded or not, the refusal's own error
        policy.check(Purpose.SCORE, date(2019, 1, 2), date(2019, 3, 29), ticket=ticket)
    with pytest.raises(TicketError):
        ledger.settle(ticket, records("a"))


@pytest.mark.spec("A2", "C11", "ARCH§10")
def test_counts_come_from_a_chain_that_verifies(tmp_path: Path, clock: SimClock) -> None:
    path = tmp_path / "research.db"
    ledger = TrialLedger.open(path, clock=clock)
    ledger.record_gate_config({"G1": {"sharpe_above": 1.5}}, reason="first gates")
    for name in ("a", "b", "c"):
        ledger.settle(ledger.open_run(run_spec(config=name)), records(name))
    assert ledger.trial_count() == 3
    with closing(sqlite3.connect(path, isolation_level=None)) as conn:
        for seq, text in conn.execute("SELECT seq, body FROM entries WHERE seq > 1").fetchall():
            body = canonical_loads(text)
            body["trial"] = "T1"  # every run in one trial, without rehashing
            conn.execute(
                "UPDATE entries SET body = ? WHERE seq = ?", (canonical_dumps(body).decode(), seq)
            )
    assert not ledger.verify().ok
    for read in (ledger.trial_count, ledger.gate_config_hash, ledger.entries):
        with pytest.raises(LedgerError):
            read()


@pytest.mark.spec("A2", "ARCH§10")
def test_an_unknown_kind_is_a_broken_chain(tmp_path: Path, clock: SimClock) -> None:
    path = tmp_path / "research.db"
    ledger = TrialLedger.open(path, clock=clock)
    ledger.record_gate_config({"G1": {"sharpe_above": 1.5}}, reason="first gates")
    _write(path, "UPDATE entries SET kind = 'bogus' WHERE seq = 1")
    for read in (ledger.trial_count, ledger.gate_config_hash, ledger.entries):
        with pytest.raises(LedgerError):
            read()


# 4. A crashed run of a configuration goes where its identical rerun goes (R4).


@pytest.mark.spec("C11")
def test_a_crashed_run_follows_its_identical_rerun(clock: SimClock) -> None:
    ledger = TrialLedger.memory(clock=clock)
    ledger.settle(ledger.open_run(run_spec(code="A")), records("X"))
    ledger.open_run(run_spec(code="B"))  # crashes before it settles
    rerun = ledger.open_run(run_spec(code="B"))
    assert ledger.trial_count() == 2  # B counts until its records are known
    settled = ledger.settle(rerun, records("X"))
    assert (settled.trial_id, settled.repeat_of) == ("T1", "R1")
    assert ledger.trial_count() == 1  # the same configuration on the same code: the same trial
