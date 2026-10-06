"""B10: DataPolicy scores and calibrates DEV only with a ticket its ledger issued (A2, C11, K5).

B2 left `ScoringTicket` structural, so anything shaped like a ticket unlocked DEV (BUILD_LOG B2,
review 2, finding 6). The policy now asks the ledger it is bound to. These tests plant each kind
of forgery, built outside quant.research.ledger, and expect the runtime token and the ledger's
record of what it issued to refuse it. Every refused read leaves a ledger entry.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from quant.core.clock import SimClock
from quant.core.errors import LockboxError, PolicyError, TicketError
from quant.data.policy import DataPolicy, Purpose
from quant.research import ledger as ledger_module
from quant.research.ledger import EntryKind, TrialLedger, TrialTicket
from tests.support.bars import SYMBOL, Ticket, store_with_days
from tests.support.ledger import LEDGER_START, run_spec

DEV_READ = (date(2019, 1, 2), date(2019, 3, 29))
SCORE_AND_CALIBRATE = [Purpose.SCORE, Purpose.CALIBRATE]
FIELDS = ("trial_id", "run_id", "first_day", "last_day", "config_hash", "code_hash", "issued_at")


@pytest.fixture
def clock() -> SimClock:
    return SimClock(LEDGER_START)


@pytest.fixture
def ledger(tmp_path: Path, clock: SimClock) -> TrialLedger:
    return TrialLedger.open(tmp_path / "research.db", clock=clock)


def fields_of(ticket: TrialTicket) -> dict[str, Any]:
    return {name: getattr(ticket, name) for name in FIELDS}


class Forged(TrialTicket):  # type: ignore[misc]
    """A subclass that skips the token check and copies a real ticket's fields and token."""

    def __init__(self, **fields: Any) -> None:
        for name, value in fields.items():
            object.__setattr__(self, name, value)


# Each forgery is built outside quant.research.ledger from a genuine ticket of `ledger`.
Forgery = Callable[[TrialLedger, TrialTicket, Path, SimClock], object]


def _plain(ledger: TrialLedger, real: TrialTicket, tmp: Path, clock: SimClock) -> object:
    return SimpleNamespace(**fields_of(real))


def _stand_in(ledger: TrialLedger, real: TrialTicket, tmp: Path, clock: SimClock) -> object:
    return Ticket(real.trial_id, real.first_day, real.last_day)  # B2's stand-in


def _subclass(ledger: TrialLedger, real: TrialTicket, tmp: Path, clock: SimClock) -> object:
    return Forged(**fields_of(real), _token=ledger_module._TOKEN)


def _without_token(ledger: TrialLedger, real: TrialTicket, tmp: Path, clock: SimClock) -> object:
    forged = object.__new__(TrialTicket)
    for name, value in fields_of(real).items():
        object.__setattr__(forged, name, value)
    return forged


def _other_token(ledger: TrialLedger, real: TrialTicket, tmp: Path, clock: SimClock) -> object:
    forged = _without_token(ledger, real, tmp, clock)
    object.__setattr__(forged, "_token", object())
    return forged


def _never_issued(ledger: TrialLedger, real: TrialTicket, tmp: Path, clock: SimClock) -> object:
    return TrialTicket(token=ledger_module._TOKEN, **fields_of(real))  # a stolen token


def _changed(ledger: TrialLedger, real: TrialTicket, tmp: Path, clock: SimClock) -> object:
    narrow = ledger.open_run(run_spec("r2", first=date(2019, 1, 1), last=date(2019, 1, 31)))
    object.__setattr__(narrow, "last_day", date(2024, 12, 31))  # widened after issue
    return narrow


def _other_ledger(ledger: TrialLedger, real: TrialTicket, tmp: Path, clock: SimClock) -> object:
    return TrialLedger.memory(clock=clock).open_run(run_spec())


def _same_file(ledger: TrialLedger, real: TrialTicket, tmp: Path, clock: SimClock) -> object:
    return TrialLedger.open(tmp / "research.db", clock=clock).open_run(run_spec("r3"))


FORGERIES: dict[str, Forgery] = {
    "plain object": _plain,
    "B2 stand-in": _stand_in,
    "subclass": _subclass,
    "built without the token": _without_token,
    "another token": _other_token,
    "stolen token, never issued": _never_issued,
    "changed after issue": _changed,
    "another ledger": _other_ledger,
    "another ledger on the same file": _same_file,
}


@pytest.mark.spec("K5", "A2", "C11", "ARCH§4.4", "ARCH§6.3")
@pytest.mark.parametrize("purpose", SCORE_AND_CALIBRATE, ids=str)
@pytest.mark.parametrize("forge", FORGERIES.values(), ids=FORGERIES.keys())
def test_k5_tickets_built_outside_the_ledger_are_refused(
    tmp_path: Path, clock: SimClock, ledger: TrialLedger, purpose: Purpose, forge: Forgery
) -> None:
    real = ledger.open_run(run_spec())
    forged = forge(ledger, real, tmp_path, clock)
    runs = ledger.trial_count()
    policy = DataPolicy(ledger=ledger)
    with pytest.raises(TicketError):
        policy.check(purpose, *DEV_READ, ticket=forged)  # type: ignore[arg-type]
    with pytest.raises(TicketError):
        ledger.verify_ticket(forged)
    refusal = ledger.entries()[-1]
    assert refusal.kind is EntryKind.REFUSAL
    assert refusal.body["purpose"] == str(purpose)
    assert (refusal.body["first_day"], refusal.body["last_day"]) == ("2019-01-02", "2019-03-29")
    assert refusal.body["error"] == "TicketError"
    assert refusal.body["ticket"] == f"{type(forged).__qualname__} {forged.trial_id}"  # type: ignore[attr-defined]
    assert ledger.trial_count() == runs
    policy.check(purpose, *DEV_READ, ticket=real)  # the real one still works
    assert ledger.verify().ok


@pytest.mark.spec("C11", "A2", "K5", "ARCH§6.3")
def test_a_bound_policy_scores_and_calibrates_dev_with_its_ledgers_ticket(
    clock: SimClock, ledger: TrialLedger
) -> None:
    ticket = ledger.open_run(run_spec(first=date(2019, 1, 1), last=date(2019, 12, 31)))
    policy = DataPolicy(ledger=ledger)
    for purpose in SCORE_AND_CALIBRATE:
        policy.check(purpose, *DEV_READ, ticket=ticket)
    store = store_with_days(date(2019, 1, 2), date(2019, 1, 3))
    tau = datetime(2019, 1, 4, tzinfo=UTC)
    view = store.view(tau, as_of=tau, policy=policy, ticket=ticket)
    assert len(view.hourly(SYMBOL, purpose=Purpose.SCORE)) == 7 + 7
    assert [e.kind for e in ledger.entries()] == [EntryKind.RUN]  # allowed reads record nothing

    with pytest.raises(PolicyError):  # 2020 is outside the ticket
        policy.check(Purpose.SCORE, date(2020, 1, 2), date(2020, 3, 31), ticket=ticket)
    refusal = ledger.entries()[-1]
    assert (refusal.kind, refusal.body["error"]) == (EntryKind.REFUSAL, "PolicyError")
    assert refusal.body["ticket"] == "TrialTicket T1"


@pytest.mark.spec("C11", "A2", "K5")
def test_a_policy_without_a_ledger_refuses_every_dev_ticket(ledger: TrialLedger) -> None:
    ticket = ledger.open_run(run_spec())
    for purpose in SCORE_AND_CALIBRATE:
        with pytest.raises(PolicyError):
            DataPolicy().check(purpose, *DEV_READ, ticket=ticket)
    DataPolicy().check(Purpose.SCORE, date(2016, 1, 4), date(2018, 12, 31))  # FIT needs none


@pytest.mark.spec("A2", "C11", "ARCH§6.3")
def test_scoring_dev_without_a_ticket_raises_and_still_leaves_a_ledger_entry(
    ledger: TrialLedger,
) -> None:
    policy = DataPolicy(ledger=ledger)
    with pytest.raises(PolicyError) as refused:
        policy.check(Purpose.SCORE, *DEV_READ)
    assert type(refused.value) is PolicyError
    (entry,) = ledger.entries()
    assert entry.kind is EntryKind.REFUSAL
    assert entry.body == {
        "purpose": "score",
        "first_day": "2019-01-02",
        "last_day": "2019-03-29",
        "error": "PolicyError",
        "reason": str(refused.value),
        "ticket": None,
    }

    # Through a view, and for Kelly's edge estimates, the same.
    store = store_with_days(date(2019, 1, 2), date(2019, 1, 3))
    tau = datetime(2019, 1, 4, tzinfo=UTC)
    with pytest.raises(PolicyError):
        store.view(tau, as_of=tau, policy=policy).hourly(SYMBOL, purpose=Purpose.SCORE)
    with pytest.raises(PolicyError):
        policy.check(Purpose.CALIBRATE, *DEV_READ)
    refusals = ledger.entries()
    assert [e.body["purpose"] for e in refusals] == ["score", "score", "calibrate"]
    assert ledger.trial_count() == 0  # a refused read saw nothing: no trial
    assert ledger.verify().ok


@pytest.mark.spec("A2", "C1", "ARCH§6.3")
def test_a_refused_lockbox_read_leaves_a_ledger_entry(ledger: TrialLedger) -> None:
    ticket = ledger.open_run(run_spec())
    policy = DataPolicy(ledger=ledger)
    with pytest.raises(LockboxError):
        policy.check(Purpose.SCORE, date(2024, 12, 1), date(2025, 1, 2), ticket=ticket)
    with pytest.raises(LockboxError):
        policy.check(Purpose.REPORT, date(2025, 3, 3), date(2025, 3, 7))
    score, report = ledger.entries()[-2:]
    assert (score.body["purpose"], score.body["error"]) == ("score", "LockboxError")
    assert score.body["ticket"] == "TrialTicket T1"
    assert (report.body["purpose"], report.body["error"]) == ("report", "LockboxError")
    assert ledger.trial_count() == 1
