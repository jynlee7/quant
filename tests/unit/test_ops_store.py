"""B7a: ops.db's pointer history, refit events and the entry freeze (C5, C7, ARCHITECTURE §9.1,
§9.2, spec §5), and the calendar's next open.

A pointer takes effect at the next session open after it is recorded, never mid-session,
across weekends and holidays. `current(kind, at)` is the latest row effective by `at`: the open
itself is the first instant it governs. History is append-only. A failed refit freezes entries
from its recording; the next passing refit unfreezes them at the open its model starts
governing; a manual acknowledgement unfreezes them at once, and is logged.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from structlog.testing import capture_logs

from quant.core.calendar import NEW_YORK, TradingCalendar
from quant.core.clock import SimClock
from quant.core.errors import ClockError, DataError

RED = pytest.mark.xfail(strict=True, reason="red:B7", raises=NotImplementedError)
XNYS = TradingCalendar.xnys()
US = timedelta(microseconds=1)
H0, H1, H2 = "a0" * 32, "a1" * 32, "a2" * 32


def ny(*args: int) -> datetime:
    """A New York wall time, in UTC."""
    return datetime(*args, tzinfo=NEW_YORK).astimezone(UTC)


class ManualClock:
    """A clock that can be set anywhere, backwards included."""

    def __init__(self, now: datetime) -> None:
        self.time = now

    def now(self) -> datetime:
        return self.time

    def sleep_until(self, when: datetime) -> None:
        raise AssertionError("the store never sleeps")


def _store(clock: Any) -> Any:
    from quant.store.ops import OpsStore

    return OpsStore.memory(clock=clock)


# --- The next open ----------------------------------------------------------------------------

NEXT_OPEN = {
    "friday evening to monday": (ny(2017, 6, 30, 17, 0), ny(2017, 7, 3, 9, 30)),
    "half day evening over the fourth of july": (ny(2017, 7, 3, 17, 0), ny(2017, 7, 5, 9, 30)),
    "thanksgiving eve": (ny(2017, 11, 22, 17, 0), ny(2017, 11, 24, 9, 30)),
    "mid-session to the next day": (ny(2017, 6, 29, 11, 0), ny(2017, 6, 30, 9, 30)),
    "exactly at an open": (ny(2017, 6, 29, 9, 30), ny(2017, 6, 30, 9, 30)),
    "before a session day's open": (ny(2017, 6, 29, 8, 0), ny(2017, 6, 29, 9, 30)),
    "a saturday": (ny(2017, 7, 1, 12, 0), ny(2017, 7, 3, 9, 30)),
    "winter time": (ny(2017, 12, 1, 17, 0), ny(2017, 12, 4, 9, 30)),
}


@RED
@pytest.mark.spec("C7", "ARCH§9.2")
@pytest.mark.parametrize("case", sorted(NEXT_OPEN))
def test_the_next_open_is_the_first_session_open_strictly_after(case: str) -> None:
    at, expected = NEXT_OPEN[case]
    opened = XNYS.next_open(at)
    assert opened == expected
    assert opened.tzinfo is UTC or opened.utcoffset() == timedelta(0)
    assert XNYS.next_open(opened - US) == opened


@RED
@pytest.mark.spec("C7")
def test_the_next_open_needs_an_aware_time() -> None:
    with pytest.raises(ClockError):
        XNYS.next_open(datetime(2017, 6, 30, 17, 0))  # noqa: DTZ001 (naive on purpose)


# --- Pointers ---------------------------------------------------------------------------------


@RED
@pytest.mark.spec("C7", "ARCH§9.2", "SPEC§5")
def test_a_pointer_governs_from_the_next_open_across_a_weekend_and_a_holiday() -> None:
    clock = SimClock(ny(2017, 6, 29, 17, 0))  # Thursday evening
    store = _store(clock)
    first = store.activate("model", H0, reason="first model")
    assert (first.effective_at, first.recorded_at) == (ny(2017, 6, 30, 9, 30), clock.now())

    clock.advance_to(ny(2017, 6, 30, 17, 0))  # Friday evening: a refit
    refit = store.activate("model", H1, reason="refit")
    monday = ny(2017, 7, 3, 9, 30)
    assert refit.effective_at == monday
    for t in (clock.now(), ny(2017, 6, 30, 23, 59), ny(2017, 7, 1, 12, 0), monday - US):
        assert store.current("model", t).artifact == H0, t
    assert store.current("model", monday).artifact == H1  # the open itself governs
    assert store.current("model", ny(2017, 7, 3, 12, 59)).artifact == H1

    clock.advance_to(ny(2017, 7, 3, 17, 0))  # Monday, a half day, evening: roll back
    back = store.rollback("model", H0, reason="rollback")
    wednesday = ny(2017, 7, 5, 9, 30)  # the Fourth is a holiday
    assert back.effective_at == wednesday
    assert store.current("model", ny(2017, 7, 4, 12, 0)).artifact == H1
    assert store.current("model", wednesday - US).artifact == H1
    assert store.current("model", wednesday).artifact == H0

    history = store.pointers("model")
    assert [(p.artifact, p.reason) for p in history] == [
        (H0, "first model"),
        (H1, "refit"),
        (H0, "rollback"),
    ]
    assert [p.seq for p in history] == sorted(p.seq for p in history)
    assert len({p.seq for p in history}) == 3


@RED
@pytest.mark.spec("C7", "ARCH§9.2")
def test_no_pointer_governs_before_the_first_takes_effect() -> None:
    clock = SimClock(ny(2017, 6, 29, 17, 0))
    store = _store(clock)
    assert store.current("model", clock.now()) is None
    store.activate("model", H0, reason="first model")
    assert store.current("model", ny(2017, 6, 30, 9, 30) - US) is None
    assert store.current("calibration", ny(2018, 1, 2, 12, 0)) is None  # kinds are separate
    with pytest.raises(ClockError):
        store.current("model", datetime(2017, 7, 1))  # noqa: DTZ001 (naive on purpose)


@RED
@pytest.mark.spec("ARCH§9.2")
def test_each_kind_keeps_its_own_history() -> None:
    clock = SimClock(ny(2017, 6, 29, 17, 0))
    store = _store(clock)
    store.activate("model", H0, reason="first model")
    store.activate("calibration", H1, reason="first calibration")
    store.activate("release", H2, reason="r1")
    friday = ny(2017, 6, 30, 9, 30)
    assert [store.current(k, friday).artifact for k in ("model", "calibration", "release")] == [
        H0,
        H1,
        H2,
    ]
    assert [p.artifact for p in store.pointers("calibration")] == [H1]


@RED
@pytest.mark.spec("ARCH§9.2")
@pytest.mark.parametrize("fault", ["kind", "artifact", "uppercase artifact", "reason"])
def test_a_bad_pointer_is_refused(fault: str) -> None:
    store = _store(SimClock(ny(2017, 6, 29, 17, 0)))
    kind, artifact, reason = "model", H0, "first model"
    if fault == "kind":
        kind = "playbook"
    elif fault == "artifact":
        artifact = "a0" * 31
    elif fault == "uppercase artifact":
        artifact = "A0" * 32
    else:
        reason = ""
    with pytest.raises(DataError):
        store.activate(kind, artifact, reason=reason)
    assert store.pointers("model") == ()


@RED
@pytest.mark.spec("C7", "ARCH§9.2", "SPEC§5")
def test_rollback_is_one_call_back_to_an_older_version() -> None:
    clock = SimClock(ny(2017, 6, 29, 17, 0))
    store = _store(clock)
    store.activate("model", H0, reason="first model")
    clock.advance_to(ny(2017, 6, 30, 17, 0))
    store.activate("model", H1, reason="refit")
    clock.advance_to(ny(2017, 7, 3, 17, 0))
    with pytest.raises(DataError):
        store.rollback("model", H2, reason="never pointed to")
    with pytest.raises(DataError):
        store.rollback("model", H1, reason="already the latest")
    assert len(store.pointers("model")) == 2
    pointer = store.rollback("model", H0, reason="bad refit")
    assert (pointer.kind, pointer.artifact, pointer.reason) == ("model", H0, "bad refit")
    assert pointer.effective_at == ny(2017, 7, 5, 9, 30)


@RED
@pytest.mark.spec("ARCH§9.1", "ARCH§9.2", "C5")
@pytest.mark.parametrize("table", ["pointers", "refit_events", "freeze_events"])
@pytest.mark.parametrize("verb", ["UPDATE", "DELETE"])
def test_ops_history_is_append_only(table: str, verb: str, tmp_path: Path) -> None:
    from quant.store.ops import FreezeCause, OpsStore, RefitOutcomeKind

    clock = SimClock(ny(2017, 6, 29, 17, 0))
    path = tmp_path / "ops.db"
    store = OpsStore.open(path, clock=clock)
    store.activate("model", H0, reason="first model")
    store.record_refit(
        at=clock.now(),
        cutoff=clock.now(),
        outcome=RefitOutcomeKind.FROZEN,
        artifact=H1,
        previous=H0,
        reasons=("transition 0->1 moved",),
    )
    store.record_freeze(FreezeCause.LIVE_DRIFT, "35-bar mean below the baseline")
    store.close()

    raw = sqlite3.connect(path)
    try:
        assert raw.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        update = f"UPDATE {table} SET seq = seq + 100"  # noqa: S608 (names from parametrize)
        delete = f"DELETE FROM {table}"  # noqa: S608 (names from parametrize)
        statement = update if verb == "UPDATE" else delete
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            raw.execute(statement)
    finally:
        raw.close()

    reopened = OpsStore.open(path, clock=clock)
    try:
        assert [p.artifact for p in reopened.pointers("model")] == [H0]
        assert len(reopened.refit_events()) == 1
        assert len(reopened.freeze_events()) == 2
    finally:
        reopened.close()


@RED
@pytest.mark.spec("ARCH§9.2", "K6")
def test_stamps_never_go_back_when_the_clock_does() -> None:
    from quant.store.ops import FreezeCause

    clock = ManualClock(ny(2017, 6, 29, 17, 0))
    store = _store(clock)
    first = store.activate("model", H0, reason="first model")
    clock.time = ny(2017, 6, 29, 16, 0)  # the clock steps back an hour
    event = store.record_freeze(FreezeCause.LIVE_DRIFT, "drift")
    assert event.recorded_at == first.recorded_at == ny(2017, 6, 29, 17, 0)
    assert event.effective_at == event.recorded_at


# --- Refit events and the freeze --------------------------------------------------------------


def _refit(
    store: Any, clock: Any, outcome: Any, artifact: str | None, reasons: tuple[str, ...]
) -> Any:
    return store.record_refit(
        at=clock.now(),
        cutoff=clock.now() - timedelta(minutes=5),
        outcome=outcome,
        artifact=artifact,
        previous=H0,
        reasons=reasons,
    )


@RED
@pytest.mark.spec("C5", "SPEC§5", "ARCH§7.3")
def test_a_failed_refit_freezes_entries_and_the_next_passing_one_unfreezes_them_at_its_open() -> (
    None
):
    from quant.store.ops import FreezeCause, RefitOutcomeKind, UnfreezeCause

    clock = SimClock(ny(2017, 5, 31, 17, 0))
    store = _store(clock)
    store.activate("model", H0, reason="first model")

    clock.advance_to(ny(2017, 6, 30, 17, 0))
    failed_at = clock.now()
    event = _refit(store, clock, RefitOutcomeKind.FROZEN, H1, ("transition 0->1 moved 0.2",))
    assert (event.outcome, event.artifact, event.previous) == (RefitOutcomeKind.FROZEN, H1, H0)
    assert event.effective_at is None
    assert event.reasons == ("transition 0->1 moved 0.2",)
    assert not store.frozen(failed_at - US)
    assert store.frozen(failed_at)  # frozen from the recording on
    assert store.frozen(ny(2017, 7, 3, 9, 30))
    # Stored but not activated: the current model keeps running.
    assert [p.artifact for p in store.pointers("model")] == [H0]
    assert store.current("model", ny(2017, 8, 1, 12, 0)).artifact == H0
    (freeze,) = store.freeze_events()
    assert (freeze.frozen, freeze.cause, freeze.effective_at) == (
        True,
        FreezeCause.REFIT_DRIFT.value,
        failed_at,
    )
    assert "transition 0->1 moved 0.2" in freeze.detail

    clock.advance_to(ny(2017, 7, 31, 17, 0))  # the next refit passes
    passed = _refit(store, clock, RefitOutcomeKind.ACTIVATED, H2, ())
    tuesday = ny(2017, 8, 1, 9, 30)
    assert passed.effective_at == tuesday
    assert store.current("model", tuesday).artifact == H2
    assert store.current("model", tuesday - US).artifact == H0
    assert store.frozen(tuesday - US)  # still frozen until the new model governs
    assert not store.frozen(tuesday)
    unfreeze = store.freeze_events()[-1]
    assert (unfreeze.frozen, unfreeze.cause, unfreeze.effective_at) == (
        False,
        UnfreezeCause.REFIT_PASSED.value,
        tuesday,
    )
    assert [e.outcome for e in store.refit_events()] == [
        RefitOutcomeKind.FROZEN,
        RefitOutcomeKind.ACTIVATED,
    ]


@RED
@pytest.mark.spec("C5", "SPEC§5")
def test_a_passing_refit_without_a_freeze_adds_no_unfreeze() -> None:
    from quant.store.ops import RefitOutcomeKind

    clock = SimClock(ny(2017, 5, 31, 17, 0))
    store = _store(clock)
    store.activate("model", H0, reason="first model")
    clock.advance_to(ny(2017, 6, 30, 17, 0))
    _refit(store, clock, RefitOutcomeKind.ACTIVATED, H1, ())
    assert store.freeze_events() == ()
    assert [p.artifact for p in store.pointers("model")] == [H0, H1]


@RED
@pytest.mark.spec("SPEC§5", "ARCH§7.3")
def test_a_rejected_refit_is_an_event_and_changes_nothing_else() -> None:
    from quant.store.ops import RefitOutcomeKind

    clock = SimClock(ny(2017, 5, 31, 17, 0))
    store = _store(clock)
    store.activate("model", H0, reason="first model")
    clock.advance_to(ny(2017, 6, 30, 17, 0))
    event = _refit(
        store, clock, RefitOutcomeKind.REJECTED, None, ("state 1: occupancy 0.0 < 0.02",)
    )
    assert (event.outcome, event.artifact, event.effective_at) == (
        RefitOutcomeKind.REJECTED,
        None,
        None,
    )
    assert [p.artifact for p in store.pointers("model")] == [H0]
    assert store.freeze_events() == ()
    assert not store.frozen(ny(2017, 7, 3, 12, 0))


@RED
@pytest.mark.spec("C5", "SPEC§5")
def test_a_manual_acknowledgement_unfreezes_at_once_and_is_logged() -> None:
    from quant.store.ops import FreezeCause, UnfreezeCause

    clock = SimClock(ny(2017, 6, 30, 11, 0))
    store = _store(clock)
    with pytest.raises(DataError):
        store.acknowledge_drift("nothing is frozen")
    store.record_freeze(FreezeCause.LIVE_DRIFT, "35-bar mean -4.2 below -3.9")
    assert store.frozen(clock.now())
    clock.advance_to(ny(2017, 6, 30, 14, 0))
    with pytest.raises(DataError):
        store.acknowledge_drift("")
    with capture_logs() as logs:
        event = store.acknowledge_drift("checked: a data outage, not a regime change")
    assert (event.frozen, event.cause) == (False, UnfreezeCause.MANUAL_ACK.value)
    assert event.effective_at == clock.now()
    assert not store.frozen(clock.now())
    assert store.frozen(clock.now() - US)
    (log,) = [entry for entry in logs if entry["event"] == "drift acknowledged"]
    assert log["note"] == "checked: a data outage, not a regime change"
    assert log["seq"] == event.seq


@RED
@pytest.mark.spec("C5")
def test_frozen_replays_the_events_in_order() -> None:
    from quant.store.ops import FreezeCause

    clock = SimClock(ny(2017, 6, 30, 11, 0))
    store = _store(clock)
    store.record_freeze(FreezeCause.LIVE_DRIFT, "first")
    clock.advance(timedelta(hours=1))
    store.acknowledge_drift("ok")
    clock.advance(timedelta(hours=1))
    store.record_freeze(FreezeCause.LIVE_DRIFT, "again")
    times = [
        ny(2017, 6, 30, 10, 59),
        ny(2017, 6, 30, 11, 0),
        ny(2017, 6, 30, 12, 0),
        ny(2017, 6, 30, 13, 0),
    ]
    assert [store.frozen(t) for t in times] == [False, True, False, True]
    with pytest.raises(ClockError):
        store.frozen(datetime(2017, 6, 30, 12))  # noqa: DTZ001 (naive on purpose)


BAD_REFITS = [
    "cutoff after at",
    "bad previous",
    "bad artifact",
    "frozen without artifact",
    "activated without artifact",
    "rejected with artifact",
    "frozen without reasons",
    "rejected without reasons",
    "activated with reasons",
    "empty reason",
]


def _bad_refits() -> dict[str, dict[str, Any]]:
    from quant.store.ops import RefitOutcomeKind

    good = {
        "outcome": RefitOutcomeKind.FROZEN,
        "artifact": H1,
        "previous": H0,
        "reasons": ("drift",),
    }
    return {
        "cutoff after at": good | {"cutoff_after": True},
        "bad previous": good | {"previous": "x"},
        "bad artifact": good | {"artifact": "A1" * 32},
        "frozen without artifact": good | {"artifact": None},
        "activated without artifact": good
        | {"outcome": RefitOutcomeKind.ACTIVATED, "artifact": None, "reasons": ()},
        "rejected with artifact": good | {"outcome": RefitOutcomeKind.REJECTED},
        "frozen without reasons": good | {"reasons": ()},
        "rejected without reasons": good
        | {"outcome": RefitOutcomeKind.REJECTED, "artifact": None, "reasons": ()},
        "activated with reasons": good | {"outcome": RefitOutcomeKind.ACTIVATED},
        "empty reason": good | {"reasons": ("",)},
    }


@RED
@pytest.mark.spec("C5", "ARCH§7.3")
@pytest.mark.parametrize("fault", BAD_REFITS)
def test_a_bad_refit_record_is_refused_and_writes_nothing(fault: str) -> None:
    clock = SimClock(ny(2017, 6, 30, 17, 0))
    store = _store(clock)
    store.activate("model", H0, reason="first model")
    args = dict(_bad_refits()[fault])
    late = args.pop("cutoff_after", False)
    with pytest.raises(DataError):
        store.record_refit(
            at=clock.now(),
            cutoff=clock.now() + (US if late else -US),
            **args,
        )
    assert store.refit_events() == ()
    assert store.freeze_events() == ()
    assert [p.artifact for p in store.pointers("model")] == [H0]
