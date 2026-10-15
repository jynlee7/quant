"""ops.db: the `current` pointers, refit outcomes and the entry freeze (ARCHITECTURE §9.1, §9.2;
spec §5; C5, C7). Built in B7.

ops.db is SQLite in WAL mode with `synchronous=FULL` (`quant.store.sqlite.connect`), so a
recorded pointer or freeze survives a power cut. This module holds its first migrations; later
steps (decision records, orders, fills, health) append theirs. It stores artifact hashes and
times only, never models: it imports nothing outside `quant.core` and `quant.store`, so the
decision path and the risk engine may read it (K1, K3).

**Every table is append-only.** SQLite triggers refuse UPDATE and DELETE on each of them
(`RAISE(ABORT, ...)`, which sqlite3 reports as an IntegrityError), so history can only grow.
Each row carries `seq`, its append order (INTEGER PRIMARY KEY, 1, 2, 3, ...), and
`recorded_at`, stamped from the store's Clock inside the write transaction. Stamps never go
back: a stamp is max(clock.now(), the last stamp written, the largest `recorded_at` in any
table when the store was opened), so a clock that steps back can't reorder history (as
market.db's stamps, B2). Times are stored as epoch microseconds UTC
(`quant.core.epoch.to_us`) and returned as aware UTC datetimes.

**Pointers** (ARCHITECTURE §9.2): one history per kind ∈ POINTER_KINDS. A row names the
artifact (its SHA-256, 64 lowercase hex characters), `effective_at`, `recorded_at` and a
`reason`. Rows are only ever appended by `activate` and `rollback`, and each takes effect at the
next session open after it is recorded, `calendar.next_open(recorded_at)` (C7): never
mid-session, and never before it was recorded. So within a kind, `effective_at` never decreases
with `seq`. `current(kind, at)` is the row with the highest `seq` among those with
`effective_at <= at`: the open itself is the first instant a new pointer governs.

**Refit outcomes** (spec §5, ARCHITECTURE §7.3): `record_refit` writes one RefitEvent and, in
the same transaction, its consequences:

- ACTIVATED: a model pointer to the new artifact, effective at the next open after the
  recording, and, when entries would still be frozen at that open, an unfreeze event
  (UnfreezeCause.REFIT_PASSED) effective at the same open: "the next refit passes every check"
  ends a freeze exactly when the new model starts governing;
- FROZEN: the artifact was stored but not activated (C5): a freeze event (FreezeCause.
  REFIT_DRIFT) effective at the recording, its detail the drift reasons joined by "; ";
- REJECTED: the event alone (a sanity reject keeps the current model; the job logs the alert).

**The freeze** (C5) is an event log. A freeze event has `frozen=True`, an unfreeze
`frozen=False`; each has a cause, a detail and `effective_at`. `frozen(at)` is a pure query:
replay, in `seq` order, the events with `effective_at <= at`; the last one decides, and with
none entries aren't frozen. A freeze stops new entries only; open positions keep their exits and
stops (spec §5), which is the switching and risk layers' business, not this store's.

Unfreezing: a passing refit (above), or `acknowledge_drift`, the logged manual path (spec §5;
the backtest simulates only the first). The live drift monitor's freeze is
`record_freeze(FreezeCause.LIVE_DRIFT, ...)`, written by the engine (B13).

Every refusal raises DataError and writes nothing. "Now" comes only from the Clock given to the
store (K6).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Final, Literal

from quant.core.calendar import TradingCalendar
from quant.core.clock import Clock
from quant.store.sqlite import Migration

PointerKind = Literal["model", "calibration", "release"]
POINTER_KINDS: Final[tuple[PointerKind, ...]] = ("model", "calibration", "release")
# The schema name `migrate` reports.
DATABASE: Final = "ops.db"

# Version 1: the pointer history, the refit events and the freeze events, each append-only.
MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        1,
        (
            """CREATE TABLE pointers (
                seq INTEGER PRIMARY KEY,
                kind TEXT NOT NULL CHECK (kind IN ('model', 'calibration', 'release')),
                artifact TEXT NOT NULL,
                effective_at INTEGER NOT NULL,
                recorded_at INTEGER NOT NULL,
                reason TEXT NOT NULL,
                CHECK (effective_at > recorded_at)
            )""",
            "CREATE INDEX pointers_by_kind ON pointers (kind, effective_at, seq)",
            """CREATE TABLE refit_events (
                seq INTEGER PRIMARY KEY,
                at INTEGER NOT NULL,
                cutoff INTEGER NOT NULL,
                outcome TEXT NOT NULL CHECK (outcome IN ('activated', 'frozen', 'rejected')),
                artifact TEXT,
                previous TEXT NOT NULL,
                reasons TEXT NOT NULL,
                effective_at INTEGER,
                recorded_at INTEGER NOT NULL
            )""",
            """CREATE TABLE freeze_events (
                seq INTEGER PRIMARY KEY,
                frozen INTEGER NOT NULL CHECK (frozen IN (0, 1)),
                cause TEXT NOT NULL,
                detail TEXT NOT NULL,
                effective_at INTEGER NOT NULL,
                recorded_at INTEGER NOT NULL
            )""",
            *(
                f"""CREATE TRIGGER {table}_no_{verb} BEFORE {verb.upper()} ON {table}
                BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END"""
                for table in ("pointers", "refit_events", "freeze_events")
                for verb in ("update", "delete")
            ),
        ),
    ),
)


class RefitOutcomeKind(StrEnum):
    """How a refit ended (spec §5, ARCHITECTURE §7.3)."""

    ACTIVATED = "activated"  # passed every check: the pointer moves at the next open
    FROZEN = "frozen"  # failed a drift check: stored, not activated, entries frozen (C5)
    REJECTED = "rejected"  # failed the sanity checks: not stored; the current model stays


class FreezeCause(StrEnum):
    """Why entries froze: a refit's drift check, or the live monitor (spec §5)."""

    REFIT_DRIFT = "refit_drift"
    LIVE_DRIFT = "live_drift"


class UnfreezeCause(StrEnum):
    """Why entries unfroze: the next refit passed every check, or you acknowledged the drift."""

    REFIT_PASSED = "refit_passed"
    MANUAL_ACK = "manual_ack"


@dataclass(frozen=True)
class Pointer:
    """One row of a pointer history (module docstring). Times are aware UTC."""

    seq: int
    kind: PointerKind
    artifact: str
    effective_at: datetime
    recorded_at: datetime
    reason: str


@dataclass(frozen=True)
class FreezeEvent:
    """One freeze (`frozen=True`) or unfreeze (`frozen=False`) event. `cause` is a FreezeCause
    value for a freeze and an UnfreezeCause value for an unfreeze."""

    seq: int
    frozen: bool
    cause: str
    detail: str
    effective_at: datetime
    recorded_at: datetime


@dataclass(frozen=True)
class RefitEvent:
    """One refit's outcome (module docstring).

    - `at`: the job's instant; `cutoff`: its data cutoff (`quant.jobs.refit.refit_cutoff`);
    - `artifact`: the new model's hash, None when REJECTED (nothing was stored);
    - `previous`: the hash of the model that was current when the job ran;
    - `reasons`: the drift findings (FROZEN) or the sanity reasons (REJECTED); () when
      ACTIVATED;
    - `effective_at`: the next open after the recording, when ACTIVATED; None otherwise.
    """

    seq: int
    at: datetime
    cutoff: datetime
    outcome: RefitOutcomeKind
    artifact: str | None
    previous: str
    reasons: tuple[str, ...]
    effective_at: datetime | None
    recorded_at: datetime


class OpsStore:
    """ops.db's pointers, refit events and freeze events (module docstring)."""

    def __init__(self, conn: sqlite3.Connection, *, clock: Clock) -> None:
        """Wrap an open connection and apply MIGRATIONS (`quant.store.sqlite.migrate`, named
        DATABASE). Every stamp comes from `clock`."""
        raise NotImplementedError("B7a: OpsStore")

    @classmethod
    def open(cls, path: Path, *, clock: Clock) -> OpsStore:
        """The store at `path`, created if missing: `connect(path, synchronous="FULL")`."""
        raise NotImplementedError("B7a: OpsStore.open")

    @classmethod
    def memory(cls, *, clock: Clock) -> OpsStore:
        """An in-memory store, for tests and backtests."""
        raise NotImplementedError("B7a: OpsStore.memory")

    def close(self) -> None:
        raise NotImplementedError("B7a: OpsStore.close")

    # --- pointers -------------------------------------------------------------------------

    def activate(
        self,
        kind: PointerKind,
        artifact: str,
        *,
        reason: str,
        calendar: TradingCalendar | None = None,
    ) -> Pointer:
        """Append a pointer of `kind` to `artifact`, effective at `calendar.next_open(stamp)`
        (XNYS when None), the stamp being this write's `recorded_at`. Raises DataError for a
        kind outside POINTER_KINDS, an artifact that isn't 64 lowercase hex characters, or an
        empty reason."""
        raise NotImplementedError("B7a: OpsStore.activate")

    def rollback(
        self,
        kind: PointerKind,
        artifact: str,
        *,
        reason: str,
        calendar: TradingCalendar | None = None,
    ) -> Pointer:
        """One call: append a pointer of `kind` back to `artifact`, an older version, effective
        at the next open after the recording, as `activate` (C7). Raises DataError, as
        `activate` does, and when `artifact` never appeared in `kind`'s history or is the
        artifact of its latest row (there is nothing to roll back to)."""
        raise NotImplementedError("B7a: OpsStore.rollback")

    def pointers(self, kind: PointerKind) -> tuple[Pointer, ...]:
        """`kind`'s whole history in `seq` order. DataError for an unknown kind."""
        raise NotImplementedError("B7a: OpsStore.pointers")

    def current(self, kind: PointerKind, at: datetime) -> Pointer | None:
        """The pointer of `kind` governing at `at`: the highest-`seq` row with `effective_at <=
        at`, None before the first takes effect. ClockError for a naive `at`, DataError for an
        unknown kind."""
        raise NotImplementedError("B7a: OpsStore.current")

    # --- refits and the freeze ------------------------------------------------------------

    def record_refit(
        self,
        *,
        at: datetime,
        cutoff: datetime,
        outcome: RefitOutcomeKind,
        artifact: str | None,
        previous: str,
        reasons: tuple[str, ...],
        calendar: TradingCalendar | None = None,
    ) -> RefitEvent:
        """Record a refit's outcome and its consequences in one transaction (module docstring).
        The model pointer of ACTIVATED is effective at `calendar.next_open(stamp)` (XNYS when
        None), as `activate`'s, and its reason names the refit ("refit at <at, ISO UTC>").

        Raises DataError, writing nothing, when: `cutoff > at`; `previous` or a given
        `artifact` isn't 64 lowercase hex characters; `artifact` is None for ACTIVATED or
        FROZEN, or given for REJECTED; `reasons` is empty for FROZEN or REJECTED, or not empty
        for ACTIVATED, or holds an empty string. ClockError for a naive `at` or `cutoff`."""
        raise NotImplementedError("B7a: OpsStore.record_refit")

    def refit_events(self) -> tuple[RefitEvent, ...]:
        """Every refit event, in `seq` order."""
        raise NotImplementedError("B7a: OpsStore.refit_events")

    def record_freeze(self, cause: FreezeCause, detail: str) -> FreezeEvent:
        """Freeze entries from this write's stamp on (`effective_at` = `recorded_at`). DataError
        for an empty detail."""
        raise NotImplementedError("B7a: OpsStore.record_freeze")

    def acknowledge_drift(self, note: str) -> FreezeEvent:
        """The manual unfreeze (spec §5): an unfreeze event, UnfreezeCause.MANUAL_ACK, detail
        `note`, effective at this write's stamp, and a structlog warning "drift acknowledged"
        with `note` and `seq`. DataError, writing nothing, for an empty note or when entries
        aren't frozen at the stamp (there is nothing to acknowledge)."""
        raise NotImplementedError("B7a: OpsStore.acknowledge_drift")

    def freeze_events(self) -> tuple[FreezeEvent, ...]:
        """Every freeze and unfreeze event, in `seq` order."""
        raise NotImplementedError("B7a: OpsStore.freeze_events")

    def frozen(self, at: datetime) -> bool:
        """Whether new entries are frozen at `at` (module docstring: the last event, in `seq`
        order, among those with `effective_at <= at`). ClockError for a naive `at`."""
        raise NotImplementedError("B7a: OpsStore.frozen")
