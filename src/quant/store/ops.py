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
  recording; with a calibration (B7b, ARCHITECTURE §7.3 step 5), a calibration pointer to it,
  effective at the same open, with the same reason; and, when entries would still be frozen at
  that open, an unfreeze event
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

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Final, Literal, cast

import structlog

from quant.core.calendar import TradingCalendar
from quant.core.canonical import canonical_dumps, canonical_loads
from quant.core.clock import Clock, ensure_utc
from quant.core.epoch import from_us, to_us
from quant.core.errors import DataError
from quant.store.sqlite import Migration, connect, migrate, transaction

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
                f"""CREATE TRIGGER {table}_no_replace BEFORE INSERT ON {table}
                WHEN EXISTS (SELECT 1 FROM {table} WHERE seq = NEW.seq)
                BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END"""  # noqa: S608 -- fixed schema names
                for table in ("pointers", "refit_events", "freeze_events")
            ),
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
        self._conn = conn
        self._clock = clock
        migrate(conn, MIGRATIONS, name=DATABASE)

    @classmethod
    def open(cls, path: Path, *, clock: Clock) -> OpsStore:
        """The store at `path`, created if missing: `connect(path, synchronous="FULL")`."""
        return cls(connect(path, synchronous="FULL"), clock=clock)

    @classmethod
    def memory(cls, *, clock: Clock) -> OpsStore:
        """An in-memory store, for tests and backtests."""
        return cls(connect(":memory:", synchronous="FULL"), clock=clock)

    def close(self) -> None:
        self._conn.close()

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
        self._validate_pointer(kind, artifact, reason)
        with transaction(self._conn):
            stamp = self._stamp()
            return self._pointer(kind, artifact, reason, stamp, calendar)

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
        self._validate_pointer(kind, artifact, reason)
        with transaction(self._conn):
            history = self.pointers(kind)
            if (
                not history
                or history[-1].artifact == artifact
                or not any(p.artifact == artifact for p in history)
            ):
                raise DataError("rollback requires an older artifact in this pointer history")
            return self._pointer(kind, artifact, reason, self._stamp(), calendar)

    def pointers(self, kind: PointerKind) -> tuple[Pointer, ...]:
        """`kind`'s whole history in `seq` order. DataError for an unknown kind."""
        self._kind(kind)
        return tuple(
            Pointer(r[0], r[1], r[2], from_us(r[3]), from_us(r[4]), r[5])
            for r in self._conn.execute(
                "SELECT * FROM pointers WHERE kind = ? ORDER BY seq", (kind,)
            )
        )

    def current(self, kind: PointerKind, at: datetime) -> Pointer | None:
        """The pointer of `kind` governing at `at`: the highest-`seq` row with `effective_at <=
        at`, None before the first takes effect. ClockError for a naive `at`, DataError for an
        unknown kind."""
        self._kind(kind)
        instant = to_us(at)
        row = self._conn.execute(
            "SELECT * FROM pointers WHERE kind = ? AND effective_at <= ? ORDER BY seq DESC LIMIT 1",
            (kind, instant),
        ).fetchone()
        return (
            None
            if row is None
            else Pointer(row[0], row[1], row[2], from_us(row[3]), from_us(row[4]), row[5])
        )

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
        calibration: str | None = None,
    ) -> RefitEvent:
        """Record a refit's outcome and its consequences in one transaction (module docstring).
        The model pointer of ACTIVATED is effective at `calendar.next_open(stamp)` (XNYS when
        None), as `activate`'s, and its reason names the refit ("refit at <at, ISO UTC>").
        `calibration` (B7b), the hash of the calibration the pass built, appends a calibration
        pointer in the same transaction, after the model pointer, with the same effective_at,
        recorded_at and reason. The RefitEvent itself doesn't name it: the pointer row does.
        DataError, writing nothing, when `calibration` is given for FROZEN or REJECTED, or isn't
        64 lowercase hex characters.

        Raises DataError, writing nothing, when: `cutoff > at`; `previous` or a given
        `artifact` isn't 64 lowercase hex characters; `artifact` is None for ACTIVATED or
        FROZEN, or given for REJECTED; `reasons` is empty for FROZEN or REJECTED, or not empty
        for ACTIVATED, or holds an empty string. ClockError for a naive `at` or `cutoff`."""
        if calibration is not None:
            raise NotImplementedError("B7b: record_refit appends the pass's calibration pointer")
        at, cutoff = ensure_utc(at), ensure_utc(cutoff)
        self._hash(previous)
        if artifact is not None:
            self._hash(artifact)
        if (
            not isinstance(outcome, RefitOutcomeKind)
            or cutoff > at
            or (artifact is None) != (outcome is RefitOutcomeKind.REJECTED)
            or (not reasons) != (outcome is RefitOutcomeKind.ACTIVATED)
            or any(not isinstance(r, str) or not r.strip() for r in reasons)
        ):
            raise DataError("invalid refit outcome, cutoff, artifact or reasons")
        with transaction(self._conn):
            stamp = self._stamp()
            effective = None
            detail = f"refit at {at.isoformat()}"
            if outcome is RefitOutcomeKind.ACTIVATED:
                pointer = self._pointer("model", cast(str, artifact), detail, stamp, calendar)
                effective = pointer.effective_at
                if self.frozen(effective):
                    self._freeze(False, UnfreezeCause.REFIT_PASSED.value, detail, effective, stamp)
            elif outcome is RefitOutcomeKind.FROZEN:
                self._freeze(True, FreezeCause.REFIT_DRIFT.value, "; ".join(reasons), stamp, stamp)
            cursor = self._conn.execute(
                "INSERT INTO refit_events (at, cutoff, outcome, artifact, previous, reasons, effective_at, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    to_us(at),
                    to_us(cutoff),
                    outcome.value,
                    artifact,
                    previous,
                    canonical_dumps(reasons),
                    None if effective is None else to_us(effective),
                    to_us(stamp),
                ),
            )
            return RefitEvent(
                cast(int, cursor.lastrowid),
                at,
                cutoff,
                outcome,
                artifact,
                previous,
                reasons,
                effective,
                stamp,
            )

    def refit_events(self) -> tuple[RefitEvent, ...]:
        """Every refit event, in `seq` order."""
        return tuple(
            RefitEvent(
                r[0],
                from_us(r[1]),
                from_us(r[2]),
                RefitOutcomeKind(r[3]),
                r[4],
                r[5],
                tuple(canonical_loads(r[6])),
                None if r[7] is None else from_us(r[7]),
                from_us(r[8]),
            )
            for r in self._conn.execute("SELECT * FROM refit_events ORDER BY seq")
        )

    def record_freeze(self, cause: FreezeCause, detail: str) -> FreezeEvent:
        """Freeze entries from this write's stamp on (`effective_at` = `recorded_at`). DataError
        for an empty detail."""
        if not isinstance(cause, FreezeCause) or not isinstance(detail, str) or not detail.strip():
            raise DataError("freeze requires a cause and detail")
        with transaction(self._conn):
            stamp = self._stamp()
            return self._freeze(True, cause.value, detail, stamp, stamp)

    def acknowledge_drift(self, note: str) -> FreezeEvent:
        """The manual unfreeze (spec §5): an unfreeze event, UnfreezeCause.MANUAL_ACK, detail
        `note`, effective at this write's stamp, and a structlog warning "drift acknowledged"
        with `note` and `seq`. DataError, writing nothing, for an empty note or when entries
        aren't frozen at the stamp (there is nothing to acknowledge)."""
        if not isinstance(note, str) or not note.strip():
            raise DataError("acknowledgement requires a note")
        with transaction(self._conn):
            stamp = self._stamp()
            if not self.frozen(stamp):
                raise DataError("entries are not frozen")
            event = self._freeze(False, UnfreezeCause.MANUAL_ACK.value, note, stamp, stamp)
        structlog.get_logger().warning("drift acknowledged", note=note, seq=event.seq)
        return event

    def freeze_events(self) -> tuple[FreezeEvent, ...]:
        """Every freeze and unfreeze event, in `seq` order."""
        return tuple(
            FreezeEvent(r[0], bool(r[1]), r[2], r[3], from_us(r[4]), from_us(r[5]))
            for r in self._conn.execute("SELECT * FROM freeze_events ORDER BY seq")
        )

    def frozen(self, at: datetime) -> bool:
        """Whether new entries are frozen at `at` (module docstring: the last event, in `seq`
        order, among those with `effective_at <= at`). ClockError for a naive `at`."""
        row = self._conn.execute(
            "SELECT frozen FROM freeze_events WHERE effective_at <= ? ORDER BY seq DESC LIMIT 1",
            (to_us(at),),
        ).fetchone()
        return bool(row[0]) if row is not None else False

    @staticmethod
    def _kind(kind: PointerKind) -> None:
        if kind not in POINTER_KINDS:
            raise DataError("unknown pointer kind")

    @staticmethod
    def _hash(artifact: str) -> None:
        if not isinstance(artifact, str) or re.fullmatch(r"[0-9a-f]{64}", artifact) is None:
            raise DataError("artifact must be a lowercase SHA-256")

    def _validate_pointer(self, kind: PointerKind, artifact: str, reason: str) -> None:
        self._kind(kind)
        self._hash(artifact)
        if not isinstance(reason, str) or not reason.strip():
            raise DataError("pointer requires a reason")

    def _stamp(self) -> datetime:
        # Called under BEGIN IMMEDIATE: also sees stamps written by other connections.
        row = self._conn.execute(
            "SELECT MAX(recorded_at) FROM (SELECT recorded_at FROM pointers UNION ALL SELECT recorded_at FROM refit_events UNION ALL SELECT recorded_at FROM freeze_events)"
        ).fetchone()
        now = to_us(self._clock.now())
        return from_us(max(now, row[0]) if row[0] is not None else now)

    def _pointer(
        self,
        kind: PointerKind,
        artifact: str,
        reason: str,
        stamp: datetime,
        calendar: TradingCalendar | None,
    ) -> Pointer:
        effective = (calendar or TradingCalendar.xnys()).next_open(stamp)
        cursor = self._conn.execute(
            "INSERT INTO pointers (kind, artifact, effective_at, recorded_at, reason) VALUES (?, ?, ?, ?, ?)",
            (kind, artifact, to_us(effective), to_us(stamp), reason),
        )
        return Pointer(cast(int, cursor.lastrowid), kind, artifact, effective, stamp, reason)

    def _freeze(
        self, frozen: bool, cause: str, detail: str, effective: datetime, stamp: datetime
    ) -> FreezeEvent:
        cursor = self._conn.execute(
            "INSERT INTO freeze_events (frozen, cause, detail, effective_at, recorded_at) VALUES (?, ?, ?, ?, ?)",
            (int(frozen), cause, detail, to_us(effective), to_us(stamp)),
        )
        return FreezeEvent(cast(int, cursor.lastrowid), frozen, cause, detail, effective, stamp)
