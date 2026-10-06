"""research.db's trial ledger, the only place a TrialTicket is built (A2, C11, K5).

ARCHITECTURE §4.4 and §10. Every scored run of the development period (DEV) is appended here
before it may read DEV, and the ledger counts the trials that G6's Deflated Sharpe deflates by.
The ledger is append-only, and each entry hashes the one before it, so `quant ledger verify`
finds a row that was edited, deleted, reordered or inserted.

**Trials (C11).** A trial is one set of decisions: two scored DEV runs are the same trial exactly
when their decision records are the same. A run's records exist only after it has read DEV, and
it may read DEV only with a ticket, which the ledger issues only after it has appended the run
(A2). Kelly's edge estimates read DEV through the same ticket (calibrate, C12), so nothing about
the records is known before the ticket exists. A run is therefore placed twice:

1. When it opens (`open_run`), by what it is: its configuration hash, its code hash and the days
   it scores. A run whose three match an earlier run's takes that run's trial as it stands,
   since the same configuration on the same code makes the same decisions. Any other run opens a
   new trial, numbered next. The `run` entry is committed, and only then is the ticket issued.
2. When it settles (`settle`), by what it did: the hash of its decision records.
   - Records equal to an earlier settled run's: the run joins that run's trial, whatever its code
     or configuration, because its decisions differ from no earlier scored run. A trial opened
     for it alone is left with no runs and stops counting (a code change that moved nothing).
   - Records new to the ledger: the run keeps the trial it opened with, unless that trial already
     has settled records, which must then differ from these. The same configuration and code
     made different decisions (changed data, or a determinism bug), so the run is a new trial.

The trial count (`trial_count`) is the number of trials holding at least one run, each run placed
by its records once it has settled and by what it is until then. A run that never settles (it
crashed or was stopped) was still allowed to read DEV, so it counts until its records show that it
repeats an earlier run; a rerun of a settled configuration adds nothing even before it settles.
Trial numbers are never reused, so a trial left empty leaves a gap. G6 uses the count after the
candidate's own run has settled.

Yardsticks (baselines and cost-sensitivity runs) are not trials (C11). They score under the ticket
of the run they measure and never settle, so they add nothing to the count.

**Tickets (K5, ARCHITECTURE §4.4).** A TrialTicket is leave for one run to score and calibrate DEV
between its first and last day. Only `open_run` builds one, with this module's private token. A
ticket is good only with the ledger object that issued it: `verify_ticket` refuses anything but a
TrialTicket itself (plain objects, lookalikes, subclasses), one built without the token, one this
ledger never issued (copies, tickets built with a stolen token, and tickets from any other ledger
object, even one open on the same file), and one changed since it was issued. DataPolicy asks the
ledger it is bound to, so a ticket is no use anywhere else. A ticket stays good after its run
settles, for the yardsticks. No lockbox ticket exists in v1 (P9): `open_run` refuses any span that
reaches the lockbox, and the go-live test (B19) is the only thing that will ever add one.

**Refusals (A2, ARCHITECTURE §6.3).** A read the bound DataPolicy refuses, and a ticket request
`open_run` refuses for its span, leaves a `refusal` entry: what was asked for, the days, the error
and what ticket was shown. A refused read saw nothing, so it is no trial; the attempt stays on the
record.

**Gate configuration (ARCHITECTURE §10).** `record_gate_config` appends the gate configuration
(gates.yaml as loaded) whenever its canonical hash differs from the last one recorded.

**The chain (ARCHITECTURE §10).** Entries are numbered 1, 2, 3, ... in the `entries` table. Each
row holds its kind, when it was appended (the ledger's clock, epoch microseconds UTC), its body as
canonical JSON, the hash of the entry before it (`prev`; GENESIS for the first) and its own hash,
`entry_hash(seq, kind, at, body, prev)`. The single `head` row names the last entry and its hash
and is written in the same transaction as every append, so a row deleted from the end, or one
added behind the ledger's back, is caught as well. Appends run under the write lock, so two
ledger objects on one file chain correctly.

| Kind | Body (canonical JSON; days are ISO dates) |
|---|---|
| run | run ("R5"), trial ("T3"), label, config_hash, code_hash, first_day, last_day |
| result | run, trial (where its records place it), records_hash, repeat_of (the earliest earlier run with the same records, or null) |
| refusal | purpose, first_day, last_day, error, reason, ticket (the `Refusal` fields) |
| gate_config | config, config_hash, previous (the last recorded config_hash, or null), reason |

research.db will also hold proposals, gate runs, walk-forward results and the fit-cache index
(ARCHITECTURE §9.1). Their tables come in later migrations, appended to MIGRATIONS here.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, final

from quant.core.clock import Clock
from quant.data.policy import Refusal, Segments
from quant.store.sqlite import Migration

GENESIS: Final = "0" * 64
"""The `prev` of the first entry."""

_TOKEN: Final = object()
"""The private token a TrialTicket's constructor demands. It never leaves this module."""

MIGRATIONS: tuple[Migration, ...] = (
    Migration(
        1,
        (
            """CREATE TABLE entries (
                seq INTEGER PRIMARY KEY CHECK (seq >= 1),
                kind TEXT NOT NULL,
                at INTEGER NOT NULL,
                body TEXT NOT NULL,
                prev TEXT NOT NULL,
                hash TEXT NOT NULL
            )""",
            # The last entry, written with every append: a truncated or extended chain shows.
            """CREATE TABLE head (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                seq INTEGER NOT NULL,
                hash TEXT NOT NULL
            )""",
        ),
    ),
)


class EntryKind(StrEnum):
    RUN = "run"  # a scored DEV run opened, and the trial it opened with
    RESULT = "result"  # a run settled: its decision records, and the trial they place it in
    REFUSAL = "refusal"  # a refused read or ticket request
    GATE_CONFIG = "gate_config"  # the gate configuration changed


def entry_hash(
    seq: int, kind: EntryKind | str, at: datetime, body: Mapping[str, Any], prev: str
) -> str:
    """An entry's hash: `sha256_hex` (quant.core.canonical) of the mapping
    {"at": at, "body": body, "kind": str(kind), "prev": prev, "seq": seq}.

    `at` is the timezone-aware time the entry was appended, so canonical JSON writes it in UTC
    with a Z. The hash covers every column of the row but the hash itself."""
    raise NotImplementedError


@dataclass(frozen=True)
class RunSpec:
    """What a scored DEV run is, declared before it reads anything (C11).

    - `config_hash`: SHA-256 hex over everything that configures the run: the release (features,
      HMM config, playbooks, strategy.md), engine.yaml, risk.yaml and costs.yaml (B11 builds it);
    - `code_hash`: SHA-256 hex naming the decision-path code and the lockfile the run executes;
    - `first_day`, `last_day`: the DEV days it will score, which its ticket covers;
    - `label`: for people, such as "r1" or "proposal 2026-11-03 #2".
    """

    config_hash: str
    code_hash: str
    first_day: date
    last_day: date
    label: str


@final
class TrialTicket:
    """Leave for one run of one trial to score and calibrate DEV from `first_day` to `last_day`
    (A2, C11, K5).

    Only `TrialLedger.open_run` builds one, after the run's entry is committed. The constructor
    raises TicketError unless `token` is this module's private token. A ticket is immutable:
    setting or deleting an attribute raises AttributeError, and `copy.copy` and `copy.deepcopy`
    raise TypeError (so does pickling, which the project never does). Equality and hashing are
    by identity. The repr shows the trial, the run and the days, never the token.

    - `trial_id`: the trial the run opened with ("T3");
    - `run_id`: the run ("R5"), the n-th run this ledger file has opened;
    - `first_day`, `last_day`, `config_hash`, `code_hash`: from the run's RunSpec;
    - `issued_at`: the ledger's clock when it issued the ticket.
    """

    __slots__ = (
        "_token",
        "code_hash",
        "config_hash",
        "first_day",
        "issued_at",
        "last_day",
        "run_id",
        "trial_id",
    )

    trial_id: str
    run_id: str
    first_day: date
    last_day: date
    config_hash: str
    code_hash: str
    issued_at: datetime
    _token: object

    def __init__(
        self,
        *,
        token: object,
        trial_id: str,
        run_id: str,
        first_day: date,
        last_day: date,
        config_hash: str,
        code_hash: str,
        issued_at: datetime,
    ) -> None:
        raise NotImplementedError

    def __setattr__(self, name: str, value: object) -> None:
        raise NotImplementedError

    def __delattr__(self, name: str) -> None:
        raise NotImplementedError

    def __copy__(self) -> TrialTicket:
        raise NotImplementedError

    def __deepcopy__(self, memo: dict[int, object]) -> TrialTicket:
        raise NotImplementedError

    def __reduce_ex__(self, protocol: object) -> tuple[object, ...]:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class Settlement:
    """Where a run's decision records place it (C11).

    `trial_id` is the run's trial now that its records are known; it differs from the ticket's
    when the run repeats an earlier run of another trial, or when its configuration and code made
    new decisions. `repeat_of` is the earliest earlier settled run with the same records, or None
    when they are new to the ledger.
    """

    run_id: str
    trial_id: str
    records_hash: str
    repeat_of: str | None


@dataclass(frozen=True)
class LedgerEntry:
    """One row of the chain, its body parsed from canonical JSON."""

    seq: int
    kind: EntryKind
    at: datetime
    body: dict[str, Any]
    prev: str
    hash: str


@dataclass(frozen=True)
class ChainReport:
    """What `verify` found: the number of entries, the head's hash (None when empty), and one
    line per problem, each naming the entry it concerns."""

    entries: int
    head: str | None
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems


class TrialLedger:
    """The trial ledger in research.db (A2, C11). Its clock stamps entries and tickets; its
    segments say where DEV and the lockbox are."""

    def __init__(
        self, conn: sqlite3.Connection, *, clock: Clock, segments: Segments | None = None
    ) -> None:
        """Migrate `conn` to MIGRATIONS (quant.store.sqlite.migrate, name "research.db") and use
        it. `segments` defaults to `Segments()`. The ledger starts with no tickets issued."""
        raise NotImplementedError

    @classmethod
    def open(cls, path: Path, *, clock: Clock, segments: Segments | None = None) -> TrialLedger:
        """The ledger in the file at `path` (WAL mode, `quant.store.sqlite.connect`), created and
        migrated if needed."""
        raise NotImplementedError

    @classmethod
    def memory(cls, *, clock: Clock, segments: Segments | None = None) -> TrialLedger:
        """A ledger in an in-memory database, for tests."""
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    # --- runs and trials (C11) ----------------------------------------------------------------

    def open_run(self, spec: RunSpec) -> TrialTicket:
        """Append a `run` entry for `spec`, then issue its ticket (A2, C11).

        Checks first, appending nothing for a bad request:
        - `config_hash` and `code_hash` must be 64 lowercase hex characters: LedgerError;
        - the span must lie inside DEV. One that reaches the lockbox raises LockboxError, since
          no lockbox ticket exists in v1 (P9); any other span outside DEV, or one that ends
          before it starts, raises PolicyError. Each of these refusals is recorded first as a
          `refusal` entry, `Refusal(purpose="ticket", ...)` with ticket None.

        Under one write transaction it then reads the chain, gives the run the next run id
        (R1, R2, ... over this file's `run` entries) and a trial as the module docstring says
        (the trial, as it stands, of the latest earlier run with the same config_hash, code_hash,
        first_day and last_day; otherwise the next trial number, T1, T2, ..., never reused), and
        appends the entry. Only after the commit does it build the ticket and keep it as issued.
        If the append fails, the error propagates and no ticket exists.
        """
        raise NotImplementedError

    def settle(self, ticket: TrialTicket, records_hash: str) -> Settlement:
        """Record the run's decision records and place the run by them (C11).

        `records_hash` is SHA-256 hex over the run's decision records in order (B11 defines it).
        The ticket must pass `verify_ticket` (TicketError otherwise), the hash must be 64
        lowercase hex characters (LedgerError), and a run settles once (LedgerError the second
        time). Appends a `result` entry as the module docstring says, and returns its Settlement.
        """
        raise NotImplementedError

    def trial_count(self) -> int:
        """The number of trials holding at least one run, as the module docstring defines it:
        the count G6's Deflated Sharpe deflates by. Refusals, gate configs and yardsticks add
        nothing."""
        raise NotImplementedError

    # --- tickets and refusals (K5, ARCHITECTURE §6.3) -------------------------------------------

    def verify_ticket(self, ticket: object) -> TrialTicket:
        """`ticket`, if this ledger object issued it and it is unchanged; otherwise TicketError.

        Refused, each with a message saying why: anything whose type is not exactly TrialTicket;
        a TrialTicket whose token is missing or not the private one; one this ledger object never
        issued, judged by identity (a copy, one built with the token by anyone else, one issued by
        another ledger object); and one whose fields differ from what was issued. It reads no
        database, so DataPolicy can call it on every read. This is DataPolicy's LedgerPort.
        """
        raise NotImplementedError

    def record_refusal(self, refusal: Refusal) -> None:
        """Append a `refusal` entry: purpose, first_day, last_day, error, reason, ticket."""
        raise NotImplementedError

    # --- gate configuration (ARCHITECTURE §10) ------------------------------------------------

    def record_gate_config(self, config: Mapping[str, Any], *, reason: str) -> LedgerEntry | None:
        """Append a `gate_config` entry if `config`'s canonical hash differs from the last one
        recorded (or none was), and return it; return None, appending nothing, when it is the
        same. `previous` is the last recorded config_hash, or None for the first."""
        raise NotImplementedError

    def gate_config_hash(self) -> str | None:
        """The config_hash of the last gate configuration recorded, or None."""
        raise NotImplementedError

    # --- the chain (ARCHITECTURE §10) ---------------------------------------------------------

    def entries(self) -> list[LedgerEntry]:
        """Every entry, in order."""
        raise NotImplementedError

    def verify(self) -> ChainReport:
        """Recheck the whole chain; see `verify_file`."""
        raise NotImplementedError


def verify_file(path: Path) -> ChainReport:
    """Recheck the chain of the ledger in the file at `path` without writing to it: no file is
    created and no migration runs. LedgerError if there is no ledger there.

    A problem is reported, naming the entry, when: the entries aren't numbered 1, 2, 3, ...
    without a gap; a kind isn't an EntryKind; a body isn't canonical JSON (it must re-encode to
    the same bytes); `prev` isn't the hash of the entry before (GENESIS for the first); a hash
    isn't `entry_hash` of its row; or the head doesn't name the last entry and its hash (an empty
    ledger has no head row). The chain holds when there are no problems.
    """
    raise NotImplementedError
