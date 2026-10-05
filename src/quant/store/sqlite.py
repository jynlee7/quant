"""SQLite access shared by the four databases (AD2, ARCHITECTURE §9.1).

Each database has a list of numbered migrations. `migrate` applies the missing ones in order, in
one transaction under the write lock, and records the version in `PRAGMA user_version`. A
database written by newer code is refused rather than misread.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from quant.core.errors import ConfigError

_SYNCHRONOUS = ("NORMAL", "FULL")
# Switching a new file to WAL takes an exclusive lock. When two connections switch the same new
# file at once, SQLite refuses one straight away, since waiting could deadlock, instead of calling
# the busy handler. So the switch is retried; once either lands, the retry is a no-op.
_WAL_TRIES = 500
_WAL_PAUSE = 0.01  # seconds: 500 tries span the 5 s busy timeout


@dataclass(frozen=True)
class Migration:
    version: int
    statements: tuple[str, ...]


def connect(path: Path | str, *, synchronous: str = "NORMAL") -> sqlite3.Connection:
    """A connection in WAL mode, with foreign keys on and explicit transactions.

    `synchronous="FULL"` is for ops.db and control.db, where a recorded decision or halt must
    survive a power cut (ARCHITECTURE §9.1).
    """
    if synchronous not in _SYNCHRONOUS:
        raise ConfigError(f"synchronous must be one of {_SYNCHRONOUS}, not {synchronous!r}")
    conn = sqlite3.connect(str(path), isolation_level=None)
    if str(path) != ":memory:":
        try:
            mode = _switch_to_wal(conn)
        except BaseException:
            conn.close()
            raise
        if mode.lower() != "wal":
            conn.close()
            raise ConfigError(f"{path}: couldn't switch to WAL mode (got {mode})")
    conn.execute(f"PRAGMA synchronous={synchronous}")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _switch_to_wal(conn: sqlite3.Connection) -> str:
    for _ in range(_WAL_TRIES - 1):
        try:
            return str(conn.execute("PRAGMA journal_mode=WAL").fetchone()[0])
        except sqlite3.OperationalError as exc:
            if (exc.sqlite_errorcode & 0xFF) != sqlite3.SQLITE_BUSY:  # busy, of any kind
                raise
            time.sleep(_WAL_PAUSE)
    return str(conn.execute("PRAGMA journal_mode=WAL").fetchone()[0])


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """One write transaction: committed if the block finishes, rolled back if it or the commit
    raises. Some errors (an interrupt, a full disk) make SQLite roll back by itself; then there
    is nothing left to roll back, and the error raised is SQLite's own."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def _user_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def migrate(conn: sqlite3.Connection, migrations: Sequence[Migration], *, name: str) -> int:
    """Apply the missing migrations. Returns the schema version.

    Several processes may open a new database at once. The first to take the write lock applies
    the migrations; the others wait for it, then find nothing left to do.
    """
    numbers = [m.version for m in migrations]
    if numbers != list(range(1, len(migrations) + 1)):
        raise ConfigError(
            f"{name}: migrations must be numbered 1, 2, 3, ... in order, not {numbers}"
        )
    if _user_version(conn) == len(migrations):
        return len(migrations)
    with transaction(conn):
        current = _user_version(conn)  # read again under the lock: another process may have won
        if current > len(migrations):
            raise ConfigError(
                f"{name} is at schema version {current}, newer than this code knows "
                f"({len(migrations)}); refusing to touch it"
            )
        for migration in migrations[current:]:
            for statement in migration.statements:
                conn.execute(statement)
            conn.execute(f"PRAGMA user_version = {migration.version}")
    return len(migrations)
