"""SQLite access shared by the four databases (AD2, ARCHITECTURE §9.1).

Each database has a list of numbered migrations. `migrate` applies the missing ones in order,
each in its own transaction, and records the version in `PRAGMA user_version`. A database
written by newer code is refused rather than misread.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from quant.core.errors import ConfigError

_SYNCHRONOUS = ("NORMAL", "FULL")


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
        mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if str(mode).lower() != "wal":
            conn.close()
            raise ConfigError(f"{path}: couldn't switch to WAL mode (got {mode})")
    conn.execute(f"PRAGMA synchronous={synchronous}")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """One write transaction: committed if the block finishes, rolled back if it raises."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def migrate(conn: sqlite3.Connection, migrations: Sequence[Migration], *, name: str) -> int:
    """Apply the missing migrations. Returns the schema version."""
    numbers = [m.version for m in migrations]
    if numbers != list(range(1, len(migrations) + 1)):
        raise ConfigError(
            f"{name}: migrations must be numbered 1, 2, 3, ... in order, not {numbers}"
        )
    current = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if current > len(migrations):
        raise ConfigError(
            f"{name} is at schema version {current}, newer than this code knows "
            f"({len(migrations)}); refusing to touch it"
        )
    for migration in migrations[current:]:
        with transaction(conn):
            for statement in migration.statements:
                conn.execute(statement)
            conn.execute(f"PRAGMA user_version = {migration.version}")
    return len(migrations)
