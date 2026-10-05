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


@dataclass(frozen=True)
class Migration:
    version: int
    statements: tuple[str, ...]


def connect(path: Path | str, *, synchronous: str = "NORMAL") -> sqlite3.Connection:
    """A connection in WAL mode, with foreign keys on and explicit transactions."""
    raise NotImplementedError("red:B2")


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    raise NotImplementedError("red:B2")


def migrate(conn: sqlite3.Connection, migrations: Sequence[Migration], *, name: str) -> int:
    """Apply the missing migrations. Returns the schema version."""
    raise NotImplementedError("red:B2")
