"""B10: the ledger's hash chain catches a planted tampering of any row (A2, P3, ARCHITECTURE §10).

Each test writes a ledger file holding every kind of entry, checks that it verifies, then edits
the file behind the ledger's back: a row edited, deleted, reordered or inserted, a well-formed row
appended, or the head rewritten. `verify_file` must report it, and `quant ledger verify` must exit
non-zero.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant.cli.main import app
from quant.core.canonical import canonical_dumps, canonical_loads
from quant.core.clock import SimClock
from quant.core.epoch import from_us
from quant.core.errors import LedgerError
from quant.research.ledger import entry_hash, verify_file
from tests.support.ledger import LEDGER_START, filled_ledger

Tamper = Callable[[sqlite3.Connection], None]


def _row(conn: sqlite3.Connection, seq: int) -> tuple[str, int, str, str, str]:
    """(kind, at, body, prev, hash) of entry `seq`."""
    row: tuple[str, int, str, str, str] = conn.execute(
        "SELECT kind, at, body, prev, hash FROM entries WHERE seq = ?", (seq,)
    ).fetchone()
    return row


def _edit_body(conn: sqlite3.Connection) -> None:
    body = canonical_loads(_row(conn, 2)[2])
    body["trial"] = "T9"
    conn.execute("UPDATE entries SET body = ? WHERE seq = 2", (canonical_dumps(body).decode(),))


def _edit_kind(conn: sqlite3.Connection) -> None:
    conn.execute("UPDATE entries SET kind = 'refusal' WHERE seq = 3")


def _edit_time(conn: sqlite3.Connection) -> None:
    conn.execute("UPDATE entries SET at = at + 1 WHERE seq = 2")


def _edit_and_rehash(conn: sqlite3.Connection) -> None:
    """Edit a row and give it a hash of its own: the next row's link breaks instead."""
    kind, at, text, prev, _ = _row(conn, 2)
    body = canonical_loads(text)
    body["trial"] = "T9"
    conn.execute(
        "UPDATE entries SET body = ?, hash = ? WHERE seq = 2",
        (canonical_dumps(body).decode(), entry_hash(2, kind, from_us(at), body, prev)),
    )


def _delete_middle(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM entries WHERE seq = 3")


def _delete_last(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM entries WHERE seq = (SELECT MAX(seq) FROM entries)")


def _reorder(conn: sqlite3.Connection) -> None:
    conn.execute("UPDATE entries SET seq = 100 WHERE seq = 2")
    conn.execute("UPDATE entries SET seq = 2 WHERE seq = 3")
    conn.execute("UPDATE entries SET seq = 3 WHERE seq = 100")


def _well_formed(conn: sqlite3.Connection, seq: int, prev: str) -> tuple[object, ...]:
    """A row that is right in itself: its hash is entry_hash of its own columns."""
    _, at, _, _, _ = _row(conn, 1)
    body = {
        "purpose": "score",
        "first_day": "2019-01-02",
        "last_day": "2019-01-03",
        "error": "PolicyError",
        "reason": "planted",
        "ticket": None,
    }
    digest = entry_hash(seq, "refusal", from_us(at), body, prev)
    return (seq, "refusal", at, canonical_dumps(body).decode(), prev, digest)


def _insert_middle(conn: sqlite3.Connection) -> None:
    conn.execute("UPDATE entries SET seq = seq + 100 WHERE seq >= 2")
    conn.execute("UPDATE entries SET seq = seq - 99 WHERE seq >= 102")
    row = _well_formed(conn, 2, _row(conn, 1)[4])
    conn.execute(
        "INSERT INTO entries (seq, kind, at, body, prev, hash) VALUES (?, ?, ?, ?, ?, ?)", row
    )


def _append_behind_its_back(conn: sqlite3.Connection) -> None:
    last = conn.execute("SELECT MAX(seq) FROM entries").fetchone()[0]
    row = _well_formed(conn, last + 1, _row(conn, last)[4])
    conn.execute(
        "INSERT INTO entries (seq, kind, at, body, prev, hash) VALUES (?, ?, ?, ?, ?, ?)", row
    )


def _rewrite_head(conn: sqlite3.Connection) -> None:
    conn.execute("UPDATE head SET seq = seq - 1, hash = (SELECT hash FROM entries WHERE seq = 5)")


TAMPERS: dict[str, Tamper] = {
    "edit a body": _edit_body,
    "edit a kind": _edit_kind,
    "edit a time": _edit_time,
    "edit and rehash a row": _edit_and_rehash,
    "delete a middle row": _delete_middle,
    "delete the last row": _delete_last,
    "reorder two rows": _reorder,
    "insert a row in the middle": _insert_middle,
    "append a row behind its back": _append_behind_its_back,
    "rewrite the head": _rewrite_head,
}


def _verify_cli(path: Path) -> tuple[int, str]:
    result = CliRunner().invoke(
        app, ["ledger", "verify", "--db", str(path)], catch_exceptions=False
    )
    return result.exit_code, result.output


@pytest.mark.spec("A2", "P3", "ARCH§10")
@pytest.mark.parametrize("tamper", TAMPERS.values(), ids=TAMPERS.keys())
def test_tampering_with_any_row_fails_verify(tmp_path: Path, tamper: Tamper) -> None:
    path = tmp_path / "research.db"
    filled_ledger(path, SimClock(LEDGER_START))
    intact = verify_file(path)
    assert (intact.ok, intact.entries) == (True, 6)
    with closing(sqlite3.connect(path)) as conn:
        tamper(conn)
        conn.commit()
    report = verify_file(path)
    assert not report.ok
    assert report.problems
    code, output = _verify_cli(path)
    assert code == 1
    assert output.strip()


@pytest.mark.spec("A2", "ARCH§10")
def test_quant_ledger_verify_passes_an_intact_ledger(tmp_path: Path) -> None:
    path = tmp_path / "research.db"
    filled_ledger(path, SimClock(LEDGER_START))
    report = verify_file(path)
    assert report.head is not None
    code, output = _verify_cli(path)
    assert code == 0
    assert output.startswith("6 entries; head ")
    assert report.head in output


@pytest.mark.spec("A2", "ARCH§10")
def test_quant_ledger_verify_creates_nothing_where_there_is_no_ledger(tmp_path: Path) -> None:
    path = tmp_path / "var" / "research.db"
    with pytest.raises(LedgerError):
        verify_file(path)
    code, output = _verify_cli(path)
    assert code == 2
    assert output.strip()
    assert not path.parent.exists()
