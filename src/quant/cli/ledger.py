"""`quant ledger`: the trial ledger in research.db (A2, ARCHITECTURE §10)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

DEFAULT_DB = Path("var/research.db")

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """The trial ledger (A2)."""


@app.command("verify")
def verify(
    db: Annotated[Path, typer.Option("--db", help="the research.db to check")] = DEFAULT_DB,
) -> None:
    """Recheck the ledger's hash chain (quant.research.ledger.verify_file).

    Exits 0 when the chain holds, printing the number of entries and the head hash. Exits 1
    when it is broken, printing each problem on its own line. Exits 2 when there is no ledger
    at `db` (LedgerError), saying so and creating nothing.
    """
    raise NotImplementedError
