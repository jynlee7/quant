"""The `quant` command (ARCHITECTURE §3). Each group of subcommands lives in its own module."""

from __future__ import annotations

import typer

from quant.cli import ledger

app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(ledger.app, name="ledger")


@app.callback()
def main() -> None:
    """SPY regime trader."""
