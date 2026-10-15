"""`quant data`: the backfill and the data-quality report on market.db (PLAN B6).

The backfill runs alone, never beside a backtest (B2 "Snapshots"). Keys come only through the
settings loader (`quant.core.settings.load_settings`), which registers them as secrets before
anything logs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from quant.core.settings import DEFAULT_ENV_FILE
from quant.data.policy import FIT_START

DEFAULT_DB = Path("var/market.db")

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Market data in market.db (spec §2)."""


@app.command("backfill")
def backfill(
    db: Annotated[Path, typer.Option("--db", help="the market.db to fill")] = DEFAULT_DB,
    env_file: Annotated[
        Path, typer.Option("--env-file", help="the dotenv file with the Alpaca keys")
    ] = DEFAULT_ENV_FILE,
    start: Annotated[
        datetime, typer.Option("--start", formats=["%Y-%m-%d"], help="the first session")
    ] = datetime(FIT_START.year, FIT_START.month, FIT_START.day),  # noqa: DTZ001 (a date)
    end: Annotated[
        datetime | None,
        typer.Option("--end", formats=["%Y-%m-%d"], help="the last session (default: latest)"),
    ] = None,
    symbol: Annotated[str, typer.Option("--symbol")] = "SPY",
) -> None:
    """Fill market.db with SIP minutes and corporate actions (quant.data.backfill).

    Resumable: sessions already stored are skipped. Exits 0 when done, printing what it did.
    Exits 2 when the settings can't be loaded (ConfigError), saying why and creating nothing.
    Exits 3 when the feed reports a split, which halts trading (ARCHITECTURE §6.1).
    """
    raise NotImplementedError("B6: quant data backfill")


@app.command("quality")
def quality(
    db: Annotated[Path, typer.Option("--db", help="the market.db to check")] = DEFAULT_DB,
    first: Annotated[
        datetime, typer.Option("--first", formats=["%Y-%m-%d"], help="the first session")
    ] = datetime(FIT_START.year, FIT_START.month, FIT_START.day),  # noqa: DTZ001 (a date)
    last: Annotated[
        datetime | None,
        typer.Option("--last", formats=["%Y-%m-%d"], help="the last session (default: latest)"),
    ] = None,
    explanations: Annotated[
        Path | None, typer.Option("--explanations", help="a JSON object: gap key -> reason")
    ] = None,
    symbol: Annotated[str, typer.Option("--symbol")] = "SPY",
) -> None:
    """Print the data-quality report (quant.data.quality) as canonical JSON. Exits 1 while a
    gap is unexplained."""
    raise NotImplementedError("B6: quant data quality")
