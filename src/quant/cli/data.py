"""`quant data`: the backfill and the data-quality report on market.db (PLAN B6).

The backfill runs alone, never beside a backtest (B2 "Snapshots"). Keys come only through the
settings loader (`quant.core.settings.load_settings`), which registers them as secrets before
anything logs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from quant.core.calendar import TradingCalendar
from quant.core.canonical import canonical_dumps
from quant.core.clock import WallClock
from quant.core.errors import ConfigError
from quant.core.schedule import Timing
from quant.core.settings import DEFAULT_ENV_FILE, load_settings
from quant.data.alpaca_data import AlpacaMarketData
from quant.data.backfill import backfill as run_backfill
from quant.data.market_db import BarStore
from quant.data.policy import FIT_START
from quant.data.quality import quality_report

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

    Resumable: sessions it already stored are skipped. It never goes past the last session
    whose minutes are all available, whatever --end says. Exits 0 when done, printing what it
    did, and on stderr any session written NOW and any action it couldn't store.
    Exits 2 when the settings can't be loaded (ConfigError), saying why and creating nothing.
    Exits 3 when the feed reports a split, which halts trading (ARCHITECTURE §6.1).
    """
    try:
        settings = load_settings(env_file)
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from None
    clock = WallClock()
    source = AlpacaMarketData(settings.alpaca, clock=clock)
    db.parent.mkdir(parents=True, exist_ok=True)
    store = BarStore.open(db, clock=clock)
    try:
        result = run_backfill(
            store, source, symbol, start.date(), None if end is None else end.date(), clock=clock
        )
        typer.echo(
            f"Fetched {len(result.fetched)}, skipped {result.skipped}, empty {len(result.empty)}; last {result.last}"
        )
        if result.revised:
            sessions = ", ".join(day.isoformat() for day in result.revised)
            typer.echo(f"Completed as NOW (history can't revise them): {sessions}", err=True)
        for conflict in result.actions.conflicts:
            typer.echo(f"Action not stored: {conflict}", err=True)
        if result.actions.halt:
            typer.echo(result.actions.halt, err=True)
            raise typer.Exit(3)
    finally:
        store.close()


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
    reasons = None
    if explanations is not None:
        reasons = json.loads(explanations.read_text())
        if not isinstance(reasons, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in reasons.items()
        ):
            raise typer.BadParameter("explanations must map gap keys to strings")
    if not db.is_file():
        raise typer.BadParameter("market.db does not exist")
    calendar = TradingCalendar.xnys()
    now = WallClock().now()
    days = calendar.sessions(first.date(), now.date())
    available = [d for d in days if Timing().available_at(calendar.session(d).close) <= now]
    end = last.date() if last is not None else (available[-1] if available else first.date())
    store = BarStore.open(db)
    try:
        report = quality_report(store, symbol, first.date(), end, explanations=reasons)
        typer.echo(canonical_dumps(report.to_json()).decode())
        if not report.ok:
            raise typer.Exit(1)
    finally:
        store.close()
