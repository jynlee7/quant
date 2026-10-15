"""`quant model`: the first model on FIT (D4, spec §4, PLAN §3 CP1). Built in B6b.

`quant model select` runs spec §4's selection with r1's config on the FIT bars of market.db
(quant.jobs.first_model), saves the artifact under `<artifacts>/models/` and writes the CP1
report under `<artifacts>/reports/`. It reads the store and writes no row to it. One full r1
selection takes minutes, so run it in the background.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from quant.cli.data import DEFAULT_DB
from quant.data.policy import FIT_START
from quant.hmm.train import TrainConfig, r1_train_config
from quant.jobs.first_model import FIT_LAST

DEFAULT_ARTIFACTS = Path("var/artifacts")

# The selection's settings: r1's (spec §4, PLAN §5). Tests swap in a smaller config here.
train_config: Callable[[], TrainConfig] = r1_train_config

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """HMM models (spec §4)."""


@app.command("select")
def select(
    db: Annotated[Path, typer.Option("--db", help="the market.db to read")] = DEFAULT_DB,
    artifacts: Annotated[
        Path, typer.Option("--artifacts", help="where models/ and reports/ go")
    ] = DEFAULT_ARTIFACTS,
    first: Annotated[
        datetime, typer.Option("--first", formats=["%Y-%m-%d"], help="the span's first day")
    ] = datetime(FIT_START.year, FIT_START.month, FIT_START.day),  # noqa: DTZ001 (a date)
    last: Annotated[
        datetime, typer.Option("--last", formats=["%Y-%m-%d"], help="the span's last day")
    ] = datetime(FIT_LAST.year, FIT_LAST.month, FIT_LAST.day),  # noqa: DTZ001 (a date)
    symbol: Annotated[str, typer.Option("--symbol")] = "SPY",
    workers: Annotated[
        int | None, typer.Option("--workers", help="training processes (default: every CPU)")
    ] = None,
    root: Annotated[
        Path, typer.Option("--root", help="the repository, for the commit and lockfile hash")
    ] = Path("."),
) -> None:
    """Select the first model on [--first, --last], which must lie inside FIT (2016-01-01 to
    2018-12-31), with `train_config()`, r1's features and label thresholds, provenance
    `code_provenance(root)`, and the wall clock for the run time.

    Prints the chosen K and covariance type, the artifact directory, both report paths and the
    selection's run time, and exits 0. Exits 2, saying why and creating nothing, when the span
    reaches outside FIT (PolicyError) or `--db` is not a file. Exits 1 when selection fails
    (ModelError), saying why.
    """
    raise NotImplementedError
