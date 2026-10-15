"""The first model, selected on FIT (D4, spec §4, PLAN §3 CP1, PLAN §4 B6). Built in B6b.

D4 fits the first model on 2016-01 → 2018-12, the FIT segment, with spec §4's selection (B4's
`select_and_train` with r1's config). This module turns a market.db into that model and the
report CP1 shows you:

1. **The span** is a pair of New York session dates [first, last] inside FIT. Anything reaching
   before FIT_START or past FIT_LAST (2018-12-31) is refused with PolicyError before the store is
   read or a file is written: DEV stays dark until trial #1 (C11), and the lockbox is never
   read for this (C1). `fit_window` turns the span into the fit window [fit_start, fit_end): New
   York midnight starting `first` and New York midnight after `last`, in UTC.
2. **The rows** (`training_data`) come only through the point-in-time view, as a backtest at
   τ = fit_end would read them: `store.view(fit_end, as_of=fit_end, policy=policy)`, hourly bars
   opening at or after fit_start, read with purpose OPERATE (a fit is an operating read, like a
   refit: ARCHITECTURE §6.3). Nothing known at or after fit_end can reach the model, so a store
   holding later bars (DEV, the lockbox, a revision made today) gives exactly the same rows as
   one holding only FIT. The bars are featurized by B3's `compute_features` from the span's
   first bar on, and every warm row is kept (all of them lie in the window by construction).
3. **The selection** is `select_and_train` on those rows with the given config, thresholds,
   provenance and workers. Its run time is measured on the injected Clock, read exactly twice,
   just before and just after selection (K6: never the wall clock directly), recorded in the
   report as `selection_seconds` and logged (B4's "done when").
4. **The artifact** is `hmm.model.save_model(model, artifacts / "models")`: the directory
   `<artifacts>/models/<artifact_hash>/` with params.npz (written and read with
   allow_pickle=False) and meta.json (ARCHITECTURE §7.2).
5. **The report** (`first_model_report`) holds everything CP1 lists, as canonical JSON in
   `<artifacts>/reports/first-model-<artifact_hash>.json`, and a short Markdown rendering
   (`render_markdown`) beside it as `first-model-<artifact_hash>.md`. Both are written to a
   temporary file in the same directory and renamed into place. The run time is in the report,
   never in the artifact: the artifact hash depends only on data, seeds and config.

Times are UTC; New York dates appear only at the edges (the span). "Now" comes only from the
Clock passed in (K6).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Final

from quant.core.clock import Clock
from quant.data.market_db import BarStore
from quant.data.policy import DEV_START, DataPolicy
from quant.features.menu import FeatureConfig
from quant.hmm.labels import LabelThresholds
from quant.hmm.train import TrainConfig, TrainingData, TrainResult

# FIT's last day: the day before DEV starts (spec §2 splits, D4).
FIT_LAST: Final = DEV_START - timedelta(days=1)
# The report's "kind", so a reader can tell it from other reports.
REPORT_KIND: Final = "first_model"
# The structlog event that records the selection's run time.
LOG_EVENT: Final = "first model selected"


def fit_window(first: date, last: date) -> tuple[datetime, datetime]:
    """(fit_start, fit_end) in UTC for the session dates [first, last]: New York midnight at the
    start of `first`, and New York midnight at the start of the day after `last`. So
    (2016-01-01, 2018-12-31) gives (2016-01-01T05:00Z, 2019-01-01T05:00Z). Raises PolicyError
    when first < FIT_START, last > FIT_LAST or last < first (module docstring, step 1)."""
    raise NotImplementedError


def training_data(
    store: BarStore,
    symbol: str,
    *,
    first: date,
    last: date,
    features: FeatureConfig,
    policy: DataPolicy | None = None,
) -> TrainingData:
    """The fit window's rows for [first, last], read through the point-in-time view (module
    docstring, step 2).

    - The window is `fit_window(first, last)`, so a span outside FIT raises PolicyError before
      anything is read.
    - The bars are `store.view(fit_end, as_of=fit_end, policy=policy).hourly(symbol,
      purpose=Purpose.OPERATE, since=fit_start)`: every read names OPERATE, and the policy (the
      default DataPolicy when None) checks it.
    - `matrix` is `compute_features(bars, features)`, its rows' available_at all in
      [fit_start, fit_end).
    - `returns[i]` is row i's bar's raw log return ln(C_t / C_prev), C_prev the close of the
      complete bar before it (the bar the feature stream consumed before it: incomplete bars
      are missing data and skipped, as `compute_features` skips them).
    - `sessions[i]` is `bar.session.toordinal()` of row i's bar.
    - `feature_config_hash` is `features.config_hash()`.

    Raises ModelError (from TrainingData) when fewer than two warm rows remain."""
    raise NotImplementedError


@dataclass(frozen=True, eq=False)
class FirstModel:
    """What `select_first_model` produced.

    - `result`: the TrainResult (model, sanity, selection);
    - `directory`: the saved artifact, `<artifacts>/models/<artifact_hash>`;
    - `report`: the report as its canonical JSON parses back (plain JSON types), equal to
      `canonical_loads(report_json.read_bytes())`;
    - `report_json`, `report_markdown`: the two report files;
    - `seconds`: the selection's run time on the Clock, the report's `selection_seconds`.
    """

    result: TrainResult
    directory: Path
    report: Mapping[str, Any]
    report_json: Path
    report_markdown: Path
    seconds: float


def select_first_model(
    store: BarStore,
    *,
    symbol: str,
    first: date,
    last: date,
    features: FeatureConfig,
    config: TrainConfig,
    thresholds: LabelThresholds,
    provenance: Mapping[str, Any],
    artifacts: Path,
    clock: Clock,
    workers: int | None = None,
    policy: DataPolicy | None = None,
) -> FirstModel:
    """Select and save the first model for the span [first, last] (module docstring, steps
    1–5): `training_data(...)`, then `select_and_train(data, thresholds=..., config=...,
    provenance=..., workers=...)` timed on `clock`, then `save_model` under `artifacts /
    "models"`, then the report's JSON and Markdown under `artifacts / "reports"`, then a
    structlog info event LOG_EVENT with `artifact` (the hash), `n_states`, `covariance_type`
    and `seconds`.

    Raises PolicyError for a span outside FIT before reading the store or creating any file or
    directory; ModelError when selection fails (every candidate set aside or rejected), after
    which no artifact or report is written."""
    raise NotImplementedError


def first_model_report(
    result: TrainResult,
    *,
    symbol: str,
    first: date,
    last: date,
    directory: Path,
    seconds: float,
) -> dict[str, Any]:
    """CP1's first-model report (PLAN §3) for a `select_and_train` result, as a canonical JSON
    object (every value has exactly one canonical form). With `info` = the model's
    `training.info`:

    - "kind": REPORT_KIND; "symbol"; "span": {"first", "last"} as ISO dates;
    - "fit_window": {"start", "end"}: the model's fit_start and fit_end;
    - "artifact": {"hash": the model's artifact hash, "path": str(directory)};
    - "chosen": {"n_states", "covariance_type"} of the model;
    - "transition_matrix": the model's transition matrix A as a K × K list of lists, row j
      being P(next state | state j);
    - "states", "selection", "validation", "sanity", "restarts", "n_rows", "loglik", "bic",
      "data_hash", "config": `info`'s entries of those names, verbatim (the state table with
      labels, v and d; the BIC and validation table and the rule that chose; the validation
      block; the sanity checks);
    - "feature_config_hash": the model's;
    - "selection_seconds": `seconds`.

    Raises ModelError when `result.selection` is None (a `train_candidate` result was not
    selected, so it has no BIC and validation table)."""
    raise NotImplementedError


def render_markdown(report: Mapping[str, Any]) -> str:
    """A short Markdown rendering of a first-model report (`first_model_report`'s object, or
    its canonical JSON parsed back), for CP1:

    - a first line `# First model: <symbol>, <span first> to <span last>`;
    - bullets naming the artifact hash in backticks, the choice as `K = <n_states>,
      <covariance_type> covariance`, the selection rule, the fit window and rows, and the run
      time as `<selection_seconds formatted .1f> s`;
    - a `## Transition matrix` table with one row per state, each entry formatted `.4f`;
    - a `## States` table with one row per state: state, label, v, d, annualized mean return,
      annualized volatility, expected duration in bars and in sessions ("inf" when infinite),
      occupancy;
    - a `## Candidates` table with one row per scored candidate in grid order, each row starting
      `| <n_states> | <covariance_type> |`, then parameters, log-likelihood, BIC, validation
      mean and standard error, and the sanity verdict; then the candidates set aside, if any.

    Ends with a newline."""
    raise NotImplementedError


def code_provenance(root: Path) -> dict[str, str | None]:
    """What the trainer records as `provenance` (ARCHITECTURE §7.2: the code commit and the
    lockfile hash), read from files, never a subprocess:

    - "commit": the 40-hex commit `root/.git/HEAD` names: the hash itself when HEAD is
      detached, else the ref it names (`ref: refs/heads/<branch>`), read from
      `root/.git/<ref>` or, when that file is absent, from `root/.git/packed-refs`. None when
      `root/.git` is not a directory or the ref can't be resolved to 40 lowercase hex;
    - "lockfile_sha256": the SHA-256 in hex of `root/uv.lock`'s bytes, None when it is missing.
    """
    raise NotImplementedError
