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
   opening at or after FIT's first New York midnight, read with purpose OPERATE (a fit is an
   operating read, like a refit: ARCHITECTURE §6.3). Nothing known at or after fit_end can
   reach the model, so a store holding later bars (DEV, the lockbox, a revision made today)
   gives exactly the same rows as one holding only FIT. The feature stream starts at the start
   of FIT, as it does live, whatever the span (ARCHITECTURE §4.2: a fit passes the bars from
   2016-01 and cuts its window with `FeatureMatrix.window`). Two checks stand behind the span's:
   a bar read from DEV's first session on is refused (PolicyError; OPERATE may read anything, so
   FIT-only can't rest on `fit_window` alone, B6 review), and every session of the span must
   give the view all its hourly bars, so a session the store learned after fit_end (written
   NOW) is refused (DataError), never silently dropped.
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
   never in the artifact: the artifact hash depends on the data, seeds, config and provenance
   (the commit, the lockfile hash and whether the tree was dirty, `code_provenance`), not on
   when or how long selection ran.

Times are UTC; New York dates appear only at the edges (the span). "Now" comes only from the
Clock passed in (K6).
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Final

import structlog

from quant.core.calendar import NEW_YORK
from quant.core.canonical import canonical_dumps, canonical_loads
from quant.core.clock import Clock
from quant.core.errors import ModelError, PolicyError
from quant.data.frames import HourlyBar
from quant.data.market_db import BarStore
from quant.data.policy import DEV_START, FIT_START, DataPolicy
from quant.features.menu import FeatureConfig
from quant.hmm.labels import LabelThresholds
from quant.hmm.model import save_model
from quant.hmm.train import TrainConfig, TrainingData, TrainResult, select_and_train
from quant.jobs._rows import training_rows

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
    if first < FIT_START or last > FIT_LAST or last < first:
        raise PolicyError(f"first model span must lie inside FIT: {FIT_START} to {FIT_LAST}")
    return (
        datetime.combine(first, time(), tzinfo=NEW_YORK).astimezone(UTC),
        datetime.combine(last + timedelta(days=1), time(), tzinfo=NEW_YORK).astimezone(UTC),
    )


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
      purpose=Purpose.OPERATE, since=<FIT_START's New York midnight>)`: every read names
      OPERATE, and the policy (the default DataPolicy when None) checks it. A bar of a session
      on or after DEV_START, or opening at or after fit_end, raises PolicyError. A session of
      [first, last] whose hourly bars the view doesn't return in full (one the store learned
      after fit_end, or never held) raises DataError naming the sessions.
    - `matrix` is `compute_features(bars, features).window(fit_start, fit_end)`: the stream
      starts at FIT's start, and the rows kept are those available in [fit_start, fit_end).
    - `returns[i]` is row i's bar's raw log return ln(C_t / C_prev), C_prev the close of the
      complete bar before it (the bar the feature stream consumed before it: incomplete bars
      are missing data and skipped, as `compute_features` skips them).
    - `sessions[i]` is `bar.session.toordinal()` of row i's bar.
    - `feature_config_hash` is `features.config_hash()`.

    Raises ModelError (from TrainingData) when fewer than two warm rows remain."""
    start, end = fit_window(first, last)

    def require_fit(bars: list[HourlyBar]) -> None:
        late = [bar for bar in bars if bar.session >= DEV_START or bar.t_open >= end]
        if late:
            raise PolicyError(
                f"first model rows must lie inside FIT, before DEV starts on {DEV_START} and before "
                f"{end.isoformat()}; read a bar of {late[0].session}"
            )

    return training_rows(
        store,
        symbol,
        start=start,
        end=end,
        first=first,
        last=last,
        features=features,
        policy=policy,
        validate=require_fit,
    )


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
    data = training_data(store, symbol, first=first, last=last, features=features, policy=policy)
    started = clock.now()
    result = select_and_train(
        data, thresholds=thresholds, config=config, provenance=provenance, workers=workers
    )
    seconds = (clock.now() - started).total_seconds()
    directory = save_model(result.model, artifacts / "models")
    report = first_model_report(
        result, symbol=symbol, first=first, last=last, directory=directory, seconds=seconds
    )
    reports = artifacts / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report_json = reports / f"first-model-{result.model.artifact_hash()}.json"
    report_markdown = report_json.with_suffix(".md")
    for path, content in (
        (report_json, canonical_dumps(report)),
        (report_markdown, render_markdown(report).encode()),
    ):
        with tempfile.NamedTemporaryFile(dir=reports, delete=False) as handle:
            temporary = Path(handle.name)
            try:
                handle.write(content)
                handle.close()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    structlog.get_logger().info(
        LOG_EVENT,
        artifact=result.model.artifact_hash(),
        n_states=result.model.params.n_states,
        covariance_type=result.model.params.covariance_type,
        seconds=seconds,
    )
    return FirstModel(result, directory, report, report_json, report_markdown, seconds)


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
    if result.selection is None:
        raise ModelError("first model report requires a selected model")
    model = result.model
    info = model.training.info
    report = {
        "kind": REPORT_KIND,
        "symbol": symbol,
        "span": {"first": first, "last": last},
        "fit_window": {"start": model.fit_start, "end": model.fit_end},
        "artifact": {"hash": model.artifact_hash(), "path": str(directory)},
        "chosen": {
            "n_states": model.params.n_states,
            "covariance_type": model.params.covariance_type,
        },
        "transition_matrix": model.params.transmat.tolist(),
        **{
            key: info[key]
            for key in (
                "states",
                "selection",
                "validation",
                "sanity",
                "restarts",
                "n_rows",
                "loglik",
                "bic",
                "data_hash",
                "config",
            )
        },
        "feature_config_hash": model.feature_config_hash,
        "selection_seconds": seconds,
    }
    normalized: dict[str, Any] = canonical_loads(canonical_dumps(report))
    return normalized


def render_markdown(report: Mapping[str, Any]) -> str:
    """A short Markdown rendering of a first-model report (`first_model_report`'s object, or
    its canonical JSON parsed back), for CP1:

    - a first line `# First model: <symbol>, <span first> to <span last>`;
    - bullets naming the artifact hash in backticks, the choice as `K = <n_states>,
      <covariance_type> covariance`, the selection rule, the fit window as the span's days with
      its exclusive end (`Fit window: 2016-01-01 to 2018-12-31 (end exclusive
      2019-01-01T05:00Z); <n_rows> rows`, so no reader takes it for a read of DEV's first day),
      and the run time as `<selection_seconds formatted .1f> s`;
    - a `## Transition matrix` table with one row per state, each entry formatted `.4f`;
    - a `## States` table with one row per state: state, label, v, d, annualized mean return,
      annualized volatility, expected duration in bars and in sessions ("inf" when infinite),
      occupancy;
    - a `## Candidates` table with one row per scored candidate in grid order, each row starting
      `| <n_states> | <covariance_type> |`, then parameters, log-likelihood, BIC, validation
      mean and standard error, and the sanity verdict; then the candidates set aside, if any.

    Ends with a newline."""
    chosen, span = report["chosen"], report["span"]
    lines = [
        f"# First model: {report['symbol']}, {span['first']} to {span['last']}",
        "",
        f"- Artifact: `{report['artifact']['hash']}`",
        f"- K = {chosen['n_states']}, {chosen['covariance_type']} covariance",
        f"- Selection rule: {report['selection']['rule']}",
        f"- Fit window: {span['first']} to {span['last']} (end exclusive {_minute(report['fit_window']['end'])}); {report['n_rows']} rows",
        f"- Selection time: {report['selection_seconds']:.1f} s",
        "",
        "## Transition matrix",
        "",
    ]

    def table(headers: list[str], rows: list[list[str]]) -> None:
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        lines.append("")

    table(
        ["State", *map(str, range(chosen["n_states"]))],
        [[str(i), *(f"{v:.4f}" for v in row)] for i, row in enumerate(report["transition_matrix"])],
    )
    lines.extend(["## States", ""])
    fields = [
        "state",
        "label",
        "v",
        "d",
        "mean_return",
        "volatility",
        "duration_bars",
        "duration_sessions",
        "occupancy",
    ]
    table(
        [
            "State",
            "Label",
            "v",
            "d",
            "Annualized return",
            "Annualized volatility",
            "Duration (bars)",
            "Duration (sessions)",
            "Occupancy",
        ],
        [
            ["inf" if row[key] is None else str(row[key]) for key in fields]
            for row in report["states"]
        ],
    )
    lines.extend(["## Candidates", ""])
    fields = [
        "n_states",
        "covariance_type",
        "n_params",
        "loglik",
        "bic",
        "validation_mean",
        "validation_se",
        "sanity_passed",
    ]
    table(
        [
            "K",
            "Covariance",
            "Parameters",
            "Log-likelihood",
            "BIC",
            "Validation mean",
            "Validation SE",
            "Sanity passed",
        ],
        [[str(row[key]) for key in fields] for row in report["selection"]["table"]],
    )
    for row in report["selection"]["set_aside"]:
        lines.append(
            f"- Set aside: K = {row['n_states']}, {row['covariance_type']}: {row['failed']}"
        )
    return "\n".join(lines).rstrip() + "\n"


def _minute(stamp: str) -> str:
    """A canonical UTC timestamp ("2019-01-01T05:00:00Z") to the minute ("2019-01-01T05:00Z")."""
    return datetime.fromisoformat(stamp).astimezone(UTC).strftime("%Y-%m-%dT%H:%MZ")


def _git_dir(root: Path) -> tuple[Path, Path] | None:
    """(the git directory, the common directory holding refs and packed-refs) of the checkout
    at `root`: `root/.git` for both in a plain checkout; for a worktree, whose `.git` is a file
    `gitdir: <path>`, that path and the directory its `commondir` file names. None when neither
    exists."""
    dot = root / ".git"
    if dot.is_dir():
        return dot, dot
    if not dot.is_file():
        return None
    text = dot.read_text().strip()
    if not text.startswith("gitdir: "):
        return None
    git = Path(text[8:])
    git = git if git.is_absolute() else root / git
    common_file = git / "commondir"
    common = git
    if common_file.is_file():
        named = Path(common_file.read_text().strip())
        common = named if named.is_absolute() else git / named
    return (git, common) if git.is_dir() and common.is_dir() else None


def _dirty(root: Path) -> bool | None:
    """Whether the tracked files at `root` differ from HEAD (`git status --porcelain
    --untracked-files=no`), None when git can't say. git runs with every `GIT_*` variable
    removed from its environment, as `quant.risk.limits.committed_bytes` does: the commit hook
    exports GIT_INDEX_FILE and friends, which would point it at another index."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            env={key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip())


def code_provenance(root: Path) -> dict[str, str | bool | None]:
    """What the trainer records as `provenance` (ARCHITECTURE §7.2: the code commit and the
    lockfile hash, and whether the tree was dirty):

    - "commit": the 40-hex commit HEAD names, read from files: the git directory is `root/.git`,
      or in a worktree the directory its `.git` file names (`_git_dir`). HEAD holds the hash
      itself when detached, else the ref it names (`ref: refs/heads/<branch>`), read from
      `<common>/<ref>` or, when that file is absent, from `<common>/packed-refs`. None when
      there is no git directory or the ref can't be resolved to 40 lowercase hex;
    - "dirty": whether tracked files differ from HEAD (`_dirty`, one `git status` run with the
      hook's GIT_* variables scrubbed); None when there is no git directory or git can't say;
    - "lockfile_sha256": the SHA-256 in hex of `root/uv.lock`'s bytes, None when it is missing.
    """
    found = _git_dir(root)
    commit = None
    if found is not None and (found[0] / "HEAD").is_file():
        git, common = found
        head = (git / "HEAD").read_text().strip()
        if head.startswith("ref: "):
            ref = head[5:]
            path = common / ref
            if ref.startswith("refs/") and ".." not in Path(ref).parts:
                if path.is_file():
                    head = path.read_text().strip()
                elif (common / "packed-refs").is_file():
                    head = next(
                        (
                            line.split()[0]
                            for line in (common / "packed-refs").read_text().splitlines()
                            if len(line.split()) == 2 and line.split()[1] == ref
                        ),
                        "",
                    )
        if re.fullmatch(r"[0-9a-f]{40}", head):
            commit = head
    lock = root / "uv.lock"
    return {
        "commit": commit,
        "dirty": None if found is None else _dirty(root),
        "lockfile_sha256": hashlib.sha256(lock.read_bytes()).hexdigest()
        if lock.is_file()
        else None,
    }
