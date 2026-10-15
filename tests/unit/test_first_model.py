"""B6: the first model on FIT and its CP1 report (D4, spec §4, PLAN §3 CP1, ARCHITECTURE §7.2).

Selection runs with tiny settings (two candidates, two restarts, a one-month validation block)
on the truncation harness's synthetic world, whose bars lie inside FIT (2016-06 to 2016-12),
so these tests stay in the fast tier. The real selection on 2016–2018 is B6b's run, not a test.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from structlog.testing import capture_logs
from typer.testing import CliRunner

import quant.cli.model as cli_model
from quant.cli.main import app
from quant.core.canonical import canonical_dumps, canonical_loads
from quant.core.errors import PolicyError
from quant.data.policy import DataPolicy, Purpose, ScoringTicket
from quant.features.menu import r1_features
from quant.hmm.labels import r1_thresholds
from quant.hmm.model import load_model
from quant.hmm.select import Candidate
from quant.hmm.train import TrainConfig, r1_train_config
from quant.jobs.first_model import (
    FIT_LAST,
    LOG_EVENT,
    REPORT_KIND,
    FirstModel,
    code_provenance,
    fit_window,
    render_markdown,
    select_first_model,
    training_data,
)
from tests.support.bars import SYMBOL
from tests.support.truncation import (
    DATA_FIRST,
    DATA_LAST,
    MODEL_FIT_END,
    fixture_training_data,
    garbage_store,
    truncated_store,
    world,
)

RED = pytest.mark.xfail(strict=True, reason="red:B6", raises=NotImplementedError)

STEP = timedelta(seconds=7.25)  # each reading of the TickingClock
PROVENANCE = {"commit": "0" * 40, "lockfile_sha256": "1" * 64}


def tiny_config() -> TrainConfig:
    """r1's config with two candidates, two restarts, 50 bootstrap draws and a one-month
    validation block: selection in well under a second on the synthetic world."""
    return r1_train_config().model_copy(
        update={
            "n_restarts": 2,
            "candidates": (
                Candidate(n_states=2, covariance_type="diag"),
                Candidate(n_states=2, covariance_type="full"),
            ),
            "bootstrap_draws": 50,
            "validation_months": 1,
        }
    )


class TickingClock:
    """A Clock each of whose readings is STEP after the one before."""

    def __init__(self, start: datetime) -> None:
        self.start, self.reads = start, 0

    def now(self) -> datetime:
        reading = self.start + self.reads * STEP
        self.reads += 1
        return reading

    def sleep_until(self, when: datetime) -> None:
        raise AssertionError("selection never sleeps")


@dataclass(frozen=True)
class SpyPolicy(DataPolicy):
    """The default policy, recording every read it is asked to check."""

    reads: list[tuple[Purpose, date, date]] = field(default_factory=list)

    def check(
        self, purpose: Purpose, first: date, last: date, *, ticket: ScoringTicket | None = None
    ) -> None:
        self.reads.append((Purpose(purpose), first, last))
        super().check(purpose, first, last, ticket=ticket)


@dataclass(frozen=True)
class Selected:
    first: FirstModel
    artifacts: Path
    clock: TickingClock
    logs: list[MutableMapping[str, Any]]


@cache
def selected(root: Path) -> Selected:
    """`select_first_model` on the synthetic world's span with the tiny config, once per module
    (`root` is the module's temporary directory)."""
    clock = TickingClock(datetime(2026, 10, 15, 12, tzinfo=UTC))
    artifacts = root / "artifacts"
    with capture_logs() as logs:
        first = select_first_model(
            world().store,
            symbol=SYMBOL,
            first=DATA_FIRST,
            last=DATA_LAST,
            features=r1_features(),
            config=tiny_config(),
            thresholds=r1_thresholds(),
            provenance=PROVENANCE,
            artifacts=artifacts,
            clock=clock,
            workers=1,
        )
    return Selected(first, artifacts, clock, logs)


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("first_model")


# --- the span --------------------------------------------------------------------------------


@RED
@pytest.mark.spec("D4", "C11", "ARCH§6.3")
def test_the_fit_window_is_new_york_midnights_inside_fit() -> None:
    assert date(2018, 12, 31) == FIT_LAST
    assert fit_window(date(2016, 1, 1), FIT_LAST) == (
        datetime(2016, 1, 1, 5, tzinfo=UTC),
        datetime(2019, 1, 1, 5, tzinfo=UTC),
    )
    assert fit_window(DATA_FIRST, DATA_LAST) == (
        datetime(2016, 6, 1, 4, tzinfo=UTC),  # daylight time
        datetime(2016, 12, 31, 5, tzinfo=UTC),
    )
    for first, last in (
        (date(2016, 1, 1), date(2019, 1, 1)),  # reaches DEV
        (date(2016, 1, 1), date(2025, 6, 30)),  # reaches the lockbox
        (date(2015, 12, 31), date(2016, 6, 30)),  # before the data
        (date(2016, 7, 1), date(2016, 6, 30)),  # empty
    ):
        with pytest.raises(PolicyError):
            fit_window(first, last)


# --- the rows --------------------------------------------------------------------------------


@RED
@pytest.mark.spec("D4", "I1", "ARCH§6.2", "ARCH§6.3")
def test_training_data_is_read_through_the_view_to_operate() -> None:
    # The rows are the B5 fixture model's own (the same bars, features, returns and sessions),
    # read through the view at τ = fit_end with purpose OPERATE, and no day past the span.
    w = world()
    policy = SpyPolicy()
    last = date(2016, 10, 31)
    data = training_data(
        w.store, SYMBOL, first=DATA_FIRST, last=last, features=r1_features(), policy=policy
    )
    assert (data.fit_start, data.fit_end) == fit_window(DATA_FIRST, last)
    assert data.fit_end == datetime(2016, 11, 1, 4, tzinfo=UTC)
    fixture = fixture_training_data(MODEL_FIT_END)
    assert data.matrix.columns == fixture.matrix.columns
    for name in ("t_open", "available_at", "values"):
        assert np.array_equal(getattr(data.matrix, name), getattr(fixture.matrix, name)), name
    assert np.array_equal(data.returns, fixture.returns)
    assert np.array_equal(data.sessions, fixture.sessions)
    assert data.feature_config_hash == r1_features().config_hash()
    assert policy.reads
    assert {purpose for purpose, _, _ in policy.reads} == {Purpose.OPERATE}
    assert max(read_last for _, _, read_last in policy.reads) <= last


@RED
@pytest.mark.spec("D4", "I1", "C11", "ARCH§6.2")
def test_training_data_ignores_what_the_store_learned_after_the_span() -> None:
    # A store that knows later bars, and a revision of the span's last bar made after it, gives
    # the rows of a store that knows only what it knew at fit_end.
    w = world()
    last = date(2016, 11, 30)
    _, end = fit_window(DATA_FIRST, last)
    cut = next(d.tau for d in w.decisions if d.tau > end)  # the first decision of December
    noisy, garbage = garbage_store(w, cut, seed=7)
    assert garbage.revised_at_cut < end <= cut  # the span's last bar, revised at the cut
    known = truncated_store(w, end)
    hashes = {
        training_data(
            store, SYMBOL, first=DATA_FIRST, last=last, features=r1_features()
        ).data_hash()
        for store in (w.store, noisy, known)
    }
    noisy.close()
    known.close()
    assert len(hashes) == 1


# --- selection, the artifact and the report ---------------------------------------------------


@RED
@pytest.mark.spec("D4", "SPEC§4", "ARCH§7.2", "PLAN§3")
def test_selection_saves_an_artifact_that_loads_without_pickles(root: Path) -> None:
    run = selected(root)
    model = run.first.result.model
    digest = model.artifact_hash()
    assert run.first.directory == run.artifacts / "models" / digest
    assert load_model(run.first.directory).artifact_hash() == digest
    with np.load(run.first.directory / "params.npz", allow_pickle=False) as arrays:
        assert np.array_equal(arrays["transmat"], model.params.transmat)
    assert (model.fit_start, model.fit_end) == fit_window(DATA_FIRST, DATA_LAST)
    assert model.fit_end <= datetime(2019, 1, 1, 5, tzinfo=UTC)  # DEV stays dark
    assert run.first.result.selection is not None
    assert run.first.result.selection.chosen in tiny_config().candidates
    assert model.training.info["provenance"] == PROVENANCE


@RED
@pytest.mark.spec("D4", "SPEC§4", "PLAN§3")
def test_the_report_holds_what_cp1_lists(root: Path) -> None:
    run = selected(root)
    model = run.first.result.model
    info = model.training.info
    digest = model.artifact_hash()
    path = run.artifacts / "reports" / f"first-model-{digest}.json"
    assert run.first.report_json == path
    raw = path.read_bytes()
    assert canonical_dumps(canonical_loads(raw)) == raw
    report = canonical_loads(raw)
    assert report == run.first.report
    assert report["kind"] == REPORT_KIND
    assert report["symbol"] == SYMBOL
    assert report["span"] == {"first": "2016-06-01", "last": "2016-12-30"}
    assert report["artifact"]["hash"] == digest
    assert report["chosen"] == {
        "n_states": model.params.n_states,
        "covariance_type": model.params.covariance_type,
    }
    assert report["transition_matrix"] == model.params.transmat.tolist()
    expected = canonical_loads(canonical_dumps(info))
    for key in ("states", "selection", "validation", "sanity", "data_hash", "n_rows", "bic"):
        assert report[key] == expected[key], key
    labels = [model.labels.label_of(k).value for k in range(model.params.n_states)]
    assert [state["label"] for state in report["states"]] == labels
    table = report["selection"]["table"]
    assert [(row["n_states"], row["covariance_type"]) for row in table] == [
        (2, "diag"),
        (2, "full"),
    ]
    assert all({"bic", "validation_mean", "validation_se"} <= set(row) for row in table)
    assert report["fit_window"] == {
        "start": "2016-06-01T04:00:00Z",
        "end": "2016-12-31T05:00:00Z",
    }


@RED
@pytest.mark.spec("D4", "SPEC§4", "PLAN§3")
def test_the_selection_run_time_is_measured_on_the_clock_and_logged(root: Path) -> None:
    run = selected(root)
    assert run.clock.reads == 2
    assert run.first.seconds == STEP.total_seconds()
    assert run.first.report["selection_seconds"] == STEP.total_seconds()
    (event,) = [entry for entry in run.logs if entry["event"] == LOG_EVENT]
    model = run.first.result.model
    assert event["artifact"] == model.artifact_hash()
    assert event["seconds"] == STEP.total_seconds()
    assert (event["n_states"], event["covariance_type"]) == (
        model.params.n_states,
        model.params.covariance_type,
    )
    # The artifact holds no wall time: the run time lives in the report alone.
    assert str(STEP.total_seconds()) not in canonical_dumps(model.meta()).decode()


@RED
@pytest.mark.spec("D4", "SPEC§4", "PLAN§3")
def test_the_markdown_report_shows_what_cp1_lists(root: Path) -> None:
    run = selected(root)
    report = run.first.report
    text = run.first.report_markdown.read_text()
    assert run.first.report_markdown == run.first.report_json.with_suffix(".md")
    assert text == render_markdown(report)
    assert text.splitlines()[0] == "# First model: SPY, 2016-06-01 to 2016-12-30"
    assert text.endswith("\n")
    chosen = report["chosen"]
    assert f"K = {chosen['n_states']}, {chosen['covariance_type']} covariance" in text
    assert f"`{report['artifact']['hash']}`" in text
    assert f"{STEP.total_seconds():.1f} s" in text
    for heading in ("## Transition matrix", "## States", "## Candidates"):
        assert heading in text
    for row in report["transition_matrix"]:
        assert all(f"{value:.4f}" in text for value in row)
    for state in report["states"]:
        assert state["label"] in text
    lines = text.splitlines()
    for row in report["selection"]["table"]:
        assert any(
            line.startswith(f"| {row['n_states']} | {row['covariance_type']} |") for line in lines
        )


@RED
@pytest.mark.spec("D4", "C11", "A2")
@pytest.mark.parametrize(
    ("first", "last"),
    [(date(2016, 6, 1), date(2019, 1, 2)), (date(2015, 12, 31), date(2016, 12, 30))],
)
def test_a_span_outside_fit_is_refused_before_anything_is_read_or_written(
    tmp_path: Path, first: date, last: date
) -> None:
    policy = SpyPolicy()
    artifacts = tmp_path / "artifacts"
    with pytest.raises(PolicyError):
        select_first_model(
            world().store,
            symbol=SYMBOL,
            first=first,
            last=last,
            features=r1_features(),
            config=tiny_config(),
            thresholds=r1_thresholds(),
            provenance=PROVENANCE,
            artifacts=artifacts,
            clock=TickingClock(datetime(2026, 10, 15, tzinfo=UTC)),
            workers=1,
            policy=policy,
        )
    assert policy.reads == []
    assert not artifacts.exists()


# --- the command -----------------------------------------------------------------------------


def _market_db(path: Path) -> Path:
    target = sqlite3.connect(path)
    world().conn.backup(target)
    target.close()
    return path


def _contents(path: Path) -> dict[str, tuple[object, ...]]:
    """Each table's row count and latest known_at (or row count alone), to show the store was
    only read. Opening it may change its journal mode, so its bytes can't show that."""
    conn = sqlite3.connect(path)
    try:
        tables = [
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ]
        contents = {}
        for table in tables:
            columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            latest = ", MAX(known_at)" if "known_at" in columns else ""
            contents[table] = conn.execute(
                f"SELECT COUNT(*){latest} FROM {table}"  # noqa: S608 — names from sqlite_master
            ).fetchone()
        return contents
    finally:
        conn.close()


@RED
@pytest.mark.spec("D4", "SPEC§4", "PLAN§3")
def test_quant_model_select_writes_the_model_and_its_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli_model, "train_config", tiny_config)
    db, artifacts = _market_db(tmp_path / "market.db"), tmp_path / "artifacts"
    before = _contents(db)
    result = CliRunner().invoke(
        app,
        [
            "model",
            "select",
            "--db",
            str(db),
            "--artifacts",
            str(artifacts),
            "--first",
            "2016-06-01",
            "--last",
            "2016-12-30",
            "--workers",
            "1",
            "--root",
            str(tmp_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    (directory,) = (artifacts / "models").iterdir()
    model = load_model(directory)
    assert model.artifact_hash() in result.output
    reports = artifacts / "reports"
    assert {p.name for p in reports.iterdir()} == {
        f"first-model-{directory.name}.json",
        f"first-model-{directory.name}.md",
    }
    assert model.training.info["provenance"] == {"commit": None, "lockfile_sha256": None}
    assert _contents(db) == before  # the store is only read


@RED
@pytest.mark.spec("D4", "C11")
def test_quant_model_select_refuses_a_span_past_fit_and_a_missing_store(tmp_path: Path) -> None:
    db, artifacts = _market_db(tmp_path / "market.db"), tmp_path / "artifacts"
    common = ["model", "select", "--artifacts", str(artifacts), "--root", str(tmp_path)]
    past = CliRunner().invoke(
        app, [*common, "--db", str(db), "--last", "2019-01-02"], catch_exceptions=False
    )
    assert past.exit_code == 2
    missing = CliRunner().invoke(
        app, [*common, "--db", str(tmp_path / "absent.db")], catch_exceptions=False
    )
    assert missing.exit_code == 2
    assert not artifacts.exists()
    assert not (tmp_path / "absent.db").exists()


# --- provenance ------------------------------------------------------------------------------


@RED
@pytest.mark.spec("ARCH§7.2")
def test_code_provenance_reads_the_commit_and_lockfile_from_files(tmp_path: Path) -> None:
    assert code_provenance(tmp_path) == {"commit": None, "lockfile_sha256": None}
    (tmp_path / "uv.lock").write_bytes(b"lock\n")
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/b06-data\n")
    loose, packed = "a" * 40, "b" * 40
    (git / "packed-refs").write_text(f"# pack-refs with: peeled\n{packed} refs/heads/b06-data\n")
    lock = hashlib.sha256(b"lock\n").hexdigest()
    assert code_provenance(tmp_path) == {"commit": packed, "lockfile_sha256": lock}
    (git / "refs" / "heads" / "b06-data").write_text(f"{loose}\n")
    assert code_provenance(tmp_path)["commit"] == loose
    (git / "HEAD").write_text(f"{'c' * 40}\n")  # detached
    assert code_provenance(tmp_path)["commit"] == "c" * 40
    (git / "HEAD").write_text("ref: refs/heads/elsewhere\n")
    assert code_provenance(tmp_path)["commit"] is None
