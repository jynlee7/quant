"""B6: the truncation harness reads minutes, corporate actions and flags, on two worlds.

B5 left two items for B6 (BUILD_LOG): garbage for dividends, splits and flags, with a layer that
reads the view's minutes and actions; and the full tier's T-NL1 on real FIT bars. The actions
layer is registered in `default_registry`, and each of its reads gets a planted leak the harness
must catch, naming the actions layer (P3). The leaks run the actions layer alone at one cut, so
the fast tier stays fast. The world is built from a spec, and a world read from a market.db file
must be the synthetic world itself when the file holds the synthetic bars.
"""

from __future__ import annotations

import dataclasses
import sqlite3
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

import quant.data.view as view_module
from quant.core.calendar import NEW_YORK
from quant.core.epoch import to_us
from quant.core.schedule import Decision
from quant.data.market_db import BarStore
from quant.data.policy import Purpose
from quant.data.validate import DataFlag
from quant.data.view import PointInTimeView
from quant.features.menu import r1_features
from quant.hmm.train import TrainingData
from quant.jobs.first_model import training_data
from tests.support.bars import SYMBOL
from tests.support.truncation import (
    ACTION_DAYS,
    FLAG_SPAN,
    HORIZON,
    MINUTE_SPAN,
    MODEL_FIT_END,
    SYNTHETIC,
    ActionsLayer,
    Registry,
    Report,
    Source,
    Variant,
    World,
    WorldSpec,
    build_world,
    check_truncation,
    default_registry,
    fixture_training_data,
    garbage_store,
    world,
)

SEED = 20160104


def _one_cut(w: World) -> tuple[datetime, ...]:
    """The model's first decision alone: late enough that the synthetic world's first dividend
    and its split are known, so the actions records aren't empty."""
    return (w.edges["model_first"].tau,)


def _assert_caught_in_actions(report: Report) -> None:
    assert not report.ok, "the planted leak was not caught"
    assert report.layers == {"actions"}, report.explain()
    assert {f.kind for f in report.failures} <= {"mismatch", "error"}, report.explain()
    assert {f.cut for f in report.failures} == set(report.cuts), report.explain()
    # Only the garbage shows the leak: the truncated store agrees with the full one.
    assert {f.variant for f in report.failures} == {Variant.GARBAGE}, report.explain()


# --- the layer -------------------------------------------------------------------------------


@pytest.mark.spec("I1", "T-NL1", "ARCH§6.2")
def test_the_actions_layer_is_registered_and_passes_clean() -> None:
    w = world()
    assert default_registry(w).names[-1] == "actions"
    report = check_truncation(w, Registry([ActionsLayer()]), _one_cut(w), seed=SEED)
    assert report.ok, report.explain()
    assert report.compared["actions"] > 0


@pytest.mark.spec("I1", "T-NL1")
def test_the_actions_layer_records_what_the_view_shows_at_tau() -> None:
    # The records hold real content, so a pass isn't the comparison of empty lists.
    w = world()
    decision = w.edges["model_first"]
    (record,) = ActionsLayer().run(Source(w.store, Variant.FULL, None), [decision], {})
    view = w.store.view(decision.tau, as_of=decision.tau)
    minutes = view.minutes(
        SYMBOL, decision.tau - MINUTE_SPAN, decision.tau + MINUTE_SPAN, purpose=Purpose.OPERATE
    )
    assert record.output["minutes"]["t"] == tuple(int(t) for t in minutes.t)
    assert len(minutes) > 0
    assert int(minutes.t[-1]) < to_us(decision.tau)
    first = decision.bar.session - ACTION_DAYS
    last = decision.bar.session + ACTION_DAYS
    assert record.output["dividends"] == (("2016-09-16", 1.0),)
    assert record.output["splits"] == (("2016-10-14", 2.0),)
    assert view.dividends(SYMBOL, first, last, purpose=Purpose.OPERATE) == [
        (date(2016, 9, 16), 1.0)
    ]
    flags = view.flags(
        SYMBOL, decision.tau - FLAG_SPAN, decision.tau + MINUTE_SPAN, purpose=Purpose.OPERATE
    )
    assert record.output["flags"] == tuple((to_us(f.t), f.kind, f.detail) for f in flags)


@pytest.mark.spec("I1", "T-NL1", "P3")
def test_the_world_holds_a_raised_flag_the_actions_layer_records() -> None:
    # Review M2/S1: with no flag in the world, every flags read compared empty lists.
    w = world()
    (record,) = ActionsLayer().run(
        Source(w.store, Variant.FULL, None), [w.edges["model_first"]], {}
    )
    assert record.output["flags"], "the actions layer recorded no flag at the model's first cut"
    assert w.conn.execute("SELECT COUNT(*) FROM data_flags WHERE raised = 1").fetchone()[0] > 0


# --- the garbage -----------------------------------------------------------------------------


@pytest.mark.spec("T-NL1", "P3")
def test_the_garbage_variant_writes_dividends_splits_and_flags() -> None:
    # Garbage no read can reach would make variant (c) vacuous for the actions layer.
    w = world()
    cut = _one_cut(w)[0]
    store, garbage = garbage_store(w, cut, seed=SEED)
    session = next(d for d in w.decisions if d.tau == cut).bar.session
    horizon = w.spec.horizon
    late = store.view(horizon, as_of=horizon)
    span = (session - ACTION_DAYS, session + ACTION_DAYS)
    dividends = dict(late.dividends(SYMBOL, *span, purpose=Purpose.OPERATE))
    splits = dict(late.splits(SYMBOL, *span, purpose=Purpose.OPERATE))
    assert dividends[garbage.dividends[0]] == 1e9
    assert np.isnan(dividends[garbage.dividends[1]])
    assert splits[garbage.splits[0]] == 1e9
    assert np.isnan(splits[garbage.splits[1]])
    # The synthetic dividend and split the cut can see are revised to -1e9, known at the cut.
    assert garbage.dividends[2] == date(2016, 9, 16)
    assert dividends[date(2016, 9, 16)] == -1e9
    assert garbage.splits[2] == date(2016, 10, 14)
    assert splits[date(2016, 10, 14)] == -1e9
    # A minute of the half hour before the cut is revised, known exactly at the cut.
    assert garbage.minute_revised_at_cut is not None
    assert cut - MINUTE_SPAN <= garbage.minute_revised_at_cut < cut
    (revised,) = store.versions(SYMBOL, garbage.minute_revised_at_cut)[-1:]
    assert (revised.known_at, revised.close) == (cut, 1e9)
    # The planted flag the cut sees is cleared, and its minute revised, after the cut.
    assert garbage.flag_cleared is not None
    assert garbage.flag_cleared not in {f.t for f in store.flags(SYMBOL)}
    assert garbage.flag_cleared in {f.t for f in store.flags(SYMBOL, as_of=cut)}
    assert store.versions(SYMBOL, garbage.flag_cleared)[-1].known_at > cut
    # The store's own flags, with no availability rule, show every garbage flag.
    standing = {f.t for f in store.flags(SYMBOL)}
    assert set(garbage.flags) <= standing
    assert len(garbage.flags) == 4
    # At the cut the view sees none of it.
    at_cut = store.view(cut, as_of=cut)
    assert dict(at_cut.dividends(SYMBOL, *span, purpose=Purpose.OPERATE)) == {
        date(2016, 9, 16): 1.0
    }
    assert at_cut.splits(SYMBOL, *span, purpose=Purpose.OPERATE) == [(date(2016, 10, 14), 2.0)]
    seen = {
        f.t
        for f in at_cut.flags(SYMBOL, cut - FLAG_SPAN, cut + MINUTE_SPAN, purpose=Purpose.OPERATE)
    }
    assert not seen & set(garbage.flags)
    assert garbage_store(w, cut, seed=SEED)[1] == garbage


# --- one planted leak for each read (P3) -----------------------------------------------------


@pytest.mark.spec("I1", "T-NL1", "P3", "ARCH§6.2")
def test_minutes_read_without_availability_are_caught_in_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    w = world()
    assert " AND available_at < :tau" in view_module._MINUTES
    monkeypatch.setattr(
        view_module, "_MINUTES", view_module._MINUTES.replace(" AND available_at < :tau", "")
    )
    report = check_truncation(w, Registry([ActionsLayer()]), _one_cut(w), seed=SEED)
    _assert_caught_in_actions(report)


@pytest.mark.spec("I1", "T-NL1", "P3", "ARCH§6.2")
@pytest.mark.parametrize("table", ["dividends", "splits"])
def test_actions_known_at_as_of_are_caught_in_actions(
    monkeypatch: pytest.MonkeyPatch, table: str
) -> None:
    # A read of known_at <= as_of sees the garbage action known exactly at the cut.
    w = world()
    sql = view_module._ACTIONS[table]
    assert sql.count("known_at < :as_of") == 2
    monkeypatch.setitem(
        view_module._ACTIONS, table, sql.replace("known_at < :as_of", "known_at <= :as_of")
    )
    report = check_truncation(w, Registry([ActionsLayer()]), _one_cut(w), seed=SEED)
    _assert_caught_in_actions(report)


@pytest.mark.spec("I1", "T-NL1", "P3", "ARCH§6.2")
def test_minutes_known_at_as_of_are_caught_in_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    # Review M1: the garbage minute known exactly at the cut lay outside the half hour read.
    w = world()
    sql = view_module._MINUTES
    assert sql.count("known_at < :as_of") == 2
    monkeypatch.setattr(
        view_module, "_MINUTES", sql.replace("known_at < :as_of", "known_at <= :as_of")
    )
    report = check_truncation(w, Registry([ActionsLayer()]), _one_cut(w), seed=SEED)
    _assert_caught_in_actions(report)


@pytest.mark.spec("I1", "T-NL1", "P3", "ARCH§6.2")
@pytest.mark.parametrize("table", ["dividends", "splits"])
def test_actions_with_an_unbounded_latest_version_are_caught_in_actions(
    monkeypatch: pytest.MonkeyPatch, table: str
) -> None:
    # Review S2/M3: a version known after the cut must not hide the one known at τ, at one cut.
    w = world()
    sql = view_module._ACTIONS[table]
    unbounded = sql.replace("\n                           AND x.known_at < :as_of)", ")")
    assert unbounded != sql
    monkeypatch.setitem(view_module._ACTIONS, table, unbounded)
    report = check_truncation(w, Registry([ActionsLayer()]), _one_cut(w), seed=SEED)
    _assert_caught_in_actions(report)


# The flags query's two latest-version subqueries, each with its known_at bound (review S1/M2).
_FLAG_VERSION = (
    "WHERE x.symbol = f.symbol AND x.t = f.t AND x.kind = f.kind\n"
    "                         AND x.known_at < :as_of)"
)
_FLAG_MINUTE_VERSION = (
    "WHERE x.symbol = m.symbol AND x.t = m.t\n"
    "                                       AND x.known_at < :as_of)"
)


@pytest.mark.spec("I1", "T-NL1", "P3", "ARCH§6.2")
@pytest.mark.parametrize("subquery", [_FLAG_VERSION, _FLAG_MINUTE_VERSION], ids=["flag", "minute"])
def test_flags_read_with_an_unbounded_latest_version_are_caught_in_actions(
    monkeypatch: pytest.MonkeyPatch, subquery: str
) -> None:
    # A clear known after the cut must not hide a flag raised at τ, nor a revision of its minute
    # known after the cut hide the minute (review S1/M2).
    w = world()
    sql = view_module._FLAGS
    assert sql.count(subquery) == 1
    monkeypatch.setattr(view_module, "_FLAGS", sql.replace(subquery, subquery.split("\n")[0] + ")"))
    report = check_truncation(w, Registry([ActionsLayer()]), _one_cut(w), seed=SEED)
    _assert_caught_in_actions(report)


@pytest.mark.spec("I1", "T-NL1", "P3", "ARCH§6.2")
def test_flags_known_at_as_of_are_caught_in_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    w = world()
    sql = view_module._FLAGS
    assert sql.count("known_at < :as_of") == 4
    monkeypatch.setattr(
        view_module, "_FLAGS", sql.replace("known_at < :as_of", "known_at <= :as_of")
    )
    report = check_truncation(w, Registry([ActionsLayer()]), _one_cut(w), seed=SEED)
    _assert_caught_in_actions(report)


@pytest.mark.spec("T-NL1", "P3")
def test_the_garbage_reads_what_the_cut_sees_without_the_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Review M6: the garbage chooses what to revise from raw rows, never through the view (the
    # code under test), so a broken view can't steer its own garbage away from the leak.
    w = world()
    cut = _one_cut(w)[0]
    _, expected = garbage_store(w, cut, seed=SEED)

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the garbage read actions or flags through the view")

    for name in ("dividends", "splits", "flags", "minutes"):
        monkeypatch.setattr(PointInTimeView, name, refuse)
    store, garbage = garbage_store(w, cut, seed=SEED)
    store.close()
    assert garbage == expected


class _StoreFlags(ActionsLayer):
    """Planted leak: the view's flags joined with `BarStore.flags(as_of=τ)`, which has no
    availability rule, so a flag on a minute that isn't available yet gets through."""

    def flags_of(self, source: Source, view: PointInTimeView, d: Decision) -> list[DataFlag]:
        start, end = d.tau - FLAG_SPAN, d.tau + MINUTE_SPAN
        own = view.flags(SYMBOL, start, end, purpose=Purpose.OPERATE)
        stored = [f for f in source.store.flags(SYMBOL, as_of=d.tau) if start <= f.t < end]
        return sorted({*own, *stored}, key=lambda f: (f.t, f.kind))


@pytest.mark.spec("I1", "T-NL1", "P3", "ARCH§6.2")
def test_flags_read_without_availability_are_caught_in_actions() -> None:
    w = world()
    report = check_truncation(w, Registry([_StoreFlags()]), _one_cut(w), seed=SEED)
    _assert_caught_in_actions(report)


# --- worlds ----------------------------------------------------------------------------------


def _all_rows(conn: sqlite3.Connection, table: str) -> Sequence[tuple[object, ...]]:
    """Every row of `table`, in key order; a minute's `source` (which names where the minutes
    were loaded from) left out."""
    columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    kept = ", ".join(c for c in columns if c != "source")
    return conn.execute(f"SELECT {kept} FROM {table} ORDER BY 1, 2, 3, 4").fetchall()  # noqa: S608


@pytest.mark.spec("T-NL1", "P4", "ARCH§13")
def test_a_world_read_from_a_market_db_file_is_the_synthetic_world(tmp_path: Path) -> None:
    # The full tier's real world is built by the same code from var/market.db. Built from a file
    # that holds the synthetic bars, it must be the synthetic world: same store, decisions, edges,
    # scaler and model parameters (the model's provenance names its source, so not its hash).
    w = world()
    path = tmp_path / "market.db"
    target = sqlite3.connect(path)
    w.conn.backup(target)
    target.close()
    spec = dataclasses.replace(SYNTHETIC, name="from a file", market_db=path)
    built = build_world(spec)
    try:
        for table in ("minute_bars", "hourly_bars", "dividends", "splits", "data_flags"):
            assert _all_rows(built.conn, table) == _all_rows(w.conn, table), table
        assert built.decisions == w.decisions
        assert dict(built.edges) == dict(w.edges)
        assert built.activation == w.activation
        assert np.array_equal(built.scaler.mean, w.scaler.mean)
        assert np.array_equal(built.scaler.std, w.scaler.std)
        assert (built.scaler.start, built.scaler.end) == (w.scaler.start, w.scaler.end)
        for name, array in w.model.arrays().items():
            assert np.array_equal(built.model.arrays()[name], array), name
        assert built.model.labels == w.model.labels
        assert (built.model.fit_start, built.model.fit_end) == (w.model.fit_start, w.model.fit_end)
    finally:
        built.store.close()


@pytest.mark.spec("T-NL1", "P4")
def test_a_market_db_missing_sessions_is_refused_with_its_reason(tmp_path: Path) -> None:
    w = world()
    path = tmp_path / "market.db"
    target = sqlite3.connect(path)
    w.conn.backup(target)
    day = date(2016, 7, 6)
    start = datetime(2016, 7, 6, 13, 30, tzinfo=UTC)  # 09:30 in New York
    target.execute(
        "DELETE FROM minute_bars WHERE t >= ? AND t < ?",
        (to_us(start), to_us(start + timedelta(hours=7))),
    )
    target.commit()
    target.close()
    spec = dataclasses.replace(SYNTHETIC, name="partial", market_db=path)
    with pytest.raises(ValueError, match=f"lacks 1 of .* {day}.*backfill"):
        build_world(spec)


def _world_file(tmp_path: Path, name: str) -> tuple[Path, WorldSpec]:
    """A market.db file holding the synthetic world, and a spec that builds a world from it."""
    path = tmp_path / f"{name}.db"
    target = sqlite3.connect(path)
    world().conn.backup(target)
    target.close()
    return path, dataclasses.replace(SYNTHETIC, name=name, market_db=path)


def _training_rows(data: TrainingData) -> tuple[object, ...]:
    m = data.matrix
    return (m.t_open.tolist(), m.available_at.tolist(), m.values.tolist(), data.returns.tolist())


@pytest.mark.spec("I1", "T-NL1", "T-NL2", "P4")
def test_a_world_files_fixture_model_reads_what_was_known_at_its_fit_end(tmp_path: Path) -> None:
    # Review S5: a revision known after fit_end must not reach the fixture model or its scaler.
    path, spec = _world_file(tmp_path, "revised later")
    conn = sqlite3.connect(path)
    t_open = to_us(datetime(2016, 10, 3, 14, 30, tzinfo=UTC))
    (version,) = conn.execute(
        "SELECT MAX(version) FROM hourly_bars WHERE t_open = ?", (t_open,)
    ).fetchone()
    conn.execute(
        "INSERT INTO hourly_bars SELECT symbol, t_open, ?, session, slot, t_close, open * 1.01,"
        " high * 1.01, low * 1.01, close * 1.01, volume, minutes, complete, available_at, ?"
        " FROM hourly_bars"
        " WHERE t_open = ? AND version = ?",
        (version + 1, to_us(MODEL_FIT_END + timedelta(days=30)), t_open, version),
    )
    conn.commit()
    conn.close()
    assert _training_rows(fixture_training_data(MODEL_FIT_END, spec)) == _training_rows(
        fixture_training_data(MODEL_FIT_END)
    )


@pytest.mark.spec("I1", "T-NL1", "D4")
def test_a_world_files_fixture_rows_are_the_first_models_rows(tmp_path: Path) -> None:
    # Review S6: the fixture takes each return from the previous complete bar, as the first
    # model's training_data does, never from an incomplete one.
    path, spec = _world_file(tmp_path, "incomplete bar")
    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE hourly_bars SET complete = 0, minutes = 59, close = close * 1.01 WHERE t_open = ?",
        (to_us(datetime(2016, 10, 3, 14, 30, tzinfo=UTC)),),
    )
    conn.commit()
    conn.close()
    fixture = fixture_training_data(MODEL_FIT_END, spec)
    reader = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        last = (MODEL_FIT_END - timedelta(microseconds=1)).astimezone(NEW_YORK).date()
        rows = training_data(
            BarStore(reader), SYMBOL, first=spec.model_first, last=last, features=r1_features()
        )
    finally:
        reader.close()
    assert _training_rows(fixture) == _training_rows(rows)


@pytest.mark.spec("T-NL1", "P4")
def test_a_market_db_holding_revisions_in_the_span_is_refused(tmp_path: Path) -> None:
    # Review S5: the world store reloads one version of each minute as history, so a file with
    # revisions in the span would be flattened; it is refused with its reason instead.
    path, spec = _world_file(tmp_path, "revised minute")
    conn = sqlite3.connect(path)
    t = to_us(datetime(2016, 7, 6, 15, 0, tzinfo=UTC))
    conn.execute(
        "INSERT INTO minute_bars SELECT symbol, t, version + 1, open, high, low, close + 0.01,"
        " volume, available_at, known_at + 3600000000, 'settle' FROM minute_bars WHERE t = ?",
        (t,),
    )
    conn.commit()
    conn.close()
    try:
        build_world(spec)
    except ValueError as exc:
        message = str(exc)
    else:
        message = ""
    assert "revision" in message, "a market.db with revisions in the span was not refused"


@pytest.mark.spec("T-NL1")
def test_a_built_world_has_its_own_store() -> None:
    # Review S10: build_world is uncached, so closing its store must not close world()'s.
    built = build_world(SYNTHETIC)
    assert built.conn is not world().conn
    built.store.close()
    assert world().store.view(HORIZON, as_of=HORIZON).hourly(SYMBOL, purpose=Purpose.OPERATE)
