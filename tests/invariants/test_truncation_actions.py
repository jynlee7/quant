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
from quant.core.epoch import to_us
from quant.core.schedule import Decision
from quant.data.policy import Purpose
from quant.data.validate import DataFlag
from quant.data.view import PointInTimeView
from tests.support.bars import SYMBOL
from tests.support.truncation import (
    ACTION_DAYS,
    FLAG_SPAN,
    MINUTE_SPAN,
    SYNTHETIC,
    ActionsLayer,
    Registry,
    Report,
    Source,
    Variant,
    World,
    build_world,
    check_truncation,
    default_registry,
    garbage_store,
    world,
)

SEED = 20160104
RED = pytest.mark.xfail(strict=True, reason="red:B6", raises=NotImplementedError)


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


@RED
@pytest.mark.spec("I1", "T-NL1", "ARCH§6.2")
def test_the_actions_layer_is_registered_and_passes_clean() -> None:
    w = world()
    assert default_registry(w).names[-1] == "actions"
    report = check_truncation(w, Registry([ActionsLayer()]), _one_cut(w), seed=SEED)
    assert report.ok, report.explain()
    assert report.compared["actions"] > 0


@RED
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


# --- the garbage -----------------------------------------------------------------------------


@RED
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
    # The synthetic dividend the cut can see is revised to -1e9, known at the cut.
    assert garbage.dividends[2] == date(2016, 9, 16)
    assert dividends[date(2016, 9, 16)] == -1e9
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


@RED
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


@RED
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


class _StoreFlags(ActionsLayer):
    """Planted leak: the view's flags joined with `BarStore.flags(as_of=τ)`, which has no
    availability rule, so a flag on a minute that isn't available yet gets through."""

    def flags_of(self, source: Source, view: PointInTimeView, d: Decision) -> list[DataFlag]:
        start, end = d.tau - FLAG_SPAN, d.tau + MINUTE_SPAN
        own = view.flags(SYMBOL, start, end, purpose=Purpose.OPERATE)
        stored = [f for f in source.store.flags(SYMBOL, as_of=d.tau) if start <= f.t < end]
        return sorted({*own, *stored}, key=lambda f: (f.t, f.kind))


@RED
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
