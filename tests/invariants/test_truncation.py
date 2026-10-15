"""B5: the T-NL1 truncation harness and T-NL2 provenance (SPEC §6, I1, I2, ARCHITECTURE §13).

Clean code must pass all three variants at every cut, and each of PLAN B5's six planted leaks
must be caught, with the failure naming the layer the leak is in (P3: a harness that can't fail
is caught). Leaks are planted with monkeypatch or a substitute layer registered in place of the
clean one, never by editing src/. The fast tier draws 50 cuts on synthetic bars; the full tier
draws 200 on real FIT bars from var/market.db (B6, PLAN P4), and skips when it is absent. B6's
actions layer, last in the default registry, is tested in test_truncation_actions.py.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import sqlite3
import statistics
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

import quant.data.view as view_module
import tests.support.truncation as truncation_module
from quant.core.canonical import sha256_hex
from quant.core.epoch import from_us, to_us
from quant.core.errors import FeatureError
from quant.core.schedule import Decision
from quant.data.frames import HourlyBar
from quant.data.market_db import HISTORY, BarStore
from quant.data.policy import Purpose
from quant.data.view import PointInTimeView
from quant.eval.smoother import smooth
from quant.features.scaler import Scaler
from quant.features.stream import FeatureState, FeatureStream, compute_features, features_for
from quant.hmm.model import HmmModel
from tests.support.bars import SYMBOL
from tests.support.truncation import (
    ACTIVATION,
    EDGES,
    HORIZON,
    LATE_FIT_END,
    MODEL_FIT_END,
    REAL,
    VIEW_DEPTH,
    FeaturesLayer,
    FilterLayer,
    Provenance,
    Record,
    Registry,
    Report,
    Source,
    Variant,
    World,
    _feature_state_hash,
    check_truncation,
    default_registry,
    draw_cuts,
    error_output,
    features_output,
    fixture_model,
    garbage_store,
    provenance_failures,
    run_layers,
    truncated_store,
    world,
)

SEED = 20160104
FAST_CUTS = 50  # PLAN P4
FULL_CUTS = 200
LAYERS = {"view", "features", "filter", "risk", "actions"}
# A planted leak shows at any cut after the model starts, so three edges are enough and keep the
# six leak tests cheap: the model's first decision, the bar after DST ends, a half day's CLOSE.
LEAK_EDGES = ("model_first", "after_dst", "half_day_close")
CMA_HALF_WIDTH = 3  # bars on each side of a centered moving average


def _leak_cuts(w: World) -> tuple[datetime, ...]:
    return tuple(w.edges[name].tau for name in LEAK_EDGES)


MISMATCH = frozenset({"mismatch"})
# A layer that peeks past τ in variant (c) reads the garbage, and may raise on it rather than
# record something different. Either way the failure is the layer's.
MISMATCH_OR_ERROR = frozenset({"mismatch", "error"})


def _assert_caught(report: Report, layer: str, kinds: frozenset[str] = MISMATCH) -> None:
    """The harness failed at every cut, and every failure names `layer`, of one of `kinds`."""
    assert not report.ok, "the planted leak was not caught"
    assert report.layers == {layer}, report.explain()
    assert {f.kind for f in report.failures} <= kinds, report.explain()
    assert {f.cut for f in report.failures} == set(report.cuts), report.explain()
    # The comparison reached the leak's layer: every layer before it (in registry order) compared
    # records, and so did the leak's own layer unless it only ever raised. Layers after it may
    # have nothing to compare when the leak shows at the first decision.
    names = list(report.compared)
    raised_only = {f.kind for f in report.failures} == {"error"}
    reached = names[: names.index(layer) + (0 if raised_only else 1)]
    assert all(report.compared[name] > 0 for name in reached), report.compared


def _where(report: Report) -> set[tuple[datetime, Variant, str, datetime | None, str]]:
    return {(f.cut, f.variant, f.layer, f.tau, f.kind) for f in report.failures}


# --- clean code ------------------------------------------------------------------------------


@pytest.mark.spec("I1", "T-NL1", "T-NL2", "P4", "ARCH§13", "ARCH§4.2")
def test_clean_code_passes_every_variant_at_fifty_cuts() -> None:
    w = world()
    cuts = draw_cuts(w, FAST_CUTS, seed=SEED)
    assert len(cuts) == FAST_CUTS
    assert {w.edges[name].tau for name in EDGES} <= set(cuts)
    report = check_truncation(w, default_registry(w), cuts, seed=SEED)
    assert report.ok, report.explain()
    assert report.cuts == cuts
    # Every layer was compared, the filter, risk verdicts and actions included, so a pass means
    # something.
    assert set(report.compared) == LAYERS
    assert all(count > 0 for count in report.compared.values()), report.compared


@pytest.mark.full
@pytest.mark.spec("I1", "T-NL1", "T-NL2", "P4", "ARCH§13")
def test_clean_code_passes_every_variant_at_two_hundred_cuts_on_real_fit_bars() -> None:
    # PLAN P4: the full tier runs 200 cuts on real bars, with a fixture model fit on 2016–2017.
    if REAL.market_db is None or not REAL.market_db.is_file():
        pytest.skip(
            f"no market.db at {REAL.market_db}: the full tier's T-NL1 runs on real FIT bars "
            "(PLAN P4); fill it with `quant data backfill`"
        )
    w = world(REAL)
    # Inside FIT only: DEV stays dark (C11). The model was fit on 2016–2017, before every
    # decision it serves (T-NL2).
    assert w.decisions[-1].bar.session <= REAL.data_last < date(2019, 1, 1)
    assert w.model.fit_start.year == 2016
    assert w.model.fit_end == REAL.model_fit_end == datetime(2018, 1, 1, tzinfo=UTC)
    assert w.decisions[0].tau < REAL.model_fit_end < w.edges["model_first"].tau
    cuts = draw_cuts(w, FULL_CUTS, seed=SEED + 1)
    report = check_truncation(w, default_registry(w), cuts, seed=SEED + 1)
    assert report.ok, report.explain()
    assert len(report.cuts) == FULL_CUTS
    assert set(report.compared) == LAYERS
    assert all(count > 0 for count in report.compared.values()), report.compared


@pytest.mark.spec("T-NL1", "P4")
def test_cuts_are_drawn_from_a_seed_and_include_the_edges() -> None:
    w = world()
    first = draw_cuts(w, FAST_CUTS, seed=SEED)
    assert draw_cuts(w, FAST_CUTS, seed=SEED) == first
    assert draw_cuts(w, FAST_CUTS, seed=SEED + 7) != first
    assert list(first) == sorted(set(first))
    taus = {d.tau for d in w.decisions}
    assert set(first) <= taus
    edges = w.edges
    assert set(edges) == set(EDGES)
    assert edges["first_warm"] == w.decisions[0]
    assert edges["model_first"].tau > ACTIVATION
    assert all(d.tau <= ACTIVATION for d in w.decisions if d.tau < edges["model_first"].tau)
    assert edges["half_day_close"].reduce_only
    assert edges["after_dst"].bar.index == 0
    assert edges["session_first"].bar.index == 0
    with pytest.raises(ValueError, match="cut"):
        draw_cuts(w, len(EDGES) - 1, seed=SEED)


# --- the variants' stores --------------------------------------------------------------------


@pytest.mark.spec("T-NL1")
def test_the_truncated_store_holds_only_what_was_known_by_the_cut() -> None:
    w = world()
    cut = w.edges["after_dst"].tau
    store = truncated_store(w, cut)
    bars = store.view(HORIZON, as_of=HORIZON).hourly(SYMBOL, purpose=Purpose.OPERATE)
    full = w.store.view(HORIZON, as_of=HORIZON).hourly(SYMBOL, purpose=Purpose.OPERATE)
    assert bars
    assert all(bar.known_at is not None and bar.known_at <= cut for bar in bars)
    assert bars == [bar for bar in full if bar.known_at is not None and bar.known_at <= cut]
    assert bars[-1].t_open == w.edges["after_dst"].bar.t_open


@pytest.mark.spec("T-NL1", "P3")
def test_the_garbage_variant_really_writes_every_kind_of_garbage() -> None:
    # Garbage the view can't see because it was never written would make variant (c) vacuous.
    w = world()
    cut = w.edges["half_day_close"].tau
    store, garbage = garbage_store(w, cut, seed=SEED)
    peek = store.view(HORIZON, as_of=HORIZON).hourly(SYMBOL, purpose=Purpose.OPERATE)
    by_open = {bar.t_open: bar for bar in peek}
    assert isinstance(by_open[garbage.nan].close, str)  # SQLite keeps the NaN as text
    assert by_open[garbage.huge].high == 1e9
    assert by_open[garbage.negative].close == -1e9
    assert len(garbage.shuffled) >= 10
    unavailable = [by_open[t] for t in garbage.unavailable]
    assert all(bar.t_close < cut <= bar.available_at for bar in unavailable)
    assert any(bar.available_at == cut for bar in unavailable)
    assert all(bar.known_at is not None and bar.known_at < cut for bar in unavailable)
    assert garbage.revised == w.edges["half_day_close"].bar.t_open
    assert garbage.revision_known_at > cut
    revised = by_open[garbage.revised]
    assert revised.version == 2
    assert revised.known_at == garbage.revision_known_at
    # A revision known exactly at the cut, of the bar before the cut's (review finding 3).
    at_the_cut = by_open[garbage.revised_at_cut]
    assert at_the_cut.version == 2
    assert at_the_cut.known_at == cut
    at_cut = store.view(cut, as_of=cut).hourly(SYMBOL, purpose=Purpose.OPERATE)
    assert at_cut[-1].t_open == garbage.revised
    assert at_cut[-1].version == 1
    assert at_cut[-2].t_open == garbage.revised_at_cut
    assert at_cut[-2].version == 1
    # Minutes before the cut that aren't available yet, and minutes after it (finding 7).
    span = (cut - timedelta(minutes=1), cut + timedelta(minutes=10))
    minutes = store.view(HORIZON, as_of=HORIZON).minutes(SYMBOL, *span, purpose=Purpose.OPERATE)
    seen = {from_us(int(t)) for t in minutes.t}
    assert set(garbage.minutes_unavailable) | set(garbage.minutes_after) <= seen
    assert len(garbage.minutes_after) == 5
    assert all(t < cut for t in garbage.minutes_unavailable)
    assert all(t >= cut for t in garbage.minutes_after)
    close = {from_us(int(t)): c for t, c in zip(minutes.t, minutes.close, strict=True)}
    assert math.isnan(close[garbage.minutes_after[0]])
    assert close[garbage.minutes_after[1]] == 1e9
    assert close[garbage.minutes_after[2]] == -1e9
    at_cut_minutes = store.view(cut, as_of=cut).minutes(SYMBOL, *span, purpose=Purpose.OPERATE)
    visible = {from_us(int(t)) for t in at_cut_minutes.t}
    assert not visible & (set(garbage.minutes_unavailable) | set(garbage.minutes_after))
    # The same seed writes the same garbage.
    assert garbage_store(w, cut, seed=SEED)[1] == garbage


# --- T-NL2 -----------------------------------------------------------------------------------


@pytest.mark.spec("T-NL2")
def test_provenance_needs_the_fit_window_to_end_strictly_before_tau() -> None:
    tau = datetime(2016, 11, 1, 14, 46, tzinfo=MODEL_FIT_END.tzinfo)
    good = Record(tau, "filter", {}, Provenance("ab" * 32, tau - timedelta(microseconds=1)))
    equal = Record(tau, "filter", {}, Provenance("ab" * 32, tau))
    late = Record(tau, "filter", {}, Provenance("ab" * 32, tau + timedelta(days=1)))
    bare = Record(tau, "view", {})
    assert provenance_failures([good, bare], cut=tau, variant=Variant.FULL) == []
    for record in (equal, late):
        (failure,) = provenance_failures([bare, record], cut=tau, variant=Variant.TRUNCATED)
        assert (failure.kind, failure.layer, failure.tau) == ("provenance", "filter", tau)


# --- the six planted leaks (P3) --------------------------------------------------------------


class CenteredAverage:
    """Planted leak 1: F5 replaced by ln(close / a centered moving average of closes), computed
    over every bar in the store. The average at bar t reads bars after t."""

    name = "features"

    def __init__(self, w: World) -> None:
        self._world = w

    def run(
        self,
        source: Source,
        decisions: Sequence[Decision],
        upstream: Mapping[str, Sequence[Record]],
    ) -> list[Record]:
        w = self._world
        everything = source.view(HORIZON).hourly(SYMBOL, purpose=Purpose.OPERATE)
        complete = [bar for bar in everything if bar.complete]
        where = {bar.t_open: i for i, bar in enumerate(complete)}
        trend = w.config.columns.index("trend")
        stream = FeatureStream(w.config)
        records = []
        for d in decisions:
            output: object
            try:
                raw = features_for(stream, source.view(d.tau), SYMBOL, d.bar).copy()
                i = where[d.bar.t_open]
                window = complete[max(0, i - CMA_HALF_WIDTH) : i + CMA_HALF_WIDTH + 1]
                raw[trend] = math.log(complete[i].close / statistics.fmean(b.close for b in window))
                output = features_output(raw, w.scaler.transform(raw, columns=w.config.columns))
            except FeatureError as exc:
                output = error_output(exc)
            records.append(Record(d.tau, self.name, output))
        return records


class FullHistoryScaler:
    """Planted leak 2: the scaler fit on every warm row in the store, not the model's window."""

    name = "features"

    def __init__(self, w: World) -> None:
        self._world = w

    def run(
        self,
        source: Source,
        decisions: Sequence[Decision],
        upstream: Mapping[str, Sequence[Record]],
    ) -> list[Record]:
        w = self._world
        everything = source.view(HORIZON).hourly(SYMBOL, purpose=Purpose.OPERATE)
        matrix = compute_features(everything, w.config)
        first = from_us(int(matrix.available_at[0]))
        scaler = Scaler.fit(matrix, start=first, end=HORIZON)
        stream = FeatureStream(w.config)
        records = []
        for d in decisions:
            output: object
            try:
                raw = features_for(stream, source.view(d.tau), SYMBOL, d.bar)
                output = features_output(raw, scaler.transform(raw, columns=w.config.columns))
            except FeatureError as exc:
                output = error_output(exc)
            records.append(Record(d.tau, self.name, output))
        return records


class Smoothed:
    """Planted leak 3 (I2): the filter swapped for eval.smoother's forward-backward posteriors
    over every row of the run, later rows included."""

    name = "filter"

    def __init__(self, model: HmmModel) -> None:
        self._model = model

    def run(
        self,
        source: Source,
        decisions: Sequence[Decision],
        upstream: Mapping[str, Sequence[Record]],
    ) -> list[Record]:
        model = self._model
        provenance = Provenance(model.artifact_hash(), model.fit_end)
        rows, taus = [], []
        for d, record in zip(decisions, upstream["features"], strict=True):
            if d.tau >= ACTIVATION and "raw" in record.output:
                rows.append(record.output["raw"])
                taus.append(d.tau)
        posteriors = (
            self.posteriors(np.array(rows, dtype=np.float64))
            if rows
            else np.empty((0, model.params.n_states))
        )
        smoothed = dict(zip(taus, posteriors.tolist(), strict=True))
        records = []
        for d in decisions:
            if d.tau in smoothed:
                records.append(
                    Record(d.tau, self.name, {"state_probs": smoothed[d.tau]}, provenance)
                )
            else:
                records.append(Record(d.tau, self.name, {"model": None}, cold=True))
        return records

    def posteriors(self, rows: np.ndarray) -> np.ndarray:
        model = self._model
        return smooth(model.params, model.scaler, rows, columns=model.scaler.columns)


class PredictProba(Smoothed):
    """Planted leak 3, as PLAN B5 words it (I2): the filter swapped for hmmlearn's
    `predict_proba`, run with the fixture model's parameters on its standardized rows. Tests
    may call hmmlearn's smoother; the I2 scan keeps it out of src/."""

    def posteriors(self, rows: np.ndarray) -> np.ndarray:
        from hmmlearn.hmm import GaussianHMM

        model, params = self._model, self._model.params
        hmm = GaussianHMM(n_components=params.n_states, covariance_type="full")
        hmm.n_features = params.n_features
        hmm.startprob_ = params.startprob.copy()
        hmm.transmat_ = params.transmat.copy()
        hmm.means_ = params.means.copy()
        hmm.covars_ = params.covars.copy()  # always full, diagonal for a diag model
        z = model.scaler.standardize(rows, columns=model.scaler.columns)
        return np.asarray(hmm.predict_proba(z), dtype=np.float64)


def _patch_view(monkeypatch: pytest.MonkeyPatch, old: str, new: str) -> None:
    """Plant a leak in the view's SQL: every hourly query with `old` rewritten to `new`."""
    for name in ("_HOURLY_LATEST_N", "_HOURLY_SINCE"):
        sql = getattr(view_module, name)
        assert old in sql, name
        monkeypatch.setattr(view_module, name, sql.replace(old, new))


@pytest.mark.spec("I1", "T-NL1", "P3", "SPEC§3")
def test_planted_leak_1_a_centered_moving_average_is_caught_in_features() -> None:
    w = world()
    registry = default_registry(w).replace(CenteredAverage(w))
    report = check_truncation(w, registry, _leak_cuts(w), seed=SEED)
    _assert_caught(report, "features", MISMATCH_OR_ERROR)


@pytest.mark.spec("I1", "T-NL1", "P3", "SPEC§3")
def test_planted_leak_2_a_scaler_fit_on_the_full_history_is_caught_in_features() -> None:
    w = world()
    registry = default_registry(w).replace(FullHistoryScaler(w))
    report = check_truncation(w, registry, _leak_cuts(w), seed=SEED)
    _assert_caught(report, "features", MISMATCH_OR_ERROR)


@pytest.mark.spec("I1", "I2", "T-NL1", "P3")
def test_planted_leak_3_a_smoother_in_place_of_the_filter_is_caught_in_filter() -> None:
    w = world()
    registry = default_registry(w).replace(Smoothed(w.model))
    _assert_caught(check_truncation(w, registry, _leak_cuts(w), seed=SEED), "filter")


@pytest.mark.spec("I1", "I2", "T-NL1", "P3")
def test_planted_leak_3_predict_proba_for_the_filter_is_caught_in_filter() -> None:
    # PLAN B5's own wording of leak 3 (review finding 8). One cut: the eval.smoother variant
    # above already runs the three edges.
    w = world()
    registry = _upto(w, "filter").replace(PredictProba(w.model))
    _assert_caught(check_truncation(w, registry, _leak_cuts(w)[:1], seed=SEED), "filter")


@pytest.mark.spec("I1", "T-NL1", "P3", "ARCH§6.2")
def test_planted_leak_4_a_view_using_at_or_before_tau_is_caught_in_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    w = world()
    _patch_view(monkeypatch, "available_at < :tau", "available_at <= :tau")
    report = check_truncation(w, default_registry(w), _leak_cuts(w), seed=SEED)
    _assert_caught(report, "view")
    assert {f.variant for f in report.failures} == {Variant.GARBAGE}


@pytest.mark.spec("I1", "T-NL1", "P3", "ARCH§6.2")
def test_planted_leak_5_availability_from_t_close_is_caught_in_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    w = world()
    _patch_view(monkeypatch, "available_at < :tau", "t_close < :tau")
    report = check_truncation(w, default_registry(w), _leak_cuts(w), seed=SEED)
    _assert_caught(report, "view")
    assert {f.variant for f in report.failures} == {Variant.GARBAGE}


@pytest.mark.spec("T-NL2", "P3")
def test_planted_leak_6_a_model_fit_past_tau_is_caught_by_provenance() -> None:
    w = world()
    late = fixture_model(LATE_FIT_END)
    assert late.fit_end == LATE_FIT_END > ACTIVATION
    registry = default_registry(w).replace(FilterLayer(late, activation=ACTIVATION))
    report = check_truncation(w, registry, _leak_cuts(w), seed=SEED)
    _assert_caught(report, "filter", frozenset({"provenance"}))
    # Every variant used the late model, so each one's records break T-NL2.
    assert {f.variant for f in report.failures} == set(Variant)


@pytest.mark.spec("T-NL1", "P3")
def test_slicing_the_full_run_hides_no_leak() -> None:
    # The full run is computed once over the whole history and sliced per cut. Recomputing it
    # afresh at each cut must find exactly the same failures for planted leak 1, so sharing one
    # run (and the layers' record caches) across cuts can't be what makes a run pass.
    w = world()
    registry = default_registry(w).replace(CenteredAverage(w))
    cuts = _leak_cuts(w)[:2]  # each fresh cut reruns the whole history
    sliced = check_truncation(w, registry, cuts, seed=SEED, slice_full=True)
    fresh = check_truncation(w, registry, cuts, seed=SEED, slice_full=False)
    _assert_caught(sliced, "features", MISMATCH_OR_ERROR)
    _assert_caught(fresh, "features", MISMATCH_OR_ERROR)
    assert _where(sliced) == _where(fresh)
    # The leak shows against the full run itself, not only through the garbage.
    assert Variant.TRUNCATED in {f.variant for f in sliced.failures}


# --- review round 1 (B5): each test plants a defect the first harness let through -------------


def _first_cut(w: World) -> tuple[datetime, ...]:
    """The first decision alone: runs (a) and (c) are one decision long, so a test whose defect
    shows from the first decision on costs little more than the full run."""
    return (w.edges["first_warm"].tau,)


def _upto(w: World, last: str) -> Registry:
    """The clean registry's layers up to and including `last`: a leak in or before `last` is
    caught there, and the layers after it would only make the full run slower."""
    layers = default_registry(w).layers
    return Registry(layers[: [layer.name for layer in layers].index(last) + 1])


def _full_history_scaler(w: World) -> Scaler:
    everything = w.store.view(HORIZON, as_of=HORIZON).hourly(SYMBOL, purpose=Purpose.OPERATE)
    matrix = compute_features(everything, w.config)
    return Scaler.fit(matrix, start=from_us(int(matrix.available_at[0])), end=HORIZON)


@pytest.mark.spec("T-NL2", "I1", "P3")
def test_review_1_a_scaler_fit_after_tau_is_caught_by_provenance() -> None:
    # Finding 1/5: a full-history scaler built once, outside the run, is the same object in every
    # variant, so only T-NL2 can see it: the scaled records must name the scaler's window.
    w = world()
    registry = _upto(w, "features").replace(FeaturesLayer(w.config, _full_history_scaler(w)))
    report = check_truncation(w, registry, _leak_cuts(w)[:1], seed=SEED)
    _assert_caught(report, "features", frozenset({"provenance"}))
    assert {f.variant for f in report.failures} == set(Variant)


@pytest.mark.spec("T-NL2", "I1")
def test_review_1_clean_features_scale_only_with_a_scaler_fit_before_tau() -> None:
    w = world()
    # The clean scaler is the fixture model's own, fit on the model's window.
    assert w.scaler.end == w.model.fit_end == MODEL_FIT_END
    assert np.array_equal(w.scaler.mean, w.model.scaler.mean)
    assert np.array_equal(w.scaler.std, w.model.scaler.std)
    run = run_layers(_upto(w, "features"), Source(w.store, Variant.FULL, None), w.decisions)
    scaled = [r for r in run.records["features"] if "scaled" in r.output]
    assert scaled
    for record in scaled:
        assert record.provenance is not None, record.tau
        assert record.provenance.fit_end < record.tau


class _Boom:
    """A filter that peeks at the whole store, then raises the same error in every run."""

    name = "filter"

    def run(
        self,
        source: Source,
        decisions: Sequence[Decision],
        upstream: Mapping[str, Sequence[Record]],
    ) -> list[Record]:
        source.view(HORIZON).hourly(SYMBOL, purpose=Purpose.OPERATE)
        raise RuntimeError("shape mismatch")


@pytest.mark.spec("T-NL1", "P3")
def test_review_2_a_layer_that_raises_the_same_error_in_every_run_fails() -> None:
    # Finding 2: equal errors counted as agreement, and the layers after it compared nothing.
    w = world()
    cuts = _first_cut(w)
    report = check_truncation(w, _upto(w, "filter").replace(_Boom()), cuts, seed=SEED)
    assert not report.ok, report.compared
    assert "filter" in report.layers, report.explain()
    assert "error" in {f.kind for f in report.failures if f.layer == "filter"}


@pytest.mark.spec("I1", "T-NL1", "P3", "ARCH§6.2")
def test_review_3_a_view_knowing_rows_at_as_of_is_caught_in_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Finding 3: no row was known exactly at a cut, so known_at <= as_of went unseen.
    w = world()
    _patch_view(monkeypatch, "known_at < :as_of", "known_at <= :as_of")
    report = check_truncation(w, _upto(w, "view"), _first_cut(w), seed=SEED)
    _assert_caught(report, "view")
    assert {f.variant for f in report.failures} == {Variant.GARBAGE}


@pytest.mark.spec("I1", "I2", "T-NL1", "P3")
def test_review_4_recomputing_the_full_run_still_catches_the_smoother() -> None:
    # Finding 4: slice_full=False ran (b) only up to τ, where a smoother has no later rows.
    w = world()
    registry = _upto(w, "filter").replace(Smoothed(w.model))
    report = check_truncation(w, registry, _leak_cuts(w)[:1], seed=SEED, slice_full=False)
    _assert_caught(report, "filter")


def _known_at_tables(conn: sqlite3.Connection) -> list[str]:
    tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
    return [
        table
        for table in tables
        if "known_at" in {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    ]


@pytest.mark.spec("T-NL1")
def test_review_6_the_truncated_store_drops_later_rows_from_every_table() -> None:
    # Finding 6: dividends, splits and calendar_cache kept rows known after the cut.
    w = world()
    cut = w.edges["after_dst"].tau
    conn = sqlite3.connect(":memory:", isolation_level=None)
    w.conn.backup(conn)
    store = BarStore(conn)
    later = cut.date() + timedelta(days=3)
    store.put_dividend(SYMBOL, ex_date=later, amount=1.0, declared=later, known_at=HISTORY)
    store.put_split(SYMBOL, ex_date=later, ratio=2.0, declared=later, known_at=HISTORY)
    conn.execute(
        "INSERT INTO calendar_cache VALUES ('test', ?, 0, 0, ?)",
        (later.isoformat(), to_us(cut + timedelta(days=3))),
    )
    conn.execute(
        "INSERT INTO data_flags VALUES (?, ?, 'test', 1, 1, '', ?)",
        (SYMBOL, to_us(cut), to_us(cut + timedelta(days=3))),
    )
    tables = _known_at_tables(conn)
    assert {"minute_bars", "hourly_bars", "dividends", "splits", "calendar_cache"} <= set(tables)
    assert {"data_flags"} <= set(tables)
    truncated = truncated_store(dataclasses.replace(w, conn=conn, store=store), cut)
    try:
        for table in tables:
            (latest,) = truncated._conn.execute(
                f"SELECT MAX(known_at) FROM {table}"  # noqa: S608 — names from sqlite_master
            ).fetchone()
            assert latest is None or latest <= to_us(cut), table
    finally:
        truncated.close()
        store.close()


class _MinutesView:
    """The view layer, also recording the minutes of the hour before τ."""

    name = "view"

    def run(
        self,
        source: Source,
        decisions: Sequence[Decision],
        upstream: Mapping[str, Sequence[Record]],
    ) -> list[Record]:
        records = []
        for d in decisions:
            view = source.view(d.tau)
            bars = view.hourly(SYMBOL, purpose=Purpose.OPERATE, last=VIEW_DEPTH)
            minutes = view.minutes(
                SYMBOL, d.tau - timedelta(hours=1), d.tau, purpose=Purpose.OPERATE
            )
            output = {
                "bars": [{"close": bar.close, "t_open": bar.t_open} for bar in bars],
                "minutes": {"t": minutes.t, "close": minutes.close},
                "digest": view.digest(),
            }
            records.append(Record(d.tau, self.name, output))
        return records


@pytest.mark.spec("I1", "T-NL1", "P3", "ARCH§6.2")
def test_review_7_minutes_read_without_availability_are_caught_in_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Finding 7: garbage was written only to hourly_bars, so a minutes query that drops the
    # availability rule was hidden by known_at in every variant.
    w = world()
    assert " AND available_at < :tau" in view_module._MINUTES
    monkeypatch.setattr(
        view_module, "_MINUTES", view_module._MINUTES.replace(" AND available_at < :tau", "")
    )
    report = check_truncation(w, Registry([_MinutesView()]), _first_cut(w), seed=SEED)
    _assert_caught(report, "view")
    assert {f.variant for f in report.failures} == {Variant.GARBAGE}


def _signed_zero_peek(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plant a lookahead whose only trace is a signed zero: the last bar's volume of a
    `last=n` read is -0.0 when the store holds a bar after τ, else 0.0. The view's digest
    hashes the rows before this, so it can't see the sign."""
    original = PointInTimeView.hourly

    def peeking(
        self: PointInTimeView,
        symbol: str,
        *,
        purpose: Purpose,
        since: datetime | None = None,
        last: int | None = None,
    ) -> list[HourlyBar]:
        bars = original(self, symbol, purpose=purpose, since=since, last=last)
        if last is None or not bars:
            return bars
        (later,) = self._conn.execute(
            "SELECT COUNT(*) FROM hourly_bars WHERE symbol = ? AND t_open > ?",
            (symbol, to_us(self.tau)),
        ).fetchone()
        return [*bars[:-1], dataclasses.replace(bars[-1], volume=-0.0 if later else 0.0)]

    monkeypatch.setattr(PointInTimeView, "hourly", peeking)


@pytest.mark.spec("I1", "T-NL1", "P3")
def test_review_9_a_lookahead_in_a_signed_zero_is_caught_in_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Finding 9: the record caches keyed on Python ==, where 0.0 == -0.0, so the full run's
    # record was handed to the truncated one.
    w = world()
    _signed_zero_peek(monkeypatch)
    report = check_truncation(w, _upto(w, "view"), _first_cut(w), seed=SEED)
    _assert_caught(report, "view")
    assert Variant.TRUNCATED in {f.variant for f in report.failures}


@pytest.mark.spec("T-NL1", "T-NL2", "P3")
def test_review_10_a_filter_that_never_leaves_cold_start_fails() -> None:
    # Finding 10: the filter's {"model": null} cold-start records counted as compared, so a
    # filter that never ran its model passed.
    w = world()
    registry = _upto(w, "filter").replace(FilterLayer(w.model, activation=HORIZON))
    report = check_truncation(w, registry, _first_cut(w), seed=SEED)
    assert not report.ok, report.compared
    assert report.layers == {"filter"}, report.explain()


class _MutatingFeatures(FeaturesLayer):
    """Writes the close read at HORIZON into each record's first scaled value after the
    fact: a leak through records the harness shares between runs."""

    def run(
        self,
        source: Source,
        decisions: Sequence[Decision],
        upstream: Mapping[str, Sequence[Record]],
    ) -> list[Record]:
        records = super().run(source, decisions, upstream)
        (peek,) = source.view(HORIZON).hourly(SYMBOL, purpose=Purpose.OPERATE, last=1)
        for record in records:
            if "scaled" in record.output:
                record.output["scaled"][0] = float(peek.close)
        return records


@pytest.mark.spec("T-NL1", "P3")
def test_review_11_a_layer_that_mutates_its_records_is_caught_in_features() -> None:
    # Finding 11: the caches gave every run the same mutable Record, so a run's mutation
    # changed the reference run's record too. Records are now read-only, so the layer can't
    # write: it raises in every run, and a layer that raises fails.
    w = world()
    registry = _upto(w, "features").replace(_MutatingFeatures(w.config, w.scaler))
    report = check_truncation(w, registry, _first_cut(w), seed=SEED)
    _assert_caught(report, "features", MISMATCH_OR_ERROR)


@pytest.mark.spec("T-NL1")
def test_review_11_a_record_is_immutable_canonical_bytes() -> None:
    tau = world().decisions[0].tau
    record = Record(tau, "features", {"raw": [1.0, -0.0]})
    with pytest.raises((TypeError, AttributeError)):
        record.output["raw"][0] = 2.0
    with pytest.raises((TypeError, AttributeError)):
        record.output = {}  # type: ignore[misc]


class _PeekingStream(FeatureStream):
    """A stream whose snapshot carries state outside `_state`: how many bars the store holds."""

    peek = 0

    def snapshot(self) -> FeatureState:
        state = super().snapshot()
        return state.model_copy(update={"returns": [*state.returns, float(self.peek)]})


class _PeekingFeatures(FeaturesLayer):
    def run(
        self,
        source: Source,
        decisions: Sequence[Decision],
        upstream: Mapping[str, Sequence[Record]],
    ) -> list[Record]:
        _PeekingStream.peek = len(source.view(HORIZON).hourly(SYMBOL, purpose=Purpose.OPERATE))
        return super().run(source, decisions, upstream)


@pytest.mark.spec("T-NL1", "P3")
def test_review_12_snapshot_state_outside_the_fast_hash_is_caught_in_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Finding 12: the state hash read FeatureStream._state, not snapshot().
    w = world()
    monkeypatch.setattr(truncation_module, "FeatureStream", _PeekingStream)
    registry = _upto(w, "features").replace(_PeekingFeatures(w.config, w.scaler))
    report = check_truncation(w, registry, _first_cut(w), seed=SEED)
    _assert_caught(report, "features")


@pytest.mark.spec("T-NL1")
def test_review_12_the_fast_state_hash_is_the_snapshot_hash_at_every_bar() -> None:
    # The harness hashes the stream's state without snapshot()'s deep copy. Guard that
    # shortcut: at every decision of the world it equals sha256_hex(stream.snapshot()).
    w = world()
    stream = FeatureStream(w.config)
    for d in w.decisions:
        with contextlib.suppress(FeatureError):
            features_for(stream, w.store.view(d.tau, as_of=d.tau), SYMBOL, d.bar)
        assert _feature_state_hash(stream) == sha256_hex(stream.snapshot()), d.tau
