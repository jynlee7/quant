"""B5: the T-NL1 truncation harness and T-NL2 provenance (SPEC §6, I1, I2, ARCHITECTURE §13).

Clean code must pass all three variants at every cut, and each of PLAN B5's six planted leaks
must be caught, with the failure naming the layer the leak is in (P3: a harness that can't fail
is caught). Leaks are planted with monkeypatch or a substitute layer registered in place of the
clean one, never by editing src/. The fast tier draws 50 cuts on synthetic bars; the full tier
draws 200. The full tier moves to real FIT bars once B6 brings them (BUILD_LOG open item).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta

import numpy as np
import pytest

import quant.data.view as view_module
from quant.core.epoch import from_us
from quant.core.errors import FeatureError
from quant.core.schedule import Decision
from quant.data.policy import Purpose
from quant.eval.smoother import smooth
from quant.features.scaler import Scaler
from quant.features.stream import FeatureStream, compute_features, features_for
from quant.hmm.model import HmmModel
from tests.support.bars import SYMBOL
from tests.support.truncation import (
    ACTIVATION,
    EDGES,
    HORIZON,
    LATE_FIT_END,
    MODEL_FIT_END,
    FilterLayer,
    Provenance,
    Record,
    Report,
    Source,
    Variant,
    World,
    check_truncation,
    default_registry,
    draw_cuts,
    error_output,
    features_output,
    fixture_model,
    garbage_store,
    provenance_failures,
    truncated_store,
    world,
)

SEED = 20160104
FAST_CUTS = 50  # PLAN P4
FULL_CUTS = 200
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
    # Every layer was compared, the filter and risk verdicts included, so a pass means something.
    assert set(report.compared) == {"view", "features", "filter", "risk"}
    assert all(count > 0 for count in report.compared.values()), report.compared


@pytest.mark.full
@pytest.mark.spec("I1", "T-NL1", "T-NL2", "P4")
def test_clean_code_passes_every_variant_at_two_hundred_cuts() -> None:
    # Synthetic bars until B6 brings real FIT bars (PLAN P4; BUILD_LOG open item).
    w = world()
    cuts = draw_cuts(w, FULL_CUTS, seed=SEED + 1)
    report = check_truncation(w, default_registry(w), cuts, seed=SEED + 1)
    assert report.ok, report.explain()
    assert len(report.cuts) == FULL_CUTS


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
    at_cut = store.view(cut, as_of=cut).hourly(SYMBOL, purpose=Purpose.OPERATE)
    assert at_cut[-1].t_open == garbage.revised
    assert at_cut[-1].version == 1
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
            smooth(
                model.params,
                model.scaler,
                np.array(rows, dtype=np.float64),
                columns=model.scaler.columns,
            )
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
                records.append(Record(d.tau, self.name, {"model": None}))
        return records


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
    # The full run is computed once and sliced per cut. Recomputing it per cut must find exactly
    # the same failures for planted leak 1, so the shortcut can't be what makes a run pass.
    w = world()
    registry = default_registry(w).replace(CenteredAverage(w))
    cuts = _leak_cuts(w)
    sliced = check_truncation(w, registry, cuts, seed=SEED, slice_full=True)
    fresh = check_truncation(w, registry, cuts, seed=SEED, slice_full=False)
    _assert_caught(sliced, "features", MISMATCH_OR_ERROR)
    _assert_caught(fresh, "features", MISMATCH_OR_ERROR)
    assert _where(sliced) == _where(fresh)
    # The leak shows against the full run itself, not only through the garbage.
    assert Variant.TRUNCATED in {f.variant for f in sliced.failures}
