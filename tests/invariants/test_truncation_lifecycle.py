"""B7b: refits and calibration in the T-NL1 truncation harness, and T-NL2 over them (the B5 open
item; PLAN B7 "T-NL1 and T-NL2 over refits"; ARCHITECTURE §7.4, §13).

`RefitLayer` runs the evening jobs through `quant.jobs.refit.refit` on each run's own store, and
`CalibrationLayer` reads the calibration governing at τ. In the synthetic world the lifecycle's
start model refits on 2016-11-30 and the refit, with the calibration it builds as of 17:00 that
day, governs from the 2016-12-01 open. Pinned here: clean code passes every variant at cuts on
both sides of that open; the refit and the calibration govern only from the open, each with its
cutoff or as-of time before every τ that uses it; and three planted leaks are caught: a refit
trained on bars after its evening, a calibration as of a time after the decisions it serves, and
a calibration whose smoother reads past its as-of time.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import quant.jobs.calibrate as calibrate_job
import quant.jobs.refit as refit_job
from quant.core.calendar import NEW_YORK
from quant.core.calibration import Calibration
from quant.core.epoch import to_us
from quant.core.regime import Label
from quant.core.schedule import Decision
from quant.data.policy import Purpose
from quant.eval.smoother import smooth
from quant.features.stream import compute_features
from quant.hmm.filter import forward_pass
from quant.hmm.labels import label_probs, top_label
from quant.hmm.model import load_model
from tests.support.bars import SYMBOL
from tests.support.truncation import (
    HORIZON,
    CalibrationLayer,
    Provenance,
    Record,
    Registry,
    Report,
    Source,
    Variant,
    World,
    check_truncation,
    default_registry,
    lifecycle_artifacts,
    lifecycle_spec,
    run_layers,
    world,
)

SEED = 20160104
REFIT_CUTOFF = datetime(2016, 11, 30, 17, tzinfo=NEW_YORK).astimezone(UTC)
REFIT_OPEN = datetime(2016, 12, 1, 9, 30, tzinfo=NEW_YORK).astimezone(UTC)
# Planted leak R: the evening job trains on every bar through the world's last session.
LATE_CUTOFF = datetime(2016, 12, 30, 17, tzinfo=NEW_YORK).astimezone(UTC)
LIFECYCLE = ("view", "features", "filter", "refit", "calibration")


def _upto_calibration(w: World) -> Registry:
    """The clean registry up to the calibration layer: the risk and actions layers after it
    would only make the full run slower."""
    layers = default_registry(w).layers
    return Registry(layers[: [layer.name for layer in layers].index("calibration") + 1])


def _lifecycle_cuts(w: World) -> tuple[datetime, ...]:
    """The model's first decision; the last decision before the refit's open (its evening job
    has run, its pointer not yet effective); the first decision after it; one a week later; and
    the world's last."""
    before = max(d.tau for d in w.decisions if d.tau < REFIT_OPEN)
    after = min(d.tau for d in w.decisions if d.tau > REFIT_OPEN)
    later = min(d.tau for d in w.decisions if d.tau > REFIT_OPEN + timedelta(days=7))
    return (w.edges["model_first"].tau, before, after, later, w.decisions[-1].tau)


@functools.cache
def _full_run() -> Any:
    w = world()
    return run_layers(_upto_calibration(w), Source(w.store, Variant.FULL, None), w.decisions)


# --- registration and the clean lifecycle ----------------------------------------------------


@pytest.mark.spec("T-NL1", "T-NL2")
def test_the_lifecycle_layers_follow_the_filter_in_the_default_registry() -> None:
    w = world()
    assert default_registry(w).names == (
        "view",
        "features",
        "filter",
        "refit",
        "calibration",
        "risk",
        "actions",
    )
    spec = lifecycle_spec(w)
    # The synthetic start model has two labels, so its calibration targets vary.
    assert spec.start.labels.labels == (Label.CALM_UP, Label.CHOP)
    assert spec.start.params is w.model.params
    assert spec.start.fit_end == w.model.fit_end


@pytest.mark.spec("T-NL1", "T-NL2", "I1", "SPEC§5", "SPEC§9", "C7", "ARCH§7.3", "ARCH§7.4")
def test_the_full_run_refits_once_and_activates_the_refit_and_its_calibration_at_the_open() -> None:
    w = world()
    run = _full_run()
    assert run.errors == {}
    refits = run.records["refit"]
    calibrations = run.records["calibration"]
    start = lifecycle_spec(w).start.artifact_hash()
    live = [r for r in refits if not r.cold]
    assert live
    assert all(r.output["job_errors"] == () for r in live)

    final = live[-1].output
    ((outcome, cutoff, artifact, effective, reasons),) = final["refits"]
    assert (outcome, cutoff, effective, reasons) == (
        "activated",
        to_us(REFIT_CUTOFF),
        to_us(REFIT_OPEN),
        (),
    )
    for record in live:
        expected = artifact if record.tau >= REFIT_OPEN else start
        assert record.output["model"] == expected, record.tau
        assert record.provenance is not None
        assert record.provenance.artifact == expected
        # T-NL2 for refits: the governing model's data cutoff precedes τ.
        assert record.provenance.fit_end < record.tau
        if record.tau >= REFIT_OPEN:
            assert record.provenance.fit_end == REFIT_CUTOFF
        assert record.output["frozen"] is False

    first, refit_calibration = None, None
    for record in calibrations:
        if record.cold:
            continue
        assert record.provenance is not None
        assert record.provenance.fit_end < record.tau  # T-NL2: as-of before every use
        if record.tau < REFIT_OPEN:
            first = first or record.output["calibration"]
            assert record.output["calibration"] == first
            assert record.provenance.fit_end == w.activation
        else:
            refit_calibration = refit_calibration or record.output["calibration"]
            assert record.output["calibration"] == refit_calibration
            assert record.provenance.fit_end == REFIT_CUTOFF
            assert record.output["sufficient"] is True
    assert first is not None
    assert refit_calibration is not None
    assert first != refit_calibration

    built = _calibration(refit_calibration)
    assert built.document.models == (start,)
    assert built.document.scored >= lifecycle_spec(w).calibration.min_bars
    # Both of the start model's labels were scored and at least one was recalibrated or passed.
    calm_chop = [v for v in built.document.labels if v.raw.base_rate not in (None, 0.0)]
    assert len(calm_chop) == 2


def _calibration(digest: str) -> Calibration:
    from quant.core.calibration import load_calibration

    return load_calibration(lifecycle_artifacts() / "calibrations" / digest)


@pytest.mark.spec("T-NL1", "T-NL2", "I1", "P4", "ARCH§13", "ARCH§7.4")
def test_clean_lifecycle_passes_every_variant_on_both_sides_of_the_refit() -> None:
    w = world()
    cuts = _lifecycle_cuts(w)
    report = check_truncation(w, _upto_calibration(w), cuts, seed=SEED)
    assert report.ok, report.explain()
    assert report.cuts == cuts
    assert set(report.compared) == set(LIFECYCLE)
    assert all(count > 0 for count in report.compared.values()), report.compared


# --- planted leaks (P3) ----------------------------------------------------------------------


def _assert_caught(report: Report, layers: set[str], kinds: set[str]) -> None:
    assert not report.ok, "the planted leak was not caught"
    assert report.layers <= layers, report.explain()
    assert {f.kind for f in report.failures} <= kinds, report.explain()
    # The cuts after the leak's refit catch it; the ones before it can't see it.
    caught = {f.cut for f in report.failures}
    assert caught == {cut for cut in report.cuts if cut > REFIT_OPEN}, report.explain()


def _late_refit(store: Any, current: Any, *, at: datetime, **kwargs: Any) -> Any:
    """Planted leak R: `refit` with its rows read up to LATE_CUTOFF, the world's last evening,
    instead of the evening's own 17:00. B7a's guard refuses rows from after `at`, so the leak
    also claims `at` is LATE_CUTOFF's evening; ops.db still stamps the real evening."""
    ops = kwargs["ops"]
    events = ops.refit_events()
    if not refit_job.refit_due(current, at, last_refit=events[-1].cutoff if events else None):
        return None
    data = refit_job.refit_data(
        store,
        kwargs["symbol"],
        start=current.fit_start,
        cutoff=LATE_CUTOFF,
        features=kwargs["features"],
    )
    rules = kwargs.pop("calibration")
    symbol, features = kwargs.pop("symbol"), kwargs.pop("features")
    kwargs.pop("policy", None)

    def build(as_of: datetime) -> Any:
        return calibrate_job.calibrate(
            store,
            ops,
            as_of=as_of,
            symbol=symbol,
            features=features,
            artifacts=kwargs["artifacts"],
            config=rules,
        )

    return refit_job.run_refit(
        data, current, at=LATE_CUTOFF + timedelta(minutes=30), calibration=build, **kwargs
    )


@pytest.mark.full  # about 3-4 s each: the fast tier's budget (B7b red)
@pytest.mark.spec("T-NL1", "T-NL2", "P3", "SPEC§5")
def test_planted_leak_a_refit_trained_on_later_bars_is_caught_in_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(refit_job, "refit", _late_refit)
    w = world()
    report = check_truncation(w, _upto_calibration(w), _lifecycle_cuts(w), seed=SEED)
    _assert_caught(report, {"refit", "calibration"}, {"mismatch", "provenance"})
    assert "refit" in report.layers


class _LateCalibration(CalibrationLayer):
    """Planted leak C: every decision served a calibration as of the world's last evening, built
    once outside the runs, so only T-NL2 can see it (as B5's planted leak 6)."""

    def __init__(self, calibration: Calibration) -> None:
        super().__init__()
        self._late = calibration

    def run(
        self,
        source: Source,
        decisions: Sequence[Decision],
        upstream: Mapping[str, Sequence[Record]],
    ) -> list[Record]:
        late = self._late
        digest = late.artifact_hash()
        return [
            Record(d.tau, self.name, {"calibration": None}, cold=True)
            if refit.cold
            else Record(d.tau, self.name, {"calibration": digest}, Provenance(digest, late.as_of))
            for d, refit in zip(decisions, upstream["refit"], strict=True)
        ]


@pytest.mark.spec("T-NL2", "P3", "ARCH§7.4", "SPEC§9")
def test_planted_leak_a_calibration_as_of_after_tau_is_caught_by_provenance(
    tmp_path: Path,
) -> None:
    from quant.core.clock import SimClock
    from quant.hmm.model import save_model
    from quant.jobs.calibrate import calibrate
    from quant.store.ops import OpsStore

    w = world()
    spec = lifecycle_spec(w)
    save_model(spec.start, tmp_path / "models")
    ops = OpsStore.memory(clock=SimClock(w.activation))
    ops.activate("model", spec.start.artifact_hash(), reason="first model")
    late = calibrate(
        w.store,
        ops,
        as_of=LATE_CUTOFF,
        symbol=SYMBOL,
        features=w.config,
        artifacts=tmp_path,
        config=spec.calibration,
    )
    assert late.as_of == LATE_CUTOFF
    registry = _upto_calibration(w).replace(_LateCalibration(late))
    cuts = (w.edges["model_first"].tau, w.edges["after_dst"].tau)
    report = check_truncation(w, registry, cuts, seed=SEED)
    assert not report.ok
    assert report.layers == {"calibration"}, report.explain()
    assert {f.kind for f in report.failures} == {"provenance"}, report.explain()
    assert {f.variant for f in report.failures} == set(Variant)


def _peeking_series(
    store: Any,
    ops: Any,
    *,
    as_of: datetime,
    symbol: str,
    features: Any,
    artifacts: Path,
    level: float,
    policy: Any = None,
) -> Any:
    """Planted leak S: calibration_series whose smoother runs over every row the store holds,
    later ones included (the view read at HORIZON), keeping only the bars before `as_of`:
    each target is smoothed with rows the as-of time hadn't seen."""
    from quant.eval.calibration import ForecastSeries, f1_interval
    from quant.jobs.calibrate import model_spans

    labels = tuple(Label)
    parts = []
    bars = store.view(HORIZON, as_of=HORIZON).hourly(symbol, purpose=Purpose.OPERATE)
    matrix = compute_features(bars, features)
    for span in model_spans(ops, as_of=as_of):
        model = load_model(artifacts / "models" / span.artifact)
        rows = matrix.window(model.fit_start, HORIZON)
        trace = forward_pass(model.params, model.scaler, rows.values, columns=rows.columns)
        smoothed = smooth(model.params, model.scaler, rows.values, columns=rows.columns)
        keep = np.nonzero(
            (rows.available_at >= to_us(span.start)) & (rows.available_at < to_us(span.end))
        )[0]
        if not len(keep):
            continue
        forecast = np.array(
            [
                [
                    min(1.0, label_probs(trace.state_probs[t], model.labels).get(lab, 0.0))
                    for lab in labels
                ]
                for t in keep
            ]
        )
        target = np.array(
            [labels.index(top_label(label_probs(smoothed[t], model.labels))) for t in keep],
            dtype=np.int64,
        )
        smoothed_labels = np.array(
            [
                [min(1.0, label_probs(smoothed[t], model.labels).get(lab, 0.0)) for lab in labels]
                for t in keep
            ]
        )
        intervals = [
            f1_interval(model, trace.state_probs[t] @ model.params.transmat, level=level)
            for t in keep
        ]
        parts.append(
            ForecastSeries(
                (span.artifact,),
                rows.available_at[keep].copy(),
                forecast,
                target,
                smoothed_labels,
                np.array([lo for lo, _ in intervals]),
                np.array([hi for _, hi in intervals]),
                rows.values[keep, rows.columns.index("ret")].copy(),
            )
        )
    return ForecastSeries.concat(parts)


@pytest.mark.full  # about 3-4 s each: the fast tier's budget (B7b red)
@pytest.mark.spec("T-NL1", "I1", "I2", "P3", "A4", "ARCH§7.4")
def test_planted_leak_a_calibration_smoothing_past_its_as_of_time_is_caught_in_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The calibration's hash is in the refit record (the evening job built it), so the leak
    # shows there first, at the first decision its calibration governs.
    monkeypatch.setattr(calibrate_job, "calibration_series", _peeking_series)
    w = world()
    cuts = _lifecycle_cuts(w)
    report = check_truncation(w, _upto_calibration(w), cuts, seed=SEED)
    assert not report.ok, "the planted leak was not caught"
    assert report.layers == {"refit"}, report.explain()
    # Against the truncated store at the first decision the refit's calibration governs, its
    # smoothed probabilities differ from the full run's, which smoothed with all of December
    # (the labels alone don't). Later cuts hold enough of December that the difference rounds
    # away, and nothing before the open can show it.
    truncated = {f.cut: f.kind for f in report.failures if f.variant is Variant.TRUNCATED}
    first_after = min(cut for cut in cuts if cut > REFIT_OPEN)
    assert truncated.get(first_after) == "mismatch", report.explain()
    assert set(truncated.values()) == {"mismatch"}
    assert all(cut > REFIT_OPEN for cut in truncated)
    # The garbage variant's peek reads the garbage, and the job can't even build the first
    # calibration: an error at every cut (B5: "either way the failure is the layer's").
    assert {f.kind for f in report.failures if f.variant is Variant.GARBAGE} <= {
        "mismatch",
        "error",
    }
