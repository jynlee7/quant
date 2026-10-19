"""B8b: the decision layer of T-NL1 and T-NL2 (PLAN §2 rule 4, ARCHITECTURE §13, SPEC §6).

`DecisionLayer` (tests.support.truncation) runs decide() itself, under the model and calibration
the lifecycle layers say govern at each τ (B7's open item: the filter layer ran the fixture
model), from a cold EngineState per run, with the risk verdicts on its own intents. Its records
are DecisionRecords and its state the EngineState hash (B5's open item). The clean layer runs in
the default registry's 50 cuts (test_truncation.py); here its records are checked for what they
hold, and two planted leaks (P3) are caught where they enter: decide reading the view at a later
τ (T-NL1, in the decide layer) and a calibration whose as-of time is after τ (T-NL2).
"""

from __future__ import annotations

import functools
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from typing import Any

import pytest

from quant.core.calibration import Calibration
from quant.core.epoch import to_us
from quant.core.schedule import Decision
from quant.data.view import PointInTimeView
from tests.support.calibration import sizing_calibration
from tests.support.truncation import (
    DecisionLayer,
    Record,
    Registry,
    Source,
    Variant,
    World,
    check_truncation,
    default_registry,
    run_layers,
    world,
)

SEED = 20160104
UPSTREAM = ("view", "features", "filter", "refit", "calibration")
# The refit's model and calibration govern from this open (test_truncation_lifecycle.py).
REFIT_OPEN = datetime(2016, 12, 1, 14, 30, tzinfo=UTC)
# Planted leak V: decide reads the view an hour past τ, where the next bar is available.
PEEK = timedelta(hours=1)
# Planted leak C: a calibration as of a time after every decision of the world.
LATE_AS_OF = datetime(2017, 1, 15, 13, 30, tzinfo=UTC)


def _upto_decide(w: World) -> Registry:
    """The clean registry up to the decision layer: the risk and actions layers after it would
    only make the runs slower."""
    layers = default_registry(w).layers
    return Registry(layers[: [layer.name for layer in layers].index("decide") + 1])


@functools.cache
def _full_run() -> Any:
    w = world()
    return run_layers(_upto_decide(w), Source(w.store, Variant.FULL, None), w.decisions)


@pytest.mark.spec("T-NL1", "T-NL2", "ARCH§13", "ARCH§4.3", "SPEC§6", "I1")
def test_the_decision_layer_records_decision_records_under_the_governing_model() -> None:
    from quant.decide.record import DecisionRecord

    w = world()
    assert _upto_decide(w).names == (*UPSTREAM, "decide")
    run = _full_run()
    assert not run.errors, run.errors
    decided = [
        (d, refit, record)
        for d, refit, record in zip(
            run.decisions, run.records["refit"], run.records["decide"], strict=True
        )
        if not record.cold
    ]
    assert decided, "the decision layer decided nothing"
    # Cold exactly where the lifecycle is: before the model's activation.
    assert all(
        r.cold == refit.cold
        for refit, r in zip(run.records["refit"], run.records["decide"], strict=True)
    )
    models = set()
    for d, refit, record in decided:
        out = record.output
        assert set(out) == set(DecisionRecord.model_fields)
        # The governing model and calibration, as the lifecycle names them (B7's open item).
        assert out["model_hash"] == refit.output["model"]
        assert out["calibration_hash"] == refit.output["calibration"]
        models.add(out["model_hash"])
        # T-NL2: the record names both, with times before τ, and its provenance is the later.
        fit_end = datetime.fromisoformat(out["model_fit_end"].replace("Z", "+00:00"))
        assert fit_end < d.tau
        assert record.provenance is not None
        assert record.provenance.fit_end < d.tau
        if out["calibration_as_of"] is not None:
            as_of = datetime.fromisoformat(out["calibration_as_of"].replace("Z", "+00:00"))
            assert as_of < d.tau
            assert record.provenance.fit_end == max(fit_end, as_of)
        # The state hash is the record's state_after; one verdict per intent.
        assert record.state == out["state_after"]
        assert len(out["verdicts"]) == len(out["intents"])
    # The refit takes over at its open, and the decision layer follows it.
    assert len(models) == 2
    after = [r.output["model_hash"] for d, _, r in decided if d.tau > REFIT_OPEN]
    before = [r.output["model_hash"] for d, _, r in decided if d.tau < REFIT_OPEN]
    assert set(after).isdisjoint(before)
    # It trades, and the risk engine judges what it asks for.
    intents = [r.output for _, _, r in decided if r.output["intents"]]
    assert intents, "the decision layer never asked for an order"
    assert any(v["verdict"] == "approved" for o in intents for v in o["verdicts"])
    # Each record's state_before is the state_after of the decision before it.
    for (_, _, prev), (_, _, cur) in pairwise(decided):
        assert cur.output["state_before"] == prev.output["state_after"]


# --- planted leaks (P3) ----------------------------------------------------------------------


class _PeekingDecision(DecisionLayer):
    """Planted leak V: the view decide gets claims τ (its `tau` and `as_of`) but reads its rows
    as a view an hour later would: the next bar is available and known by then."""

    def view_for(self, source: Source, d: Decision) -> PointInTimeView:
        view = source.view(d.tau)
        view._params.update(tau=to_us(d.tau + PEEK), as_of=to_us(d.tau + PEEK))
        return view


class _LateCalibration(DecisionLayer):
    """Planted leak C: decide sizes with a calibration as of a time after every decision."""

    def calibration_for(self, d: Decision, refit: Record) -> Calibration | None:
        return sizing_calibration(start=LATE_AS_OF - timedelta(hours=1200))


@pytest.mark.full  # about 1.5 s each: the fast tier's budget (B8b red)
@pytest.mark.spec("T-NL1", "I1", "P3", "ARCH§13")
def test_planted_leak_decide_reading_the_view_at_a_later_tau_is_caught_in_decide() -> None:
    w = world()
    cut = w.edges["after_dst"]
    assert not cut.reduce_only  # its next bar comes within the hour
    registry = _upto_decide(w).replace(_PeekingDecision())
    report = check_truncation(w, registry, (cut.tau,), seed=SEED)
    assert not report.ok, "the planted leak was not caught"
    assert report.layers == {"decide"}, report.explain()
    assert {f.kind for f in report.failures} == {"mismatch"}, report.explain()
    assert {f.variant for f in report.failures} == {Variant.TRUNCATED, Variant.GARBAGE}
    assert all(report.compared[name] > 0 for name in (*UPSTREAM, "decide")), report.compared


@pytest.mark.full  # about 1.5 s each: the fast tier's budget (B8b red)
@pytest.mark.spec("T-NL2", "P3", "ARCH§7.4", "ARCH§13")
def test_planted_leak_a_calibration_as_of_after_tau_is_caught_by_provenance() -> None:
    w = world()
    late = sizing_calibration(start=LATE_AS_OF - timedelta(hours=1200))
    assert late.as_of == LATE_AS_OF > w.decisions[-1].tau
    registry = _upto_decide(w).replace(_LateCalibration())
    cut = w.edges["model_first"]
    report = check_truncation(w, registry, (cut.tau,), seed=SEED)
    assert not report.ok, "the planted leak was not caught"
    assert report.layers == {"decide"}, report.explain()
    assert {f.kind for f in report.failures} == {"provenance"}, report.explain()
    assert {f.variant for f in report.failures} == set(Variant)
    assert {f.tau for f in report.failures} == {cut.tau}
