"""B8b: decide(), the pure decision core (ARCHITECTURE §4.1-§4.3, spec §6-§9, C4, C6, K6;
quant.decide.core).

A cold start warms the feature stream and the filter over the view from FIT's start and the
model's fit start (checked against B3's and B4's own code run on the same bars); a model swap
warms afresh and keeps the switching, trade and edge state; decide runs with file, network and
clock access disabled on an in-memory BarStore and gives the same record hash every time; rule 6
from provenance and missed decisions; Kelly 0 without an edge; the flags; and the closing-bar
decision that only reduces. Everything runs on the synthetic world (tests.support.decide).
"""

from __future__ import annotations

import builtins
import math
import os
import socket
import time
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from quant.core.calendar import TradingCalendar
from quant.core.orders import Side
from quant.core.regime import Label
from quant.core.schedule import decision_schedule
from quant.data.market_db import BarStore
from quant.data.policy import Purpose
from quant.features.stream import FeatureStream, compute_features
from quant.hmm.filter import ForwardFilter
from tests.support.calibration import SIZING_START, sizing_calibration
from tests.support.decide import DECISION_DAY, DecideFixture, r1_release

# Another model for the swap: the fixture model fit a month earlier (its window ends
# 2016-10-01, a New York midnight, before every decision here).
OTHER_FIT_END = datetime(2016, 10, 1, 4, tzinfo=UTC)


def _reference(fx: DecideFixture, model: Any) -> tuple[Any, list[float], Any, Any]:
    """B3's and B4's code on the bars the view at tau shows: the stream fed every complete bar
    through the decision bar, and the filter warmed on the model's own rows (available at or
    after its fit start) before the decision bar, then stepped on the decision bar's row.
    Returns (stream snapshot, the decision row, its FilterOutput, filter snapshot)."""
    bars = fx.view().hourly("SPY", purpose=Purpose.OPERATE)
    config = r1_release().manifest.features
    stream = FeatureStream(config)
    for bar in bars:
        if bar.complete and bar.t_open <= fx.decision.bar.t_open:
            stream.update(bar)
    matrix = compute_features([b for b in bars if b.t_open <= fx.decision.bar.t_open], config)
    t_open = matrix.t_open
    decision_us = int(fx.decision.bar.t_open.timestamp() * 1_000_000)
    start_us = int(model.fit_start.timestamp() * 1_000_000)
    keep = (matrix.available_at >= start_us) & (t_open < decision_us)
    filt = ForwardFilter.warmed(model, matrix.values[keep], columns=config.columns)
    assert int(t_open[-1]) == decision_us
    out = filt.step(matrix.values[-1], columns=config.columns)
    return stream.snapshot(), [float(x) for x in matrix.values[-1]], out, filt.snapshot()


# --- warm-up and model swap ------------------------------------------------------------------


@pytest.mark.spec("SPEC§6", "C4", "ARCH§4.2", "ARCH§4.1")
def test_a_cold_start_warms_features_and_filter_from_the_fit_start_through_the_view() -> None:
    fx = DecideFixture.get()
    out = fx.decide(fx.cold())
    stream, row, filtered, filt = _reference(fx, fx.model)
    assert out.state.features == stream
    assert out.state.filter == filt
    assert out.state.filter.model_hash == fx.model.artifact_hash()
    r = out.record
    assert r.features == row
    assert r.filter is not None
    assert r.filter.label_probs == filtered.label_probs
    assert r.filter.state_probs == [float(x) for x in filtered.state_probs]
    assert r.filter.loglik == filtered.loglik
    # C4: a cold start has no active label, and the first takeover needs 3 bars.
    assert r.active is None
    assert not r.switched
    assert out.intents == ()
    assert set(out.state.indicators) == {Label.CALM_UP, Label.CHOP}


@pytest.mark.spec("SPEC§6", "ARCH§4.3", "ARCH§7.3", "C7")
def test_a_model_swap_warms_afresh_and_keeps_switching_trade_and_edge() -> None:
    from tests.support.truncation import fixture_model

    fx = DecideFixture.get()
    other = fixture_model(OTHER_FIT_END)
    assert other.artifact_hash() != fx.model.artifact_hash()
    assert other.fit_start == fx.model.fit_start
    trade = fx.in_trade()
    state = fx.warm_state(active=Label.CHOP, trade=trade, shadows={})
    out = fx.decide(state, model=other)
    stream, _, filtered, filt = _reference(fx, other)
    # The filter runs the new model from its own fit window, not the old filter's posterior.
    assert out.state.filter == filt
    assert out.state.filter.model_hash == other.artifact_hash()
    assert out.state.features == stream
    assert out.record.filter is not None
    assert out.record.filter.label_probs == filtered.label_probs
    assert out.record.model_hash == other.artifact_hash()
    # Switching, the trade and the edge carry over.
    assert out.record.active is Label.CHOP
    assert not out.record.switched
    assert out.state.trade.in_position
    assert (out.state.trade.entry_price, out.state.trade.stop) == (trade.entry_price, trade.stop)
    assert out.state.trade.bars_in_trade == trade.bars_in_trade + 1
    assert out.state.edge == state.edge


@pytest.mark.spec("SPEC§6", "ARCH§4.2")
def test_a_warm_state_steps_once_per_decision() -> None:
    fx = DecideFixture.get()
    warm = fx.warm()
    out = fx.decide(warm)
    _, _, filtered, filt = _reference(fx, fx.model)
    assert out.state.filter == filt
    assert out.state.filter.bars == warm.filter.bars + 1
    assert out.record.filter is not None
    assert out.record.filter.label_probs == filtered.label_probs


# --- purity ----------------------------------------------------------------------------------


def _memory_store(fx: DecideFixture) -> BarStore:
    store = BarStore.memory()
    fx.world.conn.backup(store._conn)
    return store


def _forbid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every file, network and clock entry point raises from here on."""

    def refuse(*args: object, **kwargs: object) -> Any:
        raise AssertionError("decide reached I/O or a clock")

    import quant.core.clock as clock

    monkeypatch.setattr(builtins, "open", refuse)
    monkeypatch.setattr(os, "open", refuse)
    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(clock.WallClock, "now", refuse)
    monkeypatch.setattr(clock.SimClock, "now", refuse)
    monkeypatch.setattr(time, "time", refuse)
    monkeypatch.setattr(time, "time_ns", refuse)
    monkeypatch.setattr(time, "monotonic", refuse)


@pytest.mark.spec("ARCH§4.1", "K6", "ARCH§14", "I1")
def test_decide_runs_without_files_network_or_clock_and_repeats_its_record_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant.decide.core import decide

    fx = DecideFixture.get()
    store = _memory_store(fx)
    d = fx.decision
    trade_state = fx.warm_state(active=Label.CHOP, trade=fx.in_trade())
    cold_state = fx.cold()
    inputs = fx.inputs()
    # A reference run, before anything is forbidden (imports and caches warm).
    cold_ref = decide(cold_state, store.view(d.tau, as_of=d.tau), inputs)
    warm_ref = decide(trade_state, store.view(d.tau, as_of=d.tau), inputs)
    assert warm_ref.intents  # the warm run sizes an order, so the whole path runs
    _forbid(monkeypatch)
    for _ in range(2):
        cold = decide(cold_state, store.view(d.tau, as_of=d.tau), inputs)
        warm = decide(trade_state, store.view(d.tau, as_of=d.tau), inputs)
        assert cold.record.record_hash() == cold_ref.record.record_hash()
        assert warm.record.record_hash() == warm_ref.record.record_hash()
        assert warm.state.state_hash() == warm_ref.state.state_hash()
        assert warm.intents == warm_ref.intents
    monkeypatch.undo()
    # The world's own store gives the same record: decide reads only the view.
    assert fx.decide(trade_state).record.record_hash() == warm_ref.record.record_hash()
    store.close()


@pytest.mark.spec("ARCH§4.1", "I1")
def test_decide_leaves_its_state_alone_and_refuses_a_view_at_another_time() -> None:
    fx = DecideFixture.get()
    state = fx.warm_state(active=Label.CHOP, trade=fx.in_trade())
    data = state.to_json()
    fx.decide(state)
    assert state.to_json() == data
    hour_later = fx.decision.tau + timedelta(hours=1)
    later = fx.world.store.view(hour_later, as_of=hour_later)
    with pytest.raises(ValueError, match="tau"):
        fx.decide(state, later)


# --- rule 6 ----------------------------------------------------------------------------------


@pytest.mark.spec("SPEC§8", "T-NL2", "C2", "SPEC§9")
def test_a_model_fit_past_tau_is_unhealthy_and_sizes_nothing() -> None:
    fx = DecideFixture.get()
    early = next(
        d
        for d in decision_schedule(TradingCalendar.xnys(), date(2016, 10, 20))
        if not d.reduce_only
    )
    assert early.tau <= fx.model.fit_end
    out = fx.decide(fx.cold(), decision=early)
    assert out.record.health.cause == "model_provenance"
    assert not out.record.health.operating
    assert out.record.target.value == 0.0
    assert out.intents == ()


@pytest.mark.spec("SPEC§8", "T-NL2", "ARCH§7.4", "SPEC§9")
def test_a_calibration_as_of_tau_or_later_is_unhealthy() -> None:
    fx = DecideFixture.get()
    late = sizing_calibration(start=fx.decision.tau - timedelta(hours=600))
    assert late.as_of >= fx.decision.tau
    state = fx.warm_state(active=Label.CHOP, trade=fx.in_trade())
    out = fx.decide(state, calibration=late)
    assert out.record.health.cause == "calibration_provenance"
    assert out.record.calibration_as_of == late.as_of
    assert out.record.target.value == 0.0
    assert all(i.side is Side.SELL for i in out.intents)


@pytest.mark.spec("SPEC§8", "C6", "C2")
def test_a_missed_decision_replays_unhealthy_and_recovers_on_the_second_healthy_bar() -> None:
    from quant.decide.core import Flags

    fx = DecideFixture.get()
    schedule = decision_schedule(TradingCalendar.xnys(), DECISION_DAY)
    state = fx.warm_state(active=Label.CHOP, trade=fx.in_trade())
    missed = fx.decide(state, flags=Flags(missed=True))
    assert missed.record.health.cause == "missed"
    assert not missed.record.health.operating
    assert missed.record.target.value == 0.0
    nxt = fx.decide(missed.state, decision=schedule[3])
    assert nxt.record.health.cause is None
    assert not nxt.record.health.operating  # recovering
    third = fx.decide(nxt.state, decision=schedule[4])
    assert third.record.health.operating


# --- sizing through decide -------------------------------------------------------------------


@pytest.mark.spec("C12", "SPEC§9")
def test_decide_sizes_zero_without_an_edge() -> None:
    from quant.decide.edge import EdgeBook

    fx = DecideFixture.get()
    seeded = fx.warm_state(active=Label.CHOP, trade=fx.in_trade())
    empty = seeded.model_copy(update={"edge": EdgeBook.empty()})
    assert fx.decide(seeded).record.target.value > 0
    out = fx.decide(empty)
    assert out.record.target.value == 0.0
    assert out.intents == ()
    assert out.record.target.sizing is not None
    assert out.record.target.sizing["f_star"] == 0.0


@pytest.mark.spec("SPEC§9", "A4", "ARCH§7.4", "T-NL2")
def test_decide_reads_the_calibration_for_the_cap_fraction_and_p_tilde() -> None:
    fx = DecideFixture.get()
    calibration = sizing_calibration()
    assert calibration.as_of < fx.decision.tau
    assert calibration.as_of > SIZING_START
    state = fx.warm_state(active=Label.CHOP, trade=fx.in_trade())
    out = fx.decide(state, calibration=calibration)
    r = out.record
    assert (r.calibration_hash, r.calibration_as_of) == (
        calibration.artifact_hash(),
        calibration.as_of,
    )
    assert r.provenance_times() == (fx.model.fit_end, calibration.as_of)
    assert r.filter is not None
    assert r.target.sizing is not None
    p = r.filter.label_probs[Label.CHOP]
    assert r.target.sizing["cap_fraction"] == calibration.cap_fraction(Label.CHOP) == 1.0
    assert r.target.sizing["p_calibrated"] == calibration.calibrated(Label.CHOP, p)
    uncalibrated = fx.decide(state).record
    assert uncalibrated.target.sizing is not None
    assert uncalibrated.target.sizing["cap_fraction"] == 0.25


@pytest.mark.spec("SPEC§9", "SPEC§10", "C5")
def test_halted_frozen_and_locked_never_buy_and_halted_sells_reduce_only() -> None:
    from quant.decide.core import Account, Flags

    fx = DecideFixture.get()
    state = fx.warm_state(active=Label.CHOP, trade=fx.in_trade())
    held = Account(equity=100_000.0, position=300)
    halted = fx.decide(state, flags=Flags(halted=True), account=held)
    assert halted.record.target.value == 0.0
    assert halted.record.target.zero == "halted"
    (sell,) = halted.intents
    assert sell.side is Side.SELL
    assert sell.reduce_only
    assert sell.quantity <= 300
    for flags in (Flags(drift_frozen=True), Flags(daily_loss_lock=True)):
        out = fx.decide(state, flags=flags)
        assert out.intents == (), flags
        assert out.record.target.limited, flags


def _run(
    fx: DecideFixture, state: Any, decisions: Any, *, position: int
) -> list[tuple[Any, Any, int]]:
    """decide over `decisions` in order, each intent filled at once at the decision's price."""
    from quant.decide.core import Account

    out = []
    for d in decisions:
        result = fx.decide(state, decision=d, account=Account(equity=100_000.0, position=position))
        out.append((d, result, position))
        for intent in result.intents:
            position += intent.quantity if intent.side is Side.BUY else -intent.quantity
        state = result.state
    return out


@pytest.mark.spec("A3", "ARCH§5", "SPEC§9", "D9")
def test_through_the_session_the_closing_bar_only_reduces_and_nothing_goes_short() -> None:
    fx = DecideFixture.get()
    schedule = decision_schedule(TradingCalendar.xnys(), DECISION_DAY)
    state = fx.warm_state(active=Label.CHOP, trade=fx.in_trade())
    steps = _run(fx, state, schedule[2:], position=400)
    for d, result, position in steps:
        for intent in result.intents:
            assert intent.side is Side.BUY or intent.quantity <= position
            if d.reduce_only:
                assert intent.side is Side.SELL
                assert intent.reduce_only
        if d.last_fillable:
            price = result.record.target.price
            after = position + sum(
                i.quantity if i.side is Side.BUY else -i.quantity for i in result.intents
            )
            assert after * price <= 0.5 * 100_000.0 + 1e-9
    assert steps[-1][0].reduce_only
    assert math.isfinite(steps[-1][1].record.target.value)
