"""B8b review, round 1: a red test for each confirmed finding (1-7 from the review).

1. A persisted filter state that `ForwardFilter.restore` would refuse (log_probs with the wrong
   number of entries, not summing to 1, or out of step with `bars`) makes the bar unhealthy
   under rule 6 with cause MODEL; nothing escapes decide.
2. A persisted feature state that doesn't match the release's feature config (another config
   hash), or that `FeatureStream.restore` refuses, makes decide warm a fresh feature stream and
   filter over the view, as on a model swap: the bar's features and filter outputs are a cold
   warm-up's, and nothing escapes decide.
3. From a restored state, the filter steps on every warm, finite row the feature stream
   consumes, in order, as the warm-up does: a bar that became visible only after its own
   decision is stepped before the decision bar, so the restored run's filter outputs equal a
   cold warm-up's at the same tau, bit for bit.
4. Kelly's f* is 0 whenever sd^2 isn't a positive normal float or the fraction isn't finite: it
   never raises and is never inf.
5. The DecisionRecord carries decide's account (equity, position) and flags as data, so they
   change its hash.
6. The closing-bar decision (`reduce_only`) never enters a live trade: the trade state stays
   flat (exits and stops still work; the shadows are unaffected).
7. `Caps` refuses at construction a `label_caps` that misses a Label or has a value outside
   [0, 1], with pydantic's ValidationError as its Field bounds already do for the other caps.
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from quant.core.calendar import TradingCalendar
from quant.core.epoch import to_us
from quant.core.regime import Label
from quant.core.schedule import decision_schedule
from quant.data.market_db import BarStore
from quant.data.policy import Purpose
from tests.support.decide import SWITCHING, DecideFixture, r1_caps, r1_release

# --- 1: a filter snapshot restore would refuse is decided as healthy -----------------------------


def _with_filter(fx: DecideFixture, **update: Any) -> Any:
    warm = fx.warm()
    return warm.model_copy(update={"filter": warm.filter.model_copy(update=update)})


@pytest.mark.spec("SPEC§8", "C2", "ARCH§4.3")
@pytest.mark.xfail(strict=True, reason="red:B8", raises=ValueError)
def test_a_filter_state_with_an_extra_log_prob_is_a_model_failure() -> None:
    from quant.decide.health import Cause

    fx = DecideFixture.get()
    logs = fx.warm().filter.log_probs
    assert len(logs) == fx.model.params.n_states
    record = fx.decide(_with_filter(fx, log_probs=[*logs, 0.0])).record
    assert record.health.cause == Cause.MODEL.value
    assert not record.health.operating


@pytest.mark.spec("SPEC§8", "C2", "ARCH§4.3")
@pytest.mark.parametrize(
    "case",
    [
        "probabilities sum to 2",  # log_probs all 0.0: exp sums to K
        "bars 0 with log_probs set",
        "bars below 0",
        "log_probs None after bars",
    ],
)
@pytest.mark.xfail(strict=True, reason="red:B8", raises=AssertionError)
def test_a_filter_state_restore_refuses_is_a_model_failure(case: str) -> None:
    from quant.decide.health import Cause

    fx = DecideFixture.get()
    warm = fx.warm().filter
    assert warm.bars > 0
    assert warm.log_probs is not None
    updates: dict[str, dict[str, Any]] = {
        "probabilities sum to 2": {"log_probs": [0.0] * len(warm.log_probs)},
        "bars 0 with log_probs set": {"bars": 0},
        "bars below 0": {"bars": -3},
        "log_probs None after bars": {"log_probs": None},
    }
    record = fx.decide(_with_filter(fx, **updates[case])).record
    assert record.health.cause == Cause.MODEL.value
    assert not record.health.operating


# --- 2: a feature state of another config, or one restore refuses, is used as it is ---------------


@pytest.mark.spec("SPEC§6", "ARCH§4.3", "ARCH§4.2")
@pytest.mark.parametrize("perturbed", [False, True], ids=["hash only", "hash and values"])
@pytest.mark.xfail(strict=True, reason="red:B8", raises=AssertionError)
def test_a_feature_state_of_another_config_warms_afresh(perturbed: bool) -> None:
    fx = DecideFixture.get()
    warm = fx.warm()
    update: dict[str, Any] = {"config_hash": "0" * 64}
    if perturbed:
        update.update(
            ema_fast=warm.features.ema_fast * 1.01, prev_close=warm.features.prev_close * 1.01
        )
    state = warm.model_copy(update={"features": warm.features.model_copy(update=update)})
    out = fx.decide(state)
    cold = fx.decide(fx.cold()).record
    assert out.record.health.operating
    assert out.record.features == cold.features
    assert out.record.filter == cold.filter
    assert out.state.features.config_hash == r1_release().manifest.features.config_hash()


@pytest.mark.spec("SPEC§6", "ARCH§4.3", "ARCH§4.2")
@pytest.mark.xfail(strict=True, reason="red:B8", raises=KeyError)
def test_a_feature_state_restore_refuses_warms_afresh() -> None:
    fx = DecideFixture.get()
    warm = fx.warm()
    # No slot keys: FeatureStream.restore refuses it ("slot histories do not fit").
    state = warm.model_copy(
        update={"features": warm.features.model_copy(update={"slot_returns": {}})}
    )
    out = fx.decide(state)
    cold = fx.decide(fx.cold()).record
    assert out.record.features == cold.features
    assert out.record.filter == cold.filter


# --- 3: the filter skips a bar the stream consumed late -----------------------------------------


@pytest.mark.spec("SPEC§6", "ARCH§4.2", "ARCH§4.3")
@pytest.mark.xfail(strict=True, reason="red:B8", raises=AssertionError)
def test_a_late_bar_steps_the_filter_as_the_warm_up_does() -> None:
    fx = DecideFixture.get()
    late, decision = fx.previous, fx.decision
    # The world's store, with bar t (`late`'s) known only a minute after its own decision, so
    # it is invisible at tau_t and visible at tau_{t+1}.
    conn = sqlite3.connect(":memory:", isolation_level=None)
    fx.world.conn.backup(conn)
    changed = conn.execute(
        "UPDATE hourly_bars SET known_at = ? WHERE symbol = 'SPY' AND t_open = ?",
        (to_us(late.tau + timedelta(minutes=1)), to_us(late.bar.t_open)),
    ).rowcount
    assert changed == 1
    assert late.tau + timedelta(minutes=1) < decision.tau
    store = BarStore(conn)

    def view(d: Any) -> Any:
        return store.view(d.tau, as_of=d.tau)

    at_late = fx.decide(fx.cold(), view=view(late), decision=late)
    assert at_late.record.health.cause == "features"  # bar t is missing at its own decision
    restored = fx.decide(at_late.state, view=view(decision), decision=decision).record
    cold = fx.decide(fx.cold(), view=view(decision), decision=decision).record
    assert restored.features == cold.features
    assert restored.filter is not None
    assert restored.filter == cold.filter


# --- 4: Kelly with a tiny sd ----------------------------------------------------------------------


@pytest.mark.spec("C12", "SPEC§9")
@pytest.mark.parametrize(("mean", "sd"), [(1e-160, 1e-170), (1.0, 1e-200)])
@pytest.mark.xfail(strict=True, reason="red:B8", raises=ZeroDivisionError)
def test_kelly_is_zero_when_sd_squared_underflows_to_zero(mean: float, sd: float) -> None:
    from quant.decide.edge import kelly

    assert sd * sd == 0.0
    assert kelly(200, mean, sd, min_bars=100) == 0.0


@pytest.mark.spec("C12", "SPEC§9")
@pytest.mark.parametrize(
    ("mean", "sd"),
    [
        (1e-3, 1e-160),  # sd^2 subnormal, the fraction overflows to inf
        (1e-150, 1e-160),  # sd^2 subnormal, the fraction finite (1e170)
        (1e300, 1e-10),  # sd^2 normal, the fraction overflows to inf
    ],
)
@pytest.mark.xfail(strict=True, reason="red:B8", raises=AssertionError)
def test_kelly_is_zero_when_sd_squared_is_subnormal_or_the_fraction_is_not_finite(
    mean: float, sd: float
) -> None:
    from quant.decide.edge import kelly

    assert kelly(200, mean, sd, min_bars=100) == 0.0


# --- 5: the record leaves out the account and the flags ------------------------------------------


@pytest.mark.spec("ARCH§4.3", "ARCH§13")
@pytest.mark.parametrize("flag", ["drift_frozen", "daily_loss_lock"])
@pytest.mark.xfail(strict=True, reason="red:B8", raises=AssertionError)
def test_the_flags_change_the_record_hash(flag: str) -> None:
    from quant.decide.core import Flags

    fx = DecideFixture.get()
    state = fx.warm_state(active=None)
    plain = fx.decide(state).record
    flagged = fx.decide(state, flags=Flags(**{flag: True})).record
    assert plain.record_hash() != flagged.record_hash()


@pytest.mark.spec("ARCH§4.3", "ARCH§13")
@pytest.mark.xfail(strict=True, reason="red:B8", raises=AssertionError)
def test_the_account_changes_the_record_hash() -> None:
    from quant.decide.core import Account

    fx = DecideFixture.get()
    state = fx.warm_state(active=None)
    small = fx.decide(state, account=Account(equity=100_000.0, position=0)).record
    large = fx.decide(state, account=Account(equity=250_000.0, position=0)).record
    assert small.record_hash() != large.record_hash()


# --- 6: the closing-bar decision enters a trade ---------------------------------------------------

# A FIT session of the synthetic world whose closing-bar decision (16:16 ET) has CHOP's entry
# condition (z <= -1.5) under a healthy filter.
CLOSING_DAY = date(2016, 11, 10)


@pytest.mark.spec("A3", "SPEC§9", "ARCH§4.2")
@pytest.mark.xfail(strict=True, reason="red:B8", raises=AssertionError)
def test_the_closing_bar_decision_never_enters_a_trade() -> None:
    from quant.decide.indicators import IndicatorConfig, IndicatorStream, indicators_for
    from quant.decide.primitives import signal

    fx = DecideFixture.get()
    schedule = decision_schedule(TradingCalendar.xnys(), CLOSING_DAY)
    previous, closing = schedule[-2], schedule[-1]
    assert closing.reduce_only
    assert not previous.reduce_only
    warm = fx.decide(fx.cold(), decision=previous).state
    switching = warm.switching.model_copy(
        update={
            "active": Label.CHOP,
            "challenger": None,
            "count": 0,
            "since_switch": 50,
            "health_streak": SWITCHING.health_recovery_bars,
            "blocked": False,
            "reconfirm": 0,
        }
    )
    state = warm.model_copy(update={"switching": switching})
    assert not state.trade.in_position
    # The scenario: CHOP's entry condition holds on the closing bar.
    pb = r1_release().playbook(Label.CHOP)
    view = fx.view(closing)
    values = indicators_for(
        IndicatorStream(IndicatorConfig.for_playbook(pb)), view, "SPY", closing.bar
    )
    (bar,) = view.hourly("SPY", purpose=Purpose.OPERATE, last=1)
    assert bar.t_open == closing.bar.t_open
    assert bar.complete
    assert signal(pb, values, bar.close).entry
    out = fx.decide(state, decision=closing)
    assert out.record.health.operating
    assert out.record.active is Label.CHOP
    assert out.intents == ()
    assert out.record.trade_reason != "enter"
    assert not out.state.trade.in_position


# --- 7: Caps with a bad label_caps constructs ----------------------------------------------------


@pytest.mark.spec("SPEC§9", "SPEC§10")
@pytest.mark.parametrize(
    "label_caps",
    [
        {Label.CALM_UP: 1.0},
        dict.fromkeys(Label, 1.5),
        dict.fromkeys(Label, -0.1),
    ],
    ids=["missing labels", "above 1", "below 0"],
)
@pytest.mark.xfail(strict=True, reason="red:B8", raises=pytest.fail.Exception)
def test_caps_refuse_bad_label_caps_at_construction(label_caps: dict[Label, float]) -> None:
    from quant.decide.sizing import Caps

    good = r1_caps()
    with pytest.raises(ValidationError):
        Caps(
            label_caps=label_caps,
            max_position=good.max_position,
            overnight_cap=good.overnight_cap,
            single_order_cap=good.single_order_cap,
        )
