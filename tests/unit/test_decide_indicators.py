"""B8a: the streaming indicators: EMAs, ATR(n), and the N-bar mean, s.d. and z (spec §7, PLAN §5).

Each value is checked against a slow reference (tests.support.decide.ref_indicators) within
1e-12 relative, on seeded synthetic bars on the real calendar in 2016 (FIT). The EMAs are the
feature stream's own, bit for bit (ARCHITECTURE §4.2: one implementation of every feature).
The opening bar's true range includes the overnight gap (PLAN §5). Undefined values are None.
The stream snapshots to canonical JSON and continues bit for bit after a restore, and refuses
incomplete and bad bars like the feature stream, leaving its state as it was.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from typing import Any

import pytest

from quant.core.canonical import canonical_dumps, canonical_loads
from quant.core.errors import ConfigError, FeatureError
from quant.core.regime import Label
from tests.support.decide import (
    REPO,
    close_enough,
    market_bars,
    priced,
    priced_series,
    ref_indicators,
)

SPANS = (14, 35, 175)


def config(spans: tuple[int, ...] = SPANS, atr: int | None = 35, window: int | None = 35) -> Any:
    from quant.decide.indicators import IndicatorConfig

    return IndicatorConfig(ema_spans=spans, atr_bars=atr, window=window)


def stream(cfg: Any = None) -> Any:
    from quant.decide.indicators import IndicatorStream

    return IndicatorStream(cfg if cfg is not None else config())


@pytest.mark.spec("SPEC§7", "PLAN§5")
def test_every_value_matches_the_slow_reference() -> None:
    bars = market_bars()
    s = stream()
    reference = ref_indicators(bars, SPANS, 35, 35)
    for bar, ref in zip(bars, reference, strict=True):
        got = s.update(bar)
        assert got.t_open == bar.t_open
        assert set(got.ema) == set(SPANS)
        for span in SPANS:
            assert close_enough(got.ema[span], ref.ema[span])
        assert close_enough(got.atr, ref.atr)
        assert close_enough(got.mean, ref.mean)
        assert close_enough(got.sd, ref.sd)
        assert close_enough(got.z, ref.z)


@pytest.mark.spec("ARCH§4.2", "SPEC§7")
def test_the_emas_are_the_feature_streams_bit_for_bit() -> None:
    from quant.features.menu import r1_features
    from quant.features.stream import FeatureStream

    features = FeatureStream(r1_features())
    s = stream(config(spans=(35, 175), atr=None, window=None))
    for bar in market_bars():
        features.update(bar)
        got = s.update(bar)
        state = features.snapshot()
        assert got.ema[35] == state.ema_fast
        assert got.ema[175] == state.ema_slow


@pytest.mark.spec("PLAN§5", "SPEC§7")
def test_the_opening_bars_true_range_includes_the_overnight_gap() -> None:
    bars = market_bars(9)
    assert bars[6].session != bars[7].session  # 7 bars a session: bar 7 opens the next one
    first = priced(0, 100.0, low=99.0, high=101.5)
    middle = [priced(i, 100.0, low=99.5, high=100.5) for i in range(1, 7)]
    gap_down = priced(7, 95.5, low=94.0, high=96.0)  # the previous close was 100
    s = stream(config(spans=(), atr=14, window=None))
    s.update(first)
    assert s.snapshot().true_ranges == [2.5]  # first bar: H - L
    for bar in middle:
        s.update(bar)
    s.update(replace(gap_down, open=95.0))
    # max(96, 100) - min(94, 100): the gap from 100 counts, not just 96 - 94.
    assert s.snapshot().true_ranges[-1] == 6.0


@pytest.mark.spec("SPEC§7", "PLAN§5")
def test_atr_and_z_are_undefined_until_their_windows_fill() -> None:
    bars = priced_series([100.0 + (i % 5) for i in range(40)])
    s = stream(config(spans=(14,), atr=20, window=30))
    for i, bar in enumerate(bars):
        got = s.update(bar)
        assert (got.atr is None) == (i + 1 < 20)
        assert (got.mean is None) == (got.sd is None) == (got.z is None) == (i + 1 < 30)
        assert got.ema[14] is not None


@pytest.mark.spec("SPEC§7")
def test_z_is_undefined_when_every_close_in_the_window_is_equal() -> None:
    s = stream(config(spans=(), atr=None, window=14))
    got = None
    for bar in priced_series([101.25] * 20):
        got = s.update(bar)
    assert got is not None
    assert got.mean == 101.25
    assert got.sd == 0.0
    assert got.z is None
    moved = s.update(priced(20, 102.0, low=101.5, high=102.5))
    assert moved.z is not None
    assert moved.z > 0


@pytest.mark.spec("SPEC§7")
def test_an_unconfigured_indicator_is_none() -> None:
    s = stream(config(spans=(), atr=None, window=None))
    got = None
    for bar in market_bars(60):
        got = s.update(bar)
    assert got is not None
    assert dict(got.ema) == {}
    assert (got.atr, got.mean, got.sd, got.z) == (None, None, None, None)


@pytest.mark.spec("SPEC§7", "ARCH§4.3")
def test_a_restored_stream_continues_bit_for_bit() -> None:
    from quant.decide.indicators import IndicatorState, IndicatorStream

    bars = market_bars(300)
    whole = stream()
    for bar in bars[:150]:
        whole.update(bar)
    text = canonical_dumps(whole.snapshot())
    restored = IndicatorStream.restore(
        config(),
        IndicatorState.model_validate(canonical_loads(text)),
    )
    for bar in bars[150:]:
        a, b = whole.update(bar), restored.update(bar)
        assert a == b
    assert canonical_dumps(whole.snapshot()) == canonical_dumps(restored.snapshot())


@pytest.mark.spec("SPEC§7")
def test_restore_refuses_a_state_of_another_config() -> None:
    from quant.decide.indicators import IndicatorStream

    s = stream()
    for bar in market_bars(10):
        s.update(bar)
    with pytest.raises(ConfigError):
        IndicatorStream.restore(config(window=21), s.snapshot())
    too_long = s.snapshot().model_copy(update={"closes": [100.0] * 36})
    with pytest.raises(ConfigError):
        IndicatorStream.restore(config(), too_long)


@pytest.mark.spec("SPEC§7", "SPEC§8")
def test_incomplete_and_bad_bars_are_refused_and_leave_the_state() -> None:
    bars = market_bars(20)
    s = stream(config(spans=(14,), atr=14, window=14))
    for bar in bars[:10]:
        s.update(bar)
    before = canonical_dumps(s.snapshot())
    for bad in (
        replace(bars[10], complete=False),
        replace(bars[10], low=bars[10].close + 1.0),  # low above the close
        replace(bars[10], close=float("nan")),
        replace(bars[10], high=-1.0),
        bars[9],  # not after the last bar
    ):
        with pytest.raises(FeatureError):
            s.update(bad)
        assert canonical_dumps(s.snapshot()) == before
    # Skipping bar 10 is a gap: bar 11's true range is taken from bar 9's close.
    got = s.update(bars[11])
    expected = max(bars[11].high, bars[9].close) - min(bars[11].low, bars[9].close)
    assert s.snapshot().true_ranges[-1] == expected
    assert got.t_open == bars[11].t_open


@pytest.mark.spec("SPEC§7")
@pytest.mark.parametrize(
    ("spans", "atr", "window"), [((35, 14), 35, 35), ((35, 35), 35, 35), ((0,), 35, 35)]
)
def test_the_stream_refuses_a_bad_config(
    spans: tuple[int, ...], atr: int | None, window: int | None
) -> None:
    from quant.decide.indicators import IndicatorConfig, IndicatorStream

    with pytest.raises(ConfigError):
        IndicatorStream(IndicatorConfig(ema_spans=spans, atr_bars=atr, window=window))


@pytest.mark.spec("SPEC§7", "PLAN§5")
def test_each_r1_playbook_gets_the_indicators_it_needs() -> None:
    from quant.decide.indicators import IndicatorConfig
    from quant.decide.playbook import load_release

    release = load_release(REPO)
    calm = IndicatorConfig.for_playbook(release.playbook(Label.CALM_UP))
    assert (calm.ema_spans, calm.atr_bars, calm.window) == ((35, 175), 35, None)
    chop = IndicatorConfig.for_playbook(release.playbook(Label.CHOP))
    assert (chop.ema_spans, chop.atr_bars, chop.window) == ((), 35, 35)
    for label in (Label.STRESS, Label.CRASH):
        flat = IndicatorConfig.for_playbook(release.playbook(label))
        assert (flat.ema_spans, flat.atr_bars, flat.window) == ((), None, None)


@pytest.mark.spec("SPEC§7")
def test_a_close_below_ema_invalidation_adds_its_span() -> None:
    from quant.decide.indicators import IndicatorConfig
    from quant.decide.playbook import PLAYBOOK, load_release

    calm = load_release(REPO).playbook(Label.CALM_UP)
    extra = (
        {"kind": "close_below_ema", "span": 50},
        {"kind": "close_below_ema", "span": 35},  # already a trend span: no duplicate
    )
    data = calm.model_dump(mode="json")
    data["invalidation"] = [*data["invalidation"], *extra]
    with_ema = PLAYBOOK.validate_json(canonical_dumps(data))
    assert IndicatorConfig.for_playbook(with_ema).ema_spans == (35, 50, 175)


@pytest.mark.spec("SPEC§7", "SPEC§8")
def test_indicators_for_reads_the_point_in_time_view() -> None:
    from quant.core.calendar import TradingCalendar
    from quant.core.schedule import decision_schedule
    from quant.data.policy import Purpose
    from quant.decide.indicators import IndicatorStream, indicators_for
    from tests.support.bars import SYMBOL, store_with_days

    store = store_with_days(date(2016, 1, 4), date(2016, 2, 12))
    calendar = TradingCalendar.xnys()
    days = [date(2016, 2, 10), date(2016, 2, 11)]
    decisions = [d for day in days for d in decision_schedule(calendar, day)]
    live = IndicatorStream(config())
    for d in decisions:
        view = store.view(d.tau, as_of=d.tau)
        got = indicators_for(live, view, SYMBOL, d.bar)
        # The same as a fresh stream fed every complete bar up to the decision's bar.
        fresh = IndicatorStream(config())
        expected = None
        for b in view.hourly(SYMBOL, purpose=Purpose.OPERATE):
            if b.t_open <= d.bar.t_open and b.complete:
                expected = fresh.update(b)
        assert got == expected
    # The decision bar already consumed: FeatureError, as features_for does.
    last = decisions[-1]
    with pytest.raises(FeatureError):
        indicators_for(live, store.view(last.tau, as_of=last.tau), SYMBOL, last.bar)
