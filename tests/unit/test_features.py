"""B3: streaming features F1–F5 (spec §3, D3, ARCHITECTURE §4.2–§4.3, §6.1).

Bars are synthetic, built by the B2 builder on the real calendar, in FIT-era dates. The span
2016-08-01 to 2017-01-31 holds the 60-session warm-up and the half day 2016-11-25 after it.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import replace
from datetime import date
from functools import cache
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from quant.core.calendar import Slot
from quant.core.canonical import canonical_dumps, canonical_loads
from quant.core.epoch import to_us
from quant.core.errors import ConfigError, FeatureError, ModelError
from quant.data.frames import HourlyBar
from quant.features.menu import FeatureConfig, Range, Ret, Rvol, Trend, Volume, r1_features
from quant.features.scaler import Scaler
from quant.features.stream import (
    WARMUP_SESSIONS,
    FeatureState,
    FeatureStream,
    compute_features,
)
from tests.support.features import (
    bits,
    reference_features,
    stream_rows,
    synthetic_bars,
)

NEW_YORK = ZoneInfo("America/New_York")
HALF_DAY = date(2016, 11, 25)
HOURLY = (Slot.S1030, Slot.S1130, Slot.S1230, Slot.S1330, Slot.S1430)


@cache
def _bars() -> tuple[HourlyBar, ...]:
    return tuple(synthetic_bars(date(2016, 8, 1), date(2017, 1, 31)))


def _short() -> FeatureConfig:
    """Every transform, every parameter at its lower bound."""
    return FeatureConfig(
        slot_sessions=20,
        ret=Ret(),
        rvol=Rvol(bars=14),
        range=Range(),
        volume=Volume(),
        trend=Trend(fast=14, slow=70),
    )


def _minimal() -> FeatureConfig:
    """F1 and F2 only, at their upper bounds: the slot histories never fill in this span."""
    return FeatureConfig(slot_sessions=120, ret=Ret(), rvol=Rvol(bars=70))


CONFIGS = {"r1": r1_features, "short": _short, "minimal": _minimal}


@pytest.mark.spec("SPEC§3", "D3", "ARCH§4.2", "ARCH§6.1")
@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_streaming_matches_the_reference(name: str) -> None:
    config = CONFIGS[name]()
    bars = _bars()
    stream = FeatureStream(config)
    rows = [stream.update(bar) for bar in bars]
    reference = reference_features(bars, config)

    assert config.columns == reference.columns
    assert [row.t_open for row in rows] == [bar.t_open for bar in bars]
    assert [row.slot for row in rows] == [bar.slot for bar in bars]
    assert tuple(row.warm for row in rows) == reference.warm
    got = np.array([row.values for row in rows], dtype=np.float64)
    # Warm-up rows included: partial histories and undefined values must match too.
    np.testing.assert_allclose(got, reference.values, rtol=0, atol=1e-12, equal_nan=True)
    assert np.isnan(got[0, :2]).all()  # the first bar has no return
    warm = np.array(reference.warm)
    assert warm.any()
    assert np.isfinite(got[warm]).all()


@pytest.mark.spec("SPEC§3", "ARCH§14")
def test_two_runs_agree_bit_for_bit() -> None:
    bars = _bars()
    first = compute_features(bars, r1_features())
    second = compute_features(list(bars), r1_features())
    assert first.columns == second.columns
    assert first.t_open.tobytes() == second.t_open.tobytes()
    assert first.available_at.tobytes() == second.available_at.tobytes()
    assert first.values.tobytes() == second.values.tobytes()
    one, two = stream_rows(bars, r1_features()), stream_rows(bars, r1_features())
    assert [bits(row) for row in one.values()] == [bits(row) for row in two.values()]


@pytest.mark.spec("SPEC§3", "ARCH§4.2", "ARCH§4.3")
def test_a_snapshot_is_canonical_json_of_fixed_size_and_restores_bit_for_bit() -> None:
    config = r1_features()
    bars = synthetic_bars(date(2016, 1, 4), date(2016, 12, 30))
    fresh = FeatureStream(config).snapshot()
    assert fresh.config_hash == config.config_hash()
    assert (fresh.last_t_open, fresh.last_session, fresh.prev_close) == (None, None, None)
    assert fresh.sessions == 0
    assert fresh.returns == []
    for history in (
        fresh.slot_numbers,
        fresh.slot_returns,
        fresh.slot_ranges,
        fresh.slot_volumes,
    ):
        assert history == {slot.value: [] for slot in Slot}

    whole = FeatureStream(config)
    expected = [whole.update(bar) for bar in bars]
    half = len(bars) // 2
    first = FeatureStream(config)
    for bar in bars[:half]:
        first.update(bar)
    blob = canonical_dumps(first.snapshot())
    state = FeatureState.model_validate(canonical_loads(blob))
    assert state == first.snapshot()
    resumed = FeatureStream.restore(config, state)
    assert resumed.last_t_open == bars[half - 1].t_open
    rows = [resumed.update(bar) for bar in bars[half:]]
    assert [bits(row) for row in rows] == [bits(row) for row in expected[half:]]
    assert canonical_dumps(resumed.snapshot()) == canonical_dumps(whole.snapshot())

    # Fixed size: after a year each slot keeps exactly its bars of the last N sessions, however
    # long the stream runs: N, or N - 1 for the afternoon slots a half day in them lacks.
    final = whole.snapshot()
    assert final.sessions == len({bar.session for bar in bars}) == whole.sessions
    assert final.last_t_open == to_us(bars[-1].t_open)
    assert final.last_session == bars[-1].session.isoformat()
    assert final.prev_close == bars[-1].close
    assert len(final.returns) == config.rvol.bars
    n = config.slot_sessions
    recent = sorted({bar.session for bar in bars})[-n:]
    assert HALF_DAY in recent
    numbers = {slot.value: [] for slot in Slot}
    for bar in bars:
        if bar.session in recent:
            numbers[bar.slot.value].append(final.sessions - n + 1 + recent.index(bar.session))
    assert final.slot_numbers == numbers
    assert [len(numbers[slot.value]) for slot in (Slot.S1230, Slot.S1330, Slot.S1430)] == [
        n - 1
    ] * 3
    assert len(numbers[Slot.CLOSE.value]) == n
    for history in (final.slot_returns, final.slot_ranges, final.slot_volumes):
        assert sorted(history) == sorted(slot.value for slot in Slot)
        assert {slot: len(values) for slot, values in history.items()} == {
            slot: len(values) for slot, values in numbers.items()
        }
    assert final.ema_fast is not None
    assert final.ema_slow is not None


@pytest.mark.spec("SPEC§3", "ARCH§4.3")
def test_a_snapshot_restores_only_under_its_own_config() -> None:
    config = r1_features()
    stream = FeatureStream(config)
    for bar in _bars()[:50]:
        stream.update(bar)
    state = stream.snapshot()
    with pytest.raises(ConfigError):
        FeatureStream.restore(_short(), state)
    too_long = state.model_copy(update={"returns": state.returns + [0.01] * 40})
    with pytest.raises(ConfigError):
        FeatureStream.restore(config, too_long)
    no_close = state.model_copy(update={"prev_close": None})
    with pytest.raises(ConfigError):
        FeatureStream.restore(config, no_close)
    numbers = state.slot_numbers
    for bad_numbers in (
        {**numbers, "09:30": numbers["09:30"][1:]},  # fewer numbers than values
        {**numbers, "09:30": numbers["09:30"][::-1]},  # out of order
        {**numbers, "09:30": [*numbers["09:30"][:-1], state.sessions + 1]},  # a future session
        {key: value for key, value in numbers.items() if key != "CLOSE"},  # a slot missing
    ):
        with pytest.raises(ConfigError):
            FeatureStream.restore(config, state.model_copy(update={"slot_numbers": bad_numbers}))
    off = FeatureStream(_minimal())
    for bar in _bars()[:50]:
        off.update(bar)
    with_ranges = off.snapshot().model_copy(update={"slot_ranges": state.slot_ranges})
    with pytest.raises(ConfigError):  # F3 is off, so it has no histories
        FeatureStream.restore(_minimal(), with_ranges)


@pytest.mark.spec("SPEC§3", "ARCH§6.1")
def test_the_close_slot_is_normalized_against_close_history_only() -> None:
    config = r1_features()
    n = config.slot_sessions
    bars = _bars()
    rows = stream_rows(bars, config)
    k = next(i for i, bar in enumerate(bars) if bar.session == HALF_DAY and bar.slot is Slot.CLOSE)
    half_close = bars[k]
    opens_at = half_close.t_open.astimezone(NEW_YORK)
    assert (opens_at.hour, opens_at.minute) == (12, 30)  # the half day's 30-minute final bar

    # Each warm CLOSE bar against the CLOSE bars before it, by hand.
    checked, with_half_day = 0, 0
    for i, bar in enumerate(bars):
        row = rows[bar.t_open]
        if bar.slot is not Slot.CLOSE or not row.warm:
            continue
        prior = [j for j in range(i) if bars[j].slot is Slot.CLOSE][-n:]
        returns = [math.log(bars[j].close / bars[j - 1].close) for j in prior]
        ranges = [(bars[j].high - bars[j].low) / bars[j - 1].close for j in prior]
        volumes = [bars[j].volume for j in prior]
        ret = math.log(bar.close / bars[i - 1].close)
        sigma = math.sqrt(math.fsum(v * v for v in returns) / len(returns))
        rng = math.log((bar.high - bar.low) / bars[i - 1].close)
        expect = {
            "ret": ret / sigma,
            "range": rng - math.log(statistics.median(ranges)),
            "volume": math.log(bar.volume / statistics.median(volumes)),
        }
        got = dict(zip(config.columns, row.values, strict=True))
        for name, value in expect.items():
            assert got[name] == pytest.approx(value, rel=0, abs=1e-12), (bar.t_open, name)
        checked += 1
        with_half_day += k in prior
    assert checked > 40
    assert with_half_day > 30
    assert rows[half_close.t_open].warm

    # The half day's CLOSE bar feeds CLOSE history, and no hourly slot's.
    moved = list(bars)
    raised = half_close.close * 1.01
    moved[k] = replace(half_close, close=raised, high=max(half_close.high, raised))
    moved_rows = stream_rows(moved, config)
    after = bars[k + 1 :]
    by_slot = {slot: [bar.t_open for bar in after if bar.slot is slot] for slot in Slot}
    slot_normalized = [config.columns.index(name) for name in ("ret", "range", "volume")]
    for slot in HOURLY:
        for t_open in by_slot[slot]:
            same = [rows[t_open].values[c] for c in slot_normalized]
            assert same == [moved_rows[t_open].values[c] for c in slot_normalized], t_open
    later_close = by_slot[Slot.CLOSE][:n]
    assert later_close
    assert all(rows[t].values[0] != moved_rows[t].values[0] for t in later_close)


@pytest.mark.spec("SPEC§3", "ARCH§6.1")
def test_a_slot_history_spans_the_prior_n_sessions_across_a_half_day() -> None:
    # Spec §3: a slot's history is its values "over the prior 60 sessions", counted in
    # sessions, not in bars. The half day 2016-11-25 has no 12:30, 13:30 or 14:30 bar, so for
    # 60 sessions after it those slots' histories hold 59 values, not the last 60 bars.
    config = r1_features()
    n = config.slot_sessions
    bars = _bars()
    rows = stream_rows(bars, config)
    number: dict[date, int] = {}
    for bar in bars:
        number.setdefault(bar.session, len(number) + 1)

    checked, short = 0, 0
    for i, bar in enumerate(bars):
        row = rows[bar.t_open]
        if not row.warm:
            continue
        k = number[bar.session]
        prior = [
            j for j in range(i) if bars[j].slot is bar.slot and k - n <= number[bars[j].session] < k
        ]
        # The stream's first bar has a volume but no return or range ratio.
        returns = [math.log(bars[j].close / bars[j - 1].close) for j in prior if j > 0]
        ranges = [(bars[j].high - bars[j].low) / bars[j - 1].close for j in prior if j > 0]
        volumes = [bars[j].volume for j in prior]
        sigma = math.sqrt(math.fsum(v * v for v in returns) / len(returns))
        expect = {
            "ret": math.log(bar.close / bars[i - 1].close) / sigma,
            "range": math.log((bar.high - bar.low) / bars[i - 1].close)
            - math.log(statistics.median(ranges)),
            "volume": math.log(bar.volume / statistics.median(volumes)),
        }
        got = dict(zip(config.columns, row.values, strict=True))
        for name, value in expect.items():
            assert got[name] == pytest.approx(value, rel=0, abs=1e-12), (bar.t_open, name)
        checked += 1
        short += len(prior) == n - 1
    assert checked > 300
    assert short > 100  # the afternoon slots after the half day


@pytest.mark.spec("SPEC§3")
@pytest.mark.parametrize("slot_sessions", [20, 60, 120])
def test_the_warm_up_drops_60_sessions(slot_sessions: int) -> None:
    # The half day 2016-11-25 falls inside the warm-up and counts as one session.
    bars = synthetic_bars(date(2016, 10, 3), date(2017, 1, 31))
    config = r1_features().model_copy(update={"slot_sessions": slot_sessions})
    stream = FeatureStream(config)
    rows = [stream.update(bar) for bar in bars]
    sessions = sorted({bar.session for bar in bars})
    assert WARMUP_SESSIONS == 60
    assert HALF_DAY in sessions[:60]
    first_warm = sessions[60]
    assert [row.warm for row in rows] == [bar.session >= first_warm for bar in bars]
    assert stream.sessions == len(sessions)

    matrix = compute_features(bars, config)
    warm_bars = [bar for bar in bars if bar.session >= first_warm]
    assert len(matrix) == len(warm_bars)
    assert matrix.t_open.tolist() == [to_us(bar.t_open) for bar in warm_bars]
    assert matrix.available_at.tolist() == [to_us(bar.available_at) for bar in warm_bars]
    assert matrix.columns == config.columns

    last_cold = len(bars) - len(warm_bars) - 1
    assert np.isfinite(rows[last_cold].values).all()
    with pytest.raises(FeatureError):  # finite, but still the warm-up
        rows[last_cold].vector()
    assert rows[last_cold + 1].vector().tobytes() == matrix.values[0].tobytes()


def _bad(bar: HourlyBar, kind: str) -> HourlyBar:
    """`bar` made bad in one way the stream must refuse."""
    changes: dict[str, dict[str, object]] = {
        "nan-close": {"close": math.nan},
        "inf-high": {"high": math.inf},
        "zero-low": {"low": 0.0},
        "negative-close": {"close": -bar.close, "low": -bar.close},
        "high-below-close": {"high": bar.close - 0.01},
        "low-above-open": {"low": bar.open + 0.01},
        "negative-volume": {"volume": -1.0},
        "nan-volume": {"volume": math.nan},
        "incomplete": {"complete": False},
    }
    return replace(bar, **changes[kind])  # type: ignore[arg-type]


@pytest.mark.spec("SPEC§3", "I1")
@pytest.mark.parametrize(
    "kind",
    [
        "nan-close",
        "inf-high",
        "zero-low",
        "negative-close",
        "high-below-close",
        "low-above-open",
        "negative-volume",
        "nan-volume",
        "incomplete",
        "repeated",
        "earlier",
    ],
)
def test_nan_or_missing_input_is_a_model_error(kind: str) -> None:
    assert issubclass(FeatureError, ModelError)
    bars = _bars()
    stream = FeatureStream(r1_features())
    for bar in bars[:20]:
        stream.update(bar)
    before = canonical_dumps(stream.snapshot())
    if kind == "repeated":
        bad = bars[19]
    elif kind == "earlier":
        bad = bars[10]
    else:
        bad = _bad(bars[20], kind)
    with pytest.raises(FeatureError):
        stream.update(bad)
    assert canonical_dumps(stream.snapshot()) == before

    # The stream goes on from its previous bar, as if the bad one had never come.
    clean = FeatureStream(r1_features())
    for bar in bars[:20]:
        clean.update(bar)
    assert bits(stream.update(bars[21])) == bits(clean.update(bars[21]))


@pytest.mark.spec("SPEC§3", "I1")
def test_a_batch_skips_a_missing_bar_and_refuses_a_bad_one() -> None:
    bars = list(_bars())
    i = 500  # well after the warm-up
    assert bars[i].session > sorted({bar.session for bar in bars})[60]
    skipped = compute_features([*bars[:i], *bars[i + 1 :]], r1_features())
    incomplete = compute_features(
        [*bars[:i], replace(bars[i], complete=False), *bars[i + 1 :]], r1_features()
    )
    assert incomplete.t_open.tobytes() == skipped.t_open.tobytes()
    assert incomplete.values.tobytes() == skipped.values.tobytes()
    assert to_us(bars[i].t_open) not in incomplete.t_open.tolist()
    with pytest.raises(FeatureError):
        compute_features([*bars[:i], _bad(bars[i], "nan-close"), *bars[i + 1 :]], r1_features())


@pytest.mark.spec("SPEC§3")
def test_a_nan_feature_is_a_model_error() -> None:
    # Flat closes: every return is 0, so F1's RMS and F2's sum are 0 and both are undefined.
    flat = [
        replace(bar, open=200.0, high=200.5, low=199.5, close=200.0)
        for bar in synthetic_bars(date(2016, 9, 1), date(2016, 12, 30))
    ]
    stream = FeatureStream(r1_features())
    rows = [stream.update(bar) for bar in flat]
    warm = [row for row in rows if row.warm]
    assert warm
    for row in warm:
        assert math.isnan(row.values[0])
        assert math.isnan(row.values[1])
        assert np.isfinite(row.values[2:]).all()
    with pytest.raises(FeatureError):
        warm[0].vector()
    # A fit can't use a NaN row, so the batch drops it (and logs it); with every warm row NaN
    # none is left, and fitting a scaler on the window is the model error.
    matrix = compute_features(flat, r1_features())
    assert len(matrix) == 0
    with pytest.raises(FeatureError):
        Scaler.fit(matrix, start=warm[0].available_at, end=warm[-1].available_at)


@pytest.mark.full
@pytest.mark.spec("SPEC§3", "ARCH§4.2")
def test_ten_years_of_features_compute_in_under_five_seconds() -> None:
    bars = synthetic_bars(date(2016, 1, 1), date(2025, 12, 31), seed=7)
    started = time.perf_counter()
    matrix = compute_features(bars, r1_features())
    elapsed = time.perf_counter() - started
    sessions = sorted({bar.session for bar in bars})
    assert len(matrix) == sum(bar.session >= sessions[60] for bar in bars)
    assert np.isfinite(matrix.values).all()
    assert matrix.t_open[-1] == to_us(bars[-1].t_open)
    assert elapsed < 5, f"{elapsed:.2f}s"
