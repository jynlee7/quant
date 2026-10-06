"""B3 second review: the stream's overflows, the scaler's guards, restore's checks and the batch's
NaN rows (BUILD_LOG, B3)."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime
from functools import cache
from typing import Any

import numpy as np
import pytest
from structlog.testing import capture_logs

from quant.core.canonical import canonical_dumps, canonical_loads
from quant.core.epoch import to_us
from quant.core.errors import ConfigError, FeatureError
from quant.data.frames import HourlyBar
from quant.features.menu import r1_features
from quant.features.scaler import Scaler
from quant.features.stream import FeatureMatrix, FeatureState, FeatureStream, compute_features
from tests.support.features import synthetic_bars

START = datetime(2016, 1, 1, tzinfo=UTC)
END = datetime(2017, 1, 1, tzinfo=UTC)


@cache
def _bars() -> tuple[HourlyBar, ...]:
    return tuple(synthetic_bars(date(2016, 8, 1), date(2017, 1, 31)))


def _priced(bar: HourlyBar, low: float, high: float, close: float) -> HourlyBar:
    return replace(bar, open=close, high=high, low=low, close=close)


def _matrix(values: list[list[float]], columns: tuple[str, ...] = ("a",)) -> FeatureMatrix:
    times = to_us(START) + np.arange(len(values), dtype=np.int64) * 3_600_000_000
    return FeatureMatrix(columns, times, times, np.array(values, dtype=np.float64))


def _overflow(kind: str) -> tuple[list[HourlyBar], HourlyBar]:
    """Bars a stream consumes, then a valid bar whose features overflow float64."""
    bars = _bars()
    if kind == "volume-ratio":  # V_t / median volume is past float64's largest value
        before = [replace(bar, volume=1e-300) for bar in bars[:20]]
        return before, replace(bars[20], volume=1e300)
    before = [*bars[:19], _priced(bars[19], 1e-300, 1e-300, 1e-300)]
    after = {
        "return-up": _priced(bars[20], 1e300, 1e300, 1e300),  # C_t / C_{t-1} = 1e600
        "return-down": _priced(bars[20], 1e-300, 1e-300, 1e-300),
        "range": _priced(bars[20], 1e-300, 1e300, 1e-300),  # (H - L) / C_{t-1} = 1e600
    }[kind]
    if kind == "return-down":  # C_t / C_{t-1} = 1e-600, which float64 rounds to 0
        before[-1] = _priced(bars[19], 1e300, 1e300, 1e300)
    return before, after


@pytest.mark.xfail(strict=True, reason="red:B3", raises=pytest.fail.Exception)
@pytest.mark.spec("SPEC§3", "ARCH§4.3")
@pytest.mark.parametrize("kind", ["return-up", "return-down", "range", "volume-ratio"])
def test_a_bar_whose_features_overflow_is_refused_and_the_state_stays_canonical(
    kind: str,
) -> None:
    # R2: finite positive prices whose ratio float64 can't hold. The return, the range ratio or
    # V_t / median would be infinite or zero, so the bar is refused like bad data, with the
    # state as it was: never a NaN or an infinity in the state or the row.
    config = r1_features()
    before, bad = _overflow(kind)
    stream = FeatureStream(config)
    for bar in before:
        stream.update(bar)
    snapshot = canonical_dumps(stream.snapshot())
    with pytest.raises(FeatureError):
        stream.update(bad)
    assert canonical_dumps(stream.snapshot()) == snapshot
    FeatureStream.restore(config, FeatureState.model_validate(canonical_loads(snapshot)))


@pytest.mark.xfail(strict=True, reason="red:B3", raises=AssertionError)
@pytest.mark.spec("SPEC§3", "ARCH§4.3")
def test_integer_bar_values_give_a_float_state_that_round_trips() -> None:
    # C5: int volumes and prices are stored as float64, so the snapshot's canonical JSON is the
    # same before and after a JSON round trip, and its hash doesn't change on restore.
    config = r1_features()
    bars = [
        replace(bar, open=round(bar.close), high=round(bar.close) + 1, low=round(bar.close) - 1)
        for bar in _bars()[:100]
    ]
    bars = [replace(bar, close=round(bar.close), volume=round(bar.volume)) for bar in bars]
    stream = FeatureStream(config)
    for bar in bars:
        stream.update(bar)
    state = stream.snapshot()
    blob = canonical_dumps(state)
    assert canonical_dumps(FeatureState.model_validate(canonical_loads(blob))) == blob
    assert type(state.prev_close) is float
    assert type(state.ema_fast) is float
    assert all(type(v) is float for values in state.slot_volumes.values() for v in values)


@pytest.mark.xfail(strict=True, reason="red:B3", raises=pytest.fail.Exception)
@pytest.mark.spec("SPEC§3")
@pytest.mark.parametrize("value", [0.1, 1e-20, -3.3, 123456.789])
def test_a_near_constant_column_is_refused(value: float) -> None:
    # R3/C1: float64 can't hold 0.1 exactly, so the mean of a constant 0.1 column is off by a
    # rounding error and numpy's std is about 1e-17, not 0. A deviation that small next to the
    # mean is rounding, not variation, and z-scores on it would be about 1e16.
    with pytest.raises(FeatureError):
        Scaler.fit(_matrix([[value]] * 7), start=START, end=END)
    with pytest.raises(FeatureError):
        Scaler(("a",), np.array([value]), np.array([abs(value) * 1e-15]), START, END, 7)


@pytest.mark.spec("SPEC§3")
def test_a_small_but_real_deviation_is_kept() -> None:
    # The tolerance is relative: a column of tiny values that do vary still fits.
    scaler = Scaler.fit(_matrix([[1e-12], [3e-12], [2e-12]]), start=START, end=END)
    assert scaler.std[0] > 0
    centered = Scaler.fit(_matrix([[-1e-9], [1e-9]]), start=START, end=END)
    assert centered.std[0] == pytest.approx(1e-9, rel=1e-12)


@pytest.mark.xfail(strict=True, reason="red:B3", raises=RuntimeWarning)
@pytest.mark.spec("SPEC§3")
def test_a_standardized_value_that_overflows_is_a_model_error() -> None:
    # R4: finite parameters and a finite input can still give an infinite z-score. It must be
    # the model error, not a numpy warning or an infinity handed to the filter.
    scaler = Scaler.fit(_matrix([[-1e-160], [1e-160]]), start=START, end=END)
    with pytest.raises(FeatureError):
        scaler.transform(np.array([1e308]))


@pytest.mark.xfail(strict=True, reason="red:B3", raises=TypeError)
@pytest.mark.spec("SPEC§3")
def test_the_scaler_refuses_columns_it_was_not_fit_on() -> None:
    # C9: a vector of the same width from another config is scaled only if its names match the
    # columns the scaler was fit on. Input that isn't a float array is a FeatureError too.
    scaler = Scaler.fit(_matrix([[1.0, 2.0], [3.0, 5.0]], ("ret", "rvol")), start=START, end=END)
    row = np.array([2.0, 3.0])
    assert scaler.transform(row, columns=("ret", "rvol")).tolist() == [0.0, -1 / 3]
    for columns in (("ret", "trend"), ("rvol", "ret"), ("ret",)):
        with pytest.raises(FeatureError):
            scaler.transform(row, columns=columns)
    bad: Any
    for bad in ([2.0, 3.0], "23", np.array([2, 3]), np.array([[[2.0, 3.0]]])):
        with pytest.raises(FeatureError):
            scaler.transform(bad, columns=("ret", "rvol"))


@pytest.mark.xfail(strict=True, reason="red:B3", raises=FeatureError)
@pytest.mark.spec("SPEC§3")
def test_a_batch_drops_a_warm_row_with_a_nan_feature_and_logs_it() -> None:
    # C7: one flat bar (H == L, so F3 is ln 0) deep in a fit window must not abort the fit. A fit
    # can't use the row, so it is dropped and logged; the rows around it stay as they were. At
    # decision time the same row is still a model error (FeatureRow.vector, features_for).
    config = r1_features()
    bars = list(_bars())
    i = 500
    flat = replace(bars[i], open=bars[i].close, high=bars[i].close, low=bars[i].close)
    changed = [*bars[:i], flat, *bars[i + 1 :]]
    clean = compute_features(bars, config)
    with capture_logs() as logs:
        matrix = compute_features(changed, config)
    assert to_us(flat.t_open) not in matrix.t_open.tolist()
    assert len(matrix) == len(clean) - 1
    keep = clean.t_open != to_us(flat.t_open)
    assert matrix.t_open.tobytes() == clean.t_open[keep].tobytes()
    # The flat bar's close and place in the histories still count for the bars after it.
    stream = FeatureStream(config)
    rows = [stream.update(bar) for bar in changed]
    assert not np.isfinite(rows[i].values).all()
    expected = np.array([row.values for row in rows if row.warm and row is not rows[i]])
    assert matrix.values.tobytes() == expected.tobytes()
    dropped = [entry for entry in logs if entry["event"] == "feature row dropped"]
    assert len(dropped) == 1
    assert dropped[0]["log_level"] == "warning"
    assert dropped[0]["t_open"] == flat.t_open.isoformat()
    assert dropped[0]["columns"] == [config.columns[2]]


def _restore_cases(state: FeatureState) -> dict[str, dict[str, Any]]:
    one = "09:30"
    volumes, ranges, returns = state.slot_volumes, state.slot_ranges, state.slot_returns
    return {
        "negative-volume": {"slot_volumes": {**volumes, one: [-1.0, *volumes[one][1:]]}},
        "negative-range": {"slot_ranges": {**ranges, one: [-0.01, *ranges[one][1:]]}},
        "session-not-a-date": {"last_session": "yesterday"},
        "session-not-iso": {"last_session": "2016-8-1"},
        "sessions-past-the-last-bar": {"sessions": state.sessions + 3},
        "numbers-without-volumes": {"slot_volumes": {**volumes, one: volumes[one][1:]}},
        "numbers-without-returns": {
            "slot_returns": {**returns, one: returns[one][2:]},
            "slot_ranges": {**ranges, one: ranges[one][2:]},
        },
        "ranges-not-returns": {"slot_ranges": {**ranges, one: ranges[one][1:]}},
    }


@pytest.mark.xfail(strict=True, reason="red:B3", raises=pytest.fail.Exception)
@pytest.mark.spec("SPEC§3", "ARCH§4.3")
@pytest.mark.parametrize("case", sorted(_restore_cases(FeatureStream(r1_features()).snapshot())))
def test_restore_refuses_values_out_of_range_or_out_of_step(case: str) -> None:
    # C8: restore checked shapes and finiteness only. Volumes and ranges are never negative,
    # the last session is an ISO date, the session count is the last bar's session number, and
    # every slot bar in the window has its volume, and its return and range unless it was the
    # stream's first bar.
    config = r1_features()
    stream = FeatureStream(config)
    for bar in _bars()[:50]:
        stream.update(bar)
    state = stream.snapshot()
    FeatureStream.restore(config, state)  # the state as taken restores
    with pytest.raises(ConfigError):
        FeatureStream.restore(config, state.model_copy(update=_restore_cases(state)[case]))
