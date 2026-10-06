"""B3: the scaler, fit on a window and then frozen (spec §3, ARCHITECTURE §7.2)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date, datetime
from functools import cache
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from quant.core.epoch import from_us, to_us
from quant.core.errors import FeatureError
from quant.data.frames import HourlyBar
from quant.features.menu import r1_features
from quant.features.scaler import Scaler
from quant.features.stream import FeatureMatrix, compute_features
from tests.support.features import synthetic_bars, with_prices

NEW_YORK = ZoneInfo("America/New_York")
END = datetime(2017, 1, 1, tzinfo=NEW_YORK)  # the refit time: the window ends here
LATER = datetime(2018, 1, 1, tzinfo=NEW_YORK)


@cache
def _bars() -> tuple[HourlyBar, ...]:
    return tuple(synthetic_bars(date(2016, 1, 4), date(2017, 6, 30)))


def _matrix() -> FeatureMatrix:
    return compute_features(_bars(), r1_features())


def _same(one: Scaler, two: Scaler) -> bool:
    return (
        one.mean.tobytes() == two.mean.tobytes()
        and one.std.tobytes() == two.std.tobytes()
        and one.rows == two.rows
        and one.columns == two.columns
    )


@pytest.mark.spec("SPEC§3", "I1")
def test_the_scaler_ignores_data_after_its_window() -> None:
    matrix = _matrix()
    start = from_us(int(matrix.available_at[0]))
    scaler = Scaler.fit(matrix, start=start, end=END)

    inside = matrix.window(start, END)
    assert 0 < len(inside) < len(matrix)
    assert (inside.available_at < to_us(END)).all()
    assert scaler.rows == len(inside)
    assert scaler.columns == matrix.columns
    assert (scaler.start, scaler.end) == (start, END)
    np.testing.assert_allclose(scaler.mean, inside.values.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(scaler.std, inside.values.std(axis=0), rtol=1e-12)

    # Rows after the window can't reach it: cut off, or replaced by garbage.
    assert _same(Scaler.fit(inside, start=start, end=END), scaler)
    after = matrix.available_at >= to_us(END)
    garbage = matrix.values.copy()
    garbage[after] = 1e9
    junk = FeatureMatrix(matrix.columns, matrix.t_open, matrix.available_at, garbage)
    assert _same(Scaler.fit(junk, start=start, end=END), scaler)

    # End to end: bars changed after the window leave the scaler as it was.
    changed = [with_prices(b, 3.0) if b.available_at >= END else b for b in _bars()]
    assert _same(Scaler.fit(compute_features(changed, r1_features()), start=start, end=END), scaler)

    # Rows before the window's start are left out too.
    later_start = datetime(2016, 9, 1, tzinfo=NEW_YORK)
    narrow = Scaler.fit(matrix, start=later_start, end=END)
    assert narrow.rows == len(matrix.window(later_start, END)) < scaler.rows
    np.testing.assert_allclose(
        narrow.mean, matrix.window(later_start, END).values.mean(axis=0), rtol=1e-12
    )


@pytest.mark.spec("SPEC§3", "ARCH§7.2")
def test_a_fitted_scaler_is_frozen_and_applies_forward() -> None:
    matrix = _matrix()
    start = from_us(int(matrix.available_at[0]))
    scaler = Scaler.fit(matrix, start=start, end=END)
    mean, std = scaler.mean.copy(), scaler.std.copy()
    with pytest.raises(FrozenInstanceError):
        scaler.mean = np.zeros_like(mean)  # type: ignore[misc]
    with pytest.raises(ValueError, match="read-only"):
        scaler.mean[0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        scaler.std[0] = 1.0

    forward = matrix.window(END, LATER)
    z = scaler.transform(forward.values)
    assert z.shape == forward.values.shape
    assert z.dtype == np.float64
    np.testing.assert_allclose(z, (forward.values - mean) / std, rtol=1e-12, atol=0)
    assert scaler.transform(forward.values[0]).tobytes() == z[0].tobytes()
    assert scaler.mean.tobytes() == mean.tobytes()  # applying it changed nothing

    # Inside the window the standardized rows have mean 0 and deviation 1.
    inside = scaler.transform(matrix.window(start, END).values)
    np.testing.assert_allclose(inside.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(inside.std(axis=0), 1.0, rtol=1e-9)

    bad = forward.values[0].copy()
    bad[1] = np.nan
    with pytest.raises(FeatureError):
        scaler.transform(bad)
    with pytest.raises(FeatureError):
        scaler.transform(forward.values[:, :2])


@pytest.mark.spec("SPEC§3")
def test_a_scaler_refuses_a_window_it_cannot_fit() -> None:
    matrix = _matrix()
    start = from_us(int(matrix.available_at[0]))
    with pytest.raises(FeatureError):  # no rows
        Scaler.fit(matrix, start=start, end=start)
    one = from_us(int(matrix.available_at[1]))
    with pytest.raises(FeatureError):  # one row
        Scaler.fit(matrix, start=start, end=one)
    flat = matrix.values.copy()
    flat[:, 2] = 0.5
    constant = FeatureMatrix(matrix.columns, matrix.t_open, matrix.available_at, flat)
    with pytest.raises(FeatureError):  # a constant column has no deviation
        Scaler.fit(constant, start=start, end=END)
    nan = matrix.values.copy()
    nan[3, 0] = np.nan
    with pytest.raises(FeatureError):  # a matrix holds finite rows only
        FeatureMatrix(matrix.columns, matrix.t_open, matrix.available_at, nan)
