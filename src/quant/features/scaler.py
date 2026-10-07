"""The scaler: z-score parameters fit on a model's training window, then frozen (spec §3).

They are fit at refit time on the warm feature rows of the model's window and never recomputed
with data from after it. The model applies them forward until the next refit fits a new scaler,
and its artifact stores `mean` and `std` in params.npz (ARCHITECTURE §7.2, §7.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Final

import numpy as np

from quant.core.clock import ensure_utc
from quant.core.errors import FeatureError
from quant.data.frames import Floats
from quant.features.stream import FeatureMatrix, freeze_arrays

# A column whose standard deviation is no more than this times its |mean| is constant: what is
# left is the rounding error of a mean float64 can't hold exactly (about 1e-16 relative), and
# z-scores on it would be about 1e16. Real feature columns vary far more than this.
STD_RTOL: Final = 1e-9


@dataclass(frozen=True)
class Scaler:
    """Per-column mean and standard deviation of one window of feature rows.

    `start` and `end` are the window, [start, end) on the rows' `available_at`, and `rows` is
    how many rows it held. The scaler never changes once built: the dataclass is frozen and
    the arrays are read-only.
    """

    columns: tuple[str, ...]
    mean: Floats
    std: Floats
    start: datetime
    end: datetime
    rows: int

    def __post_init__(self) -> None:
        """Raise FeatureError unless `columns` is a non-empty tuple of names, `mean` and `std`
        are 1-D float64 arrays with one finite value per column, every std is > 0 and more than
        STD_RTOL times its column's |mean|, `rows` >= 2, and start < end. `start` and `end` must
        be timezone-aware (a naive one raises ClockError) and are kept in UTC. Replace the arrays
        with read-only copies (`freeze_arrays`), so writing to them raises ValueError."""
        start, end = ensure_utc(self.start), ensure_utc(self.end)
        if (
            not isinstance(self.columns, tuple)
            or not self.columns
            or any(not isinstance(name, str) or not name for name in self.columns)
            or self.rows < 2
            or start >= end
        ):
            raise FeatureError("scaler needs columns, at least two rows and start < end")
        for name in ("mean", "std"):
            values = getattr(self, name)
            if (
                not isinstance(values, np.ndarray)
                or values.dtype != np.float64
                or values.shape != (len(self.columns),)
                or not np.isfinite(values).all()
            ):
                raise FeatureError(f"scaler {name} must hold one finite float64 per column")
        if (self.std <= 0).any() or (self.std <= STD_RTOL * np.abs(self.mean)).any():
            raise FeatureError("a scaler column is constant: its deviation is only rounding")
        freeze_arrays(self, "mean", "std")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @classmethod
    def fit(cls, matrix: FeatureMatrix, *, start: datetime, end: datetime) -> Scaler:
        """Fit on `matrix.window(start, end)` and nothing else: per column, `numpy.mean` and
        `numpy.std` (ddof=0, the population standard deviation) over the window's rows.

        Rows outside the window can't reach the result, so data after the refit time can't
        either. Raises FeatureError when the window has fewer than 2 rows or a column is
        constant there: std 0, or no more than STD_RTOL times the column's |mean|, which is
        what rounding leaves of a constant that float64 can't hold exactly (0.1 gives about
        1e-17).
        """
        window = matrix.window(start, end)
        if len(window) < 2:
            raise FeatureError("a scaler window needs at least two rows")
        return cls(
            window.columns,
            window.values.mean(axis=0),
            window.values.std(axis=0),
            start,
            end,
            len(window),
        )

    def transform(self, values: Floats, *, columns: tuple[str, ...]) -> Floats:
        """`(values - mean) / std` as a new float64 array, with the frozen parameters.

        `values` is one row, shape (d,), or a matrix, shape (n, d), of float64, and `columns`
        names its columns, which must be exactly the scaler's `columns`: a vector of the same
        width from another feature config is refused. Raises FeatureError for any other names,
        shape or dtype, input that isn't a numpy array, a value that isn't finite, or a result
        that isn't (a z-score that overflows float64 is a model error, not a warning).
        """
        if (
            columns != self.columns
            or not isinstance(values, np.ndarray)
            or values.dtype != np.float64
            or values.ndim not in (1, 2)
            or values.shape[-1] != len(self.columns)
            or not np.isfinite(values).all()
        ):
            raise FeatureError(
                "scaler input must be a finite float64 row or matrix of the scaler's columns"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = (values - self.mean) / self.std
        if not np.isfinite(scaled).all():
            raise FeatureError("standardized features overflow float64")
        return scaled
