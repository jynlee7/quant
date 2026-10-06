"""The scaler: z-score parameters fit on a model's training window, then frozen (spec §3).

They are fit at refit time on the warm feature rows of the model's window and never recomputed
with data from after it. The model applies them forward until the next refit fits a new scaler,
and its artifact stores `mean` and `std` in params.npz (ARCHITECTURE §7.2, §7.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from quant.core.clock import ensure_utc
from quant.core.errors import FeatureError
from quant.data.frames import Floats
from quant.features.stream import FeatureMatrix


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
        are 1-D float64 arrays with one finite value per column, every std is > 0, `rows` >= 2,
        and start < end. `start` and `end` must be timezone-aware (a naive one raises
        ClockError) and are kept in UTC. Replace the arrays with read-only copies
        (`object.__setattr__`), so writing to them raises ValueError."""
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
            if name == "std" and (values <= 0).any():
                raise FeatureError("scaler standard deviations must be positive")
            frozen = values.copy()
            frozen.flags.writeable = False
            object.__setattr__(self, name, frozen)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @classmethod
    def fit(cls, matrix: FeatureMatrix, *, start: datetime, end: datetime) -> Scaler:
        """Fit on `matrix.window(start, end)` and nothing else: per column, `numpy.mean` and
        `numpy.std` (ddof=0, the population standard deviation) over the window's rows.

        Rows outside the window can't reach the result, so data after the refit time can't
        either. Raises FeatureError when the window has fewer than 2 rows or a column is
        constant there (std 0).
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

    def transform(self, values: Floats) -> Floats:
        """`(values - mean) / std` as a new float64 array, with the frozen parameters.

        `values` is one row, shape (d,), or a matrix, shape (n, d), with d = len(columns).
        Raises FeatureError on any other shape, or a value that isn't finite.
        """
        if (
            values.ndim not in (1, 2)
            or values.shape[-1] != len(self.columns)
            or not np.isfinite(values).all()
        ):
            raise FeatureError("scaler input must be a finite row or matrix with matching columns")
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.std
