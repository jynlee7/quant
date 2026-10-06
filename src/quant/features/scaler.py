"""The scaler: z-score parameters fit on a model's training window, then frozen (spec §3).

They are fit at refit time on the warm feature rows of the model's window and never recomputed
with data from after it. The model applies them forward until the next refit fits a new scaler,
and its artifact stores `mean` and `std` in params.npz (ARCHITECTURE §7.2, §7.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

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
        raise NotImplementedError("red:B3")

    @classmethod
    def fit(cls, matrix: FeatureMatrix, *, start: datetime, end: datetime) -> Scaler:
        """Fit on `matrix.window(start, end)` and nothing else: per column, `numpy.mean` and
        `numpy.std` (ddof=0, the population standard deviation) over the window's rows.

        Rows outside the window can't reach the result, so data after the refit time can't
        either. Raises FeatureError when the window has fewer than 2 rows or a column is
        constant there (std 0).
        """
        raise NotImplementedError("red:B3")

    def transform(self, values: Floats) -> Floats:
        """`(values - mean) / std` as a new float64 array, with the frozen parameters.

        `values` is one row, shape (d,), or a matrix, shape (n, d), with d = len(columns).
        Raises FeatureError on any other shape, or a value that isn't finite.
        """
        raise NotImplementedError("red:B3")
