"""The feature menu and the feature-config hash (spec §3, P8, PLAN §5).

v1's menu is spec §3's five transforms, F1–F5, keyed by the spec's names: `ret`, `rvol`,
`range`, `volume` and `trend`. A config picks from it. F1 (ret) and F2 (rvol) are always on, so
they are required. F3 (range), F4 (volume) and F5 (trend) are on when present and off when
absent or null. Every parameter stays inside the bounds the nightly loop may move within. There
are no other transforms in v1; adding one is a normal build step (P8).

A config is a StrictModel, so an unknown transform or parameter is refused like any unknown key,
and a value of the wrong type is refused rather than coerced. Every rejection is a
`pydantic.ValidationError`. Parameter defaults are r1's values (PLAN §5: anything not listed
keeps its spec value). As YAML or JSON, r1 is:

    slot_sessions: 60
    ret: {}
    rvol: {bars: 35}
    range: {}
    volume: {}
    trend: {fast: 35, slow: 175}

The config hash names a config in model artifacts and fit caches (ARCHITECTURE §7.2, §10).
"""

from __future__ import annotations

from typing import Final, Self

from pydantic import Field, model_validator

from quant.core.canonical import sha256_hex
from quant.core.model import StrictModel

# Feature columns always follow this order (F1–F5), skipping the transforms that are off.
MENU: Final = ("ret", "rvol", "range", "volume", "trend")

# Bounds the nightly loop may move within, inclusive at both ends (PLAN §5).
SLOT_SESSIONS: Final = (20, 120)
RVOL_BARS: Final = (14, 70)
TREND_FAST: Final = (14, 70)
TREND_SLOW: Final = (70, 350)


class Ret(StrictModel):
    """F1 ret: `r_t / sigma_slot` (spec §3). No parameters of its own: its slot history is
    `FeatureConfig.slot_sessions`. Always on (P8), written `ret: {}`."""


class Rvol(StrictModel):
    """F2 rvol: `ln sqrt(Σ r²)` over the returns of the last `bars` bars (spec §3). Always on
    (P8). r1 uses 35 bars, five full sessions."""

    bars: int = Field(default=35, ge=RVOL_BARS[0], le=RVOL_BARS[1])


class Range(StrictModel):
    """F3 range: `ln((H_t - L_t) / C_{t-1})` minus the log median of the slot's history (spec
    §3). No parameters of its own: its slot history is `FeatureConfig.slot_sessions`."""


class Volume(StrictModel):
    """F4 volume: `ln(V_t / median volume of the slot's history)` (spec §3). No parameters of
    its own: its slot history is `FeatureConfig.slot_sessions`."""


class Trend(StrictModel):
    """F5 trend: `ln(EMA_fast(C) / EMA_slow(C))` (spec §3, D3). r1 uses EMA35 / EMA175."""

    fast: int = Field(default=35, ge=TREND_FAST[0], le=TREND_FAST[1])
    slow: int = Field(default=175, ge=TREND_SLOW[0], le=TREND_SLOW[1])

    @model_validator(mode="after")
    def _fast_below_slow(self) -> Self:
        """Return self when fast < slow. Otherwise raise ValueError naming both spans, which
        pydantic reports as a ValidationError. Within the bounds that refuses only 70 / 70."""
        if self.fast >= self.slow:
            raise ValueError(f"fast span {self.fast} must be below slow span {self.slow}")
        return self


class FeatureConfig(StrictModel):
    """One selection from the menu, with its parameters (P8, PLAN §5).

    `slot_sessions` is the slot history N that F1, F3 and F4 share: each normalizes a bar
    against at most N earlier bars of its own slot (see quant.features.stream). The warm-up is
    60 sessions whatever N is (spec §3).
    """

    slot_sessions: int = Field(default=60, ge=SLOT_SESSIONS[0], le=SLOT_SESSIONS[1])
    ret: Ret
    rvol: Rvol
    range: Range | None = None
    volume: Volume | None = None
    trend: Trend | None = None

    @property
    def columns(self) -> tuple[str, ...]:
        """The names of the transforms that are on, in MENU order: the columns of every feature
        row and matrix. r1 gives ("ret", "rvol", "range", "volume", "trend"), and a config with
        only F1, F2 and F5 gives ("ret", "rvol", "trend")."""
        return tuple(name for name in MENU if getattr(self, name) is not None)

    def config_hash(self) -> str:
        """SHA-256, in hex, of this config's canonical JSON: exactly
        `quant.core.canonical.sha256_hex(self)`.

        It hashes the validated model, not the input, so key order doesn't matter, an omitted
        transform hashes like an explicit null, and a default hashes like the same value written
        out. Any change of parameter or of which transforms are on changes it.
        """
        return sha256_hex(self)


def r1_features() -> FeatureConfig:
    """The seed release r1's features (PLAN §5, D3): all five transforms on, rvol over 35 bars,
    trend EMA 35 / 175, and a slot history of 60 sessions. A new, validated config on each
    call."""
    return FeatureConfig(ret=Ret(), rvol=Rvol(), range=Range(), volume=Volume(), trend=Trend())
