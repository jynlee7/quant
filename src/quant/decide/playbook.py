"""The playbook schema, the front-matter parser, and the release loader (spec §7, ARCHITECTURE §9.2,
PLAN §5, P7).

**Files.** A release is a directory holding `strategy.md` and `playbooks/<LABEL>.md`, one per
label. Code reads only each file's YAML front matter: the text between the file's first line,
which must be exactly `---`, and the next line that is exactly `---`. Everything after that is
prose for the owner and is ignored, so editing it changes neither a playbook nor the release id.
The front matter is read with `yaml.safe_load` and must be a mapping. It is then validated
through its canonical JSON (`quant.core.canonical.canonical_dumps`, then pydantic's strict JSON
mode: `PLAYBOOK.validate_json(...)`, `Manifest.model_validate_json(...)`), so labels, times and
lists are read from their JSON text as everywhere else in the project, and a YAML `.nan` or
`.inf` is refused (canonical JSON has no form for it). A YAML timestamp with an offset is read
as that instant, in UTC; a naive one or a bare date is refused.

**Playbooks.** One StrictModel per strategy, chosen by the `strategy` key (a discriminated
union): unknown keys refused at every level (top, `params`, each invalidation), no coercion
(YAML integers are accepted for float fields; a bool is never an int, a string never a number),
no NaN or infinity, frozen. Every key below is required, `take_profit` and `stop_loss` included
(they may be null where stated). Entry and exit rules are not free-form: they are the
strategy's own, parameterized by `params` (spec §7):

- `trend_following`: enter when EMA(fast) > EMA(slow) and the close > EMA(fast); exit when
  EMA(fast) < EMA(slow). `params`: `fast` 14-70, `slow` 70-350 with fast < slow, `atr_bars`
  14-70 (PLAN §5's EMA bounds, as the features use them; ATR's window bounded like N).
- `mean_reversion`: z = (close - N-bar mean) / N-bar s.d.; enter when z <= -entry_z; exit when
  z >= -exit_z. `params`: `n` 14-70, `entry_z` 0.5-3.0, `exit_z` -1.0 to 1.0 with
  exit_z < entry_z, `atr_bars` 14-70.
- `flat`: holds no position (D8). `params` is `{}`.

Common keys: `label` (the Label the file is for), `stop_loss` (a multiple of ATR(atr_bars) at
entry, 2-8, mandatory for a trading strategy; null for `flat`, which never holds a stop),
`take_profit` (null, or a multiple of ATR at entry, 2-10; null for `flat`), `max_size` (a
fraction of equity, 0-1; sizing clamps it to the label's hard cap, it is never rejected for
exceeding the cap, spec §7; exactly 0 for `flat`), `version` (an int >= 1), `created_at` (an
aware datetime, stored in UTC; naive refused), and `invalidation` (a list, possibly empty, of
typed conditions; empty for `flat`, which is never in a trade).

All bounds are inclusive at both ends unless written as an open interval. Label rules (PLAN §5,
spec §7): a CRASH playbook must be `flat` (its hard cap is 0 in code), and a STRESS playbook's
`max_size` is at most 0.25.

**Invalidation menu** (spec §7), a discriminated union on `kind`; anything else is refused:

- `label_prob_below` {threshold in (0, 1)}: the filtered P(the playbook's label) < threshold;
- `close_below_ema` {span 14-350}: the close < EMA(span) of the close;
- `bars_in_trade_exceeds` {bars 7-70}: bars in the trade > bars;
- `trade_drawdown_exceeds` {fraction in (0, 1)}: (peak close since entry - close) / peak close
  > fraction.

quant.decide.primitives evaluates them.

**The manifest.** `strategy.md`'s front matter (`Manifest`): `release` (r1, r2, ...), `version`,
`created_at`, `features` (a `quant.features.menu.FeatureConfig`), `hmm` (`HmmSearch`, the
training config, field for field `quant.hmm.train.TrainConfig`, which decide can't import
because hmm.train imports hmmlearn, K3), `label_thresholds` (`quant.hmm.labels.LabelThresholds`),
`playbooks` (the label -> playbook file map: each of the four labels to `playbooks/<LABEL>.md`)
and `gate_report` (null until a gate report exists; r1 has none until CP2).

**The release id** is `quant.core.canonical.sha256_hex({"manifest": manifest, "playbooks":
{label.value: playbook for each label}})`: SHA-256 of the canonical JSON of the parsed front
matters. The prose bodies don't enter it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Final, Literal, Self

import yaml
from pydantic import (
    AfterValidator,
    AwareDatetime,
    Field,
    TypeAdapter,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)

from quant.core.canonical import canonical_dumps, sha256_hex
from quant.core.clock import ensure_utc
from quant.core.errors import CanonicalJSONError, ConfigError
from quant.core.model import StrictModel
from quant.core.regime import Label
from quant.features.menu import FeatureConfig
from quant.hmm.labels import LabelThresholds

STRATEGY_MD: Final = "strategy.md"
"""The manifest's file, relative to the release root."""

PLAYBOOK_DIR: Final = "playbooks"
"""The playbooks' directory, relative to the release root: `playbooks/<LABEL>.md`."""

# PLAN §5's bounds, inclusive at both ends.
EMA_FAST: Final = (14, 70)
EMA_SLOW: Final = (70, 350)
ATR_BARS: Final = (14, 70)
MR_N: Final = (14, 70)
ENTRY_Z: Final = (0.5, 3.0)
EXIT_Z: Final = (-1.0, 1.0)
STOP_LOSS: Final = (2.0, 8.0)
TAKE_PROFIT: Final = (2.0, 10.0)
MAX_SIZE: Final = (0.0, 1.0)
STRESS_MAX_SIZE: Final = 0.25
EMA_SPAN: Final = (14, 350)
BARS_IN_TRADE: Final = (7, 70)

UtcDatetime = Annotated[AwareDatetime, AfterValidator(ensure_utc)]


# --- invalidation menu ----------------------------------------------------------------------


class LabelProbBelow(StrictModel):
    """Invalidated when the filtered P(the playbook's label) < `threshold`."""

    kind: Literal["label_prob_below"]
    threshold: float = Field(gt=0.0, lt=1.0)


class CloseBelowEma(StrictModel):
    """Invalidated when the decision bar's close < EMA(`span`) of the close."""

    kind: Literal["close_below_ema"]
    span: int = Field(ge=EMA_SPAN[0], le=EMA_SPAN[1])


class BarsInTradeExceeds(StrictModel):
    """Invalidated when the trade has lasted more than `bars` decision bars."""

    kind: Literal["bars_in_trade_exceeds"]
    bars: int = Field(ge=BARS_IN_TRADE[0], le=BARS_IN_TRADE[1])


class TradeDrawdownExceeds(StrictModel):
    """Invalidated when (peak close since entry - close) / peak close > `fraction`."""

    kind: Literal["trade_drawdown_exceeds"]
    fraction: float = Field(gt=0.0, lt=1.0)


Invalidation = Annotated[
    LabelProbBelow | CloseBelowEma | BarsInTradeExceeds | TradeDrawdownExceeds,
    Field(discriminator="kind"),
]


# --- strategy params ------------------------------------------------------------------------


class TrendParams(StrictModel):
    """`trend_following`'s parameters. r1's CALM_UP: fast 35, slow 175, atr_bars 35."""

    fast: int = Field(ge=EMA_FAST[0], le=EMA_FAST[1])
    slow: int = Field(ge=EMA_SLOW[0], le=EMA_SLOW[1])
    atr_bars: int = Field(ge=ATR_BARS[0], le=ATR_BARS[1])

    @model_validator(mode="after")
    def _fast_below_slow(self) -> Self:
        """Return self when fast < slow; otherwise raise ValueError naming both (pydantic
        reports a ValidationError). Within the bounds that refuses only 70 / 70."""
        if self.fast >= self.slow:
            raise ValueError("fast must be below slow")
        return self


class MeanReversionParams(StrictModel):
    """`mean_reversion`'s parameters. r1's CHOP: n 35, entry_z 1.5, exit_z 0.0, atr_bars 35."""

    n: int = Field(ge=MR_N[0], le=MR_N[1])
    entry_z: float = Field(ge=ENTRY_Z[0], le=ENTRY_Z[1])
    exit_z: float = Field(ge=EXIT_Z[0], le=EXIT_Z[1])
    atr_bars: int = Field(ge=ATR_BARS[0], le=ATR_BARS[1])

    @model_validator(mode="after")
    def _exit_below_entry(self) -> Self:
        """Return self when exit_z < entry_z; otherwise raise ValueError naming both. Within
        the bounds that refuses exit_z in [0.5, 1.0] with entry_z <= exit_z."""
        if self.exit_z >= self.entry_z:
            raise ValueError("exit_z must be below entry_z")
        return self


class FlatParams(StrictModel):
    """`flat` has no parameters: `params: {}`."""


# --- playbooks ------------------------------------------------------------------------------


class _PlaybookBase(StrictModel):
    """The keys every playbook has."""

    label: Label
    max_size: float = Field(ge=MAX_SIZE[0], le=MAX_SIZE[1])
    version: int = Field(ge=1)
    created_at: UtcDatetime
    invalidation: tuple[Invalidation, ...]

    @model_validator(mode="after")
    def _label_rules(self) -> Self:
        """Return self when the label rules hold; otherwise raise ValueError naming the rule:
        a CRASH playbook must be `flat`, and a STRESS playbook's `max_size` must be <= 0.25
        (compared on the decimal value, `Decimal(repr(max_size))`, so exactly 0.25 passes).
        Every subclass runs it."""
        if self.label == Label.CRASH and not isinstance(self, FlatPlaybook):
            raise ValueError("CRASH must be flat")
        if self.label == Label.STRESS and Decimal(repr(self.max_size)) > Decimal("0.25"):
            raise ValueError("STRESS max_size must be <= 0.25")
        return self


class TrendPlaybook(_PlaybookBase):
    """A `trend_following` playbook (r1's CALM_UP)."""

    strategy: Literal["trend_following"]
    params: TrendParams
    stop_loss: float = Field(ge=STOP_LOSS[0], le=STOP_LOSS[1])
    take_profit: float | None = Field(ge=TAKE_PROFIT[0], le=TAKE_PROFIT[1])


class MeanReversionPlaybook(_PlaybookBase):
    """A `mean_reversion` playbook (r1's CHOP)."""

    strategy: Literal["mean_reversion"]
    params: MeanReversionParams
    stop_loss: float = Field(ge=STOP_LOSS[0], le=STOP_LOSS[1])
    take_profit: float | None = Field(ge=TAKE_PROFIT[0], le=TAKE_PROFIT[1])


class FlatPlaybook(_PlaybookBase):
    """A `flat` playbook (r1's STRESS and CRASH): `stop_loss` and `take_profit` are null,
    `max_size` is 0, and `invalidation` is empty. Its validator (on top of the label rules)
    raises ValueError when `max_size` isn't 0 or `invalidation` isn't empty."""

    strategy: Literal["flat"]
    params: FlatParams
    stop_loss: None
    take_profit: None

    @model_validator(mode="after")
    def _holds_nothing(self) -> Self:
        """Return self when `max_size == 0.0` and `invalidation == ()`; otherwise ValueError."""
        if self.max_size != 0.0 or self.invalidation != ():
            raise ValueError("flat holds nothing")
        return self


Playbook = TrendPlaybook | MeanReversionPlaybook | FlatPlaybook

PLAYBOOK: Final = TypeAdapter[Playbook](Annotated[Playbook, Field(discriminator="strategy")])
"""Validates one playbook's front matter (a mapping) into the model its `strategy` names."""


# --- the manifest ---------------------------------------------------------------------------


class HmmCandidate(StrictModel):
    """One point of the selection grid: K hidden states and the covariance type
    (`quant.hmm.select.Candidate`'s fields)."""

    n_states: int = Field(ge=2)
    covariance_type: Literal["diag", "full"]


class HmmSearch(StrictModel):
    """The HMM training config, field for field `quant.hmm.train.TrainConfig`, so its
    canonical JSON equals `r1_train_config()`'s for r1 (spec §4, PLAN §5): base_seed
    20160104, 50 restarts, K in {2, 3, 4, 5} x {diag, full} ordered K ascending with diag
    before full, n_iter 1000, tol 0.01, min_covar 0.001, 1,000 bootstrap draws and a 12-month
    validation block. TrainConfig's own checks run where it is built (hmm.train); this
    schema checks types and signs only."""

    base_seed: int = Field(ge=0)
    n_restarts: int = Field(ge=1)
    candidates: tuple[HmmCandidate, ...] = Field(min_length=1)
    n_iter: int = Field(ge=1)
    tol: float = Field(gt=0.0)
    min_covar: float = Field(gt=0.0)
    bootstrap_draws: int = Field(ge=2)
    validation_months: int = Field(ge=1)


class Manifest(StrictModel):
    """`strategy.md`'s front matter (ARCHITECTURE §9.2). `playbooks` maps each label's name to
    its file, relative to the release root; `Release` loading requires exactly the four labels,
    each to `playbooks/<LABEL>.md`."""

    release: str = Field(pattern=r"^r[1-9][0-9]*$")
    version: int = Field(ge=1)
    created_at: UtcDatetime
    features: FeatureConfig
    hmm: HmmSearch
    label_thresholds: LabelThresholds
    playbooks: Mapping[Label, str]
    gate_report: str | None

    @field_validator("playbooks", mode="after")
    @classmethod
    def _freeze_playbooks(cls, value: Mapping[Label, str]) -> Mapping[Label, str]:
        """Own an immutable copy, so the release hash cannot drift after loading."""
        return MappingProxyType(dict(value))

    @field_serializer("playbooks")
    def _serialize_playbooks(self, value: Mapping[Label, str]) -> dict[Label, str]:
        """Keep the manifest's canonical JSON map independent of its storage type."""
        return dict(value)


# --- the release ----------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Release:
    """A loaded release: its manifest, one playbook per label, and its id. Immutable:
    `playbooks` is a read-only mapping (`types.MappingProxyType`) holding every Label.
    Built by `load_release` (or `release_of` for parsed parts)."""

    manifest: Manifest
    playbooks: Mapping[Label, Playbook]
    release_id: str

    def playbook(self, label: Label) -> Playbook:
        """The playbook for `label` (every label has one)."""
        return self.playbooks[label]


def parse_front_matter(text: str) -> dict[str, Any]:
    """The YAML front matter of a release file, as a mapping.

    The file's first line must be exactly `---` (no BOM, no leading blank line), and a later
    line must be exactly `---`; the text between them is read with `yaml.safe_load`. ConfigError
    when the opening or closing `---` is missing, the YAML doesn't parse, or it isn't a mapping
    (an empty block, a list or a scalar). The body after the closing line is never parsed.
    """
    lines = text.split("\n")
    if not lines or lines[0] != "---":
        raise ConfigError("front matter must begin with --- on the first line")
    try:
        end = lines[1:].index("---") + 1
    except ValueError:
        raise ConfigError("front matter is missing its closing ---") from None
    try:
        data = yaml.safe_load("\n".join(lines[1:end]))
    except yaml.YAMLError as exc:
        raise ConfigError(str(exc)) from exc
    if not isinstance(data, dict):
        raise ConfigError("front matter must be a mapping")
    return data


def parse_playbook(text: str) -> Playbook:
    """The playbook in a playbook file's text: its front matter validated by `PLAYBOOK`.
    ConfigError, naming the problem, for whatever `parse_front_matter` refuses and for any
    schema failure (each pydantic ValidationError becomes a ConfigError)."""
    front = parse_front_matter(text)
    try:
        return PLAYBOOK.validate_json(canonical_dumps(front))
    except (ValidationError, CanonicalJSONError) as exc:
        raise ConfigError(str(exc)) from exc


def parse_manifest(text: str) -> Manifest:
    """The manifest in `strategy.md`'s text: its front matter validated as a `Manifest`.
    ConfigError as `parse_playbook`."""
    front = parse_front_matter(text)
    try:
        return Manifest.model_validate_json(canonical_dumps(front))
    except (ValidationError, CanonicalJSONError) as exc:
        raise ConfigError(str(exc)) from exc


def release_id_of(manifest: Manifest, playbooks: Mapping[Label, Playbook]) -> str:
    """`sha256_hex({"manifest": manifest, "playbooks": {label.value: playbook}})` over every
    Label (module docstring). ConfigError unless `playbooks` has exactly the four labels."""
    if set(playbooks) != set(Label):
        raise ConfigError("playbooks must contain exactly every Label")
    return sha256_hex(
        {"manifest": manifest, "playbooks": {label.value: playbooks[label] for label in Label}}
    )


def release_of(manifest: Manifest, playbooks: Mapping[Label, Playbook]) -> Release:
    """A Release of parsed parts, with its `release_id_of`. ConfigError unless `playbooks` has
    exactly the four labels and each playbook's `label` is its key. The mapping is copied, so
    the caller's dict can't change the release."""
    rid = release_id_of(manifest, playbooks)
    for label, pb in playbooks.items():
        if pb.label != label:
            raise ConfigError(f"playbook {label.value} declares label {pb.label.value}")
    return Release(manifest, MappingProxyType({label: playbooks[label] for label in Label}), rid)


def load_release(root: Path) -> Release:
    """The release in directory `root`: `root/strategy.md` and the playbook files its map names.

    ConfigError, naming the file, when: `strategy.md` or a playbook file is missing or can't be
    read as UTF-8; front matter is missing or isn't a mapping; the manifest or a playbook fails
    the schema; the map doesn't send each of the four labels to its own file, exactly
    `playbooks/<LABEL>.md` (a missing label, two labels to one file, another path); or a
    playbook's `label` isn't the label the map sends to its file. Never reads anything else.
    """

    def read(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ConfigError(f"{path}: {exc}") from exc

    try:
        manifest = parse_manifest(read(root / STRATEGY_MD))
    except ConfigError as exc:
        raise ConfigError(f"{root / STRATEGY_MD}: {exc}") from exc
    if set(manifest.playbooks) != set(Label):
        raise ConfigError("manifest playbooks must map every Label")
    pbs = {}
    for label in Label:
        rel = manifest.playbooks[label]
        if rel != f"{PLAYBOOK_DIR}/{label.value}.md":
            raise ConfigError(f"invalid playbook path for {label.value}: {rel}")
        try:
            pbs[label] = parse_playbook(read(root / rel))
        except ConfigError as exc:
            raise ConfigError(f"{root / rel}: {exc}") from exc
        if pbs[label].label != label:
            raise ConfigError(f"{root / rel}: expected label {label.value}")
    return release_of(manifest, pbs)
