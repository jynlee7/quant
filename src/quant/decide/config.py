"""config/engine.yaml: the engine's L3 configuration, loaded once and frozen (ARCHITECTURE §9.2).

The file holds spec §8's switching thresholds, spec §9's sizing constants (C12) and the decision
timing (D2). It changes only through a reviewed commit, and its `config_hash` goes into every
DecisionRecord. Like config/risk.yaml, the engine (B14) refuses to start on a file whose bytes
differ from the committed version's: `load_engine_config` takes the committed bytes, and the
engine reads them from git HEAD with `quant.risk.limits.committed_bytes(repo, ENGINE_YAML)`.
This module never imports `quant.risk` (decide stays one way from risk: the caps arrive as data).

As YAML, r1's file is:

    switching:
      takeover_prob: 0.70        # rules 1-3: a challenger takes over at P >= 0.70 ...
      persistence_bars: 3        # ... as the top label on 3 consecutive bars ...
      cooldown_bars: 7           # ... once 7 decision bars have passed since the last switch (D5)
      derisk_prob: 0.25          # rule 4: next-bar P(STRESS) + P(CRASH) > 0.25 ...
      derisk_multiplier: 0.5     # ... caps the size multiplier at 0.5
      uncertainty_gap: 0.15      # rule 5: top two label probabilities within 0.15: size 0
      health_recovery_bars: 2    # rule 6: normal operation on the 2nd consecutive healthy bar
    sizing:
      kelly_fraction: 0.25       # spec §9's 0.25 x f*
      min_edge_bars: 100         # f* = 0 with fewer counted shadow-return bars (C12)
      deadband: 0.10             # resize only when |target - current| >= 0.10 of equity
      shadow_cost_bp: 2.0        # per change of shadow position (ARCHITECTURE §10's slippage)
    timing:
      feed_delay_min: 15         # a bar is available 15 minutes after it closes (D2)
      decision_delay_min: 16     # and decided on 16 minutes after it closes

Every key is required and no other key is allowed, at every level. YAML integers are accepted
for float fields (pydantic's strict mode allows int for float); nothing else is coerced, so a
bool for an int, a string for a number, or a float for an int field is refused. Every rejection
of the file's content is a ConfigError naming the problem.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Final

import yaml
from pydantic import Field, ValidationError

from quant.core.canonical import sha256_hex
from quant.core.errors import ConfigError
from quant.core.model import StrictModel
from quant.core.schedule import Timing

ENGINE_YAML: Final = "config/engine.yaml"
"""Where the file lives, relative to the repository root."""


class SwitchingConfig(StrictModel):
    """Spec §8's switching parameters (quant.decide.switching reads them).

    - `takeover_prob`: the probability a challenger needs as the top label (rules 1-3, A1);
      in (0.5, 1], so at most one label can hold it;
    - `persistence_bars`: consecutive qualifying bars a risk-increasing takeover needs, and a
      cold start's first takeover, and an invalidation block's re-confirmation (>= 1);
    - `cooldown_bars`: decision bars after a switch before a risk-increasing switch (D5, >= 0);
    - `derisk_prob`: rule 4's threshold on next-bar P(STRESS) + P(CRASH), in (0, 1);
    - `derisk_multiplier`: rule 4's size multiplier when it fires, in [0, 1];
    - `uncertainty_gap`: rule 5's gap between the top two label probabilities, in [0, 1);
    - `health_recovery_bars`: consecutive healthy bars before normal operation resumes after an
      unhealthy one (rule 6, >= 1).
    """

    takeover_prob: float = Field(gt=0.5, le=1.0)
    persistence_bars: int = Field(ge=1)
    cooldown_bars: int = Field(ge=0)
    derisk_prob: float = Field(gt=0.0, lt=1.0)
    derisk_multiplier: float = Field(ge=0.0, le=1.0)
    uncertainty_gap: float = Field(ge=0.0, lt=1.0)
    health_recovery_bars: int = Field(ge=1)


class SizingConfig(StrictModel):
    """Spec §9's sizing constants (B8b's quant.decide.sizing and quant.decide.edge read them).

    - `kelly_fraction`: the multiple of f* in the target (0.25), in (0, 1];
    - `min_edge_bars`: counted shadow-return bars below which f* = 0 (C12: 100), >= 2 so the
      sample standard deviation exists;
    - `deadband`: the least |target - current|, as a fraction of equity, that resizes (0.10),
      in [0, 1);
    - `shadow_cost_bp`: basis points charged on each change of shadow position (2.0), >= 0.
    """

    kelly_fraction: float = Field(gt=0.0, le=1.0)
    min_edge_bars: int = Field(ge=2)
    deadband: float = Field(ge=0.0, lt=1.0)
    shadow_cost_bp: float = Field(ge=0.0)


class TimingConfig(StrictModel):
    """The decision timing (D2), in whole minutes, each >= 0. `timing()` turns it into the
    schedule's `Timing`, whose own checks (decision_delay above feed_delay, and at most the
    30-minute CLOSE bar beyond it) apply when the file is parsed."""

    feed_delay_min: int = Field(ge=0)
    decision_delay_min: int = Field(ge=0)

    def timing(self) -> Timing:
        """`Timing(feed_delay=timedelta(minutes=feed_delay_min),
        decision_delay=timedelta(minutes=decision_delay_min))`. ConfigError (from Timing) when
        the delays are inconsistent. r1's file gives `Timing()`, the schedule's defaults."""
        return Timing(
            feed_delay=timedelta(minutes=self.feed_delay_min),
            decision_delay=timedelta(minutes=self.decision_delay_min),
        )


class EngineConfig(StrictModel):
    """The whole of config/engine.yaml: `switching`, `sizing` and `timing`. Frozen all the
    way down, so nothing can change a threshold while the engine runs."""

    switching: SwitchingConfig
    sizing: SizingConfig
    timing: TimingConfig

    def config_hash(self) -> str:
        """SHA-256, in hex, of this config's canonical JSON: exactly
        `quant.core.canonical.sha256_hex(self)`. It hashes the parsed values, so comments,
        whitespace and key order in the file don't change it, and any changed value does. It
        goes into every DecisionRecord (ARCHITECTURE §9.2)."""
        return sha256_hex(self)


def parse_engine_config(data: bytes) -> EngineConfig:
    """The EngineConfig in `data`, YAML text (`yaml.safe_load`).

    ConfigError, naming the problem, for text that isn't UTF-8 or YAML, a top level that isn't
    a mapping, an unknown or missing key at any level, a value of the wrong type or out of its
    bounds (the fields' constraints above), and timing that `Timing` refuses (the parsed
    config's `timing.timing()` is built once here, so a config that parses always has a valid
    Timing). No pydantic ValidationError escapes: each becomes a ConfigError.
    """
    try:
        values = yaml.safe_load(data.decode("utf-8"))
        if not isinstance(values, dict):
            raise ConfigError("engine.yaml must contain a mapping")
        config = EngineConfig.model_validate(values)
    except (UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(f"invalid engine.yaml: {exc}") from exc
    config.timing.timing()
    return config


def load_engine_config(path: Path, *, committed: bytes) -> EngineConfig:
    """Load the engine config at `path`, frozen.

    ConfigError unless the file's bytes are exactly `committed`, the committed version's bytes
    (the engine passes `quant.risk.limits.committed_bytes(repo, ENGINE_YAML)`): one changed
    byte, whitespace and comments included, is a difference. Also ConfigError when the file
    can't be read, and whatever `parse_engine_config` refuses. The same contract as
    `quant.risk.limits.load_risk_config`.
    """
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ConfigError(str(exc)) from exc
    if data != committed:
        raise ConfigError(f"{path} differs from its committed bytes")
    return parse_engine_config(data)
