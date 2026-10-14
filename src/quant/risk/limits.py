"""config/risk.yaml: the hard limits, loaded once and frozen (spec §10, D13, ARCHITECTURE §8.1).

The engine loads the file at start and refuses to start unless its bytes are exactly the
committed version's (ARCHITECTURE §9.3, step 1). The loaded RiskConfig is immutable, nested
values included, so nothing can loosen a limit while the engine runs. Watching the file for
changes on disk, and alerting, belongs to the live engine (B14).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Final

import yaml
from pydantic import Field, ValidationError, model_validator

from quant.core.errors import ConfigError
from quant.core.model import StrictModel
from quant.core.regime import Label

RISK_YAML: Final = "config/risk.yaml"
"""Where the file lives, relative to the repository root."""


class LabelCaps(StrictModel):
    """The hard cap for each label, as a fraction of equity (spec §10). Frozen, like its parent."""

    CALM_UP: float = Field(ge=0.0, le=1.0)
    CHOP: float = Field(ge=0.0, le=1.0)
    STRESS: float = Field(ge=0.0, le=1.0)
    CRASH: float = Field(ge=0.0, le=1.0)

    def cap(self, label: Label) -> float:
        """The cap for `label`."""
        return float(getattr(self, label.value))


class KillConfig(StrictModel):
    """The kill switch's settings: `risk.yaml`'s `kill` mapping (spec §10, ARCHITECTURE §8.3 and
    §16, C2). Frozen, like its parent. Every key is required and no other key is allowed.

    The executor (quant.risk.kill):

    - `requote_s`: seconds a kill exit may stay unfilled before it is cancelled and re-quoted
      wider (spec §10: 30);
    - `widen_bp`: basis points each re-quote adds to the last one's distance from the quote
      (10, so 10, 20, 30 bp ...);
    - `max_collar_bp`: the furthest, in basis points, a kill exit ever sits from the quote (100);
    - `poll_s`: seconds between the executor's broker reads while it waits (1);
    - `deadline_s`: seconds after the executor starts by which the account must be flat and
      clean; past it the executor stops, reports INCOMPLETE and alerts (300);
    - `stop_distance_bp`: how far below the bid, in basis points, the protective stop sits that
      the executor leaves when it can't finish (100; the engine has no ATR, B9c).

    The automatic triggers (quant.risk.triggers):

    - `order_errors`, `order_error_window_s`: that many broker order errors within that many
      seconds fire the kill (ARCHITECTURE §8.3: 5 within 600);
    - `heartbeat_lost_s`: seconds without an engine heartbeat that count as a lost heartbeat
      (C2: 300, longer than the 90 s watchdog's kill-and-restart, ARCHITECTURE §16);
    - `crash_loop_starts`, `crash_loop_window_s`: that many engine starts within that many
      seconds are a crash loop (C2: 3 within 600);
    - `flatten_incomplete_s`: seconds a rule-6 flatten may stay incomplete (C2: 600);
    - `stop_missing_s`: seconds a position may go without a covering broker-side stop
      (ARCHITECTURE §16: 300).

    Every threshold is inclusive: an elapsed time or count equal to it fires.
    """

    requote_s: float = Field(gt=0.0)
    widen_bp: float = Field(ge=0.0, le=100.0)
    max_collar_bp: float = Field(gt=0.0, le=500.0)
    poll_s: float = Field(gt=0.0)
    deadline_s: float = Field(gt=0.0)
    order_errors: int = Field(ge=1)
    order_error_window_s: float = Field(gt=0.0)
    heartbeat_lost_s: float = Field(gt=0.0)
    crash_loop_starts: int = Field(ge=2)
    crash_loop_window_s: float = Field(gt=0.0)
    flatten_incomplete_s: float = Field(gt=0.0)
    stop_missing_s: float = Field(gt=0.0)
    stop_distance_bp: float = Field(gt=0.0, le=1000.0)

    @model_validator(mode="after")
    def _consistent(self) -> KillConfig:
        """ValueError (so a ValidationError) unless `poll_s < requote_s < deadline_s`: the
        executor must read the broker inside each attempt, and get more than one attempt."""
        if not self.poll_s < self.requote_s < self.deadline_s:
            raise ValueError("kill timings must satisfy poll_s < requote_s < deadline_s")
        return self


class RiskConfig(StrictModel):
    """The limits in config/risk.yaml. Every key is required and no other key is allowed.

    Exposures are fractions of the broker snapshot's equity:

    - `max_gross_exposure`: the most any BUY may leave held (1.0, D9);
    - `label_caps`: the most a BUY may leave held under each active label;
    - `uncertain_cap`: the same when rule 5 fired or no label is active (0);
    - `overnight_cap`: the same on the last decision that can fill (A3, 0.5);
    - `single_order_cap`: the most one order may buy (0.5);
    - `daily_loss_fraction`: the loss from start-of-day equity that locks entries and flattens
      (0.02);
    - `max_drawdown_fraction`: the drop from peak equity that fires the kill switch (0.10);
    - `manual_approval_notional`: dollars; a live BUY above it waits for you (D7, 25,000);
    - `max_intents_per_day`: the most intents a session may have; one more halts (C3, 20);
    - `submission_halt_at`: the raw submission that halts trading (C3, the 60th);
    - `collar_bp`: how far, in basis points, a limit may sit from its side of the quote (10);
    - `quote_max_age_s`, `snapshot_max_age_s`: seconds; the oldest quote (check 6) and broker
      snapshot (check 2) a check accepts (5 each). Reconciliation (quant.risk.reconcile) uses
      `snapshot_max_age_s` too;
    - `monitor_price_max_age_s`: seconds; the oldest price the minute monitor marks equity on
      (C8, 120). An older price marks nothing;
    - `kill`: the kill switch's settings (KillConfig).
    """

    max_gross_exposure: float = Field(gt=0.0, le=1.0)
    label_caps: LabelCaps
    uncertain_cap: float = Field(ge=0.0, le=1.0)
    overnight_cap: float = Field(ge=0.0, le=1.0)
    single_order_cap: float = Field(gt=0.0, le=1.0)
    daily_loss_fraction: float = Field(gt=0.0, lt=1.0)
    max_drawdown_fraction: float = Field(gt=0.0, lt=1.0)
    manual_approval_notional: float = Field(gt=0.0)
    max_intents_per_day: int = Field(ge=1)
    submission_halt_at: int = Field(ge=1)
    collar_bp: float = Field(gt=0.0, le=100.0)
    quote_max_age_s: float = Field(gt=0.0)
    snapshot_max_age_s: float = Field(gt=0.0)
    monitor_price_max_age_s: float = Field(gt=0.0)
    kill: KillConfig

    @model_validator(mode="after")
    def _consistent(self) -> RiskConfig:
        """ValueError (so a ValidationError) unless the limits hang together: no label cap and
        neither the overnight nor the uncertain cap above `max_gross_exposure`, CRASH's cap is 0
        (spec §7: no playbook can trade it), `uncertain_cap` is 0 (spec §10), and a kill exit's
        widest distance `kill.max_collar_bp` is at least `collar_bp`, where it starts."""
        caps = [self.label_caps.cap(label) for label in Label]
        if max(*caps, self.overnight_cap, self.uncertain_cap) > self.max_gross_exposure:
            raise ValueError("label, overnight and uncertain caps must not exceed gross exposure")
        if self.label_caps.CRASH != 0.0 or self.uncertain_cap != 0.0:
            raise ValueError("CRASH and uncertainty must cap exposure at zero")
        if self.kill.max_collar_bp < self.collar_bp:
            raise ValueError("a kill exit's widest collar must be at least the collar")
        return self


def parse_risk_config(data: bytes) -> RiskConfig:
    """The RiskConfig in `data`, YAML text (yaml.safe_load). ConfigError, naming the problem,
    for text that isn't UTF-8 or YAML, a top level that isn't a mapping, an unknown or missing
    key, a value of the wrong type, or limits that fail RiskConfig's checks. YAML integers are
    accepted for float fields; nothing else is coerced."""
    try:
        values = yaml.safe_load(data.decode("utf-8"))
        if not isinstance(values, dict):
            raise ConfigError("risk.yaml must contain a mapping")
        return RiskConfig.model_validate(values)
    except (UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(f"invalid risk.yaml: {exc}") from exc


def load_risk_config(path: Path, *, committed: bytes) -> RiskConfig:
    """Load the limits at `path`, frozen.

    ConfigError unless the file's bytes are exactly `committed`, the committed version's bytes
    (`committed_bytes`): one changed byte, whitespace and comments included, is a difference.
    Also ConfigError when the file can't be read, and whatever `parse_risk_config` refuses.
    """
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"cannot read risk limits at {path}: {exc}") from exc
    if data != committed:
        raise ConfigError(f"risk limits at {path} differ from the committed version")
    return parse_risk_config(data)


def committed_bytes(repo: Path, relpath: str = RISK_YAML) -> bytes:
    """The bytes of `relpath` as committed at HEAD in the git repository `repo`
    (`git -C repo show HEAD:relpath`).

    git runs with every `GIT_*` variable removed from its environment, because the commit hook
    exports GIT_INDEX_FILE and friends, which would point it at another repository's index.
    ConfigError when `repo` isn't a git repository, has no commits, or doesn't track `relpath`
    at HEAD.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "show", f"HEAD:{relpath}"],
            env={key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ConfigError(f"cannot read committed {relpath} from {repo}: {exc}") from exc
    return result.stdout
