"""Sizing: spec §9's target, the calibration floor, the caps, the freeze and loss locks, and the
deadband (spec §9, C12, C13, design call 9).

**The target** for the active label L and its playbook (`size`):

    target = min(cap[L] * cap_fraction, playbook.max_size, kelly_fraction * f*)
             * P~(L) * (1 - H_t) * m_rule4 * m_rule5

- `cap[L]`: L's hard cap, risk.yaml's `label_caps` passed in as data (`Caps`): decide never
  imports risk. CRASH's is 0, so CRASH sizes 0.
- `cap_fraction`: `Calibration.cap_fraction(L)` (1.0 when L and the outcome check pass, else
  C13's `failed_cap`); with no calibration at all every label fails (C13), so it is
  `quant.core.calibration.r1_calibration_config().failed_cap` (0.25, `NO_CALIBRATION_CAP`).
- `playbook.max_size`, clamped by the min to the cap (spec §7: clamped, never rejected).
- f*: `quant.decide.edge`'s Kelly fraction from the label's shadow returns (C12).
- P~(L): `Calibration.calibrated(L, P(L))`, the filtered probability through L's knots; with no
  calibration, the raw P(L).
- H_t: the filter's normalized label entropy, in [0, 1].
- m_rule4, m_rule5: the switching engine's multipliers (spec §8: 0.5 under rule 4, 0 under
  rule 5).

The product is plain float64 arithmetic in that order, clipped into [0, 1].

**Zero** (`target`): the target is 0, before the formula, while halted, on a bar that isn't
operating under rule 6, with no active label, while the active playbook is invalidation-blocked,
and whenever the active playbook's primitive is out of position (flat playbooks always are). The
formula itself gives 0 in CRASH (cap 0) and when uncertain (m_rule5 = 0).

**Rebalancing** (`rebalance`), from the target and the current exposure (shares x the
decision's last close / equity), in this order:

1. `ceiling`: the target is at most the bar's ceiling, the smallest of the max position, the
   active label's hard cap (0 with no active label) and, on the last fillable decision, the
   overnight cap (A3; `quant.decide.intents.ceiling` builds it);
2. **drift freeze and the daily-loss lock** (flags): the target may not exceed the current
   exposure (they block entries and increases only; exits and stops keep working, C5);
3. **the closing-bar decision** (`reduce_only`): the target may not exceed the current exposure;
4. **forced**: a reduction (target < current) is *forced* when rule 4, 5 or 6 fired, the engine
   is halted, or the current exposure is above the ceiling (a cap clamp, the overnight cap);
5. **deadband** (design call 9): the position moves to the target only when |target - current|
   >= `deadband` (0.10) of equity, compared exactly on decimal values (`exceeds_deadband`), or
   when the reduction is forced, or when the target is 0 (an exit to 0: a signal exit, a stop, a
   take-profit or an invalidation, exempt like forced reductions). Otherwise it stays where it is.

Everything here is pure. Comparisons on configured thresholds are exact on the decimal values
(`Fraction(repr(x))`), as switching's rules 4 and 5 are, independent of any decimal context.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from pydantic import Field, model_validator

from quant.core.calibration import PROBABILITY_SLACK, Calibration, r1_calibration_config
from quant.core.canonical import sha256_hex
from quant.core.model import StrictModel
from quant.core.regime import Label
from quant.decide import exact_decimal
from quant.decide.config import SizingConfig
from quant.decide.playbook import Playbook

NO_CALIBRATION_CAP: Final = r1_calibration_config().failed_cap
"""The cap fraction of every label when no calibration exists (C13: every label fails)."""


class Caps(StrictModel):
    """risk.yaml's limits that sizing and intents read, as data the driver builds from
    `quant.risk.limits.RiskConfig` (decide never imports risk):

    - `label_caps`: each label's hard cap (`RiskConfig.label_caps.cap(label)`), every Label;
    - `max_position`: `max_gross_exposure` (1.0, D9: no leverage);
    - `overnight_cap`: on the last fillable decision (A3: 0.5);
    - `single_order_cap`: the most one intent may move, BUY or SELL (0.5, design call 11).

    Every value is in [0, 1]. `caps_hash` goes into every DecisionRecord."""

    label_caps: dict[Label, float]
    max_position: float = Field(ge=0.0, le=1.0)
    overnight_cap: float = Field(ge=0.0, le=1.0)
    single_order_cap: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def valid_labels(self) -> Caps:
        if set(self.label_caps) != set(Label) or any(
            not 0 <= v <= 1 for v in self.label_caps.values()
        ):
            raise ValueError("label_caps must hold every Label in [0, 1]")
        return self

    def cap(self, label: Label | None) -> float:
        """`label_caps[label]`, or 0.0 for None (no active label sizes nothing). ValueError
        when `label_caps` doesn't hold every Label exactly once with a value in [0, 1]."""
        if set(self.label_caps) != set(Label) or any(
            not 0.0 <= v <= 1.0 for v in self.label_caps.values()
        ):
            raise ValueError("label_caps must hold every Label in [0, 1]")
        return 0.0 if label is None else float(self.label_caps[label])

    def caps_hash(self) -> str:
        """`quant.core.canonical.sha256_hex(self)`."""
        return sha256_hex(self)


@dataclass(frozen=True)
class Sizing:
    """The formula's factors for one bar (module docstring), each as used:

    - `label`; `cap` (hard cap), `cap_fraction`, `max_size` (the playbook's), `f_star` and
      `kelly` (= kelly_fraction x f*);
    - `base` = min(cap x cap_fraction, max_size, kelly);
    - `p_raw` (the filtered P(label)), `p_calibrated` (P~), `entropy` (H_t), `m_rule4`,
      `m_rule5`;
    - `value` = base x P~ x (1 - H_t) x m_rule4 x m_rule5, clipped into [0, 1].
    """

    label: Label
    cap: float
    cap_fraction: float
    max_size: float
    f_star: float
    kelly: float
    base: float
    p_raw: float
    p_calibrated: float
    entropy: float
    m_rule4: float
    m_rule5: float
    value: float


def size(
    label: Label,
    *,
    playbook: Playbook,
    cap: float,
    calibration: Calibration | None,
    f_star: float,
    label_prob: float,
    entropy: float,
    m_rule4: float,
    m_rule5: float,
    config: SizingConfig,
) -> Sizing:
    """The formula of the module docstring for `label` under `playbook` (its own label's
    playbook). With `calibration` None, `cap_fraction` is NO_CALIBRATION_CAP and P~ the raw
    `label_prob`. Pure. ValueError, naming it, when `playbook.label` isn't `label`, or `cap`,
    `label_prob` or `entropy` lies outside [0, 1] (`label_prob` within the calibration's
    rounding slack is clipped, as `Calibration.calibrated` clips it), or `f_star` is negative;
    ModelError from `Calibration.calibrated` for a probability it refuses."""
    if playbook.label != label:
        raise ValueError("playbook must be the label's")
    if not 0.0 <= cap <= 1.0:
        raise ValueError("cap must be in [0, 1]")
    if not (-PROBABILITY_SLACK <= label_prob <= 1 + PROBABILITY_SLACK):
        raise ValueError("label_prob must be a probability")
    if not 0.0 <= entropy <= 1.0:
        raise ValueError("entropy must be in [0, 1]")
    if not f_star >= 0:
        raise ValueError("f_star must be >= 0")
    if calibration is None:
        frac = NO_CALIBRATION_CAP
        p = min(1.0, max(0.0, float(label_prob)))
    else:
        frac = calibration.cap_fraction(label)
        p = calibration.calibrated(label, float(label_prob))
    kelly = config.kelly_fraction * f_star
    base = min(cap * frac, playbook.max_size, kelly)
    value = base * p * (1.0 - entropy) * m_rule4 * m_rule5
    value = min(1.0, max(0.0, value))
    return Sizing(
        label,
        cap,
        frac,
        playbook.max_size,
        f_star,
        kelly,
        base,
        float(label_prob),
        p,
        entropy,
        m_rule4,
        m_rule5,
        value,
    )


class ZeroReason(StrEnum):
    """Why the target is 0 before the formula (`target`), in the order they are checked."""

    HALTED = "halted"
    UNHEALTHY = "unhealthy"  # rule 6: not operating on this bar
    NO_ACTIVE = "no_active"
    BLOCKED = "blocked"  # the active playbook is invalidation-blocked (design call 6)
    OUT_OF_POSITION = "out_of_position"  # the active primitive holds no position


@dataclass(frozen=True)
class Target:
    """The bar's target exposure, a fraction of equity in [0, 1]: `value`; `zero` names why it
    is 0 before the formula (None when the formula ran); `sizing` holds the formula's factors
    (None when it didn't run)."""

    value: float
    zero: ZeroReason | None
    sizing: Sizing | None


def target(
    active: Label | None,
    *,
    playbook: Playbook | None,
    in_position: bool,
    operating: bool,
    blocked: bool,
    halted: bool,
    caps: Caps,
    calibration: Calibration | None,
    f_star: float,
    label_prob: float | None,
    entropy: float | None,
    m_rule4: float,
    m_rule5: float,
    config: SizingConfig,
) -> Target:
    """The bar's target (module docstring, "Zero" then "The target"): 0 with the first
    ZeroReason that holds (HALTED, UNHEALTHY when not `operating`, NO_ACTIVE when `active` is
    None, BLOCKED, OUT_OF_POSITION when not `in_position`); else `size(active, cap=caps.cap(
    active), ...)` and its value. `playbook`, `label_prob` and `entropy` may be None only when a
    ZeroReason applies; otherwise ValueError naming the missing one. Pure."""
    for cond, reason in (
        (halted, ZeroReason.HALTED),
        (not operating, ZeroReason.UNHEALTHY),
        (active is None, ZeroReason.NO_ACTIVE),
        (blocked, ZeroReason.BLOCKED),
        (not in_position, ZeroReason.OUT_OF_POSITION),
    ):
        if cond:
            return Target(0.0, reason, None)
    assert active is not None  # noqa: S101 - narrowed by the zero-reason checks
    if playbook is None:
        raise ValueError("playbook is required")
    if label_prob is None:
        raise ValueError("label_prob is required")
    if entropy is None:
        raise ValueError("entropy is required")
    s = size(
        active,
        playbook=playbook,
        cap=caps.cap(active),
        calibration=calibration,
        f_star=f_star,
        label_prob=label_prob,
        entropy=entropy,
        m_rule4=m_rule4,
        m_rule5=m_rule5,
        config=config,
    )
    return Target(s.value, None, s)


def exceeds_deadband(target: float, current: float, deadband: float) -> bool:
    """|target - current| >= deadband, exactly on the decimal values: `abs(Fraction(repr(
    target)) - Fraction(repr(current))) >= Fraction(repr(deadband))`, so 0.35 against 0.25 is a
    gap of exactly 0.10 and resizes (float subtraction gives 0.09999999999999998)."""
    return abs(exact_decimal(target) - exact_decimal(current)) >= exact_decimal(deadband)


@dataclass(frozen=True)
class Rebalance:
    """Where the position should go on this bar.

    - `desired`: the exposure the position should have after the decision: the target after
      steps 1-3 when it moves, else the current exposure;
    - `move`: the position moves on this bar: forced, an exit to 0, or outside the deadband,
      and the target differs from the current exposure. decide makes an intent only when it
      moves (`quant.decide.intents.order`), so a held position never trades on rounding;
    - `ceiling`: the ceiling applied (step 1);
    - `limited`: the freeze, the loss lock or the closing bar held the target at the current
      exposure (steps 2-3);
    - `forced`: the move is a forced reduction (step 4);
    - `held`: the deadband kept the position where it was (step 5).
    """

    desired: float
    move: bool
    ceiling: float
    limited: bool
    forced: bool
    held: bool


def rebalance(
    target: float,
    current: float,
    *,
    ceiling: float,
    deadband: float,
    rule_forced: bool,
    halted: bool,
    frozen: bool,
    locked: bool,
    reduce_only: bool,
) -> Rebalance:
    """Steps 1-5 of the module docstring. `rule_forced` is rule 4, 5 or 6 firing on this bar;
    `halted`, `frozen` (drift freeze) and `locked` (the daily-loss lock) are the flags;
    `reduce_only` the closing-bar decision. The position moves to the target after steps 1-3
    when the move is forced, the target is 0, or it `exceeds_deadband` the current exposure,
    and the target isn't the current exposure; `held` is True when only the deadband stopped
    it. ValueError, its message naming it, when
    `target`, `current` or `ceiling` isn't finite and >= 0 (the system is long or flat, D9).
    Pure."""
    for name, v in (("target", target), ("current", current), ("ceiling", ceiling)):
        if not (math.isfinite(v) and v >= 0):
            raise ValueError(f"{name} must be finite and >= 0")
    t = min(target, ceiling)
    limited = False
    if (frozen or locked or reduce_only) and t > current:
        t = current
        limited = True
    forced = exact_decimal(t) < exact_decimal(current) and (
        rule_forced or halted or exact_decimal(current) > exact_decimal(ceiling)
    )
    if t != current and (forced or t == 0.0 or exceeds_deadband(t, current, deadband)):
        return Rebalance(t, True, ceiling, limited, forced, False)
    return Rebalance(current, False, ceiling, limited, forced, t != current)
