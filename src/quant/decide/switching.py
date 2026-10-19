"""The switching engine: health (rule 6), takeover (rules 1-3), A1's early de-risk, uncertainty
(rule 5), early de-risk sizing (rule 4), the rule trace, and invalidation blocks (spec §8, A1,
C4, D5).

`step(state, probs, cause=..., config=..., caps=...)` evaluates one decision bar and returns a
`SwitchOutcome`. It is pure. `config` is engine.yaml's `switching` section; `caps` is each
label's hard cap (risk.yaml's `label_caps`, passed in as data: decide never imports risk).

**State** (`SwitchState`): the active label (None until the first takeover), the challenger and
its consecutive count, decision bars since the last switch (None before the first), the health
streak, and the invalidation block on the active label with its re-confirmation count.

**Each bar, in order:**

0. **Cooldown clock.** `since_switch` advances by one on every decision bar, healthy or not
   (None stays None).
1. **Health, rule 6** (quant.decide.health): `health_step(state.health_streak, cause)`. An
   unhealthy bar (a cause) has no usable probabilities: the challenger's count and a block's
   re-confirmation count reset to 0, no switch can happen, rules 1-5 are not evaluated (their
   multipliers are 1.0, their trace entries not fired with the cause as the reason), and the
   outcome isn't operating (target 0, entries blocked). A healthy bar that isn't yet operating
   (the recovery bar) is evaluated in full; only `operating` is False.
2. **Takeover, rules 1-3 and A1**, on a healthy bar. Let L = `hmm.labels.top_label(label_probs)`
   (ties to the most cautious) and P its probability. A bar *qualifies* L when P >= takeover_prob
   (0.70). When L is the active label, or the bar doesn't qualify L, the challenger is cleared
   (count 0). Otherwise, if L is the challenger its count goes up by one, else L becomes the
   challenger with count 1. Then the challenger takes over:
   - **cold start** (no active label, C4): once its count >= persistence_bars (3), whatever its
     cap, CRASH included; no cooldown applies before the first switch;
   - **de-risk, A1**: when its cap is *lower* than the active label's, on its first qualifying
     bar (count >= 1), ignoring the cooldown;
   - **risk-increasing** (an equal or higher cap, design call 4): once its count >=
     persistence_bars **and** the cooldown has expired, `since_switch >= cooldown_bars` (a
     switch at bar s allows the next at bar s + 7). The count keeps accumulating during the
     cooldown, so the switch happens on the first bar where both hold (design call 3).
   A switch sets the active label to L, clears the challenger, sets `since_switch` to 0 (a
   de-risk switch restarts the cooldown too), and clears the invalidation block.
3. **Invalidation block** (design call 6): while the active label is blocked
   (`invalidate`), a healthy bar that qualifies the active label adds one to `reconfirm`; any
   other bar resets it to 0. At persistence_bars (3) consecutive such bars, counted from the
   bar after the invalidation and with no cooldown, the block clears on that bar. A switch away
   drops the block (taking the label over again later is itself the fresh takeover).
4. **Uncertainty, rule 5**: the top two label probabilities of `label_probs` within
   uncertainty_gap of each other, `Decimal(repr(p1)) - Decimal(repr(p2)) <= Decimal(repr(gap))`
   (inclusive, design call 8: exactly 0.15 fires) -> `m_rule5` = 0.0, else 1.0. A distribution
   with one label has no second, and rule 5 doesn't fire.
5. **Early de-risk, rule 4**: `Decimal(repr(q_STRESS)) + Decimal(repr(q_CRASH)) >
   Decimal(repr(derisk_prob))` over `next_label_probs` (a missing label counts 0; strict, so
   exactly 0.25 doesn't fire) -> `m_rule4` = derisk_multiplier (0.5), else 1.0.

Rules 4-6 act on the bar that triggers them, every bar, whether or not a label is active.

**The rule trace** (spec §8's "Log"): every outcome has one `RuleTrace` per `RuleId`, in the
order R6, R1, R2, R3, A1, R5, R4, each with whether it fired and a non-empty reason:

- R6 fired: the bar isn't operating (unhealthy, or recovering);
- R1 fired: a challenger (a top label other than the active one) qualified on this bar;
- R2 fired: the challenger's count reached persistence_bars on this bar or before;
- R3 fired: the cooldown vetoed a risk-increasing switch that R1 and R2 allow on this bar;
- A1 fired: a de-risk switch happened on this bar;
- R5 fired: `m_rule5` is 0; R4 fired: `m_rule4` is the de-risk multiplier.

The DecisionRecord (B8b) adds every label and next-bar probability and the entropy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from pydantic import Field

from quant.core.model import StrictModel
from quant.core.regime import Label
from quant.decide import exact_decimal
from quant.decide.config import SwitchingConfig
from quant.decide.health import Cause, Health, health_step
from quant.hmm.labels import top_label


class RuleId(StrEnum):
    """The rule ids of spec §8 and A1, in trace order."""

    R6 = "R6"
    R1 = "R1"
    R2 = "R2"
    R3 = "R3"
    A1 = "A1"
    R5 = "R5"
    R4 = "R4"


@dataclass(frozen=True)
class RuleTrace:
    """One rule on one bar: whether it fired, and why (never empty)."""

    rule: RuleId
    fired: bool
    reason: str


@dataclass(frozen=True)
class BarProbs:
    """The filter's outputs switching reads for one bar: the filtered label probabilities, the
    next-bar ones (`FilterOutput.label_probs` and `.next_label_probs`, keyed by the model's
    labels; a label the model lacks is absent, probability 0) and the normalized entropy."""

    label_probs: Mapping[Label, float]
    next_label_probs: Mapping[Label, float]
    entropy: float


class SwitchState(StrictModel):
    """The switching state, as plain JSON data for EngineState (ARCHITECTURE §4.3).

    - `active`: the active label, None until the first takeover (C4);
    - `challenger`, `count`: the challenging label and its consecutive qualifying bars (None
      and 0 when there is none);
    - `since_switch`: decision bars since the last switch, None before the first;
    - `health_streak`: rule 6's streak (quant.decide.health);
    - `blocked`: the active label's playbook is invalidated (design call 6), False when there
      is no active label;
    - `reconfirm`: consecutive bars re-confirming a blocked active label (0 when not blocked).
    """

    active: Label | None
    challenger: Label | None
    count: int = Field(ge=0)
    since_switch: int | None = Field(ge=0)
    health_streak: int = Field(ge=0)
    blocked: bool
    reconfirm: int = Field(ge=0)

    @classmethod
    def cold(cls, config: SwitchingConfig) -> SwitchState:
        """A run's first state (C4): no active label, no challenger, no switch yet, a full
        health streak (`config.health_recovery_bars`, design call 5), not blocked."""
        return cls(
            active=None,
            challenger=None,
            count=0,
            since_switch=None,
            health_streak=config.health_recovery_bars,
            blocked=False,
            reconfirm=0,
        )


@dataclass(frozen=True)
class SwitchOutcome:
    """One bar's switching result.

    - `state`: the state after the bar;
    - `active`: `state.active`;
    - `switched`: the active label changed on this bar; `derisk`: by A1;
    - `health`: rule 6's verdict (`operating` False means target 0, entries blocked);
    - `blocked`: the active playbook is invalidation-blocked after this bar;
    - `m_rule4`, `m_rule5`: the size multipliers of rules 4 and 5;
    - `trace`: the rule trace, one entry per RuleId in trace order.
    """

    state: SwitchState
    active: Label | None
    switched: bool
    derisk: bool
    health: Health
    blocked: bool
    m_rule4: float
    m_rule5: float
    trace: tuple[RuleTrace, ...]

    @property
    def entries_allowed(self) -> bool:
        """Entries may happen under the active playbook on this bar: operating, a label is
        active, and it isn't blocked."""
        return self.health.operating and self.active is not None and not self.blocked

    def fired(self, rule: RuleId) -> bool:
        """Whether `rule` fired on this bar."""
        return next(t.fired for t in self.trace if t.rule == rule)


def step(
    state: SwitchState,
    probs: BarProbs | None,
    *,
    cause: Cause | None,
    config: SwitchingConfig,
    caps: Mapping[Label, float],
) -> SwitchOutcome:
    """One decision bar (module docstring). `cause` is None for a healthy bar, which must carry
    `probs`; an unhealthy bar's `probs` may be None and are ignored if given. ValueError, its
    message naming `probs` or `caps`, for a healthy bar without probs or with empty
    `label_probs`, or `caps` that don't hold every Label."""
    if set(caps) != set(Label):
        raise ValueError("caps must contain every Label")
    if cause is None and (probs is None or not probs.label_probs):
        raise ValueError("healthy bars require nonempty label probs")
    since = None if state.since_switch is None else state.since_switch + 1
    health = health_step(state.health_streak, cause, recovery_bars=config.health_recovery_bars)
    active, challenger, count = state.active, state.challenger, state.count
    blocked, reconfirm = state.blocked, state.reconfirm
    switched = derisk = False
    r1 = r2 = r3 = r4 = r5 = False
    reasons: dict[RuleId, str] = {}
    m_rule4 = m_rule5 = 1.0
    if cause is not None:
        challenger, count, reconfirm = None, 0, 0
        reasons = {rule: f"unhealthy: {cause.value}" for rule in RuleId}
    elif probs is not None:
        label_probs = probs.label_probs
        top = top_label(label_probs)
        qualifies = exact_decimal(label_probs[top]) >= exact_decimal(config.takeover_prob)
        if active == top or not qualifies:
            challenger, count = None, 0
        else:
            r1 = True
            if challenger == top:
                count += 1
            else:
                challenger, count = top, 1
            r2 = count >= config.persistence_bars
            cooled = since is None or since >= config.cooldown_bars
            if active is None:
                takeover = r2
            elif caps[top] < caps[active]:
                takeover = derisk = True
            else:
                takeover = r2 and cooled
                r3 = r2 and not cooled
            if takeover:
                active, challenger, count, since, switched = top, None, 0, 0, True
                blocked, reconfirm = False, 0
        if blocked and not switched:
            if active == top and qualifies:
                reconfirm += 1
                if reconfirm >= config.persistence_bars:
                    blocked, reconfirm = False, 0
            else:
                reconfirm = 0
        # Fractions of decimal representations keep sums/differences exact under any context.
        top2 = sorted(label_probs.values(), reverse=True)
        if len(top2) >= 2 and exact_decimal(top2[0]) - exact_decimal(top2[1]) <= exact_decimal(
            config.uncertainty_gap
        ):
            r5 = True
            m_rule5 = 0.0
        next_probs = probs.next_label_probs
        if exact_decimal(next_probs.get(Label.STRESS, 0.0)) + exact_decimal(
            next_probs.get(Label.CRASH, 0.0)
        ) > exact_decimal(config.derisk_prob):
            r4 = True
            m_rule4 = config.derisk_multiplier
        reasons.update(
            {
                RuleId.R1: f"top={top.value}, probability={label_probs[top]}, qualifies={qualifies}",
                RuleId.R2: f"challenger persistence {'met' if r2 else 'not met'} ({config.persistence_bars} bars)",
                RuleId.R3: f"cooldown {'veto' if r3 else 'no veto'}; previous age={state.since_switch}",
                RuleId.A1: "lower-cap takeover" if derisk else "no lower-cap takeover",
                RuleId.R5: f"top two probabilities={top2[:2]}, inclusive gap={config.uncertainty_gap}",
                RuleId.R4: f"next STRESS={next_probs.get(Label.STRESS, 0.0)}, CRASH={next_probs.get(Label.CRASH, 0.0)}; threshold > {config.derisk_prob}",
            }
        )
    if cause is None:
        reasons[RuleId.R6] = (
            f"healthy streak={health.streak}/{config.health_recovery_bars}; operating={health.operating}"
        )
    new = SwitchState(
        active=active,
        challenger=challenger,
        count=count,
        since_switch=since,
        health_streak=health.streak,
        blocked=blocked,
        reconfirm=reconfirm,
    )
    trace = (
        RuleTrace(RuleId.R6, not health.operating, reasons[RuleId.R6]),
        RuleTrace(RuleId.R1, r1, reasons[RuleId.R1]),
        RuleTrace(RuleId.R2, r2, reasons[RuleId.R2]),
        RuleTrace(RuleId.R3, r3, reasons[RuleId.R3]),
        RuleTrace(RuleId.A1, derisk, reasons[RuleId.A1]),
        RuleTrace(RuleId.R5, r5, reasons[RuleId.R5]),
        RuleTrace(RuleId.R4, r4, reasons[RuleId.R4]),
    )
    return SwitchOutcome(new, active, switched, derisk, health, blocked, m_rule4, m_rule5, trace)


def invalidate(state: SwitchState) -> SwitchState:
    """The state with the active label's playbook blocked (design call 6), its re-confirmation
    count 0, counted from the next bar. Called by decide after an invalidation hit on this bar.
    Blocking a blocked label restarts its count. ValueError, its message naming the `active`
    label, when no label is active."""
    if state.active is None:
        raise ValueError("no active")
    return state.model_copy(update={"blocked": True, "reconfirm": 0})
