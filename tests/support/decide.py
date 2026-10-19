"""Helpers for B8's decision tests: the release files, label distributions, synthetic price
bars, a slow reference for the indicators, a plain reference of the takeover rules, and (B8b)
decide() on the harness's synthetic world (`DecideFixture`).

The references are oracles. Nothing in them streams: each value is recomputed from the whole
list of bars (or probabilities) before it, with `math.fsum` sums and each EMA in closed form. They
read only plain numbers and the rules as SPEC §7-§8 and the B8 pack write them, never the code
under test.
"""

from __future__ import annotations

import functools
import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from quant.core.regime import Label
from quant.data.frames import HourlyBar
from quant.decide.config import SwitchingConfig
from quant.decide.switching import BarProbs
from tests.support.features import synthetic_bars
from tests.support.risk import config as risk_config

REPO = Path(__file__).resolve().parents[2]
ENGINE_YAML = REPO / "config" / "engine.yaml"
STRATEGY = REPO / "strategy.md"
PLAYBOOKS = {label: REPO / "playbooks" / f"{label.value}.md" for label in Label}

# r1's switching section, written out (the values of config/engine.yaml, SPEC §8).
SWITCHING = SwitchingConfig(
    takeover_prob=0.70,
    persistence_bars=3,
    cooldown_bars=7,
    derisk_prob=0.25,
    derisk_multiplier=0.5,
    uncertainty_gap=0.15,
    health_recovery_bars=2,
)

# risk.yaml's hard caps as data, the way the driver hands them to decide.
CAPS: Mapping[Label, float] = {label: risk_config().label_caps.cap(label) for label in Label}


# --- release files ---------------------------------------------------------------------------


def front_matter(path: Path) -> dict[str, Any]:
    """The YAML between a file's first two `---` lines, read with yaml.safe_load."""
    lines = path.read_text(encoding="utf-8").split("\n")
    assert lines[0] == "---"
    end = lines[1:].index("---") + 1
    data = yaml.safe_load("\n".join(lines[1:end]))
    assert isinstance(data, dict)
    return data


def playbook_text(front: Mapping[str, Any], body: str = "\nProse for the owner.\n") -> str:
    """A playbook (or manifest) file: `front` as YAML front matter, then `body`."""
    return "---\n" + yaml.safe_dump(dict(front), sort_keys=False) + "---\n" + body


def copy_release(root: Path) -> Path:
    """A copy of the repo's r1 release (strategy.md and playbooks/) under `root`."""
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy(STRATEGY, root / "strategy.md")
    (root / "playbooks").mkdir()
    for label, path in PLAYBOOKS.items():
        shutil.copy(path, root / "playbooks" / f"{label.value}.md")
    return root


# --- label distributions ---------------------------------------------------------------------


def probs(**values: float) -> dict[Label, float]:
    """A label distribution over all four labels: the named ones, and the rest of the mass
    spread evenly over the unnamed ones (CALM_UP=0.8 gives the other three 0.2 / 3 each)."""
    named = {Label(name): value for name, value in values.items()}
    rest = [label for label in Label if label not in named]
    left = 1.0 - math.fsum(named.values())
    out = dict(named)
    for label in rest:
        out[label] = left / len(rest)
    return {label: out[label] for label in Label}


def bar(
    label_probs: Mapping[Label, float], next_probs: Mapping[Label, float] | None = None
) -> BarProbs:
    """BarProbs with these label probabilities; next-bar ones default to calm ones (CALM_UP
    0.85, STRESS and CRASH 0.025 each), so rule 4 doesn't fire unless a test asks."""
    if next_probs is None:
        next_probs = {Label.CALM_UP: 0.85, Label.CHOP: 0.1, Label.STRESS: 0.025, Label.CRASH: 0.025}
    return BarProbs(dict(label_probs), dict(next_probs), 0.5)


def top(label: Label, p: float = 0.8) -> BarProbs:
    """A bar with `label` the top label at probability `p`, the rest spread evenly."""
    return bar(probs(**{label.value: p}))


# --- the takeover rules, as a plain reference ---------------------------------------------------


@dataclass(frozen=True)
class RefBar:
    """One bar for the reference: None for an unhealthy bar, else the label probabilities."""

    label_probs: Mapping[Label, float] | None


def ref_actives(
    bars: Sequence[RefBar], caps: Mapping[Label, float], config: SwitchingConfig = SWITCHING
) -> list[Label | None]:
    """The active label after each bar under rules 1-3 and A1, recomputed for every bar from
    the whole history: the last switch before it, and how many consecutive bars ending at it
    qualified its top label (healthy, top label, P >= takeover_prob; an unhealthy bar breaks the
    run)."""
    actives: list[Label | None] = []
    active: Label | None = None
    last_switch: int | None = None
    for t, current in enumerate(bars):
        if current.label_probs is not None:
            dist = current.label_probs
            leader = max((label for label in Label if label in dist), key=lambda lb: dist[lb])
            run = 0
            for back in range(t, -1, -1):
                past = bars[back].label_probs
                if past is None:
                    break
                past_top = max((lb for lb in Label if lb in past), key=lambda lb: past[lb])
                if past_top != leader or not past[leader] >= config.takeover_prob:
                    break
                # The run counts only bars since the label last became (or stayed) active.
                if back <= (last_switch if last_switch is not None else -1):
                    break
                run += 1
            if leader != active and run >= 1:
                cooled = last_switch is None or t - last_switch >= config.cooldown_bars
                if active is None:
                    switch = run >= config.persistence_bars
                elif caps[leader] < caps[active]:
                    switch = True
                else:
                    switch = run >= config.persistence_bars and cooled
                if switch:
                    active, last_switch = leader, t
        actives.append(active)
    return actives


# --- price bars ------------------------------------------------------------------------------


@functools.cache
def _template() -> tuple[HourlyBar, ...]:
    """Real hourly bar specs on the XNYS calendar (FIT, 2016), prices from seeded synthetic
    minutes. Tests replace the prices they need."""
    return tuple(synthetic_bars(date(2016, 1, 4), date(2016, 3, 31), seed=8))


def market_bars(count: int | None = None) -> list[HourlyBar]:
    """The template's synthetic bars, all of them or the first `count`."""
    bars = list(_template())
    return bars if count is None else bars[:count]


def priced(
    i: int, close: float, *, low: float | None = None, high: float | None = None
) -> HourlyBar:
    """Template bar `i` with open = close, and the given low and high (default: the close)."""
    lo = close if low is None else low
    hi = close if high is None else high
    return replace(_template()[i], open=close, high=max(hi, close), low=min(lo, close), close=close)


def priced_series(closes: Sequence[float]) -> list[HourlyBar]:
    """Consecutive template bars with these closes, open = close, high and low 0.5 either side."""
    return [priced(i, c, low=c - 0.5, high=c + 0.5) for i, c in enumerate(closes)]


# --- the indicators, as a slow reference --------------------------------------------------------


@dataclass(frozen=True)
class RefIndicators:
    ema: dict[int, float]
    atr: float | None
    mean: float | None
    sd: float | None
    z: float | None


def _ema(closes: Sequence[float], i: int, span: int) -> float:
    """EMA_i in closed form, the recursion seeded with the first close, unrolled."""
    alpha = 2.0 / (span + 1)
    keep = 1.0 - alpha
    terms = [keep**i * closes[0]]
    terms.extend(alpha * keep ** (i - j) * closes[j] for j in range(1, i + 1))
    return math.fsum(terms)


def ref_indicators(
    bars: Sequence[HourlyBar], spans: Sequence[int], atr_bars: int | None, window: int | None
) -> list[RefIndicators]:
    """Every bar's indicators as the indicators module defines them, from the bars a stream
    consumes (complete, valid, in order)."""
    closes = [b.close for b in bars]
    tr = [bars[0].high - bars[0].low] + [
        max(bars[i].high, closes[i - 1]) - min(bars[i].low, closes[i - 1])
        for i in range(1, len(bars))
    ]
    out = []
    for i in range(len(bars)):
        ema = {span: _ema(closes, i, span) for span in spans}
        atr = None
        if atr_bars is not None and i + 1 >= atr_bars:
            atr = math.fsum(tr[i + 1 - atr_bars : i + 1]) / atr_bars
        mean = sd = z = None
        if window is not None and i + 1 >= window:
            last = closes[i + 1 - window : i + 1]
            mean = math.fsum(last) / window
            sd = math.sqrt(math.fsum((c - mean) ** 2 for c in last) / window)
            z = None if len(set(last)) == 1 else (closes[i] - mean) / sd
        out.append(RefIndicators(ema, atr, mean, sd, z))
    return out


def close_enough(a: float | None, b: float | None, rel: float = 1e-12) -> bool:
    """Both None, or within `rel` relative (absolute near zero)."""
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


# --- B8b: the release, configs and decide() on the synthetic world -----------------------------
# The decide tests run on the T-NL1 harness's synthetic world (tests.support.truncation.world():
# seeded minutes inside FIT, 2016, on the real NYSE calendar), under the lifecycle's start model
# (the fixture model relabeled to CALM_UP and CHOP), so nothing is fit per test and DEV stays
# dark. The decision under test is the third of DECISION_DAY (12:46 ET: it can fill, and isn't
# the last that can), after the model's activation on 2016-11-01 and before its refit.

DECISION_DAY = date(2016, 11, 15)
DECISION_INDEX = 2


@functools.cache
def r1_release() -> Any:
    """The repository's release r1 (`quant.decide.playbook.load_release`)."""
    from tests.support.truncation import harness_release

    return harness_release()


def r1_playbook(label: Label) -> Any:
    """r1's playbook for `label`."""
    return r1_release().playbook(label)


@functools.cache
def engine_config() -> Any:
    """The committed config/engine.yaml."""
    from tests.support.truncation import harness_engine

    return harness_engine()


@functools.cache
def r1_caps() -> Any:
    """risk.yaml's limits as decide's `Caps`, the way the driver builds them."""
    from tests.support.truncation import harness_caps

    return harness_caps(risk_config())


@dataclass(frozen=True, eq=False)
class DecideFixture:
    """One decision on the synthetic world, and what decide needs for it (module comment).

    - `world`: the harness world; `model`: the lifecycle's start model (two labels);
    - `decision`, `previous`: the decision under test and the one before it (same session);
    - `bar`: the decision bar, as the view at its tau shows it; `price_before`: the previous
      bar's close.
    """

    world: Any
    model: Any
    decision: Any
    previous: Any
    bar: HourlyBar
    price_before: float

    @staticmethod
    @functools.cache
    def get() -> DecideFixture:
        from quant.core.calendar import TradingCalendar
        from quant.core.schedule import decision_schedule
        from quant.data.policy import Purpose
        from tests.support.truncation import lifecycle_spec, world

        w = world()
        schedule = decision_schedule(TradingCalendar.xnys(), DECISION_DAY)
        decision, previous = schedule[DECISION_INDEX], schedule[DECISION_INDEX - 1]
        before, bar = w.store.view(decision.tau, as_of=decision.tau).hourly(
            "SPY", purpose=Purpose.OPERATE, last=2
        )
        assert bar.t_open == decision.bar.t_open
        assert bar.complete
        assert not decision.last_fillable
        assert not decision.reduce_only
        return DecideFixture(
            w, lifecycle_spec(w).start, decision, previous, bar, float(before.close)
        )

    def view(self, decision: Any = None) -> Any:
        """The world store's view at the decision's tau (as_of = tau)."""
        d = self.decision if decision is None else decision
        return self.world.store.view(d.tau, as_of=d.tau)

    def inputs(self, **overrides: Any) -> Any:
        """DecideInputs for the decision: r1, the start model, no calibration, the committed
        engine.yaml, risk.yaml's caps, a flat $100,000 account, no flags; `overrides` replace
        any of them by field name."""
        from quant.decide.core import Account, DecideInputs, Flags

        fields: dict[str, Any] = {
            "release": r1_release(),
            "model": self.model,
            "calibration": None,
            "engine": engine_config(),
            "caps": r1_caps(),
            "decision": self.decision,
            "account": Account(equity=100_000.0, position=0),
            "flags": Flags(),
            "code_commit": "c0ffee" * 6 + "c0ff",
        }
        fields.update(overrides)
        return DecideInputs(**fields)

    def decide(self, state: Any, view: Any = None, **overrides: Any) -> Any:
        """`decide(state, view or self.view(...), self.inputs(**overrides))`."""
        from quant.decide.core import decide

        d = overrides.get("decision", self.decision)
        return decide(state, self.view(d) if view is None else view, self.inputs(**overrides))

    def cold(self, *, seeded: bool = True) -> Any:
        """`EngineState.cold(...)` with the harness's seeded edge (f* > 0 for CALM_UP and
        CHOP), or an empty one."""
        from quant.decide.state import EngineState
        from tests.support.truncation import harness_edge

        return EngineState.cold(engine_config().switching, edge=harness_edge() if seeded else None)

    def warm(self) -> Any:
        """The state after deciding `previous` from a cold start: features, filter and
        indicators warm through the bar before the decision's. Computed once."""
        return _warm(self)

    def warm_state(
        self,
        *,
        active: Label | None,
        trade: Any = None,
        shadows: Mapping[Label, Any] | None = None,
        blocked: bool = False,
    ) -> Any:
        """`warm()` with switching settled on `active` (no challenger, the cooldown long
        expired, a full health streak, `blocked`), the live `trade` (flat when None) and
        `shadows` (the warm ones when None)."""
        from quant.decide.primitives import TradeState

        warm = self.warm()
        switching = warm.switching.model_copy(
            update={
                "active": active,
                "challenger": None,
                "count": 0,
                "since_switch": None if active is None else 50,
                "health_streak": SWITCHING.health_recovery_bars,
                "blocked": blocked and active is not None,
                "reconfirm": 0,
            }
        )
        return warm.model_copy(
            update={
                "switching": switching,
                "trade": TradeState.flat() if trade is None else trade,
                "shadows": dict(warm.shadows) if shadows is None else dict(shadows),
            }
        )

    def in_trade(self, *, stop_fraction: float = 0.9) -> Any:
        """A live trade entered on the previous bar at its close, its stop `stop_fraction` of
        that close (below the decision's price)."""
        from quant.decide.primitives import TradeState

        return TradeState(
            in_position=True,
            armed=True,
            entry_price=self.price_before,
            stop=self.price_before * stop_fraction,
            take_profit=None,
            bars_in_trade=1,
            peak_close=self.price_before,
        )

    def indicators(self, label: Label) -> Any:
        """A reference: `label`'s playbook indicators for the decision bar, from a fresh
        IndicatorStream fed every bar the view at tau shows (B8a's code, not decide's)."""
        from quant.decide.indicators import IndicatorConfig, IndicatorStream, indicators_for

        stream = IndicatorStream(IndicatorConfig.for_playbook(r1_playbook(label)))
        return indicators_for(stream, self.view(), "SPY", self.decision.bar)


@functools.cache
def _warm(fx: DecideFixture) -> Any:
    return fx.decide(fx.cold(), decision=fx.previous).state


@functools.cache
def day_states() -> tuple[tuple[Any, Any], ...]:
    """(decision, the state before it) for every decision of DECISION_DAY, decided in order
    from a cold start at the session's first (seeded edge, a flat account)."""
    from quant.core.calendar import TradingCalendar
    from quant.core.schedule import decision_schedule

    fx = DecideFixture.get()
    state = fx.cold()
    out = []
    for d in decision_schedule(TradingCalendar.xnys(), DECISION_DAY):
        out.append((d, state))
        state = fx.decide(state, decision=d).state
    return tuple(out)
