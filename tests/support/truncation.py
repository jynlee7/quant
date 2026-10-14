"""The T-NL1 truncation harness and T-NL2's provenance check (B5, SPEC §6, ARCHITECTURE §13).

For each cut time τ, every record at or before τ must be identical, bit for bit, across three
runs of the decision path (SPEC §6 T-NL1):

- (a) TRUNCATED: a store holding only the data known by τ;
- (b) FULL: the store holding the whole synthetic history;
- (c) GARBAGE: (a) with garbage written after τ: NaN and ±1e9 prices, shuffled bars, a late
  revision known after τ, and bars that have closed but aren't available yet.

The decision path is a pipeline of **layers**, registered in order in a `Registry`. A layer
reads the market only through its run's `Source` and the records of the layers before it, and
gives one `Record` per decision. B5 registers what exists before `decide()` (B8): the view (B2),
features and the scaler (B3), the filter and labels (B4) and, because it is cheap, the risk
verdicts on a fixed seeded intent per bar (B9). A later step that touches the decision path
adds its layer to `default_registry` (CLAUDE.md rule 4), and B8 replaces the per-layer records
with DecisionRecords and state hashes (ARCHITECTURE §13).

Records are compared by the canonical-JSON SHA-256 of each (`Record.digest`), never
approximately. A failure names the cut, the variant, the decision and the **first layer** whose
record differs there, so a leak is reported where it enters, not in every layer downstream.

**Slicing.** The full run (b) is the engine over the whole history, so it runs once, over every
decision, and each cut compares its records up to τ. Runs (a) and (c) depend on τ by
definition, so each cut runs them over the decisions up to τ on its own store; a store is a
SQLite backup of the full one with later rows deleted, which costs about a millisecond.
`check_truncation(..., slice_full=False)` recomputes (b) per cut instead, and the test of
planted leak 1 checks that both modes catch the same failures.

**T-NL2.** A record made with a model names it (`Provenance`: the artifact hash and `fit_end`),
and `provenance_failures` demands fit_end < τ for every such record, in every variant.

Everything here is synthetic, on the real NYSE calendar inside FIT (2016). DEV stays dark.
"""

from __future__ import annotations

import math
import random
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from functools import cache, cached_property, lru_cache
from itertools import pairwise
from typing import Any, Final, Protocol

import numpy as np
from structlog.testing import capture_logs

from quant.core.calendar import TradingCalendar
from quant.core.canonical import canonical_dumps, canonical_loads, sha256_hex
from quant.core.clock import SimClock
from quant.core.epoch import from_us, to_us
from quant.core.errors import FeatureError, ModelError
from quant.core.orders import Side, TradingMode
from quant.core.regime import Label
from quant.core.schedule import Decision, decision_schedule
from quant.data.frames import MinuteFrame
from quant.data.market_db import HISTORY, BarStore
from quant.data.policy import Purpose
from quant.data.view import PointInTimeView
from quant.features.menu import FeatureConfig, r1_features
from quant.features.scaler import Scaler
from quant.features.stream import FeatureStream, compute_features, features_for
from quant.hmm.filter import ForwardFilter
from quant.hmm.labels import r1_thresholds
from quant.hmm.model import HmmModel
from quant.hmm.select import Candidate
from quant.hmm.train import TrainingData, r1_train_config, train_candidate
from quant.risk.engine import MemorySpentApprovals, RiskApprovedOrder, RiskEngine, Veto
from quant.risk.limits import RiskConfig
from tests.support.bars import SYMBOL, revise_close, synthetic_days
from tests.support.risk import config as risk_config
from tests.support.risk import context, intent

# --- the synthetic world ---------------------------------------------------------------------
# Minutes from DATA_FIRST to DATA_LAST (tests.support.bars.synthetic_days, seed DATA_SEED). The
# 60-session warm-up ends on 2016-08-24, so the first warm bar is 2016-08-25's first; the range
# holds the end of DST (2016-11-06) and a half day (2016-11-25).
DATA_FIRST: Final = date(2016, 6, 1)
DATA_LAST: Final = date(2016, 12, 30)
DATA_SEED: Final = 3
# The clean fixture model is fit on [the first warm row's available_at, MODEL_FIT_END), and the
# filter layer runs a model from ACTIVATION on: its first record is the first decision after it.
MODEL_FIT_END: Final = datetime(2016, 11, 1, tzinfo=UTC)
ACTIVATION: Final = MODEL_FIT_END
# Planted leak 6: a model whose fit window ends after the decisions it serves (T-NL2).
LATE_FIT_END: Final = datetime(2016, 12, 1, tzinfo=UTC)
# A time after every bar of the full store: what a layer that peeks reads its view at.
HORIZON: Final = datetime(2017, 1, 10, tzinfo=UTC)
# The fixture model's training: train_candidate, K=2 diag, few restarts, serial.
FIXTURE_STATES: Final = 2
FIXTURE_COVARIANCE: Final = "diag"
FIXTURE_RESTARTS: Final = 3
FIXTURE_BASE_SEED: Final = 20160104
# How many bars the view layer reads per decision (`hourly(last=VIEW_DEPTH)`).
VIEW_DEPTH: Final = 16
# The risk layer's account: $100,000, flat, priced at the view's last close.
RISK_EQUITY: Final = 100_000.0
# Edge cuts every draw includes (`edge_cuts`).
EDGES: Final = ("first_warm", "model_first", "after_dst", "half_day_close", "session_first")


class Variant(StrEnum):
    TRUNCATED = "a"  # data known by τ
    FULL = "b"  # the whole history
    GARBAGE = "c"  # data known by τ, then garbage after τ


# --- records ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Provenance:
    """The model a record was made with: its `HmmModel.artifact_hash()` and `fit_end` (T-NL2)."""

    model: str
    fit_end: datetime


@dataclass(frozen=True)
class Record:
    """One layer's output for one decision.

    - `tau`: the decision time;
    - `layer`: the layer's registered name;
    - `output`: a value with a canonical JSON form (`quant.core.canonical`): finite floats,
      string keys. A layer that refuses a decision (FeatureError, ModelError: rule 6) records
      `error_output(exc)` rather than raising;
    - `provenance`: the model behind the record, None for a layer that runs none;
    - `state`: `sha256_hex` of the layer's state after this decision (the FeatureStream's or
      ForwardFilter's snapshot, the RiskState), None for a stateless layer.
    """

    tau: datetime
    layer: str
    output: Any
    provenance: Provenance | None = None
    state: str | None = None

    def digest(self) -> str:
        """`quant.core.canonical.sha256_hex` of {"tau", "layer", "output", "provenance": null or
        {"model", "fit_end"}, "state"}. Raises CanonicalJSONError for an output without a
        canonical form, which the harness reports as that layer's failure."""
        return self._digest

    @cached_property
    def _digest(self) -> str:
        return sha256_hex(
            {
                "tau": self.tau,
                "layer": self.layer,
                "output": self.output,
                "provenance": None if self.provenance is None else asdict(self.provenance),
                "state": self.state,
            }
        )


def error_output(exc: Exception) -> dict[str, str]:
    """The output a layer records for a decision it refuses: {"error": the exception's class
    name, "message": str(exc)}."""
    return {"error": type(exc).__name__, "message": str(exc)}


# --- layers ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    """What a layer may read in one run: one variant's store. A clean layer reads only
    `view(d.tau)` for each decision d; a planted leak reads a view at a later time, which is
    exactly the bug the harness must catch."""

    store: BarStore
    variant: Variant
    cut: datetime | None  # None for the full run

    def view(self, tau: datetime) -> PointInTimeView:
        """`store.view(tau, as_of=tau)`: the data as the backtest sees it at `tau`."""
        return self.store.view(tau, as_of=tau)


class Layer(Protocol):
    """One stage of the decision path. `run` gets every decision of the run, in time order,
    and the records of the layers registered before it (by layer name, aligned with
    `decisions`), and returns one Record per decision, in order, each named `name`. Its state
    starts fresh on every call: a run is the engine from the start of the decisions."""

    @property
    def name(self) -> str: ...

    def run(
        self,
        source: Source,
        decisions: Sequence[Decision],
        upstream: Mapping[str, Sequence[Record]],
    ) -> list[Record]: ...


class Registry:
    """The layers in pipeline order. Names are unique: registering a second layer with a taken
    name raises ValueError. `replace` is how a test plants a leak: a copy with one layer
    swapped for a substitute of the same name, in the same place."""

    def __init__(self, layers: Iterable[Layer] = ()) -> None:
        self._layers: dict[str, Layer] = {}
        for layer in layers:
            self.register(layer)

    def register(self, layer: Layer) -> None:
        if layer.name in self._layers:
            raise ValueError(f"duplicate layer {layer.name}")
        self._layers[layer.name] = layer

    def replace(self, layer: Layer) -> Registry:
        """A new registry with the layer named `layer.name` swapped for `layer`. KeyError when
        no layer has that name."""
        if layer.name not in self._layers:
            raise KeyError(layer.name)
        return Registry(layer if old.name == layer.name else old for old in self.layers)

    @property
    def layers(self) -> tuple[Layer, ...]:
        return tuple(self._layers.values())

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._layers)


class ViewLayer:
    """B2's view. Per decision: `source.view(d.tau).hourly(SYMBOL, purpose=OPERATE,
    last=VIEW_DEPTH)`, recorded as {"bars": [{"t_open", "t_close", "open", "high", "low",
    "close", "volume", "minutes", "complete", "available_at"} per bar, oldest first],
    "digest": that view's `digest()`}. Versions and known_at are left out, as the digest leaves
    them out. Stateless."""

    name: Final = "view"

    def __init__(self) -> None:
        self._records: dict[tuple[Any, ...], Record] = {}

    def run(
        self,
        source: Source,
        decisions: Sequence[Decision],
        upstream: Mapping[str, Sequence[Record]],
    ) -> list[Record]:
        records = []
        fields = (
            "t_open",
            "t_close",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "minutes",
            "complete",
            "available_at",
        )
        for d in decisions:
            view = source.view(d.tau)
            bars = view.hourly(SYMBOL, purpose=Purpose.OPERATE, last=VIEW_DEPTH)
            values = tuple(tuple(getattr(bar, field) for field in fields) for bar in bars)
            key = (d.tau, view.digest(), values)
            if key not in self._records:
                self._records[key] = Record(
                    d.tau,
                    self.name,
                    {
                        "bars": [dict(zip(fields, row, strict=True)) for row in values],
                        "digest": key[1],
                    },
                )
            records.append(self._records[key])
        return records


def features_output(raw: Any, scaled: Any) -> dict[str, list[float]]:
    """The features layer's output: {"raw": the feature vector, "scaled": it standardized},
    each a list of float64 in the config's column order."""
    return {"raw": [float(x) for x in raw], "scaled": [float(x) for x in scaled]}


class FeaturesLayer:
    """B3's features and scaler. One FeatureStream per run, fed through the view:
    `features_for(stream, source.view(d.tau), SYMBOL, d.bar)` for each decision, then
    `scaler.transform` with the config's columns. Records `features_output(raw, scaled)`, or
    `error_output` when features_for or the scaler refuses (a warm-up row is a FeatureError).
    `state` is the hash of the stream's snapshot. The scaler is fixed: the clean one is fit on
    the fixture model's window (`World.scaler`)."""

    name: Final = "features"

    def __init__(self, config: FeatureConfig, scaler: Scaler) -> None:
        self.config, self.scaler = config, scaler
        self._records: dict[tuple[Any, ...], Record] = {}

    def run(
        self,
        source: Source,
        decisions: Sequence[Decision],
        upstream: Mapping[str, Sequence[Record]],
    ) -> list[Record]:
        stream = FeatureStream(self.config)
        records = []
        for d in decisions:
            output: dict[str, Any]
            try:
                raw = features_for(stream, source.view(d.tau), SYMBOL, d.bar)
                output = features_output(
                    raw, self.scaler.transform(raw, columns=self.config.columns)
                )
            except (FeatureError, ModelError) as exc:
                output = error_output(exc)
            state = _feature_state_hash(stream)
            # Both vectors (or the complete error) and the actual state participate.
            key = (d.tau, canonical_dumps(output), state)
            if key not in self._records:
                self._records[key] = Record(d.tau, self.name, output, state=state)
            records.append(self._records[key])
        return records


class FilterLayer:
    """B4's filter and labels. A decision before `activation` records {"model": null} and no
    provenance (cold start). From the first decision at or after `activation`, one
    `ForwardFilter(model)` steps on the upstream features record's "raw" row of each decision;
    a decision whose features record is an error records that error and doesn't step. The
    output holds every `FilterOutput` field: "state_probs", "next_state_probs", "label_probs"
    and "next_label_probs" (keyed by label value), "entropy", "loglik", "top_label",
    "top_label_remaining". Every record from `activation` on carries the model's Provenance,
    and `state` is the hash of the filter's snapshot.

    The model is whatever it is given: the layer itself never checks fit_end against τ. That is
    T-NL2's job (`provenance_failures`), so planted leak 6 is this class with a model fit past
    `activation`."""

    name: Final = "filter"

    def __init__(self, model: HmmModel, *, activation: datetime) -> None:
        self.model, self.activation = model, activation

    def run(
        self,
        source: Source,
        decisions: Sequence[Decision],
        upstream: Mapping[str, Sequence[Record]],
    ) -> list[Record]:
        filt = ForwardFilter(self.model)
        provenance = Provenance(self.model.artifact_hash(), self.model.fit_end)
        records = []
        for d, feature in zip(decisions, upstream["features"], strict=True):
            if d.tau < self.activation:
                records.append(Record(d.tau, self.name, {"model": None}))
                continue
            output: object = feature.output
            if "raw" in feature.output:
                try:
                    output = asdict(
                        filt.step(
                            np.array(feature.output["raw"], dtype=np.float64),
                            columns=self.model.scaler.columns,
                        )
                    )
                except (FeatureError, ModelError) as exc:
                    output = error_output(exc)
            records.append(
                Record(d.tau, self.name, output, provenance, sha256_hex(filt.snapshot()))
            )
        return records


class RiskLayer:
    """B9's risk verdicts, before `decide()` exists (B8). Per run: a fresh `RiskEngine(config)`
    in PAPER mode on its own SimClock, with MemorySpentApprovals. Per decision, in order:

    - the price is the close of the last bar in the upstream view record (no bar: record
      {"skipped": "no bar"});
    - the clock moves to `d.executes_at` + 1 s, and the engine starts that session with
      RISK_EQUITY when it is later than its current one;
    - the intent is fixed and seeded: a BUY when the decision can fill, else a reduce-only
      SELL, of 1 to 200 shares drawn from `random.Random` seeded with (`seed`, d.tau); its
      label is the upstream filter record's "top_label", or None when it has none; a BUY's stop
      is 95% of the price;
    - the context: flat broker and book, RISK_EQUITY, quote bid = ask = limit = the price,
      healthy, not frozen, not halted.

    Records the verdict without its token and µs ids: for a RiskApprovedOrder {"verdict":
    "approved", "intent", "quantity", "limit_price", "notional", "issued_at", "expires_at",
    "flagged", "halts"}; for a PendingApproval {"verdict": "pending", "intent", "quantity",
    "limit_price", "notional", "issued_at", "expires_at"}; for a Veto {"verdict": "veto",
    "intent", "check", "reason", "halts"}. `state` is the hash of the engine's RiskState."""

    name: Final = "risk"

    def __init__(self, config: RiskConfig, *, seed: int) -> None:
        self.config, self.seed = config, seed

    def run(
        self,
        source: Source,
        decisions: Sequence[Decision],
        upstream: Mapping[str, Sequence[Record]],
    ) -> list[Record]:
        if not decisions:
            return []
        clock = SimClock(decisions[0].tau)
        engine = RiskEngine(
            self.config,
            clock=clock,
            mode=TradingMode.PAPER,
            calendar=TradingCalendar.xnys(),
            spent=MemorySpentApprovals(),
        )
        records = []
        # Verdicts and RiskState are the compared outputs, not rendered diagnostic logs.
        with capture_logs():
            for d, view, filtered in zip(
                decisions, upstream["view"], upstream["filter"], strict=True
            ):
                if not view.output["bars"]:
                    records.append(Record(d.tau, self.name, {"skipped": "no bar"}))
                    continue
                price = view.output["bars"][-1]["close"]
                clock.advance_to(d.executes_at + timedelta(seconds=1))
                day = clock.now().date()
                if engine.state.session is None or day > engine.state.session:
                    engine.start_session(day, equity=RISK_EQUITY)
                label = filtered.output.get("top_label")
                order = _risk_intent(
                    Side.SELL if d.reduce_only else Side.BUY,
                    _risk_quantity(self.seed, d.tau),
                    tau=d.tau,
                    label=None if label is None else Label(label),
                    reduce_only=d.reduce_only,
                    stop=0.95 * price,
                )
                verdict = engine.check(
                    order,
                    context(clock.now(), equity=RISK_EQUITY, bid=price, ask=price, limit=price),
                )
                output: dict[str, Any] = {"intent": order.model_dump(mode="python")}
                if isinstance(verdict, Veto):
                    output.update(
                        verdict="veto",
                        check=verdict.check,
                        reason=verdict.reason,
                        halts=verdict.halts,
                    )
                else:
                    output.update(
                        verdict="approved" if isinstance(verdict, RiskApprovedOrder) else "pending"
                    )
                    for key in ("quantity", "limit_price", "notional", "issued_at", "expires_at"):
                        output[key] = getattr(verdict, key)
                    if isinstance(verdict, RiskApprovedOrder):
                        output.update(flagged=verdict.flagged, halts=verdict.halts)
                records.append(Record(d.tau, self.name, output, state=sha256_hex(engine.state)))
        return records


# --- the world -------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class World:
    """The fixture every harness run shares, built once per session (`world()`).

    - `minutes`: the synthetic minutes, DATA_FIRST to DATA_LAST;
    - `conn`, `store`: an in-memory market.db holding them as HISTORY: the full run's store,
      and the source every per-cut store is backed up from;
    - `config`: r1's FeatureConfig;
    - `decisions`: every decision from the first warm bar's to DATA_LAST's last, in order;
    - `model`: the clean fixture model, `fixture_model(MODEL_FIT_END)`;
    - `scaler`: the Scaler fit on the model's window, equal to the model's own;
    - `edges`: `edge_cuts(decisions)`.
    """

    minutes: MinuteFrame
    conn: sqlite3.Connection
    store: BarStore
    config: FeatureConfig
    decisions: tuple[Decision, ...]
    model: HmmModel
    scaler: Scaler
    edges: Mapping[str, Decision]


@cache
def world() -> World:
    """The shared World, built on first use and cached for the session."""
    minutes, store = _history()
    config = r1_features()
    matrix = compute_features(
        store.view(HORIZON, as_of=HORIZON).hourly(SYMBOL, purpose=Purpose.OPERATE), config
    )
    first = from_us(int(matrix.t_open[0]))
    cal = TradingCalendar.xnys()
    decisions = tuple(
        d
        for day in cal.sessions(first.date(), DATA_LAST)
        for d in decision_schedule(cal, day)
        if d.bar.t_open >= first
    )
    scaler = Scaler.fit(matrix, start=from_us(int(matrix.available_at[0])), end=MODEL_FIT_END)
    return World(
        minutes,
        store._conn,
        store,
        config,
        decisions,
        fixture_model(MODEL_FIT_END),
        scaler,
        edge_cuts(decisions),
    )


@cache
def fixture_model(fit_end: datetime) -> HmmModel:
    """A model fit on the full store's warm feature rows with available_at in [the first warm
    row's available_at, `fit_end`): `quant.hmm.train.train_candidate` on a TrainingData of
    those rows (each row's raw log return from the previous consumed bar's close, sessions as
    ordinals), K=FIXTURE_STATES FIXTURE_COVARIANCE, FIXTURE_RESTARTS restarts from
    FIXTURE_BASE_SEED, a one-month validation block, r1's label thresholds, workers=1. Cached
    per `fit_end`."""
    _, store = _history()
    config = r1_features()
    bars = store.view(HORIZON, as_of=HORIZON).hourly(SYMBOL, purpose=Purpose.OPERATE)
    matrix = compute_features(bars, config)
    first = from_us(int(matrix.available_at[0]))
    matrix = matrix.window(first, fit_end)
    by_open = {
        to_us(b.t_open): (math.log(b.close / previous.close), b.session.toordinal())
        for previous, b in pairwise(bars)
    }
    data = TrainingData(
        matrix,
        np.array([by_open[int(t)][0] for t in matrix.t_open], dtype=np.float64),
        np.array([by_open[int(t)][1] for t in matrix.t_open], dtype=np.int64),
        first,
        fit_end,
        config.config_hash(),
    )
    config_train = r1_train_config().model_copy(
        update={
            "n_restarts": FIXTURE_RESTARTS,
            "base_seed": FIXTURE_BASE_SEED,
            "validation_months": 1,
        }
    )
    return train_candidate(
        data,
        Candidate(n_states=FIXTURE_STATES, covariance_type=FIXTURE_COVARIANCE),
        thresholds=r1_thresholds(),
        config=config_train,
        provenance={"source": "B5 synthetic"},
        workers=1,
    ).model


def default_registry(world: World, *, risk_seed: int = 0) -> Registry:
    """The clean decision path in order: ViewLayer, FeaturesLayer(world.config, world.scaler),
    FilterLayer(world.model, activation=ACTIVATION), RiskLayer(the committed risk.yaml,
    seed=risk_seed). Later steps add their layers here (CLAUDE.md rule 4)."""
    return Registry(
        (
            ViewLayer(),
            FeaturesLayer(world.config, world.scaler),
            FilterLayer(world.model, activation=ACTIVATION),
            RiskLayer(risk_config(), seed=risk_seed),
        )
    )


# --- cut times -------------------------------------------------------------------------------


def edge_cuts(decisions: Sequence[Decision]) -> dict[str, Decision]:
    """The decisions every draw includes, keyed by EDGES:

    - "first_warm": the first decision (the first warm bar's);
    - "model_first": the first decision after ACTIVATION, the filter's first record;
    - "after_dst": the first decision after a change of New York's UTC offset (2016-11-07);
    - "half_day_close": a half day's CLOSE bar (2016-11-25), a reduce-only decision;
    - "session_first": the first bar of the session after that half day.

    KeyError when `decisions` lacks one."""

    def find(day: date, *, closing: bool = False) -> Decision:
        for d in decisions:
            if d.bar.session == day and (d.reduce_only if closing else d.bar.index == 0):
                return d
        raise KeyError(day)

    if not decisions:
        raise KeyError("first_warm")
    model_first = next((d for d in decisions if d.tau > ACTIVATION), None)
    if model_first is None:
        raise KeyError("model_first")
    return {
        "first_warm": decisions[0],
        "model_first": model_first,
        "after_dst": find(date(2016, 11, 7)),
        "half_day_close": find(date(2016, 11, 25), closing=True),
        "session_first": find(date(2016, 11, 28)),
    }


def draw_cuts(world: World, n: int, *, seed: int) -> tuple[datetime, ...]:
    """`n` distinct cut times, sorted: the τ of every edge cut, plus n - len(EDGES) other
    decisions' τ drawn by `random.Random(seed)`. The same seed gives the same cuts. ValueError,
    its message naming the number of cuts asked for, when n < len(EDGES) or n exceeds the
    decisions."""
    if not len(EDGES) <= n <= len(world.decisions):
        raise ValueError(f"invalid cut count {n}")
    edges = {d.tau for d in world.edges.values()}
    others = [d.tau for d in world.decisions if d.tau not in edges]
    return tuple(sorted(edges | set(random.Random(seed).sample(others, n - len(edges)))))


# --- the variants' stores --------------------------------------------------------------------


def truncated_store(world: World, cut: datetime) -> BarStore:
    """Variant (a): a backup of `world.conn` into a new in-memory database, with every row known
    after `cut` deleted (minutes, hourly bars, flags), as a clockless BarStore."""
    conn = sqlite3.connect(":memory:", isolation_level=None)
    world.conn.backup(conn)
    for table in ("minute_bars", "hourly_bars", "data_flags"):
        conn.execute(
            f"DELETE FROM {table} WHERE known_at > ?",  # noqa: S608 — fixed table names above
            (to_us(cut),),
        )
    return BarStore(conn)


@dataclass(frozen=True)
class Garbage:
    """What `garbage_store` wrote, so a test can check the garbage is really there.

    - `revised`: t_open of the bar revised with known_at after the cut;
    - `revision_known_at`: when that revision became known;
    - `nan`, `huge`, `negative`: t_open of the bars written with a NaN close, all prices +1e9,
      and close and low -1e9;
    - `shuffled`: t_open of the bars after the cut written with shuffled values;
    - `unavailable`: t_open of the bars that close before the cut but are available at or after
      it (one with available_at exactly the cut)."""

    revised: datetime
    revision_known_at: datetime
    nan: datetime
    huge: datetime
    negative: datetime
    shuffled: tuple[datetime, ...]
    unavailable: tuple[datetime, ...]


def garbage_store(world: World, cut: datetime, *, seed: int) -> tuple[BarStore, Garbage]:
    """Variant (c): `truncated_store(world, cut)` on a SimClock, plus garbage after the cut,
    drawn by an RNG seeded from (`seed`, `cut`), so a rerun writes the same garbage:

    1. a late revision: `tests.support.bars.revise_close` of the cut's own bar by +0.25, known at
       cut + 5 minutes (through `ingest_minutes`, as a settle arrives);
    2. raw hourly rows, inserted with SQL, never through `ingest_minutes` (which refuses them)
       or anything the harness tests, each known just before the cut (cut - 1 µs) so that only
       the view's availability rule can hide it:
       - the next 14 calendar bars after the cut's bar, their values shuffled among them, with
         one close NaN, one bar's prices all +1e9, one bar's close and low -1e9. SQLite turns
         a NaN REAL into NULL, which the schema refuses, so the NaN is stored as the text 'NaN';
       - a bar with t_close < cut and available_at == cut, and one with t_close < cut <
         available_at, both at t_opens off the calendar's grid and after the cut's bar's.

    Returns the store and what was written."""
    store = truncated_store(world, cut)
    clock = SimClock(cut)
    store = BarStore(store._conn, clock=clock)
    decision = next(d for d in world.decisions if d.tau == cut)
    bar = next(
        b
        for b in store.view(cut, as_of=cut).hourly(SYMBOL, purpose=Purpose.OPERATE, last=1)
        if b.t_open == decision.bar.t_open
    )
    revision_at = cut + timedelta(minutes=5)
    revise_close(store, clock, world.minutes, bar, 0.25, at=revision_at)
    cal = TradingCalendar.xnys()
    future = [
        b
        for day in cal.sessions(decision.bar.session, decision.bar.session + timedelta(days=14))
        for b in cal.session(day).bars
        if b.t_open > bar.t_open
    ][:14]
    values = list(range(len(future)))
    random.Random(f"{seed}:{cut.isoformat()}").shuffle(values)

    def insert(
        spec: Any,
        open_time: datetime,
        close_time: datetime,
        available: datetime,
        prices: tuple[Any, Any, Any, Any],
    ) -> None:
        store._conn.execute(
            "INSERT INTO hourly_bars VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
            (
                SYMBOL,
                to_us(open_time),
                spec.session.isoformat(),
                spec.slot.value,
                to_us(close_time),
                *prices,
                100.0,
                60,
                to_us(available),
                to_us(cut - timedelta(microseconds=1)),
            ),
        )

    for i, spec in enumerate(future):
        price = bar.close + values[i] + 1
        prices: tuple[Any, Any, Any, Any] = (price, price + 1, price - 1, price)
        if i == 0:
            prices = (price, price + 1, price - 1, "NaN")
        elif i == 1:
            prices = (1e9, 1e9, 1e9, 1e9)
        elif i == 2:
            prices = (price, price + 1, -1e9, -1e9)
        insert(spec, spec.t_open, spec.t_close, spec.t_close + timedelta(minutes=15), prices)
    unavailable = (cut - timedelta(seconds=4), cut - timedelta(seconds=3))
    for i, opened in enumerate(unavailable):
        insert(
            decision.bar,
            opened,
            cut - timedelta(seconds=1),
            cut + timedelta(seconds=i),
            (bar.close, bar.close + 1, bar.close - 1, bar.close),
        )
    return store, Garbage(
        bar.t_open,
        revision_at,
        future[0].t_open,
        future[1].t_open,
        future[2].t_open,
        tuple(b.t_open for b in future),
        unavailable,
    )


# --- runs and the check ----------------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    """One variant's records. `records[name]` holds the layer's records aligned with the run's
    decisions. A layer that raised has no entry, nor has any layer after it; `errors[name]` is
    the exception's class name and message."""

    variant: Variant
    decisions: tuple[Decision, ...]
    records: Mapping[str, tuple[Record, ...]]
    errors: Mapping[str, str]


def run_layers(registry: Registry, source: Source, decisions: Sequence[Decision]) -> Run:
    """Run every layer in order over `decisions`, each given the records of the layers before
    it. ValueError when a layer returns records that aren't one per decision, in order, with
    its name; an exception a layer raises is caught into `Run.errors`."""
    records: dict[str, tuple[Record, ...]] = {}
    errors: dict[str, str] = {}
    for layer in registry.layers:
        try:
            result = layer.run(source, decisions, records)
        except Exception as exc:
            errors[layer.name] = f"{type(exc).__name__}: {exc}"
            break
        if len(result) != len(decisions) or any(
            r.tau != d.tau or r.layer != layer.name for r, d in zip(result, decisions, strict=True)
        ):
            raise ValueError(f"unaligned records from {layer.name}")
        records[layer.name] = tuple(result)
    return Run(source.variant, tuple(decisions), records, errors)


@dataclass(frozen=True)
class Failure:
    """One way a cut failed.

    - `kind`: "mismatch" (a record's digest differs from the full run's), "error" (one run
      has no record because the layer raised there: in this variant and not the full run, the
      reverse, or both but differently), "missing" (one run lacks a record any other way) or
      "provenance" (fit_end ≥ τ);
    - `variant`: the variant compared with the full run; for "provenance", the variant whose
      record broke it;
    - `tau`: the decision where it showed.
    - `detail`: for people; tests compare the other fields."""

    cut: datetime
    variant: Variant
    layer: str
    tau: datetime | None
    kind: str
    detail: str


@dataclass(frozen=True)
class Report:
    """The outcome of `check_truncation`.

    - `cuts`: the cut times checked, in order (fewer than asked when it stopped at a failure);
    - `compared`: per layer, how many records were compared, across cuts and variants;
    - `failures`: every failure found."""

    cuts: tuple[datetime, ...]
    compared: Mapping[str, int]
    failures: tuple[Failure, ...]

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def layers(self) -> frozenset[str]:
        """The layers the failures name."""
        return frozenset(f.layer for f in self.failures)

    def explain(self) -> str:
        """The failures, one per line: cut, variant, layer, decision, kind and detail."""
        return "\n".join(
            f"{f.cut} {f.variant} {f.layer} {f.tau} {f.kind}: {f.detail}" for f in self.failures
        )


def provenance_failures(
    records: Iterable[Record], *, cut: datetime, variant: Variant
) -> list[Failure]:
    """T-NL2: a "provenance" Failure for each layer's first record with provenance whose
    fit_end is not strictly before its τ. Records without provenance pass."""
    failures = []
    seen = set()
    for record in records:
        if (
            record.tau <= cut
            and record.layer not in seen
            and record.provenance is not None
            and record.provenance.fit_end >= record.tau
        ):
            seen.add(record.layer)
            failures.append(
                Failure(
                    cut,
                    variant,
                    record.layer,
                    record.tau,
                    "provenance",
                    f"model {record.provenance.model} fit_end={record.provenance.fit_end}",
                )
            )
    return failures


def compare_runs(reference: Run, other: Run, *, cut: datetime) -> list[Failure]:
    """`other` against `reference` (the full run), over the decisions up to `cut`: walk the
    decisions in order and, at each, the layers in registry order. The first record whose
    digest differs, or that one run lacks, is the comparison's only failure, so it names the
    layer where the runs first part. A layer that raised has no records, nor have the layers
    after it, so its failure shows at the run's first decision, kind "error"; two runs whose
    layer raised the same error (`Run.errors` equal) agree there."""
    names = tuple(
        dict.fromkeys((*reference.records, *reference.errors, *other.records, *other.errors))
    )
    for i, d in enumerate(reference.decisions):
        if d.tau > cut:
            break
        for name in names:
            left_error, right_error = reference.errors.get(name), other.errors.get(name)
            if left_error is not None or right_error is not None:
                if left_error == right_error:
                    continue
                return [
                    Failure(
                        cut, other.variant, name, d.tau, "error", f"{left_error} / {right_error}"
                    )
                ]
            left, right = reference.records.get(name, ()), other.records.get(name, ())
            if i >= len(left) or i >= len(right):
                return [Failure(cut, other.variant, name, d.tau, "missing", "record missing")]
            try:
                equal = left[i].digest() == right[i].digest()
            except Exception as exc:
                return [Failure(cut, other.variant, name, d.tau, "error", str(exc))]
            if not equal:
                return [
                    Failure(cut, other.variant, name, d.tau, "mismatch", "canonical hashes differ")
                ]
    return []


def check_truncation(
    world: World,
    registry: Registry,
    cuts: Sequence[datetime],
    *,
    seed: int,
    slice_full: bool = True,
    stop_at_first: bool = False,
) -> Report:
    """T-NL1 and T-NL2 at each cut, in the order given.

    The full run (b) is `run_layers` on `world.store` over every decision, once, and each cut
    compares its records up to τ (`slice_full`); with `slice_full=False` the full store is run
    again per cut over the decisions up to τ. At each cut, (a) runs on `truncated_store` and (c)
    on `garbage_store(seed=seed)` over the decisions up to τ; `compare_runs` checks (a) and (c)
    against (b), and `provenance_failures` checks the records of all three. With
    `stop_at_first` it returns at the first cut that fails."""
    full = (
        run_layers(registry, Source(world.store, Variant.FULL, None), world.decisions)
        if slice_full
        else None
    )
    failures: list[Failure] = []
    checked = []
    compared = dict.fromkeys(registry.names, 0)
    for cut in cuts:
        decisions = tuple(d for d in world.decisions if d.tau <= cut)
        reference = (
            full
            if full is not None
            else run_layers(registry, Source(world.store, Variant.FULL, None), decisions)
        )
        checked.append(cut)
        failures.extend(
            provenance_failures(
                (r for rows in reference.records.values() for r in rows),
                cut=cut,
                variant=Variant.FULL,
            )
        )
        for variant in (Variant.TRUNCATED, Variant.GARBAGE):
            store = (
                truncated_store(world, cut)
                if variant == Variant.TRUNCATED
                else garbage_store(world, cut, seed=seed)[0]
            )
            try:
                run = run_layers(registry, Source(store, variant, cut), decisions)
                differences = compare_runs(reference, run, cut=cut)
                failures.extend(differences)
                failures.extend(
                    provenance_failures(
                        (r for rows in run.records.values() for r in rows), cut=cut, variant=variant
                    )
                )
                for index, d in enumerate(decisions):
                    for name in registry.names:
                        if differences and (d.tau, name) == (
                            differences[0].tau,
                            differences[0].layer,
                        ):
                            if differences[0].kind == "mismatch":
                                compared[name] += 1
                            break
                        if index < min(
                            len(reference.records.get(name, ())), len(run.records.get(name, ()))
                        ):
                            compared[name] += 1
                    if differences and d.tau == differences[0].tau:
                        break
            finally:
                store.close()
        if failures and stop_at_first:
            break
    return Report(tuple(checked), compared, tuple(failures))


@cache
def _history() -> tuple[MinuteFrame, BarStore]:
    minutes = synthetic_days(DATA_FIRST, DATA_LAST, seed=DATA_SEED)
    store = BarStore.memory()
    store.ingest_minutes(SYMBOL, minutes, known_at=HISTORY, source="synthetic")
    return minutes, store


def _feature_state_hash(stream: FeatureStream) -> str:
    # Serialize the actual state, including every field, before caching. Pydantic's
    # serializer preserves float64 values and signed zero. Read the state immediately
    # without deepcopying its histories; never retain a mutable reference.
    return _hash_feature_json(stream._state.model_dump_json())


@lru_cache(maxsize=2048)
def _hash_feature_json(payload: str) -> str:
    # Convert the fast serializer's spelling to the project's canonical JSON.
    return sha256_hex(canonical_loads(payload))


# Intent construction validates a deterministic decision id. Reuse only exact inputs;
# the engine and all of its checks still run afresh for every variant.
_risk_intent = lru_cache(maxsize=4096)(intent)


@lru_cache(maxsize=4096)
def _risk_quantity(seed: int, tau: datetime) -> int:
    return random.Random(f"{seed}:{tau.isoformat()}").randint(1, 200)
