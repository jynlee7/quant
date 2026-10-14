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

import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any, Final, Protocol

from quant.core.schedule import Decision
from quant.data.frames import MinuteFrame
from quant.data.market_db import BarStore
from quant.data.view import PointInTimeView
from quant.features.menu import FeatureConfig
from quant.features.scaler import Scaler
from quant.hmm.model import HmmModel
from quant.risk.limits import RiskConfig

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
        raise NotImplementedError


def error_output(exc: Exception) -> dict[str, str]:
    """The output a layer records for a decision it refuses: {"error": the exception's class
    name, "message": str(exc)}."""
    raise NotImplementedError


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
        raise NotImplementedError


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
        raise NotImplementedError

    def register(self, layer: Layer) -> None:
        raise NotImplementedError

    def replace(self, layer: Layer) -> Registry:
        """A new registry with the layer named `layer.name` swapped for `layer`. KeyError when
        no layer has that name."""
        raise NotImplementedError

    @property
    def layers(self) -> tuple[Layer, ...]:
        raise NotImplementedError

    @property
    def names(self) -> tuple[str, ...]:
        raise NotImplementedError


class ViewLayer:
    """B2's view. Per decision: `source.view(d.tau).hourly(SYMBOL, purpose=OPERATE,
    last=VIEW_DEPTH)`, recorded as {"bars": [{"t_open", "t_close", "open", "high", "low",
    "close", "volume", "minutes", "complete", "available_at"} per bar, oldest first],
    "digest": that view's `digest()`}. Versions and known_at are left out, as the digest leaves
    them out. Stateless."""

    name: Final = "view"

    def run(
        self,
        source: Source,
        decisions: Sequence[Decision],
        upstream: Mapping[str, Sequence[Record]],
    ) -> list[Record]:
        raise NotImplementedError


def features_output(raw: Any, scaled: Any) -> dict[str, list[float]]:
    """The features layer's output: {"raw": the feature vector, "scaled": it standardized},
    each a list of float64 in the config's column order."""
    raise NotImplementedError


class FeaturesLayer:
    """B3's features and scaler. One FeatureStream per run, fed through the view:
    `features_for(stream, source.view(d.tau), SYMBOL, d.bar)` for each decision, then
    `scaler.transform` with the config's columns. Records `features_output(raw, scaled)`, or
    `error_output` when features_for or the scaler refuses (a warm-up row is a FeatureError).
    `state` is the hash of the stream's snapshot. The scaler is fixed: the clean one is fit on
    the fixture model's window (`World.scaler`)."""

    name: Final = "features"

    def __init__(self, config: FeatureConfig, scaler: Scaler) -> None:
        raise NotImplementedError

    def run(
        self,
        source: Source,
        decisions: Sequence[Decision],
        upstream: Mapping[str, Sequence[Record]],
    ) -> list[Record]:
        raise NotImplementedError


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
        raise NotImplementedError

    def run(
        self,
        source: Source,
        decisions: Sequence[Decision],
        upstream: Mapping[str, Sequence[Record]],
    ) -> list[Record]:
        raise NotImplementedError


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
        raise NotImplementedError

    def run(
        self,
        source: Source,
        decisions: Sequence[Decision],
        upstream: Mapping[str, Sequence[Record]],
    ) -> list[Record]:
        raise NotImplementedError


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


def world() -> World:
    """The shared World, built on first use and cached for the session."""
    raise NotImplementedError


def fixture_model(fit_end: datetime) -> HmmModel:
    """A model fit on the full store's warm feature rows with available_at in [the first warm
    row's available_at, `fit_end`): `quant.hmm.train.train_candidate` on a TrainingData of
    those rows (each row's raw log return from the previous consumed bar's close, sessions as
    ordinals), K=FIXTURE_STATES FIXTURE_COVARIANCE, FIXTURE_RESTARTS restarts from
    FIXTURE_BASE_SEED, a one-month validation block, r1's label thresholds, workers=1. Cached
    per `fit_end`."""
    raise NotImplementedError


def default_registry(world: World, *, risk_seed: int = 0) -> Registry:
    """The clean decision path in order: ViewLayer, FeaturesLayer(world.config, world.scaler),
    FilterLayer(world.model, activation=ACTIVATION), RiskLayer(the committed risk.yaml,
    seed=risk_seed). Later steps add their layers here (CLAUDE.md rule 4)."""
    raise NotImplementedError


# --- cut times -------------------------------------------------------------------------------


def edge_cuts(decisions: Sequence[Decision]) -> dict[str, Decision]:
    """The decisions every draw includes, keyed by EDGES:

    - "first_warm": the first decision (the first warm bar's);
    - "model_first": the first decision after ACTIVATION, the filter's first record;
    - "after_dst": the first decision after a change of New York's UTC offset (2016-11-07);
    - "half_day_close": a half day's CLOSE bar (2016-11-25), a reduce-only decision;
    - "session_first": the first bar of the session after that half day.

    KeyError when `decisions` lacks one."""
    raise NotImplementedError


def draw_cuts(world: World, n: int, *, seed: int) -> tuple[datetime, ...]:
    """`n` distinct cut times, sorted: the τ of every edge cut, plus n - len(EDGES) other
    decisions' τ drawn by `random.Random(seed)`. The same seed gives the same cuts. ValueError,
    its message naming the number of cuts asked for, when n < len(EDGES) or n exceeds the
    decisions."""
    raise NotImplementedError


# --- the variants' stores --------------------------------------------------------------------


def truncated_store(world: World, cut: datetime) -> BarStore:
    """Variant (a): a backup of `world.conn` into a new in-memory database, with every row known
    after `cut` deleted (minutes, hourly bars, flags), as a clockless BarStore."""
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
        raise NotImplementedError

    @property
    def layers(self) -> frozenset[str]:
        """The layers the failures name."""
        raise NotImplementedError

    def explain(self) -> str:
        """The failures, one per line: cut, variant, layer, decision, kind and detail."""
        raise NotImplementedError


def provenance_failures(
    records: Iterable[Record], *, cut: datetime, variant: Variant
) -> list[Failure]:
    """T-NL2: a "provenance" Failure for each layer's first record with provenance whose
    fit_end is not strictly before its τ. Records without provenance pass."""
    raise NotImplementedError


def compare_runs(reference: Run, other: Run, *, cut: datetime) -> list[Failure]:
    """`other` against `reference` (the full run), over the decisions up to `cut`: walk the
    decisions in order and, at each, the layers in registry order. The first record whose
    digest differs, or that one run lacks, is the comparison's only failure, so it names the
    layer where the runs first part. A layer that raised has no records, nor have the layers
    after it, so its failure shows at the run's first decision, kind "error"; two runs whose
    layer raised the same error (`Run.errors` equal) agree there."""
    raise NotImplementedError


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
    raise NotImplementedError
