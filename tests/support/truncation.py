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

Records are compared by their digests (`Record.digest`), never approximately. A Record is
immutable: it holds its output as canonical JSON bytes (`Record.payload`), digested when it is
made, and hands layers downstream a read-only copy parsed back from those bytes. A failure
names the cut, the variant, the decision and the **first layer** whose record differs there,
so a leak is reported where it enters, not in every layer downstream. A layer that raises (not
a per-decision refusal, which is a record) fails the check in any run, since it and every layer
after it then compare nothing; and a check in which some layer compared no record beyond a
cold start fails as "vacuous", so a pass always means something.

**Slicing.** The full run (b) is the engine over the whole history, so it runs once, over every
decision, and each cut compares its records up to τ. Runs (a) and (c) depend on τ by
definition, so each cut runs them over the decisions up to τ on its own store; a store is a
SQLite backup of the full one with later rows deleted, which costs about a millisecond.
`check_truncation(..., slice_full=False)` recomputes (b) per cut, still over the whole history,
and the slicing test checks that both modes find the same failures.

**Caching.** Layers reuse a Record when its inputs are exactly those of one made before,
across runs, cuts and registries. Keys are exact, never Python `==` on values (where
0.0 == -0.0 and 1 == 1.0 == True): `cached_record` keys on the output's canonical bytes and
the record's other fields as exact types, and the view layer on the `repr` of its values,
built only from exact types (None, bool, int, float, str), whose repr determines type and
value. Since Records are immutable, a reused one is indistinguishable from one made afresh.

**T-NL2.** A record made with a fitted artifact names it (`Provenance`: the artifact's hash and
the end of its fit window): the filter's model, and the scaler of the features layer's scaled
vectors. `provenance_failures` demands fit_end < τ for every such record, in every variant.

Everything here is synthetic, on the real NYSE calendar inside FIT (2016). DEV stays dark.
"""

from __future__ import annotations

import hashlib
import math
import random
import sqlite3
from collections import OrderedDict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from functools import cache, lru_cache
from itertools import chain, pairwise
from types import MappingProxyType
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
    """The fitted artifact a record was made with (T-NL2): its hash (`HmmModel.artifact_hash()`
    for a model, `scaler_hash` for a scaler) and the end of its fit window (`fit_end`, a
    scaler's `end`)."""

    artifact: str
    fit_end: datetime

    def __post_init__(self) -> None:
        if type(self.artifact) is not str or not isinstance(self.fit_end, datetime):
            raise TypeError("a Provenance names its artifact hash (str) and fit_end (datetime)")


_UNPARSED: Final = object()


def _check_fields(layer: object, provenance: object, state: object, cold: object) -> None:
    """TypeError unless `layer` is a str, `provenance` a Provenance or None, `state` a str or
    None and `cold` a bool: exact types, so a cache key built from them is exact too."""
    if (
        type(layer) is not str
        or type(cold) is not bool
        or (state is not None and type(state) is not str)
        or (provenance is not None and type(provenance) is not Provenance)
    ):
        raise TypeError("a Record's layer and state are str, its cold a bool")


def _freeze(value: Any) -> Any:
    """Parsed canonical JSON made read-only: objects as MappingProxyType, arrays as tuples."""
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


class Record:
    """One layer's output for one decision. Immutable: setting or deleting an attribute raises
    AttributeError, and `output` is read-only.

    - `tau`: the decision time;
    - `layer`: the layer's registered name;
    - `payload`: the output's canonical JSON bytes (`quant.core.canonical.canonical_dumps`),
      taken when the Record is made. An output without a canonical form raises
      CanonicalJSONError there, inside the layer, which the harness reports as its failure. A
      layer that refuses a decision (FeatureError, ModelError: rule 6) records
      `error_output(exc)` rather than raising;
    - `output`: the payload parsed back, read-only (objects as mappings, arrays as tuples), so
      what a later layer reads is exactly what is compared;
    - `provenance`: the fitted artifact behind the record, None for a layer that uses none;
    - `state`: `sha256_hex` of the layer's state after this decision (the FeatureStream's or
      ForwardFilter's snapshot, the RiskState), None for a stateless layer;
    - `cold`: a placeholder for a decision before the layer starts (the filter's cold start).
      It is compared like any record but doesn't count as one (`Report.compared`).
    """

    __slots__ = ("_digest", "_output", "cold", "layer", "payload", "provenance", "state", "tau")
    tau: datetime
    layer: str
    payload: bytes
    provenance: Provenance | None
    state: str | None
    cold: bool
    _digest: str
    _output: Any

    def __init__(
        self,
        tau: datetime,
        layer: str,
        output: Any,
        provenance: Provenance | None = None,
        state: str | None = None,
        *,
        cold: bool = False,
    ) -> None:
        _check_fields(layer, provenance, state, cold)
        payload = canonical_dumps(output)
        header = canonical_dumps(
            {
                "tau": tau,
                "layer": layer,
                "provenance": None if provenance is None else asdict(provenance),
                "state": state,
                "cold": cold,
            }
        )
        for name, value in (
            ("tau", tau),
            ("layer", layer),
            ("payload", payload),
            ("provenance", provenance),
            ("state", state),
            ("cold", cold),
            ("_digest", hashlib.sha256(header + b"\n" + payload).hexdigest()),
            ("_output", _UNPARSED),
        ):
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"a Record is immutable: can't set {name}")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"a Record is immutable: can't delete {name}")

    def __repr__(self) -> str:
        return f"Record({self.tau.isoformat()}, {self.layer!r}, {self._digest[:12]})"

    @property
    def output(self) -> Any:
        if self._output is _UNPARSED:
            object.__setattr__(self, "_output", _freeze(canonical_loads(self.payload)))
        return self._output

    def digest(self) -> str:
        """SHA-256, in hex, of the canonical JSON of {"tau", "layer", "provenance": null or
        {"artifact", "fit_end"}, "state", "cold"}, a newline, and `payload`. Canonical JSON
        holds no raw newline, so the split is unambiguous."""
        return self._digest


def error_output(exc: Exception) -> dict[str, str]:
    """The output a layer records for a decision it refuses: {"error": the exception's class
    name, "message": str(exc)}."""
    return {"error": type(exc).__name__, "message": str(exc)}


class RecordCache:
    """Records reused across runs, cuts and registries under exact keys, least recently used
    first out past `size`. Reuse is safe because a Record is immutable and its key determines
    its canonical bytes: a hit is indistinguishable from making the Record afresh."""

    def __init__(self, size: int) -> None:
        self._size = size
        self._records: OrderedDict[Hashable, Record] = OrderedDict()

    def get(self, key: Hashable) -> Record | None:
        record = self._records.get(key)
        if record is not None:
            self._records.move_to_end(key)
        return record

    def put(self, key: Hashable, record: Record) -> Record:
        self._records[key] = record
        if len(self._records) > self._size:
            self._records.popitem(last=False)
        return record


# The full run's 620 decisions per layer, many times over.
_VIEW_RECORDS: Final = RecordCache(4096)
_RECORDS: Final = RecordCache(16384)


def cached_record(
    tau: datetime,
    layer: str,
    output: Any,
    provenance: Provenance | None = None,
    state: str | None = None,
    *,
    cold: bool = False,
) -> Record:
    """`Record(tau, layer, output, provenance, state, cold=cold)`, or the one made before
    from exactly the same inputs (`_RECORDS`): the key is the output's canonical bytes, the
    layer, τ and the provenance's fit_end as epoch microseconds, the artifact hash, the state
    hash and `cold`, each checked to be of an exact type first (`_check_fields`), so equal keys
    mean equal canonical bytes."""
    _check_fields(layer, provenance, state, cold)
    key = (
        layer,
        canonical_dumps(output),
        to_us(tau),
        None if provenance is None else (provenance.artifact, to_us(provenance.fit_end)),
        state,
        cold,
    )
    record = _RECORDS.get(key)
    if record is None:
        record = _RECORDS.put(key, Record(tau, layer, output, provenance, state, cold=cold))
    return record


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
    "digest": that view's `digest()`}, times as epoch microseconds (`to_us`, as the store keeps
    them). Versions and known_at are left out, as the digest leaves them out. Stateless.

    Records are reused (`_VIEW_RECORDS`) under an exact key (`_exact_repr` of the layer name,
    the decision, the digest and every recorded value), never under the view's digest alone:
    the digest is part of what is under test."""

    name: Final = "view"

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
            values = tuple(
                (
                    to_us(bar.t_open),
                    to_us(bar.t_close),
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    bar.volume,
                    bar.minutes,
                    bar.complete,
                    to_us(bar.available_at),
                )
                for bar in bars
            )
            digest = view.digest()
            key = _exact_repr(self.name, to_us(d.tau), digest, values)
            record = None if key is None else _VIEW_RECORDS.get(key)
            if record is None:
                record = Record(
                    d.tau,
                    self.name,
                    {
                        "bars": [dict(zip(fields, row, strict=True)) for row in values],
                        "digest": digest,
                    },
                )
                if key is not None:
                    _VIEW_RECORDS.put(key, record)
            records.append(record)
        return records


def features_output(raw: Any, scaled: Any = None) -> dict[str, list[float]]:
    """The features layer's output: {"raw": the feature vector, "scaled": it standardized},
    each a list of float64 in the config's column order; just {"raw"} when `scaled` is None."""
    output = {"raw": [float(x) for x in raw]}
    if scaled is not None:
        output["scaled"] = [float(x) for x in scaled]
    return output


def scaler_hash(scaler: Scaler) -> str:
    """`sha256_hex` of a scaler's every field: its Provenance artifact (T-NL2)."""
    return sha256_hex(
        {
            "columns": list(scaler.columns),
            "mean": scaler.mean,
            "std": scaler.std,
            "start": scaler.start,
            "end": scaler.end,
            "rows": scaler.rows,
        }
    )


class FeaturesLayer:
    """B3's features and scaler. One FeatureStream per run, fed through the view:
    `features_for(stream, source.view(d.tau), SYMBOL, d.bar)` for each decision. Before
    `activation` it records `features_output(raw)`, unscaled and without provenance; from the
    first decision at or after `activation`, `features_output(raw, scaled)` with `scaled` from
    `scaler.transform`, and every record carries the scaler's Provenance (`scaler_hash`, its
    window's `end`). Records `error_output` when features_for or the scaler refuses (a warm-up
    row is a FeatureError). `state` is the hash of the stream's snapshot (`_feature_state_hash`).

    The scaler is fixed and whatever it is given; the clean one is the fixture model's own,
    fit on the model's window, and activates with the model (`World.scaler`, ACTIVATION). The
    layer never checks the scaler's window against τ: that is T-NL2's job, so a scaler fit
    after the decisions it scales is caught by `provenance_failures`."""

    name: Final = "features"

    def __init__(
        self, config: FeatureConfig, scaler: Scaler, *, activation: datetime = ACTIVATION
    ) -> None:
        self.config, self.scaler, self.activation = config, scaler, activation
        self._provenance = Provenance(scaler_hash(scaler), scaler.end)

    def run(
        self,
        source: Source,
        decisions: Sequence[Decision],
        upstream: Mapping[str, Sequence[Record]],
    ) -> list[Record]:
        stream = FeatureStream(self.config)
        records = []
        for d in decisions:
            scaling = d.tau >= self.activation
            output: dict[str, Any]
            try:
                raw = features_for(stream, source.view(d.tau), SYMBOL, d.bar)
                scaled = (
                    self.scaler.transform(raw, columns=self.config.columns) if scaling else None
                )
                output = features_output(raw, scaled)
            except (FeatureError, ModelError) as exc:
                output = error_output(exc)
            state = _feature_state_hash(stream)
            provenance = self._provenance if scaling else None
            records.append(cached_record(d.tau, self.name, output, provenance, state))
        return records


class FilterLayer:
    """B4's filter and labels. A decision before `activation` records {"model": null}, no
    provenance and `cold` (a cold start, which `Report.compared` doesn't count). From the first decision at or after `activation`, one
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
                records.append(cached_record(d.tau, self.name, {"model": None}, cold=True))
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
                cached_record(d.tau, self.name, output, provenance, sha256_hex(filt.snapshot()))
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
                    records.append(cached_record(d.tau, self.name, {"skipped": "no bar"}))
                    continue
                price = view.output["bars"][-1]["close"]
                clock.advance_to(d.executes_at + timedelta(seconds=1))
                day = clock.now().date()
                if engine.state.session is None or day > engine.state.session:
                    engine.start_session(day, equity=RISK_EQUITY)
                label = filtered.output.get("top_label")
                order = intent(
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
                records.append(
                    cached_record(d.tau, self.name, output, state=sha256_hex(engine.state))
                )
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
    - `scaler`: the Scaler fit on the model's window, [the first warm row's available_at,
      MODEL_FIT_END): the model's own (same mean and std), with its window, so the features
      layer can name it in Provenance;
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
    """The clean decision path in order: ViewLayer, FeaturesLayer(world.config, world.scaler,
    activation=ACTIVATION), FilterLayer(world.model, activation=ACTIVATION), RiskLayer(the
    committed risk.yaml, seed=risk_seed). Later steps add their layers here (CLAUDE.md
    rule 4)."""
    return Registry(
        (
            ViewLayer(),
            FeaturesLayer(world.config, world.scaler, activation=ACTIVATION),
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


def known_at_tables(conn: sqlite3.Connection) -> tuple[str, ...]:
    """Every table of the database with a known_at column, by name."""
    tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    return tuple(
        table
        for table in sorted(tables)
        if "known_at" in {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    )


def truncated_store(world: World, cut: datetime) -> BarStore:
    """Variant (a): a backup of `world.conn` into a new in-memory database, with every row known
    after `cut` deleted from every table that has a known_at column (`known_at_tables`: minutes,
    hourly bars, dividends, splits, the calendar cache and flags), as a clockless BarStore."""
    conn = sqlite3.connect(":memory:", isolation_level=None)
    world.conn.backup(conn)
    for table in known_at_tables(conn):
        conn.execute(
            f"DELETE FROM {table} WHERE known_at > ?",  # noqa: S608 — names from sqlite_master
            (to_us(cut),),
        )
    return BarStore(conn)


@dataclass(frozen=True)
class Garbage:
    """What `garbage_store` wrote, so a test can check the garbage is really there.

    - `revised`: t_open of the bar revised with known_at after the cut;
    - `revision_known_at`: when that revision became known;
    - `revised_at_cut`: t_open of the bar revised with known_at exactly the cut, which only a
      view reading known_at <= as_of sees;
    - `nan`, `huge`, `negative`: t_open of the bars written with a NaN close, all prices +1e9,
      and close and low -1e9;
    - `shuffled`: t_open of the bars after the cut written with shuffled values;
    - `unavailable`: t_open of the bars that close before the cut but are available at or after
      it (one with available_at exactly the cut);
    - `minutes_unavailable`: t of the minutes before the cut but available at or after it (one
      with available_at exactly the cut);
    - `minutes_after`: t of the minutes at or after the cut, with a NaN close, all prices +1e9,
      close and low -1e9, then shuffled real values."""

    revised: datetime
    revision_known_at: datetime
    revised_at_cut: datetime
    nan: datetime
    huge: datetime
    negative: datetime
    shuffled: tuple[datetime, ...]
    unavailable: tuple[datetime, ...]
    minutes_unavailable: tuple[datetime, ...]
    minutes_after: tuple[datetime, ...]


def garbage_store(world: World, cut: datetime, *, seed: int) -> tuple[BarStore, Garbage]:
    """Variant (c): `truncated_store(world, cut)` on a SimClock, plus garbage after the cut,
    drawn by an RNG seeded from (`seed`, `cut`), so a rerun writes the same garbage:

    1. revisions through `ingest_minutes`, as a settle arrives (`tests.support.bars.revise_close`):
       the bar before the cut's bar by -0.25, known exactly at the cut, and the cut's own bar by
       +0.25, known at cut + 5 minutes;
    2. raw hourly rows, inserted with SQL, never through `ingest_minutes` (which refuses them)
       or anything the harness tests, each known just before the cut (cut - 1 µs) so that only
       the view's availability rule can hide it:
       - the next 14 calendar bars after the cut's bar, their values shuffled among them, with
         one close NaN, one bar's prices all +1e9, one bar's close and low -1e9. SQLite turns
         a NaN REAL into NULL, which the schema refuses, so the NaN is stored as the text 'NaN';
       - a bar with t_close < cut and available_at == cut, and one with t_close < cut <
         available_at, both at t_opens off the calendar's grid and after the cut's bar's;
    3. raw minute rows, known at cut - 1 µs likewise:
       - two minutes off the grid before the cut (cut - 30 s and cut - 20 s), available at the
         cut and a second after it;
       - five minutes from the cut on: a NaN close (as text), all prices +1e9, close and low
         -1e9, then two with the cut's bar's last values shuffled.

    Dividends, splits and flags get no garbage: no layer reads them yet (B6 open item).

    Returns the store and what was written."""
    store = truncated_store(world, cut)
    clock = SimClock(cut)
    store = BarStore(store._conn, clock=clock)
    decision = next(d for d in world.decisions if d.tau == cut)
    previous, bar = store.view(cut, as_of=cut).hourly(SYMBOL, purpose=Purpose.OPERATE, last=2)
    if bar.t_open != decision.bar.t_open:
        raise ValueError(f"the cut's bar isn't visible at {cut.isoformat()}")
    revise_close(store, clock, world.minutes, previous, -0.25, at=cut)
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

    def insert_minute(t: datetime, available: datetime, prices: tuple[Any, ...]) -> None:
        store._conn.execute(
            "INSERT INTO minute_bars VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, 'garbage')",
            (
                SYMBOL,
                to_us(t),
                *prices,
                100.0,
                to_us(available),
                to_us(cut - timedelta(microseconds=1)),
            ),
        )

    quiet = (bar.close, bar.close + 1, bar.close - 1, bar.close)
    minutes_unavailable = (cut - timedelta(seconds=30), cut - timedelta(seconds=20))
    for i, t in enumerate(minutes_unavailable):
        insert_minute(t, cut + timedelta(seconds=i), quiet)
    last = world.minutes.select(world.minutes.t < to_us(bar.t_close))
    real = [float(last.open[-1]), float(last.high[-1]), float(last.low[-1]), float(last.close[-1])]
    minutes_after = tuple(cut + timedelta(minutes=i) for i in range(5))
    for i, t in enumerate(minutes_after):
        minute: tuple[Any, ...]
        if i == 0:
            minute = (bar.close, bar.close + 1, bar.close - 1, "NaN")
        elif i == 1:
            minute = (1e9, 1e9, 1e9, 1e9)
        elif i == 2:
            minute = (bar.close, bar.close + 1, -1e9, -1e9)
        else:
            random.Random(f"{seed}:{cut.isoformat()}:{i}").shuffle(real)
            minute = tuple(real)
        insert_minute(t, t + timedelta(minutes=1), minute)
    return store, Garbage(
        bar.t_open,
        revision_at,
        previous.t_open,
        future[0].t_open,
        future[1].t_open,
        future[2].t_open,
        tuple(b.t_open for b in future),
        unavailable,
        minutes_unavailable,
        minutes_after,
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

    - `kind`: "mismatch" (a record's digest differs from the full run's), "error" (the layer
      raised in this variant's run, the full run, or both, even with the same error: it and
      every layer after it then have no records to compare), "missing" (one run lacks a record
      any other way), "provenance" (fit_end ≥ τ) or "vacuous" (a check that found nothing else
      compared no record of this layer beyond a cold start, at any cut);
    - `variant`: the variant compared with the full run; for "provenance", the variant whose
      record broke it; FULL for "vacuous";
    - `tau`: the decision where it showed (None for "vacuous", reported at the last cut).
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
    - `compared`: per layer in registry order, how many records were compared, across cuts and
      variants, cold-start records (`Record.cold`) not counted; a record whose digest differs
      counts, a missing or unmade one doesn't;
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
                    f"artifact {record.provenance.artifact} fit_end={record.provenance.fit_end}",
                )
            )
    return failures


def compare_runs(
    reference: Run, other: Run, *, cut: datetime
) -> tuple[list[Failure], dict[str, int]]:
    """`other` against `reference` (the full run), over the decisions up to `cut`: walk the
    decisions in order and, at each, the layers in order. The first record whose digest
    differs, or that one run lacks, is the comparison's only failure, so it names the layer
    where the runs first part. A layer that raised in either run has no records, nor have the
    layers after it, so it fails at the first decision, kind "error", even when both runs raised
    the same error: a layer that compares nothing must not pass. A layer's per-decision
    refusals are records (`error_output`), compared like any other on (decision, layer,
    message) through the digest.

    Returns the failures (at most one) and, per layer, how many records were compared before
    the walk stopped, the differing one included and cold-start records not counted."""
    names = tuple(
        dict.fromkeys((*reference.records, *reference.errors, *other.records, *other.errors))
    )
    compared = dict.fromkeys(names, 0)
    for i, d in enumerate(reference.decisions):
        if d.tau > cut:
            break
        for name in names:
            left_error, right_error = reference.errors.get(name), other.errors.get(name)
            if left_error is not None or right_error is not None:
                detail = f"full: {left_error} / {other.variant}: {right_error}"
                return [Failure(cut, other.variant, name, d.tau, "error", detail)], compared
            left, right = reference.records.get(name, ()), other.records.get(name, ())
            if i >= len(left) or i >= len(right):
                failure = Failure(cut, other.variant, name, d.tau, "missing", "record missing")
                return [failure], compared
            if not left[i].cold:
                compared[name] += 1
            if left[i].digest() != right[i].digest():
                failure = Failure(cut, other.variant, name, d.tau, "mismatch", "digests differ")
                return [failure], compared
    return [], compared


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

    The full run (b) is `run_layers` on `world.store` over every decision of the whole history,
    once, and each cut compares its records up to τ (`slice_full`); with `slice_full=False` it
    is run afresh at each cut, still over the whole history, never only up to τ (a smoother
    would see no later rows there). At each cut, (a) runs on `truncated_store` and (c) on
    `garbage_store(seed=seed)` over the decisions up to τ; `compare_runs` checks (a) and (c)
    against (b), and `provenance_failures` checks the records of all three. When nothing else
    failed, a layer that compared no record (cold starts aside) at any cut fails "vacuous". With
    `stop_at_first` it returns at the first cut that fails. ValueError for no cuts."""
    if not cuts:
        raise ValueError("no cut times to check")

    def full_run() -> Run:
        return run_layers(registry, Source(world.store, Variant.FULL, None), world.decisions)

    full = full_run() if slice_full else None
    failures: list[Failure] = []
    checked = []
    compared = dict.fromkeys(registry.names, 0)
    for cut in cuts:
        decisions = tuple(d for d in world.decisions if d.tau <= cut)
        reference = full if full is not None else full_run()
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
                differences, counts = compare_runs(reference, run, cut=cut)
                failures.extend(differences)
                failures.extend(
                    provenance_failures(
                        (r for rows in run.records.values() for r in rows), cut=cut, variant=variant
                    )
                )
                for name, count in counts.items():
                    compared[name] = compared.get(name, 0) + count
            finally:
                store.close()
        if failures and stop_at_first:
            break
    if not failures:
        failures.extend(
            Failure(
                checked[-1],
                Variant.FULL,
                name,
                None,
                "vacuous",
                "no record compared beyond a cold start",
            )
            for name in registry.names
            if compared[name] == 0
        )
    return Report(tuple(checked), compared, tuple(failures))


@cache
def _history() -> tuple[MinuteFrame, BarStore]:
    minutes = synthetic_days(DATA_FIRST, DATA_LAST, seed=DATA_SEED)
    store = BarStore.memory()
    store.ingest_minutes(SYMBOL, minutes, known_at=HISTORY, source="synthetic")
    return minutes, store


# FeatureStream.snapshot as B3 wrote it: a deep copy of `_state` and nothing more.
_STREAM_SNAPSHOT: Final = FeatureStream.snapshot


def _feature_state_hash(stream: FeatureStream) -> str:
    """`sha256_hex(stream.snapshot())`, without snapshot()'s deep copy when that is safe.

    The shortcut serializes `stream._state` directly (pydantic keeps float64 values and signed
    zero), and is taken only when the stream's class uses B3's own `snapshot`, which copies
    `_state` and nothing else. Any other stream (a subclass, a patched method) is hashed through
    its public `snapshot()`. The shortcut's equality with the snapshot hash is guarded at every
    bar of the world by `test_review_12_the_fast_state_hash_is_the_snapshot_hash_at_every_bar`."""
    if type(stream).snapshot is not _STREAM_SNAPSHOT:
        return sha256_hex(stream.snapshot())
    return _hash_feature_json(stream._state.model_dump_json())


@lru_cache(maxsize=2048)
def _hash_feature_json(payload: str) -> str:
    # Convert the fast serializer's spelling to the project's canonical JSON.
    return sha256_hex(canonical_loads(payload))


def _risk_quantity(seed: int, tau: datetime) -> int:
    return _quantity(f"{seed}:{tau.isoformat()}")


@lru_cache(maxsize=4096)
def _quantity(seed: str) -> int:
    # Keyed on the exact seed string the draw uses.
    return random.Random(seed).randint(1, 200)


# Types whose repr names the type and the value exactly: a float's repr round-trips (and
# keeps -0.0), and an int's, a bool's and a str's are distinct from each other's.
_EXACT_TYPES: Final = frozenset({type(None), bool, int, float, str})


def _exact_repr(
    name: str, tau_us: int, digest: str, rows: tuple[tuple[Any, ...], ...]
) -> str | None:
    """`repr` of the arguments as a cache key when every value in `rows` is of an exact type,
    else None (don't cache). Two equal keys hold the same types and values, so the same
    canonical JSON: never Python ==, where 0.0 == -0.0 and 1 == 1.0 == True."""
    if not set(map(type, chain.from_iterable(rows))) <= _EXACT_TYPES:
        return None
    return repr((name, tau_us, digest, rows))
