"""The model type and its artifact: `params.npz` plus `meta.json`, never pickles (ARCHITECTURE §7.2).

A trained model is an `HmmModel`: the Gaussian HMM's parameters (`GaussianParams`), the frozen
scaler its features are standardized with (`ScalerParams`), its fit window, its feature-config
hash, its label table and its training record. `hmm.train` (B4b) builds one as follows: fit on
the scaled fit-window rows, symmetrize each covariance, run `hmm.filter.forward_pass` over the
same rows and label with `hmm.labels.label_states`, then construct the model.

**The artifact** is the directory `<root>/<artifact_hash>/` with two files:

- `params.npz`: exactly the arrays PARAM_NAMES names, float64, as `<name>.npy` members. The
  file is a ZIP written member by member in PARAM_NAMES order, uncompressed (ZIP_STORED), each
  with the fixed `ZipInfo` date 1980-01-01 00:00:00 and written by
  `numpy.lib.format.write_array(..., allow_pickle=False)`. That makes the bytes a function of
  the arrays alone, so equal models give byte-identical files. It reads back with
  `numpy.load(path, allow_pickle=False)`.
- `meta.json`: the canonical JSON (quant.core.canonical) of
  `{"artifact_hash": <hash>, "meta": <ModelMeta>}`.

**The artifact hash** is `quant.core.canonical.sha256_hex` of

    {"meta": <ModelMeta>,
     "params": {name: {"dtype": "float64", "shape": [...], "sha256": <hex>} for each name}}

where `<hex>` is the SHA-256 of the array's little-endian, C-order bytes
(`numpy.ascontiguousarray(a, dtype="<f8").tobytes()`). It names the model everywhere: the
artifact directory, the filter's snapshots (`FilterState.model_hash`) and decision records.
Every value that defines the model is inside it, so the same seeds and data give the same hash.

**Covariances** are stored full, shape (K, d, d), for both covariance types. A diag model's
off-diagonal entries are exactly 0.

Times are UTC. Nothing here imports hmmlearn or eval (K3 reaches this module through
hmm.filter).
"""

from __future__ import annotations

import hashlib
import re
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt
from pydantic import AwareDatetime

from quant.core.canonical import canonical_dumps, canonical_loads, sha256_hex
from quant.core.clock import ensure_utc
from quant.core.errors import ClockError, ModelError
from quant.core.model import StrictModel
from quant.hmm.labels import LabelTable

Floats = npt.NDArray[np.float64]
CovarianceType = Literal["diag", "full"]

# The arrays in params.npz, in the order they are written. No other member is allowed.
PARAM_NAMES: Final = ("startprob", "transmat", "means", "covars", "scaler_mean", "scaler_std")
# meta.json's format version. Bump it when the artifact's layout changes.
ARTIFACT_FORMAT: Final = 1
# Probability vectors must sum to 1 within this.
SUM_TOLERANCE: Final = 1e-10


def _check_array(values: Floats, shape: tuple[int, ...], name: str) -> None:
    if (
        not isinstance(values, np.ndarray)
        or values.dtype != np.float64
        or values.shape != shape
        or not np.isfinite(values).all()
    ):
        raise ModelError(f"{name} must be a finite float64 array of shape {shape}")


def _freeze_arrays(owner: object, *names: str) -> None:
    for name in names:
        values = getattr(owner, name).copy()
        values.flags.writeable = False
        object.__setattr__(owner, name, values)


@dataclass(frozen=True, eq=False)
class GaussianParams:
    """A Gaussian HMM with K states over d standardized features.

    - `startprob` (K,): the start probabilities π.
    - `transmat` (K, K): the transition matrix A, rows summing to 1, a_jk = P(k next | j now).
    - `means` (K, d) and `covars` (K, d, d): each state's Gaussian, covariances always full.

    Immutable: equality is by value through `HmmModel.artifact_hash`, not `==`.
    """

    covariance_type: CovarianceType
    startprob: Floats
    transmat: Floats
    means: Floats
    covars: Floats

    def __post_init__(self) -> None:
        """Raise ModelError unless:

        - `covariance_type` is "diag" or "full";
        - every array is float64 with the shapes above, K ≥ 1 and d ≥ 1, and every value is
          finite;
        - `startprob` and every row of `transmat` are ≥ 0 and sum to 1 within SUM_TOLERANCE;
        - every covariance is exactly symmetric (Σ == Σᵀ, elementwise) and positive definite
          (its Cholesky factorization succeeds);
        - for "diag", every off-diagonal entry is exactly 0.

        Replace each array with a read-only copy (`object.__setattr__`), so the caller's array
        can't change the model and writing to the model's raises ValueError.
        """
        if self.covariance_type not in ("diag", "full"):
            raise ModelError("covariance type must be diag or full")
        if not isinstance(self.means, np.ndarray) or self.means.ndim != 2:
            raise ModelError("means must have shape (K, d)")
        k, d = self.means.shape
        if k < 1 or d < 1:
            raise ModelError("a model needs at least one state and feature")
        for name, shape in (
            ("startprob", (k,)),
            ("transmat", (k, k)),
            ("means", (k, d)),
            ("covars", (k, d, d)),
        ):
            _check_array(getattr(self, name), shape, name)
        for probs in (self.startprob, self.transmat):
            if (probs < 0).any() or not np.allclose(
                probs.sum(axis=-1), 1.0, rtol=0, atol=SUM_TOLERANCE
            ):
                raise ModelError("probabilities must be nonnegative and sum to one")
        for covariance in self.covars:
            if not np.array_equal(covariance, covariance.T):
                raise ModelError("covariances must be exactly symmetric")
            if self.covariance_type == "diag" and np.any(
                covariance != np.diag(np.diag(covariance))
            ):
                raise ModelError("a diag covariance must have zero off-diagonal entries")
            try:
                np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError as exc:
                raise ModelError("covariances must be positive definite") from exc
        _freeze_arrays(self, "startprob", "transmat", "means", "covars")

    @property
    def n_states(self) -> int:
        """K."""
        return int(self.startprob.shape[0])

    @property
    def n_features(self) -> int:
        """d."""
        return int(self.means.shape[1])


@dataclass(frozen=True, eq=False)
class ScalerParams:
    """The frozen z-score parameters the model's features are standardized with (spec §3).

    `columns` names the d feature columns in order; `mean` and `std` are (d,). B3's
    `quant.features.scaler.Scaler` carries the same three values; the trainer copies them in.
    """

    columns: tuple[str, ...]
    mean: Floats
    std: Floats

    def __post_init__(self) -> None:
        """Raise ModelError unless `columns` is a non-empty tuple of distinct strings, and
        `mean` and `std` are 1-D float64 arrays with one finite value per column, every std
        > 0. Replace the arrays with read-only copies."""
        if (
            not isinstance(self.columns, tuple)
            or not self.columns
            or any(not isinstance(name, str) for name in self.columns)
            or len(set(self.columns)) != len(self.columns)
        ):
            raise ModelError("scaler columns must be a nonempty tuple of distinct strings")
        for name in ("mean", "std"):
            _check_array(getattr(self, name), (len(self.columns),), name)
        if (self.std <= 0).any():
            raise ModelError("scaler standard deviations must be positive")
        _freeze_arrays(self, "mean", "std")

    def standardize(self, rows: Floats) -> Floats:
        """`(rows - mean) / std`, exactly that expression, as a new float64 array. `rows` is one
        row (d,) or a matrix (n, d). Raises ModelError on any other shape or dtype, or a value
        that isn't finite."""
        if (
            not isinstance(rows, np.ndarray)
            or rows.ndim not in (1, 2)
            or rows.shape[-1] != len(self.columns)
        ):
            raise ModelError("scaler input must be a row or matrix with its column count")
        _check_array(rows, rows.shape, "scaler input")
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = (rows - self.mean) / self.std
        if not np.isfinite(scaled).all():
            raise ModelError("standardized features overflow float64")
        return scaled


class TrainingRecord(StrictModel):
    """How the model was trained (spec §4, ARCHITECTURE §7.1), as stored in meta.json.

    - `base_seed`: the configured base seed.
    - `seeds` and `restart_logliks`: one entry per restart, in restart order: its seed and its
      final training log-likelihood. At least one restart.
    - `best`: the index of the winning restart.
    - `info`: everything else the trainer records (data hash, the BIC and validation table,
      the validation block's predictive log-likelihood distribution, sanity checks, the code
      commit, library and BLAS versions). Any value with a canonical JSON form. It is part of
      the artifact hash.
    """

    base_seed: int
    seeds: tuple[int, ...]
    restart_logliks: tuple[float, ...]
    best: int
    info: dict[str, Any]


class ModelMeta(StrictModel):
    """meta.json's `meta` object: every non-array value of an HmmModel. `HmmModel.meta()` builds
    it, and loading parses it with `ModelMeta.model_validate_json` (strict, so JSON strings
    become datetimes and labels, and arrays become tuples)."""

    format: int
    covariance_type: CovarianceType
    n_states: int
    n_features: int
    feature_columns: tuple[str, ...]
    feature_config_hash: str
    fit_start: AwareDatetime
    fit_end: AwareDatetime
    labels: LabelTable
    training: TrainingRecord


@dataclass(frozen=True, eq=False)
class HmmModel:
    """A trained, labeled model: the unit that is saved, loaded, filtered and swapped.

    `fit_start` and `fit_end` are the fit window, [fit_start, fit_end) on the rows'
    `available_at`. Every decision that uses the model must come at or after `fit_end` (T-NL2).
    `feature_config_hash` is the `FeatureConfig.config_hash()` the features were computed under.
    """

    params: GaussianParams
    scaler: ScalerParams
    fit_start: datetime
    fit_end: datetime
    feature_config_hash: str
    labels: LabelTable
    training: TrainingRecord

    def __post_init__(self) -> None:
        """Raise ModelError unless the parts agree: `scaler` has `params.n_features` columns,
        `labels.states` has `params.n_states` rows, `feature_config_hash` is 64 lowercase hex
        characters, `training` has as many seeds as log-likelihoods (≥ 1) with
        0 ≤ best < len(seeds) and an `info` with a canonical JSON form, and
        fit_start < fit_end. A naive fit_start or fit_end raises ClockError. Both are stored in
        UTC (`object.__setattr__`)."""
        start, end = ensure_utc(self.fit_start), ensure_utc(self.fit_end)
        if (
            len(self.scaler.columns) != self.params.n_features
            or len(self.labels.states) != self.params.n_states
            or re.fullmatch(r"[0-9a-f]{64}", self.feature_config_hash) is None
            or start >= end
        ):
            raise ModelError("model parts, feature hash or fit window do not agree")
        training = self.training
        if (
            not training.seeds
            or len(training.seeds) != len(training.restart_logliks)
            or not 0 <= training.best < len(training.seeds)
        ):
            raise ModelError("training record needs matching restarts and a valid best index")
        try:
            canonical_dumps(training)
        except ValueError as exc:
            raise ModelError("training record has no canonical JSON form") from exc
        object.__setattr__(self, "fit_start", start)
        object.__setattr__(self, "fit_end", end)

    def meta(self) -> ModelMeta:
        """The model's ModelMeta, with `format` = ARTIFACT_FORMAT."""
        return ModelMeta(
            format=ARTIFACT_FORMAT,
            covariance_type=self.params.covariance_type,
            n_states=self.params.n_states,
            n_features=self.params.n_features,
            feature_columns=self.scaler.columns,
            feature_config_hash=self.feature_config_hash,
            fit_start=self.fit_start,
            fit_end=self.fit_end,
            labels=self.labels,
            training=self.training,
        )

    def arrays(self) -> dict[str, Floats]:
        """The arrays of params.npz, keyed in PARAM_NAMES order: the four of `params`, then the
        scaler's `mean` as "scaler_mean" and `std` as "scaler_std"."""
        return {
            "startprob": self.params.startprob,
            "transmat": self.params.transmat,
            "means": self.params.means,
            "covars": self.params.covars,
            "scaler_mean": self.scaler.mean,
            "scaler_std": self.scaler.std,
        }

    def artifact_hash(self) -> str:
        """The artifact hash defined in the module docstring, as 64 lowercase hex characters.
        Deterministic: it depends only on `meta()` and `arrays()`."""
        return sha256_hex(
            {
                "meta": self.meta(),
                "params": {
                    name: {
                        "dtype": "float64",
                        "shape": list(array.shape),
                        "sha256": hashlib.sha256(
                            np.ascontiguousarray(array, dtype="<f8").tobytes()
                        ).hexdigest(),
                    }
                    for name, array in self.arrays().items()
                },
            }
        )


def save_model(model: HmmModel, root: Path) -> Path:
    """Write the model's artifact to `root/<artifact_hash>/` and return that directory.

    The files are written into a temporary directory inside `root` and renamed into place, so
    a reader never sees half an artifact. `root` is created if missing. Saving a model whose
    directory already exists checks it with `load_model` and returns it unchanged, so saving is
    idempotent; an existing directory that doesn't load as this model raises ModelError and is
    left alone.
    """
    digest = model.artifact_hash()
    directory = root / digest
    try:
        root.mkdir(parents=True, exist_ok=True)
        if directory.exists():
            load_model(directory)
            return directory
        with tempfile.TemporaryDirectory(prefix=".model-", dir=root) as temporary:
            staged = Path(temporary) / digest
            staged.mkdir()
            with zipfile.ZipFile(
                staged / "params.npz", "w", compression=zipfile.ZIP_STORED
            ) as archive:
                for name, array in model.arrays().items():
                    info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                    with archive.open(info, "w") as member:
                        np.lib.format.write_array(member, array, allow_pickle=False)
            (staged / "meta.json").write_bytes(
                canonical_dumps({"artifact_hash": digest, "meta": model.meta()})
            )
            try:
                staged.rename(directory)
            except OSError:
                # Another writer may have published the same artifact while we wrote it.
                if not directory.exists():
                    raise
                load_model(directory)
        return directory
    except (OSError, ValueError) as exc:
        raise ModelError(f"cannot save model artifact at {directory}") from exc


def load_model(directory: Path) -> HmmModel:
    """The model saved in `directory`, with every array bit for bit as saved.

    Reads params.npz only with `numpy.load(path, allow_pickle=False)`, so an object array is
    refused, never unpickled. Raises ModelError when a file is missing or unreadable, params.npz
    holds an object array or any member other than PARAM_NAMES' `<name>.npy`, an array isn't
    float64, meta.json isn't canonical JSON of {"artifact_hash", "meta"}, the meta doesn't
    parse or contradicts the arrays, the model fails its own checks, or the hash recomputed from
    what was read differs from meta.json's `artifact_hash` or from the directory's name.
    """
    try:
        raw = (directory / "meta.json").read_bytes()
        document = canonical_loads(raw)
        if (
            not isinstance(document, dict)
            or set(document) != {"artifact_hash", "meta"}
            or canonical_dumps(document) != raw
        ):
            raise ModelError("meta.json must be the canonical artifact document")
        meta = ModelMeta.model_validate_json(canonical_dumps(document["meta"]))
        with zipfile.ZipFile(directory / "params.npz") as archive:
            if sorted(archive.namelist()) != sorted(f"{name}.npy" for name in PARAM_NAMES):
                raise ModelError("params.npz must hold exactly the six parameter arrays")
        with np.load(directory / "params.npz", allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in PARAM_NAMES}
        model = HmmModel(
            params=GaussianParams(
                meta.covariance_type, *(arrays[name] for name in PARAM_NAMES[:4])
            ),
            scaler=ScalerParams(meta.feature_columns, arrays["scaler_mean"], arrays["scaler_std"]),
            fit_start=meta.fit_start,
            fit_end=meta.fit_end,
            feature_config_hash=meta.feature_config_hash,
            labels=meta.labels,
            training=meta.training,
        )
        if meta != model.meta():
            raise ModelError("model metadata contradicts its parameters or format")
        if (
            model.artifact_hash() != document["artifact_hash"]
            or directory.name != document["artifact_hash"]
        ):
            raise ModelError("model artifact hash does not match its contents or directory")
        return model
    except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile, ClockError) as exc:
        raise ModelError(f"cannot load model artifact at {directory}") from exc
