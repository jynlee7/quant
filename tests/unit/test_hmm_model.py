"""B4: the model type and its artifact, params.npz plus meta.json, never pickles (ARCHITECTURE §7.2)."""

from __future__ import annotations

import hashlib
import io
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quant.core.canonical import canonical_dumps, canonical_loads, sha256_hex
from quant.core.errors import ClockError, ModelError
from quant.core.regime import Label
from tests.support.hmm import (
    DEFAULT_LABELS,
    FEATURE_HASH,
    FIT_END,
    FIT_START,
    Truth,
    four_sparse,
    identity_scaler,
    label_table,
    model_of,
    params_of,
    three_full,
    training_record,
    two_diag,
)

PARAM_NAMES = ("startprob", "transmat", "means", "covars", "scaler_mean", "scaler_std")


def _model(truth: Truth | None = None, **kwargs: Any) -> Any:
    truth = truth or three_full()
    return model_of(truth, DEFAULT_LABELS["three_full"] if truth.k == 3 else None, **kwargs)


def _expected_hash(model: Any) -> str:
    """The artifact hash as ARCHITECTURE §7.2's module docstring defines it, computed here."""
    arrays = model.arrays()
    return sha256_hex(
        {
            "meta": model.meta(),
            "params": {
                name: {
                    "dtype": "float64",
                    "shape": list(arrays[name].shape),
                    "sha256": hashlib.sha256(
                        np.ascontiguousarray(arrays[name], dtype="<f8").tobytes()
                    ).hexdigest(),
                }
                for name in PARAM_NAMES
            },
        }
    )


@pytest.mark.spec("ARCH§7.2", "SPEC§4", "D4")
@pytest.mark.parametrize("make", [three_full, two_diag, four_sparse])
def test_an_artifact_round_trips_bit_for_bit(tmp_path: Path, make: Any) -> None:
    from quant.hmm.model import ARTIFACT_FORMAT, load_model, save_model

    truth = make()
    model = model_of(truth, DEFAULT_LABELS[make.__name__])
    directory = save_model(model, tmp_path / "models")
    assert directory == tmp_path / "models" / model.artifact_hash()
    assert sorted(p.name for p in directory.iterdir()) == ["meta.json", "params.npz"]

    loaded = load_model(directory)
    assert loaded.artifact_hash() == model.artifact_hash()
    before, after = model.arrays(), loaded.arrays()
    assert list(after) == list(PARAM_NAMES)
    for name in PARAM_NAMES:
        assert after[name].dtype == np.float64
        assert after[name].shape == before[name].shape
        assert after[name].tobytes() == before[name].tobytes()
        assert not after[name].flags.writeable
    assert loaded.meta() == model.meta()
    assert canonical_dumps(loaded.meta()) == canonical_dumps(model.meta())

    # What the artifact carries (ARCHITECTURE §7.2).
    meta = loaded.meta()
    assert meta.format == ARTIFACT_FORMAT
    assert meta.n_states == truth.k
    assert meta.n_features == truth.d
    assert meta.covariance_type == truth.covariance_type
    assert meta.feature_columns == loaded.scaler.columns
    assert meta.feature_config_hash == FEATURE_HASH
    assert (meta.fit_start, meta.fit_end) == (FIT_START, FIT_END)  # D4's window, T-NL2's input
    assert [s.label for s in meta.labels.states] == DEFAULT_LABELS[make.__name__]
    assert meta.training.seeds == (20160104, 20160105, 20160106)
    assert meta.training.best == 1
    assert loaded.fit_end.tzinfo == UTC
    assert loaded.params.covariance_type == truth.covariance_type
    assert loaded.params.covars.shape == (truth.k, truth.d, truth.d)


@pytest.mark.spec("ARCH§7.2", "ARCH§14")
def test_the_files_and_the_hash_are_deterministic(tmp_path: Path) -> None:
    from quant.hmm.model import save_model

    one = save_model(_model(), tmp_path / "a")
    two = save_model(_model(), tmp_path / "b")  # built again from fresh copies
    assert one.name == two.name
    for name in ("params.npz", "meta.json"):
        assert (one / name).read_bytes() == (two / name).read_bytes()

    # meta.json is canonical JSON of the hash and the meta.
    raw = (one / "meta.json").read_bytes()
    document = canonical_loads(raw)
    assert canonical_dumps(document) == raw
    assert sorted(document) == ["artifact_hash", "meta"]
    assert document["artifact_hash"] == one.name

    # params.npz holds exactly the six arrays, as fixed-date, uncompressed members.
    with zipfile.ZipFile(one / "params.npz") as archive:
        infos = archive.infolist()
    assert [i.filename for i in infos] == [f"{name}.npy" for name in PARAM_NAMES]
    assert all(i.date_time == (1980, 1, 1, 0, 0, 0) for i in infos)
    assert all(i.compress_type == zipfile.ZIP_STORED for i in infos)


@pytest.mark.spec("ARCH§7.2", "SPEC§4")
def test_the_artifact_hash_follows_its_definition_and_every_value() -> None:
    model = _model()
    digest = model.artifact_hash()
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)
    assert digest == _expected_hash(model)
    assert _model().artifact_hash() == digest

    truth = three_full()
    nudged_means = truth.means.copy()
    nudged_means[1, 0] = np.nextafter(nudged_means[1, 0], np.inf)  # one ulp
    changes = [
        model_of(
            Truth("full", truth.startprob, truth.transmat, nudged_means, truth.covars),
            DEFAULT_LABELS["three_full"],
        ),
        _model(fit_end=datetime(2019, 2, 1, tzinfo=UTC)),
        _model(training=training_record(base_seed=7)),
        model_of(truth, [Label.CHOP, Label.CALM_UP, Label.CRASH]),
        model_of(truth, DEFAULT_LABELS["three_full"], scaler=_scaler(std=[1.0, 2.0])),
    ]
    hashes = {m.artifact_hash() for m in changes}
    assert digest not in hashes
    assert len(hashes) == len(changes)


def _scaler(std: list[float]) -> Any:
    from quant.hmm.model import ScalerParams

    return ScalerParams(("f0", "f1"), np.zeros(2), np.array(std))


@pytest.mark.spec("ARCH§7.2")
def test_loading_never_unpickles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from quant.hmm.model import load_model, save_model

    directory = save_model(_model(), tmp_path)
    calls: list[dict[str, Any]] = []
    real_load = np.load

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", spy)
    load_model(directory)
    assert calls
    assert all(call.get("allow_pickle") is False for call in calls)
    monkeypatch.undo()

    # An object array in params.npz is refused, not unpickled: its payload's __reduce__ would
    # run code on unpickling, and must not. (Claude C9: a plain object array failed the shape
    # check even when unpickled, so it couldn't tell.)
    planted = tmp_path / "planted" / directory.name
    shutil.copytree(directory, planted)
    with np.load(directory / "params.npz", allow_pickle=False) as npz:
        arrays = {name: npz[name] for name in npz.files}
    payload = np.empty(1, dtype=object)
    payload[0] = _Payload()
    arrays["means"] = payload
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)  # type: ignore[arg-type]
    (planted / "params.npz").write_bytes(buffer.getvalue())
    _UNPICKLED.clear()
    with pytest.raises(ModelError):
        load_model(planted)
    assert not _UNPICKLED


_UNPICKLED: list[str] = []


def _unpickled(note: str) -> None:
    _UNPICKLED.append(note)


class _Payload:
    """Unpickling this calls `_unpickled`: the side effect a pickle exploit would have."""

    def __reduce__(self) -> tuple[Any, tuple[str]]:
        return (_unpickled, ("ran",))


def _copy(directory: Path, where: Path) -> Path:
    target = where / directory.name
    shutil.copytree(directory, target)
    return target


@pytest.mark.spec("ARCH§7.2")
def test_a_tampered_or_incomplete_artifact_is_refused(tmp_path: Path) -> None:
    from quant.hmm.model import load_model, save_model

    directory = save_model(_model(), tmp_path / "good")
    meta = (directory / "meta.json").read_bytes()
    with np.load(directory / "params.npz", allow_pickle=False) as npz:
        arrays = {name: npz[name] for name in npz.files}

    def variant(name: str) -> Path:
        return _copy(directory, tmp_path / name)

    # A changed value in meta.json.
    bad = variant("meta")
    document = canonical_loads(meta)
    document["meta"]["training"]["base_seed"] = 1
    (bad / "meta.json").write_bytes(canonical_dumps(document))
    # A changed array.
    changed = variant("array")
    altered = dict(arrays)
    altered["transmat"] = arrays["transmat"][[1, 0, 2]][:, [1, 0, 2]]
    np.savez(changed / "params.npz", **altered)  # type: ignore[arg-type]
    # An extra array.
    extra = variant("extra")
    np.savez(extra / "params.npz", **arrays, spare=np.zeros(1))  # type: ignore[arg-type]
    # A float32 array.
    narrow = variant("narrow")
    np.savez(narrow / "params.npz", **{**arrays, "means": arrays["means"].astype(np.float32)})  # type: ignore[arg-type]
    # meta.json not canonical (spaces), though the same JSON.
    loose = variant("loose")
    (loose / "meta.json").write_text(meta.decode().replace(",", ", ", 1))
    # A missing file.
    missing = variant("missing")
    (missing / "params.npz").unlink()
    # The directory renamed: its name no longer matches the hash.
    renamed = tmp_path / "renamed" / ("0" * 64)
    shutil.copytree(directory, renamed)

    for path in (bad, changed, extra, narrow, loose, missing, renamed):
        with pytest.raises(ModelError):
            load_model(path)


@pytest.mark.spec("ARCH§7.2")
def test_saving_is_idempotent_and_never_overwrites(tmp_path: Path) -> None:
    from quant.hmm.model import load_model, save_model

    model = _model()
    first = save_model(model, tmp_path)
    stamp = (first / "meta.json").stat().st_mtime_ns
    assert save_model(model, tmp_path) == first
    assert (first / "meta.json").stat().st_mtime_ns == stamp
    assert [p.name for p in tmp_path.iterdir()] == [first.name]  # no temporary left behind

    # A directory under the model's name that isn't this model is left alone.
    (first / "meta.json").write_bytes(b"{}")
    with pytest.raises(ModelError):
        save_model(model, tmp_path)
    assert (first / "meta.json").read_bytes() == b"{}"
    with pytest.raises(ModelError):
        load_model(first)


def _bad_params() -> list[tuple[str, dict[str, Any]]]:
    t = three_full()
    base = {
        "covariance_type": "full",
        "startprob": t.startprob,
        "transmat": t.transmat,
        "means": t.means,
        "covars": t.covars,
    }
    asym = t.covars.copy()
    asym[0, 0, 1] += 1e-9
    singular = t.covars.copy()
    singular[1] = [[1.0, 1.0], [1.0, 1.0]]
    rows = t.transmat.copy()
    rows[0] = [0.9, 0.2, -0.1]
    unnormalized = t.transmat.copy()
    unnormalized[2, 2] += 1e-6
    nan_means = t.means.copy()
    nan_means[0, 0] = np.nan
    return [
        ("covariance type", {**base, "covariance_type": "spherical"}),
        ("startprob sum", {**base, "startprob": np.array([0.5, 0.3, 0.3])}),
        ("negative transition", {**base, "transmat": rows}),
        ("row sum", {**base, "transmat": unnormalized}),
        ("asymmetric", {**base, "covars": asym}),
        ("not positive definite", {**base, "covars": singular}),
        ("diag with off-diagonals", {**base, "covariance_type": "diag"}),
        ("float32", {**base, "means": t.means.astype(np.float32)}),
        ("nan", {**base, "means": nan_means}),
        ("shape", {**base, "means": t.means[:2]}),
        ("covars shape", {**base, "covars": t.covars[:, :1, :1]}),
    ]


@pytest.mark.spec("SPEC§4", "ARCH§7.2")
@pytest.mark.parametrize(("why", "fields"), _bad_params(), ids=[w for w, _ in _bad_params()])
def test_bad_parameters_are_refused(why: str, fields: dict[str, Any]) -> None:
    from quant.hmm.model import GaussianParams

    with pytest.raises(ModelError):
        GaussianParams(**fields)


@pytest.mark.spec("SPEC§4", "ARCH§7.2")
def test_the_model_checks_its_parts_and_freezes_its_arrays() -> None:
    from quant.hmm.model import HmmModel, ScalerParams

    truth = three_full()
    params = params_of(truth)
    source = truth.means
    assert params.means is not source
    with pytest.raises(ValueError, match="read-only"):
        params.means[0, 0] = 9.0
    assert (params.n_states, params.n_features) == (3, 2)

    def build(**changes: Any) -> Any:
        fields = {
            "params": params,
            "scaler": identity_scaler(2),
            "fit_start": FIT_START,
            "fit_end": FIT_END,
            "feature_config_hash": FEATURE_HASH,
            "labels": label_table(DEFAULT_LABELS["three_full"]),
            "training": training_record(),
        }
        return HmmModel(**{**fields, **changes})

    assert build().fit_end == FIT_END
    for changes in (
        {"scaler": identity_scaler(3)},
        {"labels": label_table([Label.CHOP, Label.CHOP])},
        {"feature_config_hash": "AB" * 32},
        {"feature_config_hash": "ab" * 31},
        {"fit_end": FIT_START},
        {"training": training_record().model_copy(update={"best": 3})},
        {"training": training_record().model_copy(update={"restart_logliks": (-1.0,)})},
    ):
        with pytest.raises(ModelError):
            build(**changes)
    with pytest.raises(ClockError):
        build(fit_end=datetime(2019, 1, 1))  # noqa: DTZ001 - naive on purpose

    for mean, std, columns in (
        (np.zeros(2), np.array([1.0, 0.0]), ("a", "b")),
        (np.zeros(2), np.ones(2), ("a", "a")),
        (np.zeros(2), np.ones(2), ("a",)),
        (np.array([np.nan, 0.0]), np.ones(2), ("a", "b")),
    ):
        with pytest.raises(ModelError):
            ScalerParams(columns, mean, std)
