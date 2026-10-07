"""B4a review, part 2: an immutable model and training record, the scaler's column guard, the
top label's name, durable artifact writes and the winning restart (BUILD_LOG, B4)."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quant.core.errors import ModelError
from tests.support.hmm import (
    DEFAULT_LABELS,
    identity_scaler,
    model_of,
    sample,
    three_full,
    training_record,
)


def _model(training: Any = None, scaler: Any = None) -> Any:
    return model_of(three_full(), DEFAULT_LABELS["three_full"], scaler, training=training)


@pytest.mark.xfail(strict=True, reason="red:B4", raises=pytest.fail.Exception)
@pytest.mark.spec("ARCH§7.2")
def test_the_training_record_of_a_model_cannot_change() -> None:
    """Codex R4: `training.info` was a plain dict, so assigning to it changed the artifact hash
    of a model that a filter had already cached, and its own snapshots stopped restoring."""
    from quant.hmm.filter import ForwardFilter

    record = training_record().model_copy(
        update={"info": {"note": "synthetic", "table": [1, {"k": 2}]}}
    )
    model = _model(record)
    digest = model.artifact_hash()
    snapshot = ForwardFilter(model).snapshot()
    with pytest.raises(TypeError):
        model.training.info["note"] = "updated"
    with pytest.raises((TypeError, AttributeError)):
        model.training.info["table"].append(3)
    with pytest.raises(TypeError):
        model.training.info["table"][1]["k"] = 3
    assert model.artifact_hash() == digest
    ForwardFilter.restore(model, snapshot)


@pytest.mark.xfail(strict=True, reason="red:B4", raises=AssertionError)
@pytest.mark.spec("ARCH§7.2")
def test_the_callers_training_info_cannot_change_a_built_model() -> None:
    """Codex R4: the model kept the caller's dict, so the caller could change the model."""
    info: dict[str, Any] = {"note": "synthetic", "table": [1, 2]}
    model = _model(training_record().model_copy(update={"info": info}))
    digest = model.artifact_hash()
    info["note"] = "updated"
    info["table"].append(3)
    assert model.artifact_hash() == digest


@pytest.mark.xfail(strict=True, reason="red:B4", raises=pytest.fail.Exception)
@pytest.mark.spec("ARCH§7.2")
def test_a_models_arrays_cannot_be_made_writable_again() -> None:
    """Codex R4: a read-only copy that owns its memory can be flipped back to writable."""
    model = _model(scaler=identity_scaler(2))
    for array in model.arrays().values():
        with pytest.raises(ValueError, match="WRITEABLE"):
            array.flags.writeable = True


@pytest.mark.xfail(strict=True, reason="red:B4", raises=pytest.fail.Exception)
@pytest.mark.spec("SPEC§3")
def test_the_models_scaler_refuses_a_constant_column() -> None:
    """Claude C5: B3's scaler refuses a std that is only rounding (≤ STD_RTOL × |mean|), and the
    model's copy of it must too, also on load."""
    from quant.hmm.model import ScalerParams

    with pytest.raises(ModelError):
        ScalerParams(("a", "b"), np.array([0.1, 0.0]), np.array([1e-17, 1.0]))


@pytest.mark.xfail(strict=True, reason="red:B4", raises=AttributeError)
@pytest.mark.spec("SPEC§3")
def test_the_models_scaler_is_b3s_scaler() -> None:
    """Claude C5: the trainer converts B3's Scaler with a check, and both standardize alike."""
    from quant.features.scaler import Scaler
    from quant.hmm.model import ScalerParams

    start, end = datetime(2016, 1, 4, tzinfo=UTC), datetime(2019, 1, 1, tzinfo=UTC)
    scaler = Scaler(
        ("ret", "rvol"), np.array([0.001, 0.02]), np.array([0.01, 0.005]), start, end, 9
    )
    params = ScalerParams.from_scaler(scaler)
    assert params.columns == scaler.columns
    assert params.mean.tobytes() == scaler.mean.tobytes()
    assert params.std.tobytes() == scaler.std.tobytes()
    rows = np.random.default_rng(5).normal(0.0, 0.02, (20, 2))
    ours = params.standardize(rows, columns=("ret", "rvol"))
    assert ours.tobytes() == scaler.transform(rows, columns=("ret", "rvol")).tobytes()


@pytest.mark.xfail(strict=True, reason="red:B4", raises=TypeError)
@pytest.mark.spec("SPEC§3", "SPEC§6")
def test_the_filter_refuses_rows_of_other_columns() -> None:
    """Claude C5: a row of the right width from another feature config was filtered silently.
    Every entry point takes the rows' column names, which must be the scaler's."""
    from quant.eval.smoother import smooth
    from quant.hmm.filter import ForwardFilter, forward_pass

    model = _model()
    _, x = sample(three_full(), 20, seed=17)
    right, wrong = model.scaler.columns, ("rvol", "ret")
    fresh = ForwardFilter(model)
    assert (
        fresh.step(x[0], columns=right).loglik
        == ForwardFilter(model).run(x[:1], columns=right)[0].loglik
    )
    for call in (
        lambda: ForwardFilter(model).step(x[0], columns=wrong),
        lambda: ForwardFilter(model).run(x, columns=wrong),
        lambda: ForwardFilter.warmed(model, x, columns=wrong),
        lambda: forward_pass(model.params, model.scaler, x, columns=wrong),
        lambda: smooth(model.params, model.scaler, x, columns=wrong),
        lambda: model.scaler.standardize(x, columns=wrong),
    ):
        with pytest.raises(ModelError):
            call()
    assert fresh.bars == 1


@pytest.mark.xfail(strict=True, reason="red:B4", raises=AssertionError)
@pytest.mark.spec("SPEC§6")
def test_the_filter_names_its_most_probable_label_the_top_label() -> None:
    """Claude C7: `active_label` read like the switching engine's active label (SPEC §8), which
    has hysteresis. The filter's is the most probable label, so it is `top_label`, with
    `top_label_remaining` its expected remaining duration."""
    from quant.hmm.filter import FilterOutput, ForwardFilter
    from quant.hmm.labels import top_label

    fields = FilterOutput.__dataclass_fields__
    assert "active_label" not in fields
    assert {"top_label", "top_label_remaining"} <= set(fields)
    model = _model()
    _, x = sample(three_full(), 30, seed=23)
    for out in ForwardFilter(model).run(x, columns=model.scaler.columns):
        assert out.top_label is top_label(out.label_probs)
        assert out.top_label_remaining is None or out.top_label_remaining >= 1.0


@pytest.mark.xfail(strict=True, reason="red:B4", raises=AssertionError)
@pytest.mark.spec("ARCH§7.2")
def test_saving_a_model_is_durable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Claude C10: without fsync, a power loss can publish the renamed directory before its
    files' bytes, leaving a truncated artifact that blocks every later save. Each file, the
    staged directory and the root are synced."""
    from quant.hmm.model import save_model

    synced: set[tuple[int, int]] = set()
    real = os.fsync

    def spy(fd: int) -> None:
        stat = os.fstat(fd)
        synced.add((stat.st_dev, stat.st_ino))
        real(fd)

    monkeypatch.setattr(os, "fsync", spy)
    root = tmp_path / "models"
    directory = save_model(_model(), root)
    for path in (directory / "params.npz", directory / "meta.json", directory, root):
        stat = path.stat()
        assert (stat.st_dev, stat.st_ino) in synced, path


@pytest.mark.xfail(strict=True, reason="red:B4", raises=AssertionError)
@pytest.mark.spec("ARCH§7.2")
def test_saving_clears_the_temporary_directories_a_crash_left(tmp_path: Path) -> None:
    """Claude C10: a crash mid-save leaves a `.model-*` directory behind. The next save, which
    holds the root's lock, removes it."""
    from quant.hmm.model import load_model, save_model

    root = tmp_path / "models"
    stale = root / ".model-crashed" / ("ab" * 32)
    stale.mkdir(parents=True)
    (stale / "params.npz").write_bytes(b"PK")
    directory = save_model(_model(), root)
    load_model(directory)
    assert not [p for p in root.iterdir() if p.name.startswith(".model-")]


@pytest.mark.xfail(strict=True, reason="red:B4", raises=pytest.fail.Exception)
@pytest.mark.spec("SPEC§4", "ARCH§7.2")
def test_the_best_restart_is_the_highest_log_likelihood() -> None:
    """Claude C11: the restart with the highest training log-likelihood wins (SPEC §4), a tie
    going to the lowest seed index. A record naming another restart is a trainer bug."""
    record = training_record()
    assert record.best == int(np.argmax(record.restart_logliks))
    tied = record.model_copy(update={"restart_logliks": (-5.0, -1.0, -1.0), "best": 1})
    assert _model(tied).training.best == 1
    for update in (
        {"best": 0},
        {"best": 2},
        {"restart_logliks": (-5.0, -1.0, -1.0), "best": 2},
    ):
        with pytest.raises(ModelError):
            _model(record.model_copy(update=update))
