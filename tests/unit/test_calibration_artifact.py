"""B7b: the calibration artifact sizing reads (ARCHITECTURE §7.4, §9.1; spec §9).

An immutable directory `artifacts/calibrations/<sha256>/` holding canonical JSON and an
allow_pickle=False `.npz` of knots, read by `quant.core.calibration`, which imports neither
eval nor scikit-learn (K3). Pinned here: the hash is stable and covers every value, saving is
idempotent and never overwrites, a tampered or renamed artifact is refused, and a calibration
may be used only strictly after its as-of time (T-NL2).
"""

from __future__ import annotations

import functools
import subprocess
import sys
import zipfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from quant.core.calibration import (
    DOCUMENT_FILE,
    KNOTS_FILE,
    CalibrationSource,
    r1_calibration_config,
)
from quant.core.canonical import canonical_dumps, canonical_loads
from quant.core.errors import ClockError, ModelError
from quant.core.regime import Label
from tests.support.calibration import after, covered_share, exact, series

N = 4035  # 4,000 scored bars


@functools.cache
def _timid() -> object:
    """The exact forecaster squeezed to 0.35 and 0.65: CALM_UP and CHOP fail as they are and
    pass recalibrated (two knots each). CRASH and STRESS never hold and are always forecast 0:
    they fail, and their recalibration is the constant map to 0 (one knot each)."""
    from quant.eval.calibration import build_calibration

    forecast, target = exact(N)
    forecast = np.where(forecast < 0.5, 0.35, 0.65)
    return build_calibration(
        series(N, label_forecast=forecast, target=target, covered=covered_share(N)),
        as_of=after(N),
        source=CalibrationSource.BACKTEST,
        config=r1_calibration_config(),
    )


def _rebuild(calibration: object, **document: object) -> object:
    from quant.core.calibration import Calibration

    return Calibration(calibration.document.model_copy(update=document), calibration.knots)


# --- the hash and the files -----------------------------------------------------------------


@pytest.mark.spec("ARCH§7.4", "ARCH§9.1")
def test_a_saved_calibration_loads_back_with_the_same_hash(tmp_path: Path) -> None:
    from quant.core.calibration import load_calibration, save_calibration

    calibration = _timid()
    digest = calibration.artifact_hash()
    directory = save_calibration(calibration, tmp_path / "calibrations")
    assert directory == tmp_path / "calibrations" / digest
    assert sorted(p.name for p in directory.iterdir()) == sorted([DOCUMENT_FILE, KNOTS_FILE])
    loaded = load_calibration(directory)
    assert loaded.artifact_hash() == digest
    assert loaded.document == calibration.document
    assert set(loaded.knots) == set(Label)
    assert [len(loaded.knots[label][0]) for label in Label] == [1, 1, 2, 2]
    for label, (x, y) in calibration.knots.items():
        assert np.array_equal(loaded.knots[label][0], x)
        assert np.array_equal(loaded.knots[label][1], y)

    raw = (directory / DOCUMENT_FILE).read_bytes()
    document = canonical_loads(raw)
    assert canonical_dumps(document) == raw  # canonical JSON
    assert set(document) == {"artifact_hash", "calibration"}
    assert document["artifact_hash"] == digest
    with zipfile.ZipFile(directory / KNOTS_FILE) as archive:
        assert archive.namelist() == [
            f"{label.value}_{axis}.npy" for label in Label for axis in "xy"
        ]
        assert {info.compress_type for info in archive.infolist()} == {zipfile.ZIP_STORED}
        assert {info.date_time for info in archive.infolist()} == {(1980, 1, 1, 0, 0, 0)}
    with np.load(directory / KNOTS_FILE, allow_pickle=False) as archive:
        assert np.array_equal(archive["CALM_UP_x"], calibration.knots[Label.CALM_UP][0])


@pytest.mark.spec("ARCH§7.4", "ARCH§9.1")
def test_the_hash_is_stable_and_the_bytes_reproducible(tmp_path: Path) -> None:
    from quant.core.calibration import save_calibration

    first = _timid()
    _timid.cache_clear()
    second = _timid()
    assert first is not second
    assert first.artifact_hash() == second.artifact_hash()
    one = save_calibration(first, tmp_path / "a")
    two = save_calibration(second, tmp_path / "b")
    for name in (DOCUMENT_FILE, KNOTS_FILE):
        assert (one / name).read_bytes() == (two / name).read_bytes()


@pytest.mark.spec("ARCH§7.4", "ARCH§9.1")
@pytest.mark.parametrize(
    "change",
    ["as_of", "source", "models", "knot value", "config", "outcome", "data hash"],
)
def test_every_value_is_in_the_hash(change: str) -> None:
    from quant.core.calibration import Calibration

    calibration = _timid()
    document = calibration.document
    if change == "knot value":
        x, y = calibration.knots[Label.CALM_UP]
        y = y.copy()
        y[0] = np.nextafter(y[0], 1.0)
        other = Calibration(document, {**calibration.knots, Label.CALM_UP: (x, y)})
    elif change == "as_of":
        other = _rebuild(calibration, as_of=document.as_of + timedelta(microseconds=1))
    elif change == "source":
        other = _rebuild(calibration, source=CalibrationSource.FILLS)
    elif change == "models":
        other = _rebuild(calibration, models=("ef" * 32,))
    elif change == "config":
        other = _rebuild(calibration, config=document.config.model_copy(update={"failed_cap": 0.2}))
    elif change == "outcome":
        other = _rebuild(
            calibration,
            outcome=document.outcome.model_copy(update={"covered": document.outcome.covered - 1}),
        )
    else:
        other = _rebuild(calibration, data_hash="0" * 64)
    assert other.artifact_hash() != calibration.artifact_hash()


@pytest.mark.spec("ARCH§7.4", "ARCH§9.1")
def test_saving_again_is_idempotent_and_never_an_overwrite(tmp_path: Path) -> None:
    from quant.core.calibration import save_calibration

    calibration = _timid()
    root = tmp_path / "calibrations"
    directory = save_calibration(calibration, root)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    assert save_calibration(calibration, root) == directory
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    assert [p.name for p in root.iterdir()] == [directory.name]  # no staging left behind

    # A directory of that name that doesn't hold this calibration is refused and left alone.
    (directory / KNOTS_FILE).write_bytes(b"not a zip")
    with pytest.raises(ModelError):
        save_calibration(calibration, root)
    assert (directory / KNOTS_FILE).read_bytes() == b"not a zip"


@pytest.mark.spec("ARCH§7.4", "ARCH§9.1")
def test_a_calibration_is_immutable() -> None:
    from pydantic import ValidationError

    calibration = _timid()
    x, y = calibration.knots[Label.CALM_UP]
    for array in (x, y):
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 0.5
    with pytest.raises(TypeError):
        calibration.knots[Label.CRASH] = (x, y)  # type: ignore[index]
    with pytest.raises(ValidationError):
        calibration.document.bars = 1  # type: ignore[misc]
    with pytest.raises(AttributeError):
        calibration.document = calibration.document  # type: ignore[misc]


@pytest.mark.spec("ARCH§7.4", "ARCH§9.1")
def test_knots_handed_in_are_copied() -> None:
    from quant.core.calibration import Calibration

    calibration = _timid()
    x, y = (a.copy() for a in calibration.knots[Label.CALM_UP])
    mine = Calibration(calibration.document, {**calibration.knots, Label.CALM_UP: (x, y)})
    digest = mine.artifact_hash()
    y[0] = 0.0
    assert mine.artifact_hash() == digest


@pytest.mark.spec("ARCH§7.4", "ARCH§9.1")
@pytest.mark.parametrize(
    "tamper",
    [
        "document value",
        "knot value",
        "renamed",
        "extra member",
        "pickled member",
        "missing knots",
        "missing document",
        "non-canonical json",
        "float32 knots",
    ],
)
def test_a_tampered_artifact_is_refused(tamper: str, tmp_path: Path) -> None:
    from quant.core.calibration import load_calibration, save_calibration

    calibration = _timid()
    directory = save_calibration(calibration, tmp_path)
    document_path, knots_path = directory / DOCUMENT_FILE, directory / KNOTS_FILE
    arrays = {
        f"{label.value}_{axis}": values
        for label, pair in calibration.knots.items()
        for axis, values in zip("xy", pair, strict=True)
    }

    def rewrite_knots(members: dict[str, object]) -> None:
        with zipfile.ZipFile(knots_path, "w", compression=zipfile.ZIP_STORED) as archive:
            for name, array in members.items():
                with archive.open(f"{name}.npy", "w") as member:
                    np.lib.format.write_array(member, np.asarray(array), allow_pickle=True)

    if tamper == "document value":
        document = canonical_loads(document_path.read_bytes())
        document["calibration"]["bars"] += 1
        document_path.write_bytes(canonical_dumps(document))
    elif tamper == "knot value":
        rewrite_knots({**arrays, "CALM_UP_y": arrays["CALM_UP_y"] * 0.5})
    elif tamper == "renamed":
        directory = directory.rename(directory.with_name("0" * 64))
    elif tamper == "extra member":
        rewrite_knots({**arrays, "CRASH_x": np.array([0.5])})
    elif tamper == "pickled member":
        rewrite_knots({**arrays, "CALM_UP_x": np.array([0.35, 0.65], dtype=object)})
    elif tamper == "missing knots":
        knots_path.unlink()
    elif tamper == "missing document":
        document_path.unlink()
    elif tamper == "non-canonical json":
        document_path.write_bytes(document_path.read_bytes() + b" ")
    else:
        rewrite_knots({name: a.astype(np.float32) for name, a in arrays.items()})
    with pytest.raises(ModelError):
        load_calibration(directory)


@pytest.mark.spec("ARCH§7.4", "SPEC§9")
@pytest.mark.parametrize(
    "fault",
    [
        "knots missing",
        "knot count",
        "knots unasked",
        "labels out of order",
        "sufficient",
        "passed",
        "scored",
    ],
)
def test_a_calibration_whose_parts_disagree_is_refused(fault: str) -> None:
    from quant.core.calibration import Calibration

    calibration = _timid()
    document = calibration.document
    knots = dict(calibration.knots)
    if fault == "knots missing":
        del knots[Label.CHOP]
        with pytest.raises(ModelError):
            Calibration(document, knots)
        return
    if fault == "knot count":
        # CRASH's verdict counts one knot: two are refused.
        knots[Label.CRASH] = knots[Label.CHOP]
        with pytest.raises(ModelError):
            Calibration(document, knots)
        return
    if fault == "knots unasked":
        # A calibration of no bars recalibrates nothing: knots handed in are refused.
        from quant.eval.calibration import ForecastSeries, build_calibration

        empty = build_calibration(
            ForecastSeries.empty(),
            as_of=after(0),
            source=CalibrationSource.BACKTEST,
            config=r1_calibration_config(),
        )
        with pytest.raises(ModelError):
            Calibration(empty.document, {Label.CHOP: knots[Label.CHOP]})
        return
    update: dict[str, object] = {
        "labels out of order": {"labels": tuple(reversed(document.labels))},
        "sufficient": {"sufficient": False},
        "scored": {"scored": document.scored - 1},
        "passed": {
            "labels": tuple(
                v.model_copy(update={"passed": not v.passed}) if v.label is Label.CRASH else v
                for v in document.labels
            )
        },
    }[fault]
    with pytest.raises(ModelError):
        _rebuild(calibration, **update)


# --- using one ------------------------------------------------------------------------------


@pytest.mark.spec("T-NL2", "ARCH§7.4")
def test_a_calibration_is_used_only_strictly_after_its_as_of_time() -> None:
    calibration = _timid()
    as_of = calibration.as_of
    calibration.require_before(as_of + timedelta(microseconds=1))
    for tau in (as_of, as_of - timedelta(hours=1)):
        with pytest.raises(ModelError):
            calibration.require_before(tau)
    with pytest.raises(ClockError):
        calibration.require_before(as_of.replace(tzinfo=None) + timedelta(days=1))


@pytest.mark.spec("SPEC§9", "ARCH§7.4")
def test_sizing_reads_the_calibrated_probability_and_the_cap_fraction() -> None:
    calibration = _timid()
    assert calibration.passed(Label.CALM_UP)
    assert calibration.passed(Label.CHOP)
    assert calibration.document.outcome.passed
    assert calibration.cap_fraction(Label.CALM_UP) == 1.0
    assert calibration.cap_fraction(Label.CRASH) == 0.25  # never present: no skill shown
    # P̃: the knots map 0.35 → 0.05 and 0.65 → 0.95, linearly between, clipped outside.
    assert abs(calibration.calibrated(Label.CALM_UP, 0.5) - 0.5) < 1e-12
    assert abs(calibration.calibrated(Label.CALM_UP, 0.1) - 0.05) < 1e-12
    assert abs(calibration.calibrated(Label.CALM_UP, 0.9) - 0.95) < 1e-12
    # STRESS never held in the scored bars: its map sends every forecast to 0.
    assert calibration.calibrated(Label.STRESS, 0.3) == 0.0
    # A label probability that left [0, 1] by rounding is clipped; anything else is refused.
    assert abs(calibration.calibrated(Label.CALM_UP, 1.0000000000000002) - 0.95) < 1e-12
    assert abs(calibration.calibrated(Label.CALM_UP, -1e-17) - 0.05) < 1e-12
    for p in (1.1, -0.01, float("nan"), float("inf")):
        with pytest.raises(ModelError):
            calibration.calibrated(Label.CALM_UP, p)


@pytest.mark.spec("K3", "ARCH§7.4")
def test_the_artifact_module_imports_neither_eval_nor_scikit_learn() -> None:
    # Sizing reads calibrations as plain data (ARCHITECTURE §7.4, K3).
    code = (
        "import sys, quant.core.calibration\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('sklearn', 'hmmlearn')"
        " or m == 'quant.eval' or m.startswith('quant.eval.'))\n"
        "print(','.join(bad))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""
