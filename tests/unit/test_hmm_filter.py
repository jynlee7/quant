"""B4: our log-space forward filter and every output spec §6 lists, on synthetic HMM data."""

from __future__ import annotations

import math
import subprocess
import sys
from collections.abc import Callable

import numpy as np
import pytest

from quant.core.canonical import canonical_dumps, canonical_loads
from quant.core.errors import ModelError
from quant.core.regime import Label
from tests.support.hmm import (
    DEFAULT_LABELS,
    TRUTHS,
    Truth,
    brute_force,
    four_sparse,
    hmmlearn_score,
    model_of,
    naive_filter,
    sample,
    three_full,
    two_diag,
)

TOLERANCE = 1e-10  # PLAN B4: the filter matches the naive reference within 1e-10


def _model(name: str) -> object:
    return model_of(TRUTHS[name](), DEFAULT_LABELS[name])


def _same(one: object, two: object) -> bool:
    """Two FilterOutputs equal bit for bit."""
    a, b = vars(one), vars(two)
    if a.keys() != b.keys():
        return False
    for key, value in a.items():
        other = b[key]
        if isinstance(value, np.ndarray):
            if value.dtype != other.dtype or value.tobytes() != other.tobytes():
                return False
        elif isinstance(value, float):
            if not isinstance(other, float) or value.hex() != other.hex():
                return False
        elif isinstance(value, dict):
            if list(value) != list(other):
                return False
            if any(value[k].hex() != other[k].hex() for k in value):
                return False
        elif value != other:
            return False
    return True


@pytest.mark.spec("SPEC§6")
def test_the_references_agree_with_brute_force_and_hmmlearn() -> None:
    """The oracles themselves: the naive filter and smoother equal path enumeration on a few
    bars, and hmmlearn's smoother and score on many."""
    from tests.support.hmm import hmmlearn_posteriors, naive_smoother

    for make in TRUTHS.values():
        truth = make()
        _, few = sample(truth, 6, seed=1)
        filtered, smoothed = brute_force(truth, few)
        np.testing.assert_allclose(naive_filter(truth, few).probs, filtered, atol=1e-14)
        np.testing.assert_allclose(naive_smoother(truth, few), smoothed, atol=1e-14)
        _, many = sample(truth, 1000, seed=2)
        np.testing.assert_allclose(
            naive_smoother(truth, many), hmmlearn_posteriors(truth, many), atol=1e-10
        )
        assert naive_filter(truth, many).loglik.sum() == pytest.approx(
            hmmlearn_score(truth, many), rel=1e-12, abs=1e-8
        )


@pytest.mark.spec("I2", "K3")
def test_the_decision_path_never_loads_hmmlearn_or_eval() -> None:
    """Importing the filter, the labels and the model type pulls in neither hmmlearn nor eval,
    in a fresh interpreter (K3 checks the imports statically; this checks what actually runs)."""
    code = (
        "import sys\n"
        "import quant.hmm.filter, quant.hmm.labels, quant.hmm.model\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] == 'hmmlearn'"
        " or m == 'quant.eval' or m.startswith('quant.eval.'))\n"
        "print(','.join(bad))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""


@pytest.mark.spec("SPEC§6")
@pytest.mark.parametrize("name", sorted(TRUTHS))
def test_the_filter_matches_the_naive_reference(name: str) -> None:
    from quant.hmm.filter import ForwardFilter

    truth = TRUTHS[name]()
    _, x = sample(truth, 1500, seed=11)
    reference = naive_filter(truth, x)
    outputs = ForwardFilter(_model(name)).run(x)

    assert len(outputs) == len(x)
    probs = np.array([o.state_probs for o in outputs])
    nexts = np.array([o.next_state_probs for o in outputs])
    loglik = np.array([o.loglik for o in outputs])
    assert probs.dtype == np.float64
    np.testing.assert_allclose(probs, reference.probs, rtol=0, atol=TOLERANCE)
    np.testing.assert_allclose(nexts, reference.next_probs, rtol=0, atol=TOLERANCE)
    np.testing.assert_allclose(loglik, reference.loglik, rtol=0, atol=TOLERANCE)

    # On a few bars, the filter equals P(s_t | x_0..x_t) by enumerating every path.
    _, few = sample(truth, 6, seed=12)
    filtered, _ = brute_force(truth, few)
    few_probs = np.array([o.state_probs for o in ForwardFilter(_model(name)).run(few)])
    np.testing.assert_allclose(few_probs, filtered, rtol=0, atol=1e-12)


@pytest.mark.spec("SPEC§6")
def test_run_step_and_forward_pass_agree_bit_for_bit() -> None:
    from quant.hmm.filter import ForwardFilter, emission_logpdf, forward_pass

    truth = four_sparse()
    model = _model("four_sparse")
    _, x = sample(truth, 400, seed=21)

    ran = ForwardFilter(model).run(x)
    stepper = ForwardFilter(model)
    stepped = [stepper.step(row) for row in x]
    assert stepper.bars == len(x)
    assert all(_same(a, b) for a, b in zip(ran, stepped, strict=True))

    trace = forward_pass(model.params, model.scaler, x)
    assert trace.state_probs.shape == trace.log_probs.shape == (len(x), truth.k)
    assert trace.loglik.shape == (len(x),)
    assert trace.state_probs.tobytes() == np.array([o.state_probs for o in ran]).tobytes()
    assert trace.loglik.tobytes() == np.array([o.loglik for o in ran]).tobytes()
    # Impossible states are log 0: state 3 can't start, so bar 0 gives it -inf exactly.
    assert trace.log_probs[0, 3] == -math.inf
    assert trace.state_probs[0, 3] == 0.0

    # Emissions come row by row: a row's densities don't depend on the rows around it.
    whole = emission_logpdf(model.params, x)
    assert whole.shape == (len(x), truth.k)
    for t in (0, 199, 399):
        assert whole[t].tobytes() == emission_logpdf(model.params, x[t : t + 1])[0].tobytes()


@pytest.mark.spec("SPEC§6")
def test_the_filter_stays_finite_on_outliers() -> None:
    """Rows far from every state underflow a probability-space filter; ours stays finite."""
    from quant.hmm.filter import ForwardFilter

    truth = three_full()
    _, x = sample(truth, 5000, seed=31)
    x[100] = [50.0, -50.0]
    x[1000] = [1e9, -1e9]
    x[1001] = [-1e9, 1e9]
    x[2500:2510] = 1e6
    outputs = ForwardFilter(_model("three_full")).run(x)
    for out in outputs:
        assert np.isfinite(out.state_probs).all()
        assert np.isfinite(out.next_state_probs).all()
        assert abs(out.state_probs.sum() - 1.0) < 1e-12
        assert math.isfinite(out.loglik)
        assert 0.0 <= out.entropy <= 1.0
        assert all(math.isfinite(p) for p in out.label_probs.values())


@pytest.mark.full
@pytest.mark.spec("SPEC§6")
def test_the_filter_stays_finite_over_100000_bars() -> None:
    from quant.hmm.filter import ForwardFilter

    truth = three_full()
    _, x = sample(truth, 100_000, seed=32)
    outputs = ForwardFilter(_model("three_full")).run(x)
    assert len(outputs) == 100_000
    probs = np.array([o.state_probs for o in outputs])
    loglik = np.array([o.loglik for o in outputs])
    assert np.isfinite(probs).all()
    assert np.isfinite(loglik).all()
    assert np.abs(probs.sum(axis=1) - 1.0).max() < 1e-12
    assert loglik.sum() == pytest.approx(hmmlearn_score(truth, x), rel=1e-10)


@pytest.mark.spec("SPEC§6")
@pytest.mark.parametrize("name", ["four_sparse", "two_diag"])
def test_next_bar_probabilities_are_p_times_a_and_entropy_is_in_0_1(name: str) -> None:
    from quant.hmm.filter import ForwardFilter
    from quant.hmm.labels import label_probs

    truth = TRUTHS[name]()
    model = _model(name)
    _, x = sample(truth, 800, seed=41)

    table = model.labels
    n_labels = len(table.labels)
    for out in ForwardFilter(model).run(x):
        np.testing.assert_allclose(
            out.next_state_probs, out.state_probs @ truth.transmat, rtol=0, atol=1e-12
        )
        assert abs(out.next_state_probs.sum() - 1.0) < 1e-12
        expected_next = label_probs(out.next_state_probs, table)
        assert list(out.next_label_probs) == list(expected_next)
        for label, p in expected_next.items():
            assert out.next_label_probs[label] == pytest.approx(p, abs=1e-15)

        assert 0.0 <= out.entropy <= 1.0
        h = -sum(p * math.log(p) for p in out.label_probs.values() if p > 0)
        assert out.entropy == pytest.approx(min(1.0, h / math.log(n_labels)), abs=1e-12)


@pytest.mark.spec("SPEC§4", "SPEC§6")
def test_states_that_share_a_label_add_up() -> None:
    from quant.hmm.filter import ForwardFilter

    truth = four_sparse()  # labels CHOP, CALM_UP, CHOP, CRASH
    _, x = sample(truth, 300, seed=51)
    for out in ForwardFilter(_model("four_sparse")).run(x):
        p = out.state_probs
        assert list(out.label_probs) == [Label.CRASH, Label.CALM_UP, Label.CHOP]
        assert Label.STRESS not in out.label_probs  # no stress-like state, no STRESS label
        assert out.label_probs[Label.CHOP] == pytest.approx(p[0] + p[2], abs=1e-15)
        assert out.label_probs[Label.CALM_UP] == pytest.approx(p[1], abs=1e-15)
        assert out.label_probs[Label.CRASH] == pytest.approx(p[3], abs=1e-15)
        assert sum(out.label_probs.values()) == pytest.approx(1.0, abs=1e-12)
        q = out.next_state_probs
        assert out.next_label_probs[Label.CHOP] == pytest.approx(q[0] + q[2], abs=1e-15)


@pytest.mark.spec("SPEC§6")
def test_the_active_label_and_its_expected_remaining_duration() -> None:
    from quant.hmm.filter import ForwardFilter

    truth = four_sparse()
    a = truth.transmat
    _, x = sample(truth, 300, seed=61)
    seen = set()
    for out in ForwardFilter(_model("four_sparse")).run(x):
        best = max(out.label_probs.values())
        assert out.label_probs[out.active_label] == best
        seen.add(out.active_label)
        p = out.state_probs
        if out.active_label is Label.CHOP:  # states 0 and 2
            states = [0, 2]
            w = p[states] / p[states].sum()
            sub = a[np.ix_(states, states)]
            expected = float(w @ np.linalg.inv(np.eye(2) - sub) @ np.ones(2))
        elif out.active_label is Label.CALM_UP:
            expected = 1.0 / (1.0 - a[1, 1])
        else:
            expected = 1.0 / (1.0 - a[3, 3])
        assert out.remaining_duration == pytest.approx(expected, rel=1e-12)
        assert out.remaining_duration >= 1.0
    assert {Label.CHOP, Label.CALM_UP} <= seen

    # A model whose states all share one label can never leave it: no finite duration, and
    # no uncertainty between labels.
    one = model_of(truth, [Label.CHOP] * 4)
    for out in ForwardFilter(one).run(x[:50]):
        assert out.remaining_duration is None
        assert out.entropy == 0.0
        assert out.label_probs == {Label.CHOP: pytest.approx(1.0, abs=1e-12)}


@pytest.mark.spec("SPEC§6")
@pytest.mark.parametrize("make", [three_full, two_diag], ids=["three_full", "two_diag"])
def test_the_predictive_log_likelihoods_add_up_to_the_total(make: Callable[[], Truth]) -> None:
    from quant.hmm.filter import ForwardFilter

    truth = make()
    _, x = sample(truth, 2000, seed=71)
    model = model_of(truth, [Label.CHOP] * truth.k)
    total = sum(out.loglik for out in ForwardFilter(model).run(x))
    assert total == pytest.approx(hmmlearn_score(truth, x), rel=1e-12, abs=1e-8)


@pytest.mark.spec("SPEC§3", "SPEC§6")
def test_the_filter_standardizes_with_the_models_frozen_scaler() -> None:
    from quant.hmm.filter import ForwardFilter
    from quant.hmm.model import ScalerParams

    truth = three_full()
    _, z = sample(truth, 300, seed=81)
    mean = np.array([0.001, -2.5])
    std = np.array([0.02, 3.0])
    raw = z * std + mean
    scaler = ScalerParams(("ret", "rvol"), mean, std)
    scaled = model_of(truth, DEFAULT_LABELS["three_full"], scaler)
    plain = model_of(truth, DEFAULT_LABELS["three_full"])

    standardized = (raw - mean) / std
    assert scaler.standardize(raw).tobytes() == standardized.tobytes()
    ours = ForwardFilter(scaled).run(raw)
    theirs = ForwardFilter(plain).run(standardized)
    assert all(_same(a, b) for a, b in zip(ours, theirs, strict=True))


@pytest.mark.spec("SPEC§6", "ARCH§4.3")
def test_the_filter_state_survives_a_snapshot_and_restore() -> None:
    from quant.hmm.filter import FilterState, ForwardFilter

    truth = four_sparse()
    model = _model("four_sparse")
    _, x = sample(truth, 800, seed=91)

    fresh = ForwardFilter(model).snapshot()
    assert fresh.bars == 0
    assert fresh.log_probs is None
    assert fresh.model_hash == model.artifact_hash()

    # After one bar, states 2 and 3 are impossible (π is 0 there): log 0 is stored as None.
    one = ForwardFilter(model)
    one.step(x[0])
    first = one.snapshot()
    assert first.log_probs is not None
    assert first.log_probs[2] is None
    assert first.log_probs[3] is None

    whole = ForwardFilter(model)
    outputs = whole.run(x)
    half = ForwardFilter(model)
    half.run(x[:400])
    state = half.snapshot()
    text = canonical_dumps(state)
    restored_state = FilterState.model_validate(canonical_loads(text))
    assert canonical_dumps(restored_state) == text
    restored = ForwardFilter.restore(model, restored_state)
    assert restored.bars == 400
    later = restored.run(x[400:])
    assert all(_same(a, b) for a, b in zip(outputs[400:], later, strict=True))
    assert canonical_dumps(restored.snapshot()) == canonical_dumps(whole.snapshot())

    # One bar restored from the bar-0 snapshot, with its -inf entries.
    again = ForwardFilter.restore(
        model, FilterState.model_validate(canonical_loads(canonical_dumps(first)))
    )
    assert _same(again.step(x[1]), outputs[1])


@pytest.mark.spec("SPEC§6", "ARCH§4.3")
def test_restore_refuses_a_state_that_doesnt_fit() -> None:
    from quant.hmm.filter import FilterState, ForwardFilter

    truth = three_full()
    model = _model("three_full")
    _, x = sample(truth, 50, seed=101)
    f = ForwardFilter(model)
    f.run(x)
    good = f.snapshot()
    other = model_of(truth, [Label.CHOP, Label.CHOP, Label.STRESS])  # a different model
    assert other.artifact_hash() != model.artifact_hash()
    with pytest.raises(ModelError):
        ForwardFilter.restore(other, good)

    lp = good.log_probs
    assert lp is not None
    bad = [
        good.model_copy(update={"log_probs": lp[:2]}),
        good.model_copy(update={"log_probs": None}),
        good.model_copy(update={"bars": 0}),
        good.model_copy(update={"bars": -1}),
        good.model_copy(update={"log_probs": [v + 0.1 if v is not None else None for v in lp]}),
    ]
    for state in bad:
        with pytest.raises(ModelError):
            ForwardFilter.restore(model, FilterState.model_validate(state.model_dump()))


@pytest.mark.spec("SPEC§3", "SPEC§6")
def test_a_bad_row_is_a_model_error_and_changes_nothing() -> None:
    from quant.hmm.filter import ForwardFilter

    truth = three_full()
    _, x = sample(truth, 20, seed=111)
    f = ForwardFilter(_model("three_full"))
    f.run(x[:10])
    before = canonical_dumps(f.snapshot())
    for row in (
        np.array([np.nan, 0.0]),
        np.array([np.inf, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([[0.0, 0.0]]),
        np.array([0, 0], dtype=np.int64),
        np.array([0.0, 0.0], dtype=np.float32),
    ):
        with pytest.raises(ModelError):
            f.step(row)  # type: ignore[arg-type]
        assert canonical_dumps(f.snapshot()) == before
    assert f.bars == 10


@pytest.mark.spec("SPEC§6", "D4")
def test_a_swapped_in_model_warms_up_over_its_own_fit_window() -> None:
    """Spec §6: a new model's filter starts from π on its fit window's first row and runs
    forward to the swap, seeing only the past."""
    from quant.hmm.filter import ForwardFilter

    truth = two_diag()
    model = _model("two_diag")
    _, x = sample(truth, 600, seed=121)
    window, after = x[:450], x[450:]

    warmed = ForwardFilter.warmed(model, window)
    stepped = ForwardFilter(model)
    for row in window:
        stepped.step(row)
    assert warmed.bars == 450
    assert canonical_dumps(warmed.snapshot()) == canonical_dumps(stepped.snapshot())

    continuous = ForwardFilter(model).run(x)
    assert all(_same(a, b) for a, b in zip(continuous[450:], warmed.run(after), strict=True))

    empty = ForwardFilter.warmed(model, np.empty((0, truth.d)))
    assert canonical_dumps(empty.snapshot()) == canonical_dumps(ForwardFilter(model).snapshot())
