"""B8a: seed release r1 and its loader (spec §7, ARCHITECTURE §9.2, PLAN §5, P7).

PLAN B8's "done when": r1's files validate against the schema. Their values are PLAN §5's,
exactly, fixed before any data was looked at (P7). The manifest names the features, the HMM
search and the label thresholds that the code's own r1 functions give, and the label ->
playbook map sends each label to its own file. The release id is the hash of the parsed front
matters, so the prose bodies never change it. The loader refuses a missing playbook, a bad map,
a playbook filed under another label, missing or non-mapping front matter, and schema failures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quant.core.canonical import canonical_dumps, sha256_hex
from quant.core.errors import ConfigError
from quant.core.regime import Label
from tests.support.decide import REPO, copy_release, front_matter, playbook_text


@pytest.mark.spec("P7", "PLAN§5", "SPEC§7", "D8")
def test_r1_calm_up_is_plan_5s_trend_following() -> None:
    from quant.decide.playbook import LabelProbBelow, TrendParams, TrendPlaybook, load_release

    playbook = load_release(REPO).playbook(Label.CALM_UP)
    assert isinstance(playbook, TrendPlaybook)
    assert playbook.label == Label.CALM_UP
    assert playbook.strategy == "trend_following"
    assert playbook.params == TrendParams(fast=35, slow=175, atr_bars=35)
    assert (playbook.stop_loss, playbook.take_profit, playbook.max_size) == (4.0, None, 1.0)
    assert playbook.invalidation == (LabelProbBelow(kind="label_prob_below", threshold=0.30),)


@pytest.mark.spec("P7", "PLAN§5", "SPEC§7", "D8")
def test_r1_chop_is_plan_5s_mean_reversion() -> None:
    from quant.decide.playbook import (
        BarsInTradeExceeds,
        LabelProbBelow,
        MeanReversionParams,
        MeanReversionPlaybook,
        load_release,
    )

    playbook = load_release(REPO).playbook(Label.CHOP)
    assert isinstance(playbook, MeanReversionPlaybook)
    assert playbook.params == MeanReversionParams(n=35, entry_z=1.5, exit_z=0.0, atr_bars=35)
    assert (playbook.stop_loss, playbook.take_profit, playbook.max_size) == (3.0, None, 0.5)
    assert playbook.invalidation == (
        BarsInTradeExceeds(kind="bars_in_trade_exceeds", bars=21),
        LabelProbBelow(kind="label_prob_below", threshold=0.30),
    )


@pytest.mark.spec("P7", "PLAN§5", "SPEC§7", "D8")
@pytest.mark.parametrize("label", [Label.STRESS, Label.CRASH])
def test_r1_stress_and_crash_stand_aside(label: Label) -> None:
    from quant.decide.playbook import FlatPlaybook, load_release

    playbook = load_release(REPO).playbook(label)
    assert isinstance(playbook, FlatPlaybook)
    assert playbook.strategy == "flat"
    assert (playbook.stop_loss, playbook.take_profit, playbook.max_size) == (None, None, 0.0)
    assert playbook.invalidation == ()


@pytest.mark.spec("P7", "PLAN§5", "ARCH§9.2")
def test_r1_manifest_names_the_codes_r1_configs() -> None:
    from quant.decide.playbook import load_release
    from quant.features.menu import r1_features
    from quant.hmm.labels import r1_thresholds
    from quant.hmm.train import r1_train_config

    manifest = load_release(REPO).manifest
    assert manifest.release == "r1"
    assert manifest.version == 1
    assert manifest.features == r1_features()
    assert canonical_dumps(manifest.hmm) == canonical_dumps(r1_train_config())
    assert manifest.label_thresholds == r1_thresholds()
    assert manifest.playbooks == {label: f"playbooks/{label.value}.md" for label in Label}
    assert manifest.gate_report is None


@pytest.mark.spec("ARCH§9.2", "SPEC§7")
def test_the_release_id_is_the_hash_of_the_parsed_front_matters() -> None:
    from quant.decide.playbook import load_release, parse_manifest, parse_playbook

    release = load_release(REPO)
    manifest = parse_manifest((REPO / "strategy.md").read_text(encoding="utf-8"))
    playbooks = {
        label.value: parse_playbook(
            (REPO / "playbooks" / f"{label.value}.md").read_text(encoding="utf-8")
        )
        for label in Label
    }
    assert release.release_id == sha256_hex({"manifest": manifest, "playbooks": playbooks})
    assert set(release.playbooks) == set(Label)


@pytest.mark.spec("ARCH§9.2", "SPEC§7")
def test_prose_never_changes_the_release_id_but_front_matter_does(tmp_path: Path) -> None:
    from quant.decide.playbook import load_release

    original = load_release(REPO).release_id
    root = copy_release(tmp_path / "r1")
    for path in [root / "strategy.md", *sorted((root / "playbooks").iterdir())]:
        path.write_text(path.read_text(encoding="utf-8") + "\nA new paragraph.\n", "utf-8")
    assert load_release(root).release_id == original

    chop = root / "playbooks" / "CHOP.md"
    front = front_matter(chop)
    front["stop_loss"] = 3.5
    chop.write_text(playbook_text(front), "utf-8")
    assert load_release(root).release_id != original


@pytest.mark.spec("ARCH§9.2")
def test_a_release_is_immutable() -> None:
    from pydantic import ValidationError

    from quant.decide.playbook import load_release

    release = load_release(REPO)
    with pytest.raises(TypeError):
        release.playbooks[Label.CRASH] = release.playbooks[Label.CALM_UP]  # type: ignore[index]
    with pytest.raises(AttributeError):
        release.release_id = "x"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        release.manifest.release = "r2"


@pytest.mark.spec("ARCH§9.2")
def test_release_of_hashes_parsed_parts_like_the_loader() -> None:
    from quant.decide.playbook import load_release, release_id_of, release_of

    loaded = load_release(REPO)
    built = release_of(loaded.manifest, dict(loaded.playbooks))
    assert built.release_id == loaded.release_id == release_id_of(loaded.manifest, loaded.playbooks)
    misfiled = {**loaded.playbooks, Label.CRASH: loaded.playbooks[Label.STRESS]}
    with pytest.raises(ConfigError):
        release_of(loaded.manifest, misfiled)
    short = {label: pb for label, pb in loaded.playbooks.items() if label != Label.CHOP}
    with pytest.raises(ConfigError):
        release_of(loaded.manifest, short)
    with pytest.raises(ConfigError):
        release_id_of(loaded.manifest, short)


# --- what the loader refuses ----------------------------------------------------------------


def _missing_playbook(root: Path) -> None:
    (root / "playbooks" / "STRESS.md").unlink()


def _two_labels_one_file(root: Path) -> None:
    _edit_manifest(root, lambda m: m["playbooks"].update(CHOP="playbooks/CALM_UP.md"))


def _label_missing_from_map(root: Path) -> None:
    _edit_manifest(root, lambda m: m["playbooks"].pop("CRASH"))


def _another_path(root: Path) -> None:
    (root / "playbooks" / "chop_v2.md").write_bytes((root / "playbooks" / "CHOP.md").read_bytes())
    _edit_manifest(root, lambda m: m["playbooks"].update(CHOP="playbooks/chop_v2.md"))


def _unknown_label_in_map(root: Path) -> None:
    _edit_manifest(root, lambda m: m["playbooks"].update(BULL="playbooks/CALM_UP.md"))


def _misfiled_playbook(root: Path) -> None:
    # STRESS.md holding CRASH's (flat, valid) playbook: the label isn't the map's.
    front = front_matter(root / "playbooks" / "STRESS.md")
    front["label"] = "CRASH"
    (root / "playbooks" / "STRESS.md").write_text(playbook_text(front), "utf-8")


def _no_front_matter(root: Path) -> None:
    (root / "strategy.md").write_text("# Strategy r1\n\nNo front matter.\n", "utf-8")


def _list_front_matter(root: Path) -> None:
    (root / "playbooks" / "CHOP.md").write_text("---\n- 1\n- 2\n---\nbody\n", "utf-8")


def _schema_failure(root: Path) -> None:
    front = front_matter(root / "playbooks" / "CALM_UP.md")
    front["stop_loss"] = 9.0
    (root / "playbooks" / "CALM_UP.md").write_text(playbook_text(front), "utf-8")


def _manifest_unknown_key(root: Path) -> None:
    _edit_manifest(root, lambda m: m.update(owner="me"))


def _manifest_bad_release(root: Path) -> None:
    _edit_manifest(root, lambda m: m.update(release="R1"))


def _not_utf8(root: Path) -> None:
    (root / "playbooks" / "CRASH.md").write_bytes(b"---\nlabel: \xff\n---\n")


def _edit_manifest(root: Path, edit: object) -> None:
    front = front_matter(root / "strategy.md")
    edit(front)  # type: ignore[operator]
    (root / "strategy.md").write_text(playbook_text(front), "utf-8")


@pytest.mark.spec("ARCH§9.2", "SPEC§7")
@pytest.mark.parametrize(
    "spoil",
    [
        _missing_playbook,
        _two_labels_one_file,
        _label_missing_from_map,
        _another_path,
        _unknown_label_in_map,
        _misfiled_playbook,
        _no_front_matter,
        _list_front_matter,
        _schema_failure,
        _manifest_unknown_key,
        _manifest_bad_release,
        _not_utf8,
    ],
    ids=lambda f: f.__name__.strip("_"),
)
def test_the_loader_refuses_a_broken_release(tmp_path: Path, spoil: object) -> None:
    from quant.decide.playbook import load_release

    root = copy_release(tmp_path / "r1")
    load_release(root)  # the copy itself loads
    spoil(root)  # type: ignore[operator]
    with pytest.raises(ConfigError):
        load_release(root)


@pytest.mark.spec("ARCH§9.2")
def test_the_loader_refuses_a_missing_manifest(tmp_path: Path) -> None:
    from quant.decide.playbook import load_release

    root = copy_release(tmp_path / "r1")
    (root / "strategy.md").unlink()
    with pytest.raises(ConfigError):
        load_release(root)
