"""The deployed build manifest: what "zero retraining" is checked against.

Before the manifest, a procedure's `requires.detector_classes` was checked only
against its own entity list. A procedure could name a class the detector has
never seen, list it under `requires`, and pass - the check was self-referential.

Each test below is one change that LOOKS like an afternoon's authoring and is
actually a data campaign. The point is that the tooling says so at load, before
anyone has promised a date.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from parikshak.pdl import ProcedureError, load_procedure
from parikshak.pdl.build import BUILDS_DIR, BuildError, PerceptionBuild, check_build, onboard

ROOT = Path(__file__).resolve().parent.parent
BUILD_PATH = BUILDS_DIR / "parikshak-perception-1.2.0.json"
PROCEDURES = sorted(p for p in (ROOT / "procedures").glob("*.yaml") if not p.name.startswith("_"))
CSP1 = ROOT / "procedures" / "csp1_colloid_sample_processing.yaml"


class Collect:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def err(self, where, msg):
        self.errors.append(f"{where}: {msg}")

    def warn(self, where, msg):
        self.warnings.append(f"{where}: {msg}")


@pytest.fixture(scope="module")
def build() -> PerceptionBuild:
    return PerceptionBuild.load(BUILD_PATH)


def doc_of(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def errors(doc: dict, build: PerceptionBuild) -> str:
    rep = Collect()
    check_build(doc, build, rep)
    return " | ".join(rep.errors)


# -- the manifest ----------------------------------------------------------
def test_the_build_every_procedure_names_exists():
    for path in PROCEDURES:
        target = doc_of(path)["requires"]["min_model_version"]
        assert (BUILDS_DIR / f"{target}.json").exists(), f"{path.name} targets missing {target}"


def test_a_build_can_be_found_by_name(build):
    assert PerceptionBuild.named("parikshak-perception-1.2.0").name == build.name


def test_state_heads_are_part_of_the_build(build):
    assert "closed" in build.state_heads["glovebox_latch"]
    assert "vial" not in build.state_heads, "vials have no state classifier in this build"


def test_the_build_state_heads_agree_with_csp1(build):
    """The manifest was derived from what CSP-1 was trained against. If the two
    disagree, one of them is describing a model that does not exist."""
    for name, entity in doc_of(CSP1)["entities"].items():
        if entity.get("states"):
            assert tuple(entity["states"]) == build.state_heads[entity["detector_class"]], name


@pytest.mark.parametrize("bad,match", [
    ('{"build": "x"', "not valid JSON"),
    ('{"build": "x"}', "missing"),
])
def test_malformed_manifests_are_refused(tmp_path, bad, match):
    p = tmp_path / "b.json"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(BuildError, match=match):
        PerceptionBuild.load(p)


def test_a_missing_manifest_is_refused(tmp_path):
    with pytest.raises(BuildError, match="no build manifest"):
        PerceptionBuild.load(tmp_path / "nope.json")


# -- every shipped procedure fits ------------------------------------------
@pytest.mark.parametrize("path", PROCEDURES, ids=lambda p: p.stem)
def test_every_shipped_procedure_fits_the_deployed_build(path, build):
    doc = doc_of(path)
    assert errors(doc, build) == ""
    assert not onboard(doc, build).retraining_required


@pytest.mark.parametrize("path", PROCEDURES, ids=lambda p: p.stem)
def test_every_shipped_procedure_loads_through_the_build_gate(path, build):
    load_procedure(path, build=build)


# -- the things that look like authoring and are not -----------------------
def test_a_new_detector_class_needs_a_new_dataset(build):
    doc = doc_of(CSP1)
    doc["requires"]["detector_classes"].append("torque_wrench")
    assert "NEW detection dataset" in errors(doc, build)


def test_a_new_state_for_a_known_class_needs_a_retrained_head(build):
    """The easy one to miss. `state_is(latch, half_open)` reads like a procedure
    edit, and it is a retrained state classifier."""
    doc = doc_of(CSP1)
    doc["entities"]["latch"]["states"].append("half_open")
    assert "retrained state head" in errors(doc, build)


def test_states_on_a_class_with_no_state_head_are_refused(build):
    doc = doc_of(CSP1)
    doc["entities"]["vial_a"]["states"] = ["full", "empty", "unknown"]
    assert "no state classifier" in errors(doc, build)


def test_a_new_motion_class_needs_a_retrained_tcn(build):
    doc = doc_of(CSP1)
    doc["motion_classes"]["classes"].append("unscrew")
    assert "retrained TCN" in errors(doc, build)


def test_a_new_capability_is_a_new_perception_component(build):
    doc = doc_of(CSP1)
    doc["requires"]["capabilities"].append("thermal_imaging")
    assert "new perception component" in errors(doc, build)


def test_targeting_a_different_build_is_refused(build):
    doc = doc_of(CSP1)
    doc["requires"]["min_model_version"] = "parikshak-perception-2.0.0"
    assert "deployed build is" in errors(doc, build)


def test_the_onboarding_report_names_what_is_missing(build):
    doc = doc_of(CSP1)
    doc["requires"]["detector_classes"].append("torque_wrench")
    report = onboard(doc, build)
    assert report.retraining_required
    classes = next(r for r in report.rows if r[0] == "detector classes")
    assert not classes[2] and "torque_wrench" in classes[1]


def test_the_loader_refuses_a_procedure_that_needs_retraining(tmp_path, build):
    doc = doc_of(CSP1)
    doc["requires"]["detector_classes"].append("torque_wrench")
    doc["entities"]["wrench"] = {"detector_class": "torque_wrench", "label": "Wrench"}
    p = tmp_path / "p.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    with pytest.raises(ProcedureError, match="NEW detection dataset"):
        load_procedure(p, build=build)


# -- the command-line gate -------------------------------------------------
def test_validator_cli_checks_against_the_build(tmp_path):
    ok = subprocess.run(
        [sys.executable, "tools/validate_procedure.py", str(CSP1), "--strict",
         "--build", str(BUILD_PATH)], cwd=ROOT, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout[-800:]

    doc = copy.deepcopy(doc_of(CSP1))
    doc["entities"]["latch"]["states"].append("half_open")
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    bad = subprocess.run(
        [sys.executable, "tools/validate_procedure.py", str(bad_path), "--strict",
         "--build", str(BUILD_PATH)], cwd=ROOT, capture_output=True, text=True)
    assert bad.returncode == 1
    assert "retrained state head" in bad.stdout


def test_validator_cli_refuses_a_missing_manifest(tmp_path):
    r = subprocess.run(
        [sys.executable, "tools/validate_procedure.py", str(CSP1),
         "--build", str(tmp_path / "missing.json")], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2
