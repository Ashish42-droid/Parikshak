#!/usr/bin/env python3
"""Run every gate. This is the CI job, minus the repo.

    python tools/check.py

When the repo exists, the GitHub Actions step is `python tools/check.py` and
nothing else changes. Keeping the gate runnable locally is what stops it from
rotting between pushes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


class Runner:
    def __init__(self) -> None:
        self.failed: list[str] = []

    def step(self, name: str, argv: list[str]) -> None:
        print(f"\n=== {name} " + "=" * max(0, 60 - len(name)))
        rc = subprocess.run(argv, cwd=ROOT).returncode
        if rc != 0:
            self.failed.append(name)
            print(f"--- FAIL ({name}), exit {rc}")


def schema_check() -> int:
    """Structural validation of every procedure against the JSON Schema."""
    import yaml
    from jsonschema import Draft202012Validator

    schema = json.loads((ROOT / "schema" / "procedure.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    total = 0
    for path in sorted((ROOT / "procedures").glob("*.yaml")):
        if path.name.startswith("_"):
            continue  # the template fails by design; see PLAN.md section 20
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        errors = list(validator.iter_errors(doc))
        print(f"  {path.name:<46} {len(errors)} structural errors")
        total += len(errors)
    return 1 if total else 0


def main() -> int:
    if sys.argv[1:2] == ["--schema-only"]:
        return schema_check()

    import yaml

    r = Runner()
    procedures = [p for p in sorted((ROOT / "procedures").glob("*.yaml"))
                  if not p.name.startswith("_")]
    for path in procedures:
        # Against the build the procedure names, not just against itself. A
        # procedure's own requires list is self-referential; the build manifest
        # is what one set of trained weights actually provides.
        target = yaml.safe_load(path.read_text(encoding="utf-8"))["requires"]["min_model_version"]
        r.step(f"validate {path.name} against {target}",
               [PY, "tools/validate_procedure.py", str(path.relative_to(ROOT)), "--strict",
                "--build", str(Path("builds") / f"{target}.json")])
    # The zero-retraining verdict, for every procedure. If a procedure edit ever
    # starts needing a data campaign, this is the step that says so.
    r.step("onboard: zero retraining",
           [PY, "tools/onboard.py", *[str(p.relative_to(ROOT)) for p in procedures]])
    r.step("validator negative suite", [PY, "tools/test_validator.py"])
    r.step("json schema", [PY, "tools/check.py", "--schema-only"])
    r.step("pytest", [PY, "-m", "pytest"])
    # The golden corpus replayed through the real engine, checked against the
    # expectations that travel inside each trace. This is the regression suite
    # that every recorded run will join.
    traces = sorted((ROOT / "traces" / "golden").glob("*.jsonl"))
    if traces:
        r.step("replay golden corpus",
               [PY, "-m", "parikshak.replay", *[str(t) for t in traces], "--check"])
        # Metrics over the committed corpus. The full 130-trace degraded corpus
        # is ~300 MB and takes minutes, so it is a deliberate command
        # (tools/eval_report.py) rather than part of every gate run.
        r.step("eval report (golden)",
               [PY, "tools/eval_report.py", "--traces", "traces/golden"])
        # The operator display and the voice path, rendered from a real run.
        # Cheap, and it means a change that breaks the screen fails the gate
        # rather than surfacing during a rehearsal.
        r.step("operator display + voice",
               [PY, "-m", "parikshak.replay", "traces/golden/skip_S08_latch.jsonl",
                "--view", "--speak"])

    # The second procedure through the identical engine. PLAN.md descope item 6:
    # never cut this - it is the claim that a new experiment is a file.
    crx2 = sorted((ROOT / "traces" / "golden_crx2").glob("*.jsonl"))
    if crx2:
        r.step("replay CRX-2 golden corpus",
               [PY, "-m", "parikshak.replay", *[str(t) for t in crx2], "--check",
                "--procedure", "procedures/crx2_colloid_resuspension.yaml"])

    print("\n" + "=" * 66)
    if r.failed:
        print("FAILED: " + ", ".join(r.failed))
        return 1
    print("ALL GATES GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
