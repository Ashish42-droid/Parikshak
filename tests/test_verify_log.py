"""The log verifier a reviewer runs on the ground.

A hash chain is only worth something if the command that checks it says INTACT
for an untouched log, BROKEN for an edited one - and says WHERE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import verify_log  # noqa: E402

from parikshak.belief.trace import iter_trace  # noqa: E402
from parikshak.engine.runner import ProcedureEngine  # noqa: E402
from parikshak.pdl import load_procedure  # noqa: E402


def logged_run(tmp_path: Path) -> Path:
    proc = load_procedure(ROOT / "procedures" / "csp1_colloid_sample_processing.yaml")
    log = tmp_path / "skip_S08.log.jsonl"
    engine = ProcedureEngine(proc, run_id="skip_S08", log_path=log)
    for frame in iter_trace(ROOT / "traces" / "golden" / "skip_S08_latch.jsonl"):
        engine.step(frame)
    engine.finish()
    return log


def test_an_untouched_log_verifies(tmp_path, capsys):
    log = logged_run(tmp_path)
    assert verify_log.main([str(log)]) == 0
    assert "INTACT" in capsys.readouterr().out


def test_an_edited_alert_breaks_the_chain_at_that_record(tmp_path, capsys):
    log = logged_run(tmp_path)
    lines = log.read_text(encoding="utf-8").splitlines()
    i = next(n for n, line in enumerate(lines) if json.loads(line)["kind"] == "alert")
    record = json.loads(lines[i])
    record["payload"]["text"] = "All good, nothing to see."
    lines[i] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert verify_log.main([str(log)]) == 1
    out = capsys.readouterr().out
    assert "BROKEN" in out and f"record {i}" in out


def test_an_unreadable_log_is_reported_not_crashed(tmp_path, capsys):
    bad = tmp_path / "garbage.log.jsonl"
    bad.write_text("not json\n", encoding="utf-8")
    assert verify_log.main([str(bad)]) == 2
    assert "UNREADABLE" in capsys.readouterr().out
