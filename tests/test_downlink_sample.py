"""The sample downlink package: log first, checksummed, and honest about gaps."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import make_downlink_sample as sample  # noqa: E402

from parikshak.engine.logger import verify_log_file  # noqa: E402


def test_the_package_sends_the_log_first_and_verifies(tmp_path):
    manifest = sample.build(tmp_path, eval_report=None)

    kinds = [a["kind"] for a in manifest["artifacts"]]
    assert kinds[0] in ("run_log", "run_log_text")
    assert manifest["artifacts"][0]["priority"] == 0
    assert manifest["_problems"] == []
    log = next(tmp_path.glob("*.log.jsonl"))
    assert verify_log_file(log)[0]
    assert manifest["log_vs_video"]["ratio"] > 1000
    assert manifest["not_included"], "missing evidence must be stated, not implied"
    assert (tmp_path / "README.txt").exists()
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["run_id"]
