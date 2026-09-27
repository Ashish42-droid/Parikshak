#!/usr/bin/env python3
"""Verify a run log's hash chain.

    python tools/verify_log.py runs/live/live-20260915-090700.log.jsonl
    python tools/verify_log.py runs/live/*.log.jsonl

Each record in a PARIKSHAK run log carries the SHA-256 of the record before it,
so editing, deleting or reordering any line breaks every link after it. This is
the command a reviewer runs on a downlinked log - on the ground, with none of
the engine present - and the one the demo runs on stage.

Exit code 0 when every chain is intact, 1 when any is broken, 2 when a file
cannot be read.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from parikshak.engine.logger import read_log, verify_log_file  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify run log hash chains.")
    ap.add_argument("logs", nargs="+", type=Path)
    args = ap.parse_args(argv)

    worst = 0
    for path in args.logs:
        try:
            records = read_log(path)
            ok, detail = verify_log_file(path)
        except (OSError, ValueError) as exc:
            print(f"UNREADABLE  {path}: {exc}")
            worst = max(worst, 2)
            continue
        size_kb = path.stat().st_size / 1024
        print(f"{'INTACT ' if ok else 'BROKEN '}  {path}  "
              f"({len(records)} records, {size_kb:.1f} KB) - {detail}")
        if not ok:
            worst = max(worst, 1)
    return worst


if __name__ == "__main__":
    sys.exit(main())
