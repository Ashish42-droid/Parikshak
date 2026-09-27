#!/usr/bin/env python3
"""One self-contained HTML file with every demo run replayed and embedded.

    python tools/make_demo_page.py                 # runs/demo/parikshak_demo.html
    python tools/make_demo_page.py --only csp1-skip-latch crx2-nominal

Double-click the result to open it: no Python, no server and no network are
needed (fonts fall back to system ones offline). Every replay is run through the
engine when this tool runs, and the tool fails if any run no longer shows what
its title says. `runs/demo/page.html` is the same page without the document
wrapper, for publishing as a web page.

For the live version, which replays on demand: `python -m demo`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from demo.server import page  # noqa: E402
from demo.scenarios import (  # noqa: E402
    DEFAULT,
    GROUP_ORDER,
    SCENARIOS,
    catalogue,
    replay,
    results,
)


def bundle(only: set[str] | None = None, report=print) -> dict:
    data = {}
    for sc in SCENARIOS:
        if only and sc.id not in only:
            continue
        t0 = time.monotonic()
        run = replay(sc.id)
        data[sc.id] = run
        alerts = sum(e["type"] == "alert" for e in run["events"])
        verdict = ("as expected" if run["expectation"]["met"]
                   else "NOT AS EXPECTED: " + "; ".join(run["expectation"]["problems"]))
        report(f"  {sc.id:<20} {len(run['frames']):4d} frames, {alerts} alert(s), {verdict} "
               f"({time.monotonic() - t0:.1f} s)")
    if not data:
        raise SystemExit(f"no scenario matched {sorted(only or ())}")
    return {
        "catalogue": {"default": DEFAULT if DEFAULT in data else next(iter(data)),
                      "groups": list(GROUP_ORDER),
                      "scenarios": [s for s in catalogue() if s["id"] in data]},
        "data": data,
        "results": results(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the self-contained PARIKSHAK demo page.")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "demo")
    ap.add_argument("--only", nargs="*", default=None, help="scenario ids to include")
    args = ap.parse_args(argv)

    b = bundle(set(args.only) if args.only else None)
    args.out.mkdir(parents=True, exist_ok=True)
    full = args.out / "parikshak_demo.html"
    bare = args.out / "page.html"
    full.write_text(page(b), encoding="utf-8", newline="\n")
    bare.write_text(page(b, standalone=False), encoding="utf-8", newline="\n")
    print(f"\n  {full}  ({full.stat().st_size / 1e6:.1f} MB) - double-click to open")
    print(f"  {bare}  (same page, for publishing)")
    unmet = [k for k, v in b["data"].items() if not v["expectation"]["met"]]
    if unmet:
        print("  runs that no longer show what their title says: " + ", ".join(unmet))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
