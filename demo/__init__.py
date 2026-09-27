"""The PARIKSHAK demo in a browser.

    python -m demo                          # opens http://127.0.0.1:8765
    python tools/make_demo_page.py          # one self-contained HTML file to share

Every run shown is a real replay through the same engine the live system and
the desktop window use. See `scenarios` for what is captured and `server` for
the web server.

**Why this lives outside `parikshak/`.** The flight package promises that no code
in it can open a socket, and tests/test_offline.py enforces that for every module
under `parikshak/`. A web server is a presentation tool for people on the ground,
not on-board software, so it sits here and imports the engine - the engine never
imports it.
"""
