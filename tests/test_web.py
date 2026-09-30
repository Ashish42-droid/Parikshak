"""The browser demo shows real engine output, and says so when it stops matching.

Every scenario is a replay through the engine. These tests hold the demo to
that: the alert shown for the skipped latch is the engine's own, with its
reason; the log shown is verified by the real chain check; a correct run under
noise shows no alert; and the served and embedded pages both work.
"""

from __future__ import annotations

import pytest

from demo import scenarios as S
from demo.server import create_app, page


def test_every_scenario_points_at_files_that_exist():
    for sc in S.SCENARIOS:
        assert (S.ROOT / sc.procedure).exists(), sc.id
        assert (S.ROOT / sc.trace).exists(), sc.id
    assert S.DEFAULT in {s.id for s in S.SCENARIOS}
    assert {s.group for s in S.SCENARIOS} <= set(S.GROUP_ORDER)
    assert len({s.id for s in S.SCENARIOS}) == len(S.SCENARIOS)


def test_the_skipped_latch_carries_the_engines_alert_its_reason_and_a_verified_log():
    run = S.replay("csp1-skip-latch")
    assert run["expectation"]["met"], run["expectation"]["problems"]
    alerts = [e for e in run["events"] if e["type"] == "alert"]
    assert any(a["step"] == "S08" and a["kind"] == "SKIP" and a["reason"] for a in alerts), alerts
    assert len(run["frames"]) == run["summary"]["frames"]

    s08 = [s["id"] for s in run["procedure"]["steps"]].index("S08")
    assert run["status_codes"][run["frames"][-1]["st"][s08]] == "SKIPPED"
    assert run["log"]["verified"] and run["log"]["records"] > 0
    assert [e["t"] for e in run["events"]] == sorted(e["t"] for e in run["events"])


def test_a_correct_run_under_heavy_noise_shows_no_alert():
    run = S.replay("csp1-nominal-noisy")
    assert not [e for e in run["events"] if e["type"] == "alert"]


def test_the_server_serves_the_page_and_every_endpoint():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    home = client.get("/")
    assert home.status_code == 200 and "PARIKSHAK" in home.text and "<!doctype html>" in home.text
    cat = client.get("/api/scenarios").json()
    assert cat["default"] == S.DEFAULT and len(cat["scenarios"]) == len(S.SCENARIOS)
    assert client.get("/api/scenario/no-such-run").status_code == 404
    assert client.get("/api/scenario/crx2-wrong-vial").json()["expectation"]["met"]
    assert set(client.get("/api/results").json()) == {"csp1", "crx2", "bench", "soak"}


def test_an_embedded_page_needs_no_server_and_is_publishable():
    bundle = {"catalogue": {"default": "x", "groups": [], "scenarios": []},
              "data": {"x": {"note": "</script> must not end the tag"}}, "results": {}}
    bare = page(bundle, standalone=False)
    assert "window.PARIKSHAK_BUNDLE=" in bare
    assert page(bundle).startswith("<!doctype html>")


def test_recording_endpoints():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    fake_video = b"FAKEMEDIA_WEBM_VIDEO_DATA_FOR_PARIKSHAK_TEST"
    res = client.post(
        "/api/tracker/save_recording",
        files={"file": ("test_run_exp.webm", fake_video, "video/webm")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "saved"
    assert "test_run_exp.webm" in data["filename"]
    assert data["size_bytes"] == len(fake_video)
    assert data["url"].startswith("/recordings/")

    recs = client.get("/api/tracker/recordings").json()
    assert "recordings" in recs
    assert recs["count"] >= 1
    assert any("test_run_exp.webm" in r["name"] for r in recs["recordings"])


def test_report_endpoints():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    gen_res = client.post("/api/tracker/generate_report")
    assert gen_res.status_code == 200
    gen_data = gen_res.json()
    assert gen_data["status"] == "generated"
    assert "text_file" in gen_data
    assert "pdf_file" in gen_data
    assert "preview_text" in gen_data
    assert "PARIKSHAK" in gen_data["preview_text"]

    txt_res = client.get("/api/tracker/download_report/text")
    assert txt_res.status_code == 200
    assert "PARIKSHAK ON-BOARD MISSION PROCEDURE WITNESS REPORT" in txt_res.text
    assert "1. PROCEDURE STEP EXECUTION AUDIT CHRONOLOGY" in txt_res.text

    pdf_res = client.get("/api/tracker/download_report/pdf")
    assert pdf_res.status_code == 200
    assert pdf_res.headers["content-type"] == "application/pdf"
    assert pdf_res.content.startswith(b"%PDF")

    rep_list = client.get("/api/tracker/reports").json()
    assert "reports" in rep_list
    assert rep_list["count"] >= 1


