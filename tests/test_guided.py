"""The guided run: the engine walking someone through an experiment.

The page chooses only what the crew does. Every verdict here - the tick, the
skip, the wrong object, the "cannot verify" - is the engine's own, reached from
belief frames it has never seen before. These tests hold that boundary: a
correct run is silent, a skipped step is caught without anyone telling the
engine it was skipped, and a step done out of sight is not called a mistake.
"""

from __future__ import annotations

import pytest

from demo import guided


def drive(run, choose=lambda sid: "do", limit: int = 25) -> list[str]:
    """Act on every step until the run has no active step left."""
    acted = []
    while run.active and len(acted) < limit:
        sid = run.active
        acted.append(sid)
        run.perform(choose(sid))
    return acted


def test_both_experiments_are_offered_with_their_own_steps():
    keys = {e["key"] for e in guided.experiments()}
    assert keys == {"csp1", "crx2"}
    _sid, run = guided.start("crx2")
    assert run.static()["procedure"]["id"] == "CRX-2"
    assert len(run.static()["procedure"]["steps"]) == 8
    with pytest.raises(KeyError):
        guided.start("nope")


def test_a_correct_run_is_completed_and_silent():
    _sid, run = guided.start("crx2")
    acted = drive(run)
    summary = run.finish()
    assert acted == ["S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08"]
    assert summary["complete"] == 8, summary["status"]
    assert summary["alerts"] == [], summary["alerts"]
    assert summary["log"]["verified"] and summary["log"]["records"] > 0


def test_a_skipped_step_is_caught_with_its_reason():
    _sid, run = guided.start("csp1")
    drive(run, lambda sid: "skip" if sid == "S08" else "do")
    summary = run.finish()
    assert "S08" in summary["skipped"], summary["status"]
    skips = [a for a in summary["alerts"] if a["kind"] == "SKIP" and a["step"] == "S08"]
    assert skips and skips[0]["reason"], summary["alerts"]


def test_taking_the_confusable_object_is_reported():
    _sid, run = guided.start("crx2")
    drive(run, lambda sid: "wrong" if sid == "S02" else "do")
    summary = run.finish()
    assert [a for a in summary["alerts"] if a["kind"] == "WRONG_OBJECT"], summary["alerts"]


def test_a_step_done_out_of_sight_is_not_called_a_mistake():
    _sid, run = guided.start("crx2")
    drive(run, lambda sid: "blocked" if sid == "S05" else "do")
    summary = run.finish()
    assert not [a for a in summary["alerts"] if a["kind"] == "SKIP"], summary["alerts"]
    assert summary["status"]["S05"] in ("COMPLETE", "UNVERIFIED"), summary["status"]


def test_the_page_is_offered_the_actions_that_step_allows():
    _sid, run = guided.start("csp1")
    ids = {a.id for a in run.actions()}
    assert {"do", "skip", "blocked", "wait"} <= ids
    assert "wrong" not in ids, "S01 has no confusable object"
    while run.active != "S03":
        run.perform("do")
    assert "wrong" in {a.id for a in run.actions()}
    with pytest.raises(ValueError):
        run.perform("hazard")          # not offered at S03


def test_a_finished_run_refuses_further_actions_and_keeps_its_summary():
    _sid, run = guided.start("crx2")
    drive(run)
    first = run.finish()
    assert run.finish() == first
    assert run.view()["finished"] and run.view()["actions"] == []
    with pytest.raises(ValueError):
        run.perform("do")


def test_the_server_runs_one_from_start_to_finish():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from demo.server import create_app

    client = TestClient(create_app())
    assert client.get("/api/guided/experiments").json()["experiments"]
    started = client.post("/api/guided/start", json={"experiment": "crx2"}).json()
    session = started["session"]
    assert started["static"]["procedure"]["id"] == "CRX-2"
    assert started["state"]["actions"]

    state = started["state"]
    while state["active"] and not state["finished"]:
        state = client.post("/api/guided/act",
                            json={"session": session, "action": "do"}).json()["state"]
    assert state["summary"] is None or state["summary"]["alerts"] == []
    assert client.post("/api/guided/start", json={"experiment": "nope"}).status_code == 404
    assert client.post("/api/guided/act",
                       json={"session": "gone", "action": "do"}).status_code == 410
