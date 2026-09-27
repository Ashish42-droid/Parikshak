"""A local web server for the demo.

    python -m demo [--port 8765] [--no-browser]

Runs offline: every scenario is replayed by the engine on this machine the first
time it is opened. The page is the same file `tools/make_demo_page.py` embeds
the replays into for sharing - with no embedded data it asks this server.
"""

from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from pathlib import Path
from typing import Any
import numpy as np

from demo import guided
from demo.scenarios import DEFAULT, GROUP_ORDER, catalogue, replay_json, results

STATIC = Path(__file__).with_name("static")
PAGE = STATIC / "index.html"

#: The page is authored without <html>/<head>/<body> so the same file can be
#: published as it is; served locally it is wrapped here.
_SHELL_HEAD = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
               '<meta name="viewport" content="width=device-width, initial-scale=1, '
               'viewport-fit=cover"></head><body>')
_SHELL_TAIL = "</body></html>"


def page(bundle: dict[str, Any] | None = None, *, standalone: bool = True) -> str:
    """The demo page, optionally with every replay embedded.

    `standalone` wraps it in a full HTML document, for serving or opening from
    disk; without it the bare page is returned, ready to publish.
    """
    body = PAGE.read_text(encoding="utf-8")
    if bundle is not None:
        data = json.dumps(bundle, separators=(",", ":")).replace("</", "<\\/")
        body = body.replace("<!--BUNDLE-->", f"<script>window.PARIKSHAK_BUNDLE={data};</script>")
    return f"{_SHELL_HEAD}{body}{_SHELL_TAIL}" if standalone else body


def create_app():
    from fastapi import FastAPI, HTTPException, UploadFile, File
    from fastapi.responses import HTMLResponse, Response
    from parikshak.perception.tracker_service import get_tracker_service
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    from pathlib import Path

    app = FastAPI(title="PARIKSHAK demo", docs_url=None, redoc_url=None)
    tracker_svc = get_tracker_service()

    static_p = Path("demo/static")
    if static_p.exists():
        app.mount("/static", StaticFiles(directory=str(static_p)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return page()

    @app.get("/presentation", response_class=HTMLResponse)
    def presentation_page():
        p = Path("demo/static/sih_presentation.html")
        if p.exists():
            return HTMLResponse(p.read_text(encoding="utf-8"))
        raise HTTPException(404, "Presentation page not found")

    @app.get("/api/download_pptx")
    def download_pptx():
        p = Path("PARIKSHAK_SIH2026_Submission.pptx")
        if p.exists():
            return FileResponse(
                path=str(p),
                filename="PARIKSHAK_SIH2026_Submission.pptx",
                media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            )
        raise HTTPException(404, "PowerPoint deck not found")

    @app.get("/api/scenarios")
    def scenarios() -> dict[str, Any]:
        return {"default": DEFAULT, "groups": list(GROUP_ORDER), "scenarios": catalogue()}

    @app.get("/api/scenario/{scenario_id}")
    def scenario(scenario_id: str) -> Response:
        try:
            return Response(replay_json(scenario_id), media_type="application/json")
        except KeyError:
            raise HTTPException(404, f"no scenario {scenario_id!r}") from None

    @app.get("/api/results")
    def measured() -> dict[str, Any]:
        return results()

    # -- guided runs: the engine walking someone through an experiment ----
    @app.get("/api/guided/experiments")
    def guided_experiments() -> dict[str, Any]:
        return {"experiments": guided.experiments()}

    @app.post("/api/guided/start")
    def guided_start(payload: dict) -> dict[str, Any]:
        try:
            session_id, run = guided.start(str(payload.get("experiment", "")))
        except KeyError as exc:
            raise HTTPException(404, f"no experiment {exc}") from None
        return {"session": session_id, "static": run.static(), "state": run.view()}

    @app.post("/api/guided/act")
    def guided_act(payload: dict) -> dict[str, Any]:
        try:
            run = guided.get(str(payload.get("session", "")))
        except KeyError:
            raise HTTPException(410, "this run is no longer open - start a new one") from None
        try:
            run.perform(str(payload.get("action", "")))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return {"state": run.view()}

    # -- Live YOLO and OpenCV Human Activity Tracker endpoints -----------
    @app.get("/api/tracker/experiments")
    def tracker_experiments() -> dict[str, Any]:
        return {
            "experiments": [
                {
                    "id": "WBP-1",
                    "title": "Water Bottle Protocol (Activity Benchmark)",
                    "category": "Interactive Benchmark & Prototype",
                    "rack": "BENCH-1 (Desktop / Tabletop)",
                    "steps_count": 4,
                    "target_object": "Water Bottle",
                    "description": "Validates 4 steps: bottle identified -> hand grasps and lifts -> drinks water at mouth (held >= 1.5s) -> bottle returned to table surface and released.",
                },
                {
                    "id": "CRX-2",
                    "title": "CRX-2 : Colloid Resuspension and Cold Return",
                    "category": "ISRO Space Experiment (Microgravity)",
                    "rack": "MSG-A (Microgravity Science Glovebox)",
                    "steps_count": 8,
                    "target_object": "Colloid Vial B",
                    "description": "Full space flight procedure: foot restraint lock -> cold locker retrieval -> processing unit check -> 10x resuspension agitation -> tray settling -> restowage.",
                },
                {
                    "id": "CSP-1",
                    "title": "CSP-1 : Colloid Sample Processing",
                    "category": "ISRO Space Experiment (Microgravity)",
                    "rack": "MSG-A (Microgravity Science Glovebox)",
                    "steps_count": 14,
                    "target_object": "Sample Cartridge & Vial",
                    "description": "Complete 14-step microgravity colloid sample processing protocol with glovebox latch and cartridge lock verification.",
                },
            ]
        }

    @app.post("/api/tracker/set_experiment")
    def tracker_set_experiment(payload: dict) -> dict[str, Any]:
        exp_id = str(payload.get("experiment_id", "WBP-1"))
        return tracker_svc.set_experiment(exp_id)

    @app.post("/api/tracker/reset")
    def tracker_reset() -> dict[str, Any]:
        return tracker_svc.reset()

    @app.post("/api/tracker/frame")
    def tracker_process_frame(payload: dict) -> dict[str, Any]:
        frame_data = str(payload.get("frame", ""))
        if not frame_data:
            raise HTTPException(400, "Missing frame data")
        res = tracker_svc.process_b64_frame(frame_data)
        if "error" in res:
            raise HTTPException(400, res["error"])
        return res

    @app.get("/api/tracker/telemetry")
    def tracker_telemetry() -> dict[str, Any]:
        with tracker_svc.lock:
            if not tracker_svc.last_telemetry:
                dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                _, telem = tracker_svc.tracker.process_frame(dummy)
                tracker_svc.last_telemetry = telem
            return tracker_svc.last_telemetry

    @app.post("/api/tracker/upload_video")
    async def tracker_upload_video(file: UploadFile = File(...)) -> dict[str, Any]:
        upload_path = Path("runs/uploads") / file.filename
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        content = await file.read()
        upload_path.write_bytes(content)
        return tracker_svc.load_video_file(upload_path)

    @app.post("/api/tracker/load_demo_video")
    def tracker_load_demo_video() -> dict[str, Any]:
        demo_path = Path("runs/uploads/demo_bottle_run.mp4")
        if not demo_path.exists():
            tracker_svc.generate_demo_video(str(demo_path))
        return tracker_svc.load_video_file(demo_path)

    @app.get("/api/tracker/next_video_frame")
    def tracker_next_video_frame() -> dict[str, Any]:
        res = tracker_svc.get_next_video_frame()
        if "error" in res:
            raise HTTPException(400, res["error"])
        return res

    @app.post("/api/tracker/simulate")
    def tracker_simulate(payload: dict) -> dict[str, Any]:
        event_name = str(payload.get("event", "nominal_step"))
        return tracker_svc.simulate_event(event_name)

    @app.get("/api/tracker/cameras")
    def tracker_cameras() -> dict[str, Any]:
        return {"cameras": tracker_svc.list_available_cameras()}

    @app.post("/api/tracker/start_camera")
    def tracker_start_camera(payload: dict | None = None) -> dict[str, Any]:
        raw_idx = (payload or {}).get("camera_index", -1)
        try:
            idx = int(raw_idx)
        except (ValueError, TypeError):
            idx = -1
        return tracker_svc.start_local_camera(idx)

    @app.post("/api/tracker/stop_camera")
    def tracker_stop_camera() -> dict[str, Any]:
        return tracker_svc.stop_local_camera()

    @app.get("/api/tracker/camera_frame")
    def tracker_camera_frame() -> dict[str, Any]:
        return tracker_svc.get_camera_frame_b64()

    @app.get("/api/tracker/camera_stream")
    def tracker_camera_stream():
        import time
        from fastapi.responses import StreamingResponse

        def stream_generator():
            while True:
                frame_bytes = tracker_svc.get_camera_frame_mjpeg()
                if frame_bytes is None:
                    time.sleep(0.04)
                    continue
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
                time.sleep(0.033)

        return StreamingResponse(stream_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

    return app




def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Open the PARIKSHAK demo in a browser.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = ap.parse_args(argv)

    import uvicorn

    url = f"http://127.0.0.1:{args.port}/"
    print(f"PARIKSHAK demo at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.2, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, log_level="warning")
    return 0
