"""Unit tests verifying YoloExperimentTracker and TrackerService logic for MOA-1 Multi-Object Experiment."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from parikshak.perception.yolo_tracker import YoloExperimentTracker
from parikshak.perception.tracker_service import TrackerService
from parikshak.perception.report_generator import (
    generate_structured_text_report,
    generate_pdf_report,
)


def create_blank_frame(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestMoa1Tracker:
    def test_moa1_initialization_has_7_steps(self):
        tracker = YoloExperimentTracker("MOA-1")
        assert tracker.experiment_id == "MOA-1"
        assert len(tracker.steps) == 7

        expected_ids = ["S01", "S02", "S03", "S04", "S05", "S06", "S07"]
        assert [s.id for s in tracker.steps] == expected_ids

        assert tracker.steps[0].id == "S01"
        assert tracker.steps[0].status == "active"
        assert "Chair" in tracker.steps[0].name

        assert tracker.steps[1].id == "S02"
        assert "Sit" in tracker.steps[1].name

        assert tracker.steps[2].id == "S03"
        assert "Smartphone" in tracker.steps[2].name or "Phone" in tracker.steps[2].name

        assert tracker.steps[3].id == "S04"
        assert "Return" in tracker.steps[3].name or "Desk" in tracker.steps[3].name

        assert tracker.steps[4].id == "S05"
        assert "Bottle" in tracker.steps[4].name

        assert tracker.steps[5].id == "S06"
        assert "Drink" in tracker.steps[5].name

        assert tracker.steps[6].id == "S07"
        assert "Release" in tracker.steps[6].name or "Return" in tracker.steps[6].name

    def test_moa1_reset_restores_clean_state(self):
        tracker = YoloExperimentTracker("MOA-1")
        tracker.moa1_chair_pulled = True
        tracker.moa1_is_seated = True
        tracker.moa1_phone_picked = True
        tracker.moa1_phone_stowed = True
        tracker.moa1_bottle_lifted = True
        tracker.moa1_water_consumed = True
        tracker.moa1_bottle_returned = True
        tracker.protocol_complete = True

        tracker.reset()

        assert tracker.step_idx == 0
        assert tracker.moa1_chair_pulled is False
        assert tracker.moa1_is_seated is False
        assert tracker.moa1_phone_picked is False
        assert tracker.moa1_phone_stowed is False
        assert tracker.moa1_bottle_lifted is False
        assert tracker.moa1_water_consumed is False
        assert tracker.moa1_bottle_returned is False
        assert tracker.protocol_complete is False
        assert tracker.steps[0].status == "active"
        for s in tracker.steps[1:]:
            assert s.status == "pending"

    def test_moa1_process_dummy_frame_telemetry(self):
        tracker = YoloExperimentTracker("MOA-1")
        frame = create_blank_frame()
        t = 100.0

        annotated, telem = tracker.process_frame(frame, current_time=t)

        assert telem["experiment_id"] == "MOA-1"
        assert telem["active_step_id"] == "S01"
        assert len(telem["steps"]) == 7
        assert "chair" in telem
        assert "posture" in telem
        assert "phone" in telem
        assert "bottle" in telem
        assert "drinking" in telem
        assert "geometry" in telem

        # Check full body skeletal pose telemetry fields
        geo = telem["geometry"]
        assert "body_points_count" in geo
        assert "body_points_total" in geo
        assert geo["body_points_total"] == 17
        assert "skeleton" in geo

        # Annotated frame should have same dimensions
        assert annotated.shape == frame.shape

    def test_moa1_service_full_nominal_execution_simulation(self):
        service = TrackerService("MOA-1")
        assert service.experiment_id == "MOA-1"
        assert service.tracker.steps[0].id == "S01"

        # 1. Pull chair
        t1 = service.simulate_event("moa1_pull_chair")
        assert service.tracker.moa1_chair_pulled is True
        assert service.tracker.steps[0].status == "completed"
        assert service.tracker.steps[1].status == "active"
        assert t1["chair"]["pulled"] is True

        # 2. Sit down
        t2 = service.simulate_event("moa1_sit")
        assert service.tracker.moa1_is_seated is True
        assert service.tracker.steps[1].status == "completed"
        assert service.tracker.steps[2].status == "active"
        assert t2["posture"]["is_seated"] is True
        assert t2["posture"]["knee_angle_deg"] == 92.5

        # 3. Pick up phone
        t3 = service.simulate_event("moa1_phone_pickup")
        assert service.tracker.moa1_phone_picked is True
        assert service.tracker.steps[2].status == "completed"
        assert service.tracker.steps[3].status == "active"
        assert t3["phone"]["picked"] is True

        # 4. Stow phone
        t4 = service.simulate_event("moa1_phone_stow")
        assert service.tracker.moa1_phone_stowed is True
        assert service.tracker.steps[3].status == "completed"
        assert service.tracker.steps[4].status == "active"
        assert t4["phone"]["stowed"] is True

        # 5. Lift bottle
        t5 = service.simulate_event("moa1_bottle_lift")
        assert service.tracker.moa1_bottle_lifted is True
        assert service.tracker.steps[4].status == "completed"
        assert service.tracker.steps[5].status == "active"
        assert t5["bottle"]["lifted"] is True

        # 6. Drink water
        t6 = service.simulate_event("moa1_drink")
        assert service.tracker.moa1_water_consumed is True
        assert service.tracker.steps[5].status == "completed"
        assert service.tracker.steps[6].status == "active"
        assert t6["drinking"]["water_consumed"] is True
        assert t6["drinking"]["hold_duration_s"] >= 1.5

        # 7. Return bottle & release
        t7 = service.simulate_event("moa1_bottle_return")
        assert service.tracker.moa1_bottle_returned is True
        assert service.tracker.protocol_complete is True
        assert service.tracker.steps[6].status == "completed"
        assert t7["is_complete"] is True
        assert t7["compliance_score"] == 100

    def test_moa1_service_deviation_skip_drink(self):
        service = TrackerService("MOA-1")
        service.simulate_event("moa1_pull_chair")
        service.simulate_event("moa1_sit")
        service.simulate_event("moa1_phone_pickup")
        service.simulate_event("moa1_phone_stow")
        service.simulate_event("moa1_bottle_lift")

        # Simulate skipping drinking step
        service.simulate_event("moa1_skip_drink")
        assert service.tracker.steps[5].id == "S06"
        assert service.tracker.steps[5].status == "skipped"
        assert service.tracker.moa1_water_consumed is False

        # Alert should be present
        alerts = service.tracker.alerts
        assert len(alerts) > 0
        skip_alerts = [a for a in alerts if a.step_id == "S06" and a.kind == "skipped"]
        assert len(skip_alerts) >= 1
        assert "skipped" in skip_alerts[0].message.lower()

    def test_moa1_structured_text_report_generation(self):
        service = TrackerService("MOA-1")
        # Run through nominal steps
        for ev in ["moa1_pull_chair", "moa1_sit", "moa1_phone_pickup", "moa1_phone_stow",
                   "moa1_bottle_lift", "moa1_drink", "moa1_bottle_return"]:
            service.simulate_event(ev)

        report = generate_structured_text_report(service)

        assert "PARIKSHAK ON-BOARD MISSION PROCEDURE" in report
        assert "MOA-1" in report
        assert "Compliance Rating    : 100%" in report
        assert "[STEP 01] S01" in report
        assert "[STEP 02] S02" in report
        assert "[STEP 03] S03" in report
        assert "[STEP 04] S04" in report
        assert "[STEP 05] S05" in report
        assert "[STEP 06] S06" in report
        assert "[STEP 07] S07" in report
        assert "Cryptographic Hash (SHA-256):" in report

    def test_moa1_pdf_report_generation(self):
        service = TrackerService("MOA-1")
        for ev in ["moa1_pull_chair", "moa1_sit", "moa1_phone_pickup", "moa1_phone_stow",
                   "moa1_bottle_lift", "moa1_drink", "moa1_bottle_return"]:
            service.simulate_event(ev)

        pdf_bytes = generate_pdf_report(service)
        assert isinstance(pdf_bytes, (bytes, bytearray))
        assert len(pdf_bytes) > 500
        # Valid PDF file header
        assert pdf_bytes.startswith(b"%PDF")

    def test_moa1_voice_alert_on_skipped_step(self):
        service = TrackerService("MOA-1")
        # Trigger skip via generic inject_skip
        telem = service.simulate_event("inject_skip")
        assert telem["recent_alert"] is not None
        assert telem["recent_alert"]["kind"] == "skipped"
        assert telem["recent_alert"]["tts"] is not None
        assert len(telem["recent_alert"]["tts"]) > 0
        assert "skipped" in telem["recent_alert"]["tts"].lower() or "warning" in telem["recent_alert"]["tts"].lower()

    def test_moa1_voice_alert_on_out_of_order_step(self):
        service = TrackerService("MOA-1")
        # Trigger out of order via moa1_out_of_order
        telem = service.simulate_event("moa1_out_of_order")
        assert telem["recent_alert"] is not None
        assert telem["recent_alert"]["kind"] == "out_of_order"
        assert telem["recent_alert"]["tts"] is not None
        assert len(telem["recent_alert"]["tts"]) > 0
        assert "out of order" in telem["recent_alert"]["tts"].lower()

    def test_all_experiments_voice_alerts_skip_and_order(self):
        # BCX-1
        bcx_svc = TrackerService("BCX-1")
        bcx_order = bcx_svc.simulate_event("inject_out_of_order")
        assert bcx_order["recent_alert"]["kind"] == "out_of_order"
        assert "out of order" in bcx_order["recent_alert"]["tts"].lower()

        bcx_skip = bcx_svc.simulate_event("inject_skip")
        assert bcx_skip["recent_alert"]["kind"] == "skipped"
        assert "skipped" in bcx_skip["recent_alert"]["tts"].lower()

        # WBP-1
        wbp_svc = TrackerService("WBP-1")
        wbp_order = wbp_svc.simulate_event("inject_out_of_order")
        assert wbp_order["recent_alert"]["kind"] == "out_of_order"
        assert "out of order" in wbp_order["recent_alert"]["tts"].lower()

        wbp_skip = wbp_svc.simulate_event("inject_skip")
        assert wbp_skip["recent_alert"]["kind"] == "skipped"
        assert "skipped" in wbp_skip["recent_alert"]["tts"].lower()

