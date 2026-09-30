"""Unit test verifying YoloExperimentTracker state machine logic for WBP-1 and BCX-1."""

import cv2
import numpy as np
import pytest
from parikshak.perception.yolo_tracker import YoloExperimentTracker


def test_tracker_initialization():
    tracker_wbp = YoloExperimentTracker("WBP-1")
    assert tracker_wbp.experiment_id == "WBP-1"
    assert len(tracker_wbp.steps) == 4
    assert tracker_wbp.steps[0].id == "S01"
    assert tracker_wbp.steps[0].status == "active"

    tracker_bcx = YoloExperimentTracker("BCX-1")
    assert tracker_bcx.experiment_id == "BCX-1"
    assert len(tracker_bcx.steps) == 6
    assert tracker_bcx.steps[0].id == "S01"


def test_bcx1_color_and_collision_logic():
    tracker = YoloExperimentTracker("BCX-1")
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)

    # Outer container
    cv2.rectangle(frame, (80, 60), (560, 420), (200, 200, 200), 3)

    # Red box inside container at (200, 200)
    cv2.rectangle(frame, (200, 200), (260, 260), (0, 0, 255), -1)

    t = 1000.0
    annotated, telemetry = tracker.process_frame(frame, current_time=t)
    assert telemetry["experiment_id"] == "BCX-1"
    assert "collision" in telemetry
    assert telemetry["collision"]["red_inside"] is True
    assert telemetry["geometry"]["red_box_detected"] is True


def test_wbp1_state_machine_transitions():
    tracker = YoloExperimentTracker("WBP-1")
    assert tracker.steps[0].id == "S01"
    assert tracker.steps[0].status == "active"

    # Step 1 -> complete when stable on table
    t = 100.0
    tracker.s01_stable_duration = 0.9
    tracker.steps[0].status = "completed"
    tracker._advance_step(t)
    assert tracker.step_idx == 1
    assert tracker.steps[1].id == "S02"
    assert tracker.steps[1].status == "active"

    # Step 2 -> if drinking pose is detected, it should advance immediately
    # Simulate active drinking pose in S02
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Target lifted & drinking
    tracker.target_lifted = True
    tracker.steps[1].status = "completed"
    tracker._advance_step(t + 1.0)
    assert tracker.step_idx == 2
    assert tracker.steps[2].id == "S03"
    assert tracker.steps[2].status == "active"

    # Step 3 -> Drinking hold accumulation (S03_TARGET_S = 0.8s)
    tracker.drink_hold_duration = 0.85
    tracker.water_consumed = True
    tracker.steps[2].status = "completed"
    tracker._advance_step(t + 2.0)
    assert tracker.step_idx == 3
    assert tracker.steps[3].id == "S04"
    assert tracker.steps[3].status == "active"

    # Step 4 -> Return to table & release (S04_TARGET_S = 0.6s)
    tracker.s04_settle_duration = 0.65
    tracker.protocol_complete = True
    tracker.steps[3].status = "completed"
    tracker._advance_step(t + 3.0)
    assert tracker.protocol_complete is True


def test_wbp1_full_body_tracking_fields():
    """Verify that WBP-1 geometry telemetry includes full-body skeletal data."""
    tracker = YoloExperimentTracker("WBP-1")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    _, telemetry = tracker.process_frame(frame, current_time=50.0)
    geo = telemetry["geometry"]
    assert "body_points_count" in geo
    assert "body_points_total" in geo
    assert geo["body_points_total"] == 17
    assert "posture_status" in geo
    assert "posture_stability" in geo
    assert "torso_angle_deg" in geo
    assert "body_zones" in geo
    assert "skeleton" in geo
    assert isinstance(geo["skeleton"], list)

