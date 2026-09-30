import time
import cv2
import numpy as np
import pytest
from parikshak.perception.yolo_tracker import YoloExperimentTracker


def create_blank_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def draw_outer_container(frame, x1=80, y1=60, x2=560, y2=420):
    # White rectangular border representing container box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 4)


def draw_red_box(frame, x1=120, y1=150, x2=200, y2=230):
    # Pure bright red in BGR
    frame[y1:y2, x1:x2] = (20, 20, 240)


def draw_yellow_box(frame, x1=380, y1=150, x2=460, y2=230):
    # Pure bright yellow in BGR
    frame[y1:y2, x1:x2] = (20, 230, 235)


def feed_frames(tracker, frame, start_time, duration, step=0.2):
    t = start_time
    end = start_time + duration
    last_telem = None
    while t <= end:
        _, last_telem = tracker.process_frame(frame, current_time=t)
        t += step
    return t, last_telem


class TestBcx1CollisionTracker:
    def test_bcx1_init_has_6_steps(self):
        tracker = YoloExperimentTracker()
        tracker.init_procedure("BCX-1")
        assert len(tracker.steps) == 6
        step_ids = [s.id for s in tracker.steps]
        assert step_ids == ["S01", "S02", "S03", "S04", "S05", "S06"]
        assert tracker.steps[0].id == "S01"
        assert "Container" in tracker.steps[0].name
        assert tracker.steps[1].id == "S02"
        assert "Color" in tracker.steps[1].name

    def test_step1_container_locking_and_persistence(self):
        """Problem 1: Step 1 box should lock and not get detached."""
        tracker = YoloExperimentTracker()
        tracker.init_procedure("BCX-1")
        frame = create_blank_frame()
        draw_outer_container(frame, 90, 70, 550, 410)

        # Process frames over 1.0s to complete S01 and lock container
        t0 = 100.0
        t1, telem1 = feed_frames(tracker, frame, t0, 1.0, step=0.2)
        assert tracker.bcx1_container_locked is True
        assert tracker.steps[0].status == "completed"
        locked_coords = tracker.bcx1_locked_container
        assert locked_coords is not None
        assert abs(locked_coords[0] - 90) < 15

        # Frame with noisy occlusion - container should NOT get detached
        noisy_frame = create_blank_frame()
        _, telem2 = tracker.process_frame(noisy_frame, current_time=t1 + 0.1)
        assert tracker.bcx1_container_locked is True
        assert tracker.bcx1_locked_container == locked_coords

    def test_step2_color_verification(self):
        """Problem 3: Color detection should be an explicit step with high accuracy."""
        tracker = YoloExperimentTracker()
        tracker.init_procedure("BCX-1")
        frame = create_blank_frame()
        draw_outer_container(frame, 80, 60, 560, 420)

        # Advance through S01 (1.0s)
        t, _ = feed_frames(tracker, frame, 100.0, 1.0, step=0.2)
        assert tracker.steps[0].status == "completed"
        assert tracker.steps[1].status == "active"
        assert tracker.steps[1].id == "S02"

        # Present both Red and Yellow boxes
        draw_red_box(frame, 120, 150, 200, 230)
        draw_yellow_box(frame, 380, 150, 460, 230)

        # Advance through S02 (1.0s)
        t, _ = feed_frames(tracker, frame, t + 0.1, 1.0, step=0.2)
        assert tracker.bcx1_colors_verified is True
        assert tracker.steps[1].status == "completed"
        assert tracker.steps[2].status == "active"
        assert tracker.steps[2].id == "S03"

    def test_out_of_order_yellow_before_red_warning(self):
        """Problem 2: Warning given when user skips Red Box and places Yellow Box first."""
        tracker = YoloExperimentTracker()
        tracker.init_procedure("BCX-1")
        frame = create_blank_frame()
        draw_outer_container(frame, 80, 60, 560, 420)
        draw_red_box(frame, 120, 150, 200, 230)
        draw_yellow_box(frame, 380, 150, 460, 230)

        # Advance S01 and S02
        t, _ = feed_frames(tracker, frame, 100.0, 1.0, step=0.2)
        t, _ = feed_frames(tracker, frame, t + 0.1, 1.0, step=0.2)
        assert tracker.steps[2].status == "active" # S03

        # In S03, remove Red box and put Yellow box inside container
        f_yellow_only = create_blank_frame()
        draw_outer_container(f_yellow_only, 80, 60, 560, 420)
        tracker.bcx1_last_red_box = None
        tracker.bcx1_last_seen_red_time = 0.0
        draw_yellow_box(f_yellow_only, 250, 200, 330, 280) # inside container

        t, telem = feed_frames(tracker, f_yellow_only, t + 0.1, 0.4, step=0.2)
        assert telem["recent_alert"] is not None
        assert telem["recent_alert"]["kind"] == "out_of_order"
        assert "S03" in telem["recent_alert"]["step_id"]
        assert "Red Box first" in telem["recent_alert"]["message"]

    def test_step6_separation_detected(self):
        """Problem 4: Separating boxes from collision is properly detected and completes."""
        tracker = YoloExperimentTracker()
        tracker.init_procedure("BCX-1")
        frame = create_blank_frame()
        draw_outer_container(frame, 80, 60, 560, 420)

        # Fast forward to S05 (Collision)
        for s in tracker.steps[:4]:
            s.status = "completed"
        tracker.step_idx = 4
        tracker.steps[4].status = "active" # S05 Collide
        tracker.bcx1_container_locked = True
        tracker.bcx1_colors_verified = True
        tracker.bcx1_red_placed = True
        tracker.bcx1_yellow_placed = True

        # Collide boxes inside container
        f_col = create_blank_frame()
        draw_outer_container(f_col, 80, 60, 560, 420)
        draw_red_box(f_col, 250, 200, 310, 260)
        draw_yellow_box(f_col, 305, 200, 365, 260) # touching

        t, _ = feed_frames(tracker, f_col, 200.0, 1.0, step=0.2)
        assert tracker.steps[4].status == "completed" # S05 complete
        assert tracker.steps[5].status == "active"    # S06 Separate

        # Separate boxes inside container (distance >= 20px)
        f_sep = create_blank_frame()
        draw_outer_container(f_sep, 80, 60, 560, 420)
        draw_red_box(f_sep, 150, 200, 210, 260)
        draw_yellow_box(f_sep, 380, 200, 440, 260) # > 100px separation

        # Feed separation frames for 1.0s (target is 0.7s)
        t, telem_done = feed_frames(tracker, f_sep, t + 0.1, 1.0, step=0.2)
        assert tracker.steps[5].status == "completed"
        assert tracker.protocol_complete is True
        assert telem_done["is_complete"] is True

    def test_alert_expires_after_4_seconds(self):
        """Problem 5: Alerts should not stick forever in telemetry."""
        tracker = YoloExperimentTracker()
        tracker.init_procedure("BCX-1")
        frame = create_blank_frame()
        draw_outer_container(frame)

        t = 100.0
        tracker._trigger_alert("S01", "critical", "hazard", "Test Alert", "Test TTS", t)
        _, telem1 = tracker.process_frame(frame, current_time=t + 1.0)
        assert telem1["recent_alert"] is not None

        # After 4.5 seconds, recent_alert should be None
        _, telem2 = tracker.process_frame(frame, current_time=t + 4.5)
        assert telem2["recent_alert"] is None

    def test_full_body_tracking_telemetry_and_hud(self):
        """Verifies full 17-keypoint skeleton telemetry structure and HUD rendering."""
        tracker = YoloExperimentTracker("BCX-1")
        frame = create_blank_frame()
        draw_outer_container(frame)

        # 1. Verify telemetry fields on blank frame
        _, telem = tracker.process_frame(frame, current_time=100.0)
        assert "geometry" in telem
        geo = telem["geometry"]
        assert "body_points_total" in geo
        assert geo["body_points_total"] == 17
        assert "body_points_count" in geo
        assert "posture_status" in geo
        assert "posture_stability" in geo
        assert "torso_angle_deg" in geo
        assert "body_zones" in geo
        assert set(geo["body_zones"].keys()) == {"head", "torso", "upper_limbs", "lower_limbs"}

        # 2. Test HUD rendering with simulated 17-point astronaut skeleton
        mock_kpts = []
        from parikshak.perception.yolo_tracker import COCO_KEYPOINTS
        assert len(COCO_KEYPOINTS) == 17

        # Simulate astronaut in foot restraint (standing upright in frame)
        coords = [
            (320, 100), # nose
            (312, 92),  # left_eye
            (328, 92),  # right_eye
            (302, 96),  # left_ear
            (338, 96),  # right_ear
            (285, 140), # left_shoulder
            (355, 140), # right_shoulder
            (260, 200), # left_elbow
            (380, 200), # right_elbow
            (240, 260), # left_wrist
            (400, 260), # right_wrist
            (295, 270), # left_hip
            (345, 270), # right_hip
            (290, 360), # left_knee
            (350, 360), # right_knee
            (290, 440), # left_ankle (anchored in foot restraint)
            (350, 440), # right_ankle
        ]
        for idx, (kx, ky) in enumerate(coords):
            mock_kpts.append({
                "id": idx,
                "name": COCO_KEYPOINTS[idx],
                "x": float(kx),
                "y": float(ky),
                "conf": 0.95,
                "visible": True,
            })

        mock_skeleton = {
            "detected": True,
            "keypoints": mock_kpts,
            "visible_indices": list(range(17)),
            "body_points_count": 17,
            "body_points_total": 17,
            "zones": {"head": 5, "torso": 4, "upper_limbs": 4, "lower_limbs": 4},
            "wrists": [{"side": "left", "point": (240.0, 260.0), "conf": 0.95}, {"side": "right", "point": (400.0, 260.0), "conf": 0.95}],
            "mouth_region": (320.0, 128.0),
            "sh_mid": (320.0, 140.0),
            "hip_mid": (320.0, 270.0),
            "torso_angle_deg": 0.0,
            "arm_angles": {"left_elbow_deg": 135.0, "right_elbow_deg": 135.0},
            "leg_angles": {"left_knee_deg": 175.0, "right_knee_deg": 175.0},
            "posture_status": "FOOT_RESTRAINT_ANCHORED",
            "posture_stability": "STABLE",
            "centroid": (320.0, 220.0),
            "velocity_px_s": 1.2,
            "anchored": True,
        }

        # Render full-body cybernetic rig on test frame
        test_img = create_blank_frame()
        tracker._render_skeletal_rig(test_img, mock_skeleton)

        # Image should now contain cybernetic glowing lines and reticles (mean > 0)
        assert np.mean(test_img) > 1.0

