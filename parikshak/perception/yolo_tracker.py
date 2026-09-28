"""YOLO and OpenCV-based human activity and procedure compliance tracker.

Supports both:
  1. The interactive benchmark: WBP-1 Water Bottle Protocol
  2. Space experiment tracking (vials, tools, latches, gloves, crew actions)

Tracks object bounding boxes, hand positions, mouth/face keypoints, spatial
contact, and sequence order to detect nominal execution and compliance deviations.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np


@dataclass
class StepState:
    id: str
    name: str
    prompt: str
    status: str = "pending"  # pending, active, completed, skipped, failed
    started_at: float | None = None
    completed_at: float | None = None
    hold_start: float | None = None
    elapsed_s: float = 0.0


@dataclass
class DeviationAlert:
    step_id: str
    severity: str  # advisory, caution, critical
    kind: str      # out_of_order, skipped, wrong_object, premature, timeout
    message: str
    timestamp: float
    spoken_tts: str


class YoloExperimentTracker:
    """End-to-end vision tracker and procedure validator using YOLO and OpenCV."""

    # Calibration and verification hold targets (in seconds)
    S01_TARGET_S = 1.0  # Stable on table
    S02_TARGET_S = 1.0  # Grasp and lift
    S03_TARGET_S = 1.4  # Drinking hold
    S04_TARGET_S = 1.0  # Return to table & release

    def __init__(self, experiment_id: str = "WBP-1") -> None:
        self.experiment_id = experiment_id.upper()
        self.load_models()
        self.init_procedure(self.experiment_id)
        self.reset()

    def load_models(self) -> None:
        from ultralytics import YOLO
        # Lightweight models running real-time on CPU
        self.det_model = YOLO("yolov8n.pt")
        self.pose_model = YOLO("yolov8n-pose.pt")
        # Warm up models on dummy tensor so first live camera frame has zero cold-start delay
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        self.det_model(dummy, imgsz=320, verbose=False)
        self.pose_model(dummy, imgsz=320, verbose=False)

    def init_procedure(self, experiment_id: str) -> None:
        if experiment_id == "WBP-1":
            self.title = "Water Bottle Protocol (Activity Benchmark)"
            self.rack_id = "BENCH-1"
            self.steps = [
                StepState("S01", "Locate water bottle on table",
                          "Ensure the water bottle is resting on the table surface."),
                StepState("S02", "Grasp and lift bottle",
                          "Grasp the water bottle and lift it off the table."),
                StepState("S03", "Drink water from bottle",
                          "Bring the bottle to your mouth and drink water."),
                StepState("S04", "Return bottle to table surface",
                          "Place the water bottle back onto the table surface and release it."),
            ]
            self.expected_target = "bottle"
            self.confusables = ["cup", "wine glass", "mug"]
        elif experiment_id == "CRX-2":
            self.title = "CRX-2 : Colloid Resuspension and Cold Return"
            self.rack_id = "MSG-A (Space Station Rack)"
            self.steps = [
                StepState("S01", "Secure in foot restraint",
                          "Secure yourself in the foot restraint."),
                StepState("S02", "Retrieve vial B from cold locker",
                          "Retrieve vial B, the orange banded vial, from cold locker L one."),
                StepState("S03", "Confirm processing unit idle",
                          "Check processing unit is idle and latch is closed."),
                StepState("S04", "Resuspend vial B by agitation",
                          "Agitate vial B ten times to resuspend the colloid."),
                StepState("S05", "Place vial B in tray T1",
                          "Place vial B in tray T one and let it settle."),
                StepState("S06", "Confirm uniform suspension",
                          "Inspect vial B and confirm suspension is uniform."),
                StepState("S07", "Return vial B to cold locker",
                          "Return vial B to cold locker L one."),
                StepState("S08", "Verify glovebox latch closed",
                          "Verify glovebox latch is closed and locked."),
            ]
            self.expected_target = "bottle"  # or vial
            self.confusables = ["cup"]
        else:
            # Generic / custom space procedure
            self.title = f"Procedure {experiment_id}"
            self.rack_id = "PAYLOAD-1"
            self.steps = [
                StepState("S01", "Identify payload sample", "Identify and inspect the payload sample."),
                StepState("S02", "Retrieve sample from rack", "Grasp and retrieve sample from rack."),
                StepState("S03", "Execute procedure protocol", "Perform protocol operation on sample."),
                StepState("S04", "Restow and secure", "Return sample and secure in restraint."),
            ]
            self.expected_target = "bottle"
            self.confusables = ["cup"]

    def reset(self) -> None:
        self.step_idx = 0
        for s in self.steps:
            s.status = "pending"
            s.started_at = None
            s.completed_at = None
            s.hold_start = None
            s.elapsed_s = 0.0

        if self.steps:
            self.steps[0].status = "active"
            self.steps[0].started_at = time.time()

        self.alerts: list[DeviationAlert] = []
        self.recent_alerts: list[DeviationAlert] = []
        self.frame_count = 0
        self.start_wall_time = time.time()
        self.last_nominal_seen = time.time()
        self.last_frame_time: float | None = None
        self.baseline_table_y: float | None = None
        self.initial_bottle_y: float | None = None

        # Step 1: Stability on table calibration (target: 1.0s)
        self.s01_stable_start: float | None = None
        self.s01_stable_duration: float = 0.0

        # Step 2: Grasp and lift verification (target: 1.0s)
        self.s02_lift_start: float | None = None
        self.s02_lift_duration: float = 0.0
        self.target_lifted = False

        # Step 3: Sustained drinking verification (target: 1.4s)
        self.drink_hold_start: float | None = None
        self.drink_hold_duration: float = 0.0
        self.last_drinking_seen_time: float = 0.0
        self.water_consumed = False
        self.abandon_table_start: float | None = None

        # Step 4: Return to table and hands released (target: 1.0s)
        self.s04_settle_start: float | None = None
        self.s04_settle_duration: float = 0.0
        self.last_s04_seen_time: float = 0.0
        self.protocol_complete = False

        # Cached mouth position for when bottle occludes face during drinking
        self.last_known_mouth: tuple[float, float] | None = None
        self.last_known_mouth_time: float = 0.0

    def process_frame(self, frame: np.ndarray, current_time: float | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Processes one video frame: runs YOLO detection + pose, checks procedure step logic,
        draws Mission Control HUD, and returns telemetry.
        """
        if current_time is None:
            current_time = time.time()

        self.frame_count += 1
        h, w = frame.shape[:2]
        if w != 640 or h != 480:
            frame = cv2.resize(frame, (640, 480))
            h, w = 480, 640

        # Check if dummy blank frame (used during reset / initial handshake)
        is_dummy_frame = (np.mean(frame) < 1.0)

        # 1. Run YOLO Object Detection with imgsz=320 for sub-100ms CPU inference
        det_results = self.det_model(frame, imgsz=320, verbose=False)[0]
        detected_objects = []
        target_box = None
        target_conf = 0.0
        confusable_box = None

        for box in det_results.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = self.det_model.names[cls_id]
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

            if conf < 0.25:
                continue

            detected_objects.append({
                "class": cls_name,
                "confidence": conf,
                "box": (x1, y1, x2, y2),
                "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            })

            # Check if this matches our target (e.g. bottle or container)
            if cls_name == "bottle" or (self.expected_target == "bottle" and cls_name in ["bottle", "wine glass", "cup", "vase"]):
                if conf > target_conf:
                    target_box = (x1, y1, x2, y2)
                    target_conf = conf

            # Check if secondary confusable object (e.g. cup/mug distinct from target)
            elif cls_name in self.confusables:
                confusable_box = (x1, y1, x2, y2)

        # 2. Run YOLO Pose Estimation with imgsz=320
        pose_results = self.pose_model(frame, imgsz=320, verbose=False)[0]
        person_detected = False
        wrists = []
        head_point = None
        mouth_region = None

        if len(pose_results.keypoints) > 0 and pose_results.keypoints.data.shape[1] >= 17:
            kpts = pose_results.keypoints.data[0].cpu().numpy()
            person_detected = True

            # Keypoint indices in COCO:
            # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
            # 5: left_shoulder, 6: right_shoulder
            # 9: left_wrist, 10: right_wrist
            nose = kpts[0]
            left_eye = kpts[1]
            right_eye = kpts[2]
            left_shoulder = kpts[5]
            right_shoulder = kpts[6]

            if nose[2] > 0.20:
                head_point = (float(nose[0]), float(nose[1]))
                mouth_region = (head_point[0], head_point[1] + 28.0)  # estimated mouth position below nose
                self.last_known_mouth = mouth_region
                self.last_known_mouth_time = current_time
            elif left_eye[2] > 0.20 and right_eye[2] > 0.20:
                # If nose is covered by bottle during drinking, estimate mouth from eye center!
                eye_mid_x = (float(left_eye[0]) + float(right_eye[0])) / 2.0
                eye_mid_y = (float(left_eye[1]) + float(right_eye[1])) / 2.0
                mouth_region = (eye_mid_x, eye_mid_y + 45.0)
                self.last_known_mouth = mouth_region
                self.last_known_mouth_time = current_time
            elif left_eye[2] > 0.20:
                mouth_region = (float(left_eye[0]) + 15.0, float(left_eye[1]) + 45.0)
                self.last_known_mouth = mouth_region
                self.last_known_mouth_time = current_time
            elif right_eye[2] > 0.20:
                mouth_region = (float(right_eye[0]) - 15.0, float(right_eye[1]) + 45.0)
                self.last_known_mouth = mouth_region
                self.last_known_mouth_time = current_time
            elif left_shoulder[2] > 0.20 and right_shoulder[2] > 0.20:
                # Fallback to shoulder center if face completely occluded
                sh_x = (float(left_shoulder[0]) + float(right_shoulder[0])) / 2.0
                sh_y = (float(left_shoulder[1]) + float(right_shoulder[1])) / 2.0
                mouth_region = (sh_x, sh_y - 95.0)
                self.last_known_mouth = mouth_region
                self.last_known_mouth_time = current_time
            elif self.last_known_mouth is not None and (current_time - self.last_known_mouth_time) < 10.0:
                # Use recent cached mouth position if bottle/hand completely occluded face while drinking
                mouth_region = self.last_known_mouth

            left_wrist = kpts[9]
            if left_wrist[2] > 0.25:
                wrists.append({"side": "left", "point": (float(left_wrist[0]), float(left_wrist[1])), "conf": float(left_wrist[2])})

            right_wrist = kpts[10]
            if right_wrist[2] > 0.25:
                wrists.append({"side": "right", "point": (float(right_wrist[0]), float(right_wrist[1])), "conf": float(right_wrist[2])})

        # 3. Geometric Reasoning & Spatial Relationships
        hand_contact_target = False
        hand_contact_confusable = False
        closest_wrist_dist = 9999.0

        if target_box is not None:
            bx1, by1, bx2, by2 = target_box
            b_width = bx2 - bx1
            b_height = by2 - by1

            for w_info in wrists:
                wx, wy = w_info["point"]
                dist_x = max(bx1 - wx, 0.0, wx - bx2)
                dist_y = max(by1 - wy, 0.0, wy - by2)
                dist = math.hypot(dist_x, dist_y)
                closest_wrist_dist = min(closest_wrist_dist, dist)

                # Wrist inside or in realistic proximity (< 75px or 1.25x bottle width)
                if dist < max(75.0, b_width * 1.25):
                    hand_contact_target = True

        if confusable_box is not None:
            cx1, cy1, cx2, cy2 = confusable_box
            for w_info in wrists:
                wx, wy = w_info["point"]
                dist_x = max(cx1 - wx, 0.0, wx - cx2)
                dist_y = max(cy1 - wy, 0.0, wy - cy2)
                dist = math.hypot(dist_x, dist_y)
                if dist < 85.0:
                    hand_contact_confusable = True

        # Distance to mouth/face & Drinking Pose Check
        dist_to_mouth = 9999.0
        near_mouth = False
        is_drinking_pose = False

        if target_box is not None and mouth_region is not None:
            bx1, by1, bx2, by2 = target_box
            b_height = by2 - by1
            b_width = bx2 - bx1
            top_center = ((bx1 + bx2) / 2.0, by1)  # top drinking opening of bottle
            top_left = (bx1, by1)
            top_right = (bx2, by1)
            mx, my = mouth_region

            # Distance from mouth to bottle top / cap / upper corners (handles tilted bottles)
            dist_top = math.hypot(top_center[0] - mx, top_center[1] - my)
            dist_tl = math.hypot(top_left[0] - mx, top_left[1] - my)
            dist_tr = math.hypot(top_right[0] - mx, top_right[1] - my)
            dist_to_mouth = min(dist_top, dist_tl, dist_tr)

            # Distance from mouth to bottle bounding box edge
            dist_x = max(bx1 - mx, 0.0, mx - bx2)
            dist_y = max(by1 - my, 0.0, my - by2)
            box_dist_to_mouth = math.hypot(dist_x, dist_y)

            # Check if mouth coordinates overlap or are inside bottle upper half / near upper opening
            mouth_overlap = (bx1 - 35.0 <= mx <= bx2 + 35.0) and (by1 - 40.0 <= my <= by1 + b_height * 0.75)

            # Realistic mouth contact threshold for 480p camera:
            # Handles bottle held directly to mouth, tilted sideways, or overlapping mouth region
            if (dist_to_mouth < 110.0 or box_dist_to_mouth < 45.0 or mouth_overlap) and (by1 < my + 60.0):
                near_mouth = True

        # Table surface baseline (calibrated during S01 when resting on table)
        table_zone_y = h * 0.65
        ref_table = self.baseline_table_y or table_zone_y
        in_table_zone = False
        if target_box is not None:
            b_bottom = target_box[3]
            # In table zone if resting on or below the table reference line
            if b_bottom >= (ref_table - 35.0) or (b_bottom > h * 0.52):
                in_table_zone = True

        # Vertical Lift Calculation
        is_lifted = False
        lift_delta = 0.0
        if target_box is not None:
            b_bottom = target_box[3]
            b_center_y = (target_box[1] + target_box[3]) / 2.0
            if self.initial_bottle_y is not None:
                lift_delta = self.initial_bottle_y - b_center_y

            # True physical lift: bottle bottom lifted > 25px off baseline table surface
            if ((ref_table - b_bottom) > 25.0 or lift_delta > 25.0) and b_bottom < (ref_table - 15.0):
                is_lifted = True

        # Genuine Drinking Pose requires:
        # 1. Bottle top or box is near mouth zone
        # 2. Bottle is raised (lifted or in upper frame near face)
        # 3. Hand is holding the bottle (hand contact or wrist nearby)
        if target_box is not None and near_mouth and (is_lifted or target_box[1] < (h * 0.60)):
            if hand_contact_target or closest_wrist_dist < 110.0 or len(wrists) == 0:
                is_drinking_pose = True

        # Grasp detection for Step 4 release checking:
        # Active grasp requires wrist directly touching/overlapping the bottle
        is_grasping_target = False
        if target_box is not None:
            bx1, by1, bx2, by2 = target_box
            b_width = bx2 - bx1
            for w_info in wrists:
                wx, wy = w_info["point"]
                dist_x = max(bx1 - wx, 0.0, wx - bx2)
                dist_y = max(by1 - wy, 0.0, wy - by2)
                dist = math.hypot(dist_x, dist_y)
                if dist < max(28.0, b_width * 0.45):
                    is_grasping_target = True

        # Hands Released Condition:
        # User has let go of the bottle if wrists are not actively gripping it,
        # or if wrist is pulled away, or if no hands are visible in camera frame.
        hands_released = False
        if len(wrists) == 0:
            hands_released = True
        elif closest_wrist_dist >= 35.0 or not is_grasping_target:
            hands_released = True

        # Returned to Table Surface Condition:
        # Bottle has descended from mouth/drinking zone down onto table/desk plane
        is_returned_to_table = False
        if target_box is not None:
            b_bottom = target_box[3]
            b_top = target_box[1]
            away_from_mouth = (not near_mouth) and (dist_to_mouth > 95.0 or mouth_region is None or (mouth_region and b_top > (mouth_region[1] + 20.0)))
            in_surface_zone = in_table_zone or (not is_lifted) or (b_bottom >= (ref_table - 40.0)) or (b_bottom > h * 0.50)
            if away_from_mouth and in_surface_zone:
                is_returned_to_table = True

        # Frame delta time for smooth hold accumulation
        dt = (current_time - self.last_frame_time) if self.last_frame_time else 0.05
        dt = min(max(dt, 0.01), 0.35)
        self.last_frame_time = current_time

        # 4. Procedure Compliance State Machine with Strict Preconditions
        active_step = self.steps[self.step_idx] if self.step_idx < len(self.steps) else None

        # Check for wrong object deviation
        if hand_contact_confusable and not hand_contact_target and confusable_box is not None:
            self._trigger_alert(
                step_id=active_step.id if active_step else "S02",
                severity="caution",
                kind="wrong_object",
                message="Wrong object grasped! Picked up the cup/mug instead of the target bottle.",
                tts="Wrong object grasped. Please use the water bottle.",
                t=current_time,
            )

        # Execute step state transitions only on real frames (not dummy blank frame)
        if active_step is not None and not is_dummy_frame:
            active_step.elapsed_s = current_time - (active_step.started_at or current_time)

            # ==========================================
            # STEP 1: S01 - Locate bottle on table
            # ==========================================
            if active_step.id == "S01":
                # Precondition: None (first step).
                # Requirements:
                # 1. Target bottle detected in frame.
                # 2. Bottle is resting in table zone (not held up in mid-air).
                # 3. Must stay resting stably on table for >= 1.0 seconds to calibrate baseline.
                if target_box is not None and in_table_zone and not is_lifted:
                    if self.s01_stable_start is None:
                        self.s01_stable_start = current_time
                    self.s01_stable_duration = current_time - self.s01_stable_start

                    # Calibrate table surface and bottle baseline
                    self.baseline_table_y = target_box[3]
                    self.initial_bottle_y = (target_box[1] + target_box[3]) / 2.0

                    if self.s01_stable_duration >= self.S01_TARGET_S:
                        active_step.status = "completed"
                        active_step.completed_at = current_time
                        self._advance_step(current_time)
                        active_step = self.steps[self.step_idx] if self.step_idx < len(self.steps) else None
                else:
                    self.s01_stable_start = None
                    self.s01_stable_duration = 0.0

            # ==========================================
            # STEP 2: S02 - Grasp and lift bottle
            # ==========================================
            elif active_step.id == "S02":
                # Strict Precondition: S01 MUST be completed!
                if self.steps[0].status != "completed":
                    # Cannot complete S02 without S01
                    pass
                else:
                    # Requirements:
                    # 1. Hand grasp detected on bottle (hand_contact_target or wrist < 90px)
                    # 2. Bottle lifted vertically off table surface (is_lifted == True and lift_delta > 25px)
                    # 3. Must maintain grasp + lift for >= 1.0 seconds (genuine physical hold)
                    grasp_detected = hand_contact_target or (closest_wrist_dist < 90.0)
                    if grasp_detected and is_lifted:
                        if self.s02_lift_start is None:
                            self.s02_lift_start = current_time
                        self.s02_lift_duration += dt
                        if self.s02_lift_duration >= self.S02_TARGET_S:
                            self.target_lifted = True
                            active_step.status = "completed"
                            active_step.completed_at = current_time
                            self._advance_step(current_time)
                            active_step = self.steps[self.step_idx] if self.step_idx < len(self.steps) else None
                    else:
                        self.s02_lift_start = None
                        self.s02_lift_duration = max(0.0, self.s02_lift_duration - (dt * 0.5))

            # ==========================================
            # STEP 3: S03 - Drink water from bottle
            # ==========================================
            elif active_step.id == "S03":
                # Precondition: S01 completed
                if self.steps[0].status != "completed":
                    pass
                else:
                    # Sustained drinking hold accumulation
                    if is_drinking_pose:
                        if self.drink_hold_start is None:
                            self.drink_hold_start = current_time
                        self.drink_hold_duration += dt
                        self.last_drinking_seen_time = current_time

                        # Check if genuine drinking duration (target: 1.4s) is achieved
                        if self.drink_hold_duration >= self.S03_TARGET_S:
                            self.water_consumed = True
                            active_step.status = "completed"
                            active_step.completed_at = current_time
                            self._advance_step(current_time)
                            active_step = self.steps[self.step_idx] if self.step_idx < len(self.steps) else None
                    else:
                        # Allow brief pose flicker (up to 1.2s) without wiping accumulated drinking progress
                        if (current_time - self.last_drinking_seen_time) > 1.2:
                            self.drink_hold_start = None
                            self.drink_hold_duration = max(0.0, self.drink_hold_duration - (dt * 0.3))

                    # Check for genuine premature lowering / skipping:
                    # Only if bottle is returned to table, hands released, and held there for >= 2.5s
                    if in_table_zone and not is_lifted and not self.water_consumed and hands_released:
                        if self.abandon_table_start is None:
                            self.abandon_table_start = current_time

                        abandon_time = current_time - self.abandon_table_start
                        if abandon_time < 2.5:
                            # Prompt user that drinking has not finished
                            self._trigger_alert(
                                step_id="S03",
                                severity="caution",
                                kind="incomplete",
                                message="Drink water from bottle! Bottle was lowered before drinking was completed.",
                                tts="Please drink water before returning the bottle.",
                                t=current_time,
                            )
                        else:
                            # User set bottle down on table and let go without drinking
                            self._trigger_alert(
                                step_id="S03",
                                severity="caution",
                                kind="skipped",
                                message="Step S03 Skipped! Bottle placed on table without drinking water.",
                                tts="Warning. Step three skipped. Water was not consumed.",
                                t=current_time,
                            )
                            active_step.status = "skipped"
                            active_step.completed_at = current_time
                            self._advance_step(current_time)
                            active_step = self.steps[self.step_idx] if self.step_idx < len(self.steps) else None
                    else:
                        self.abandon_table_start = None

            # ==========================================
            # STEP 4: S04 - Return bottle to table & release
            # ==========================================
            elif active_step.id == "S04":
                # Precondition: Step 1 completed (bottle was part of protocol)
                if self.steps[0].status != "completed":
                    pass
                else:
                    # Requirements:
                    # 1. Bottle returned to table surface (lowered from mouth, in table zone)
                    # 2. Hands released and pulled away from bottle
                    # 3. Stable released state accumulated for >= 1.0s
                    if is_returned_to_table and hands_released:
                        if self.s04_settle_start is None:
                            self.s04_settle_start = current_time
                        self.s04_settle_duration += dt
                        self.last_s04_seen_time = current_time

                        if self.s04_settle_duration >= self.S04_TARGET_S:
                            self.protocol_complete = True
                            active_step.status = "completed"
                            active_step.completed_at = current_time
                            self._advance_step(current_time)
                            active_step = self.steps[self.step_idx] if self.step_idx < len(self.steps) else None
                            self._trigger_alert(
                                step_id="S04",
                                severity="info",
                                kind="nominal_completion",
                                message="Water Bottle Protocol Complete! Bottle returned to table and hands released.",
                                tts="Protocol verified nominal. Bottle returned to table surface and released.",
                                t=current_time,
                            )
                    else:
                        # Allow brief pose/detection flicker (up to 1.2s) without wiping accumulated progress
                        if (current_time - self.last_s04_seen_time) > 1.2:
                            self.s04_settle_start = None
                            self.s04_settle_duration = max(0.0, self.s04_settle_duration - (dt * 0.4))

        # 5. Draw High-Tech Mission Control Visual HUD (OpenCV)
        annotated = frame.copy()
        self._render_hud(
            annotated,
            target_box=target_box,
            confusable_box=confusable_box,
            wrists=wrists,
            mouth_region=mouth_region,
            hand_contact_target=hand_contact_target,
            is_lifted=is_lifted,
            lift_delta=lift_delta,
            near_mouth=near_mouth,
            is_drinking_pose=is_drinking_pose,
            dist_to_mouth=dist_to_mouth,
            in_table_zone=in_table_zone,
            active_step=active_step,
            current_time=current_time,
            is_returned_to_table=is_returned_to_table,
            hands_released=hands_released,
            closest_wrist_dist=closest_wrist_dist,
        )

        # 6. Generate Telemetry Package
        completed_count = sum(1 for s in self.steps if s.status == "completed")
        compliance_pct = int((completed_count / len(self.steps)) * 100) if self.steps else 100
        drinking_pct = min(100, int((self.drink_hold_duration / self.S03_TARGET_S) * 100))

        step_telemetry = []
        for s in self.steps:
            is_step_verifying = False
            step_pct = 0
            if s.status == "completed":
                step_pct = 100
            elif s.status == "active":
                if s.id == "S01":
                    is_step_verifying = (self.s01_stable_duration > 0.0)
                    step_pct = min(100, int((self.s01_stable_duration / self.S01_TARGET_S) * 100))
                elif s.id == "S02":
                    is_step_verifying = (self.s02_lift_duration > 0.0)
                    step_pct = min(100, int((self.s02_lift_duration / self.S02_TARGET_S) * 100))
                elif s.id == "S03":
                    is_step_verifying = (is_drinking_pose or self.drink_hold_duration > 0.0)
                    step_pct = min(100, int((self.drink_hold_duration / self.S03_TARGET_S) * 100))
                elif s.id == "S04":
                    is_step_verifying = ((is_returned_to_table and hands_released) or self.s04_settle_duration > 0.0)
                    step_pct = min(100, int((self.s04_settle_duration / self.S04_TARGET_S) * 100))

            step_telemetry.append({
                "id": s.id,
                "name": s.name,
                "prompt": s.prompt,
                "status": s.status,
                "elapsed_s": round(s.elapsed_s, 1),
                "is_verifying": is_step_verifying,
                "verification_pct": step_pct,
            })

        telemetry = {
            "experiment_id": self.experiment_id,
            "experiment_title": self.title,
            "rack_id": self.rack_id,
            "frame_idx": self.frame_count,
            "active_step_id": active_step.id if active_step else "DONE",
            "active_step_name": active_step.name if active_step else "Protocol Completed",
            "prompt": active_step.prompt if active_step else "Experiment sequence completed nominally.",
            "compliance_score": compliance_pct,
            "is_complete": self.protocol_complete,
            "drinking": {
                "in_progress": is_drinking_pose,
                "hold_duration_s": round(self.drink_hold_duration, 2),
                "target_duration_s": self.S03_TARGET_S,
                "progress_pct": drinking_pct,
                "water_consumed": self.water_consumed,
            },
            "settle": {
                "in_progress": is_returned_to_table and hands_released,
                "is_returned": is_returned_to_table,
                "hands_released": hands_released,
                "hold_duration_s": round(self.s04_settle_duration, 2),
                "target_duration_s": self.S04_TARGET_S,
                "progress_pct": min(100, int((self.s04_settle_duration / self.S04_TARGET_S) * 100)),
                "is_complete": self.protocol_complete,
            },
            "steps": step_telemetry,
            "geometry": {
                "target_detected": target_box is not None,
                "target_confidence": round(target_conf, 2),
                "person_detected": person_detected,
                "hand_contact": hand_contact_target,
                "closest_wrist_px": round(closest_wrist_dist, 1) if closest_wrist_dist < 9000 else None,
                "is_lifted": is_lifted,
                "lift_delta_px": round(lift_delta, 1),
                "near_mouth": near_mouth,
                "is_drinking_pose": is_drinking_pose,
                "in_table_zone": in_table_zone,
                "dist_to_mouth_px": round(dist_to_mouth, 1) if dist_to_mouth < 9000 else None,
            },
            "recent_alert": (
                {
                    "step_id": self.recent_alerts[-1].step_id,
                    "severity": self.recent_alerts[-1].severity,
                    "kind": self.recent_alerts[-1].kind,
                    "message": self.recent_alerts[-1].message,
                    "tts": self.recent_alerts[-1].spoken_tts,
                }
                if self.recent_alerts
                else None
            ),
            "alert_count": len(self.alerts),
        }

        return annotated, telemetry

    def _advance_step(self, current_time: float) -> None:
        self.step_idx += 1
        if self.step_idx < len(self.steps):
            next_step = self.steps[self.step_idx]
            next_step.status = "active"
            next_step.started_at = current_time

    def _trigger_alert(self, step_id: str, severity: str, kind: str, message: str, tts: str, t: float) -> None:
        # Avoid duplicate spam within 3.5 seconds
        if self.recent_alerts and (t - self.recent_alerts[-1].timestamp) < 3.5:
            if self.recent_alerts[-1].kind == kind and self.recent_alerts[-1].step_id == step_id:
                return

        alert = DeviationAlert(step_id, severity, kind, message, t, tts)
        self.alerts.append(alert)
        self.recent_alerts.append(alert)
        if len(self.recent_alerts) > 5:
            self.recent_alerts.pop(0)

    def _render_hud(
        self,
        img: np.ndarray,
        target_box: tuple[float, float, float, float] | None,
        confusable_box: tuple[float, float, float, float] | None,
        wrists: list[dict],
        mouth_region: tuple[float, float] | None,
        hand_contact_target: bool,
        is_lifted: bool,
        lift_delta: float,
        near_mouth: bool,
        is_drinking_pose: bool,
        dist_to_mouth: float,
        in_table_zone: bool,
        active_step: StepState | None,
        current_time: float,
        is_returned_to_table: bool = False,
        hands_released: bool = False,
        closest_wrist_dist: float = 9999.0,
    ) -> None:
        """Renders high-aesthetic Mission Control HUD directly on the OpenCV frame."""
        h, w = img.shape[:2]

        # Draw Target Bounding Box
        if target_box is not None:
            x1, y1, x2, y2 = [int(v) for v in target_box]
            # Color: Cyan if drinking, Green if grasped, Amber if resting
            if is_drinking_pose:
                color = (255, 210, 0)  # Bright Cyan
                tag = "TARGET: DRINKING ACTION ACTIVE"
            elif near_mouth:
                color = (220, 200, 50)  # Cyan-Gold
                tag = "TARGET: AT MOUTH ZONE"
            elif hand_contact_target and is_lifted:
                color = (60, 220, 100)  # Bright Emerald
                tag = f"TARGET: GRASPED (LIFT +{int(lift_delta)}px)"
            elif hand_contact_target:
                color = (80, 210, 160)
                tag = "TARGET: GRASPED (TABLE)"
            else:
                color = (0, 200, 255)  # Gold/Yellow
                tag = "TARGET: BOTTLE (TABLE)"

            # Box & Corner Brackets
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            corner_len = min(20, (x2 - x1) // 3)
            # Top-left corner
            cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, 4)
            cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, 4)
            # Top-right corner
            cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, 4)
            cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, 4)
            # Bottom-left corner
            cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, 4)
            cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, 4)
            # Bottom-right corner
            cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, 4)
            cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, 4)

            # Label badge
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(img, (x1, y1 - 22), (x1 + tw + 10, y1), color, -1)
            cv2.putText(img, tag, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (10, 15, 20), 1, cv2.LINE_AA)

        # Draw Confusable Object Bounding Box
        if confusable_box is not None:
            cx1, cy1, cx2, cy2 = [int(v) for v in confusable_box]
            cv2.rectangle(img, (cx1, cy1), (cx2, cy2), (50, 50, 220), 2)
            cv2.putText(img, "CONFUSABLE (CUP)", (cx1, cy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 50, 220), 1, cv2.LINE_AA)

        # Draw Wrists / Hands
        for w_info in wrists:
            wx, wy = int(w_info["point"][0]), int(w_info["point"][1])
            cv2.circle(img, (wx, wy), 7, (0, 240, 255), -1)
            cv2.circle(img, (wx, wy), 10, (0, 240, 255), 1)

            # Proximity line to target if grasping
            if target_box is not None and hand_contact_target:
                bx = int((target_box[0] + target_box[2]) / 2.0)
                by = int((target_box[1] + target_box[3]) / 2.0)
                cv2.line(img, (wx, wy), (bx, by), (60, 255, 120), 2, cv2.LINE_AA)

        # Draw Mouth / Head Region indicator & Contact Vector
        if mouth_region is not None:
            mx, my = int(mouth_region[0]), int(mouth_region[1])
            if is_drinking_pose:
                # Green contact circle
                cv2.circle(img, (mx, my), 18, (60, 240, 100), 2)
                cv2.putText(img, "DRINK CONTACT", (mx - 50, my - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 240, 100), 1, cv2.LINE_AA)
                # Line from bottle top to mouth
                if target_box is not None:
                    bx = int((target_box[0] + target_box[2]) / 2.0)
                    by = int(target_box[1])
                    cv2.line(img, (bx, by), (mx, my), (60, 240, 100), 2, cv2.LINE_AA)
            else:
                cv2.circle(img, (mx, my), 14, (255, 140, 50), 2)
                cv2.putText(img, "DRINK ZONE", (mx - 40, my - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 140, 50), 1, cv2.LINE_AA)

        # Table Surface Reference Line
        table_y = int(self.baseline_table_y or (h * 0.65))
        cv2.line(img, (20, table_y), (w - 20, table_y), (100, 120, 140), 1, cv2.LINE_AA)
        cv2.putText(img, "TABLE SURFACE REFERENCE", (30, table_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 140, 160), 1, cv2.LINE_AA)

        # Header Bar (Dark Glassmorphic Banner)
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (15, 20, 28), -1)
        cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
        cv2.line(img, (0, 55), (w, 55), (45, 91, 216), 2)

        # Header Text
        cv2.putText(img, f"PARIKSHAK AI WITNESS | {self.experiment_id}", (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"RACK: {self.rack_id}", (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 180, 210), 1, cv2.LINE_AA)

        # Status Badge (Nominal vs Alert)
        # Status Badge (Nominal vs Verifying vs Alert)
        has_alert = len(self.recent_alerts) > 0 and (current_time - self.recent_alerts[-1].timestamp) < 4.0
        if has_alert:
            badge_color = (40, 40, 220)  # Red
            badge_text = f"ALERT: {self.recent_alerts[-1].kind.upper()}"
        elif self.protocol_complete:
            badge_color = (60, 190, 80)  # Green
            badge_text = "PROTOCOL VERIFIED NOMINAL"
        elif active_step and (
            (active_step.id == "S01" and self.s01_stable_duration > 0)
            or (active_step.id == "S02" and self.s02_lift_duration > 0)
            or (active_step.id == "S03" and self.drink_hold_duration > 0)
            or (active_step.id == "S04" and self.s04_settle_duration > 0)
        ):
            badge_color = (40, 150, 240)  # Amber
            badge_text = f"VERIFYING: {active_step.id}"
        else:
            badge_color = (220, 140, 40)  # Cyan/Blue
            badge_text = f"MONITORING: {active_step.id if active_step else 'NOMINAL'}"

        cv2.rectangle(img, (w - 280, 12), (w - 20, 44), badge_color, -1)
        cv2.putText(img, badge_text, (w - 270, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Center Action Bar depending on active step
        if active_step and active_step.id == "S01":
            # Show table stability calibration status
            calib_pct = min(1.0, self.s01_stable_duration / self.S01_TARGET_S)
            bar_w = int(260 * calib_pct)
            cv2.rectangle(img, (w // 2 - 130, 62), (w // 2 + 130, 88), (20, 25, 35), -1)
            cv2.rectangle(img, (w // 2 - 130, 62), (w // 2 - 130 + bar_w, 88), (60, 200, 120), -1)
            cv2.rectangle(img, (w // 2 - 130, 62), (w // 2 + 130, 88), (80, 140, 240), 1)
            calib_text = f"VERIFYING TABLE: {int(calib_pct * 100)}% ({self.s01_stable_duration:.1f}s / {self.S01_TARGET_S:.1f}s)" if self.s01_stable_duration > 0 else "PLACE BOTTLE ON TABLE"
            cv2.putText(img, calib_text, (w // 2 - 120, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        elif active_step and active_step.id == "S02":
            # Grasp and Vertical Lift Progress Bar
            pct = min(1.0, self.s02_lift_duration / self.S02_TARGET_S)
            bar_w = int(260 * pct)
            cv2.rectangle(img, (w // 2 - 130, 62), (w // 2 + 130, 88), (20, 25, 35), -1)
            grasp_detected = hand_contact_target or (closest_wrist_dist < 90.0)
            fill_color = (60, 230, 100) if (is_lifted and grasp_detected) else (40, 180, 240)
            cv2.rectangle(img, (w // 2 - 130, 62), (w // 2 - 130 + bar_w, 88), fill_color, -1)
            cv2.rectangle(img, (w // 2 - 130, 62), (w // 2 + 130, 88), (80, 140, 240), 1)

            if is_lifted and grasp_detected:
                lift_label = f"VERIFYING LIFT: {int(pct * 100)}% ({self.s02_lift_duration:.1f}s / {self.S02_TARGET_S:.1f}s)"
            elif grasp_detected:
                lift_label = "BOTTLE GRASPED -> LIFT OFF TABLE"
            else:
                lift_label = "GRASP AND LIFT BOTTLE"

            cv2.putText(img, lift_label, (w // 2 - 120, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        elif active_step and active_step.id == "S03":
            # Sustained Drinking Action Progress Bar
            pct = min(1.0, self.drink_hold_duration / self.S03_TARGET_S)
            bar_w = int(260 * pct)
            cv2.rectangle(img, (w // 2 - 130, 62), (w // 2 + 130, 88), (20, 25, 35), -1)
            fill_color = (60, 230, 100) if is_drinking_pose else (40, 180, 240)
            cv2.rectangle(img, (w // 2 - 130, 62), (w // 2 - 130 + bar_w, 88), fill_color, -1)
            cv2.rectangle(img, (w // 2 - 130, 62), (w // 2 + 130, 88), (80, 140, 240), 1)

            if is_drinking_pose:
                drink_label = f"VERIFYING DRINK: {int(pct * 100)}% ({self.drink_hold_duration:.1f}s / {self.S03_TARGET_S:.1f}s)"
            elif self.drink_hold_duration > 0:
                drink_label = f"HOLD TO MOUTH ({self.drink_hold_duration:.1f}s / {self.S03_TARGET_S:.1f}s)"
            else:
                drink_label = "BRING BOTTLE TO MOUTH TO DRINK"

            cv2.putText(img, drink_label, (w // 2 - 120, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        elif active_step and active_step.id == "S04":
            # Return to Table Surface & Hand Release Progress Bar
            pct = min(1.0, self.s04_settle_duration / self.S04_TARGET_S)
            bar_w = int(260 * pct)
            cv2.rectangle(img, (w // 2 - 130, 62), (w // 2 + 130, 88), (20, 25, 35), -1)
            fill_color = (60, 230, 100) if (is_returned_to_table and hands_released) else (40, 180, 240)
            cv2.rectangle(img, (w // 2 - 130, 62), (w // 2 - 130 + bar_w, 88), fill_color, -1)
            cv2.rectangle(img, (w // 2 - 130, 62), (w // 2 + 130, 88), (80, 140, 240), 1)

            if not is_returned_to_table:
                settle_label = "PLACE BOTTLE ON TABLE SURFACE"
            elif not hands_released:
                settle_label = "BOTTLE ON TABLE -> RELEASE HANDS"
            else:
                settle_label = f"VERIFYING RELEASE: {int(pct * 100)}% ({self.s04_settle_duration:.1f}s / {self.S04_TARGET_S:.1f}s)"

            cv2.putText(img, settle_label, (w // 2 - 120, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        elif self.protocol_complete:
            cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 + 140, 88), (20, 35, 25), -1)
            cv2.rectangle(img, (w // 2 - 140, 62), (w // 2 + 140, 88), (60, 220, 100), -1)
            cv2.putText(img, "ALL 4 STEPS VERIFIED NOMINAL", (w // 2 - 125, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # Bottom Prompt & Alert Bar
        cv2.rectangle(overlay, (0, h - 60), (w, h), (15, 20, 28), -1)
        cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)
        cv2.line(img, (0, h - 60), (w, h - 60), (45, 91, 216), 1)

        if has_alert:
            alert = self.recent_alerts[-1]
            cv2.putText(img, f"WARNING: {alert.message}", (25, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 80, 255), 2, cv2.LINE_AA)
        elif active_step:
            step_prompt = f"[{active_step.id}] {active_step.prompt}"
            cv2.putText(img, step_prompt, (25, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 245, 250), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "All procedure steps verified successfully.", (25, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 220, 120), 1, cv2.LINE_AA)

