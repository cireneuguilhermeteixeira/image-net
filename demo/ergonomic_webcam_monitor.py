#!/usr/bin/env python3
"""Local ergonomic monitor using face landmarks from a webcam."""

from __future__ import annotations

import argparse
import math
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np


LEFT_EYE = 33
RIGHT_EYE = 263
NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor viewing distance, head tilt, and break time locally."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--too-close", type=float, default=1.35, help="Ratio above calibrated face size")
    parser.add_argument("--tilt-degrees", type=float, default=12.0)
    parser.add_argument("--warning-seconds", type=float, default=2.0)
    parser.add_argument("--break-minutes", type=float, default=20.0)
    return parser.parse_args()


def pixel(landmark, width: int, height: int) -> tuple[int, int]:
    return int(landmark.x * width), int(landmark.y * height)


def draw_status(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    height = 52 + 31 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (540, height), (8, 17, 29), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    for index, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (28, 45 + index * 31), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    if args.too_close <= 1:
        raise SystemExit("--too-close must be greater than 1")
    if args.tilt_degrees <= 0 or args.warning_seconds < 0 or args.break_minutes <= 0:
        raise SystemExit("Tilt and break duration must be positive; warning duration cannot be negative")
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    recent_eye_distances: deque[float] = deque(maxlen=12)
    recent_tilts: deque[float] = deque(maxlen=12)
    calibrated_distance: float | None = None
    bad_posture_since: float | None = None
    session_started = time.monotonic()

    print("Sit comfortably and press C to calibrate. Press B after a break or Q to quit.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            lines: list[tuple[str, tuple[int, int, int]]] = []
            posture_issue = False

            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark
                left = landmarks[LEFT_EYE]
                right = landmarks[RIGHT_EYE]
                eye_distance = math.hypot(right.x - left.x, right.y - left.y)
                tilt = math.degrees(math.atan2(right.y - left.y, right.x - left.x))
                recent_eye_distances.append(eye_distance)
                recent_tilts.append(tilt)
                smooth_distance = float(np.mean(recent_eye_distances))
                smooth_tilt = float(np.mean(recent_tilts))

                for index in (LEFT_EYE, RIGHT_EYE, NOSE_TIP, FOREHEAD, CHIN):
                    cv2.circle(frame, pixel(landmarks[index], width, height), 4, (66, 217, 200), -1)
                cv2.line(frame, pixel(left, width, height), pixel(right, width, height), (199, 243, 107), 2)

                if calibrated_distance is None:
                    lines.append(("Press C to calibrate your normal distance", (199, 243, 107)))
                else:
                    distance_ratio = smooth_distance / calibrated_distance
                    too_close = distance_ratio > args.too_close
                    too_tilted = abs(smooth_tilt) > args.tilt_degrees
                    posture_issue = too_close or too_tilted
                    distance_text = f"Distance: {distance_ratio:.2f}x calibrated"
                    lines.append((distance_text, (80, 120, 255) if too_close else (80, 220, 120)))
                    lines.append((f"Head tilt: {smooth_tilt:+.1f} degrees", (80, 120, 255) if too_tilted else (80, 220, 120)))
            else:
                lines.append(("Face not detected", (80, 120, 255)))

            now = time.monotonic()
            if posture_issue:
                bad_posture_since = bad_posture_since or now
            else:
                bad_posture_since = None
            if bad_posture_since and now - bad_posture_since >= args.warning_seconds:
                lines.append(("POSTURE ALERT: move back or straighten your head", (80, 120, 255)))

            break_seconds = args.break_minutes * 60
            remaining = max(0.0, break_seconds - (now - session_started))
            if remaining == 0:
                lines.append(("BREAK REMINDER: look at something far away", (80, 180, 255)))
            else:
                lines.append((f"Next eye break: {int(remaining // 60):02d}:{int(remaining % 60):02d}", (170, 182, 200)))

            lines.append(("Local only | C calibrates | B resets break | Q quits", (170, 182, 200)))
            draw_status(frame, lines)
            cv2.imshow("Ergonomic Webcam Monitor", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c") and recent_eye_distances:
                calibrated_distance = float(np.mean(recent_eye_distances))
                bad_posture_since = None
                session_started = time.monotonic()
                print(f"Calibrated at eye-distance ratio {calibrated_distance:.4f}")
            if key == ord("b"):
                session_started = time.monotonic()
                print("Break timer reset")
    finally:
        face_mesh.close()
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
