#!/usr/bin/env python3
"""Lightweight real-time ImageNet-1K classification from a webcam.

Pipeline: capture a frame, isolate its center, apply the preprocessing expected
by MobileNetV3, infer 1,000 class probabilities, and display the highest scores.
Inference is intentionally skipped on some frames so the camera remains fluid.
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify the center of a webcam frame with ImageNet-1K."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--top-k", type=int, default=3, choices=range(1, 6), metavar="1-5")
    parser.add_argument(
        "--every",
        type=int,
        default=5,
        metavar="FRAMES",
        help="Run inference every N frames (default: 5)",
    )
    parser.add_argument(
        "--crop-scale",
        type=float,
        default=0.65,
        metavar="RATIO",
        help="Size of the square focus area, from 0.3 to 1.0 (default: 0.65)",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.65,
        metavar="RATIO",
        help="Weight given to the previous prediction, from 0 to 0.95 (default: 0.65)",
    )
    return parser.parse_args()


def focus_crop(frame: np.ndarray, scale: float) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Return a centered square crop and its (left, top, right, bottom) bounds."""
    # ImageNet models classify the whole input rather than locating each object.
    # A focus area reduces background noise and tells the presenter where to
    # place the object that should dominate the prediction.
    height, width = frame.shape[:2]
    side = max(1, int(min(height, width) * scale))
    left = (width - side) // 2
    top = (height - side) // 2
    return frame[top : top + side, left : left + side], (left, top, left + side, top + side)


def draw_panel(
    frame: np.ndarray,
    predictions: list[tuple[str, float]],
    inference_ms: float | None,
    device: torch.device,
    fps: float,
) -> None:
    """Draw predictions without changing the frame used for inference."""
    # Draw on a copy first, then blend it to obtain a translucent background.
    # This panel is added after inference, so its text never reaches the model.
    panel_height = 104 + 36 * len(predictions)
    overlay = frame.copy()
    cv2.rectangle(overlay, (14, 14), (520, panel_height), (7, 17, 29), -1)
    cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)

    cv2.putText(frame, "MobileNetV3 Small | ImageNet-1K", (30, 47),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (66, 217, 200), 2, cv2.LINE_AA)
    timing = "loading first prediction..." if inference_ms is None else f"inference {inference_ms:.1f} ms | camera {fps:.1f} FPS | {device}"
    cv2.putText(frame, timing, (30, 76), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (180, 194, 211), 1, cv2.LINE_AA)

    for rank, (label, confidence) in enumerate(predictions, start=1):
        y = 111 + (rank - 1) * 36
        cv2.putText(frame, f"{rank}. {label[:32]}", (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.57, (244, 247, 251), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{confidence:5.1f}%", (432, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.57, (199, 243, 107), 1, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    if args.every < 1:
        raise SystemExit("--every must be at least 1")
    if not 0.3 <= args.crop_scale <= 1.0:
        raise SystemExit("--crop-scale must be between 0.3 and 1.0")
    if not 0.0 <= args.smoothing <= 0.95:
        raise SystemExit("--smoothing must be between 0 and 0.95")

    print("Loading MobileNetV3 Small weights (the first run may download about 10 MB)...")

    # The weights object is more than a checkpoint: torchvision also stores the
    # matching preprocessing recipe and all 1,000 ImageNet class names in it.
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights).eval()
    transform = weights.transforms()
    categories = weights.meta["categories"]

    # CUDA is used when available; otherwise this small network runs on the CPU.
    # eval() above disables training-only behavior such as dropout updates.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    predictions: list[tuple[str, float]] = []
    smoothed: torch.Tensor | None = None
    inference_ms: float | None = None
    frame_number = 0
    frame_times: deque[float] = deque(maxlen=30)
    print("Point the camera at one centered object. Press Q or Esc to quit.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Could not read a frame from the camera.")
                break

            # Mirroring makes the preview behave like a familiar webcam view.
            frame = cv2.flip(frame, 1)
            crop, bounds = focus_crop(frame, args.crop_scale)
            frame_number += 1

            # Classifying every frame wastes work because adjacent webcam frames
            # are nearly identical. Reusing the latest result keeps the UI light.
            if frame_number == 1 or frame_number % args.every == 0:
                # OpenCV supplies BGR pixels, while PIL/torchvision expect RGB.
                image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

                # The official transform resizes, center-crops, converts to a
                # tensor, and normalizes pixels exactly as training expected.
                batch = transform(image).unsqueeze(0).to(device)

                # GPU operations are asynchronous. Synchronizing around the
                # model call makes the displayed inference time meaningful.
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    # MobileNet returns 1,000 raw logits. Softmax converts them
                    # into relative probabilities whose sum is one.
                    current = model(batch).softmax(dim=1)[0]
                if device.type == "cuda":
                    torch.cuda.synchronize()
                inference_ms = (time.perf_counter() - started) * 1_000

                # An exponential moving average prevents labels from flickering
                # when two classes receive similar scores in consecutive frames.
                smoothed = current if smoothed is None else (
                    args.smoothing * smoothed + (1.0 - args.smoothing) * current
                )

                # topk returns class indexes; categories maps them to readable
                # ImageNet labels such as "coffee mug" or "banana".
                scores, class_ids = smoothed.topk(args.top_k)
                predictions = [
                    (categories[class_id], score * 100)
                    for score, class_id in zip(scores.tolist(), class_ids.tolist())
                ]

            # Camera FPS is averaged over a short rolling window for stability.
            now = time.perf_counter()
            frame_times.append(now)
            fps = 0.0
            if len(frame_times) > 1:
                elapsed = frame_times[-1] - frame_times[0]
                if elapsed > 0:
                    fps = (len(frame_times) - 1) / elapsed

            # Only after inference do we add the focus rectangle and UI panel.
            left, top, right, bottom = bounds
            cv2.rectangle(frame, (left, top), (right, bottom), (66, 217, 200), 2)
            cv2.putText(frame, "place one object inside this area", (left + 8, bottom - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (66, 217, 200), 1, cv2.LINE_AA)
            draw_panel(frame, predictions, inference_ms, device, fps)
            cv2.imshow("ImageNet Webcam | Q to quit", frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
