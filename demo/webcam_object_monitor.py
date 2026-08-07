#!/usr/bin/env python3
"""Real-time ImageNet classification with temporal smoothing and target alerts."""

from __future__ import annotations

import argparse
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


@dataclass(frozen=True)
class InferenceResult:
    predictions: list[Prediction]
    elapsed_ms: float
    sequence: int


class ClassifierWorker:
    """Run inference off the camera thread and keep only the newest frame."""

    def __init__(self, smoothing_window: int, top_k: int) -> None:
        weights = MobileNet_V3_Large_Weights.DEFAULT
        self.model = mobilenet_v3_large(weights=weights).eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.transform = weights.transforms()
        self.categories = weights.meta["categories"]
        self.top_k = top_k
        self._history: deque[torch.Tensor] = deque(maxlen=smoothing_window)
        self._frames: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self._result: InferenceResult | None = None
        self._result_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._sequence = 0

    def start(self) -> None:
        self._thread.start()

    def submit(self, frame_bgr: np.ndarray) -> None:
        if self._frames.full():
            return
        self._frames.put_nowait(frame_bgr.copy())

    def latest(self) -> InferenceResult | None:
        with self._result_lock:
            return self._result

    def close(self) -> None:
        while not self._frames.empty():
            try:
                self._frames.get_nowait()
            except queue.Empty:
                break
        self._frames.put_nowait(None)
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while True:
            frame = self._frames.get()
            if frame is None:
                return
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            batch = self.transform(image).unsqueeze(0).to(self.device)
            started = time.perf_counter()
            with torch.inference_mode():
                probabilities = self.model(batch).softmax(dim=1)[0].cpu()
            elapsed_ms = (time.perf_counter() - started) * 1_000

            self._history.append(probabilities)
            smoothed = torch.stack(tuple(self._history)).mean(dim=0)
            scores, class_ids = smoothed.topk(self.top_k)
            predictions = [
                Prediction(self.categories[class_id], score * 100)
                for score, class_id in zip(scores.tolist(), class_ids.tolist())
            ]
            self._sequence += 1
            with self._result_lock:
                self._result = InferenceResult(predictions, elapsed_ms, self._sequence)


class TargetWatcher:
    """Trigger after a target remains in the classifier output for several updates."""

    def __init__(
        self,
        targets: list[str],
        minimum_confidence: float,
        stable_updates: int,
        cooldown_seconds: float,
    ) -> None:
        self.targets = [target.casefold() for target in targets]
        self.minimum_confidence = minimum_confidence
        self.stable_updates = stable_updates
        self.cooldown_seconds = cooldown_seconds
        self._candidate: str | None = None
        self._count = 0
        self._last_triggered = 0.0
        self._last_sequence = -1

    def update(self, result: InferenceResult | None) -> Prediction | None:
        if result is None or result.sequence == self._last_sequence:
            return None
        self._last_sequence = result.sequence
        match = None
        match = next(
            (
                prediction
                for prediction in result.predictions
                if prediction.confidence >= self.minimum_confidence
                and any(target in prediction.label.casefold() for target in self.targets)
            ),
            None,
        )

        label = match.label if match else None
        if label == self._candidate:
            self._count += 1
        else:
            self._candidate = label
            self._count = 1 if label else 0

        ready = self._count >= self.stable_updates
        outside_cooldown = time.monotonic() - self._last_triggered >= self.cooldown_seconds
        if match and ready and outside_cooldown:
            self._last_triggered = time.monotonic()
            self._count = 0
            return match
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor a webcam with MobileNetV3 pretrained on ImageNet-1K."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--top-k", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--smoothing", type=int, default=6, choices=range(1, 31))
    parser.add_argument(
        "--watch",
        action="append",
        default=[],
        metavar="TEXT",
        help="Alert when a class containing TEXT is stable; may be repeated",
    )
    parser.add_argument("--min-confidence", type=float, default=20.0)
    parser.add_argument("--stable-updates", type=int, default=3)
    parser.add_argument("--cooldown", type=float, default=10.0)
    parser.add_argument("--capture-dir", type=Path, default=Path("captures"))
    return parser.parse_args()


def draw_panel(
    frame: np.ndarray,
    result: InferenceResult | None,
    fps: float,
    watch_targets: list[str],
) -> None:
    overlay = frame.copy()
    panel_height = 145 if result else 75
    cv2.rectangle(overlay, (12, 12), (510, panel_height), (8, 17, 29), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.putText(frame, f"Camera {fps:4.1f} FPS", (28, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (66, 217, 200), 2)
    if result is None:
        cv2.putText(frame, "Loading first prediction...", (28, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (220, 228, 238), 1)
        return
    cv2.putText(frame, f"Inference {result.elapsed_ms:.1f} ms", (255, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 182, 200), 1)
    for index, prediction in enumerate(result.predictions):
        y = 70 + index * 27
        text = f"{index + 1}. {prediction.label}: {prediction.confidence:.1f}%"
        cv2.putText(frame, text, (28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (244, 247, 251), 1, cv2.LINE_AA)
    if watch_targets:
        watched = ", ".join(watch_targets)
        cv2.putText(frame, f"Watching: {watched}", (28, panel_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (199, 243, 107), 1)


def save_evidence(frame: np.ndarray, prediction: Prediction, capture_dir: Path) -> Path:
    capture_dir.mkdir(parents=True, exist_ok=True)
    safe_label = prediction.label.replace(" ", "_").replace("/", "-")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = capture_dir / f"{timestamp}-{safe_label}.jpg"
    cv2.imwrite(str(path), frame)
    return path


def main() -> int:
    args = parse_args()
    if not 0 <= args.min_confidence <= 100:
        raise SystemExit("--min-confidence must be between 0 and 100")
    if args.stable_updates < 1:
        raise SystemExit("--stable-updates must be at least 1")

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    worker = ClassifierWorker(args.smoothing, args.top_k)
    worker.start()
    watcher = TargetWatcher(args.watch, args.min_confidence, args.stable_updates, args.cooldown)
    previous_time = time.perf_counter()
    fps = 0.0
    alert_text = ""
    alert_until = 0.0

    print("Press Q to quit or S to save a snapshot.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            worker.submit(frame)
            result = worker.latest()
            now = time.perf_counter()
            instant_fps = 1.0 / max(now - previous_time, 0.001)
            fps = instant_fps if fps == 0 else fps * 0.9 + instant_fps * 0.1
            previous_time = now

            triggered = watcher.update(result) if args.watch else None
            if triggered:
                path = save_evidence(frame, triggered, args.capture_dir)
                alert_text = f"ALERT: {triggered.label} saved to {path.name}"
                alert_until = time.monotonic() + 3
                print(alert_text)

            draw_panel(frame, result, fps, args.watch)
            if time.monotonic() < alert_until:
                cv2.putText(frame, alert_text, (24, frame.shape[0] - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (80, 220, 120), 2)
            cv2.imshow("ImageNet Webcam Object Monitor", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                path = save_evidence(frame, Prediction("manual", 100), args.capture_dir)
                print(f"Snapshot saved to {path}")
    finally:
        worker.close()
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
