# ImageNet Proof of Concept

This repository contains a short, demo-first introduction to ImageNet and practical computer vision. The complete session is designed to take **5–10 minutes**, with most of the time spent running live POCs.

## Contents

- `slides/index.html` — self-contained presentation (5 slides, no build step)
- `SPEAKER_NOTES.md` — an English script with timing and demo cues
- `demo/imagenet_demo.py` — classifies one local or remote image
- `demo/webcam_object_monitor.py` — real-time ImageNet classification with stable target alerts
- `demo/ergonomic_webcam_monitor.py` — local posture, viewing-distance, and break monitor
- `requirements.txt` — Python dependencies for all POCs

## Open the slides

Open `slides/index.html` in any modern browser, or serve the folder locally:

```bash
python3 -m http.server 8000
```

Then visit `http://localhost:8000/slides/`.

Controls:

- `→`, `Space`, or click the right side: next slide
- `←` or click the left side: previous slide
- `Home` / `End`: first / last slide
- `F`: toggle fullscreen

## Install the demos

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Run each webcam POC before presenting so camera permissions and dependencies are already confirmed.

## POC 0 — Single-image baseline

```bash
python3 demo/imagenet_demo.py
```

The first run downloads the pretrained weights (about 21 MB) and a sample dog image. To use your own image:

```bash
python3 demo/imagenet_demo.py path/to/image.jpg
```

Or pass an image URL:

```bash
python3 demo/imagenet_demo.py https://example.com/image.jpg
```

Run this once before presenting so the MobileNetV3 weights are cached. If the venue has unreliable internet, also download a test image and pass its local path.

## POC 1 — Webcam object monitor

```bash
python3 demo/webcam_object_monitor.py
```

This demo classifies the complete camera frame while keeping the UI responsive with a background inference worker. It averages predictions over several frames to reduce flicker.

Use the target-watching mode to model a simple monitored station. The following command saves a timestamped frame after a matching ImageNet class remains visible for three inference updates:

```bash
python3 demo/webcam_object_monitor.py \
  --watch "coffee mug" \
  --watch "water bottle" \
  --min-confidence 15
```

Controls:

- `Q` or `Esc`: quit
- `S`: save a manual snapshot in `captures/`

Practical examples include a prototype intake station, visual inventory checkpoint, or presence alert. Because ImageNet classification describes the whole frame, production systems that must locate several objects should use an object detector instead.

## POC 2 — Ergonomic webcam monitor

```bash
python3 demo/ergonomic_webcam_monitor.py
```

Sit in a comfortable position and press `C`. The app calibrates the apparent distance between your eyes, then warns when you remain too close to the screen or tilt your head for too long. Frames stay in memory and are not recorded.

Controls:

- `C`: calibrate the normal viewing position
- `B`: reset the eye-break timer after taking a break
- `Q` or `Esc`: quit

For a short presentation, accelerate the break reminder:

```bash
python3 demo/ergonomic_webcam_monitor.py --break-minutes 0.25
```

The distance value is a relative visual estimate, not a physical measurement or medical assessment.

## Troubleshooting

If camera `0` is unavailable, try another device index:

```bash
python3 demo/webcam_object_monitor.py --camera 1
```

Close conferencing software that may already be using the camera. On Linux, verify that your user can access `/dev/video*`.

## Suggested presentation flow

The speaker script uses about two minutes for five slides and four to five minutes for the live POCs. Use only POC 1 for a five-minute version, or run both webcam demos for a seven-to-nine-minute session.
