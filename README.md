# ImageNet Proof of Concept

This repository contains a short, demo-first introduction to ImageNet and practical computer vision. The complete session is designed to take **5–10 minutes**, with most of the time spent running live POCs.

## Contents

- `slides/index.html` — self-contained presentation (7 slides, no build step)
- `SPEAKER_NOTES.md` — an English script with timing and demo cues
- `demo/imagenet_demo.py` — classifies one local or remote image
- `demo/imagenet_webcam.py` — lightweight live ImageNet-1K classification from a webcam
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

Run the webcam POC before presenting so camera permissions and model weights are already confirmed.

## POC 1 — ImageNet classification

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

## POC 2 — ImageNet webcam classifier

```bash
python3 demo/imagenet_webcam.py
```

Place one object at a time inside the focus square. The app uses MobileNetV3 Small with ImageNet-1K weights and shows the three most likely classes. Frames stay in memory and are not recorded.

Controls:

- `Q` or `Esc`: quit

For a slower computer, run inference less often:

```bash
python3 demo/imagenet_webcam.py --every 10
```

The default model has about 2.5 million parameters and its weights are about 10 MB. Predictions are smoothed to reduce label flicker. This is closed-set classification: the answer is always chosen from ImageNet-1K's 1,000 classes.

## Troubleshooting

If camera `0` is unavailable, try another device index:

```bash
python3 demo/imagenet_webcam.py --camera 1
```

Close conferencing software that may already be using the camera. On Linux, verify that your user can access `/dev/video*`.

## Suggested presentation flow

The speaker script uses about three minutes for theory and code walkthroughs, plus three to four minutes for the two POCs. For a five-minute version, use only the webcam demo; it already demonstrates the complete ImageNet inference pipeline.
