# ImageNet Proof of Concept

This repository contains a short, presentation-ready introduction to ImageNet and a live image-classification demo. The complete session is designed to take **5–10 minutes**.

## Contents

- `slides/index.html` — self-contained presentation (6 slides, no build step)
- `SPEAKER_NOTES.md` — an English script with timing and demo cues
- `demo/imagenet_demo.py` — classifies a local or remote image with a model pretrained on ImageNet-1K
- `requirements.txt` — Python dependencies for the demo

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

## Run the demo

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
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

Run the demo once before presenting so the model weights are cached. If the venue has unreliable internet, also download a test image and pass its local path.

## Suggested presentation flow

The slide script includes a normal path of approximately 6 minutes and optional lines that can be omitted to stay closer to 5 minutes. The live demo should take about 60 seconds.
