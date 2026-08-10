#!/usr/bin/env python3
"""Classify one image with MobileNetV3 pretrained on ImageNet-1K.

This small command-line example exposes the same four stages used by the live
demo: load an image, preprocess it, run inference, and rank ImageNet labels.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


DEFAULT_IMAGE = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify an image using MobileNetV3 pretrained on ImageNet-1K."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=DEFAULT_IMAGE,
        help="Local image path or HTTP(S) URL (default: PyTorch's sample dog image)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        choices=range(1, 11),
        metavar="1-10",
        help="Number of predictions to display (default: 5)",
    )
    return parser.parse_args()


def load_image(source: str) -> Image.Image:
    """Load an RGB image from a URL or local path."""
    if source.startswith(("http://", "https://")):
        request = urllib.request.Request(
            source, headers={"User-Agent": "ImageNet-POC/1.0"}
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            image_bytes = response.read()
        image = Image.open(io.BytesIO(image_bytes))
    else:
        image = Image.open(Path(source).expanduser())
    return image.convert("RGB")


def classify(image: Image.Image, top_k: int) -> tuple[list[tuple[str, float]], float, str]:
    """Return ranked labels, inference time in milliseconds, and device name."""
    # DEFAULT selects pretrained ImageNet-1K weights. The weights object also
    # carries the precise preprocessing pipeline and the category names.
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights)
    model.eval()

    # Move both the network and its input to the same available device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Resize, crop and normalize with the recipe used during model training;
    # unsqueeze adds the batch dimension expected by PyTorch: [1, 3, H, W].
    batch = weights.transforms()(image).unsqueeze(0).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        # The output contains one score per ImageNet-1K class. Softmax changes
        # those raw logits into relative probabilities.
        probabilities = model(batch).softmax(dim=1)[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1_000

    # Keep only the strongest predictions and translate indexes into labels.
    scores, class_ids = probabilities.topk(top_k)
    categories = weights.meta["categories"]
    predictions = [
        (categories[class_id], score * 100)
        for score, class_id in zip(scores.tolist(), class_ids.tolist())
    ]
    return predictions, elapsed_ms, str(device)


def main() -> int:
    args = parse_args()
    try:
        image = load_image(args.image)
        predictions, elapsed_ms, device = classify(image, args.top_k)
    except (FileNotFoundError, IsADirectoryError, UnidentifiedImageError) as error:
        print(f"Could not read the image: {error}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, TimeoutError) as error:
        print(f"Could not download the image: {error}", file=sys.stderr)
        return 1
    except RuntimeError as error:
        print(f"Model inference failed: {error}", file=sys.stderr)
        return 1

    print("\nImageNet-1K classification")
    print(f"Source: {args.image}")
    print(f"Device: {device} | Inference: {elapsed_ms:.1f} ms\n")
    for rank, (label, confidence) in enumerate(predictions, start=1):
        bar = "█" * round(confidence / 4)
        print(f"{rank:>2}. {label:<24} {confidence:6.2f}%  {bar}")
    print("\nNote: confidence is relative to the model's 1,000 training classes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
