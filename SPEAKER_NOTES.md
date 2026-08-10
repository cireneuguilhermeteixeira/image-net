# ImageNet and Live Webcam POC — Speaker Notes

**Target duration:** 6–8 minutes

**Format:** about 2½ minutes of theory, 1½ minutes of still-image classification, and 2–3 minutes of live webcam classification

**Before presenting:** Confirm camera permission, cache the MobileNet weights, and keep both demo commands open in separate terminal tabs.

## Slide 1 — ImageNet (0:00–0:25)

“I will briefly explain ImageNet and pretrained models, then show the same idea on one image and on a live webcam. Both POCs use MobileNetV3 weights trained on ImageNet-1K.”

## Slide 2 — ImageNet Is the Textbook (0:25–1:15)

“ImageNet is not a neural network. It is a labeled image dataset and a benchmark. The complete database contains about 14.2 million images organized into 21,841 concepts.”

“Most developers use ImageNet-1K, a subset with 1,000 categories and about 1.28 million training images. The dataset supplies examples; a model is the system that learns from them.”

## Slide 3 — Training Turns Examples into Weights (1:15–2:20)

“MobileNetV3 is the architecture: the mathematical structure of the network. During training, it predicts labels, measures errors, and adjusts millions of numeric weights.”

“The final weights contain reusable visual patterns. Our demo downloads only those weights, not the ImageNet database.”

“AlexNet demonstrated the impact of deep networks and GPUs on ImageNet in 2012. MobileNet came later with an architecture optimized for efficient inference. Both can have ImageNet-trained weights.”

## Slide 4 — POC 1: Classify One Image (2:20–3:50)

Run:

```bash
python3 demo/imagenet_demo.py
```

Explain the code shown on the slide:

1. “The weights object identifies the pretrained checkpoint, preprocessing recipe, and 1,000 class names.”
2. “The image is transformed exactly as the model expects: resized, cropped, converted to values, and normalized.”
3. “The model produces 1,000 raw scores. Softmax converts them into relative probabilities.”
4. “Top-k selects the five highest scores and maps their indexes to ImageNet labels.”

“The model identifies the sample as a Samoyed. Similar white animals appear below it because their visual features overlap. Confidence is relative to the 1,000 available classes, not universal certainty.”

## Slide 5 — POC 2: Live ImageNet Webcam (3:50–6:40)

Run:

```bash
python3 demo/imagenet_webcam.py
```

Hold one familiar object inside the focus square, such as a mug, keyboard, banana, backpack, or water bottle. Then swap it for a second object and watch the Top-3 change.

Explain the code shown on the slide:

1. “MobileNetV3 Small is loaded once with pretrained ImageNet-1K weights.”
2. “The focus square isolates the object and the official weights transform applies the expected resize, crop, and normalization.”
3. “The network returns 1,000 scores, one for each ImageNet-1K category.”
4. “Softmax and Top-k turn those scores into the three labels shown on screen.”
5. “To keep the app light, inference runs every five frames and predictions are smoothed between updates.”

“MobileNetV3 Small has about 2.5 million parameters and a roughly 10 MB weights file. Processing is local and webcam frames are not stored.”

Conclude:

“ImageNet is the labeled visual vocabulary; MobileNetV3 is the efficient architecture; pretrained weights connect them. The webcam turns that pipeline into a live, tangible demonstration.”

## Timing options

For a five-minute version, skip the still-image terminal demo and use only the webcam. For a longer version, compare confusing objects and discuss why closed-set classification must always choose one of its 1,000 labels.

## Demo recovery

- If the camera is busy, close conferencing applications or retry with `--camera 1`.
- If an ImageNet label is unexpected, use it to explain closed-set classification.
- If predictions are unstable, use one well-lit object, fill most of the focus square, and keep the background simple.
- If the computer is slow, restart with `--every 10`.
- If no GUI can be shown, use the final slide to explain the focus crop, preprocessing, inference, smoothing, and Top-3 labels.
