# ImageNet and Live Webcam — Speaker Notes

**Target duration:** 7–9 minutes

**Format:** about 3½ minutes of history and theory, 1½ minutes of still-image classification, and 2–3 minutes of live webcam classification

**Before presenting:** Confirm camera permission, cache the MobileNet weights, and keep both demo commands open in separate terminal tabs.

## Slide 1 — ImageNet (0:00–0:25)

“I will briefly explain ImageNet and pretrained models, then show the same idea on one image and on a live webcam. Both examples use MobileNetV3 weights trained on ImageNet-1K.”

## Slide 2 — Why ImageNet Was Created (0:25–1:35)

“ImageNet was introduced at CVPR in 2009 by Jia Deng and colleagues at Princeton. At the time, datasets such as Caltech-101 had relatively few classes and examples. They were useful, but too small and controlled to represent the enormous variation of objects in real photographs.”

“Meanwhile, the web already contained billions of images. The problem was how to organize and label them reliably enough for research. The team used WordNet as a semantic hierarchy. For each concept, or synset, they collected candidate images from several search engines, expanded and translated the search terms, removed duplicates, and asked people on Amazon Mechanical Turk to verify whether each image really matched the concept.”

“The 2009 paper described 3.2 million images. By 2014, ImageNet contained about 14.2 million annotated images across 21,841 concepts. ImageNet is the dataset—the organized examples—not a neural network.”

## Slide 3 — What Changed (1:35–2:45)

“In 2010, the ImageNet Large Scale Visual Recognition Challenge, or ILSVRC, turned part of the database into a standard competition. Its best-known classification task uses 1,000 categories and roughly 1.2 million training images. This subset became known as ImageNet-1K.”

“In 2012, AlexNet combined a deep convolutional network, GPUs, ReLU activations, and dropout. It reached 15.3 percent Top-5 error in the challenge, compared with 26.2 percent for the second-best entry. That gap changed the direction of computer vision: engineered features gave way to features learned directly from data.”

“The next challenge was deployment. Large networks were accurate but expensive. MobileNet introduced efficient operations for constrained hardware, and MobileNetV3, published in 2019, combined hardware-aware architecture search with design improvements. It provided Large and Small variants for different resource budgets.”

## Slide 4 — Training Turns Examples into Weights (2:45–3:35)

“MobileNetV3 is the architecture: the mathematical structure of the network. During training, it predicts labels, measures errors, and adjusts millions of numeric weights.”

“The final weights contain reusable visual patterns. Our demo downloads only those weights, not the ImageNet database.”

“The important distinction is: ImageNet supplies labeled examples; MobileNetV3 defines the network architecture; training converts those examples into numeric weights; inference uses the weights to predict labels for new images.”

“Our demo downloads only pretrained weights and the list of 1,000 categories. It does not download or retrain on the ImageNet dataset.”

## Slide 5 — Classify One Image (3:35–4:50)

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

## Slide 6 — Inside the Live Classifier (4:50–5:50)

Explain the code shown on the slide before opening the camera:

1. “MobileNetV3 Small is loaded once with pretrained ImageNet-1K weights.”
2. “The focus square isolates the object and the official weights transform applies the expected resize, crop, and normalization.”
3. “The network returns 1,000 scores, one for each ImageNet-1K category.”
4. “Softmax and Top-k turn those scores into the three labels shown on screen.”
5. “To keep the app light, inference runs every five frames and predictions are smoothed between updates.”

“MobileNetV3 Small has about 2.5 million parameters and a roughly 10 MB weights file. Processing is local and webcam frames are not stored.”

## Slide 7 — Live Demo (5:50–7:40)

Run:

```bash
python3 demo/imagenet_webcam.py
```

Hold one familiar object inside the focus square, such as a mug, keyboard, banana, backpack, or water bottle. Read the Top-3 rather than only the first answer. Then swap it for a second object and discuss what changed.

“If a prediction looks wrong, remember that this is closed-set classification. The model cannot invent a new label: it must choose among the 1,000 categories it learned from ImageNet-1K. Background, framing, lighting, and how much of the object is visible all affect the answer.”

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
