# ImageNet and Ergonomic Webcam POC — Speaker Notes

**Target duration:** 6–8 minutes

**Format:** about 2½ minutes of theory, 1½ minutes of ImageNet classification, and 2–3 minutes of ergonomic webcam demonstration

**Before presenting:** Confirm camera permission, cache the MobileNet weights, and keep both demo commands open in separate terminal tabs.

## Slide 1 — How Machines Learn to See (0:00–0:25)

“I will briefly explain ImageNet and pretrained models, then show two POCs: classification of one image and a practical ergonomic webcam assistant. The final slide connects the webcam behavior back to its code.”

## Slide 2 — ImageNet Is the Textbook (0:25–1:15)

“ImageNet is not a neural network. It is a labeled image dataset and a benchmark. The complete database contains about 14.2 million images organized into 21,841 concepts.”

“Most developers use ImageNet-1K, a subset with 1,000 categories and about 1.28 million training images. The dataset supplies examples; a model is the system that learns from them.”

## Slide 3 — Training Turns Examples into Weights (1:15–2:20)

“MobileNetV3 is the architecture: the mathematical structure of the network. During training, it predicts labels, measures errors, and adjusts millions of numeric weights.”

“The final weights contain reusable visual patterns. Our demo downloads only those weights, not the ImageNet database.”

“AlexNet demonstrated the impact of deep networks and GPUs on ImageNet in 2012. MobileNet came later with an architecture optimized for efficient inference. Both can have ImageNet-trained weights, but neither one is ImageNet itself.”

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

## Slide 5 — POC 2: Ergonomic Webcam Monitor (3:50–6:40)

Run:

```bash
python3 demo/ergonomic_webcam_monitor.py --break-minutes 0.25
```

Sit comfortably and press `C`. Move closer and tilt your head until the warnings appear.

Explain the code shown on the slide:

1. “MediaPipe processes each frame and returns facial landmarks.”
2. “The app selects the left-eye and right-eye coordinates.”
3. “The distance between the eyes provides a relative estimate of viewing distance.”
4. “The angle of the eye line provides head tilt.”
5. “Calibration stores the normal eye distance for this user. Measurements are smoothed and must remain outside the threshold before an alert appears.”

“This POC does not use ImageNet. It uses a model specialized in facial geometry. Processing is local and frames are not stored. It is an ergonomic aid, not a medical device or calibrated distance sensor.”

Conclude:

“ImageNet classification answers what is visible. Face landmarks answer where facial points are. Useful computer-vision systems select the model that matches the real question.”

## Timing options

For a five-minute version, shorten the theory and demonstrate only one posture warning. For a longer version, classify a second local image and discuss the Top-5 results.

## Demo recovery

- If the camera is busy, close conferencing applications or retry with `--camera 1`.
- If an ImageNet label is unexpected, use it to explain closed-set classification.
- If the face monitor fails, improve frontal lighting and center the face.
- If no GUI can be shown, use the final slide to explain landmarks, geometry, calibration, and alerts.
