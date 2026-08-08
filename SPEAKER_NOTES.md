# ImageNet and Webcam POCs — Speaker Notes

**Target duration:** 7–9 minutes

**Format:** about 2½ minutes of theory, 4–5 minutes of demos, and 1 minute explaining the code

**Before presenting:** Confirm camera permission, cache the MobileNet weights, and keep both webcam commands open in separate terminal tabs.

## Slide 1 — How Machines Learn to See (0:00–0:25)

“I will briefly explain the theory behind ImageNet and pretrained models, then spend most of the session on two webcam POCs. At the end, we will connect the live behavior back to the code.”

## Slide 2 — ImageNet Is the Textbook (0:25–1:15)

“ImageNet is not a neural network. It is a labeled image dataset and a benchmark. The complete database contains about 14.2 million images organized into 21,841 concepts.”

“Most developers use ImageNet-1K, a subset with 1,000 categories and about 1.28 million training images. The dataset supplies the examples; a model is the system that learns from them.”

## Slide 3 — Training Turns Examples into Weights (1:15–2:20)

“MobileNetV3 is the architecture: the mathematical structure of the network. During training, it repeatedly predicts labels, measures its errors, and adjusts millions of numeric weights.”

“The final weights contain reusable visual patterns. Our demo downloads only those weights, not the ImageNet database.”

“AlexNet demonstrated the impact of deep networks and GPUs on ImageNet in 2012. MobileNet came later with an architecture optimized for efficient inference. Both can have ImageNet-trained weights, but neither one is ImageNet itself.”

## Slide 4 — Two Questions, Two Models (2:20–2:55)

“The first POC asks what is visible in the complete frame. MobileNet returns ImageNet labels and confidence scores.”

“The second asks where parts of the face are. MediaPipe returns coordinates for facial landmarks. That is why it can measure relative distance and head tilt. Classification identifies; landmarks locate and track.”

## Live Demo 1 — Webcam Object Monitor (2:55–5:10)

```bash
python3 demo/webcam_object_monitor.py
```

Hold one clear object near the camera and explain:

“The camera can produce around 30 frames per second, but CPU inference is slower. The application therefore keeps camera capture on the main thread and classifies frames in a background worker.”

Press `Q`, then demonstrate a monitored workflow using a label observed in the first run:

```bash
python3 demo/webcam_object_monitor.py --watch "coffee mug" --min-confidence 15
```

“Predictions are averaged across recent frames. An alert requires repeated matches and has a cooldown, reducing false triggers and duplicate evidence.”

“Because this is classification, it describes the complete frame. A production application that must locate several objects should use an object detector.”

## Live Demo 2 — Ergonomic Webcam Assistant (5:10–7:10)

```bash
python3 demo/ergonomic_webcam_monitor.py --break-minutes 0.25
```

Sit comfortably and press `C`. Move closer and tilt your head until the warnings appear.

“This POC uses MediaPipe rather than ImageNet. It estimates viewing distance from the apparent distance between the eyes and calculates tilt from the angle of the eye line. Measurements are smoothed and must remain outside the threshold before an alert appears.”

“Processing is local and frames are not stored. This is an ergonomic aid, not a medical device or calibrated distance sensor.”

## Slide 5 — Inside the Webcam Monitor (7:10–8:20)

Return to the final slide and explain the four important lines:

1. “The weights object links the pretrained model, preprocessing recipe, and ImageNet class names.”
2. “`worker.submit(frame)` sends a copy of the newest camera frame to the inference thread.”
3. “`worker.latest()` lets the UI reuse the last completed prediction without blocking.”
4. “The history average reduces flicker, and the watcher applies confidence, stability, and cooldown rules before saving evidence.”

Conclude:

“The model provides perception, but the useful POC comes from combining it with camera handling, concurrency, smoothing, thresholds, and user feedback.”

## Timing options

For a five-minute version, omit the ergonomic demo. For a longer version, invite the audience to choose an object and discuss why ImageNet sometimes returns an unexpected label.

## Demo recovery

- If the camera is busy, close conferencing applications or retry with `--camera 1`.
- If an object label is unexpected, use it to explain closed-set classification.
- If the face monitor fails, improve frontal lighting and center the face.
- If no GUI can be shown, use Slides 4 and 5 to explain the two pipelines and code.
