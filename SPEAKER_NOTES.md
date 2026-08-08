# Computer Vision POCs — Speaker Notes

**Target duration:** 7–9 minutes

**Format:** approximately 2 minutes of slides and 5 minutes of live demos

**Before presenting:** Run all demos once, confirm camera permission, cache the MobileNet weights, and keep the three commands open in separate terminal tabs.

## Slide 1 — From a Dataset to a Useful System (0:00–0:25)

“This session is demo-first. I will use ImageNet to establish one concept, then show two webcam applications that are closer to real products: an object monitor and an ergonomic assistant.”

## Slide 2 — Keep the Layers Separate (0:25–1:10)

“ImageNet is a labeled dataset. MobileNetV3 is a neural-network architecture. Training the architecture on ImageNet produces weights: millions of numbers that encode what the model learned.”

“Our first script is only a baseline. It receives one image and returns ranked labels. The next POC turns that same model into a continuous system.”

### Optional baseline — 30 seconds

```bash
python3 demo/imagenet_demo.py
```

“The model does not query ImageNet at runtime. It uses the downloaded weights.”

## Slide 3 — POC 1: Webcam Object Monitor (1:10–3:45)

Start with the normal real-time view:

```bash
python3 demo/webcam_object_monitor.py
```

Hold one clear object near the camera. Explain:

“Classification describes the complete frame, so this works best like a visual intake station: one prominent object at a time. Inference runs in a background thread, allowing the camera window to remain responsive. The displayed result is averaged over recent predictions, which reduces label flicker.”

Press `Q`, then demonstrate the monitored workflow with an object whose ImageNet label you observed:

```bash
python3 demo/webcam_object_monitor.py --watch "coffee mug" --min-confidence 15
```

“The alert requires the target to appear in several inference updates. It then saves a timestamped frame and waits through a cooldown. That pattern could prototype an intake checkpoint, inventory event, or presence alert.”

Mention the limitation:

“This classifier cannot draw a box around each object. A production system with several objects would replace it with an object detector, while keeping much of the surrounding pipeline.”

## Slide 4 — POC 2: Ergonomic Webcam Assistant (3:45–6:20)

```bash
python3 demo/ergonomic_webcam_monitor.py --break-minutes 0.25
```

Sit comfortably and press `C`. Then move closer to the camera and tilt your head until the alerts appear.

“This POC does not use ImageNet. It uses a specialized face-landmark model. The distance between the eyes provides a relative estimate of viewing distance, and the line between the eyes provides head tilt.”

“It calibrates to the current user, smooths measurements across frames, and waits before raising an alert. All frames are processed locally and are not stored.”

Press `B` to reset the accelerated break timer.

“This is an ergonomic aid, not a calibrated physical measurement or medical device.”

## Slide 5 — A Model Is Only One Component (6:20–7:10)

“The useful behavior did not come from a model alone. It also required camera handling, asynchronous inference, smoothing, thresholds, persistence, user feedback, and clear limitations.”

“ImageNet is valuable because it provides reusable visual knowledge. Real applications often combine that knowledge with specialized models—or replace classification entirely when the business question is about location, motion, or geometry.”

## Timing options

For a five-minute version, skip the static baseline and run only the webcam object monitor. For a nine-minute version, run all three scripts and invite the audience to choose an object for the watch mode.

## Demo recovery

- If the camera is busy, close conferencing applications and retry with `--camera 1`.
- If an ImageNet label is unexpected, use it as an example of closed-set classification and domain limitations.
- If the face monitor does not detect a face, improve frontal lighting and move into the center of the frame.
- If a GUI cannot be shown, explain the pipeline using Slides 3 and 4; both contain the exact commands.
