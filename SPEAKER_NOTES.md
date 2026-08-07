# ImageNet POC — Speaker Notes

**Target duration:** 6 minutes

**Language:** English

**Before presenting:** Open `slides/index.html` and run the demo once so the model weights and sample image are cached.

## Slide 1 — Teaching Machines to See (0:00–0:35)

“Today I will explain what ImageNet is, why it mattered, and show a small image-classification demo. ImageNet is not a software library or a neural network. It is a labeled image dataset and a benchmark.”

**Transition:** “Let us start with the data.”

## Slide 2 — A Dataset at Unprecedented Scale (0:35–1:25)

“The complete ImageNet database organizes more than 14 million images into 21,841 concepts. The subset most people use is ImageNet-1K: 1,000 classes, about 1.28 million training images, and 50,000 validation images.”

“The key distinction is simple: ImageNet is the dataset. A neural network studies those images during training and stores what it learns in its weights.”

## Slide 3 — The Benchmark That Changed the Field (1:25–2:15)

“ImageNet gave researchers a large, shared benchmark. In 2012, AlexNet demonstrated that deep convolutional networks trained with GPUs could outperform earlier approaches by a large margin.”

“That led to a reusable idea called transfer learning: train a model on a large dataset first, then adapt its learned features to a smaller, specialized task.”

## Slide 4 — From Pixels to Predictions (2:15–3:15)

“Our pipeline loads an image, applies the exact resize and normalization expected by the pretrained weights, runs MobileNetV3, and ranks its predictions.”

“Top-1 means the first prediction is correct. Top-5 means the correct answer appears among the five best predictions. The model we use is compact: its weights file is about 21 megabytes.”

## Slide 5 — Live POC (3:15–4:45)

“TorchVision connects the architecture, preprocessing recipe, and class names through one weights object. The model does not query the ImageNet database during inference. It uses knowledge already encoded in the downloaded weights.”

Switch to the terminal and run:

```bash
python3 demo/imagenet_demo.py
```

Point out the result:

“The highest prediction is Samoyed. Similar-looking classes such as Arctic fox and Pomeranian also appear in the Top-5. The percentages are relative to the model’s 1,000 available classes, not universal certainty.”

## Slide 6 — Useful Baseline, Real Limits (4:45–5:50)

“ImageNet pretraining is useful, but the model has a fixed vocabulary. It must choose among its 1,000 labels, even when none is appropriate. Dataset bias and domain shift can also reduce reliability on real deployment data.”

“The three concepts to remember are: ImageNet is the dataset, MobileNet is the architecture, and the pretrained weights are what the model learned from the dataset. Thank you.”

## Timing adjustment

To stay close to 5 minutes, keep the terminal demo to the default image. To extend toward 7 minutes, classify one additional local image and discuss why the Top-5 changes.
