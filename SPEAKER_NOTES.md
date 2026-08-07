# ImageNet POC — Speaker Notes

**Target duration:** 7 minutes (acceptable range: 5–10 minutes)  
**Language:** English  
**Before presenting:** Open `slides/index.html`, increase browser zoom if needed, and run the demo once so its weights and sample image are cached.

## Slide 1 — Teaching Machines to See (0:00–0:35)

“Today I will explain ImageNet, why it changed computer vision, and show a small working classification demo. The important distinction is that ImageNet is not a software library. It is a large labeled image dataset and a benchmark. The code in the demo uses a neural network that learned from one of its subsets.”

**Transition:** “Let us start with what is actually inside ImageNet.”

## Slide 2 — A Dataset at Unprecedented Scale (0:35–1:20)

“The complete ImageNet database organizes more than 14 million images into 21,841 concepts, called synsets, using the WordNet hierarchy. Most machine-learning discussions actually mean ImageNet-1K: the competition subset with 1,000 classes, about 1.28 million training images, and 50,000 validation images.”

“A synset groups synonyms around one concept. This makes the labels hierarchical rather than just a flat list.”

**Optional detail:** “The official test set contains 100,000 images, but its labels are not the part developers normally use for local evaluation.”

## Slide 3 — The Benchmark That Changed the Field (1:20–2:05)

“ImageNet made it possible to compare visual-recognition systems at a scale that exposed meaningful differences. In 2012, AlexNet showed that deep convolutional networks trained with GPUs could outperform earlier approaches by a large margin. That result helped trigger the modern deep-learning era.”

“The reusable idea is transfer learning: a model first learns general visual patterns on ImageNet, then we adapt it to a smaller, specialized dataset.”

**Transition:** “So what happens when such a model receives one image?”

## Slide 4 — From Pixels to a Prediction (2:05–2:55)

“The pipeline has four steps. First, load the image. Second, resize, crop, and normalize it exactly as expected by the pretrained weights. Third, run a forward pass through the network. Finally, softmax converts the output scores into a distribution over 1,000 labels.”

“For this POC I chose MobileNetV3-Large because it is compact: around 5.5 million parameters and a 21 megabyte weights file. The current default TorchVision weights report 75.3 percent Top-1 and 92.6 percent Top-5 accuracy on ImageNet-1K.”

## Slide 5 — How Success Is Measured (2:55–3:40)

“Top-1 accuracy means the first prediction must exactly match the label. Top-5 accuracy is more forgiving: the correct label can appear anywhere among the five highest scores.”

“Top-5 is useful because categories can be visually similar—for example, several breeds of dog. But neither metric proves that the model understands the scene. It only measures performance against this specific label set and validation data.”

## Slide 6 — Live Demo: 16 Lines That Classify an Image (3:40–5:05)

“Here is the core of the demo. TorchVision connects the model, its preprocessing recipe, and its exact category names through one weights object. That prevents common mistakes such as using the wrong normalization or label order.”

Now switch to the terminal and run:

```bash
python3 demo/imagenet_demo.py
```

Point out the Top-5 labels, confidence values, device, and inference time. Then say:

“The model has never been told that this particular image contains a dog. It is matching learned visual features to the 1,000 ImageNet classes. These percentages are relative to those available classes, so they should not be interpreted as universal certainty.”

If time allows, classify a local image:

```bash
python3 demo/imagenet_demo.py path/to/image.jpg
```

**Fallback if the terminal is unavailable:** Stay on the slide and walk through the highlighted lines from top to bottom.

## Slide 7 — Powerful Baseline, Important Limits (5:05–6:05)

“ImageNet pretraining is a strong baseline, but it has boundaries. A fixed set of labels creates a closed world: the model must choose among its 1,000 options even when none is appropriate. Data collected from the web also carries representation and labeling biases. Performance can drop under domain shift—for example, medical scans or unusual camera conditions.”

“Finally, dataset access and image copyright require care. The safe engineering pattern is to validate on data that represents the real deployment context, inspect failure cases, and add human review when decisions matter.”

## Slide 8 — Three Ideas to Remember (6:05–6:45)

“First, ImageNet provided data and a common benchmark at unprecedented scale. Second, pretrained weights turn that research investment into a practical starting point for new applications. Third, benchmark accuracy is not the same as real-world reliability.”

“The POC is intentionally small, but the same flow—preprocess, infer, rank, validate—is the foundation of many production vision systems. Thank you.”

**If questions are expected:** Keep the terminal ready to try another image and discuss why the predictions change.

## Timing shortcuts

To finish near 5 minutes:

- Omit the optional detail on Slide 2.
- Use only the default image in the demo.
- Give one example, rather than three, on Slide 7.

To extend toward 9–10 minutes:

- Run the demo on a second, out-of-domain image.
- Ask the audience to predict the Top-5 before showing the result.
- Discuss transfer learning and fine-tuning after Slide 3.
