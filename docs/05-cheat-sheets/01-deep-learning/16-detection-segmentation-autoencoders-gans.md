---
title: "Cheat Sheet: Detection, Segmentation, Autoencoders and GANs"
sidebar_position: 16
---

:::tip[What this sheet covers]
The vision-task and generative-model round of the revision question bank on one screen: classification vs detection vs segmentation, boxes, IoU, NMS and mAP, semantic vs instance segmentation, autoencoders (reconstruction, denoising, anomaly detection), VAEs, GANs and mode collapse. Full notes: [Object Detection and Segmentation](../../03-ai-ml/03-deep-learning/08-vision-tasks-and-generative-models/01-object-detection-and-segmentation.md) and [Autoencoders and GANs](../../03-ai-ml/03-deep-learning/08-vision-tasks-and-generative-models/02-autoencoders-and-gans.md).
:::

![The same cat photo three ways: one label, one box with score 0.997, a pixel mask](/img/deep-learning/vision-generative/three-tasks.jpg)

## 🗺️ The Ideas in One Table

| Idea | One line | Number from the runs |
|---|---|---|
| Classification / detection / segmentation | one label / class + score + box per object / class per pixel | DETR: cup 0.9975 on the coffee photo |
| Box formats | $(x_{min}, y_{min}, x_{max}, y_{max})$ or $(c_x, c_y, w, h)$, often normalised | cup (172.4, 17.7, 410.8, 287.1) = (291.6, 152.4, 238.4, 269.4) |
| IoU | overlap ÷ union; a match at 0.5 or more | (0,0,4,4) vs (2,2,6,6): 4/28 = 0.143 |
| NMS | keep best, delete IoU above threshold, repeat | 8 candidates → boxes 0, 4, 7; score threshold 0.5 drops 7 |
| mAP | area under precision-recall, mean over classes (COCO: over IoU 0.5–0.95) | toy AP@0.5 = 0.8667 |
| DETR | 100 queries + "no object", Hungarian matching, no NMS | logits (1, 100, 92), boxes (1, 100, 4) |
| Semantic vs instance | class per pixel vs mask per object | two cats: "cat" 78.3% vs cat 1 37.7% + cat 2 37.9% |
| U-Net | encoder-decoder + concatenated skip connections | (3, 64, 64) → (5, 64, 64) logits |
| Autoencoder | encoder → small latent code → decoder; reconstruction loss | 8-number code: test MSE 0.0145 (PCA 0.0246) |
| Anomaly detection | train on normal, score by reconstruction error | unseen 9s: AUC 0.8836 |
| GAN | generator vs discriminator, balance at D ≈ 0.5 | D(real) 0.511, D(fake) 0.457, D loss 1.343 ≈ 2 ln 2 |
| Mode collapse | realistic but not diverse | 100× faster D: 2 of 8 modes, 93.4% on a mode |

## 🎤 Answer These Out Loud

<details>
<summary><strong>Classification vs detection vs segmentation, with examples?</strong></summary>

Classification: one label for the image ("contains a dog"). Detection: class + score + bounding box per object ("a dog at this box"). Segmentation: a class for every pixel ("these pixels are the dog"); instance segmentation also separates dog 1, dog 2, dog 3.
</details>

<details>
<summary><strong>Explain IoU and NMS.</strong></summary>

IoU = intersection area ÷ union area, 0 to 1; a prediction matches ground truth at IoU ≥ 0.5. NMS sorts boxes by score, keeps the best, deletes boxes whose IoU with it exceeds a threshold and repeats, usually per class. It removes duplicates, not isolated false positives (a score threshold does), and DETR doesn't need it.
</details>

<details>
<summary><strong>How do you evaluate a detector?</strong></summary>

Match predictions to ground truth by IoU in score order (duplicates are false positives), trace precision and recall over all score thresholds, take the area under the curve as AP per class, and average for mAP; COCO also averages over IoU 0.5 to 0.95.
</details>

<details>
<summary><strong>What is an autoencoder and what is it for?</strong></summary>

Encoder compresses the input into a small latent code, decoder rebuilds it, trained only on reconstruction loss (no labels). The bottleneck stops copying. Uses: compression, visualisation, denoising (noisy in, clean target: MSE 0.0524 → 0.0237) and anomaly detection (high reconstruction error = unusual). A plain autoencoder can't generate; a VAE can.
</details>

<details>
<summary><strong>What is a GAN and what is mode collapse?</strong></summary>

A generator turns noise into fakes; a discriminator scores real vs fake; they train in competition, and the generator learns only through the discriminator's gradient. Balance looks like D ≈ 0.5 on both and D loss ≈ 2 ln 2 = 1.386. Mode collapse: the generator produces only a few kinds of output; samples look real but variety is missing (2 of 8 modes with a 100× faster discriminator).
</details>

## 🚨 Traps

- Box format mix-ups (xyxy vs cxcywh vs COCO xywh; pixels vs normalised; $y$ grows down).
- A detection score isn't a probability of being right: DETR said "carrot" 0.8612 on a cat's nose.
- Class-agnostic NMS deletes overlapping boxes of different classes (cup and bowl, IoU 0.9865).
- Transposed conv: kernel 3, stride 2, padding 1 turns 4 into 7; use `output_padding=1` for 8.
- Pixel accuracy misleads with dominant background; use mIoU or Dice ($2\,\text{IoU}/(1 + \text{IoU})$).
- No bottleneck, no learning: an autoencoder that can copy learns nothing.
- Detach fakes in the discriminator step; use the non-saturating generator loss.
- Realistic GAN samples can hide mode collapse; measure diversity (FID).
