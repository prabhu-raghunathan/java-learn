---
title: "Object Detection and Segmentation: From One Label to Every Pixel"
sidebar_position: 1
---

import {BoxNmsLab} from '@site/src/components/Widgets';

:::tip[TL;DR (30-Second Interview Pitch)]
**Image classification** gives one label for the whole image ("cat"). **Object detection** finds every object and gives each a **class, a confidence score and a bounding box** (4 numbers). **Segmentation** labels **every pixel**: **semantic** segmentation says which class each pixel belongs to, **instance** segmentation also says which object, and **panoptic** does both for the whole image. Detectors are judged with **IoU** (overlap ÷ union of two boxes), clean up duplicate boxes with **non-maximum suppression (NMS)** and are scored with **mAP**. Segmenters use an **encoder-decoder** (U-Net style) that shrinks the image to understand it and grows it back to label pixels. On this page a real DETR detector finds the coffee cup with score 0.9975, also reports a "carrot" on a cat's nose with 0.8612 (a confident false positive), and a real Mask2Former model splits two cats that semantic segmentation calls simply "cat" (78.3% of pixels) into cat 1 (37.7%) and cat 2 (37.9%).
:::

<FormulaKey />

:::info[Where this page sits]
**Before:** [Transfer Learning with ResNet-18](../02-cnns/04-transfer-learning-resnet18.md): CNNs that classify a whole image.
**This page:** the computer-vision part of the **Module III revision question bank** (questions 41–43, awareness level in the course), expanded into full notes and checked with real pretrained models: DETR (`facebook/detr-resnet-50`) for detection and Mask2Former (`facebook/mask2former-swin-small-coco-panoptic`) for segmentation, on two scikit-image sample photos.
**Next:** [Autoencoders and GANs](./02-autoencoders-and-gans.md), the other half of the same revision round.
:::

## 🧭 Start Here

The CNN pages of this course answer one question per image: *what is in it?* A classifier looks at a photo of a cat and outputs "cat". Real applications usually need more. A self-driving car must know **where** each pedestrian is, not just that the scene contains pedestrians. A medical tool must outline the exact **pixels** of a tumour. A photo app must cut out one person from a group. Each of these asks for a richer output than a single label, and the three standard tasks are defined by how rich that output is.

This page walks up that ladder. It starts by comparing the three tasks (Chapter 1), then explains the building blocks of detection: how a box is written down (Chapter 2), how two boxes are compared with **IoU** (Chapter 3), how detectors are built (Chapter 4), what a real detector actually outputs (Chapter 5), how duplicate boxes are removed with **NMS** (Chapter 6) and how detectors are scored with **mAP** (Chapter 7). The second half is segmentation: how a network can output a label per pixel with an **encoder-decoder** (Chapter 8), and the difference between **semantic, instance and panoptic** segmentation (Chapter 9).

Every number comes from a real run on a CPU (PyTorch 2.15, torchvision 0.29, transformers 5.22.2): the detections and masks are the real outputs of pretrained models on scikit-image's `chelsea` (a cat) and `coffee` (a cup on a saucer) sample photos, and a "two cats" image made by placing the cat photo next to its mirror image. IoU, NMS and AP values are computed with `torchvision.ops` and NumPy.

| New concept | One-line meaning |
|---|---|
| **Object detection** | Find every object: class + score + bounding box |
| **Bounding box** | The smallest axis-aligned rectangle around an object (4 numbers) |
| **IoU** | Intersection over union: how much two boxes or masks overlap (0 to 1) |
| **Anchor / proposal** | A candidate box the detector refines |
| **NMS** | Non-maximum suppression: keep the best box, delete its near-duplicates |
| **Precision / recall** | Share of detections that are right / share of objects that are found |
| **AP, mAP** | Area under the precision-recall curve; averaged over classes (and IoU thresholds) |
| **Semantic segmentation** | A class label for every pixel |
| **Instance segmentation** | A separate mask for every object, even of the same class |
| **Panoptic segmentation** | Semantic for "stuff" (sky, road) + instance for "things" (cars, cats) |
| **Encoder-decoder (U-Net)** | Downsample to understand, upsample to label pixels, with skip connections |
| **Transposed convolution** | A learned upsampling layer that makes feature maps larger |

:::analogy[Three levels of describing a photo]
Classification is a friend saying "there's a cat in your photo". Detection is the friend circling the cat with a marker. Segmentation is the friend carefully colouring in exactly the cat's pixels, and instance segmentation is using a different colour for each cat.
:::

## 1. Three Tasks: Classification, Detection, Segmentation

**Image classification** maps a whole image to one label (or one probability per class). Its output is a vector of $C$ scores for $C$ classes, and it says nothing about where the object is or how many there are. It is the right tool when the image is "about" one thing: is this X-ray normal, is this product photo a shoe or a bag.

**Object detection** outputs a *list*, one entry per object found, and each entry has three parts: a **class** ("cup"), a **confidence score** (0.9975) and a **bounding box** (four numbers locating it). The list can be empty or contain many objects of many classes, so the output size varies from image to image. That makes detection harder than classification in two ways: the model must answer *what* and *where* at the same time, and it must decide *how many* objects there are, which is why duplicate removal and score thresholds appear later.

**Image segmentation** outputs a label for **every pixel**, so its output has the same height and width as the image. A box around a cat also contains background; a segmentation mask contains only cat pixels, so it is the most detailed spatial answer. **Semantic segmentation** labels pixels by class only; **instance segmentation** also separates individual objects; **panoptic segmentation** combines both (Chapter 9).

![The same cat photo three ways: classification says cat for the whole image; detection draws a box labelled cat with score 0.997; segmentation colours exactly the cat's pixels](/img/deep-learning/vision-generative/three-tasks.jpg)

*How to read it:* left to right, the output gets more detailed: one word, then a rectangle, then a pixel-exact region. The box (middle) is DETR's real output; the mask (right) is Mask2Former's real cat segment covering 82.6% of the image.

| Task | Question it answers | Output for one image | Example output |
|---|---|---|---|
| Classification | What is in the image? | $C$ class scores | "cat" |
| Object detection | What objects, and where? | $N$ × (class, score, 4 box numbers), $N$ varies | "cat 0.9965 at (0.3, 0.0, 426.6, 299.9)" |
| Semantic segmentation | Which class is each pixel? | $H \times W$ class map | every cat pixel = "cat" |
| Instance segmentation | Which object is each pixel? | one $H \times W$ mask per object | cat 1 mask, cat 2 mask |

:::tip[🎯 Interview angle]
**"What is the difference between image classification, object detection and image segmentation?"** Classification predicts one label for the whole image ("this image contains a dog"). Detection predicts each object's class and location as a bounding box ("a dog at this box"), so it handles many objects. Segmentation predicts a class for every pixel ("these exact pixels are the dog"), the most detailed output; instance segmentation additionally separates individual objects of the same class.
:::

## 2. Bounding Boxes and How They Are Written

A **bounding box** is the smallest axis-aligned rectangle that contains an object. "Axis-aligned" means its sides are horizontal and vertical, so four numbers describe it. Image coordinates start at the **top-left corner**, with $x$ growing to the right and $y$ growing **downward** (the opposite of a school graph), and they are measured either in pixels or as fractions of the image width and height.

There are two common ways to write the same four numbers. **Corner format** (`xyxy`) gives the top-left and bottom-right corners, $(x_{min}, y_{min}, x_{max}, y_{max})$; Pascal VOC, torchvision and most evaluation code use it. **Centre format** (`cxcywh`) gives the centre point and the size, $(c_x, c_y, w, h)$; YOLO and DETR predict boxes this way, usually **normalised** to 0–1 by the image size, because a centre and a size are easier for a network to regress than two corners that must stay ordered. COCO annotation files use a third variant, $(x_{min}, y_{min}, w, h)$. Mixing up formats is one of the most common detection bugs: the numbers look plausible but every box is wrong.

<Formula name="Box format conversion" tag="xyxy and cxcywh" words="The centre is the midpoint of the two corners; width and height are the differences. Going back, each corner is the centre minus or plus half the size." remember="Centre is the average, size is the difference.">

$$
\begin{aligned}
c_x &= \tfrac{x_{min} + x_{max}}{2}, \quad c_y = \tfrac{y_{min} + y_{max}}{2} \\
w &= x_{max} - x_{min}, \quad h = y_{max} - y_{min}
\end{aligned}
$$

- $(x_{min}, y_{min})$: top-left corner; $(x_{max}, y_{max})$: bottom-right corner
- $(c_x, c_y)$: box centre; $w, h$: box width and height; divide by the image width and height to normalise

**Example:** DETR's cup box in corner format is (172.4, 17.7, 410.8, 287.1) on the 600 × 400 coffee photo. In centre format that is (291.6, 152.4, 238.4, 269.4), and normalised (0.486, 0.381, 0.397, 0.674), exactly the raw numbers DETR's box head produced before post-processing.

</Formula>

![One box in two formats on the 600 by 400 coffee image, and the IoU of two boxes: overlap 4, union 28, IoU 0.143](/img/deep-learning/diagrams/box-formats-iou.svg)

*How to read it:* the left half shows the cup box with its two corners (purple) and its centre (red), and the three ways of writing it. The right half is the IoU example from the next chapter.

A detector therefore has to answer two questions for each object, as the course notes put it: **where is it** (the 4 box numbers, a *regression* problem) and **what is it** (the class probabilities, a *classification* problem). That is why detection losses have two parts (Chapter 4).

:::danger[Interview Trap: "a box is four numbers, the format doesn't matter"]
`(172.4, 17.7, 410.8, 287.1)` read as `cxcywh` would describe a completely different box. Always state the format, whether coordinates are in pixels or normalised, and whether $y$ grows downward. Converting with a library (`torchvision.ops.box_convert`) avoids hand-written mistakes.
:::

:::tip[🎯 Interview angle]
**"What is a bounding box and how do you represent it?"** It is the smallest axis-aligned rectangle around an object, described by four numbers: either corners $(x_{min}, y_{min}, x_{max}, y_{max})$ or centre and size $(c_x, c_y, w, h)$, often normalised to 0–1. A detector predicts these box numbers (regression) together with class probabilities (classification) for every object.
:::

## 3. IoU: Measuring How Well Two Boxes Agree

To train or evaluate a detector you constantly need to answer: *is this predicted box the same object as that ground-truth box?* The standard measure is **intersection over union (IoU)**, also called the Jaccard index. Take the area where the two boxes overlap (the intersection) and divide it by the area covered by either box (the union). Identical boxes give 1, boxes that don't touch give 0, and partial overlaps fall in between.

IoU is used everywhere in detection. **Evaluation:** a prediction counts as a correct detection (a *true positive*) only if its IoU with a ground-truth box of the same class is at least a threshold, traditionally **0.5** (Pascal VOC), while COCO averages over thresholds from 0.5 to 0.95. **Training:** anchors or proposals are labelled "object" or "background" by their IoU with ground truth. **Post-processing:** NMS deletes boxes whose IoU with a better box is too high (Chapter 6). The same formula works for **masks** in segmentation, counting pixels instead of box area (Chapter 9).

<Formula name="Intersection over union (IoU)" tag="Box and mask overlap" words="Overlap area divided by the area covered by either box. One means a perfect match, zero means no overlap." remember="Overlap over everything covered.">

$$
\operatorname{IoU}(\data{A}, \pred{B}) = \frac{\lvert \data{A} \cap \pred{B} \rvert}{\lvert \data{A} \cup \pred{B} \rvert} = \frac{\lvert \data{A} \cap \pred{B} \rvert}{\lvert \data{A} \rvert + \lvert \pred{B} \rvert - \lvert \data{A} \cap \pred{B} \rvert}
$$

- $\data{A}$: the ground-truth box; $\pred{B}$: the predicted box; $\lvert \cdot \rvert$: area
- the intersection of two corner-format boxes has width $\max(0, \min(x_{max}) - \max(x_{min}))$, and the same for height

**Example:** $\data{A} = (0, 0, 4, 4)$ and $\pred{B} = (2, 2, 6, 6)$ overlap in a 2 × 2 square: $\operatorname{IoU} = 4 / (16 + 16 - 4) = 4/28 = 0.143$. Sliding $\pred{B}$ only halfway still gives a low IoU: $(10, 10, 50, 50)$ vs $(20, 20, 60, 60)$ is 0.391, below 0.5.

</Formula>

The second example is worth remembering: two 40 × 40 boxes shifted by only a quarter of their size in each direction already fall below 0.5. IoU is strict, which is why a detector's box regression has to be precise, not just roughly in the right place.

:::analogy[Two people highlighting the same paragraph]
Two students highlight what they think is the key passage. IoU is the part they both highlighted divided by everything either of them highlighted. If one highlights a whole page and the other one sentence, they "agree" on that sentence, but the IoU is tiny because the union is huge.
:::

:::tip[🎯 Interview angle]
**"What is IoU and where is it used?"** Intersection area divided by union area of two boxes (or masks), from 0 to 1. It decides whether a prediction matches a ground-truth object during evaluation (typically IoU ≥ 0.5, or averaged over 0.5–0.95 for COCO), assigns anchors to objects during training, and drives NMS. Two equal boxes shifted by a quarter of their width diagonally already score only 0.39.
:::

## 4. How Detectors Are Built

A classifier ends in one vector of class scores. A detector needs a variable number of (class, box) outputs, and the three families of detectors differ in how they produce them. All of them start the same way: a **CNN backbone** (a ResNet, for example, pretrained on ImageNet) turns the image into feature maps, the same feature maps the classification pages built.

**Two-stage detectors** (R-CNN, Fast R-CNN, Faster R-CNN) first propose regions that might contain an object, then classify each one. In Faster R-CNN a small **region proposal network** slides over the feature map and, at each position, scores a set of **anchors** (reference boxes of a few sizes and aspect ratios) as "object or not" and adjusts them. The best few hundred proposals are cropped from the feature map (**RoI pooling / RoI Align** turns each into a fixed-size grid) and a second head predicts the final class and a refined box. Two passes make it accurate and comparatively slow. **One-stage detectors** (YOLO, SSD, RetinaNet) skip the proposal step: they predict a class and a box offset for **every anchor at every grid position** in a single pass. That is fast, but it produces thousands of overlapping candidates, most of them background, so they rely on score thresholds and NMS, and (RetinaNet) on the **focal loss**, which down-weights the many easy background examples.

**DETR** (DEtection TRansformer) treats detection as **set prediction**. After the CNN backbone, a Transformer encoder-decoder uses a fixed number of learned **object queries** (100 in `detr-resnet-50`); each query outputs one class (including a special **"no object"** class) and one box. During training, the predictions are matched one-to-one to the ground-truth objects with the **Hungarian algorithm**, so two queries can't both be rewarded for the same cat. That matching is why DETR doesn't need NMS. The real model's output shapes show this design: class logits $(1, 100, 92)$ = 100 queries × (91 COCO class ids + 1 "no object"), and boxes $(1, 100, 4)$ in normalised $c_x c_y w h$. It has 41,524,768 parameters.

![Three detector families: two-stage proposes regions then classifies each; one-stage predicts a box for every anchor then needs NMS; DETR predicts a set of 100 query outputs and needs no NMS](/img/deep-learning/diagrams/detector-families.svg)

*How to read it:* each row is a pipeline from image features (blue) to final detections; grey boxes are post-processing. Only the one-stage and two-stage rows need NMS to remove duplicates.

<Formula name="Detection loss" tag="What and where" words="A detector is trained on two things at once: a classification loss for what each box contains and a box regression loss for where it is, counted only for boxes matched to a real object." remember="Class loss plus box loss.">

$$
\loss{L} = \loss{L_{cls}}(\pred{p}, \data{c}) + \hyper{\lambda} \cdot \mathbb{1}[\data{c} \neq \varnothing] \cdot \loss{L_{box}}(\pred{\hat b}, \data{b})
$$

- $\pred{p}$: predicted class probabilities; $\data{c}$: the true class ($\varnothing$ = background / no object)
- $\pred{\hat b}, \data{b}$: predicted and true box; $\loss{L_{box}}$ is smooth-L1 (Faster R-CNN) or L1 + generalised IoU (DETR)
- $\hyper{\lambda}$: how much the box term weighs; the indicator means background predictions get no box loss

**Example:** an anchor that overlaps no object is trained only to say "background"; there is no correct box to regress it toward, so its box loss is switched off.

</Formula>

| Family | Examples | How boxes appear | Duplicates | Speed vs accuracy |
|---|---|---|---|---|
| Two-stage | R-CNN, Faster R-CNN, Mask R-CNN | Proposals, then refine and classify | NMS | Accurate, slower |
| One-stage | YOLO, SSD, RetinaNet | Dense predictions per anchor / grid cell | NMS + score threshold | Fast, real-time |
| Set prediction | DETR, Deformable DETR | 100 queries, one-to-one matching | None by design | Simple pipeline, slow to train |

:::tip[🎯 Interview angle]
**"How does an object detector produce a variable number of boxes?"** By predicting a fixed, large set of candidates and filtering. Two-stage detectors propose regions and classify each; one-stage detectors predict a class and box for every anchor and remove duplicates with NMS; DETR predicts 100 query outputs, one of which can be "no object", and is trained with one-to-one Hungarian matching so it needs no NMS. All are trained with a classification loss plus a box regression loss.
:::

## 5. What a Real Detector Outputs

Running DETR on the coffee photo makes the abstract design concrete. The model returns 100 predictions no matter what is in the image; each has a probability over the 91 classes plus "no object". The post-processing step takes, for each query, the highest real-class probability as its **score**, converts the normalised box to pixels and keeps the predictions above a **score threshold**. With a threshold of 0.5 the coffee photo yields three detections: cup 0.9975, dining table 0.9925 and spoon 0.9796. Lowering the threshold lets more through: 3 queries score above 0.9, 4 above 0.1 and 6 above 0.01. The fourth is a second "spoon" at 0.2931 covering the lower half of the same spoon, and the fifth a "bowl" at 0.0436 sitting on almost exactly the cup's box (IoU 0.9865 with it).

The cat photo shows the other classic error. DETR finds the cat at 0.9965, and then reports a **carrot** at **0.8612**, a small box on the cat's orange nose. A confident false positive like this is a reminder that a detection score is the model's belief, not a probability of being right, and that the threshold has to be chosen on validation data for the use case. On the two-cats image, the two real cats come out at 0.9961 and 0.9958, followed by weaker overlapping "cat" boxes at 0.1628 and 0.0711 that span both cats.

![DETR detections on the cat photo (scores above 0.5: cat 0.997 and a false carrot 0.861 on the nose) and on the coffee photo (above 0.1: cup, dining table, spoon 0.980 and a weaker duplicate spoon 0.293)](/img/deep-learning/vision-generative/detr-detections.jpg)

*How to read it:* every rectangle is a real DETR output, labelled with its class and score. The left image uses a 0.5 threshold, the right one 0.1, which is why the weaker duplicate spoon appears.

| Image | Detections (score, class) | Lesson |
|---|---|---|
| coffee, threshold 0.5 | cup 0.9975, dining table 0.9925, spoon 0.9796 | Large "stuff-like" objects (table) count too |
| coffee, threshold 0.1 | + spoon 0.2931 (part of the same spoon) | Low thresholds admit partial duplicates |
| cat, threshold 0.5 | cat 0.9965, carrot 0.8612 | Confident false positive on the nose |
| two cats, threshold 0.5 | cat 0.9961, cat 0.9958 | Two objects of one class, two boxes |

:::danger[Interview Trap: "the score is the probability the box is correct"]
A detection score is the model's class confidence for that query or anchor. It isn't calibrated, it says nothing about how well the box fits, and it can be high for a wrong class (carrot 0.8612). Choose the threshold on validation data, per use case, trading missed objects against false alarms.
:::

:::tip[🎯 Interview angle]
**"Your detector outputs 100 boxes per image. How do you get the final detections?"** Take each prediction's best class and score, drop predictions below a score threshold tuned on validation data, convert boxes to pixel coordinates, and (for anchor-based detectors) remove duplicates with NMS. On a real DETR run, a 0.5 threshold turned 100 queries into 3 detections for the coffee photo, but still kept a confident false "carrot" on the cat photo.
:::

## 6. Non-Maximum Suppression: One Box per Object

Anchor-based detectors fire many times on the same object: neighbouring anchors all overlap the cat, and each refines into a slightly different box with a high score. **Non-maximum suppression (NMS)** reduces each cluster to one box with a simple greedy rule. Sort the boxes by score. Take the highest-scoring box and **keep** it. Delete every remaining box whose IoU with it is above a threshold (typically 0.5), because they are probably the same object. Repeat with the highest-scoring box still left, until none remain.

In the toy example below, a detector outputs 8 candidate boxes for two cats. NMS keeps box 0 (score 0.95) and deletes boxes 1, 2 and 3, whose IoUs with it are 0.84, 0.79 and 0.75. It then keeps box 4 (0.92) and deletes boxes 5 and 6 (IoU 0.84 and 0.81). Box 7 (score 0.40), a stray box between the cats, overlaps box 0 by only 0.17 and box 4 by 0.15, so NMS **keeps** it: NMS removes duplicates, not false positives. A **score threshold** applied first (0.5 here) removes it. The threshold matters in both directions: at 0.9 nothing overlaps enough and all 8 boxes survive; at a very low value, two genuinely different but overlapping objects (people in a crowd) would wrongly be merged into one.

<Animation src="vision/iou-then-nms" caption="NMS on 8 candidate boxes for two cats. The best box is kept (green), boxes overlapping it with IoU above 0.5 are deleted (red), and the loop repeats; the weak stray box 7 survives because it overlaps nothing much." />

NMS can be run **per class** or **class-agnostic**. torchvision's `batched_nms` only compares boxes of the same class, which is usually right (a person on a bicycle should keep both boxes). On the coffee photo it keeps both DETR's cup (0.9975) and the "bowl" (0.0436) even though their boxes overlap with IoU 0.9865, because they are different classes; class-agnostic `nms` deletes the bowl. Variants such as **Soft-NMS** lower the scores of overlapping boxes instead of deleting them, which helps in crowded scenes.

<BoxNmsLab />

| Step | NMS action | Toy example (IoU threshold 0.5) |
|---|---|---|
| 1 | (Optional) drop boxes below the score threshold | threshold 0.5 drops box 7 (0.40) |
| 2 | Sort the remaining boxes by score | 0, 4, 1, 5, 2, 3, 6 (7) |
| 3 | Keep the best box | keep box 0 (0.95) |
| 4 | Delete boxes with IoU above the threshold with it | delete 1, 2, 3 (IoU 0.84, 0.79, 0.75) |
| 5 | Repeat with the best box left | keep 4; delete 5, 6 (IoU 0.84, 0.81) |
| 6 | Stop when no boxes remain | kept: 0, 4 (and 7 without step 1) |

:::analogy[Choosing one photo from a burst]
Your phone takes 8 shots of the same moment. You keep the sharpest one and delete the others that show the same scene. A shot of something different in the corner of the room isn't a duplicate, so you keep it too, even if it's blurry: deciding whether it's worth keeping at all is a separate question (the score threshold).
:::

:::tip[🎯 Interview angle]
**"Explain non-maximum suppression."** Sort detections by score, keep the best, delete the remaining boxes whose IoU with it exceeds a threshold (about 0.5), and repeat. It removes duplicate detections of the same object, usually per class. It doesn't remove isolated false positives (a score threshold does), and a threshold that's too low merges nearby objects, which is why crowded scenes use Soft-NMS or NMS-free detectors like DETR.
:::

## 7. Evaluating Detectors: Precision, Recall and mAP

Accuracy doesn't make sense for detection, because there is no fixed number of predictions to compare. Instead, each prediction is matched to the ground truth: sorted by score, a prediction is a **true positive (TP)** if its IoU with an unmatched ground-truth box of the same class is at least 0.5, otherwise it's a **false positive (FP)** (including a second, duplicate box on an already-matched object). Ground-truth objects nobody matched are **false negatives**. From these, **precision** = TP ÷ (TP + FP) is "how many of my detections are right", and **recall** = TP ÷ (number of ground-truth objects) is "how many objects did I find".

Lowering the score threshold trades precision for recall, so detectors are summarised over **all** thresholds at once. Walk down the score-sorted predictions, computing precision and recall after each one: that traces the **precision-recall curve**. **Average precision (AP)** is the area under it (after making precision monotonically decreasing, the "interpolated" curve). **mAP** averages AP over all classes; COCO's headline metric, written AP@[.5:.95], also averages over 10 IoU thresholds from 0.5 to 0.95, rewarding tightly fitting boxes.

<Formula name="Average precision (AP)" tag="Detection metric" words="Sort predictions by score, compute precision and recall after each one, make precision non-increasing from the right, and add up precision times each step in recall." remember="Area under the precision-recall curve.">

$$
\begin{aligned}
\text{precision}_k &= \frac{\mathrm{TP}_k}{k}, \qquad \text{recall}_k = \frac{\mathrm{TP}_k}{\data{N_{gt}}} \\
\mathrm{AP} &= \sum_{k} (r_{k} - r_{k-1}) \cdot \max_{j \geq k} \text{precision}_j
\end{aligned}
$$

- $\mathrm{TP}_k$: true positives among the top $k$ predictions; $\data{N_{gt}}$: number of ground-truth objects
- $r_k$: recall after $k$ predictions; mAP = mean of AP over classes (and over IoU thresholds for COCO)

**Example:** 3 ground-truth objects, 6 predictions whose matches in score order are TP, TP, FP, FP, TP, FP. Precision runs 1, 1, 0.67, 0.5, 0.6, 0.5 and recall 0.33, 0.67, 0.67, 0.67, 1, 1. AP $= \frac13 \cdot 1 + \frac13 \cdot 1 + \frac13 \cdot 0.6 = 0.8667$.

</Formula>

| Metric | Meaning | Where used |
|---|---|---|
| AP@0.5 | AP with IoU threshold 0.5 | Pascal VOC |
| AP@0.75 | Stricter: boxes must fit tightly | COCO breakdown |
| AP@[.5:.95] (COCO "mAP") | Mean AP over IoU 0.5, 0.55, …, 0.95 | COCO leaderboard |
| AP small / medium / large | AP by object size | Diagnosing small-object failures |

:::tip[🎯 Interview angle]
**"How do you evaluate an object detector?"** Match predictions to ground truth by IoU (≥ 0.5) in score order to get TP and FP, trace the precision-recall curve over all score thresholds, take the area under it as AP per class, and average over classes for mAP; COCO also averages over IoU thresholds 0.5–0.95. Duplicates count as false positives, which is how NMS problems show up in the metric.
:::

## 8. Segmentation Networks: Shrink to Understand, Grow to Label Pixels

A classification CNN repeatedly halves the feature maps with pooling or stride until a small grid of rich features remains, which is ideal for "what is in the image" but throws away *where*. Segmentation needs an output as large as the input, one prediction per pixel. A **fully convolutional network (FCN)** replaces the final fully connected layers with convolutions, so it outputs a coarse grid of class scores, and then **upsamples** it back to the image size. The output tensor is $(N, C, H, W)$: for every pixel, one score per class. The predicted mask is the **argmax** over the class dimension, and training uses **cross-entropy per pixel**, averaged over all pixels.

Upsampling can be fixed (bilinear interpolation) or learned. A **transposed convolution** (sometimes loosely called "deconvolution") is a learned upsampler: with stride 2 it spreads each input value over a 2 × 2 region weighted by a learned kernel, roughly doubling the height and width. **U-Net**, designed for medical images, makes the encoder-decoder symmetric: an **encoder** (contracting path) downsamples to capture *what* is in the image, a **decoder** (expanding path) upsamples back, and **skip connections** copy the encoder's high-resolution feature maps across and **concatenate** them with the decoder's. The deep features say "this is a cat"; the skipped shallow features say exactly where the cat's edges are. Without skips, upsampled masks come out blurry.

![U-Net style encoder-decoder with real shapes: 3 x 64 x 64 input, conv to 16 x 64 x 64, pool and conv to 32 x 32 x 32, transposed conv back to 16 x 64 x 64, concatenated with the skip to 32 x 64 x 64, and a 1x1 conv to 5 class scores per pixel](/img/deep-learning/diagrams/unet-segmentation.svg)

*How to read it:* data flows left to right; the dashed red arrow is the skip connection. Every shape is from a real run of the `TinyUNet` in the implementation section; a real U-Net repeats the down and up steps four times.

<Formula name="Transposed convolution output size" tag="Learned upsampling" words="The reverse of the convolution output-size formula: each step of the input moves the output by the stride, then the kernel adds its own width and padding trims both sides." remember="Input minus one, times stride, plus kernel.">

$$
H_{out} = (H_{in} - 1) \cdot \hyper{S} - 2\hyper{P} + \hyper{F} + \hyper{P_{out}}
$$

- $H_{in}$: input height (same rule for width); $\hyper{S}$: stride; $\hyper{P}$: padding; $\hyper{F}$: kernel size
- $\hyper{P_{out}}$: `output_padding`, an extra row added on one side to resolve size ambiguity

**Example:** from 4 × 4 with $\hyper{F} = 2, \hyper{S} = 2, \hyper{P} = 0$: $(4-1) \cdot 2 + 2 = 8$. With $\hyper{F} = 4, \hyper{S} = 2, \hyper{P} = 1$: $6 - 2 + 4 = 8$. With $\hyper{F} = 3, \hyper{S} = 2, \hyper{P} = 1$: $6 - 2 + 3 = 7$, which is why `output_padding=1` is needed to reach 8.

</Formula>

| Architecture | Idea | Known for |
|---|---|---|
| FCN | Convolutions only, upsample the coarse score map | First end-to-end semantic segmentation |
| U-Net | Symmetric encoder-decoder with concatenated skip connections | Medical images, small datasets |
| DeepLab | Dilated (atrous) convolutions keep resolution; ASPP for multi-scale context | Street scenes |
| Mask R-CNN | Faster R-CNN + a small mask head per detected box | Instance segmentation |
| Mask2Former | Transformer queries, each predicting a class and a mask | One model for semantic, instance and panoptic |

:::danger[Interview Trap: "segmentation is classification applied to each pixel separately"]
The loss is per-pixel cross-entropy, but the network is not run per pixel: one forward pass predicts all pixels together, and each pixel's prediction uses a large receptive field of context. Classifying a pixel from its own colour alone would be hopeless (a grey pixel could be road, sky or a cat).
:::

:::tip[🎯 Interview angle]
**"How does a network output a label for every pixel?"** An encoder downsamples to build semantic features, a decoder upsamples back to the input resolution (bilinear or transposed convolutions), and a 1 × 1 convolution gives $C$ class scores per pixel; the mask is the argmax and training uses per-pixel cross-entropy. U-Net adds skip connections that concatenate high-resolution encoder features into the decoder so boundaries stay sharp.
:::

## 9. Semantic, Instance and Panoptic Segmentation

**Semantic segmentation** assigns each pixel a class and nothing more. With three dogs in a photo, every dog pixel is "dog", and the output can't tell you how many dogs there are or which pixels belong to which. **Instance segmentation** gives each object its own mask, so the three dogs become dog 1, dog 2 and dog 3; it is harder because it combines pixel-level classification with object-level separation (detection). It usually covers only countable **things** (people, cars, animals), not amorphous **stuff** (sky, road, grass). **Panoptic segmentation** unifies the two: every pixel gets a class, and pixels of "things" also get an instance id.

The two-cats image shows the difference with real model output. Mask2Former's semantic output labels 78.3% of the pixels "cat", plus curtain (13.95%) and wall (7.76%), as one undifferentiated cat region. Its panoptic output on the same image produces two separate cat segments, cat 1 covering 37.7% of the pixels (score 0.9338) and cat 2 covering 37.9% (score 0.9691), plus a curtain segment. Counting the cats, measuring each one or cutting one out needs the instance output.

![Two cats side by side: the semantic map colours all cat pixels the same blue; the panoptic map colours cat 1 blue and cat 2 orange](/img/deep-learning/vision-generative/semantic-vs-instance.jpg)

*How to read it:* grey regions are background classes (curtain, wall). The middle panel is the semantic output, the right panel the panoptic one, both from the same Mask2Former forward pass.

<Animation src="vision/semantic-vs-instance" caption="The real masks on a grid: first every cat pixel gets the label cat (semantic), then the same pixels split into cat 1 and cat 2 (instance). Only the second answers how many cats there are." />

Masks are compared with the same overlap idea as boxes. **Mask IoU** counts pixels; the **Dice coefficient**, common in medical imaging, is twice the overlap divided by the sum of the two mask sizes. The two always rank predictions the same way, and Dice is always at least as large as IoU. Semantic segmentation reports **mean IoU (mIoU)** over classes; **pixel accuracy** alone is misleading when one class (background) dominates.

<Formula name="Dice coefficient" tag="Mask overlap" words="Twice the overlapping pixels divided by the total pixels in both masks. It ranks predictions like IoU but is always at least as large." remember="Twice the overlap over both sizes.">

$$
\operatorname{Dice}(\data{A}, \pred{B}) = \frac{2\,\lvert \data{A} \cap \pred{B} \rvert}{\lvert \data{A} \rvert + \lvert \pred{B} \rvert} = \frac{2\,\operatorname{IoU}}{1 + \operatorname{IoU}}
$$

- $\data{A}$: ground-truth mask; $\pred{B}$: predicted mask; $\lvert \cdot \rvert$: number of pixels

**Example:** two 5 × 5 masks offset by 2 pixels diagonally overlap in 9 pixels; union 41. IoU $= 9/41 = 0.2195$ and Dice $= 18/50 = 0.36$.

</Formula>

| | Semantic | Instance | Panoptic |
|---|---|---|---|
| Output | One class per pixel | One mask + class per object | Class per pixel + id per object |
| Three dogs | All dog pixels = "dog" | dog 1, dog 2, dog 3 | dog 1, dog 2, dog 3 + grass, sky |
| Covers background stuff? | Yes | Usually no | Yes |
| Typical models | FCN, U-Net, DeepLab, SegFormer | Mask R-CNN | Panoptic FPN, Mask2Former |
| Typical metric | mIoU | mask AP | PQ (panoptic quality) |

:::analogy[Colouring book vs name tags]
Semantic segmentation is colouring every dog in the picture with the same brown crayon. Instance segmentation is giving each dog a different colour and a name tag. Panoptic is doing that and also colouring the grass and the sky.
:::

:::tip[🎯 Interview angle]
**"Semantic vs instance segmentation, with three dogs?"** Semantic segmentation labels every pixel's class, so all three dogs are one "dog" region and you can't count them. Instance segmentation produces a separate mask per object (dog 1, 2, 3), which is harder because it needs object-level separation on top of pixel classification, typically with a detector plus a mask head (Mask R-CNN). Panoptic segmentation combines both for every pixel in the image.
:::

## ⚙️ Going Deeper (Detection and Segmentation Internals)

**Anchors and box parameterisation.** Anchor-based detectors don't regress raw coordinates. They predict offsets relative to an anchor: $t_x = (x - x_a)/w_a$, $t_y = (y - y_a)/h_a$, $t_w = \log(w / w_a)$, $t_h = \log(h / h_a)$. The log makes width changes scale-invariant and keeps predicted sizes positive. Anchor sizes and aspect ratios are hyperparameters, often chosen by clustering the training boxes (YOLOv2).

**Feature pyramids.** Small objects vanish in a heavily downsampled feature map. A **Feature Pyramid Network (FPN)** builds top-down feature maps at several resolutions and assigns small objects to high-resolution levels and large ones to coarse levels. Most modern detectors and Mask R-CNN use one.

**Class imbalance and focal loss.** A one-stage detector evaluates tens of thousands of anchors per image and almost all are background. Plain cross-entropy is dominated by these easy negatives. Focal loss multiplies the cross-entropy by $(1 - p_t)^{\gamma}$, so well-classified examples contribute almost nothing and training focuses on hard ones.

**RoI Align.** Faster R-CNN's RoI pooling rounds box coordinates to the feature grid, which shifts masks by up to a cell. Mask R-CNN's RoI Align samples the feature map with bilinear interpolation at exact positions, which matters for pixel-accurate masks.

**DETR's trade-off.** Hungarian matching removes NMS and hand-designed anchors, but DETR needs long training schedules and is weaker on small objects; Deformable DETR attends only to a few sampled points per query to fix both. The "no object" class is how a fixed set of 100 queries represents a variable number of objects.

**Resolution and the input pipeline.** The DETR processor resized the 600 × 400 coffee photo to 1200 × 800 (and the 451 × 300 cat to 1202 × 800) before inference; Mask2Former used 384 × 384. Boxes and masks must be mapped back to the original size (`target_sizes`) or they won't line up with the image. Segmentation networks output masks at a reduced resolution (Mask2Former's mask logits are 96 × 96) and upsample them.

**Losses for segmentation.** Per-pixel cross-entropy is the default; with heavy class imbalance (a small tumour in a large scan) **Dice loss** ($1 - \text{Dice}$ computed on soft probabilities) or a combination is common, because it measures overlap directly instead of being dominated by background pixels.

## 💻 Python Implementation

:::tip[Run it yourself]
`pip install torch torchvision transformers timm scikit-image`. The blocks use scikit-image's `coffee` sample photo; outputs are from a real run on a CPU (PyTorch 2.15, torchvision 0.29, transformers 5.22.2).
:::

### 1. Run a pretrained detector (DETR)

The model always returns 100 query predictions: class logits over 91 COCO ids + "no object", and a normalised $c_x c_y w h$ box each.

```python title="detect.py" showLineNumbers
import torch
from PIL import Image
from transformers import DetrImageProcessor, DetrForObjectDetection
from skimage import data

image = Image.fromarray(data.coffee())                                  # 600 x 400 sample photo
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").eval()

inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    out = model(**inputs)
# highlight-next-line
print(out.logits.shape, out.pred_boxes.shape)      # torch.Size([1, 100, 92]) torch.Size([1, 100, 4])
```

### 2. Turn 100 queries into detections

Keep each query's best real class above a score threshold and convert boxes to pixel corners.

```python showLineNumbers
# highlight-next-line
result = processor.post_process_object_detection(out, threshold=0.5, target_sizes=[image.size[::-1]])[0]
for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
    print(f"{model.config.id2label[label.item()]:<13} {score.item():.4f}  {[round(v, 1) for v in box.tolist()]}")
# dining table  0.9925  [-0.1, 1.0, 599.8, 395.5]
# cup           0.9975  [172.4, 17.7, 410.8, 287.1]
# spoon         0.9796  [325.1, 65.3, 425.8, 328.2]
```

### 3. Convert box formats and compute IoU

`torchvision.ops` has tested versions of both; `target_sizes` takes (height, width), hence `image.size[::-1]` above.

```python showLineNumbers
from torchvision.ops import box_convert, box_iou

cup = torch.tensor([[172.4, 17.7, 410.8, 287.1]])
print(box_convert(cup, "xyxy", "cxcywh"))          # tensor([[291.6000, 152.4000, 238.4000, 269.4000]])
print(box_iou(torch.tensor([[0., 0., 4., 4.]]), torch.tensor([[2., 2., 6., 6.]])))   # tensor([[0.1429]])
```

### 4. Non-maximum suppression

The 8 candidate boxes from Chapter 6; NMS returns the indices it keeps, best first.

```python showLineNumbers
from torchvision.ops import nms, batched_nms

boxes = torch.tensor([[10, 20, 46, 80], [12, 18, 48, 78], [8, 24, 44, 84], [14, 22, 50, 82],
                      [56, 30, 92, 86], [58, 28, 94, 84], [54, 34, 90, 88], [30, 40, 70, 70]], dtype=torch.float)
scores = torch.tensor([0.95, 0.90, 0.80, 0.60, 0.92, 0.85, 0.55, 0.40])
# highlight-next-line
print(nms(boxes, scores, iou_threshold=0.5))        # tensor([0, 4, 7])   stray box 7 survives NMS
keep = nms(boxes[scores >= 0.5], scores[scores >= 0.5], 0.5)
print(keep)                                         # tensor([0, 4])      a score threshold removes it
```

```python title="❌ Class-agnostic NMS deletes a different class" showLineNumbers
b = torch.tensor([[172.4, 17.7, 410.8, 287.1], [173.5, 17.7, 412.6, 287.5]])   # DETR's cup and bowl, IoU 0.9865
s = torch.tensor([0.9975, 0.0436])
# error-next-line
print(nms(b, s, 0.5))                               # ❌ tensor([0])     the bowl is gone
# success-next-line
print(batched_nms(b, s, torch.tensor([0, 1]), 0.5)) # ✅ tensor([0, 1])  per-class: different labels never suppress each other
```

### 5. Average precision from TP/FP flags

The interpolated AP of Chapter 7: 3 ground-truth objects, 6 predictions in score order.

```python showLineNumbers
import numpy as np

def average_precision(tp, n_gt):
    tp = np.array(tp)
    precision = np.cumsum(tp) / np.arange(1, len(tp) + 1)
    recall = np.cumsum(tp) / n_gt
    p = np.concatenate([[0], precision, [0]]); r = np.concatenate([[0], recall, [1]])
    # highlight-next-line
    for i in range(len(p) - 2, -1, -1):              # make precision non-increasing from the right
        p[i] = max(p[i], p[i + 1])
    idx = np.where(r[1:] != r[:-1])[0]
    return float(np.sum((r[idx + 1] - r[idx]) * p[idx + 1]))

print(round(average_precision([1, 1, 0, 0, 1, 0], n_gt=3), 4))   # 0.8667
```

### 6. A tiny U-Net: per-pixel class scores

Encoder, transposed-convolution decoder and one skip connection; the shapes match the U-Net diagram.

```python title="tiny_unet.py" showLineNumbers
import torch.nn as nn

class TinyUNet(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU())
        self.down = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
        self.up = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.head = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        e = self.enc(x)                              # (N, 16, 64, 64)
        u = self.up(self.down(e))                    # (N, 32, 32, 32) -> (N, 16, 64, 64)
        # highlight-next-line
        return self.head(torch.cat([u, e], dim=1))   # skip connection: concat -> (N, 32, 64, 64) -> (N, 5, 64, 64)

torch.manual_seed(0)
net = TinyUNet()
logits = net(torch.randn(2, 3, 64, 64))
print(logits.shape, logits.argmax(1).shape)          # torch.Size([2, 5, 64, 64]) torch.Size([2, 64, 64])
target = torch.randint(0, 5, (2, 64, 64))            # one class id per pixel
print(round(nn.CrossEntropyLoss()(logits, target).item(), 4))   # 1.6167  (about ln 5 = 1.609 before training)
for k, s, p in [(2, 2, 0), (4, 2, 1), (3, 2, 1)]:
    print(k, s, p, nn.ConvTranspose2d(8, 4, k, s, p)(torch.randn(1, 8, 4, 4)).shape[-1])
# 2 2 0 8
# 4 2 1 8
# 3 2 1 7
```

### 7. Mask IoU and Dice

Two 5 × 5 masks offset by 2 pixels.

```python showLineNumbers
A = np.zeros((10, 10), bool); A[2:7, 2:7] = True
B = np.zeros((10, 10), bool); B[4:9, 4:9] = True
inter, union = (A & B).sum(), (A | B).sum()
print(inter, union, round(inter / union, 4), round(2 * inter / (A.sum() + B.sum()), 4))   # 9 41 0.2195 0.36
```

## 🚨 Interview Gotchas & Edge Cases

:::warning[The Most Common Question]
**"What's the difference between classification, detection and segmentation, and between semantic and instance segmentation?"** Classification: one label per image. Detection: class + score + box per object. Segmentation: a label per pixel. Semantic segmentation can't separate objects of the same class (three dogs are one "dog" region); instance segmentation gives each object its own mask. Be ready to sketch the output shapes: $C$ scores, $N \times (4 + C)$, $C \times H \times W$.
:::

- **Box format mix-ups** (xyxy vs cxcywh vs COCO xywh, pixels vs normalised, $y$ downward) silently produce wrong boxes.
- **IoU is strict**: equal boxes shifted by a quarter of their size diagonally score 0.39, below the usual 0.5.
- **NMS removes duplicates, not false positives**: the stray box 7 survives; only a score threshold drops it.
- **Class-agnostic vs per-class NMS**: per-class kept both DETR's cup and bowl (IoU 0.9865); class-agnostic kept only the cup.
- **NMS threshold too low** merges overlapping real objects (crowds); too high leaves duplicates.
- **DETR needs no NMS**: one-to-one Hungarian matching during training.
- **Scores aren't probabilities of being right**: carrot 0.8612 on a cat's nose.
- **Duplicates count as false positives in mAP**, so bad NMS lowers AP.
- **COCO mAP averages over IoU 0.5–0.95**; AP@0.5 is a much easier number.
- **Pixel accuracy misleads** when background dominates; report mIoU or Dice.
- **Transposed convolution sizes**: kernel 3, stride 2, padding 1 gives 7 from 4, not 8; use `output_padding=1`.
- **Map outputs back to the original image size**: models resize inputs (1200 × 800 for DETR here, 384 × 384 for Mask2Former).

## ⚡ Essential Interview Recall

A one-screen revision list. Every point is explained in the chapters above.

**The three tasks**
- Classification: one label per image. Detection: class + score + box per object. Segmentation: class per pixel.
- Semantic (class per pixel) vs instance (mask per object) vs panoptic (both, whole image).

**Boxes and IoU**
- Corner $(x_{min}, y_{min}, x_{max}, y_{max})$ vs centre $(c_x, c_y, w, h)$, often normalised; $y$ grows down.
- $\operatorname{IoU} = \lvert \data{A} \cap \pred{B} \rvert / \lvert \data{A} \cup \pred{B} \rvert$; match if IoU ≥ 0.5.
- $(0,0,4,4)$ vs $(2,2,6,6)$: 4/28 = 0.143.

**Detectors**
- Two-stage (Faster R-CNN): proposals → RoI pooling → class + box. One-stage (YOLO, SSD): dense anchors, NMS. DETR: 100 queries + "no object", Hungarian matching, no NMS.
- Loss = classification + $\hyper{\lambda}$ × box regression (only for matched boxes).
- Real DETR: logits (1, 100, 92), boxes (1, 100, 4); cup 0.9975; false carrot 0.8612.

**NMS and mAP**
- NMS: sort, keep best, delete IoU above threshold, repeat; per class; doesn't remove isolated false positives.
- Precision = TP/(TP+FP), recall = TP/objects; AP = area under the P-R curve; mAP = mean over classes (COCO: and IoU 0.5–0.95).

**Segmentation**
- Output $(N, C, H, W)$, argmax per pixel, per-pixel cross-entropy.
- Encoder-decoder; transposed conv $H_{out} = (H_{in} - 1)\hyper{S} - 2\hyper{P} + \hyper{F}$; U-Net skip connections keep edges sharp.
- Two cats: semantic "cat" 78.3%; panoptic cat 1 37.7%, cat 2 37.9%.
- Dice $= 2\operatorname{IoU}/(1 + \operatorname{IoU})$; mIoU for semantic, mask AP for instance.

## 📐 Formula Sheet

| Concept | Formula | In plain words |
|---|---|---|
| Box centre and size | $c_x = \tfrac{x_{min} + x_{max}}{2}$, $\ w = x_{max} - x_{min}$ (same for $y$, $h$) | Centre is the average, size the difference |
| IoU | $\dfrac{\lvert \data{A} \cap \pred{B} \rvert}{\lvert \data{A} \rvert + \lvert \pred{B} \rvert - \lvert \data{A} \cap \pred{B} \rvert}$ | Overlap over everything covered |
| Detection loss | $\loss{L} = \loss{L_{cls}} + \hyper{\lambda}\, \mathbb{1}[\data{c} \neq \varnothing]\, \loss{L_{box}}$ | What plus where (box loss only for objects) |
| Precision, recall | $\mathrm{TP}_k / k$, $\ \mathrm{TP}_k / \data{N_{gt}}$ | Right detections / found objects |
| Average precision | $\sum_k (r_k - r_{k-1}) \max_{j \geq k} \text{precision}_j$ | Area under the interpolated P-R curve |
| Transposed conv size | $H_{out} = (H_{in} - 1)\hyper{S} - 2\hyper{P} + \hyper{F} + \hyper{P_{out}}$ | Learned upsampling output size |
| Dice | $\dfrac{2 \lvert \data{A} \cap \pred{B} \rvert}{\lvert \data{A} \rvert + \lvert \pred{B} \rvert} = \dfrac{2\,\operatorname{IoU}}{1 + \operatorname{IoU}}$ | Mask overlap, at least as large as IoU |
| Anchor offsets | $t_x = (x - x_a)/w_a$, $\ t_w = \log(w / w_a)$ | Boxes regressed relative to an anchor |

## Concept-Check Answers (from the question bank)

These are the revision question bank's computer-vision questions (Phase 5, Q41–Q43), answered in full; the bank page has all 51.

<details>
<summary><strong>Q41. What is the difference between image classification, object detection, and image segmentation? Give one example of each.</strong></summary>

Classification predicts one label for the whole image ("this image contains a dog"). Detection identifies each object's class and its location as a bounding box ("there is a dog at this box"), with a confidence score, so it handles many objects per image. Segmentation assigns a class to every pixel ("these exact pixels belong to the dog"). So classification is image-level, detection is object-level with location, and segmentation is pixel-level, the most detailed spatial output (Chapter 1).

</details>

<details>
<summary><strong>Q42. What is a bounding box in object detection? What are common ways to represent it?</strong></summary>

The smallest axis-aligned rectangle that localises an object. It is written as corners $(x_{min}, y_{min}, x_{max}, y_{max})$ or as centre and size $(c_x, c_y, w, h)$, in pixels or normalised to 0–1 (COCO files also use $(x_{min}, y_{min}, w, h)$). A detector predicts both the box (where is the object, a regression) and class probabilities (what is it, a classification). DETR's cup box (172.4, 17.7, 410.8, 287.1) is (291.6, 152.4, 238.4, 269.4) in centre format (Chapter 2).

</details>

<details>
<summary><strong>Q43. What is the difference between semantic segmentation and instance segmentation? Use the example of multiple dogs in one image.</strong></summary>

Semantic segmentation labels every pixel with a class but doesn't separate objects of the same class: with three dogs, all dog pixels are "dog". Instance segmentation gives each object its own mask: dog 1, dog 2, dog 3. Instance segmentation is harder because it needs pixel-level classification and object-level separation together. On the two-cats image, semantic output was one "cat" region (78.3% of pixels) and panoptic output split it into cat 1 (37.7%) and cat 2 (37.9%) (Chapter 9).

</details>
