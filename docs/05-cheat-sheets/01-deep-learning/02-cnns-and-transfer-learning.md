---
title: "Cheat Sheet: CNNs and Transfer Learning"
sidebar_position: 2
---

:::tip[What this sheet covers]
Convolution, output sizes and parameter counts, the full CNN classifier, training and dropout, AlexNet, a real image pipeline (cats vs dogs), evaluation metrics, ResNet and transfer learning. Full notes: [CNN Fundamentals](../../03-ai-ml/03-deep-learning/02-cnns/01-cnn-fundamentals.md) · [CNN Architecture, Training and AlexNet](../../03-ai-ml/03-deep-learning/02-cnns/02-cnn-architecture-training-alexnet.md) · [CNN in Practice: Cats vs Dogs](../../03-ai-ml/03-deep-learning/02-cnns/03-cnn-practical-cats-vs-dogs.md) · [Transfer Learning with ResNet18](../../03-ai-ml/03-deep-learning/02-cnns/04-transfer-learning-resnet18.md).
:::

## 🎤 Answer These Out Loud

<details>
<summary><strong>Why CNNs instead of fully connected networks for images?</strong></summary>

Two problems with FC layers: **parameter explosion** (32×32×3 → 100 neurons = 307,300 params; at 128×128×3 it is 4.9 M; it scales with $H \cdot W$) and **loss of spatial structure** (flattening forgets which pixels were neighbours). Convolution fixes both with **local connectivity** (each output sees a small patch) and **weight sharing** (one kernel reused everywhere): a few hundred parameters, independent of image size, translation-equivariant.
</details>

<details>
<summary><strong>What does a convolution actually compute?</strong></summary>

At each position, element-wise multiply the kernel with the patch under it and sum: a **dot product** $w^\top x$ (plus a bias), across **all input channels**. Slide by the stride and repeat; the grid of results is one **feature map** per filter. It is linear and local, so ReLU is needed between convs. PyTorch actually computes cross-correlation (no kernel flip), which makes no difference for learning.
</details>

<details>
<summary><strong>Output shape and parameter count of <code>Conv2d(C_in, C_out, F, stride=S, padding=P)</code>?</strong></summary>

$(N, C_{out}, \lfloor (H - F + 2P)/S \rfloor + 1, \lfloor (W - F + 2P)/S \rfloor + 1)$. Parameters $C_{out}(C_{in}F^2 + 1)$; a 3×3 filter on 64 channels has 577 params, not 10. Stride, padding and image size don't change the parameter count. "Same" padding: $P = (F - 1)/2$ with $S = 1$. Kernels are odd so they have a centre.
</details>

<details>
<summary><strong>Walk me through a CNN classifier end to end.</strong></summary>

`(N,3,32,32)` → Conv 3→16 + ReLU → Pool → `(N,16,16,16)` → Conv 16→32 + ReLU → Pool → `(N,32,8,8)` → Flatten `(N,2048)` → FC 64 + ReLU (+ dropout) → FC 10 logits. Parameters: conv 448 + 4,640; `fc1` 131,136 (96% of 136,874). Loss: `CrossEntropyLoss` on raw logits with integer labels; its gradient is $p - y$; shared kernels sum their gradient over every position. Regularize with dropout, augmentation, weight decay, early stopping or global average pooling.
</details>

<details>
<summary><strong>How does dropout work, and why is it off at test time?</strong></summary>

During training each activation is zeroed with probability $p$ and survivors are scaled by $1/(1-p)$ (**inverted dropout**), so the expected value is unchanged. It stops neurons co-adapting and approximates an ensemble of thinned networks. At evaluation it is the identity (`model.eval()`), so predictions are deterministic. Typical $p$: 0.5 on FC layers, 0.1–0.3 on conv (prefer `Dropout2d`).
</details>

<details>
<summary><strong>Why was AlexNet a breakthrough?</strong></summary>

ImageNet 2012 (1.2 M images, 1,000 classes): top-5 error ~26% → ~15%. Ingredients: **ReLU** (no saturation, ~6× faster training), **depth** (5 conv + 3 FC), **two GPUs**, **data augmentation**, **dropout** in the FC layers (which held ~94% of ~60 M params). The lesson: deep CNN + large labelled data + GPU compute, features learned rather than hand-designed (SIFT/HOG).
</details>

<details>
<summary><strong>Build an image classifier from a folder of images.</strong></summary>

Labels from file/folder names, check balance → **split first** (stratified 70/15/15, no overlap) → preprocess identically: resize, BGR→RGB, scale 0–1 (or mean/std), HWC→CHW float32 → `Dataset` + `DataLoader` (shuffle train only, augment train only) → small CNN baseline or a pretrained backbone → cross-entropy on logits, Adam, `train()`/`eval()` + `no_grad()`, watch train vs val, keep the best checkpoint → test **once**: confusion matrix, per-class precision/recall/F1.
</details>

<details>
<summary><strong>Precision, recall, F1: when does accuracy lie?</strong></summary>

Precision $TP/(TP+FP)$: when it says positive, how often is it right. Recall $TP/(TP+FN)$: of all real positives, how many it found. F1 is their harmonic mean. Accuracy lies on imbalanced data: 95% cats → "always cat" scores 95% with 0% dog recall. Name the positive class first. In sklearn's confusion matrix rows = actual, columns = predicted: `[[TN, FP], [FN, TP]]`.
</details>

<details>
<summary><strong>What is transfer learning and why does it work?</strong></summary>

Reuse a network trained on a huge dataset as a feature extractor: $f_{new}(X) = g_\phi(h_{\theta^*}(X))$. Early layers learn universal features (edges, textures, colour blobs) that transfer to any image task; later layers are more specific. Replace the head, freeze some or all of the backbone, train on your data. Cats vs dogs: scratch ~81% → pretrained ResNet18 ~98%.
</details>

<details>
<summary><strong>Feature extraction or fine-tuning?</strong></summary>

**Feature extraction**: freeze everything, train only the new head (1,026 params for 2 classes); fast, almost can't overfit, best for small data similar to ImageNet (98.19%). **Fine-tuning**: also unfreeze the deepest block(s) (`layer4` + `fc` = 8,394,754 params) with a ~10× smaller LR (1e-4), ideally after training the head first, to avoid catastrophic forgetting (98.74%). Unfreeze top-down; unfreeze more the further your domain is from ImageNet (X-rays, satellite).
</details>

## 📐 Formulas

| Concept | Formula | In plain words |
|---|---|---|
| FC layer params | $d_{out}(d_{in} + 1)$, $d_{in} = H \cdot W \cdot C$ | Grows with the image |
| Conv output pixel | $\pred{Y}_{i,j} = \sum_{c,u,v}\param{K}_{c,u,v}\data{X}_{c,\,iS+u,\,jS+v} + \param{b}$ | Kernel · patch |
| Output size | $\lfloor (\hyper{W} - \hyper{F} + 2\hyper{P})/\hyper{S} \rfloor + 1$ | Positions that fit |
| Same padding | $\hyper{P} = (\hyper{F} - 1)/2$, $\hyper{S} = 1$ | Keep the size |
| Conv params | $\hyper{C_{out}}(\hyper{C_{in}}\hyper{F}^2 + 1)$ | Independent of image size |
| Max pooling 2×2 | $\pred{Y}_{i,j} = \max \data{X}_{2i+u, 2j+v}$ | Halve H, W; channels and params unchanged |
| Receptive field | $r_\ell = r_{\ell-1} + (\hyper{F_\ell} - 1)\prod_{i<\ell}\hyper{S_i}$ | How much of the image one unit sees |
| MACs | $H_{out}W_{out}\hyper{C_{out}}\hyper{C_{in}}\hyper{F}^2$ | Few params, lots of compute |
| Global average pooling | $\pred{g_c} = \frac{1}{HW}\sum_{i,j}\pred{A_{c,i,j}}$ | One number per map |
| Softmax / CE | $\pred{p_i} = e^{\pred{z_i}}/\sum_j e^{\pred{z_j}}$, $\loss{\mathcal L} = -\log\pred{p_{true}}$ | Probabilities; surprise at the truth |
| Starting loss | $\approx \ln C$ (2.30 for 10 classes) | Uniform guess |
| CE gradient | $\pred{p} - \data{y}$ | Predicted minus true |
| Shared-kernel gradient | $\partial\loss{\mathcal L}/\partial\param{K} = \sum_{i,j}\partial\loss{\mathcal L}/\partial\pred{Y_{ij}} \cdot \data{X_{patch(i,j)}}$ | Every position votes |
| Dropout (train) | $\pred{\tilde h} = \data{m} \odot \pred{h}/(1 - \hyper{p})$, $\mathbb E[\pred{\tilde h}] = \pred{h}$ | Zero some, boost the rest |
| Standardize | $\data{x'} = (\data{x} - \mu_c)/\sigma_c$ | ImageNet mean/std for pretrained models |
| Batches per epoch | $\lceil N/\hyper{B} \rceil$ | 8,400 / 64 → 132 |
| Tensor memory | $N \cdot C \cdot H \cdot W \cdot 4$ bytes | float32 |
| Precision / recall / F1 | $\frac{TP}{TP+FP}$, $\frac{TP}{TP+FN}$, $2\frac{PR}{P+R}$ | |
| Residual block | $\pred{y} = F(\data{x};\param{W}) + \data{x}$, $\partial\pred{y}/\partial\data{x} = \partial F/\partial\data{x} + I$ | Identity keeps gradients alive |
| Transfer learning | $f_{new}(\data{X}) = g_{\param{\phi}}(h_{\param{\theta^*}}(\data{X}))$ | Pretrained extractor + new head |
| L2 / weight decay | $\loss{L} + \frac{\hyper{\lambda}}{2}\sum\param{w}^2$; $\param{w} \leftarrow (1 - \hyper{\eta\lambda})\param{w} - \hyper{\eta}\nabla\loss{L}$ | Shrink every weight |
| L1 | $\loss{L} + \hyper{\lambda}\sum\lvert\param{w}\rvert$, gradient $\hyper{\lambda}\,\text{sign}(\param{w})$ | Pushes weights to exactly 0 |

## 🔢 Numbers to Quote

| Fact | Number |
|---|---|
| FC 3,072 → 100 | 307,300 params (128×128×3: 4.9 M) |
| `Conv2d(3, 5, 3)` / `Conv2d(3, 32, 3)` | 140 / 896 params |
| SmallCNN (32×32 input) | 136,874 params; conv 5,088; `fc1` 131,136 (96%) |
| SimpleCNN (128×128 input) | 4,287,938 params; `fc1` 4,194,432 (97.8%); flatten 32,768 |
| ANN vs CNN on CIFAR-size input | 3.08 M vs 82.8 K params |
| AlexNet | ~60 M params, ~94% in FC; 227×227×3 → 6×6×256 = 9,216 features |
| Adam LR | 1e-3 default; 0.1 blew loss from 2.3 to 177 |
| Cats vs dogs data | 12,000 balanced → 8,400 / 1,800 / 1,800; 1.54 GiB if preprocessed eagerly |
| Cats vs dogs results | scratch ~0.81 val; feature extraction 0.9819; fine-tune 0.9874 (train 0.998) |
| ResNet18 | 11,689,512 params (1,000 classes); 11,177,538 with a 2-way head; `layer4` 8,393,728 (72%) |
| GPU vs CPU matmul (4000×4000) | ~14× faster when timed properly (warm-up + `synchronize()`) |

## 📏 Shape Flows

| Model | Flow |
|---|---|
| CNN block | `(1,3,32,32)` → Conv 3→8, p1 → `(1,8,32,32)` → ReLU → Pool → `(1,8,16,16)` |
| SmallCNN | `(N,3,32,32)` → `(N,16,32,32)` → `(N,16,16,16)` → `(N,32,16,16)` → `(N,32,8,8)` → `(N,2048)` → `(N,64)` → `(N,10)` |
| SimpleCNN | `(64,3,128,128)` → 32 → 64 → 128 channels at 128 → 64 → 32 → 16 px → `(64,32768)` → `(64,128)` → `(64,2)` |
| ResNet18 | 224 → 112 → 56 (stem) → `layer1` 56 → `layer2` 28 → `layer3` 14 → `layer4` 7×7×512 → avg pool `(512,)` → `fc` |

## ⚖️ Quick Comparisons

| | **Stride-2 conv** | **Max pool 2×2** |
|---|---|---|
| Halves H, W | Yes | Yes |
| Parameters | Yes (learns how to downsample) | None |
| Channels | Can change | Unchanged |

| | **Feature extraction** | **Fine-tuning** |
|---|---|---|
| Trainable | Head only (1,026) | `layer4` + head (8,394,754) |
| Learning rate | ~1e-3 for the head | ~1e-4 (10× smaller) |
| Overfitting risk | Very low | Moderate |
| Best when | Small data, similar to ImageNet | More data or a different domain |

| Dataset size / similarity to ImageNet | Strategy |
|---|---|
| Small + similar | Feature extraction |
| Large or different | Fine-tune more layers, top-down |
| Huge and very different | Possibly train from scratch |

## 💻 Code You Should Be Able to Write

```python title="small_cnn.py" showLineNumbers
import torch
import torch.nn as nn

def conv_out(n, k, s=1, p=0):
    return (n - k + 2 * p) // s + 1

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 8 * 8, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, num_classes))

    def forward(self, x):
        return self.head(self.features(x))                  # raw logits

model = SmallCNN()
x = torch.randn(4, 3, 32, 32)
print(model(x).shape, sum(p.numel() for p in model.parameters()))   # torch.Size([4, 10]) 136874
print(conv_out(32, 3, 1, 1), conv_out(32, 2, 2), conv_out(224, 7, 2, 3))   # 32 16 112
# highlight-next-line
loss = nn.CrossEntropyLoss()(model(x), torch.tensor([0, 1, 2, 3]))   # logits + integer labels
print(round(loss.item(), 1))                                           # 2.2, close to ln 10 = 2.30
```

```python title="transfer_resnet18.py" showLineNumbers
import torch.nn as nn
from torchvision import models

model = models.resnet18(weights=None)        # weights=models.ResNet18_Weights.DEFAULT in practice
for p in model.parameters():
    p.requires_grad = False                  # freeze the backbone
# highlight-next-line
model.fc = nn.Linear(model.fc.in_features, 2)   # new head: trainable by default
count = lambda ps: sum(p.numel() for p in ps)
print(count(model.parameters()), count(p for p in model.parameters() if p.requires_grad))   # 11177538 1026
for p in model.layer4.parameters():
    p.requires_grad = True                   # fine-tuning: unfreeze the last stage too
print(count(p for p in model.parameters() if p.requires_grad))   # 8394754
```

Preprocessing pipeline from memory: `cv2.imread` (check for `None`) → `cv2.resize(img, (W, H))` (**width first**) → `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` → `/ 255.0` → `np.transpose(img, (2, 0, 1))` → float32 tensor; for pretrained models `transforms.Normalize(ImageNet mean, std)` after `ToTensor()`.

## 🚨 Traps

- HWC (OpenCV/PIL/NumPy) vs NCHW (PyTorch): `permute(2, 0, 1)` / `np.transpose`, `unsqueeze(0)` for one image.
- Always **floor** the output size: (8 − 3)/2 + 1 = 3.5 → 3.
- Flatten size mismatch (`mat1 and mat2 shapes cannot be multiplied`) when the input size changes: use a dummy forward pass, `nn.LazyLinear` or global average pooling. `torch.flatten(x)` without `start_dim=1` also flattens the batch.
- Few parameters ≠ cheap: conv layers dominate FLOPs and activation memory.
- Stacked convs without ReLU collapse into one linear filter.
- Forgetting `model.eval()` leaves dropout on; forgetting `model.train()` afterwards leaves it off.
- Loss far above $\ln C$ at the start → a bug; stuck at $\ln C$ → nothing learned. Overfit one small batch first.
- OpenCV is BGR; `cv2.imread` returns `None` silently; `cv2.resize` takes `(width, height)`.
- Model on GPU, data on CPU → `Expected all tensors to be on the same device`.
- Re-running a notebook training cell continues training: rebuild the model and optimizer.
- Pretrained model with wrong preprocessing (no ImageNet normalization, BGR, 0–255) silently costs accuracy.
- Augment training data only; never label-changing augmentations (flipping text/digits).
- `Normalize` before `ToTensor` fails.
- Forgetting to replace `fc` trains a 1,000-way classifier without any error.
- Frozen BatchNorm still updates running statistics in `train()` mode.
- Save the best weights with `copy.deepcopy(model.state_dict())`; a plain `state_dict()` keeps changing.
- `Adam(weight_decay=…)` ≠ decoupled weight decay: use `AdamW`. `ReduceLROnPlateau.step(val_loss)` needs the metric.
- `ImageFolder` assigns labels alphabetically: check `class_to_idx`.
- Train accuracy below validation with heavy augmentation is normal.
- `torch.load(..., weights_only=False)` on untrusted files can execute code.
