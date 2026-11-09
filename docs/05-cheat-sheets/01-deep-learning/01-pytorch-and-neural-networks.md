---
title: "Cheat Sheet: PyTorch, Calculus and Neural Networks"
sidebar_position: 1
---

:::tip[What this sheet covers]
Tensors, autograd and the training loop; the calculus behind backprop; neurons, activations, losses, backpropagation and generalization. Full notes: [PyTorch Fundamentals](../../03-ai-ml/03-deep-learning/01-pytorch-and-neural-networks/01-pytorch-fundamentals.md) · [The Calculus Behind PyTorch](../../03-ai-ml/03-deep-learning/01-pytorch-and-neural-networks/02-calculus-behind-pytorch.md) · [Artificial Neural Networks](../../03-ai-ml/03-deep-learning/01-pytorch-and-neural-networks/03-artificial-neural-networks.md).
:::

## 🎤 Answer These Out Loud

<details>
<summary><strong>What is a tensor, and how is it different from a NumPy array?</strong></summary>

A multi-dimensional array of numbers **plus** the history of how it was computed (for autograd), and it can live on a **CPU or GPU**. NumPy = numbers; tensor = numbers + computation history + GPU acceleration. Three properties matter: **`shape`** (most bugs), **`dtype`** (default `float32`; `torch.tensor(np_array)` keeps `float64`), **`device`** (`cpu`, `cuda:0`, `mps`).
</details>

<details>
<summary><strong>How does autograd work internally?</strong></summary>

It is **reverse-mode automatic differentiation** with a **dynamic (define-by-run)** graph: every operation on a tensor with `requires_grad=True` records a `grad_fn` (`AddBackward0`, `MulBackward0`…) that saves what its local derivative needs. `loss.backward()` walks the graph in reverse from a **scalar**, multiplying upstream gradients by local derivatives and summing where branches meet, and writes results into **leaf** tensors' `.grad`. The graph is freed afterwards. Reverse mode wins because training has one output (the loss) and millions of inputs (the weights).
</details>

<details>
<summary><strong>Why does PyTorch accumulate gradients instead of overwriting them?</strong></summary>

On purpose: when a large batch doesn't fit in GPU memory you can run several micro-batches, let their gradients add up, and call `step()` once, which equals one large batch. The cost is that you must call `zero_grad()` once per update; place it at the start of the iteration or right after `step()`, never between `backward()` and `step()`.
</details>

<details>
<summary><strong>Walk me through forward and backward propagation for a 2-layer network.</strong></summary>

**Forward:** $Z^{(1)} = XW_1^\top + b_1$ → $A^{(1)} = \text{ReLU}(Z^{(1)})$ → $\hat y = A^{(1)}W_2^\top + b_2$ → $L = \text{MSE}(\hat y, y)$; shapes $(N,d) \to (N,h) \to (N,o)$. **Backward:** $\delta_2 = \frac{2}{N}(\hat y - y)$; $\nabla W_2 = \delta_2^\top A^{(1)}$; $\delta_1 = (\delta_2 W_2) \odot \text{ReLU}'(Z^{(1)})$; $\nabla W_1 = \delta_1^\top X$; bias gradients are batch sums. **Update:** $W \leftarrow W - \eta\nabla W$. Pattern: weight gradient = error × what flowed in; previous error = error × weights × activation slope.
</details>

<details>
<summary><strong>Why is the bias gradient a sum over the batch?</strong></summary>

The bias was **broadcast** across the $N$ samples in the forward pass (one $b$ added to every row), so in the backward pass the gradients from all rows flow into the same parameter and **add up**. Broadcast forward ⇒ sum backward.
</details>

<details>
<summary><strong>Compare sigmoid, tanh and ReLU.</strong></summary>

Sigmoid: range $(0, 1)$, slope $\sigma(1-\sigma) \le 0.25$, saturates at both ends, not zero-centred (use for a binary output probability). Tanh: range $(-1, 1)$, slope $1 - \tanh^2 \le 1$, zero-centred, still saturates. ReLU: $\max(0, z)$, slope exactly 1 for $z > 0$ and 0 otherwise; cheap, no saturation on the positive side, but neurons can die. Default in hidden layers: ReLU.
</details>

<details>
<summary><strong>Why can't a linear model fit $y = x^2$, and how does a ReLU network do it?</strong></summary>

A linear model's second derivative is 0 while $x^2$'s is 2, and stacked linear layers are still linear. A ReLU network is **piecewise linear**: each hidden neuron adds one kink at $x = -b_j/w_j$, and the output layer adds the hinges together. More neurons → more kinks → a closer fit.
</details>

<details>
<summary><strong>MSE vs cross-entropy?</strong></summary>

MSE $\frac{1}{N}\sum(\hat y - y)^2$ is for regression and penalizes large errors heavily. Binary cross-entropy $-\frac{1}{N}\sum[y\log\hat y + (1-y)\log(1-\hat y)]$ is for classification: it punishes **confident wrong** predictions very heavily ($-\log$ of a probability near 0 → ∞), which gives strong gradients where MSE would give weak ones. Pairings: identity + MSE, sigmoid + BCE, softmax + CE.
</details>

<details>
<summary><strong>How do you recognize under- and overfitting?</strong></summary>

**Underfitting** (high bias): training and validation loss both high, the model is too simple. **Overfitting** (high variance): training loss low but validation loss rising, the model memorizes noise. Capacity ↑ → bias ↓, variance ↑. Even a tiny network overfits a tiny dataset (5 points vs 97 parameters): capacity is relative to data size.
</details>

<details>
<summary><strong>What does <code>model.eval()</code> do that <code>torch.no_grad()</code> doesn't?</strong></summary>

`model.eval()` changes layer behaviour (dropout off, BatchNorm uses running statistics). `torch.no_grad()` stops building the autograd graph (saves memory and time). They are independent: inference needs **both**, and you must call `model.train()` again before the next training epoch.
</details>

## 📐 Formulas

| Concept | Formula | In plain words |
|---|---|---|
| Linear layer | $\pred{Y} = \data{X}\param{W}^\top + \param{b}$, `weight` is `(out, in)` | Weighted sum of the inputs, plus a shift |
| One neuron | $\pred{z} = \param{w}^\top\data{x} + \param{b}$, $\pred{a} = \phi(\pred{z})$ | Weigh, sum, shift, bend |
| Linear layers collapse | $\param{W_2}(\param{W_1}\data{x}+\param{b_1})+\param{b_2} = \param{W'}\data{x} + \param{b'}$ | No activation, no depth |
| Derivative | $\frac{df}{dx} = \lim_{h \to 0}\frac{f(x+h) - f(x)}{h}$ | Slope over a tiny step |
| Rules | $(x^n)' = nx^{n-1}$, $(uv)' = u'v + uv'$, $(e^x)' = e^x$, $(\ln x)' = 1/x$ | |
| Chain rule (one path) | $\frac{dz}{dx} = \frac{dz}{dy}\cdot\frac{dy}{dx}$ | Multiply local slopes |
| Chain rule (several paths) | $\frac{\partial f}{\partial a} = \sum_{\text{paths}}\prod \text{local slopes}$ | Add every route |
| Backprop step | node gradient = upstream gradient × local derivative | What every `grad_fn` does |
| MSE | $\loss{L} = \frac{1}{N}\sum(\pred{\hat y} - \data{y})^2$ | Average squared miss |
| BCE | $\loss{L} = -\frac{1}{N}\sum[\data{y}\log\pred{\hat y} + (1-\data{y})\log(1-\pred{\hat y})]$ | Punish confident wrong answers |
| Gradient descent | $\param{\theta} \leftarrow \param{\theta} - \hyper{\eta}\,\loss{\nabla_\theta L}$ | Small step downhill |
| GD on $(w - 3)^2$ | distance × $(1 - 2\hyper{\eta})$ per step | Below 1 converges, above 1 diverges |
| Linear model gradients | $\frac{\partial \loss{L}}{\partial \param{w}} = 2(\pred{\hat y} - \data{y})\data{x}$, $\frac{\partial \loss{L}}{\partial \param{b}} = 2(\pred{\hat y} - \data{y})$ | Error × input, and error |
| Sigmoid | $\sigma(z) = \frac{1}{1 + e^{-z}}$, $\sigma' = \sigma(1 - \sigma) \le 0.25$ | Squash to (0, 1) |
| Tanh / ReLU slopes | $1 - \tanh^2 z \le 1$; ReLU' $\in \lbrace 0, 1 \rbrace$ | |
| Output error | $\loss{\delta_2} = \frac{2}{N}(\pred{\hat y} - \data{y})$ | How wrong, which direction |
| Weight gradient | $\partial\loss{L}/\partial\param{W} = \loss{\delta}^\top(\text{layer input})$ | Error × what came in |
| Pass error back | $\loss{\delta_1} = (\loss{\delta_2}\param{W_2}) \odot \phi'(\pred{Z^{(1)}})$ | Through weights, gated by the slope |
| Bias gradient | $\partial\loss{L}/\partial\param{b} = \sum_i \loss{\delta_i}$ | Broadcast forward, sum backward |
| Matrix layer | $\frac{\partial \loss{L}}{\partial \param{W_{ij}}} = g_i x_j$, $\frac{\partial \loss{L}}{\partial \data{x}} = \param{W}^\top g$ | Upstream × input; back through the transpose |
| Softmax + CE | $\partial \loss{L} / \partial \pred{z} = \pred{p} - \data{y}$ | Predicted minus true |
| Vanishing bound | $\prod_\ell \sigma'(z_\ell) \le 0.25^L$ | Sigmoid slopes shrink the signal |

## 🔢 Numbers to Quote

- Derivative of $x^2$ at 3 by shrinking $h$: 7, 6.1, 6.001 → **6**.
- $f = (a+b)ab + ab$ at $a = 2, b = 3$: $c = 5$, $d = 6$, $e = 30$, $f = 36$; $\partial f/\partial a = 24$, $\partial f/\partial b = 18$ (running `backward()` twice gives 48 and 36: accumulation).
- Sigmoid neuron $x = 1, w = 0.5, b = 0, y = 1$: $z = 0.5$, $a = 0.6225$, $L = 0.1425$, $\partial L/\partial w = -0.1774$.
- GD on $(w - 3)^2$ from 0: $\eta = 0.1$ → 0.6, 1.08, 1.464 …; $\eta = 1.0$ bounces 0, 6, 0, 6; $\eta = 1.1$ → 6.6, −1.32, 8.18 (diverges).
- Notebook learning-rate experiment: too small leaves loss 0.11 after 500 epochs; 0.05–0.2 reaches 0.0012; ~1.0 kills all 16 ReLUs (loss frozen at 33.6); 2.0 → `nan`.
- Linear regression run (true $w = 3$, $b = 2$): loss 351 → 49 → 7.9 → 2.2 → ~1.2.

## ⚖️ Quick Comparisons

| | **`loss.backward()`** | **`optimizer.step()`** |
|---|---|---|
| Does | Computes $\partial L/\partial\theta$ into `.grad` | Changes the weights using `.grad` |
| Changes weights? | No | Yes, the only place |

| | **`model.eval()`** | **`torch.no_grad()`** |
|---|---|---|
| Affects | Dropout, BatchNorm behaviour | Graph building / memory |
| Use at inference | Yes | Yes (both together) |

| Activation | Range | Max slope | Zero-centred | Main risk |
|---|---|---|---|---|
| Sigmoid | (0, 1) | 0.25 | No | Vanishing gradients |
| Tanh | (−1, 1) | 1 | Yes | Saturation |
| ReLU | [0, ∞) | 1 | No | Dead neurons |

| Training symptom | Meaning |
|---|---|
| Divergence (loss → `inf`/`nan`) | LR too high, unscaled inputs |
| Oscillation | LR slightly too high |
| Plateau, gradient norm ≈ 0 | Vanishing gradients, dead ReLUs, bad init, LR too small |

## 💻 Code You Should Be Able to Write

```python title="training_loop.py" showLineNumbers
import torch
import torch.nn as nn

torch.manual_seed(0)
X = torch.linspace(0, 10, 100).unsqueeze(1)                 # (N, 1)
y = 3 * X + 2 + torch.randn_like(X)                         # (N, 1): same shape as preds
model = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(300):
    model.train()
    # highlight-start
    optimizer.zero_grad()                                   # 1. clear old .grad
    preds = model(X)                                        # 2. forward
    loss = criterion(preds, y)                              # 3. scalar loss
    loss.backward()                                         # 4. fill every .grad
    optimizer.step()                                        # 5. update weights
    # highlight-end

model.eval()
with torch.no_grad():                                       # inference: eval() AND no_grad()
    print(round(loss.item(), 2), model(torch.tensor([[4.0]])).item() > 10)   # 1.05 True
```

Idioms to have ready: `x.detach().cpu().numpy()` (tensor → NumPy), `torch.from_numpy(a).float()` (NumPy float64 → float32), `model.to(device)` (in place) vs `x = x.to(device)` (returns a copy), `super().__init__()` first in every `nn.Module`, call `model(x)` never `model.forward(x)`.

## 🚨 Traps

- Forgetting `zero_grad()` → gradients pile up → oscillation/divergence. Intentional accumulation every *k* micro-batches simulates a larger batch.
- `preds` `(N, 1)` minus `y` `(N,)` broadcasts to `(N, N)`: wrong loss, no error. Keep shapes identical (`unsqueeze(1)`).
- `backward()` on a non-scalar needs `gradient=` (or `.mean()`/`.sum()` first); a second `backward()` on the same graph fails (buffers freed).
- `.grad` is `None` on intermediate tensors (use `.retain_grad()`), before the first backward, and after `zero_grad()` (set-to-none).
- Updating weights by hand outside `torch.no_grad()` records the update in the graph.
- Unscaled inputs (0–100 instead of 0–10) blow up the loss at the same LR: normalize features.
- Sigmoid + `BCEWithLogitsLoss` applies sigmoid twice; MSE for classification learns slowly when confidently wrong.
- Never put ReLU (or sigmoid) on a regression output; it restricts the predictions' range.
- Identical (e.g. zero) weight init never breaks symmetry: all hidden neurons stay identical.
- Shuffle before a train/validation split; sorted data gives a misleading validation curve.
- Large LR + ReLU kills neurons; count live ones with `(a1 > 0).any(dim=0)`.
- `.item()` inside a hot GPU loop forces a sync each iteration.
- Float32 finite-difference checks are unreliable; use float64 and central differences (`gradcheck`).
