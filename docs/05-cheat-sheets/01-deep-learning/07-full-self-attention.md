---
title: "Cheat Sheet: Scaled Dot-Product Self-Attention"
sidebar_position: 7
---

:::tip[What this sheet covers]
Self- vs cross-attention, Q/K/V projections and shapes, the score matrix, why we scale by √d_k, softmax saturation, the attention matrix and Z = AV, and the code you should be able to write. Full notes: [Full Self-Attention: Scaled Dot-Product Attention Step by Step](../../03-ai-ml/03-deep-learning/06-attention-and-transformers/02-full-self-attention-scaled-dot-product.md).
:::

![Scaled dot-product self-attention](/img/deep-learning/diagrams/self-attention-pipeline.svg)

## 🎤 Answer These Out Loud

<details>
<summary><strong>Write the attention formula and walk through the shapes.</strong></summary>

$\operatorname{softmax}(QK^\top/\sqrt{d_k})V$. $X$ is $n \times d_{model}$; $Q = XW_Q$, $K = XW_K$ are $n \times d_k$, $V = XW_V$ is $n \times d_v$. $QK^\top$ is $n \times n$; scale and row-wise softmax keep $n \times n$ with rows summing to 1; $AV$ is $n \times d_v$. Notebook: $5 \times 4 \to 5 \times 3 \to 5 \times 5 \to 5 \times 5 \to 5 \times 3$. Only $W_Q, W_K, W_V$ are learned (36 numbers in the toy).
</details>

<details>
<summary><strong>Why divide by √d_k? Why not d_k?</strong></summary>

With independent unit-variance entries, $\operatorname{Var}(q \cdot k) = d_k$, so scores grow like $\sqrt{d_k}$ (measured std 1.38 at $d_k = 2$ up to 11.02 at 128; scaled: about 1 at every size). Large scores make softmax near one-hot ([10, 20, 5, 15, 2] → 0.993) with vanishing gradients (Jacobian max 0.183 → 0.0067). Dividing by $d_k$ would leave variance $1/d_k$, giving near-uniform attention. Assumes sensible init and normalized inputs (the notebook's $N(0,1)$ weights give $\operatorname{var}(q) \approx 3.7$ and stay saturated).
</details>

<details>
<summary><strong>Self-attention vs cross-attention?</strong></summary>

Self: $Q$, $K$, $V$ from the same sequence; tokens attend within it (encoders, GPT). Cross: $Q$ from the decoder, $K$/$V$ from the encoder output (encoder-decoder models such as the original Transformer and T5).
</details>

<details>
<summary><strong>Why Q, K, V instead of using X directly?</strong></summary>

Three roles (search, be found, contribute) need different representations. $XX^\top$ would be symmetric and mostly self-matching. Separate learned projections give an asymmetric, learnable score matrix: $S_{good,not} = -8.66$ but $S_{not,good} = 4.21$ in the toy.
</details>

<details>
<summary><strong>How do you read S and A?</strong></summary>

Rows = query tokens (who asks), columns = key tokens (who is compared). $S$ is raw and unbounded (−8.73 to 22.47 in the toy; compare within a row only). $A$ is row-wise softmax: each row sums to 1; columns don't (0.41 to 1.82). $z_i = \sum_j A_{ij} v_j$.
</details>

<details>
<summary><strong>In the toy run "good" gives "not" 0.0003. Did attention fail?</strong></summary>

No: the embeddings and $W$s are random and untrained. Training learns $W_Q$, $W_K$ so useful pairs score high, with several heads capturing different relations. The toy shows the mechanism and shapes only.
</details>

<details>
<summary><strong>Why can't you subtract x_good from z_good?</strong></summary>

$x_{good} \in \mathbb{R}^{d_{model}} = \mathbb{R}^4$, $z_{good} \in \mathbb{R}^{d_v} = \mathbb{R}^3$. Compare $z_{good}$ with $v_{good}$ (same space). In a full layer, $W_O$ projects back to $d_{model}$ and the residual gives $x + W_O z$.
</details>

<details>
<summary><strong>Why multi-head attention?</strong></summary>

One head gives one softmax distribution per token, so it compromises between relations (negation, subject–verb, coreference). $h$ heads with $d_k = d_{model}/h$ attend to different relations in parallel; concatenate and project with $W_O$. Original Transformer: $h = 8$, $d_k = 64$, $d_{model} = 512$.
</details>

## 📐 Formulas

| Concept | Formula | In plain words |
|---|---|---|
| Projections | $\pred{Q} = \data{X}\param{W_Q}$, $\pred{K} = \data{X}\param{W_K}$, $\pred{V} = \data{X}\param{W_V}$ | Three learned views |
| Scores | $\pred{S} = \pred{Q}\pred{K}^\top$, $\pred{S_{ij}} = \pred{q_i}^\top\pred{k_j}$ | Every query vs every key |
| Variance | $\operatorname{Var}(\pred{q}^\top\pred{k}) \approx \hyper{d_k}$ | Grows with size |
| Weights | $\pred{A} = \operatorname{softmax}(\pred{Q}\pred{K}^\top/\sqrt{\hyper{d_k}})$, rows sum to 1 | Scale, then normalize rows |
| Output | $\pred{Z} = \pred{A}\pred{V}$, $\pred{z_i} = \sum_j \pred{A_{ij}}\pred{v_j}$ | Weighted average of values |
| Softmax Jacobian | $\pred{A_i}(\delta_{ij} - \pred{A_j})$ | Near zero when saturated |
| Multi-head | $\operatorname{Concat}(\text{head}_1..\text{head}_h)\param{W_O}$ | Heads in parallel |

## 🔢 Numbers Worth Quoting

| Fact | Number |
|---|---|
| Toy setup | seed 42, $n = 5$, $d_{model} = 4$, $d_k = d_v = 3$; $W$ are 4 × 3 |
| One score | $S_{good,not} = q_{good} \cdot k_{not} = -8.661$; $S_{not,good} = 4.207$ |
| Score spread | $S$ from −8.73 to 22.47; std 6.12 → 3.53 after ÷√3 |
| Std of q·k (1000 pairs) | raw 1.38, 2.09, 2.94, 4.00, 5.32, 7.88, 11.02 for $d_k$ = 2…128; scaled 0.94–1.04 |
| Softmax sharpness | max weight 0.242 (small) → 0.632 (medium) → 0.993 (large scores) |
| "good" row | raw softmax [0.002, 0.002, 0.483, 0.000, 0.513] → scaled [0.020, 0.021, 0.471, 0.000, 0.488] |
| $z_{good}$ | [0.7465, −0.0026, −0.7641]; $v_{good}$ = [0.6715, −0.2694, −0.7881] |
| Saturation from init | var(q) 3.72 with $N(0,1)$ weights vs 0.96 with $N(0, 1/d_{model})$ |

## 💻 Code You Should Be Able to Write

```python title="self_attention.py" showLineNumbers
import torch, torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_k, bias=False)

    def forward(self, X, mask=None):                    # X: (batch, n, d_model)
        Q, K, V = self.W_Q(X), self.W_K(X), self.W_V(X)
        scores = Q @ K.transpose(-2, -1) / K.shape[-1] ** 0.5
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))   # before softmax
        # highlight-next-line
        A = torch.softmax(scores, dim=-1)                # over keys: rows sum to 1
        return A @ V, A
```

## 🚨 Traps

- Softmax over `dim=-1` (keys), not over queries.
- Masks as $-\infty$ **before** softmax.
- Weights live in `__init__`; never re-draw them per call (the notebook's function does).
- `nn.Linear.weight` is $(d_{out}, d_{in})$, the transpose of the maths' $W$.
- $QK^\top$ is not symmetric; only $d_k$ must match, $d_v$ is free.
- Read attention from $A$, not from raw $S$.
- √d_k fixes dependence on size, not bad initialization.
- Random weights → meaningless patterns; attention maps are not explanations.
- Cost: $O(n^2 d_k)$ time, $O(n^2)$ memory per head per layer.
