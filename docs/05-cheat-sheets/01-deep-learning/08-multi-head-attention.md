---
title: "Cheat Sheet: Multi-Head Attention"
sidebar_position: 8
---

:::tip[What this sheet covers]
Why one head isn't enough, per-head projections and shapes, Concat and $W^O$, the paper's formula and ablation, efficient implementation and PyTorch gotchas. Full notes: [Multi-Head Attention: Many Attention Patterns in Parallel](../../03-ai-ml/03-deep-learning/06-attention-and-transformers/03-multi-head-attention.md).
:::

![Multi-Head Attention (paper Figure 2, right)](/img/deep-learning/diagrams/paper-figure2-multi-head.svg)

## 🎤 Answer These Out Loud

<details>
<summary><strong>Why multi-head attention instead of one head?</strong></summary>

One head = one softmax distribution per token, so one relation at a time; needing several forces it to average their values ("with a single attention head, averaging inhibits this"). $h$ heads with their own $W_i^Q, W_i^K, W_i^V$ attend to different relations in different subspaces. Trained toy: 2 heads → previous-token head 0.996 and first-token head 0.999; 1 head splits about 0.47/0.51. Paper Table 3 (A): $h = 1$ 24.9 vs $h = 8$ 25.8 BLEU; $h = 32$ 25.4.
</details>

<details>
<summary><strong>Write the formula and the shapes.</strong></summary>

$\operatorname{MultiHead}(Q,K,V) = \operatorname{Concat}(\text{head}_1..\text{head}_h)W^O$, $\text{head}_i = \operatorname{Attention}(QW_i^Q, KW_i^K, VW_i^V)$. $W_i^{Q,K}: d_{model} \times d_k$, $W_i^V: d_{model} \times d_v$, $W^O: hd_v \times d_{model}$; $d_k = d_v = d_{model}/h$ (512/8 = 64). Toy: $X$ 5 × 4 → heads 5 × 2 → Concat 5 × 4 → $W^O$ 4 × 4 → 5 × 4.
</details>

<details>
<summary><strong>Do more heads mean more parameters?</strong></summary>

No: with $d_k = d_{model}/h$, weights are always $4d_{model}^2$ (toy 64; base 1,048,576), the same as one full-width head. More heads = more, narrower patterns; more attention-matrix memory ($h \cdot n^2$).
</details>

<details>
<summary><strong>What do Concat and W^O each do?</strong></summary>

Concat places head outputs side by side along features (no parameters, no mixing: $H_{The}$ = [−1.063, 1.911 | 0.175, −1.367]). $W^O$ mixes features across heads and maps back to $d_{model}$ ($Z_{The}$ = [−0.137, 3.136, −7.395, 0.296]), so residual connections work.
</details>

<details>
<summary><strong>How do libraries implement it?</strong></summary>

One $d_{model} \times d_{model}$ matrix per Q/K/V (heads side by side), reshape (batch, $n$, $d_{model}$) → (batch, $h$, $n$, $d_k$), batched attention, transpose + reshape back (= Concat), then $W^O$. Identical to the per-head loop (difference 0.0). PyTorch stores all three as `in_proj_weight` (3·d × d) and matches NumPy to $8.9 \times 10^{-16}$.
</details>

<details>
<summary><strong>Where does multi-head attention appear in the Transformer?</strong></summary>

Three ways: encoder self-attention (no mask), decoder masked self-attention, encoder–decoder attention (Q from decoder, K/V from encoder). Inside an encoder layer: MHA → Add & Norm → FFN → Add & Norm, shape-preserving, stacked $N = 6$.
</details>

## 📐 Formulas

| Concept | Formula | In plain words |
|---|---|---|
| Per-head projections | $\pred{Q^{(r)}} = \data{X}\param{W_Q^{(r)}}$, etc. | Same X, different lens |
| Head width | $\hyper{d_k} = \hyper{d_{model}}/\hyper{h}$ | Split the width |
| One head | $\operatorname{softmax}(\pred{Q^{(r)}}\pred{K^{(r)}}^\top/\sqrt{\hyper{d_k}})\pred{V^{(r)}}$ | Full attention per head |
| Concat | $\pred{H} \in \mathbb{R}^{n \times \hyper{h}\hyper{d_v}}$ | Side by side |
| Output | $\pred{Z} = \pred{H}\param{W^O}$ | Mix the heads |
| Parameters | $4\hyper{d_{model}}^2$ | Same as one head |

## 🔢 Numbers Worth Quoting

| Fact | Number |
|---|---|
| Toy setup | seed 42, $n$ 5, $d_{model}$ 4, $h$ 2, $d_k$ 2 |
| "good" row | head 1 [0.444, 0.423, 0.010, 0.118, 0.004]; head 2 [0.923, 0.021, 0.053, 0.001, 0.003] |
| Head difference | max $|A^{(1)} - A^{(2)}|$ = 0.807 |
| Trained specialization | 2 heads: 0.996 previous / 0.999 first token; 1 head ≈0.47/0.51 |
| Paper (base) | $h$ 8, $d_k$ 64, MHA weights 1,048,576 |
| Table 3 (A) | $h$ 1: 24.9 · 4: 25.5 · 8: 25.8 · 16: 25.8 · 32: 25.4 |
| PyTorch averaged "good" row | [0.683, 0.222, 0.032, 0.060, 0.003] (matches neither head) |

## 💻 Code You Should Be Able to Write

```python title="multi_head_attention.py" showLineNumbers
import torch, torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        assert d_model % h == 0
        self.h, self.d_k = h, d_model // h
        self.W_q, self.W_k, self.W_v, self.W_o = (nn.Linear(d_model, d_model) for _ in range(4))

    def forward(self, q, k, v, mask=None):              # (batch, n, d_model)
        B = q.size(0)
        split = lambda x: x.view(B, -1, self.h, self.d_k).transpose(1, 2)   # (B, h, n, d_k)
        Q, K, V = split(self.W_q(q)), split(self.W_k(k)), split(self.W_v(v))
        scores = Q @ K.transpose(-2, -1) / self.d_k ** 0.5
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        A = torch.softmax(scores, dim=-1)
        # highlight-next-line
        out = (A @ V).transpose(1, 2).reshape(B, -1, self.h * self.d_k)    # Concat
        return self.W_o(out), A
```

## 🚨 Traps

- Concat doesn't mix heads; $W^O$ does.
- Scale by $\sqrt{d_k}$ of the head, not $\sqrt{d_{model}}$.
- $d_{model}$ must be divisible by $h$.
- Identical head weights = one head (rank of Concat halves).
- `nn.MultiheadAttention` averages weights over heads by default; `average_attn_weights=False` for per-head. `batch_first` defaults to False.
- Random heads differ meaninglessly; which head learns which job is arbitrary.
- Too many heads → $d_k$ too small → worse (Table 3).
