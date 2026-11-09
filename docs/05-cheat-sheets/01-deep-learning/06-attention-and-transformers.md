---
title: "Cheat Sheet: Attention and Why Transformers"
sidebar_position: 6
---

:::tip[What this sheet covers]
Why recurrence became the bottleneck, the seq2seq fixed-context problem, soft alignment and context vectors, Query–Key–Value, reading a self-attention matrix, "Attention Is All You Need" and the Transformer big picture. Full notes: [From LSTM to Attention: Why Transformers Became Necessary](../../03-ai-ml/03-deep-learning/06-attention-and-transformers/01-from-lstm-to-attention-why-transformers.md).
:::

![Recurrence chain vs self-attention](/img/deep-learning/diagrams/chain-vs-attention.svg)

## 🎤 Answer These Out Loud

<details>
<summary><strong>Why did Transformers replace RNNs/LSTMs?</strong></summary>

Recurrence is **sequential** ($h_t$ needs $h_{t-1}$: $O(n)$ dependent steps, GPUs idle), links distant tokens through an **$O(n)$ path**, and squeezes the past into **one fixed-size state**. Self-attention: $O(1)$ sequential steps, $O(1)$ path, all positions in parallel, pure matrix multiplication. Costs: $O(n^2)$ compute/memory in context length and no built-in order. LSTM improved memory; the Transformer redesigned the computation.
</details>

<details>
<summary><strong>What was wrong with the original encoder-decoder, and how did attention fix it?</strong></summary>

The encoder's last state $c = h_T$ was the only thing the decoder saw, identical at every output step: long sentences lost detail, and every word got the same summary. Bahdanau (2014) kept all encoder states and computed per step $e_{t,i} = \text{score}(s_{t-1}, h_i)$, $\alpha_{t,i} = \text{softmax}$, $c_t = \sum_i \alpha_{t,i} h_i$: a fresh context per output word.
</details>

<details>
<summary><strong>Why softmax, and why soft rather than hard attention?</strong></summary>

Softmax makes weights positive, comparable and summing to 1 (a weighted **average**), is differentiable and amplifies score gaps exponentially. Hard selection (argmax) has zero/undefined gradient and keeps one token only. Softmax is shift-invariant (scores + 100 → same weights) and scaling sharpens (×5 → "not" 0.731). It has no parameters: **relevance comes from the scores**, not from softmax.
</details>

<details>
<summary><strong>What is a context vector? How is it different from averaging?</strong></summary>

$c = \sum_i \alpha_i v_i$. With weights 1/n it is the average; with input-dependent weights it emphasises what matters now. "The movie was not good": scores [0.2, 0.4, 0.1, 2.0, 1.8] → weights [0.071, 0.086, 0.064, 0.428, 0.351] → context [0.381, 0.472] vs average [0.28, 0.24]. A learned, per-prediction weighted average.
</details>

<details>
<summary><strong>Explain Query, Key, Value.</strong></summary>

Library: the **query** is your search, **keys** are the index cards, **values** are the book contents. $q = xW_Q$ (what I look for), $k = xW_K$ (what I offer for matching), $v = xW_V$ (what I pass on). Scores $q \cdot k$ → softmax → weights on values. Three projections because searching, being found and contributing are different jobs. "it" in "...because it was tired" matches the key of "animal".
</details>

<details>
<summary><strong>How do you read a self-attention matrix?</strong></summary>

$n \times n$. **Row** = the token asking (a softmax distribution, sums to 1); **column** = the token attended to (no constraint). New vectors $= AV$. Toy: the good row [0.05, 0.05, 0.10, 0.55, 0.25] turns good [1.0, 0.1] into [0.275, 0.58]; equal attention collapses every token to [0.28, 0.24]. Hand-made toy matrices ≠ learned ones; and attention weights are a hint, not an explanation.
</details>

<details>
<summary><strong>What did "Attention Is All You Need" actually claim?</strong></summary>

Vaswani et al. 2017: build the sequence model **without recurrence or convolution**; attention is the backbone, not an add-on (attention already existed, since 2014). Not literal: it still uses embeddings, positional encoding, multi-head attention, feed-forward layers, residuals and layer norm. 28.4 BLEU on WMT14 EN-DE; base of BERT, GPT and LLMs.
</details>

<details>
<summary><strong>Walk through a Transformer layer. Why positional encodings?</strong></summary>

Embeddings $X \in \mathbb{R}^{n\times d}$ + positions → $N$ × [self-attention → add & norm → position-wise feed-forward → add & norm] → contextual vectors. Attention = **communication across tokens**; feed-forward = **transformation within each token**. Self-attention is permutation-equivariant (shuffle in → shuffle out, measured difference $4.4\times10^{-16}$), so order must be injected (sinusoidal, learned, relative, RoPE). "bank" differs in "river bank" vs "bank account".
</details>

## 📐 Formulas

| Concept | Formula | In plain words |
|---|---|---|
| Recurrence | $\pred{h_t} = f(\data{x_t}, \pred{h_{t-1}})$ | Each state waits for the last |
| Fixed context | $\pred{c} = \pred{h_T}$, $\pred{y_t} = g(\pred{y_{t-1}}, \pred{s_{t-1}}, \pred{c})$ | One summary for all outputs |
| Dynamic context | $\pred{c_t} = \sum_i \pred{\alpha_{t,i}}\pred{h_i}$ | New look-back per output |
| Attention weights | $\pred{\alpha_i} = e^{\pred{e_i}} / \sum_j e^{\pred{e_j}}$ | Positive, sum to 1 |
| Context vector | $\pred{c} = \sum_i \pred{\alpha_i}\pred{v_i}$ | Weighted average of values |
| Projections | $\pred{Q} = \data{X}\param{W_Q}$, $\pred{K} = \data{X}\param{W_K}$, $\pred{V} = \data{X}\param{W_V}$ | Three views of each token |
| Scaled dot-product | $\operatorname{softmax}(\pred{Q}\pred{K}^\top/\sqrt{\hyper{d_k}})\pred{V}$ | Match, then mix |
| Self-attention matrix | $\pred{H'} = \pred{A}\pred{V}$, rows of $\pred{A}$ sum to 1 | Rows ask, columns answer |
| Residual | $\pred{x} + \operatorname{Sublayer}(\pred{x})$ | Keep input, add update |

## 🔢 Numbers Worth Quoting

| Fact | Number |
|---|---|
| Scores → weights | [0.2, 0.4, 0.1, 2.0, 1.8] → [0.0708, 0.0864, 0.0640, 0.4282, 0.3506]; "not" + "good" = 78% |
| Context vs average | [0.3813, 0.4719] vs [0.28, 0.24] |
| Same scores × 5 | [0.0001, 0.0002, 0.0001, 0.7308, 0.2688] (near-hard) |
| "movie" score 0.4 → 1.0 → 2.5 | weight 0.086 → 0.147 → 0.436 |
| Toy self-attention | good [1.0, 0.1] → [0.275, 0.58]; not [0.0, 1.0] → [0.525, 0.355] |
| Per-layer cost (n tokens, width d) | self-attention $O(n^2 d)$, $O(1)$ sequential, $O(1)$ path; recurrent $O(n d^2)$, $O(n)$, $O(n)$ |
| Transformer (2017) | 28.4 BLEU WMT14 EN-DE; 3.5 days on 8 GPUs |

## ⚖️ RNN/LSTM vs Self-Attention

| | RNN / LSTM | Self-attention |
|---|---|---|
| Sequential steps per layer | $O(n)$ | $O(1)$ |
| Path between two tokens | $O(n)$ | $O(1)$ |
| Compute per layer | $O(n d^2)$ | $O(n^2 d)$ |
| Memory of the past | One fixed-size state | All token vectors |
| Word order | Built in | Must add positional information |
| Generation memory per step | Constant | Grows with context (KV cache) |
| Best for | Streaming, on-device, short sequences | Large-scale training, long-range context |

## 💻 Code You Should Be Able to Write

```python title="attention_numpy.py" showLineNumbers
import numpy as np

def softmax(x, axis=-1):
    # highlight-next-line
    e = np.exp(x - x.max(axis=axis, keepdims=True))     # subtract the max: no overflow
    return e / e.sum(axis=axis, keepdims=True)

def attention(Q, K, V):
    scores = Q @ K.T / np.sqrt(K.shape[-1])              # (n_q, n_k)
    weights = softmax(scores, axis=-1)                   # each row sums to 1
    return weights @ V, weights
```

## 🚨 Traps

- Softmax normalizes; it doesn't decide relevance (no parameters).
- Rows of the attention matrix sum to 1; columns don't.
- Attention predates the Transformer (2014); the Transformer removed recurrence (2017).
- "All you need" still includes positions, feed-forward, residuals, layer norm, multi-head.
- Self-attention is order-blind without positional information.
- Naive `exp(x) / sum(exp(x))` overflows: subtract the max.
- Attention maps are not explanations; hand-drawn heatmaps are not model outputs.
- Quadratic cost: doubling context length quadruples attention memory and compute.
- Transformers train in parallel but still generate one token at a time.
