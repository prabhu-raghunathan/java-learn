---
title: "Cheat Sheet: Recurrent Neural Networks"
sidebar_position: 4
---

:::tip[What this sheet covers]
Why sequences need memory, padding and masking, the RNN cell and its shapes, unrolling and weight sharing, backpropagation through time, many-to-one and many-to-many models, training in PyTorch, and vanishing/exploding gradients. Full notes: [Sequence Data and Recurrence](../../03-ai-ml/03-deep-learning/04-rnns/01-sequence-data-and-recurrence.md) · [RNN Architecture, Forward Pass and BPTT](../../03-ai-ml/03-deep-learning/04-rnns/02-rnn-architecture-forward-pass-bptt.md) · [RNNs in Practice](../../03-ai-ml/03-deep-learning/04-rnns/03-rnns-in-practice-training-and-limits.md).
:::

## 🎤 Answer These Out Loud

<details>
<summary><strong>Why do we need RNNs? Why not a CNN or an ANN?</strong></summary>

Sequences (text, speech, sensor data, prices) are **ordered**, **variable-length** and need **memory** of earlier elements: "I love deep learning" and "learning deep love I" share every word but not the meaning. ANNs/CNNs assume fixed-size inputs and keep no memory between inputs (a 1-D CNN only sees a bounded window). An RNN reads one element per step and carries a hidden state, the same weights at every step.
</details>

<details>
<summary><strong>Write the RNN equations and explain each term.</strong></summary>

$h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$, $y_t = W_y h_t + b_y$, $h_0 = 0$. $W_x x_t$ brings in the current input, $W_h h_{t-1}$ the memory of everything before, tanh adds non-linearity and keeps the state in $(-1, 1)$. Shapes: $x_t$ $(B, D)$, $h_t$ $(B, H)$, $W_x$ $(H, D)$, $W_h$ $(H, H)$, $W_y$ $(O, H)$.
</details>

<details>
<summary><strong>Why tanh and not ReLU in a vanilla RNN?</strong></summary>

The same update is applied many times: tanh is **bounded** (the state can't grow without limit), **zero-centred** (positive and negative signals stay balanced) and has a maximum slope of 1 (vs sigmoid's 0.25). ReLU is unbounded, so repeated multiplication by $W_h$ can explode the state. Note tanh does *not* prevent vanishing gradients; its slope ≤ 1 makes them more likely.
</details>

<details>
<summary><strong>What does unrolling mean, and why weight sharing?</strong></summary>

Unrolling redraws the loop as $T$ copies of the cell, one per time step, connected through the hidden state: a $T$-layer feedforward network whose layers **share weights**. Sharing keeps the parameter count independent of length (700 vs 7,000 for 10 steps with $D = 10, H = 20, O = 5$), lets one model handle any length, and learns a rule once ("not flips sentiment") wherever it appears.
</details>

<details>
<summary><strong>How is an RNN trained?</strong></summary>

**Backpropagation through time**: forward over all steps storing every $h_t$; loss $L = \sum_t L_t$ (or only $L_T$ for many-to-one); backprop through the unrolled graph; each step back multiplies by $\text{diag}(1 - h_t^2)W_h$; a shared weight's gradient is the **sum** over steps; one optimizer step per sequence (or chunk). In PyTorch, autograd does all of this. Memory is $O(T)$; long sequences use truncated BPTT (carry the state, `detach()` between chunks).
</details>

<details>
<summary><strong>Explain vanishing and exploding gradients in RNNs.</strong></summary>

The gradient to an early state is a product of one Jacobian per step: $\frac{\partial L_T}{\partial h_k} = \frac{\partial L_T}{\partial h_T}\prod_{j=k+1}^{T}\text{diag}(1 - h_j^2)W_h$. If the factors are below 1 it shrinks exponentially ($0.5^{15} = 0.000031$; measured 4.0 → 0.0000067 over 19 steps), so early steps learn nothing; if above 1 it explodes ($1.5^{15} = 437.89$), giving loss spikes and `NaN`. Exploding → **clip**; vanishing → **LSTM/GRU**, orthogonal init, attention.
</details>

<details>
<summary><strong>Many-to-one vs many-to-many?</strong></summary>

Same recurrence; the difference is where you read outputs. **Many-to-one** reads only the final state ($y = W_{hy}h_T + b_y$): sentiment, sequence classification (alternatives: mean/max pooling, bidirectional final states, attention pooling). **Many-to-many** emits a softmax at every step: tagging, next-token prediction, trained with cross-entropy against the input shifted by one. The same token can get different predictions at different positions because $h_t$ carries the context.
</details>

<details>
<summary><strong>How do you batch sentences of different lengths?</strong></summary>

Pad to the longest with a reserved `<PAD>` = 0 (and `<UNK>` = 1), keep the lengths (or a mask), and use `pack_padded_sequence` (or take each row's state at its true last index). Without packing, pads keep updating the hidden state and the final state of short sentences is wrong. Truncate very long sequences to a maximum length chosen from the data.
</details>

<details>
<summary><strong>How do you know your hand-written backprop is correct?</strong></summary>

A **gradient check**: compare against autograd or central finite differences $\frac{L(W+\epsilon) - L(W-\epsilon)}{2\epsilon}$ on a tiny example; agreement to ~$10^{-7}$ (float32) or ~$10^{-12}$ (float64). A wrong gradient can still lower the loss: the notebook's shortcut (last step only, $W_{hh}$ never updated) reached 0.0165 while full BPTT reached 0.0080, matching autograd to $2.2 \times 10^{-16}$.
</details>

## 📐 Formulas

| Concept | Formula | In plain words |
|---|---|---|
| Hidden state | $\pred{h_t} = \tanh(\param{W_x}\data{x_t} + \param{W_h}\pred{h_{t-1}} + \param{b})$, $\pred{h_0} = 0$ | Input plus memory, squashed |
| Output | $\pred{y_t} = \param{W_y}\pred{h_t} + \param{b_y}$ | Linear read of the memory |
| Unrolled | $\pred{h_3} = f(\data{x_3}, f(\data{x_2}, f(\data{x_1}, \pred{h_0})))$ | Same $f$ every step |
| Many-to-one | $\pred{y} = \param{W_{hy}}\pred{h_T} + \param{b_y}$ | Decide once at the end |
| Per-step softmax | $\pred{p_t} = \text{softmax}(\pred{y_t})$ | A distribution at every step |
| Parameter count | $HD + H^2 + OH + H + O$; `nn.RNN`: $HD + H^2 + 2H$ | Independent of $T$ |
| Total loss | $\loss{L} = \sum_t \loss{L_t}$ | Sum over steps |
| One-step Jacobian | $\partial\pred{h_t}/\partial\pred{h_{t-1}} = \text{diag}(1 - \pred{h_t}^2)\param{W_h}$ | tanh slope × recurrent weights |
| Gradient to an early state | $\frac{\partial\loss{L_T}}{\partial\pred{h_k}} = \frac{\partial\loss{L_T}}{\partial\pred{h_T}}\prod_{j=k+1}^{T}\text{diag}(1 - \pred{h_j}^2)\param{W_h}$ | One factor per step back |
| Shared-weight gradient | $\frac{\partial\loss{L}}{\partial\param{W}} = \sum_t\frac{\partial\loss{L_t}}{\partial\param{W}}$ | Sum every use |
| Bound | $\lVert\partial\loss{L_T}/\partial\pred{h_k}\rVert \le \lVert\partial\loss{L_T}/\partial\pred{h_T}\rVert\,\sigma_{\max}(\param{W_h})^{T-k}$ | Below 1 shrinks, above 1 can grow |
| BPTT by hand | $da_t = (1 - h_t^2)\odot dh_t$; $dW_{xh} \mathrel{+}= da_t x_t^\top$; $dW_{hh} \mathrel{+}= da_t h_{t-1}^\top$; $dh_{t-1} = W_{hh}^\top da_t$ | Through tanh, accumulate, pass back |
| BCE with logits | $\loss{L} = -[\data{y}\log\sigma(\pred{z}) + (1-\data{y})\log(1 - \sigma(\pred{z}))]$ | Binary loss on a raw score |
| Clipping | if $\lVert g\rVert > c$: $g \leftarrow g \cdot c/\lVert g\rVert$ | Cap the step, keep the direction |

## 🔢 Numbers to Quote

- Toy hidden-state trace (notebook 12 weights): $h_1 = [0.462, 0.291, -0.537]$ … $h_4 = [0.464, 0.878, -0.771]$.
- Weight sharing: 700 vs 7,000 parameters for 10 steps; `nn.RNN(10, 20)` = 640.
- Measured BPTT gradient norms, $W_{hh} = 0.5 \times$ orthogonal, $T = 20$: 4.0 at $h_{20}$ → 0.0000067 at $h_1$; scale 3.0 → 2,460.
- $0.5^{15} = 0.000031$; $1.5^{15} = 437.89$; $\tanh(0.5h)$ ×10 shrinks 2.241 → 0.00147.
- Training from scratch (50 epochs): shortcut gradient 0.0165 vs full BPTT 0.0080.
- MSE + sigmoid at prediction reports 0.731 / 0.5; `BCEWithLogitsLoss` gives logits ±6 → 0.998 / 0.002.
- Gated preview: forget gate 0.95 keeps $0.95^{20} = 0.358$ after 20 steps vs $0.5^{20} = 9.54 \times 10^{-7}$.

## ⚖️ Input–Output Patterns

| Pattern | Example | How |
|---|---|---|
| One-to-one | Image classification | No recurrence (plain ANN) |
| Many-to-one | Sentiment, spam, ECG | Final hidden state → output |
| One-to-many | Captioning, music generation | Input as the initial state, generate step by step |
| Many-to-many, aligned | POS tagging, next-token training | Output at every step |
| Many-to-many, encoder-decoder | Translation, summarization | Encoder RNN → state → decoder RNN |

## 💻 Code You Should Be Able to Write

```python title="sentiment_rnn.py" showLineNumbers
import torch
import torch.nn as nn

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64, cell=nn.LSTM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = cell(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):                        # x: (B, T) token ids
        out, state = self.rnn(self.embedding(x))  # out: (B, T, H)
        h_n = state[0] if isinstance(state, tuple) else state   # LSTM returns (h_n, c_n)
        # highlight-next-line
        return self.fc(h_n[-1])                  # (B, 1) logit from the last layer's final state

x = torch.randint(1, 100, (8, 12))
for cell in (nn.RNN, nn.GRU, nn.LSTM):
    m = SentimentRNN(100, cell=cell)
    logits = m(x)
    loss = nn.BCEWithLogitsLoss()(logits, torch.ones(8, 1))
    loss.backward()
    nn.utils.clip_grad_norm_(m.parameters(), 1.0)   # after backward(), before step()
    print(cell.__name__, tuple(logits.shape), sum(p.numel() for p in m.rnn.parameters()))
# RNN (8, 1) 6272
# GRU (8, 1) 18816        3x the RNN
# LSTM (8, 1) 25088       4x the RNN
```

## 🚨 Traps

- Padding with an id that is also a word: reserve `<PAD>` = 0, `<UNK>` = 1.
- Forgetting to pack or mask: pads keep updating the state.
- `nn.RNN` defaults to `(T, B, D)`; batch-first data without `batch_first=True` silently treats the batch as time.
- `h_n` is always `(layers × directions, B, H)`; `output[:, -1]` equals `h_n[-1]` only for an unpadded single-direction RNN.
- Carrying state across batches without `h.detach()` grows the graph and fails on the second backward.
- Not clipping: one long sequence can produce `NaN` weights. Clip after `backward()`, before `step()`; clipping does nothing for vanishing gradients.
- `nn.RNN` has two bias vectors: $HD + H^2 + 2H$ parameters.
- With $h_0 = 0$, step 1 contributes nothing to $\partial L/\partial W_h$.
- MSE training + sigmoid at prediction distorts every probability: train with `BCEWithLogitsLoss`.
- A wrong hand-written gradient can still reduce the loss: gradient-check it.
- Ambiguous training data (first word = last word = majority) makes "memory" tests meaningless.
- RNNs can't parallelize over time: one reason Transformers took over.
