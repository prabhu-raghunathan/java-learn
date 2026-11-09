---
title: "Cheat Sheet: The Transformer (Figure 1)"
sidebar_position: 9
---

:::tip[What this sheet covers]
Every box of the paper's Figure 1 in order, the hyperparameters, training recipe, results and ablations of "Attention Is All You Need", and the questions interviewers ask about it. Full notes: [The Transformer: Figure 1 Box by Box](../../03-ai-ml/03-deep-learning/06-attention-and-transformers/04-transformer-architecture-figure-1.md).
:::

![The Transformer - model architecture](/img/deep-learning/diagrams/transformer-figure-1.svg)

## 🗺️ Figure 1 in One Table (bottom to top)

| Box | What it does | Base model |
|---|---|---|
| Inputs | Source token ids (BPE), read all at once | ~37K shared vocab |
| Input Embedding | Lookup × $\sqrt{d_{model}}$, shared with Output Embedding and Linear | 37,000 × 512 |
| Positional Encoding (⊕) | Add sinusoids $\sin/\cos(pos/10000^{2i/d})$ | 512 wide, no params |
| Multi-Head Attention (enc) | Self-attention, no mask | 8 heads × 64 |
| Add & Norm | $\operatorname{LayerNorm}(x + \operatorname{Dropout}(\operatorname{Sublayer}(x)))$ | dropout 0.1 |
| Feed Forward | $\max(0, xW_1 + b_1)W_2 + b_2$ per position | 512 → 2048 → 512 |
| N× (enc) | Stack; last output = memory $z$ | N = 6 |
| Outputs (shifted right) | $[\langle s \rangle, y_1..y_{m-1}]$ → predict $[y_1..y_m]$ | teacher forcing |
| Output Embedding + PE | Same as input side (shared weights) | |
| Masked Multi-Head Attention | Self-attention, $-\infty$ on future positions | lower-triangular |
| Multi-Head Attention (dec) | Q from decoder, K = V = encoder output | $m \times n$ scores |
| Add & Norm, Feed Forward (dec) | As encoder; 3 sub-layers per decoder layer | N = 6 |
| Linear | Tied to the embeddings: $y E^\top$ | 512 → 37,000 |
| Softmax → Output Probabilities | Next-token distribution | beam 4, α 0.6 |

## 🎤 Answer These Out Loud

<details>
<summary><strong>Walk through Figure 1.</strong></summary>

Encoder: embeddings ×√d_model + sinusoidal PE → 6 × [self-MHA → Add & Norm → FFN → Add & Norm]. Decoder: target shifted right → embeddings + PE → 6 × [masked self-MHA → Add & Norm → MHA(Q = decoder, K = V = encoder) → Add & Norm → FFN → Add & Norm] → Linear (tied) → Softmax. $d_{model}$ 512, $h$ 8, $d_k$ 64, $d_{ff}$ 2048.
</details>

<details>
<summary><strong>Why positional encodings? Why sinusoids?</strong></summary>

Attention is order-blind (toy model without PE: 0.5% exact reversal vs 100%). Added sinusoids: unique, bounded, parameter-free; $PE_{pos+k} = R_k PE_{pos}$ (rotation per sin/cos pair, verified to $6 \times 10^{-15}$), so $PE_{pos} \cdot PE_{pos+k}$ depends only on $k$. Learned embeddings: 25.7 vs 25.8 BLEU; sinusoids may extrapolate to longer sequences.
</details>

<details>
<summary><strong>Why the decoder mask and the right shift?</strong></summary>

Training feeds the whole shifted target in parallel (teacher forcing); the mask ($-\infty$ for $j > i$) stops position $i$ seeing the token it must predict. Toy without mask: 100% teacher-forced accuracy but 0.1% when generating. At generation time there's no future to hide.
</details>

<details>
<summary><strong>What's in Add & Norm, and Post-LN vs Pre-LN?</strong></summary>

Residual + LayerNorm per position (mean 0, variance 1, learned γ, β), dropout 0.1 on the sub-layer output before adding. Paper = Post-LN, $\operatorname{LN}(x + F(x))$; modern LLMs mostly Pre-LN, $x + F(\operatorname{LN}(x))$, which is more stable when deep and less dependent on warmup.
</details>

<details>
<summary><strong>How was it trained?</strong></summary>

WMT14 En–De 4.5M pairs, ~25K + 25K tokens per batch, 8 P100s (base 100K steps / 12 h; big 300K / 3.5 days). Adam (0.9, 0.98, 1e-9); $lrate = d^{-0.5}\min(s^{-0.5}, s \cdot 4000^{-1.5})$ (peak 0.000699); residual dropout 0.1; label smoothing 0.1 (worse perplexity, better BLEU); checkpoint averaging.
</details>

<details>
<summary><strong>Why self-attention instead of RNN or CNN?</strong></summary>

Table 1: self-attention $O(n^2 d)$, $O(1)$ sequential, $O(1)$ path; recurrent $O(n d^2)$, $O(n)$, $O(n)$; conv $O(k n d^2)$, $O(1)$, $O(\log_k n)$. More parallel, shortest paths, cheaper than RNNs when $n < d$; restricted attention $O(r n d)$ for long inputs.
</details>

<details>
<summary><strong>How do BERT, GPT and T5 relate to it?</strong></summary>

BERT = encoder only (bidirectional, masked-token pre-training). GPT = decoder only (masked self-attention + FFN, next-token prediction, no cross-attention). T5/BART = full encoder–decoder.
</details>

## 🔢 Numbers Worth Quoting

| Fact | Number |
|---|---|
| Base / big | N 6/6, $d_{model}$ 512/1024, $d_{ff}$ 2048/4096, h 8/16, 65M/213M params |
| BLEU En–De | 28.4 big, 27.3 base (> 2 BLEU over previous best incl. ensembles) |
| BLEU En–Fr | 41.8 (abstract, Table 2); §6.1 text says 41.0 |
| Training cost | base $3.3 \times 10^{18}$, big $2.3 \times 10^{19}$ FLOPs |
| Table 3 | h = 1: 24.9; N = 2: 23.7; no dropout: 24.6; no label smoothing: 25.3 (PPL 4.67); learned PE: 25.7; big: 26.4 |
| Learning rate | 0.000175 (1K), 0.000699 (4K peak), 0.000349 (16K), 0.000140 (100K) |
| Params per layer (base) | MHA 1,050,624 · FFN 2,099,712 · enc layer 3,152,384 · dec layer 4,204,032 |
| Parsing | 91.3 F1 WSJ-only, 92.7 semi-supervised |
| Toy model (reverse 6 digits) | 168,128 params; 100% exact; cross-attention anti-diagonal 1.00 |

## 💻 Code You Should Be Able to Write

```python title="encoder_decoder_layers.py" showLineNumbers
import torch, torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self, d, h, d_ff, p=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.ReLU(), nn.Linear(d_ff, d))
        self.n1, self.n2, self.drop = nn.LayerNorm(d), nn.LayerNorm(d), nn.Dropout(p)

    def forward(self, x, pad_mask=None):
        x = self.n1(x + self.drop(self.attn(x, x, x, key_padding_mask=pad_mask)[0]))   # Add & Norm
        return self.n2(x + self.drop(self.ff(x)))                                       # Add & Norm

class DecoderLayer(nn.Module):
    def __init__(self, d, h, d_ff, p=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.ReLU(), nn.Linear(d_ff, d))
        self.n1, self.n2, self.n3, self.drop = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d), nn.Dropout(p)

    def forward(self, y, memory):
        n = y.size(1)
        # highlight-next-line
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)   # True = blocked (future)
        y = self.n1(y + self.drop(self.self_attn(y, y, y, attn_mask=causal)[0]))
        y = self.n2(y + self.drop(self.cross_attn(y, memory, memory)[0]))     # Q = decoder, K = V = encoder
        return self.n3(y + self.drop(self.ff(y)))
```

## 🚨 Traps

- PE is **added**, not concatenated; fixed sinusoids in the paper.
- Mask = training-time necessity; future scores → $-\infty$ **before** softmax.
- Cross-attention: K and V from the **encoder output**, the same for every decoder layer; no causal mask.
- Dropout before the residual add, and on embedding + PE.
- Post-LN in the paper; know Pre-LN.
- Weight tying + ×√d_model in the embedding layers.
- Label smoothing lowers confidence on purpose (toy: 0.92 on correct tokens).
- 41.8 vs 41.0 En–Fr inconsistency in the paper.
