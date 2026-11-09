---
title: "Cheat Sheet: Stacked Encoder Blocks"
sidebar_position: 11
---

:::tip[What this sheet covers]
Stacking Transformer encoder blocks, the attention mask, masked mean pooling, parameter counts and the 1-block vs 3-block experiment from notebook 18 (the file named *Hugging_Face_Pipeline_Complete*, whose content is the stacked-encoder mini project). Full notes: [Stacked Encoder Blocks: a Mini Sentiment Project](../../03-ai-ml/03-deep-learning/06-attention-and-transformers/06-stacked-encoder-blocks-mini-project.md).
:::

![Mini sentiment project: embeddings, stacked encoder blocks, masked mean pooling, classifier](/img/deep-learning/diagrams/stacked-encoder-classifier.svg)

## 🗺️ The Pipeline in One Table

| Step | What happens | Shape (batch 4) |
|---|---|---|
| Tokenize + encode | lower-case, split on spaces, `<UNK>` = 1, truncate/pad to 6 with `<PAD>` = 0 | ids `[4, 6]`, mask `[4, 6]` |
| Embeddings | token embedding + learned position embedding | `[4, 6, 32]` |
| Encoder block × N | masked MHSA → Add & Norm → FFN (32 → 64 → 32, ReLU) → Add & Norm | `[4, 6, 32]` each; attention `[4, 4, 6, 6]` |
| Masked mean pooling | $\sum_t m_t X_t / \max(\sum_t m_t, 10^{-9})$ | `[4, 32]` |
| Head | Linear(32, 2) → cross-entropy | `[4, 2]` |

## 🎤 Answer These Out Loud

<details>
<summary><strong>Why stack encoder blocks?</strong></summary>

Each block is one round of "attend to all tokens, then transform each token". Stacking lets later blocks work on already-contextualized vectors (relations of relations, negation scope, phrase meaning). Shapes never change, so a stack is an `nn.ModuleList` and a loop; each block has its own weights.
</details>

<details>
<summary><strong>Why `nn.ModuleList`, not a Python list or `nn.Sequential`?</strong></summary>

A Python list doesn't register the blocks: no parameters in the optimizer, no `.to()`, no `state_dict`. `nn.Sequential` registers them but can't pass the attention mask to every block. `ModuleList` + explicit loop does both.
</details>

<details>
<summary><strong>How is padding ignored?</strong></summary>

Twice: padded **keys** get −1e9 added to their scores before the softmax (mask broadcast as `[B, 1, 1, T]`), so their weights are exactly 0; and **pooling** averages only the positions where the mask is 1. `padding_idx` alone only zeroes the `<PAD>` embedding.
</details>

<details>
<summary><strong>Is the 3-block model better?</strong></summary>

Both reach 100% training accuracy on 18 sentences, which proves nothing. Held-out (16 new combinations), 20 seeds: 1 block 84.1% ± 6.4, 3 blocks 86.3% ± 7.6: within noise. Depth needs data, which is why BERT-style encoders are pre-trained.
</details>

<details>
<summary><strong>Mean pooling or [CLS]?</strong></summary>

Masked mean pooling uses every real token and is robust when training from scratch (also used by sentence-embedding models). [CLS] works when the model is trained to summarize there (BERT). An unmasked mean leaks padding into the sentence vector.
</details>

## 🔢 Numbers Worth Quoting

| Fact | Number |
|---|---|
| Data | 18 phrases, vocab 35 (`<PAD>` 0, `<UNK>` 1), max_len 6, batch 4 |
| Model | D 32, 4 heads × 8, d_ff 64, ReLU, post-LN, Adam 3e-3, 60 epochs |
| Parameters | embeddings 1,312 + 8,544 per block + head 66 → 9,922 (1 block), 27,010 (3 blocks) |
| Final training loss | 0.000441 (1 block), 0.000492 (3 blocks); both 100% training accuracy |
| Held-out, 20 seeds | 84.1% ± 6.4 (1 block) vs 86.3% ± 7.6 (3 blocks) |
| "this movie is terrible" | 1 block NEGATIVE 0.9598, 3 blocks POSITIVE 0.9180 |
| Hidden-state norm, "not good movie" | 8.73 → 5.62 → 5.74 → 6.01 (LayerNorm keeps ≈ √32 = 5.66) |
| Mean absolute change per block | 0.85, 0.59, 0.63 |

## 💻 Code You Should Be Able to Write

```python title="encoder_stack_and_pooling.py" showLineNumbers
import torch, torch.nn as nn

class EncoderStack(nn.Module):
    def __init__(self, d_model=32, heads=4, d_ff=64, layers=3):
        super().__init__()
        # highlight-next-line
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model, heads, d_ff, 0.0, batch_first=True)
                                     for _ in range(layers)])
    def forward(self, x, mask):                         # mask: 1 = real token, 0 = pad
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=mask.eq(0))   # PyTorch wants True = ignore
        return x

def masked_mean(x, mask):
    m = mask.unsqueeze(-1).type_as(x)
    # highlight-next-line
    return (x * m).sum(1) / m.sum(1).clamp(min=1e-9)

torch.manual_seed(0)
x, mask = torch.randn(2, 6, 32), torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]])
out = EncoderStack()(x, mask)
print(out.shape, masked_mean(out, mask).shape)          # torch.Size([2, 6, 32]) torch.Size([2, 32])
print(sum(p.numel() for p in EncoderStack(layers=1).parameters()))   # 8544
```

## 🚨 Traps

- Shape never changes through a stack: `[B, T, D]` in and out.
- Training accuracy on a tiny set measures memorization, not depth's value.
- `padding_idx` ≠ masking; mask the attention keys **and** the pooling.
- PyTorch's `src_key_padding_mask` uses **True = ignore**, the opposite of a Hugging Face `attention_mask`.
- −1e9 in float16 overflows; use `torch.finfo(dtype).min`.
- Heads split D; they don't add parameters.
- Attention maps are diagnostics, not explanations.
