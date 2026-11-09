---
title: "Cheat Sheet: BERT"
sidebar_position: 10
---

:::tip[What this sheet covers]
BERT's Figure 1 (pre-training and fine-tuning) and Figure 2 (input representation), the two pre-training tasks, every fine-tuning head, the hyperparameters, results and ablations, and the questions interviewers ask about the paper. Full notes: [BERT: Pre-training and Fine-tuning, Figure by Figure](../../03-ai-ml/03-deep-learning/06-attention-and-transformers/05-bert-pretraining-and-fine-tuning.md).
:::

![Overall pre-training and fine-tuning procedures for BERT](/img/deep-learning/diagrams/bert-figure-1.svg)

## 🗺️ BERT in One Table

| Piece | What it is | BERT-base |
|---|---|---|
| Architecture | Transformer **encoder** only, unmasked self-attention (left + right context in every layer) | L 12, H 768, A 12, FFN 3072 (GELU), 110M |
| Input format | [CLS] A [SEP] B [SEP] (single sentence: [CLS] A [SEP]) | max 512 tokens |
| Input embedding | token (WordPiece) + segment ($E_A$/$E_B$) + learned position, then LayerNorm | vocab 30,000 (released 30,522) |
| Outputs | $C$ = final [CLS] vector; $T_i$ = final vector of token $i$ | each in $\mathbb{R}^{768}$ |
| Task 1: Masked LM | choose 15% of tokens: 80% [MASK], 10% random, 10% unchanged; predict originals | loss at chosen positions only |
| Task 2: NSP | B is the real next span 50% (IsNext) / random 50% (NotNext); classify from $C$ | 97–98% accuracy |
| Pre-training | BooksCorpus 800M + Wikipedia 2,500M words; batch 256 × 512; 1M steps; Adam 1e-4, warmup 10k, linear decay | 4 days, 16 TPU chips |
| Classification head | $\operatorname{softmax}(CW^\top)$, $W \in \mathbb{R}^{K \times H}$ | only new weights |
| SQuAD head | start $\operatorname{softmax}(S \cdot T_i)$, end $\operatorname{softmax}(E \cdot T_j)$; best $S \cdot T_i + E \cdot T_j$, $j \ge i$ | $2H$ new weights |
| SQuAD 2.0 | no answer = span at [CLS]; answer iff best span $> S \cdot C + E \cdot C + \tau$ | $\tau$ tuned on dev |
| Fine-tuning | all parameters; batch 16/32, lr 5e-5 / 3e-5 / 2e-5, 2–4 epochs | ≤ 1 h on a TPU |

![BERT input representation](/img/deep-learning/diagrams/bert-figure-2.svg)

## 🎤 Answer These Out Loud

<details>
<summary><strong>What problem did BERT solve?</strong></summary>

GPT and ELMo pre-trained with unidirectional language models: GPT sees only left context, ELMo concatenates two separately trained one-direction LSTMs. BERT pre-trains a deep bidirectional Transformer encoder (made trainable by the masked LM) and fine-tunes all of it with one new output layer per task.
</details>

<details>
<summary><strong>Why can't BERT use a normal language-modelling loss?</strong></summary>

In a bidirectional multi-layer encoder each word would indirectly see itself, so predicting the input is trivial. BERT hides the targets in the input ([MASK]) instead of masking the attention. Toy check: a causal-mask variant recovers a masked first token 4.65% of the time (chance 4.3%), the bidirectional one 100%.
</details>

<details>
<summary><strong>Explain 80/10/10.</strong></summary>

Of the 15% chosen positions: 80% [MASK], 10% random token, 10% unchanged, all predicted. [MASK] never appears in fine-tuning, so mixing in random and unchanged tokens reduces the mismatch and forces a good representation of every token (the model can't tell which are corrupted or graded). Overall: 12% / 1.5% / 1.5% of all tokens. Table 8: [MASK]-only hurts feature-based NER (94.0 vs 94.9).
</details>

<details>
<summary><strong>Is NSP useful?</strong></summary>

In the paper, removing it hurt QNLI (88.4 → 84.9), MNLI (84.4 → 83.9) and SQuAD (88.5 → 87.9). RoBERTa later dropped it without loss by training on long contiguous text; most later encoders omit it. $C$ is not a good sentence embedding without fine-tuning.
</details>

<details>
<summary><strong>How do you fine-tune BERT for classification, QA and NER?</strong></summary>

Classification: $W \in \mathbb{R}^{K \times H}$ on $C$, cross-entropy. QA: start/end vectors $S$, $E$ dotted with each paragraph $T_i$, softmax, best span with $j \ge i$, loss = gold start + end log-likelihood. NER: classifier on each word's first sub-token, no CRF. Always update all parameters (2–4 epochs, lr 2e-5 to 5e-5).
</details>

<details>
<summary><strong>Fine-tuning vs feature-based?</strong></summary>

Fine-tuning updates everything (best accuracy). Feature-based freezes BERT and feeds its layers to a task model: concat of the last 4 layers gave 96.1 NER dev F1 vs 96.4 fine-tuned. Use it when a task needs its own architecture or to precompute features once.
</details>

<details>
<summary><strong>What did the ablations show?</strong></summary>

Left-to-right instead of MLM (both without NSP): −10.1 SQuAD F1, −9.0 MRPC. No NSP: QNLI −3.5. Bigger models better everywhere, even MRPC with 3,600 examples. 1M vs 500k steps: about +1.0% MNLI. Fine-tuning robust to masking ratios.
</details>

<details>
<summary><strong>BERT vs GPT?</strong></summary>

Same size (base), but BERT's attention is bidirectional and GPT's is left-only. BERT: MLM + NSP, BooksCorpus + Wikipedia, learns [CLS]/[SEP]/segments in pre-training, 128K-word batches, task-specific fine-tuning lr. GPT: next-token LM, BooksCorpus, special tokens only at fine-tuning, 32K-word batches, lr 5e-5 for all. GPT can generate text; BERT can't (encoder only).
</details>

## 🔢 Numbers Worth Quoting

| Fact | Number |
|---|---|
| Sizes | base L 12 / H 768 / A 12 / 110M (109,482,240 exact); large 24 / 1024 / 16 / 340M (335,141,888) |
| Params per layer | $12H^2 + 13H$ = 7,087,872 (base); embeddings 23.8M |
| Masking | 15% chosen → 12% [MASK], 1.5% random, 1.5% unchanged |
| Pre-training | 3.3B words; 256 × 512 tokens; 1M steps ≈ 40 epochs; lr 1e-4; warmup 10k; 90% steps at length 128 |
| GLUE | BERT-large 82.1 average (test, excluding WNLI), leaderboard 80.5 vs GPT 72.8; MNLI 86.7 |
| SQuAD v1.1 | test F1 93.2 (ensemble + TriviaQA), 91.8 single; dev F1 88.5 base, 90.9 large |
| SQuAD 2.0 | test F1 83.1 (+5.1) |
| SWAG | 86.3% (+27.1 over ESIM+ELMo, +8.3 over GPT) |
| NER (CoNLL-2003) | fine-tuned 96.6 / 92.8 (large dev / test); feature-based concat last 4: 96.1 dev |
| NSP accuracy | 97–98% (toy BERT on this site: 97.8%) |
| Attention cost 512 vs 128 | 16× more scores |

## 💻 Code You Should Be Able to Write

```python title="bert_masking_and_heads.py" showLineNumbers
import random, torch, torch.nn as nn

def mask_tokens(tok, rng, vocab_size, special, mask_id, rate=0.15):
    inp, lab = list(tok), [-100] * len(tok)                 # -100 = ignored by cross_entropy
    cand = [i for i, t in enumerate(tok) if t not in special]
    for i in rng.sample(cand, max(1, round(rate * len(cand)))):
        lab[i] = tok[i]
        r = rng.random()
        # highlight-start
        if r < 0.8:   inp[i] = mask_id                       # 80% [MASK]
        elif r < 0.9: inp[i] = rng.randrange(vocab_size)     # 10% random token
        # highlight-end
    return inp, lab                                          # remaining 10%: unchanged

class SpanHead(nn.Module):                                   # SQuAD: only S and E are new
    def __init__(self, H):
        super().__init__()
        self.S, self.E = nn.Parameter(torch.randn(H) * 0.02), nn.Parameter(torch.randn(H) * 0.02)
    def forward(self, T):                                    # T: (batch, n, H) from BERT
        return T @ self.S, T @ self.E                        # start and end scores per token

inp, lab = mask_tokens(list(range(10, 30)), random.Random(0), 30, {10}, 0)   # token 10 plays [CLS]
print(inp)   # [10, 11, 0, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 0, 0, 25, 26, 27, 28, 29]
print(lab)   # 12 at index 2, 23 at 13, 24 at 14; -100 everywhere else (3 of 19 chosen = round(2.85))
```

## 🚨 Traps

- 15% are **chosen**, not all replaced by [MASK]; loss only at those positions.
- Learned position embeddings (not sinusoids), maximum 512 tokens.
- Segment embeddings: only A and B; [CLS] and the first [SEP] belong to A.
- $C$ without fine-tuning is not a sentence embedding (trained only for NSP).
- SQuAD: best pair with $j \ge i$, not two independent argmaxes.
- Fine-tuning updates **all** weights; the head is the only new part.
- BERT is encoder-only: no left-to-right generation.
- "128,000 tokens/batch" is 256 × 512 = 131,072.
