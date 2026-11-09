---
title: "Cheat Sheet: Hugging Face Pipelines"
sidebar_position: 12
---

:::tip[What this sheet covers]
What `pipeline()` hides (tokenizer → model → postprocessing), the default sentiment model, tokenizer outputs, logits and the softmax score, batching and the gotchas, from notebook 19 (and the Hugging Face cells of notebook 18). Full notes: [Hugging Face Pipelines and What They Hide](../../03-ai-ml/03-deep-learning/07-hugging-face-qna-and-rag/01-hugging-face-pipelines.md).
:::

![What pipeline("sentiment-analysis") does: preprocess, forward, postprocess](/img/deep-learning/diagrams/hf-pipeline-internals.svg)

## 🗺️ Pipeline vs Manual, Stage by Stage

| Stage | Pipeline does | By hand |
|---|---|---|
| Load | default model for the task (warns; pin it) | `AutoTokenizer.from_pretrained(name)`, `AutoModelForSequenceClassification.from_pretrained(name).eval()` |
| Preprocess | lower-case, WordPiece, [CLS]/[SEP], ids, mask, pad, truncate | `tokenizer(texts, padding=True, truncation=True, return_tensors="pt")` |
| Forward | model under `no_grad` | `with torch.no_grad(): logits = model(**batch).logits` (`[B, 2]`) |
| Postprocess | softmax, argmax, `id2label` | `F.softmax(logits, -1)`, `.argmax(-1)`, `model.config.id2label[i]` |
| Output | `{'label': 'POSITIVE', 'score': 0.99987}` | whatever you need: logits, all probabilities, hidden states |

## 🎤 Answer These Out Loud

<details>
<summary><strong>What does pipeline("sentiment-analysis") load and return?</strong></summary>

With no model given, `distilbert-base-uncased-finetuned-sst-2-english` (revision 714eb0f): DistilBERT, 6 × 768 × 12, fine-tuned on SST-2, 66,955,010 parameters. It returns one `{'label', 'score'}` per input; score = softmax probability of the winning label. Pin `model=` and `revision=` in production.
</details>

<details>
<summary><strong>What does the tokenizer return?</strong></summary>

`input_ids` (WordPiece ids with [CLS] 101 and [SEP] 102), `attention_mask` (1 real, 0 pad) and `token_type_ids` (segments; DistilBERT ignores them). "The movie was surprisingly good." → `[101, 1996, 3185, 2001, 10889, 2204, 1012, 102]`.
</details>

<details>
<summary><strong>How do logits become the score?</strong></summary>

Softmax over `[z0, z1]`; with two classes $p_1 = \sigma(z_1 - z_0)$. Logits `[-4.2341, 4.6189]` → margin 8.853 → 0.99986. Scores saturate near 1 above a margin of about 5, so a high score is not evidence of correctness ("It was fine, I guess." → POSITIVE 0.99976).
</details>

<details>
<summary><strong>Why go manual instead of using the pipeline?</strong></summary>

Control over batching, padding, truncation and devices; access to logits, hidden states and attentions; custom postprocessing (thresholds, neutral zones); and the same tokenizer and model objects are what you fine-tune.
</details>

## 🔢 Numbers Worth Quoting

| Fact | Number |
|---|---|
| Default sentiment model | `distilbert-base-uncased-finetuned-sst-2-english`, revision 714eb0f, 268 MB safetensors |
| Parameters | 66,955,010 = encoder 66,362,880 + head 592,130 |
| DistilBERT vs BERT-base | about 40% smaller, 60% faster, about 97% of GLUE performance |
| Vocabulary / max length | 30,522 WordPieces (uncased) / 512 tokens |
| Notebook scores | love 0.999874 · terrible 0.999726 · not bad 0.998976 · not terrible 0.984081 · fine, I guess 0.999756 |
| Margin → score | 4.6 → 0.99, 8.853 → 0.99986, 9.2 → 0.9999 |
| Truncation | 802-token text → 512 kept |

## 💻 Code You Should Be Able to Write

```python title="manual_sentiment.py" showLineNumbers
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForSequenceClassification.from_pretrained(name).eval()

texts = ["I love this movie.", "It was fine, I guess."]
# highlight-next-line
batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    probs = F.softmax(model(**batch).logits, dim=-1)
for text, p in zip(texts, probs):
    i = p.argmax().item()
    print(text, model.config.id2label[i], round(p[i].item(), 6))
# I love this movie. POSITIVE 0.999874
# It was fine, I guess. POSITIVE 0.999756
```

## 🚨 Traps

- Unpinned pipeline defaults can change; pin model + revision.
- Tokenizer and model must come from the same checkpoint.
- Pass `attention_mask` with padded batches.
- `truncation=True` silently drops tokens past 512.
- DistilBERT ignores `token_type_ids`.
- `model.eval()` + `torch.no_grad()` for inference.
- `output_attentions` needs `attn_implementation="eager"` (sdpa returns none).
- "Could not import module 'pipeline'" = environment problem (PyTorch missing/incompatible).
