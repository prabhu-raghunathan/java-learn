---
title: "Cheat Sheet: Extractive Document QnA"
sidebar_position: 13
---

:::tip[What this sheet covers]
How an extractive QnA model picks a span, how notebook 20 turns it into a document QnA system over the BERT paper (page-wise extraction, overlapping chunks, brute-force reading, ranking, threshold, source tracking), why it fails, and how answers become fine-tuning labels. Full notes: [Extractive Document QnA over a PDF](../../03-ai-ml/03-deep-learning/07-hugging-face-qna-and-rag/02-extractive-document-qna.md). Span maths in depth: [BERT, Chapter 10](../../03-ai-ml/03-deep-learning/06-attention-and-transformers/05-bert-pretraining-and-fine-tuning.md#10-token-level-heads-squad-spans-unanswerable-questions-and-ner).
:::

![Brute-force document QnA: once per document, extract and chunk; for every question, read all 78 chunks, rank, threshold, cite](/img/deep-learning/diagrams/docqa-pipeline.svg)

## 🗺️ The System in One Table

| Step | What happens | Number from the run |
|---|---|---|
| Load | `AutoTokenizer` + `AutoModelForQuestionAnswering`, `distilbert-base-cased-distilled-squad` | 65,192,450 params; head `Linear(768, 2)` = 1,538 |
| Tokenize | `[CLS] question [SEP] context [SEP]`; `sequence_ids`, `offset_mapping` | 48 tokens for the 3-sentence example |
| Span | best $s_i + e_j$ with both ends in the context, $i \le j$, length ≤ 40 | 16–22, score 22.8177 |
| Extract PDF | PyMuPDF page by page, collapse whitespace | 16 pages, 64,105 characters |
| Length wall | pages 1–4 vs `max_length=384` | 4,556 tokens; model saw 8.0% |
| Chunk | 180 words, overlap 40 (step 140), min 30 | 78 chunks; 1.74 tokens/word; 71–432 tokens |
| Read all | reader on every chunk, rank by span score | about 10 s per question on CPU |
| Abstain | rank-1 score below 5.0 → "No reliable answer found" | favourite language 3.14 refused |
| Cite | page, chunk id, preview | "BERT stands for" → page 1, chunk 0 |

## 🎤 Answer These Out Loud

<details>
<summary><strong>How does an extractive QnA model produce an answer?</strong></summary>

Encoder over `[CLS] q [SEP] context [SEP]`; `Linear(H, 2)` gives each token a start and an end logit; choose the context-only pair maximizing $s_i + e_j$ with $i \le j$ and a length cap; slice the original text with `offset_mapping` (decode re-spaces punctuation: "encoder - only").
</details>

<details>
<summary><strong>How do you handle a document longer than 512 tokens?</strong></summary>

Chunk it with overlap (words or sentences), or let the tokenizer make strided windows (`return_overflowing_tokens=True, stride=128`: 19 windows for pages 1–4); read each and compare candidates. `truncation=True` alone silently drops text (92% of pages 1–4).
</details>

<details>
<summary><strong>Why does brute-force reading of all chunks rank badly?</strong></summary>

Logits are only comparable within one forward pass. Each chunk's best span is scored in isolation, so answer-shaped text elsewhere wins: right answer 20.123 vs "evaluation server" 20.049; 3 of 5 questions wrong at rank 1 with the right span in the top 3. Retrieval (RAG) picks relevant chunks first.
</details>

<details>
<summary><strong>Is the confidence score the probability that the answer is right?</strong></summary>

No. The notebook averages start and end softmaxes over all positions (0.5896 vs a span probability of 0.3382), and a SQuAD v1.1 model never abstains: "Which city was BERT invented in?" → "Transformers", 0.9964, span score 6.34 (passes a threshold of 5.0). Use a SQuAD 2.0 model (no-answer at [CLS]), calibrate thresholds, cite sources.
</details>

<details>
<summary><strong>How does a SQuAD record become training labels?</strong></summary>

`answer_start` is a character index (16). With `offset_mapping`, the start label is the context token containing character 16 (token 15) and the end label the token containing the last answer character (token 27). Pass `start_positions`, `end_positions` → loss = mean of the two cross-entropies (0.8377 on that record).
</details>

## 🔢 Numbers Worth Quoting

| Fact | Number |
|---|---|
| Reader | DistilBERT-base-cased, distilled SQuAD v1.1: 6 layers, 768, vocab 28,996 |
| Parameters | 65,192,450 = encoder 65,190,912 + `qa_outputs` 1,538 |
| Small example | "encoder-only Transformer", span 22.8177, notebook confidence 0.5896 |
| Unanswerable | "Who invented BERT?" → span score −11.9487 (vs 22.8 answerable) |
| BERT paper | 16 pages, 64,105 characters, about 17,900 tokens |
| Chunks | 180/40 words → 78 chunks; mean 293.83 tokens with the question |
| Near tie | 20.123 (right, page 1) vs 20.049 ("evaluation server", page 6) |
| Fine-tuning label | answer_start 16 → tokens 15..27, loss 0.8377 |

## 💻 Code You Should Be Able to Write

```python title="best_span.py" showLineNumbers
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

name = "distilbert-base-cased-distilled-squad"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForQuestionAnswering.from_pretrained(name).eval()

def best_span(question, context, max_length=500, L=40):
    enc = tok(question, context, truncation="only_second", max_length=max_length,
              return_offsets_mapping=True, return_tensors="pt")
    off, sid = enc.pop("offset_mapping")[0], enc.sequence_ids(0)
    with torch.no_grad():
        o = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    s, e = o.start_logits[0], o.end_logits[0]
    k = torch.arange(len(s)); c = torch.tensor([x == 1 for x in sid])
    # highlight-next-line
    ok = c[:, None] & c[None, :] & (k[None, :] >= k[:, None]) & (k[None, :] - k[:, None] < L)
    sc = (s[:, None] + e[None, :]).masked_fill(~ok, float("-inf"))
    i, j = divmod(sc.argmax().item(), len(s))
    return context[off[i][0]:off[j][1]], round(sc[i, j].item(), 4)

print(best_span("What type of model is BERT?",
                "BERT is an encoder-only Transformer model. It is widely used for natural language "
                "processing tasks such as question answering, text classification, and named entity recognition."))
# ('encoder-only Transformer', 22.8177)
```

## 🚨 Traps

- No `pipeline("question-answering")` in transformers 5: use `AutoModelForQuestionAnswering`.
- Two independent argmaxes can give end < start or a span in the question.
- `decode` ≠ source text; slice with offsets.
- SQuAD v1.1 models always answer; confidence ≠ correctness.
- Span scores aren't comparable across chunks.
- `truncation=True` drops text silently; budget in tokens (1.74 per word here).
- Literal "[CLS]"/"[SEP]" in documents become special tokens: `split_special_tokens=True`.
- `answer_start` is in characters; labels are tokens.
