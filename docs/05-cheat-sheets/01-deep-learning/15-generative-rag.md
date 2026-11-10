---
title: "Cheat Sheet: Generative RAG End to End"
sidebar_position: 15
---

:::tip[What this sheet covers]
The whole generative RAG system from notebook 22 on one screen: Transformer families, PDF → clean → sentence chunks → embeddings → FAISS, hybrid BM25 + dense search, token-budgeted prompts, FLAN-T5 decoding, evaluation by stage, routing, grounding and refusal, the Streamlit app, and when to fine-tune. Full notes: [Generative RAG End to End](../../03-ai-ml/03-deep-learning/07-hugging-face-qna-and-rag/04-generative-rag-end-to-end.md).
:::

![The whole RAG system: prepare the document once, then for each question search, gate on cosine, route to the extractive reader or the generator, and check grounding before answering](/img/deep-learning/diagrams/rag-end-to-end.svg)

## 🗺️ The Pipeline in One Table

| Step | What happens | Number from the run (BERT paper, CPU) |
|---|---|---|
| Read | `fitz.open(path)`, `page.get_text()` per page | 16 pages; 0 pages under 20 words |
| Clean | NFKC, join hyphen + newline, collapse spaces; keep `[CLS]`/`[SEP]`/`[MASK]` | paper uses them 28 / 25 / 11 times |
| Chunk | whole sentences, `max_words=250`, 2-sentence overlap, page + id | 54 chunks (notebook), 55 (this page) |
| Tokens | measure with the model's tokenizer | ~1.7 tokens/word; 48 of 55 chunks over MiniLM's 256 |
| Embed + index | MiniLM 384-d unit vectors in `IndexFlatIP` | cosine = dot product |
| Search | dense + BM25 fused by RRF ($c$ = 60) | recall@1: dense 0.50, BM25 0.50, hybrid 0.75 |
| Gate | refuse if best cosine below 0.2 | FIFA question 0.1408; lowest real question 0.3669 |
| Route | exact-fact patterns → extractive, else explain words → generative | check "how many" before "how" |
| Prompt | instruction + question + most relevant sentences, 360-token budget | 410 tokens: 33 + 13 + 3 + 359 + 2 |
| Generate | FLAN-T5-base, 4 beams, `early_stopping=False`, `no_repeat_ngram_size=3` | about 7–8 s per answer on CPU |
| Check | refuse if support below 0.5 | "22,500M" support 0.0 → refused |

## 🎤 Answer These Out Loud

<details>
<summary><strong>What is RAG and how is it different from using an LLM?</strong></summary>

Retrieve the most relevant chunks of your document, then generate the answer conditioned on the question **and** those chunks: $P(\pred{y} \mid \data{q}, \data{C_k})$. An LLM alone answers from training memory with no source; RAG answers from current, citable context. Retrieval failure (answer chunk not retrieved) and generation failure (right context, wrong answer) are different bugs.
</details>

<details>
<summary><strong>Encoder-only, decoder-only, encoder-decoder?</strong></summary>

Encoder-only (BERT, MiniLM): bidirectional attention, understanding, embeddings, start/end logits, can't generate. Decoder-only (GPT): causal mask, autoregressive generation. Encoder-decoder (FLAN-T5): bidirectional encoder reads the prompt once, causal decoder writes with cross-attention. flan-t5-small 76,961,152 params, flan-t5-base 247,577,856, 512 input tokens.
</details>

<details>
<summary><strong>Why hybrid search?</strong></summary>

Embeddings blur exact rare terms; BM25 matches them. "How many words are in English Wikipedia?": dense ranked the three answer chunks 4, 6, 22; BM25 ranked them 1, 2, 3; RRF put them at 1, 2, 4. RRF adds $1/(60 + \text{rank})$ from each list, so no score normalisation is needed.
</details>

<details>
<summary><strong>How do you build the prompt for a 512-token model?</strong></summary>

Instruction with a fixed fallback sentence, question, context, "Answer:". Budget in generator tokens. Keep metadata out (flan-t5-small answered with "[Source 1 | Page 3 | Chunk 6 | Score 0.5080]"). Pick the most relevant sentences, not the first tokens of each chunk: same model, 1 → 5 of 12 facts right.
</details>

<details>
<summary><strong>Greedy or beam search?</strong></summary>

Greedy takes the top token each step and can't revisit: NSP step 1 was We 0.134 vs To 0.113, and it dropped the "why". 4 beams found a complete answer with a better average log-probability (−0.0966 vs −0.1395). `early_stopping=True` returned a shorter, worse answer here. Keep `do_sample=False` for factual RAG.
</details>

<details>
<summary><strong>How do you stop hallucinations?</strong></summary>

Gate on retrieval (best cosine below 0.2 → refuse), route exact facts to an extractive reader, and check support (share of answer words in the context; below 0.5 → refuse). Limits: support misses wrong answers built from document words ("MNLI…", support 1.00), and a SQuAD v1.1 reader can't refuse ("BooksCorpus" for a GPU question).
</details>

<details>
<summary><strong>Your answers are bad. Fine-tune?</strong></summary>

Not first. Evaluate by stage on known-answer questions: 2 of 12 failures were retrieval misses. Fix extraction, cleaning, chunking, retrieval and prompts first; fine-tune the specific weak component (embedder, reranker, router, generator, reader) and measure before and after.
</details>

## 🔢 Numbers Worth Quoting

| Fact | Number |
|---|---|
| 12 factual questions, answer right | notebook config 1; generator with sentence selection 5; extractive reader 5; either 7 |
| Retrieval misses | 2 of 12 (answer chunks at hybrid ranks 10 and 11) |
| Recall@1 / @3 / @5 / @10 | dense 0.50 / 0.67 / 0.83 / 0.83; BM25 0.50 / 0.75 / 0.83 / 1.00; hybrid 0.75 / 0.83 / 0.83 / 0.92 |
| Notebook chunk tokens (FLAN-T5) | 124–522, mean 373.39, median 382 |
| Notebook balanced prompt | 67 instruction, 360 available, 306 used, 375 total |
| Greedy vs beam (NSP) | greedy 27 tokens, −3.768 total (−0.1395 avg); beam −0.0966 |
| Refusals | FIFA best cosine 0.1408; "22,500M" support 0.0 |

## 💻 Code You Should Be Able to Write

```python title="rag_core.py" showLineNumbers
def hybrid_search(index, bm25, question, k=5, c=60):
    dense_order = [r for r, _ in dense_search(index, question, index.ntotal)]
    bm25_order = list(np.argsort(-bm25.scores(question)))
    fused = Counter()
    for order in (dense_order, bm25_order):
        for rank, row in enumerate(order):
            # highlight-next-line
            fused[int(row)] += 1 / (c + rank + 1)                    # reciprocal rank fusion
    return [row for row, _ in fused.most_common(k)]

def support(answer, context):
    a = [w for w in words(answer) if w not in STOP]
    return sum(w in set(words(context)) for w in a) / len(a) if a else 0.0

# rag_answer: refuse if best cosine < 0.2 → route → extract (top 3 chunks) or generate (beam 4)
#             → refuse if support < 0.5 → return answer, route, reason, sources
```

## 🚨 Traps

- An LLM without retrieval isn't RAG.
- Don't delete document words while "cleaning" (`[CLS]` in an NLP paper).
- Budget in tokens; MiniLM truncates at 256, FLAN-T5 reads 512.
- High retrieval score ≠ answer present (figure labels scored 0.5080).
- No metadata inside the prompt; small models copy it.
- Count every passage you add to the budget, trimmed ones included.
- Router rules: exact-fact patterns before "how".
- SQuAD v1.1 readers always return a span.
- Word-overlap grounding catches invented words, not wrong quotes.
- Streamlit: `@st.cache_resource` or it re-indexes every click.
