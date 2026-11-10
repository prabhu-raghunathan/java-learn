---
title: "Cheat Sheet: Retriever + Reader with FAISS"
sidebar_position: 14
---

:::tip[What this sheet covers]
Why brute-force reading doesn't scale, sentence embeddings (all-MiniLM-L6-v2), cosine vs L2 on unit vectors, top-k retrieval, FAISS flat vs approximate indexes, score fusion, and how to evaluate retrieval separately from reading, from notebook 21. Full notes: [Retriever + Reader: Semantic Search with Embeddings and FAISS](../../03-ai-ml/03-deep-learning/07-hugging-face-qna-and-rag/03-retriever-reader-with-faiss.md).
:::

![Retriever + reader: chunks are embedded and indexed once; each question is embedded, the top-k chunks are found by cosine similarity, and only those are read](/img/deep-learning/diagrams/rr-architecture.svg)

## 🗺️ The Pipeline in One Table

| Step | What happens | Number from the run |
|---|---|---|
| Embed chunks (once) | `SentenceTransformer("all-MiniLM-L6-v2").encode(texts)` | (78, 384), unit length; 48 of 78 chunks over the 256-token limit |
| Index (once) | `faiss.IndexFlatIP(384).add(X)` (float32) | `ntotal` 78; row i = `chunks_data[i]` |
| Embed question | same model | about 8 ms on CPU |
| Search | `index.search(q, k)` → scores, rows | 0.006 ms for 78 vectors (flat) |
| Read | extractive reader on each of the k chunks | 5 calls instead of 78 |
| Pick | reader span score, or α·mm(cos) + (1−α)·mm(span) | α = 0.4 in the notebook |
| Cost | 1.16 s per question at k = 5 vs 18.70 s brute force (CPU) | 16× faster |

## 🎤 Answer These Out Loud

<details>
<summary><strong>How does dense retrieval work?</strong></summary>

A bi-encoder embeds every chunk once (mean pooling + L2 normalize → unit vectors). Per question: embed it with the same model, score all chunks with `E @ q` (cosine for unit vectors), keep the top k, map rows back to metadata, hand the chunks to the reader or LLM.
</details>

<details>
<summary><strong>Cosine, inner product or L2?</strong></summary>

For unit vectors they rank identically: $\cos = q \cdot v$ and $\lVert q - v \rVert^2 = 2 - 2\cos$ (notebook: 1.5268836 = 2 − 2 × 0.2365582). `IndexFlatL2` returns squared distances. If the model doesn't normalize, normalize both index and query (`faiss.normalize_L2`, in place), or IP and L2 will also respond to vector length.
</details>

<details>
<summary><strong>Is FAISS faster than NumPy?</strong></summary>

Not with flat indexes: they're exact scans (200,000 × 384: 27 ms vs 29 ms per query, same results). Approximate indexes are the speed-up: HNSW 0.17 ms at recall@10 0.96, IVF 0.3 ms at nprobe 8 (synthetic clustered data).
</details>

<details>
<summary><strong>How do you debug a wrong answer?</strong></summary>

Measure recall@k on labelled questions: is a chunk with the answer in the top k? BERT paper, 12 questions: recall@1/3/5/10 = 0.25/0.42/0.67/0.83. Brute force 8/12 vs top-5 retrieve-then-read 6/12; both misses were retrieval misses. Then fix chunking, k, hybrid search, reranking; only then the reader.
</details>

<details>
<summary><strong>What's wrong with min-max score blending?</strong></summary>

Min-max over the top k always maps the best candidate to 1 and the worst to 0: a 0.022 cosine gap (0.2366 vs 0.2143) became 1.0 vs 0.0 and flipped the notebook's answer. Use stable scales (raw cosine, softmax over span scores) or rank fusion, tuned on labelled data.
</details>

## 🔢 Numbers Worth Quoting

| Fact | Number |
|---|---|
| all-MiniLM-L6-v2 | 6 layers, 384-d, 12 heads, 22,713,216 params, truncates at 256 word pieces |
| "disadvantages" query | weaknesses 0.582, limitations 0.425, cat 0.006 |
| Reader time per chunk | 0.268 s CPU (0.022 s on the notebook's GPU) |
| Per question | brute force 18.70 s vs retrieve top 5 1.16 s (CPU) |
| Recall@k, 12 questions | 0.25 / 0.42 / 0.67 / 0.83 at k = 1 / 3 / 5 / 10 |
| Right answers | brute force 8, top-5 α 0.4: 6, retrieval-only ranking 3 |
| Vector memory | 1,536 bytes per 384-d float32 vector; 200,000 → 307.2 MB |

## 💻 Code You Should Be Able to Write

```python title="dense_retrieval.py" showLineNumbers
import numpy as np, faiss
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
E = embedder.encode(chunk_texts).astype("float32")          # (N, 384), unit length
index = faiss.IndexFlatIP(E.shape[1])
index.add(E)
assert index.ntotal == len(chunk_texts)                      # row i ↔ chunk i

def retrieve(question, k=5):
    q = embedder.encode([question]).astype("float32")        # shape (1, 384)
    # highlight-next-line
    scores, rows = index.search(q, k)
    return [(int(r), float(s)) for r, s in zip(rows[0], scores[0]) if r != -1]

# retrieve("Which corpus is used for pre-training besides English Wikipedia?")
# → [(1, 0.5174), (18, 0.475), (47, 0.4727), (19, 0.4657), (67, 0.4551)]   (rows of the 78 BERT-paper chunks)
```

## 🚨 Traps

- Same embedding model for chunks and questions.
- FAISS wants float32 `(n, d)`; `normalize_L2` is in place and returns `None`.
- FAISS returns row positions: keep metadata aligned or use `IndexIDMap`.
- Flat indexes are exact brute force; ANN indexes need a recall check.
- Embedding models truncate (MiniLM: 256 word pieces): chunk to the model.
- Vague questions retrieve reference lists; short tail chunks score high.
- Top-1 retrieval is rarely enough; evaluate recall@k first.
- Per-query min-max destroys absolute scale.
