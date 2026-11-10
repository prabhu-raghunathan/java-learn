---
title: "Retriever + Reader: Semantic Search with Embeddings and FAISS"
sidebar_position: 3
---

import {CosineVsL2, RetrieverReaderLab} from '@site/src/components/Widgets';

:::tip[TL;DR (30-Second Interview Pitch)]
Brute-force document QnA runs the expensive reader on **every** chunk, so its cost grows as $N \times T_{reader}$ and irrelevant chunks get a chance to win. A **retriever + reader** system splits the job: a cheap **retriever** finds the few chunks likely to hold the answer, and the **reader** (the extractive BERT model) reads only those. Notebook 21 builds the retriever from **sentence embeddings**: `all-MiniLM-L6-v2` turns every chunk and the question into a 384-number **unit vector**, **cosine similarity** (a dot product, since the vectors are unit length) ranks the chunks, and **FAISS** stores the vectors in an index and returns the top-$k$. On the BERT paper's 78 chunks, reading the top 5 instead of all 78 cut the time per question from 18.7 s to 1.16 s on a CPU. The catches are measurable: the retriever's **recall@5 was 0.67** on a 12-question test (it missed the answer chunk a third of the time), so the system answered 6 of 12 correctly against 8 of 12 for brute force; `IndexFlatIP` is an exact scan, no faster than NumPy (approximate indexes such as **HNSW** are what scale); and the notebook's min-max score blending stretches a 0.022 cosine gap into 0 vs 1.
:::

<FormulaKey />

:::info[Where this page sits]
**Before:** [Extractive Document QnA over a PDF](./02-extractive-document-qna.md) built brute-force QnA over the BERT paper and ended on its two problems: cost, and uncalibrated scores across chunks.
**This page:** notebook 21, *Retrival_Ext_QnA_Complete*: rebuild the pipeline, then add text embeddings, cosine-similarity search, a manual top-$k$ retriever, FAISS indexes, and a retriever + reader pipeline with combined scoring. Cosine similarity itself is defined on [NLP Foundations](../03-nlp-foundations/01-nlp-foundations-words-to-vectors.md#4-one-hot-encoding) and [NumPy](../../01-python-for-data-exploration/01-numpy/02-numpy-broadcasting-linalg-aggregations.md); here it is used, not re-derived.
**Next:** notebook 22, generative RAG: the same retriever, but an LLM writes the answer from the retrieved chunks.
:::

## 🧭 Start Here

The previous page ended with a system that worked and a system that couldn't grow. To answer one question it ran DistilBERT on all 78 chunks of the BERT paper, and it ranked the 78 answers by span scores that aren't comparable across chunks. Notebook 21 fixes the architecture rather than the scores: before reading anything, **search**. A search component (the **retriever**) picks the handful of chunks most likely to contain the answer; the reading component (the **reader**) extracts the span from those few. This page builds that system the way the notebook does, then measures it honestly.

The first half is about meaning as numbers. A **sentence-embedding model** turns any text into a fixed-length vector so that texts with similar meaning point in similar directions; "What are the disadvantages of the method?" lands close to "The method has some weaknesses…" though they share no keyword. Comparing the question's vector with every chunk's vector by **cosine similarity** and keeping the top $k$ is already a working retriever (Chapters 1–5). The second half is about engineering and judgement: wiring the retriever to the reader and combining their scores (Chapters 6–7), moving the vectors into **FAISS** and understanding what it does and doesn't speed up (Chapter 8), and finally **evaluating** the retriever separately from the reader, which is how you find out where a wrong answer came from (Chapters 9–10).

Every output on this page comes from re-running the notebook's code on CPU (transformers 5.22.2, sentence-transformers, faiss-cpu) against the same BERT paper PDF. The notebook's numbers are reproduced exactly (for example the cosine 0.2365582 and the L2 distance 1.5268836 in its FAISS cells); its Colab GPU timings differ from CPU timings, and both are quoted where they matter. The evaluation in Chapter 9 is new: 12 factual questions with known answers, written for this page because the notebook's questions ("What is the main idea of this document?") have no single right answer to check against.

| New concept | One-line meaning |
|---|---|
| **Retriever** | Finds the chunks most likely to contain the answer (search) |
| **Reader** | Extracts the answer span from one chunk (the QnA model of the previous page) |
| **Sentence embedding** | A fixed-length vector for a whole text, placing similar meanings close together |
| **Bi-encoder** | Question and chunks are embedded separately, so chunk vectors can be computed once in advance |
| **Semantic search** | Ranking texts by embedding similarity instead of shared keywords |
| **Top-$k$ retrieval** | Keep the $k$ most similar chunks and pass only those to the reader |
| **FAISS** | Facebook AI Similarity Search: a library of vector indexes for nearest-neighbour search |
| **`IndexFlatL2` / `IndexFlatIP`** | Exact FAISS indexes that score every stored vector by L2 distance / inner product |
| **ANN index** | Approximate nearest neighbour (IVF, HNSW): skips most vectors, trading a little recall for speed |
| **Combined score** | $\alpha$ × normalized retrieval score + $(1-\alpha)$ × normalized reader score |
| **Recall@$k$** | Fraction of questions whose answer-bearing chunk is among the top $k$ retrieved |

![Retriever + reader: chunks are embedded and indexed once; each question is embedded, the top-k chunks are found by cosine similarity, and only those are read](/img/deep-learning/diagrams/rr-architecture.svg)

*How to read it:* the top row runs once per document; the bottom row runs per question. FAISS only stores vectors, so the chunk text and page numbers live in a separate Python list that the returned row numbers index into.

:::analogy[A library: the catalogue finds the shelf, you read the page]
Brute force is reading every book in the library to answer one question. A retriever is the catalogue: it doesn't know the answer, but it sends you to the right shelf in seconds. The reader is you opening the few books on that shelf and finding the exact sentence. The notebook's own version: the retriever finds the room, the reader finds the sentence in the room. If the catalogue sends you to the wrong shelf, no amount of careful reading helps.
:::

## 1. Why Brute Force Doesn't Scale

Notebook 21 first rebuilds the previous system quickly, with three upgrades already folded in. Its reader is the robust version: it masks question and special tokens out of the span search, scores the top-20 starts × top-20 ends, slices answers out of the original text with `offset_mapping`, and handles long contexts with `stride=128` windows. Those are exactly the fixes the [previous page recommended](./02-extractive-document-qna.md#4-from-start-and-end-logits-to-an-answer-span). The text cleaning is also stronger (Unicode NFKC, ligatures, line-break hyphens), and the chunker still makes 78 chunks of up to 180 words with a 40-word overlap.

Then it measures the cost. If a document has $N$ chunks and one reader call takes $T_{reader}$ seconds, brute force costs $N \times T_{reader}$ per question: linear in the size of the corpus. On the lecture's Colab GPU one reader call took 0.022 s, so the 78-chunk paper took about 2 s per question. On a CPU it took **0.268 s per chunk** and **19.26 s per question** (the notebook's reader pads every input to 500 tokens, which makes CPU calls slower than they need to be). Scaled to 10,000 chunks, a modest company wiki, that's 2,676 s, **about 45 minutes per question** on this CPU.

If a retriever first picks $k$ chunks, the reader runs only $k$ times. The cost becomes the retriever's own time plus $k \times T_{reader}$, and since retrieval is a cheap vector comparison, the reader calls drop by a factor of $N / k$: 200× for 1,000 chunks at $k = 5$, 2,000× for 10,000. Measured end to end on this paper with $k = 5$: **1.16 s** per question instead of 18.70 s on CPU (0.13 s instead of 2.0 s on the notebook's GPU).

<Formula name="Cost per question" tag="Brute force vs retrieve-then-read" words="Brute force pays one reader call per chunk. Retrieve-then-read pays one question embedding, one vector search, and one reader call per retrieved chunk." remember="Read k chunks, not N.">

$$
\begin{aligned}
T_{brute} &= \data{N} \cdot T_{reader} \\
T_{rr} &= T_{embed} + T_{search} + \hyper{k} \cdot T_{reader}, \qquad \hyper{k} \ll \data{N}
\end{aligned}
$$

- $\data{N}$: chunks in the corpus; $\hyper{k}$: chunks retrieved; $T_{reader}$: one reader forward pass.
- $T_{embed}$: embedding the question (about 8 ms here); $T_{search}$: comparing it with the stored vectors (0.003 ms for 78 vectors).

**Example:** $\data{N}$ = 78, $T_{reader}$ = 0.268 s: brute force about 20.9 s estimated, 18.70 s measured; with $\hyper{k}$ = 5, 1.16 s measured.

</Formula>

Speed is half the motivation. The other half is the ranking problem from the previous page: a SQuAD reader always returns its best-looking span, so every irrelevant chunk is a chance for a confident wrong answer to out-score the right one. Reading fewer, more relevant chunks removes most of those chances, provided the retriever keeps the right one. Whether it does is an empirical question, and Chapter 9 answers it.

:::tip[🎯 Interview angle]
**"Why doesn't brute-force reading scale, and what replaces it?"** It runs the reader once per chunk per question, so latency grows linearly with the corpus: 0.27 s per chunk on CPU is 45 minutes per question at 10,000 chunks. A retriever embeds the question and finds the top-$k$ similar chunks with a cheap vector search, and the reader runs only $k$ times. On the BERT paper that was 1.16 s instead of 18.7 s at $k = 5$. It also stops irrelevant chunks from producing confident wrong spans, as long as the retriever's recall is good.
:::

## 2. Retriever, Reader, Embedding Model, Index: Four Different Jobs

Notebook 21 insists on keeping four components apart, and interviewers probe the same distinctions. The **embedding model** (`all-MiniLM-L6-v2`) turns text into a vector; it answers nothing. The **index** (FAISS) stores vectors and returns the nearest ones to a query vector; it knows nothing about text, pages or answers. The **retriever** is the procedure that uses both: embed the question, search the index, map the returned row numbers back to chunk metadata, return the top-$k$ chunks. The **reader** (`distilbert-base-cased-distilled-squad`) takes a question and one chunk and returns a span.

The embedding model and the reader are both Transformer encoders, but they are trained for different jobs and used differently. The reader sees question and chunk **together** in one input, so its attention can compare every question token with every chunk token; that is accurate but must be recomputed for every (question, chunk) pair. The embedding model sees each text **alone**, which is why it is called a **bi-encoder**: chunk vectors can be computed once, stored, and compared with any future question by a dot product. That separation is what makes retrieval cheap, and it is also its weakness, because the question and chunk never "see" each other inside the model.

| Component | Input → output | Runs | Cost per question |
|---|---|---|---|
| Embedding model (MiniLM) | text → 384-number unit vector | chunks once, question every time | one small forward pass for the question |
| Index (FAISS) | query vector → top-$k$ row numbers + scores | every question | a scan (flat) or a partial scan (ANN) |
| Retriever | question → top-$k$ chunks with metadata | every question | embed + search + list lookup |
| Reader (DistilBERT-SQuAD) | question + one chunk → span + score | $k$ times per question | $k$ full forward passes |

:::danger[Interview Trap: "FAISS answers the question"]
FAISS is not a QnA model and not even a text tool: it finds nearest vectors and returns their row numbers and scores. The embedding model decides what "near" means, your own list maps rows back to chunks, and the reader (or an LLM) produces the answer.
:::

:::tip[🎯 Interview angle]
**"What's the difference between the embedding model and the reader, if both are BERT-like?"** The embedding model is a bi-encoder: it encodes question and chunk separately into vectors, so chunk vectors are precomputed and comparison is a dot product, which is cheap but coarse. The reader is a cross-encoder style model: it reads question and chunk together in one input, so attention compares them token by token, which is accurate but must run per pair. Retrieve with the bi-encoder, read (or rerank) with the cross-encoder.
:::

## 3. Sentence Embeddings: Meaning as a Direction

The previous pages produced a vector per **token**. Retrieval needs one vector per **text**: per chunk and per question. `all-MiniLM-L6-v2` is a small sentence-embedding model built for exactly that. It is a 6-layer, 384-wide, 12-head encoder (22,713,216 parameters) with an uncased 30,522-WordPiece vocabulary, followed by two extra steps that the `SentenceTransformer` wrapper prints as its three modules: `Transformer`, `Pooling` and `Normalize`. **Mean pooling** averages the final token vectors (ignoring padding) into one 384-number vector, and **L2 normalization** scales it to length 1. So every embedding comes out as a **unit vector**: the norms of the notebook's four sample sentences are exactly `[1. 1. 1. 1.]`.

What makes these vectors useful is how the model was trained. Its model card says it was fine-tuned with a **contrastive objective** on over 1 billion sentence pairs (question–answer pairs, title–body pairs, paraphrases and similar): within each batch, the cosine similarity of every possible pair is computed and a cross-entropy loss pushes each sentence toward its true partner and away from the others. After that training, texts that tend to answer or paraphrase each other end up pointing in similar directions, regardless of wording. This is the contextual, sentence-level successor of the static word vectors of [NLP Foundations](../03-nlp-foundations/01-nlp-foundations-words-to-vectors.md#5-word-embeddings): one vector for the whole meaning of a text, computed from context.

The notebook's demonstration is the one to remember. The query "What are the disadvantages of the method?" is compared with five sentences, none of which contains "disadvantages". "The method has some **weaknesses** when applied to noisy data." scores **0.582**; "The **limitations** of the proposed approach are discussed in this section." scores **0.425**; a sentence about a new training strategy 0.227; one about strong benchmark performance 0.126; and "The cat is sleeping on the sofa." **0.006**. A keyword search would have found none of them; the embedding ranks the two relevant ones first.

<Animation src="rr/semantic-neighbours" caption="Each sentence is drawn at its real angle from the query, arccos of the cosine between their 384-number vectors. Weaknesses and limitations are closest; the unrelated sentence sits at almost 90 degrees." />

<Formula name="Sentence embedding (mean pooling + normalize)" tag="all-MiniLM-L6-v2" words="Average the final vectors of the real tokens, then scale the average to length one. The result is one direction per text." remember="Average the tokens, then unit length.">

$$
\begin{aligned}
\pred{u} &= \frac{\sum_{t} \data{m_t}\, \pred{h_t}}{\sum_t \data{m_t}} \\
\pred{v} &= \frac{\pred{u}}{\lVert \pred{u} \rVert_2}, \qquad \lVert \pred{v} \rVert_2 = 1
\end{aligned}
$$

- $\pred{h_t}$: final 384-number vector of token $t$; $\data{m_t}$: attention mask (1 for real tokens, 0 for padding).
- $\pred{v}$: the sentence embedding, a unit vector.

**Example:** "The model has several limitations." → a 384-number vector whose first values are 0.0143, −0.0644, 0.0059, … and whose length is 1.0.

</Formula>

There is a limit the notebook never checks. The model card states that input longer than **256 word pieces is truncated**, and the model was fine-tuned on sequences of at most 128 tokens. A 180-word chunk of this paper is about 250 MiniLM tokens on average, up to 393. Measured: **48 of the 78 chunks exceed 256 tokens**, and **10.3%** of all chunk tokens never reach the embedding. Whatever sits at the end of those chunks is invisible to the retriever, even though the reader would see it. Chunk size should be chosen for the **embedding model's** limit as well as the reader's: with 120-word chunks, only 2 of 110 exceed it.

:::analogy[A compass bearing for every paragraph]
An embedding gives every text a compass bearing in a 384-dimensional space. Texts about the same thing point the same way, however they're worded: "weaknesses" and "limitations" both point roughly north-north-east of "disadvantages". Cosine similarity just measures how far apart two bearings are. Length carries nothing, which is why the model normalizes it away.
:::

:::tip[🎯 Interview angle]
**"How does a sentence-embedding model produce one vector for a whole text, and why does it work for search?"** A Transformer encodes the tokens, mean pooling averages the token vectors over the attention mask, and the result is L2-normalized to a unit vector. It works for search because it was fine-tuned contrastively on pairs that belong together (question–answer, paraphrases), so related texts point in similar directions even without shared words. Watch its input limit: `all-MiniLM-L6-v2` truncates at 256 word pieces, which cut off part of 48 of the notebook's 78 chunks.
:::

## 4. Cosine Similarity, Dot Product and L2 on Unit Vectors

Cosine similarity, the dot product divided by both lengths, was introduced on the [NLP Foundations page](../03-nlp-foundations/01-nlp-foundations-words-to-vectors.md#4-one-hot-encoding) and computed in bulk on the [NumPy page](../../01-python-for-data-exploration/01-numpy/02-numpy-broadcasting-linalg-aggregations.md) (normalize the rows, then one matrix product gives every similarity). The notebook re-checks it on toy vectors: same direction 0.9999999999, perpendicular 0.0, opposite −0.9999999999 (the 1e-10 in its denominator keeps those just short of ±1).

What matters for retrieval is what happens when all vectors have length 1, as MiniLM's do. Then the denominator is 1 and **cosine similarity is just the dot product**. And the squared Euclidean distance becomes a simple function of it: expanding $\lVert q - v \rVert^2 = \lVert q \rVert^2 + \lVert v \rVert^2 - 2\, q \cdot v = 2 - 2\cos\theta$. So for unit vectors the three measures always rank documents identically: highest cosine = highest dot product = smallest L2 distance. That is visible in the notebook's own output: `IndexFlatL2` returned distance **1.5268836** for chunk 51, and $2 - 2 \times 0.2365582$ (its cosine from `IndexFlatIP`) is exactly 1.5268836. Both indexes returned the same five chunks in the same order.

<Formula name="Unit vectors: cosine, dot product and L2 agree" tag="Retrieval metrics" words="When every vector has length one, cosine similarity is the dot product, and squared Euclidean distance is two minus twice the cosine. All three rank neighbours the same way." remember="Unit length: IP = cos, L2² = 2 − 2cos.">

$$
\begin{aligned}
\lVert \pred{q} \rVert = \lVert \pred{v} \rVert = 1 \;\Rightarrow\; \cos\theta &= \pred{q} \cdot \pred{v} \\
\lVert \pred{q} - \pred{v} \rVert_2^2 &= 2 - 2\cos\theta
\end{aligned}
$$

- $\pred{q}$: question embedding; $\pred{v}$: chunk embedding; $\theta$: the angle between them.

**Example:** "What is the main idea of this document?" vs chunk 51: cosine 0.2365582, `IndexFlatL2` distance 1.5268836 = 2 − 2 × 0.2365582.

</Formula>

This also explains a detail the notebook gets slightly wrong. Its "Version 2" FAISS index normalizes the embeddings with `faiss.normalize_L2` before switching to inner product, presenting that as the improvement over the L2 index. With this model the embeddings were already unit length, so normalizing changed nothing (`np.allclose` before and after is `True`) and the L2 index was already a cosine ranking. The habit is still right, because many embedding models do **not** normalize their output, and for those the metric matters: with unequal lengths, inner product favours long vectors and L2 distance penalizes them, so the two can disagree with cosine. Note also that FAISS's `IndexFlatL2` returns the **squared** distance, not the distance the notebook's formula calls Euclidean.

<CosineVsL2 />

:::danger[Interview Trap: "inner product search is cosine search"]
Only for unit-length vectors. Normalize both the stored vectors and every query (`faiss.normalize_L2`, in place, float32), or use a model that normalizes its output. Mixing a normalized index with an unnormalized query silently ranks by length as well as direction.
:::

:::tip[🎯 Interview angle]
**"L2 index or inner-product index for sentence embeddings?"** If the embeddings are unit length, it doesn't matter for ranking: $\lVert q - v \rVert^2 = 2 - 2\cos\theta$, so both give the cosine order (the notebook's two indexes returned identical results). Use `IndexFlatIP` so the scores read directly as cosines, and normalize stored and query vectors if the model doesn't. If vectors aren't normalized, inner product and L2 can rank differently from cosine, because they also respond to vector length.
:::

## 5. A Manual Top-k Retriever, and What It Retrieves

The notebook's first retriever is deliberately plain. Embed all 78 chunks once (a `(78, 384)` array, 2.76 s on CPU, 0.32 s on the GPU); for each question, embed it, compute the cosine with every chunk in a Python loop, sort, and return the top $k$ chunks with their page, chunk id, word count and score. The loop is fine for 78 chunks; the vectorized form is one matrix product, `E @ q`, followed by `np.argpartition` to find the top $k$ without sorting everything. Either way, retrieval is an exact scan over all chunk vectors: the reader is no longer the bottleneck, but the retriever still touches every vector.

<Formula name="Top-k retrieval" tag="Dense retrieval" words="Score every chunk by its cosine with the question, which for unit vectors is one matrix product, and keep the k highest-scoring chunks." remember="One matmul, then top-k.">

$$
\begin{aligned}
\pred{s} &= \data{E}\, \pred{q} \in \mathbb{R}^{\data{N}} \\
R_{\hyper{k}}(q) &= \operatorname{top\text{-}k}_i\; \pred{s_i}
\end{aligned}
$$

- $\data{E}$: the $\data{N} \times 384$ matrix of unit chunk embeddings; $\pred{q}$: the unit question embedding.
- $R_{\hyper{k}}(q)$: the indices of the $\hyper{k}$ largest scores, mapped back to chunk metadata.

**Example:** "What full form of BERT?" → chunk 14 (page 4, 0.5556), chunk 13 (page 4, 0.517), chunk 58 (page 12, 0.4975), chunk 61 (page 13, 0.4821), chunk 0 (page 1, 0.4669). The answer is in chunk 0, retrieved only fifth.

</Formula>

Reading the notebook's retrieval outputs closely teaches three practical lessons. **Vague questions retrieve noise.** For "What is the main idea of this document?", "What is the main contribution of the document?" and the like, every top-5 cosine is around 0.21–0.25, and the winners are chunks from the **references** section (chunks 51, 53, 54, 57) and a figure caption: lists of paper titles look "about" research contributions to an embedding model. The reader then dutifully extracts paper titles, such as "Attention is all you need" or "Distributed representations of sentences and documents". Specific questions get much higher, more meaningful scores (0.55 for the full form of BERT, 0.52 for the pre-training corpus).

**Short chunks win.** "What are the limitations mentioned?" retrieves chunk 40 first with 0.3045: it is a 27-word leftover at the end of page 8. The notebook's new chunker has no minimum length, so 6 chunks have fewer than 60 words, and a short text's mean-pooled vector isn't diluted by other topics, so it tends to score high on vague questions. Filtering or merging tail chunks fixes this.

**The answer is often not at rank 1.** Even for "What full form of BERT?", chunk 0 (the abstract, which states the full form) is fifth, behind the "Input/Output Representations" section that mentions BERT's inputs many times. That is the strongest argument for retrieving $k > 1$ chunks and letting the reader decide.

:::tip[🎯 Interview angle]
**"How do you implement top-k semantic retrieval without a vector database?"** Embed all chunks once into an $N \times d$ matrix of unit vectors; per question, embed it, compute `scores = E @ q` (cosines), take the top $k$ with `argpartition` and map indices back to chunk metadata. It's exact and fast for thousands of chunks. Watch the failure modes: vague questions match boilerplate like reference lists, short fragments score high, and the answer chunk is often not rank 1, so retrieve several and let the reader choose.
:::

## 6. Retrieve, Then Read

Wiring the two halves together is short. `retrieve_then_answer_simple` retrieves the top $k$ = 5 chunks, runs the notebook's reader on each, keeps the valid spans, and ranks them by the **reader's span score**, returning the best answer with its page, chunk, retrieval rank and scores. Five reader calls instead of 78.

<Animation src="rr/retrieve-then-read" caption="All 78 cosines for the corpus question, the 5 kept, and the reader's span in each. Retrieval rank 1 holds no answer; the reader finds BooksCorpus in the chunk ranked 4, and the combined score picks it." />

The best case for the architecture looks like this, on a question from the evaluation set: "Which corpus is used for pre-training besides English Wikipedia?" The retriever's top 5 are chunk 1 (0.5174), chunk 18 (0.4750), chunk 47 (0.4727), chunk 19 (0.4657) and chunk 67 (0.4551). The highest-ranked chunk contains no answer; the reader extracts "Introduction Language" there with a span score of −4.65. In chunk 19, ranked only fourth by the retriever, the reader finds "**BooksCorpus**" with a span score of **25.28**, and chunk 67 repeats it (18.78). Retrieval narrowed 78 chunks to 5 and kept the right one; reading picked it out. Neither half alone would have got there: retrieval rank 1 is wrong, and the reader alone would have to read everything.

The notebook's own demonstration question shows the opposite case. For "What is the main idea of this document?", all five retrieved chunks are bibliography or figure text, so the reader's best span is a cited paper title ("Attention is all you need", span score 0.73). The pipeline worked as designed; the question and the retrieval were the problem. The reader can only choose among what the retriever hands it.

| Stage | "Which corpus is used besides English Wikipedia?" | "What is the main idea of this document?" |
|---|---|---|
| Top cosine | 0.5174 | 0.2366 |
| Answer chunk retrieved? | yes, rank 4 (and 5) | no clear answer chunk exists |
| Reader's best span | BooksCorpus (25.28) | Attention is all you need (0.73) |
| Outcome | right | meaningless |

:::tip[🎯 Interview angle]
**"Walk me through a retrieve-then-read query."** Embed the question with the same model used for the chunks; find the top-$k$ chunks by cosine; run the extractive reader on each with the question; keep valid spans; pick the answer by the reader's span score (or a combination with the retrieval score); return the answer with its page and chunk. In the BERT-paper example the right chunk was retrieval rank 4, and the reader's span score (25.28) is what found it, which is why $k$ is more than 1.
:::

## 7. Combining Retrieval and Reader Scores

The notebook's "better" pipeline argues, reasonably, that an answer from a highly relevant chunk deserves more trust than one from a weakly retrieved chunk, so it combines both signals. Because cosines live around 0.2–0.6 and span scores around −10 to 25, it first **min-max normalizes** each list over the valid candidates (subtract the minimum, divide by the range) and then mixes them with a weight $\alpha$ = 0.4 on retrieval.

<Formula name="Combined answer score (notebook)" tag="Score fusion" words="Rescale the retrieval cosines and the reader span scores of the candidates to 0 to 1 each, then take a weighted average with weight alpha on retrieval." remember="Min-max both, then blend with alpha.">

$$
\begin{aligned}
\operatorname{mm}(x_j) &= \frac{x_j - \min_i x_i}{\max_i x_i - \min_i x_i + \epsilon} \\
\pred{f_j} &= \hyper{\alpha}\, \operatorname{mm}(\pred{s^{ret}_j}) + (1 - \hyper{\alpha})\, \operatorname{mm}(\pred{s^{read}_j})
\end{aligned}
$$

- $\pred{s^{ret}_j}$: cosine of candidate $j$'s chunk; $\pred{s^{read}_j}$: its span score; $\epsilon$ = 1e-10.
- $\hyper{\alpha}$: weight on retrieval (0.4 in the notebook); the answer is the candidate with the largest $\pred{f_j}$.

**Example:** the corpus question, $\hyper{\alpha}$ = 0.4: final scores 0.4646, 0.4647, 0.1126, **0.6677** (BooksCorpus, rank 4), 0.4836, so BooksCorpus wins.

</Formula>

The idea of fusing signals is right; min-max over the top $k$ is a fragile way to do it. It rescales **relative to whatever candidates happen to be present**, so the best one always gets 1 and the worst always gets 0, whatever their absolute quality. For the "main idea" question, the five cosines span only 0.2143 to 0.2366, a difference of 0.022 between equally poor chunks; min-max stretches that into a full 0-to-1 range, so the rank-1 chunk gets the whole retrieval bonus (0.4) and rank 5 none. That flips the answer: reader-only ranking picked "Attention is all you need" (rank 5), the combined score picks "Distributed representations of sentences and documents" (rank 1, final 0.9745). Both are paper titles from the references, so neither is right, but the flip was caused by noise. Two more edge cases: with a single valid candidate, both normalized scores are 0 (range 0), and adding or removing one distant candidate changes everyone else's normalized scores.

<Animation src="rr/minmax-stretch" caption="The main-idea question's five retrieval cosines, 0.2366 down to 0.2143, before and after min-max. A 0.022 gap becomes a 1.0 gap, so retrieval rank 1 gets all the credit and rank 5 none." />

Better options keep scores on a fixed scale. Cosine is already bounded, so it can be used raw; span scores can be turned into a probability over the candidates with a softmax, or replaced by a calibrated score; and $\alpha$ should be chosen on labelled questions, not by hand. On the 12-question evaluation of Chapter 9 with the notebook's chunks and $k$ = 5, the notebook's blend ($\alpha$ = 0.4) answered 6 correctly, reader-only ranking ($\alpha$ = 0) 5, and retrieval-only ranking ($\alpha$ = 1) just 3: the reader's span score carries most of the signal about *where in the retrieved set the answer is*.

:::analogy[Grading on a curve in a class of five]
Min-max is grading on a curve in a class of five: the top student gets 100 and the bottom gets 0, even if all five scored between 21 and 24 out of 100 on the real test. If you then average that curved grade with another subject, a 1-mark difference on the original test has become worth 100 points. Fine when the class is large and spread out; misleading when everyone is equally weak.
:::

:::danger[Interview Trap: "normalize both scores to 0–1 and average them"]
Per-query min-max over the top-$k$ removes the absolute level of the scores: weak candidates look perfect if they're the best of a bad set, and tiny gaps become maximal. Use scores on a fixed scale (raw cosine, softmax-normalized reader scores, or rank-based fusion such as reciprocal rank fusion) and tune the weights on labelled data.
:::

:::tip[🎯 Interview angle]
**"How would you combine the retriever's score and the reader's score?"** Both carry information: retrieval says the chunk is on topic, the reader says a strong span exists in it. Put them on comparable, stable scales first (cosine is already bounded; turn span scores into probabilities over candidates), then combine with a weight tuned on a labelled set, or use rank-based fusion. Avoid min-max over the top $k$: it maps the best candidate to 1 and the worst to 0 regardless of absolute quality, so a 0.022 cosine gap became a full point in the notebook's example.
:::

## 8. FAISS: Exact Indexes, Approximate Indexes, and What Actually Gets Faster

**FAISS** (Facebook AI Similarity Search) is a library of data structures for finding the nearest vectors to a query vector. The notebook uses its simplest family, the **flat** indexes. `faiss.IndexFlatL2(384)` and `faiss.IndexFlatIP(384)` store the vectors as they are (`is_trained` is `True` immediately, because there's nothing to learn), `index.add(X)` appends a float32 array (`ntotal` goes from 0 to 78), and `index.search(Q, k)` returns two `(n_queries, k)` arrays: the scores (squared L2 distances, smaller is better; or inner products, larger is better) and the **row numbers** of the matches, with `-1` padding when fewer than $k$ exist.

Three practical points sit around those calls. FAISS wants **float32, C-contiguous, 2-D** arrays: a single query must be shaped `(1, 384)`. `faiss.normalize_L2` normalizes **in place** and returns `None`, which is why the notebook normalizes a `.copy()`. And FAISS returns **row positions, not your ids**: the notebook relies on row $i$ being `chunks_data[i]`, and its improved retriever checks `faiss_index.ntotal == len(chunks)` because a mismatch would silently attach the wrong page to an answer. Wrapping the index in `faiss.IndexIDMap` and adding vectors with `add_with_ids` makes the index return your own ids instead (for example 1051, 1057, 1071 for chunks 51, 57, 71 when ids are offset by 1000); `faiss.write_index` / `read_index` save and reload it so chunks are embedded only once.

Then the claim worth correcting. The notebook presents FAISS as "faster and more scalable" than the manual retriever, and its timing table shows them almost equal (0.0100 s vs 0.0092 s per query here; 0.042 vs 0.038 s on the GPU). That's expected, because **a flat index is the same exact brute-force scan** as `E @ q`, just in optimized C++. Splitting the time shows where it goes: embedding the question took about **8 ms**, the NumPy scan of 78 vectors **0.0025 ms** and the FAISS flat search **0.006 ms**. At 200,000 synthetic 384-dimensional vectors, single-threaded, a NumPy scan took 28.8 ms per query and `IndexFlatIP` 27.0 ms, with identical top-10 results. FAISS's real speed-ups come from **approximate** nearest-neighbour (ANN) indexes, which avoid scoring most vectors.

![Three ways to find the nearest vectors: exact flat scan, IVF clusters, HNSW graph](/img/deep-learning/diagrams/faiss-index-types.svg)

*How to read it:* a flat index scores every vector; IVF clusters the vectors with k-means and scans only the clusters nearest the query; HNSW walks a proximity graph greedily towards the query.

**IVF** (inverted file, `IndexIVFFlat`) first learns `nlist` centroids with k-means (`index.train`), assigns each vector to its nearest centroid, and at query time scans only the `nprobe` closest clusters. **HNSW** (hierarchical navigable small world, `IndexHNSWFlat`) links every vector to a few near neighbours in a layered graph and answers a query by greedy hops from an entry point, with `efSearch` controlling how wide the search beam is. Both trade a little **recall** (the fraction of the true top-$k$ they return) for large speed-ups, tuned per query by `nprobe` or `efSearch`.

| Index (200,000 × 384, single thread) | ms per query | Recall@10 vs exact | Build |
|---|---|---|---|
| NumPy `X @ q` + `argpartition` | 28.84 | 1.0 (exact) | none |
| `IndexFlatIP` | 27.05 | 1.0 (exact) | none (307.2 MB of float32) |
| `IndexHNSWFlat` (M = 32), efSearch 16 / 64 / 256 | 0.092 / 0.167 / 0.711 | 0.807 / 0.960 / 0.990 | 29.3 s |
| `IndexIVFFlat` (nlist 1024), nprobe 1 / 8 / 32 | 0.094 / 0.316 / 0.967 | 0.997 / 1.0 / 1.0 | 20.4 s (train + add) |

*Synthetic data: 2,000 clusters of unit vectors, queries are noisy copies of stored vectors (mean cosine 0.958 to their source). Clustered data flatters IVF; real embeddings usually sit in between. One run per setting.*

:::analogy[Sorting the post by postcode, and asking friends of friends]
A flat index reads the address on every letter in the sorting office. IVF sorts letters into postcode bins first and only searches the one or two bins whose postcode is closest: much faster, and only occasionally wrong when a letter sits near a bin boundary. HNSW is finding someone by asking a person who knows people near them, who asks someone nearer still: a few hops instead of a full search, with a small chance of getting stuck one hop away.
:::

:::tip[🎯 Interview angle]
**"Is FAISS faster than computing cosine similarity with NumPy?"** Not with `IndexFlatIP`/`IndexFlatL2`: those are exact scans, and on 200,000 vectors they took 27 ms vs NumPy's 29 ms single-threaded with identical results. FAISS pays off with approximate indexes: HNSW answered in 0.17 ms at 0.96 recall@10, IVF in 0.3 ms with nprobe 8. Choose by corpus size and latency budget, and measure recall against an exact index. For a 78-chunk PDF, question embedding (8 ms) dwarfs search time either way.
:::

## 9. Evaluating Retriever and Reader Separately

When a retrieve-then-read system gives a wrong answer, there are two very different causes: the retriever never returned the chunk with the answer (**retrieval failure**), or it did and the reader didn't pick the right span from it (**reader failure**). The notebook names this debugging step but stops before doing it, and its questions have no checkable answers. So this page adds a small labelled test: 12 factual questions about the BERT paper (its full form, the masking rate, BERT-large's layers, the pre-training corpora, the GLUE score, the parameter count, the activation function, the vocabulary size, the learning rate, the QA dataset…), each with the answer string to look for.

The retriever's metric is **recall@k**: the fraction of questions for which at least one chunk containing the answer is among the top $k$. With the notebook's 180-word chunks it was **0.25 at k = 1, 0.42 at k = 3, 0.67 at k = 5 and 0.83 at k = 10**. Two questions ("What does BERT stand for?" and "Which activation function does BERT use?") had no answer chunk even in the top 10. The end-to-end metric is simply how many answers were right.

<Formula name="Recall at k" tag="Retriever evaluation" words="For each labelled question, check whether any chunk containing the answer is among the top k retrieved; recall at k is the fraction of questions where it is." remember="Was the answer even handed over?">

$$
\text{Recall@}\hyper{k} = \frac{1}{\lvert \data{Q} \rvert} \sum_{q \in \data{Q}} \mathbb{1}\big[\, R_{\hyper{k}}(q) \cap \data{G}(q) \neq \varnothing \,\big]
$$

- $\data{Q}$: the labelled questions; $\data{G}(q)$: the chunks that contain the answer to $q$; $R_{\hyper{k}}(q)$: the top-$\hyper{k}$ retrieved chunks.

**Example:** notebook chunks, $\hyper{k}$ = 5: the answer chunk was retrieved for 8 of 12 questions, Recall@5 = 0.67.

</Formula>

| System (12 questions, notebook's 78 chunks unless stated) | Reader calls | Right answers |
|---|---|---|
| Brute force: reader on all 78 chunks, best span score | 78 | **8** |
| Retrieve top 1, read it | 1 | 3 |
| Retrieve top 5, rank by reader score ($\alpha$ = 0) | 5 | 5 |
| Retrieve top 5, notebook blend ($\alpha$ = 0.4) | 5 | 6 |
| Retrieve top 5, rank by retrieval only ($\alpha$ = 1) | 5 | 3 |
| Retrieve top 10, notebook blend ($\alpha$ = 0.4) | 10 | 7 |
| Notebook's exact `faiss_retrieve_then_answer` (k 5, $\alpha$ 0.4, min 10 answer characters) | 5 | 6 |
| 120-word chunks (110), top 5, $\alpha$ = 0.4 | 5 | 6 |

The table says something the notebook's narrative doesn't: on a single small document, **retrieval traded accuracy for speed**. Brute force read every chunk and got 8 right; retrieving 5 chunks was 16× faster and got 6. The gap is entirely retrieval failure. The two questions brute force got right and the top-5 pipeline got wrong are "What does BERT stand for?" (the answer chunk wasn't even in the top 10) and "What is the size of the WordPiece vocabulary?" (answer chunk at rank 6): in both, the reader never saw the right chunk. On every question where the answer chunk was in the top 5, the two systems agreed. The reader is not the bottleneck here; recall is. That is the normal state of real retrieve-then-read systems, and it's why the first thing to measure is recall@k on labelled questions, before tuning the reader or the score blend.

<RetrieverReaderLab />

Smaller chunks (120 words, which fit MiniLM's 256-token limit) raised recall@5 from 0.67 to 0.83 but not the end-to-end count at $k$ = 5 (6 of 12): with more, shorter chunks, more of them are near-duplicates that hand the reader extra distractors. The usual next steps in production are **hybrid search** (combine dense retrieval with keyword search such as BM25, which catches exact terms like "gelu" or "30,000" that embeddings blur), a **cross-encoder reranker** over the top 20–50, better chunk boundaries, metadata filters (here, excluding the references section would have removed the vague questions' noise), and **query rewriting**.

:::analogy[Two exams, two grades]
Grade the retriever and the reader separately, like two people in a relay. If the baton never reaches the second runner, it doesn't matter how fast they are. Recall@k is whether the baton was handed over; reader accuracy given the right chunk is how well the second runner runs. Here the handover failed a third of the time, so training the second runner harder would not help.
:::

:::tip[🎯 Interview angle]
**"Your RAG/QA system gives wrong answers. How do you debug it?"** Separate retrieval from reading with a labelled set. Measure recall@k: is a chunk containing the answer in the top $k$? If not, it's a retrieval problem: chunking (size versus the embedding model's token limit, short fragments), the embedding model, $k$, hybrid search, reranking, filters. If yes, check the reader on that chunk. On the BERT paper, recall@5 was 0.67 and brute force (no retrieval) beat retrieve-then-read 8 to 6 of 12, so retrieval was the main bottleneck.
:::

## 10. From Retrieve-then-Read to Generative RAG

The notebook ends where the next one begins. What has been built is **retrieval-based extractive QnA**: retrieval chooses the context, an extractive reader copies an answer span out of it. Replace the reader with a generator (an LLM given the question plus the retrieved chunks as context and instructed to answer from them) and the same pipeline becomes **retrieval-augmented generation**. Everything before the reader carries over unchanged: chunking, embeddings, the vector index, top-$k$ retrieval, metadata for citations, and recall@k as the first metric to check.

| | Brute-force extractive (previous page) | Retrieve + extractive reader (this page) | Retrieve + generator (notebook 22) |
|---|---|---|---|
| Chunks read | all $N$ | top $k$ | top $k$ |
| Answer | copied span | copied span | generated text |
| Answer can cover several chunks? | no | no | yes |
| Hallucination risk | none | none | present; reduced by grounding |
| Main failure | cost, uncalibrated ranking | retrieval recall | retrieval recall, unfaithful generation |

Production systems swap FAISS for a **vector database** (Chroma, Pinecone, Weaviate, Milvus, Elasticsearch with vector search). They are the same idea with persistence, metadata filtering, updates and replication around an ANN index.

:::tip[🎯 Interview angle]
**"How does this retriever-reader system relate to RAG?"** It is the retrieval half of RAG with an extractive reader as the answerer. RAG keeps chunking, embedding, indexing and top-$k$ retrieval, and replaces the span extractor with an LLM that writes an answer conditioned on the retrieved chunks. Retrieval quality (recall@k) still bounds answer quality, and grounding plus citations are what limit hallucination.
:::

## ⚙️ Going Deeper (Retrieval Internals)

**Why bi-encoders are fast and cross-encoders are accurate.** A bi-encoder computes $v_i = f(c_i)$ once per chunk and $q = f(\text{question})$ once per query; relevance is $q \cdot v_i$, so a million chunks cost one matrix product, or an ANN lookup. A cross-encoder computes $g(\text{question}, c_i)$ jointly for every pair, so attention can match the question's words to the chunk's words, but nothing can be precomputed. Standard practice is retrieve-then-rerank: a bi-encoder fetches 50–100 candidates and a cross-encoder reorders them. The extractive reader is itself a cross-encoder over (question, chunk).

**Memory.** A float32 384-dimensional vector is 1,536 bytes: 78 chunks take 119,808 bytes, a million chunks about 1.5 GB and 200,000 the 307.2 MB measured above. HNSW adds the graph links on top (M = 32 neighbours per node at the base layer). Product quantization (`IndexIVFPQ`) compresses vectors to a few dozen bytes each at some cost in recall, which is how billion-scale indexes fit in memory.

**Training length vs inference length.** `all-MiniLM-L6-v2` was fine-tuned on sequences of at most 128 tokens and truncates at 256 by default. Raising `max_seq_length` up to its 512 position limit makes it read longer chunks, but the model never learned to summarize text that long, so quality can drop. The robust fix is chunking to the model: here, 120-word chunks (2 of 110 over 256 tokens) instead of 180-word chunks (48 of 78).

**Exact search is fine longer than you think.** On a CPU core, an exact scan of 200,000 vectors took about 28 ms. For corpora up to a few hundred thousand chunks with modest traffic, a flat index (or a NumPy matrix product) is simpler, exact and fast enough. Reach for IVF/HNSW when the corpus or the query rate grows, and always report recall against the exact result.

**Cleaning has side effects.** The notebook's improved cleaner joins every line-break hyphen (`(\w)-\s*\n\s*(\w)` → `\1\2`), fixing "representa-/tion" but also turning the genuine compounds "left-to-/right" into "left-toright" (3 times) and "right-to-/left" into "right-toleft". Dictionary-aware dehyphenation fixes this. Whatever the cleaning, apply the same normalization to anything compared against the chunks later.

**Evaluating at scale.** Twelve questions is a sanity check, not a benchmark. Real evaluations use hundreds of labelled questions with recall@k and MRR (mean reciprocal rank of the first relevant chunk) for the retriever, exact match and F1 for the reader, and end-to-end accuracy; differences of one or two questions out of twelve, like those in Chapter 9, are within noise.

## 💻 Python Implementation

:::tip[Run it yourself]
Every block was run in order in one Python session (Python 3.13, sentence-transformers with `all-MiniLM-L6-v2`, faiss-cpu, transformers 5.22.2, CPU), after rebuilding the notebook's cleaned pages, its 78 chunks (`chunks_data`) and its reader (`qa_on_single_chunk`) from Sections 2–4. Outputs are copied from that run.
:::

### 1. Load the embedding model and look at what it outputs

Three modules, unit-length 384-number vectors, a 256-token input limit.

```python showLineNumbers
import numpy as np
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
E4 = embedder.encode(["The model has several limitations.", "The method has some disadvantages.",
                      "The cat is sleeping on the sofa.", "The results show strong performance."])
print(E4.shape, E4.dtype)                        # (4, 384) float32
# highlight-next-line
print(np.linalg.norm(E4, axis=1).round(6))       # [1. 1. 1. 1.]   already unit length
print([type(m).__name__ for m in embedder])      # ['Transformer', 'Pooling', 'Normalize']
print(embedder.max_seq_length, sum(p.numel() for p in embedder.parameters()))   # 256 22713216
```

### 2. Semantic similarity without shared words

Because the vectors are unit length, one matrix product gives the cosines.

```python showLineNumbers
query = "What are the disadvantages of the method?"
cands = ["The limitations of the proposed approach are discussed in this section.",
         "The model achieves strong performance on several benchmark datasets.",
         "The cat is sleeping on the sofa.",
         "The method has some weaknesses when applied to noisy data.",
         "The paper introduces a new training strategy."]
qv = embedder.encode(query)
C = embedder.encode(cands)
# highlight-next-line
sims = C @ qv                                    # cosine = dot product for unit vectors
for i in np.argsort(-sims):
    print(f"{sims[i]:.6f}  {cands[i]}")
# 0.582322  The method has some weaknesses when applied to noisy data.
# 0.425186  The limitations of the proposed approach are discussed in this section.
# 0.227246  The paper introduces a new training strategy.
# 0.126243  The model achieves strong performance on several benchmark datasets.
# 0.006088  The cat is sleeping on the sofa.
```

### 3. Embed the chunks, and check the token limit

The notebook never checks how many chunks MiniLM truncates.

```python showLineNumbers
texts = [c["chunk_text"] for c in chunks_data]
E = embedder.encode(texts)
print(E.shape)                                   # (78, 384)
n_tok = [len(embedder.tokenizer(t)["input_ids"]) for t in texts]
# error-next-line
print(sum(n > 256 for n in n_tok), max(n_tok), round(100 * sum(max(0, n - 256) for n in n_tok) / sum(n_tok), 1))
# ❌ 48 393 10.3   → 48 chunks truncated, 10.3% of chunk tokens never embedded
```

### 4. A vectorized top-k retriever

One matrix product and `argpartition` replace the notebook's per-chunk Python loop; same results.

```python showLineNumbers
def retrieve(question, k=5):
    s = E @ embedder.encode(question)            # cosine with all 78 chunks
    # highlight-start
    top = np.argpartition(-s, k)[:k]             # the k best, unordered, in O(N)
    top = top[np.argsort(-s[top])]               # order just those k
    # highlight-end
    return [(int(i), chunks_data[i]["page_number"], round(float(s[i]), 4), chunks_data[i]["word_count"]) for i in top]

for r in retrieve("What are the limitations mentioned?"):
    print(r)
# (40, 8, 0.3045, 27)    ← a 27-word tail chunk wins
# (39, 8, 0.2213, 167)
# (67, 14, 0.2027, 180)
# (23, 5, 0.2021, 35)
# (65, 13, 0.1904, 31)
```

### 5. FAISS L2 and inner-product indexes agree on unit vectors

Same rows, and the L2 scores are squared distances equal to 2 − 2 × cosine.

```python showLineNumbers
import faiss

X = E.astype("float32")                          # FAISS wants float32, shape (n, d)
l2 = faiss.IndexFlatL2(X.shape[1]); l2.add(X)
ip = faiss.IndexFlatIP(X.shape[1]); ip.add(X)
q = embedder.encode(["What is the main idea of this document?"]).astype("float32")   # shape (1, 384)
D, I = l2.search(q, 5)
S, J = ip.search(q, 5)
print(I[0], D[0].round(4))                       # [51 57 71 53 54] [1.5269 1.5459 1.5589 1.5684 1.5715]
print(J[0], S[0].round(4))                       # [51 57 71 53 54] [0.2366 0.227  0.2206 0.2158 0.2143]
# success-next-line
print(np.allclose(D[0], 2 - 2 * S[0], atol=1e-5))   # ✅ True: squared L2 = 2 − 2·cos
```

### 6. `normalize_L2` works in place, and was a no-op here

It returns `None` and overwrites its argument, so normalize a copy; MiniLM vectors are already unit length.

```python showLineNumbers
Y = E.copy().astype("float32")
# highlight-next-line
faiss.normalize_L2(Y)                            # in place; returns None
print(np.allclose(Y, E))                         # True   (nothing to normalize)
Z = 2 * E.astype("float32")
faiss.normalize_L2(Z)
print(np.allclose(Z, E, atol=1e-6))              # True   (length removed, direction kept)
```

### 7. Your own ids, and saving the index

`IndexIDMap` returns your ids instead of row positions; `write_index` avoids re-embedding.

```python showLineNumbers
ids = np.array([1000 + c["chunk_id"] for c in chunks_data], dtype="int64")
idx = faiss.IndexIDMap(faiss.IndexFlatIP(X.shape[1]))
# highlight-next-line
idx.add_with_ids(X, ids)
print(idx.search(q, 3)[1][0], idx.ntotal)        # [1051 1057 1071] 78
faiss.write_index(ip, "/tmp/bert_chunks.faiss")
ip2 = faiss.read_index("/tmp/bert_chunks.faiss")
print(ip2.ntotal, ip2.search(q, 1)[1][0])        # 78 [51]
```

### 8. Retrieve, read, and blend the scores

The notebook's pipeline in a few lines: top 5, the notebook's reader on each, min-max both scores, $\alpha$ = 0.4.

```python showLineNumbers
def min_max(v, eps=1e-10):
    v = np.asarray(v, dtype=np.float32)
    return (v - v.min()) / (v.max() - v.min() + eps)

def retrieve_then_read(question, k=5, alpha=0.4):
    s = E @ embedder.encode(question)
    top = np.argsort(-s)[:k]
    cands = []
    for rank, i in enumerate(top, start=1):
        r = qa_on_single_chunk(question, chunks_data[i], qa_tokenizer, qa_model, device="cpu",
                               max_answer_tokens=40, min_answer_chars=2, max_answer_chars=400)
        if r["status"] == "valid_answer":
            cands.append({"rank": rank, "chunk": int(i), "cos": float(s[i]), "span": r["span_score"], "answer": r["answer"]})
    nr, ns = min_max([c["cos"] for c in cands]), min_max([c["span"] for c in cands])
    for c, a, b in zip(cands, nr, ns):
        # highlight-next-line
        c["final"] = round(float(alpha * a + (1 - alpha) * b), 4)
    return max(cands, key=lambda c: c["final"]), cands

best, cands = retrieve_then_read("Which corpus is used for pre-training besides English Wikipedia?")
for c in cands: print(c["rank"], c["chunk"], round(c["cos"], 4), round(c["span"], 2), c["final"], c["answer"])
# 1 1 0.5174 -4.65 0.4646 Introduction Language
# 2 18 0.475 10.59 0.4647 NotNext
# 3 47 0.4727 -8.27 0.1126 EMNLP. Association for Computational Linguistics
# 4 19 0.4657 25.28 0.6677 BooksCorpus
# 5 67 0.4551 18.78 0.4836 BooksCorpus
print("best:", best["answer"], "from retrieval rank", best["rank"])   # best: BooksCorpus from retrieval rank 4
```

### What was wrong in the notebook, and the fix

| Notebook code or claim | Problem | Fix |
|---|---|---|
| "Version 2: normalize, then `IndexFlatIP`" as the improvement | MiniLM already outputs unit vectors; normalizing changed nothing and the L2 index already ranked by cosine (identical rows) | know your model's output; normalize only when needed; use IP so scores read as cosines (blocks 1, 5, 6) |
| `IndexFlatL2` described as Euclidean distance | it returns **squared** L2 | $D = 2 - 2\cos$ for unit vectors (block 5) |
| "FAISS is faster and more scalable" with flat indexes | flat search is the same exact scan as NumPy (0.0025 vs 0.006 ms for 78 vectors; 29 vs 27 ms for 200,000); question embedding dominates | use IVF/HNSW for scale, check recall (Chapter 8) |
| 180-word chunks for MiniLM | 48 of 78 exceed the 256-token limit; 10.3% of tokens never embedded | chunk to the embedding model's limit (block 3) |
| Chunker without a minimum length | 6 tail chunks under 60 words; a 27-word chunk wins "limitations" | drop or merge short tails |
| Min-max over the top $k$ before blending | stretches a 0.022 cosine gap to 0 vs 1; single candidate → all zeros | stable scales, tuned weights (Chapter 7) |
| `min_answer_chars=10` in the retrieval pipelines, 3 in brute force | an unfair comparison, and short answers ("15%", "gelu", "110M") can only be returned padded with extra words ("15% of tokens") | one setting for all pipelines |
| Python loop over every chunk for cosine | slow beyond a few thousand chunks | `E @ q` + `argpartition` (block 4) |
| Dehyphenation regex | merges genuine compounds ("left-toright") | dictionary-aware dehyphenation |
| No retrieval evaluation | can't tell retrieval failures from reader failures | recall@k on labelled questions (Chapter 9) |

## 🚨 Interview Gotchas & Edge Cases

:::warning[The Most Common Question]
**"How does semantic search work in a RAG or document-QA system, and how do you know it's working?"** Chunk the documents; embed every chunk once with a sentence-embedding (bi-encoder) model into unit vectors; store them in a vector index (FAISS flat for small corpora, IVF/HNSW at scale) with a separate id-to-metadata map. At query time, embed the question with the same model, find the top-$k$ chunks by cosine (inner product on unit vectors), and pass them to a reader or LLM with citations. To know it works, measure **recall@k** on labelled questions before anything else: on the BERT paper it was 0.67 at $k$ = 5, and retrieval misses explained almost all wrong answers.
:::

- **Use the same embedding model** for chunks and queries; changing models means re-embedding the corpus.
- **Unit length decides the metric**: cosine, inner product and squared L2 give the same ordering only for normalized vectors.
- **`IndexFlatL2` returns squared distances**, smaller is better; `IndexFlatIP` returns similarities, larger is better.
- **FAISS needs float32 `(n, d)` arrays**; `normalize_L2` is in place and returns `None`.
- **FAISS returns row positions**: keep `ntotal == len(chunks)` or use `IndexIDMap`.
- **Flat indexes are exact brute force**; speed-ups need ANN indexes and a recall check.
- **Embedding models truncate** (256 word pieces for MiniLM): long chunks lose their ends silently.
- **Vague questions retrieve boilerplate** (reference lists, captions); short fragments score high.
- **Top-1 retrieval is rarely enough**: the answer chunk was rank 4 for the corpus question, rank 5 for "What full form of BERT?".
- **Min-max per query destroys absolute scale**; tune score fusion on labelled data.
- **Retrieval can lower accuracy** on small corpora (8 → 6 of 12 here) while cutting cost 16×; decide with measurements.
- **Bi-encoder scores aren't probabilities**: a cosine of 0.24 is "weak" only relative to this model and corpus.

## ⚡ Essential Interview Recall

A one-screen revision list. Every point is explained in the chapters above.

**Architecture**
- Brute force: $N \cdot T_{reader}$; retrieve-then-read: $T_{embed} + T_{search} + k \cdot T_{reader}$. Measured 18.70 s vs 1.16 s per question (CPU, $k$ = 5).
- Embedding model (text → vector), index (vectors → nearest rows), retriever (question → top-$k$ chunks), reader (question + chunk → span).
- Bi-encoder for retrieval (precomputable), cross-encoder for reading/reranking (accurate, per pair).

**Embeddings and similarity**
- `all-MiniLM-L6-v2`: 6 layers, 384-d, 22,713,216 params, mean pooling + L2 normalize, truncates at 256 word pieces; contrastive training on 1B+ pairs.
- "disadvantages" → weaknesses 0.582, limitations 0.425, cat 0.006.
- Unit vectors: $\cos = q \cdot v$, $\lVert q - v \rVert^2 = 2 - 2\cos$ (1.5268836 = 2 − 2 × 0.2365582).
- Top-$k$ = `E @ q` + `argpartition`.

**FAISS**
- `IndexFlatL2` (squared distance), `IndexFlatIP` (inner product): exact, no training; float32 `(n, d)`; `normalize_L2` in place; row ids → metadata; `IndexIDMap`, `write_index`.
- Flat ≈ NumPy (27 vs 29 ms at 200k); HNSW 0.17 ms at recall 0.96; IVF 0.3 ms at nprobe 8.

**Scoring and evaluation**
- Notebook blend: min-max both, $f = \alpha \cdot ret + (1-\alpha) \cdot read$, $\alpha$ = 0.4; min-max stretches a 0.022 gap to 1.0.
- 12 labelled questions: recall@1/3/5/10 = 0.25/0.42/0.67/0.83; brute force 8, top-5 blend 6, retrieval-only 3.
- Debug order: recall@k first (retrieval failure?), then the reader on the right chunk.

## 📐 Formula Sheet

| Concept | Formula | In plain words |
|---|---|---|
| Brute-force cost | $T_{brute} = \data{N} \cdot T_{reader}$ | one reader call per chunk |
| Retrieve-then-read cost | $T_{rr} = T_{embed} + T_{search} + \hyper{k} \cdot T_{reader}$ | embed, search, read $k$ |
| Mean pooling | $\pred{u} = \sum_t \data{m_t} \pred{h_t} / \sum_t \data{m_t}$ | average the real tokens' vectors |
| Normalize | $\pred{v} = \pred{u} / \lVert \pred{u} \rVert_2$ | scale to length 1 |
| Cosine on unit vectors | $\cos\theta = \pred{q} \cdot \pred{v}$ | the dot product |
| Squared L2 on unit vectors | $\lVert \pred{q} - \pred{v} \rVert_2^2 = 2 - 2\cos\theta$ | same ranking as cosine |
| Retrieval scores | $\pred{s} = \data{E} \pred{q}$ | one matrix product |
| Top-$k$ | $R_{\hyper{k}}(q) = \operatorname{top\text{-}k}_i\, \pred{s_i}$ | keep the $k$ best chunks |
| Min-max | $\operatorname{mm}(x_j) = (x_j - \min x)/(\max x - \min x + \epsilon)$ | rescale the candidates to 0–1 |
| Combined score | $\pred{f_j} = \hyper{\alpha}\operatorname{mm}(\pred{s^{ret}_j}) + (1-\hyper{\alpha})\operatorname{mm}(\pred{s^{read}_j})$ | blend retrieval and reader |
| Recall@$k$ | $\frac{1}{\lvert \data{Q} \rvert}\sum_q \mathbb{1}[R_{\hyper{k}}(q) \cap \data{G}(q) \neq \varnothing]$ | was the answer chunk retrieved? |
| Reader-call reduction | $\data{N} / \hyper{k}$ | 200× for 1,000 chunks at $k$ = 5 |
| Vector memory | $\data{N} \cdot d \cdot 4$ bytes | 1,536 bytes per 384-d float32 vector |

## Concept-Check Answers (from the notebook)

### Why brute-force QnA is not scalable

<details>
<summary><strong>Q1. Why does brute-force QnA become slow for large documents?</strong></summary>

The reader must run on every chunk for every question, so time grows linearly with the number of chunks: about 0.27 s per chunk on CPU here, so roughly 45 minutes per question at 10,000 chunks.
</details>

<details>
<summary><strong>Q2. In brute-force QnA, how many chunks are passed to the reader?</strong></summary>

All of them: $N$ reader calls per question (78 for the BERT paper).
</details>

<details>
<summary><strong>Q3. Why can irrelevant chunks still produce misleading answers?</strong></summary>

A SQuAD-style extractive reader always extracts its most answer-like span from whatever context it gets, often with a high score, and those scores aren't calibrated across chunks; an irrelevant chunk can therefore out-score the relevant one.
</details>

<details>
<summary><strong>Q4. What is the role of the retriever?</strong></summary>

To find the few chunks most likely to contain the answer, here by embedding the question and ranking chunks by cosine similarity, so the reader only reads those.
</details>

<details>
<summary><strong>Q5. What is the role of the reader?</strong></summary>

To extract the exact answer span from each retrieved chunk (start and end positions), with a span score used for ranking.
</details>

<details>
<summary><strong>Q6. Why is the Retriever → Reader architecture better?</strong></summary>

It applies the expensive reader to $k$ chunks instead of $N$ (1.16 s vs 18.70 s per question here) and removes most irrelevant chunks before they can produce misleading answers. Its quality is bounded by the retriever's recall, which must be measured.
</details>

### Text embeddings and semantic meaning

<details>
<summary><strong>Q1. What is a text embedding?</strong></summary>

A fixed-length numerical vector representing a text's meaning; `all-MiniLM-L6-v2` produces 384 numbers of unit length for any sentence or short paragraph.
</details>

<details>
<summary><strong>Q2. Why are embeddings useful for document search?</strong></summary>

They let us compare texts by meaning instead of exact words: chunks whose vectors point in a similar direction to the question's are likely relevant, even with different wording.
</details>

<details>
<summary><strong>Q3. Is an embedding model the same as a QnA reader model?</strong></summary>

No. The embedding model maps one text to a vector (for search); the reader takes a question and a context together and returns an answer span.
</details>

<details>
<summary><strong>Q4. What does it mean when two embeddings are close to each other?</strong></summary>

The texts are probably semantically related (high cosine similarity), as judged by what the model learned in training; it is a relevance signal, not proof.
</details>

<details>
<summary><strong>Q5. Why can embeddings work better than exact keyword matching?</strong></summary>

They capture related meanings: "disadvantages" matches "weaknesses" (0.582) and "limitations" (0.425) without sharing a word. (Keywords still win on exact identifiers and numbers, which is why hybrid search exists.)
</details>

### Cosine similarity for comparing embeddings

<details>
<summary><strong>Q1. What does cosine similarity measure?</strong></summary>

The cosine of the angle between two vectors: how similarly they point, ignoring their lengths (1 same direction, 0 perpendicular, −1 opposite).
</details>

<details>
<summary><strong>Q2. Why do we use cosine similarity with embeddings?</strong></summary>

Embedding models encode meaning in the direction of the vector; cosine compares directions. For unit-length embeddings it is just a dot product, so it is cheap to compute for all chunks at once.
</details>

<details>
<summary><strong>Q3. If two text embeddings have high cosine similarity, what does it usually mean?</strong></summary>

The texts are semantically related, for example a question and a passage that answers it.
</details>

<details>
<summary><strong>Q4. Does cosine similarity extract the final answer?</strong></summary>

No. It only scores relevance; the reader extracts the answer.
</details>

<details>
<summary><strong>Q5. In the Retriever + Reader architecture, cosine similarity belongs to which part?</strong></summary>

The retriever.
</details>

### Manual semantic retriever with top-k cosine similarity

<details>
<summary><strong>Q1. What is the input to the manual semantic retriever?</strong></summary>

A question, the chunk list with metadata, the precomputed chunk embeddings, the embedding model and `top_k`.
</details>

<details>
<summary><strong>Q2. What is the output of the manual semantic retriever?</strong></summary>

A ranked list of the top-$k$ chunks with cosine scores, chunk ids, page numbers, text and previews.
</details>

<details>
<summary><strong>Q3. Why create embeddings for all chunks before asking questions?</strong></summary>

So the document becomes searchable vectors once; each question then needs only one embedding and a fast comparison, instead of re-encoding the document.
</details>

<details>
<summary><strong>Q4. Why is the question also converted into an embedding?</strong></summary>

To compare it with the chunk vectors in the same space; it must use the same model as the chunks.
</details>

<details>
<summary><strong>Q5. Does the retriever extract the final answer?</strong></summary>

No; it returns candidate chunks.
</details>

<details>
<summary><strong>Q6. Why retrieve top-k chunks instead of only the top-1 chunk?</strong></summary>

The most similar chunk often isn't the one with the answer: the corpus question's answer was at rank 4 and the full-form question's at rank 5. Recall@1 was 0.25 on the 12-question test vs 0.67 at $k$ = 5.
</details>

### Combining the manual retriever with the BERT reader

<details>
<summary><strong>Q1. What are the two main stages in the Retriever + Reader pipeline?</strong></summary>

Retrieval (find the top-$k$ relevant chunks) and reading (extract an answer span from each and pick the best).
</details>

<details>
<summary><strong>Q2. Why should we not send every chunk to the reader?</strong></summary>

It is slow (linear in the corpus) and gives irrelevant chunks the chance to produce high-scoring wrong spans.
</details>

<details>
<summary><strong>Q3. In the simple version, how were answers ranked?</strong></summary>

By the reader's span score alone (start logit + end logit).
</details>

<details>
<summary><strong>Q4. In the improved version, which two scores were combined?</strong></summary>

The retrieval cosine and the reader's span score, each min-max normalized, combined as $\alpha \cdot$ retrieval $+ (1-\alpha) \cdot$ reader with $\alpha$ = 0.4.
</details>

<details>
<summary><strong>Q5. Why do we normalize scores before combining them?</strong></summary>

They're on different scales (cosines about 0.2–0.6, span scores about −10 to 25), so without rescaling the span score would dominate. Min-max per query is fragile, though: it forces the best candidate to 1 and the worst to 0 whatever their absolute quality.
</details>

<details>
<summary><strong>Q6. Is this system generative RAG?</strong></summary>

No. It is retrieval-based extractive QnA: the answer is copied from a chunk, not generated.
</details>

<details>
<summary><strong>Q7. What is the main difference between our current system and a generative RAG system?</strong></summary>

The answering component: an extractive reader that copies a span here, an LLM that writes an answer conditioned on the retrieved chunks in RAG. Retrieval is the same.
</details>

### FAISS introduction and first vector index

<details>
<summary><strong>Q1. What is FAISS used for?</strong></summary>

Fast nearest-neighbour search over vectors: storing chunk embeddings and returning the closest ones to a query embedding.
</details>

<details>
<summary><strong>Q2. Is FAISS a QnA model?</strong></summary>

No. It knows nothing about text or answers; it only compares vectors.
</details>

<details>
<summary><strong>Q3. What does FAISS return after search?</strong></summary>

Two arrays of shape `(n_queries, k)`: scores (squared L2 distances or inner products) and the row numbers of the matching vectors (`-1` if fewer than $k$ exist).
</details>

<details>
<summary><strong>Q4. Why map FAISS indices back to chunks_data?</strong></summary>

FAISS returns row positions only; page numbers, chunk ids and text live in our own list, which must stay aligned with the index rows (or use `IndexIDMap` for explicit ids).
</details>

<details>
<summary><strong>Q5. In L2 search, does a smaller or larger distance mean more relevant?</strong></summary>

Smaller (closer).
</details>

<details>
<summary><strong>Q6. In cosine-style FAISS search with normalized vectors and inner product, does a smaller or larger score mean more relevant?</strong></summary>

Larger (higher cosine).
</details>

<details>
<summary><strong>Q7. Why normalize vectors before using IndexFlatIP for cosine-style search?</strong></summary>

Inner product equals cosine only for unit-length vectors; otherwise longer vectors get higher scores regardless of direction. (MiniLM's output is already unit length, so here normalization changed nothing.)
</details>

### Building a clean FAISS semantic retriever

<details>
<summary><strong>Q1. What does faiss_semantic_retrieve() take as input?</strong></summary>

A question, the chunk list, the FAISS index, the embedding model, `top_k` and an optional minimum retrieval score.
</details>

<details>
<summary><strong>Q2. What does faiss_semantic_retrieve() return?</strong></summary>

A ranked list of retrieved chunks with FAISS row index, chunk id, page, text, preview, word count and retrieval score.
</details>

<details>
<summary><strong>Q3. Why normalize the question embedding before searching?</strong></summary>

The index holds normalized chunk vectors; the query must be unit length too for inner product to equal cosine.
</details>

<details>
<summary><strong>Q4. Why is faiss_index.ntotal == len(chunks) an important check?</strong></summary>

Results are row positions used to look up chunk metadata; if the counts differ, a returned row can map to the wrong chunk and the answer gets the wrong source.
</details>

<details>
<summary><strong>Q5. What is the difference between a FAISS index position and a chunk ID?</strong></summary>

The position is where the vector sits inside the FAISS index; the chunk id is our own metadata identifier. They coincide here only because chunks were added in order.
</details>

<details>
<summary><strong>Q6. Does FAISS extract the final answer?</strong></summary>

No; the reader does.
</details>

<details>
<summary><strong>Q7. Why may FAISS not look much faster for a very small PDF?</strong></summary>

Embedding the question (about 8 ms) dominates, and a flat index is an exact scan just like NumPy (0.006 vs 0.0025 ms for 78 vectors). Even at 200,000 vectors a flat index matched NumPy; approximate indexes (IVF, HNSW) are where the speed-up comes from.
</details>

### Combining the FAISS retriever with the BERT reader

<details>
<summary><strong>Q1. What are the two main components of our FAISS-based QnA system?</strong></summary>

The FAISS retriever (with the MiniLM embedding model) and the BERT-style extractive reader.
</details>

<details>
<summary><strong>Q2. What does FAISS do in this pipeline?</strong></summary>

Finds the top-$k$ chunk vectors most similar to the question vector.
</details>

<details>
<summary><strong>Q3. What does the BERT reader do in this pipeline?</strong></summary>

Extracts an answer span from each retrieved chunk and scores it.
</details>

<details>
<summary><strong>Q4. Why is this better than brute-force QnA?</strong></summary>

The reader processes $k$ chunks instead of all of them (5 vs 78), which is much faster and avoids most irrelevant chunks; its accuracy then depends on retrieval recall.
</details>

<details>
<summary><strong>Q5. In the simple version, how did we rank final answers?</strong></summary>

By the reader span score only.
</details>

<details>
<summary><strong>Q6. In the improved version, which two scores were combined?</strong></summary>

The FAISS retrieval score (cosine) and the reader span score, after min-max normalization, with $\alpha$ = 0.4 on retrieval.
</details>

<details>
<summary><strong>Q7. Why is metadata important in FAISS-based QnA?</strong></summary>

FAISS returns only row numbers and scores; metadata gives the page, chunk id, preview and text needed for the reader and for citing the source.
</details>

<details>
<summary><strong>Q8. Is this system a generative RAG system?</strong></summary>

No. It is retrieval-based extractive QnA; generative RAG uses an LLM to write the answer from the retrieved chunks.
</details>
