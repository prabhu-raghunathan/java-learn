---
title: "Extractive Document QnA over a PDF"
sidebar_position: 2
---

import {ChunkPlanner, DocQaRanker} from '@site/src/components/Widgets';

:::tip[TL;DR (30-Second Interview Pitch)]
**Extractive question answering** doesn't write an answer: it **highlights** one. A BERT-style encoder fine-tuned on SQuAD reads `[CLS] question [SEP] context [SEP]`, a single `Linear(768, 2)` head gives every token a **start logit** and an **end logit**, and the answer is the context slice between the best valid start and end. With `AutoModelForQuestionAnswering` and `distilbert-base-cased-distilled-squad` (65,192,450 parameters), notebook 20 turns that into a **document QnA system** over the 16-page BERT paper: extract text page by page, cut it into **78 overlapping 180-word chunks** (the model sees at most 512 tokens), run the reader on **every** chunk, rank the candidates by span score, refuse to answer below a threshold, and return the page and chunk as evidence. It works for "What does BERT stand for?" and fails instructively elsewhere: raw span scores aren't comparable across chunks, so a wrong span ("evaluation server", 20.049) nearly beats the right one (20.123), and "Which city was BERT invented in?" is answered "Transformers" with 0.9964 confidence. Those failures are why retrieval (RAG) comes next.
:::

<FormulaKey />

:::info[Where this page sits]
**Before:** [Hugging Face Pipelines and What They Hide](./01-hugging-face-pipelines.md) opened `pipeline()` into tokenizer → model → postprocessing for sentiment.
**This page:** notebook 20, *Bert_QnA_Complete*: the same Auto classes with a question-answering head, then a document QnA project over the BERT paper PDF. The span-scoring maths itself was derived on the BERT page ([Chapter 10, Token-level Heads](../06-attention-and-transformers/05-bert-pretraining-and-fine-tuning.md#10-token-level-heads-squad-spans-unanswerable-questions-and-ner)), so here it is only recapped and used.
**Next:** notebook 21, retriever + reader: embed the chunks, retrieve the top few with a vector index, and read only those.
:::

## 🧭 Start Here

So far every Hugging Face model we've used answered one fixed question: is this sentence positive or negative? The obvious next wish is to ask a document anything. There are two ways a model can answer a question about a text. A **generative** model writes a new sentence (that's what chat LLMs do, and notebook 22 builds one). An **extractive** model is humbler and safer: it can only point at a stretch of the text you gave it and say "the answer is from here to here". That is exactly what BERT was fine-tuned to do on SQuAD, and it's what this page uses.

The page is built like the notebook. First we load a QnA model with `AutoTokenizer` + `AutoModelForQuestionAnswering` and answer one question about a three-sentence paragraph by hand: tokenize the pair, get start and end logits, pick the best span, decode it (Chapters 1–4). Then we ask what the model's "confidence" means and what it does when the answer isn't there (Chapters 5–6). Then the real project: read a PDF page by page, discover the context-length wall, cut the text into overlapping chunks, ask every chunk, rank, filter, threshold and cite the source (Chapters 7–11). Finally we see how the same records become fine-tuning data (Chapter 12) and why brute force leads to RAG (Chapter 13).

Every output on this page comes from re-running the notebook's code on CPU (transformers 5.22.2, PyTorch 2.15, PyMuPDF) against the same BERT paper PDF the lecture uploaded. Every number matches the notebook's Colab run to all printed digits. Where the notebook's code has a flaw, the implementation section says what it is and shows the fix, run for real.

| New concept | One-line meaning |
|---|---|
| **Extractive QnA** | Answer = a span copied from the given context, chosen by start and end positions |
| **Generative QnA** | Answer = new text written by a decoder (an LLM), possibly not in the context |
| **`AutoModelForQuestionAnswering`** | Encoder + a `Linear(H, 2)` head that outputs `start_logits` and `end_logits` per token |
| **SQuAD** | Stanford Question Answering Dataset: question, Wikipedia paragraph, answer span |
| **Span score** | $s_i + e_j$: start logit of token $i$ plus end logit of token $j$ |
| **`sequence_ids`** | Per token: 0 = question, 1 = context, `None` = special token |
| **`offset_mapping`** | Per token: its (start, end) character positions in the original string |
| **Chunking** | Cutting long text into pieces that fit the model's input limit |
| **Overlap / stride** | Words (or tokens) repeated between neighbouring chunks so no answer is cut in half |
| **Brute-force document QnA** | Run the reader on every chunk and rank the answers |
| **Threshold** | Minimum score below which the system says "No reliable answer found" |
| **Source tracking** | Returning page number, chunk id and context preview with every answer |
| **`answer_start`** | Character index where the answer begins: the label format for QnA fine-tuning |

![Brute-force document QnA: once per document, extract and chunk; for every question, read all 78 chunks, rank, threshold, cite](/img/deep-learning/diagrams/docqa-pipeline.svg)

*How to read it:* the top row runs once per PDF and fills a chunk store; the bottom row runs for every question and touches every chunk, which is the cost this page ends up wanting to remove.

:::analogy[An open-book exam where you may only use a highlighter]
Extractive QnA is an open-book exam with one rule: you can't write, you can only highlight. If the book contains the answer, a good student highlights exactly the right words. If it doesn't, the student still highlights *something*, because the rules don't allow a blank. Everything this page does (chunking, ranking, thresholds, citations) is about handing the student the right page and noticing when the highlight is a guess.
:::

## 1. Extractive vs Generative Question Answering

In **extractive question answering** the input is a question plus a **context** passage, and the output is a span of that passage. Asked "What type of model is BERT?" with the context "BERT is an encoder-only Transformer model. It is widely used for…", the model returns `encoder-only Transformer`: characters 12 to 36 of the context, nothing more. It can't rephrase, combine two sentences or answer from general knowledge. That sounds like a weakness, and for open-ended explanations it is, but it buys two properties that matter in production: the answer is always **verifiable** (it's literally in the source) and the model **can't invent facts** that aren't in the text.

A **generative** QnA model (T5, GPT-style LLMs) has a decoder that writes the answer token by token. It can say "BERT stands for Bidirectional Encoder Representations from Transformers; it is an encoder-based model designed for deep bidirectional understanding", which is friendlier, but nothing forces those words to come from the context, so it can **hallucinate**. That's why later notebooks pair generators with retrieved context.

Architecturally, the change from the sentiment model of the previous page is only the **task head**. The sentiment model put a classifier on the [CLS] vector and predicted a label for the whole sequence. The QnA model keeps every token's output vector and predicts two numbers *per token*: how likely this token is to start the answer and how likely it is to end it. The backbone, tokenizer and pre-training are the same family; the head decides the task.

| | Sentiment (previous page) | Extractive QnA (this page) | Generative QnA (notebook 22) |
|---|---|---|---|
| Input | one sentence | question + context pair | question (+ retrieved context) |
| Head | classifier on [CLS] → `[B, 2]` | `Linear(H, 2)` on every token → `[B, T]` twice | decoder + vocabulary softmax |
| Output | label + score | start, end → a span of the context | new text |
| Can invent facts? | n/a | no, only copies | yes (hallucination risk) |
| Answer when it's not in the text? | n/a | still picks a span | may make one up |

:::tip[🎯 Interview angle]
**"What's the difference between extractive and generative QnA?"** Extractive QnA predicts a start and an end position in the given context and returns that span, so the answer is always verbatim from the source and easy to verify, but it can't synthesize or rephrase. Generative QnA writes a new answer with a decoder; it is more flexible but can hallucinate, so it must be grounded in retrieved context. Architecturally, extractive QnA is an encoder with a per-token start/end head.
:::

## 2. The Model: `distilbert-base-cased-distilled-squad`

The notebook loads `distilbert-base-cased-distilled-squad` with `AutoTokenizer` and `AutoModelForQuestionAnswering`. It has to: in the lecture's Colab the shortcut `pipeline("question-answering")` raised an unknown-task error, and that wasn't a fluke. In transformers 5.22.2 the pipeline registry has no `question-answering` task at all (it still lists `document-question-answering` and `table-question-answering`, which are different tasks), so loading the model classes directly is the only route. It's also the better one for learning and for fine-tuning later, as the [previous page](./01-hugging-face-pipelines.md#5-opening-the-pipeline-three-stages) argued for sentiment.

The checkpoint's name says what it is. **DistilBERT**: a 6-layer, 768-wide, 12-head student of BERT-base (introduced on the previous page). **base-cased**: the tokenizer keeps capital letters (vocabulary 28,996 WordPieces), which helps with names and acronyms in answers. **distilled-squad**: it was fine-tuned on SQuAD v1.1 with knowledge distillation from a larger BERT teacher. Its config also says `sinusoidal_pos_embds: true`, the fixed sine/cosine positions from the original Transformer, although the 512 × 768 position table is still stored and counted as a parameter.

`AutoModelForQuestionAnswering` resolves to `DistilBertForQuestionAnswering`, and printing it shows the whole task head: `qa_outputs = Linear(in_features=768, out_features=2, bias=True)`. Applied to every token's final vector $T_i$, its first output is the start logit and its second the end logit. Those two rows of weights are exactly the start vector $S$ and end vector $E$ of the BERT paper, plus a bias each, so the head has only $2 \times 768 + 2 = 1{,}538$ parameters on top of the encoder's 65,190,912.

![DistilBertForQuestionAnswering: one Linear(768, 2) turns every token vector into a start and an end score](/img/deep-learning/diagrams/docqa-qa-head.svg)

*How to read it:* every token, question and context alike, gets a start and an end logit; only spans whose both ends fall in the context (green) are legal answers.

| Part | Parameters | How it's made up |
|---|---|---|
| Word embeddings | 22,268,928 | 28,996 × 768 |
| Position embeddings | 393,216 | 512 × 768 (filled with sinusoids, still trainable in this checkpoint) |
| Embedding LayerNorm | 1,536 | 2 × 768 |
| 6 Transformer blocks | 42,527,232 | 6 × 7,087,872 (attention 2,362,368 + FFN 4,722,432 + 2 LayerNorms 3,072) |
| `qa_outputs` | 1,538 | 768 × 2 + 2 |
| **Total** | **65,192,450** | all trainable |

:::danger[Interview Trap: "the QnA model has a special answer-generating layer"]
It doesn't. The entire task head is one linear layer from 768 to 2: a start score and an end score per token. Everything that makes it good at QnA lives in the fine-tuned encoder weights; the head just reads them out.
:::

:::tip[🎯 Interview angle]
**"What does `AutoModelForQuestionAnswering` add on top of the encoder?"** A single `Linear(hidden, 2)` applied to every token's final hidden state, producing `start_logits` and `end_logits` of shape `[batch, seq_len]`. For DistilBERT that's 1,538 parameters on a 65.19M-parameter encoder; the two weight rows are BERT's start and end vectors $S$ and $E$.
:::

## 3. Tokenizing a Question–Context Pair

The tokenizer packs the pair exactly the way BERT was trained on SQuAD: `[CLS] question [SEP] context [SEP]`. For "What type of model is BERT?" and the three-sentence context, that's **48 tokens**, so `input_ids` and `attention_mask` both have shape `[1, 48]`. The tokenizer also returns `token_type_ids` (0 for the question segment, 1 for the context) because it is a `BertTokenizer`, but DistilBERT has no segment embeddings and ignores them, as we saw on the [previous page](./01-hugging-face-pipelines.md#6-the-tokenizer-input_ids-attention_mask-and-special-tokens).

Because this tokenizer is **cased**, the acronym "BERT" isn't in its vocabulary as one piece: it becomes `B`, `##ER`, `##T`. "encoder" becomes `en`, `##code`, `##r`. That matters for decoding later, because a span is a range of these pieces, not of words.

Two tokenizer outputs that the notebook doesn't use are what make the rest of this page correct. **`sequence_ids()`** tells you, for every token position, which segment it belongs to: `None` for [CLS] and [SEP], 0 for question tokens, 1 for context tokens. **`offset_mapping`** (ask for it with `return_offsets_mapping=True`) gives each token's `(start, end)` character positions inside its own original string: for the context's `B ##ER ##T is` it's `(0, 1), (1, 3), (3, 4), (5, 7)`. With these two, you can restrict answers to the context and cut the answer out of the original text instead of re-assembling it from tokens.

| Position | 0 | 1–9 | 10 | 11–46 | 47 |
|---|---|---|---|---|---|
| Token | [CLS] | `What type of model is B ##ER ##T ?` | [SEP] | `B ##ER ##T is an en ##code ##r - only …` | [SEP] |
| `sequence_ids` | None | 0 | None | 1 | None |
| Can be part of the answer? | no | no | no | yes | no |

:::tip[🎯 Interview angle]
**"How do you know which tokens belong to the context?"** Call `encoding.sequence_ids(i)`: it returns `None` for special tokens, 0 for the first text (the question) and 1 for the second (the context). Combined with `return_offsets_mapping=True`, which gives every token's character span, it lets you mask non-context positions before choosing a span and map the chosen tokens back to exact characters of the source.
:::

## 4. From Start and End Logits to an Answer Span

One forward pass gives `start_logits` and `end_logits`, each of shape `[1, 48]`: one score per token position. The notebook first takes the naive route, `argmax` of each independently. Here it works: start index 16 (`en`) and end index 22 (`##former`), decoding to "encoder - only Transformer". But the two argmaxes are chosen separately, so nothing prevents the end from landing before the start, or 300 tokens after it, or on the question.

The fix, derived on the BERT page ([Chapter 10](../06-attention-and-transformers/05-bert-pretraining-and-fine-tuning.md#10-token-level-heads-squad-spans-unanswerable-questions-and-ner)), is to score **pairs**: every candidate span $(i, j)$ gets $s_i + e_j$, and we take the best pair that is **valid**. The notebook's `select_best_answer_span` enforces two of the three validity rules: $j \ge i$ and a maximum answer length (40 tokens). It loops over all $O(n \cdot L)$ pairs in Python, calling `.item()` for each, which is slow but correct for those two rules. The rule it skips is the third: **both ends must be context tokens**. In this example it doesn't matter (the best span is 16–22, inside the context), but on PDF chunks the unrestricted search sometimes lands on [CLS] or [SEP], which decode to an empty string; that's the real reason the notebook later needs an "answer quality filter".

<Formula name="Best valid span (recap)" tag="Span selection" words="Every pair of a start token i and an end token j gets the sum of their logits; only pairs inside the context, in order and at most L tokens long, are allowed; the best allowed pair is the answer." remember="Best in-context pair, not two argmaxes.">

$$
\begin{aligned}
\pred{\text{score}}(i, j) &= \pred{s_i} + \pred{e_j} \\
\mathcal{V} &= \{(i, j) : i, j \in \text{context},\; i \le j < i + \hyper{L}\} \\
(\hat\imath, \hat\jmath) &= \arg\max_{(i, j) \in \mathcal{V}} \pred{\text{score}}(i, j)
\end{aligned}
$$

- $\pred{s_i}, \pred{e_j}$: start logit of token $i$ and end logit of token $j$ (the two outputs of `qa_outputs`).
- $\mathcal{V}$: the valid spans: both ends in the context (`sequence_ids == 1`), ordered, at most $\hyper{L}$ tokens.
- $\hyper{L}$: maximum answer length in tokens (the notebook uses 40; Hugging Face's examples use 30).

**Example:** "What type of model is BERT?": best span 16–22 with $\pred{s_{16}} + \pred{e_{22}}$ = 22.8177, the same pair the two independent argmaxes found.

</Formula>

Decoding is the last trap. `tokenizer.decode(input_ids[16:23])` re-joins WordPieces with spaces around punctuation, giving "encoder **-** only Transformer". The original text says "encoder-only Transformer". Slicing the original string with the offsets, `context[offsets[16][0] : offsets[22][1]]`, returns it exactly; the same fix turns the notebook's "Ming - Wei Chang" back into "Ming-Wei Chang" and "at - tention" into "at- tention" (the PDF's own line-break hyphen). For a system that cites sources, returning the source's exact characters is the right behaviour.

:::analogy[Two highlighters, but only on the page you were given]
The start and end logits are the green and pink highlighters from the BERT page. What this chapter adds is the rule of the exam hall: you may only highlight the *passage*, not the question sheet or the cover page, and the pink mark must come after the green one. Then, instead of retyping what you highlighted (and getting the spacing wrong), you photocopy exactly that stretch of the page.
:::

:::tip[🎯 Interview angle]
**"Why not take argmax of the start and end logits separately?"** Because they're independent: the end can come before the start, the span can be absurdly long, or either end can sit in the question or on a special token. Score every pair with $s_i + e_j$, mask out pairs that aren't inside the context, ordered and within a maximum length, and take the best remaining pair; then map it back to characters with `offset_mapping` rather than `decode`, so the answer is the exact source text.
:::

## 5. Confidence Is Not a Probability of Being Right

The notebook's QnA function also reports a **confidence score**: the softmax probability of the chosen start token plus that of the chosen end token, divided by two. For "What type of model is BERT?" that's 0.5896. It's a reasonable first instinct, but it isn't the probability of anything you'd want. Both softmaxes run over **all 48 positions**, including the question tokens, [CLS] and [SEP]; averaging two probabilities doesn't give the probability of the pair; and a start of 0.9 with an end of 0.1 scores the same 0.5 as a solid 0.5 and 0.5.

There are two cleaner choices. Treating start and end as independent, the probability of the span is the **product** $P^{start}_i \cdot P^{end}_j$ (0.3372 here). Or restrict to the spans you'd actually allow and softmax the span scores over that set: the **span probability**, the share of probability mass the chosen span gets among all valid spans (0.3382 here). Either way the honest number is about a third, not 0.59. The model is splitting its belief between "encoder-only Transformer" and near-identical spans such as "encoder-only Transformer model", which is fine: they're all right.

<Formula name="Notebook confidence vs span probability" tag="Scoring a span" words="The notebook averages two separate softmax probabilities over the whole input. A span probability normalizes the span scores over the valid context spans only, so the numbers sum to one over the answers the system could actually return." remember="Average of two is not a probability.">

$$
\begin{aligned}
\text{conf}_{nb} &= \tfrac{1}{2}\big(\operatorname{softmax}(\pred{s})_{\hat\imath} + \operatorname{softmax}(\pred{e})_{\hat\jmath}\big) \\
\pred{P}(\hat\imath, \hat\jmath) &= \frac{e^{\pred{s_{\hat\imath}} + \pred{e_{\hat\jmath}}}}{\sum_{(i, j) \in \mathcal{V}} e^{\pred{s_i} + \pred{e_j}}}
\end{aligned}
$$

- $\text{conf}_{nb}$: the notebook's score; each softmax runs over every position of the input.
- $\pred{P}(\hat\imath, \hat\jmath)$: probability of the chosen span among the valid spans $\mathcal{V}$ of Chapter 4.

**Example:** "What type of model is BERT?": $\text{conf}_{nb}$ = 0.5896; start × end = 0.3372; span probability = 0.3382.

</Formula>

The deeper problem is that **no** score from one forward pass says whether the answer is *correct*. A SQuAD-trained reader is trained to put its mass on the most answer-like span in front of it. Show it a chunk where nothing answers the question and it still concentrates its mass somewhere: over the BERT paper, "Which city was BERT invented in?" gets the answer "Transformers" with a notebook confidence of **0.9964**. This is the QnA version of the previous page's lesson that a near-certain sentiment score is not correctness.

:::danger[Interview Trap: "the softmax confidence tells you the answer is right"]
It tells you how peaked the model's start and end distributions are over the text it was shown, nothing more. "Which city was BERT invented in?" → "Transformers" at 0.9964. Calibrate on labelled data, compare against a no-answer score with a model trained for it, and always show the source.
:::

:::tip[🎯 Interview angle]
**"How would you put a confidence on an extractive answer?"** Not by averaging the start and end softmaxes over the whole sequence. Use the joint probability of the span among valid context spans (or $P^{start} \cdot P^{end}$), and treat it as a ranking signal, not a probability of correctness: a SQuAD v1.1 model gives 0.9964 to a nonsense answer. For "is the answer even here?" use a SQuAD 2.0-style model that compares against a [CLS] no-answer score, then calibrate the threshold on held-out questions.
:::

## 6. When the Answer Isn't in the Context

The notebook makes this explicit with "Who invented BERT?" against the three-sentence context, which names no inventors. The model answers "encoder - only Transformer model" anyway, with a confidence of 0.3851 and a span score of **−11.9487** (compare 22.8 for a real answer). Then it adds one sentence to the context, "It was introduced by Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova", asks "Who introduced BERT?", and gets all four names with span score 24.2352 and confidence 0.9225. Same model, same question type: **context quality decides answer quality**.

Why does it always answer? Because `distilbert-base-cased-distilled-squad` was fine-tuned on **SQuAD v1.1**, where every question has an answer in its paragraph. It never learned to say "not here". SQuAD 2.0 added unanswerable questions, and BERT handles them by training the no-answer case to point at [CLS] and answering only if the best span beats the [CLS] score $s_{[CLS]} + e_{[CLS]}$ by a tuned margin (the rule is on the [BERT page](../06-attention-and-transformers/05-bert-pretraining-and-fine-tuning.md#10-token-level-heads-squad-spans-unanswerable-questions-and-ner)). You can compute that margin for this v1.1 model too, but its [CLS] score was never trained to mean anything: for "Who invented BERT?" the margin is still +3.83, and over the PDF the margin ranks the wrong answers about as often as the right ones (Chapter 10). For a system that must abstain, pick a SQuAD 2.0 checkpoint.

| Question (3-sentence context) | Answer | Notebook confidence | Span score |
|---|---|---|---|
| What type of model is BERT? | encoder - only Transformer | 0.5896 | 22.8177 |
| Which tasks use BERT? | question answering, text classification, and named entity recognition | 0.6600 | 20.9337 |
| What is BERT used for? | natural language processing tasks | 0.7192 | 19.4224 |
| Which task involves finding entities? | named entity recognition | 0.6261 | 8.6876 |
| Who invented BERT? *(not in the context)* | encoder - only Transformer model | 0.3851 | −11.9487 |
| Who introduced BERT? *(after adding the authors)* | Jacob Devlin, Ming - Wei Chang, Kenton Lee, and Kristina Toutanova | 0.9225 | 24.2352 |

Notice that the span score separated the unanswerable case (−11.9) from the answerable ones far better than the confidence did (0.39 vs 0.59–0.72). Within **one** context, raw span scores are informative. Chapter 10 shows why that stops being true across different chunks.

:::tip[🎯 Interview angle]
**"What does an extractive model do with a question its context can't answer?"** A SQuAD v1.1 model still returns the most answer-like span, because it never saw unanswerable training examples: "Who invented BERT?" against a passage without names returns "encoder-only Transformer model". You need a SQuAD 2.0-style model that scores a no-answer option at [CLS] (answer only if the best span beats it by a tuned threshold), and in a document system you need retrieval so the right context is in front of the reader in the first place.
:::

## 7. Reading a PDF Page by Page

To answer questions about a document we first need its text. The notebook uploads the BERT paper (NAACL 2019, 16 pages including the appendix) and extracts it with **PyMuPDF** (`import fitz`; `page.get_text()` per page), collapsing all whitespace with one regex. It keeps the pages **separate**, as a list of `{page_number, text, num_characters}` records, instead of gluing the document into one string. That one design choice is what later lets the system say "the answer came from page 1".

The extraction looks healthy: 16 pages, **64,105 characters** in total, 4,006.56 per page on average (page 1: 4,061 characters, starting "Proceedings of NAACL-HLT 2019…"). The notebook's health check is a simple floor: under 500 characters for the whole document means the PDF is probably **scanned**, i.e. pictures of pages with no text layer, and needs **OCR** (optical character recognition, a separate pipeline) before any of this works.

"Healthy" still isn't clean. A two-column research paper comes out with artefacts that matter for QnA. Line-break hyphens survive as "repre- sentation" and "pre- train" (334 places in this PDF); the "fi" ligature comes out as the single character "ﬁ" (152 times); tables come out as runs of numbers; and the paper's own figures contain the literal strings "[CLS]" and "[SEP]". That last one is a genuine trap: the tokenizer recognizes those strings as its **special tokens** and gives them ids 101 and 102, so a chunk of text can contain real separator tokens in the middle of the context. In this PDF, 21 of the 78 chunks do (57 special-token ids in total). The fix is one argument, `split_special_tokens=True`, which tokenizes the text "[CLS]" as ordinary characters.

| Extraction artefact | Example from this PDF | Effect | Fix |
|---|---|---|---|
| Line-break hyphen | "repre- sentation", "at- tention" | extra tokens, odd answers ("at- tention") | rejoin `letter- letter` (careful: "ﬁne- tuned" was really "fine-tuned") |
| Ligature | "ﬁne-tuning" | the cased tokenizer splits "ﬁ" differently from "fi" | Unicode NFKC normalization |
| Literal special tokens | "[CLS]", "[SEP]" in figure text | become ids 101/102 inside the context | `split_special_tokens=True` |
| Tables, references | rows of numbers, author lists | plausible-looking wrong spans | skip or chunk by section |
| Scanned pages | almost no characters | nothing to read | OCR first |

:::tip[🎯 Interview angle]
**"Why extract a PDF page by page, and what goes wrong?"** Keeping pages separate preserves metadata (page numbers) that you need for citations, debugging and later fine-tuning data. Extraction of text PDFs is easy but noisy: line-break hyphens, ligatures, headers, tables and references all leak into the text, and scanned PDFs have no text at all and need OCR. A quick health check (characters per page) catches the scanned case early.
:::

## 8. The Context-Length Wall

With text in hand, the notebook tries the obvious thing: paste pages 1–4 (17,149 characters) in as the context. "What does BERT stand for?" comes back right, "Bidirectional Encoder Representations from Transformers", with confidence 0.8434. Then it checks the length. Question plus pages 1–4 is **4,556 tokens**; the model's position table has 512 entries, and the notebook's function truncates at `max_length=384`. So `truncation=True` silently threw most of the text away.

How much? Using the offsets of the last context token that survived truncation: the model saw only the first **1,374 of 17,149 characters**, i.e. **8.0%** of the four pages, roughly the abstract. The question was answered correctly only because the answer happens to sit in the abstract. "What tasks can BERT be fine-tuned for?" got "question answering and language inference" with a low span score of 7.88 from the same 8%. Anything asked about page 3 would have been answered from text that doesn't contain it, and nothing in the output would say so. Hugging Face even prints a warning for the untruncated call ("Token indices sequence length is longer than the specified maximum sequence length for this model (4556 > 512)"), but with `truncation=True` there is no warning at all.

:::analogy[Reading a book through a letterbox]
The model reads through a letterbox exactly 384 (at most 512) tokens wide. Push four pages through and it reads the first strip, then the rest falls on the floor unread. It will still answer every question confidently from that strip. Chunking is sliding the book past the letterbox one strip at a time.
:::

:::danger[Interview Trap: `truncation=True` is not "handling" long inputs]
It's discarding them. For QnA, either chunk the document yourself or let the tokenizer cut overlapping windows with `return_overflowing_tokens=True` and `stride`, run every window, and combine the answers. Always measure how many tokens were dropped.
:::

:::tip[🎯 Interview angle]
**"What happens if the context is longer than the model's max length?"** With `truncation=True` (or `"only_second"`), the tokenizer drops the end of the context without an error; for the BERT paper's pages 1–4, 92% of the characters never reached the model. For QnA you split the context into overlapping windows (your own chunks, or the tokenizer's `return_overflowing_tokens=True` with a `stride`), run each window and pick the best span across windows.
:::

## 9. Cutting the Document into Overlapping Chunks

The notebook's `chunk_pages_by_words` cuts each page's words into windows of `chunk_size = 180` words, and starts every next window `overlap = 40` words before the previous one ended: the **step** is 140 words. Page 1 has 584 words, so its chunks cover words 0–180, 140–320, 280–460 and 420–584 (the last is shorter, 164 words). A final window under `min_chunk_words = 30` would be dropped as too small to be useful. Each chunk keeps `chunk_id`, `page_number`, its word range, its text and lengths. Over the 16 pages this gives **78 chunks**.

<Animation src="docqa/chunk-window" caption="Page 1 of the BERT paper cut into 180-word windows that step by 140. The yellow strips are the 40 words each chunk shares with the previous one; the answer to What does BERT stand for sits inside chunk 0." />

Why overlap at all? Chunk boundaries are blind: they fall every 140 words regardless of sentences. Without overlap, an answer straddling a boundary would be cut in half and no chunk could return it whole. With 40 shared words, any answer up to 40 words long that is cut by one boundary lies entirely inside the neighbour. The price is redundancy: each boundary's 40 words are read twice, and near-duplicate answers can appear from neighbouring chunks.

<Formula name="Number of overlapping chunks" tag="Chunking" words="Windows of C words start every C minus O words; the number of windows is enough steps to reach the end, at least one; a final window shorter than the minimum is dropped." remember="Step is size minus overlap.">

$$
\begin{aligned}
\hyper{\text{step}} &= \hyper{C} - \hyper{O} \\
n_{chunks} &= \max\!\left(1,\; \left\lceil \frac{\data{W} - \hyper{O}}{\hyper{C} - \hyper{O}} \right\rceil\right)
\end{aligned}
$$

- $\data{W}$: words on the page; $\hyper{C}$: chunk size (180); $\hyper{O}$: overlap (40, must be less than $\hyper{C}$); $\hyper{m}$: `min_chunk_words` (30).
- A final window shorter than $\hyper{m}$ words is dropped, so subtract one if that happens.

**Example:** page 1, $\data{W}$ = 584: $\lceil 544 / 140 \rceil$ = 4 chunks. Page 4, 811 words: $\lceil 771 / 140 \rceil$ = 6. Summed over the 16 pages: 78.

</Formula>

The chunk size is chosen in **words** but the model's limit is in **tokens**, so the notebook measures both. On this PDF a word costs **1.74 tokens** on average (1.37 to 2.34 per chunk, higher where there are numbers, citations and broken hyphens), and a question + chunk pair is 71 to 432 tokens, mean 293.83. All 78 fit under 512, and the notebook uses `max_length=500` for chunk QnA. But 8 chunks are longer than 384 tokens, so the two cells that use `max_length=384` (the unsupported-question and failure-analysis cells) silently truncate the ends of those 8 chunks: the notebook mixes the two limits.

<ChunkPlanner />

The tokenizer can also do this job in **token** space: `return_overflowing_tokens=True` with `stride=128` cuts the context into windows that each fit `max_length` and share 128 tokens with the previous window, repeating the question in every window. Pages 1–4 at `max_length=384` become **19 windows**. Word chunks keep human-readable boundaries and page metadata; token windows guarantee the limit exactly. Production systems often chunk by sentences or sections first and use token windows as a safety net.

<Formula name="Tokenizer windows with a stride" tag="return_overflowing_tokens" words="Each window holds max_length minus the question and the three special tokens of context, and starts stride tokens before the previous window ended." remember="Room per window, step = room − stride.">

$$
\begin{aligned}
\hyper{R} &= \hyper{\text{max\_length}} - \data{Q} - 3 \\
n_{windows} &= 1 + \left\lceil \frac{\data{N} - \hyper{R}}{\hyper{R} - \hyper{\text{stride}}} \right\rceil
\end{aligned}
$$

- $\data{Q}$: question tokens (without [CLS]/[SEP]); $\data{N}$: context tokens; $\hyper{R}$: context room per window.

**Example:** "What does BERT stand for?" has $\data{Q}$ = 8 WordPieces, so $\hyper{R}$ = 384 − 8 − 3 = 373; pages 1–4 have $\data{N}$ = 4,545 context tokens: $1 + \lceil 4172 / 245 \rceil$ = 1 + 18 = 19 windows (the last is 146 tokens long).

</Formula>

:::analogy[Overlapping photos for a panorama]
When you shoot a panorama you overlap each photo with the last, so nothing at the seams goes missing and the software can stitch them. Chunks overlap for the same reason: a sentence on the seam appears whole in at least one photo. Overlap too little and seams lose content; overlap too much and you take twice as many photos of the same wall.
:::

:::tip[🎯 Interview angle]
**"How do you choose chunk size and overlap?"** Size from the model's token limit: measure tokens per word on your documents (1.74 here), leave room for the question and special tokens, and stay under max_length (180 words ≈ 313 tokens + question fits 384–512). Overlap so that typical answers or sentences crossing a boundary appear whole in a neighbour (40 words here, about 20%). Too small chunks lose context; too large get truncated or dilute retrieval; more overlap means more chunks and duplicate answers.
:::

## 10. Asking Every Chunk and Ranking the Answers

The first document QnA system is **brute force**: for each question, run the extractive QnA function on all 78 chunks, keep each chunk's best span with its score and metadata, sort the candidates by **span score**, and show the top $k$. For "What does BERT stand for?" the winner is right: "Bidirectional Encoder Representations from Transformers" from chunk 0 on page 1, span score 20.123. Every chunk costs one forward pass, so one question costs 78 passes: about **10 seconds on a CPU** for this single 16-page paper (on a GPU it's much faster, but it still grows with questions × chunks).

<Animation src="docqa/chunk-ranking" caption="The best span score of each of the 78 chunks for What does BERT stand for. The right answer on page 1 wins by only 0.074 over evaluation server on page 6, and 26 chunks clear a threshold of 5." />

The top-5 list shows why one answer is not enough. Rank 2 is "evaluation server" (page 6) at 20.049, just **0.074** behind the right answer. Ranks 3 and 5 are the paper's title from the appendix and references. Then the notebook asks five questions and only two come back right at rank 1:

| Question | Rank-1 answer | Span score | Page | Right? |
|---|---|---|---|---|
| What does BERT stand for? | Bidirectional Encoder Representations from Transformers | 20.1230 | 1 | yes |
| What is BERT designed to pre-train? | Deep Bidirectional Transformers for Language Understanding | 22.8195 | 12 | partly (the title, not "deep bidirectional representations from unlabeled text", which is rank 3 at 21.68) |
| What tasks can BERT be fine-tuned for? | masking strategies | 14.1154 | 16 | no |
| What is masked language modeling? | at - tention | 18.5630 | 6 | no (the right "randomly masks some of the tokens from the input" is rank 3 at 14.33) |
| What dataset is used for question answering? | Multi - granularity hierarchical attention fusion networks | 18.3737 | 12 | no ("SQuAD v1.1" is rank 3 at 17.15) |

The underlying reason is that **span scores are not comparable across chunks**. A logit is meaningful relative to the other logits in the *same* forward pass; there's no normalization that makes 18.6 in chunk 26 mean the same as 18.6 in chunk 3. Each chunk is scored as if it were the only passage in the world, and the SQuAD reader always finds its best-looking span. Chunks full of answer-shaped phrases (titles, table headers, references) can out-score the chunk that actually answers the question. The right answer is often in the top 3, which says the reader is fine and the **ranking** is the weak link.

Better per-chunk scores don't fix this. Ranking by the margin over each chunk's [CLS] score, or by span probability, moves the winners around (by margin, "Stanford Question Answering Dataset" wins the dataset question, but "understanding capability" wins "What does BERT stand for?") without making them reliably right. The fix is upstream: decide which chunks are **relevant** to the question before reading them. That's retrieval, the next notebook.

:::analogy[Asking every librarian at once]
Brute-force QnA is shouting your question across a library with 78 librarians, each holding one page. Every librarian shouts back their best guess with the same confidence, because none of them knows what the others hold. The loudest voice isn't the best answer. A retriever is the reference desk that sends your question only to the three librarians holding relevant pages.
:::

:::tip[🎯 Interview angle]
**"You run a QnA model over every chunk and pick the highest score. What's wrong with that?"** Two things. Cost: one forward pass per chunk per question, so it scales with corpus size (10 s per question for one 16-page PDF on CPU). Quality: start/end logits are only comparable within one forward pass; across chunks they aren't calibrated, so answer-shaped text in irrelevant chunks can win (a wrong span lost by only 0.074 here, and 3 of 5 questions got wrong rank-1 answers while the right one sat in the top 3). Add a retriever to pick relevant chunks first, and optionally a reranker.
:::

## 11. Filters, Thresholds and Source Tracking

Three cheap safeguards turn the ranked list into a usable answer. A **quality filter** (`is_reasonable_answer`) drops candidates that are empty, one character, pure punctuation or a special token. It's needed because the notebook's span search isn't restricted to the context: for the unsupported questions, the raw highest-scoring "answers" are empty strings (the span landed on [CLS]/[SEP] and decoded to nothing, scores 11.99 and 10.08). Restricting the search to context tokens (Chapter 4) removes the cause; the filter stays useful as a guard against punctuation-only spans.

A **threshold** makes the system abstain: if the rank-1 span score is below 5.0, return "No reliable answer found in the document." The notebook's test shows the intended behaviour: "What is the author's favorite programming language?" tops out at 3.1413 ("ESIM + ELMo sys - tem") and is refused, while "What does BERT stand for?" (20.123) is answered. But 5.0 is a hand-picked number on uncalibrated scores, and it fails both ways. "Which city was BERT invented in?" scores 6.3379 and is answered **"Transformers"** (confidence 0.9964); "What tasks can BERT be fine-tuned for?" scores 14.1154 and is answered **"masking strategies"**. Neither question is helped by any threshold that still lets real answers through.

**Source tracking** is the safeguard that always works: return the page number, chunk id and a context preview with every answer, so a human can check it in seconds. The notebook's final call shows why: "Is BERT Encoder or Decoder only model?" returns "**BERT SQuAD**" from page 5, chunk 23 (span score 13.815), and the preview ("…the BERT SQuAD model can be trained in around 30 minutes on a single Cloud TPU…") makes the error obvious. The question itself is a poor fit for extraction too: it asks the system to choose between two options, and the paper never writes "BERT is an encoder-only model".

<DocQaRanker />

| Question | Filtered rank-1 answer | Score | Notebook confidence | Output at threshold 5.0 |
|---|---|---|---|---|
| What does BERT stand for? | Bidirectional Encoder Representations from Transformers | 20.1230 | 0.8709 | answered (right) |
| What is the author's favorite programming language? | ESIM + ELMo sys - tem | 3.1413 | 0.6546 | refused (right) |
| What tasks can BERT be fine-tuned for? | masking strategies | 14.1154 | 0.7470 | answered (wrong) |
| Which city was BERT invented in? | Transformers | 6.3379 | 0.9964 | answered (wrong) |
| Is BERT Encoder or Decoder only model? | BERT SQuAD | 13.8150 | 0.7672 | answered (wrong) |

:::analogy[A receipt with every answer]
A threshold is a cashier who refuses notes that look too worn: useful, but forgeries can look crisp. Source tracking is the receipt stapled to every answer, saying which page and which paragraph it came from. You may still be handed a wrong answer, but you can check it in five seconds instead of trusting it blindly.
:::

:::tip[🎯 Interview angle]
**"How do you stop a document QnA system from returning nonsense?"** Filter degenerate spans (empty, punctuation, special tokens; better, restrict the search to context tokens), abstain below a score threshold calibrated on labelled questions (a hand-picked 5.0 on raw logits lets "Transformers" through for "Which city was BERT invented in?"), use a SQuAD 2.0 model that can predict no-answer, add retrieval so irrelevant chunks are never read, and always return the source (page, chunk, preview) so users can verify.
:::

## 12. From QnA Records to Fine-tuning Data

Everything so far used a model someone else fine-tuned. To adapt it to your own documents (contracts, manuals, policies), you fine-tune on your own labelled examples, and the notebook ends by showing their shape. It's the **SQuAD format**: an `id`, a `question`, a `context`, and `answers` with the answer `text` and its `answer_start`, the **character** index where it begins. For "BERT stands for Bidirectional Encoder Representations from Transformers.", the answer starts at character **16**; the check `context[16 : 16 + len(text)] == text` is the first thing to run on any dataset, because off-by-one starts are the most common labelling bug.

The model, though, is trained on **token** positions, so each record is converted with the offset mapping, exactly the tool from Chapter 3. The start label is the context token whose character span contains character 16 (token 15, `B`, of `B ##id ##ire ##ctional`); the end label is the context token containing the answer's last character, character 70 (token 27, `Transformers`). Passing `start_positions=15` and `end_positions=27` to the model makes it return the loss, the average of the start and end cross-entropies (derived on the BERT page): **0.8377** on this one example, before any training. When an answer falls outside a chunk or window, the convention is to label that window's start and end as position 0 ([CLS]), which is exactly how SQuAD 2.0 models learn the no-answer option.

<Animation src="docqa/offsets-to-labels" caption="The SQuAD record stores answer_start = 16 as a character index. The offset mapping finds the token whose span contains character 16 (token 15) and the one containing the last character (token 27): those are the training labels." />

<Formula name="Character span to token labels" tag="Fine-tuning prep" words="The start label is the context token whose character range contains answer_start; the end label is the context token whose range contains the answer's last character." remember="Characters in, token positions out.">

$$
\begin{aligned}
\data{a_{end}} &= \data{a_{start}} + \lvert \data{\text{answer}} \rvert \\
\data{y_{start}} &= t : \text{off}_t^{(0)} \le \data{a_{start}} < \text{off}_t^{(1)} \\
\data{y_{end}} &= t : \text{off}_t^{(0)} < \data{a_{end}} \le \text{off}_t^{(1)} \\
\loss{L} &= -\tfrac{1}{2}\log \operatorname{softmax}(\pred{s})_{\data{y_{start}}} \\
&\quad\; - \tfrac{1}{2}\log \operatorname{softmax}(\pred{e})_{\data{y_{end}}}
\end{aligned}
$$

- $\data{a_{start}}$: `answer_start` (characters); $\data{a_{end}}$: exclusive character end; $\text{off}_t$: token $t$'s `(start, end)` offsets.
- $\data{y_{start}}, \data{y_{end}}$: the token labels (`start_positions`, `end_positions`); $t$ ranges over context tokens only.

**Example:** $\data{a_{start}}$ = 16, answer length 55, $\data{a_{end}}$ = 71: tokens 15 (`B`, offsets 16–17) and 27 (`Transformers`, offsets 59–71); the model's loss on this record is 0.8377.

</Formula>

:::tip[🎯 Interview angle]
**"What does a QnA fine-tuning example look like, and how is it turned into labels?"** SQuAD format: question, context, answer text and `answer_start` as a character index. Tokenize question + context with `return_offsets_mapping=True`, find the context token containing the start character and the one containing the last answer character, and pass those as `start_positions` / `end_positions`; the loss is the mean of the two cross-entropies. With long contexts, use strided windows and label windows that don't contain the answer as (0, 0), the [CLS] no-answer position.
:::

## 13. From Brute Force to RAG

The notebook closes by naming the two problems its system has, and the next notebooks solve them in order. The first is **cost and ranking**: reading every chunk with the expensive reader is slow and lets irrelevant chunks win. The cure is a **retriever**: turn every chunk once into an embedding vector, store the vectors in an index (FAISS), embed the question, fetch the top few most similar chunks, and run the reader only on those. That's **retrieval-augmented** QnA, notebook 21. The second is **expressiveness**: extraction can only copy spans, so it can't answer "Is BERT encoder-only or decoder-only?" in a sentence of its own. A generator (an LLM) can, if it's fed the retrieved chunks and told to stay grounded in them; that's generative RAG, notebook 22.

| | This page: brute-force extractive | Retriever + reader (notebook 21) | Generative RAG (notebook 22) |
|---|---|---|---|
| Chunks read per question | all 78 | top-k retrieved | top-k retrieved |
| Ranking signal | raw span score | similarity, then span score | similarity |
| Answer | span from one chunk | span from one chunk | generated text grounded in chunks |
| Hallucination risk | none (copies) | none (copies) | present, reduced by grounding |
| Main weakness | cost, uncalibrated ranking | retriever misses | unfaithful generation |

The pieces built here carry straight over: page-wise extraction, overlapping chunks with metadata, the reader function and source tracking. Only the "ask every chunk" step is replaced.

:::tip[🎯 Interview angle]
**"How does this extractive system become a RAG system?"** Keep extraction, chunking and metadata. Precompute an embedding per chunk and index them; at query time embed the question, retrieve the top-k chunks by similarity, and run the extractive reader (or an LLM) only on those, returning the answer with its source. Retrieval cuts cost from all chunks to k and removes most irrelevant chunks that fooled the reader's uncalibrated scores.
:::

## ⚙️ Going Deeper (Document QnA Internals)

**Vectorized span search.** The notebook's double loop calls `.item()` for every pair, $O(n \cdot L)$ Python operations per chunk. The same search is one broadcast: `score = s[:, None] + e[None, :]`, set invalid pairs (outside the context, $j < i$, $j - i \ge L$) to `-inf` with a mask, take `argmax` of the flattened matrix and `divmod` by $n$. For $n$ = 432 that's a 432 × 432 matrix: trivial next to the forward pass. Hugging Face's own postprocessing instead takes the top-20 starts and top-20 ends and checks the 400 pairs, which is cheaper still and almost always finds the same span.

**Why the [CLS] margin doesn't rescue a SQuAD v1.1 model.** In SQuAD 2.0 training, unanswerable windows are labelled (0, 0), so $s_{[CLS]} + e_{[CLS]}$ learns to be high exactly when there is no answer. `distilbert-base-cased-distilled-squad` never saw such labels, so its [CLS] logits are a by-product. Measured over the PDF, ranking by margin moves the correct "Bidirectional Encoder Representations from Transformers" from rank 1 to rank 3 for "What does BERT stand for?" (margin 19.39 vs 23.68 for "understanding capability"), while promoting "Stanford Question Answering Dataset" to rank 1 for the dataset question. The margin is a different uncalibrated number, not a better one.

**Batching chunks.** Reading 78 chunks one by one took about 10 s per question on this CPU. Padding all 78 into one `[78, 432]` batch took 19.7 s in a single run, because every chunk is padded to the longest one and attention cost grows with the square of the length. On a GPU, batching is what you'd do (it's how Hugging Face processes overflowing windows), ideally with chunks sorted by length so padding is minimal.

**Word chunks vs token windows.** The notebook's word chunks are per page, so an answer that crosses a page break can't be found, and every chunk's token count varies (71–432). The tokenizer's `return_overflowing_tokens=True, stride=128` produces windows of exactly `max_length` with `overflow_to_sample_mapping` to tell which original example each window came from, plus offsets for exact answers. Real systems usually chunk on structure first (headings, paragraphs, sentences) to keep units meaningful and use token limits as a hard cap.

**Cleaning trade-offs.** Rejoining "word- word" fixes 334 broken words but also merges real hyphenated compounds broken at a line end ("ﬁne- tuned" → "finetuned"); NFKC turns "ﬁ" into "fi". Together they cut the document from 17,892 to 16,704 tokens (6.6% fewer) and remove odd answers like "at- tention". Cleaning should happen before chunking, and the same normalization must be applied to anything you later compare against (labels, retrieved text).

**Distillation in one line.** DistilBERT is trained to match a larger teacher's soft output distribution (plus the usual hard labels), so it inherits much of the teacher's behaviour with half the layers. The `-distilled-squad` checkpoint applies the same idea during SQuAD fine-tuning. It's a good default reader for CPU demos; larger readers (BERT-large, RoBERTa) score higher on SQuAD but cost more per chunk, which brute force multiplies by the number of chunks.

## 💻 Python Implementation

:::tip[Run it yourself]
Every block was run in order in one Python session (Python 3.13, transformers 5.22.2, PyTorch 2.15 CPU, PyMuPDF), against the BERT paper PDF (`17 Bert_Model.pdf`, 16 pages). Outputs are copied from that run into the trailing comments.
:::

### 1. Load the reader and look at its head

`AutoModelForQuestionAnswering` resolves the checkpoint to `DistilBertForQuestionAnswering`; the head is one `Linear(768, 2)`.

```python showLineNumbers
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

name = "distilbert-base-cased-distilled-squad"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForQuestionAnswering.from_pretrained(name).eval()

n = lambda m: sum(p.numel() for p in m.parameters())
print(type(model).__name__)                      # DistilBertForQuestionAnswering
print(f"{n(model):,}")                           # 65,192,450
# highlight-next-line
print(model.qa_outputs)                          # Linear(in_features=768, out_features=2, bias=True)
print(n(model.distilbert.embeddings.word_embeddings), n(model.distilbert.transformer.layer[0]), n(model.qa_outputs))
                                                 # 22268928 7087872 1538
print(tok.model_max_length, model.config.vocab_size, model.config.n_layers)   # 512 28996 6
```

### 2. Tokenize the pair with segment ids and character offsets

`sequence_ids` marks question (0), context (1) and special tokens (`None`); offsets are character spans in each original string.

```python showLineNumbers
question = "What type of model is BERT?"
context = ("BERT is an encoder-only Transformer model. It is widely used for natural language "
           "processing tasks such as question answering, text classification, and named entity recognition.")
# highlight-next-line
enc = tok(question, context, return_offsets_mapping=True, return_tensors="pt")
offsets = enc.pop("offset_mapping")[0]
seq_ids = enc.sequence_ids(0)
tokens = tok.convert_ids_to_tokens(enc["input_ids"][0])
print(len(tokens))                               # 48
print(tokens[:12])        # ['[CLS]', 'What', 'type', 'of', 'model', 'is', 'B', '##ER', '##T', '?', '[SEP]', 'B']
print(seq_ids[:12])       # [None, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, 1]
print(offsets[11:15].tolist())                   # [[0, 1], [1, 3], [3, 4], [5, 7]]
```

### 3. Best valid span, vectorized, and an exact answer from the offsets

The notebook loops over pairs and never masks the question or special tokens, then rebuilds the answer with `decode`. One masked broadcast and a character slice fix both.

```python showLineNumbers
with torch.no_grad():
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
s, e = out.start_logits[0], out.end_logits[0]
print(s.shape, e.shape)                          # torch.Size([48]) torch.Size([48])
print(s.argmax().item(), e.argmax().item())      # 16 22   (independent argmaxes)

L = 40
idx = torch.arange(len(s))
in_ctx = torch.tensor([sid == 1 for sid in seq_ids])
score = s[:, None] + e[None, :]                  # score[i, j] = s_i + e_j
# success-start
valid = (in_ctx[:, None] & in_ctx[None, :]
         & (idx[None, :] >= idx[:, None]) & (idx[None, :] - idx[:, None] < L))
score = score.masked_fill(~valid, float("-inf"))
# success-end
i, j = divmod(score.argmax().item(), len(s))
print(i, j, round(score[i, j].item(), 4))        # 16 22 22.8177
# error-next-line
print(tok.decode(enc["input_ids"][0][i:j + 1]))  # ❌ encoder - only Transformer   (the notebook's output)
# success-next-line
print(context[offsets[i][0]:offsets[j][1]])      # ✅ encoder-only Transformer      (exact source text)
```

### 4. What the "confidence" numbers mean

The notebook averages two softmaxes taken over all 48 positions; the span probability normalizes over valid context spans only.

```python showLineNumbers
p_start, p_end = s.softmax(-1)[i].item(), e.softmax(-1)[j].item()
# error-next-line
print(round((p_start + p_end) / 2, 4))           # ❌ 0.5896  the notebook's "confidence"
print(round(p_start * p_end, 4))                 # 0.3372  independent start × end
# success-next-line
print(round(score.flatten().softmax(-1)[i * len(s) + j].item(), 4))   # ✅ 0.3382  share among valid spans
null = (s[0] + e[0]).item()
print(round(null, 4), round(score[i, j].item() - null, 4))           # -2.3005 25.1182  ([CLS] score, margin)
```

### 5. Page-wise extraction and the context-length wall

Pages 1–4 are 4,556 tokens; with truncation at 384 the model sees 8% of them.

```python showLineNumbers
import re, fitz                                  # PyMuPDF

def clean_text(t): return re.sub(r"\s+", " ", t).strip()

doc = fitz.open("17 Bert_Model.pdf")
pages = [{"page_number": k + 1, "text": clean_text(p.get_text())} for k, p in enumerate(doc)]
total = sum(len(p["text"]) for p in pages)
print(len(pages), total, round(total / len(pages), 2))   # 16 64105 4006.56
print([len(p["text"]) for p in pages[:5]])               # [4061, 4531, 3701, 4853, 3852]

first4 = clean_text(" ".join(p["text"] for p in pages[:4]))
ids = tok("What does BERT stand for?", first4)["input_ids"]
print(len(first4), len(ids))                             # 17149 4556
# highlight-next-line
enc4 = tok("What does BERT stand for?", first4, truncation="only_second", max_length=384, return_offsets_mapping=True)
seen = max(o[1] for o, sid in zip(enc4["offset_mapping"], enc4.sequence_ids()) if sid == 1)
print(seen, round(100 * seen / len(first4), 1))          # 1374 8.0   (characters seen, % of pages 1-4)
```

### 6. Overlapping word chunks with metadata

The notebook's chunker, compacted: 180-word windows stepping by 140, per page, keeping page and word range.

```python showLineNumbers
def chunk_pages_by_words(pages, chunk_size=180, overlap=40, min_chunk_words=30):
    chunks = []
    for page in pages:
        words, start = page["text"].split(), 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            if end - start >= min_chunk_words:
                chunks.append({"chunk_id": len(chunks), "page_number": page["page_number"],
                               "start_word_index": start, "end_word_index": end,
                               "text": " ".join(words[start:end])})
            if end == len(words):
                break
            # highlight-next-line
            start = end - overlap                # step forward chunk_size - overlap words
    return chunks

chunks = chunk_pages_by_words(pages)
print(len(chunks))                                              # 78
print([(c["start_word_index"], c["end_word_index"]) for c in chunks[:4]])
                                                 # [(0, 180), (140, 320), (280, 460), (420, 584)]
lens = [len(tok("What does BERT stand for?", c["text"])["input_ids"]) for c in chunks]
print(min(lens), round(sum(lens) / len(lens), 2), max(lens))    # 71 293.83 432
```

### 7. Letting the tokenizer cut strided windows instead

Token-space windows that always fit `max_length` and overlap by `stride` tokens.

```python showLineNumbers
win = tok("What does BERT stand for?", first4, truncation="only_second", max_length=384,
          # highlight-next-line
          stride=128, return_overflowing_tokens=True, return_offsets_mapping=True)
print(len(win["input_ids"]), [len(x) for x in win["input_ids"]][-2:])   # 19 [384, 146]
print(win["overflow_to_sample_mapping"][:3])                            # [0, 0, 0]
```

### 8. Brute-force document QnA with a threshold

The fixed reader (context-only spans, exact text) run on every chunk, ranked by span score; the second question shows both the threshold's weakness and the special-token trap.

```python showLineNumbers
def best_span(question, ctx, max_length=500, L=40):
    enc = tok(question, ctx, truncation="only_second", max_length=max_length,
              return_offsets_mapping=True, return_tensors="pt")
    off, sid = enc.pop("offset_mapping")[0], enc.sequence_ids(0)
    with torch.no_grad():
        o = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    s, e = o.start_logits[0], o.end_logits[0]
    k = torch.arange(len(s)); c = torch.tensor([x == 1 for x in sid])
    sc = (s[:, None] + e[None, :]).masked_fill(
        ~(c[:, None] & c[None, :] & (k[None, :] >= k[:, None]) & (k[None, :] - k[:, None] < L)), float("-inf"))
    i, j = divmod(sc.argmax().item(), len(s))
    return ctx[off[i][0]:off[j][1]], round(sc[i, j].item(), 4)

def document_qa(question, chunks, threshold=5.0, top_k=3):
    cands = []
    for ch in chunks:                            # one forward pass per chunk
        answer, span_score = best_span(question, ch["text"])
        cands.append({"answer": answer, "span_score": span_score,
                      "page": ch["page_number"], "chunk": ch["chunk_id"]})
    cands.sort(key=lambda r: r["span_score"], reverse=True)
    best = cands[0]
    status = "answered" if best["span_score"] >= threshold else "below_threshold"
    return status, cands[:top_k]

for q in ["What does BERT stand for?", "Which city was BERT invented in?"]:
    status, top = document_qa(q, chunks)
    print(q, status)
    for r in top: print("   ", r)
# What does BERT stand for? answered
#     {'answer': 'Bidirectional Encoder Representations from Transformers', 'span_score': 20.123, 'page': 1, 'chunk': 0}
#     {'answer': 'evaluation server', 'span_score': 20.049, 'page': 6, 'chunk': 27}
#     {'answer': 'Pre-training of Deep Bidirectional Transformers for Language Understanding', 'span_score': 19.4828, 'page': 12, 'chunk': 58}
# Which city was BERT invented in? answered
#     {'answer': '[CLS]', 'span_score': 8.5253, 'page': 5, 'chunk': 22}
#     {'answer': '[CLS]', 'span_score': 8.3018, 'page': 7, 'chunk': 33}
#     {'answer': '[CLS]', 'span_score': 8.2111, 'page': 5, 'chunk': 21}
```

The city question is "answered" with the literal text `[CLS]` from the paper's figure captions: the next block shows why.

### 9. Literal special tokens inside the document

`[CLS]` and `[SEP]` written in the PDF become real special tokens unless you split them.

```python showLineNumbers
x = tok("Q?", "we pack [CLS] and [SEP] tokens")
# error-next-line
print(tok.convert_ids_to_tokens(x["input_ids"]))
# ❌ ['[CLS]', 'Q', '?', '[SEP]', 'we', 'pack', '[CLS]', 'and', '[SEP]', 'token', '##s', '[SEP]']
# success-next-line
y = tok("Q?", "we pack [CLS] and [SEP] tokens", split_special_tokens=True)
print(tok.convert_ids_to_tokens(y["input_ids"]))
# ✅ ['[CLS]', 'Q', '?', '[SEP]', 'we', 'pack', '[', 'C', '##LS', ']', 'and', '[', 'SE', '##P', ']', 'token', '##s', '[SEP]']
```

### 10. A SQuAD record becomes token labels and a loss

`answer_start` is a character index; the offsets turn it into `start_positions` / `end_positions`.

```python showLineNumbers
ex = {"question": "What does BERT stand for?",
      "context": "BERT stands for Bidirectional Encoder Representations from Transformers.",
      "answers": {"text": ["Bidirectional Encoder Representations from Transformers"], "answer_start": [16]}}
a_text, a_start = ex["answers"]["text"][0], ex["answers"]["answer_start"][0]
a_end = a_start + len(a_text)                    # exclusive character end
print(ex["context"][a_start:a_end] == a_text)    # True   (always check the label first)
enc = tok(ex["question"], ex["context"], return_offsets_mapping=True)
sid, off = enc.sequence_ids(), enc["offset_mapping"]
# highlight-start
start_pos = next(k for k, (o, t) in enumerate(zip(off, sid)) if t == 1 and o[0] <= a_start < o[1])
end_pos = next(k for k, (o, t) in enumerate(zip(off, sid)) if t == 1 and o[0] < a_end <= o[1])
# highlight-end
print(start_pos, end_pos, tok.decode(enc["input_ids"][start_pos:end_pos + 1]))
                                                 # 15 27 Bidirectional Encoder Representations from Transformers
with torch.no_grad():
    o = model(input_ids=torch.tensor([enc["input_ids"]]), attention_mask=torch.tensor([enc["attention_mask"]]),
              start_positions=torch.tensor([start_pos]), end_positions=torch.tensor([end_pos]))
print(round(o.loss.item(), 4))                   # 0.8377
```

### What was wrong in the notebook, and the fix

| Notebook code | Problem | Fix (blocks above) |
|---|---|---|
| `select_best_answer_span` | searches every position, including question, [CLS] and [SEP]; that's why unsupported questions' top "answers" were empty strings and a filter was needed | mask to `sequence_ids == 1` (block 3) |
| `tokenizer.decode(answer_ids)` | re-spaces punctuation: "encoder - only", "Ming - Wei Chang" | slice the source with `offset_mapping` (block 3) |
| `confidence = (p_start + p_end) / 2` | average of two softmaxes over all positions; not a probability of the span (0.5896 vs 0.3382) | span probability over valid spans, used only for ranking (block 4) |
| `truncation=True, max_length=384` on pages 1–4 | silently drops 92% of the text | chunk, or `return_overflowing_tokens` + `stride` (blocks 5–7) |
| `max_length=384` in the failure cells, 500 elsewhere | 8 chunks longer than 384 tokens lose their ends in those cells | one consistent limit at or above the longest chunk (432) |
| Ranking by raw span score, threshold 5.0 | logits aren't comparable across chunks; "Transformers" (6.34) passes for the city question | retrieval first, calibrated threshold, SQuAD 2.0 reader (Chapter 13) |
| Default tokenization of PDF text | literal "[CLS]"/"[SEP]" in 21 chunks become special tokens | `split_special_tokens=True` (block 9) |

## 🚨 Interview Gotchas & Edge Cases

:::warning[The Most Common Question]
**"How does an extractive QnA model find the answer, and how do you make it work on a document longer than 512 tokens?"** The encoder reads `[CLS] question [SEP] context [SEP]`; a `Linear(H, 2)` head gives every token a start and an end logit; the answer is the context span maximizing $s_i + e_j$ with $i \le j$, a length cap and both ends in the context, mapped back to text with the offset mapping. For long documents, split the context into overlapping chunks (or tokenizer windows with a stride), run the reader on each, and compare candidates, ideally only over chunks a retriever picked, since raw span scores aren't calibrated across chunks. Return the source with the answer.
:::

- **Independent argmaxes** can produce end < start or a span in the question; score pairs.
- **`decode` is not the source text**: use `offset_mapping` to slice the original string.
- **`pipeline("question-answering")` doesn't exist in transformers 5**: load `AutoModelForQuestionAnswering`.
- **SQuAD v1.1 models never abstain**: they always return a span ("Who invented BERT?" → "encoder-only Transformer model").
- **Confidence ≠ correctness**: "Which city was BERT invented in?" → "Transformers" at 0.9964.
- **Span scores are per forward pass**: comparing them across chunks is comparing different scales.
- **`truncation=True` drops text silently**: 4,556 tokens → 384 kept, 8% of the characters.
- **Words ≠ tokens**: 1.74 tokens per word on this paper; budget in tokens.
- **Overlap duplicates answers**: the same span can come from two neighbouring chunks; deduplicate.
- **PDF text contains your special tokens**: `split_special_tokens=True`.
- **Hyphen and ligature artefacts** change tokenization and answers ("at- tention"); normalize before chunking.
- **`answer_start` is a character index**, labels are token indices; verify `context[start:start+len] == text`.
- **DistilBERT ignores `token_type_ids`**; the segment information it needs comes from the [SEP] token and position.

## ⚡ Essential Interview Recall

A one-screen revision list. Every point is explained in the chapters above.

**The model**
- Extractive = copy a span; generative = write text (can hallucinate).
- `AutoModelForQuestionAnswering` = encoder + `Linear(768, 2)`: `start_logits`, `end_logits`, each `[B, T]`.
- `distilbert-base-cased-distilled-squad`: 6 layers, 768, cased vocab 28,996, 65,192,450 params (head 1,538), SQuAD v1.1.
- No `question-answering` pipeline in transformers 5.22.2.

**Picking the answer**
- Score $\pred{s_i} + \pred{e_j}$ over valid pairs: context-only (`sequence_ids == 1`), $i \le j$, length ≤ $\hyper{L}$.
- Slice the context with `offset_mapping`; `decode` re-spaces punctuation.
- Notebook confidence (mean of two softmaxes) 0.5896 vs span probability 0.3382.
- Unanswerable: v1.1 models still answer; SQuAD 2.0 compares with the [CLS] score + τ.

**Documents**
- PyMuPDF page-wise: 16 pages, 64,105 characters; scanned PDFs need OCR.
- Pages 1–4 = 4,556 tokens; at 384 the model saw 8%.
- Chunks: 180 words, 40 overlap, step 140 → 78 chunks; 1.74 tokens per word; 71–432 tokens.
- Tokenizer alternative: `return_overflowing_tokens=True`, `stride` (19 windows for pages 1–4).

**Ranking and safety**
- Brute force: 78 forward passes per question, about 10 s on CPU.
- Right answer 20.123 vs "evaluation server" 20.049; 3 of 5 questions wrong at rank 1, right answer often top-3.
- Filter degenerate spans, threshold (5.0 fails: "Transformers" 6.34), always return page + chunk + preview.
- Fine-tuning record: question, context, text, `answer_start` (16) → token labels 15, 27 → loss 0.8377.
- Next: retrieve top-k chunks first (RAG), then read.

## 📐 Formula Sheet

| Concept | Formula | In plain words |
|---|---|---|
| Span score | $\pred{\text{score}}(i, j) = \pred{s_i} + \pred{e_j}$ | start logit plus end logit |
| Valid spans | $\mathcal{V} = \{(i, j) : i, j \in \text{context}, i \le j < i + \hyper{L}\}$ | in the context, ordered, not too long |
| Best span | $(\hat\imath, \hat\jmath) = \arg\max_{\mathcal{V}} \pred{s_i} + \pred{e_j}$ | best allowed pair |
| Notebook confidence | $\tfrac{1}{2}(\operatorname{softmax}(\pred{s})_{\hat\imath} + \operatorname{softmax}(\pred{e})_{\hat\jmath})$ | average of two separate probabilities |
| Span probability | $\pred{P}(\hat\imath, \hat\jmath) = e^{\pred{s_{\hat\imath}} + \pred{e_{\hat\jmath}}} / \sum_{\mathcal{V}} e^{\pred{s_i} + \pred{e_j}}$ | share among the answers you could return |
| No-answer margin | $(\pred{s_{\hat\imath}} + \pred{e_{\hat\jmath}}) - (\pred{s_{[CLS]}} + \pred{e_{[CLS]}})$ | how far the best span beats "no answer" (meaningful only for SQuAD 2.0 models) |
| Chunk step | $\hyper{\text{step}} = \hyper{C} - \hyper{O}$ | size minus overlap |
| Chunks per page | $\max(1, \lceil (\data{W} - \hyper{O}) / (\hyper{C} - \hyper{O}) \rceil)$ | windows to cover the page |
| Window room | $\hyper{R} = \hyper{\text{max\_length}} - \data{Q} - 3$ | context tokens per window |
| Tokenizer windows | $1 + \lceil (\data{N} - \hyper{R}) / (\hyper{R} - \hyper{\text{stride}}) \rceil$ | windows for $\data{N}$ context tokens |
| Character end | $\data{a_{end}} = \data{a_{start}} + \lvert \data{\text{answer}} \rvert$ | exclusive end of the answer |
| Token labels | $\text{off}_t^{(0)} \le \data{a_{start}} < \text{off}_t^{(1)}$, $\text{off}_t^{(0)} < \data{a_{end}} \le \text{off}_t^{(1)}$ | tokens containing the first and last character |
| QnA loss | $\loss{L} = \tfrac{1}{2}(-\log \operatorname{softmax}(\pred{s})_{\data{y_{start}}} - \log \operatorname{softmax}(\pred{e})_{\data{y_{end}}})$ | mean of start and end cross-entropy |
| Head size | $2\hyper{H} + 2$ | 1,538 for $\hyper{H}$ = 768 |

## Concept-Check Answers (from the notebook)

### Lecture workflow

<details>
<summary><strong>Q1. Why are we avoiding pipeline("question-answering") in this fresh notebook?</strong></summary>

Because the installed `transformers` didn't recognize `"question-answering"` as a pipeline task, and in transformers 5.22.2 the task is indeed absent from the registry. Loading `AutoTokenizer` + `AutoModelForQuestionAnswering` works, and it exposes every step (tokens, logits, span selection) that we need for understanding, debugging and later fine-tuning.
</details>

<details>
<summary><strong>Q2. What is the main difference between extractive QnA and generative QnA?</strong></summary>

Extractive QnA selects a span of the given context (start and end positions), so the answer is verbatim from the source. Generative QnA writes a new answer with a decoder; it can rephrase and synthesize but may not stick to the context (hallucination).
</details>

<details>
<summary><strong>Q3. In sentiment classification the model predicts a class label. What does an extractive QnA model predict?</strong></summary>

Two scores for every token: `start_logits` and `end_logits`, each `[batch, seq_len]`. The best valid (start, end) pair inside the context defines the answer span.
</details>

<details>
<summary><strong>Q4. Why is it useful to track the answer text and answer position?</strong></summary>

Supervised QnA fine-tuning needs labels in the form "answer text + `answer_start`" (a character index), which are converted to start and end token positions with the offset mapping. Tracking positions from the start makes predictions directly reusable as training or evaluation data.
</details>

<details>
<summary><strong>Q5. Why is document QnA a natural bridge between basic Transformer inference and RAG?</strong></summary>

It introduces the long-context problem: a document doesn't fit in one model input, so it has to be chunked and the right chunk found. Brute-force reading of all chunks is slow and ranks poorly, which motivates adding retrieval before reading: that is RAG.
</details>

### Colab setup and library verification

<details>
<summary><strong>Q1. Why use AutoTokenizer and AutoModelForQuestionAnswering instead of pipeline("question-answering")?</strong></summary>

They load the checkpoint's own tokenizer and a model with the QnA head directly, independent of the pipeline registry (which lacks that task in this version), and they show the intermediate tensors. The same objects are the ones you fine-tune.
</details>

<details>
<summary><strong>Q2. Which library provides AutoTokenizer and AutoModelForQuestionAnswering?</strong></summary>

Hugging Face `transformers`.
</details>

<details>
<summary><strong>Q3. If transformers imports but the model fails to download, what could be the reason?</strong></summary>

No internet or blocked access to the Hugging Face Hub, a typo in the model name, a gated or private model needing authentication, a version incompatibility, or not enough disk or memory.
</details>

<details>
<summary><strong>Q4. Why install datasets and evaluate even though we aren't fine-tuning yet?</strong></summary>

They are the next step: `datasets` loads and preprocesses SQuAD-style training data, and `evaluate` provides QnA metrics (exact match, F1). Installing them now keeps the notebook ready for the fine-tuning follow-up without changing the environment later.
</details>

<details>
<summary><strong>Q5. Why verify the environment before loading large pretrained models?</strong></summary>

It catches missing packages, incompatible versions, a missing GPU or wrong device early, before spending time and bandwidth downloading and loading hundreds of megabytes of weights.
</details>

### Loading the pretrained QnA model

<details>
<summary><strong>Q1. What are the two main Hugging Face classes used in this section?</strong></summary>

`AutoTokenizer` (loads the checkpoint's tokenizer; here a cased `BertTokenizer`) and `AutoModelForQuestionAnswering` (loads `DistilBertForQuestionAnswering`, the encoder plus the span head).
</details>

<details>
<summary><strong>Q2. If the tokenized pair has sequence length 64, what is the shape of start_logits?</strong></summary>

`[1, 64]`: one start score per token for the single pair (and `end_logits` is also `[1, 64]`).
</details>

<details>
<summary><strong>Q3. If the model loads but runs slowly, what could be one possible reason?</strong></summary>

It is running on the CPU (no GPU, or the model/inputs weren't moved to `cuda`), the inputs are long (attention grows quadratically with length), or you are processing many chunks one at a time.
</details>

<details>
<summary><strong>Q4. How is a QnA model's output different from a sentiment model's?</strong></summary>

A sentiment model outputs one vector of class logits per sequence (`[B, num_labels]`, from the [CLS] position). A QnA model outputs two scores per token (`start_logits` and `end_logits`, each `[B, T]`) from a `Linear(H, 2)` applied to every position.
</details>

<details>
<summary><strong>Q5. Why are start_logits and end_logits important for later fine-tuning?</strong></summary>

Fine-tuning trains exactly these outputs: the loss is the cross-entropy of the start logits against the gold start token plus that of the end logits against the gold end token (averaged). Passing `start_positions` and `end_positions` to the model returns that loss (0.8377 on the notebook's example record).
</details>

### Reusable manual extractive QnA function

<details>
<summary><strong>Q1. Why create a reusable QnA function instead of repeating the forward pass?</strong></summary>

The same steps (tokenize, forward, choose span, decode, score) run for every question and, later, for every chunk. A function keeps them consistent, testable and easy to fix in one place (for example adding the context mask or offset slicing).
</details>

<details>
<summary><strong>Q2. For one question-context pair with 100 tokens, what is the shape of start_logits?</strong></summary>

`[1, 100]`.
</details>

<details>
<summary><strong>Q3. Why is taking argmax(start_logits) and argmax(end_logits) independently sometimes problematic?</strong></summary>

The two choices ignore each other: the end can fall before the start, the span can be far too long, or either end can land in the question or on a special token. Score pairs with $s_i + e_j$ and choose the best valid pair instead.
</details>

<details>
<summary><strong>Q4. Why can an extractive QnA model fail if the context doesn't contain the answer?</strong></summary>

It can only copy from the context, and a SQuAD v1.1 model was never trained to abstain, so it returns the most answer-like span anyway ("Who invented BERT?" → "encoder-only Transformer model", span score −11.95).
</details>

<details>
<summary><strong>Q5. How does this function connect to future QnA fine-tuning?</strong></summary>

Its inputs (question, context) and outputs (start, end positions, answer text) are the same fields a fine-tuning record uses; fine-tuning adds gold start and end labels and trains the logits this function reads.
</details>

<details>
<summary><strong>Q6. Why is extractive QnA an important intermediate step before RAG?</strong></summary>

It teaches answering strictly from provided context. RAG keeps that idea and adds a retrieval step that chooses which context to provide from a large collection.
</details>

### Uploading and reading a PDF page-wise

<details>
<summary><strong>Q1. Why extract PDF text page-wise instead of combining the whole document immediately?</strong></summary>

To keep page numbers as metadata, so every chunk and answer can be traced to its page for citations, debugging and building labelled data later.
</details>

<details>
<summary><strong>Q2. What type of PDF is best for this lecture?</strong></summary>

A text-based PDF whose text you can select and copy: research papers, lecture notes, documentation, reports.
</details>

<details>
<summary><strong>Q3. If the extracted text is almost empty, what is the likely reason?</strong></summary>

The PDF is scanned or image-only, so there is no text layer; it needs OCR before text-based QnA.
</details>

<details>
<summary><strong>Q4. Why can passing the entire PDF text directly to the model fail?</strong></summary>

The model accepts at most 512 tokens (384 in the notebook's function). With truncation the rest is silently dropped (pages 1–4: 4,556 tokens, only 8% of the characters seen), so the answer may never reach the model; without truncation, inputs over 512 positions fail.
</details>

<details>
<summary><strong>Q5. Why is it useful to preserve page numbers while extracting text?</strong></summary>

They make answers verifiable and debuggable, and they help when turning reviewed answers into labelled fine-tuning or evaluation examples.
</details>

<details>
<summary><strong>Q6. How does page-wise extraction help in building a document QnA or RAG system?</strong></summary>

It gives the document structure and metadata from the first step; RAG indexes chunks with metadata (page, section, source) and returns them as citations.
</details>

### Chunking long PDF text

<details>
<summary><strong>Q1. Why do we need to chunk long PDF documents before applying a QnA model?</strong></summary>

Because of the maximum input length: a whole document would be truncated. Chunks small enough to fit (180 words ≈ 313 tokens here) let every part of the document be read.
</details>

<details>
<summary><strong>Q2. Why use overlap between consecutive chunks?</strong></summary>

So that text cut by a chunk boundary appears whole in the neighbouring chunk: with 40 shared words, any answer up to 40 words crossing one boundary is fully inside one chunk.
</details>

<details>
<summary><strong>Q3. If a question + chunk becomes 300 tokens, what is the shape of input_ids for one pair?</strong></summary>

`[1, 300]`, and `start_logits` / `end_logits` are `[1, 300]` as well.
</details>

<details>
<summary><strong>Q4. If many chunks are still longer than the model limit, what should we change?</strong></summary>

Reduce `chunk_size` (for example 120 words with overlap 30), or switch to token-based windows (`return_overflowing_tokens=True` with a `stride`) that fit `max_length` exactly. Measure tokens per word on your own documents first (1.74 here).
</details>

<details>
<summary><strong>Q5. Why store chunk_id and page_number for each chunk?</strong></summary>

For source tracking: to show and verify where an answer came from, and to build fine-tuning datasets and RAG indexes that need those references.
</details>

<details>
<summary><strong>Q6. How is chunking related to Retrieval-Augmented Generation?</strong></summary>

Chunks are the units RAG embeds, indexes and retrieves; the retrieved chunks are what the reader or generator sees.
</details>

### QnA over PDF chunks

<details>
<summary><strong>Q1. Why run QnA over chunks instead of the full PDF?</strong></summary>

The full PDF (about 17,900 tokens) exceeds the model's input length; each chunk fits.
</details>

<details>
<summary><strong>Q2. Why keep the page number with each answer?</strong></summary>

Source tracking: users can verify the answer on that page, which makes the system trustworthy and easy to debug.
</details>

<details>
<summary><strong>Q3. Why do we rank candidate answers?</strong></summary>

Every chunk produces its own best span, so a question yields 78 candidates; ranking picks one (and shows a few alternatives). The ranking signal matters: raw span scores are not calibrated across chunks.
</details>

<details>
<summary><strong>Q4. If the best answer is wrong but another top-k answer is correct, what does that tell us?</strong></summary>

The reader found the answer but the ranking failed (for example "randomly masks some of the tokens from the input" at rank 3). Improve ranking: retrieve relevant chunks first, rerank, or validate answers.
</details>

<details>
<summary><strong>Q5. Why is brute-force QnA over every chunk not ideal for very large documents?</strong></summary>

Cost grows with questions × chunks (78 forward passes, about 10 s per question on CPU for one paper), and more irrelevant chunks means more chances for a wrong span to out-score the right one.
</details>

<details>
<summary><strong>Q6. How does this chunk-based QnA system lead to RAG?</strong></summary>

RAG replaces "read every chunk" with "retrieve the top-k relevant chunks, then read those", keeping the same chunks, metadata and reader.
</details>

### Failure case analysis

<details>
<summary><strong>Q1. Why can an extractive QnA model return an answer when the correct answer isn't present?</strong></summary>

It is trained (on SQuAD v1.1) to pick the most plausible span from whatever context it is given and has no no-answer option, so it always returns something ("Transformers" for "Which city was BERT invented in?").
</details>

<details>
<summary><strong>Q2. Why inspect the context preview along with the answer?</strong></summary>

To check that the text actually supports the answer. Plausible but wrong spans are common ("BERT SQuAD" for "Is BERT Encoder or Decoder only model?"), and the preview exposes them immediately.
</details>

<details>
<summary><strong>Q3. If the correct answer is in top-k but not rank 1, what needs improvement?</strong></summary>

The ranking: better scores across chunks (retrieval similarity, a reranker, calibrated scores) or answer validation.
</details>

<details>
<summary><strong>Q4. What is the benefit of a minimum score threshold?</strong></summary>

It lets the system abstain ("No reliable answer found") when the best candidate is weak, as for the favourite-programming-language question (3.14 below 5.0). It must be tuned on labelled questions, because a hand-picked threshold on raw logits still passes confident wrong answers (6.34 for "Transformers").
</details>

<details>
<summary><strong>Q5. Why is brute-force QnA over all chunks not scalable?</strong></summary>

Every question runs the full model on every chunk, so latency and compute grow linearly with the corpus; a retriever shortlists a few chunks instead.
</details>

<details>
<summary><strong>Q6. Why is source grounding important in document QnA systems?</strong></summary>

It lets users verify answers, and in legal, medical, academic or enterprise settings an unsupported answer can be harmful; grounding also makes errors visible for debugging.
</details>

### Final project summary and roadmap

<details>
<summary><strong>Q1. What is the final document QnA pipeline built in this lecture?</strong></summary>

PDF → page-wise text extraction → cleaning → overlapping chunks with metadata → extractive QnA on every chunk → ranking → filter and threshold → answer with page number, chunk id and context preview.
</details>

<details>
<summary><strong>Q2. What does the QnA head predict?</strong></summary>

A start logit and an end logit for every token; the best valid pair in the context gives the answer span.
</details>

<details>
<summary><strong>Q3. What fields are required to fine-tune an extractive QnA model?</strong></summary>

`question`, `context`, answer `text` and `answer_start` (character index), usually with an `id`. They are converted into `start_positions` and `end_positions` token labels with the offset mapping.
</details>

<details>
<summary><strong>Q4. Why is brute-force QnA over all chunks not enough for large-scale systems?</strong></summary>

It is too slow and ranks poorly across many chunks; retrieval selects the relevant chunks first.
</details>

<details>
<summary><strong>Q5. How is LLM-based document QnA different from extractive QnA?</strong></summary>

An LLM generates a natural-language answer, possibly combining several passages, while extractive QnA copies one span. The LLM is more flexible but can hallucinate, so it must be grounded in retrieved context.
</details>

<details>
<summary><strong>Q6. Why is source tracking important in document QnA systems?</strong></summary>

It allows verification, builds trust, supports debugging and accountability, and turns reviewed answers into future training data.
</details>
