---
title: "Generative RAG End to End: From a PDF to a Grounded Answer"
sidebar_position: 4
---

import {RagSystemLab, PromptBudgetLab} from '@site/src/components/Widgets';

:::tip[TL;DR (30-Second Interview Pitch)]
**Retrieval-Augmented Generation (RAG)** answers questions about a document by doing two things in order: **retrieve** the few passages that are about the question, then let a **generative model** write the answer *from those passages only*. Notebook 22 builds the whole thing over the BERT paper with free models: PyMuPDF reads the PDF, sentence-aware chunks are embedded with `all-MiniLM-L6-v2`, **FAISS** finds the closest chunks, and the encoder-decoder **FLAN-T5** writes the answer. Measured on 12 factual questions, the honest picture is: the notebook's prompt got **1 of 12** right, choosing the most relevant *sentences* for the prompt raised it to **5 of 12**, an extractive reader also got **5 of 12** (different ones), and **7 of 12** were answered by at least one of them. So a real system needs four things on top of "retrieve + generate": **hybrid search** (meaning + keywords), a **router** that sends exact-fact questions to an extractive reader and explanation questions to the generator, a **grounding check** that refuses answers whose words aren't in the context, and **evaluation** that tells retrieval failures apart from generation failures.
:::

<FormulaKey />

:::info[Where this page sits]
**Before:** [Retriever + Reader with FAISS](./03-retriever-reader-with-faiss.md) found chunks by embedding search and *extracted* the answer span from them.
**This page:** notebook 22, *From Retriever-Reader PDF QnA to Generative RAG Assistant using LLMs and Streamlit*. It is written to be read **on its own**: every step from the raw PDF to the final app is explained again here from scratch (reading, cleaning, chunking, embeddings, FAISS), then the new parts are added (Transformer families, prompts, generation, routing, grounding, the app, fine-tuning). You don't need the earlier pages to follow it.
**Next:** the one-screen [Generative RAG cheat sheet](../../../05-cheat-sheets/01-deep-learning/15-generative-rag.md).
:::

## 🧭 Start Here

Imagine you have a 16-page research paper and you want to ask it questions in plain English: "What corpus was BERT trained on?", "Why does BERT use next sentence prediction?", "How does BERT differ from GPT?". A large language model on its own would answer from whatever it memorised during training, which may be out of date, may be about a different paper, and can be confidently invented. What we want instead is a system that **looks the answer up in this document**, writes it in readable language, and shows **which page it came from**. That system is called **Retrieval-Augmented Generation**, RAG for short, and this page builds one end to end.

The page follows the order in which data flows through the system, and each chapter is one stage. First the document is prepared, once: the PDF is **read page by page** (Chapter 3), the text is **cleaned** (Chapter 4), cut into **chunks** of whole sentences (Chapter 5), measured in **tokens** (Chapter 6), turned into **embedding vectors** (Chapter 7) and stored in a **FAISS index** (Chapter 8). Then, for every question, the system **searches** for the right chunks by meaning and by keywords (Chapters 8 and 9), packs them into a **prompt** (Chapter 10) and lets **FLAN-T5 generate** the answer token by token (Chapter 11). The last part is about making it trustworthy: **measuring** where it fails (Chapter 12), **routing** each question to the right kind of answerer (Chapter 13), **checking grounding and refusing** when the document doesn't support an answer (Chapter 14), wrapping it in a **Streamlit app** (Chapter 15), and knowing **when fine-tuning helps** (Chapter 16). Chapters 1 and 2 come first because they explain *why* a generator is needed and *which kind* of Transformer can generate.

Every number on this page comes from a real run on a CPU over the same BERT paper PDF the course uses (transformers 5.22.2, sentence-transformers, faiss-cpu, PyMuPDF). The notebook's own outputs are reproduced exactly where they are quoted (54 chunks, FLAN-T5-small answering with a copied source header, the token budget that printed 0). The evaluation in Chapter 12 is new: 12 factual questions with known answers, 5 explanation questions and 2 questions the paper can't answer, written for this page so that "does it work?" has a measurable answer. The final, improved pipeline is in the 💻 Python Implementation section, and every output there is from a real run.

| New concept | One-line meaning |
|---|---|
| **RAG** | Retrieve relevant passages first, then generate an answer from them |
| **Retriever** | The search part: finds the chunks most related to the question |
| **Extractive reader** | A model that copies an answer span out of a passage (BERT-style QnA) |
| **Generator** | A model that writes new text token by token (FLAN-T5 here) |
| **Encoder / decoder** | The understanding half / the writing half of a Transformer |
| **Chunk** | A piece of the document small enough to embed and to fit in a prompt |
| **Token** | The unit a model actually reads; one word is often 1–3 tokens |
| **Embedding** | A list of 384 numbers that places texts with similar meaning close together |
| **Cosine similarity** | How closely two embeddings point the same way (1 = same direction) |
| **FAISS** | A library that stores embeddings and returns the nearest ones fast |
| **BM25** | Classic keyword scoring: rare words that match count most |
| **Hybrid search (RRF)** | Merge the meaning ranking and the keyword ranking by rank position |
| **Prompt** | Instruction + question + retrieved context, sent to the generator |
| **Greedy / beam search** | Pick the single best next token / keep several candidate answers in parallel |
| **Grounding** | Whether the answer is actually supported by the retrieved text |
| **Router** | Decides per question: extract a span, generate an answer, or refuse |

![The whole RAG system: prepare the document once (read, clean, chunk, embed, index), then for each question search, gate on cosine, route to the extractive reader or the generator, and check grounding before answering](/img/deep-learning/diagrams/rag-end-to-end.svg)

*How to read it:* the top band runs **once per document** and produces the chunk list and the index. The bottom band runs **once per question**. Two gates can stop a question and return the refusal sentence: a low best cosine (the document isn't about this) and low support (the generated answer uses words the context never had).

:::analogy[An open-book exam with a strict examiner]
A plain LLM is a student answering from memory: fluent, fast, and sometimes inventing. RAG turns it into an **open-book exam**. The retriever is the student flipping to the right pages, the generator is the student writing the answer in their own words from those pages, and the grounding check is the examiner who asks "show me where it says that" and marks the answer zero if the student can't. A good student also knows when to write "the book doesn't cover this".
:::

## 1. Why Generate? Extractive Answers vs Generated Answers

The previous system in this course was a **retriever + reader**. The retriever found the five most related chunks; the reader, a BERT model fine-tuned on the SQuAD question-answering dataset, looked inside each chunk and returned the **span** of text most likely to be the answer, a start position and an end position. That design is called **extractive** question answering because the answer is always *extracted*, copied character for character, from the passage. It has real strengths: the answer can't contain words that aren't in the document, and for a fact that appears verbatim ("BooksCorpus", "15%", "30,000") it is precise.

Its weakness shows up as soon as the answer isn't one contiguous phrase. Ask "What is next sentence prediction and why is it used?" and the honest answer combines a definition from one sentence with a purpose from another. A span-picker can only return one of them, or a long stretch of text that happens to contain both. Summaries ("What is the main contribution?"), comparisons ("How does BERT differ from GPT?") and explanations ("Why is BERT bidirectional?") all have this shape: the answer has to be **written**, not copied. That needs a **generative** model, one that produces new text one token at a time.

Generation brings a new risk. A model that can write anything can also write something the document never said, which is called **hallucination**. That is why the generator in RAG is never asked a bare question: it is given the retrieved passages and told to answer *only* from them. Formally, a plain LLM models $P(\pred{answer} \mid \data{question})$, while RAG models $P(\pred{answer} \mid \data{question}, \data{retrieved\ context})$. Using an LLM alone is therefore **not** RAG; the retrieval step is what makes it RAG, and the retrieved sources are what make the answer checkable.

<Formula name="RAG in one line" tag="Retrieve, then generate" words="First pick the k chunks most similar to the question, then generate the answer conditioned on the question and only those chunks." remember="Search first, then write.">

$$
\begin{aligned}
\data{C_k} &= \operatorname{top\text{-}k}_{c}\ \operatorname{sim}(\data{q}, \data{c}) \\
\pred{\hat y} &\sim P_{\param{\theta}}(\pred{y} \mid \data{q}, \data{C_k})
\end{aligned}
$$

- $\data{q}$: the question; $\data{c}$: one chunk of the document
- $\data{C_k}$: the $\hyper{k}$ chunks with the highest similarity to the question
- $P_{\param{\theta}}$: the generator with its learned weights $\param{\theta}$
- $\pred{\hat y}$: the generated answer

**Example:** for "Which corpus is used for pre-training besides English Wikipedia?", the retriever's top 5 on this page are chunks 46, 13, 12, 4 and 47; chunk 13 (page 5) contains "BooksCorpus (800M words)".

</Formula>

| | Extractive QnA (reader) | Generative RAG (generator) |
|---|---|---|
| Output | A span copied from the passage | New text written token by token |
| Exact facts, names, numbers | **Strong**: copies them exactly | Can paraphrase or garble them |
| Explanations, summaries, comparisons | Weak: one span only | **Strong** |
| Combining several passages | Weak | **Strong** |
| Hallucination risk | Low (but can pick a wrong span) | Higher: must be checked |
| Can say "not in the document" | Not a SQuAD v1.1 model: it always returns some span | Yes, if the prompt allows it |

:::danger[Interview Trap: "RAG means using an LLM"]
An LLM without retrieval is just an LLM. RAG is defined by the **retrieval** step that injects external, up-to-date, citable context before generation. And retrieval failure and generation failure are different problems: if the right chunk never reaches the prompt, no generator can fix it.
:::

:::tip[🎯 Interview angle]
**"When would you choose extractive QnA over generative RAG?"** For exact-fact questions where the answer appears verbatim (a number, a name, a full form), extractive QnA copies it precisely and can't invent words. For explanations, summaries and comparisons that need several passages combined, a generator is needed. In practice a good system uses both and **routes** each question, which on this page answers 7 of 12 factual questions with at least one of the two, against 5 of 12 for either alone.
:::

## 2. Three Transformer Families: Who Can Read, Who Can Write

All the models in this pipeline are **Transformers**, but they are built from different halves of the original Transformer, and that decides what they can do. A Transformer layer lets every token look at other tokens through **self-attention** and mix in their information. The key design choice is *which* tokens each position is allowed to look at. That choice splits Transformers into three families.

An **encoder-only** model (BERT, DistilBERT, and the MiniLM embedding model) lets every token attend to **every other token**, left and right. That is called **bidirectional** attention, and it gives each token a representation built from its whole sentence, which is ideal for *understanding*: classifying a sentence, comparing two texts, or scoring which token starts and ends an answer. What it can't do is write: it has no mechanism for producing a new token after the last one. That is exactly why the BERT reader of the earlier pages **extracts**: its output is one **start logit** and one **end logit** per input token (a score for "the answer starts here" and "the answer ends here"), and the answer is the best-scoring span between them.

A **decoder-only** model (GPT, Llama, Mistral) uses a **causal mask**: each token may attend only to tokens on its **left**. During training it learns to predict the next token from all the previous ones, and at inference it generates by repeating that step: predict a token, append it, predict the next. This step-by-step process is called **autoregressive generation**. An **encoder-decoder** model (T5, FLAN-T5, BART) has both halves: an encoder reads the whole input bidirectionally, and a decoder writes the output autoregressively while attending to the encoder's representation through **cross-attention**. That split, "read the context fully, then write an answer about it", is a natural fit for RAG, translation and summarisation.

![Encoder-only: every token sees every token (full attention grid). Decoder-only: each token sees only earlier tokens (lower-triangular grid). Encoder-decoder: a bidirectional encoder reads the input; a causal decoder writes the output and cross-attends to the encoder](/img/deep-learning/diagrams/transformer-families.svg)

*How to read it:* each grid shows which positions (columns) a token (row) may attend to. A full grid is bidirectional; a lower triangle is causal. The encoder-decoder's decoder has its own causal grid plus a cross-attention link to every encoder position.

<Formula name="Autoregressive generation" tag="How decoders write" words="The probability of a whole answer is the product of the probability of each token given all the tokens before it. An encoder-decoder also conditions every step on the encoded input x." remember="One token at a time, left to right.">

$$
\begin{aligned}
P(\pred{y} \mid \data{x}) &= \prod_{t=1}^{\hyper{m}} P_{\param{\theta}}(\pred{y_t} \mid \pred{y_{1}}, \dots, \pred{y_{t-1}}, \data{x}) \\
\log P(\pred{y} \mid \data{x}) &= \sum_{t=1}^{\hyper{m}} \log P_{\param{\theta}}(\pred{y_t} \mid \pred{y_{\lt t}}, \data{x})
\end{aligned}
$$

- $\data{x}$: the input (for RAG: instruction + question + context)
- $\pred{y_t}$: the token generated at step $t$; $\pred{y_{\lt t}}$: all tokens before it
- $\hyper{m}$: the answer length (stops at the end-of-sequence token or `max_new_tokens`)
- the log form turns the product into a sum, which is what decoding algorithms actually add up

**Example:** the greedy answer to the NSP question on this page is 27 tokens long; its log-probabilities add up to −3.768, an average of −0.1395 per token.

</Formula>

| Family | Attention | Strong at | Examples | Role on this page |
|---|---|---|---|---|
| **Encoder-only** | Bidirectional, all tokens | Understanding, classification, span extraction, embeddings | BERT, DistilBERT, MiniLM | Embedding model, extractive reader |
| **Decoder-only** | Causal, left only | Open-ended generation, chat | GPT, Llama, Mistral | (the usual "LLM"; not used here) |
| **Encoder-decoder** | Encoder bidirectional + decoder causal + cross-attention | Input-to-output tasks: summarise, translate, answer from context | T5, FLAN-T5, BART | **The generator** |

**Why FLAN-T5?** FLAN-T5 is Google's T5 encoder-decoder after **instruction tuning**: it was fine-tuned on a large collection of tasks phrased as instructions ("Answer the question using the context…"), so it follows a prompt without further training. It is also small enough for a free Colab or a laptop CPU. `google/flan-t5-small` has 76,961,152 parameters; `google/flan-t5-base`, which the notebook switches to later, has 247,577,856 (12 encoder + 12 decoder layers, hidden size 768, a 32,128-token vocabulary). Both read at most **512 input tokens**, a limit that drives several design decisions below. Swapping the BERT reader for FLAN-T5 changes only the *answer-producing* component: the retriever is still needed to supply the context.

:::analogy[A reader, a storyteller and an interpreter]
An encoder-only model is a careful **reader** who sees the whole page at once and can underline the answer, but never writes a sentence of their own. A decoder-only model is a **storyteller** who keeps adding the next word to what they've said so far. An encoder-decoder is an **interpreter**: they listen to the entire speech first (encoder), then deliver it in their own words (decoder), glancing back at their notes (cross-attention) for every sentence.
:::

:::tip[🎯 Interview angle]
**"Why can't BERT generate answers, and what would you use instead?"** BERT is encoder-only: bidirectional attention gives great token representations, and the QnA head turns them into start and end logits, so it can only select a span. Generation needs a causal decoder that predicts the next token from the previous ones. For RAG over a document, an encoder-decoder such as FLAN-T5 fits naturally: the encoder reads the retrieved context fully and the decoder writes the answer while cross-attending to it; decoder-only LLMs do the same job with the context simply placed before the question.
:::

## 3. Step 1: Read the PDF Page by Page

Everything downstream depends on getting the text out of the PDF correctly, so the pipeline starts with **PyMuPDF** (imported as `fitz`). `fitz.open(path)` opens the file and gives access to its pages; `page.get_text()` returns that page's text as one string. The notebook extracts **page by page** rather than as one long string for a simple reason: every chunk made later inherits its page number, and the page number is what lets the final app say "Source: page 5". Without it, an answer can't be traced, debugged or trusted.

On the BERT paper this gives 16 pages; the first four have 584, 669, 583 and 811 words, and no page has fewer than 20 words. That last check matters. A page with almost no words usually means the PDF is **scanned**: its pages are images, `get_text()` returns nothing, and the document needs **OCR** (optical character recognition, e.g. Tesseract) before any RAG can work. Text-based PDFs, where you can select and copy the text, are what this pipeline expects.

Research papers add their own extraction problems, and it pays to look at the raw text before going further. **Two-column layouts** can interleave lines from both columns; **tables and equations** come out as runs of numbers and symbols ("80.6/80.1 66.1 82.3 93.2…"); **figures** come out as their labels ("BERT E E1 E... EN Tok 1... Tok N"), which becomes important in Chapter 12; and **headers, footers and page numbers** ("4173" at the top of every page here) get mixed into the text. Better extractors can read text *blocks* with their positions (`page.get_text("blocks")`), skip empty pages, and attach document metadata (title, file name) to every page.

| Extraction problem | What it looks like in the text | Remedy |
|---|---|---|
| Scanned PDF | Pages with 0 words | OCR first |
| Two columns | Sentences from both columns interleaved | Block-based extraction, sort by column |
| Tables, equations | Runs of numbers and symbols | Detect and keep separately, or describe |
| Figure labels | "Tok 1... Tok N Masked Sentence A" | Drop or tag figure regions |
| Headers, page numbers | "4173" before the first sentence | Strip repeated header patterns |

:::tip[🎯 Interview angle]
**"Why extract text page by page, and what do you check before chunking?"** Page-wise extraction keeps the page number attached to every piece of text, which is how the final answer can cite its source. Before chunking, check word counts per page (near-zero means a scanned PDF that needs OCR) and read a sample: poor extraction becomes poor chunks, poor retrieval and poor answers, and no later stage can recover text that was never extracted.
:::

## 4. Step 2: Clean the Text Without Destroying It

Raw PDF text is noisy in predictable ways. Words are broken across lines with a hyphen ("repre-\nsentation"), ligatures are stored as single characters ("ﬁ" for "fi", "ﬂ" for "fl"), there are invisible characters (zero-width spaces, byte-order marks), runs of spaces and newlines, and stray spaces before punctuation ("mechanisms ."). **Cleaning** removes that noise because noisy chunks make noisy embeddings, and noisy embeddings make retrieval worse. A word like "ﬁne-tuning" with a ligature doesn't even match "fine-tuning" in keyword search.

Each cleaning step should have a clear target. **Unicode NFKC normalisation** turns compatibility characters into their plain forms (the "ﬁ" ligature becomes "fi"). **Joining hyphenated line breaks** (`(\w)-\n(\w)` → the two halves joined) repairs "repre-\nsentation" into "representation"; it must only fire on a hyphen *followed by a newline*, or it would also break real hyphens. **Collapsing whitespace** and **removing the space before punctuation** tidy the rest. On a test string, the page's cleaner turns "BERT is de-\nsigned to pre-\ntrain deep ﬁne-tuned  models ." into "BERT is designed to pretrain deep fine-tuned models."

The danger is cleaning **too aggressively**. Technical documents use symbols that carry meaning: brackets, slashes, percent signs, Greek letters, equations, version numbers. Stripping "all special characters" destroys them. The notebook's cleaner made a subtler mistake: it deleted every occurrence of `[CLS]`, `[SEP]` and `[MASK]` because they look like tokenizer artifacts. In most PDFs they would be. But the BERT paper is *about* those tokens: its raw text contains `[CLS]` 28 times, `[SEP]` 25 times and `[MASK]` 11 times, as words in its own sentences. After the notebook's cleaning, the sentence explaining them reads "is a special symbol added in front of every input example, and is a special separator token", which no retriever or reader can use. The fix is simply not to delete words that are part of the document's content; the tokenizer is told separately not to treat them as special (Chapter 13).

| Cleaning step | Example | Keep it? |
|---|---|---|
| NFKC normalisation | "ﬁne" → "fine" | ✅ yes |
| Remove zero-width characters | invisible `\u200b` → space | ✅ yes |
| Join hyphenated line breaks | "repre-\nsentation" → "representation" | ✅ yes (hyphen + newline only) |
| Collapse whitespace, fix " ." | "models  ." → "models." | ✅ yes |
| Delete `[CLS]`, `[SEP]`, `[MASK]` | "[CLS] is a special symbol" → "is a special symbol" | ❌ not when the document talks about them |
| Remove all special characters | "15%", "$H=768$", "⟨Question, Answer⟩" damaged | ❌ never for technical text |

:::danger[Interview Trap: cleaning that deletes content]
"Remove special tokens" sounds harmless until the document is about NLP. A useful habit, from the notebook: compare **raw and cleaned word counts** per page. A small drop means you removed noise; a big drop means your cleaner is eating content.
:::

:::analogy[Washing vegetables, not peeling them away]
Cleaning text is washing vegetables: rinse off the dirt (broken hyphens, odd spaces, invisible characters) but keep the vegetable. A cleaner that deletes every bracketed token is a peeler that keeps going until the carrot is gone. On the BERT paper it peeled away the very words the paper explains.
:::

:::tip[🎯 Interview angle]
**"What can go wrong when you clean PDF text for RAG?"** Under-cleaning leaves ligatures, broken hyphenation and noise that weaken embeddings and break keyword matches; over-cleaning deletes meaning. On the BERT paper, deleting `[CLS]`, `[SEP]` and `[MASK]` removed 64 occurrences of the paper's own subject matter and garbled the sentences that define them. Clean with targeted rules, never "strip all symbols", and compare raw vs cleaned word counts to catch content loss.
:::

## 5. Step 3: Sentence-Aware Chunking

A whole document can't be embedded or put in a prompt at once, so it is cut into **chunks**. The simplest method, a fixed number of words per chunk, cuts sentences in half: one chunk ends with "BERT is pre-trained on" and the next begins "BooksCorpus and English Wikipedia", and neither half answers the question. **Sentence-aware chunking** first splits the text into sentences and then packs *whole sentences* into chunks until a size limit is reached. Ideas stay intact, and a retrieved chunk reads like a real paragraph.

Two parameters control it. **`max_words`** (250 here) caps the chunk size, because chunks still have to fit the embedding model and the prompt; a chunk is closed when the next sentence would push it over. **`sentence_overlap`** (2 here) copies the last two sentences of each chunk into the start of the next, so an idea that straddles a boundary appears whole in at least one chunk. Every chunk also stores its **page number** and an **id**, which travel with it through retrieval to the final source display.

Splitting sentences in a PDF is harder than splitting on full stops. "e.g.", "i.e.", "Fig. 1", "et al." and "vs." all contain a period followed by a space, and equations, references and table rows often have no punctuation at all. The page's splitter protects common abbreviations before splitting on `. ! ?` followed by a space and a capital letter, digit, quote or bracket: "Transformers use self-attention. Fig. 1 shows the architecture, e.g. the encoder. It removes recurrence." splits into exactly three sentences. On the BERT paper the notebook's chunker makes **54 chunks** (79–249 words, average 220.74); the page's version, which keeps `[CLS]`-style words and protects abbreviations, makes **55 chunks** (44–249 words, average 216.71).

| Chunking strategy | Splits on | Strength | Weakness |
|---|---|---|---|
| Fixed words | Every N words | Simple, even sizes | Cuts sentences and ideas |
| **Sentence-aware** (this page) | Sentence boundaries, up to `max_words` | Whole sentences, readable chunks | Sentence splitting in PDFs is imperfect |
| Section-aware | Headings (Abstract, Method, Results) | Chunks never mix sections; section metadata helps search | Needs reliable heading detection |
| Token-aware | A token budget, not words | Guarantees chunks fit the model | Needs the model's tokenizer |
| Semantic | Where the topic changes (embedding similarity between neighbouring sentences drops) | Each chunk is one topic | Slower, uneven sizes |

The notebook's recommended long-term order is **section-aware + token-aware + semantic-aware**: respect the document's structure, make sure each chunk fits the models, and split where the meaning changes. Whatever the strategy, **look at the chunks** before embedding them: a chunk made of reference-list entries or figure labels will be retrieved for the wrong questions, as Chapter 8 shows.

:::analogy[Cutting a cake along the layers]
Fixed-word chunking is cutting a layered cake with a ruler: every slice is the same size but each one has half a layer of something. Sentence-aware chunking cuts between the layers, so every slice has complete layers. Overlap is leaving a thin strip of the previous layer on each slice so nothing that sits on a boundary is lost.
:::

:::tip[🎯 Interview angle]
**"How do you choose a chunking strategy for RAG?"** Start sentence-aware so chunks hold complete ideas, with a size cap that fits both the embedding model and the generator's prompt (measured in tokens, not just words), and a small overlap of one or two sentences so boundary ideas survive. Store page and section metadata on every chunk. Then inspect the chunks: references, tables and figure labels produce chunks that match many questions and answer none, so they're worth tagging or filtering.
:::

## 6. Step 4: Count Tokens, Not Words

Models don't read words; they read **tokens**, the pieces a tokenizer cuts text into. Common words are one token, rare words split into several ("binarized" becomes "bin", "ar", "ized"), and numbers, symbols and reference lists split into many. So a chunk's word count says little about whether it fits a model. The notebook measured its 54 chunks with the FLAN-T5 tokenizer: **124 to 522 tokens**, mean 373.39, median 382, 90th percentile 480.3, about **1.7 tokens per word** on average. The longest chunks were all from the **reference list** (522, 516 and 514 tokens for about 245 words each), because author names, years and venues tokenize badly.

Token counts matter at two places in the pipeline, and both have hard limits. The **embedding model** `all-MiniLM-L6-v2` reads at most **256 tokens** and silently **truncates** the rest: the vector describes only the beginning of the chunk. On the notebook's chunks, 50 of 54 were longer than 256 tokens and 29.5% of all chunk tokens were never seen by the embedder; on this page's chunks, 48 of 55 are over the limit. A sentence near the end of a long chunk is therefore invisible to dense search, which is one reason keyword search (Chapter 9) helps. The **generator** FLAN-T5 reads at most **512 input tokens**, and that budget has to hold the instruction, the question and all the retrieved context; anything past 512 is cut off, usually the end of the prompt, which is where the question and the "Answer:" cue are if the prompt is built carelessly.

<Formula name="Prompt token budget" tag="What fits in the generator" words="The context gets whatever the input limit leaves after the fixed parts of the prompt, capped by a maximum you choose." remember="Fixed parts first, context gets the rest.">

$$
\begin{aligned}
\hyper{B_{ctx}} &= \min(\hyper{L_{max}} - n_{instr} - n_{cue} - n_{safety},\ \hyper{B_{cap}}) \\
&= \min(512 - 67 - 2 - 12,\ 360) = 360
\end{aligned}
$$

- $\hyper{L_{max}}$: the model's input limit (512 for FLAN-T5)
- $n_{instr}$: tokens in the instruction + question (67 in the notebook's balanced prompt)
- $n_{cue}$: the "Answer:" marker (2 tokens); $n_{safety}$: a small margin (12)
- $\hyper{B_{cap}}$: the maximum context you allow (360 in the notebook)

**Example:** the notebook's balanced prompt had 360 tokens available, used 306 of them for context, and the final prompt was 375 tokens, under 512.

</Formula>

| Limit | Model | What happens past it | Seen on this paper |
|---|---|---|---|
| 256 tokens | MiniLM embedder | Truncated silently: the vector ignores the rest | 48 of 55 chunks too long |
| 384 tokens per window | DistilBERT reader | Split into overlapping windows (stride 128) | Handled |
| 512 tokens | FLAN-T5 generator | Prompt truncated, usually its end | Must budget the context |

:::tip[🎯 Interview angle]
**"Why measure chunks in tokens rather than words?"** Model limits are in tokens, and the tokens-per-word ratio varies a lot: about 1.7 on average here, more for references, tables and math. An embedding model with a 256-token limit silently ignores the end of longer chunks, and a generator with a 512-token limit cuts the prompt. Chunk and budget with the model's own tokenizer.
:::

## 7. Step 5: Embeddings and Cosine Similarity

To search by meaning, every chunk is turned into an **embedding**: a fixed-length vector of numbers produced by a model trained so that texts with similar meaning get vectors pointing in similar directions. The notebook uses `sentence-transformers/all-MiniLM-L6-v2`, a small encoder-only model (6 layers) that outputs **384 numbers** per text. "What corpus was BERT trained on?" and "BERT is pre-trained on BooksCorpus and English Wikipedia" share few words but land close together; a sentence about cooking lands far away. Embedding all 54 notebook chunks gives a matrix of shape $(N, d) = (54, 384)$, one row per chunk.

Closeness is measured by **cosine similarity**, the cosine of the angle between two vectors: 1 when they point the same way, 0 when they are unrelated (at right angles), negative when they point in opposite directions. Computing it normally needs a division by both vector lengths, but the notebook asks the model for **normalized** embeddings (`normalize_embeddings=True`), which scales every vector to length exactly 1 (it checks: minimum, maximum and average norm are all 1.0). For unit vectors, cosine similarity is simply the **dot product**, which is the cheapest operation a search library can do.

Three rules keep embedding search correct. The question must be embedded with the **same model** as the chunks; vectors from different models live in different spaces and their dot products mean nothing. The vectors are stored as **float32**, the type FAISS requires. And the vectors are kept **separate from the chunk records** (`chunks_data`): the vector matrix is for search, the list of dictionaries (text, page, id) is for people, and row $i$ of one matches entry $i$ of the other.

<Formula name="Cosine similarity" tag="Meaning closeness" words="The dot product of two vectors divided by both their lengths. For unit-length vectors the division disappears and cosine is just the dot product." remember="Unit vectors: cosine equals dot product.">

$$
\begin{aligned}
\cos(\data{a}, \data{b}) &= \frac{\data{a} \cdot \data{b}}{\lVert \data{a} \rVert\, \lVert \data{b} \rVert} \\
\lVert \data{a} \rVert = \lVert \data{b} \rVert = 1 &\Rightarrow \cos(\data{a}, \data{b}) = \data{a} \cdot \data{b} = \textstyle\sum_i a_i b_i
\end{aligned}
$$

- $\data{a}, \data{b}$: two embeddings (a question and a chunk), 384 numbers each
- $\lVert \cdot \rVert$: vector length; normalising makes it 1

**Example:** $\data{a} = (0.6, 0.8)$ and $\data{b} = (1, 0)$ are both unit length, so $\cos = 0.6 \cdot 1 + 0.8 \cdot 0 = 0.6$. On the paper, "What does BERT stand for?" has a best chunk cosine of 0.4719; "Who won the 2018 FIFA World Cup?" reaches only 0.1408.

</Formula>

:::analogy[Arrows on a map of meaning]
Think of each text as an arrow drawn from the centre of a huge map, pointing toward its topic. Two texts about the same thing point the same way, whatever their length or wording. Cosine similarity ignores how long the arrows are and only asks how much they point the same way, so normalising every arrow to length 1 loses nothing.
:::

:::tip[🎯 Interview angle]
**"Why normalize embeddings, and why must the question use the same model?"** After L2 normalisation every vector has length 1, so cosine similarity equals the dot product, which an inner-product index computes very fast. The question must be embedded by the same model because each model defines its own vector space: a question vector from model A compared with chunk vectors from model B produces numbers with no meaning.
:::

## 8. Step 6: FAISS and Top-k Retrieval

With 54 vectors, a NumPy matrix product would find the best chunks instantly. **FAISS** (Facebook AI Similarity Search) is the tool for when there are millions: a library of **vector indexes** that store vectors and return the nearest ones to a query. The notebook uses `IndexFlatIP`: "Flat" means it stores every vector as is and compares the query with **all** of them (exact search), and "IP" means it scores by **inner product**, which equals cosine because the vectors are normalised. You `add` the chunk matrix once; each `search(q, k)` returns two arrays of shape (1, k): the **scores** and the **row indices** of the top $k$ vectors (an index of −1 means fewer than $k$ results exist).

FAISS stores only numbers. It knows nothing about text, pages or chunk ids, so the returned row numbers are used to look up the chunk records: `chunks_data[i]` gives the text and page for row $i$. That is why the order of the vector matrix and the chunk list must never drift apart.

A retrieval score is a **similarity**, not a guarantee. Two results from the notebook show the difference. For the vague question "What is the main idea discussed in this document?", the top five cosines were only 0.2209, 0.2080, 0.2030, 0.2006 and 0.1972, and three of the five chunks were pages of the **reference list**: a vague question matches text that is "about papers" in general. For "What is the full form of BERT?", the top chunk scored a high 0.5080 but was **figure-label noise** ("BERT BERT E E1 E... EN Tok 1... Tok N"), while the chunk that actually says "BERT, which stands for Bidirectional Encoder Representations from Transformers" ranked only **12th**. A high score means "talks about BERT a lot", not "contains the answer". A **very low** best score is still useful as a signal: if nothing in the document is even moderately similar, the question is probably outside it (Chapter 14).

| FAISS index | Search | Speed at scale | Recall | Use when |
|---|---|---|---|---|
| `IndexFlatIP` / `IndexFlatL2` | Exact: compares with every vector | Linear in N | 100% | Small and medium corpora, and as the exact baseline |
| `IndexIVFFlat` | Probes only the nearest clusters | Much faster | Slightly below 100% | Millions of vectors |
| `IndexHNSWFlat` | Walks a neighbour graph | Very fast queries | Slightly below 100% | Low-latency search |

:::danger[Interview Trap: "high retrieval score = right answer"]
Retrieval scores rank chunks by topical similarity. Figure labels, tables and reference lists often score high because they repeat key terms. Evaluate retrieval with questions whose answer chunk you know (recall@k, Chapter 12), not by the scores alone.
:::

:::tip[🎯 Interview angle]
**"What does FAISS return, and what are the limits of `IndexFlatIP`?"** It returns the top-k scores and row indices; the rows are mapped back to chunk text and metadata kept outside FAISS. `IndexFlatIP` is exact, so it is accurate but scans every vector, which grows linearly with the corpus; for millions of vectors you move to approximate indexes such as IVF or HNSW, trading a little recall for large speed-ups.
:::

## 9. Step 7: Keyword Search (BM25) and Hybrid Retrieval

Embeddings are good at meaning and bad at **exact strings**. Questions full of rare, specific terms ("How many words are in English Wikipedia?", "Which activation function, GELU?") need a search that rewards matching those exact words. The classic method is **BM25**, a refined TF-IDF score. Each question word contributes more if it is **rare** across chunks (its *inverse document frequency*, IDF, is high), more if it appears **often** in the chunk (with diminishing returns, controlled by $k_1$), and the score is adjusted for **chunk length** so that long chunks don't win just by being long (controlled by $b$). Common words like "the" or "what" contribute almost nothing.

On "How many words are in English Wikipedia?" the difference is stark. The three chunks that contain the answer "2,500M" are chunks 13, 46 and 47. Dense (embedding) search ranks them **4th, 6th and 22nd**, and its top three are 38, 51 and 9. BM25 ranks them exactly **1st, 2nd and 3rd**, because "english" appears in only 2 of the 55 chunks and "wikipedia" in only 5, giving them high IDF. Neither method is always better, so the page combines them with **reciprocal rank fusion (RRF)**: each chunk gets $1/(c + \text{rank})$ from each list, the scores are added, and the sum is sorted. RRF uses only **rank positions**, so it doesn't matter that cosines live around 0.2–0.6 and BM25 scores around 0–15.

<Formula name="BM25" tag="Keyword relevance" words="Sum over the question words: how rare the word is across chunks, times a saturating function of how often it appears in this chunk, adjusted for chunk length." remember="Rare words that match count most.">

$$
\begin{aligned}
\operatorname{BM25}(\data{q}, \data{d}) &= \sum_{w \in \data{q}} \operatorname{IDF}(w) \cdot \operatorname{TF}(w, \data{d}) \\
\operatorname{TF}(w, \data{d}) &= \frac{f(w, \data{d})\,(\hyper{k_1} + 1)}{f(w, \data{d}) + \hyper{k_1}\,\big(1 - \hyper{b} + \hyper{b}\,\lvert \data{d} \rvert / \overline{\lvert d \rvert}\big)} \\
\operatorname{IDF}(w) &= \ln\!\Big(1 + \frac{\data{N} - n_w + 0.5}{n_w + 0.5}\Big)
\end{aligned}
$$

- $f(w, \data{d})$: how many times word $w$ appears in chunk $\data{d}$; $\operatorname{TF}$: its saturating, length-adjusted version
- $\lvert \data{d} \rvert$: chunk length in words; $\overline{\lvert d \rvert}$: average chunk length (217.2 here)
- $\data{N}$: number of chunks; $n_w$: number of chunks containing $w$
- $\hyper{k_1} = 1.5$: how fast repeated matches saturate; $\hyper{b} = 0.75$: how much length is penalised

**Example:** with $\data{N} = 55$ chunks, "english" appears in 2, so $\operatorname{IDF} = \ln(1 + 53.5/2.5) = 3.109$; "wikipedia" appears in 5, so $\operatorname{IDF} = \ln(1 + 50.5/5.5) = 2.321$.

</Formula>

<Formula name="Reciprocal rank fusion (RRF)" tag="Hybrid search" words="Each ranking gives a chunk one over c plus its rank; add the contributions and sort. Chunks ranked well by both lists rise to the top." remember="Fuse ranks, not raw scores.">

$$
\operatorname{RRF}(\data{d}) = \sum_{r \in \{\text{dense},\ \text{BM25}\}} \frac{1}{\hyper{c} + \operatorname{rank}_r(\data{d})}, \qquad \hyper{c} = 60
$$

- $\operatorname{rank}_r(\data{d})$: the chunk's position (1 = best) in ranking $r$
- $\hyper{c}$: a damping constant; 60 is the standard value

**Example:** chunk 46 is dense rank 4 and BM25 rank 2: $1/64 + 1/62 = 0.03175$, hybrid rank 1. Chunk 13 is dense rank 22 but BM25 rank 1: $1/82 + 1/61 = 0.02859$, hybrid rank 4. Chunk 38, dense rank 1, falls to hybrid rank 11 because BM25 doesn't like it.

</Formula>

<Animation src="rag/hybrid-fusion" caption="The real dense and BM25 rankings for How many words are in English Wikipedia. Watch the three answer chunks (green) climb from dense ranks 4, 6 and 22 to hybrid ranks 1, 2 and 4." />

Over the 12 factual test questions (Chapter 12), each with a known answer chunk, hybrid search finds the answer chunk at rank 1 three times out of four, where either method alone manages one time in two:

| Recall@k (12 factual questions, 55 chunks) | @1 | @3 | @5 | @10 |
|---|---|---|---|---|
| Dense (MiniLM + FAISS) | 0.50 | 0.67 | 0.83 | 0.83 |
| BM25 (keywords) | 0.50 | 0.75 | 0.83 | 1.00 |
| **Hybrid (RRF)** | **0.75** | **0.83** | **0.83** | 0.92 |

:::analogy[Two judges, one scoreboard]
Dense search is a judge who scores on overall impression; BM25 is a judge who checks the exact technical details. Each has blind spots. RRF doesn't average their raw marks (they use different scales); it adds points for *placing*, so a chunk both judges rank near the top wins, and one only a single judge loves drops back.
:::

:::tip[🎯 Interview angle]
**"Why use hybrid retrieval, and why fuse with RRF?"** Embeddings capture paraphrase and meaning but miss exact rare terms, numbers and acronyms; BM25 is the reverse. Hybrid retrieval runs both and fuses them. RRF uses only rank positions, so it needs no score normalisation or weight tuning; on this paper it raised recall@1 from 0.50 (either method alone) to 0.75.
:::

## 10. Step 8: Building the Prompt

The generator doesn't search; it only sees what you put in its **prompt**. A RAG prompt has three parts: an **instruction** ("Answer the question using only the context below. If the context does not contain the answer, say: The provided document does not contain enough information to answer this question."), the **question**, and the retrieved **context**, followed by an "Answer:" cue that tells an instruction-tuned model where its output begins. The instruction does two jobs: it limits the model to the document (reducing answers from outside knowledge) and it gives the model a fixed **fallback sentence** for when the context doesn't contain the answer. The notebook's `build_strict_rag_prompt` adds stronger rules than `build_rag_prompt` for exactly that reason.

Building the prompt well turned out to be the hardest part of the notebook, and its failures are worth studying one by one. **First, metadata in the prompt.** The notebook labelled each passage with a header like "[Source 1 | Page 3 | Chunk 6 | Score 0.5080]" so sources could be traced. FLAN-T5-small, asked "What is the full form of BERT?", answered with exactly that header: it **copied the metadata** because it looked like the most answer-shaped text in the prompt. Source information belongs in the app's source panel, **not** in the text the model reads. **Second, characters are not tokens.** The context was capped at `max_context_characters = 3500`, but the number of tokens in 3,500 characters depends on the text (references and tables use far more tokens per character), and the model's limit is in tokens; one notebook prompt was 3,882 characters long.

**Third, a budget that lies.** The notebook's first token-aware prompt printed "Context token budget used: 0" while the prompt actually contained about 360 tokens of passage text (429 tokens in total). The loop adds whole passages while they fit and trims the one that doesn't, but it appended the trimmed passage without adding its tokens to the counter. The top chunk (459 tokens) was already bigger than the 360-token budget, so the entire context became one trimmed chunk, the figure-label noise of Chapter 8, and the counter stayed at 0. **Fourth, a fair budget that keeps the wrong text.** The balanced version gives each of the top 3 passages an equal share (about 100 tokens each) and trims each passage to its **first** 100 tokens. With chunks of 459, 340 and 384 tokens, that keeps roughly a quarter of each. The prompt fits (375 tokens), but the full form of BERT isn't in any of the kept pieces, and flan-t5-base answered just "BERT".

<PromptBudgetLab />

The page's fix is to choose context by **relevance at the sentence level**. The top 5 hybrid chunks are split into sentences, every sentence is embedded, the sentences are sorted by cosine to the question, and the most relevant ones are added until the 360-token budget is used. The chosen sentences are then put back in **reading order** (by chunk rank, then position) so the context reads naturally. For "What is next sentence prediction and why is it used?" this produces a 410-token prompt: 33 tokens of instruction, 13 for the question line, 3 for "Context:", 359 tokens of the most relevant sentences, and 2 for "Answer:". On the 12 factual questions, the same flan-t5-base model went from **1 of 12** right with the notebook's balanced prompt to **5 of 12** with sentence selection.

![Prompt anatomy for the NSP question: instruction 33 tokens, question 13, Context label 3, selected sentences 359, Answer cue 2, total 410 of 512. Kept out of the prompt: page numbers, chunk ids, scores](/img/deep-learning/diagrams/rag-prompt-anatomy.svg)

*How to read it:* each bar segment is a real token count from the FLAN-T5 tokenizer. Almost the whole budget goes to context; the metadata lives outside the prompt and is shown to the user by the app.

| Prompt design | Result on this paper |
|---|---|
| Passages with "[Source … Score …]" headers (flan-t5-small) | Answer was the header itself |
| Character cap 3,500 | Token count unpredictable; one prompt 3,882 chars |
| Token-aware, counter not updated | Printed "used: 0" with ~360 context tokens present |
| Balanced: first ~100 tokens of each top-3 chunk | Fits (375 tokens) but cuts the answer: 1 of 12 facts right |
| **Most relevant sentences, reading order, 360-token budget** | **5 of 12 facts right** |

:::danger[Interview Trap: "more context is always better"]
Too much context can exceed the input limit (and get truncated), slow generation, bury the relevant sentence among irrelevant ones, and give the model more wrong text to copy. Pick the most relevant pieces within a token budget, and keep metadata out of the model's input.
:::

:::analogy[Packing a small suitcase]
The prompt is a cabin bag with a strict weight limit (512 tokens). Packing the first item from every drawer (the first 100 tokens of each chunk) fills the bag fairly but leaves behind what you actually need. Packing by "what will I use on this trip?" (the most relevant sentences) fits the same bag with the right things. And you don't pack the labels from the drawers.
:::

:::tip[🎯 Interview angle]
**"How do you build a RAG prompt for a small model with a 512-token limit?"** Instruction with a fixed fallback answer, the question, then context selected by relevance, all counted in the generator's tokens with a fixed budget. Don't put source metadata in the model's input, since small models copy it; show sources in the UI instead. On this paper, switching from the first tokens of each chunk to the most relevant sentences took factual accuracy from 1 to 5 of 12 with the same model.
:::

## 11. Step 9: Generating the Answer with FLAN-T5

Generation has three steps. The prompt is **tokenized** into input ids (with `truncation=True, max_length=512` as a safety net). `model.generate()` runs the encoder once over the whole prompt, then runs the decoder step by step: at each step it outputs a probability for every one of the 32,128 tokens in the vocabulary, a **decoding strategy** picks the next token, and the step repeats until the end-of-sequence token or `max_new_tokens`. Finally the output ids are **decoded** back into text with `skip_special_tokens=True` and trimmed.

The decoding strategy matters more than it seems. **Greedy decoding** (`num_beams=1`) takes the single most likely token at every step. It is fast, but it can't change its mind: on the NSP question the very first step was nearly a three-way tie, "We" 0.134, "To" 0.113, "Pre" 0.106, and greedy committed to "We", producing "We pre-train for a binarized next sentence prediction task that can be trivially generated from any monolingual corpus" (27 tokens, total log-probability −3.768, an average of −0.1395 per token). That copies the *how* but drops the *why*. **Beam search** (`num_beams=4`) keeps the 4 best partial answers at every step and finally returns the one with the best length-normalised log-probability. It found "To train a model that understands sentence relationships, we pre-train for a binarized next sentence prediction task that can be trivially generated from any monolingual corpus." with an average log-probability of −0.0966 per token, a better score than greedy's −0.1395, and a better answer: it starts with the purpose. On this CPU it took about 7–8 seconds.

<Animation src="rag/token-by-token" caption="FLAN-T5-base answering the NSP question greedily, one token per step, with the three most likely candidates at each step. Step 1 is a near tie that greedy decides once and never revisits." />

<Formula name="Beam search score" tag="Choosing among candidates" words="Each candidate answer is scored by its total log-probability divided by its length to the power alpha; beam search keeps the best B partial answers at each step and returns the best finished one." remember="Keep B candidates, pick best average.">

$$
\operatorname{score}(\pred{y}) = \frac{1}{|\pred{y}|^{\hyper{\alpha}}} \sum_{t=1}^{|\pred{y}|} \log P_{\param{\theta}}(\pred{y_t} \mid \pred{y_{\lt t}}, \data{x})
$$

- $|\pred{y}|$: answer length in tokens; $\hyper{\alpha}$: `length_penalty` (1.0 by default, so the score is the average log-probability per token)
- $\hyper{B}$: `num_beams`, the number of candidates kept at each step (1 = greedy)

**Example:** greedy's 27-token answer has sum −3.768, so its average is −0.1395; the 4-beam answer scores −0.0966, so beam search prefers it.

</Formula>

A few `generate()` settings control the rest of the behaviour, and two of them changed answers on this page. **`early_stopping`** decides when beam search stops: with `True` it stops as soon as 4 finished candidates exist, and on the NSP question that returned the short "to train a model that understands sentence relationships" (score −0.2685, clearly worse); with `False`, which the final pipeline uses, it keeps searching while a better candidate is still possible. **`no_repeat_ngram_size=3`** forbids repeating any 3-token sequence, which stops small models from looping. `do_sample=False` makes the output deterministic (no random sampling), which is what you want when answers must be reproducible and checkable.

| Setting | Value here | Effect |
|---|---|---|
| `num_beams` | 4 (1 = greedy) | More candidates explored; slower; often more complete answers |
| `early_stopping` | `False` | Don't stop at the first 4 finished beams; True gave a worse answer here |
| `no_repeat_ngram_size` | 3 | No repeated 3-token phrases (stops loops) |
| `max_new_tokens` | 80 | Upper bound on answer length |
| `do_sample` | `False` | Deterministic output; no temperature randomness |
| `max_length` (input) | 512 | Truncate the prompt as a safety net |

:::analogy[Choosing a route turn by turn vs keeping options open]
Greedy decoding is driving by always taking the turn that looks best right now; one early wrong turn and you're stuck on a worse road. Beam search is a navigator who keeps the four most promising routes in mind at every junction and only at the end picks the one with the best overall time.
:::

:::tip[🎯 Interview angle]
**"Greedy vs beam search for RAG answers?"** Greedy picks the most likely token each step and can't revisit an early choice; beam search keeps B partial answers and returns the one with the best length-normalised log-probability. On the NSP question, greedy's first step was a near tie (0.134 vs 0.113) and it dropped the "why"; 4 beams found a complete answer with a better per-token score (−0.0966 vs −0.1395). Keep `do_sample=False` for factual RAG, and check `early_stopping`: here `True` returned a worse, shorter answer.
:::

## 12. Measuring It: Where Do Answers Go Wrong?

A RAG demo that answers three hand-picked questions proves nothing. The notebook's own questions ("What is the main idea of this document?") have no single right answer, so this page adds a small test set: **12 factual questions** whose answers appear verbatim in the paper (each checked with a pattern, so "2,500M" counts but "22,500M" doesn't), **5 explanation questions** judged by reading, and **2 questions the paper can't answer**. The value of a test set is not the score itself but what it lets you do: for every wrong answer, find **which stage** failed.

The idea is to follow the answer back through the pipeline. If the chunk containing the answer isn't in the top 5 retrieved, it's a **retrieval failure** and nothing after retrieval can fix it. If it was retrieved but the answer sentence didn't make it into the prompt, it's a **prompt failure**. If the answer was in the prompt and the model still got it wrong, it's a **generation failure** (or a reader failure, for the extractive model). On this paper, the measurements give a clear picture:

| # | Factual question | Answer chunk's hybrid rank | Generator (FLAN-T5-base) | Extractive reader (DistilBERT) |
|---|---|---|---|---|
| 0 | What does BERT stand for? | 10 (not retrieved) | ✗ "BERT" | ✗ "OpenAI GPT" |
| 1 | What percentage of input tokens are masked? | 1 | ✓ 15% | ✓ 15% |
| 2 | How many layers does BERT-large have? | 1 | ✓ "L=24, H=1024, A=16, …" | ✗ "4173" |
| 3 | Which corpus besides English Wikipedia? | 1 | ✓ BooksCorpus | ✓ BooksCorpus |
| 4 | How many words are in English Wikipedia? | 1 | ✗ "22,500M" | ✓ 2,500M |
| 5 | What GLUE score does BERT obtain? | 2 | ✗ "8.6% absolute accuracy improvement" | ✓ 80.5 |
| 6 | Which task predicts whether sentence B follows A? | 1 | ✗ a sentence about the 50% split | ✗ "Sentence" |
| 7 | How many parameters does BERT-base have? | 1 | ✓ "Total Parameters=110M" | ✗ "4173" |
| 8 | Which activation function does BERT use? | 11 (not retrieved) | ✗ "self-attention" | ✗ "bidirectional self-attention" |
| 9 | What is the size of the WordPiece vocabulary? | 1 | ✓ 30,000 | ✓ 30,000 |
| 10 | What learning rate is used for pre-training? | 1 | ✗ Adam's β values | ✗ 5e-5 (the fine-tuning rate) |
| 11 | Which dataset has 100k crowdsourced QA pairs? | 1 | ✗ "MNLI …" | ✗ "MNLI …" |
| | **Right** | 10 of 12 retrieved | **5 of 12** | **5 of 12** |

Read the table by stage. **Retrieval** failed on 2 questions (0 and 8): their answer chunks ranked 10th and 11th, outside the top 5, so both models were doomed. The full form of BERT sits in the very first chunk of the paper, mixed with the title, author names and abstract, and that chunk doesn't look like an answer to either search. On the 10 questions where the answer was in the prompt, the **generator** was right 5 times and the **extractive reader** 5 times, but **not the same 5**: together they cover 7 questions (1, 2, 3, 4, 5, 7, 9). The reader's two "4173" answers are the paper's page-header number, extracted as if it were a count of layers; the generator's "22,500M" is a number that appears nowhere in the paper. For comparison, the notebook's own configuration (dense top-3, balanced first-tokens prompt) got **1 of 12** with the generator.

| Failure point | Example on this paper | First fix |
|---|---|---|
| PDF extraction | Figure labels become text ("Tok 1... Tok N") | Better parser, drop figure regions |
| Cleaning | `[CLS]`, `[SEP]`, `[MASK]` deleted from sentences | Don't delete document words |
| Chunking | Full form buried in the title/author chunk | Section-aware chunking |
| Retrieval | Questions 0 and 8: answer chunk ranked 10th, 11th | Hybrid search, better chunks, reranker |
| Prompt | First-tokens budget cut the answer out (1 of 12) | Sentence-level selection (5 of 12) |
| Generation | "22,500M", Adam's β for "learning rate" | Extractive route for facts, grounding check |
| Grounding | An invented number presented as fact | Support check, refusal |
| Routing | A "How many…" question sent to the generator | Route exact-fact questions to the extractor |

:::danger[Interview Trap: "the LLM got it wrong"]
Most wrong RAG answers here are not the model's fault: two were retrieval misses that no model could answer. The notebook's own lesson: ask "where did the pipeline fail?", not "why did the LLM fail?". Measure recall@k for retrieval separately from answer accuracy.
:::

:::tip[🎯 Interview angle]
**"How do you evaluate a RAG system?"** Build a small set of questions with known answers and known source chunks. Measure retrieval on its own (recall@k: is the answer chunk in the top k?), then answer accuracy given the retrieved context, and attribute every failure to a stage: extraction, cleaning, chunking, retrieval, prompt, generation, grounding or routing. On this paper that showed 2 of 12 failures were retrieval misses and that the generator and the extractive reader fail on different questions, which is what justifies routing.
:::

## 13. Routing: The Right Answerer for Each Question

The measurements suggest the obvious design: don't use one answering method for everything. **Exact-fact** questions (a full form, a number, a dataset name, a percentage) go to the **extractive reader**, which copies the answer exactly; **explanation, summary and comparison** questions go to the **generator**, which can combine and rephrase; and questions the document can't answer get the **refusal** sentence. Choosing the route before answering is called **query routing**.

The notebook proposes starting with **rules**, and that is what this page implements: two regular expressions over the question. An `EXACT` pattern ("how many", "how much", "what percentage", "which", "stand for", "full form", "what is the size", "what value") sends the question to the extractive route; otherwise an `EXPLAIN` pattern ("why", "how", "explain", "summarise", "compare", "differ", "describe") sends it to the generator; anything else defaults to extractive. The **order** matters. A first, naive version checked only for explanation words, so "**How** many words are in English Wikipedia?" matched "how" and went to the generator, which answered "22,500M". Checking the exact-fact patterns first fixes it.

How much does routing buy? With these rules, all 12 factual questions go to the extractive reader, so the router scores the reader's **5 of 12** on facts, and all 5 explanation questions go to the generator, where the extractive reader would have returned fragments such as "conceptually simple and empirically powerful" for "Why is BERT called bidirectional?". Routing doesn't reach the **7 of 12** that a perfect choice between the two models would get on facts, because the rules decide by question wording, not by which model happens to be right. Better routers, in the notebook's order of sophistication, are: rules → similarity to example questions of each type → a small trained classifier → an LLM that classifies the question.

| Question type | Example | Route |
|---|---|---|
| Exact fact (full form, value, name) | What is the full form of BERT? | Extractive |
| Number, percentage, size | How many words are in English Wikipedia? | Extractive |
| Definition | Define masked language modeling. | Extractive or short generative |
| Explanation | Why is BERT bidirectional? | Generative |
| Summary | Summarize the main contribution. | Generative |
| Comparison | How does BERT differ from OpenAI GPT? | Generative |
| Not in the document | Who won the 2018 FIFA World Cup? | Refuse |

:::tip[🎯 Interview angle]
**"How would you route questions in a document chatbot?"** Classify each question before answering: exact facts to an extractive reader, explanations, summaries and comparisons to a generator, and unsupported questions to a refusal. Start with simple rules and test them: here the order of the rules mattered, since "How many…" contains "how" and was first misrouted to the generator. Move to a trained classifier or an LLM router when rules stop scaling.
:::

## 14. Grounding and Refusal: Knowing When Not to Answer

A fluent answer is not a correct one. In document QnA an answer is **grounded** when it can be traced back to the retrieved text, and the system should check that before showing it. The notebook frames this as a support status (fully supported, partially supported, weakly supported, insufficient context); this page implements the two simplest checks that give that status a number.

The first check happens **before** answering: if the best cosine between the question and any chunk is below a threshold (0.2 here), nothing in the document is about the question, so the system **refuses** without calling any model. "Who won the 2018 FIFA World Cup?" has a best cosine of only 0.1408 and is refused; left to answer, the generator said "United States" with beam search and "Brazil" with greedy decoding, both invented. The lowest best-cosine among the 12 real factual questions is 0.3669, so the gate doesn't block any of them. The second check happens **after** generating: the **support score** is the fraction of the answer's content words (ignoring words like "the" and "of") that also appear in the context the model was given. "22,500M" has support 0.0, since that token isn't in the context, so the system refuses instead of showing an invented number.

<Formula name="Support score" tag="A simple grounding check" words="The share of the answer's content words that also occur in the context given to the model. Answers built from words the context never contained score low and are refused." remember="Every answer word should be in the context.">

$$
\begin{aligned}
\pred{A} &= \operatorname{words}(\pred{a}) \setminus \text{stop} \\
\operatorname{support}(\pred{a}, \data{C}) &= \frac{\lvert \{\, w \in \pred{A} : w \in \operatorname{words}(\data{C}) \,\} \rvert}{\lvert \pred{A} \rvert}
\end{aligned}
$$

- $\pred{a}$: the generated answer; $\data{C}$: the context in the prompt
- $\pred{A}$: the answer's content words; stop: very common words (the, of, is, what…) that say nothing about grounding
- refuse when $\operatorname{support} \lt \hyper{\tau}$, with $\hyper{\tau} = 0.5$ here

**Example:** "22,500M" has one content word, not in the context: support $0/1 = 0.0$, refused. "8.6% absolute accuracy improvement" has 3 of its 4 content words in the context: support $0.75$, accepted, though it is the wrong answer to "What GLUE score does BERT obtain?".

</Formula>

The example shows the limit of word-overlap checks: they catch **invented** content, but not **wrong answers assembled from document words**. "MNLI Multi-Genre Natural Language Inference" (support 1.00) is a real phrase from the paper and the wrong dataset. Stronger checks compare each answer sentence with the evidence using a **natural-language-inference** model or an LLM judge. The other limit is the extractive route: a reader fine-tuned on **SQuAD v1.1** never saw unanswerable questions in training, so it **always returns some span**. Asked "What GPU was used to train BERT?" (the paper pre-trained on Cloud TPUs and only remarks that fine-tuning takes a few hours on a GPU, without naming one), it answered "BooksCorpus" with a span score of 13.03, while the generator, given the same retrieved chunks, correctly replied that the document doesn't contain the information. A reader trained on **SQuAD 2.0**, which includes unanswerable questions and a no-answer option, would be the fix.

<Animation src="rag/route-and-refuse" caption="Three real questions through the same pipeline. The corpus question is extracted from page 5, the NSP question is generated with support 1.00, and the FIFA question is refused at the cosine gate, 0.141 below 0.2." />

<RagSystemLab />

:::analogy[A fact-checker at the door]
The cosine gate is a receptionist who says "we don't handle that here" before anyone does any work. The support check is a fact-checker who reads the draft answer and circles every word that isn't in the source notes; too many circles and the answer doesn't go out. But the fact-checker only checks *words*, so a sentence copied from the wrong paragraph still gets through.
:::

:::tip[🎯 Interview angle]
**"How do you reduce hallucination in RAG?"** Tell the model to answer only from the context with a fixed fallback, then verify: refuse when retrieval is weak (best cosine below a threshold) and when the answer isn't supported by the context (here, the share of answer words present in the context). Word-overlap checks catch invented content like "22,500M" but not wrong-but-document-worded answers, so stronger systems use an NLI or LLM judge per claim. And use a reader trained with unanswerable questions (SQuAD 2.0) if the extractive route must be able to say "not found".
:::

## 15. The Whole System, and a Streamlit App

Put together, the system has two phases. **Indexing**, once per document: read the PDF, clean, chunk into sentence-aware chunks with page numbers, embed the chunks into a FAISS index and build the BM25 statistics. **Answering**, once per question: check the best cosine (refuse if too low), retrieve the top 5 chunks by hybrid search, route the question; on the extractive route run the reader over the top 3 chunks and return the best span with its page; on the generative route build the sentence-selected prompt, generate with beam search, and refuse if the support score is too low. Every result carries the **route**, a short **reason** (span score and page, or support score) and the **sources** (chunk id and page) for display. The function `rag_answer` in the implementation section is exactly this.

**Streamlit** turns that function into a web app with a few lines of Python. The notebook's proposed app has a PDF upload or path, a question box, an answer box, a source panel, the route used, a grounding status and a debug toggle. The one Streamlit-specific idea that matters is **caching**: Streamlit re-runs the whole script on every interaction, so without care it would re-read the PDF, re-embed every chunk and reload the models on each question. Decorating the loading function with `@st.cache_resource` builds the chunks and indexes once and reuses them. Tested with Streamlit's `AppTest`, the app answers "Which corpus is used for pre-training besides English Wikipedia?" with "BooksCorpus" and the caption "Route: extractive · span score 23.91, page 5", showing 5 source expanders, and refuses the FIFA question with "Route: refused · top cosine 0.141".

| App part | What it shows | Why |
|---|---|---|
| PDF input | Path or upload | The document to index |
| Question box | Free text | The user's question |
| Answer box | Answer or refusal sentence | The result |
| Route + reason | "extractive · span score 23.91, page 5" | Transparency, debugging |
| Source panel | Page and chunk text for each retrieved chunk | Lets the user verify |
| Cached indexes | `@st.cache_resource` | Index once, not on every click |

:::tip[🎯 Interview angle]
**"Walk me through your RAG system end to end."** Offline: parse the PDF page by page, clean carefully, chunk by sentences with page metadata, embed with a sentence-transformer into a FAISS inner-product index, and build BM25 statistics. Online: gate on the best cosine, retrieve with hybrid search (RRF), route the question, extract with a QnA reader or generate with FLAN-T5 from a token-budgeted, sentence-selected prompt, check support, and return the answer with its route and sources. In a Streamlit app, cache the indexes with `@st.cache_resource`.
:::

## 16. When Does Fine-Tuning Help?

When a RAG system gives poor answers, "fine-tune the model" is the tempting fix, and usually the wrong first one. Fine-tuning changes a model's **behaviour**: its style, its output format, its skill at a specific task, its understanding of domain vocabulary. It doesn't fix missing or wrong **context**. On this page, two of twelve failures were retrieval misses, several were prompt problems, and one was a cleaning bug; none of those would be fixed by training the generator.

The notebook's rule of thumb is to fix the pipeline first and fine-tune only the component that the evaluation shows is weak. A RAG system has several fine-tunable parts: the **embedding model** (so domain questions land near domain chunks), a **reranker** (a cross-encoder that re-scores the top chunks more accurately), the **router** (a small classifier trained on labelled question types), the **generator** (for answer style, format and following the context), and the **extractive reader** (for domain-specific spans, or on SQuAD 2.0 data so it can refuse). Each needs its own data: question–relevant-chunk pairs for the embedder and reranker, labelled questions for the router, and question–context–answer triples for the generator and reader.

| Problem | Fine-tune first? | Better first fix |
|---|---|---|
| Wrong chunks retrieved | No | Improve retrieval (hybrid, chunking, reranker) |
| PDF text noisy | No | Better parsing and cleaning |
| Answer not in the document | No | Retrieve better, or refuse |
| Prompt too long / truncated | No | Token-aware prompt building |
| Exact facts answered poorly | No | Extractive route, hybrid retrieval |
| Answer style unsuitable | Maybe | Instruction fine-tuning of the generator |
| Domain language misunderstood | Maybe | Domain-adapted embeddings |
| Consistent output format needed | Yes, possibly | Instruction fine-tuning |

The roadmap is the same evaluation loop as Chapter 12: build a strong baseline, write evaluation questions, find the main weakness, fine-tune only that component, and **measure before and after** on the same questions.

:::tip[🎯 Interview angle]
**"Your RAG answers are poor. Do you fine-tune the LLM?"** Not first. Evaluate stage by stage: if the answer chunk isn't retrieved, fix chunking or retrieval (or fine-tune the embedder or add a reranker); if it's retrieved but cut from the prompt, fix prompt building; if the context is right and the answers are still poor or badly formatted, then fine-tune the generator. Fine-tuning changes behaviour, not knowledge of what's in your document; that's retrieval's job.
:::

## ⚙️ Going Deeper (RAG Internals)

**Where the time goes.** Measured on a CPU over the notebook's chunks (one run each): embedding a question 0.012 s, hybrid retrieval 0.011 s, building the sentence-selected prompt 1.12 s (it embeds every sentence of the top 5 chunks), generation 1.67 s with `early_stopping=True` (the final `early_stopping=False` setting took about 7–8 s for the 30-token NSP answer), and the extractive reader 0.42 s. Retrieval is essentially free; the models dominate. Sentence embeddings for every chunk could be computed once at indexing time to remove most of the prompt-building cost.

**Encoder-decoder inference.** FLAN-T5's encoder runs **once** over the whole prompt; the decoder then runs once per generated token, attending to its own previous tokens (causal self-attention, with cached keys and values) and to all encoder outputs (cross-attention). So a long prompt costs one big encoder pass, and a long answer costs many small decoder passes. Beam search multiplies the decoder work by the number of beams.

**Embedding truncation.** `all-MiniLM-L6-v2` has `max_seq_length = 256`; longer inputs are cut, and the vector describes only the start of the chunk. With 48 of 55 chunks over the limit, dense search is blind to the end of most chunks. Remedies: smaller chunks (by tokens), embedding sentences or overlapping windows and pooling, or a long-context embedding model.

**Bi-encoder vs cross-encoder.** The embedder is a **bi-encoder**: question and chunk are encoded separately, so chunk vectors are computed once and search is a dot product. A **cross-encoder reranker** reads question and chunk *together* through one Transformer and outputs a relevance score; it is far more accurate and far slower, so it is used only to re-order the top 20–50 retrieved chunks. Retrieval misses like questions 0 and 8 (answer chunks at hybrid ranks 10 and 11) are exactly what a reranker over a top-20 list can recover.

**BM25 details.** The keyword tokenizer keeps numbers and their punctuation together (`2,500m`, `15%`, `1e-4` stay single tokens), lower-cases text and skips stop words in the query. IDF uses the "+1 inside the log" form, which keeps IDF positive even for words in more than half the chunks. $k_1 = 1.5$ means a word's contribution saturates after a few occurrences; $b = 0.75$ scales the length penalty.

**Why RRF works without tuning.** RRF ignores score scales entirely; with $c = 60$, the difference between rank 1 and rank 2 ($1/61$ vs $1/62$) is small, so a chunk needs support from **both** lists to win. Weighted score fusion ($\alpha \cdot$ dense $+ (1-\alpha) \cdot$ BM25) can work too, but needs score normalisation and a tuned $\alpha$, and min-max normalisation exaggerates tiny gaps.

**Special tokens as text.** Because the cleaned text keeps literal `[CLS]` and `[SEP]`, the extractive reader's tokenizer must not turn them into its real special tokens. Without `split_special_tokens=True`, the string "[CLS]" in the document becomes the actual classification token, and the reader returned "[CLS]" as an answer. With it, the reader tokenizes "[CLS]" as ordinary characters.

**Extractive span search, masked.** The reader scores every (start, end) pair as start logit + end logit, but only pairs inside the **context** tokens, with end ≥ start and length below 40 tokens; the question and padding positions are masked out. `offset_mapping` maps the winning token pair back to character positions in the original text, so the returned span is exact. Long chunks are split into 384-token windows with a 128-token stride, and the best span over all windows wins.

**What the support score can't see.** Word overlap is order-blind and claim-blind: it can't tell "BERT outperforms GPT" from "GPT outperforms BERT", and it can't tell the right number from another number in the same context. Claim-level checks use **NLI** (does the context *entail* this sentence?) or an LLM judge, at the cost of another model call.

## 💻 Python Implementation

:::tip[Run it yourself]
The final pipeline is one file, `rag_pipeline.py`, shown block by block below; each block's outputs come from running it on the BERT paper (`17 Bert_Model.pdf`) on a CPU with transformers 5.22.2, sentence-transformers, faiss-cpu and PyMuPDF. `pip install pymupdf sentence-transformers faiss-cpu transformers torch streamlit`.
:::

### 1. Read the PDF page by page

PyMuPDF opens the file; each page's text is stored with its 1-based page number.

```python title="rag_pipeline.py: block 1" showLineNumbers
import re, math, unicodedata
from collections import Counter
import fitz  # PyMuPDF

def read_pdf(path):
    doc = fitz.open(path)
    # highlight-next-line
    pages = [{"page": i + 1, "text": page.get_text()} for i, page in enumerate(doc)]
    doc.close()
    return pages

pages = read_pdf("17 Bert_Model.pdf")
print(len(pages), [len(p["text"].split()) for p in pages[:4]])   # 16 [584, 669, 583, 811]
```

### 2. Clean the text without deleting words

Targeted rules only. The notebook's version also deleted `[CLS]`, `[SEP]` and `[MASK]`; this one keeps them because the paper uses them as words.

```python title="rag_pipeline.py: block 2" showLineNumbers
def clean_text(text):
    text = unicodedata.normalize("NFKC", text or "")                  # "ﬁ" -> "fi", odd spaces -> normal
    text = text.replace("\u200b", " ").replace("\ufeff", " ")         # zero-width space, byte-order mark
    # highlight-next-line
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)                      # "repre-\nsentation" -> "representation"
    text = re.sub(r"\s+", " ", text)                                  # newlines and runs of spaces -> one space
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)                      # "word ." -> "word."
    return text.strip()

raw = " ".join(p["text"] for p in pages)
print(raw.count("[CLS]"), raw.count("[SEP]"))                         # 28 25
print(clean_text("BERT is de-\nsigned to pre-\ntrain deep ﬁne-tuned  models ."))
# BERT is designed to pretrain deep fine-tuned models.
```

```python title="❌ The notebook's cleaner deletes the paper's own words" showLineNumbers
for token in ["[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"]:
    # error-next-line
    cleaned = cleaned.replace(token, " ")    # ❌ "[CLS] is a special symbol..." -> " is a special symbol..."
```

### 3. Split into sentences and pack them into chunks

Abbreviations are protected before splitting; chunks hold whole sentences, up to 250 words, with a 2-sentence overlap and the page number.

```python title="rag_pipeline.py: block 3" showLineNumbers
ABBREV = {"e.g.": "e<DOT>g<DOT>", "i.e.": "i<DOT>e<DOT>", "Fig.": "Fig<DOT>", "et al.": "et al<DOT>", "vs.": "vs<DOT>"}

def split_sentences(text):
    for a, b in ABBREV.items():
        text = text.replace(a, b)
    # highlight-next-line
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\(\[])', text)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]

def make_chunks(pages, max_words=250, overlap_sentences=2):
    chunks = []
    for p in pages:
        current, n_words = [], 0
        for s in split_sentences(clean_text(p["text"])):
            w = len(s.split())
            if current and n_words + w > max_words:                   # chunk is full: close it
                chunks.append({"id": len(chunks), "page": p["page"], "text": " ".join(current)})
                # highlight-next-line
                current = current[-overlap_sentences:]                 # carry the last 2 sentences over
                n_words = sum(len(x.split()) for x in current)
                while current and n_words + w > max_words:
                    n_words -= len(current.pop(0).split())
            current.append(s); n_words += w
        if current:
            chunks.append({"id": len(chunks), "page": p["page"], "text": " ".join(current)})
    return chunks

print(split_sentences("Transformers use self-attention. Fig. 1 shows the architecture, e.g. the encoder. It removes recurrence."))
# ['Transformers use self-attention.', 'Fig. 1 shows the architecture, e.g. the encoder.', 'It removes recurrence.']
chunks = make_chunks(pages)
wc = [len(c["text"].split()) for c in chunks]
print(len(chunks), min(wc), max(wc), round(sum(wc) / len(wc), 2))    # 55 44 249 216.71
```

### 4. Embed the chunks and build a FAISS index

Normalized float32 vectors in an inner-product index, so the scores are cosines.

```python title="rag_pipeline.py: block 4" showLineNumbers
import numpy as np, faiss
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

def build_index(chunks):
    # highlight-next-line
    vectors = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])                      # inner product = cosine for unit vectors
    index.add(vectors)
    return index

def dense_search(index, question, k=5):
    q = embedder.encode([question], normalize_embeddings=True).astype("float32")
    scores, rows = index.search(q, k)
    return [(int(r), float(s)) for r, s in zip(rows[0], scores[0]) if r != -1]

index = build_index(chunks)
print(index.ntotal, index.d)                                         # 55 384
print(dense_search(index, "What does BERT stand for?", 3))
# [(54, 0.47192516922950745), (9, 0.454640656709671), (8, 0.44709596037864685)]
lens = [len(embedder.tokenizer(c["text"])["input_ids"]) for c in chunks]
print(sum(l > 256 for l in lens), "of", len(chunks))                 # 48 of 55 (longer than MiniLM's 256)
```

### 5. Keyword search (BM25) and hybrid fusion (RRF)

BM25 from scratch in a few lines, then reciprocal rank fusion of the full dense and BM25 rankings.

```python title="rag_pipeline.py: block 5" showLineNumbers
WORD = re.compile(r"[a-z0-9]+(?:[.,%-][a-z0-9]+)*%?")
STOP = set("a an the of to in and or is are was were be by for on with as at from that this which what how why does do did it its".split())

def words(text):
    return WORD.findall(text.lower())

class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.docs = [Counter(words(c["text"])) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / len(self.lens)
        self.df = Counter(w for d in self.docs for w in d)
        self.k1, self.b, self.n = k1, b, len(chunks)

    def scores(self, question):
        s = np.zeros(self.n)
        for w in [w for w in words(question) if w not in STOP]:
            if w not in self.df:
                continue
            idf = math.log(1 + (self.n - self.df[w] + 0.5) / (self.df[w] + 0.5))
            for i, d in enumerate(self.docs):
                f = d[w]
                if f:
                    s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
        return s

def hybrid_search(index, bm25, question, k=5, c=60):
    n = index.ntotal
    dense_order = [r for r, _ in dense_search(index, question, n)]
    bm25_order = list(np.argsort(-bm25.scores(question)))
    fused = Counter()
    for rank, row in enumerate(dense_order):
        # highlight-next-line
        fused[row] += 1 / (c + rank + 1)                              # reciprocal rank fusion
    for rank, row in enumerate(bm25_order):
        fused[int(row)] += 1 / (c + rank + 1)
    return [row for row, _ in fused.most_common(k)]

bm25 = BM25(chunks)
q = "How many words are in English Wikipedia?"
print([r for r, _ in dense_search(index, q, 3)])                     # [38, 51, 9]   no answer chunk
print([int(i) for i in np.argsort(-bm25.scores(q))[:3]])             # [13, 46, 47]  all three answer chunks
print(hybrid_search(index, bm25, q, 3))                              # [46, 47, 50]
```

### 6. Build the prompt from the most relevant sentences

No metadata in the prompt; sentences chosen by cosine to the question within a 360-token budget, then put back in reading order.

```python title="rag_pipeline.py: block 6" showLineNumbers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

gen_name = "google/flan-t5-base"
gen_tok = AutoTokenizer.from_pretrained(gen_name)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_name).eval()

INSTRUCTION = ("Answer the question using only the context below. If the context does not contain the answer, "
               "say: The provided document does not contain enough information to answer this question.")

def n_tokens(text):
    return len(gen_tok(text, add_special_tokens=False)["input_ids"])

def build_prompt(question, chunks, rows, budget=360):
    sents = [(r, i, s) for r in rows for i, s in enumerate(split_sentences(chunks[r]["text"])) if len(s.split()) >= 3]
    S = embedder.encode([s for _, _, s in sents], normalize_embeddings=True)
    q = embedder.encode([question], normalize_embeddings=True)[0]
    picked, used = [], 0
    # highlight-next-line
    for j in np.argsort(-(S @ q)):                                    # most relevant sentences first
        t = n_tokens(sents[j][2])
        if used + t <= budget:
            picked.append(j); used += t
    picked.sort(key=lambda j: (rows.index(sents[j][0]), sents[j][1]))  # back into reading order
    context = " ".join(sents[j][2] for j in picked)
    return f"{INSTRUCTION}\n\nQuestion: {question}\n\nContext: {context}\n\nAnswer:", context

q = "What is next sentence prediction and why is it used?"
rows = hybrid_search(index, bm25, q, 5)
prompt, context = build_prompt(q, chunks, rows)
print(rows, n_tokens(prompt), n_tokens(context))                     # [12, 43, 9, 19, 3] 410 359
```

```python title="❌ The notebook's passage headers end up in the answer" showLineNumbers
source_header = f"[Source {chunk['rank']} | Page {chunk['page_number']} | Chunk {chunk['chunk_id']} | Score {chunk['retrieval_score']:.4f}]"
# error-next-line
block = source_header + "\n" + chunk_text     # ❌ metadata inside the text the model reads
# flan-t5-small, "What is the full form of BERT?" -> "[Source 1 | Page 3 | Chunk 6 | Score 0.5080]"
```

### 7. Generate with FLAN-T5

Beam search with 4 beams, no early stopping, no repeated 3-grams, deterministic.

```python title="rag_pipeline.py: block 7" showLineNumbers
def generate(prompt, beams=4, max_new_tokens=80):
    inputs = gen_tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        # highlight-start
        out = gen_model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=beams,
                                 do_sample=False, early_stopping=False, no_repeat_ngram_size=3)
        # highlight-end
    return gen_tok.decode(out[0], skip_special_tokens=True).strip()

print(generate(prompt))              # about 7-8 s on this CPU
# To train a model that understands sentence relationships, we pre-train for a binarized next sentence
# prediction task that can be trivially generated from any monolingual corpus.
print(generate(prompt, beams=1))     # greedy: the "why" is lost
# We pre-train for a binarized next sentence prediction task that can be trivially generated from any monolingual corpus
```

### 8. The extractive reader, for exact facts

DistilBERT fine-tuned on SQuAD; the span search is restricted to context tokens; `split_special_tokens=True` keeps the paper's literal "[CLS]" from becoming a real special token.

```python title="rag_pipeline.py: block 8" showLineNumbers
from transformers import AutoModelForQuestionAnswering

qa_tok = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad").eval()

def extract(question, context, max_len=40):
    # highlight-start
    enc = qa_tok(question, context, truncation="only_second", max_length=384, stride=128, split_special_tokens=True,
                 return_overflowing_tokens=True, return_offsets_mapping=True, padding=True, return_tensors="pt")
    # highlight-end
    offsets = enc.pop("offset_mapping"); enc.pop("overflow_to_sample_mapping")
    with torch.no_grad():
        out = qa_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    best = (float("-inf"), "")
    for w in range(enc["input_ids"].shape[0]):                       # one row per 384-token window
        s, e = out.start_logits[w], out.end_logits[w]
        ctx = torch.tensor([x == 1 for x in enc.sequence_ids(w)])
        i = torch.arange(len(s))
        ok = ctx[:, None] & ctx[None, :] & (i[None, :] >= i[:, None]) & (i[None, :] - i[:, None] < max_len)
        score = (s[:, None] + e[None, :]).masked_fill(~ok, float("-inf"))
        a, b = divmod(score.argmax().item(), len(s))
        if score[a, b] > best[0]:
            best = (score[a, b].item(), context[offsets[w][a][0]:offsets[w][b][1]])
    return best[1], round(best[0], 2)

q = "Which corpus is used for pre-training besides English Wikipedia?"
print(extract(q, chunks[hybrid_search(index, bm25, q, 1)[0]]["text"]))   # ('BooksCorpus', 13.86)
```

### 9. Route, ground, refuse: the whole system

The cosine gate, the rule-based router, the support check and the result with its sources.

```python title="rag_pipeline.py: block 9" showLineNumbers
EXACT = re.compile(r"\b(how many|how much|what percentage|which|stand for|full form|what is the size|what value)\b", re.I)
EXPLAIN = re.compile(r"\b(why|how|explain|summari[sz]e|compare|differ|describe)\b", re.I)
REFUSAL = "The provided document does not contain enough information to answer this question."

def route(question):
    # highlight-next-line
    if EXACT.search(question):                                        # check exact facts first: "How many" contains "how"
        return "extractive"
    return "generative" if EXPLAIN.search(question) else "extractive"

def support(answer, context):
    a = [w for w in words(answer) if w not in STOP]
    c = set(words(context))
    return sum(w in c for w in a) / len(a) if a else 0.0

def rag_answer(question, chunks, index, bm25, k=5, min_cosine=0.2, min_support=0.5):
    best_cosine = dense_search(index, question, 1)[0][1]
    rows = hybrid_search(index, bm25, question, k)
    sources = [(chunks[r]["id"], chunks[r]["page"]) for r in rows]
    # highlight-next-line
    if best_cosine < min_cosine:                                      # nothing in the document is about this
        return {"route": "refused", "answer": REFUSAL, "why": f"top cosine {best_cosine:.3f}", "sources": sources}
    if route(question) == "extractive":
        cands = [extract(question, chunks[r]["text"]) + (chunks[r]["page"],) for r in rows[:3]]
        ans, score, page = max(cands, key=lambda x: x[1])
        return {"route": "extractive", "answer": ans, "why": f"span score {score}, page {page}", "sources": sources}
    prompt, context = build_prompt(question, chunks, rows)
    ans = generate(prompt)
    sup = support(ans, context)
    # highlight-next-line
    if sup < min_support:                                             # answer uses words the context never had
        return {"route": "refused", "answer": REFUSAL, "why": f"support {sup:.2f}: {ans!r}", "sources": sources}
    return {"route": "generative", "answer": ans, "why": f"support {sup:.2f}", "sources": sources}
```

The same call on five kinds of question (outputs trimmed to route, answer and reason):

| Question | Route | Answer | Reason |
|---|---|---|---|
| Which corpus is used for pre-training besides English Wikipedia? | extractive | BooksCorpus | span score 23.91, page 5 |
| What is next sentence prediction and why is it used? | generative | To train a model that understands sentence relationships, we pre-train for a binarized next sentence prediction task… | support 1.00 |
| How does BERT differ from OpenAI GPT? | generative | BERT uses a bidirectional Transformer | support 1.00 |
| Who won the 2018 FIFA World Cup? | refused | (refusal sentence) | top cosine 0.141 |
| What GPU was used to train BERT? | extractive | ❌ BooksCorpus | span score 13.03, page 14: a SQuAD v1.1 reader can't refuse |

### 10. The Streamlit app

`@st.cache_resource` builds the chunks and indexes once per PDF path instead of on every interaction. Run with `streamlit run app.py`.

```python title="app.py" showLineNumbers
import streamlit as st
from rag_pipeline import read_pdf, make_chunks, build_index, BM25, rag_answer

st.title("Ask the BERT paper")

# highlight-next-line
@st.cache_resource
def load(path):
    chunks = make_chunks(read_pdf(path))
    return chunks, build_index(chunks), BM25(chunks)

pdf_path = st.text_input("PDF path", "17 Bert_Model.pdf")
question = st.text_input("Your question")
show_sources = st.checkbox("Show sources", value=True)

if question:
    chunks, index, bm25 = load(pdf_path)
    with st.spinner("Searching and answering..."):
        result = rag_answer(question, chunks, index, bm25)
    st.subheader("Answer")
    st.write(result["answer"])
    st.caption(f"Route: {result['route']} · {result['why']}")      # Route: extractive · span score 23.91, page 5
    if show_sources:
        for chunk_id, page in result["sources"]:
            with st.expander(f"Page {page} · chunk {chunk_id}"):
                st.write(chunks[chunk_id]["text"][:600])
```

### What was wrong in the notebook, and the fix

| Notebook behaviour | Effect on the BERT paper | Fix on this page |
|---|---|---|
| Cleaner deletes `[CLS]`, `[SEP]`, `[MASK]` | 64 of the paper's own words removed; definitions garbled | Keep them; `split_special_tokens=True` in the reader |
| Passage headers "[Source … Score …]" in the prompt | flan-t5-small answered with the header | Metadata only in the app's source panel |
| `max_context_characters=3500` | Token count unpredictable | Budget in generator tokens |
| Token-aware prompt prints "budget used: 0" | Trimmed passage never counted; context was one trimmed noise chunk | Count every passage added, trimmed or not |
| Balanced prompt keeps the first ~100 tokens of each chunk | Answer often cut out: 1 of 12 facts right | Most relevant sentences in reading order: 5 of 12 |
| Dense-only retrieval | Answer chunk for "full form" ranked 12th; recall@1 0.50 | Hybrid BM25 + dense with RRF: recall@1 0.75 |
| One answering method for every question | Exact facts paraphrased or invented | Rule-based router, extractive route for facts |
| No refusal check after generation | "22,500M", "United States" shown as answers | Cosine gate and support check |
| `early_stopping=True` with beams (tried here) | Shorter, worse NSP answer | `early_stopping=False` |

## 🚨 Interview Gotchas & Edge Cases

:::warning[The Most Common Question]
**"How do you make a RAG system stop hallucinating?"** No single setting does it; it takes a chain of checks. Retrieve well (hybrid search, good chunks), give the model only relevant context with an instruction to answer from it and a fixed fallback sentence, keep metadata out of the prompt, route exact facts to an extractive reader, refuse when the best retrieval score is too low, and verify the answer against the context before showing it. Then measure with a test set: on this paper the support check alone turned "22,500M" and "United States" into refusals.
:::

- **RAG is not "an LLM with a PDF".** Without a retrieval step there's nothing to ground on and nothing to cite.
- **High retrieval score ≠ answer present.** Figure labels scored 0.5080 for "full form of BERT" while the real definition ranked 12th.
- **Vague questions retrieve the bibliography.** "What is the main idea?" pulled three reference-list chunks into the top 5.
- **Cleaning can delete content.** `[CLS]`, `[SEP]` and `[MASK]` are words in an NLP paper; deleting them garbled the definitions.
- **Words aren't tokens.** About 1.7 tokens per word here, more for references; MiniLM silently truncates at 256 tokens (48 of 55 chunks).
- **Small models copy whatever looks like an answer**, including your "[Source … Score …]" headers.
- **Budgets must count every piece added.** The notebook's counter missed the trimmed passage and printed 0.
- **First-N-tokens trimming keeps the wrong text.** Select sentences by relevance instead.
- **Greedy decoding can't revisit a near-tie** (0.134 vs 0.113 at step 1); `early_stopping=True` with beams returned a worse answer here.
- **Rule order in routers matters.** "How many…" contains "how".
- **A SQuAD v1.1 reader never says "not found"**; it answered "BooksCorpus" for a GPU question.
- **Word-overlap grounding misses wrong-but-quoted answers** ("MNLI…" for the SQuAD question had support 1.00).
- **Generator and reader fail on different questions** (5 + 5 right, 7 by at least one), which is the case for routing or answer agreement.
- **Fine-tuning is rarely the first fix**: 2 of 12 failures were retrieval misses that no fine-tuned generator could fix.
- **Streamlit re-runs the script on every interaction**; without `@st.cache_resource` it re-indexes on each question.

## ⚡ Essential Interview Recall

A one-screen revision list. Every point is explained in the chapters above.

**RAG basics**
- RAG = **retrieve** the top-$\hyper{k}$ chunks, then **generate** $P(\pred{y} \mid \data{q}, \data{C_k})$; an LLM alone isn't RAG.
- Extractive = copy a span (precise, can't invent words); generative = write new text (explains, combines, can hallucinate).
- Retrieval failure (right chunk never retrieved) ≠ generation failure (right chunk, wrong answer).

**Transformer families**
- Encoder-only (BERT, MiniLM): bidirectional; understanding, embeddings, start/end logits.
- Decoder-only (GPT): causal; autoregressive generation.
- Encoder-decoder (FLAN-T5): bidirectional encoder + causal decoder with cross-attention; input-to-output tasks.
- flan-t5-small 76,961,152 params; flan-t5-base 247,577,856; 512 input tokens.

**Preparing the document**
- Page-wise extraction keeps page numbers; zero-word pages mean OCR is needed.
- Clean with targeted rules (NFKC, hyphen + newline, whitespace); never "strip all symbols"; compare raw vs cleaned counts.
- Sentence-aware chunks: `max_words=250`, 2-sentence overlap, page + id metadata; 54 notebook chunks, 55 on this page.
- Measure in tokens: ~1.7 tokens per word; MiniLM truncates at 256, FLAN-T5 at 512.

**Retrieval**
- MiniLM: 384-dim unit vectors; cosine = dot product; float32; same model for question and chunks.
- `IndexFlatIP`: exact inner-product search, returns scores + row ids; map rows back to chunk records.
- BM25 rewards rare exact words ($k_1 = 1.5$, $b = 0.75$); RRF fuses ranks with $1/(60 + \text{rank})$.
- Recall@1: dense 0.50, BM25 0.50, **hybrid 0.75** on 12 factual questions.

**Prompt and generation**
- Prompt = instruction (+ fallback sentence) + question + context + "Answer:"; no metadata inside.
- Budget in tokens: $\min(512 - 67 - 2 - 12, 360) = 360$ in the notebook.
- Sentence-selected context: 1 → **5 of 12** facts right with the same model.
- Greedy vs beam: beam keeps $\hyper{B}$ candidates, scores by average log-probability (−0.0966 vs greedy −0.1395 on NSP).
- `early_stopping=False`, `no_repeat_ngram_size=3`, `do_sample=False`.

**Making it trustworthy**
- Evaluate per stage; 2 of 12 failures were retrieval misses.
- Route: exact facts → extractive, explanations → generative, check exact patterns first.
- Refuse if best cosine $\lt 0.2$ (FIFA: 0.1408) or support $\lt 0.5$ ("22,500M": 0.0).
- Support catches invented words, not wrong answers made of document words; SQuAD v1.1 readers can't refuse.
- Streamlit: `@st.cache_resource` for the indexes; show route, reason and sources.
- Fine-tune only the component evaluation shows is weak, and measure before and after.

:::speak[Explain RAG and how you would make it reliable]
RAG, retrieval-augmented generation, answers questions from a specific document by searching it first and then generating. I'd read the PDF page by page, clean it carefully, split it into sentence-aware chunks that carry their page number, embed them with a sentence-transformer, and store the vectors in a FAISS inner-product index, plus BM25 statistics for keyword search. For each question I'd retrieve with hybrid search, fusing dense and BM25 ranks with reciprocal rank fusion, then route: exact facts go to an extractive reader that copies the span, explanations go to a generator like FLAN-T5 with a token-budgeted prompt of the most relevant sentences and an instruction to answer only from them. Before showing anything I'd check grounding: refuse when the best cosine is low or when the answer's words aren't in the context. And I'd evaluate stage by stage on questions with known answers, because on the BERT paper two of twelve failures were retrieval misses that no generator could fix.
:::

## 📐 Formula Sheet

| Concept | Formula | In plain words |
|---|---|---|
| RAG | $\data{C_k} = \operatorname{top\text{-}k}_c \operatorname{sim}(\data{q}, \data{c})$, $\ \pred{\hat y} \sim P_{\param{\theta}}(\pred{y} \mid \data{q}, \data{C_k})$ | Retrieve the k most similar chunks, generate from them |
| Autoregressive generation | $P(\pred{y} \mid \data{x}) = \prod_t P_{\param{\theta}}(\pred{y_t} \mid \pred{y_{\lt t}}, \data{x})$ | One token at a time, each conditioned on the previous ones |
| Log form | $\log P(\pred{y} \mid \data{x}) = \sum_t \log P_{\param{\theta}}(\pred{y_t} \mid \pred{y_{\lt t}}, \data{x})$ | Decoders add log-probabilities |
| Prompt token budget | $\hyper{B_{ctx}} = \min(\hyper{L_{max}} - n_{instr} - n_{cue} - n_{safety},\ \hyper{B_{cap}})$ | Context gets what the fixed parts leave |
| Cosine similarity | $\cos(\data{a}, \data{b}) = \dfrac{\data{a} \cdot \data{b}}{\lVert \data{a} \rVert \lVert \data{b} \rVert}$ | Angle-based closeness of two embeddings |
| Unit vectors | $\lVert \data{a} \rVert = \lVert \data{b} \rVert = 1 \Rightarrow \cos = \data{a} \cdot \data{b}$ | After normalising, cosine is a dot product |
| BM25 | $\sum_{w} \operatorname{IDF}(w) \dfrac{f\,(\hyper{k_1}+1)}{f + \hyper{k_1}(1 - \hyper{b} + \hyper{b}\,\lvert \data{d} \rvert/\overline{\lvert d \rvert})}$ | Rare matching words count most, saturating, length-adjusted |
| IDF | $\ln\!\big(1 + \frac{\data{N} - n_w + 0.5}{n_w + 0.5}\big)$ | Fewer chunks contain the word, higher its weight |
| RRF | $\sum_r \dfrac{1}{\hyper{c} + \operatorname{rank}_r(\data{d})}$, $\hyper{c} = 60$ | Add rank-based points from each ranking |
| Beam score | $\dfrac{1}{\lvert \pred{y} \rvert^{\hyper{\alpha}}} \sum_t \log P_{\param{\theta}}(\pred{y_t} \mid \pred{y_{\lt t}}, \data{x})$ | Average log-probability (α = 1) |
| Support | $\dfrac{\lvert \{w \in \pred{a} \setminus \text{stop} : w \in \data{C}\} \rvert}{\lvert \pred{a} \setminus \text{stop} \rvert}$ | Share of answer words found in the context |
| Refusal rule | refuse if $\max \cos \lt 0.2$ or support $\lt \hyper{\tau} = 0.5$ | Don't answer when retrieval or grounding is weak |

## Concept-Check Answers (from the notebook)

All 76 concept-check questions from notebook 22, grouped by notebook section. Answers are corrected and enriched with this page's measurements where the notebook's original answer was incomplete.

### Introduction: Why a New Notebook?

<details>
<summary><strong>Q1. In the previous Retriever + Reader system, what was the role of FAISS?</strong></summary>

FAISS was the **search** part of the retriever. It stored the embedding of every chunk and, for each question's embedding, returned the indices and scores of the most similar chunks. It never answered anything: the reader then extracted the answer from the chunks FAISS pointed to (Chapter 8).

</details>

<details>
<summary><strong>Q2. Why is a BERT-style QnA reader called extractive?</strong></summary>

Because its answer is always a span **extracted** from the given passage. The model outputs a start score and an end score for every token, and the answer is the text between the best start and end positions, copied exactly. It can't produce words that aren't in the passage (Chapters 1–2).

</details>

<details>
<summary><strong>Q3. Why do we need a generative model in this new notebook?</strong></summary>

Many useful answers aren't one copied span: explanations, summaries and comparisons need information combined from several sentences and written in readable language. A generator writes new text token by token from the question and the retrieved context, which a span-picker can't do (Chapter 1).

</details>

<details>
<summary><strong>Q4. Is using an LLM alone the same as RAG?</strong></summary>

No. RAG means **retrieving** external context first and generating the answer from it. An LLM on its own answers from what it memorised in training, which can be outdated, about something else, or invented, and gives no source to check (Chapter 1).

</details>

<details>
<summary><strong>Q5. What are the two main components of RAG?</strong></summary>

**Retrieval** and **generation**. The retriever finds the chunks most relevant to the question (embeddings + FAISS, and on this page BM25 too); the generator writes the answer conditioned on the question and those chunks, $P(\pred{y} \mid \data{q}, \data{C_k})$.

</details>

<details>
<summary><strong>Q6. Why should we show source chunks along with the generated answer?</strong></summary>

So the answer can be **verified**. Showing the page and text the answer was based on lets the user check that it's grounded rather than hallucinated, and lets the developer debug whether a wrong answer came from retrieval or from generation (Chapters 14–15).

</details>

<details>
<summary><strong>Q7. What is the difference between retrieval failure and generation failure?</strong></summary>

A **retrieval failure** means the chunk containing the answer never reached the prompt (on this paper: 2 of 12 factual questions, answer chunks ranked 10th and 11th). A **generation failure** means the right context was in the prompt but the model still wrote a wrong or unsupported answer (for example "22,500M"). They need different fixes, so evaluation must tell them apart (Chapter 12).

</details>

### Section 1: Recap of Retriever + Reader PDF QnA

<details>
<summary><strong>Q1. Why was running QnA on every chunk not scalable?</strong></summary>

The reader is an expensive Transformer call, and brute force runs it once per chunk, so the cost per question grows linearly with the number of chunks. A large document or a collection of documents makes every question slow and costly; a retriever cuts the reader's work to the top few chunks.

</details>

<details>
<summary><strong>Q2. What is the role of the retriever in a Retriever + Reader system?</strong></summary>

It narrows the search: it compares the question with every chunk (cheaply, as vectors) and passes only the most relevant few to the reader. It decides *where* to look, not *what* the answer is.

</details>

<details>
<summary><strong>Q3. What is the role of the reader?</strong></summary>

The reader looks inside each retrieved chunk and extracts the answer span, using start and end logits over the chunk's tokens. The best-scoring span across the retrieved chunks becomes the answer.

</details>

<details>
<summary><strong>Q4. Why is an extractive answer sometimes less useful than a generated answer?</strong></summary>

It can only return one copied phrase. For "why", "explain", "summarise" or "compare" questions the answer spans several sentences or chunks; the extractive reader on this page answered "Why is BERT called bidirectional?" with the fragment "conceptually simple and empirically powerful".

</details>

<details>
<summary><strong>Q5. Give one example where extractive QnA is still better than generative QnA.</strong></summary>

Exact facts that appear verbatim. "How many words are in English Wikipedia?": the extractive reader returned exactly "2,500M", while FLAN-T5 generated "22,500M", a number that isn't in the paper. Numbers, names, full forms, dataset names and metric values are safer to copy than to regenerate.

</details>

<details>
<summary><strong>Q6. Why does generative answering introduce hallucination risk?</strong></summary>

A generator can produce any token sequence, not only text from the context. If the context is weak, missing or confusing, it may fill the gap with fluent text from its training ("United States" for a FIFA question) or garble a fact ("22,500M"). That's why RAG constrains it with an instruction and checks grounding afterwards.

</details>

<details>
<summary><strong>Q7. What is the next architectural change we are making after Retriever + Reader?</strong></summary>

Replacing (or complementing) the extractive reader with a **generative model**: retriever + generator, which is generative RAG. The retriever stays; what changes is the component that produces the answer.

</details>

### Section 2: Why BERT Extracts but Does Not Generate

<details>
<summary><strong>Q1. Why are BERT-style models called encoder-only models?</strong></summary>

They use only the Transformer **encoder** stack: bidirectional self-attention layers that turn input tokens into contextual vectors. They have no decoder that produces tokens one after another, so they don't generate text (Chapter 2).

</details>

<details>
<summary><strong>Q2. Why are encoder-only models strong for understanding tasks?</strong></summary>

Every token attends to every other token, left and right, so each token's vector is built from the whole input. That rich, bidirectional context is exactly what classification, similarity, embeddings and span selection need.

</details>

<details>
<summary><strong>Q3. In extractive QnA, what do start logits and end logits represent?</strong></summary>

One score per input token for "the answer starts here" (start logits) and "the answer ends here" (end logits). The answer is the valid span, inside the context, with end after start and limited length, that maximises start logit + end logit.

</details>

<details>
<summary><strong>Q4. What does a decoder model do during generation?</strong></summary>

It produces the output one token at a time: given the tokens generated so far (and, in an encoder-decoder, the encoded input), it outputs a probability for every vocabulary token, a decoding rule picks one, it's appended, and the step repeats until an end token or a length limit.

</details>

<details>
<summary><strong>Q5. What is autoregressive generation?</strong></summary>

Generating a sequence where each token depends on all previously generated tokens: $P(\pred{y} \mid \data{x}) = \prod_t P(\pred{y_t} \mid \pred{y_{\lt t}}, \data{x})$. Each output becomes part of the input for the next step.

</details>

<details>
<summary><strong>Q6. Why are encoder-decoder models useful for tasks like summarization and RAG?</strong></summary>

The encoder reads the entire input (question + context) bidirectionally once, and the decoder writes a new output while cross-attending to that representation at every step. "Read fully, then write about it" is exactly the shape of summarisation, translation and answering from retrieved context.

</details>

<details>
<summary><strong>Q7. Why is `google/flan-t5-small` a suitable teaching choice for this notebook?</strong></summary>

It is small (76,961,152 parameters), runs on a free Colab or a CPU, is an encoder-decoder, and is **instruction-tuned**, so it follows a prompt like "Answer the question using only the context" without extra training. Its weakness, visible here, is quality: it copied a source header as its answer, which is why the notebook moves to flan-t5-base (247,577,856 parameters).

</details>

<details>
<summary><strong>Q8. Is the shift from BERT reader to FLAN-T5 generator a replacement of retrieval?</strong></summary>

No. Only the answer-producing component changes. The generator still needs the retriever to supply the relevant chunks; without retrieval it would answer from memory, which is not RAG.

</details>

### Section 3: Colab Setup, Library Installation, and Imports

<details>
<summary><strong>Q1. Why do we need `PyMuPDF` in this notebook?</strong></summary>

To open the PDF (`fitz.open`) and extract text page by page (`page.get_text()`), keeping page numbers for source display (Chapter 3).

</details>

<details>
<summary><strong>Q2. What is the role of `sentence-transformers`?</strong></summary>

It loads the embedding model (`all-MiniLM-L6-v2`) that turns chunks and questions into 384-dimensional vectors, with `normalize_embeddings=True` producing unit vectors for cosine search (Chapter 7).

</details>

<details>
<summary><strong>Q3. What is the role of FAISS?</strong></summary>

It stores the chunk embeddings in an index and performs the nearest-neighbour search that returns the top-k most similar chunks to the question vector (Chapter 8).

</details>

<details>
<summary><strong>Q4. Why do we need Hugging Face Transformers?</strong></summary>

It provides the tokenizers and models: FLAN-T5 (`AutoModelForSeq2SeqLM`) for generation and, on this page, the DistilBERT SQuAD reader (`AutoModelForQuestionAnswering`) for the extractive route, plus `generate()` with greedy and beam search.

</details>

<details>
<summary><strong>Q5. Why are we checking whether GPU is available?</strong></summary>

Generation runs much faster on a GPU. On a CPU it still works but is slower (about 7–8 s for one 4-beam answer on this page), so you may choose a smaller model, fewer beams or shorter prompts. The notebook sets `device = "cuda" if torch.cuda.is_available() else "cpu"`.

</details>

<details>
<summary><strong>Q6. Why is `google/flan-t5-small` preferred over a very large LLM for this teaching notebook?</strong></summary>

It fits free hardware, loads in seconds, needs no API key, and still demonstrates every mechanism of RAG (prompting, generation, grounding). Large LLMs give better answers but hide the mechanics behind cost and hardware requirements.

</details>

<details>
<summary><strong>Q7. What does the helper function `print_wrapped()` do?</strong></summary>

It prints long text wrapped to a fixed line width (100 characters by default) using `textwrap.fill`, so chunks and prompts are readable in the notebook output.

</details>

### Section 4: Uploading PDF and Extracting Page-Wise Text

<details>
<summary><strong>Q1. Why do we extract text page by page instead of extracting the whole PDF as one long string?</strong></summary>

Page-wise extraction keeps the **page number** attached to every piece of text. Every chunk inherits it, so the final answer can cite "page 5", and debugging can show exactly where retrieved text came from.

</details>

<details>
<summary><strong>Q2. Why is page number important in a RAG system?</strong></summary>

It makes answers **traceable**: the user can open the page and verify the claim, and the developer can see whether retrieval found the right part of the document.

</details>

<details>
<summary><strong>Q3. What does `fitz.open(pdf_path)` do?</strong></summary>

It opens the PDF and returns a document object whose pages can be iterated; each page object can then extract its text.

</details>

<details>
<summary><strong>Q4. What does `page.get_text()` return?</strong></summary>

A string with the text content of that page, in reading order as PyMuPDF determines it, including line breaks; on the BERT paper the first four pages have 584, 669, 583 and 811 words.

</details>

<details>
<summary><strong>Q5. If most pages show zero words after extraction, what is the likely issue?</strong></summary>

The PDF is **scanned** (pages are images, not text). It needs OCR before chunking, or the pipeline has nothing to index.

</details>

<details>
<summary><strong>Q6. Why should we inspect extracted text before chunking?</strong></summary>

Extraction errors (interleaved columns, figure labels, tables, headers) flow into every later stage: noisy chunks, noisy embeddings, wrong retrieval, wrong answers. Seeing the raw text early is the cheapest place to catch them.

</details>

<details>
<summary><strong>Q7. What kind of PDF is best for this notebook?</strong></summary>

A text-based PDF (you can select and copy the text), ideally single-column with few tables and figures. Scanned PDFs need OCR; complex layouts need block-aware extraction.

</details>

### Section 5: Text Cleaning Pipeline with Special Token Removal

<details>
<summary><strong>Q1. Why should we clean PDF text before chunking?</strong></summary>

Raw PDF text has broken hyphenation, ligatures, invisible characters and messy whitespace. Those make chunks harder to read, weaken embeddings and break exact keyword matches ("ﬁne-tuning" doesn't match "fine-tuning"), so retrieval quality drops (Chapter 4).

</details>

<details>
<summary><strong>Q2. Why do we remove tokens like `[CLS]` and `[SEP]`?</strong></summary>

The notebook removes them because in most documents they are tokenizer artifacts, not content. On the BERT paper that reasoning backfires: the paper uses `[CLS]` 28 times, `[SEP]` 25 times and `[MASK]` 11 times as words, and deleting them garbles the sentences that define them. The safer rule is to remove only what you've confirmed is noise, and to tell the reader's tokenizer not to treat literal "[CLS]" text as a special token (`split_special_tokens=True`).

</details>

<details>
<summary><strong>Q3. Why should we not remove all special characters from technical documents?</strong></summary>

Symbols carry meaning in technical text: percentages, equations, brackets, version numbers, Greek letters, slashes in "80.6/80.1". Removing them destroys facts and makes some questions unanswerable.

</details>

<details>
<summary><strong>Q4. What is the purpose of fixing hyphenated line breaks?</strong></summary>

To rejoin words split across lines, such as "repre-\nsentation" into "representation", so they read correctly, embed correctly and match keyword searches. The rule should only fire on a hyphen followed by a line break.

</details>

<details>
<summary><strong>Q5. Why do we compare raw word count and cleaned word count?</strong></summary>

It's a quick check that cleaning removed noise rather than content. A small difference is expected; a large drop means the cleaner is deleting real words.

</details>

<details>
<summary><strong>Q6. What can happen if the cleaning function is too aggressive?</strong></summary>

It deletes information the answers depend on: on this paper, aggressive special-token removal turned "[CLS] is a special symbol added in front of every input example" into "is a special symbol added in front of every input example", which no model can use to answer a question about [CLS].

</details>

<details>
<summary><strong>Q7. Why is text cleaning important for retrieval quality?</strong></summary>

Retrieval compares embeddings and keywords of chunks with the question. Noise dilutes the embeddings and breaks word matches, so the right chunk scores lower and wrong chunks can outrank it.

</details>

### Section 6: Sentence-Aware Chunking with Metadata

<details>
<summary><strong>Q1. Why is sentence-aware chunking better than word-based chunking?</strong></summary>

It packs whole sentences into chunks, so no sentence (and usually no idea) is cut in half. Retrieved chunks read like complete paragraphs, which helps both embeddings and the generator (Chapter 5).

</details>

<details>
<summary><strong>Q2. What does `sentence_overlap` mean?</strong></summary>

The number of sentences from the end of one chunk repeated at the start of the next (2 here), so an idea that crosses a chunk boundary appears whole in at least one chunk.

</details>

<details>
<summary><strong>Q3. Why should we preserve page number in every chunk?</strong></summary>

The chunk is what retrieval returns, so the chunk must carry its source; otherwise the answer can't cite where it came from.

</details>

<details>
<summary><strong>Q4. Why can poor chunks lead to poor retrieval?</strong></summary>

Each chunk becomes one embedding. A chunk that mixes topics, cuts a definition in half, or is mostly references or figure labels gets a vector that doesn't represent any answerable content, so it is retrieved for the wrong questions and the right chunk is missed.

</details>

<details>
<summary><strong>Q5. Why do we still use `max_words` in sentence-aware chunking?</strong></summary>

Chunks must still fit the models: MiniLM embeds only 256 tokens and the generator's prompt is limited to 512. A size cap keeps chunks manageable; ideally the cap is measured in tokens.

</details>

<details>
<summary><strong>Q6. Why is sentence splitting not always perfect for PDFs?</strong></summary>

Periods appear in abbreviations ("e.g.", "Fig. 1", "et al."), numbers and references; equations, tables and bullet points often have no sentence punctuation; and extraction can break lines oddly. The page's splitter protects common abbreviations, but no rule set is perfect.

</details>

<details>
<summary><strong>Q7. What is the difference between sentence-aware chunking and semantic chunking?</strong></summary>

Sentence-aware chunking respects sentence boundaries and a size limit but ignores topic; semantic chunking places boundaries where the meaning changes, for example where the embedding similarity between neighbouring sentences drops.

</details>

<details>
<summary><strong>Q8. Why should we inspect chunks before creating embeddings?</strong></summary>

Chunk quality sets the ceiling for retrieval and answers. Looking at chunk sizes and samples reveals reference-list chunks, figure noise, tiny fragments and split definitions before they become hard-to-debug retrieval errors.

</details>

### Section 7: Creating Embeddings for Sentence-Aware Chunks

<details>
<summary><strong>Q1. Why do we convert chunks into embeddings?</strong></summary>

To compare meaning numerically. Embeddings place semantically similar texts close together, so the question's vector can be compared with every chunk's vector to find related chunks even without shared keywords (Chapter 7).

</details>

<details>
<summary><strong>Q2. What does it mean for two chunks to be close in embedding space?</strong></summary>

The embedding model considers them similar in meaning: their vectors point in nearly the same direction, so their cosine similarity is high.

</details>

<details>
<summary><strong>Q3. Why should chunk embeddings and question embeddings be created using the same model?</strong></summary>

Each model defines its own vector space. Only vectors from the same model are comparable; mixing models makes similarity scores meaningless.

</details>

<details>
<summary><strong>Q4. What is the shape of the chunk embedding matrix?</strong></summary>

$(N, d)$: one row per chunk, one column per embedding dimension. In the notebook it is $(54, 384)$; on this page $(55, 384)$.

</details>

<details>
<summary><strong>Q5. Why do we normalize embeddings?</strong></summary>

So every vector has length 1. Then cosine similarity equals the dot product, and an inner-product index (`IndexFlatIP`) returns cosine scores directly. The notebook verifies minimum, maximum and average norm are all 1.0.

</details>

<details>
<summary><strong>Q6. Why do we convert embeddings to `float32`?</strong></summary>

FAISS works with 32-bit floats; it's also half the memory of float64 with no meaningful loss for similarity search.

</details>

<details>
<summary><strong>Q7. Why do we keep `chunk_embeddings` separate from `chunks_data`?</strong></summary>

They serve different consumers: the matrix is numeric and goes into FAISS; `chunks_data` holds the readable text and metadata (page, id) for prompts and display. Row $i$ of one corresponds to entry $i$ of the other.

</details>

<details>
<summary><strong>Q8. Why is FAISS needed if we can already calculate similarity using NumPy?</strong></summary>

For 54 chunks NumPy is fine. FAISS matters at scale: it offers optimised exact search and approximate indexes (IVF, HNSW) that search millions of vectors fast, plus GPU support.

</details>

### Section 8: Building FAISS Index and Retrieving Top-k Chunks

<details>
<summary><strong>Q1. What problem does FAISS solve in our RAG pipeline?</strong></summary>

Fast nearest-neighbour search: given the question vector, return the k chunk vectors with the highest similarity, without writing the search loop yourself (Chapter 8).

</details>

<details>
<summary><strong>Q2. Why are we using `IndexFlatIP`?</strong></summary>

The embeddings are normalised, so inner product equals cosine similarity. A Flat index does exact search, which is accurate and plenty fast for a few dozen chunks.

</details>

<details>
<summary><strong>Q3. Why does inner product behave like cosine similarity in our setup?</strong></summary>

Cosine is $\data{a} \cdot \data{b} / (\lVert \data{a} \rVert \lVert \data{b} \rVert)$. With both lengths equal to 1, the denominator is 1, so cosine equals the dot product.

</details>

<details>
<summary><strong>Q4. What does FAISS return after a search?</strong></summary>

Two arrays of shape (number of queries, k): the similarity **scores** and the **row indices** of the nearest vectors, best first (−1 for missing results).

</details>

<details>
<summary><strong>Q5. Why do we need `chunks_data` after FAISS returns indices?</strong></summary>

FAISS stores only vectors. The indices must be mapped back to the chunk records to get the text, page number and chunk id.

</details>

<details>
<summary><strong>Q6. What is the difference between retrieval score and answer correctness?</strong></summary>

The score measures similarity between question and chunk, not whether the chunk contains the answer. For "What is the full form of BERT?" the top chunk (0.5080) was figure-label noise while the defining chunk ranked 12th.

</details>

<details>
<summary><strong>Q7. What does a low top retrieval score indicate?</strong></summary>

That nothing in the document is strongly related to the question: the question may be outside the document, or phrased very differently. This page refuses when the best cosine is below 0.2; the FIFA question scores 0.1408.

</details>

<details>
<summary><strong>Q8. What is one limitation of `IndexFlatIP`?</strong></summary>

It compares the query with every stored vector, so search time and memory grow linearly with the corpus. Large collections use approximate indexes such as IVF or HNSW.

</details>

<details>
<summary><strong>Q9. Why might hybrid retrieval be better than semantic retrieval alone?</strong></summary>

Keyword search (BM25) catches exact rare terms, numbers and acronyms that embeddings blur. On "How many words are in English Wikipedia?" dense search ranked the answer chunks 4th, 6th and 22nd while BM25 ranked them 1st to 3rd; fused with RRF, recall@1 over 12 questions rose from 0.50 to 0.75 (Chapter 9).

</details>

### Section 9: Building a RAG Prompt from Retrieved Chunks

<details>
<summary><strong>Q1. Why do we need a prompt after retrieving chunks?</strong></summary>

The generator only sees its input text. The prompt packages the task instruction, the question and the retrieved context into one input the model can act on (Chapter 10).

</details>

<details>
<summary><strong>Q2. What are the three main parts of a RAG prompt?</strong></summary>

The **instruction** (answer only from the context, with a fallback sentence), the **retrieved context**, and the **user question**, usually ending with an "Answer:" cue.

</details>

<details>
<summary><strong>Q3. Why should the prompt instruct the model to use only the provided context?</strong></summary>

To keep the answer grounded in the document instead of the model's training memory, and to give it an explicit way out ("The provided document does not contain enough information…") when the context lacks the answer.

</details>

<details>
<summary><strong>Q4. Why do we include page number and chunk ID inside the context blocks?</strong></summary>

The notebook does it for traceability and debugging, but it backfired: flan-t5-small answered "What is the full form of BERT?" with "[Source 1 | Page 3 | Chunk 6 | Score 0.5080]". Keep source metadata in the application (it already knows which chunks it retrieved) and show it in the UI, not inside the text the model reads; the notebook's later prompts remove it for this reason.

</details>

<details>
<summary><strong>Q5. Why can too much retrieved context hurt generation?</strong></summary>

It can exceed the 512-token input limit and get truncated, slow generation, bury the relevant sentence among irrelevant text, and give a small model more wrong text to copy. Selecting the most relevant sentences within a budget took the same model from 1 to 5 of 12 facts.

</details>

<details>
<summary><strong>Q6. What is the role of `max_context_characters`?</strong></summary>

It caps how much retrieved text goes into the prompt (3,500 characters by default). Because model limits are in tokens, a token budget measured with the generator's tokenizer is the more reliable control.

</details>

<details>
<summary><strong>Q7. What is the difference between `build_rag_prompt()` and `build_strict_rag_prompt()`?</strong></summary>

`build_rag_prompt()` gives a general grounded instruction; `build_strict_rag_prompt()` adds stronger rules (no outside knowledge, no guessing) and a fixed fallback response, to reduce unsupported answers.

</details>

<details>
<summary><strong>Q8. At this stage, has the system generated an answer yet?</strong></summary>

No. Retrieval has selected chunks and the prompt has been built; generation with FLAN-T5 comes next (Chapter 11).

</details>
