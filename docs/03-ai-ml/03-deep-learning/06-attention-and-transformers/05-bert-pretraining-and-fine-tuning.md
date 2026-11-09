---
title: "BERT: Pre-training and Fine-tuning, Figure by Figure"
sidebar_position: 5
---

import {BertInputExplorer, MlmMaskingPlayground, SpanPredictor, BertSizeCalculator} from '@site/src/components/Widgets';

:::tip[TL;DR (30-Second Interview Pitch)]
**BERT** ("Bidirectional Encoder Representations from Transformers", Devlin et al., NAACL 2019) is the **left half of Transformer Figure 1**: a stack of $L$ encoder layers with no decoder and no causal mask, so every token's representation is conditioned on its **left and right context in every layer**. Because a bidirectional model can't be trained as an ordinary language model (each word would "see itself"), BERT is **pre-trained** on unlabeled text with two tasks: **Masked LM** (hide 15% of the WordPiece tokens, replacing them with [MASK] 80% of the time, a random token 10%, unchanged 10%, and predict the originals) and **Next Sentence Prediction** (is sentence B the real next sentence? read from the [CLS] vector $C$). The input is the sum of **token, segment (A/B) and position** embeddings (Figure 2). For a downstream task you add **one small output layer** (a $K \times H$ classifier on $C$, or a start vector $S$ and end vector $E$ for SQuAD spans) and **fine-tune all parameters** for 2–4 epochs. BERT-base ($L$=12, $H$=768, $A$=12, 110M parameters) and BERT-large ($L$=24, $H$=1024, $A$=16, 340M) set new state of the art on eleven tasks, including GLUE 80.5, MultiNLI 86.7% and SQuAD v1.1 test F1 93.2.
:::

<FormulaKey />

:::info[Where this page sits]
**Before:** [The Transformer: Figure 1 Box by Box](./04-transformer-architecture-figure-1.md) built every box of the encoder that BERT reuses unchanged.
**This page:** notebook 17, the paper **"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"** (arXiv 1810.04805), organized around its **Figure 1** (pre-training and fine-tuning) and **Figure 2** (input representation), with Figures 3 and 4 and every results table. A tiny BERT is pre-trained and fine-tuned on this page so that every mechanism comes with real numbers.
**Next:** notebook 18, Hugging Face pipelines: using pre-trained BERT-family models without writing the model yourself.
:::

## 🧭 Start Here

Before BERT, almost every NLP task trained its own model, often from nothing but word vectors. The 2018 idea that changed this was **transfer learning for text**: first train one big network on a huge amount of unlabeled text so it learns how language works, then adapt it to each task with a little labeled data. Two families already did this. **ELMo** trained left-to-right and right-to-left LSTM language models and handed their hidden states to task-specific models as extra features. **OpenAI GPT** trained a Transformer decoder as a left-to-right language model and then fine-tuned the whole thing on each task. Both were held back by the same thing: a language model predicts the next word, so it may only look **left**.

BERT's argument is that this restriction is the bottleneck. To understand "bank" in "he sat on the bank of the river", the useful word ("river") is to the right. BERT therefore uses the Transformer **encoder**, where every token attends to every other token, and invents a training task that makes sense for a model that sees everything: hide some words and ask the model to fill them in, like a cloze test. A second task, deciding whether two sentences follow each other, teaches it about sentence pairs. After this **pre-training**, the same network is **fine-tuned** on each downstream task with only a tiny output layer added.

The paper's Figure 1 is the whole story in one picture, and this page follows it. On the **left** (pre-training), an unlabeled pair of sentences A and B, with some tokens masked, goes in at the bottom as embeddings $E$; BERT turns them into output vectors $C, T_1, \ldots, T_N, T_{[SEP]}, T_1', \ldots, T_M'$; the masked positions feed the **Mask LM** heads and $C$ feeds the **NSP** head. On the **right** (fine-tuning), the *same* parameters are copied into one model per task (MNLI, NER, SQuAD); for question answering the input is a question and a paragraph, and the output is a **start/end span** in the paragraph. Figure 2 zooms into the bottom row: how each input embedding $E$ is built.

![Overall pre-training and fine-tuning procedures for BERT](/img/deep-learning/diagrams/bert-figure-1.svg)

*How to read it:* our redraw of the paper's Figure 1 with every label kept. Pink = input tokens, yellow = input embeddings $E$, green = final hidden vectors ($C$ for [CLS], $T_i$ for token $i$), blue = the BERT encoder (the faint lines are bidirectional self-attention: every output sees every input). Red arrows are the output heads. Dotted arrows: the pre-trained parameters initialize every fine-tuned model.

The chapters build the idea in order:

1. Why left-to-right pre-training was the limitation (Figure 3: BERT vs GPT vs ELMo).
2. The model: Transformer encoder, $L$ layers, hidden size $H$, $A$ heads.
3. The input: token + segment + position embeddings, [CLS] and [SEP] (Figure 2).
4. Why a bidirectional model can't be trained as a normal language model.
5. Pre-training task 1, the Masked LM, with its 80/10/10 rule.
6. Pre-training task 2, Next Sentence Prediction.
7. Pre-training data and procedure.
8. Fine-tuning: one model, many tasks (Figure 4).
9. Sentence-level heads: GLUE classification and SWAG.
10. Token-level heads: SQuAD spans, unanswerable questions, NER.
11. The feature-based alternative.
12. What the ablations prove.
13. A tiny BERT built from these pieces, pre-trained and fine-tuned.

| New concept | One-line meaning |
|---|---|
| **Pre-training** | Train once on unlabeled text with self-made labels (masked words, next sentence) |
| **Fine-tuning** | Start from the pre-trained weights, add a small head, train everything on labeled task data |
| **Feature-based** | Freeze the pre-trained model and feed its vectors to a separate task model (ELMo style) |
| **Bidirectional** | Each token's vector depends on words to its left *and* right, in every layer |
| **WordPiece** | Sub-word tokenizer with a 30,000-piece vocabulary; rare words split into pieces (play ##ing) |
| **[CLS]** | Special first token; its final vector $C$ summarizes the sequence for classification |
| **[SEP]** | Special separator token between and after sentences |
| **Segment embedding** | Learned vector saying "this token belongs to sentence A" or "to sentence B" |
| **Masked LM (MLM)** | Hide 15% of tokens and predict them from both sides |
| **NSP** | Binary task: is sentence B the actual next sentence after A? |
| **$C$, $T_i$** | Final hidden vectors of [CLS] and of token $i$, each in $\mathbb{R}^H$ |
| **$L$, $H$, $A$** | Number of layers, hidden size, number of attention heads |

:::analogy[A medical residency]
Pre-training is medical school: years of reading everything (unlabeled text), quizzed by covering up words in textbooks and asking what goes there. Fine-tuning is a short residency in one specialty (sentiment, question answering): the same doctor, the same knowledge, a few weeks of supervised cases. Nobody trains a cardiologist from scratch without medical school, and before BERT that's roughly what NLP did for every task.
:::

## 1. Why BERT: Unidirectional Language Models Were the Bottleneck

There are two ways to reuse a pre-trained language representation. In the **feature-based** approach (ELMo), the pre-trained network is frozen and its vectors are *extra inputs* to a task-specific architecture that you still have to design and train. In the **fine-tuning** approach (OpenAI GPT), you add very few task-specific parameters and train *all* the pre-trained parameters on the downstream task. Both used the same pre-training objective: a **unidirectional** language model, predicting the next word from the words before it.

The paper's claim is that this objective limits the architecture. GPT is a left-to-right Transformer: every token may attend only to the tokens on its left. For a sentence-level task that's sub-optimal; for a **token-level** task such as question answering it can be very harmful, because the meaning of a word in the paragraph often depends on what comes after it, and the answer must be chosen with both the question and the full paragraph in view. ELMo gets both directions, but only by training two separate models (one left-to-right LSTM, one right-to-left) and **concatenating** their states: no single layer ever combines left and right context.

BERT's fix is a **deep** bidirectional model: a Transformer encoder whose self-attention is unmasked, so in every layer each token's vector mixes information from the entire sequence. Figure 3 of the paper shows the three architectures side by side.

![Differences in pre-training model architectures](/img/deep-learning/diagrams/bert-figure-3.svg)

*How to read it:* each "Trm" is a Transformer layer at one position, each "Lstm" an LSTM cell. In **BERT** every input position connects to every position in the next layer. In **OpenAI GPT** position $i$ only connects to positions $\ge i$ in the layer above (the causal mask from the Transformer's decoder). **ELMo** runs a left-to-right and a right-to-left LSTM separately and concatenates their outputs at the end.

| | ELMo | OpenAI GPT | BERT |
|---|---|---|---|
| Network | 2 separate LSTM language models | Transformer **decoder**-style (masked self-attention) | Transformer **encoder** (unmasked self-attention) |
| Context of each token | left *or* right, joined only at the end | left only | left **and** right, in every layer |
| Pre-training objective | next word / previous word | next word | **Masked LM + NSP** |
| How it's used downstream | **feature-based** (frozen vectors into a task model) | **fine-tuning** | **fine-tuning** (feature-based also works, chapter 11) |
| Training corpus | — | BooksCorpus (800M words) | BooksCorpus + English Wikipedia (3.3B words) |

:::analogy[Reading through a keyhole]
A left-to-right model reads a sentence through a keyhole that slides from left to right: when it reaches "bank" it hasn't seen "river" yet. ELMo sends two readers, one from each end, and staples their notes together afterwards. BERT reads the whole page at once, twelve times over, and every reading pass can use anything on the page.
:::

:::tip[🎯 Interview angle]
**"What limitation of GPT and ELMo did BERT address?"** Both pre-trained with unidirectional language models. GPT's self-attention only sees left context, which hurts token-level tasks like question answering; ELMo combines two independently trained one-directional LSTMs only by shallow concatenation. BERT pre-trains a *deep* bidirectional Transformer encoder, made possible by the masked-LM objective, and then fine-tunes all of it.
:::

## 2. The Model: a Transformer Encoder, L × H × A

BERT's architecture is not new: it is the multi-layer **Transformer encoder** of "Attention Is All You Need", almost unchanged. The paper names its three size knobs: $L$ = number of layers (Transformer blocks), $H$ = hidden size (the paper's $d_{model}$), $A$ = number of self-attention heads (the paper's $h$). The feed-forward inner size is always $4H$. Each layer is exactly the encoder layer of the previous page: multi-head self-attention → Add & Norm → feed-forward → Add & Norm, with no mask, so every position attends to every position.

![BERT keeps only the left half of Transformer Figure 1](/img/deep-learning/diagrams/bert-encoder-only.svg)

*How to read it:* the original Figure 1 with the decoder faded. BERT keeps the encoder stack. Its outputs $T_1 \ldots T_n$ (one vector per token) go straight to small task heads instead of to a decoder.

Two sizes are reported. **BERT-base** ($L$ = 12, $H$ = 768, $A$ = 12, 110M parameters) was chosen to have the same size as OpenAI GPT so the two could be compared fairly; the only architectural difference between them is the attention mask. **BERT-large** ($L$ = 24, $H$ = 1024, $A$ = 16, 340M parameters) shows what scale adds. A few details differ from the 2017 Transformer: positions use **learned** embeddings (a 512 × $H$ table) instead of sinusoids, a **segment** embedding is added to every token, the feed-forward activation is **GELU** instead of ReLU (following GPT), and there is no decoder, so no cross-attention and no causal mask.

| | Transformer base (2017) encoder | BERT-base | BERT-large |
|---|---|---|---|
| Layers | 6 | $L$ = 12 | $L$ = 24 |
| Hidden size | $d_{model}$ = 512 | $H$ = 768 | $H$ = 1024 |
| Heads (head size) | 8 (64) | $A$ = 12 (64) | $A$ = 16 (64) |
| Feed-forward | 2048, ReLU | 3072 = 4H, GELU | 4096 = 4H, GELU |
| Positions | sinusoidal, fixed | learned, 512 | learned, 512 |
| Vocabulary | ~37K BPE (shared) | 30,000 WordPiece | 30,000 WordPiece |
| Parameters | encoder + decoder 65M | 110M | 340M |

Where do 110 million parameters come from? Each encoder layer has four $H \times H$ attention projections, a feed-forward layer $H \to 4H \to H$, and two LayerNorms; the embeddings are three lookup tables. Counting every weight and bias:

<Formula name="BERT parameter count" tag="Model size" words="Each encoder layer holds 12 H squared weights (4 in attention, 8 in the feed-forward network) plus 13 H biases and LayerNorm parameters; the embeddings are vocabulary, position and segment tables of width H." remember="Twelve H squared per layer.">

$$
\begin{aligned}
\text{attention} &= 4\hyper{H}^2 + 4\hyper{H} \\
\text{FFN} &= 8\hyper{H}^2 + 5\hyper{H} \\
\text{layer} &= \text{attention} + \text{FFN} + 4\hyper{H} \\
&= 12\hyper{H}^2 + 13\hyper{H} \\
\text{embeddings} &= (\hyper{V} + 514)\hyper{H} + 2\hyper{H} \\
\text{total} &= \text{embeddings} + \hyper{L} \cdot \text{layer} + \hyper{H}^2 + \hyper{H}
\end{aligned}
$$

- $\hyper{H}$: hidden size; $\hyper{L}$: number of layers; $\hyper{V}$: vocabulary size; 514 = 512 positions + 2 segments.
- Attention = the Q, K, V and output projections with biases; FFN = $H \to 4H \to H$ with biases; $4H$ = two LayerNorms.
- The last term, $H^2 + H$, is the pooler: the released code's dense + tanh layer on $C$ (used by NSP and classifiers).
- The number of heads $\hyper{A}$ does not appear: heads split $H$ into slices, they don't add weights.

**Example:** BERT-base with the released 30,522-piece vocabulary: embeddings 23,837,184 + 12 × 7,087,872 + pooler 590,592 = **109,482,240** (exactly the parameter count of HuggingFace's `BertModel`); BERT-large: **335,141,888**. The paper rounds these to 110M and 340M.

</Formula>

<BertSizeCalculator />

:::danger[Interview Trap: "BERT is a new architecture"]
It isn't. The paper says its implementation is "almost identical" to the original Transformer encoder and spends one paragraph on the architecture. BERT's contributions are the **pre-training tasks** (MLM, NSP) that make a deep bidirectional encoder trainable, the **input format** that handles single sentences and pairs, and the **fine-tuning recipe** that reuses one model for eleven tasks.
:::

:::tip[🎯 Interview angle]
**"What is BERT's architecture, and how big is BERT-base?"** A stack of Transformer encoder layers with unmasked (bidirectional) self-attention. BERT-base has $L$ = 12 layers, $H$ = 768, $A$ = 12 heads of 64, feed-forward 3072 with GELU, learned position embeddings up to 512 tokens, and about 110M parameters (109.5M exactly with the 30,522-piece vocabulary); BERT-large is 24 / 1024 / 16 with 340M. Per layer the count is $12H^2 + 13H$, so the heads don't change the size.
:::

## 3. Input Representation (Figure 2)

One input format has to serve every task: a single sentence (sentiment), a pair of sentences (entailment, paraphrase), a question with a paragraph (QA). BERT packs everything into **one token sequence**. In the paper a "sentence" means any contiguous span of text, not necessarily a linguistic sentence; a "sequence" is the whole packed input. The first token is always the special **[CLS]** (classification) token. Two sentences are separated by a special **[SEP]** token, and a final [SEP] ends the sequence: `[CLS] A [SEP] B [SEP]`. A single sentence is `[CLS] A [SEP]`.

Words are split with **WordPiece**, a sub-word tokenizer with a 30,000-token vocabulary. Frequent words are single pieces; rare words are split greedily into the longest pieces the vocabulary contains, with continuation pieces marked by `##`. So "playing" becomes `play ##ing` and "flightless" becomes `flight ##less`. The vocabulary stays small, and no word is ever out of vocabulary.

Each token's **input embedding** is the **sum of three learned embeddings**: a **token** embedding (which piece it is), a **segment** embedding ($E_A$ for every token of sentence A including the first [CLS] and [SEP], $E_B$ for every token of sentence B), and a **position** embedding ($E_0, E_1, \ldots$ up to 511). Summing rather than concatenating keeps the width at $H$. The sum is then layer-normalized and passed through dropout before the first encoder layer. The segment embedding is how BERT knows which sentence a token came from, even though both sentences are in one sequence and attend to each other freely.

![BERT input representation](/img/deep-learning/diagrams/bert-figure-2.svg)

*How to read it:* the paper's Figure 2 example, column by column. The input pair "my dog is cute" / "he likes playing" is tokenized (playing → play ##ing), wrapped with [CLS] and [SEP], and each token's three embeddings are added.

<Animation src="bert/input-embeddings" caption="Each row is a lookup table. Watch the three rows for one column collapse into a single vector: that sum is what enters encoder layer 1." />

<Formula name="BERT input embedding" tag="Figure 2" words="The vector fed to the first encoder layer for token i is the sum of its token embedding, its segment embedding and its position embedding, then LayerNorm." remember="Token plus segment plus position.">

$$
\pred{x_i} = \operatorname{LayerNorm}\big(\param{E_{tok}}[\data{t_i}] + \param{E_{seg}}[\data{s_i}] + \param{E_{pos}}[i]\big) \in \mathbb{R}^{\hyper{H}}
$$

- $\data{t_i}$: WordPiece id of token $i$; $\data{s_i} \in \lbrace A, B \rbrace$: its segment; $i$: its position (0 to 511).
- $\param{E_{tok}}$ ($V \times H$), $\param{E_{seg}}$ ($2 \times H$), $\param{E_{pos}}$ ($512 \times H$): learned tables.
- $\pred{x_i}$: the input vector (dropout is applied after the LayerNorm in training).

**Example:** in Figure 2, the token "he" at position 6 in sentence B gets $E_{he} + E_B + E_6$.

</Formula>

<BertInputExplorer />

:::analogy[A name badge with three lines]
Every token walks into the encoder wearing a badge: line one says who it is (the token embedding), line two which team it's on (segment A or B), line three its seat number (position). Because the badge is one sum, it's the same size whatever is written on it, and the encoder learns to read all three lines at once.
:::

:::danger[Interview Trap: positional encodings in BERT]
BERT does **not** use the sinusoidal positional encoding of the Transformer paper. Its positions are a **learned** table of 512 vectors. That's why BERT can't take inputs longer than 512 tokens without new position vectors, and why its pre-training spends the last 10% of steps on length 512: to train those embeddings (chapter 7).
:::

:::tip[🎯 Interview angle]
**"What goes into BERT's input embeddings?"** Three learned embeddings summed per token: WordPiece token embedding, segment embedding (A or B) and learned position embedding (up to 512), followed by LayerNorm and dropout. Every sequence starts with [CLS], whose final vector $C$ is used for classification, and sentences are separated and terminated by [SEP]. A pair is packed as [CLS] A [SEP] B [SEP] with segment A on the first part and B on the second.
:::

## 4. The Problem with Bidirectional Language Models

A standard language model is trained to predict each word from the words before it. That objective only works for a left-to-right (or right-to-left) model, and the paper explains why in one sentence: with bidirectional conditioning, each word could indirectly **"see itself"**. In a multi-layer encoder, position 5's output in layer 2 depends on every position's layer-1 output, including position 5's own input token. If you asked every position to predict its own input word, the model would simply copy it through the network and learn nothing about language.

Left-to-right models avoid this with the causal mask: position $i$ can't see positions $\ge i$, so it can't see the word it must predict. But then it can't see the right context either. BERT's way out is to keep full bidirectional attention and **remove the answer from the input** instead: replace some tokens with [MASK] and predict only those. The model now has both sides of every masked word and no way to copy it. That is the masked language model of the next chapter.

How much does the right-hand context actually buy? This page's tiny BERT (chapter 13) lives in a synthetic language where each "sentence" is six numbers forming an arithmetic progression modulo 23 (for example 14 19 1 6 11 16, step 5). We trained the same encoder twice with the same masked-LM data: once bidirectional, once with a causal mask so each position sees only its left. Then we masked one token of sentence A at each position and measured how often the original was recovered:

| Masked position in sentence A | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Left-to-right (causal mask) | 4.65% | 19.65% | 100% | 100% | 100% | 100% |
| Bidirectional (BERT) | **100%** | **100%** | 100% | 100% | 100% | 100% |

The left-to-right model at position 1 sees only [CLS], so it can only guess one of 23 values (chance is 4.3%); at position 2 it knows one value but not the step (one of 5, 20%). With right context, every position is recoverable. The same causal model's NSP accuracy was 50.6%, pure chance: its $C$ vector sits at position 0 and, under a causal mask, sees nothing but [CLS].

<Animation src="bert/left-vs-bidirectional" caption="Same encoder, same data, one difference: the attention mask. Left-to-right fails exactly where the needed context lies to the right." />

:::tip[🎯 Interview angle]
**"Why can't you train BERT with a normal language-modelling objective?"** Because in a bidirectional multi-layer encoder, each position's output depends on its own input token through the other positions, so predicting every input word would be trivial copying ("seeing itself"). A causal mask fixes that but throws away right context. BERT keeps bidirectional attention and instead hides the target tokens in the input (masked LM).
:::

## 5. Pre-training Task #1: Masked LM

The **masked language model** (the paper also calls it a Cloze task, after a 1953 readability test where readers fill in deleted words) works like this. The training data generator chooses **15% of the WordPiece positions** in each sequence at random. The final hidden vectors $T_i$ at those positions go through an output softmax over the vocabulary, as in an ordinary language model, and the loss is the cross-entropy against the **original** token. Unlike a denoising autoencoder, BERT predicts only the masked positions, not the whole input.

If every chosen token were simply replaced by [MASK], there would be a **mismatch between pre-training and fine-tuning**: [MASK] never appears in real downstream inputs, so the model would be learning features for a situation it never meets again. The paper therefore corrupts the chosen tokens in three ways. If token $i$ is chosen, it is replaced by **[MASK] 80% of the time**, by a **random token 10%** of the time, and **left unchanged 10%** of the time. In all three cases $T_i$ must predict the original token. Using the appendix's example sentence "my dog is hairy" with "hairy" chosen: 80% "my dog is [MASK]", 10% "my dog is apple", 10% "my dog is hairy".

![Masked LM: which positions are predicted, and what the model is shown there](/img/deep-learning/diagrams/bert-mlm-masking.svg)

*How to read it:* of all tokens, 15% are chosen; of those, 80/10/10 go to [MASK], random, unchanged. Overall that is 12% [MASK], 1.5% random, 1.5% unchanged-but-graded.

Why does this help? The encoder **doesn't know which words it will be asked to predict, or which have been replaced by random words**, so it is forced to keep a good contextual representation of *every* input token: a real-looking word might be a random substitute, and an unchanged word might be graded. The unchanged 10% also biases the representation towards the actual observed word. And because random replacement touches only 1.5% of all tokens (10% of 15%), it doesn't seem to harm the model's language understanding.

<Animation src="bert/masking" caption="Chosen tokens are corrupted three ways, but all are graded against the original. The bar is a numpy simulation of the rule over one million tokens." />

<Formula name="Masked LM loss" tag="Pre-training task 1" words="Average, over the chosen positions only, of the negative log probability the model assigns to the original token, using a softmax over the whole vocabulary on top of T_i." remember="Predict only what was hidden.">

$$
\loss{\mathcal{L}_{MLM}} = -\frac{1}{|\mathcal{M}|}\sum_{i \in \mathcal{M}} \log \operatorname{softmax}\big(\pred{T_i}\,\param{W_{tok}}^{\top} + \param{b}\big)_{\data{t_i}}
$$

- $\mathcal{M}$: the chosen 15% of positions; $\data{t_i}$: the original token at position $i$.
- $\pred{T_i}$: BERT's final hidden vector at position $i$ (after a small dense + GELU + LayerNorm transform in the released code).
- $\param{W_{tok}}$: the output vocabulary matrix, tied to the input token embeddings in the released code.

**Example:** a numpy simulation of the rule over 1,000,000 tokens (seed 0) chose 15.00%; 12.01% of all tokens became [MASK], 1.50% random, 1.49% unchanged; of the chosen, 80.07% / 9.98% / 9.95%.

</Formula>

<MlmMaskingPlayground />

The appendix tested the mixing ratios (Table 8, BERT-base dev results). MNLI is fine-tuned; NER is reported both fine-tuned and **feature-based** (frozen BERT, last four layers concatenated), where the mismatch should hurt most because the model can't adapt:

| [MASK] | Same | Random | MNLI (fine-tune) | NER (fine-tune) | NER (feature-based) |
|---|---|---|---|---|---|
| **80%** | **10%** | **10%** | **84.2** | **95.4** | **94.9** |
| 100% | 0% | 0% | 84.3 | 94.9 | 94.0 |
| 80% | 0% | 20% | 84.1 | 95.2 | 94.6 |
| 80% | 20% | 0% | 84.4 | 95.2 | 94.7 |
| 0% | 20% | 80% | 83.7 | 94.8 | 94.6 |
| 0% | 0% | 100% | 83.6 | 94.9 | 94.6 |

Fine-tuning is surprisingly robust to the strategy. As expected, **[MASK]-only** is the worst for the feature-based NER (94.0), where the representation never gets to adapt to [MASK]-free inputs, and **random-only** is clearly worse too. The 80/10/10 mix is best or near best everywhere.

A cost of the MLM: each batch only gives a learning signal on 15% of tokens, against 100% for a left-to-right model. The appendix checked this (Figure 5): MLM pre-training does converge slightly slower, but its downstream accuracy overtakes the left-to-right model almost immediately.

:::analogy[A proofreading exam]
The examiner circles about one word in seven. Most circled words are blacked out, a few are swapped for a wrong word, and a few are left as they were, and you must write the correct word for every circled spot. Since you can't tell which uncircled words are safe or which visible words are traps, you end up reading every word carefully, and that careful reading is the skill being trained.
:::

:::danger[Interview Trap: "15% of tokens are replaced with [MASK]"]
15% are **chosen for prediction**; only 80% of those (12% of all tokens) become [MASK]. 1.5% become random words and 1.5% stay unchanged but are still predicted. The loss covers all 15%, and none of the other 85%.
:::

:::tip[🎯 Interview angle]
**"Explain BERT's masking strategy and why it's 80/10/10."** Choose 15% of WordPiece positions; replace the token with [MASK] 80% of the time, a random token 10%, keep it 10%; predict the original at all chosen positions with cross-entropy. Pure [MASK] would create a pre-train/fine-tune mismatch because [MASK] never appears downstream; random and unchanged tokens make the model keep a good representation of every token since it can't tell which are corrupted or graded. Table 8 shows fine-tuning is robust to the mix, while [MASK]-only hurts the feature-based setting most (94.0 vs 94.9 NER F1).
:::

## 6. Pre-training Task #2: Next Sentence Prediction

Many important tasks, such as question answering and natural language inference, depend on the **relationship between two sentences**, which a language model doesn't directly learn. BERT therefore adds a binary task that can be generated from any text corpus for free. For each pre-training example, sentence B is the **actual next span** of text after A **50% of the time** (label **IsNext**) and a **random span** from the corpus the other 50% (label **NotNext**). The final vector $C$ of the [CLS] token goes through a binary classifier. The appendix's examples:

| Input | Label |
|---|---|
| [CLS] the man went to [MASK] store [SEP] he bought a gallon [MASK] milk [SEP] | **IsNext** |
| [CLS] the man [MASK] to the store [SEP] penguin [MASK] are flight ##less birds [SEP] | **NotNext** |

Note that both tasks run on the same example: the sentences are masked for MLM while $C$ is trained on NSP. The final pre-trained model achieves **97–98% NSP accuracy**. The paper warns in a footnote that $C$ is **not a meaningful sentence representation without fine-tuning**, precisely because it was trained only for NSP.

<Formula name="NSP loss and the total pre-training loss" tag="Pre-training task 2" words="NSP is a two-way classification from the CLS vector. The pre-training loss is the mean masked LM loss plus the mean NSP loss." remember="Total equals MLM plus NSP.">

$$
\begin{aligned}
\pred{p} &= \operatorname{softmax}\big(\operatorname{tanh}(\pred{C}\param{W_p} + \param{b_p})\,\param{W_{nsp}}\big) \\
\loss{\mathcal{L}_{NSP}} &= -\log \pred{p}_{\data{y}}, \qquad \data{y} \in \lbrace \text{IsNext}, \text{NotNext} \rbrace \\
\loss{\mathcal{L}} &= \loss{\mathcal{L}_{MLM}} + \loss{\mathcal{L}_{NSP}}
\end{aligned}
$$

- $\pred{C}$: final hidden vector of [CLS]; $\param{W_p}, \param{b_p}$: the pooler (dense + tanh) of the released code; $\param{W_{nsp}}$: $H \times 2$.
- $\data{y}$: IsNext if B really followed A in the document.

**Example:** with 50/50 labels an untrained classifier sits at $\ln 2 = 0.693$; this page's tiny BERT went from 0.703 at step 1 to 0.105 at step 4,000 and 97.8% held-out NSP accuracy.

</Formula>

Did NSP help? The ablation (Table 5, chapter 12) trained BERT-base without it: MNLI dropped from 84.4 to 83.9, QNLI from 88.4 to 84.9, SQuAD F1 from 88.5 to 87.9. Later work disagreed: RoBERTa (Liu et al., 2019) found that removing NSP and packing full-length contiguous text matched or improved results, and most later encoders dropped it. In an interview, give both: the BERT paper found it helped QA and NLI; follow-up work attributed that mostly to how the input was constructed, and NSP is generally considered a weak task now.

:::tip[🎯 Interview angle]
**"What is NSP and is it useful?"** Next Sentence Prediction: given [CLS] A [SEP] B [SEP], classify from $C$ whether B is the real next span (50%) or a random one (50%). It was meant to teach sentence relationships for QA and NLI, and the paper's ablation shows removing it hurt QNLI (88.4 → 84.9), MNLI and SQuAD. RoBERTa later showed NSP can be dropped without loss when you train on long contiguous text, so modern encoders usually omit it.
:::

## 7. Pre-training Data and Procedure

Pre-training follows the left half of Figure 1. The corpus is **BooksCorpus** (800M words) plus **English Wikipedia** (2,500M words, text passages only, no lists, tables or headers). The paper stresses that a **document-level** corpus is critical, rather than a shuffled sentence-level corpus like the Billion Word Benchmark, because the inputs are long contiguous spans and NSP needs real "next" text.

Each training input is built by sampling two spans ("sentences", usually much longer than one linguistic sentence) such that the pair is at most **512 tokens**. Span A gets segment embedding A, span B gets B; 50% of the time B really follows A. Masking is applied after WordPiece tokenization at a uniform 15%, with no special treatment of word pieces (a later released variant, "whole word masking", masks all pieces of a word together).

| Setting (paper Appendix A.2) | Value |
|---|---|
| Batch | 256 sequences × 512 tokens (the paper writes 128,000 tokens; 256 × 512 = 131,072) |
| Steps | 1,000,000, about 40 epochs over 3.3B words |
| Optimizer | Adam, learning rate 1e-4, $\beta_1$ = 0.9, $\beta_2$ = 0.999, L2 weight decay 0.01 |
| Schedule | linear warmup over the first 10,000 steps, then linear decay |
| Regularization | dropout 0.1 on all layers |
| Activation | GELU (as in GPT) instead of ReLU |
| Loss | mean MLM likelihood + mean NSP likelihood |
| Sequence length | 128 tokens for 90% of steps, 512 for the last 10% |
| Hardware | BERT-base: 4 Cloud TPUs (16 chips); BERT-large: 16 Cloud TPUs (64 chips); 4 days each |

The length schedule is a cost trick. Attention is quadratic in sequence length: a 512-token sequence has $(512/128)^2 = 16$ times as many attention scores per head as a 128-token one. So most of training uses short sequences, and the final 10% at length 512 exists mainly to **learn the position embeddings** for positions 128 to 511. Is all this training needed? Appendix C.1 answers yes: BERT-base gains almost 1.0% MNLI accuracy from 1M steps compared with 500k.

:::analogy[Practising scales before concerts]
A pianist practises mostly short exercises (cheap, many repetitions) and only in the last weeks rehearses full-length pieces. BERT does 90% of its practice on 128-token passages and saves the expensive 512-token passages for the end, mostly so it learns what the later seats (positions 128–511) mean.
:::

:::tip[🎯 Interview angle]
**"How was BERT pre-trained?"** On BooksCorpus and English Wikipedia (3.3B words), with document-level spans packed into pairs of at most 512 tokens, MLM + NSP loss, batch 256, 1M steps (about 40 epochs), Adam with learning rate 1e-4, 10k warmup steps and linear decay, weight decay 0.01, dropout 0.1, GELU. 90% of steps used length 128 and the last 10% length 512 to learn the later position embeddings, because attention cost grows with the square of the length. BERT-base took 4 days on 16 TPU chips.
:::

## 8. Fine-tuning: One Model, Many Tasks

Fine-tuning is the right half of Figure 1, and the paper calls it straightforward because self-attention already handles both single texts and text pairs. For each task you plug the task's inputs into the same `[CLS] A [SEP] B [SEP]` format, add one output layer, initialize everything else from the pre-trained checkpoint, and **train all parameters end to end**. Each task gets its own fine-tuned copy, even though all copies start from the same pre-trained weights.

Sentence A and sentence B from pre-training map onto very different tasks: (1) the two sentences of a **paraphrase** pair, (2) the hypothesis and premise of an **entailment** pair, (3) the **question and passage** in question answering, and (4) a **degenerate text–∅ pair** (only sentence A) for single-sentence classification or tagging. At the output, the token vectors $T_i$ feed a layer for **token-level** tasks (tagging, QA spans), and $C$ feeds a layer for **sequence-level** classification (entailment, sentiment). Earlier pair models often encoded the two texts separately and then applied cross-attention between them; encoding the concatenated pair with self-attention does both at once, because self-attention across the joint sequence *is* bidirectional cross-attention between the two sentences.

![Illustrations of Fine-tuning BERT on Different Tasks](/img/deep-learning/diagrams/bert-figure-4.svg)

*How to read it:* the paper's Figure 4. (a) and (b) are sequence-level tasks: a class label from $C$. (c) and (d) are token-level: a start/end span from the paragraph's $T_i$, or one tag per token (here a person-name tag, B-PER).

Fine-tuning is cheap compared with pre-training: every result in the paper can be reproduced in at most 1 hour on a single Cloud TPU or a few hours on a GPU, starting from the same pre-trained model (the SQuAD model trains in about 30 minutes on a TPU to 91.0 dev F1). Most hyperparameters stay as in pre-training (dropout stays 0.1); the paper found this range works well across tasks:

| Fine-tuning hyperparameter (A.3) | Values to search |
|---|---|
| Batch size | 16, 32 |
| Learning rate (Adam) | 5e-5, 3e-5, 2e-5 |
| Epochs | 2, 3, 4 |

Large datasets (100k+ labeled examples) are far less sensitive to these choices than small ones, so the paper recommends simply trying all combinations and keeping the best on the dev set.

:::analogy[One engine, many vehicles]
Pre-training builds a single engine. Fine-tuning bolts it into a bus, a truck or a race car: a different body and controls (the output layer) and a short tune-up of the whole machine (all parameters updated) for the job. Nobody designs a new engine for each vehicle.
:::

:::tip[🎯 Interview angle]
**"How do you fine-tune BERT for a new task?"** Format the input as [CLS] A [SEP] (B [SEP]), add a task head (a $K \times H$ classifier on $C$ for sequence classification, or per-token layers on $T_i$ for tagging and span extraction), load the pre-trained weights, and train everything with a small learning rate (2e-5 to 5e-5), batch 16 or 32, for 2 to 4 epochs, choosing by dev score. Only the head is new; everything else is updated, not frozen.
:::

## 9. Sentence-level Heads: GLUE Classification and SWAG

For **GLUE** (eight sentence and sentence-pair classification datasets: MNLI, QQP, QNLI, SST-2, CoLA, STS-B, MRPC, RTE), the input is packed as in chapter 3 and the aggregate representation is $C \in \mathbb{R}^H$, the final vector of [CLS]. The **only new parameters** are the classification weights $W \in \mathbb{R}^{K \times H}$, where $K$ is the number of labels, and the loss is the standard $\log(\operatorname{softmax}(CW^\top))$. The paper fine-tuned with batch 32 for 3 epochs and picked the best learning rate among 5e-5, 4e-5, 3e-5 and 2e-5 on the dev set. BERT-large was sometimes unstable on small datasets, so they ran several random restarts (same checkpoint, different data shuffling and classifier initialization) and kept the best on dev.

<Formula name="Sequence classification head" tag="GLUE fine-tuning" words="Multiply the CLS vector by a K by H weight matrix to get one score per label, softmax, and use cross-entropy against the true label." remember="One matrix on C.">

$$
\pred{P}(\data{y} = k) = \operatorname{softmax}(\pred{C}\,\param{W}^\top)_k, \qquad \loss{\mathcal{L}} = -\log \pred{P}(\data{y})
$$

- $\pred{C} \in \mathbb{R}^{H}$: final hidden vector of [CLS]; $\param{W} \in \mathbb{R}^{K \times H}$: the only new weights.
- $K$: number of labels (3 for MNLI: entailment, contradiction, neutral).

**Example:** BERT-base MNLI adds $3 \times 768 = 2{,}304$ new weights (plus 3 biases in practice) to a 110M-parameter model.

</Formula>

| System (GLUE test) | MNLI-m/mm | QQP | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE | Average |
|---|---|---|---|---|---|---|---|---|---|
| Pre-OpenAI SOTA | 80.6/80.1 | 66.1 | 82.3 | 93.2 | 35.0 | 81.0 | 86.0 | 61.7 | 74.0 |
| BiLSTM+ELMo+Attn | 76.4/76.1 | 64.8 | 79.8 | 90.4 | 36.0 | 73.3 | 84.9 | 56.8 | 71.0 |
| OpenAI GPT | 82.1/81.4 | 70.3 | 87.4 | 91.3 | 45.4 | 80.0 | 82.3 | 56.0 | 75.1 |
| BERT-base | 84.6/83.4 | 71.2 | 90.5 | 93.5 | 52.1 | 85.8 | 88.9 | 66.4 | 79.6 |
| **BERT-large** | **86.7/85.9** | **72.1** | **92.7** | **94.9** | **60.5** | **86.5** | **89.3** | **70.1** | **82.1** |

Both BERT models beat every system on every task, by 4.5% (base) and 7.0% (large) average accuracy over the prior state of the art. BERT-base and GPT are nearly identical architectures apart from the attention mask, so the base-vs-GPT gap (79.6 vs 75.1) is mostly the effect of bidirectionality and the pre-training tasks. On the official leaderboard, which also counts the problematic WNLI set, BERT-large scored **80.5** against GPT's 72.8. BERT-large's biggest margins over GPT are on two of the smallest datasets: CoLA (+15.1, 8.5k examples) and RTE (+14.1, 2.5k).

**SWAG** (113k sentence-pair completion examples) asks which of four continuations is most plausible. BERT builds **four input sequences**, each [CLS] sentence [SEP] continuation [SEP], and the only new parameter is a **vector** whose dot product with each sequence's $C$ is that choice's score; a softmax over the four scores gives the prediction. Fine-tuned for 3 epochs at learning rate 2e-5 with batch 16, BERT-large reached 86.3% test accuracy, 27.1 points above the authors' own ESIM+ELMo baseline and 8.3 above GPT.

<Formula name="Multiple-choice head (SWAG)" tag="Choose 1 of 4" words="Run BERT once per candidate pair, score each candidate with one dot product on its CLS vector, and softmax across the candidates." remember="One vector, four forward passes.">

$$
\pred{P}(\text{choice } c) = \frac{e^{\param{v} \cdot \pred{C_c}}}{\sum_{c'=1}^{4} e^{\param{v} \cdot \pred{C_{c'}}}}
$$

- $\pred{C_c}$: [CLS] vector of the sequence "sentence + candidate $c$"; $\param{v} \in \mathbb{R}^H$: the only new parameters.

**Example:** BERT-large adds just 1,024 weights for SWAG.

</Formula>

:::tip[🎯 Interview angle]
**"How does BERT do sentence-pair classification, and what are the new parameters?"** Pack [CLS] A [SEP] B [SEP] with segments A and B, run BERT, take $C$, apply $W \in \mathbb{R}^{K \times H}$ and softmax, train with cross-entropy while updating all of BERT. The only new weights are the $K \times H$ matrix. For multiple choice (SWAG), encode each candidate pair separately and score each $C$ with one shared vector, softmaxed across candidates.
:::

## 10. Token-level Heads: SQuAD Spans, Unanswerable Questions and NER

**SQuAD v1.1** (100k crowd-sourced question/answer pairs) gives a question and a Wikipedia passage containing the answer; the task is to predict the answer **text span** in the passage. As Figure 1 shows, the question is sentence A and the passage is sentence B in one packed sequence. The only new parameters are a **start vector** $S \in \mathbb{R}^H$ and an **end vector** $E \in \mathbb{R}^H$ (unfortunately the paper reuses the letter $E$, which also names the input embeddings). The probability that word $i$ starts the answer is a dot product $S \cdot T_i$ followed by a softmax over all words in the paragraph, and the same with $E$ for the end. A candidate span from $i$ to $j$ is scored $S \cdot T_i + E \cdot T_j$, and the prediction is the **highest-scoring span with $j \ge i$**. Training maximizes the log-likelihoods of the correct start and end positions.

<Formula name="SQuAD span head" tag="Token-level QA" words="Score every paragraph token as a possible start with S and as a possible end with E; softmax each set of scores; the answer is the start-end pair with the largest summed score that doesn't end before it starts." remember="Two vectors, best i to j.">

$$
\begin{aligned}
\pred{P^{start}_i} &= \frac{e^{\param{S} \cdot \pred{T_i}}}{\sum_{j} e^{\param{S} \cdot \pred{T_j}}}, \qquad \pred{P^{end}_j} = \frac{e^{\param{E} \cdot \pred{T_j}}}{\sum_{k} e^{\param{E} \cdot \pred{T_k}}} \\
(\hat\imath, \hat\jmath) &= \arg\max_{j \ge i}\; \param{S} \cdot \pred{T_i} + \param{E} \cdot \pred{T_j} \\
\loss{\mathcal{L}} &= -\log \pred{P^{start}_{\data{s}}} - \log \pred{P^{end}_{\data{e}}}
\end{aligned}
$$

- $\pred{T_i}$: final vector of paragraph token $i$; sums run over paragraph tokens.
- $\param{S}, \param{E} \in \mathbb{R}^{H}$: start and end vectors, the only new parameters ($2H$ = 1,536 for BERT-base).
- $\data{s}, \data{e}$: gold start and end positions.

**Example:** this page's tiny BERT, question "11 20", paragraph 19 22 2 5 8 11 14 17 20 0: the best start is "11" ($S \cdot T_9$ = 6.07), the best end "20" ($E \cdot T_{12}$ = 6.24), span score 6.07 + 6.24 = 12.30, answer "11 14 17 20".

</Formula>

![SQuAD fine-tuning: a start vector S and an end vector E](/img/deep-learning/diagrams/bert-span-head.svg)

*How to read it:* the question tokens (grey) can't be the answer; each paragraph token's $T_i$ gets a start score and an end score, and the best pair with end after start is the answer.

<Animation src="bert/span-scoring" caption="Real start and end scores from the fine-tuned tiny BERT. One start bar and one end bar stand out, and the span between them is the answer." />

<SpanPredictor />

The paper fine-tuned for 3 epochs at learning rate 5e-5 with batch 32, after first fine-tuning on TriviaQA as modest data augmentation. Results (EM = exact match, F1 = token overlap):

| SQuAD v1.1 system | Dev EM | Dev F1 | Test EM | Test F1 |
|---|---|---|---|---|
| Human | – | – | 82.3 | 91.2 |
| #1 leaderboard ensemble (nlnet) | – | – | 86.0 | 91.7 |
| BERT-base (single) | 80.8 | 88.5 | – | – |
| BERT-large (single) | 84.1 | 90.9 | – | – |
| BERT-large (single + TriviaQA) | 84.2 | 91.1 | 85.1 | 91.8 |
| **BERT-large (ensemble + TriviaQA)** | **86.2** | **92.2** | **87.4** | **93.2** |

The single BERT model beat the top leaderboard **ensemble**, and without TriviaQA it lost only 0.1–0.4 F1.

**SQuAD 2.0** allows questions with **no answer** in the paragraph. BERT handles them with almost no change: an unanswerable question is treated as a span that **starts and ends at [CLS]**, so the start/end softmaxes now also include the [CLS] position. At prediction time, compare the no-answer score $s_{null} = S \cdot C + E \cdot C$ with the best non-null span score $\hat s_{i,j} = \max_{j \ge i} S \cdot T_i + E \cdot T_j$, and answer only if $\hat s_{i,j} > s_{null} + \tau$, where the threshold $\tau$ is tuned on the dev set to maximize F1. Fine-tuned for 2 epochs (learning rate 5e-5, batch 48, no TriviaQA), BERT-large scored **83.1 test F1**, +5.1 over the previous best system.

<Formula name="SQuAD 2.0 no-answer rule" tag="Unanswerable questions" words="The no-answer option is the span that starts and ends on CLS. Answer only if the best real span beats it by more than a tuned threshold." remember="Beat CLS by tau to answer.">

$$
\begin{aligned}
s_{null} &= \param{S} \cdot \pred{C} + \param{E} \cdot \pred{C} \\
\text{answer} &\iff \max_{j \ge i}\big(\param{S} \cdot \pred{T_i} + \param{E} \cdot \pred{T_j}\big) > s_{null} + \hyper{\tau}
\end{aligned}
$$

- $\pred{C}$: [CLS] vector; $\hyper{\tau}$: threshold chosen on the dev set.

**Example:** for the tiny BERT's example (trained without unanswerable questions), $s_{null}$ = −4.644 + −0.933 = −5.58, far below the best span's 12.30, so it answers for any $\tau$ below 17.88.

</Formula>

**Named entity recognition** (CoNLL-2003) is token tagging: every token gets a label such as B-PER or O. BERT uses a case-preserving WordPiece model, includes the maximal document context, and feeds the **first sub-token's** representation of each word to a token-level classifier over the NER labels, without a CRF on top. Fine-tuned, BERT-large reaches 96.6 dev / 92.8 test F1 and BERT-base 96.4 / 92.4.

:::analogy[Two highlighters]
Answering a SQuAD question is like handing a reader a green highlighter for "the answer starts here" and a pink one for "it ends here". The reader rates every word for each pen, and you take the best green-then-pink pair. For SQuAD 2.0 the reader may also put both pens on the title page ([CLS]), meaning "it isn't in here", and you only believe a highlighted answer if it clearly beats that.
:::

:::danger[Interview Trap: taking start and end argmaxes independently]
The two softmaxes are separate, but the prediction is not "argmax start, argmax end". You need the best **pair** with $j \ge i$ (implementations also cap the answer length, often at 30 tokens), otherwise you can predict an end before the start. For SQuAD 2.0 the [CLS] span is a competitor that must be beaten by $\tau$.
:::

:::tip[🎯 Interview angle]
**"How does BERT answer SQuAD questions?"** Pack [CLS] question [SEP] paragraph [SEP]; learn a start vector $S$ and end vector $E$; start probability of token $i$ is softmax over the paragraph of $S \cdot T_i$, end likewise with $E$; predict the span maximizing $S \cdot T_i + E \cdot T_j$ with $j \ge i$; train on the sum of the start and end log-likelihoods. For SQuAD 2.0, unanswerable questions point to [CLS], and you answer only if the best span beats $S \cdot C + E \cdot C$ by a threshold $\tau$ tuned on dev.
:::

## 11. The Feature-based Approach with BERT

Fine-tuning everything isn't always possible or desirable. The paper lists two advantages of the **feature-based** approach: not every task can be expressed with a Transformer encoder plus one layer, so some need their own architecture on top; and it's much cheaper to compute an expensive representation of the training data **once** and then run many experiments with cheap models on top.

To compare, the authors froze BERT-base, extracted activations from one or more layers (no fine-tuning of any BERT parameter), and fed them to a randomly initialized two-layer 768-dimensional BiLSTM before the NER classifier (CoNLL-2003 dev F1):

| Representation fed to the BiLSTM | Dev F1 |
|---|---|
| Fine-tuning BERT-base (for comparison) | 96.4 |
| Embeddings (layer 0) | 91.0 |
| Second-to-last hidden layer | 95.6 |
| Last hidden layer | 94.9 |
| Weighted sum of the last four hidden layers | 95.9 |
| **Concatenate the last four hidden layers** | **96.1** |
| Weighted sum of all 12 layers | 95.5 |

Concatenating the top four layers is only **0.3 F1** behind fine-tuning the whole model, so BERT works both ways. Two details are worth remembering: the second-to-last layer beats the last one (the last layer is the most specialized for the pre-training tasks), and the raw embedding layer alone is far behind (91.0): the contextualization is where the value is.

:::tip[🎯 Interview angle]
**"Can you use BERT without fine-tuning?"** Yes, as a frozen feature extractor. On CoNLL-2003 NER, concatenating the last four layers of frozen BERT-base into a BiLSTM got 96.1 dev F1 versus 96.4 for full fine-tuning. Reasons to do it: a task needs its own architecture, or you want to compute features once and run many cheap experiments. The second-to-last layer is often better than the last one.
:::

## 12. What the Ablations Show

**Pre-training tasks and bidirectionality (Table 5).** All rows use the BERT-base architecture, the same data, fine-tuning and hyperparameters. "No NSP" keeps MLM but drops NSP. "LTR & No NSP" is a left-to-right language model without NSP (left-only at fine-tuning too), i.e. a GPT-like model trained on BERT's data and input format. "+ BiLSTM" adds a randomly initialized BiLSTM on top of it during fine-tuning.

| Pre-training (BERT-base) | MNLI-m | QNLI | MRPC | SST-2 | SQuAD F1 |
|---|---|---|---|---|---|
| **BERT-base (MLM + NSP)** | **84.4** | **88.4** | **86.7** | **92.7** | **88.5** |
| No NSP | 83.9 | 84.9 | 86.5 | 92.6 | 87.9 |
| LTR & No NSP | 82.1 | 84.3 | 77.5 | 92.1 | 77.8 |
| + BiLSTM | 82.1 | 84.1 | 75.7 | 91.6 | 84.9 |

Removing NSP hurts QNLI, MNLI and SQuAD. Replacing MLM with a left-to-right LM (compare "No NSP" with "LTR & No NSP") hurts **everything**, with large drops on MRPC (−9.0) and SQuAD (−10.1 F1): for span prediction, a token's left-only vector has no idea what follows it. The BiLSTM recovers part of SQuAD (84.9) but hurts the GLUE tasks. The paper also explains why ELMo-style "train a left-to-right and a right-to-left model and concatenate" is worse: (a) twice as expensive as one bidirectional model; (b) unnatural for QA, since the right-to-left model couldn't condition the answer on the question; (c) strictly less powerful than a deep bidirectional model that uses both directions in every layer.

**Model size (Table 6).** All other settings fixed; dev accuracy averaged over 5 fine-tuning restarts; "LM (ppl)" is the masked-LM perplexity on held-out training data.

| $L$ | $H$ | $A$ | LM (ppl) | MNLI-m | MRPC | SST-2 |
|---|---|---|---|---|---|---|
| 3 | 768 | 12 | 5.84 | 77.9 | 79.8 | 88.4 |
| 6 | 768 | 3 | 5.24 | 80.6 | 82.2 | 90.7 |
| 6 | 768 | 12 | 4.68 | 81.9 | 84.8 | 91.3 |
| 12 | 768 | 12 | 3.99 | 84.4 | 86.7 | 92.9 |
| 12 | 1024 | 16 | 3.54 | 85.7 | 86.9 | 93.3 |
| 24 | 1024 | 16 | 3.23 | 86.6 | 87.8 | 93.7 |

Bigger is strictly better on all four columns, **even on MRPC with only 3,600 labeled training examples**. That was the paper's surprising claim: scaling to extreme model sizes helps very small downstream tasks too, *provided the model has been sufficiently pre-trained*, because fine-tuning adds only a few new parameters and reuses everything else.

**Training steps (Appendix C.1).** BERT-base gains almost 1.0% MNLI from 1M pre-training steps versus 500k; MLM converges slightly slower than left-to-right (it predicts only 15% of tokens per batch) but beats it in accuracy almost from the start.

:::tip[🎯 Interview angle]
**"Which parts of BERT matter most, according to the paper?"** Bidirectionality matters most: swapping the masked LM for a left-to-right LM (both without NSP, same data) costs 10.1 SQuAD F1 and 9.0 MRPC points. NSP adds a smaller gain on QNLI, MNLI and SQuAD. Size helps monotonically, even on 3,600-example MRPC, as long as pre-training is long enough. Fine-tuning is robust to the exact masking mix.
:::

## 13. Putting It Together: a Tiny BERT, Pre-trained and Fine-tuned

The implementation section builds a BERT whose classes follow the figures: `BertEmbeddings` (Figure 2: token + segment + learned position, LayerNorm, dropout), `Bert` (a stack of post-LN Transformer encoder layers with GELU and no mask), `PreTrainingHeads` (Mask LM with output weights tied to the token embeddings, and NSP on the pooled $C$), and the fine-tuning heads `SequenceClassifier` ($W$ on $C$) and `SpanHead` ($S$ and $E$). With $H$ = 64, $L$ = 4, $A$ = 4 the encoder has 202,880 parameters.

The "language" is synthetic so that it runs on a laptop CPU and every behaviour is checkable. A sentence is six numbers forming an arithmetic progression modulo 23 with a step from 1 to 5 (e.g. 7 12 17 22 4 9). For NSP, sentence B either continues A's progression (IsNext) or is a random progression (NotNext). Masking follows the paper exactly: 15% of positions (2 of the 12 here), 80/10/10. Pre-training uses AdamW (learning rate 1e-3, weight decay 0.01), 10% linear warmup then linear decay, batch 128, 4,000 steps, about 330 seconds on 2 CPU cores.

| Pre-training result (4,000 held-out pairs) | Value |
|---|---|
| MLM loss, step 1 → 4,000 | 3.300 → 0.147 (3.300 = ln 27, uniform over the vocabulary) |
| NSP loss, step 1 → 4,000 | 0.703 → 0.105 |
| MLM accuracy on chosen positions | 98.6% |
| NSP accuracy | **97.8%** (the paper's model: 97–98%) |
| Same model with a causal mask: NSP accuracy | 50.6% ($C$ sees only [CLS]) |

Then the pre-trained encoder is fine-tuned on two downstream tasks, and compared with the same architecture trained **from scratch** on the same labeled data:

- **Step-size classification** (sequence level, like SST-2): given one sentence, predict its step (5 classes) from $C$. This needs exactly what MLM taught: how neighbouring numbers relate.
- **Span extraction** (token level, like SQuAD): the question is two numbers, the paragraph is a 10-number progression containing both, and the answer is the span from the first to the second.

| Fine-tuning task | Labeled examples | Pre-trained BERT | From scratch |
|---|---|---|---|
| Step-size classification (10 epochs, lr 1e-3) | 100 | **82.9%** | 39.4% |
| Step-size classification (4 epochs, lr 5e-4) | 1,000 | **100%** | 56.8% |
| Span extraction, exact match (10 epochs, lr 5e-4) | 500 | 51.2% | **96.0%** |
| Span extraction, exact match (6 epochs, lr 5e-4) | 2,000 | 99.7% | 100% |

The classification rows are the paper's story in miniature: with 100 labels, pre-training more than doubles accuracy. The span rows are an honest counterexample. Our span task is pure **string matching** (find the paragraph tokens equal to the question tokens), which a fresh encoder learns fastest; the pre-trained encoder first has to bend representations tuned for predicting *missing* numbers, so with 500 examples it lags, and with 2,000 both solve it. Transfer helps when the downstream task needs what pre-training taught, which for real language (syntax, meaning, world knowledge) is almost always the case, and it is the reason BERT's largest GLUE gains came on the smallest datasets.

:::tip[🎯 Interview angle]
**"Walk me through implementing BERT pre-training from scratch."** Embeddings = token + segment + learned position, LayerNorm, dropout; $L$ bidirectional encoder layers (GELU FFN of $4H$); an MLM head (dense + GELU + LayerNorm, then logits tied to the token embeddings) applied at masked positions; an NSP head on the pooled [CLS] vector. Data: sample sentence pairs, 50% real next, mask 15% with 80/10/10, labels −100 elsewhere. Loss = MLM cross-entropy + NSP cross-entropy, AdamW with warmup and linear decay. Then swap the heads for a task head and fine-tune everything.
:::

## ⚙️ Going Deeper (BERT Internals)

**GELU.** BERT's feed-forward uses $\operatorname{GELU}(x) = x\,\Phi(x)$, where $\Phi$ is the standard normal CDF (often computed with the tanh approximation $0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$). It's a smooth ReLU: near zero it lets small negative values through with a small weight, which gives slightly better optimization than ReLU's hard corner.

**The pooler.** The released code passes $C$ through a dense layer with tanh (the "pooler", $H^2 + H$ parameters) before NSP, and HuggingFace's sequence classifiers reuse it. The paper's description simply says "$C$". Because the pooler was trained only for NSP, many practitioners prefer mean-pooling the $T_i$ for sentence embeddings, and models like Sentence-BERT fine-tune specifically for that.

**Where the parameters are.** In BERT-base, the embeddings are 23.8M (21.8%, mostly the 30,522 × 768 token table) and each encoder layer is 7.09M (attention 2.36M, feed-forward 4.72M). The feed-forward layers hold twice as many weights as attention. In BERT-large, the 24 layers are 90% of the 335M.

**Attention cost and the 512 limit.** Self-attention builds an $n \times n$ score matrix per head per layer: BERT-base at $n$ = 512 holds 12 × 512² ≈ 3.1M scores per layer. Inputs longer than 512 tokens must be truncated or split into overlapping windows with a stride (the standard SQuAD recipe for long passages), because there are no position embeddings beyond 511.

**Pre-LN vs post-LN.** BERT keeps the original post-LN layer (LayerNorm after the residual addition), and its learning-rate warmup (10,000 steps) is partly there to keep the early updates of a post-LN stack stable.

**The pre-train/fine-tune mismatch in practice.** The [MASK] token never appears downstream, which is why the 80/10/10 rule exists; Table 8 shows the problem is worst when the model can't adapt (feature-based). Later models attacked the same issue differently: ELECTRA replaces masked prediction with "was this token replaced?" on every token, which also fixes the 15%-of-tokens signal problem.

**What came after.** RoBERTa: same architecture, no NSP, dynamic masking (a new mask each epoch rather than fixed in preprocessing), bigger batches, more data and longer training. ALBERT: factorized embeddings and cross-layer parameter sharing for a smaller model, and sentence-order prediction instead of NSP. DistilBERT: a 6-layer student distilled from BERT-base. All keep the Figure 2 input and the fine-tuning recipe.

**Cased vs uncased, and the real vocabulary.** The released English models have an uncased vocabulary of 30,522 WordPieces (the paper says 30,000; about 1,000 slots are unused placeholders); the cased model is recommended when capitalization matters, as for NER, which is why the paper's NER experiment uses a case-preserving WordPiece model.

## 💻 PyTorch Implementation

:::tip[Run it yourself]
Verified on Python 3.13 and PyTorch 2.15 (CPU); every output below is copied from a real run. Blocks 1–4 are one file, `tiny_bert.py`; blocks 5 and 6 import it (block 6 needs the `pre.pt` saved by block 5). Each pre-training run takes about 330 seconds on 2 CPU cores, fine-tuning about a minute in total. Block 7 needs `pip install transformers` but downloads nothing.
:::

**Note on the reference material.** Notebook 17 is the paper itself, so there is no notebook code to correct. The paper gives no pseudo-code for the heads; the MLM transform (dense + GELU + LayerNorm, output tied to the token embeddings) and the tanh pooler on $C$ follow Google's released implementation, which the HuggingFace parameter counts in block 7 confirm.

### 1. The synthetic language, packing (Figure 2) and the 80/10/10 masking

A "sentence" is six numbers in an arithmetic progression mod 23; NSP pairs are built 50/50, and `mask_tokens` is the paper's procedure line for line.

```python title="tiny_bert.py (part 1)" showLineNumbers
"""Tiny BERT: pre-train with MLM + NSP on a synthetic 'language', then fine-tune.
Classes mirror BERT Figure 1 / Figure 2. CPU only."""
import math, random, time
import torch, torch.nn as nn, torch.nn.functional as F

# ---------- the synthetic language ----------
PAD, CLS, SEP, MASK = 0, 1, 2, 3
M = 23                      # number values 0..22 -> token ids 4..26
V = 4 + M                   # vocabulary size 27
SENT = 6                    # tokens per sentence
MAXLEN = 1 + 2 * SENT + 2   # [CLS] A [SEP] B [SEP] = 15

def progression(start, step, n):
    return [4 + (start + k * step) % M for k in range(n)]

def make_pair(rng):
    """Sentence A and B; 50% B continues A (IsNext=1), 50% random (NotNext=0)."""
    s, d = rng.randrange(M), rng.randint(1, 5)
    a = progression(s, d, SENT)
    if rng.random() < 0.5:
        return a, progression(s + SENT * d, d, SENT), 1
    while True:
        s2, d2 = rng.randrange(M), rng.randint(1, 5)
        b = progression(s2, d2, SENT)
        if b != progression(s + SENT * d, d, SENT):
            return a, b, 0

def pack(a, b=None):
    """Figure 2: tokens, segment ids (A=0, B=1) and positions."""
    tok = [CLS] + a + [SEP]
    seg = [0] * len(tok)
    if b is not None:
        tok += b + [SEP]
        seg += [1] * (len(b) + 1)
    return tok, seg

def mask_tokens(tok, rng, rate=0.15):
    """BERT masking: choose 15% of real tokens; 80% [MASK], 10% random, 10% unchanged.
    Returns corrupted input and labels (-100 = not predicted)."""
    inp, lab = list(tok), [-100] * len(tok)
    cand = [i for i, t in enumerate(tok) if t not in (CLS, SEP, PAD)]
    k = max(1, round(rate * len(cand)))
    for i in rng.sample(cand, k):
        lab[i] = tok[i]
        r = rng.random()
        # highlight-start
        if r < 0.8:
            inp[i] = MASK
        elif r < 0.9:
            inp[i] = rng.randrange(4, V)
        # highlight-end
        # else: keep the original token
    return inp, lab
```

### 2. BertEmbeddings and the bidirectional encoder

Token + segment + learned position embeddings, then a post-LN encoder stack with GELU. The `causal` switch exists only for the left-to-right ablation.

```python title="tiny_bert.py (part 2)" showLineNumbers
# ---------- the model (Figure 1 left / Figure 2) ----------
class BertEmbeddings(nn.Module):
    """Token + Segment + Position embeddings, summed, then LayerNorm and dropout."""
    def __init__(self, H, vocab=V, max_len=MAXLEN, p=0.1):
        super().__init__()
        self.token = nn.Embedding(vocab, H)
        self.segment = nn.Embedding(2, H)
        self.position = nn.Embedding(max_len, H)   # learned, not sinusoidal
        self.norm, self.drop = nn.LayerNorm(H), nn.Dropout(p)
    def forward(self, tok, seg):
        pos = torch.arange(tok.size(1), device=tok.device)
        # highlight-next-line
        return self.drop(self.norm(self.token(tok) + self.segment(seg) + self.position(pos)))

class Bert(nn.Module):
    """L Transformer encoder layers (bidirectional: no causal mask). Returns T_1..T_n; C = T[:, 0]."""
    def __init__(self, H=64, L=4, A=4, p=0.1, causal=False):
        super().__init__()
        self.causal = causal                       # True = left-to-right ablation (GPT-style)
        self.emb = BertEmbeddings(H, p=p)
        layer = nn.TransformerEncoderLayer(H, A, 4 * H, p, activation="gelu",
                                           batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, L, enable_nested_tensor=False)
    def forward(self, tok, seg):
        pad = tok.eq(PAD)
        n = tok.size(1)
        # highlight-next-line
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool), 1) if self.causal else None
        return self.encoder(self.emb(tok, seg), mask=causal, src_key_padding_mask=pad)
```

### 3. Pre-training heads (Mask LM + NSP) and the pre-training loop

The MLM logits reuse the token-embedding matrix; NSP reads the pooled $C$. The loss is the sum of the two mean losses, with warmup then linear decay.

```python title="tiny_bert.py (part 3)" showLineNumbers
class PreTrainingHeads(nn.Module):
    """Mask LM head (decoder tied to token embeddings) + NSP head on C."""
    def __init__(self, bert, H):
        super().__init__()
        self.bert = bert
        self.transform = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.LayerNorm(H))
        self.mlm_bias = nn.Parameter(torch.zeros(V))
        self.pooler = nn.Sequential(nn.Linear(H, H), nn.Tanh())
        self.nsp = nn.Linear(H, 2)
        for prm in self.parameters():               # BERT init: weights ~ N(0, 0.02)
            if prm.dim() > 1:
                nn.init.normal_(prm, std=0.02)
    def forward(self, tok, seg):
        T = self.bert(tok, seg)
        # highlight-next-line
        mlm_logits = self.transform(T) @ self.bert.emb.token.weight.T + self.mlm_bias
        nsp_logits = self.nsp(self.pooler(T[:, 0]))
        return mlm_logits, nsp_logits

def pretrain_batch(rng, n):
    toks, segs, labs, nsp = [], [], [], []
    for _ in range(n):
        a, b, is_next = make_pair(rng)
        tok, seg = pack(a, b)
        inp, lab = mask_tokens(tok, rng)
        toks.append(inp); segs.append(seg); labs.append(lab); nsp.append(is_next)
    t = lambda x: torch.tensor(x)
    return t(toks), t(segs), t(labs), t(nsp)

def warmup_linear(opt, warm, total):
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / (total - warm))))

def pretrain(steps=4000, H=64, L=4, A=4, seed=0, causal=False, log=True):
    torch.manual_seed(seed); rng = random.Random(seed)
    model = PreTrainingHeads(Bert(H, L, A, causal=causal), H)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
    sch = warmup_linear(opt, steps // 10, steps)
    for step in range(steps):
        tok, seg, lab, nsp = pretrain_batch(rng, 128)
        mlm_logits, nsp_logits = model(tok, seg)
        loss_mlm = F.cross_entropy(mlm_logits.reshape(-1, V), lab.reshape(-1), ignore_index=-100)
        loss_nsp = F.cross_entropy(nsp_logits, nsp)
        # highlight-next-line
        loss = loss_mlm + loss_nsp          # sum of the two means, as in A.2
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if log and (step == 0 or (step + 1) % 1000 == 0):
            print(f"step {step+1:5d}  mlm {loss_mlm.item():.3f}  nsp {loss_nsp.item():.3f}")
    return model
```

### 4. Fine-tuning heads: classifier on C, start/end vectors for spans

`SequenceClassifier` adds only $W$; `SpanHead` adds only $S$ and $E$ and masks non-paragraph positions. `best_span` searches all pairs with $j \ge i$.

```python title="tiny_bert.py (part 4)" showLineNumbers
# ---------- fine-tuning (Figure 1 right) ----------
class SequenceClassifier(nn.Module):
    """GLUE-style head: only new parameters are W in R^{K x H}; logits = C W^T."""
    def __init__(self, bert, H, K):
        super().__init__()
        self.bert, self.W = bert, nn.Linear(H, K)
    def forward(self, tok, seg):
        C = self.bert(tok, seg)[:, 0]
        return self.W(C)

class SpanHead(nn.Module):
    """SQuAD head: only new parameters are a start vector S and an end vector E."""
    def __init__(self, bert, H):
        super().__init__()
        self.bert = bert
        self.S = nn.Parameter(torch.randn(H) * 0.02)
        self.E = nn.Parameter(torch.randn(H) * 0.02)
    def forward(self, tok, seg):
        T = self.bert(tok, seg)                       # (B, n, H)
        # highlight-next-line
        start, end = T @ self.S, T @ self.E           # S.T_i and E.T_j
        para = seg.eq(1) & tok.ne(SEP)                # answers live in the paragraph only
        neg = torch.finfo(T.dtype).min
        return start.masked_fill(~para, neg), end.masked_fill(~para, neg)

def best_span(start, end, max_len=None):
    """argmax over i <= j of S.T_i + E.T_j (one example, 1-D tensors)."""
    score = start[:, None] + end[None, :]
    n = score.size(0)
    # highlight-next-line
    ok = torch.ones(n, n, dtype=torch.bool).triu()
    if max_len is not None:
        ok &= ~torch.ones(n, n, dtype=torch.bool).triu(max_len)
    score = score.masked_fill(~ok, float("-inf"))
    i, j = divmod(score.argmax().item(), n)
    return i, j, score[i, j].item()

def step_example(rng):
    """Single sentence ([CLS] A [SEP]) labelled with its step size 1..5 -> class 0..4."""
    d = rng.randint(1, 5)
    tok, seg = pack(progression(rng.randrange(M), d, SENT))
    return tok, seg, d - 1

PARA = 10
def span_example(rng):
    """Question = two values [q1, q2]; paragraph = 10-token progression; answer = span q1..q2."""
    para = progression(rng.randrange(M), rng.randint(1, 5), PARA)
    i, j = sorted(rng.sample(range(PARA), 2))
    tok, seg = pack([para[i], para[j]], para)
    off = 1 + 2 + 1                                   # [CLS] q1 q2 [SEP]
    return tok, seg, off + i, off + j
```

### 5. Pre-train bidirectionally and left-to-right, then evaluate

One packed and masked example first (Figure 1, left), then both models. The second run is identical except for the causal mask.

```python title="run_pretrain.py" showLineNumbers
import random, time, torch
from tiny_bert import *

NAMES = {PAD: "[PAD]", CLS: "[CLS]", SEP: "[SEP]", MASK: "[MASK]"}
show = lambda ids: " ".join(NAMES.get(t, str(t - 4)) for t in ids)

rng = random.Random(3)                          # one pre-training example, as in Figure 1 (left)
a, b, is_next = make_pair(rng)
tok, seg = pack(a, b)
inp, lab = mask_tokens(tok, rng)
print("tokens  ", show(tok))
print("segments", seg)
print("input   ", show(inp))
print("labels  ", [NAMES.get(t, t - 4) if t != -100 else "." for t in lab])
print("IsNext" if is_next else "NotNext")

def evaluate(model, seed=123):
    model.eval(); r = random.Random(seed)
    tok, seg, lab, nsp = pretrain_batch(r, 4000)
    with torch.no_grad():
        ml, nl = model(tok, seg)
    m = lab != -100
    print(f"  NSP accuracy {(nl.argmax(-1) == nsp).float().mean():.3f}   "
          f"MLM accuracy {(ml.argmax(-1)[m] == lab[m]).float().mean():.3f}")
    accs = []
    for pos in range(1, 1 + SENT):              # mask one token of sentence A (IsNext pairs)
        ok = 0
        for _ in range(2000):
            while True:
                a, b, y = make_pair(r)
                if y == 1: break
            t, s = pack(a, b); orig, t[pos] = t[pos], MASK
            with torch.no_grad():
                ml, _ = model(torch.tensor([t]), torch.tensor([s]))
            ok += ml[0, pos].argmax().item() == orig
        accs.append(ok / 2000)
    print("  masked-token accuracy by position:", " ".join(f"{x:.4f}" for x in accs))

# highlight-next-line
for causal in (False, True):
    print("\nleft-to-right (causal mask)" if causal else "\nbidirectional (BERT)")
    t0 = time.time()
    model = pretrain(4000, causal=causal)
    print(f"  {time.time() - t0:.0f} s, encoder parameters {sum(p.numel() for p in model.bert.parameters()):,}")
    evaluate(model)
    if not causal:
        torch.save(model.state_dict(), "pre.pt")

# Output:
# tokens   [CLS] 7 12 17 22 4 9 [SEP] 11 16 21 3 8 13 [SEP]
# segments [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
# input    [CLS] 7 12 17 22 4 9 [SEP] 11 [MASK] 21 3 [MASK] 13 [SEP]
# labels   ['.', '.', '.', '.', '.', '.', '.', '.', '.', 16, '.', '.', 8, '.', '.']
# NotNext
#
# bidirectional (BERT)
# step     1  mlm 3.300  nsp 0.703
# step  1000  mlm 0.608  nsp 0.420
# step  2000  mlm 0.176  nsp 0.159
# step  3000  mlm 0.213  nsp 0.179
# step  4000  mlm 0.147  nsp 0.105
#   336 s, encoder parameters 202,880
#   NSP accuracy 0.978   MLM accuracy 0.986
#   masked-token accuracy by position: 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000
#
# left-to-right (causal mask)
# step     1  mlm 3.300  nsp 0.703
# step  1000  mlm 1.065  nsp 0.693
# step  2000  mlm 0.795  nsp 0.693
# step  3000  mlm 0.824  nsp 0.691
# step  4000  mlm 0.926  nsp 0.693
#   331 s, encoder parameters 202,880
#   NSP accuracy 0.506   MLM accuracy 0.766
#   masked-token accuracy by position: 0.0465 0.1965 1.0000 1.0000 1.0000 1.0000
```

The bidirectional model recovers a masked token at every position of sentence A; the left-to-right model is at chance at position 1 (1/23 = 4.3%) and at one-in-five at position 2, and its NSP accuracy is a coin flip because $C$ sits at position 0 and, under a causal mask, sees only itself.

### 6. Fine-tune vs train from scratch

The same architecture and the same labeled data, starting from `pre.pt` or from random weights.

```python title="run_finetune.py" showLineNumbers
import random, torch, torch.nn.functional as F
from tiny_bert import *

T = lambda x: torch.tensor(x)

def encoder(pretrained):
    torch.manual_seed(1)
    heads = PreTrainingHeads(Bert(64, 4, 4), 64)
    if pretrained:
        heads.load_state_dict(torch.load("pre.pt"))
    # highlight-next-line
    return heads.bert                       # drop the MLM and NSP heads, keep the encoder

def batches(data, bs, rng):
    data = data[:]; rng.shuffle(data)
    for k in range(0, len(data), bs):
        yield data[k:k + bs]

def fine_tune_classifier(pretrained, n_train, epochs, lr, bs=16):
    rng = random.Random(7); train = [step_example(rng) for _ in range(n_train)]
    r2 = random.Random(99); test = [step_example(r2) for _ in range(2000)]
    torch.manual_seed(2); model = SequenceClassifier(encoder(pretrained), 64, K=5)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    for _ in range(epochs):
        model.train()
        for b in batches(train, bs, rng):
            tok, seg, y = map(T, zip(*b))
            loss = F.cross_entropy(model(tok, seg), y)         # log softmax(C W^T)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); tok, seg, y = map(T, zip(*test))
    with torch.no_grad():
        return (model(tok, seg).argmax(-1) == y).float().mean().item()

def fine_tune_span(pretrained, n_train, epochs, lr, bs=16):
    rng = random.Random(7); train = [span_example(rng) for _ in range(n_train)]
    r2 = random.Random(99); test = [span_example(r2) for _ in range(1000)]
    torch.manual_seed(2); model = SpanHead(encoder(pretrained), 64)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    for _ in range(epochs):
        model.train()
        for b in batches(train, bs, rng):
            tok, seg, i, j = map(T, zip(*b))
            s, e = model(tok, seg)
            loss = (F.cross_entropy(s, i) + F.cross_entropy(e, j)) / 2   # mean of start and end log-likelihoods
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); exact = 0
    with torch.no_grad():
        for tok, seg, i, j in test:
            s, e = model(T([tok]), T([seg]))
            exact += best_span(s[0], e[0])[:2] == (i, j)
    return exact / len(test), model

for n, ep, lr in [(100, 10, 1e-3), (1000, 4, 5e-4)]:
    print(f"step-size classifier, {n} labels: pre-trained {fine_tune_classifier(True, n, ep, lr):.3f}"
          f"  from scratch {fine_tune_classifier(False, n, ep, lr):.3f}")
for n, ep in [(500, 10), (2000, 6)]:
    em_pre, model = fine_tune_span(True, n, ep, 5e-4)
    em_new, _ = fine_tune_span(False, n, ep, 5e-4)
    print(f"span extraction, {n} labels: pre-trained EM {em_pre:.3f}  from scratch EM {em_new:.3f}")

r = random.Random(5); tok, seg, i, j = span_example(r)     # the example on this page
with torch.no_grad():
    s, e = model(T([tok]), T([seg]))
bi, bj, score = best_span(s[0], e[0])
print("question", [t - 4 for t in tok[1:3]], " paragraph", [t - 4 for t in tok[4:14]])
print(f"predicted span {bi}..{bj} = {[t - 4 for t in tok[bi:bj + 1]]}  gold {i}..{j}  score {score:.2f}")

# Output:
# step-size classifier, 100 labels: pre-trained 0.829  from scratch 0.394
# step-size classifier, 1000 labels: pre-trained 1.000  from scratch 0.568
# span extraction, 500 labels: pre-trained EM 0.512  from scratch EM 0.960
# span extraction, 2000 labels: pre-trained EM 0.997  from scratch EM 1.000
# question [11, 20]  paragraph [19, 22, 2, 5, 8, 11, 14, 17, 20, 0]
# predicted span 9..12 = [11, 14, 17, 20]  gold 9..12  score 12.30
```

### 7. Checking the paper's numbers

Parameter counts against HuggingFace's `BertModel`, the masking shares, and the pre-training arithmetic.

```python title="paper_numbers.py" showLineNumbers
from transformers import BertConfig, BertModel        # random weights: nothing is downloaded
import numpy as np

count = lambda m: sum(p.numel() for p in m.parameters())
base = BertModel(BertConfig())                        # defaults = BERT-base: L 12, H 768, A 12, vocab 30,522
large = BertModel(BertConfig(hidden_size=1024, num_hidden_layers=24,
                             num_attention_heads=16, intermediate_size=4096))
print(count(base), count(base.embeddings), count(base.encoder.layer[0]), count(base.pooler))
# 109482240 23837184 7087872 590592
print(count(large))                                   # 335141888
H = 768
print(12 * H * H + 13 * H, (30522 + 512 + 2) * H + 2 * H)      # 7087872 23837184   the formula matches

rng = np.random.default_rng(0); N = 1_000_000         # the 15% / 80-10-10 rule on a million tokens
chosen = rng.random(N) < 0.15
r = rng.random(N)
mask, rand, same = chosen & (r < 0.8), chosen & (r >= 0.8) & (r < 0.9), chosen & (r >= 0.9)
print([round(float(x.mean()), 4) for x in (chosen, mask, rand, same)])   # [0.15, 0.1201, 0.015, 0.0149]
print(256 * 512, round(128_000 * 1_000_000 / 3.3e9, 1), (512 / 128) ** 2)   # 131072 38.8 16.0
```

## 🚨 Interview Gotchas & Edge Cases

:::warning[The Most Common Question]
**"Explain BERT: architecture, pre-training and fine-tuning."**

BERT is a stack of Transformer **encoder** layers (base: $L$ = 12, $H$ = 768, $A$ = 12, 110M; large: 24 / 1024 / 16, 340M) with unmasked self-attention, so every token sees its left and right context in every layer. The input is [CLS] A [SEP] B [SEP], and each token's embedding is the sum of WordPiece token, segment (A/B) and learned position embeddings. It is pre-trained on BooksCorpus + Wikipedia with **Masked LM** (choose 15% of tokens; 80% [MASK], 10% random, 10% unchanged; predict the originals) and **Next Sentence Prediction** (50% real next sentence, classified from the [CLS] vector $C$). For a task you add one output layer (a $K \times H$ classifier on $C$, or start/end vectors $S$, $E$ over the $T_i$ for SQuAD) and fine-tune all parameters for 2–4 epochs at learning rate 2e-5 to 5e-5.
:::

- **15% chosen, not 15% masked:** 12% [MASK], 1.5% random, 1.5% unchanged; all 15% are predicted, the other 85% get no loss.
- **The loss is only at chosen positions,** so MLM learns from 15% of tokens per batch and converges a little slower than a left-to-right LM.
- **[CLS] is not a sentence embedding out of the box:** $C$ was trained only for NSP. Fine-tune it, or mean-pool the $T_i$, or use a sentence-embedding model.
- **Segment embeddings exist** (only two: A and B), and sentence A includes [CLS] and the first [SEP].
- **Position embeddings are learned and stop at 512;** longer texts need truncation or sliding windows.
- **BERT is encoder-only:** it can't generate text left to right the way GPT does; use it for understanding tasks (classification, tagging, extraction, retrieval).
- **"Sentence" in the paper means any span of text,** usually much longer than a linguistic sentence.
- **WordPiece:** 30,000 pieces in the paper (30,522 in the released uncased vocabulary); continuation pieces start with `##`; for word-level tags use the first sub-token.
- **Fine-tuning updates all weights;** the feature-based alternative freezes BERT (concat of the last four layers was only 0.3 F1 behind on NER).
- **SQuAD decoding needs $j \ge i$** and usually a maximum answer length; SQuAD 2.0 compares against the [CLS] no-answer score plus a threshold $\tau$.
- **Small datasets are unstable for BERT-large:** the paper used several random restarts and picked by dev score.
- **NSP was later dropped** (RoBERTa): its benefit in the paper is real but smaller than bidirectionality's, and other input constructions do as well.
- **Batch arithmetic:** 256 × 512 = 131,072 tokens, which the paper rounds to "128,000 tokens/batch".
- **The paper reuses the letter $E$** for input embeddings and for the SQuAD end vector.

## ⚡ Essential Interview Recall

A one-screen revision list. Every point is explained in the chapters above.

**The idea**
- Pre-train once on unlabeled text, fine-tune everything per task with one new output layer.
- Deep bidirectional: unmasked encoder self-attention, both contexts in every layer (GPT: left only; ELMo: two one-way LSTMs concatenated).

**Architecture**
- Transformer encoder only. Base: L = 12, H = 768, A = 12, FFN 3072, 110M (109.5M exactly). Large: 24 / 1024 / 16, FFN 4096, 340M.
- Per layer $12H^2 + 13H$ parameters; heads don't change the count. GELU, post-LN, dropout 0.1.

**Input (Figure 2)**
- [CLS] A [SEP] B [SEP]; WordPiece 30,000; token + segment + learned position (512) embeddings, summed, LayerNorm.
- $C$ = final [CLS] vector; $T_i$ = final vector of token $i$.

**Pre-training (Figure 1, left)**
- MLM: 15% chosen → 80% [MASK] / 10% random / 10% unchanged; cross-entropy at chosen positions only.
- NSP: 50% IsNext / 50% NotNext from $C$; 97–98% accuracy.
- Loss = MLM + NSP. BooksCorpus 800M + Wikipedia 2,500M words; batch 256 × 512; 1M steps (~40 epochs); Adam 1e-4, warmup 10k, linear decay; 90% of steps at length 128, last 10% at 512; 4 days.

**Fine-tuning (Figure 1, right; Figure 4)**
- Classification: $\operatorname{softmax}(CW^\top)$, $W \in \mathbb{R}^{K \times H}$. Multiple choice: one vector dotted with each candidate's $C$.
- SQuAD: start $\operatorname{softmax}(S \cdot T_i)$, end $\operatorname{softmax}(E \cdot T_j)$, best $S \cdot T_i + E \cdot T_j$ with $j \ge i$. SQuAD 2.0: answer iff best span $> S \cdot C + E \cdot C + \tau$.
- NER: classifier on the first sub-token of each word, no CRF.
- Hyperparameters: batch 16/32, lr 5e-5 / 3e-5 / 2e-5, 2–4 epochs; minutes to hours.

**Results and ablations**
- GLUE 80.5 (leaderboard), MNLI 86.7%, SQuAD v1.1 test F1 93.2, SQuAD 2.0 test F1 83.1, SWAG 86.3%.
- Left-to-right instead of MLM: −10.1 SQuAD F1, −9.0 MRPC; no NSP: QNLI 88.4 → 84.9.
- Bigger is better even on 3,600-example MRPC; feature-based (concat last 4 layers) 96.1 vs 96.4 fine-tuned NER F1.

**This page's tiny BERT**
- H = 64, L = 4, A = 4 (202,880 encoder parameters): NSP 97.8%, MLM 98.6%; causal-mask variant 4.65% at position 1 vs 100% bidirectional.
- 100 labels: pre-trained 82.9% vs scratch 39.4% on step-size classification.

:::speak[Explain BERT in 60 seconds]
BERT is the encoder half of the Transformer, pre-trained so it can be reused for almost any language-understanding task. Earlier models like GPT pre-trained as left-to-right language models, so a word's representation never saw what came after it, which hurts tasks like question answering. BERT uses unmasked self-attention so every token sees both sides in every layer, and because a bidirectional model can't be trained to predict the next word, it uses a masked language model instead: pick 15% of the tokens, replace most with a [MASK] token, a few with random words, leave a few as they are, and predict the originals. A second task, next sentence prediction, classifies from the [CLS] vector whether sentence B really follows A. The input is the sum of token, segment and position embeddings. To use it, I add one small layer, say a classifier on the [CLS] vector or start and end vectors for span extraction, and fine-tune everything for a few epochs at a learning rate around 2e-5 to 5e-5. That recipe set the state of the art on eleven tasks in 2018.
:::

## 📐 Formula Sheet

| Concept | Formula | In plain words |
|---|---|---|
| Input embedding | $\pred{x_i} = \operatorname{LN}(\param{E_{tok}}[\data{t_i}] + \param{E_{seg}}[\data{s_i}] + \param{E_{pos}}[i])$ | Token + segment + position, then LayerNorm |
| Parameters per layer | $12\hyper{H}^2 + 13\hyper{H}$ | Attention $4H^2$, FFN $8H^2$, plus biases and LayerNorms |
| Total parameters | $(\hyper{V} + 514)\hyper{H} + 2\hyper{H} + \hyper{L}(12\hyper{H}^2 + 13\hyper{H}) + \hyper{H}^2 + \hyper{H}$ | Embeddings + layers + pooler |
| Masked LM loss | $-\frac{1}{\lvert\mathcal{M}\rvert}\sum_{i \in \mathcal{M}} \log \operatorname{softmax}(\pred{T_i}\param{W_{tok}}^\top + \param{b})_{\data{t_i}}$ | Cross-entropy at the chosen 15% only |
| Masking shares | $0.15 \times (0.8, 0.1, 0.1) = (12\%, 1.5\%, 1.5\%)$ | [MASK], random, unchanged-but-graded |
| NSP loss | $-\log \operatorname{softmax}(\operatorname{tanh}(\pred{C}\param{W_p} + \param{b_p})\param{W_{nsp}})_{\data{y}}$ | Two-way classification from [CLS] |
| Pre-training loss | $\loss{\mathcal{L}} = \loss{\mathcal{L}_{MLM}} + \loss{\mathcal{L}_{NSP}}$ | Sum of the two means |
| Classification head | $\operatorname{softmax}(\pred{C}\param{W}^\top)$, $\param{W} \in \mathbb{R}^{K \times H}$ | One matrix on $C$ |
| Multiple choice | $\operatorname{softmax}_c(\param{v} \cdot \pred{C_c})$ | One vector, one score per candidate |
| Span start / end | $\pred{P^{start}_i} = \operatorname{softmax}_i(\param{S} \cdot \pred{T_i})$, $\pred{P^{end}_j} = \operatorname{softmax}_j(\param{E} \cdot \pred{T_j})$ | Two dot products per token |
| Span prediction | $\arg\max_{j \ge i}\, \param{S} \cdot \pred{T_i} + \param{E} \cdot \pred{T_j}$ | Best start-end pair, end not before start |
| Span loss | $-\log \pred{P^{start}_{\data{s}}} - \log \pred{P^{end}_{\data{e}}}$ | Gold start and end log-likelihoods |
| SQuAD 2.0 | answer iff $\max_{j \ge i}(\param{S} \cdot \pred{T_i} + \param{E} \cdot \pred{T_j}) > \param{S} \cdot \pred{C} + \param{E} \cdot \pred{C} + \hyper{\tau}$ | Beat the [CLS] no-answer span |
| GELU | $x\,\Phi(x) \approx 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$ | Smooth ReLU in the FFN |
| Attention cost ratio | $(512/128)^2 = 16$ | Why 90% of steps use length 128 |

## Interview Questions on the Paper

<details>
<summary><strong>Q1. What does BERT stand for, and what is the one-sentence idea?</strong></summary>

Bidirectional Encoder Representations from Transformers. Pre-train a deep bidirectional Transformer encoder on unlabeled text with a masked language model (plus next sentence prediction), then fine-tune it with one extra output layer for each downstream task.
</details>

<details>
<summary><strong>Q2. What are the two strategies for using pre-trained language representations, and which does BERT use?</strong></summary>

Feature-based (ELMo: frozen pre-trained vectors are extra features for a task-specific architecture) and fine-tuning (GPT: few new parameters, all pre-trained parameters trained on the task). BERT is a fine-tuning approach, though section 5.3 shows its features also work well when frozen.
</details>

<details>
<summary><strong>Q3. Why can't a deep bidirectional model be trained with a standard language-model objective?</strong></summary>

Because each word could indirectly see itself: in a multi-layer bidirectional encoder every position's output depends on its own input token through other positions, so predicting the input would be trivial. MLM removes the targets from the input instead of masking the attention.
</details>

<details>
<summary><strong>Q4. Describe the masking procedure exactly.</strong></summary>

After WordPiece tokenization, choose 15% of token positions uniformly at random. For each chosen position, replace the token with [MASK] 80% of the time, a random token 10%, or leave it unchanged 10%. Predict the original token at every chosen position with cross-entropy; no loss elsewhere.
</details>

<details>
<summary><strong>Q5. Why not always use [MASK]?</strong></summary>

[MASK] never appears during fine-tuning, so a model trained only on [MASK] inputs faces a mismatch. With random and unchanged tokens mixed in, the encoder can't tell which tokens are corrupted or graded and must keep a good representation of every token. Table 8: [MASK]-only drops feature-based NER from 94.9 to 94.0 F1.
</details>

<details>
<summary><strong>Q6. Doesn't the random replacement teach the model wrong things?</strong></summary>

It affects only 1.5% of all tokens (10% of 15%), which the paper found doesn't harm language understanding, and the label is still the original token, so the model learns to correct implausible words from context.
</details>

<details>
<summary><strong>Q7. What is Next Sentence Prediction and how is its data built?</strong></summary>

For each example, sentence B is the real next span after A 50% of the time (IsNext) and a random span from the corpus otherwise (NotNext). The [CLS] vector $C$ is classified into the two labels. The final model reaches 97–98% NSP accuracy.
</details>

<details>
<summary><strong>Q8. What are [CLS], [SEP] and segment embeddings for?</strong></summary>

[CLS] is the first token of every input; its final vector $C$ is the aggregate representation for classification (and NSP). [SEP] separates sentences and ends the sequence. Segment embeddings ($E_A$, $E_B$) are added to every token to mark which sentence it belongs to.
</details>

<details>
<summary><strong>Q9. How is BERT's input embedding built?</strong></summary>

Sum of the WordPiece token embedding, the segment embedding and a learned position embedding (positions 0–511), followed by LayerNorm and dropout (Figure 2).
</details>

<details>
<summary><strong>Q10. Give BERT-base and BERT-large sizes and why base has that size.</strong></summary>

Base: L = 12, H = 768, A = 12, 110M. Large: L = 24, H = 1024, A = 16, 340M. Feed-forward 4H. Base matches OpenAI GPT's size so the two can be compared; the main architectural difference is that GPT's attention is masked to the left.
</details>

<details>
<summary><strong>Q11. How is BERT fine-tuned for GLUE, and what are the new parameters?</strong></summary>

The input is packed as for pre-training; the final [CLS] vector $C$ goes into a classification layer $W \in \mathbb{R}^{K \times H}$ with loss $\log(\operatorname{softmax}(CW^\top))$; all parameters are fine-tuned (batch 32, 3 epochs, best of lr 5e-5 / 4e-5 / 3e-5 / 2e-5 on dev). $W$ is the only new weight matrix.
</details>

<details>
<summary><strong>Q12. How does BERT do SQuAD v1.1 and v2.0?</strong></summary>

v1.1: question = A, passage = B; learn start vector $S$ and end vector $E$; $P^{start}_i = \operatorname{softmax}(S \cdot T_i)$ over the paragraph, same for the end; predict the best span $S \cdot T_i + E \cdot T_j$ with $j \ge i$; loss = sum of gold start and end log-likelihoods. v2.0: unanswerable questions have start = end = [CLS]; predict an answer only when the best span score exceeds $S \cdot C + E \cdot C + \tau$, with $\tau$ tuned on dev.
</details>

<details>
<summary><strong>Q13. What did the ablations show?</strong></summary>

Removing NSP hurts QNLI (88.4 → 84.9), MNLI and SQuAD. Replacing MLM by a left-to-right LM hurts all tasks, especially MRPC and SQuAD (77.8 F1 vs 87.9 without NSP); a BiLSTM on top helps SQuAD but not GLUE. Larger models improve every task, even MRPC with 3,600 examples. 1M pre-training steps beat 500k by about 1% MNLI. Fine-tuning is robust to the masking mix.
</details>

<details>
<summary><strong>Q14. Why does pre-training use sequence length 128 for 90% of the steps?</strong></summary>

Attention cost is quadratic in length, so 512-token sequences are disproportionately expensive (16× the attention scores of 128). The last 10% of steps at 512 are there to learn the position embeddings for positions beyond 128.
</details>

<details>
<summary><strong>Q15. When would you use BERT as a feature extractor instead of fine-tuning, and which layers?</strong></summary>

When the task needs its own architecture, or when you want to compute expensive features once and run many cheap experiments. On CoNLL-2003 NER, concatenating the last four hidden layers of frozen BERT-base gave 96.1 dev F1, only 0.3 behind fine-tuning (96.4); the second-to-last layer alone (95.6) beat the last (94.9).
</details>

<details>
<summary><strong>Q16. How does BERT differ from GPT beyond the attention mask?</strong></summary>

GPT trained on BooksCorpus only; BERT on BooksCorpus and Wikipedia. GPT introduced [SEP] and [CLS] only at fine-tuning; BERT learns them and the A/B segment embeddings during pre-training. GPT used batches of 32,000 words, BERT 128,000 (both 1M steps). GPT used one fine-tuning learning rate (5e-5) for all tasks; BERT picks a task-specific one on dev. The ablations attribute most of the gains to bidirectionality and the two pre-training tasks.
</details>
