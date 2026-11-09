---
title: "Hugging Face Pipelines and What They Hide"
sidebar_position: 1
---

import {BertTokenizerLive, LogitsToConfidence} from '@site/src/components/Widgets';

:::tip[TL;DR (30-Second Interview Pitch)]
Hugging Face `transformers` turns a pre-trained model into one line: `pipeline("sentiment-analysis")("I love this movie.")` returns `{'label': 'POSITIVE', 'score': 0.99987}`. With no model named, it downloads the task's default, **`distilbert-base-uncased-finetuned-sst-2-english`** (a 6-layer DistilBERT fine-tuned on SST-2 movie reviews, 66,955,010 parameters). Inside, a pipeline does three things: **preprocess** (an `AutoTokenizer` lower-cases, splits into WordPieces, adds [CLS]/[SEP] and returns `input_ids` and `attention_mask`), **forward** (an `AutoModelForSequenceClassification` runs the encoder and a small head on the [CLS] vector, giving **logits** of shape `[batch, 2]`), and **postprocess** (softmax, argmax, `config.id2label`). Writing those steps by hand gives you control over batching, padding, truncation, devices and fine-tuning. The `score` is a softmax probability, which for two classes is the sigmoid of the logit margin, so it saturates near 1 quickly: high confidence is not correctness.
:::

<FormulaKey />

:::info[Where this page sits]
**Before:** [Stacked Encoder Blocks: a Mini Sentiment Project](../06-attention-and-transformers/06-stacked-encoder-blocks-mini-project.md) built a 27,010-parameter sentiment encoder by hand and ended where pre-trained encoders begin.
**This page:** notebook 19, *Hugging_Face_Project_1*. Its title promises "From Pipeline Preview to Document Question Answering", but the notebook covers only the pipeline and what it hides (its sections 0–2); the question-answering part is notebook 20. This page also absorbs the two Hugging Face preview cells from the end of notebook 18, which failed there with `ModuleNotFoundError`.
**Next:** notebook 20, extractive question answering over a PDF with a DistilBERT model fine-tuned on SQuAD.
:::

## 🧭 Start Here

On the previous page we built every piece of a sentiment classifier ourselves: a vocabulary, ids, padding, a mask, embeddings, encoder blocks, pooling and a head, and trained it on 18 phrases. Real NLP systems almost never start from random weights. They download an encoder that has already been **pre-trained** on billions of words (BERT, DistilBERT, RoBERTa, …), usually already **fine-tuned** for the task, and run it. **Hugging Face `transformers`** is the library that makes this routine: the **Hub** hosts hundreds of thousands of models, and a handful of classes load the right tokenizer and model for any of them by name.

The fastest way in is a **pipeline**: give it a task name, get a callable that takes raw text and returns a readable answer. That is perfect for a first try and terrible for understanding, debugging or production, because everything we built by hand last time is hidden inside it. So the notebook does what we'll do here: run the pipeline, notice what it can't tell us (a near-certain score on a sentence it might be getting wrong), then **open the box** and execute the same three steps manually with `AutoTokenizer` and `AutoModelForSequenceClassification`, until the manual numbers match the pipeline's exactly.

Every output on this page comes from re-running the notebook's code (transformers 5.22.2, PyTorch 2.15, CPU). The pipeline downloads the same model revision the notebook used (`714eb0f`), and every score matches the notebook's Colab run to all printed digits.

| New concept | One-line meaning |
|---|---|
| **Hugging Face Hub** | Online registry of pre-trained models, tokenizers and datasets, addressed by name (`user/model`) |
| **`pipeline(task)`** | One callable that does tokenization, the model forward pass and postprocessing for a task |
| **Default model** | The model a pipeline downloads when you don't name one (here DistilBERT fine-tuned on SST-2) |
| **`AutoTokenizer`** | Loads the exact tokenizer a model was trained with, from its name |
| **`AutoModelFor…`** | Loads the model architecture plus the right task head (`…SequenceClassification`, `…QuestionAnswering`) |
| **`input_ids` / `attention_mask`** | Token ids (with [CLS], [SEP], padding) and 1/0 marks for real tokens vs padding |
| **Logits** | Raw, unnormalized class scores the model outputs |
| **`id2label`** | The model config's map from class index to label name |
| **DistilBERT** | A 6-layer BERT distilled from BERT-base: about 40% smaller and 60% faster, keeping about 97% of its GLUE performance |

:::analogy[A vending machine and its back panel]
A pipeline is a vending machine: put in a sentence, out drops a labelled can. That's all you need until a can comes out wrong. Then you open the back panel and find the same parts we built last time: a coin reader (tokenizer), a mechanism (the model) and a label printer (postprocessing). Knowing the panel is what lets you fix, tune or rebuild the machine.
:::

## 1. From a Hand-built Encoder to a Pre-trained One

Our mini encoder had 27,010 parameters and had seen 18 phrases. The model the pipeline loads has **66,955,010** parameters: 66,362,880 in a DistilBERT encoder that was pre-trained on English text, and 592,130 in a classification head trained on **SST-2** (the binary Stanford Sentiment Treebank, sentences from movie reviews labelled positive or negative). The skeleton is the same as ours: token + position embeddings, a stack of encoder blocks, a head that turns contextual vectors into two logits. Everything else is scale and training data.

**DistilBERT** (Sanh et al., 2019) is BERT-base compressed by **knowledge distillation**: a 6-layer student is trained to imitate the outputs of the 12-layer BERT teacher, keeping about 97% of BERT's language-understanding performance while being 40% smaller and 60% faster. It keeps BERT's WordPiece vocabulary and [CLS]/[SEP] format, but drops the **segment (token-type) embeddings** and the pooler. This is why the pipeline uses it as the default: good accuracy at a size that runs on a laptop CPU.

The name on the Hub encodes its history: `distilbert-base-uncased` (lower-cased vocabulary) `-finetuned-sst-2-english`. Pre-training happened once, at great cost; fine-tuning on SST-2 adapted it to sentiment; we just download and run it. That division of labour (pre-train once, fine-tune cheaply, reuse everywhere) is the BERT recipe from two pages ago, delivered as a library.

| | Our mini encoder (notebook 18) | `distilbert-base-uncased-finetuned-sst-2-english` |
|---|---|---|
| Tokenizer | whitespace, 35 entries | WordPiece, 30,522 entries, lower-cased |
| Max length | 6 | 512 |
| Blocks × width × heads | 3 × 32 × 4 | 6 × 768 × 12 |
| FFN | 64, ReLU | 3072, GELU |
| Sentence vector | masked mean pooling | final [CLS] vector → `pre_classifier` (768 × 768) → ReLU → dropout 0.2 → `classifier` (768 × 2) |
| Parameters | 27,010 | 66,955,010 |
| Training | 18 phrases from scratch | pre-trained, then fine-tuned on SST-2 (≈67k sentences) |

:::tip[🎯 Interview angle]
**"Why use a pre-trained model instead of training a Transformer classifier from scratch?"** Pre-training on billions of words has already taught the encoder syntax, word meaning and context, which a small labeled dataset could never teach (our 18-phrase model couldn't generalize reliably). You reuse that knowledge and only fine-tune a small head plus the encoder on your task, which needs far less data and compute. DistilBERT is a common default because it keeps about 97% of BERT's quality at 60% of its size.
:::

## 2. Setting Up, and Why Notebook 18's Cells Failed

The notebook installs only what this lecture needs: `transformers` (pipelines, tokenizers, models), `accelerate` (device placement), `sentencepiece` (needed by some non-WordPiece tokenizers), `safetensors` (the safe, fast weight format the Hub serves), `pypdf` and `pymupdf` (for the PDF work in notebook 20), and `protobuf<6.0.0` (pinned to avoid a conflict with packages pre-installed on Colab). It then checks the runtime: in Colab, Python 3.12, PyTorch 2.11 with CUDA and a Tesla T4 GPU; the lecture also runs on CPU, just slower.

Notebook 18's last two cells had failed on the instructor's local machine with `ModuleNotFoundError: Could not import module 'pipeline'. Are this object's requirements defined correctly?`. That message comes from `transformers`' **lazy import** system: `from transformers import pipeline` only resolves the real module when it's first used, and if importing it fails (typically because a backend such as PyTorch is missing, or its version is incompatible with the installed `transformers`), you get this generic error instead of the underlying one. The notebook's fix, starting a clean Colab runtime with freshly installed packages, is the right one. Locally, check `python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"` and upgrade them together.

The first run of a pipeline is slow because it downloads the model (`config.json`, `vocab.txt`, `tokenizer_config.json` and a 268 MB `model.safetensors`) into the local cache (`~/.cache/huggingface`). Later runs load from disk. Colab also warns that no `HF_TOKEN` is set: public models download fine without one, with lower rate limits; a token is required only for gated or private models.

:::danger[Interview Trap: "No model was supplied, defaulted to …"]
The pipeline printed: *No model was supplied, defaulted to distilbert/distilbert-base-uncased-finetuned-sst-2-english and revision 714eb0f. Using a pipeline without specifying a model name and revision in production is not recommended.* Defaults can change between library versions, and a model's weights can change on the Hub. In production, pin both: `pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", revision="714eb0f")`.
:::

:::tip[🎯 Interview angle]
**"Your pipeline import fails with 'Could not import module pipeline'. What do you check?"** That's transformers' lazy import hiding the real error, almost always an environment problem: is PyTorch (or TensorFlow) installed, and are its version and the transformers version compatible? Check both versions, import torch directly to see the real error, and reinstall them together in a clean environment. The first run being slow is just the model download into the cache.
:::

## 3. `pipeline()`: Inference in One Line

`pipeline("sentiment-analysis")` builds an object that you call with a string or a list of strings. For each input it returns a dictionary with two keys: **`label`**, the predicted class name, and **`score`**, the probability the model assigns to that class. On the notebook's five sentences:

| Sentence | label | score |
|---|---|---|
| I love this movie. | POSITIVE | 0.999874 |
| This movie is terrible. | NEGATIVE | 0.999726 |
| The story was not bad. | POSITIVE | 0.998976 |
| The acting was boring. | NEGATIVE | 0.999776 |
| The film was surprisingly good. | POSITIVE | 0.999860 |

All five are right, including the negation "not bad", which our 18-phrase model could only get right because that exact phrase was in its training data. The output is already post-processed: no logits, no softmax, no ids. Passing a list sends the sentences through the model as a batch. Pipelines also take `device=0` (first GPU), `batch_size=…`, `truncation=True`, and `top_k=None` to return the score of every class instead of just the winner.

Pipelines exist for many tasks with the same interface: `"text-classification"` (the same as `"sentiment-analysis"`), `"question-answering"`, `"token-classification"` (NER), `"summarization"`, `"translation"`, `"text-generation"`, `"fill-mask"`, `"zero-shot-classification"`, `"feature-extraction"` and more. Each task string maps to a default model and to the right `AutoModelFor…` class.

:::tip[🎯 Interview angle]
**"What does `pipeline('sentiment-analysis')` do at a high level, and what does it return?"** It loads a default (or named) pre-trained model and its tokenizer and wraps tokenization, padding and attention masks, the forward pass, softmax and label decoding into one call. It returns one `{'label': …, 'score': …}` per input, where score is the softmax probability of the predicted label.
:::

## 4. Confidence Is Not Correctness

The `score` looks like a measure of how sure the model is that it's right. It isn't. It's the softmax probability of the top class, which only says how strongly the model prefers that class *given what it learned*. The notebook makes the point with harder sentences:

| Sentence | label | score | Comment |
|---|---|---|---|
| The movie was not terrible. | POSITIVE | 0.984081 | Lowest score of all: double negative, mildly positive |
| The movie was not good. | NEGATIVE | 0.999773 | Negation handled confidently |
| The movie was okay, but not amazing. | NEGATIVE | 0.995725 | Mixed feeling forced into one of two classes |
| I expected more from this film. | NEGATIVE | 0.991327 | Implicit disappointment, no sentiment word |
| It was fine, I guess. | POSITIVE | **0.999756** | Lukewarm, yet near-certain |

"It was fine, I guess." is lukewarm at best, yet it gets 0.99976, higher than "not terrible". The model has only two classes, so it can't say "neutral"; it was fine-tuned on movie-review sentences with clear polarity; and its score saturates (chapter 7). A model can be **confidently wrong**, especially on ambiguous, sarcastic, domain-specific or out-of-distribution text. In practice you check **calibration** on held-out data, add a neutral class or a threshold below which you defer to a human, and evaluate on examples from your own domain.

:::analogy[A confident friend with two words]
Ask a friend who is only allowed to answer "loved it" or "hated it" what they thought of a so-so film. They will pick one, and they might say it loudly. The loudness tells you about their vocabulary, not about the film.
:::

:::danger[Interview Trap: "score 0.9998 means 99.98% chance it is right"]
Softmax outputs are not calibrated probabilities of being correct. Fine-tuned Transformers are often over-confident, and a two-class model has no way to express "neither". Measure accuracy and calibration on held-out data before trusting scores as probabilities.
:::

:::tip[🎯 Interview angle]
**"Why shouldn't we treat a high confidence score as a guarantee?"** A high score only means the model strongly prefers one class; it is not a calibrated probability of being correct. Models can be confidently wrong on ambiguous, sarcastic, domain-specific or out-of-distribution text ("It was fine, I guess." → POSITIVE 0.99976). Use calibration checks, thresholds or a neutral class, and domain evaluation.
:::

## 5. Opening the Pipeline: Three Stages

A pipeline is three functions in a row, and the rest of this page writes each one by hand.

![What pipeline("sentiment-analysis") does: preprocess, forward, postprocess](/img/deep-learning/diagrams/hf-pipeline-internals.svg)

*How to read it:* text enters on the left. **Preprocess** (green box 1) is the tokenizer: it produces `input_ids` and `attention_mask`. **Forward** (box 2) runs the model in inference mode and produces logits. **Postprocess** (box 3) applies softmax, takes the argmax and maps the index to a label name with the model's config.

The two classes that do the work are loaded with **Auto classes**. `AutoTokenizer.from_pretrained(name)` reads the model's `tokenizer_config.json` and builds the matching tokenizer class (here `BertTokenizer`, because DistilBERT uses BERT's WordPiece vocabulary). `AutoModelForSequenceClassification.from_pretrained(name)` reads `config.json`, sees `"model_type": "distilbert"`, builds `DistilBertForSequenceClassification` and loads the weights. "Auto" means you never have to know the concrete class, so the same code works for BERT, RoBERTa or DeBERTa just by changing the name. The task suffix (`ForSequenceClassification`, `ForQuestionAnswering`, `ForTokenClassification`, …) decides which head sits on top of the shared encoder.

`model.eval()` switches off dropout (the head has dropout 0.2, the encoder 0.1), so the same input always gives the same output; `torch.no_grad()` stops PyTorch from recording the computation for backpropagation, which saves memory and time. The model's **`config.id2label`** is `{0: 'NEGATIVE', 1: 'POSITIVE'}`: it is what lets a pipeline print a word instead of an index.

<Animation src="hf/pipeline-steps" caption="The three stages on one sentence, with the real ids, logits and probabilities. Every number here is what the pipeline computes internally." />

:::tip[🎯 Interview angle]
**"What do `AutoTokenizer` and `AutoModelForSequenceClassification` do?"** They read a model's config files on the Hub and instantiate the right concrete tokenizer and architecture (here `BertTokenizer` and `DistilBertForSequenceClassification`), loading the matching vocabulary and weights. The `For…` suffix selects the task head. Using Auto classes makes code model-agnostic: swap the name to try another checkpoint.
:::

## 6. The Tokenizer: `input_ids`, `attention_mask` and Special Tokens

Calling `tokenizer("The movie was surprisingly good.", return_tensors="pt")` returns three tensors. **`input_ids`** is `[[101, 1996, 3185, 2001, 10889, 2204, 1012, 102]]`: one id per token, as a `[1, 8]` PyTorch tensor. Converting the ids back gives `['[CLS]', 'the', 'movie', 'was', 'surprisingly', 'good', '.', '[SEP]']`, and decoding gives the string `[CLS] the movie was surprisingly good. [SEP]`. The tokenizer lower-cased the text (this is an *uncased* model), split off the full stop as its own token, and wrapped the sentence in the special tokens the model was trained with: **[CLS]** (id 101) at the start, whose final vector the classifier reads, and **[SEP]** (id 102) at the end. **`attention_mask`** is all ones, because a single sentence needs no padding.

**Tokens are not words.** Words that aren't in the 30,522-entry vocabulary are split greedily into the longest known pieces, continuation pieces marked with `##`: "Transformers tokenization is unbelievably subword-based!" becomes `transformers`, `token`, `##ization`, `is`, `un`, `##bel`, `##ie`, `##va`, `##bly`, `sub`, `##word`, `-`, `based`, `!`. Common words are single tokens, rare ones become several, and nothing is ever out of vocabulary. This is the WordPiece scheme from the [BERT page](../06-attention-and-transformers/05-bert-pretraining-and-fine-tuning.md).

The third tensor, **`token_type_ids`** (all zeros here), is BERT's segment id, which marks sentence A vs B in a pair. The tokenizer returns it because it is a `BertTokenizer`, but **DistilBERT has no segment embeddings**: passing it changes nothing (the logits are identical with or without it). For a batch, the tokenizer pads shorter sequences with **[PAD]** (id 0) and sets their mask to 0: tokenizing "The movie was surprisingly good." and "Bad." together gives `[101, 2919, 1012, 102, 0, 0, 0, 0]` with mask `[1, 1, 1, 1, 0, 0, 0, 0]` for the second.

<BertTokenizerLive />

| Output | Meaning | Example ("The movie was surprisingly good.") |
|---|---|---|
| `input_ids` | Vocabulary id of every token, with [CLS] 101 and [SEP] 102 | `[101, 1996, 3185, 2001, 10889, 2204, 1012, 102]` |
| `attention_mask` | 1 = real token, 0 = padding | `[1, 1, 1, 1, 1, 1, 1, 1]` |
| `token_type_ids` | Segment A (0) or B (1); ignored by DistilBERT | `[0, 0, 0, 0, 0, 0, 0, 0]` |

:::danger[Interview Trap: using a different tokenizer from the model's]
Ids only mean something for the vocabulary the model was trained with: id 2204 is "good" for `bert-base-uncased`, something else for RoBERTa. Always load the tokenizer from the same checkpoint name as the model, and don't lower-case input yourself for a *cased* model (or forget to for a custom uncased setup).
:::

:::tip[🎯 Interview angle]
**"What does the tokenizer return for a BERT-style model, and what are [CLS] and [SEP]?"** `input_ids` (vocabulary ids including special tokens), `attention_mask` (1 for real tokens, 0 for padding) and, for BERT tokenizers, `token_type_ids` (segment A/B). [CLS] starts every sequence and its final vector is used for classification; [SEP] ends a sentence and separates pairs. Subword tokenization means tokens aren't words: rare words split into `##` pieces.
:::

## 7. Logits, Softmax and the Label

Calling `model(**encoded)` inside `torch.no_grad()` returns a `SequenceClassifierOutput` whose `logits` are `tensor([[-4.2341, 4.6189]])`, shape `[1, 2]`: one row per input, one column per class (index 0 = NEGATIVE, 1 = POSITIVE). **Logits** are raw scores on any scale. Softmax turns them into probabilities, `[1.4294e-04, 9.9986e-01]`; the argmax is 1, and `id2label[1]` is POSITIVE. That is exactly the pipeline's `{'label': 'POSITIVE', 'score': 0.99986}`.

Inside the model, the logits come from one place: the final hidden state of the **[CLS]** token. DistilBERT outputs a `[1, 8, 768]` tensor (one 768-dimensional vector per token); the head takes position 0, applies `pre_classifier` (768 → 768), ReLU, dropout and `classifier` (768 → 2). Recomputing that by hand from `output_hidden_states` reproduces the logits exactly.

With two classes, softmax has a simple closed form: the probability of class 1 is the **sigmoid of the margin** $z_1 - z_0$. For this sentence the margin is $4.6189 - (-4.2341) = 8.853$, and $\sigma(8.853) = 0.99986$. The sigmoid saturates fast: a margin of 4.6 already gives 0.99, a margin of 9.2 gives 0.9999. That is why almost every score from this model starts with 0.99: once the margin passes about 5, the score barely moves, whatever the model "thinks".

<Formula name="Softmax over two logits" tag="Postprocessing" words="Exponentiate the class scores and normalize them to sum to 1. With two classes this equals the sigmoid of the difference between the two logits." remember="Two classes: sigmoid of the margin.">

$$
\begin{aligned}
\pred{p_i} &= \frac{e^{\pred{z_i}}}{\sum_j e^{\pred{z_j}}} \\
\pred{p_1} &= \frac{e^{\pred{z_1}}}{e^{\pred{z_0}} + e^{\pred{z_1}}} = \frac{1}{1 + e^{-(\pred{z_1} - \pred{z_0})}} = \sigma(\pred{z_1} - \pred{z_0})
\end{aligned}
$$

- $\pred{z_0}, \pred{z_1}$: logits for NEGATIVE and POSITIVE; $\pred{p_i}$: their probabilities.
- The pipeline's `score` is $\max_i \pred{p_i}$ and its `label` is $\operatorname{id2label}[\arg\max_i \pred{p_i}]$.

**Example:** logits $[-4.2341, 4.6189]$: margin 8.853, $\pred{p_1} = 0.99986$, label POSITIVE.

</Formula>

<Formula name="Sequence-classification head of DistilBERT" tag="Forward" words="Take the final vector of the CLS token, pass it through a 768 by 768 layer, ReLU, dropout and a 768 by 2 layer to get two logits." remember="CLS vector, two linear layers.">

$$
\pred{z} = \param{W_c}\,\operatorname{Dropout}\big(\operatorname{ReLU}(\param{W_p}\,\pred{h_{[CLS]}} + \param{b_p})\big) + \param{b_c}
$$

- $\pred{h_{[CLS]}} \in \mathbb{R}^{768}$: final hidden vector at position 0.
- $\param{W_p}$: $768 \times 768$ (`pre_classifier`); $\param{W_c}$: $2 \times 768$ (`classifier`); dropout 0.2 is off in `eval()`.

**Example:** head parameters $768 \cdot 768 + 768 + 768 \cdot 2 + 2 = 592{,}130$; encoder 66,362,880; total 66,955,010.

</Formula>

![DistilBertForSequenceClassification](/img/deep-learning/diagrams/distilbert-sequence-classification.svg)

*How to read it:* WordPiece tokens become word + learned position embeddings (no segment embeddings), pass through 6 Transformer blocks, and only the [CLS] position's final vector feeds the two-layer head.

<LogitsToConfidence />

<Animation src="hf/confidence-saturation" caption="Real margins from this model on four sentences. Past a margin of about 5 the curve is flat, so lukewarm and enthusiastic sentences get almost the same score." />

:::tip[🎯 Interview angle]
**"What are logits and how does the pipeline turn them into a label and score?"** Logits are the head's raw, unnormalized class scores, shape `[batch, num_labels]`. Softmax normalizes them into probabilities; argmax picks the class; `config.id2label` names it; the score is the winning probability. For two classes that probability is the sigmoid of the logit margin, so scores saturate near 1 once the margin exceeds about 5.
:::

## 8. Batching, Padding and a Clean Prediction Function

The notebook ends by wrapping the manual steps in `manual_sentiment_prediction(text, tokenizer, model)`, which tokenizes one text, runs the model, applies softmax and returns the label and both probabilities. It reproduces the pipeline's outputs (I love this movie → POSITIVE 0.9999; This movie is terrible → NEGATIVE 0.9997). The natural improvement is to predict a **batch**: tokenize all texts at once with `padding=True, truncation=True`, run one forward pass, and read one row per text. That is what the pipeline does when you pass a list, and it's much faster on a GPU than a Python loop over sentences.

Padding is safe *only* together with the attention mask. Run "Bad." alone and padded inside a batch with a longer sentence: with the mask, the logits are identical (`[4.6731, -3.7037]` both times). Drop the mask (pass only `input_ids`) and the four [PAD] positions are attended to like real tokens: the logits become `[4.5069, -3.6528]`. Here the label doesn't change, but the representation does, and on longer padding or borderline inputs the prediction can flip.

Truncation matters for long inputs. DistilBERT has 512 position embeddings, so 512 tokens is a hard limit: a 200-sentence review tokenizes to 802 tokens, `transformers` warns that the sequence is longer than the maximum, and running it untruncated would crash on position 512. With `truncation=True` the tokenizer keeps the first 512 tokens (510 plus [CLS] and [SEP]), and **the model never sees the rest**. Long documents need a different strategy (chunking), which is exactly where notebook 20 starts.

| Concern | Pipeline | Manual tokenizer + model |
|---|---|---|
| Lines of code | 1 | about 10 |
| Batching | Pass a list; `batch_size=` | Pad the batch yourself (`padding=True`) |
| Truncation | `truncation=True` argument | `truncation=True, max_length=…` |
| Device | `device=0` | `model.to("cuda")`, move every tensor |
| Access to logits, hidden states, attentions | Limited | Full (`output_hidden_states=True`, …) |
| Custom postprocessing (thresholds, neutral zone, calibration) | Hard | Easy |
| Fine-tuning | Not possible | Same objects go straight into training |
| Best for | Quick demos, prototyping | Debugging, production, fine-tuning, RAG pipelines |

:::analogy[A recipe card vs cooking from the pantry]
The pipeline is a meal kit: everything pre-measured, one instruction, dinner in minutes. The tokenizer and model are the pantry: more steps, but you choose the portions (batch size), cut what doesn't fit the pan (truncation), cook several dishes at once (batching) and change the recipe (fine-tuning).
:::

:::tip[🎯 Interview angle]
**"How do you run a Hugging Face classifier efficiently on many texts?"** Tokenize the whole batch with `padding=True, truncation=True, return_tensors='pt'`, move the model and tensors to the GPU, call the model once under `torch.no_grad()` with `model.eval()`, then softmax and `id2label` per row. Always pass the attention mask with padded inputs, and remember that truncation silently drops everything past 512 tokens.
:::

## ⚙️ Going Deeper (Hugging Face Internals)

**What `from_pretrained` downloads.** For this model: `config.json` (architecture, `id2label`, dropout rates, `max_position_embeddings: 512`), `tokenizer_config.json` and `vocab.txt` (232 KB, 30,522 WordPieces), and `model.safetensors` (268 MB of float32 weights). Files are cached by revision hash, so a pinned `revision` always loads identical weights.

**Pipeline internals.** A `Pipeline` subclass implements `preprocess` (tokenizer), `_forward` (model under `no_grad`) and `postprocess` (for text classification: softmax, or sigmoid for multi-label models, controlled by `function_to_apply`). `__call__` handles batching, device placement and lists. Writing your own subclass is how custom postprocessing gets plugged in while keeping the pipeline interface.

**Attention implementations.** Recent `transformers` versions run attention with PyTorch's fused `scaled_dot_product_attention` ("sdpa") by default. It doesn't return the attention weights: asking for `output_attentions=True` gives an empty tuple and a warning. Load with `attn_implementation="eager"` when you need the attention maps (for inspection like on the previous page); keep sdpa for speed otherwise.

**Why only [CLS] is used.** `DistilBertForSequenceClassification` reads `hidden_state[:, 0]`, the [CLS] position, because fine-tuning trained the head (and, through backpropagation, the encoder) to put a sentence summary there. Mean pooling over tokens (our notebook-18 model, Sentence-BERT) is the alternative when no model has been trained to use [CLS].

**`token_type_ids` and DistilBERT.** The tokenizer is a `BertTokenizer`, so it returns segment ids, but `DistilBertForSequenceClassification.forward` has no `token_type_ids` parameter (it accepts `input_ids`, `attention_mask`, `inputs_embeds`, `labels`, `position_ids` and absorbs extra keyword arguments). For sentence pairs, DistilBERT relies on [SEP] alone to separate them.

**Speed and precision.** On a GPU, half precision (`torch_dtype=torch.float16`) halves memory and speeds up inference with negligible change in the logits for classification; dynamic quantization (`torch.quantization.quantize_dynamic`) does the same for CPU. Batches should be sorted by length so less padding is wasted.

## 💻 Python Implementation

:::tip[Run it yourself]
Verified with transformers 5.22.2 and PyTorch 2.15 on CPU; every output below is copied from a real run (`pip install -U transformers torch` is enough; the model downloads on first use). Each block is a standalone script. The pipeline's scores match the notebook's Colab run (transformers 5.12.1, Tesla T4) to every printed digit.
:::

**What was changed from the notebooks.** Notebook 18's two Hugging Face cells failed with `ModuleNotFoundError` (an environment problem, chapter 2); their content (batched tokenization with padding and truncation) is block 6. Block 2 pins the model and revision instead of relying on the pipeline default. Blocks 6 and 7 add batching, the attention-mask check, the head recomputation and truncation, which the notebook only mentions.

### 1. The pipeline in one line

Five sentences, one call. The first run downloads the default model.

```python title="hf_pipeline_demo.py" showLineNumbers
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")       # no model given: the task's default model is downloaded

sentences = ["I love this movie.", "This movie is terrible.", "The story was not bad.",
             "The acting was boring.", "The film was surprisingly good."]
for sentence, output in zip(sentences, sentiment_pipeline(sentences)):
    print(f"{sentence:33s} {output}")

# Output:
# [transformers] No model was supplied, defaulted to distilbert/distilbert-base-uncased-finetuned-sst-2-english and revision 714eb0f.
# Using a pipeline without specifying a model name and revision in production is not recommended.
# I love this movie.                {'label': 'POSITIVE', 'score': 0.9998736381530762}
# This movie is terrible.           {'label': 'NEGATIVE', 'score': 0.9997261166572571}
# The story was not bad.            {'label': 'POSITIVE', 'score': 0.9989763498306274}
# The acting was boring.            {'label': 'NEGATIVE', 'score': 0.9997760653495789}
# The film was surprisingly good.   {'label': 'POSITIVE', 'score': 0.9998598098754883}
```

### 2. Pin the model, then probe ambiguous sentences

Naming the model and revision removes the warning; `top_k=None` returns every class.

```python title="hf_ambiguous.py" showLineNumbers
from transformers import pipeline

clf = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
               revision="714eb0f")                         # pin model and revision, as the warning asks
ambiguous = ["The movie was not terrible.", "The movie was not good.", "The movie was okay, but not amazing.",
             "I expected more from this film.", "It was fine, I guess."]
for sentence, output in zip(ambiguous, clf(ambiguous)):
    print(f"{sentence:37s} {output['label']:8s} {output['score']:.6f}")
print(clf("It was fine, I guess.", top_k=None))           # every class, not just the winner

# Output:
# The movie was not terrible.           POSITIVE 0.984081
# The movie was not good.               NEGATIVE 0.999773
# The movie was okay, but not amazing.  NEGATIVE 0.995725
# I expected more from this film.       NEGATIVE 0.991327
# It was fine, I guess.                 POSITIVE 0.999756
# [{'label': 'POSITIVE', 'score': 0.9997562766075134}, {'label': 'NEGATIVE', 'score': 0.00024378078524023294}]
```

### 3. Load the tokenizer and model with Auto classes

`AutoTokenizer` and `AutoModelForSequenceClassification` resolve the concrete classes from the checkpoint name.

```python title="hf_auto_classes.py" showLineNumbers
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()                                               # inference mode: dropout off

print(type(tokenizer).__name__, type(model).__name__)
print(model.config.id2label, model.config.label2id)
print(sum(p.numel() for p in model.parameters()))

# Output:
# BertTokenizer DistilBertForSequenceClassification
# {0: 'NEGATIVE', 1: 'POSITIVE'} {'NEGATIVE': 0, 'POSITIVE': 1}
# 66955010
```

### 4. What the tokenizer returns

Ids, mask, segment ids, the tokens behind the ids, and WordPiece splitting a rare word.

```python title="hf_tokenizer.py" showLineNumbers
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
encoded = tokenizer("The movie was surprisingly good.", return_tensors="pt")
print(list(encoded.keys()))
print(encoded["input_ids"])
print(encoded["attention_mask"])
print(tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]))
print(tokenizer.decode(encoded["input_ids"][0]))
print(tokenizer.tokenize("Transformers tokenization is unbelievably subword-based!"))

# Output:
# ['input_ids', 'token_type_ids', 'attention_mask']
# tensor([[  101,  1996,  3185,  2001, 10889,  2204,  1012,   102]])
# tensor([[1, 1, 1, 1, 1, 1, 1, 1]])
# ['[CLS]', 'the', 'movie', 'was', 'surprisingly', 'good', '.', '[SEP]']
# [CLS] the movie was surprisingly good. [SEP]
# ['transformers', 'token', '##ization', 'is', 'un', '##bel', '##ie', '##va', '##bly', 'sub', '##word', '-', 'based', '!']
```

### 5. The forward pass by hand

Logits, softmax, argmax, `id2label`, and the two-class sigmoid identity.

```python title="hf_forward.py" showLineNumbers
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()

encoded = tokenizer("The movie was surprisingly good.", return_tensors="pt")
with torch.no_grad():                                      # no gradients needed for inference
    # highlight-next-line
    output = model(**encoded)
logits = output.logits
probabilities = F.softmax(logits, dim=-1)
predicted_id = probabilities.argmax(dim=-1).item()
print(logits, logits.shape)
print(probabilities)
print(predicted_id, model.config.id2label[predicted_id])
print(torch.sigmoid(logits[0, 1] - logits[0, 0]).item())  # two classes: softmax = sigmoid of the margin

# Output:
# tensor([[-4.2341,  4.6189]]) torch.Size([1, 2])
# tensor([[1.4294e-04, 9.9986e-01]])
# 1 POSITIVE
# 0.9998570680618286
```

### 6. Batching, padding and the attention mask

One forward pass for many texts; padding is harmless with the mask and leaks without it.

```python title="hf_batching.py" showLineNumbers
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()

@torch.no_grad()
def predict(texts):
    # highlight-next-line
    batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    probs = F.softmax(model(**batch).logits, dim=-1)
    return [(model.config.id2label[p.argmax().item()], round(p.max().item(), 4)) for p in probs]

batch = tokenizer(["The movie was surprisingly good.", "Bad."], padding=True, return_tensors="pt")
print(batch["input_ids"].tolist())
print(batch["attention_mask"].tolist())
print(predict(["I love this movie.", "This movie is terrible.", "Bad."]))

with torch.no_grad():
    padded = model(**batch).logits[1]
    alone = model(**tokenizer("Bad.", return_tensors="pt")).logits[0]
    # error-next-line
    no_mask = model(input_ids=batch["input_ids"]).logits[1]
print(padded.tolist(), alone.tolist())                     # padding + mask: identical to running it alone
print(no_mask.tolist())                                    # mask forgotten: the pads leak in

# Output:
# [[101, 1996, 3185, 2001, 10889, 2204, 1012, 102], [101, 2919, 1012, 102, 0, 0, 0, 0]]
# [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0]]
# [('POSITIVE', 0.9999), ('NEGATIVE', 0.9997), ('NEGATIVE', 0.9998)]
# [4.673137187957764, -3.703674554824829] [4.673137187957764, -3.703674554824829]
# [4.5069355964660645, -3.652789354324341]
```

### 7. Inside the head, and truncation

Recompute the logits from the [CLS] hidden state; see what happens past 512 tokens.

```python title="hf_inside_head.py" showLineNumbers
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
encoded = tokenizer("The movie was surprisingly good.", return_tensors="pt")

with torch.no_grad():
    out = model(**encoded, output_hidden_states=True)
    cls_vector = out.hidden_states[-1][:, 0]               # final vector of [CLS]
    # highlight-next-line
    manual = model.classifier(torch.relu(model.pre_classifier(cls_vector)))
print(len(out.hidden_states), out.hidden_states[-1].shape)
print(manual, out.logits)

long_text = " ".join(["the movie was good"] * 200)
print(len(tokenizer(long_text)["input_ids"]), tokenizer(long_text, truncation=True, return_tensors="pt")["input_ids"].shape)

# Output:
# 7 torch.Size([1, 8, 768])
# tensor([[-4.2341,  4.6189]]) tensor([[-4.2341,  4.6189]])
# [transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (802 > 512). Running this sequence through the model will result in indexing errors
# 802 torch.Size([1, 512])
```

## 🚨 Interview Gotchas & Edge Cases

:::warning[The Most Common Question]
**"What does a Hugging Face pipeline do internally?"**

Three stages. **Preprocess:** the model's own tokenizer (loaded with `AutoTokenizer`) normalizes the text, splits it into subword tokens, adds special tokens ([CLS] … [SEP] for BERT-style models), maps them to ids and builds the attention mask, padding and truncating batches. **Forward:** the model (loaded with `AutoModelFor<Task>`) runs in eval mode under `torch.no_grad()`; for sentiment, DistilBERT encodes the tokens and a head on the [CLS] vector outputs logits `[batch, 2]`. **Postprocess:** softmax, argmax and `config.id2label` give `{'label', 'score'}`. With no model specified it downloads a task default (here DistilBERT fine-tuned on SST-2), which you should pin by name and revision in production.
:::

- **The pipeline default can change**: pin `model=` and `revision=` (the notebook's warning says so).
- **First call is slow** because of the download into `~/.cache/huggingface`; later calls load from disk.
- **Score ≠ probability of being right**: "It was fine, I guess." → POSITIVE 0.99976.
- **Two-class score = sigmoid(margin)**: saturates near 1 once the margin passes about 5.
- **No neutral class**: a 2-label model must pick POSITIVE or NEGATIVE.
- **Same checkpoint for tokenizer and model**, or the ids mean different words.
- **Uncased model**: the tokenizer lower-cases; don't feed it pre-tokenized or cased-specific text.
- **`token_type_ids` are ignored by DistilBERT** (no segment embeddings), though its BertTokenizer returns them.
- **Always pass `attention_mask` with padding**: without it, "Bad." in a padded batch got different logits.
- **512-token limit**: `truncation=True` silently drops the rest (802 → 512 tokens); long documents need chunking.
- **`model.eval()` + `torch.no_grad()`** for inference: dropout off, no autograd memory.
- **`output_attentions=True` needs `attn_implementation="eager"`** in recent versions (sdpa returns none).
- **`ModuleNotFoundError: Could not import module 'pipeline'`** is a lazy-import error hiding an environment problem (PyTorch missing or incompatible).

## ⚡ Essential Interview Recall

A one-screen revision list. Every point is explained in the chapters above.

**Pipeline**
- `pipeline("sentiment-analysis")` → default `distilbert-base-uncased-finetuned-sst-2-english`, revision 714eb0f; returns `{'label', 'score'}`.
- Three stages: preprocess (tokenizer) → forward (model) → postprocess (softmax, argmax, id2label).
- Pin model and revision; `top_k=None` for all classes; `device`, `batch_size`, `truncation`.

**The model**
- DistilBERT: 6 layers × 768 × 12 heads, FFN 3072 GELU, WordPiece 30,522, max 512, no segment embeddings.
- 66,955,010 parameters = encoder 66,362,880 + head 592,130 (`pre_classifier` 768×768 → ReLU → dropout 0.2 → `classifier` 768×2 on [CLS]).
- About 40% smaller and 60% faster than BERT-base, about 97% of its GLUE performance (distillation).

**Tokenizer**
- `input_ids` `[101, 1996, 3185, 2001, 10889, 2204, 1012, 102]` = [CLS] the movie was surprisingly good . [SEP].
- `attention_mask` 1/0; `token_type_ids` returned but unused by DistilBERT; [PAD] = 0.
- Subwords: "tokenization" → token ##ization.

**Logits → label**
- logits `[-4.2341, 4.6189]` → softmax `[0.00014, 0.99986]` → POSITIVE.
- $p_1 = \sigma(z_1 - z_0)$; margin 8.853 → 0.99986.

**Practice**
- Batch with `padding=True, truncation=True`; keep the mask; `eval()` + `no_grad()`.
- Confidence is not correctness: "not terrible" 0.984 (lowest), "fine, I guess" 0.99976.

## 📐 Formula Sheet

| Concept | Formula | In plain words |
|---|---|---|
| Softmax | $\pred{p_i} = e^{\pred{z_i}} / \sum_j e^{\pred{z_j}}$ | Logits to probabilities |
| Two-class softmax | $\pred{p_1} = \sigma(\pred{z_1} - \pred{z_0}) = 1/(1 + e^{-(\pred{z_1} - \pred{z_0})})$ | Probability = sigmoid of the margin |
| Pipeline output | $\text{label} = \operatorname{id2label}[\arg\max_i \pred{p_i}]$, $\text{score} = \max_i \pred{p_i}$ | Winning class and its probability |
| DistilBERT head | $\pred{z} = \param{W_c}\operatorname{Dropout}(\operatorname{ReLU}(\param{W_p}\pred{h_{[CLS]}} + \param{b_p})) + \param{b_c}$ | Two linear layers on the [CLS] vector |
| Head parameters | $768^2 + 768 + 768 \cdot 2 + 2 = 592{,}130$ | `pre_classifier` + `classifier` |
| Cross-entropy (fine-tuning) | $\loss{L} = -\log \pred{p_{\data{y}}}$ | What SST-2 fine-tuning minimized |

## Concept-Check Answers (from the notebook)

### First Hugging Face sentiment pipeline

<details>
<summary><strong>Q1. What does pipeline("sentiment-analysis") do at a high level?</strong></summary>

It loads a pre-trained sentiment-analysis model (by default `distilbert-base-uncased-finetuned-sst-2-english`) with its tokenizer and applies the full inference pipeline to raw text. Internally it handles tokenization (including padding and attention masks), the model forward pass, probability calculation with softmax, and label decoding through `id2label`.
</details>

<details>
<summary><strong>Q2. What are the two main fields returned by the sentiment-analysis pipeline?</strong></summary>

`label` and `score`. `label` is the predicted sentiment class (POSITIVE or NEGATIVE); `score` is the softmax probability the model assigns to that predicted class, for example `{'label': 'POSITIVE', 'score': 0.9998736381530762}` for "I love this movie.".
</details>

<details>
<summary><strong>Q3. If the pipeline cell takes time during the first run, what is one likely reason?</strong></summary>

The model and tokenizer files are being downloaded from the Hugging Face Hub and cached (here a 268 MB `model.safetensors` plus config and vocabulary). Later runs load from the local cache. Model loading itself and, on Colab, an unauthenticated (no `HF_TOKEN`) download with lower rate limits can add time.
</details>

<details>
<summary><strong>Q4. Why should we not treat a high confidence score as a guarantee that the prediction is correct?</strong></summary>

A high score only means the model strongly prefers one class; it is a softmax output, not a calibrated probability of being right. Models can be confidently wrong, especially on ambiguous, sarcastic, domain-specific or out-of-distribution text: "It was fine, I guess." gets POSITIVE 0.99976. With two classes the score is the sigmoid of the logit margin, which saturates near 1 for any large margin.
</details>
