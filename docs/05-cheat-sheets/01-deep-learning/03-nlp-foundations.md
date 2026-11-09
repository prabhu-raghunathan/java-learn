---
title: "Cheat Sheet: NLP Foundations"
sidebar_position: 3
---

:::tip[What this sheet covers]
Turning text into numbers: preprocessing, vocabularies, one-hot vectors, embeddings, Word2Vec, co-occurrence and GloVe, TF-IDF, Naive Bayes, linear SVMs and honest evaluation. Full notes: [NLP Foundations: Words to Vectors](../../03-ai-ml/03-deep-learning/03-nlp-foundations/01-nlp-foundations-words-to-vectors.md) · [NLP in Practice](../../03-ai-ml/03-deep-learning/03-nlp-foundations/02-nlp-in-practice-preprocessing-word2vec-sentiment.md).
:::

## 🎤 Answer These Out Loud

<details>
<summary><strong>How do you turn text into something a neural network can use?</strong></summary>

Preprocess (tokenize, lowercase, optionally remove stop words **keeping negations**, lemmatize; do less for Transformers) → vocabulary (word → index) with `<PAD>`, `<UNK>` and a frequency cut-off → **don't feed raw indices** (they invent an order) → one-hot fixes that but is $|V|$-dimensional and every pair has cosine 0 → **embeddings**: an $|V| \times d$ table of dense learned vectors (`nn.Embedding`), trained from scratch or initialized from Word2Vec/GloVe/FastText.
</details>

<details>
<summary><strong>Stemming vs lemmatization?</strong></summary>

**Stemming** chops suffixes by rules (Porter): fast, no dictionary, but gives non-words (*studies → studi*, *machine → machin*) and merges unrelated words (*universe/university → univers*). **Lemmatization** maps to the dictionary form (*studies → study*, *better → good*) and needs the **part of speech**: NLTK assumes noun by default, so *running* stays *running* unless tagged as a verb. More accurate, slower.
</details>

<details>
<summary><strong>Why are word embeddings better than one-hot vectors?</strong></summary>

One-hot vectors are huge ($|V|$ dims), sparse, and all **orthogonal** (cosine 0, distance $\sqrt 2$), so "cat" is as far from "dog" as from "carburettor". Embeddings are short (50–300 dims), dense and **learned**, and by the **distributional hypothesis** words used in similar contexts get similar vectors; similarity is measured with **cosine**. A lookup is just one-hot × table. They even support analogies: *king − man + woman ≈ queen* (excluding the query words).
</details>

<details>
<summary><strong>How does Word2Vec learn embeddings?</strong></summary>

It invents a self-supervised task: slide a context window over the text to make (centre, context) pairs ("i love deep learning", window 1 → 6 pairs), then **skip-gram** predicts each context word from the centre word, or **CBOW** predicts the centre word from the averaged context. The score is a dot product, $P(o \mid c) = \text{softmax}(u_o \cdot v_c)$, made cheap with **negative sampling**. Keep the input vectors $v$ as the embeddings. It needs a lot of text: on ~300 tokens the vectors barely move from their random init; on ~1 M words they become meaningful.
</details>

<details>
<summary><strong>What are the limits of static embeddings?</strong></summary>

One vector per word regardless of sense ("bank"), so contextual models (BERT) are needed for polysemy. Antonyms come out close (*good* ≈ *bad*, similarity 0.60 on Brown) because they share contexts, which hurts sentiment. Out-of-vocabulary words get nothing (subword models fix this). They also absorb social biases from the corpus.
</details>

<details>
<summary><strong>Word2Vec vs GloVe?</strong></summary>

Word2Vec learns from **local** prediction (one window at a time). GloVe starts from the **global co-occurrence matrix** $X_{ij}$ (how often $j$ appears near $i$; symmetric, $|V|^2$, sparse) and fits dense vectors so that $w_i \cdot \tilde w_j + b_i + \tilde b_j \approx \log X_{ij}$. Both give static embeddings of similar quality.
</details>

<details>
<summary><strong>What is TF-IDF and how does scikit-learn compute it?</strong></summary>

Term frequency × inverse document frequency: high for words **frequent in this document but rare in the corpus**. scikit-learn's default uses a smoothed IDF, $\ln\frac{1 + N}{1 + \text{df}} + 1$ (never zero, never divides by zero), then scales each document vector to length 1. It ignores word order ("man bites dog" and "dog bites man" get cosine 1.0) and, by default, lowercases and drops one-character tokens.
</details>

<details>
<summary><strong>Naive Bayes vs linear SVM for text?</strong></summary>

**Multinomial NB**: $\hat y = \arg\max_c P(c)\prod_t P(t \mid c)^{\text{count}}$ with Laplace smoothing; assumes words are independent given the class; very fast, data-efficient, over-confident probabilities. **Linear SVM**: maximum-margin hyperplane $\text{sign}(w \cdot x + b)$ with hinge loss and regularization `C`; excellent on sparse, high-dimensional TF-IDF; no probabilities without calibration. Both are standard baselines before deep models.
</details>

<details>
<summary><strong>How do you evaluate a small text classifier honestly?</strong></summary>

**Split first** and fit the vectorizer on training data only (a `Pipeline` guarantees it); fitting on everything leaks the test vocabulary and IDF. With tiny data, a single 16-review test set swings ±25 points between random splits (37.5%–93.75% over 20 splits), so use **repeated cross-validation** and report the spread (NB 72.8% ± 10.1%, SVM 67.4% ± 11.7%). Read the confusion matrix and per-class precision/recall, naming the positive class.
</details>

## 📐 Formulas

| Concept | Formula | In plain words |
|---|---|---|
| Sentence as tokens | $\data{S} = [\data{w_1}, \ldots, \data{w_n}]$ | List of tokens |
| Vocabulary index | $f(\data{w_i}) = i$ | Every distinct word gets a number |
| One-hot | $\data{x_w} \in \lbrace 0, 1 \rbrace^{\lvert V \rvert}$, one 1 at $f(w)$ | One dimension per word |
| Cosine similarity | $\frac{\data{a}\cdot\data{b}}{\lVert\data{a}\rVert\lVert\data{b}\rVert}$ | Compare directions |
| Embedding lookup | $\pred{e_w} = \data{x_w}^\top\param{E} = \param{E}[f(w)]$, $\param{E} \in \mathbb R^{\lvert V \rvert \times \hyper{d}}$ | Pick a row |
| Analogy | $\arg\max_{w \notin \lbrace a,b,c \rbrace}\cos(\param{v_b} - \param{v_a} + \param{v_c}, \param{v_w})$ | Add the relationship offset |
| Training pairs | $\sum_i (\min(i-1, \hyper{w}) + \min(n-i, \hyper{w}))$ | ~$2w$ per word, fewer at the edges |
| Skip-gram | $P(\data{o}\mid\data{c}) = \frac{\exp(\param{u_o}\cdot\param{v_c})}{\sum_w \exp(\param{u_w}\cdot\param{v_c})}$, $\loss{L} = -\log P$ | Dot-product score, softmax |
| GloVe | $\param{w_i}\cdot\param{\tilde w_j} + \param{b_i} + \param{\tilde b_j} \approx \log\data{X_{ij}}$ | Dot products match log counts |
| Smoothed IDF | $\ln\frac{1 + \data{N}}{1 + \data{\text{df}(t)}} + 1$ | Rare across documents → high |
| TF-IDF | $\text{tf}(t,d) \times \text{idf}(t)$, then L2-normalize | Frequent here, rare elsewhere |
| Naive Bayes | $\pred{\hat y} = \arg\max_c P(c)\prod_t P(t\mid c)^{\data{\text{count}}}$ | Prior × every word's vote |
| Laplace smoothing | $P(t\mid c) = \frac{\text{count}(t,c) + \hyper{\alpha}}{\sum_{t'}\text{count}(t',c) + \hyper{\alpha}\lvert V \rvert}$ | Add one to every count |
| SVM | $\min\frac12\lVert\param{w}\rVert^2 + \hyper{C}\sum_i\max(0, 1 - \data{y_i}(\param{w}\cdot\data{x_i} + \param{b}))$; margin $2/\lVert\param{w}\rVert$ | Wide margin, few violations |

## 🔢 Numbers to Quote

- "I am learning machine learning and loving it" → stop words removed `[learning, machine, learning, loving]` → stems `[learn, machin, learn, love]` → lemmas `[learn, machine, learn, love]`.
- Embedding table 10,000 words × 100 dims = 1 M parameters; co-occurrence matrix for 10,000 words = 100 M cells.
- Word2Vec on 298 tokens: the "delivery" vector moved 0.0003 (from a random vector of length ~0.087). On the 1 M-word Brown corpus: *week → month, year, season, day*; *good*'s nearest neighbour is *bad* (0.60).
- NLTK English stop words: 198, including *not, no, nor, don't*.
- Averaged Word2Vec vectors of `[product, good]` vs `[product, not, good]`: cosine 0.89.
- Review 1 TF-IDF: *box* 0.4187, *order* 0.2920 (norm 10.2606 over 80 reviews).
- Toy Naive Bayes: P(negative | "good late") = 0.623.
- Leaky single split: NB 62.5% vs SVM 43.75%; honest repeated 5-fold CV: NB 72.8% ± 10.1%, SVM 67.4% ± 11.7%.

## ⚖️ Quick Comparisons

| Representation | Size | Similarity between words | Learned? | Order? |
|---|---|---|---|---|
| Raw index | 1 number | Fake ordering | No | No |
| One-hot | $\lvert V \rvert$ | Always 0 | No | No |
| Embedding | 50–300 | Meaningful (cosine) | Yes | No (per word) |
| TF-IDF (document) | $\lvert V \rvert$, sparse | Shared important words | No (counts) | No |

| | **Skip-gram** | **CBOW** |
|---|---|---|
| Predicts | Context from centre word | Centre word from averaged context |
| Speed | Slower (one example per pair) | Faster |
| Strength | Small data, rare words | Large data, frequent words |

| Preprocessing step | Keep it for… | Skip / be careful for… |
|---|---|---|
| Lowercasing | Classical models | Named entities (*US* vs *us*, *Apple*) |
| Stop-word removal | Topic classification | Sentiment (negations), QA, translation, Transformers |
| Stemming / lemmatization | Small classical models | Transformers (use their own subword tokenizer) |

## 💻 Code You Should Be Able to Write

```python title="text_classifier.py" showLineNumbers
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

texts = ["great product loved it", "terrible quality not good", "fast delivery happy", "late delivery box damaged",
         "excellent experience", "not happy at all", "works well good value", "broken on arrival disappointing"] * 3
labels = [1, 0, 1, 0, 1, 0, 1, 0] * 3
X_tr, X_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.25, stratify=labels, random_state=0)
# highlight-next-line
clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), MultinomialNB())   # vectorizer fitted on train only
clf.fit(X_tr, y_tr)
print(clf.score(X_te, y_te), clf.predict(["not good delivery"]))   # 1.0 [0]   (toy data: repeated sentences)
```

Also have ready: a vocabulary built with `sorted(set(tokens))` plus `<PAD>` = 0 and `<UNK>` = 1; `nn.Embedding(vocab_size, d, padding_idx=0)`; gensim `Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1, workers=1, seed=42)`.

## 🚨 Traps

- Removing stop words deletes negations: "this product is not good" → "product good".
- Lemmatizing without POS tags treats every word as a noun.
- Building a vocabulary from `set()` without sorting gives different indices each run; save the mapping with the model.
- No `<UNK>` token → `KeyError` on the first unseen word.
- Analogy search must exclude the query words.
- Word2Vec on a tiny corpus learns noise; gensim is reproducible only with `workers=1`, `seed` and `PYTHONHASHSEED`.
- Cleaning punctuation before tokenizing turns *didn't* into `didnt`, which no negation list contains.
- `TfidfVectorizer` drops one-character tokens and lowercases by default; `.toarray()` on a big sparse matrix can exhaust memory.
- Fitting the vectorizer before splitting leaks test information.
- Averaging word vectors loses order and dilutes negation.
- t-SNE plots don't preserve global distances or cluster sizes; perplexity must be below the number of points.
- Naive Bayes probabilities are over-confident; `LinearSVC` gives none: calibrate if needed.
- Say which class is "positive" before talking about false positives.
