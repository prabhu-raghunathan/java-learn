---
title: "NLP in Practice: Text Preprocessing, Word2Vec, TF-IDF & Sentiment Classification"
sidebar_position: 2
---

:::tip[TL;DR (30-Second Interview Pitch)]
A classical NLP pipeline is: **clean** (lowercase, strip punctuation/numbers/extra spaces) → **tokenize** → remove **stop words** → **POS-aware lemmatization**, wrapped in **one function** used identically at training and prediction time. **Word2Vec** (gensim) turns words into vectors, but it needs a large corpus and gives *word* vectors, not document features. For classifying whole reviews, **TF-IDF** (term frequency × inverse document frequency) gives every document a sparse vector that up-weights words that are frequent in it but rare elsewhere. **Multinomial Naive Bayes** and a **linear SVM** on TF-IDF are the classic text-classification baselines. Evaluate honestly: **split before fitting the vectorizer** (or use a `Pipeline`), keep **negations** like *not* for sentiment, and with tiny data use **cross-validation**, not a single 16-review test set.
:::

<FormulaKey />

:::info[Where this page sits]
**Before:** [NLP Foundations: From Words to Vectors](./01-nlp-foundations-words-to-vectors.md) explains preprocessing, vocabularies, one-hot encoding, embeddings, Word2Vec and co-occurrence in theory.
**This page:** the practical notebook (*9 NLP Text Processing*): a reusable preprocessing pipeline on messy customer reviews, Word2Vec with gensim, t-SNE, TF-IDF, and Naive Bayes vs SVM sentiment classification, including corrections to three mistakes in the notebook's evaluation.
**Next:** [Sequence Data and the Idea of Recurrence](../04-rnns/01-sequence-data-and-recurrence.md): RNNs, which read text as a sequence instead of a bag of words.
:::

## 🧭 Start Here

The previous page explained *what* the representations are. This one builds them on real, messy text: e-commerce customer reviews full of capitals, exclamation marks, prices, typos and emoticons ("PAYMENT failed again!!! txn id 99882 :( pls help").

The journey has three acts:

1. **Clean and normalize the text** with a reusable function. Every design choice (remove numbers? keep "not"?) depends on the task.
2. **Learn word vectors with Word2Vec**, visualize them with t-SNE, and find out what really happens when you train embeddings on a few dozen sentences.
3. **Classify sentiment.** Turn each review into a **TF-IDF** vector, train **Naive Bayes** and a **linear SVM**, and evaluate them honestly.

Along the way we'll correct three things in the notebook: its TF-IDF leaks test data, its 16-review test set is far too small to rank the two models, and its Word2Vec "clusters" are random initialization rather than learned meaning. Each correction is a classic interview question in disguise.

| New concept | One-line meaning |
|---|---|
| **Regex cleaning** | Pattern-based find-and-replace to strip punctuation, digits and extra spaces |
| **`word_tokenize`** | NLTK's tokenizer: splits punctuation and contractions (*don't → do + n't*) |
| **POS tagging** | Labelling each word as noun, verb, adjective… so the lemmatizer can find the right base form |
| **gensim `Word2Vec`** | Library that trains word embeddings from tokenized sentences |
| **t-SNE** | A method that squashes high-dimensional vectors into 2-D for plotting, keeping neighbours together |
| **Bag-of-words** | A document as word counts, ignoring order |
| **TF-IDF** | Word count in a document × how rare the word is across documents |
| **Multinomial Naive Bayes** | Probabilistic classifier that multiplies per-word class probabilities |
| **Linear SVM** | Classifier that finds the separating hyperplane with the widest margin |
| **Data leakage** | Test data influencing anything that was fitted, which inflates the score |
| **Cross-validation** | Averaging scores over several train/test splits for a stable estimate |

:::analogy[A text pipeline is a laundry line]
Raw reviews arrive like a basket of mixed laundry: different sizes, colours and stains. Cleaning is sorting and washing, tokenization is unfolding each item, lemmatization is pairing socks that are really the same, and TF-IDF is hanging everything on numbered pegs so the classifier can see at a glance which items stand out. Use a different routine for the test basket and nothing lines up.
:::

## 1. Raw Text Is Messy: Cleaning It

The notebook starts with five deliberately noisy sentences:

| Raw | After `basic_cleaning` |
|---|---|
| `I LOVE Machine Learning!!!` | `i love machine learning` |
| `   NLP is AMAZING... isn't it???   ` | `nlp is amazing isnt it` |
| `Deep Learning costs $1000 sometimes!!!` | `deep learning costs sometimes` |
| `AI & ML are transforming the world in 2025!!!` | `ai ml are transforming the world in` |
| `I am learning    NLP     from Masai` | `i am learning nlp from masai` |

Every problem in the raw column inflates the vocabulary or adds noise. "Machine" and "machine" would be two words. "Learning!!!" would be a third. Leading, trailing and repeated spaces make naive `split(" ")` produce empty tokens. A model that sees these as different symbols has fewer examples of each and learns less from each.

The cleaning function applies four regular-expression steps, **in this order**:

1. **Lowercase**: `text.lower()`.
2. **Remove punctuation and special characters**: `re.sub(r"[^\w\s]", "", text)` deletes everything that is not a word character (`\w` = letters, digits, underscore) or whitespace (`\s`). Note the side effect: *isn't* becomes `isnt`, a new token that no stop-word list contains.
3. **Remove numbers (optional)**: `re.sub(r"\d+", "", text)`.
4. **Collapse whitespace**: `re.sub(r"\s+", " ", text).strip()` turns any run of spaces, tabs and newlines into one space and trims the ends.

**Cleaning is task-dependent.** Every step deletes information, so ask whether that information matters for *your* task. Numbers matter in "5-star rating", "delivered in 2 days" or "₹999 is too expensive". Punctuation matters for emotion ("great!!!" vs "great."), for emoticons (":(" is a strong negative signal that this cleaner deletes), for questions ("?"), and for decimals and URLs. Case matters for named entities (*Apple*, *US*). The notebook's defaults are fine for a topic or sentiment classifier on short reviews, but they are choices, not laws.

| Step | Pattern | Replace with | Effect on `"   NLP is AMAZING... isn't it???   "` |
|---|---|---|---|
| 1. Lowercase | `str.lower()` | — | `"   nlp is amazing... isn't it???   "` |
| 2. Punctuation | `[^\w\s]` (anything not a word char or space) | `""` | `"   nlp is amazing isnt it   "`: *isn't* → `isnt` |
| 3. Numbers | `\d+` (runs of digits) | `""` | unchanged here; `$1000` disappears in sentence 3 |
| 4. Whitespace | `\s+` then `.strip()` | `" "` | `"nlp is amazing isnt it"` |

:::tip[🎯 Interview angle]
**"How would you clean text for a classifier?"** Lowercase, strip punctuation/special characters, optionally remove numbers, normalize whitespace, then tokenize. Then immediately add: **it depends on the task.** Keep numbers when quantities matter, punctuation and emoticons for sentiment, case for named entities, and do almost no cleaning for Transformer models, which bring their own tokenizer.
:::

## 2. Tokenization in Practice

After cleaning, each review is still one string. **Tokenization** breaks it into tokens, the units everything downstream counts, looks up and embeds. On clean, lowercase text without punctuation, `sentence.split()` and NLTK's `word_tokenize` give identical results (they did for all five notebook sentences).

The difference shows on *raw* text, which is where a real tokenizer earns its keep:

| Input | `str.split()` | `nltk.word_tokenize` |
|---|---|---|
| `don't` | `["don't"]` | `['do', "n't"]` |
| `I can't pay $1,000.50 today, e-mail me!` | `['I', "can't", 'pay', '$1,000.50', 'today,', 'e-mail', 'me!']` | `['I', 'ca', "n't", 'pay', '$', '1,000.50', 'today', ',', 'e-mail', 'me', '!']` |

`split()` glues punctuation onto words ("today," and "me!" become distinct tokens from "today" and "me"). `word_tokenize` follows the Penn Treebank conventions: punctuation becomes its own token, contractions split into a word and a clitic (*can't → ca + n't*, so the negation `n't` survives as a separate, recognizable token), but numbers like "1,000.50" and hyphenated words like "e-mail" stay whole.

**Order matters here.** The notebook removes punctuation *before* tokenizing, so `word_tokenize` never sees the apostrophe and *didn't* becomes `didnt`. Tokenizing first would give `did` + `n't`, which the stop-word list then removes. That's a real design choice, and it changes whether negation survives.

Modern NLP mostly uses **subword tokenizers** (BPE, WordPiece, SentencePiece) learned from data: frequent words stay whole and rare words split into pieces (*tokenization → token + ization*). They have a fixed vocabulary size, almost no out-of-vocabulary words, and they are what BERT and GPT use. You'll meet them in the Transformer section.

:::tip[🎯 Interview angle]
**"Why not just `split()`?"** It leaves punctuation attached ("great!" ≠ "great"), can't split contractions, and has no idea about numbers, URLs or hyphens. A proper tokenizer (NLTK, spaCy) handles those consistently, and subword tokenizers (BPE/WordPiece) also eliminate most out-of-vocabulary words. **"What is a token?"** The unit of text a model processes: a word, punctuation mark, or subword piece.
:::

## 3. Stop Words, Stemming and Lemmatization in Practice

NLTK's English stop-word list (198 words in the version used here) holds pronouns, articles, auxiliaries and prepositions: *i, am, is, the, in, from, it…* Removing them from the five cleaned sentences leaves the content words: `['love', 'machine', 'learning']`, `['nlp', 'amazing', 'isnt']`, `['deep', 'learning', 'costs', 'sometimes']`, `['ai', 'ml', 'transforming', 'world']`, `['learning', 'nlp', 'masai']`. Note that `isnt` survives: the list contains "isn't" and "isn" but not the cleaned form `isnt`.

Then the three normalizers compared on the same tokens:

| Original | Porter stemmer | Lemmatizer (no POS) | Lemmatizer + POS |
|---|---|---|---|
| love, machine, learning | love, **machin**, learn | love, machine, learning | love, machine, learning |
| nlp, amazing, isnt | nlp, **amaz**, isnt | nlp, amazing, isnt | nlp, amazing, isnt |
| deep, learning, costs, sometimes | deep, learn, cost, **sometim** | deep, learning, cost, sometimes | deep, learning, cost, sometimes |
| ai, ml, transforming, world | ai, ml, transform, world | ai, ml, transforming, world | ai, ml, transforming, world |
| learning, nlp, masai | learn, nlp, masai | learning, nlp, masai | **learn**, nlp, masai |

The stemmer always cuts, sometimes too far (*machin, amaz, sometim*). The plain lemmatizer treats every word as a noun, so it only fixed plurals (*costs → cost*). The POS-aware version only changed one word, and that is instructive: in "learning nlp masai" the tagger called *learning* a verb (→ *learn*), but in "love machine learning" and "deep learning" it tagged *learning* as a noun ("machine learning" the field), and the noun lemma of "learning" is "learning". **POS-aware lemmatization is context-sensitive**: the same word can get different lemmas in different sentences, which is linguistically right, but means the vocabulary isn't always as small as you might expect.

The classic demonstration is `["running", "better", "studies"]`: without POS → `['running', 'better', 'study']`; with POS (tagged VBG, JJR, NNS) → `['run', 'good', 'study']`.

**How POS mapping works.** `nltk.pos_tag` returns Penn Treebank tags (`VBG` = verb gerund, `JJR` = comparative adjective, `NNS` = plural noun, `RB` = adverb…). WordNet's lemmatizer only knows four classes, so a small helper maps tags by their first letter: J → adjective, V → verb, N → noun, R → adverb, anything else → noun.

**The cost.** POS tagging runs a statistical model on every sentence, which is far slower than a stemmer's string rules. On millions of documents, or in a latency-sensitive API, that can dominate the preprocessing time. It also needs grammatical sentences: on keyword-style queries or tweets the tagger guesses badly. Tagging the *tokens after stop-word removal* (as the notebook does) also removes the grammar around them and makes the tags less accurate. Tagging the full sentence first and removing stop words afterwards is more accurate.

:::danger[Interview Trap: "Stop-word removal is always safe"]
For sentiment it's actively harmful: NLTK's list contains **not, no, nor**, and contractions like *don't, isn't, wasn't*. "This product is good" and "This product is not good" both preprocess to `['product', 'good']` with the standard list. The fix (Section 7) is a task-specific stop-word list that keeps negations.
:::

:::tip[🎯 Interview angle]
**"Why tag parts of speech before lemmatizing?"** The lemma depends on the word's role: *better* → *good* only as an adjective, *running* → *run* only as a verb; NLTK's default is noun. **"Stemming or POS lemmatization: which is more accurate, and what does it cost?"** Lemmatization (real words, irregular forms); it's slower because tagging runs a model per sentence, which matters at scale or in low-latency services, and it is less reliable on ungrammatical text.
:::

## 4. One Reusable Preprocessing Function

The notebook combines every step into `preprocess_text(text)`: lowercase → strip punctuation → strip numbers → collapse spaces → tokenize → remove stop words → POS-tag → lemmatize. Applied to "I am learning NLP and building AI systems in 2025!!!" it returns `['learn', 'nlp', 'building', 'ai', 'system']`.

Wrapping the steps in one function isn't just tidiness. It enforces the most important rule of any ML pipeline: **training data and prediction-time data must go through exactly the same transformation.** The model learned weights for tokens like `learn` and `system`. If the production API lowercases but forgets to lemmatize, a user's "systems" becomes the unknown token `systems`, the model never saw it, and accuracy quietly drops with no error message. This mismatch is called **training–serving skew**, and it's one of the most common causes of models that work in the notebook but fail in production. One function, used in the training notebook, the evaluation code and the deployed service, removes the risk.

The same function is reused for the 54-review dataset with `df["raw_text"].apply(preprocess_text)`. One pandas detail from the notebook is worth knowing: when six extra noisy reviews are appended with `pd.concat`, their `processed_tokens` column comes out as **NaN**, because that column was computed before the concat. Derived columns must be recomputed (or the function applied) after adding rows, which the notebook does a few cells later.

| Raw review | Processed tokens |
|---|---|
| My order arrived 2 days late!!! Very disappointed :( | order, arrive, day, late, disappointed |
| The phone cover quality is AMAZING, really loved it. | phone, cover, quality, amaze, really, love |
| Payment failed 3 times but money got deducted!!! | payment, fail, time, money, get, deduct |
| Customer support was rude and didn't solve my issue. | customer, support, rude, didnt, solve, issue |

Notice *amazing → amaze* (tagged as a verb here), *got → get*, and *didnt* surviving because punctuation was removed before the stop-word check.

:::analogy[One preprocessing function is one recipe card]
If the head chef writes the recipe on one card and every cook uses that card, the dish tastes the same at every branch. If the evening shift cooks from memory and skips a step, customers get a different dish and nobody notices until the complaints arrive. Training–serving skew is the evening shift cooking from memory.
:::

:::tip[🎯 Interview angle]
**"What happens if preprocessing differs between training and inference?"** Training–serving skew: tokens the model never saw, a shifted feature distribution, and silently lower accuracy with no error. Fix it by packaging preprocessing with the model: one function, or better a scikit-learn `Pipeline`, or the tokenizer saved alongside a Transformer.
:::

## 5. Training Word2Vec with gensim

`gensim.models.Word2Vec` takes a list of tokenized sentences and does everything from the previous page: builds the vocabulary, generates (centre, context) pairs with a sliding window, and trains the vectors. The parameters map directly onto the theory:

| Parameter | Meaning | Notebook value |
|---|---|---|
| `vector_size` | Embedding dimension $d$ | 10 (demo), 50 (reviews) |
| `window` | Context words on each side | 3 |
| `min_count` | Ignore words rarer than this | 1 (keeps every word) |
| `sg` | 0 = CBOW (default), 1 = skip-gram | 0 |
| `epochs` | Passes over the corpus | 5 (default) |
| `workers` | Training threads | 4 |
| `seed` | Random seed for initialization | default |

After training, `model.wv.index_to_key` is the vocabulary sorted by frequency, `model.wv["delivery"]` is that word's vector (shape `(50,)`), and `model.wv.most_similar("delivery")` lists the words with the highest cosine similarity.

**What the notebook's results really show.** On the 54 customer reviews (298 tokens after preprocessing, vocabulary 210 words, most seen once), the "most similar" words to *delivery* came out as *receive, chat, link, ever, id*, and for *payment* as *service, checkout, soft, performance*, all with similarities around 0.2–0.4. They look half-plausible, but they are noise. We checked directly:

- After training, the *delivery* vector had moved by **0.0003** from its random starting point, whose length is about **0.087**. Training barely touched it.
- The nearest neighbours of *delivery* after training (*loud, checkout, coupon*) are **identical** to its neighbours in the *untrained* random vectors.
- Retraining with a different seed gives completely different "neighbours" (*defective, fit, next, page, happy*).

The reason is data: each word appears once or twice, and there are a few hundred training pairs in total, against 210 × 50 × 2 ≈ 21,000 parameters. Word2Vec needs hundreds of thousands to billions of words. On the one-million-word Brown corpus the same call gives *week → month, year, season, day* and solves king − man + woman = queen (see the [previous page](./01-nlp-foundations-words-to-vectors.md)). So the practical lesson is: **on small domain data, use pretrained vectors** (GloVe, FastText, or Word2Vec trained on news/web text), optionally fine-tuned, rather than training your own.

**Reproducibility.** gensim results change between runs unless you set `seed`, use `workers=1` (multi-threading makes the update order random), and fix Python's hash seed (`PYTHONHASHSEED`), which affects vocabulary ordering.

**Embeddings are domain-specific.** Trained on product reviews, *cheap* sits near *quality* and *fabric*; trained on finance news, near *stock* and *valuation*. "Python" in programming text vs zoology text gets different neighbours. That's why the same word can have different vectors in different models, and why domain-specific pretrained embeddings exist.

:::danger[Interview Trap: "The neighbours look sensible, so the embeddings are good"]
With little data, cosine-similarity neighbours are dominated by the **random initialization**, and the human brain finds patterns in anything. Always compare against a baseline (untrained vectors, a different seed) or evaluate on a standard similarity/analogy benchmark before trusting home-trained embeddings.
:::

:::tip[🎯 Interview angle]
**"What does Word2Vec take as input and output?"** Input: tokenized sentences (lists of tokens). Output: one dense vector per vocabulary word (here 50 numbers). **"Why are results weak on small datasets?"** Too few co-occurrences per word for the vectors to move away from their random start; more parameters than training signal. Use pretrained embeddings instead. **"Why do embeddings differ across domains?"** Meaning is learned from context, and the contexts differ.
:::

## 6. Visualizing Embeddings with t-SNE

A 50-dimensional vector can't be plotted. **t-SNE** (t-distributed Stochastic Neighbour Embedding) maps each vector to a 2-D point so that **points that were close neighbours in 50-D stay close in 2-D**. It works by turning distances into "probability of being neighbours" in both spaces and moving the 2-D points until the two sets of probabilities match.

Read t-SNE plots carefully, because they are easy to over-interpret:

- **It preserves local neighbourhoods, not distances.** Cluster sizes, gaps between clusters, and the global layout carry little meaning. Two clusters far apart on the plot aren't necessarily far apart in the original space.
- **The result depends on `perplexity`** (roughly, how many neighbours each point considers) and on the random seed. Different settings give different pictures. Perplexity must be smaller than the number of points: with 10 points, `perplexity=30` raises *"perplexity (30) must be less than n_samples (10)"*. The notebook used 5 for its 210 words.
- **It is for looking, not modelling.** t-SNE has no `transform` for new points, and the 2-D coordinates shouldn't be used as features.

So when the notebook observes "delivery, payment and support clusters", remember Section 5: those vectors are near-random, and t-SNE can make random points look clustered. Clusters overlap when words share contexts (e.g. *refund* appears in both payment and customer-service reviews), when the corpus is small, or simply because squashing 50 dimensions into 2 must distort something. **PCA** is the faster, deterministic alternative: it preserves global structure but separates clusters less well. **UMAP** is a popular modern alternative to t-SNE.

| | **PCA** | **t-SNE** |
|---|---|---|
| Type | Linear projection | Non-linear, neighbour-preserving |
| Preserves | Global variance and distances (approximately) | Local neighbourhoods only |
| Deterministic | Yes | No (seed, perplexity) |
| New points | `transform` works | No transform; refit everything |
| Speed | Fast | Slow on large data |
| Use for | Quick overview, preprocessing | Visual exploration of clusters |

:::tip[🎯 Interview angle]
**"Why use t-SNE, and does it preserve distances?"** To eyeball high-dimensional embeddings in 2-D. It preserves **local** neighbourhoods, not global distances, so cluster sizes and inter-cluster gaps are not meaningful, and results change with perplexity and seed. Use it for exploration only, never as features or as proof that embeddings are good.
:::

## 7. From Word Vectors to Document Features

Now the task changes. Instead of studying individual words, we want to classify a **whole review** as positive or negative. A classifier needs one fixed-length feature vector per review, and Word2Vec gives us one vector per *word*. How do we get from words to a document?

**The tempting shortcut: average the word vectors.** It gives a fixed-size vector and works surprisingly well for topic detection. But for sentiment it has three serious problems:

- **Word order is lost**: "not bad, actually good" and "not good, actually bad" contain the same words.
- **Negation is diluted**: one *not* among several content words barely moves the average. Using Brown-corpus Word2Vec vectors, the averages of `[product, good]` and `[product, not, good]` have cosine similarity **0.89** (skip-gram), almost the same vector.
- **Every word counts equally**, so frequent filler words wash out the few words that carry sentiment.

And with the standard stop-word list it's worse: "This product is good" and "This product is not good" both preprocess to **`['product', 'good']`**, identical before any vectors are involved.

So the notebook switches to **TF-IDF**, the classic strong baseline for document classification. It represents each document as a sparse vector with one dimension per vocabulary word, whose value says how *important* that word is *to this document*. Importance has two parts:

- **TF, term frequency**: how often the word appears in this document. More mentions → more important here.
- **IDF, inverse document frequency**: how *rare* the word is across all documents. A word in every review ("product", "order") can't distinguish anything, so it is down-weighted; a word in only two reviews ("defective", "crushed") is a strong, specific signal.

Multiplying them gives high scores to words that are **frequent in this document but rare in the corpus**. Then each document vector is scaled to length 1 (L2 normalization), so long and short reviews are comparable.

scikit-learn's `TfidfVectorizer` uses a **smoothed** IDF that adds 1 to the counts (as if one extra document contained every word), which avoids dividing by zero and keeps every IDF ≥ 1. Let's verify the notebook's first review by hand. With all 80 reviews, "order arrive two day late box damage" has seven words, each appearing once, so TF = 1 for each and the vector is just the IDFs, normalized:

| Word | Reviews containing it (df) | IDF $= \ln\frac{1 + 80}{1 + \text{df}} + 1$ | ÷ norm 10.2606 = TF-IDF |
|---|---|---|---|
| box, damage, two | 2 | $\ln(81/3) + 1 = 4.2958$ | **0.4187** |
| late | 3 | 4.0082 | 0.3906 |
| day | 5 | 3.6027 | 0.3511 |
| arrive | 6 | 3.4485 | 0.3361 |
| order | 10 | 2.9966 | **0.2920** |

Exactly the notebook's output. *Order* is the most common word in this review set, so it scores lowest; *box* and *damage* are rare, so they score highest. A word appearing in all 80 reviews would get IDF $\ln(81/81) + 1 = 1$, the minimum.

The result is an 80 × 234 matrix (80 reviews, 234 vocabulary words), almost entirely zeros. Sparse, high-dimensional, but very informative per dimension, which is exactly what linear classifiers like.

<Formula name="TF-IDF" tag="Document features" words="Weight each word in a document by how often it appears there times how rare it is across all documents. Then scale each document's vector to length one." remember="Frequent here, rare elsewhere.">

$$
\begin{aligned}
\text{tfidf}(t, d) &= \data{\text{tf}(t, d)} \times \text{idf}(t),
\qquad
\text{idf}(t) = \ln\frac{1 + \data{N}}{1 + \data{\text{df}(t)}} + 1 \\[4pt]
\data{\mathbf{x}_d} &\leftarrow \frac{\data{\mathbf{x}_d}}{\lVert \data{\mathbf{x}_d} \rVert_2}
\end{aligned}
$$

- $\data{\text{tf}(t, d)}$: count of term $t$ in document $d$
- $\data{N}$: number of documents; $\data{\text{df}(t)}$: number of documents containing $t$
- The "+1"s are scikit-learn's default smoothing (`smooth_idf=True`); textbook IDF is $\ln(N / \text{df})$
- **Example:** "box" appears in 2 of 80 reviews → idf $= \ln(81/3) + 1 = 4.2958$; after L2 normalization in review 1 → **0.4187**

</Formula>

<Animation src="nlp/tfidf-weights" caption="TF-IDF for the first review, word by word. Each word starts with term frequency 1, gets multiplied by its IDF, and the vector is divided by its length 10.26. The common word order ends lowest at 0.292, the rare words box and damage highest at 0.419." />

| Document representation | Keeps word order | Handles negation | Size | Good first choice for |
|---|---|---|---|---|
| Averaged Word2Vec | No | Poorly (diluted) | $d$ (dense) | Topic similarity with good pretrained vectors |
| Bag-of-words counts | No | Only as its own token | $\lvert V \rvert$ (sparse) | Naive Bayes |
| **TF-IDF** | No | Only as its own token | $\lvert V \rvert$ (sparse) | **Classical classifiers: NB, linear SVM, logistic regression** |
| TF-IDF with bigrams | Partly (adjacent pairs) | Better ("not_good" becomes a feature) | Larger | Sentiment baselines |
| RNN / Transformer | **Yes** | **Yes** | Learned | When data and compute allow |

:::analogy[TF-IDF is a highlighter that avoids the obvious]
Highlighting every word in a review is useless, and so is highlighting "product", which every review contains. TF-IDF highlights what is said *a lot here* but *rarely elsewhere*, like *crushed* or *refund*: the words that make this review different from the rest.
:::

:::tip[🎯 Interview angle]
**"Explain TF-IDF."** Term frequency in the document × inverse document frequency across the corpus, so words frequent in this document but rare overall score highest; common words are down-weighted; rows are L2-normalized. **"Why does it pair well with NB and SVM?"** It produces sparse, high-dimensional, mostly independent features in which a few words carry the signal, which is exactly where linear models and Naive Bayes shine. **"Weakness?"** No word order or semantics; fix negation with bigrams (`ngram_range=(1, 2)`).
:::

## 8. Preparing the Sentiment Dataset

Sentiment classification is **supervised learning**: each example pairs an input (the review text) with a target label (*positive* or *negative*), and the model learns the mapping from labelled examples. Without labels there is nothing to learn the mapping *from*; you could only cluster reviews, not name what each cluster means.

The notebook uses 80 short reviews, exactly **40 positive and 40 negative**, so a model that always answers one class scores 50%: that's the baseline to beat. It's larger than the five-sentence toy examples (more vocabulary, more examples of each sentiment word), which makes the comparison more meaningful. But 80 is still *tiny* for NLP, and as Section 10 shows, too small to rank two models reliably. Real sentiment datasets have thousands to millions of reviews (IMDb: 50,000; Amazon reviews: millions).

**Task-specific preprocessing.** For sentiment, the notebook builds a modified stop-word list: NLTK's list **minus** the negations `{not, no, nor, never, neither, none}`. With it, "This product is not good" → `['product', 'not', 'good']`, while the standard list gives `['product', 'good']`. Those three negation words then become some of the strongest features the models learn (Section 9: *not* is the single most negative-leaning word in both models). POS-aware lemmatization still helps by merging inflected forms (*damaged → damage*, *arrived → arrive*, *loved → love*), so each sentiment word's evidence is pooled into one feature.

Two limits remain. The list doesn't contain cleaned contractions like `didnt` or `wasnt`, which survive only because punctuation was stripped first, and they become separate features from `not`. And a bag-of-words model still can't tell *what* is negated ("not good" vs "good, not bad"); bigrams or sequence models are needed for that.

:::tip[🎯 Interview angle]
**"Why should preprocessing change with the task?"** Because the information you delete must be irrelevant to *that* task: negations are noise for topic classification but decisive for sentiment, numbers matter for price complaints, case matters for named entities. **"Why keep negation?"** Removing *not* maps "good" and "not good" to identical features and flips meaning.
:::

## 9. Naive Bayes and Linear SVM

Both models learn a weight per vocabulary word, but from different philosophies.

**Multinomial Naive Bayes** is probabilistic. It applies **Bayes' rule** to ask: given the words in this review, which class is more probable? It makes the "naive" assumption that, *within a class*, each word appears independently of the others. That is obviously false ("customer" and "support" travel together), but it turns an impossible estimate into a simple product: $P(\text{class}) \times$ the probability of each word under that class. Each $P(\text{word} \mid \text{class})$ is just the word's share of all words in that class's training documents, with **Laplace smoothing** (add 1 to every count) so that a word never seen in a class doesn't zero out the whole product. In practice it multiplies (adds, in log space) per-word evidence.

A hand-checkable example. Train on four tiny documents: positive "good fast", "good quality"; negative "late bad", "not good late". Vocabulary: \{bad, fast, good, late, not, quality\}, so $\lvert V \rvert = 6$. Classify "good late":

- Positive: 4 words in total. $P(\text{good} \mid +) = \frac{2 + 1}{4 + 6} = 0.3$, $P(\text{late} \mid +) = \frac{0 + 1}{10} = 0.1$. Score $= 0.5 \times 0.3 \times 0.1 = 0.015$.
- Negative: 5 words. $P(\text{good} \mid -) = \frac{1 + 1}{11} = 0.182$, $P(\text{late} \mid -) = \frac{2 + 1}{11} = 0.273$. Score $= 0.5 \times 0.182 \times 0.273 = 0.0248$.
- Normalized: P(negative) $= 0.0248 / (0.015 + 0.0248) = $ **0.623**, exactly what scikit-learn's `MultinomialNB` reports. "Late" is twice as common in negative reviews, which outweighs "good".

Naive Bayes trains in a single pass of counting, needs very little data, is hard to overfit and gives a strong baseline. That's why it's still the first thing to try on text. Its weaknesses: the independence assumption makes its probabilities over-confident, and correlated words get double-counted.

**A linear SVM** is geometric. Each review is a point in a 234-dimensional TF-IDF space, and the SVM looks for the flat **hyperplane** $\mathbf{w} \cdot \mathbf{x} + b = 0$ that separates positive from negative points with the **widest margin**, the largest gap to the nearest points on either side. Only those nearest points (the **support vectors**) define the boundary. The parameter `C` trades margin width against training errors (small `C` = wide margin, more tolerant of mistakes = more regularization). Prediction is the sign of $\mathbf{w} \cdot \mathbf{x} + b$, so each word gets a weight: positive weights push towards *positive*, negative towards *negative*.

SVMs are excellent on text because TF-IDF data is high-dimensional and sparse: with more dimensions than documents, the classes are usually linearly separable, and maximizing the margin is a strong defence against overfitting in exactly that situation. Linear SVMs (and logistic regression) were the state of the art for text classification before deep learning.

Both models, trained on all 80 reviews, learn sensible word weights. The most negative SVM weights are **not** (−0.76), *terrible*, *disappointing*, *status*, *defective*, *wrong*; the most positive are *experience*, *happy*, *excellent*, *premium*, *satisfied*, *easy*. Naive Bayes agrees: *not, late, status, app, wrong, defective, terrible* lean negative; *experience, excellent, happy, fast, smoothly, comfortable* lean positive. Notice *status* and *app* (from "app showed pending status", "app freezes"): they are reliable negative words *in this dataset* but not in general, a reminder that models learn the dataset's quirks along with real sentiment.

<Formula name="Multinomial Naive Bayes" tag="Bayes' rule with independent words" words="Score each class by its prior probability times the probability of every word in the document under that class, using add-one smoothing so unseen words do not zero the product. Pick the class with the higher score." remember="Prior times every word's vote.">

$$
\begin{aligned}
\pred{\hat y} &= \arg\max_{c}\; P(c)\prod_{t \in d} P(t \mid c)^{\data{\text{count}(t, d)}} \\[4pt]
P(t \mid c) &= \frac{\data{\text{count}(t, c)} + \hyper{\alpha}}{\sum_{t'} \data{\text{count}(t', c)} + \hyper{\alpha}\,\lvert V \rvert}
\end{aligned}
$$

- $P(c)$: fraction of training documents in class $c$ (here 0.5 each)
- $\hyper{\alpha} = 1$: Laplace (add-one) smoothing
- **Example:** "good late" → positive $0.5 \times \frac{3}{10} \times \frac{1}{10} = 0.015$, negative $0.5 \times \frac{2}{11} \times \frac{3}{11} = 0.0248$ → **negative**, P = 0.623

</Formula>

<Formula name="Linear SVM" tag="Maximum-margin hyperplane" words="Find the weight vector and bias whose hyperplane separates the two classes with the widest possible gap, allowing a few violations controlled by C. Classify a new document by which side of the plane it falls on." remember="Widest street between the classes.">

$$
\pred{\hat y} = \text{sign}\big(\param{\mathbf{w}} \cdot \data{\mathbf{x}} + \param{b}\big),
\qquad
\min_{\param{\mathbf{w}}, \param{b}} \;\tfrac{1}{2}\lVert \param{\mathbf{w}} \rVert^2 + \hyper{C}\sum_i \max\big(0,\ 1 - \data{y_i}(\param{\mathbf{w}} \cdot \data{\mathbf{x}_i} + \param{b})\big)
$$

- Margin width $= 2 / \lVert \param{\mathbf{w}} \rVert$: minimizing $\lVert \param{\mathbf{w}} \rVert$ widens it
- $\max(0, 1 - y\,f(x))$: **hinge loss**, zero for points on the correct side beyond the margin (`LinearSVC` uses its squared version by default)
- **Learned weights (all 80 reviews):** *not* −0.76, *terrible* −0.63; *experience* +0.91, *happy* +0.77

</Formula>

| | **Multinomial Naive Bayes** | **Linear SVM** |
|---|---|---|
| Approach | Probabilistic (Bayes' rule) | Geometric (maximum margin) |
| Key assumption | Words independent given the class | Classes roughly linearly separable |
| Training | One counting pass, extremely fast | Optimization, still fast for linear |
| Small data | **Very good**, low variance | Good, but more variance |
| Outputs probabilities | Yes (often over-confident) | No (decision scores; calibrate if needed) |
| Correlated features | Double-counts them | Handles them |
| Tuning | `alpha` (smoothing) | `C` (regularization) |

:::analogy[Naive Bayes is a jury; SVM is a fence]
Naive Bayes is a jury in which every word testifies independently ("I usually appear in complaints"), and the verdict multiplies their testimonies. An SVM builds the widest possible fence between the positive and negative reviews, and judges a new review by which side of the fence it lands on.
:::

:::tip[🎯 Interview angle]
**"Why is Naive Bayes still useful despite its naive assumption?"** Ranking classes only needs the *argmax* to be right, not calibrated probabilities, and NB gets it right surprisingly often; it's very fast, data-efficient and a strong baseline. **"Why is SVM strong on text?"** High-dimensional sparse TF-IDF features are usually linearly separable, and the maximum margin controls overfitting in the more-features-than-samples regime.
:::

## 10. Evaluating Honestly (and What the Notebook Got Wrong)

The notebook splits the 80 reviews 80/20 (stratified: 64 train, 16 test), trains both models and reports **Naive Bayes 62.5% vs SVM 43.75%**, with these confusion matrices (labels ordered positive, negative):

| | NB: pred + | NB: pred − | SVM: pred + | SVM: pred − |
|---|---|---|---|---|
| **Actual positive** | 6 | 2 | 4 | 4 |
| **Actual negative** | 4 | 4 | 5 | 3 |

An SVM scoring *worse than a coin flip* should set off alarms. Three problems hide in this setup, and each is a classic interview topic.

**Problem 1: data leakage.** The notebook fits `TfidfVectorizer` on **all 80 reviews** and only then splits. So the vocabulary and every IDF value were computed with the test reviews included: the test set helped build the features. In real deployment, new reviews arrive *after* the vectorizer was fitted, so their unseen words get no column at all. The fix is to **split the raw text first, then fit the vectorizer on the training text only**, which a scikit-learn `Pipeline` does automatically (it re-fits the vectorizer inside every split and cross-validation fold). With the fix, the training vocabulary shrinks from 234 to **199** words, and **38 of the 69 tokens in the 16 test reviews are words never seen in training**. On this split the accuracies happen to stay 62.5% and 43.75% (NB's confusion matrix changes to `[[5, 3], [3, 5]]`), but on real data leakage typically inflates scores, sometimes enormously.

**Problem 2: the test set is far too small.** With 16 test reviews, each review is worth 6.25 percentage points, and the result depends heavily on *which* 16 you picked. Repeating the same split with 20 different random seeds gave:

- Naive Bayes: anywhere from **50%** to **93.75%** (mean 74.4%)
- SVM: anywhere from **37.5%** to **93.75%** (mean 66.9%)

The notebook's single split happened to be an unlucky one for the SVM. Comparing two models on one 16-example split is close to meaningless.

**Problem 3: so use cross-validation.** **Repeated stratified k-fold cross-validation** (5 folds × 10 repeats = 50 train/test splits, each with a properly re-fitted pipeline) gives **NB 72.8% ± 10.1%** and **SVM 67.4% ± 11.7%**. Naive Bayes is slightly ahead, which fits its reputation on tiny datasets, but the two ranges overlap heavily, and the honest conclusion is "both are around 70%, NB a little better, with high uncertainty because 80 reviews is too few". More data would help far more than switching models.

**Reading the metrics.** As on the [CNN evaluation page](../02-cnns/03-cnn-practical-cats-vs-dogs.md), accuracy alone hides the *kind* of mistakes. NB's report on the notebook split: negative precision 0.67 / recall 0.50, positive precision 0.60 / recall 0.75. It catches positives well but misses half the negatives. If *negative* is the class you care about (a support team triaging complaints), a **negative review predicted as positive is a false negative**: a complaint that slips through unnoticed, usually the costlier error. Two models can have the same accuracy with very different mistake patterns, which is why you always read the confusion matrix.

<Animation src="nlp/split-variance" caption="The same two pipelines evaluated on 20 different random 64/16 splits of the 80 reviews. Each dot is one split's test accuracy. Watch how widely they scatter, from 37.5 to 93.75 percent, while the cross-validation means sit at 72.8 for Naive Bayes and 67.4 for the SVM." />

| Evaluation setup | NB accuracy | SVM accuracy | Trust it? |
|---|---|---|---|
| Notebook: TF-IDF fit on all data, one split | 62.5% | 43.75% | ❌ Leaky and noisy |
| Fixed: `Pipeline`, same single split | 62.5% | 43.75% | ⚠️ No leak, still one tiny split |
| 20 random single splits | 50%–93.75% | 37.5%–93.75% | ⚠️ Shows how unstable one split is |
| **Repeated 5-fold CV (50 splits)** | **72.8% ± 10.1%** | **67.4% ± 11.7%** | ✅ Best estimate available on 80 reviews |

:::danger[Interview Trap: "Fit the vectorizer, then split"]
Anything that **learns from data** (TF-IDF vocabulary and IDF, scalers, PCA, imputers, tokenizers trained on the corpus) must be fitted on the **training split only** and merely *applied* to validation and test. Fitting on everything leaks test information into the features. Put the vectorizer and the model in one scikit-learn `Pipeline` and the problem disappears, including inside cross-validation.
:::

:::analogy[One tiny test set is a single coin-toss tournament]
Deciding which of two players is better from one round of 16 coin-tossed matches can crown either of them. Cross-validation is playing many tournaments with reshuffled draws and comparing average results: slower, but now the winner probably deserves it.
:::

:::tip[🎯 Interview angle]
**"Why is accuracy alone not enough?"** It hides the error type and is misleading on imbalanced data; read precision, recall and the confusion matrix. **"Negative review predicted positive: what error is that?"** With *negative* as the class of interest, a **false negative** (a missed complaint); with *positive* as the positive class, it's a **false positive**: always say which class you call positive. **"Your model scored 44% on 16 test reviews; what do you do?"** Check for leakage (fit transforms on train only), then use cross-validation, then get more data.
:::

## ⚙️ Going Deeper (Internals)

**What `TfidfVectorizer` does by default**
- Lowercases, tokenizes with the regex `(?u)\b\w\w+\b` (so **single-character tokens are dropped**), counts terms, applies smoothed IDF $\ln\frac{1+N}{1+\text{df}} + 1$, and L2-normalizes each row. Useful options: `ngram_range=(1, 2)` for bigrams (captures "not good"), `min_df` / `max_df` to drop very rare or very common terms, `sublinear_tf=True` to use $1 + \ln(\text{tf})$, and `stop_words`.
- The output is a SciPy **sparse CSR matrix**: only non-zero values are stored. An 80 × 234 matrix with ~5 words per review is >97% zeros. Calling `.toarray()` on a large corpus can exhaust memory.

**Why Naive Bayes works on TF-IDF even though it's defined on counts**
- `MultinomialNB` only needs non-negative feature values; with TF-IDF it effectively uses fractional "counts". Classic theory assumes raw counts (use `CountVectorizer`), but TF-IDF inputs often work as well or better in practice.

**Naive Bayes in log space**
- Multiplying hundreds of small probabilities underflows to 0, so implementations add **log probabilities**: $\log P(c) + \sum_t \text{count}(t, d)\log P(t \mid c)$. That makes NB a **linear classifier** in the features, like the SVM.

**`LinearSVC` details**
- Default loss is **squared hinge**, regularization L2, `C=1.0`. It returns decision scores, not probabilities: wrap it in `CalibratedClassifierCV` if you need probabilities. `SVC(kernel="linear")` solves the same problem much more slowly on large data. Non-linear kernels rarely help on high-dimensional sparse text.

**Train/serving consistency**
- A fitted `Pipeline([("tfidf", TfidfVectorizer()), ("clf", MultinomialNB())])` can be saved with `joblib.dump` and loaded in a service, so the exact vocabulary and IDF values travel with the model. A custom preprocessing function must be shipped too (or wrapped as the vectorizer's `preprocessor` / `tokenizer`).

**Bigrams and negation**
- With `ngram_range=(1, 2)`, "not good" yields the features `not`, `good` and `not good`. The bigram lets a linear model learn that "not good" is negative even though "good" alone is positive, a cheap partial fix for word order.

:::danger[Interview Trap: "TF-IDF + linear models are obsolete"]
They're still the first baseline in industry: they train in seconds, need little data, are interpretable (you can list each word's weight), and on many short-text tasks with limited data they compete with much heavier models. Always build one before fine-tuning a Transformer, so you know what the expensive model actually adds.
:::

## 💻 Python Implementation

:::tip[Run it yourself]
Verified on Python 3.13, NLTK 3.10, gensim 4.4 and scikit-learn 1.9; outputs in the comments come from those runs. NLTK data: `punkt_tab`, `stopwords`, `wordnet`, `averaged_perceptron_tagger_eng` (and `brown` for block 6). `data` stands for the notebook's 80 labelled reviews (40 positive, 40 negative).
:::

### 1. Basic cleaning

Lowercase, strip punctuation and digits, collapse whitespace.

```python showLineNumbers
import re

def basic_cleaning(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)                         # punctuation & symbols
    text = re.sub(r"\d+", "", text)                             # numbers (task-dependent!)
    return re.sub(r"\s+", " ", text).strip()                    # collapse & trim spaces

print(basic_cleaning("   NLP is AMAZING... isn't it???   "))   # nlp is amazing isnt it
print(basic_cleaning("Deep Learning costs $1000 sometimes!!!"))  # deep learning costs sometimes
```

### 2. `split()` vs `word_tokenize`

A real tokenizer separates punctuation and contractions.

```python showLineNumbers
from nltk.tokenize import word_tokenize

text = "I can't pay $1,000.50 today, e-mail me!"
# error-next-line
print(text.split())             # ❌ ['I', "can't", 'pay', '$1,000.50', 'today,', 'e-mail', 'me!']
# success-next-line
print(word_tokenize(text))      # ✅ ['I', 'ca', "n't", 'pay', '$', '1,000.50', 'today', ',', 'e-mail', 'me', '!']
print(word_tokenize("don't"))   # ['do', "n't"]
```

### 3. The reusable preprocessing function

One function for training and prediction alike; an optional stop-word set makes it task-specific.

```python title="preprocessing.py" showLineNumbers
import re
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

STOP_WORDS = set(stopwords.words("english"))                    # 198 words, includes 'not', 'no', 'nor'
NEGATIONS = {"not", "no", "nor", "never", "neither", "none"}
# highlight-next-line
STOP_WORDS_SENTIMENT = STOP_WORDS - NEGATIONS                   # keep negations for sentiment
lemmatizer = WordNetLemmatizer()

def to_wordnet_pos(tag):
    return {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}.get(tag[0], wordnet.NOUN)

def preprocess_text(text, stop_words=STOP_WORDS):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in word_tokenize(text) if t not in stop_words]
    return [lemmatizer.lemmatize(w, to_wordnet_pos(tag)) for w, tag in pos_tag(tokens)]

print(preprocess_text("I am learning NLP and building AI systems in 2025!!!"))
# ['learn', 'nlp', 'building', 'ai', 'system']
# error-next-line
print(preprocess_text("This product is not good"))                         # ❌ ['product', 'good']
# success-next-line
print(preprocess_text("This product is not good", STOP_WORDS_SENTIMENT))   # ✅ ['product', 'not', 'good']
```

### 4. POS tags change the lemma

The same three words with and without part-of-speech information.

```python showLineNumbers
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

words = ["running", "better", "studies"]
lem = WordNetLemmatizer()
print(pos_tag(words))                                           # [('running', 'VBG'), ('better', 'JJR'), ('studies', 'NNS')]
print([lem.lemmatize(w) for w in words])                        # ['running', 'better', 'study']
print([lem.lemmatize(w, to_wordnet_pos(t)) for w, t in pos_tag(words)])   # ['run', 'good', 'study']
```

### 5. Word2Vec on tiny data learns (almost) nothing

Compare the trained vectors with their untrained starting point.

```python title="w2v_tiny.py" showLineNumbers
import numpy as np
from gensim.models import Word2Vec

sentences = [preprocess_text(r) for r in reviews]               # the notebook's 54 reviews → 298 tokens

untrained = Word2Vec(vector_size=50, window=3, min_count=1, workers=1, seed=42)
untrained.build_vocab(sentences)                                # random initial vectors, no training
model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, workers=1, seed=42)

print(len(model.wv.index_to_key))                               # 210 words, most seen once
print(np.linalg.norm(model.wv["delivery"] - untrained.wv["delivery"]))   # ≈ 0.0003: barely moved
print(np.linalg.norm(untrained.wv["delivery"]))                 # ≈ 0.087
# error-start
print(model.wv.most_similar("delivery", topn=3))                # [('loud', 0.31), ('checkout', 0.31), ('coupon', 0.3)]
print(untrained.wv.most_similar("delivery", topn=3))            # identical: these "neighbours" are random init
# error-end
```

### 6. Word2Vec where it works: a real corpus

A million tokens gives meaningful neighbours, and shows the antonym problem.

```python title="w2v_brown.py" showLineNumbers
import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import brown

corpus = [[w.lower() for w in s if w.isalpha()] for s in brown.sents()]   # 981,716 tokens
sg = Word2Vec(corpus, vector_size=100, window=5, min_count=5, sg=1, epochs=10, workers=1, seed=42)

print(sg.wv.most_similar("coffee", topn=2))                     # [('cups', 0.81), ('cup', 0.81)]
# highlight-next-line
print(sg.wv.most_similar("good", topn=1))                       # [('bad', 0.6)]: antonyms share contexts

avg = lambda ws: np.mean([sg.wv[w] for w in ws], axis=0)
a, b = avg(["product", "good"]), avg(["product", "not", "good"])
print(round(float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)), 3))     # 0.891: averaging dilutes "not"
```

### 7. t-SNE needs perplexity below the number of points

Visualization only; nothing here is a feature.

```python showLineNumbers
import numpy as np
from sklearn.manifold import TSNE

words = model.wv.index_to_key                                   # 210 words
coords = TSNE(n_components=2, perplexity=5, random_state=42).fit_transform(model.wv[words])
print(coords.shape)                                             # (210, 2)

# error-next-line
TSNE(perplexity=30).fit_transform(np.random.rand(10, 5))        # ❌ ValueError: perplexity (30) must be less than n_samples (10)
```

### 8. TF-IDF by hand vs `TfidfVectorizer`

Smoothed IDF, then L2 normalization, reproduces the notebook's numbers.

```python title="tfidf_check.py" showLineNumbers
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

texts = [" ".join(preprocess_text(r["text"], STOP_WORDS_SENTIMENT)) for r in data]   # 80 reviews
vec = TfidfVectorizer()
X = vec.fit_transform(texts)
print(X.shape)                                                  # (80, 234): sparse CSR matrix

print(texts[0])                                                 # order arrive two day late box damage
print(math.log(81 / 3) + 1)                                     # 4.2958… = idf('box'), in 2 of 80 reviews
idf = np.array([vec.idf_[vec.vocabulary_[w]] for w in texts[0].split()])
print(np.round(idf / np.linalg.norm(idf), 4))                   # [0.292 0.3361 0.4187 0.3511 0.3906 0.4187 0.4187]
print(np.round(X[0, [vec.vocabulary_["box"], vec.vocabulary_["order"]]].toarray(), 4))   # [[0.4187 0.292 ]]
```

### 9. Naive Bayes by hand

Laplace-smoothed probabilities on four tiny documents match scikit-learn.

```python showLineNumbers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

docs = ["good fast", "good quality", "late bad", "not good late"]
labels = ["pos", "pos", "neg", "neg"]
cv = CountVectorizer()
nb = MultinomialNB(alpha=1.0).fit(cv.fit_transform(docs), labels)

pos = 0.5 * (2 + 1) / (4 + 6) * (0 + 1) / (4 + 6)              # 0.015
neg = 0.5 * (1 + 1) / (5 + 6) * (2 + 1) / (5 + 6)              # 0.0248
print(round(neg / (pos + neg), 4))                              # 0.6231
# highlight-next-line
print(nb.predict_proba(cv.transform(["good late"])).round(4))   # [[0.6231 0.3769]]: classes ['neg', 'pos']
```

### 10. The leak, and the fix with a `Pipeline`

Fit the vectorizer only on training text: the pipeline does it for you.

```python title="sentiment_pipeline.py" showLineNumbers
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

labels = [r["sentiment"] for r in data]

# error-start
X_all = TfidfVectorizer().fit_transform(texts)                  # ❌ vocabulary + IDF learned from test reviews too
X_tr, X_te, y_tr, y_te = train_test_split(X_all, labels, test_size=0.2, random_state=42, stratify=labels)
# error-end

# success-start
t_tr, t_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
nb_pipe = make_pipeline(TfidfVectorizer(), MultinomialNB()).fit(t_tr, y_tr)   # ✅ fitted on 64 training texts only
# success-end
print(len(nb_pipe[0].vocabulary_))                              # 199 (not 234): test-only words are unknown
print(nb_pipe.score(t_te, y_te))                                # 0.625
print(make_pipeline(TfidfVectorizer(), LinearSVC()).fit(t_tr, y_tr).score(t_te, y_te))   # 0.4375
```

### 11. Cross-validation instead of one tiny split

Fifty properly re-fitted splits give a stable estimate with error bars.

```python showLineNumbers
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
for name, clf in [("NB", MultinomialNB()), ("SVM", LinearSVC())]:
    scores = cross_val_score(make_pipeline(TfidfVectorizer(), clf), texts, labels, cv=cv)
    print(f"{name}: {scores.mean():.3f} ± {scores.std():.3f}")
# NB: 0.728 ± 0.101
# SVM: 0.674 ± 0.117
```

### 12. What each model learned, and predicting new reviews

Inspect word weights; the same preprocessing runs at prediction time.

```python showLineNumbers
import numpy as np

svm = make_pipeline(TfidfVectorizer(), LinearSVC()).fit(texts, labels)
coef, vocab = svm[1].coef_[0], svm[0].get_feature_names_out()   # classes ['negative', 'positive']
order = np.argsort(coef)
print([vocab[i] for i in order[:4]])                            # ['not', 'terrible', 'disappointing', 'status']
print([vocab[i] for i in order[-4:]])                           # ['premium', 'excellent', 'happy', 'experience']

new = ["Delivery was late and the packaging was damaged", "This is not a good purchase"]
# highlight-next-line
new_clean = [" ".join(preprocess_text(t, STOP_WORDS_SENTIMENT)) for t in new]   # same preprocessing as training
print(new_clean)                                                # ['delivery late packaging damage', 'not good purchase']
print(svm.predict(new_clean))                                   # ['negative' 'negative']
```

## 🚨 Interview Gotchas & Edge Cases

:::warning[The Most Common Question]
**"Build a sentiment classifier for customer reviews. Walk me through it."**

1. **Data:** labelled reviews, check class balance, hold out a test set first (stratified).
2. **Preprocess** with one shared function: lowercase, clean, tokenize, remove stop words **except negations**, lemmatize. Keep it identical at serving time.
3. **Features:** TF-IDF (with bigrams for "not good"), fitted **on training data only**, inside a `Pipeline`.
4. **Models:** Naive Bayes and a linear SVM (or logistic regression) as fast, interpretable baselines.
5. **Evaluate:** cross-validation for model selection, then the test set once; confusion matrix, per-class precision/recall/F1, with the business-critical class defined as positive.
6. **Improve:** more data, bigrams, pretrained embeddings, then a fine-tuned Transformer (BERT) when data and latency allow, compared against the baseline.
:::

- **Fitting the TF-IDF vectorizer before splitting** leaks the test vocabulary and IDF into training. Split first, or use a `Pipeline`.
- **A 16-example test set** can swing accuracy by ±25 points between random splits. Use cross-validation for comparisons, and report the spread.
- **Standard stop-word lists delete negations**: "good" and "not good" become identical. Use a sentiment-specific list.
- **Cleaning punctuation before tokenizing** turns *didn't* into `didnt`, a token no stop-word list or negation list contains.
- **`TfidfVectorizer` silently drops one-character tokens** (default `token_pattern`), and lowercases by default.
- **Calling `.toarray()`** on a large sparse TF-IDF matrix can blow up memory.
- **Word2Vec on a few hundred tokens** produces vectors that are basically their random initialization. Use pretrained embeddings for small domain data.
- **gensim isn't reproducible** with `workers > 1`. Use `workers=1`, `seed` and `PYTHONHASHSEED`.
- **t-SNE plots** don't preserve distances or cluster sizes, depend on perplexity and seed, and require perplexity < number of points.
- **Averaging word vectors** throws away order and dilutes negation (cosine 0.89 between "product good" and "product not good").
- **Appending rows to a DataFrame** leaves previously derived columns as NaN for the new rows. Recompute them.
- **Naive Bayes probabilities are over-confident**, and `LinearSVC` gives no probabilities. Calibrate if downstream code needs them.
- **Models learn dataset quirks**: *status* and *app* rank among the most negative words here only because of how these 80 reviews were written.
- **Saying "false positive" without naming the positive class** is ambiguous. State which class is positive first.

## ⚡ Essential Interview Recall

A one-screen revision list. Every point is explained in the chapters above.

**Cleaning and tokenization**
- Regex cleaning: lowercase → `[^\w\s]` → `\d+` → `\s+`; *isn't* → `isnt`. **Task-dependent**: keep numbers, punctuation, emoticons or case when they carry meaning.
- `word_tokenize` splits punctuation and contractions (*don't → do + n't*); `split()` doesn't. Transformers use subword tokenizers (BPE/WordPiece).

**Normalization**
- NLTK stop words (198) include *not, no, nor*; keep negations for sentiment.
- Porter: *machin, amaz, sometim*. Lemmatizer without POS: noun only. With POS: *running → run, better → good*. POS tagging is slower and context-sensitive.
- One preprocessing function for train and serve: avoids **training–serving skew**.

**Word2Vec and t-SNE**
- gensim: `vector_size`, `window`, `min_count`, `sg` (0 CBOW / 1 skip-gram), `epochs`, `workers`, `seed`.
- On 298 tokens the vectors barely move (0.0003) and neighbours = random init; on 1 M tokens they're meaningful. Use pretrained vectors for small data. Reproducible only with `workers=1` + seed + `PYTHONHASHSEED`.
- t-SNE: 2-D view preserving **local neighbourhoods only**; perplexity < n; visualization, not features.

**TF-IDF**
- $\text{tf} \times \text{idf}$, idf $= \ln\frac{1+N}{1+\text{df}} + 1$ (scikit-learn), then L2-normalize rows. Frequent here + rare elsewhere = high. Review 1: *box* 0.4187, *order* 0.2920.
- Averaged embeddings lose order and dilute negation; TF-IDF (+ bigrams) is the classical document baseline.

**Classifiers**
- **Multinomial NB**: $P(c)\prod P(t \mid c)^{\text{count}}$, Laplace smoothing, independence assumption; fast, data-efficient baseline. Toy example: P(neg | "good late") = 0.623.
- **Linear SVM**: maximum-margin hyperplane $\text{sign}(\mathbf{w} \cdot \mathbf{x} + b)$, hinge loss, `C`; excellent on sparse high-dimensional text.
- Top learned words: *not, terrible, disappointing* (negative); *experience, happy, excellent* (positive).

**Evaluation**
- Notebook: NB 62.5% vs SVM 43.75% on 16 test reviews, with **leakage** (vectorizer fitted on all 80).
- Fix: split first / `Pipeline` (vocabulary 199, 38 of 69 test tokens unseen). 20 splits: 37.5%–93.75%. **Repeated 5-fold CV: NB 72.8% ± 10.1%, SVM 67.4% ± 11.7%.**
- Read the confusion matrix and per-class precision/recall; name the positive class.

## 📐 Formula Sheet

Every equation on this page in one place.

| Concept | Formula | In plain words |
|---|---|---|
| Cleaning | `[^\w\s]` → ∅, `\d+` → ∅, `\s+` → one space | Strip symbols, drop digits, squeeze spaces |
| Term frequency | $\data{\text{tf}(t, d)}$ = count of $t$ in $d$ | How often here |
| Smoothed IDF | $\text{idf}(t) = \ln\frac{1 + \data{N}}{1 + \data{\text{df}(t)}} + 1$ | How rare across documents |
| TF-IDF | $\text{tf}(t, d) \times \text{idf}(t)$, then $\data{\mathbf{x}_d} / \lVert \data{\mathbf{x}_d} \rVert_2$ | Frequent here, rare elsewhere, unit length |
| Naive Bayes decision | $\pred{\hat y} = \arg\max_c P(c) \prod_{t} P(t \mid c)^{\data{\text{count}(t,d)}}$ | Prior times every word's vote |
| Laplace smoothing | $P(t \mid c) = \frac{\data{\text{count}(t,c)} + \hyper{\alpha}}{\sum_{t'} \data{\text{count}(t',c)} + \hyper{\alpha}\lvert V \rvert}$ | Add one to every count |
| SVM decision | $\pred{\hat y} = \text{sign}(\param{\mathbf{w}} \cdot \data{\mathbf{x}} + \param{b})$ | Which side of the plane |
| SVM objective | $\min \tfrac{1}{2}\lVert \param{\mathbf{w}} \rVert^2 + \hyper{C}\sum_i \max(0, 1 - \data{y_i}(\param{\mathbf{w}} \cdot \data{\mathbf{x}_i} + \param{b}))$ | Wide margin, few violations |
| Margin width | $2 / \lVert \param{\mathbf{w}} \rVert$ | Smaller weights, wider street |
| Accuracy / Precision / Recall / F1 | $\frac{TP+TN}{\text{all}}$, $\frac{TP}{TP+FP}$, $\frac{TP}{TP+FN}$, $\frac{2PR}{P+R}$ | As on the CNN evaluation page |

## Concept-Check Answers (from the notebook)

### Section 1: Raw text exploration

<details>
<summary><strong>1. Why can't we feed raw text directly into ML models?</strong></summary>

Models need numeric feature vectors of a fixed shape. Raw strings are variable-length symbol sequences full of noise (case, punctuation, spacing, numbers), so they must be cleaned, tokenized and converted to numbers (counts, TF-IDF, embeddings) first.
</details>

<details>
<summary><strong>2. What problem occurs if "Machine" and "machine" are treated differently?</strong></summary>

They become two vocabulary entries for one word: the vocabulary grows, each version gets half the training evidence, and what the model learns about one doesn't transfer to the other.
</details>

<details>
<summary><strong>3. Why are extra spaces problematic?</strong></summary>

Naive splitting on single spaces produces empty or whitespace tokens, and inconsistent spacing makes identical texts look different (different strings, different hashes). Collapsing whitespace (`\s+` → one space, then `strip()`) normalizes them.
</details>

### Section 2: Basic cleaning

<details>
<summary><strong>1. Why do we convert text to lowercase?</strong></summary>

So that case variants (*Love, LOVE, love*) map to one token, shrinking the vocabulary and pooling their evidence, at the cost of distinctions like *US/us* or *Apple/apple*.
</details>

<details>
<summary><strong>2. When should we NOT remove numbers?</strong></summary>

When quantities carry meaning for the task: ratings ("5-star"), durations ("arrived in 2 days"), prices, versions, dates, product codes or measurements.
</details>

<details>
<summary><strong>3. When can punctuation be useful?</strong></summary>

For sentiment and emotion ("!!!", "?!"), emoticons (":(" is strongly negative), questions ("?"), sentence boundaries, abbreviations, decimals and URLs, and for any model that reads full sentences (Transformers).
</details>

### Section 3: Tokenization

<details>
<summary><strong>1. Why is tokenization required?</strong></summary>

Models work on discrete units: counting words, building a vocabulary, looking up embeddings and creating features all need the text split into tokens first.
</details>

<details>
<summary><strong>2. Why is simple split not always sufficient?</strong></summary>

`split()` leaves punctuation attached ("great!", "today,"), can't separate contractions (*don't*), and doesn't understand numbers, URLs or hyphens. `word_tokenize` gives `['do', "n't"]` and splits off punctuation consistently.
</details>

<details>
<summary><strong>3. What is a token?</strong></summary>

The basic unit a model processes after splitting: usually a word, but also punctuation, a contraction piece (*n't*) or a subword (*token + ization*).
</details>

### Section 4: Stop words, stemming, lemmatization

<details>
<summary><strong>1. Why remove stop words?</strong></summary>

They are very frequent and carry little topic-specific meaning, so removing them shrinks the vocabulary and feature vectors and reduces noise for bag-of-words models.
</details>

<details>
<summary><strong>2. Why is stemming sometimes problematic?</strong></summary>

It chops blindly: non-words (*machin, amaz, sometim, studi*), unrelated words merged (*universe/university → univers*, *organization/organ → organ*), and irregular forms missed (*ran, better*).
</details>

<details>
<summary><strong>3. When should we avoid stop-word removal?</strong></summary>

Sentiment analysis (negations), question answering (*who, what*), machine translation, summarization, phrase-sensitive tasks, and any sequence model or Transformer that needs the full sentence.
</details>

<details>
<summary><strong>4. Why is POS tagging needed before lemmatization?</strong></summary>

The correct lemma depends on the word's grammatical role: *better → good* only as an adjective, *running → run* only as a verb. WordNet's lemmatizer assumes a noun otherwise.
</details>

<details>
<summary><strong>5. Which is more accurate: stemming or POS lemmatization?</strong></summary>

POS-aware lemmatization: it returns real dictionary words and handles irregular forms (*running, better, studies → run, good, study*), whereas stemming produces crude, sometimes wrong stems.
</details>

<details>
<summary><strong>6. When might POS tagging slow down the pipeline?</strong></summary>

On large corpora (millions of documents) or real-time, low-latency services, because the tagger runs a statistical model over every sentence. It's also less reliable on ungrammatical text (tweets, search queries), so the extra cost may not pay off.
</details>

### Section 5: Full preprocessing pipeline

<details>
<summary><strong>1. Why build preprocessing as a function?</strong></summary>

Reusability and, above all, **consistency**: the exact same steps run on training data, evaluation data and live user input, and changes happen in one place.
</details>

<details>
<summary><strong>2. Where will this function be used?</strong></summary>

On the training dataset, on validation/test data, on new user input at prediction time (the deployed API or batch scoring job), and when building embeddings or features.
</details>

<details>
<summary><strong>3. What happens if preprocessing is inconsistent?</strong></summary>

Training–serving skew: production text produces tokens or features the model never saw, the feature distribution shifts, and accuracy drops silently with no error.
</details>

### Section 6A: How Word2Vec learns

<details>
<summary><strong>1. What does Word2Vec take as input?</strong></summary>

A corpus of tokenized sentences: a list of token lists, e.g. `[['love', 'machine', 'learning'], ['nlp', 'amazing', 'isnt'], ...]`.
</details>

<details>
<summary><strong>2. What is the output for each word?</strong></summary>

A dense vector of length `vector_size` (10 in the demo, 50 for the reviews), accessed with `model.wv[word]`.
</details>

<details>
<summary><strong>3. Why do similar-context words become close in vector space?</strong></summary>

Training adjusts vectors so a word predicts its neighbours (or vice versa); words with similar neighbours receive similar updates and converge to similar vectors. That's the distributional hypothesis, and it needs enough data to actually happen.
</details>

### Section 6B: Reusable Word2Vec training

<details>
<summary><strong>1. Why wrap Word2Vec training into a function?</strong></summary>

To reuse the same configuration across datasets, experiment with parameters (vector size, window, `min_count`, skip-gram vs CBOW) in one place, and keep training reproducible and deployable.
</details>

<details>
<summary><strong>2. Why can embeddings differ across domains?</strong></summary>

Meaning is learned from context, and contexts differ: *cheap* in product reviews sits near *quality* and *fabric*; in finance text, near *stock* and *valuation*. Each corpus produces its own geometry.
</details>

<details>
<summary><strong>3. Why may results on small datasets be weak?</strong></summary>

Each word appears only a few times, so there are too few co-occurrence examples to move vectors away from their random initialization. Here the vectors moved by 0.0003 and the "neighbours" matched the untrained ones exactly. Use pretrained embeddings or far more text.
</details>

### Section 7: t-SNE

<details>
<summary><strong>1. Why do we use t-SNE?</strong></summary>

To project high-dimensional embeddings (50-D) into 2-D so humans can see which words sit near each other.
</details>

<details>
<summary><strong>2. Does t-SNE preserve exact distances?</strong></summary>

No. It preserves local neighbourhoods; global distances, cluster sizes and gaps between clusters are not meaningful, and the layout changes with perplexity and random seed.
</details>

<details>
<summary><strong>3. Why do some clusters overlap?</strong></summary>

Words used across several topics (*refund* in payment and support reviews) genuinely share contexts; small data gives noisy vectors; and compressing 50 dimensions into 2 must distort some relationships.
</details>

### Section 8: From words to documents

<details>
<summary><strong>1. What does Word2Vec mainly represent: words or full documents?</strong></summary>

**Words**: one vector per vocabulary word. Document vectors need an extra step (averaging, TF-IDF weighting, Doc2Vec, or a sequence model).
</details>

<details>
<summary><strong>2. Why is naive averaging of word embeddings risky for sentiment?</strong></summary>

It ignores word order, gives every word equal weight, and dilutes negation: the averages of "product good" and "product not good" have cosine similarity 0.89, nearly the same vector.
</details>

<details>
<summary><strong>3. Why can TF-IDF work well with Naive Bayes and SVM?</strong></summary>

It gives sparse, high-dimensional document vectors in which informative, discriminative words get high weights and common words low weights. Naive Bayes thrives on word-occurrence features, and linear SVMs excel with many sparse features where classes are close to linearly separable.
</details>

### Section 9: Labelled dataset

<details>
<summary><strong>1. Why do we need labels for sentiment classification?</strong></summary>

The model learns the mapping from text to sentiment by example; labels supply the correct answer to learn from and to measure errors against.
</details>

<details>
<summary><strong>2. What makes this a supervised learning problem?</strong></summary>

Each input (review text) is paired with a known target (positive/negative), and the model is trained to predict the target from the input.
</details>

<details>
<summary><strong>3. Why is a slightly larger dataset better than a tiny toy one here?</strong></summary>

More vocabulary coverage and more examples per sentiment word, so the models learn real patterns and the train/test comparison is less random. 80 reviews is still small: different splits swing accuracy from 37.5% to 93.75%.
</details>

### Section 10: Preprocessing for sentiment

<details>
<summary><strong>1. Why should preprocessing change depending on the task?</strong></summary>

Information that is noise for one task is signal for another: negations decide sentiment but not topic, numbers matter for delivery complaints, case matters for named entities. Preprocessing must keep what the task needs.
</details>

<details>
<summary><strong>2. Why is keeping negation important in sentiment analysis?</strong></summary>

Without *not*, "This product is good" and "This product is not good" become the identical `['product', 'good']`. With it, *not* becomes the single strongest negative feature both models learn.
</details>

<details>
<summary><strong>3. What is the role of POS-aware lemmatization here?</strong></summary>

It merges inflected forms into one base word (*damaged → damage, arrived → arrive, loved → love*), so each sentiment word's evidence is pooled into one feature, giving a smaller, denser vocabulary of real words.
</details>

### Section 11: TF-IDF

<details>
<summary><strong>1. Why do we need TF-IDF after preprocessing?</strong></summary>

Preprocessing still leaves lists of words; the classifiers need fixed-length numeric vectors. TF-IDF turns each review into one, weighting words by importance.
</details>

<details>
<summary><strong>2. What kind of words get high TF-IDF scores?</strong></summary>

Words frequent in the given document but rare across the corpus, the distinctive ones (*box, damage* = 0.4187 in review 1, vs *order* = 0.2920).
</details>

<details>
<summary><strong>3. Why do common words across many documents get lower importance?</strong></summary>

A word that appears everywhere can't distinguish one document (or class) from another, so its IDF is low: a word in all documents gets the minimum smoothed IDF of 1.
</details>

### Section 12: Training NB and SVM

<details>
<summary><strong>1. Why do we split the dataset into training and testing sets?</strong></summary>

To measure performance on data the model hasn't seen, which estimates real-world generalization. Everything fitted (including the TF-IDF vectorizer) must use only the training part.
</details>

<details>
<summary><strong>2. Why is SVM often strong for text classification?</strong></summary>

Text features are high-dimensional and sparse, classes are often linearly separable there, and maximizing the margin controls overfitting when there are more features than samples.
</details>

<details>
<summary><strong>3. Why is Naive Bayes still useful even though it is simple?</strong></summary>

It's extremely fast, needs little data, rarely overfits, gives a strong baseline, and its ranking of classes is often correct even though its independence assumption is false. On these 80 reviews it beat the SVM in cross-validation (72.8% vs 67.4%).
</details>

### Section 13: Evaluation

<details>
<summary><strong>1. Why is accuracy alone not enough?</strong></summary>

It hides which classes the errors fall in and is misleading on imbalanced data. Two models with equal accuracy can make very different mistakes; precision, recall and the confusion matrix reveal them.
</details>

<details>
<summary><strong>2. What does the confusion matrix tell us?</strong></summary>

The count of every actual/predicted combination: correct predictions on the diagonal, each error type off it. Here NB caught 6 of 8 positives but only 4 of 8 negatives.
</details>

<details>
<summary><strong>3. If a negative review is predicted as positive, what kind of error is that?</strong></summary>

It depends on which class you call positive. If **positive sentiment** is the positive class, it's a **false positive**. If **negative** reviews are the class of interest (finding complaints), it's a **false negative**: a missed complaint, usually the costlier mistake. State the positive class first.
</details>
