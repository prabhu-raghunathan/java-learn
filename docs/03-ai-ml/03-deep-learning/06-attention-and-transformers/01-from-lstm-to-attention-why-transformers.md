---
title: "From LSTM to Attention: Why Transformers Became Necessary"
sidebar_position: 1
---

import {AttentionWeightsPlayground, SelfAttentionMatrix} from '@site/src/components/Widgets';

:::tip[TL;DR (30-Second Interview Pitch)]
LSTMs fixed the RNN's memory, but not its **chain**: token $t$ still waits for token $t-1$, information between distant words travels through every step in between, and the whole past is squeezed into one hidden vector. In seq2seq translation the squeeze was extreme: the decoder saw only **one fixed context** $c = h_T$. **Attention** replaces that with a fresh, weighted look-back at every step: score every token, turn the scores into weights with **softmax** ($\alpha_i \ge 0$, $\sum \alpha_i = 1$), and take the **weighted sum of value vectors**. On "The movie was not good", scores [0.2, 0.4, 0.1, 2.0, 1.8] become weights [0.071, 0.086, 0.064, 0.428, 0.351], and the context [0.381, 0.472] is shaped by "not" and "good", unlike the plain average [0.28, 0.24]. Scores come from **Query–Key** similarity; content comes from **Values**. The **Transformer** (Vaswani et al., 2017) made attention the backbone itself: every token talks to every token in one parallel step (path length O(1), cost O(n²·d)), with positional information, feed-forward layers, residuals and layer norm around it.
:::

<FormulaKey />

:::info[Where this page sits]
**Before:** [LSTM in Practice: Text Generation and Temperature Sampling](../05-lstm-and-gru/02-lstm-text-generation-temperature-sampling.md) ended with what recurrence still can't do.
**This page:** notebook *14 Transformer 1* (the conceptual lecture): why recurrence is a bottleneck, the fixed-context problem, selective focus, soft alignment, weighted context vectors, Query–Key–Value, attention through examples and the self-attention matrix, "Attention Is All You Need", and the Transformer big picture.
**Next:** notebook *15 Full Self-Attention* formalizes $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$ with learned projections.
:::

## 🧭 Start Here

For four pages we have been fighting one problem: how does a model remember what it read? RNNs added a hidden state, LSTMs and GRUs added gates so the memory survives longer. They work well, and yet almost every major model since 2018 (BERT, GPT, modern translation systems, vision and multimodal models) dropped recurrence. This page answers **why**. The answer is not "LSTMs forget". It is that the **shape of the computation** is wrong for long text and modern hardware: one token at a time, one vector to carry everything, and long chains between related words.

The fix is an idea humans use without thinking: **selective focus**. When you read "The movie was not good", you do not weigh every word equally; "not" and "good" decide the sentiment. Attention gives a model that ability explicitly: for each prediction it scores every token, turns the scores into weights, and builds a custom summary in which relevant tokens dominate. Self-attention lets **every token do this for every other token**, all at once. The Transformer is the architecture built entirely around that operation.

This is a concept lecture. The numbers on this page come from the notebook's toy examples (hand-picked scores, 2-dimensional "value" vectors and a hand-designed attention matrix), all re-run and verified. Where those values come from in a real model (learned projections $W_Q, W_K, W_V$ and the scaled dot product) is the subject of the next page. The chain of ideas:

1. Static text (TF-IDF) → sequences (RNN) → better memory (LSTM/GRU).
2. Recurrence is still a bottleneck: **sequential**, **long paths**, **compressed state**.
3. Seq2seq made it worse: one **fixed context** vector for the whole sentence.
4. **Selective focus**: look back at the relevant tokens, differently at each step.
5. **Soft alignment**: softmax turns relevance scores into weights.
6. The **weighted context vector**: a learned, weighted average.
7. **Query–Key–Value**: where the scores and the content come from.
8. **Self-attention**: a token-by-token weight matrix; meaning updated by context.
9. **"Attention Is All You Need"**: attention as the backbone, not an add-on.
10. The **Transformer big picture**: embeddings, positions, attention, feed-forward, residuals, normalization.

| New concept | One-line meaning |
|---|---|
| **Recurrence** | $h_t = f(x_t, h_{t-1})$: each state depends on the previous one, so steps run in order |
| **Path length** | Number of computation steps information needs to travel between two tokens |
| **Fixed-length context** | The seq2seq decoder's single summary $c = h_T$ of the whole source sentence |
| **Attention** | A weighted look-up over all tokens, with weights computed for the current need |
| **Relevance score** $e_i$ | A raw number saying how useful token $i$ is right now |
| **Attention weight** $\alpha_i$ | The softmax of the scores: non-negative, summing to 1 |
| **Soft alignment** | Every token gets some weight (vs hard selection of exactly one) |
| **Context vector** | $c = \sum_i \alpha_i v_i$: the weighted summary used for the prediction |
| **Query / Key / Value** | What a token looks for / what it offers for matching / what it passes on |
| **Self-attention** | Attention where a sequence attends to itself: every token to every token |
| **Attention matrix** | $n \times n$ weights; row = the token asking, column = the token attended to |
| **Transformer** | An architecture built on self-attention, without recurrence or convolution |

:::analogy[Reading with a highlighter]
An LSTM reads a contract once, left to right, and must answer every later question from the notes it scribbled on one index card. Attention lets it keep the whole contract on the desk and, for each question, highlight the clauses that matter before answering. Different questions, different highlights; the contract itself never has to fit on the card.
:::

## 1. The Road So Far: From Counting Words to Gated Memory

NLP models went through three phases before attention. **Static representations** such as TF-IDF turned a document into word-importance counts. They are fast and surprisingly strong for topic classification, but they have **no word order and no context**: "dog bites man" and "man bites dog" contain the same words, so they get the same vector (their TF-IDF cosine similarity is 1.0, as measured on the [LSTM page](../05-lstm-and-gru/01-lstm-and-gru-gated-memory.md)).

**Sequential models** fixed order. An RNN reads one token at a time and carries a hidden state, $h_t = f(x_t, h_{t-1})$, so "dog bites man" and "man bites dog" now produce different states. But the vanilla RNN's memory is fragile: gradients shrink or blow up as they are multiplied back through many steps.

**Gated models**, LSTM and GRU, gave that memory a protected path: an additive cell state and learned gates that decide what to keep, write and reveal. They powered sentiment analysis, sequence labelling, translation and text generation for years. So the question this lecture asks is a fair one: if LSTMs were already good, why did the field move on?

![Recurrence chain vs self-attention](/img/deep-learning/diagrams/chain-vs-attention.svg)

*How to read it:* top, the recurrent view: information about "The" reaches "good" only by passing through every state in between (4 sequential steps for 5 tokens, $n - 1$ in general). Bottom, the self-attention view: every pair of tokens is connected directly, so any token reaches any other in one step, and all $n^2$ pairs are computed at the same time.

:::tip[🎯 Interview angle]
**"Summarize the evolution from TF-IDF to Transformers in one minute."** TF-IDF: word importance, no order or context. RNN: order via a hidden state, but fragile memory. LSTM/GRU: gated, additive memory that survives longer, but still a step-by-step chain into one fixed-size state. Attention: look back at all tokens with learned weights. Transformer: make attention the backbone, so every token interacts with every token in parallel.
:::

## 2. Why Recurrence Is Still a Bottleneck

Look at the recurrence $h_t = f(x_t, h_{t-1})$ again. It says the state at step $t$ **cannot be computed** until the state at step $t-1$ exists. For a 100-word sentence, word 100 waits for word 99, which waits for word 98, and so on: 100 dependent steps, no matter how many processors you have. That is **problem 1: sequential dependency**. GPUs and TPUs are built for the opposite pattern, huge matrix multiplications over many values at once. A recurrent model can parallelize across the batch and inside each step's matrix multiply, but **never across time**, so long sequences leave most of the hardware idle.

**Problem 2: long information paths.** If token 1 should influence token 50, its information has to be carried forward through 49 state updates. Each update mixes in new input, squashes values through nonlinearities and multiplies by weights. LSTM gates make this chain far more forgiving than a vanilla RNN's, but it is still a chain: the longer it is, the harder optimization becomes, the more the information is diluted, and the slower the model learns distant relationships.

**Problem 3: hidden-state compression.** At every step, $h_t$ is a single fixed-size vector that must summarize *everything so far*: the subject, the tense, the sentiment, names, negations, which noun a pronoun might refer to. A 512-dimensional vector holding all of that for a long paragraph is a very demanding job, and whatever doesn't fit is lost.

The key insight is that LSTM improved the **memory** but did not remove the **chain**. The researchers' question became: why must token 50 wait for token 49, and why can't tokens interact directly? The original Transformer paper quantified the difference per layer, for sequence length $n$, representation size $d$ and convolution kernel width $k$:

| Layer type | Compute per layer | Sequential operations | Maximum path length |
|---|---|---|---|
| **Self-attention** | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| **Recurrent** (RNN/LSTM) | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| **Convolutional** | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k n)$ |

Self-attention wins on the two columns that matter for learning and hardware (constant sequential depth and constant path length) and pays with compute and memory that grow with $n^2$. For typical sentences, where $n$ is smaller than $d$, it is even cheaper per layer than recurrence.

<Animation src="attention/path-length" caption="Two ways for information from The to reach good. Watch the recurrent dot hop through every token, four sequential steps, while the self-attention dot takes one direct arc, with all the other pairs drawn at the same time." />

:::danger[Interview Trap: "LSTMs can't be parallelized at all"]
They can be parallelized **across the batch** and **inside** each time step (the gate matrix multiplies run on the GPU). What they cannot do is compute step $t$ before step $t-1$, so the time dimension is serial. Self-attention removes exactly that serial dependency during training. (At generation time even a Transformer decoder produces one token at a time; the parallelism win is in training and in reading the prompt.)
:::

:::tip[🎯 Interview angle]
**"Why is recurrence a bottleneck if LSTMs already solve vanishing gradients?"** Three reasons that gates don't fix: computation is **sequential** in time ($O(n)$ dependent steps, so GPUs sit idle), information between distant tokens travels an **$O(n)$-long path**, and the whole past is **compressed into one fixed-size state**. LSTM improved the memory, not the chain. Self-attention gives $O(1)$ sequential steps and $O(1)$ path length, at the price of $O(n^2)$ pairwise work.
:::

## 3. The Fixed-Length Context Problem in Seq2seq

The weakness showed up most sharply in **sequence-to-sequence** (seq2seq) translation, the dominant design around 2014. An **encoder** LSTM reads the source sentence $x_1, \ldots, x_T$, and its final hidden state $h_T$ becomes the **context vector** $c$. A **decoder** LSTM then generates the translation one word at a time, and every step is conditioned on that same $c$: $y_t = g(y_{t-1}, s_{t-1}, c)$, where $s_{t-1}$ is the decoder's own state.

So the entire meaning of the source sentence, however long, has to pass through one vector. Take "The student who studied all night before the difficult exam finally passed confidently." To translate the end of the sentence well, the decoder needs the subject (student), the earlier event (studied), the result (passed) and the manner (confidently), and all of it must survive the trip through $c$. The notebook's picture is pouring twenty facts into one small bottle: some survive, some blur. Longer sentences mean more facts per bottle, which is why these models degraded noticeably as sentences grew.

There is a second, subtler problem. Different output words need **different** parts of the input: generating a verb needs the source verb, a noun needs the noun phrase, a pronoun needs its antecedent. But a fixed $c$ gives every decoding step the **same** summary. What the decoder really wants is to look back at the source and pick out what matters *for the word it is producing right now*.

![Encoder-decoder with one fixed context vector](/img/deep-learning/diagrams/seq2seq-bottleneck.svg)

*How to read it:* the encoder (left) reads $x_1 \ldots x_4$ and hands only its last state $h_T$ to the decoder as $c$. The red arrows show the same $c$ feeding every decoder step, which is the bottleneck: every output word sees one identical summary.

<Formula name="Seq2seq with a fixed context" tag="The bottleneck" words="The encoder compresses the whole source sentence into its last hidden state, and every decoder step is conditioned on that one vector." remember="One bottle for the whole sentence.">

$$
\begin{aligned}
\pred{h_t} &= f(\data{x_t}, \pred{h_{t-1}}), \qquad \pred{c} = \pred{h_T} \\
\pred{y_t} &= g(\pred{y_{t-1}}, \pred{s_{t-1}}, \pred{c})
\end{aligned}
$$

- $\pred{h_t}$: encoder state; $\pred{s_{t-1}}$: decoder state; $\pred{c}$: the single context vector
- the same $\pred{c}$ appears in every output step $t$

**Example:** for a 4-word source and a 3-word translation, the decoder's three steps all read the same $\pred{c} = \pred{h_4}$.

</Formula>

Bahdanau, Cho and Bengio (2014) fixed this by keeping **all** encoder states $h_1, \ldots, h_T$ and computing a **new context for every output step**: score each encoder state against the current decoder state, softmax the scores, and take the weighted sum. That was the first widely used attention mechanism, still attached to an RNN.

![Encoder-decoder with attention](/img/deep-learning/diagrams/seq2seq-attention.svg)

*How to read it:* every encoder state $h_i$ sends a line to the $\Sigma$ node; the line's thickness stands for its weight $\alpha_{t,i}$ at this output step (thicker means more attention; the thicknesses are illustrative). The weighted sum $c_t$ goes into the decoder together with its previous state $s_{t-1}$. At the next step the weights are recomputed, so $c_{t+1}$ can focus somewhere else.

<Formula name="Attention in seq2seq" tag="Dynamic context" words="At every output step, score each encoder state against the decoder state, turn the scores into weights with softmax, and use the weighted sum of encoder states as this step's context." remember="New look-back for every output word.">

$$
\begin{aligned}
\pred{e_{t,i}} &= \operatorname{score}(\pred{s_{t-1}}, \pred{h_i}) \\
\pred{\alpha_{t,i}} &= \frac{e^{\pred{e_{t,i}}}}{\sum_j e^{\pred{e_{t,j}}}} \\
\pred{c_t} &= \sum_i \pred{\alpha_{t,i}}\,\pred{h_i}
\end{aligned}
$$

- $\pred{e_{t,i}}$: how relevant source position $i$ is for output step $t$
- $\pred{\alpha_{t,i}}$: the normalized weight; for each $t$ the weights sum to 1
- $\pred{c_t}$: a different context vector at every output step

**Example:** in implementation block 4 below, three decoder steps over six random encoder states put their largest weight on positions 2, 2 and 5 (0.298, 0.416 and 0.585), so the three context vectors differ, while the fixed design would reuse $\pred{h_6}$ three times.

</Formula>

:::danger[Interview Trap: "Attention was invented by the Transformer"]
Attention came first, in 2014, as a helper for RNN encoder-decoders (Bahdanau et al.; Luong et al. followed in 2015 with multiplicative scores). The Transformer's contribution in 2017 was to **drop the RNN** and build the whole model from attention.
:::

:::tip[🎯 Interview angle]
**"What was the problem with the original encoder-decoder, and how did attention fix it?"** The encoder compressed the whole source into its last state $h_T$, and every decoder step used that same vector, so long sentences lost information and every output word saw the same summary. Attention keeps all encoder states and, at each decoder step, scores them against the decoder state, softmaxes the scores and takes a weighted sum, giving a fresh, step-specific context $c_t$.
:::

## 4. Selective Focus: The Intuition Behind Attention

Humans do not read with uniform attention. In "The movie was not good at all", the sentiment is decided by **not** and **good**; "the", "was" and "at" barely matter. In "The product is good but not durable", a question about durability should focus on **not durable**, not on the earlier "good". In "Riya gave Neha her notebook because she trusted her", working out who "she" is means comparing the candidate names directly.

Recurrent models had no explicit mechanism for this. Whatever was relevant had to be **carried inside the hidden state**, mixed together with everything else, hoping it was still there when needed. Attention flips the question around: instead of storing everything and hoping, the model **asks, at the moment of prediction, which tokens matter right now**, and retrieves them. The focus is dynamic (it changes with each prediction) and task-dependent (sentiment, translation and question answering focus on different words in the same sentence).

That is the core idea in one line: **same memory for everything** is replaced by **dynamic, task-dependent focus**. Everything else on this page (softmax weights, context vectors, queries and keys, the Transformer) is machinery for doing this well.

:::analogy[A spotlight, not a floodlight]
A floodlight lights the whole stage evenly; you see everything and nothing stands out. A spotlight operator follows the action, brighter on whoever is speaking, dim on the rest, and the focus moves scene by scene. Attention is the spotlight operator, and it re-aims for every word the model predicts.
:::

:::tip[🎯 Interview angle]
**"Explain attention to a non-technical person."** When the model makes a prediction, it looks back over all the words and decides how much each one matters for *this* prediction, then builds its answer mostly from the important ones. In "the movie was not good", it leans on "not" and "good". The focus is recomputed for every prediction, so different outputs can look at different words.
:::

## 5. Soft Alignment: Turning Relevance Scores into Weights

Suppose the model must focus on the relevant words. The simplest option is **hard selection**: pick exactly one token (the highest-scoring one) and ignore the rest. It is intuitive but bad for learning. Choosing an index is a step function: tiny changes in the scores either don't change the choice at all (zero gradient) or flip it completely (a jump). Gradient descent gets no useful signal, so training is unstable. It also throws away information when two words matter together, as "not" and "good" do.

**Soft selection**, or **soft alignment**, gives every token a weight instead: large for relevant tokens, small for the rest. Everyone contributes, but unequally, and the weights change smoothly with the scores, so gradients flow. To be usable, weights should be **positive**, **comparable** across tokens and **sum to 1**, so they behave like a distribution of attention. The **softmax** function does exactly this: exponentiate each score (making it positive and amplifying differences), then divide by the total.

Take the notebook's sentiment example. The relevance scores for "The movie was not good" are 0.2, 0.4, 0.1, 2.0 and 1.8. Exponentiating gives 1.221, 1.492, 1.105, 7.389 and 6.050, which sum to 17.257. Dividing gives the weights in the table: "not" and "good" together take 78% of the attention, and the three filler words share the remaining 22%. Nothing is ignored completely, but the result is clearly dominated by the negation and the adjective.

| Token | Raw score $e_i$ | $e^{e_i}$ | Weight $\alpha_i$ |
|---|---|---|---|
| The | 0.2 | 1.221 | **0.0708** |
| movie | 0.4 | 1.492 | **0.0864** |
| was | 0.1 | 1.105 | **0.0640** |
| not | 2.0 | 7.389 | **0.4282** |
| good | 1.8 | 6.050 | **0.3506** |
| **Sum** | | 17.257 | **1.0000** |

<Formula name="Attention weights (softmax over scores)" tag="Soft alignment" words="Exponentiate each relevance score and divide by the total, so every weight is positive and all weights add up to 1; bigger scores get exponentially bigger weights." remember="Exponentiate, then share out.">

$$
\pred{\alpha_i} = \frac{e^{\pred{e_i}}}{\sum_{j=1}^{n} e^{\pred{e_j}}}, \qquad \pred{\alpha_i} > 0, \qquad \sum_i \pred{\alpha_i} = 1
$$

- $\pred{e_i}$: raw relevance score of token $i$ (any real number)
- $\pred{\alpha_i}$: its attention weight
- adding the same constant to every score changes nothing; multiplying all scores by a factor above 1 sharpens the weights

**Example:** scores [0.2, 0.4, 0.1, 2.0, 1.8] → weights [0.0708, 0.0864, 0.0640, 0.4282, 0.3506]; the same scores × 5 → [0.0001, 0.0002, 0.0001, 0.7308, 0.2688], almost a hard choice.

</Formula>

<Animation src="attention/soft-alignment" caption="The five relevance scores pass through softmax. Watch the bars for not and good grow to 0.428 and 0.351 while the others stay small, then see the weighted context 0.38 and 0.47 next to the plain average 0.28 and 0.24." />

Two properties are worth saying out loud. Softmax **only depends on differences** between scores: adding 100 to every score gives exactly the same weights. And it is **exponential**, so a gap of 2 in score is a factor of $e^2 \approx 7.4$ in weight. Raising one score increases its weight and lowers all the others, because they share a fixed total of 1: raising "movie" from 0.4 to 1.0 moves its weight from 0.086 to 0.147 and lowers "not" from 0.428 to 0.400; at 2.5 "movie" becomes the largest weight (0.436).

**Try it yourself.** Drag the scores and the sharpness. Watch the weights always sum to 1, see that equal scores give the plain average, and see how a large sharpness turns soft attention into nearly hard selection.

<AttentionWeightsPlayground />

:::danger[Interview Trap: "Softmax decides what is relevant"]
Softmax has **no learnable parameters**; it only normalizes. Relevance is decided by whatever produces the **scores**. In this lecture the scores were picked by hand; in a Transformer they come from Query–Key dot products whose projection matrices are learned. "The model attends to *not*" really means "the learned scoring function gave *not* a high score".
:::

:::tip[🎯 Interview angle]
**"Why use softmax in attention instead of hard selection or plain normalization?"** Hard selection (argmax) is not differentiable, so it can't be trained with gradient descent, and it keeps only one token. Softmax is smooth, makes every weight positive even for negative scores, sums to 1 so the result is a weighted *average* on a stable scale, and amplifies score differences exponentially so the most relevant tokens dominate. Dividing raw scores by their sum would fail on negative scores.
:::

## 6. The Weighted Context Vector: A Learned Weighted Average

Weights alone don't produce anything; they say how much of each token to use. Each token also carries a vector of information, its **value vector** $v_i$. The **context vector** is the weighted sum of those vectors, $c = \sum_i \alpha_i v_i$: important tokens contribute a lot, unimportant ones a little, and the result has the same size as one value vector.

The notebook uses 2-dimensional toy vectors so the arithmetic is visible. Read dimension 1 loosely as a "positive meaning" signal and dimension 2 as a "negation" signal (real embeddings are not this interpretable; this is for intuition). "good" is [1.0, 0.1], "not" is [0.0, 1.0], and the filler words have small values. Each vector is scaled by its weight, and the scaled vectors are added up. "good" contributes 0.351 to the positive dimension, "not" contributes 0.428 to the negation dimension, and the other three together add only 0.031 and 0.009. The context, **[0.3813, 0.4719]**, carries **both** halves of "not good": a positive word and its negation. A plain average treats every word as equally important and gives [0.28, 0.24], diluting the signal with "The", "movie" and "was".

| Token | Weight $\alpha_i$ | Value $v_i$ | $\alpha_i v_i$ |
|---|---|---|---|
| The | 0.0708 | [0.1, 0.0] | [0.0071, 0.0000] |
| movie | 0.0864 | [0.2, 0.1] | [0.0173, 0.0086] |
| was | 0.0640 | [0.1, 0.0] | [0.0064, 0.0000] |
| not | 0.4282 | [0.0, 1.0] | [0.0000, 0.4282] |
| good | 0.3506 | [1.0, 0.1] | [0.3506, 0.0351] |
| **Context** $c = \sum$ | | | **[0.3813, 0.4719]** |
| Plain average | 0.2 each | | [0.28, 0.24] |

<Formula name="Weighted context vector" tag="Attention output" words="Scale every token's value vector by its attention weight and add them up; the result is a summary dominated by the tokens that matter for the current prediction." remember="Weighted average of values.">

$$
\pred{c} = \sum_{i=1}^{n} \pred{\alpha_i}\,\pred{v_i} = \pred{\boldsymbol{\alpha}}^\top \pred{V}
$$

- $\pred{v_i}$: value vector of token $i$; $\pred{V}$: all value vectors stacked as rows ($n \times d_v$)
- equal weights $1/n$ turn it into the plain average
- the context has the same dimension as one value vector

**Example:** weights [0.0708, 0.0864, 0.0640, 0.4282, 0.3506] on the toy values give $\pred{c}$ = [0.3813, 0.4719], versus the average [0.28, 0.24].

</Formula>

This "summary" is not fixed. For sentiment, the scores favour "not" and "good". For topic classification, the same sentence might put most weight on "movie". In translation, every output word gets its own weights. Attention therefore builds a **customized summary per prediction**: a weighted average, but a **learned** one.

:::danger[Interview Trap: "Attention is just averaging"]
It is a weighted average, but the weights are computed from the input itself, for each query, by learned functions. That makes it **content-based routing**: the same layer mixes different tokens for different sentences and different positions. A fixed average or a convolution kernel uses the same mixing weights for every input.
:::

:::tip[🎯 Interview angle]
**"What is a context vector, and how is it different from averaging the embeddings?"** It is $c = \sum_i \alpha_i v_i$: the value vectors weighted by attention. With equal weights it *is* the average; with learned, input-dependent weights it emphasises what matters for the current prediction. On "The movie was not good" the attention context is [0.381, 0.472], shaped by "not" and "good", while the average [0.28, 0.24] is diluted by filler words.
:::

## 7. Query, Key and Value: Where Scores and Content Come From

So far the scores were given. In a real model they have to be **computed**, and the computation must answer three questions: what is the current token looking for, which tokens match that, and what information should be taken from them? The **Query–Key–Value** framework gives each question its own vector.

Think of searching a library. The **query** is what you are looking for ("books on deep learning"). Each book has **key** metadata on its index card: title, subject, tags. You compare your query against the keys to decide which books are relevant. Then you read the **value**: the actual content of the chosen books. Comparing query to keys gives the scores; the scores become weights; the weights mix the values.

In NLP every token plays all three roles. From each token's representation $x_i$ the model computes a **query** $q_i = x_i W_Q$ (what this token wants), a **key** $k_i = x_i W_K$ (what this token offers for matching) and a **value** $v_i = x_i W_V$ (the information it carries). Take "The animal didn't cross the road because it was tired" and the token **it**. Its query might encode "a singular noun that can be tired". The keys of "animal" and "road" both describe singular nouns, but "animal" matches "can be tired" better, so its score is higher, and the **value** of "animal" flows strongly into the new representation of "it".

Why three separate vectors instead of one? Because the roles conflict. What a word *searches for* (a verb looking for its subject) is not what it *advertises* (being a verb), and neither is the *content* it should pass on. With one vector for everything, "match" would just mean "similar", and every token would mostly attend to itself and its synonyms. Separate learned projections let search behaviour, matching identity and content transfer specialize independently.

![Query, key and value](/img/deep-learning/diagrams/qkv-retrieval.svg)

*How to read it:* the token "it" is projected by $W_Q$ into a query $q$. Every candidate word is projected by $W_K$ into a key and by $W_V$ into a value. The query is compared with each key ($q \cdot k$), softmax turns the three scores into weights, each value is multiplied by its weight, and $\Sigma$ adds them into the output for "it".

The pipeline for one token, in order:

| # | Step | Result |
|---|---|---|
| 1 | Build the query from the current token | $q = x W_Q$ |
| 2 | Compare it with every key | scores $q \cdot k_j$ |
| 3 | Softmax the scores | weights $\alpha_j$ |
| 4 | Weight the values | $\alpha_j v_j$ |
| 5 | Add them up | context output $\sum_j \alpha_j v_j$ |

The notebook only previews the formula (its cell shows a broken placeholder where the equation should be); the full derivation, including why the scores are divided by $\sqrt{d_k}$, is the next page's topic:

<Formula name="Scaled dot-product attention (preview)" tag="Next lecture" words="Compare every query with every key by dot product, scale by the square root of the key size, softmax each row into weights, and use the weights to mix the value vectors." remember="Match queries to keys, mix values.">

$$
\begin{aligned}
\operatorname{Attention}(\pred{Q}, \pred{K}, \pred{V}) &= \operatorname{softmax}\!\left(\frac{\pred{Q}\pred{K}^\top}{\sqrt{\hyper{d_k}}}\right)\pred{V} \\
\pred{Q} = \data{X}\param{W_Q},\quad \pred{K} &= \data{X}\param{W_K},\quad \pred{V} = \data{X}\param{W_V}
\end{aligned}
$$

- $\data{X}$: token representations ($n \times d$); $\param{W_Q}, \param{W_K}, \param{W_V}$: learned projections
- $\pred{Q}\pred{K}^\top$: the $n \times n$ score matrix; softmax is applied to each row
- $\hyper{d_k}$: the size of the query/key vectors

**Example:** with 5 tokens, $\pred{Q}\pred{K}^\top$ is 5 × 5, each row of weights sums to 1, and the output has one new vector per token.

</Formula>

:::analogy[Dating-app profiles]
Your **query** is what you're looking for, everyone's **key** is their public profile, and their **value** is who they actually are once you meet. Matching compares your wish list with profiles, but what you end up spending time with is the person, not the profile. Keeping the profile and the person separate is exactly why keys and values are different vectors.
:::

:::tip[🎯 Interview angle]
**"Explain queries, keys and values."** Each token is projected three ways by learned matrices. The query says what this token is looking for, the keys say what each token offers for matching, and the values are the content to pass on. Query–key dot products give relevance scores, softmax turns them into weights, and the weights mix the values. Separate projections let a token search for one thing, be found for another, and contribute a third.
:::

## 8. Attention Through Examples, and the Self-Attention Matrix

Attention becomes concrete once you ask, sentence by sentence, **which words should look at which**. In "Rahul gave Aman his laptop because he trusted him", the token *he* should attend to the candidates *Rahul* and *Aman*, and *his* to possible owners. In "The movie was not interesting", the representation of *interesting* must take *not* into account or the sentiment flips. In "The bouquet of roses on the table near the window smells fresh", the verb *smells* agrees with *bouquet*, not with the nearer nouns *roses*, *table* or *window*, and attention can connect them directly across the distance. In translation, generating *eau* should look at *water*. In question answering ("Who developed laws of motion?" over "Newton developed laws of motion."), the answer comes from attending to *Newton*, *developed* and *laws of motion*. Language meaning lives in **relationships between words**, and attention learns those relationships for each input.

In **self-attention** a sentence attends to **itself**: every token is a query, and every token (including itself) is a key and value. The weights then form an $n \times n$ **attention matrix**. **Row $i$** answers "when token $i$ is updated, how much does it take from each token?", so every row is one softmax distribution and **sums to 1**. **Column $j$** shows how much attention token $j$ receives from all the others; columns are not normalized (here they sum to 0.9, 1.05, 0.9, 1.15 and 1.0). The new representation of token $i$ is row $i$ times the value matrix, so the whole layer is one matrix product: $\text{new} = A V$.

The notebook's toy matrix was designed by hand so that *good* and *not* attend strongly to each other and the other words mostly to themselves:

| Query ↓ / Key → | The | movie | was | not | good |
|---|---|---|---|---|---|
| **The** | 0.60 | 0.20 | 0.10 | 0.05 | 0.05 |
| **movie** | 0.10 | 0.55 | 0.15 | 0.10 | 0.10 |
| **was** | 0.10 | 0.20 | 0.45 | 0.15 | 0.10 |
| **not** | 0.05 | 0.05 | 0.10 | 0.30 | **0.50** |
| **good** | 0.05 | 0.05 | 0.10 | **0.55** | 0.25 |

With the same value vectors as before, *good* starts as [1.0, 0.1], pure "positive". After attention it becomes **[0.275, 0.580]**: it has absorbed most of *not*'s negation signal and is no longer simply positive. *not* goes from [0.0, 1.0] to **[0.525, 0.355]**, now aware of what it negates. Equal attention (0.2 everywhere) would give every token the same vector, the average [0.28, 0.24], and all distinctions would vanish. That is the central power of self-attention: **each token's meaning is updated by looking at the tokens relevant to it**.

<Animation src="attention/heatmap-row" caption="The toy self-attention matrix next to the value vectors. Watch the good row get multiplied with the values to give a new good of 0.275 and 0.580, then the not row give 0.525 and 0.355: each row is one weighted average." />

<Formula name="Self-attention as a matrix product" tag="Every token at once" words="Stack every token's attention weights as the rows of a matrix; multiplying it by the value matrix gives every token's new, context-aware vector in one step." remember="Rows ask, columns answer.">

$$
\begin{aligned}
\pred{H'} &= \pred{A}\,\pred{V}, \qquad \pred{A} \in \mathbb{R}^{n \times n}, \quad \textstyle\sum_j \pred{A_{ij}} = 1 \\
\pred{h'_i} &= \sum_j \pred{A_{ij}}\,\pred{v_j}
\end{aligned}
$$

- $\pred{A_{ij}}$: how much token $i$ attends to token $j$ (row = query token, column = attended token)
- $\pred{V}$: value vectors as rows ($n \times d_v$); $\pred{H'}$: new token vectors

**Example:** the row for good is [0.05, 0.05, 0.10, 0.55, 0.25], so $\pred{h'_{\text{good}}}$ = 0.05·[0.1, 0] + 0.05·[0.2, 0.1] + 0.10·[0.1, 0] + 0.55·[0, 1] + 0.25·[1, 0.1] = [0.275, 0.580].

</Formula>

**Try it yourself.** Click a row to see which tokens that word attends to and its new vector. Then switch to equal attention and watch every token collapse onto the same average.

<SelfAttentionMatrix />

:::danger[Interview Trap: "The heatmap shows what the model learned"]
This matrix was **written by hand** for teaching. In a real model it is **computed** for every input from $\operatorname{softmax}(QK^\top/\sqrt{d_k})$, so it changes with every sentence, and each layer and each head has its own. Even for real models, attention weights are a hint about information flow, not a faithful explanation of a prediction (see Going Deeper).
:::

:::tip[🎯 Interview angle]
**"How do you read a self-attention matrix?"** Each row is one token acting as a query: a softmax distribution over all tokens, summing to 1. Each column shows how much a token is attended to; columns don't need to sum to 1. The new vector for token $i$ is row $i$ times the value matrix, so the whole layer is $AV$. In "not good", the *good* row puts 0.55 on *not*, which turns *good* from [1.0, 0.1] into [0.275, 0.58]: context-aware rather than plainly positive.
:::

## 9. "Attention Is All You Need" (2017)

In 2017 Vaswani and colleagues at Google published **"Attention Is All You Need"** and introduced the **Transformer**. Before it, strong sequence models were built on RNNs, LSTMs, GRUs or convolutions, often with attention **added** so the decoder could look back at the encoder states. Attention existed, but it was a helper; **recurrence was still the backbone**. The paper's bold claim was that the backbone itself could be attention: a sequence model needs **neither recurrence nor convolution**. Attention should not just support the model; it can **be** the model.

The architectural shift is from **sequential, memory-based** processing, $h_t = f(x_t, h_{t-1})$, to **matrix-based interaction**, token $i$ ↔ token $j$ for all pairs in parallel. For $n$ tokens, self-attention builds an $n \times n$ interaction structure in one step, so "smells" can attend to "bouquet" directly instead of waiting for the information to crawl through every word between them. For years, "sequence" had meant "recurrence"; the paper showed that step-by-step processing was an assumption, not a requirement.

The title is easy to misread. It does **not** mean a single softmax formula solves everything, or that position information, feed-forward layers, residual connections or normalization are unnecessary. The Transformer uses all of them: embeddings, positional encoding, **multi-head** attention, feed-forward networks, residual connections and layer normalization. The precise reading is: **recurrence is not necessary as the central engine of sequence modeling**.

Why it scaled so well, in five points: tokens interact directly; the computation is mostly matrix multiplication; training parallelizes across sequence positions; long-range dependencies have short paths; and residuals plus normalization let the network be stacked deep. On the paper's main benchmark, WMT 2014 English-to-German translation, the large Transformer reached 28.4 BLEU, a new state of the art, after 3.5 days of training on 8 GPUs, a fraction of the cost of the previous best models. The real impact was bigger than translation: the same block became the base of **BERT** (encoder), **GPT** (decoder), today's large language models, vision Transformers and multimodal systems.

In one sentence: **LSTM improved memory; the Transformer redesigned the computation**, replacing sequential recurrence with parallel, attention-based token interaction.

:::analogy[Relay race vs conference call]
An RNN is a relay race: the message passes hand to hand, runner by runner, and the last runner can only know what survived every handover. A Transformer layer is a conference call: everyone hears everyone at the same time and decides whom to listen to. A call costs more connections as the group grows ($n^2$), but nobody has to wait for the baton.
:::

:::tip[🎯 Interview angle]
**"What was new in 'Attention Is All You Need'? Attention already existed."** Attention existed as an add-on to RNN encoder-decoders. The paper removed recurrence and convolution entirely and built the model from self-attention plus feed-forward layers, residuals, layer norm and positional encodings. That made training parallel across positions and gave every token an $O(1)$ path to every other, which is why it scaled on GPUs/TPUs and became the base of BERT, GPT and modern LLMs.
:::

## 10. The Transformer Big Picture

A Transformer processes a sentence less like a chain, $x_1 \to x_2 \to \cdots$, and more like a **set of interacting tokens**, $x_i \leftrightarrow x_j$. At a high level it has seven parts: input embeddings, positional information, self-attention layers, feed-forward layers, residual connections, layer normalization and an output layer. This lecture covers what each one is for; the mathematics comes in the next notebooks.

**Embeddings.** Text can't be multiplied, so every token becomes a vector $x \in \mathbb{R}^d$ and a sentence of $n$ tokens becomes a matrix $X \in \mathbb{R}^{n \times d}$. All attention operations are matrix products on $X$.

**Positional information.** Here is a surprise: plain self-attention **does not know word order**. Every token compares itself with every other token by content, so if you shuffle the words, each token gets the same new vector as before, just in a different place (in implementation block 5, the outputs for shuffled input match the shuffled outputs to within $4.4 \times 10^{-16}$). "dog bites man" and "man bites dog" would be indistinguishable. So the Transformer **adds position information** to the embeddings, either fixed sinusoidal positional encodings or learned position embeddings, before the first layer.

**Self-attention** is the communication step: each token updates itself by looking at the others. After it, *good* in "The movie was not good" no longer means "good" in isolation; it means "good, in the context of *not*".

**The feed-forward network** is the thinking step: the same small two-layer network is applied to **each token separately**, transforming what the token has gathered. The notebook's memorable distinction: **attention = communication across tokens; feed-forward = transformation within each token**. A Transformer layer alternates the two.

**Residual connections and layer normalization** make deep stacks trainable. A residual adds a sublayer's input back to its output ($x + \text{Sublayer}(x)$), so information and gradients have a direct path through many layers, the same idea as in ResNet. Layer normalization keeps each token vector's activations on a stable scale.

![Transformer block, big picture](/img/deep-learning/diagrams/transformer-big-picture.svg)

*How to read it:* tokens become embeddings, positional information is added ($+$), and the result enters the green block, which is repeated $N$ times: self-attention (tokens talk to each other), add the residual and normalize, then the feed-forward network (each token refined on its own), add the residual and normalize. What comes out is one **contextual vector per token**.

After several layers every token's vector reflects the relevant parts of the whole sentence. That is why Transformer representations are **contextual**: "bank" in "river bank" attends strongly to *river* and ends up near water-and-land meanings, while "bank" in "bank account" attends to *account* and ends up near finance. Static embeddings such as Word2Vec give "bank" one vector for both. Transformers come in three shapes, encoder-only (BERT), decoder-only (GPT) and encoder-decoder (the original translation model), which the following notebooks cover.

:::analogy[A study group]
In each round, students first discuss (attention: everyone listens to whoever is relevant to them), then each goes off and works through their own notes (feed-forward). They keep their earlier notes and add to them rather than starting over (residual connections). After a few rounds, every student's notes reflect the whole group's discussion, which is what a contextual embedding is.
:::

:::tip[🎯 Interview angle]
**"Walk me through a Transformer layer at a high level, and why does it need positional encodings?"** Token embeddings plus positional information go into $N$ identical layers. Each layer does self-attention (tokens exchange information), add-and-norm, a position-wise feed-forward network (each token transformed independently), add-and-norm. Positional information is required because self-attention is permutation-equivariant: without it, shuffling the words would just shuffle the outputs, so "dog bites man" and "man bites dog" would look the same.
:::

## ⚙️ Going Deeper (Attention Internals)

**Additive vs multiplicative scores.** Bahdanau attention scores with a small network, $e = v^\top \tanh(W s + U h)$ (**additive**). Luong attention uses dot products, $e = s^\top W h$ or $s^\top h$ (**multiplicative**). The Transformer uses the scaled dot product $q \cdot k / \sqrt{d_k}$: it is one matrix multiply for all pairs, so it is much faster on GPUs. The $\sqrt{d_k}$ keeps dot products of large vectors from becoming so big that softmax saturates into near one-hot weights with tiny gradients (next page).

**Numerically stable softmax.** The notebook computes `np.exp(scores) / np.sum(np.exp(scores))`. That is fine for scores near 1, but `exp(1000)` overflows to infinity and the result becomes `nan`. Because softmax is unchanged by adding a constant, implementations subtract the maximum score first, so the largest exponent is $e^0 = 1$. `torch.softmax` and `F.softmax` do this internally. Block 1 shows both.

**Permutation equivariance.** Self-attention without positional information treats its input as a **set**: permuting the input rows permutes the output rows in the same way and changes nothing else. That is both a strength (no built-in bias about distance, so long-range links are as easy as short ones) and the reason position must be injected explicitly. Variants differ in how: absolute sinusoidal or learned embeddings added to $X$ (original Transformer, BERT), relative position biases, or rotary embeddings (RoPE) applied to $q$ and $k$ in most modern LLMs.

**Quadratic cost.** The score matrix is $n \times n$ for every head in every layer, so compute and memory grow with $n^2$. Doubling the context quadruples the attention cost. This is why context length is expensive, and why there are memory-efficient kernels (FlashAttention, which computes exact attention in tiles without storing the full matrix), sparse and sliding-window attention, and linear-attention approximations.

**Self-attention vs cross-attention.** In self-attention $Q$, $K$ and $V$ come from the same sequence. In **cross-attention** (encoder-decoder models) the queries come from the decoder and the keys and values from the encoder, which is exactly the Bahdanau set-up of chapter 3 rewritten in Q/K/V form.

**Masking.** A decoder that generates text must not look at future tokens during training, so the scores for positions $j > i$ are set to $-\infty$ before softmax (their weights become exactly 0). Padding tokens are masked the same way. This is the **causal mask** of GPT-style models.

**Attention is not explanation.** Heatmaps are tempting to read as "why the model decided", but a weight only says how much of a value vector was mixed in at one layer and one head; values are themselves mixtures from earlier layers, and the residual path carries information past attention entirely. Studies have found that very different attention patterns can give the same prediction. Use attention maps as a debugging hint, not as proof.

## 💻 Python Implementation

:::tip[Run it yourself]
Verified on Python 3.13 with NumPy 2.5. Outputs are copied from real runs into the trailing comments. Each block is self-contained.
:::

**What was corrected from the notebook.** The softmax subtracts the maximum score, so it can't overflow (block 1). Cell 22 used 0.5 for "movie" while every other cell and table used 0.4; this page uses 0.4 throughout and shows both results. The notebook's markdown cells contain broken placeholders where the softmax, context-vector and attention formulas should be; the formulas are restored in the chapters above.

### 1. From raw scores to attention weights

Softmax with the max-subtraction trick, and the properties discussed in chapter 5.

```python title="attention_weights.py" showLineNumbers
import numpy as np

tokens = ["The", "movie", "was", "not", "good"]
raw_scores = np.array([0.2, 0.4, 0.1, 2.0, 1.8])        # the notebook's table (cell 22 used 0.5 for "movie")

def softmax(scores):
    # highlight-next-line
    exp_scores = np.exp(scores - scores.max())           # subtract the max: same result, no overflow
    return exp_scores / exp_scores.sum()

attention_weights = softmax(raw_scores)
for token, score, weight in zip(tokens, raw_scores, attention_weights):
    print(f"{token:>6} | raw {score:.1f} | weight {weight:.4f}")
#    The | raw 0.2 | weight 0.0708
#  movie | raw 0.4 | weight 0.0864
#    was | raw 0.1 | weight 0.0640
#    not | raw 2.0 | weight 0.4282
#   good | raw 1.8 | weight 0.3506
print(attention_weights.sum())                           # 1.0
print(np.round(softmax(np.array([0.2, 0.5, 0.1, 2.0, 1.8])), 4))   # [0.0701 0.0947 0.0635 0.4243 0.3474]
print(np.round(softmax(raw_scores + 100), 4))           # [0.0708 0.0864 0.064  0.4282 0.3506]  shift: no change
print(np.round(softmax(raw_scores * 5), 4))             # [1.000e-04 2.000e-04 1.000e-04 7.308e-01 2.688e-01]  sharper
```

```python title="❌ The notebook's softmax overflows on large scores" showLineNumbers
import numpy as np
np.seterr(all="ignore")
# error-next-line
naive = np.exp(np.array([1000.0, 999.0]))               # ❌ exp(1000) = inf
print(naive / naive.sum())                               # [nan nan]
```

### 2. The weighted context vector

Scale each value vector by its weight and add; compare with the plain average.

```python title="context_vector.py" showLineNumbers
import numpy as np

tokens = ["The", "movie", "was", "not", "good"]
weights = np.exp([0.2, 0.4, 0.1, 2.0, 1.8]); weights /= weights.sum()
value_vectors = np.array([
    [0.1, 0.0],   # The
    [0.2, 0.1],   # movie
    [0.1, 0.0],   # was
    [0.0, 1.0],   # not
    [1.0, 0.1],   # good
])
weighted = weights[:, None] * value_vectors             # each row scaled by its weight
print(np.round(weighted, 4))
# [[0.0071 0.    ]
#  [0.0173 0.0086]
#  [0.0064 0.    ]
#  [0.     0.4282]
#  [0.3506 0.0351]]
# highlight-next-line
context = weighted.sum(axis=0)
print(np.round(context, 4))                              # [0.3813 0.4719]
print(np.round(value_vectors.mean(axis=0), 4))          # [0.28 0.24]       plain average
print(np.round(weights @ value_vectors, 4))             # [0.3813 0.4719]   same thing as one matrix product
```

### 3. The self-attention matrix

Rows sum to 1, columns don't; every token's new vector in one product.

```python title="attention_matrix.py" showLineNumbers
import numpy as np

tokens = ["The", "movie", "was", "not", "good"]
A = np.array([
    [0.60, 0.20, 0.10, 0.05, 0.05],
    [0.10, 0.55, 0.15, 0.10, 0.10],
    [0.10, 0.20, 0.45, 0.15, 0.10],
    [0.05, 0.05, 0.10, 0.30, 0.50],
    [0.05, 0.05, 0.10, 0.55, 0.25],
])
V = np.array([[0.1, 0.0], [0.2, 0.1], [0.1, 0.0], [0.0, 1.0], [1.0, 0.1]])
print(A.sum(axis=1), A.sum(axis=0))      # [1. 1. 1. 1. 1.] [0.9  1.05 0.9  1.15 1.  ]
# highlight-next-line
new = A @ V                               # every token's new, context-aware vector
for t, v, n in zip(tokens, V, new):
    print(f"{t:>6} {v} -> {np.round(n, 4)}")
#    The [0.1 0. ] -> [0.16  0.075]
#  movie [0.2 0.1] -> [0.235 0.165]
#    was [0.1 0. ] -> [0.195 0.18 ]
#    not [0. 1.] -> [0.525 0.355]
#   good [1.  0.1] -> [0.275 0.58 ]
print(np.round(np.full(5, 0.2) @ V, 4))  # [0.28 0.24]   equal attention: every token gets the average
```

### 4. Fixed context vs a dynamic context per decoder step

The seq2seq bottleneck and its fix, with random states (seeded) standing in for trained encoder and decoder states.

```python title="seq2seq_context.py" showLineNumbers
import numpy as np
rng = np.random.default_rng(1)
H = rng.normal(size=(6, 3))              # 6 encoder states
S = rng.normal(size=(3, 3))              # 3 decoder states
# error-next-line
c_fixed = H[-1]                          # ❌ classic seq2seq: one vector for every output step
scores = S @ H.T                         # (3, 6): each decoder step scores every encoder state
alpha = np.exp(scores - scores.max(axis=1, keepdims=True)); alpha /= alpha.sum(axis=1, keepdims=True)
# success-next-line
contexts = alpha @ H                     # ✅ (3, 3): a different context per output step
print(np.round(alpha, 3))
# [[0.084 0.298 0.178 0.107 0.25  0.084]
#  [0.211 0.416 0.204 0.095 0.035 0.038]
#  [0.004 0.307 0.072 0.021 0.585 0.01 ]]
print(np.round(contexts, 3))
# [[-0.557  0.407  0.139]
#  [-0.554  0.667  0.354]
#  [-0.856  0.229 -0.109]]
print(np.round(c_fixed, 3))              # [ 0.599  0.04  -0.292]
```

Here the "score" is a plain dot product between decoder and encoder states (Luong-style); Bahdanau used a small additive network instead.

### 5. Why the Transformer needs positions

Plain self-attention ignores order: shuffling the tokens only shuffles the outputs.

```python title="permutation_equivariance.py" showLineNumbers
import numpy as np
rng = np.random.default_rng(0)
X = rng.normal(size=(5, 4))              # 5 token vectors, no position information

def self_attention(X):
    scores = X @ X.T / np.sqrt(X.shape[1])               # every token vs every token
    w = np.exp(scores - scores.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    return w @ X

perm = np.array([4, 2, 0, 3, 1])         # shuffle the word order
out = self_attention(X)
out_perm = self_attention(X[perm])
# highlight-next-line
print(np.abs(out_perm - out[perm]).max())  # 4.440892098500626e-16   same outputs, just shuffled
print(np.abs(out_perm - out).max() > 0.1)  # True   (rows are in a different order)
```

This toy uses $X$ directly as queries, keys and values; real layers apply $W_Q$, $W_K$, $W_V$ first, which doesn't change the conclusion.

## 🚨 Interview Gotchas & Edge Cases

:::warning[The Most Common Question]
**"Why did Transformers replace RNNs and LSTMs?"**

1. **Sequential computation:** $h_t$ needs $h_{t-1}$, so training can't parallelize over time; self-attention computes all positions at once ($O(1)$ sequential steps vs $O(n)$).
2. **Long paths:** in an RNN information between distant tokens crosses $O(n)$ steps; in self-attention every pair is one step apart.
3. **Fixed-size bottleneck:** an RNN squeezes the past into one state (and seq2seq squeezed the whole source into $c = h_T$); attention reads all token vectors directly with input-dependent weights.
4. **Hardware fit:** attention is matrix multiplication, which GPUs and TPUs are built for, so Transformers scale to huge data and models.
5. **Costs:** $O(n^2)$ compute and memory in context length, and no built-in order (positional encodings). RNNs still make sense for streaming and on-device use with constant memory per step.
:::

- **Softmax normalizes; it doesn't decide.** Relevance comes from the learned scores (query–key similarity).
- **Rows sum to 1, columns don't.** A row is one token's distribution over the others; a column is how much a token is attended to.
- **Attention existed before 2017** (Bahdanau 2014, Luong 2015); the Transformer removed recurrence.
- **"All you need" is not literal:** the Transformer also needs positional information, feed-forward layers, residuals, layer norm and multi-head attention.
- **Self-attention is order-blind** without positional information: permuting the input permutes the output.
- **Naive softmax overflows:** subtract the max score; frameworks do this for you.
- **Hand-made heatmaps teach, they don't describe a model.** Real attention weights are computed per input, per layer, per head.
- **Attention weights are not explanations.** Values are mixtures, and residual paths bypass attention.
- **Quadratic memory:** a long context costs $n^2$ per head per layer; doubling $n$ quadruples it.
- **Parallel in training, sequential in generation:** a Transformer decoder still produces one token at a time (with a KV cache).

## ⚡ Essential Interview Recall

A one-screen revision list. Every point is explained in the chapters above.

**Why move beyond LSTM**
- TF-IDF (no order) → RNN (order, fragile memory) → LSTM/GRU (gated memory) → attention → Transformer.
- Recurrence: sequential in time, $O(n)$ path between distant tokens, one fixed-size state.
- Per layer: self-attention $O(n^2 d)$ compute, $O(1)$ sequential, $O(1)$ path; recurrent $O(n d^2)$, $O(n)$, $O(n)$.

**Seq2seq**
- Fixed context $c = h_T$ for every output step: information squeezed, same summary for every word.
- Bahdanau (2014): $c_t = \sum_i \alpha_{t,i} h_i$, a new context per output step.

**Soft alignment and context**
- $\alpha_i = \operatorname{softmax}(e_i)$: positive, sum to 1, differentiable; shift-invariant; scaling sharpens.
- "The movie was not good": scores [0.2, 0.4, 0.1, 2.0, 1.8] → weights [0.071, 0.086, 0.064, 0.428, 0.351].
- Context $c = \sum_i \alpha_i v_i$ = [0.381, 0.472] vs average [0.28, 0.24].

**Query, Key, Value**
- Query = what a token looks for, Key = what it offers for matching, Value = what it passes on; $q = xW_Q$, $k = xW_K$, $v = xW_V$.
- Preview: $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$.

**Self-attention matrix**
- $n \times n$; row = query token (sums to 1), column = attended token; new vectors $= AV$.
- Toy: good [1.0, 0.1] → [0.275, 0.58]; not [0.0, 1.0] → [0.525, 0.355]; equal attention → all [0.28, 0.24].

**Transformer**
- Vaswani et al. 2017: attention as the backbone, no recurrence or convolution; 28.4 BLEU on WMT14 EN-DE.
- Embeddings $X \in \mathbb{R}^{n \times d}$ + positional information → $N$ × (self-attention + add & norm → feed-forward + add & norm) → contextual vectors.
- Attention = communication across tokens; feed-forward = transformation within each token.
- Positions are needed because self-attention is permutation-equivariant; "bank" gets different vectors in "river bank" and "bank account".

## 📐 Formula Sheet

Every equation on this page in one place.

| Concept | Formula | In plain words |
|---|---|---|
| Recurrence | $\pred{h_t} = f(\data{x_t}, \pred{h_{t-1}})$ | Each state waits for the previous one |
| Seq2seq fixed context | $\pred{c} = \pred{h_T}$, $\pred{y_t} = g(\pred{y_{t-1}}, \pred{s_{t-1}}, \pred{c})$ | One summary for every output step |
| Attention scores (seq2seq) | $\pred{e_{t,i}} = \operatorname{score}(\pred{s_{t-1}}, \pred{h_i})$ | How relevant source $i$ is for output $t$ |
| Attention weights | $\pred{\alpha_i} = e^{\pred{e_i}} / \sum_j e^{\pred{e_j}}$ | Positive weights that sum to 1 |
| Dynamic context | $\pred{c_t} = \sum_i \pred{\alpha_{t,i}}\pred{h_i}$ | A new look-back per output step |
| Context vector | $\pred{c} = \sum_i \pred{\alpha_i}\pred{v_i} = \pred{\boldsymbol{\alpha}}^\top\pred{V}$ | Weighted average of values |
| Projections | $\pred{Q} = \data{X}\param{W_Q}$, $\pred{K} = \data{X}\param{W_K}$, $\pred{V} = \data{X}\param{W_V}$ | Three learned views of each token |
| Scaled dot-product attention | $\operatorname{softmax}(\pred{Q}\pred{K}^\top / \sqrt{\hyper{d_k}})\,\pred{V}$ | Match queries to keys, mix values |
| Self-attention matrix | $\pred{H'} = \pred{A}\pred{V}$, $\sum_j \pred{A_{ij}} = 1$ | Rows ask, columns answer |
| Additive score (Bahdanau) | $\pred{e} = \param{v}^\top\tanh(\param{W}\pred{s} + \param{U}\pred{h})$ | A small network scores the pair |
| Embedding matrix | $\data{X} \in \mathbb{R}^{n \times d}$ | One vector per token |
| Residual connection | $\pred{x} + \operatorname{Sublayer}(\pred{x})$ | Keep the input, add the update |
| Per-layer cost | self-attention $O(n^2 d)$; recurrent $O(n d^2)$ | Quadratic in length vs in width |

## Concept-Check Answers (from the notebook)

### Opening: the historical progression

<details>
<summary><strong>Q1. Why was TF-IDF limited for language understanding?</strong></summary>

It represents a document by word-importance counts, with no word order and no context, so "dog bites man" and "man bites dog" get the same vector, and a word has the same weight whatever surrounds it (negation, sarcasm and phrases are invisible).
</details>

<details>
<summary><strong>Q2. What problem did RNN solve?</strong></summary>

Order and memory: it reads tokens one at a time and carries a hidden state $h_t = f(x_t, h_{t-1})$, so the representation depends on the sequence, not just the bag of words.
</details>

<details>
<summary><strong>Q3. Why was LSTM considered superior to vanilla RNN?</strong></summary>

Its additive cell state and learned gates let information and gradients survive many more steps, so it learns longer-range dependencies than a vanilla RNN, whose gradients vanish or explode through repeated multiplication.
</details>

<details>
<summary><strong>Q4. If LSTM was strong, what new limitation might still remain?</strong></summary>

The chain itself: processing is still sequential (no parallelism over time), distant tokens are linked only through a long path of updates, and the past must fit into one fixed-size state.
</details>

### Why recurrence is still a bottleneck

<details>
<summary><strong>Q1. Why are RNN/LSTM difficult to parallelize?</strong></summary>

Because $h_t$ depends on $h_{t-1}$: step $t$ can't start until step $t - 1$ is done, so the time dimension is strictly serial. Only the batch and the matrix multiplies inside a step can run in parallel.
</details>

<details>
<summary><strong>Q2. Why does long-distance dependency remain hard even in LSTM?</strong></summary>

Information from token 1 must pass through every intermediate state to reach token 50. Gates make the path more forgiving, but every step still mixes in new input and risks diluting the old information, and the gradient still travels the whole chain back.
</details>

<details>
<summary><strong>Q3. What does hidden-state compression mean?</strong></summary>

At each step one fixed-size vector $h_t$ must summarize everything read so far (subject, tense, sentiment, names, negation, dependencies). Whatever doesn't fit is blurred or lost, and longer inputs make it worse.
</details>

<details>
<summary><strong>Q4. Which is better for hardware: sequential recurrence or matrix parallel computation?</strong></summary>

Matrix parallel computation. GPUs and TPUs are built to do huge matrix multiplications at once; one-token-at-a-time recurrence leaves most of that capacity idle.
</details>

### Fixed-length context limitation

<details>
<summary><strong>Q1. In old seq2seq models, what does the vector $h_T$ represent?</strong></summary>

The encoder's final hidden state, used as the context $c$: a single summary of the entire source sentence that the decoder conditions every output word on.
</details>

<details>
<summary><strong>Q2. Why does sentence length worsen this problem?</strong></summary>

More words means more facts to fit into the same fixed-size vector, and the earliest words have passed through the most updates, so more information is diluted or lost.
</details>

<details>
<summary><strong>Q3. Why may different output tokens need different parts of the input?</strong></summary>

Each output word corresponds to different source words: a verb needs the source verb, a noun its noun phrase, a pronoun its antecedent. A single summary gives every step the same information regardless of what it needs.
</details>

<details>
<summary><strong>Q4. What is better: one fixed summary or dynamic access to source tokens?</strong></summary>

Dynamic access: keep all encoder states and compute a new weighted context for each output step, so every word can draw on the source positions it needs. That is attention.
</details>

### Selective focus intuition

<details>
<summary><strong>Q1. Why is equal treatment of all words inefficient?</strong></summary>

Most words contribute little to a given prediction; weighting them equally dilutes the few that matter. In the toy example the plain average [0.28, 0.24] is much weaker than the attention context [0.381, 0.472].
</details>

<details>
<summary><strong>Q2. In sentiment analysis, why is "not" often critical?</strong></summary>

It reverses the polarity of what follows: "good" becomes negative in "not good". If the model doesn't attend to "not", it reads the sentence as positive.
</details>

<details>
<summary><strong>Q3. Attention is closer to storing everything in one vector, or dynamically retrieving relevant parts?</strong></summary>

Dynamically retrieving relevant parts: it keeps all token vectors and computes, for each prediction, how much to take from each.
</details>

<details>
<summary><strong>Q4. Give one NLP task where selective focus helps.</strong></summary>

Machine translation (each output word looks at its source word), sentiment analysis (focus on "not" and "good"), pronoun resolution ("she" looks at the candidate names) or question answering (focus on the answer span).
</details>

### Soft alignment mechanism

<details>
<summary><strong>Q1. Why is soft selection easier to train than hard selection?</strong></summary>

Soft weights change smoothly with the scores, so gradients flow to the scoring function. Hard selection is a step function (argmax): its gradient is zero almost everywhere and it jumps when the choice flips, so learning is unstable.
</details>

<details>
<summary><strong>Q2. Why must weights sum to 1?</strong></summary>

So the context is a weighted **average** on the same scale as the value vectors, whatever the sentence length, and the weights are comparable as a distribution of attention.
</details>

<details>
<summary><strong>Q3. If one token has a much larger score, what happens after softmax?</strong></summary>

Its weight approaches 1 and the others approach 0, because softmax is exponential: multiplying the toy scores by 5 gives "not" 0.731 and three tokens about 0.0001 each. Attention becomes nearly hard.
</details>

<details>
<summary><strong>Q4. Why can multiple words still matter in soft attention?</strong></summary>

Weights are shared out, not winner-take-all: several tokens with high scores all get large weights ("not" 0.428 and "good" 0.351), and their values are combined in the context.
</details>

### Query–Key–Value conceptual framework

<details>
<summary><strong>Q1. Query represents what?</strong></summary>

What the current token is looking for: its search request, $q = xW_Q$.
</details>

<details>
<summary><strong>Q2. Key is used for what?</strong></summary>

Matching: each token's key is compared with the query (dot product) to produce the relevance score; it describes what that token offers.
</details>

<details>
<summary><strong>Q3. Value contains what?</strong></summary>

The information a token passes on when attended to; the output is the weighted sum of values.
</details>

<details>
<summary><strong>Q4. Why not use one single vector for all three roles?</strong></summary>

The roles differ: what a token searches for, how it should be found and what it should contribute are different things. One vector would make matching mean plain similarity (tokens mostly attending to themselves). Separate learned projections let each role specialize.
</details>

<details>
<summary><strong>Q5. In pronoun resolution, which earlier token's key may match the pronoun query?</strong></summary>

The antecedent's. In "The animal didn't cross the road because it was tired", the query of "it" should match the key of "animal" (something that can be tired) more than the key of "road".
</details>

### Coding demo: from raw scores to attention weights

<details>
<summary><strong>Q1. Why do we need softmax after raw relevance scores?</strong></summary>

Raw scores can be negative, are on an arbitrary scale and don't sum to 1. Softmax makes them positive, comparable and normalized, so they can weight an average, and it stays differentiable.
</details>

<details>
<summary><strong>Q2. Why should attention weights sum to 1?</strong></summary>

So the output is a weighted average with a stable scale, independent of how many tokens there are, and the weights read as a distribution of focus.
</details>

<details>
<summary><strong>Q3. In this example, why did "not" receive high attention?</strong></summary>

Because it was given the highest raw score (2.0), chosen by hand to reflect that negation decides the sentiment. Softmax turned that into the largest weight, 0.428. In a real model the high score would come from learned query–key similarity.
</details>

<details>
<summary><strong>Q4. If the raw score of "movie" is increased, what will happen to its attention weight?</strong></summary>

It increases, and every other weight decreases, because the total is fixed at 1: at 1.0 "movie" gets 0.147 (from 0.086) and "not" drops to 0.400; at 2.5 "movie" becomes the largest weight, 0.436.
</details>

<details>
<summary><strong>Q5. Does softmax decide relevance by itself, or does it only normalize scores?</strong></summary>

It only normalizes; it has no parameters. Relevance comes from whatever produces the scores: hand-picked here, query–key similarity with learned projections in a Transformer.
</details>

### Coding demo: the weighted context vector

<details>
<summary><strong>Q1. What is a context vector?</strong></summary>

The attention output: the weighted sum of the value vectors, $c = \sum_i \alpha_i v_i$, a summary of the sequence customized for the current prediction ([0.3813, 0.4719] in the example).
</details>

<details>
<summary><strong>Q2. Why is the context vector not a simple average?</strong></summary>

Its weights come from relevance, so important tokens dominate. A simple average gives every token 1/n, which dilutes the signal ([0.28, 0.24] vs [0.381, 0.472]).
</details>

<details>
<summary><strong>Q3. What happens if a token has a very small attention weight?</strong></summary>

It still contributes, but very little: its value vector is scaled almost to zero ("was" with 0.064 adds [0.0064, 0]). It is softly suppressed, not removed.
</details>

<details>
<summary><strong>Q4. In the phrase "not good", why should both words contribute?</strong></summary>

The meaning is the combination: "good" brings the positive content and "not" reverses it. In the example the context carries both, 0.351 from "good" in the positive dimension and 0.428 from "not" in the negation dimension; either alone would mislead.
</details>

<details>
<summary><strong>Q5. In real Transformer attention, are the value vectors manually created?</strong></summary>

No. They are computed as $V = XW_V$, from learned embeddings and a learned projection matrix, trained end to end. The 2-dimensional vectors here were made up for teaching.
</details>

### Coding demo: tiny sentence attention heatmap

<details>
<summary><strong>Q1. In an attention matrix, what does each row represent?</strong></summary>

One query token: its attention distribution over all tokens, i.e. how much it takes from each when it is updated.
</details>

<details>
<summary><strong>Q2. What does each column represent?</strong></summary>

One token being attended to: how much attention it receives from each query token.
</details>

<details>
<summary><strong>Q3. Why should each row sum to 1?</strong></summary>

Each row is a softmax distribution for one token, so its new vector is a weighted average of the values. Columns have no such constraint (0.9 to 1.15 here).
</details>

<details>
<summary><strong>Q4. In the phrase "not good", why should "good" attend to "not"?</strong></summary>

Because "not" changes what "good" means. Attending to it (0.55 in the toy matrix) moves "good" from [1.0, 0.1] to [0.275, 0.58], a representation that already includes the negation.
</details>

<details>
<summary><strong>Q5. Is the heatmap shown here manually created or learned by the model?</strong></summary>

Manually created for intuition. In a real model the matrix is computed for each input as $\operatorname{softmax}(QK^\top/\sqrt{d_k})$, with learned projections, separately for every layer and head.
</details>

### "Attention Is All You Need"

<details>
<summary><strong>Q1. Did attention exist before the Transformer?</strong></summary>

Yes. It was introduced in 2014 for RNN encoder-decoders (Bahdanau et al.), where it helped the decoder look back at encoder states; recurrence remained the backbone.
</details>

<details>
<summary><strong>Q2. What was the bold claim of "Attention Is All You Need"?</strong></summary>

That a strong sequence model can be built with attention as its core, without recurrence or convolution: attention doesn't just support the model, it can be the model.
</details>

<details>
<summary><strong>Q3. Why was removing recurrence important?</strong></summary>

It removed the step-by-step dependency, so all positions are processed in parallel during training, and it shortened the path between any two tokens to one step, making long-range relations easier to learn.
</details>

<details>
<summary><strong>Q4. Why are Transformers more suitable for GPU/TPU training?</strong></summary>

Their work is mostly large matrix multiplications over all tokens at once ($QK^\top$, $AV$, feed-forward layers), which is exactly what accelerators are fast at; there is no serial loop over time.
</details>

<details>
<summary><strong>Q5. Does the Transformer use only one attention formula, or does it also need other components?</strong></summary>

It needs other components: embeddings, positional encoding, multi-head attention, position-wise feed-forward networks, residual connections and layer normalization (plus an output layer). "All you need" means no recurrence, not attention alone.
</details>

### Transformer big picture

<details>
<summary><strong>Q1. What is the central mechanism inside a Transformer?</strong></summary>

Self-attention: every token builds its new representation as a weighted combination of all tokens' values, with weights from query–key similarity.
</details>

<details>
<summary><strong>Q2. Why does a Transformer need positional information?</strong></summary>

Self-attention compares tokens by content only and is permutation-equivariant: shuffling the input just shuffles the output. Without added positions, "dog bites man" and "man bites dog" would look the same.
</details>

<details>
<summary><strong>Q3. What is the difference between attention and feed-forward layers?</strong></summary>

Attention is communication across tokens (each token gathers information from others). The feed-forward network is transformation within each token (the same network applied to every position independently).
</details>

<details>
<summary><strong>Q4. Why are residual connections useful?</strong></summary>

They add each sublayer's input back to its output, giving information and gradients a direct path through many layers, so deep stacks train stably; layer normalization keeps the activations on a stable scale.
</details>

<details>
<summary><strong>Q5. Why is the word "bank" a good example for contextual representation?</strong></summary>

It has different meanings in "river bank" and "bank account". A static embedding gives it one vector; in a Transformer "bank" attends to "river" or "account" and gets a different, context-specific representation.
</details>
