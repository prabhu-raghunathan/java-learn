---
title: "Full Self-Attention: Scaled Dot-Product Attention Step by Step"
sidebar_position: 2
---

import {AttentionPipelineExplorer, ScalingSharpnessPlayground} from '@site/src/components/Widgets';

:::tip[TL;DR (30-Second Interview Pitch)]
Self-attention lets every token of **one** sequence attend to every token of the **same** sequence. Stack the token embeddings as $X$ ($n \times d_{model}$), project them with three learned matrices into queries $Q = XW_Q$, keys $K = XW_K$ and values $V = XW_V$ ($n \times d_k$), compare every query with every key in one product $S = QK^\top$ ($n \times n$), **divide by $\sqrt{d_k}$** so the scores don't grow with the vector size, apply a **row-wise softmax** to get the attention matrix $A$ (every row sums to 1), and mix the values: $Z = AV$. Each row of $Z$ is a **context-aware** version of one token. The $\sqrt{d_k}$ exists because a dot product of $d_k$ unit-variance terms has standard deviation $\sqrt{d_k}$ (measured: 1.38 at $d_k = 2$, 11.02 at $d_k = 128$; after scaling, about 1 at every size). Without it, softmax saturates toward one-hot and its gradients vanish.
:::

<FormulaKey />

:::info[Where this page sits]
**Before:** [From LSTM to Attention: Why Transformers Became Necessary](./01-from-lstm-to-attention-why-transformers.md) built the intuition with hand-picked scores and a hand-drawn attention matrix.
**This page:** notebook *15 Full Self-Attention*: from encoder–decoder attention to self-attention, the input matrix $X$, the $Q$/$K$/$V$ projections and their shapes, the score matrix $QK^\top$ and the meaning of one score, the raw-score heatmap, why we scale by $\sqrt{d_k}$, the softmax sharpness problem, the full formula, the attention matrix $A$, the contextual output $Z = AV$, original vs updated embeddings, and the bridge to multi-head attention.
**Next:** notebook *16 Multi-Head Attention* runs several of these attention computations in parallel.
:::

## 🧭 Start Here

The previous page explained **why** attention exists, using numbers we chose by hand: scores of 2.0 for "not" and 1.8 for "good", value vectors with obvious meanings, an attention matrix drawn so "good" looks at "not". A real Transformer can't be told any of that. It has to **compute** the scores from the tokens themselves, for every pair of tokens, with parameters it can learn. This page does that computation completely, step by step, on the same five-token sentence "The movie was not good".

The whole mechanism is five matrix operations. Each token's embedding is turned into a **query** (what it's looking for), a **key** (how it can be found) and a **value** (what it passes on) by three learned matrices. Every query is compared with every key by a dot product, giving an $n \times n$ table of raw scores. The scores are divided by $\sqrt{d_k}$, turned into weights by a softmax over each row, and used to take a weighted average of the value vectors. The result is one new vector per token, built from the whole sentence. That is the formula at the heart of every Transformer:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

The notebook uses **random, untrained** numbers ($X$ and the three weight matrices come from `np.random.seed(42)`) with $n = 5$ tokens, $d_{model} = 4$ and $d_k = d_v = 3$. That makes every number on this page reproducible, and it teaches an important lesson: with untrained weights, the attention pattern has **no linguistic meaning**. In this run "good" gives "not" only 0.0003 of its attention. The mechanism, the shapes and the arithmetic are what to learn here; training is what makes the pattern meaningful. The chain of ideas:

1. Self-attention: $Q$, $K$ and $V$ all come from the same sequence.
2. The input matrix $X$: one row per token, still isolated.
3. Projections $Q = XW_Q$, $K = XW_K$, $V = XW_V$, and their shapes.
4. The score matrix $S = QK^\top$: every query against every key.
5. What one score $S_{ij}$ means, and how to read the raw-score heatmap.
6. Why raw scores grow with $d_k$, and why we divide by $\sqrt{d_k}$.
7. The softmax sharpness problem.
8. The full formula, step by step with shapes.
9. The attention matrix $A$: rows are distributions.
10. The output $Z = AV$: contextual token representations.
11. Original embedding vs updated representation, and the bridge to multi-head attention.

| New concept | One-line meaning |
|---|---|
| **Self-attention** | Attention where queries, keys and values come from the same sequence |
| **Cross-attention** | Queries from one sequence (decoder), keys and values from another (encoder) |
| **$X$** | The $n \times d_{model}$ matrix of token embeddings, one row per token |
| **$W_Q$, $W_K$, $W_V$** | Learned $d_{model} \times d_k$ projection matrices |
| **$Q$, $K$, $V$** | Query, key and value matrices, $n \times d_k$ (values $n \times d_v$) |
| **$d_k$** | Size of each query/key vector (here 3; 64 per head in the original Transformer) |
| **Score matrix $S = QK^\top$** | $n \times n$ raw compatibilities; $S_{ij} = q_i \cdot k_j$ |
| **Scaling $1/\sqrt{d_k}$** | Keeps score variance about 1 whatever the vector size |
| **Row-wise softmax** | Turns each row of scores into non-negative weights summing to 1 |
| **Attention matrix $A$** | $n \times n$ weights; row $i$ = how token $i$ spreads its attention |
| **Contextual output $Z = AV$** | One new vector per token: a weighted mix of all value vectors |
| **Softmax saturation** | Large score gaps make softmax nearly one-hot, with vanishing gradients |

:::analogy[A room full of people, each with three cards]
Everyone in a meeting holds three cards: a **question card** (what I need to hear about), a **name tag** (what topics I can speak to) and a **notes card** (what I'd actually say). Each person reads every name tag, decides how relevant each speaker is to their question, and then listens to everyone at once with the volume set by relevance. Each person leaves with their own blend of everyone's notes. That's one self-attention layer, run for all five words of the sentence at the same time.
:::

## 1. From Encoder–Decoder Attention to Self-Attention

The first attention mechanism (Bahdanau, 2014, covered on the previous page) connected **two** sequences. In translation, the decoder was writing the French sentence and the encoder had read the English one. The decoder's current state acted as the **query**, and the encoder's outputs served as both **keys** (to match against) and **values** (to read from). That is **encoder–decoder attention**, today usually called **cross-attention**: information flows across sequences, from source to target.

That setup is **asymmetric**. The decoder looks at the encoder, but the words of the English sentence never look at **each other** through attention; their interactions still had to pass through the encoder's recurrent chain. Self-attention removes the restriction: **queries, keys and values all come from the same sequence**. Every token attends to every token in its own sentence, itself included, so "good" can look directly at "not" and "movie", and "not" can look at "good".

This shift is why meaning becomes **contextual**. A word's representation is no longer a fixed lookup; it is rebuilt from the words around it. "good" next to "not" ends up represented differently from "good" in "a good movie", because in each sentence it mixes in different neighbours.

| Mechanism | Query source | Key/value source | Interaction |
|---|---|---|---|
| **Encoder–decoder (cross) attention** | Decoder | Encoder | Across two sequences |
| **Self-attention** | Same sequence | Same sequence | Within one sequence |

![Encoder-decoder (cross) attention vs self-attention](/img/deep-learning/diagrams/cross-vs-self-attention.svg)

*How to read it:* left, the decoder token "eau" sends one query to the three encoder outputs, which provide keys and values: two different sequences. Right, a single sequence: "good" scores every token including itself (the self-loop), and every other token does the same simultaneously. In self-attention $Q$, $K$ and $V$ are three projections of the same matrix $X$.

:::tip[🎯 Interview angle]
**"What is the difference between self-attention and cross-attention?"** In self-attention, queries, keys and values are all projections of the same sequence, so tokens attend to each other within it. In cross-attention, queries come from one sequence (the decoder) and keys and values from another (the encoder output). Encoders and GPT-style decoders use self-attention; encoder-decoder models like the original Transformer and T5 also have cross-attention in the decoder.
:::

## 2. The Input Matrix X: Tokens as Rows

Attention works on matrices, so the sentence must become one. Each token is first mapped to a vector by an embedding lookup, a row of $d_{model}$ numbers. Stacking those rows gives the input matrix $X$, with **one row per token** (row index = position in the sentence) and **one column per feature** (column index = embedding dimension). For "The movie was not good" with $d_{model} = 4$, $X$ is $5 \times 4$.

Packing the sentence into one matrix is what makes Transformers fast. Instead of looping over tokens, every later step is a single matrix multiplication over all tokens at once, which is exactly what GPUs are good at. It also means the batch dimension adds naturally: a batch of sentences is a 3-D tensor of shape (batch, $n$, $d_{model}$).

At this stage the rows are **independent**. Row 5 ("good") knows nothing about row 4 ("not"); each is just the embedding of its own word, identical in every sentence the word appears in. You *could* compare two rows directly, for example $x_{good} \cdot x_{not} = 1.514$, but that number is **static**: it is the same in "not good" and in "good, not bad", it isn't normalized, and nothing about it was learned for the task of combining words. Nothing in $X$ can tell "not good" from "good". Self-attention is the mechanism that updates each row using the others.

| Token | $x_i$ (row of $X$, from `np.random.seed(42)`) |
|---|---|
| The | [0.4967, −0.1383, 0.6477, 1.5230] |
| movie | [−0.2342, −0.2341, 1.5792, 0.7674] |
| was | [−0.4695, 0.5426, −0.4634, −0.4657] |
| not | [0.2420, −1.9133, −1.7249, −0.5623] |
| good | [−1.0128, 0.3142, −0.9080, −1.4123] |

<Formula name="Input embedding matrix" tag="Starting point" words="Stack one embedding row per token; rows are positions in the sentence, columns are features." remember="Rows are tokens, columns are features.">

$$
\data{X} = \begin{bmatrix} \data{x_1} \\ \vdots \\ \data{x_n} \end{bmatrix} \in \mathbb{R}^{n \times \hyper{d_{model}}}, \qquad \data{x_i} \in \mathbb{R}^{\hyper{d_{model}}}
$$

- $n$: number of tokens; $\hyper{d_{model}}$: embedding size (here 4; 512 in the original Transformer)
- $\data{x_i}$: the embedding of token $i$, before any interaction

**Example:** "The movie was not good" with $\hyper{d_{model}} = 4$ gives a 5 × 4 matrix; $\data{x_{good}} \cdot \data{x_{not}} = 1.514$, a static number that ignores context.

</Formula>

:::tip[🎯 Interview angle]
**"Why does a Transformer represent a sentence as a matrix instead of processing tokens one by one?"** Stacking the token embeddings into $X \in \mathbb{R}^{n \times d_{model}}$ turns every step of attention into a matrix multiplication over all tokens at once, which parallelizes on GPUs and extends naturally to batches. The rows start independent; attention is what lets each row be updated from the others.
:::

## 3. Query, Key and Value Projections

Self-attention doesn't use $X$ directly to compare tokens. It first creates **three different views** of every token with three learned weight matrices: $Q = XW_Q$, $K = XW_K$ and $V = XW_V$. Each token then has a **query** vector (what am I looking for?), a **key** vector (what do I contain, for matching?) and a **value** vector (what will I pass forward if someone attends to me?).

Why not compare raw embeddings, $XX^\top$? Because then "matching" could only mean "having similar embeddings", and every token would score highest with itself and its synonyms. A negation word should be found by the adjective it modifies, not by other negation words. With separate projections, the model can learn that the **query** of "good" asks for sentiment modifiers, the **key** of "not" advertises "I am a modifier", and the **value** of "not" carries the information "flip the polarity". Because $W_Q \ne W_K \ne W_V$, the same row of $X$ produces three different vectors.

The projections are ordinary matrix multiplications with learned matrices of shape $d_{model} \times d_k$. With $d_{model} = 4$ and $d_k = 3$, each $W$ is $4 \times 3$, and $(5 \times 4)(4 \times 3) = 5 \times 3$: every token gets a 3-number query, key and value. In the notebook the three $W$ matrices are random; in a trained model they are learned by gradient descent along with everything else. They are the **only parameters** of a single attention head: $3 \times 4 \times 3 = 36$ numbers here.

<Formula name="Query, key and value projections" tag="Three learned views" words="Multiply the same input matrix by three different learned matrices to get each token's query, key and value." remember="One X, three lenses.">

$$
\begin{aligned}
&\pred{Q} = \data{X}\param{W_Q}, \qquad \pred{K} = \data{X}\param{W_K}, \qquad \pred{V} = \data{X}\param{W_V} \\
&\param{W_Q}, \param{W_K}, \param{W_V} \in \mathbb{R}^{\hyper{d_{model}} \times \hyper{d_k}}
\end{aligned}
$$

- $\param{W_Q}, \param{W_K}, \param{W_V}$: learned projection matrices (the parameters of the layer)
- $\pred{Q}, \pred{K}, \pred{V} \in \mathbb{R}^{n \times \hyper{d_k}}$: one query, key and value row per token

**Example:** $(5 \times 4)(4 \times 3) = 5 \times 3$; the query of "good" is $\pred{q_{good}}$ = [−0.4751, 0.5663, −2.1041] and its key is $\pred{k_{good}}$ = [0.5940, 1.1215, −1.6942]: same token, different vectors.

</Formula>

:::danger[Interview Trap: "Q, K and V are three copies of the input"]
They are three **different linear projections** of the input, with independently learned weights. If $W_Q = W_K$, the score matrix $QK^\top$ would be symmetric (token $i$ would rate $j$ exactly as $j$ rates $i$), and attention couldn't express one-directional relations such as "the adjective looks for its negation, but the negation looks for its verb".
:::

:::tip[🎯 Interview angle]
**"Why do we need Q, K and V instead of using X directly?"** Because the three roles differ: what a token searches for, how it can be found and what it contributes are different things. Three learned projections let each role specialize. Using $X$ directly would make matching mean raw embedding similarity, give a symmetric score matrix, and make each token mostly attend to itself.
:::

## 4. The Shapes of Q, K and V

Self-attention is almost entirely matrix multiplication, so most bugs and most interview questions are about **shapes**. The rule is simple: the projections change the **feature** dimension, never the **token** dimension. $X$ has $n$ rows, so $Q$, $K$ and $V$ have $n$ rows too: row 5 of each is still "good". Only the width changes from $d_{model}$ to $d_k$.

Two constraints matter. A query must be compared with a key by a dot product, $q_i^\top k_j$, which is only defined when both have the **same length**: $q_i$ and $k_j$ must both live in $\mathbb{R}^{d_k}$. The values are never dot-producted with anything; they are only averaged, so their size $d_v$ can be different. The notebook sets $d_v = d_k = 3$ for simplicity, and the original Transformer also uses $d_v = d_k$ (64 per head), but the formula works for any $d_v$, and the output $Z$ takes the values' width.

| Matrix | General shape | Notebook | One row |
|---|---|---|---|
| $X$ | $n \times d_{model}$ | 5 × 4 | $x_i \in \mathbb{R}^4$ |
| $W_Q$, $W_K$ | $d_{model} \times d_k$ | 4 × 3 | |
| $W_V$ | $d_{model} \times d_v$ | 4 × 3 | |
| $Q$, $K$ | $n \times d_k$ | 5 × 3 | $q_i, k_i \in \mathbb{R}^3$ |
| $V$ | $n \times d_v$ | 5 × 3 | $v_i \in \mathbb{R}^3$ |

:::tip[🎯 Interview angle]
**"Which dimensions in attention must match, and which are free?"** $Q$, $K$ and $V$ keep the sequence length $n$ of $X$. Queries and keys must have the same width $d_k$ because they are dot-producted. The value width $d_v$ is free, and it sets the output width. In practice $d_k = d_v = d_{model}/h$ per head (64 in the original Transformer with $d_{model} = 512$ and $h = 8$).
:::

## 5. The Score Matrix S = QKᵀ

Now the core interaction. Each token asks, "how relevant is every other token to me?" The answer is a dot product between its **query** and every token's **key**. Doing this for all pairs at once is a single product: $S = QK^\top$. Transposing $K$ turns it into $d_k \times n$, so $(n \times d_k)(d_k \times n) = n \times n$: a **square** matrix with one row and one column per token, because every token is compared with every token of the same sequence.

**Rows are queries (who is asking); columns are keys (who is being looked at).** Row $i$ lists how token $i$ scores every token: $S_{i,:} = [q_i \cdot k_1, \ldots, q_i \cdot k_n]$. This is where self-attention's $O(n^2)$ cost comes from: five tokens give 25 scores, a 1,000-token prompt gives a million, per head per layer.

These are **raw** scores. They are unbounded, can be negative, and don't sum to anything in particular. In the notebook's run they range from −8.73 to +22.47. They tell you which keys match a query better, but they can't be used as averaging weights yet.

![Where one score comes from: S = QK^T](/img/deep-learning/diagrams/score-matrix-shapes.svg)

*How to read it:* the highlighted row of $Q$ (the query of "good", length $d_k$) meets the highlighted column of $K^\top$ (the key of "not") in one dot product, which fills the single dark cell of $S$ at row "good", column "not". Repeating this for every row-column pair fills the whole $n \times n$ matrix.

| $S = QK^\top$ (query ↓, key →) | The | movie | was | not | good |
|---|---|---|---|---|---|
| **The** | 2.257 | 3.089 | −4.421 | 6.430 | −5.105 |
| **movie** | 1.010 | 4.089 | −0.287 | −8.350 | −1.599 |
| **was** | 0.008 | 0.625 | 0.966 | −5.088 | 0.378 |
| **not** | −2.835 | −8.727 | 0.091 | **22.475** | 4.207 |
| **good** | −1.605 | −1.493 | 3.856 | **−8.661** | 3.918 |

<Formula name="Attention score matrix" tag="Every query vs every key" words="Multiply the queries by the transposed keys; entry i, j is the dot product of token i's query with token j's key." remember="Rows ask, columns answer.">

$$
\pred{S} = \pred{Q}\pred{K}^\top \in \mathbb{R}^{n \times n}, \qquad \pred{S_{ij}} = \pred{q_i}^\top \pred{k_j} = \sum_{\ell=1}^{\hyper{d_k}} \pred{q_{i\ell}}\,\pred{k_{j\ell}}
$$

- row $i$: query token $i$ compared with all keys; column $j$: key token $j$ as seen by all queries
- $(n \times \hyper{d_k})(\hyper{d_k} \times n) = n \times n$, so the cost is $O(n^2 \hyper{d_k})$

**Example:** $\pred{S_{good,not}}$ = (−0.4751)(4.5272) + (0.5663)(−0.9301) + (−2.1041)(2.8439) = −8.661.

</Formula>

:::tip[🎯 Interview angle]
**"What is QKᵀ, and why is it square?"** It is the matrix of all query-key dot products: entry $(i, j)$ says how well token $i$'s query matches token $j$'s key. It is $n \times n$ because in self-attention the same $n$ tokens are both the askers and the answerers. That square is why attention costs $O(n^2)$ time and memory in the sequence length. The entries are raw, unbounded scores, not weights yet.
:::

## 6. What One Score Means, and How to Read the Raw-Score Heatmap

Zoom in on a single entry. $S_{ij}$ answers: **how strongly is token $i$ trying to attend to token $j$?** Token $i$ is the one **asking** (its query), token $j$ the one **being compared** (its key). The order matters: $S_{good,not}$ (does "good" want "not"?) and $S_{not,good}$ (does "not" want "good"?) use different vectors and are generally different numbers: −8.661 and 4.207 in this run.

There is nothing mysterious inside an entry. It is a sum of $d_k$ products, one per feature: multiply the query's first number by the key's first number, and so on, and add. The notebook checks this by computing $q_{good} \cdot k_{not}$ by hand and comparing it with $S[4, 3]$: both give −8.66147. The whole row for "good", [−1.605, −1.493, 3.856, −8.661, 3.918], answers the question "when we update 'good', which tokens should contribute?" before any normalization. In a trained model we'd hope $S_{good,not}$ is large, because "not" changes what "good" means. Here it is the **most negative** score in the row, because the weights are random.

A **heatmap** of $S$ (rows = query tokens, columns = key tokens, colour = value) shows the whole interaction map at once: scan along a row to see what one token is drawn to. But a bright cell in the **raw** heatmap is not yet "attention". The values are unscaled and unnormalized, and they live on very different scales row to row. The "not" row spans from −8.7 to 22.5 while the "was" row stays between −5.1 and 1.0, so a colour scale shared by all rows hides most of the structure. Raw scores this large also foreshadow the problem of the next two chapters: after softmax, one token can swallow nearly all the weight.

:::danger[Interview Trap: "A brighter cell in QKᵀ means more attention"]
Only **within a row**, and only before you account for the softmax. Raw scores in different rows are not comparable (each row is normalized separately), a raw score of 4 can mean 50% or 0.01% depending on the rest of its row, and the final weights depend on score **differences** after scaling. Always read attention from $A$, not from $S$.
:::

:::tip[🎯 Interview angle]
**"What does a single entry S_ij represent, and is it symmetric?"** It is the dot product of token $i$'s query with token $j$'s key: how strongly token $i$ (the asker) is drawn to token $j$ (the one compared). It is not symmetric, because $q_i \cdot k_j$ uses different vectors from $q_j \cdot k_i$ (here $S_{good,not} = -8.66$ but $S_{not,good} = 4.21$). Raw scores are unnormalized, so they are only comparable within a row.
:::

## 7. Why Scaling by √d_k Is Needed

A dot product adds up $d_k$ terms. If the entries of queries and keys are roughly independent with mean 0 and variance 1, each product $q_\ell k_\ell$ also has variance 1, and the variance of a sum of independent terms is the sum of the variances. So $\operatorname{Var}(q \cdot k) = d_k$, and the typical size (standard deviation) of a score grows like $\sqrt{d_k}$. This has nothing to do with how relevant tokens are; it's purely a side effect of the vector length. With $d_k = 64$ (the original Transformer's per-head size), scores are about 8 times larger than with $d_k = 1$.

The notebook measures it: 1,000 random query-key pairs at each size. The raw standard deviation follows the theory $\sqrt{d_k}$ closely (1.38 vs 1.41 at $d_k = 2$, 4.00 vs 4.00 at 16, 11.02 vs 11.31 at 128). Dividing every score by $\sqrt{d_k}$ brings the standard deviation back to about 1 (between 0.94 and 1.04) at **every** size. The scaled scores have the same ranking as the raw ones; only their magnitude is controlled.

Why $\sqrt{d_k}$ and not $d_k$? Because the goal is to undo the growth of the **standard deviation**, which is $\sqrt{d_k}$. Dividing by $d_k$ would overshoot: the scores' standard deviation would become $1/\sqrt{d_k}$ (0.125 at $d_k = 64$), every score would be squeezed toward 0, and softmax would return nearly uniform weights regardless of relevance.

In the notebook's own example, scaling divides the whole $5 \times 5$ score matrix by $\sqrt{3} = 1.732$, so its standard deviation drops from 6.12 to 3.53. Notice that 3.53 is still well above 1. The variance argument assumes query and key entries with variance about 1, but the notebook's weights are drawn from $N(0, 1)$ **without** any initialization scaling, so each entry of $q$ sums 4 random products and has variance about $d_{model} = 4$ (measured: 3.72 on average over 2,000 random draws). With properly initialized weights (variance $1/d_{model}$, like Xavier or PyTorch's default), the entries have variance 0.96 and the argument holds. The $\sqrt{d_k}$ factor fixes the dependence on $d_k$; it relies on sensible initialization and normalization (LayerNorm before attention, in modern models) for the rest.

<Animation src="self-attention/scaling" caption="Standard deviation of 1000 random dot products at each d_k. Watch the raw bars climb from 1.38 to 11.02, roughly the square root of d_k, while the scaled bars stay near 1 at every size." />

<Formula name="Variance of a dot product" tag="Why scores grow" words="Adding d_k independent unit-variance products gives variance d_k, so a score's typical size grows like the square root of d_k; dividing by that square root brings the variance back to 1." remember="Variance adds; divide by the root.">

$$
\begin{aligned}
\operatorname{Var}(\pred{q_i}^\top \pred{k_j}) &= \sum_{\ell=1}^{\hyper{d_k}} \operatorname{Var}(\pred{q_{i\ell}}\,\pred{k_{j\ell}}) \approx \hyper{d_k} \\
\operatorname{Var}\!\left(\frac{\pred{q_i}^\top \pred{k_j}}{\sqrt{\hyper{d_k}}}\right) &\approx 1
\end{aligned}
$$

- assumes entries of $\pred{q}$ and $\pred{k}$ are independent with mean 0 and variance 1
- dividing by $\hyper{d_k}$ instead would give variance $1/\hyper{d_k}$: scores squashed toward 0

**Example:** measured standard deviations, raw vs scaled: $\hyper{d_k}$ = 2: 1.38 vs 0.98; 16: 4.00 vs 1.00; 128: 11.02 vs 0.97.

</Formula>

:::danger[Interview Trap: "The √d_k normalizes the scores into probabilities"]
Scaling only shrinks the scores by a constant; they are still unbounded and can be negative. **Softmax** makes them probabilities. And $\sqrt{d_k}$ only removes the dependence on vector length under the unit-variance assumption: as this notebook shows, badly scaled weights (variance 3.7 per entry) still produce saturated rows after scaling.
:::

:::tip[🎯 Interview angle]
**"Why divide by √d_k in attention? Why not d_k?"** A dot product of two $d_k$-dimensional vectors with independent unit-variance entries has variance $d_k$, so its standard deviation grows like $\sqrt{d_k}$. Large scores push softmax into saturation, where it is nearly one-hot and its gradients vanish. Dividing by $\sqrt{d_k}$ restores unit variance at any head size; dividing by $d_k$ would over-shrink the scores to variance $1/d_k$ and make attention nearly uniform.
:::

## 8. The Softmax Sharpness Problem

Each row of scores goes through a softmax: exponentiate, then divide by the row total. Softmax is not only about **ranking**; it is very sensitive to **magnitude**. Because it exponentiates, the weight ratio between two tokens is $e^{\text{score gap}}$. A gap of 1 is a factor of 2.7, a gap of 5 is a factor of 148, and a gap of 10 is a factor of 22,000. Multiply all the scores by a constant and the ranking stays the same, but the distribution gets sharper.

The notebook shows this with three score vectors that rank the tokens almost the same way. Close-together scores [1.0, 1.2, 0.8, 1.1, 0.9] give a smooth distribution: the largest weight is 0.242 and every token keeps between 16% and 24%. Spread-out scores [2, 4, 1, 3, 0.5] give 0.632 to the top token. Large scores [10, 20, 5, 15, 2] give **0.993** to one token and between 0.00000002 and 0.007 to the others: almost **one-hot**.

A nearly one-hot row is a problem for two reasons. It throws away context, because the token copies one neighbour instead of blending several. Worse, it stops learning. The derivative of a softmax output with respect to a score is $\partial A_i / \partial s_j = A_i(\delta_{ij} - A_j)$, and when one $A$ is near 1 and the rest near 0, every one of these terms is near 0. For the small scores the largest entry of that gradient matrix is 0.183; for the large scores it is 0.0067, about 27 times smaller. Early in training, when the scores are essentially random, saturated attention would lock in arbitrary choices and give the model almost no signal to fix them. Scaling by $\sqrt{d_k}$ keeps the scores in the range where softmax is still soft.

In the notebook's own "good" row the effect is modest because $d_k$ is only 3. Raw softmax gives [0.0021, 0.0023, 0.4825, 0.0000018, 0.5132]; scaled softmax gives [0.0201, 0.0214, 0.4705, 0.0003, 0.4876]. The small weights grow roughly tenfold (0.002 to 0.02), which is exactly the "less confident too early" behaviour scaling is for. With realistic head sizes the difference is far larger: try the widget below.

<Animation src="self-attention/sharpness" caption="Three score vectors with nearly the same ranking. Watch the largest weight grow from 0.242 to 0.632 to 0.993 as the numbers get bigger, until one token takes almost everything." />

<Formula name="Row-wise softmax with scaling" tag="Attention weights" words="Divide every score by the square root of d_k, exponentiate, and divide by the row total; each row becomes non-negative weights that sum to 1." remember="Shrink, exponentiate, share out.">

$$
\begin{aligned}
\pred{A_{ij}} &= \frac{\exp\!\left(\pred{S_{ij}} / \sqrt{\hyper{d_k}}\right)}{\sum_{m=1}^{n} \exp\!\left(\pred{S_{im}} / \sqrt{\hyper{d_k}}\right)} \\
0 \le \pred{A_{ij}} &\le 1, \qquad \textstyle\sum_{j} \pred{A_{ij}} = 1
\end{aligned}
$$

- softmax is applied to each **row** separately (one distribution per query token)
- weight ratios are exponential in score gaps: $\pred{A_{ij}}/\pred{A_{im}} = e^{(\pred{S_{ij}} - \pred{S_{im}})/\sqrt{\hyper{d_k}}}$

**Example:** scores [10, 20, 5, 15, 2] → largest weight 0.993 (near one-hot); [1.0, 1.2, 0.8, 1.1, 0.9] → largest weight 0.242 (smooth).

</Formula>

**Try it yourself.** The row is the notebook's "good" row, with its raw scores rebuilt for any $d_k$ (at $d_k = 3$ they are exactly the notebook's). Slide $d_k$ up toward 64 and 512 and compare raw softmax with scaled softmax.

<ScalingSharpnessPlayground />

:::analogy[A microphone with the gain turned up]
Softmax is a mixing desk that normalizes total volume. When the inputs are at sensible levels, you hear every voice, loudest first. Turn the gain up on every channel equally and the desk's limiter lets only the loudest voice through; everyone else is cut to silence. Dividing by $\sqrt{d_k}$ is turning the gain back down so the quieter voices stay in the mix.
:::

:::tip[🎯 Interview angle]
**"What happens to softmax when the scores are large, and why is that bad for training?"** Softmax depends on score differences exponentially, so large scores make it nearly one-hot: one token gets almost all the weight. Then the softmax Jacobian $A_i(\delta_{ij} - A_j)$ is close to zero everywhere, so the gradients that would teach the scoring function vanish, and the token ignores the rest of its context. Scaling by $\sqrt{d_k}$ keeps scores at unit variance so attention stays soft early in training.
:::

## 9. The Full Scaled Dot-Product Attention Formula

All the pieces now fit into one line. Attention works in two stages: first decide **how much** each token should attend to each other token ($A$), then **use** those weights to combine information from the value vectors ($Z$). Written out, there are four steps:

| # | Step | Formula | Shape (general) | Notebook |
|---|---|---|---|---|
| 1 | Raw scores | $S = QK^\top$ | $n \times n$ | 5 × 5 |
| 2 | Scale | $S / \sqrt{d_k}$ | $n \times n$ | 5 × 5 |
| 3 | Row-wise softmax | $A = \operatorname{softmax}(S / \sqrt{d_k})$ | $n \times n$, rows sum to 1 | 5 × 5 |
| 4 | Mix the values | $Z = AV$ | $(n \times n)(n \times d_v) = n \times d_v$ | 5 × 3 |

Each piece has one job. $QK^\top$ measures compatibility, $\sqrt{d_k}$ stabilizes magnitude, softmax turns compatibility into a distribution, and multiplying by $V$ produces the new representations. Note what is **learned** and what isn't: the only parameters are $W_Q$, $W_K$ and $W_V$; the scaling and the softmax are fixed operations. Everything the layer learns about which tokens should attend to which is encoded in those three matrices.

![Scaled dot-product self-attention, shapes for the notebook example](/img/deep-learning/diagrams/self-attention-pipeline.svg)

*How to read it:* $X$ (5 × 4) splits into three projections. $Q$ and $K$ meet in $QK^\top$ to give the 5 × 5 score matrix $S$, which is divided by $\sqrt{d_k}$ and softmaxed row by row into $A$. $A$ then multiplies $V$, which arrives along the bottom path untouched by the scoring, to give $Z$, one 3-number context-aware row per token. Values never influence the weights; queries and keys never appear in the output directly.

<Formula name="Scaled dot-product attention" tag="The Transformer's core" words="Score every query against every key, scale by the square root of the key size, softmax each row into weights, and use the weights to average the value vectors." remember="Match, scale, normalize, mix.">

$$
\pred{Z} = \operatorname{Attention}(\pred{Q}, \pred{K}, \pred{V}) = \operatorname{softmax}\!\left(\frac{\pred{Q}\pred{K}^\top}{\sqrt{\hyper{d_k}}}\right)\pred{V}
$$

- $\pred{Q}, \pred{K} \in \mathbb{R}^{n \times \hyper{d_k}}$, $\pred{V} \in \mathbb{R}^{n \times \hyper{d_v}}$, $\pred{Z} \in \mathbb{R}^{n \times \hyper{d_v}}$
- softmax is row-wise; the only learned parameters are $\param{W_Q}, \param{W_K}, \param{W_V}$

**Example:** in the notebook, $(5 \times 3)(3 \times 5) \to 5 \times 5$ scores and weights, then $(5 \times 5)(5 \times 3) \to \pred{Z}$ of shape 5 × 3; $\pred{z_{good}}$ = [0.7465, −0.0026, −0.7641].

</Formula>

**Try it yourself.** Step through the notebook's actual matrices. Click any cell to see the exact arithmetic that produced it: the dot product behind a score, the division by $\sqrt{3}$, the softmax fraction, or the weighted sum behind an output.

<AttentionPipelineExplorer />

:::tip[🎯 Interview angle]
**"Write the attention formula and walk through the shapes."** $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$. With $n$ tokens: $Q, K$ are $n \times d_k$, so $QK^\top$ is $n \times n$; scaling and the row-wise softmax keep $n \times n$ with rows summing to 1; multiplying by $V$ ($n \times d_v$) gives $n \times d_v$, one output per token. Time and memory are $O(n^2 d_k)$ and $O(n^2)$ for the score matrix.
:::

## 10. The Attention Weight Matrix A

$A = \operatorname{softmax}(QK^\top / \sqrt{d_k})$ is the **attention weight matrix**: how much each token attends to every other token. Its rows and columns read the same way as in $S$, but now the numbers are meaningful as weights. **Row $i$** belongs to query token $i$: it says how token $i$ spreads its attention, and therefore how much of each value vector goes into its new representation. **Column $j$** belongs to key/value token $j$: it shows how much attention token $j$ receives from all the others.

Every row is a **probability distribution**: all entries between 0 and 1, summing to exactly 1, because softmax was applied to each row. That makes $Z = AV$ a weighted **average** rather than an arbitrary sum. Columns have no such constraint. In this run they sum to 0.407, 1.177, 0.853, 1.818 and 0.745, so "not" is the most attended-to token overall.

Here is the notebook's $A$. It is easier to read than the raw heatmap because every row is on the same 0-to-1 scale. It also shows what random weights do. "not" puts 0.99997 of its attention on itself, a near one-hot row, because its raw score with itself was 22.47 (12.98 even after scaling). "The" sends 0.808 to "not". "good" splits its attention between "was" (0.471) and itself (0.488) and gives "not" only 0.0003. None of this is linguistics; it is the arithmetic of untrained weights.

| $A$ (query ↓, key →) | The | movie | was | not | good | row sum |
|---|---|---|---|---|---|---|
| **The** | 0.073 | 0.117 | 0.002 | **0.808** | 0.001 | 1 |
| **movie** | 0.131 | **0.777** | 0.062 | 0.001 | 0.029 | 1 |
| **was** | 0.183 | 0.262 | **0.319** | 0.010 | 0.227 | 1 |
| **not** | 0.000 | 0.000 | 0.000 | **1.000** | 0.000 | 1 |
| **good** | 0.020 | 0.021 | 0.471 | 0.000 | **0.488** | 1 |

| Query token | Attends most to | Weight |
|---|---|---|
| The | not | 0.8075 |
| movie | movie | 0.7768 |
| was | was | 0.3185 |
| not | not | 1.0000 |
| good | good | 0.4876 |

:::danger[Interview Trap: "good ignores not, so self-attention failed"]
These weights come from **random** $W_Q$ and $W_K$. Nothing has been trained, so there is no reason for "good" to look at "not". In a trained model the projections are learned so that useful pairs get high scores (and with multiple heads, different heads pick up different relations). The toy run shows the mechanism and the shapes, not what a model learns.
:::

:::tip[🎯 Interview angle]
**"What do the rows and columns of the attention matrix mean? Why do rows sum to 1?"** Row $i$ is query token $i$'s distribution of attention over all tokens; it sums to 1 because softmax is applied row-wise, which makes each output a weighted average of the values. Column $j$ is how much attention token $j$ receives; columns needn't sum to 1. In a causal (GPT-style) model the upper triangle is masked to zero, so each row only covers earlier tokens.
:::

## 11. Contextual Token Representations: Z = AV

The weights alone are not the output; they only say how much **value** information to borrow from each token. The output is $Z = AV$: $(n \times n)(n \times d_v) = n \times d_v$, one new row per token. Row $i$ is $z_i = \sum_j A_{ij} v_j$, a weighted mixture of **all** value vectors, mixed according to token $i$'s row of $A$.

Expanded for "good": $z_{good} = A_{good,The}\,v_{The} + A_{good,movie}\,v_{movie} + A_{good,was}\,v_{was} + A_{good,not}\,v_{not} + A_{good,good}\,v_{good}$. With this run's weights, "was" contributes [0.392, 0.105, −0.405] and "good" itself contributes [0.327, −0.131, −0.384]; together those two tokens hold 96% of the weight. The other three add tiny amounts, and "not", at weight 0.0003, adds about [−0.001, −0.000, 0.001]. The total is $z_{good}$ = [0.7465, −0.0026, −0.7641]. The notebook computes it both by the loop and by the matrix product and gets the same answer.

That is why $z_i$ is called a **contextual** representation. $x_{good}$ is the same in every sentence; $z_{good}$ is built from this sentence's tokens and would be different in "a good movie". Attention doesn't copy one token; it creates a weighted average of information from several. In a trained model this is how "good" absorbs the negation from "not": its row of $A$ puts weight on "not", and the value of "not" flows into $z_{good}$.

<Animation src="self-attention/z-equals-av" caption="The good row of A multiplies each value vector. Watch the was and good contributions dominate and the not contribution almost vanish, then add up to z_good of 0.746, -0.003, -0.764." />

<Formula name="Contextual output" tag="Z = AV" words="Each token's new vector is its row of attention weights times the value vectors: a weighted average of every token's value." remember="Each row is a custom average.">

$$
\pred{Z} = \pred{A}\pred{V} \in \mathbb{R}^{n \times \hyper{d_v}}, \qquad \pred{z_i} = \sum_{j=1}^{n} \pred{A_{ij}}\,\pred{v_j}
$$

- $\pred{A_{ij}}$: how much of token $j$'s value goes into token $i$'s new vector
- $\pred{z_i}$: the context-aware representation of token $i$

**Example:** $\pred{z_{good}}$ = 0.0201·$\pred{v_{The}}$ + 0.0214·$\pred{v_{movie}}$ + 0.4705·$\pred{v_{was}}$ + 0.0003·$\pred{v_{not}}$ + 0.4876·$\pred{v_{good}}$ = [0.7465, −0.0026, −0.7641].

</Formula>

:::analogy[A smoothie recipe per customer]
The value vectors are the fruits on the counter. Each customer (token) has their own recipe card (a row of $A$) saying what fraction of each fruit to blend, and the fractions always add up to one cup. Every customer leaves with a different smoothie made from the same fruits. "good"'s recipe in this run is roughly half "was", half "good", and a pinch of everything else.
:::

:::tip[🎯 Interview angle]
**"What does Z = AV compute, and why is z_i called contextual?"** Each row $z_i = \sum_j A_{ij} v_j$ is a weighted average of all tokens' value vectors, with weights from token $i$'s attention row. It is contextual because it depends on the other tokens in this particular sentence, unlike the static embedding $x_i$. $V$ supplies the content; $A$ only decides the mix.
:::

## 12. Original Embedding vs Attention-Updated Representation

The notebook ends the computation by comparing $x_{good}$, the input embedding, with $z_{good}$, the attention output. $x_{good}$ = [−1.0128, 0.3142, −0.9080, −1.4123] is the token **before** any interaction: the same vector in every sentence. $z_{good}$ = [0.7465, −0.0026, −0.7641] is "good" **after** collecting information from the whole sentence. The transformation chain is $x_i \to (q_i, k_i, v_i) \to A_{i,:} \to z_i$.

There is a catch the notebook points out: the two vectors don't even have the same size. $x_{good}$ lives in $\mathbb{R}^{d_{model}} = \mathbb{R}^4$ and $z_{good}$ in $\mathbb{R}^{d_v} = \mathbb{R}^3$, so you can't subtract them element by element, and plotting them side by side only compares shapes loosely. The technically valid comparison is between $z_{good}$ and $v_{good}$ = [0.6715, −0.2694, −0.7881]: both live in the same 3-dimensional value space. $v_{good}$ is what "good" would contribute **on its own**; $z_{good}$ is what it becomes after mixing in "was" (0.471) and small amounts of the others. The difference between them is exactly the context that attention added.

Even that comparison has **no semantic meaning** here. With random weights, the direction in which $z_{good}$ moves away from $v_{good}$ is arbitrary; it isn't "more negative" or "more neutral" in any linguistic sense. What matters is the mechanism, $Z = AV$. In a real Transformer, the attention output is also projected back to $d_{model}$ by an output matrix $W_O$ and **added** to the input through a residual connection ($x + \text{Attention}(x)$), so input and output live in the same space and can be compared directly. That's the next notebook's territory.

| Vector | Space | Value | What it is |
|---|---|---|---|
| $x_{good}$ | $\mathbb{R}^4$ ($d_{model}$) | [−1.0128, 0.3142, −0.9080, −1.4123] | Static embedding, no context |
| $v_{good}$ | $\mathbb{R}^3$ ($d_v$) | [0.6715, −0.2694, −0.7881] | "good"'s own contribution |
| $z_{good}$ | $\mathbb{R}^3$ ($d_v$) | [0.7465, −0.0026, −0.7641] | "good" after attention (mixed with "was" and others) |

:::tip[🎯 Interview angle]
**"How do you compare a token's representation before and after attention?"** Compare like with like. The attention output $z_i$ lives in the value space ($d_v$), so in a single head it should be compared with $v_i$, not with the embedding $x_i$ ($d_{model}$). In a full Transformer layer, $W_O$ projects the output back to $d_{model}$ and the residual connection adds it to $x_i$, so the updated representation is $x_i + W_O z_i$ in the same space as the input.
:::

## 13. Summary and the Bridge to Multi-Head Attention

The complete flow, start to finish: embed and stack the tokens into $X$; project into $Q$, $K$, $V$; compare all queries with all keys, $S = QK^\top$; scale by $\sqrt{d_k}$; softmax each row into $A$; mix the values, $Z = AV$. Three things happen along the way: token-to-token **compatibility** ($QK^\top$), **normalization** into weights (scaled softmax), and **mixing** into contextual representations ($AV$). The output $Z$ is no longer a set of isolated token vectors: every row carries information from the whole sequence.

One limitation is built into what we have. A single head produces **one** attention distribution per token, and therefore one kind of relationship at a time. But language has many relationships at once: "good" relates to "not" through negation, to "movie" as the thing being judged, and to "was" grammatically. One softmax row has to compromise between them, and because softmax concentrates weight, a single head tends to capture one dominant relation and blur the others.

**Multi-head attention** runs $h$ attention computations in parallel, each with its own $W_Q^{(i)}$, $W_K^{(i)}$, $W_V^{(i)}$ and a smaller $d_k = d_{model}/h$, so each head can specialize in a different relationship. Their outputs are concatenated and mixed by an output matrix $W_O$: $\operatorname{MultiHead}(X) = \operatorname{Concat}(\text{head}_1, \ldots, \text{head}_h)W_O$. The original Transformer uses $h = 8$ heads of size 64 for $d_{model} = 512$. That is the next notebook.

:::tip[🎯 Interview angle]
**"Why do we need multi-head attention if single-head attention already connects every token?"** One head gives each token a single softmax distribution, so it must compromise between different relationships (negation, subject–verb, coreference) and tends to lock onto one. Multiple heads, each with its own smaller projections, attend to different relations in parallel; their outputs are concatenated and projected by $W_O$. The total cost is about the same as one full-width head.
:::

## ⚙️ Going Deeper (Self-Attention Internals)

**Cost and memory.** Computing $QK^\top$ costs $O(n^2 d_k)$ multiply-adds and storing $A$ takes $O(n^2)$ memory per head per layer; the projections cost $O(n \cdot d_{model} \cdot d_k)$. For short sequences the projections and feed-forward layers dominate; for long contexts the $n^2$ terms do. FlashAttention computes exactly the same $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$ in tiles, using the max-subtraction trick incrementally ("online softmax"), so the full $n \times n$ matrix is never stored in GPU memory.

**Batched shapes in practice.** Real code works on tensors of shape (batch, heads, $n$, $d_k$). `Q @ K.transpose(-2, -1)` transposes only the last two axes, and softmax uses `dim=-1` (over keys, so each query's row sums to 1). Using the wrong axis (`dim=-2`) silently normalizes columns instead, a classic bug that still trains but badly.

**`nn.Linear` stores the transpose.** `nn.Linear(d_model, d_k).weight` has shape $(d_k, d_{model})$ and computes $xW^\top + b$. The maths $XW_Q$ and the code `self.W_Q(X)` are the same thing with the weight stored transposed. Attention projections are often created with `bias=False`, as in the implementation below.

**Masking.** Two masks are added to the scaled scores **before** softmax, by setting forbidden entries to $-\infty$ so their weights become exactly 0: a **padding mask** (ignore pad tokens in a batch) and a **causal mask** (GPT-style decoders: token $i$ may only attend to positions $j \le i$, so the upper triangle of $A$ is zero). Masking after softmax would break the "rows sum to 1" property.

**Scaling as a temperature.** Dividing by $\sqrt{d_k}$ is the same as a softmax temperature $T = \sqrt{d_k}$. Some models instead learn a temperature or normalize $q$ and $k$ first (QK-norm) to keep attention logits from growing during training, which is a known cause of training instability in very large models.

**Initialization matters.** The $\sqrt{d_k}$ argument assumes unit-variance query and key entries. That needs the projections initialized with variance about $1/d_{model}$ (Xavier/Glorot, PyTorch's `nn.Linear` default is similar in scale) and the inputs normalized (LayerNorm). The notebook's $W \sim N(0, 1)$ gives entries of variance about $d_{model}$, which is why its rows stay saturated after scaling.

**Permutation equivariance.** Nothing in $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$ refers to position: shuffle the rows of $X$ and the rows of $Z$ shuffle the same way. Word order has to be added to $X$ as positional encodings (previous page).

**$d_v$ is free.** Only $d_k$ has to match between queries and keys. $d_v$ sets the output width; multi-head attention uses $d_v = d_{model}/h$ so the concatenated heads come back to $d_{model}$.

## 💻 Python Implementation

:::tip[Run it yourself]
Verified on Python 3.13 with NumPy 2.5 and PyTorch 2.15 (CPU). Blocks 1–5 continue one session, in order, reproducing the notebook's random numbers (`np.random.seed(42)`); blocks 6–8 are self-contained.
:::

**What was corrected from the notebook.** The final `scaled_dot_product_attention()` function creates new random $W_Q$, $W_K$, $W_V$ **inside** the function, so every call uses different "learnable" weights and gives a different result (block 6); weights belong to the layer and must persist across calls. The notebook also stores standard deviations in lists named `average_raw_scores` (the values are standard deviations, not averages), and its scaling demo only reproduces if the cells run in order, because it continues the same random stream. Block 8 measures why scaling alone doesn't remove the saturation in this example.

### 1. The input matrix and the projections

```python title="self_attention_setup.py" showLineNumbers
import numpy as np
np.random.seed(42)

def row_softmax(M):
    M = M - M.max(axis=1, keepdims=True)                 # stable: subtract each row's max
    E = np.exp(M)
    return E / E.sum(axis=1, keepdims=True)

tokens = ["The", "movie", "was", "not", "good"]
X = np.random.randn(5, 4)                                # n = 5 tokens, d_model = 4
print(np.dot(X[4], X[3]))                                # 1.5140775469995646   static, context-free

d_k = 3
W_Q = np.random.randn(4, d_k)                            # learnable in a real model
W_K = np.random.randn(4, d_k)
W_V = np.random.randn(4, d_k)
# highlight-next-line
Q, K, V = X @ W_Q, X @ W_K, X @ W_V
print(Q.shape, K.shape, V.shape)                         # (5, 3) (5, 3) (5, 3)
print(np.round(Q[4], 4), np.round(K[4], 4))              # [-0.4751  0.5663 -2.1041] [ 0.594   1.1215 -1.6942]
```

### 2. The score matrix and one score by hand

```python title="score_matrix.py" showLineNumbers
# highlight-next-line
S = Q @ K.T                                              # (5, 3) @ (3, 5) -> (5, 5)
print(S.shape)                                           # (5, 5)
print(np.round(S[4], 4))                                 # [-1.605  -1.4927  3.8561 -8.6615  3.9177]   row "good"
i, j = tokens.index("good"), tokens.index("not")
print(np.dot(Q[i], K[j]), S[i, j])                       # -8.66147014985341 -8.66147014985341
print(S[i, j], S[j, i])                                  # -8.66147014985341 4.207336971086857   not symmetric
print(S.min().round(3), S.max().round(3))                # -8.727 22.475
```

### 3. Why √d_k: dot products grow with dimension

```python title="scaling_demo.py" showLineNumbers
dims, raw_std, scaled_std = [2, 4, 8, 16, 32, 64, 128], [], []
for d in dims:
    r = np.array([np.dot(np.random.randn(d), np.random.randn(d)) for _ in range(1000)])
    raw_std.append(r.std())
    # highlight-next-line
    scaled_std.append((r / np.sqrt(d)).std())
print(np.round(np.sqrt(dims), 2))                        # [ 1.41  2.    2.83  4.    5.66  8.   11.31]   theory
print(np.round(raw_std, 2))                              # [ 1.38  2.09  2.94  4.    5.32  7.88 11.02]
print(np.round(scaled_std, 2))                           # [0.98 1.04 1.04 1.   0.94 0.99 0.97]
S_scaled = S / np.sqrt(d_k)
print(S.std().round(2), S_scaled.std().round(2))         # 6.12 3.53
```

### 4. Softmax sharpness, and the full formula

```python title="softmax_and_attention.py" showLineNumbers
def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

for s in ([1.0, 1.2, 0.8, 1.1, 0.9], [2.0, 4.0, 1.0, 3.0, 0.5], [10.0, 20.0, 5.0, 15.0, 2.0]):
    print(np.round(softmax(np.array(s)), 4))
# [0.198  0.2419 0.1621 0.2188 0.1792]
# [0.0855 0.6316 0.0314 0.2324 0.0191]
# [0.     0.9933 0.     0.0067 0.    ]
print(np.round(softmax(S[4]), 4))                        # [0.0021 0.0023 0.4825 0.     0.5132]   raw
print(np.round(softmax(S_scaled[4]), 4))                 # [2.010e-02 2.140e-02 4.705e-01 3.000e-04 4.876e-01]   scaled

# success-start
A = row_softmax(S / np.sqrt(d_k))                        # (5, 5), rows sum to 1
Z = A @ V                                                # (5, 3)
# success-end
print(A.sum(axis=1))                                     # [1. 1. 1. 1. 1.]
print(np.round(A.sum(axis=0), 3))                        # [0.407 1.177 0.853 1.818 0.745]   columns: no rule
for t, row in zip(tokens, A):
    print(t, tokens[row.argmax()], round(row.max(), 4))
# The not 0.8075
# movie movie 0.7768
# was was 0.3185
# not not 1.0
# good good 0.4876
```

### 5. z_good as a weighted sum of value vectors

```python title="z_good.py" showLineNumbers
g = tokens.index("good")
contributions = A[g][:, None] * V                        # one row per token: A[good, j] * v_j
print(np.round(contributions, 4))
# [[ 5.200e-03  1.840e-02  2.000e-02]
#  [ 2.320e-02  5.800e-03  3.600e-03]
#  [ 3.918e-01  1.048e-01 -4.048e-01]
#  [-1.100e-03 -3.000e-04  1.300e-03]
#  [ 3.274e-01 -1.313e-01 -3.842e-01]]
# highlight-next-line
print(np.round(contributions.sum(axis=0), 4), np.round(Z[g], 4))   # [ 0.7465 -0.0026 -0.7641] [ 0.7465 -0.0026 -0.7641]
print(np.round(V[g], 4))                                 # [ 0.6715 -0.2694 -0.7881]   same space as z_good
print(X[g].shape, Z[g].shape)                            # (4,) (3,)   x_good can't be subtracted from z_good
```

### 6. ❌ Weights created inside the function vs ✅ weights that persist

```python title="❌ The notebook's function draws new 'learnable' weights on every call" showLineNumbers
def scaled_dot_product_attention(X, d_k=3):
    n, d_model = X.shape
    # error-next-line
    W_Q, W_K, W_V = (np.random.randn(d_model, d_k) for _ in range(3))   # ❌ new random weights each call
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    return row_softmax(Q @ K.T / np.sqrt(d_k)) @ V

print(np.round(scaled_dot_product_attention(X)[4], 4))   # [-0.594  -0.2032 -0.0664]
print(np.round(scaled_dot_product_attention(X)[4], 4))   # [1.5906 0.6001 0.0543]   same input, different answer
```

```python title="✅ Weights are passed in (they belong to the layer, not the call)" showLineNumbers
# success-next-line
def scaled_dot_product_attention(X, W_Q, W_K, W_V):    # ✅ same weights every call, trainable
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    A = row_softmax(Q @ K.T / np.sqrt(K.shape[1]))
    return A @ V, A
```

### 7. The same layer in PyTorch

```python title="self_attention_torch.py" showLineNumbers
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k, bias=False)   # weight shape (d_k, d_model): stored transposed
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_k, bias=False)

    def forward(self, X):                                # X: (batch, n, d_model)
        Q, K, V = self.W_Q(X), self.W_K(X), self.W_V(X)
        # highlight-next-line
        A = torch.softmax(Q @ K.transpose(-2, -1) / K.shape[-1] ** 0.5, dim=-1)   # rows over keys
        return A @ V, A

torch.manual_seed(0)
attn = SelfAttention(4, 3)
Z, A = attn(torch.randn(2, 5, 4))                        # a batch of 2 sentences
print(Z.shape, A.shape)                                  # torch.Size([2, 5, 3]) torch.Size([2, 5, 5])
print(sum(p.numel() for p in attn.parameters()))         # 36
print(attn.W_Q.weight.shape)                             # torch.Size([3, 4])

np.random.seed(42)                                       # the notebook's matrices again
Xn = np.random.randn(5, 4); Wq, Wk, Wv = (np.random.randn(4, 3) for _ in range(3))
Qt, Kt, Vt = (torch.tensor(Xn @ W, dtype=torch.float64) for W in (Wq, Wk, Wv))
print(torch.round(F.scaled_dot_product_attention(Qt, Kt, Vt)[4], decimals=4))
# tensor([ 0.7465, -0.0026, -0.7641], dtype=torch.float64)   built-in matches NumPy (max diff 2.2e-16)
```

### 8. Why √d_k didn't stop the saturation here

```python title="init_variance.py" showLineNumbers
import numpy as np
rng = np.random.default_rng(0); n, d_model, d_k = 5, 4, 3

def row_softmax(M):
    M = M - M.max(1, keepdims=True); E = np.exp(M); return E / E.sum(1, keepdims=True)

for name, scale in [("W ~ N(0, 1)          (notebook)", 1.0), ("W ~ N(0, 1/d_model)  (Xavier-like)", 1 / np.sqrt(d_model))]:
    q_var, top = [], []
    for _ in range(2000):
        X = rng.normal(size=(n, d_model)); W_Q, W_K = rng.normal(scale=scale, size=(2, d_model, d_k))
        Q, K = X @ W_Q, X @ W_K
        q_var.append(Q.var()); top.append(row_softmax(Q @ K.T / np.sqrt(d_k)).max(1).mean())
    print(f"{name}: var(q) {np.mean(q_var):.2f}, mean largest weight per row {np.mean(top):.3f}")
# W ~ N(0, 1)          (notebook): var(q) 3.72, mean largest weight per row 0.690
# W ~ N(0, 1/d_model)  (Xavier-like): var(q) 0.96, mean largest weight per row 0.404
```

With 5 tokens, perfectly uniform attention would give a largest weight of 0.2: the properly scaled version stays much softer.

## 🚨 Interview Gotchas & Edge Cases

:::warning[The Most Common Question]
**"Why do we divide by √d_k in scaled dot-product attention?"**

A dot product of two $d_k$-dimensional vectors with independent, zero-mean, unit-variance entries has variance $d_k$, so typical scores grow like $\sqrt{d_k}$ (measured: standard deviation 1.38 at $d_k = 2$ up to 11.02 at $d_k = 128$). Softmax is exponential in score differences, so large scores make it nearly one-hot (scores [10, 20, 5, 15, 2] give 0.993 to one token), its Jacobian $A_i(\delta_{ij} - A_j)$ goes to zero, and gradients vanish (27× smaller in that example). Dividing by $\sqrt{d_k}$ restores unit variance (about 1.0 at every $d_k$) so attention stays soft and trainable. Dividing by $d_k$ would over-shrink to variance $1/d_k$ and make attention nearly uniform. It assumes well-initialized, normalized inputs.
:::

- **Softmax over the right axis:** `dim=-1` (over keys). Normalizing over queries silently produces wrong attention.
- **Rows sum to 1, columns don't;** in a causal model the upper triangle of $A$ is exactly 0.
- **$QK^\top$ is not symmetric** unless $W_Q = W_K$: $S_{good,not} = -8.66$ vs $S_{not,good} = 4.21$.
- **Only $d_k$ must match;** $d_v$ is free and sets the output width.
- **Learnable means persistent:** projection weights belong to the layer (`nn.Linear` in `__init__`), never re-drawn per call.
- **`nn.Linear.weight` is $(d_{out}, d_{in})$:** the transpose of the maths' $W$.
- **Masks go before softmax,** as $-\infty$, never after (that would break the row sums).
- **Overflow:** subtract the row max before `exp`; frameworks do it for you.
- **Random weights produce meaningless patterns:** a toy run where "good" ignores "not" says nothing about attention's ability.
- **Saturation can survive scaling** if the projections are badly initialized or the inputs unnormalized.
- **Quadratic cost:** $O(n^2 d_k)$ time and $O(n^2)$ memory per head per layer for the score matrix.
- **Compare like with like:** $z_i$ lives in $\mathbb{R}^{d_v}$; compare it with $v_i$, not $x_i$ (or with $x_i + W_O z_i$ in a full layer).

## ⚡ Essential Interview Recall

A one-screen revision list. Every point is explained in the chapters above.

**Self-attention vs cross-attention**
- Self: $Q$, $K$, $V$ from the same sequence. Cross: $Q$ from the decoder, $K$/$V$ from the encoder.
- Meaning becomes contextual: each token is rebuilt from its own sentence.

**Inputs and projections**
- $X \in \mathbb{R}^{n \times d_{model}}$: rows = tokens, columns = features; rows start independent.
- $Q = XW_Q$, $K = XW_K$, $V = XW_V$; $W \in \mathbb{R}^{d_{model} \times d_k}$ are the only parameters (36 in the toy).
- Query = what I look for, key = how I'm found, value = what I pass on.
- $q_i, k_j$ both in $\mathbb{R}^{d_k}$ (dot product); $d_v$ free.

**Scores**
- $S = QK^\top \in \mathbb{R}^{n \times n}$, $S_{ij} = q_i \cdot k_j$: row = asker, column = compared; not symmetric.
- Raw scores are unbounded; compare only within a row.

**Scaling and softmax**
- $\operatorname{Var}(q \cdot k) \approx d_k$ → divide by $\sqrt{d_k}$ (std 1.38 → 11.02 raw vs about 1 scaled).
- Not $d_k$: that gives variance $1/d_k$, near-uniform attention.
- Large scores → near one-hot softmax ([10, 20, 5, 15, 2] → 0.993) → vanishing gradients.
- Toy "good" row: raw softmax small weights 0.002 → scaled 0.02.

**Formula and outputs**
- $Z = \operatorname{softmax}(QK^\top/\sqrt{d_k})V$; shapes $5 \times 4 \to 5 \times 3 \to 5 \times 5 \to 5 \times 5 \to 5 \times 3$.
- $A$: rows are distributions (sum 1), columns how much a token is attended (0.41–1.82 here).
- $z_i = \sum_j A_{ij} v_j$; $z_{good}$ = [0.7465, −0.0026, −0.7641], 96% from "was" and "good".
- Compare $z_i$ with $v_i$ (same space), not $x_i$; random weights → no semantic meaning.

**Beyond one head**
- One head = one distribution per token; multi-head runs $h$ heads with $d_k = d_{model}/h$, concat, $W_O$.
- Cost $O(n^2 d_k)$; masks ($-\infty$) before softmax; positions must be added to $X$.

## 📐 Formula Sheet

Every equation on this page in one place.

| Concept | Formula | In plain words |
|---|---|---|
| Input matrix | $\data{X} \in \mathbb{R}^{n \times \hyper{d_{model}}}$ | One embedding row per token |
| Projections | $\pred{Q} = \data{X}\param{W_Q}$, $\pred{K} = \data{X}\param{W_K}$, $\pred{V} = \data{X}\param{W_V}$ | Three learned views of each token |
| Projection shapes | $\param{W} \in \mathbb{R}^{\hyper{d_{model}} \times \hyper{d_k}}$, $\pred{Q}, \pred{K}, \pred{V} \in \mathbb{R}^{n \times \hyper{d_k}}$ | Width changes, token count doesn't |
| Score matrix | $\pred{S} = \pred{Q}\pred{K}^\top \in \mathbb{R}^{n \times n}$ | Every query vs every key |
| One score | $\pred{S_{ij}} = \pred{q_i}^\top\pred{k_j} = \sum_\ell \pred{q_{i\ell}}\pred{k_{j\ell}}$ | A sum of $\hyper{d_k}$ products |
| Dot-product variance | $\operatorname{Var}(\pred{q}^\top\pred{k}) \approx \hyper{d_k}$ | Scores grow like $\sqrt{\hyper{d_k}}$ |
| Scaled scores | $\pred{S_{scaled}} = \pred{Q}\pred{K}^\top / \sqrt{\hyper{d_k}}$ | Back to unit variance |
| Attention weights | $\pred{A_{ij}} = \exp(\pred{S_{ij}}/\sqrt{\hyper{d_k}}) / \sum_m \exp(\pred{S_{im}}/\sqrt{\hyper{d_k}})$ | Row-wise softmax |
| Row constraint | $\sum_j \pred{A_{ij}} = 1$, $0 \le \pred{A_{ij}} \le 1$ | Each row is a distribution |
| Softmax Jacobian | $\partial \pred{A_i} / \partial \pred{s_j} = \pred{A_i}(\delta_{ij} - \pred{A_j})$ | Near zero when saturated |
| Contextual output | $\pred{Z} = \pred{A}\pred{V}$, $\pred{z_i} = \sum_j \pred{A_{ij}}\pred{v_j}$ | Weighted average of values |
| Full formula | $\operatorname{softmax}(\pred{Q}\pred{K}^\top/\sqrt{\hyper{d_k}})\,\pred{V}$ | Match, scale, normalize, mix |
| Multi-head (preview) | $\operatorname{Concat}(\text{head}_1, \ldots, \text{head}_h)\param{W_O}$ | Several heads in parallel |

## Concept-Check Answers (from the notebook)

### From encoder–decoder attention to self-attention

<details>
<summary><strong>Q1. In encoder–decoder attention, where do Q, K, V come from?</strong></summary>

Queries come from the **decoder** (its current state), keys and values from the **encoder** outputs. The decoder attends across to the source sentence.
</details>

<details>
<summary><strong>Q2. In self-attention, why do we use the same matrix X for Q, K, V?</strong></summary>

Because the goal is for tokens of one sequence to attend to each other. Each token must both ask (query) and be found and read (key, value) within the same sentence, so all three are projections of the same $X$, with different learned weights.
</details>

<details>
<summary><strong>Q3. What is the limitation of treating embeddings independently?</strong></summary>

Each token's vector is fixed regardless of its neighbours, so "good" has the same representation in "not good" and "very good". The model can't capture negation, coreference or any context-dependent meaning.
</details>

<details>
<summary><strong>Q4. In "The movie was not good", why must "good" attend to "not"?</strong></summary>

"not" reverses the polarity of "good". Unless information from "not" flows into the representation of "good", the phrase reads as positive.
</details>

### The input matrix X

<details>
<summary><strong>Q1. What does each row of matrix X represent?</strong></summary>

The embedding of one token: row $i$ is $x_i \in \mathbb{R}^{d_{model}}$, the vector for the $i$-th word.
</details>

<details>
<summary><strong>Q2. Why do we use a matrix instead of processing tokens one by one?</strong></summary>

So every attention step is one matrix multiplication over all tokens at once: vectorized, parallel on GPUs, and easy to batch.
</details>

<details>
<summary><strong>Q3. What is the shape of X in our example?</strong></summary>

$5 \times 4$: five tokens with $d_{model} = 4$.
</details>

<details>
<summary><strong>Q4. Why is the dot product between two token embeddings insufficient for context modeling?</strong></summary>

It is static: $x_{good} \cdot x_{not} = 1.514$ in every sentence. It isn't normalized, isn't learned for the task, ignores sentence structure, and can only measure raw embedding similarity rather than a learned "who should look at whom".
</details>

<details>
<summary><strong>Q5. What is missing in X that self-attention will introduce?</strong></summary>

Interaction between tokens: context awareness and dependency modeling. Each row will be updated using information from the other rows.
</details>

### Query, Key, Value projections

<details>
<summary><strong>Q1. Why do we create Q, K and V instead of directly using X?</strong></summary>

Each token plays three different roles: searching, being matched and contributing content. Separate learned projections let each role specialize, and they make the score matrix asymmetric and learnable instead of raw embedding similarity.
</details>

<details>
<summary><strong>Q2. What is the shape of W_Q in our example?</strong></summary>

$4 \times 3$ ($d_{model} \times d_k$).
</details>

<details>
<summary><strong>Q3. What is the shape of Q?</strong></summary>

$5 \times 3$ ($n \times d_k$): one 3-number query per token.
</details>

<details>
<summary><strong>Q4. Why are Q, K and V different even though they come from the same X?</strong></summary>

They use different weight matrices, $W_Q \ne W_K \ne W_V$, so the same row of $X$ is projected three different ways: $q_{good}$ = [−0.475, 0.566, −2.104] but $k_{good}$ = [0.594, 1.121, −1.694].
</details>

<details>
<summary><strong>Q5. What does the Value matrix V represent?</strong></summary>

The information each token passes on when attended to. The output is a weighted average of value rows; values never affect the weights.
</details>

### Shapes of Q, K and V

<details>
<summary><strong>Q1. Why do Q, K and V have the same number of rows as X?</strong></summary>

The projections act on each row separately (they change the feature width, not the token count), so row $i$ of each is still token $i$.
</details>

<details>
<summary><strong>Q2. Why must q_i and k_j have the same dimension?</strong></summary>

They are combined by a dot product, $q_i^\top k_j$, which is only defined for vectors of equal length $d_k$.
</details>

<details>
<summary><strong>Q3. What is the shape of each query vector in our example?</strong></summary>

$(3,)$: $q_i \in \mathbb{R}^3$.
</details>

<details>
<summary><strong>Q4. What is the shape of each key vector in our example?</strong></summary>

$(3,)$: $k_i \in \mathbb{R}^3$.
</details>

<details>
<summary><strong>Q5. What is the shape of each value vector in our example?</strong></summary>

$(3,)$: $v_i \in \mathbb{R}^3$, because $d_v = d_k = 3$ here.
</details>

<details>
<summary><strong>Q6. Why are we using d_v = d_k in this lecture?</strong></summary>

For simplicity. Values are only averaged, never dot-producted, so $d_v$ could differ; it just sets the output width. The original Transformer also uses $d_v = d_k$ (64 per head).
</details>

### The attention score matrix QKᵀ

<details>
<summary><strong>Q1. What does S_ij represent?</strong></summary>

The dot product of token $i$'s query and token $j$'s key: how strongly token $i$ is drawn to token $j$, before normalization.
</details>

<details>
<summary><strong>Q2. Why is S a square matrix?</strong></summary>

Because the same $n$ tokens are both the queries (rows) and the keys (columns): $(n \times d_k)(d_k \times n) = n \times n$.
</details>

<details>
<summary><strong>Q3. What is the shape of S in our example?</strong></summary>

$5 \times 5$.
</details>

<details>
<summary><strong>Q4. What does each row of S represent?</strong></summary>

One query token compared with every key: $[q_i \cdot k_1, \ldots, q_i \cdot k_n]$, its raw relevance scores for all tokens.
</details>

<details>
<summary><strong>Q5. Why can we not directly use S as attention weights?</strong></summary>

The scores are unbounded, can be negative and don't sum to 1 (they range from −8.73 to 22.47 here), so they can't serve as averaging weights. They also grow with $d_k$. We scale them and apply a row-wise softmax first.
</details>

### Meaning of S_ij

<details>
<summary><strong>Q1. What does S_ij represent?</strong></summary>

$q_i^\top k_j = \sum_\ell q_{i\ell} k_{j\ell}$: the raw compatibility of token $i$'s query with token $j$'s key.
</details>

<details>
<summary><strong>Q2. In S_ij, which token is asking and which token is being compared?</strong></summary>

Token $i$ (the row, via its query) is asking; token $j$ (the column, via its key) is being compared. Swapping them uses different vectors: $S_{good,not} = -8.66$ but $S_{not,good} = 4.21$.
</details>

<details>
<summary><strong>Q3. Why is S_good,not important in "The movie was not good"?</strong></summary>

It measures how much "good" wants information from "not", the word that flips its meaning. In a trained model it should be high; in this random run it is −8.66, the lowest in the row.
</details>

<details>
<summary><strong>Q4. What does one full row of S represent?</strong></summary>

How one token scores every token in the sentence: for "good", [−1.605, −1.493, 3.856, −8.661, 3.918] answers "which tokens should contribute when we update 'good'?", before normalization.
</details>

<details>
<summary><strong>Q5. Why are raw scores not final attention weights?</strong></summary>

They may be negative or very large and don't sum to 1. They must be scaled by $\sqrt{d_k}$ and normalized row-wise by softmax.
</details>

### Why scale by √d_k

<details>
<summary><strong>Q1. Why can raw dot products become large when d_k increases?</strong></summary>

A dot product sums $d_k$ products. With independent unit-variance entries the variances add, so $\operatorname{Var}(q \cdot k) \approx d_k$ and the standard deviation grows like $\sqrt{d_k}$ (measured 1.38 at $d_k = 2$, 11.02 at 128).
</details>

<details>
<summary><strong>Q2. What happens when large scores are passed into softmax?</strong></summary>

Softmax becomes very sharp, nearly one-hot: one token gets almost all the weight, the others almost zero, and the gradients for most tokens become tiny.
</details>

<details>
<summary><strong>Q3. Why do we divide by √d_k instead of d_k?</strong></summary>

The standard deviation grows like $\sqrt{d_k}$, so dividing by $\sqrt{d_k}$ restores unit variance. Dividing by $d_k$ would leave variance $1/d_k$, shrinking scores toward 0 and making attention nearly uniform.
</details>

<details>
<summary><strong>Q4. Does scaling change the shape of the score matrix?</strong></summary>

No. It divides every entry by the same constant; the shape (and the ranking within each row) is unchanged.
</details>

<details>
<summary><strong>Q5. What is the shape of S_scaled in our example?</strong></summary>

$5 \times 5$. (Its standard deviation drops from 6.12 to 3.53.)
</details>

### The softmax sharpness problem

<details>
<summary><strong>Q1. What does softmax do to a vector of scores?</strong></summary>

Exponentiates each score and divides by the total, giving non-negative weights that sum to 1, larger for larger scores, with ratios exponential in the score gaps.
</details>

<details>
<summary><strong>Q2. Why does softmax become sharp when scores are large?</strong></summary>

Weight ratios are $e^{\text{gap}}$, so larger scores mean larger gaps and exponentially larger ratios: [10, 20, 5, 15, 2] gives 0.993 to the top score.
</details>

<details>
<summary><strong>Q3. What does it mean when attention becomes almost one-hot?</strong></summary>

The token copies essentially one other token's value and ignores the rest of the context, as with the "not" row here (0.99997 on itself).
</details>

<details>
<summary><strong>Q4. Why is one-hot attention too early in training problematic?</strong></summary>

The early scores are essentially random, so a one-hot choice is arbitrary. And a saturated softmax has near-zero gradients ($A_i(\delta_{ij} - A_j) \approx 0$; the largest entry fell from 0.183 to 0.0067 in the notebook's examples), so the model gets almost no signal to correct it. Learning becomes slow and unstable.
</details>

<details>
<summary><strong>Q5. How does scaling by √d_k help?</strong></summary>

It keeps scores at unit variance, so softmax stays in its soft range where all tokens get some weight and gradients flow. In the notebook's "good" row the small weights rise from about 0.002 to 0.02.
</details>

### The scaled dot-product attention formula

<details>
<summary><strong>Q1. Write the full scaled dot-product attention formula.</strong></summary>

$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\dfrac{QK^\top}{\sqrt{d_k}}\right)V$, with softmax applied row-wise.
</details>

<details>
<summary><strong>Q2. What is the shape of A in our example?</strong></summary>

$5 \times 5$.
</details>

<details>
<summary><strong>Q3. Why does each row of A sum to 1?</strong></summary>

Softmax is applied to each row separately and always produces non-negative values summing to 1. Each row is one token's attention distribution.
</details>

<details>
<summary><strong>Q4. What does Z = AV represent?</strong></summary>

The contextual token representations: row $i$ is the weighted average of all value vectors using token $i$'s attention weights.
</details>

<details>
<summary><strong>Q5. Why is z_i called a contextual token representation?</strong></summary>

It is built from the value vectors of every token in the sentence, weighted by relevance, so it depends on the context rather than only on token $i$.
</details>

### The attention weight matrix and Z = AV

<details>
<summary><strong>Q1. What is the formula for the contextual output matrix?</strong></summary>

$Z = AV = \operatorname{softmax}(QK^\top/\sqrt{d_k})\,V$.
</details>

<details>
<summary><strong>Q2. What is the shape of Z in our example?</strong></summary>

$5 \times 3$: $(5 \times 5)(5 \times 3)$.
</details>

<details>
<summary><strong>Q3. What does each row z_i represent?</strong></summary>

The updated, context-aware vector of token $i$: $z_i = \sum_j A_{ij} v_j$. For "good" it is [0.7465, −0.0026, −0.7641].
</details>

<details>
<summary><strong>Q4. Why is z_i called contextual?</strong></summary>

It mixes information from all tokens in the sentence according to attention, so the same word gets a different $z$ in a different sentence.
</details>

<details>
<summary><strong>Q5. What is the role of V in computing Z?</strong></summary>

$V$ supplies the content that gets mixed; $A$ only decides the proportions. Values never influence the weights.
</details>

<details>
<summary><strong>Q6. Why should we not overinterpret the attention weights in this toy example?</strong></summary>

The embeddings and projection matrices are random and untrained, so the pattern ("good" gives "not" 0.0003, "The" gives "not" 0.808) reflects arbitrary numbers, not language. The example teaches mechanism and shapes only.
</details>

### Attention heatmaps

<details>
<summary><strong>Q1. What does a row in the attention heatmap represent?</strong></summary>

One query token's attention distribution: how much it takes from each key/value token.
</details>

<details>
<summary><strong>Q2. What does a column in the attention heatmap represent?</strong></summary>

One key/value token: how much attention it receives from each query token.
</details>

<details>
<summary><strong>Q3. Why is the attention heatmap easier to interpret than the raw score heatmap?</strong></summary>

Every entry is a weight between 0 and 1 and every row sums to 1, so all rows share one scale and read as proportions. Raw scores are unbounded, can be negative and differ wildly in scale between rows (−8.7 to 22.5 in the "not" row).
</details>

<details>
<summary><strong>Q4. Why does every row of the attention heatmap sum to 1?</strong></summary>

Because it shows $A$, the row-wise softmax of the scaled scores.
</details>

<details>
<summary><strong>Q5. If "good" does not attend strongly to "not" in this toy example, should we conclude self-attention failed?</strong></summary>

No. The weights are random and untrained. Training learns $W_Q$ and $W_K$ so that useful pairs score high (and multiple heads capture several relations). The toy run only shows the computation.
</details>

<details>
<summary><strong>Q6. What is the main purpose of visualizing A?</strong></summary>

To see how each token distributes its attention: to understand and debug the mechanism (shapes, row sums, saturation such as the "not" row at 1.0). It isn't proof of what the model "understands".
</details>

### Original embedding vs attention-updated embedding

<details>
<summary><strong>Q1. What is the difference between x_i and z_i?</strong></summary>

$x_i$ is the static input embedding of token $i$, before any interaction ($\mathbb{R}^{d_{model}}$). $z_i$ is its representation after self-attention, a weighted mix of all tokens' values ($\mathbb{R}^{d_v}$).
</details>

<details>
<summary><strong>Q2. Why can we not directly subtract x_good and z_good in this example?</strong></summary>

They have different sizes: $x_{good} \in \mathbb{R}^4$ ($d_{model}$) and $z_{good} \in \mathbb{R}^3$ ($d_k$). Element-wise subtraction is undefined, and even the shared coordinates mean different things.
</details>

<details>
<summary><strong>Q3. Why is comparing v_good with z_good more technically valid?</strong></summary>

Both live in the same 3-dimensional value space: $v_{good}$ = [0.6715, −0.2694, −0.7881] is what "good" contributes alone, and $z_{good}$ = [0.7465, −0.0026, −0.7641] is that space after mixing in the other tokens.
</details>

<details>
<summary><strong>Q4. What does z_good contain that v_good alone does not?</strong></summary>

Information from the other tokens' values, in proportion to "good"'s attention: here mainly "was" (0.471), plus small amounts of "The", "movie" and "not".
</details>

<details>
<summary><strong>Q5. Why should we not attach semantic meaning to the direction of change in this toy example?</strong></summary>

The embeddings and projections are random, so the direction from $v_{good}$ to $z_{good}$ is arbitrary. Only the mechanism, $Z = AV$, carries over to real models.
</details>
