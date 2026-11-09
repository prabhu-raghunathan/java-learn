---
title: "Multi-Head Attention: Many Attention Patterns in Parallel"
sidebar_position: 3
---

import {MultiHeadExplorer, HeadSplitCalculator} from '@site/src/components/Widgets';

:::tip[TL;DR (30-Second Interview Pitch)]
One attention head gives every token **one** softmax distribution over the sentence, so it can follow only one kind of relationship at a time; if it needs two, it has to **average** them. **Multi-head attention** runs $h$ scaled dot-product attentions in parallel, each with its **own** $W_Q^{(r)}, W_K^{(r)}, W_V^{(r)}$ that project the same input into a smaller subspace of size $d_k = d_{model}/h$. Each head produces an $n \times d_k$ output; the outputs are **concatenated** back to $n \times d_{model}$ and mixed by a learned **output projection** $W^O$: $\operatorname{MultiHead}(Q,K,V) = \operatorname{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$. Because each head is $h$ times narrower, the cost and parameter count equal one full-width head ($4d_{model}^2$ weights: 1,048,576 for the paper's $d_{model} = 512$, $h = 8$, $d_k = 64$). In a tiny trained model, one head learns "look at the previous token" (weight 0.996) and the other "look at the first token" (0.999); a single head has to split about 50/50.
:::

<FormulaKey />

:::info[Where this page sits]
**Before:** [Full Self-Attention: Scaled Dot-Product Attention Step by Step](./02-full-self-attention-scaled-dot-product.md) computed **one** head end to end.
**This page:** notebook *16 Multi-Head Attention*, cross-checked against **"Attention Is All You Need"** (Vaswani et al., 2017), §3.2.2, Figure 2 and Table 3: why one head is not enough, per-head projections and shapes, attention per head, comparing heads, concatenation, the output projection $W^O$, the full formula, single vs multi-head, a NumPy function, and where it sits in the encoder block. In the paper's **Figure 1** this is every orange box: "Multi-Head Attention" in the encoder, and "Masked Multi-Head Attention" and "Multi-Head Attention" in the decoder.
**Next:** [The Transformer: Figure 1 Box by Box](./04-transformer-architecture-figure-1.md) assembles every block of the architecture.
:::

## 🧭 Start Here

On the previous page we built one complete self-attention computation: queries, keys and values, scores, $\sqrt{d_k}$ scaling, a row-wise softmax and a weighted sum of values. That is what the paper calls **one attention head**. It has a structural limitation. For each token it produces exactly **one** probability distribution over the sentence. Language has many relationships at once: in "The movie was not good", "good" relates to "not" through negation, to "movie" as the thing being judged and to "was" grammatically. One softmax row has to compromise between all of them, and because softmax tends to concentrate weight, it usually either picks one relationship or blurs several together.

The paper's fix is simple: don't build one big attention, build several smaller ones side by side. Each **head** gets its own learned projections, so it looks at the same sentence through a different lens and can produce a different attention pattern. Their outputs are glued together and passed through one more learned matrix that mixes them. The paper puts the motivation in one sentence: multi-head attention lets the model "jointly attend to information from different representation subspaces at different positions", whereas with a single head, averaging inhibits this. Because each head is narrower, the total cost doesn't grow.

The notebook uses the same five-token sentence, now with $d_{model} = 4$ and $h = 2$ heads of size $d_k = 2$, and random (untrained) weights from `np.random.seed(42)`. As before, the random weights show the **mechanism and the shapes**, not linguistics. To show what heads do when trained, this page adds a small experiment: a model trained on a toy task needing two different "where to look" rules, where the two heads split the jobs between them. The chain of ideas:

1. One head = one attention pattern per token (and averaging when it needs more).
2. Multi-head attention = $h$ attentions in parallel.
3. Each head has its own $W_Q^{(r)}, W_K^{(r)}, W_V^{(r)}$.
4. Shapes: $d_k = d_{model}/h$.
5. Computing $Q^{(r)}, K^{(r)}, V^{(r)}$ for each head.
6. Scaled dot-product attention inside each head.
7. Comparing the heads' attention patterns.
8. Concatenation: heads side by side.
9. The output projection $W^O$: heads mixed.
10. The full formula.
11. Single-head vs multi-head, and trained heads specializing.
12. Packaging it as a function, and how libraries really do it.
13. Where it sits in the encoder block.

| New concept | One-line meaning |
|---|---|
| **Attention head** | One complete scaled dot-product attention with its own $W_Q, W_K, W_V$ |
| **Multi-head attention (MHA)** | $h$ heads in parallel on the same input, then Concat and $W^O$ |
| **$h$** | Number of heads (2 in the notebook; 8 in the paper's base model) |
| **$d_k = d_v = d_{model}/h$** | Width of each head's queries/keys and values (2 here; 64 in the paper) |
| **Representation subspace** | The smaller space a head projects into; different heads, different subspaces |
| **$\text{head}_r$** | Head $r$'s output, $n \times d_v$ |
| **Concat** | Placing head outputs side by side: $n \times h d_v$ |
| **Output projection $W^O$** | Learned $h d_v \times d_{model}$ matrix that mixes the heads |
| **Head specialization** | Trained heads learning different relations (previous token, first token, syntax…) |
| **Fused projection** | One big $d_{model} \times d_{model}$ matrix per Q/K/V, reshaped into heads (how libraries do it) |

:::analogy[A panel of specialist readers]
Give one paragraph to a panel of readers, each with their own highlighter colour and their own question: one marks negations, one marks who did what, one marks time words. Each reads the whole paragraph but highlights different things. At the end, an editor takes all the highlighted copies, lays them side by side and writes one combined summary. The readers are the heads, laying the copies side by side is Concat, and the editor is $W^O$.
:::

## 1. Why One Attention Head Is Not Enough

Recall single-head attention. The input $X \in \mathbb{R}^{n \times d_{model}}$ is projected into $Q = XW_Q$, $K = XW_K$, $V = XW_V$ with $W \in \mathbb{R}^{d_{model} \times d_k}$; the scores $QK^\top$ are $n \times n$; after scaling and softmax we get the attention matrix $A$, and the output is $Z = AV$. The important word is **one**: one $A$, so one pattern of who-looks-at-whom per token.

In the notebook's single head (5 tokens, $d_{model} = 4$, $d_k = 2$), row "good" of $A$ is [0.444, 0.423, 0.010, 0.118, 0.004]: "good" looks mostly at "The" and "movie". Whatever this head learns, it has to fit every relationship "good" takes part in into that single row. A sentence needs many at once: negation ("not" → "good"), what is being described ("movie" ← "good"), subject–verb agreement, nearby phrase structure, long-distance links. If one relationship dominates, the others are lost; if the row spreads weight over several positions, the output is an **average** of their values, and the separate signals get blurred together.

The paper names this exactly: with a single attention head, averaging inhibits attending to different representation subspaces at different positions. The cure is to let the model compute **several** attention matrices, each free to specialize. That's chapter 2.

:::tip[🎯 Interview angle]
**"What's the limitation of single-head attention?"** It produces one softmax distribution per token, so one attention pattern. If a token needs information from several places for different reasons, the single head either picks one or averages their values, blurring the signals. Multi-head attention gives several independent distributions per token, so different relations can be attended to cleanly in parallel.
:::

## 2. The Big Picture: Several Attentions in Parallel

Multi-head attention changes the question from "what is the single best way for tokens to attend to each other?" to "can several heads learn **different kinds** of relationships at the same time?" For "The movie was not good", one head might learn the negation link "not" → "good", another the phrase structure "movie" ↔ "good", another short-range neighbours. They all run at once.

Every head receives the **same input** $X$, but each has its **own** projection matrices, so each computes its own $Q^{(r)}, K^{(r)}, V^{(r)}$ and its own attention matrix $A^{(r)}$. Head $r$ computes $\text{head}_r = \operatorname{Attention}(Q^{(r)}, K^{(r)}, V^{(r)})$, an $n \times d_k$ output. With two heads in the notebook, we get two $5 \times 2$ outputs, which are later concatenated (chapter 8) and mixed (chapter 9).

The paper's Figure 2 (right) draws exactly this. Read it bottom to top: V, K and Q each pass through a **Linear** layer (the projection; drawn as a stack because there are $h$ of them, one per head); the projected versions go into **Scaled Dot-Product Attention**, also stacked $h$ times; the $h$ results are combined by **Concat**; and a final **Linear** ($W^O$) produces the output.

![Multi-Head Attention (paper Figure 2, right)](/img/deep-learning/diagrams/paper-figure2-multi-head.svg)

*How to read it:* the three bottom inputs are V, K and Q (in self-attention all three are the same $X$). Each stacked "Linear" is $h$ separate projections $W_i^V, W_i^K, W_i^Q$, one per head. The stacked "Scaled Dot-Product Attention" box is $h$ independent attentions (the "h" beside it). "Concat" joins the $h$ outputs, and the top "Linear" is $W^O$. Labels follow the paper's figure word for word.

:::tip[🎯 Interview angle]
**"Describe multi-head attention in one breath."** Project the same queries, keys and values $h$ times with different learned matrices into smaller $d_k = d_{model}/h$ subspaces, run scaled dot-product attention independently in each, concatenate the $h$ outputs, and apply a final learned linear layer $W^O$. Each head can learn a different attention pattern, and the total cost is about the same as one full-width head.
:::

## 3. Separate Projection Matrices for Each Head

Why do the heads need different matrices? Because if every head used the same $W_Q$, $W_K$ and $W_V$, every head would compute exactly the same $Q$, $K$, $V$, the same attention matrix and the same output. Concatenating $h$ identical copies adds no information (block 5 in the implementation shows the concatenated matrix has rank 2 instead of 4 when the two heads are identical). The projections are what let heads specialize; they are not a technical detail.

Each head $r$ has $W_Q^{(r)}, W_K^{(r)}, W_V^{(r)} \in \mathbb{R}^{d_{model} \times d_k}$, and computes $Q^{(r)} = XW_Q^{(r)}$, $K^{(r)} = XW_K^{(r)}$, $V^{(r)} = XW_V^{(r)}$. In the notebook, $X$ is $5 \times 4$ and each $W$ is $4 \times 2$, so each head's $Q$, $K$, $V$ are $5 \times 2$. The same row of $X$ comes out differently in each head: "The" gets the query [−1.387, 0.729] in head 1 and [0.045, −2.232] in head 2. The notebook checks with `np.allclose` that $Q^{(1)} \ne Q^{(2)}$, $K^{(1)} \ne K^{(2)}$ and $V^{(1)} \ne V^{(2)}$.

In plain language: $W_Q^{(r)}$ controls **what head $r$ asks**, $W_K^{(r)}$ controls **how tokens can be found** by head $r$, and $W_V^{(r)}$ controls **what information head $r$ passes on**. A head that tracks negation needs queries that look for modifiers, keys that advertise "I am a negation", and values that carry "flip the polarity"; a head that tracks the subject needs entirely different ones.

<Formula name="Per-head projections" tag="Different lens per head" words="Each head multiplies the same input by its own three learned matrices, giving its own queries, keys and values in a smaller subspace." remember="Same X, h different lenses.">

$$
\begin{aligned}
&\pred{Q^{(r)}} = \data{X}\param{W_Q^{(r)}}, \quad \pred{K^{(r)}} = \data{X}\param{W_K^{(r)}}, \\
&\pred{V^{(r)}} = \data{X}\param{W_V^{(r)}}, \qquad r = 1, \ldots, \hyper{h} \\
&\param{W_Q^{(r)}}, \param{W_K^{(r)}} \in \mathbb{R}^{\hyper{d_{model}} \times \hyper{d_k}}, \quad \param{W_V^{(r)}} \in \mathbb{R}^{\hyper{d_{model}} \times \hyper{d_v}}
\end{aligned}
$$

- $r$: head index; every head has its own three matrices
- $\pred{Q^{(r)}}, \pred{K^{(r)}} \in \mathbb{R}^{n \times \hyper{d_k}}$, $\pred{V^{(r)}} \in \mathbb{R}^{n \times \hyper{d_v}}$

**Example:** row "The" of $\pred{Q^{(1)}}$ = [−1.387, 0.729] but of $\pred{Q^{(2)}}$ = [0.045, −2.232]: same token, different heads, different queries.

</Formula>

:::tip[🎯 Interview angle]
**"Why does each head need its own W_Q, W_K, W_V?"** So that each head projects the input into a different subspace and can compute a different attention pattern. With shared matrices all heads would be identical copies, and concatenating copies adds nothing; the multi-head layer would collapse to one head.
:::

## 4. The Shape Setup: d_k = d_model / h

Most multi-head confusion is about shapes, so fix them first. $n$ is the number of tokens (5), $d_{model}$ the embedding width (4), $h$ the number of heads (2), and $d_k$ the width inside each head. The standard choice, and the paper's, is $d_k = d_v = d_{model}/h$: here $4 / 2 = 2$; in the paper $512 / 8 = 64$.

The reason is the concatenation at the end. Each head outputs $n \times d_v$; $h$ of them side by side give $n \times h d_v$, and with $d_v = d_{model}/h$ that is exactly $n \times d_{model}$, the shape we started with. So the multi-head layer can be dropped into the network, followed by a residual connection ($X + \text{output}$ needs equal shapes), and stacked. Each head works in a **smaller** subspace, but together they cover the full model width.

This choice also keeps the **cost fixed**. A single head of full width $d_{model}$ needs $3d_{model}^2$ projection weights plus $d_{model}^2$ for the output; $h$ heads of width $d_{model}/h$ need $3 \cdot h \cdot d_{model} \cdot (d_{model}/h) = 3d_{model}^2$ plus the same $d_{model}^2$. The paper says it directly: due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality. You get $h$ attention patterns for the price of one.

| Symbol | Meaning | Notebook | Paper (base) |
|---|---|---|---|
| $n$ | Tokens | 5 | sentence length |
| $d_{model}$ | Embedding / layer width | 4 | 512 |
| $h$ | Heads | 2 | 8 |
| $d_k = d_v$ | Per-head width, $d_{model}/h$ | 2 | 64 |
| $W_Q^{(r)}, W_K^{(r)}, W_V^{(r)}$ | Per-head projections | 4 × 2 | 512 × 64 |
| $\text{head}_r$ | Per-head output | 5 × 2 | $n$ × 64 |
| Concat $H$ | All heads side by side | 5 × 4 | $n$ × 512 |
| $W^O$ | Output projection, $h d_v \times d_{model}$ | 4 × 4 | 512 × 512 |
| MHA weights (no biases) | $4 d_{model}^2$ | 64 | 1,048,576 |

![Multi-head attention shapes in the notebook example](/img/deep-learning/diagrams/mha-toy-shapes.svg)

*How to read it:* $X$ ($5 \times 4$) goes to both heads. Each head has its own three $4 \times 2$ matrices, its own $5 \times 5$ attention matrix and a $5 \times 2$ output. Concat joins them into $H$ ($5 \times 4$), and $W_O$ ($4 \times 4$) gives $Z$, the same shape as $X$.

**Try it yourself.** Pick a configuration or drag the sliders. Note that the total stays $4d_{model}^2$ whatever $h$ is, and that $d_{model}$ must be divisible by $h$. At $d_{model} = 512$ the widget also shows the paper's measured result for that number of heads.

<HeadSplitCalculator />

:::tip[🎯 Interview angle]
**"Why is d_k set to d_model / h? Does adding heads add parameters?"** So that the $h$ concatenated head outputs have width $h \cdot d_k = d_{model}$, matching the input for the residual connection and for stacking. It also keeps the parameters and compute fixed: $h$ heads of width $d_{model}/h$ cost the same $4d_{model}^2$ weights as one full-width head. More heads means more, narrower attention patterns, not a bigger layer.
:::

## 5. Computing Q, K and V for Each Head

With the shapes fixed, each head computes its own three matrices from the same $X$. Head 1 uses $W_Q^{(1)}, W_K^{(1)}, W_V^{(1)}$; head 2 uses $W_Q^{(2)}, W_K^{(2)}, W_V^{(2)}$. Shape-wise $(5 \times 4)(4 \times 2) = 5 \times 2$ for every one of the six products.

The point to absorb is **where the difference comes from**. The input didn't change; the projection matrices did. Before any attention is computed, each head already sees the sentence through its own learned lens. Head 1's keys put "good" at [−0.730, 3.562], head 2's at [1.304, −1.424]. So the same query–key comparison in the two heads compares different numbers and can rank the tokens differently.

Different **value** projections matter as much as different query/key projections. The values are what actually gets passed forward after weighting, so separate $W_V^{(r)}$ let each head extract a different kind of information from the tokens it attends to, not just look at different tokens.

| Head 1 | Q⁽¹⁾ | K⁽¹⁾ | V⁽¹⁾ |
|---|---|---|---|
| The | [−1.387, 0.729] | [1.029, −2.945] | [0.590, −1.348] |
| movie | [−2.102, 0.850] | [0.891, −2.973] | [1.340, 0.452] |
| was | [0.137, −0.893] | [−0.421, 2.201] | [−1.107, 1.088] |
| not | [1.812, 2.269] | [0.567, −1.104] | [1.383, −0.977] |
| good | [0.657, −0.850] | [−0.730, 3.562] | [−1.136, 2.316] |

:::tip[🎯 Interview angle]
**"If every head gets the same input, why do they compute different things?"** Because the projections differ: $Q^{(r)} = XW_Q^{(r)}$ with a different learned $W$ per head. The same token gets a different query, key and value in every head, so each head compares and mixes different vectors. Separate value projections also let each head pass on a different kind of information.
:::

## 6. Scaled Dot-Product Attention Inside Each Head

Each head now runs exactly the attention from the previous page, on its own matrices, completely independently. The heads share no scores or weights. For head $r$: scores $S^{(r)} = Q^{(r)}K^{(r)\top}$ ($5 \times 2$ times $2 \times 5$ gives $5 \times 5$), scaled by $\sqrt{d_k} = \sqrt{2}$ (the per-head width, not $d_{model}$), row-wise softmax into $A^{(r)}$ ($5 \times 5$, every row sums to 1), then $\text{head}_r = A^{(r)}V^{(r)}$ ($5 \times 5$ times $5 \times 2$ gives $5 \times 2$).

The paper's Figure 2 (left) draws this inner computation as a stack of operations, read bottom to top: **MatMul** of Q and K, **Scale**, **Mask (opt.)**, **SoftMax**, and a second **MatMul** with V. The optional mask sets illegal positions to $-\infty$ before the softmax; the encoder doesn't need it, but the decoder's "Masked Multi-Head Attention" in Figure 1 uses it so a position can't look at later positions. The next page covers it.

![Scaled Dot-Product Attention (paper Figure 2, left)](/img/deep-learning/diagrams/paper-figure2-scaled-dot-product.svg)

*How to read it:* Q and K enter the first MatMul ($QK^\top$), the result is scaled by $1/\sqrt{d_k}$, optionally masked (illegal links set to $-\infty$), and softmaxed row by row; the second MatMul multiplies the weights by V. Every head in Figure 2 (right) runs this exact box. Labels follow the paper word for word.

<Formula name="One attention head" tag="head_r" words="Each head is ordinary scaled dot-product attention on its own projected queries, keys and values, scaled by the square root of the per-head width." remember="Each head: its own full attention.">

$$
\begin{aligned}
\pred{\text{head}_r} &= \operatorname{softmax}\!\left(\frac{\pred{Q^{(r)}}\pred{K^{(r)}}^\top}{\sqrt{\hyper{d_k}}}\right)\pred{V^{(r)}} \\
&= \operatorname{softmax}\!\left(\frac{\data{X}\param{W_Q^{(r)}}(\data{X}\param{W_K^{(r)}})^\top}{\sqrt{\hyper{d_k}}}\right)\data{X}\param{W_V^{(r)}}
\end{aligned}
$$

- $\pred{A^{(r)}} = \operatorname{softmax}(\cdot) \in \mathbb{R}^{n \times n}$, rows sum to 1
- $\pred{\text{head}_r} \in \mathbb{R}^{n \times \hyper{d_v}}$; scale by $\sqrt{\hyper{d_k}}$ of the head (here $\sqrt 2$), not $\sqrt{\hyper{d_{model}}}$

**Example:** head 1 output for "The" = [−1.063, 1.911]; head 2 output for "The" = [0.175, −1.367].

</Formula>

:::tip[🎯 Interview angle]
**"Inside one head, what happens, and what is d_k in the scaling?"** Exactly single-head scaled dot-product attention: $QK^\top$, divide by $\sqrt{d_k}$, optional mask, row-wise softmax, multiply by $V$. The $d_k$ is the per-head width ($d_{model}/h$, 64 in the paper), because that's the length of the vectors being dot-producted. Heads share nothing until Concat.
:::

## 7. Visualizing Attention Patterns Across Heads

Each head has its own $5 \times 5$ attention matrix, and putting them side by side shows whether the heads look at the sentence differently. Read them as before: **row** = query token (how it spreads its attention), **column** = key token (how much it is attended to), and $A_{ij}^{(r)}$ = how much token $i$ attends to token $j$ **in head $r$**.

They are very different. The largest gap between the two matrices is 0.807, in row "not": head 1 sends 0.854 of "not"'s attention to "good", head 2 sends only 0.047 there. Row "good" is [0.444, 0.423, 0.010, 0.118, 0.004] in head 1 (split between "The" and "movie") and [0.923, 0.021, 0.053, 0.001, 0.003] in head 2 (almost all on "The"). The notebook plots both heatmaps and their difference $A^{(1)} - A^{(2)}$ to show where they disagree.

| Query token | Head 1 attends most to | Head 2 attends most to |
|---|---|---|
| The | good (0.712) | not (0.516) |
| movie | good (0.775) | good (0.774) |
| was | movie (0.427) | The (0.771) |
| not | good (0.854) | The (0.566) |
| good | The (0.444) | The (0.923) |

In a trained Transformer such differences correspond to real specialization. The paper's appendix shows heads in layer 5 of 6 following long-distance dependencies (many heads link "making" to "more difficult"), heads apparently doing anaphora resolution (linking "its" to "Law" and "application"), and heads tracking sentence structure. In this notebook the weights are random, so the two patterns are different but **meaningless**. Chapter 11 trains a small model to show real specialization.

:::danger[Interview Trap: "Different heatmaps prove the heads learned different things"]
With random weights, heads differ just because their random matrices differ. Specialization is a result of **training**, and even trained heads aren't guaranteed to be interpretable. Studies of trained models found many heads can be pruned with little loss, so not every head carries a distinct, meaningful role.
:::

:::tip[🎯 Interview angle]
**"How would you check whether heads learned different things?"** Visualize each head's attention matrix per layer (row = query, column = key) on real inputs, look for consistent patterns (previous token, separators, syntactic heads, coreference), and measure them quantitatively, for example average weight on a relation, or the accuracy drop when a head is masked out. On random weights the patterns differ but mean nothing.
:::

## 8. Concatenating the Heads

Each head has produced its own contextual representation: $\text{head}_1$ and $\text{head}_2$, each $5 \times 2$. The next layer expects one matrix, so the outputs are **concatenated along the feature dimension** (in NumPy, `axis=1`, because the shape is (tokens, features)). The result $H$ is $5 \times (2 + 2) = 5 \times 4$: for every token, the first two numbers come from head 1 and the next two from head 2. In general $H \in \mathbb{R}^{n \times h d_v}$, which equals $n \times d_{model}$ when $d_v = d_{model}/h$.

Concatenation **does not mix** the heads. It only places their outputs side by side, like taping two reports together. Row "The" of $H$ is [−1.063, 1.911, 0.175, −1.367]: exactly head 1's row followed by head 2's row. Nothing yet combines what head 1 found with what head 2 found. That's the job of the next step.

<Formula name="Concatenation" tag="Side by side" words="Place the h head outputs next to each other along the feature axis; each token's row becomes its head-1 features followed by its head-2 features and so on." remember="Concat stacks, it doesn't mix.">

$$
\begin{aligned}
\pred{H} &= \operatorname{Concat}(\pred{\text{head}_1}, \ldots, \pred{\text{head}_h}) \in \mathbb{R}^{n \times \hyper{h}\hyper{d_v}} \\
\hyper{h}\hyper{d_v} &= \hyper{d_{model}} \text{ when } \hyper{d_v} = \hyper{d_{model}}/\hyper{h}
\end{aligned}
$$

- concatenation is along features (`axis=1` for an $n \times d$ matrix; `dim=-1` for batched tensors)
- no learned parameters; nothing is mixed

**Example:** $\pred{H_{The}}$ = [−1.063, 1.911 | 0.175, −1.367]: head 1's two features, then head 2's.

</Formula>

:::tip[🎯 Interview angle]
**"What does Concat do in multi-head attention, and why isn't it the last step?"** It joins the $h$ head outputs along the feature axis into an $n \times h d_v$ matrix, which is $n \times d_{model}$ with the standard split. It has no parameters and doesn't combine the heads: each token's row is just its head outputs placed one after another. The learned $W^O$ afterwards is what mixes information across heads.
:::

## 9. The Output Projection W^O

After Concat, the model needs to **combine** what the heads found: maybe "negation from head 1 plus subject from head 2" is what the next layer needs. The output projection does this: $Z = HW^O$. Each output feature is a learned weighted sum of **all** features from **all** heads, so information from different heads finally interacts.

Shapes: the paper defines $W^O \in \mathbb{R}^{h d_v \times d_{model}}$. With the standard $h d_v = d_{model}$ that is square, $d_{model} \times d_{model}$, which is how the notebook writes it ($4 \times 4$; $512 \times 512$ in the paper). $(5 \times 4)(4 \times 4) = 5 \times 4$, so $Z$ has **exactly the shape of $X$**: one $d_{model}$-wide contextual vector per token, ready for the residual addition $X + Z$ and for the next layer. In the notebook row "The" of $Z$ is [−0.137, 3.136, −7.395, 0.296]; compare it with $H$'s row above: every entry now depends on all four $H$ entries.

$W^O$ is also why the model can afford narrow heads. Each head only sees a $d_k$-wide slice, but $W^O$ maps the combined $h d_v$ features back into the full model space, where the next layers can use any combination of them.

<Formula name="Output projection" tag="Mix the heads" words="Multiply the concatenated head outputs by a learned matrix so every output feature can combine information from every head; the result has the model width again." remember="W^O is the editor.">

$$
\pred{Z} = \pred{H}\param{W^O}, \qquad \param{W^O} \in \mathbb{R}^{\hyper{h}\hyper{d_v} \times \hyper{d_{model}}}, \qquad \pred{Z} \in \mathbb{R}^{n \times \hyper{d_{model}}}
$$

- notebook: $(5 \times 4)(4 \times 4) = 5 \times 4$; paper base: $512 \times 512$
- $\pred{Z}$ has the same shape as $\data{X}$, so $\data{X} + \pred{Z}$ (residual) is defined

**Example:** $\pred{H_{The}}$ = [−1.063, 1.911, 0.175, −1.367] becomes $\pred{Z_{The}}$ = [−0.137, 3.136, −7.395, 0.296].

</Formula>

:::analogy[Translators and an editor]
Two translators each produce a partial translation in their own shorthand. Stapling the two pages together (Concat) gives the editor everything, but nothing has been reconciled yet. The editor (W^O) reads both and writes the final paragraph, taking a phrase from one and a correction from the other. Without the editor, the next reader would get two disconnected drafts.
:::

:::tip[🎯 Interview angle]
**"What's the role of W^O, and what's its shape?"** It mixes the concatenated head outputs so information from different heads can combine, and maps them back to the model width: $W^O \in \mathbb{R}^{h d_v \times d_{model}}$, which is $d_{model} \times d_{model}$ (512 × 512 in the base model) with the standard split. Its output has the input's shape, so residual connections and stacking work.
:::

## 10. The Full Multi-Head Attention Formula

Put together, multi-head attention is five steps: start from $X$; project into $Q^{(r)}, K^{(r)}, V^{(r)}$ per head; run scaled dot-product attention independently in each head; concatenate; apply $W^O$. In other words, multiple parallel self-attention operations followed by concatenation and an output projection. Each head learns a different attention view, and $W^O$ learns how to combine the views.

The paper writes it with general $Q$, $K$, $V$ inputs, not just $X$, because the same block is used three ways in Figure 1: encoder self-attention ($Q = K = V$ = previous encoder layer), decoder masked self-attention ($Q = K = V$ = previous decoder layer, with the mask), and encoder–decoder attention ($Q$ from the decoder, $K = V$ = encoder output). For self-attention, substitute $Q = K = V = X$, which is the notebook's form.

<Formula name="Multi-head attention (paper eq. in §3.2.2)" tag="The orange boxes of Figure 1" words="Run h attention heads in parallel, each on its own projections of the queries, keys and values, concatenate their outputs and mix them with one more learned matrix." remember="Project h ways, attend, concat, mix.">

$$
\begin{aligned}
&\operatorname{MultiHead}(\pred{Q}, \pred{K}, \pred{V}) \\
&\quad = \operatorname{Concat}(\pred{\text{head}_1}, \ldots, \pred{\text{head}_h})\,\param{W^O} \\
&\pred{\text{head}_i} = \operatorname{Attention}(\pred{Q}\param{W_i^Q},\, \pred{K}\param{W_i^K},\, \pred{V}\param{W_i^V})
\end{aligned}
$$

- $\param{W_i^Q}, \param{W_i^K} \in \mathbb{R}^{\hyper{d_{model}} \times \hyper{d_k}}$, $\param{W_i^V} \in \mathbb{R}^{\hyper{d_{model}} \times \hyper{d_v}}$, $\param{W^O} \in \mathbb{R}^{\hyper{h}\hyper{d_v} \times \hyper{d_{model}}}$
- paper: $\hyper{h} = 8$, $\hyper{d_k} = \hyper{d_v} = \hyper{d_{model}}/\hyper{h} = 64$; self-attention: $\pred{Q} = \pred{K} = \pred{V} = \data{X}$

**Example:** notebook shapes $5 \times 4 \to$ two heads of $5 \times 2 \to H$: $5 \times 4 \to Z$: $5 \times 4$.

</Formula>

<Animation src="multi-head/split-concat" caption="Both heads read the same X but produce different attention matrices. Watch their 5 by 2 outputs slide side by side into H, then W_O mix them into Z with the same 5 by 4 shape as X." />

**Try it yourself.** Click a row in either head to see what that token attends to in each head, its two head outputs, the concatenated row and the final mixed row.

<MultiHeadExplorer />

:::tip[🎯 Interview angle]
**"Write the multi-head attention formula."** $\operatorname{MultiHead}(Q,K,V) = \operatorname{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$ with $\text{head}_i = \operatorname{Attention}(QW_i^Q, KW_i^K, VW_i^V)$ and $\operatorname{Attention} = \operatorname{softmax}(QK^\top/\sqrt{d_k})V$. Shapes: $W_i^{Q,K} \in \mathbb{R}^{d_{model} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$, $W^O \in \mathbb{R}^{hd_v \times d_{model}}$; the paper used $h = 8$, $d_k = d_v = 64$. The same block serves encoder self-attention, masked decoder self-attention and encoder–decoder attention.
:::

## 11. Single-Head vs Multi-Head Attention

Side by side: single-head attention computes **one** attention matrix $A \in \mathbb{R}^{n \times n}$ and one output; multi-head attention computes $h$ attention matrices $A^{(1)}, \ldots, A^{(h)}$, concatenates the outputs and mixes them with $W^O$. The notebook's comparison (its section 11) shows the shapes: single-head output $5 \times 2$, multi-head output $5 \times 4$. The real difference is the **number of attention patterns**, not the final shape: with the standard split, a full-width single head would output $5 \times 4$ too.

With random weights you can't see the benefit, so here is a small experiment (block 7 in the implementation). A toy model reads 8 random tokens and, at every position, must predict **two** things: the **previous** token and the **first** token. Those are two different "where to look" rules. It is trained for 1,500 Adam steps, once with **1 head** and once with **2 heads** of the same total width ($d_{model} = 32$).

- With **2 heads**, the heads **divide the work**. Head 1's average weight on the previous token is 0.996; head 2's average weight on the first token is 0.999. Each is an almost perfect specialist. Their remaining weight on the other target is 0.14, which is just position 1, where the previous token *is* the first token (1 of 7 rows, 1/7 ≈ 0.143). The result repeats on seeds 1, 2 and 3; only *which* head takes which job changes (seed 2 swaps them), because nothing in the architecture prefers one head for one job.
- With **1 head**, the task still reaches 100% accuracy, but the single softmax **splits**: about 0.47 on the first token and 0.51 on the previous token in every row. Its output is a 50/50 average of two value vectors, and the next layer has to untangle them. This is the "averaging inhibits" effect the paper describes, measured.

The paper measured the trade-off at full scale (Table 3, rows A), keeping the total computation constant on English-to-German. Single-head attention ($h = 1$, $d_k = 512$) scored 24.9 dev BLEU; $h = 4$: 25.5; $h = 8$: 25.8; $h = 16$: 25.8; $h = 32$ ($d_k = 16$): 25.4. Single-head is 0.9 BLEU worse than the best setting, and quality also drops with too many heads, because each head becomes too narrow to compute good compatibilities (rows B show that reducing $d_k$ hurts).

<Animation src="multi-head/specialization" caption="Attention of the trained toy model, averaged over 2000 test sequences. Watch the single head split each row about evenly between the first token and the previous token, while with two heads one head takes the previous-token diagonal and the other the first-token column." />

| | Single head | Multi-head ($h$ heads) |
|---|---|---|
| Attention matrices per layer | 1 | $h$ |
| Patterns per token | One distribution | $h$ distributions |
| Per-head width | $d_{model}$ | $d_{model}/h$ |
| Weights (no bias) | $4d_{model}^2$ | $4d_{model}^2$ (same) |
| Toy experiment (prev + first token) | ≈0.47 / 0.51 split per row | Head 1: 0.996 prev; head 2: 0.999 first |
| Paper Table 3 (A), dev BLEU | 24.9 ($h = 1$) | 25.8 ($h = 8$), 25.4 ($h = 32$) |

:::analogy[One spotlight vs several]
A single spotlight that must follow two actors stands between them and lights both dimly. Two spotlights can each follow one actor in full brightness. The lighting budget is the same (the same total wattage spread over two lamps), but the picture is far clearer.
:::

:::tip[🎯 Interview angle]
**"What does multi-head attention buy you over a single head with the same total size?"** Several independent attention distributions per token instead of one. A single head that needs two relations must average them (in a toy experiment, about 0.47/0.51 between two targets), while two heads each attend sharply to one (0.996 and 0.999). The cost is the same, $4d_{model}^2$ weights. The paper's ablation: $h = 1$ scored 24.9 BLEU vs 25.8 for $h = 8$, and too many heads (32) dropped to 25.4 as heads became too narrow.
:::

## 12. Packaging Multi-Head Attention as a Function, and How Libraries Do It

The notebook's last computational step packages the flow into a reusable `multi_head_attention(X, W_Q_list, W_K_list, W_V_list, W_O)`. It loops over heads, computes each head's $Q$, $K$, $V$, attention and output, concatenates, applies $W^O$, and returns four objects: the final output $Z$, the list of attention matrices, the list of head outputs and the concatenation $H$. Returning the intermediates is useful for teaching and debugging: you can plot every head's attention or check every shape. The required inputs are $X$, three lists of per-head projection matrices and $W^O$. With the notebook's numbers it returns $Z$ of shape $5 \times 4$, identical to the step-by-step computation.

Production code doesn't loop over heads. It stores **one** $d_{model} \times d_{model}$ matrix each for Q, K and V: the per-head matrices placed side by side, $W_Q = [W_Q^{(1)} \mid \cdots \mid W_Q^{(h)}]$. It does one big multiply, then **reshapes** the result from (batch, $n$, $d_{model}$) to (batch, $h$, $n$, $d_k$) so all heads run as one batched matrix multiplication. Concat is the reverse reshape. Block 5 shows this gives exactly the same $Z$ as the loop (difference 0.0). PyTorch's `nn.MultiheadAttention` goes further and stores all three as one `in_proj_weight` of shape $(3d_{model}, d_{model})$. Loaded with the notebook's weights it reproduces the NumPy output to $8.9 \times 10^{-16}$ (block 6).

| | Notebook loop | Fused (libraries) |
|---|---|---|
| Projection weights | $3h$ matrices $d_{model} \times d_k$ | 3 (or 1) matrices $d_{model} \times d_{model}$ |
| Heads computed | One at a time in a Python loop | All at once as a batch dimension |
| Concat | `np.concatenate(..., axis=1)` | `transpose` + `reshape` |
| Result | Identical | Identical |

:::danger[Interview Trap: "nn.MultiheadAttention returns each head's attention"]
By default it returns attention weights **averaged over heads** (`average_attn_weights=True`). For the notebook's weights the averaged "good" row is [0.683, 0.222, 0.032, 0.060, 0.003], which matches neither head (head 1: [0.444, 0.423, 0.010, 0.118, 0.004]; head 2: [0.923, 0.021, 0.053, 0.001, 0.003]). Pass `average_attn_weights=False` to get per-head weights of shape (batch, $h$, $n$, $n$). Also note `batch_first=False` is the default (sequence first), and `embed_dim` is the **total** width, split across `num_heads`.
:::

:::tip[🎯 Interview angle]
**"How is multi-head attention implemented efficiently?"** Not with a loop. Use one $d_{model} \times d_{model}$ projection per Q/K/V (the per-head matrices side by side), reshape (batch, $n$, $d_{model}$) to (batch, $h$, $n$, $d_k$), run all heads as one batched matmul and softmax, then transpose and reshape back (that's Concat) and apply $W^O$. It is mathematically identical to looping over heads, just one big GPU kernel.
:::

## 13. Where Multi-Head Attention Sits: the Encoder Block

Multi-head attention never appears alone. In Figure 1 it is the first sub-layer of every **encoder layer**, followed by "Add & Norm", a "Feed Forward" network, and another "Add & Norm". The notebook's last section builds this block with toy shapes to show where MHA fits: $Z_{MHA} = \operatorname{MultiHead}(X)$, then a residual connection and layer normalization $\tilde U = \operatorname{LayerNorm}(X + Z_{MHA})$, then a position-wise feed-forward network $F = \operatorname{FFN}(\tilde U)$ (here $4 \to 8 \to 4$ with a ReLU), then $Y = \operatorname{LayerNorm}(\tilde U + F)$. This is exactly the paper's $\operatorname{LayerNorm}(x + \operatorname{Sublayer}(x))$ around each of the two sub-layers.

The key property is that the block is **shape-preserving**: $5 \times 4$ in, $5 \times 4$ out. The residual additions need it ($X + Z_{MHA}$ only works if both are $5 \times 4$, which is why $W^O$ maps back to $d_{model}$). The FFN expands each token internally to $d_{ff} = 8$ (2048 in the paper) and compresses it back. Layer normalization rescales each token's row to mean 0 and variance 1 (block 8 confirms row means of 0 and variances of 1.000 for $Y$). Because the output has the input's shape, $N$ identical blocks can be stacked: $N = 6$ in the paper, the "N×" beside the encoder in Figure 1.

The notebook says the division of labour well: multi-head attention updates each token using information from all other tokens; the rest of the block stabilizes, transforms and refines those representations. The next page walks through every box of Figure 1 (positional encoding, Add & Norm, Feed Forward, the masked and encoder–decoder attentions, Linear and Softmax) in order.

<Formula name="Encoder layer around multi-head attention" tag="Figure 1, left" words="Add the attention output back to the input and normalize, then do the same around a per-token feed-forward network; the shape stays n by d_model so layers can be stacked." remember="Attend, add, norm; transform, add, norm.">

$$
\begin{aligned}
\pred{\tilde U} &= \operatorname{LayerNorm}(\data{X} + \operatorname{MHA}(\data{X}, \data{X}, \data{X})) \\
\pred{Y} &= \operatorname{LayerNorm}(\pred{\tilde U} + \operatorname{FFN}(\pred{\tilde U})) \\
\operatorname{FFN}(x) &= \max(0, x\param{W_1} + \param{b_1})\param{W_2} + \param{b_2}
\end{aligned}
$$

- MHA = MultiHead; $\data{X}, \pred{\tilde U}, \pred{Y} \in \mathbb{R}^{n \times \hyper{d_{model}}}$; FFN inner width $\hyper{d_{ff}}$ (8 here, 2048 in the paper)
- stacked $\hyper{N}$ times ($\hyper{N} = 6$ in the paper)

**Example:** notebook: $5 \times 4 \to 5 \times 4 \to$ FFN $4 \to 8 \to 4 \to 5 \times 4$; every row of $\pred{Y}$ has mean 0 and variance 1.000.

</Formula>

:::tip[🎯 Interview angle]
**"Where does multi-head attention sit in a Transformer encoder layer?"** It's the first of two sub-layers: multi-head self-attention, then Add & Norm (residual plus LayerNorm), then a position-wise feed-forward network (512 → 2048 → 512 with ReLU in the paper), then Add & Norm again. Every sub-layer keeps the $n \times d_{model}$ shape so the residuals work and $N = 6$ identical layers can be stacked.
:::

## ⚙️ Going Deeper (Multi-Head Internals)

**Three uses of the same block (paper §3.2.3).** The Transformer uses multi-head attention three ways. *Encoder self-attention:* Q, K, V all come from the previous encoder layer, so every position attends to every input position. *Decoder masked self-attention:* Q, K, V from the previous decoder layer, with future positions masked to $-\infty$ to keep generation auto-regressive. *Encoder–decoder attention:* queries from the decoder, keys and values from the encoder output, so every output position can look at the whole input (the classic seq2seq attention). Only the inputs and the mask change; the block is identical.

**Why narrower heads still work.** Each head's $d_k$-dimensional dot product is a low-rank similarity: $QK^\top = XW_Q W_K^\top X^\top$ has rank at most $d_k$. A single full-width head has one rank-$d_{model}$ similarity; $h$ heads have $h$ separate rank-$d_k$ similarities, each with its own softmax. The paper's rows (B) show the limit: shrinking $d_k$ too far (16 or 32 with $h = 8$) costs quality, because compatibility gets too coarse.

**Cost.** Projections cost $O(n \cdot d_{model}^2)$ and the attention itself $O(h \cdot n^2 \cdot d_k) = O(n^2 d_{model})$, the same as one full-width head. Memory for the attention matrices is $O(h \cdot n^2)$, so more heads cost more activation memory even at the same parameter count.

**Fewer key/value heads (modern variants).** At inference, the stored keys and values of past tokens (the **KV cache**) dominate memory. **Multi-query attention** shares one K/V head across all query heads, and **grouped-query attention** shares each K/V head among a group of query heads. Both shrink the cache while keeping many query heads. They are common in recent large models.

**Heads are redundant and prunable.** "Are Sixteen Heads Really Better than One?" (Michel et al., 2019) found that many heads in a trained Transformer can be removed at test time with little loss, and some layers work with a single head. Multi-head capacity helps training, but not every head ends up essential.

**Learnable LayerNorm.** The notebook's `simple_layer_norm` only standardizes each row. The real LayerNorm (Ba et al., 2016, the paper's reference [1]) also has a learned gain $\gamma$ and bias $\beta$ per feature: $\gamma \odot (x - \mu)/\sqrt{\sigma^2 + \epsilon} + \beta$. The paper also applies dropout ($P_{drop} = 0.1$) to each sub-layer output before the residual addition. Both are covered on the next page.

**Symmetry between heads.** Heads are interchangeable: permuting them (and the matching blocks of $W^O$) gives the same function. That's why, in the toy experiment, which head becomes the "previous token" specialist depends on the random seed.

## 💻 Python Implementation

:::tip[Run it yourself]
Verified on Python 3.13, NumPy 2.5 and PyTorch 2.15 (CPU); outputs copied from real runs. Blocks 1–5 continue one session (`np.random.seed(42)`, the notebook's draw order); blocks 6–8 are self-contained except where they say so.
:::

**What was corrected or clarified from the notebook.** The notebook writes $W_O \in \mathbb{R}^{d_{model} \times d_{model}}$; the paper's general shape is $hd_v \times d_{model}$ (equal only with the standard split), and the code already uses `(h * d_k, d_model)`. In section 11 the "single head" draws its weights first, so it re-uses the exact numbers that sections 1–10 called head 1, and section 11's two "multi-head" heads are new random draws, not the heads analyzed earlier. The `simple_layer_norm` has no learnable gain and bias, and there is no dropout (see Going Deeper). Block 5 adds the fused implementation libraries use, block 6 the PyTorch equivalent, and block 7 a trained example, because random weights can't show specialization.

### 1. Per-head projections

```python title="multi_head_setup.py" showLineNumbers
import numpy as np
np.random.seed(42)

def row_softmax(M):
    M = M - M.max(axis=1, keepdims=True)
    E = np.exp(M)
    return E / E.sum(axis=1, keepdims=True)

def attention(Q, K, V):
    A = row_softmax(Q @ K.T / np.sqrt(Q.shape[-1]))      # scale by the per-head d_k
    return A, A @ V

tokens = ["The", "movie", "was", "not", "good"]
n, d_model, h = 5, 4, 2
d_k = d_model // h                                       # 2
X = np.random.randn(n, d_model)

W_Q, W_K, W_V = [], [], []
for r in range(h):                                       # same draw order as the notebook
    W_Q.append(np.random.randn(d_model, d_k))
    W_K.append(np.random.randn(d_model, d_k))
    W_V.append(np.random.randn(d_model, d_k))
W_O = np.random.randn(h * d_k, d_model)                  # (4, 4)

# highlight-next-line
Q1, Q2 = X @ W_Q[0], X @ W_Q[1]
print(Q1.shape, np.round(Q1[0], 3), np.round(Q2[0], 3))  # (5, 2) [-1.387  0.729] [ 0.045 -2.232]
print(np.allclose(Q1, Q2))                               # False
```

### 2. Attention inside each head

```python title="per_head_attention.py" showLineNumbers
As, heads = [], []
for r in range(h):
    # highlight-next-line
    A, out = attention(X @ W_Q[r], X @ W_K[r], X @ W_V[r])
    As.append(A); heads.append(out)
print(As[0].shape, heads[0].shape)                       # (5, 5) (5, 2)
print(np.round(As[0][4], 3))                             # [0.444 0.423 0.01  0.118 0.004]   head 1, row "good"
print(np.round(As[1][4], 3))                             # [0.923 0.021 0.053 0.001 0.003]   head 2, row "good"
print(As[0].sum(axis=1), As[1].sum(axis=1))              # [1. 1. 1. 1. 1.] [1. 1. 1. 1. 1.]
print(round(np.abs(As[0] - As[1]).max(), 3))             # 0.807
for t, a1, a2 in zip(tokens, As[0], As[1]):
    print(t, tokens[a1.argmax()], round(a1.max(), 3), tokens[a2.argmax()], round(a2.max(), 3))
# The good 0.712 not 0.516
# movie good 0.775 good 0.774
# was movie 0.427 The 0.771
# not good 0.854 The 0.566
# good The 0.444 The 0.923
```

### 3. Concat, then W_O

```python title="concat_and_project.py" showLineNumbers
# highlight-next-line
H = np.concatenate(heads, axis=1)                        # side by side along features
print(H.shape, np.round(H[0], 3))                        # (5, 4) [-1.063  1.911  0.175 -1.367]
print(np.round(heads[0][0], 3), np.round(heads[1][0], 3))  # [-1.063  1.911] [ 0.175 -1.367]   just placed together
# highlight-next-line
Z = H @ W_O                                              # heads mixed, back to d_model
print(Z.shape, np.round(Z[0], 3))                        # (5, 4) [-0.137  3.136 -7.395  0.296]
print(Z.shape == X.shape)                                # True
```

### 4. The notebook's function

```python title="multi_head_function.py" showLineNumbers
def multi_head_attention(X, W_Q_list, W_K_list, W_V_list, W_O):
    attention_matrices, head_outputs = [], []
    for W_q, W_k, W_v in zip(W_Q_list, W_K_list, W_V_list):
        A, out = attention(X @ W_q, X @ W_k, X @ W_v)
        attention_matrices.append(A); head_outputs.append(out)
    H_concat = np.concatenate(head_outputs, axis=1)
    return H_concat @ W_O, attention_matrices, head_outputs, H_concat

Z_f, A_f, heads_f, H_f = multi_head_attention(X, W_Q, W_K, W_V, W_O)
print(np.allclose(Z_f, Z), H_f.shape, len(A_f))          # True (5, 4) 2
```

### 5. The fused version libraries use (and why heads must differ)

```python title="fused_heads.py" showLineNumbers
W_Qb, W_Kb, W_Vb = (np.concatenate(Ws, axis=1) for Ws in (W_Q, W_K, W_V))   # each (4, 4)
# highlight-start
Qh = (X @ W_Qb).reshape(n, h, d_k).transpose(1, 0, 2)  # (h, n, d_k): heads become a batch axis
Kh = (X @ W_Kb).reshape(n, h, d_k).transpose(1, 0, 2)
Vh = (X @ W_Vb).reshape(n, h, d_k).transpose(1, 0, 2)
# highlight-end
S = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(d_k)            # (h, n, n) all heads at once
A = np.exp(S - S.max(-1, keepdims=True)); A /= A.sum(-1, keepdims=True)
Z_fused = (A @ Vh).transpose(1, 0, 2).reshape(n, h * d_k) @ W_O   # transpose + reshape = Concat
print(np.abs(Z_fused - Z).max())                         # 0.0

H_same = np.concatenate([heads[0], heads[0]], axis=1)    # two identical heads
print(np.linalg.matrix_rank(H), np.linalg.matrix_rank(H_same))   # 4 2   copies add no information
print(3 * h * d_model * d_k + h * d_k * d_model)         # 64 weights = 4 * d_model**2
```

### 6. PyTorch's nn.MultiheadAttention with the same weights

```python title="torch_mha.py" showLineNumbers
import torch, torch.nn as nn                             # continues the session above

mha = nn.MultiheadAttention(embed_dim=4, num_heads=2, bias=False, batch_first=True).double()
with torch.no_grad():
    # in_proj_weight stacks W_Q, W_K, W_V, each stored transposed: shape (3 * 4, 4)
    mha.in_proj_weight.copy_(torch.tensor(np.concatenate([W_Qb.T, W_Kb.T, W_Vb.T], axis=0)))
    mha.out_proj.weight.copy_(torch.tensor(W_O.T))
x = torch.tensor(X)[None]                                # (batch=1, n=5, d_model=4)
# highlight-next-line
out, w = mha(x, x, x, average_attn_weights=False)        # per-head weights
print(out.shape, w.shape)                                # torch.Size([1, 5, 4]) torch.Size([1, 2, 5, 5])
print(np.abs(out[0].detach().numpy() - Z).max())         # 8.881784197001252e-16
# error-next-line
_, w_avg = mha(x, x, x)                                  # ❌ default: averaged over heads
print(torch.round(w_avg[0, 4], decimals=4))
# tensor([0.6833, 0.2220, 0.0318, 0.0595, 0.0034], dtype=torch.float64, grad_fn=<RoundBackward1>)
```

### 7. Trained heads specialize

```python title="heads_specialize.py" showLineNumbers
import torch, torch.nn as nn

V, L, D, STEPS = 10, 8, 32, 1500                         # vocab, length, width, Adam steps

class Toy(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, D), nn.Embedding(L, D)
        self.mha = nn.MultiheadAttention(D, heads, batch_first=True)
        self.out_prev, self.out_first = nn.Linear(D, V), nn.Linear(D, V)

    def forward(self, x):
        e = self.tok(x) + self.pos(torch.arange(L))
        z, w = self.mha(e, e, e, average_attn_weights=False)
        return self.out_prev(z), self.out_first(z), w

def batch(b=256):                                        # targets: previous token AND first token
    x = torch.randint(0, V, (b, L))
    return x, x[:, :-1], x[:, :1].expand(b, L - 1)

def run(heads, seed=0):
    torch.manual_seed(seed)
    m, ce = Toy(heads), nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    for _ in range(STEPS):
        x, yp, yf = batch(); p, f, _ = m(x)
        loss = ce(p[:, 1:].reshape(-1, V), yp.reshape(-1)) + ce(f[:, 1:].reshape(-1, V), yf.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    torch.manual_seed(123); x, yp, yf = batch(2000)
    with torch.no_grad():
        p, f, w = m(x)
    W, rows = w.mean(0), torch.arange(1, L)              # batch-averaged attention per head
    return ((p[:, 1:].argmax(-1) == yp).float().mean().item(), (f[:, 1:].argmax(-1) == yf).float().mean().item(),
            W[:, rows, rows - 1].mean(-1), W[:, rows, 0].mean(-1))

for heads in (1, 2):
    acc_p, acc_f, to_prev, to_first = run(heads)
    print(heads, round(acc_p, 3), round(acc_f, 3), [f"prev {a:.3f} / first {b:.3f}" for a, b in zip(to_prev, to_first)])
# 1 1.0 1.0 ['prev 0.579 / first 0.541']
# 2 1.0 1.0 ['prev 0.996 / first 0.141', 'prev 0.143 / first 0.999']
```

The single head's two numbers add to more than 1 because position 1 counts for both (there, the previous token *is* the first token); in every other row it puts about 0.47 on the first token and 0.51 on the previous one.

### 8. The encoder block around it

```python title="encoder_block.py" showLineNumbers
import numpy as np
np.random.seed(42)                                       # notebook section 13 draw order
n, d_model, h, d_ff = 5, 4, 2, 8
d_k = d_model // h
X = np.random.randn(n, d_model)
W_Q, W_K, W_V = [], [], []
for _ in range(h):
    W_Q.append(np.random.randn(d_model, d_k)); W_K.append(np.random.randn(d_model, d_k)); W_V.append(np.random.randn(d_model, d_k))
W_O = np.random.randn(h * d_k, d_model)
W_1, b_1 = np.random.randn(d_model, d_ff), np.random.randn(d_ff)
W_2, b_2 = np.random.randn(d_ff, d_model), np.random.randn(d_model)

def row_softmax(M):
    M = M - M.max(axis=1, keepdims=True); E = np.exp(M); return E / E.sum(axis=1, keepdims=True)

def layer_norm(M, eps=1e-6):                             # no learnable gain/bias, as in the notebook
    return (M - M.mean(axis=1, keepdims=True)) / np.sqrt(M.var(axis=1, keepdims=True) + eps)

heads = [row_softmax((X @ q) @ (X @ k).T / np.sqrt(d_k)) @ (X @ v) for q, k, v in zip(W_Q, W_K, W_V)]
Z_mha = np.concatenate(heads, axis=1) @ W_O
# highlight-start
U_norm = layer_norm(X + Z_mha)                           # Add & Norm
F = np.maximum(0, U_norm @ W_1 + b_1) @ W_2 + b_2        # Feed Forward: 4 -> 8 -> 4
Y = layer_norm(U_norm + F)                               # Add & Norm
# highlight-end
print(X.shape, Z_mha.shape, F.shape, Y.shape)            # (5, 4) (5, 4) (5, 4) (5, 4)
print(np.round(Y[0], 3))                                 # [ 0.346  1.288 -1.483 -0.151]
print(np.round(Y.mean(axis=1), 6), np.round(Y.var(axis=1), 3))   # [-0.  0.  0.  0. -0.] [1. 1. 1. 1. 1.]
```

## 🚨 Interview Gotchas & Edge Cases

:::warning[The Most Common Question]
**"Why multi-head attention instead of one big attention head?"**

One head gives each token a single softmax distribution, so it can follow one relationship at a time; needing several forces it to average their values, which blurs them (the paper: "with a single attention head, averaging inhibits this"). Multi-head attention projects Q, K, V $h$ times into $d_k = d_{model}/h$ subspaces with separate learned matrices, runs attention in each, concatenates and mixes with $W^O$. Each head can specialize (in a toy model, previous-token 0.996 vs first-token 0.999, while one head split about 50/50), and the cost equals one full-width head ($4d_{model}^2$ weights). The paper measured 24.9 BLEU for $h = 1$ vs 25.8 for $h = 8$; 32 heads dropped to 25.4 as each head got too narrow.
:::

- **Heads don't add parameters** with $d_k = d_{model}/h$: always $4d_{model}^2$ (plus biases).
- **$d_{model}$ must be divisible by $h$;** libraries raise an error otherwise.
- **Scale by $\sqrt{d_k}$ of the head** (64), not $\sqrt{d_{model}}$ (512).
- **Concat doesn't mix;** $W^O$ does. Leaving out $W^O$ means heads never interact inside the layer.
- **$W^O$ is $hd_v \times d_{model}$;** square only with the standard split.
- **Same weights for all heads = one head** (rank of Concat drops from 4 to 2 in the toy).
- **Random heads look different but mean nothing;** specialization comes from training.
- **`nn.MultiheadAttention` averages attention over heads by default;** use `average_attn_weights=False`. `batch_first` defaults to False.
- **Which head learns what is arbitrary** (heads are interchangeable); don't hard-code "head 3 = syntax".
- **Too many heads hurts** when $d_k$ gets too small (Table 3: $h = 32$, $d_k = 16$ → 25.4).
- **More heads cost activation memory** ($h \times n^2$ attention entries), even at constant parameters.

## ⚡ Essential Interview Recall

A one-screen revision list. Every point is explained in the chapters above.

**Why and what**
- One head = one attention distribution per token → must average several relations.
- MHA = $h$ scaled dot-product attentions in parallel on separate learned projections, then Concat, then $W^O$.
- Paper: "jointly attend to information from different representation subspaces at different positions".

**Shapes**
- $d_k = d_v = d_{model}/h$: notebook 4/2 = 2; paper 512/8 = 64.
- $W_i^{Q,K}: d_{model} \times d_k$, $W_i^V: d_{model} \times d_v$, $\text{head}_i: n \times d_v$, Concat: $n \times hd_v$, $W^O: hd_v \times d_{model}$, output $n \times d_{model}$.
- Weights $4d_{model}^2$ regardless of $h$ (toy 64; base 1,048,576).

**Formula**
- $\operatorname{MultiHead}(Q,K,V) = \operatorname{Concat}(\text{head}_1..\text{head}_h)W^O$, $\text{head}_i = \operatorname{Attention}(QW_i^Q, KW_i^K, VW_i^V)$.
- Inside a head: MatMul → Scale → Mask (opt.) → SoftMax → MatMul (Figure 2, left).

**Toy numbers (seed 42)**
- "good": head 1 [0.444, 0.423, 0.010, 0.118, 0.004], head 2 [0.923, 0.021, 0.053, 0.001, 0.003]; max $|A^{(1)} - A^{(2)}|$ = 0.807.
- $H_{The}$ = [−1.063, 1.911, 0.175, −1.367] → $Z_{The}$ = [−0.137, 3.136, −7.395, 0.296].

**Evidence**
- Trained toy: 2 heads → 0.996 previous-token head + 0.999 first-token head; 1 head → about 0.47/0.51 split.
- Table 3 (A): $h$ = 1: 24.9, 4: 25.5, 8: 25.8, 16: 25.8, 32: 25.4 dev BLEU.

**Implementation and context**
- Fused: one $d_{model} \times d_{model}$ matrix per Q/K/V, reshape to (batch, $h$, $n$, $d_k$); identical to the loop.
- PyTorch averages head weights by default; matches NumPy to $8.9 \times 10^{-16}$.
- Encoder layer: MHA → Add & Norm → FFN (512 → 2048 → 512) → Add & Norm, shape-preserving, $N = 6$.
- Three uses: encoder self, masked decoder self, encoder–decoder (Q from decoder, K/V from encoder).

## 📐 Formula Sheet

Every equation on this page in one place.

| Concept | Formula | In plain words |
|---|---|---|
| Per-head projections | $\pred{Q^{(r)}} = \data{X}\param{W_Q^{(r)}}$, $\pred{K^{(r)}} = \data{X}\param{W_K^{(r)}}$, $\pred{V^{(r)}} = \data{X}\param{W_V^{(r)}}$ | Same X, different lens per head |
| Head width | $\hyper{d_k} = \hyper{d_v} = \hyper{d_{model}}/\hyper{h}$ | Split the width across heads |
| One head | $\pred{\text{head}_r} = \operatorname{softmax}(\pred{Q^{(r)}}\pred{K^{(r)}}^\top/\sqrt{\hyper{d_k}})\pred{V^{(r)}}$ | Full attention per head |
| Concat | $\pred{H} = \operatorname{Concat}(\pred{\text{head}_1}, \ldots, \pred{\text{head}_h}) \in \mathbb{R}^{n \times \hyper{h}\hyper{d_v}}$ | Side by side, not mixed |
| Output projection | $\pred{Z} = \pred{H}\param{W^O}$, $\param{W^O} \in \mathbb{R}^{\hyper{h}\hyper{d_v} \times \hyper{d_{model}}}$ | Mix heads, back to model width |
| Multi-head attention | $\operatorname{Concat}(\pred{\text{head}_1}..\pred{\text{head}_h})\param{W^O}$, $\pred{\text{head}_i} = \operatorname{Attention}(\pred{Q}\param{W_i^Q}, \pred{K}\param{W_i^K}, \pred{V}\param{W_i^V})$ | Project h ways, attend, concat, mix |
| Parameter count | $3\hyper{h} \cdot \hyper{d_{model}}\hyper{d_k} + \hyper{h}\hyper{d_v}\hyper{d_{model}} = 4\hyper{d_{model}}^2$ | Same as one full-width head |
| Encoder sub-layer | $\operatorname{LayerNorm}(\data{x} + \operatorname{Sublayer}(\data{x}))$ | Add & Norm |
| Feed-forward | $\operatorname{FFN}(x) = \max(0, x\param{W_1} + \param{b_1})\param{W_2} + \param{b_2}$ | Per-token two-layer MLP |

## Concept-Check Answers (from the notebook)

### Section 1: why one attention head is not enough

<details>
<summary><strong>Q1. What is the shape of the input embedding matrix X in this example?</strong></summary>

$5 \times 4$: five tokens, each with $d_{model} = 4$ features.
</details>

<details>
<summary><strong>Q2. Why does the attention score matrix QKᵀ have shape 5 × 5?</strong></summary>

$Q$ is $5 \times 2$ and $K^\top$ is $2 \times 5$, so the product is $5 \times 5$: every one of the 5 query tokens compared with every one of the 5 key tokens.
</details>

<details>
<summary><strong>Q3. What does one row of the attention weight matrix A represent?</strong></summary>

How one query token distributes its attention over all tokens (a softmax distribution summing to 1). Row "good" here is [0.444, 0.423, 0.010, 0.118, 0.004].
</details>

<details>
<summary><strong>Q4. Why is a single attention head limited?</strong></summary>

It produces one attention matrix, so one pattern per token. Language has many simultaneous relations (negation, subject–object, adjective–noun, syntax, long-distance dependencies). One head must pick one or average their values, which blurs them.
</details>

<details>
<summary><strong>Q5. In "The movie was not good", why might "good" need to attend to "not"?</strong></summary>

"not" reverses the meaning of "good". Unless "good"'s representation takes in information from "not", the phrase reads as positive.
</details>

### Section 2: multi-head attention big picture

<details>
<summary><strong>Q1. Why do Head 1 and Head 2 receive the same input matrix X?</strong></summary>

They are parallel attentions over the same sequence: each must be able to look at every token. They differ in how they project it, not in what they see.
</details>

<details>
<summary><strong>Q2. If both heads receive the same X, why can their attention matrices be different?</strong></summary>

Each head has its own $W_Q^{(r)}, W_K^{(r)}, W_V^{(r)}$, so it computes different queries and keys and therefore different scores and weights (max difference 0.807 here).
</details>

<details>
<summary><strong>Q3. What is the shape of Q⁽¹⁾ in this example?</strong></summary>

$5 \times 2$: $(5 \times 4)(4 \times 2)$.
</details>

<details>
<summary><strong>Q4. What is the shape of the attention matrix A⁽²⁾?</strong></summary>

$5 \times 5$: every query token against every key token.
</details>

<details>
<summary><strong>Q5. In simple words, what does multi-head attention allow the model to do?</strong></summary>

Look at the sentence in several ways at once: several attention patterns in parallel, so different heads can capture different relationships, then combine them.
</details>

### Section 3: separate projection matrices for each head

<details>
<summary><strong>Q1. Why should different heads have different projection matrices?</strong></summary>

So each head maps the input into a different subspace and can learn a different attention pattern and extract different information.
</details>

<details>
<summary><strong>Q2. What would happen if all heads used the same W_Q, W_K and W_V?</strong></summary>

All heads would compute identical $Q$, $K$, $V$, attention and outputs. Concatenating copies adds nothing (the concatenation's rank falls from 4 to 2 in the toy), so the layer would behave like one head.
</details>

<details>
<summary><strong>Q3. What is the shape of W_Q⁽¹⁾ in this example?</strong></summary>

$4 \times 2$ ($d_{model} \times d_k$).
</details>

<details>
<summary><strong>Q4. What is the shape of Q⁽²⁾ in this example?</strong></summary>

$5 \times 2$.
</details>

<details>
<summary><strong>Q5. In simple language, what role does W_V⁽ʳ⁾ play?</strong></summary>

It decides what information head $r$ extracts from each token and passes on. The values are the content that gets mixed by the attention weights.
</details>

### Section 4: shape setup

<details>
<summary><strong>Q1. What does n represent in X ∈ ℝ^(n × d_model)?</strong></summary>

The number of tokens in the sequence: 5 here.
</details>

<details>
<summary><strong>Q2. In this example, what is the value of d_model?</strong></summary>

4: each token is a 4-dimensional vector.
</details>

<details>
<summary><strong>Q3. If d_model = 4 and h = 2, what is d_k?</strong></summary>

$4 / 2 = 2$.
</details>

<details>
<summary><strong>Q4. What is the shape of W_Q⁽ʳ⁾ for each head?</strong></summary>

$4 \times 2$.
</details>

<details>
<summary><strong>Q5. Why do we often choose d_k = d_model / h?</strong></summary>

So the concatenated output has width $h \cdot d_k = d_{model}$, the same as the input (needed for residuals and stacking), and so the total parameters and compute equal one full-width head ($4d_{model}^2$).
</details>

### Section 5: computing Q, K, V for each head

<details>
<summary><strong>Q1. What is the input matrix used by both heads?</strong></summary>

The same $X \in \mathbb{R}^{5 \times 4}$.
</details>

<details>
<summary><strong>Q2. Why are Q⁽¹⁾ and Q⁽²⁾ usually different?</strong></summary>

They use different matrices, $W_Q^{(1)} \ne W_Q^{(2)}$. The input is the same; the projection changes ("The": [−1.387, 0.729] vs [0.045, −2.232]).
</details>

<details>
<summary><strong>Q3. What is the shape of V⁽¹⁾ in this example?</strong></summary>

$5 \times 2$.
</details>

<details>
<summary><strong>Q4. What does K⁽ʳ⁾ represent for head r?</strong></summary>

Head $r$'s key vectors: how each token can be matched by head $r$'s queries, which decides how relevant each token is to each query in that head.
</details>

<details>
<summary><strong>Q5. Why is it useful for each head to have a different value matrix V⁽ʳ⁾?</strong></summary>

Values are what gets passed forward after weighting. Different value projections let each head extract a different kind of information, not just attend to different tokens.
</details>

### Section 6: scaled dot-product attention per head

<details>
<summary><strong>Q1. What is the formula for the output of head r?</strong></summary>

$\text{head}_r = \operatorname{softmax}\!\left(Q^{(r)}K^{(r)\top}/\sqrt{d_k}\right)V^{(r)}$.
</details>

<details>
<summary><strong>Q2. Why does Q⁽ʳ⁾K⁽ʳ⁾ᵀ produce a 5 × 5 matrix?</strong></summary>

$(5 \times 2)(2 \times 5) = 5 \times 5$: each of the 5 queries against each of the 5 keys.
</details>

<details>
<summary><strong>Q3. Why do we divide the attention scores by √d_k?</strong></summary>

Dot products of $d_k$-dimensional vectors grow in spread like $\sqrt{d_k}$; dividing keeps them at unit variance so softmax doesn't saturate (near one-hot, vanishing gradients). Here $d_k$ is the per-head width, 2.
</details>

<details>
<summary><strong>Q4. What does one row of A⁽ʳ⁾ represent?</strong></summary>

How one query token distributes its attention over all tokens in head $r$; it sums to 1.
</details>

<details>
<summary><strong>Q5. What is the shape of each head output in this example?</strong></summary>

$5 \times 2$: $(5 \times 5)(5 \times 2)$.
</details>

### Section 7: visualizing attention across heads

<details>
<summary><strong>Q1. What does a row of an attention heatmap represent?</strong></summary>

A query token: how it spreads its attention over all key tokens.
</details>

<details>
<summary><strong>Q2. What does a column of an attention heatmap represent?</strong></summary>

A key token: how much attention each query token gives it.
</details>

<details>
<summary><strong>Q3. What does the value A_ij⁽ʳ⁾ mean?</strong></summary>

The attention weight from query token $i$ to key token $j$ in head $r$.
</details>

<details>
<summary><strong>Q4. Why can Head 1 and Head 2 produce different heatmaps?</strong></summary>

They compute attention in different projected spaces, because their $W_Q$, $W_K$, $W_V$ differ.
</details>

<details>
<summary><strong>Q5. Why should we avoid strong semantic interpretation of the heatmaps in this toy example?</strong></summary>

The projections are random and untrained, so the patterns reflect random numbers, not language. They demonstrate mechanism and shapes only; specialization appears with training (chapter 11).
</details>

### Section 8: concatenating heads

<details>
<summary><strong>Q1. What is the shape of each head output in this example?</strong></summary>

$5 \times 2$.
</details>

<details>
<summary><strong>Q2. Along which axis do we concatenate the head outputs?</strong></summary>

The feature axis: `axis=1` for a (tokens, features) matrix (`dim=-1` for batched tensors).
</details>

<details>
<summary><strong>Q3. What is the shape of the concatenated matrix H?</strong></summary>

$5 \times 4$.
</details>

<details>
<summary><strong>Q4. Does concatenation itself mix information across heads?</strong></summary>

No. It only places the head outputs side by side ($H_{The}$ = head 1's [−1.063, 1.911] followed by head 2's [0.175, −1.367]).
</details>

<details>
<summary><strong>Q5. Why is concatenation not the final step of multi-head attention?</strong></summary>

The heads still need to be combined. The learned output projection $W^O$ mixes features from all heads into the final representation.
</details>

### Section 9: output projection W^O

<details>
<summary><strong>Q1. What is the shape of the concatenated matrix H in this example?</strong></summary>

$5 \times 4$.
</details>

<details>
<summary><strong>Q2. What is the shape of the output projection matrix W_O?</strong></summary>

$4 \times 4$ here. In general $hd_v \times d_{model}$, which is $d_{model} \times d_{model}$ when $hd_v = d_{model}$.
</details>

<details>
<summary><strong>Q3. What is the formula for the final multi-head attention output after output projection?</strong></summary>

$Z = HW^O = \operatorname{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$.
</details>

<details>
<summary><strong>Q4. Why is concatenation alone not enough?</strong></summary>

It doesn't learn how to combine the heads; each output feature would still come from a single head. $W^O$ lets every output feature use information from every head.
</details>

<details>
<summary><strong>Q5. Why is it useful that the final output shape matches X?</strong></summary>

The block can then be followed by the residual addition $X + Z$ and by the next layer, and identical layers can be stacked, all expecting $n \times d_{model}$.
</details>

### Section 10: full multi-head attention formula

<details>
<summary><strong>Q1. Write the formula for head_r using Q⁽ʳ⁾, K⁽ʳ⁾, V⁽ʳ⁾.</strong></summary>

$\text{head}_r = \operatorname{softmax}\!\left(Q^{(r)}K^{(r)\top}/\sqrt{d_k}\right)V^{(r)}$.
</details>

<details>
<summary><strong>Q2. Write the formula for head_r directly in terms of X and the projection matrices.</strong></summary>

$\text{head}_r = \operatorname{softmax}\!\left(XW_Q^{(r)}(XW_K^{(r)})^\top/\sqrt{d_k}\right)XW_V^{(r)}$.
</details>

<details>
<summary><strong>Q3. What is the formula for MultiHead(X)?</strong></summary>

$\operatorname{MultiHead}(X) = \operatorname{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$ with $\text{head}_r = \operatorname{Attention}(XW_Q^{(r)}, XW_K^{(r)}, XW_V^{(r)})$; in the paper's general form, $\operatorname{MultiHead}(Q, K, V)$.
</details>

<details>
<summary><strong>Q4. Why do we concatenate the heads before applying W_O?</strong></summary>

To collect all heads' representations into one $n \times hd_v$ matrix, so one learned matrix $W^O$ can mix across them in a single multiplication.
</details>

<details>
<summary><strong>Q5. What is the final output shape of multi-head attention in this example?</strong></summary>

$5 \times 4$, the same as $X$.
</details>

### Section 11: single-head vs multi-head

<details>
<summary><strong>Q1. How many attention matrices does single-head attention produce?</strong></summary>

One, $A \in \mathbb{R}^{n \times n}$.
</details>

<details>
<summary><strong>Q2. How many attention matrices does multi-head attention produce when h = 2?</strong></summary>

Two: $A^{(1)}$ and $A^{(2)}$, each $5 \times 5$.
</details>

<details>
<summary><strong>Q3. Why can different heads learn different attention patterns?</strong></summary>

Each has its own projection matrices, so each computes attention in its own representation space and gets its own gradients during training.
</details>

<details>
<summary><strong>Q4. What is the main representational advantage of multi-head attention?</strong></summary>

It captures several relationships in parallel without averaging them together. In the trained toy, 2 heads became a 0.996 previous-token head and a 0.999 first-token head, while one head split about 0.47/0.51.
</details>

<details>
<summary><strong>Q5. Why do we apply W_O after concatenating the heads?</strong></summary>

Concatenation only places the head outputs side by side; $W^O$ learns how to combine information from different heads into the final $d_{model}$-wide representation.
</details>

### Section 12: NumPy function for multi-head attention

<details>
<summary><strong>Q1. Why is it useful to package multi-head attention into a function?</strong></summary>

The same steps run in every layer and every forward pass. A function makes the code reusable, testable and easier to debug, and it's the shape that becomes an `nn.Module` in practice.
</details>

<details>
<summary><strong>Q2. What are the inputs required by the multi_head_attention() function?</strong></summary>

The input matrix $X$, lists of per-head $W_Q$, $W_K$ and $W_V$ matrices, and the output projection $W^O$.
</details>

<details>
<summary><strong>Q3. Why does the function return attention matrices separately?</strong></summary>

So each head's pattern can be inspected, visualized and debugged (and checked for row sums), which the final output alone doesn't reveal.
</details>

<details>
<summary><strong>Q4. What is the shape of H_concat in this example?</strong></summary>

$5 \times 4$.
</details>

<details>
<summary><strong>Q5. What is the final output shape returned by the function?</strong></summary>

$5 \times 4$, the same as $X$.
</details>
