---
title: "The Transformer: Figure 1 Box by Box"
sidebar_position: 4
---

import {Figure1Explorer, PositionalEncodingExplorer, LearningRateSchedule} from '@site/src/components/Widgets';

:::tip[TL;DR (30-Second Interview Pitch)]
The Transformer ("Attention Is All You Need", Vaswani et al., NIPS 2017) is an **encoder–decoder** built only from attention and position-wise feed-forward layers: no recurrence, no convolution. **Encoder** (left of Figure 1): Input Embedding (×√d_model) + Positional Encoding, then N = 6 identical layers of [Multi-Head self-Attention → Add & Norm → Feed Forward → Add & Norm]. **Decoder** (right): Outputs (shifted right) → Output Embedding + Positional Encoding, then N = 6 layers of [**Masked** Multi-Head Attention → Add & Norm → **encoder–decoder** Multi-Head Attention (queries from the decoder, keys and values from the encoder) → Add & Norm → Feed Forward → Add & Norm], then Linear → Softmax → Output Probabilities. Every sub-layer is $\operatorname{LayerNorm}(x + \operatorname{Sublayer}(x))$ with width $d_{model} = 512$; attention has $h = 8$ heads of 64; the FFN is 512 → 2048 → 512. Trained with Adam plus a warmup learning-rate schedule, dropout 0.1 and label smoothing 0.1, it reached 28.4 BLEU on WMT14 English–German, beating all previous models including ensembles, at a fraction of the training cost.
:::

<FormulaKey />

:::info[Where this page sits]
**Before:** [Multi-Head Attention: Many Attention Patterns in Parallel](./03-multi-head-attention.md) built the orange boxes.
**This page:** the paper **"Attention Is All You Need"** (arXiv 1706.03762), organized around its **Figure 1**, using your notes from notebooks 14–16 for the attention parts. Every label of Figure 1 gets its own chapter, in the order data flows through it (encoder bottom to top, then decoder bottom to top, then the output head), followed by the paper's "why self-attention", training and results sections. A small Transformer, written so its classes mirror Figure 1, is trained on this page to show every box doing its job with real numbers.
**Next:** BERT (notebook 17) keeps only the left half of Figure 1, the encoder.
:::

## 🧭 Start Here

Everything from the last three pages (attention, scaling, softmax, queries/keys/values, multiple heads) is **one box type** in the figure below. The rest of the Transformer is the scaffolding that turns those boxes into a model that can read a sentence in one language and write it in another. Learn this figure label by label and you can explain any modern language model: BERT is the left half, GPT is the right half without the middle attention, and T5 is the whole thing.

The paper solves **sequence transduction**: map an input sequence $(x_1, \ldots, x_n)$ to an output sequence $(y_1, \ldots, y_m)$, as in English → German translation. Like earlier seq2seq models it has two halves. The **encoder** (left) reads the whole input **at once** and turns it into continuous representations $z = (z_1, \ldots, z_n)$, one vector per input token. The **decoder** (right) generates the output **one token at a time**, each step consuming the tokens it has already produced. The paper calls this **auto-regressive**. What is new is what's inside the boxes: stacked self-attention and point-wise fully connected layers, with no recurrence anywhere.

Read the figure from the bottom. On the left, the input tokens are embedded, a positional encoding is added (the ⊕ with the wave symbol), and the result goes through the grey encoder block, repeated **N×**. On the right, the target sentence **shifted right** is embedded the same way and goes through the grey decoder block, also repeated N×. The decoder block has one extra attention box in the middle, whose two left-hand inputs come from the encoder's output. At the top, a **Linear** layer and a **Softmax** turn the decoder's vectors into **Output Probabilities** for the next token. The chapters follow exactly this order.

Throughout the page, a **toy Transformer** built from these boxes (6-token inputs, $d_{model} = 64$, 4 heads, 2 layers, 168,128 parameters) is trained to "translate" a digit sequence into its reverse, using the paper's own optimizer, learning-rate schedule, dropout and label smoothing. It learns the task perfectly: 1,000 of 1,000 unseen sequences reversed exactly. That gives real numbers for every box, and two ablations show what breaks when you remove positional encoding or the decoder's mask.

![The Transformer - model architecture](/img/deep-learning/diagrams/transformer-figure-1.svg)

*How to read it:* our redraw of the paper's Figure 1, with every label word for word. Left grey box = one encoder layer (repeated N×), right grey box = one decoder layer (repeated N×). The lines that bend around each orange or blue box and enter an "Add & Norm" are the residual connections. The three short arrows under each "Multi-Head Attention" box are its Q, K and V inputs; in the decoder's middle attention, two of them come from the encoder.

**Try it yourself.** Click any box in the figure to see what it does, its formula, its shape in the paper's base model and in this page's toy model, and the section of the paper that describes it.

<Figure1Explorer />

| Figure 1 label | One-line meaning | Chapter |
|---|---|---|
| **Inputs** | Source token ids, all read at once | 2 |
| **Input Embedding** | Learned lookup token → $d_{model}$ vector, ×$\sqrt{d_{model}}$ | 3 |
| **Positional Encoding** (⊕) | Sinusoidal position vector added to each embedding | 4 |
| **Multi-Head Attention** (encoder) | Self-attention: every input token looks at every input token | 5 |
| **Add & Norm** | $\operatorname{LayerNorm}(x + \operatorname{Sublayer}(x))$: residual plus layer normalization | 6 |
| **Feed Forward** | Per-position two-layer MLP, 512 → 2048 → 512 | 7 |
| **N×** (encoder) | 6 identical layers stacked; output = memory for the decoder | 8 |
| **Outputs (shifted right)** | Target with a start token in front: input $y_{i-1}$, predict $y_i$ | 9 |
| **Output Embedding** + Positional Encoding | Same as the input side (shared weights) | 10 |
| **Masked Multi-Head Attention** | Decoder self-attention that can't see future positions | 11 |
| **Multi-Head Attention** (decoder) | Encoder–decoder attention: Q from decoder, K and V from encoder | 12 |
| Decoder **Add & Norm**, **Feed Forward**, **N×** | Same machinery, three sub-layers per layer | 13 |
| **Linear** | Decoder vector → one score per vocabulary token (tied to the embeddings) | 14 |
| **Softmax** → **Output Probabilities** | Scores → next-token distribution; decode step by step | 15 |

:::analogy[A translation office]
The encoder side is a team of readers who all read the whole English document at the same time, discuss it six rounds over, and pin a richly annotated copy on the wall. The decoder side is a writer producing the German version one word at a time. Before each word, the writer rereads what they've written so far (masked self-attention: no peeking at words not yet written), glances at the annotated English on the wall to find the relevant part (encoder–decoder attention), thinks (feed-forward), and picks the next word from a ranked list of candidates (Linear + Softmax).
:::

## 1. The Big Picture: an Encoder–Decoder Without Recurrence

Before the Transformer, the best translation systems were encoder–decoders built from RNNs or LSTMs (and some from convolutions), usually with an attention mechanism connecting the two halves. The paper keeps the **encoder–decoder structure** (paper §3) but replaces everything inside it: the encoder and decoder are stacks of self-attention and position-wise feed-forward layers. Its abstract describes the Transformer as based **solely** on attention, dispensing with recurrence and convolutions entirely.

Two consequences follow from that design, and both appear all over the figure. First, the encoder processes **all input positions in parallel**: there is no $h_t = f(h_{t-1}, x_t)$ chain, so a GPU can process the whole sentence at once. Second, with no recurrence and no convolution, the model has **no built-in notion of order**, so order must be injected explicitly (Positional Encoding, chapter 4). The decoder is still auto-regressive at **generation** time (it produces one token, feeds it back, produces the next), but during **training** it also runs all positions in parallel, thanks to the shifted-right outputs and the mask (chapters 9 and 11).

All sub-layers and embeddings produce vectors of the same width, $d_{model} = 512$ in the base model, so residual connections can add them. The stack depth is $N = 6$ for both encoder and decoder. Those two numbers, plus $h = 8$ heads and an FFN inner width of $d_{ff} = 2048$, define the base model (65 million parameters in the paper's Table 3).

| Hyperparameter | Base | Big | Toy model on this page |
|---|---|---|---|
| $N$ (layers per stack) | 6 | 6 | 2 |
| $d_{model}$ | 512 | 1024 | 64 |
| $d_{ff}$ | 2048 | 4096 | 128 |
| $h$ (heads) | 8 | 16 | 4 |
| $d_k = d_v$ | 64 | 64 | 16 |
| $P_{drop}$ | 0.1 | 0.3 | 0.1 |
| $\epsilon_{ls}$ (label smoothing) | 0.1 | 0.1 | 0.1 |
| Parameters | 65M | 213M | 168,128 |

:::tip[🎯 Interview angle]
**"Describe the Transformer architecture at a high level."** An encoder–decoder where both halves are stacks of $N = 6$ identical layers built from multi-head attention and position-wise feed-forward networks, each sub-layer wrapped as $\operatorname{LayerNorm}(x + \operatorname{Sublayer}(x))$. The encoder reads the whole input in parallel into one $d_{model}$-vector per token. The decoder generates auto-regressively, using masked self-attention over its own outputs and encoder–decoder attention over the encoder's vectors. Linear + Softmax gives next-token probabilities. No recurrence or convolution; order comes from added positional encodings.
:::

## 2. Inputs

At the very bottom left of Figure 1 is **Inputs**: the source sentence as a sequence of token ids. The paper used **byte-pair encoding** (BPE), which splits rare words into frequent sub-word pieces. For English–German it used a **shared source–target vocabulary of about 37,000 tokens** (§5.1); English–French used 32,000 word-pieces. A shared vocabulary is what makes the embedding sharing of chapter 3 possible.

The encoder receives **all** input tokens at once: a sentence of $n$ tokens is one tensor of $n$ ids (plus a batch dimension). Sentences of different lengths are padded to the same length within a batch and the padding is masked out in attention. The paper batched sentence pairs by approximate length, about 25,000 source and 25,000 target tokens per training batch.

In the toy model the input vocabulary is the 10 digits plus a start token (11 ids in all), and an input is six digits, for example [3, 1, 4, 1, 5, 9]. Its translation is the reverse, [9, 5, 1, 4, 1, 3].

:::tip[🎯 Interview angle]
**"What does the Transformer take as input, and why sub-word tokens?"** A sequence of token ids for the whole source sentence at once. The paper used byte-pair encoding with a shared ~37K English–German vocabulary: sub-words keep the vocabulary small while handling rare words, and sharing it between source and target lets one embedding matrix serve both sides and the output layer.
:::

## 3. Input Embedding

**Input Embedding** turns each token id into a vector: a learned lookup table $E \in \mathbb{R}^{V \times d_{model}}$, one row per vocabulary entry, so token $t$ becomes the row $E[t]$, a 512-dimensional vector. These rows are trained along with everything else, so tokens used in similar ways end up with similar vectors (the same idea as Word2Vec, but learned for the task).

Two details from §3.4 are easy to miss and popular in interviews. First, the paper **shares one weight matrix** between the Input Embedding, the Output Embedding and the pre-softmax **Linear** layer (weight tying, following Press and Wolf). With a 37,000 × 512 matrix that saves two copies of about 19 million parameters, and it ties the input and output meaning of each token together. Second, in the embedding layers the weights are **multiplied by $\sqrt{d_{model}}$** ($\sqrt{512} \approx 22.6$). Because the same matrix also serves as the output layer, it's kept at a small scale ($\sim 1/\sqrt{d_{model}}$ per entry is common); multiplying by $\sqrt{d_{model}}$ brings the embeddings up to unit scale, comparable to the positional encodings they're about to be added to, which lie between −1 and 1.

The toy model shows why the scale matters. With PyTorch's default embedding initialization (standard deviation 1), multiplying by $\sqrt{64} = 8$ and reusing the matrix as the output layer made the very first training loss **45.09**, far above the $\ln 11 = 2.40$ of a uniform guess. Initializing the shared matrix with standard deviation $d_{model}^{-1/2}$ instead gave a first loss of **4.99**, and training went smoothly.

<Formula name="Embedding with scaling" tag="Input / Output Embedding" words="Look up the token's learned row in the shared embedding matrix and multiply it by the square root of d_model." remember="Look up, then scale by root d.">

$$
\data{e_i} = \param{E}[\data{x_i}] \cdot \sqrt{\hyper{d_{model}}}, \qquad \param{E} \in \mathbb{R}^{V \times \hyper{d_{model}}}
$$

- $V$: vocabulary size (about 37,000, shared by source and target)
- $\param{E}$ is shared with the Output Embedding and the pre-softmax Linear layer

**Example:** toy model: $\param{E}$ is 11 × 64; [3, 1, 4, 1, 5, 9] becomes a 6 × 64 matrix, scaled by $\sqrt{64} = 8$.

</Formula>

:::tip[🎯 Interview angle]
**"Why multiply the embeddings by √d_model?"** The paper shares the embedding matrix with the output projection, which keeps its entries small. Multiplying by $\sqrt{d_{model}}$ rescales the looked-up vectors to roughly unit size, so the positional encodings (bounded in $[-1, 1]$) don't swamp them. And it reduces parameters by sharing one $V \times d_{model}$ matrix across input embedding, output embedding and pre-softmax Linear.
:::

## 4. Positional Encoding (and the ⊕)

Self-attention treats its input as a **set**: if you shuffle the tokens, every token gets the same output as before, just in a different place (shown on the first page of this section). Without extra information, "dog bites man" and "man bites dog" would look identical. Since the model has no recurrence and no convolution, the paper **injects position information** by **adding** a positional encoding to every embedding at the bottom of both stacks (§3.5). That's the ⊕ in Figure 1. The encodings have the same width $d_{model}$ as the embeddings, so the two can be summed.

The paper's encoding uses sines and cosines of different frequencies: each pair of dimensions $(2i, 2i+1)$ is a sine and a cosine of the same angle $pos / 10000^{2i/d_{model}}$. Low dimensions oscillate fast (dimension 0 has wavelength $2\pi \approx 6.3$ positions), high dimensions slowly; the wavelengths form a geometric progression from $2\pi$ to $10000 \cdot 2\pi$. Like the digits of a clock (seconds, minutes, hours), the fast dimensions distinguish neighbouring positions and the slow ones give coarse position. Each position gets a unique vector, every vector has the same length ($\sqrt{d_{model}/2} = 16$ for 512), and $PE(0)$ is simply $[0, 1, 0, 1, \ldots]$.

The paper chose this form because of a specific property: for any fixed offset $k$, $PE_{pos+k}$ is a **linear function** of $PE_{pos}$. Each sine–cosine pair is a point on a circle, and moving $k$ positions rotates that point by a fixed angle $\omega_i k$, which is a $2 \times 2$ rotation matrix that doesn't depend on $pos$. Block 6 verifies it (error $6 \times 10^{-15}$) and shows the consequence: the dot product $PE_{pos} \cdot PE_{pos+k}$ depends only on the distance $k$ (249.10 for $k = 1$ and 189.60 for $k = 5$, whether $pos$ is 10 or 20). The authors hypothesized this would make it easy to attend by **relative** position. They also tried **learned** positional embeddings and found nearly identical results (Table 3, row E: 25.7 vs 25.8 BLEU), and chose the sinusoids because they **may extrapolate** to sequences longer than those seen in training.

The toy model shows how essential this box is. Trained identically but **without** positional encoding, it gets only **25.2%** of output tokens right and **0.5%** of sequences exactly reversed (vs 100% with it), because reversing a sequence is impossible if you can't tell the positions apart.

<Formula name="Sinusoidal positional encoding (paper §3.5)" tag="Positional Encoding" words="Each pair of dimensions is a sine and a cosine of the position divided by a frequency that shrinks geometrically across the pairs; the vector is added to the embedding." remember="Clock hands at different speeds.">

$$
\begin{aligned}
PE_{(pos,\, 2i)} &= \sin\!\left(pos / 10000^{2i/\hyper{d_{model}}}\right) \\
PE_{(pos,\, 2i+1)} &= \cos\!\left(pos / 10000^{2i/\hyper{d_{model}}}\right) \\
\data{x} &= \data{e_{pos}} + PE_{pos}
\end{aligned}
$$

- $pos$: token position (0, 1, 2, …); $i$: dimension pair index; wavelengths $2\pi$ to $10000 \cdot 2\pi$
- no parameters; the same encoding is added on the encoder and decoder sides

**Example:** $d_{model} = 512$: $PE_1$ starts [0.8415, 0.5403, 0.8219, 0.5697, 0.8020, 0.5974]; $\lVert PE_{pos} \rVert^2 = 256$ for every $pos$.

</Formula>

<Formula name="Shift property" tag="Why sinusoids" words="Moving k positions rotates every sine-cosine pair by a fixed angle, so the encoding at pos plus k is a fixed linear map of the encoding at pos, the same for every pos." remember="Shifting = rotating each pair.">

$$
\begin{aligned}
\begin{bmatrix} \sin \omega_i(pos + k) \\ \cos \omega_i(pos + k) \end{bmatrix} &= R(\omega_i k) \begin{bmatrix} \sin \omega_i\, pos \\ \cos \omega_i\, pos \end{bmatrix} \\
R(\theta) &= \begin{bmatrix} \cos \theta & \sin \theta \\ -\sin \theta & \cos \theta \end{bmatrix} \\
\omega_i &= 10000^{-2i/\hyper{d_{model}}}
\end{aligned}
$$

- the matrix depends on $k$ only, not on $pos$, so $PE_{pos+k} = R_k\, PE_{pos}$
- consequence: $PE_{pos} \cdot PE_{pos+k}$ depends only on $k$

**Example:** $d_{model} = 512$: $PE_{10} \cdot PE_{11}$ = $PE_{20} \cdot PE_{21}$ = 249.10; with $k = 5$ both are 189.60.

</Formula>

<Animation src="transformer/positional-encoding" caption="Four dimensions of an 8-dimensional positional encoding as waves over position. Watch the fast wave separate neighbouring positions while the slow ones barely move, and read the full encoding vector off the vertical line as it moves." />

**Try it yourself.** Move the position and switch $d_{model}$. Watch the vector change, see the waves of different speeds, and confirm that the similarity curve (and the two dot products) don't depend on the position at all.

<PositionalEncodingExplorer />

:::danger[Interview Trap: "Positional encodings are concatenated to the embeddings"]
They are **added** (the ⊕), not concatenated: both are $d_{model}$ wide, so the sum keeps the width and adds no parameters. The model learns to keep token content and position in roughly separable directions. Also, the original encodings are **fixed**, not learned; BERT and GPT-2 later used learned position embeddings, and many recent models use relative schemes such as rotary embeddings.
:::

:::tip[🎯 Interview angle]
**"Why does the Transformer need positional encodings, and why sinusoids?"** Self-attention is permutation-equivariant, so without position information the order of words is invisible (our toy model's reversal accuracy drops from 100% to 0.5% without it). The paper adds sinusoids of geometrically spaced frequencies: unique per position, bounded, parameter-free, and $PE_{pos+k}$ is a rotation of $PE_{pos}$, which should make relative positions easy to attend to. Learned embeddings scored about the same (25.7 vs 25.8 BLEU); sinusoids were kept because they may extrapolate to longer sequences.
:::

## 5. Multi-Head Attention in the Encoder (Self-Attention)

The first box inside the encoder block is **Multi-Head Attention**, exactly the block from the previous page. Look at its three input arrows: they branch from the **same** line. That's what makes it **self-attention**: queries, keys and values all come from the same place, the output of the previous encoder layer (or the embeddings, for the first layer). Every position in the encoder can attend to **every** position of the input, in both directions, and nothing is masked (except padding).

Inside, the input $x$ ($n \times 512$) is projected by 8 heads into queries, keys and values of size 64, each head computes $\operatorname{softmax}(QK^\top/\sqrt{64})V$, the 8 outputs are concatenated back to 512, and $W^O$ mixes them. The output has the same shape as the input, $n \times 512$. This is the step where tokens exchange information: after it, each position's vector contains information gathered from the whole sentence.

The paper's appendix visualizes what trained encoder heads do: in layer 5 of 6, many heads link the verb "making" to the distant words that complete the phrase "making … more difficult", and two heads seem to do **anaphora resolution**, linking "its" to the noun it refers to. Different heads learn different jobs.

<Formula name="Encoder self-attention" tag="Multi-Head Attention (encoder)" words="Queries, keys and values are all the previous layer's output, so every input position attends to every input position." remember="Q = K = V = previous layer.">

$$
\begin{aligned}
\operatorname{MHA}(\data{x}, \data{x}, \data{x}) &= \operatorname{Concat}(\pred{\text{head}_1}, \ldots, \pred{\text{head}_h})\param{W^O} \\
\pred{\text{head}_i} &= \operatorname{softmax}\!\left(\frac{\data{x}\param{W_i^Q}(\data{x}\param{W_i^K})^\top}{\sqrt{\hyper{d_k}}}\right)\data{x}\param{W_i^V}
\end{aligned}
$$

- MHA = MultiHead; $\data{x}$: the previous encoder layer's output ($n \times \hyper{d_{model}}$); $\hyper{h} = 8$, $\hyper{d_k} = 64$
- no mask (apart from padding): attention looks both left and right

**Example:** toy model: 6 × 64 in, 4 heads of 16, 6 × 64 out.

</Formula>

:::tip[🎯 Interview angle]
**"What kind of attention does the encoder use?"** Bidirectional multi-head self-attention: Q, K and V are all the previous layer's output, so each position attends to all positions, left and right, with no causal mask (only padding is masked). It's where tokens exchange information; the output keeps the $n \times d_{model}$ shape.
:::

## 6. Add & Norm

Every sub-layer in Figure 1 is followed by an **Add & Norm** box, and every one has a line bending around the sub-layer into it. That line is the **residual connection** (He et al.'s ResNet idea): the sub-layer's input $x$ is **added** to its output. Then **layer normalization** normalizes the sum. The paper writes the output of each sub-layer as $\operatorname{LayerNorm}(x + \operatorname{Sublayer}(x))$ (§3.1), and adds one more detail in §5.4: **dropout** ($P_{drop} = 0.1$) is applied to the sub-layer's output **before** it is added. So in full: $\operatorname{LayerNorm}(x + \operatorname{Dropout}(\operatorname{Sublayer}(x)))$.

**Why the residual (Add).** It gives information and gradients a direct path through the network. A sub-layer only has to learn a **correction** to its input, not a whole new representation, and the gradient flows back through the "+ x" path unchanged, so stacks of 6 (or 100) layers still train. It's also why every sub-layer and embedding must output exactly $d_{model} = 512$: you can only add vectors of equal size. (This is why multi-head attention ends with $W^O$ mapping back to 512, and why the FFN projects back down from 2048.)

**Why layer normalization (Norm).** LayerNorm (Ba et al., 2016) takes **one position's** vector of $d_{model}$ numbers, subtracts its mean and divides by its standard deviation, then applies a learned per-feature gain $\gamma$ and bias $\beta$. It keeps activations on a stable scale layer after layer, which makes optimization steadier. It normalizes across the **features of one token**, not across the batch like BatchNorm, so it doesn't depend on batch size or on other sequences of different lengths in the batch. In the notebook's encoder block every row of the output had mean 0 and variance 1 (with $\gamma = 1$, $\beta = 0$).

![Inside one Add & Norm (post-LN, as in the paper)](/img/deep-learning/diagrams/add-and-norm.svg)

*How to read it:* the input $x$ goes through the sub-layer (multi-head attention or feed-forward), dropout is applied, and the result is added to the untouched $x$ that came around the top (the residual). The sum is layer-normalized per position. The output has the same shape as $x$, ready for the next sub-layer.

<Formula name="Add & Norm (paper §3.1, §5.4)" tag="Residual + LayerNorm" words="Add the sub-layer's output (after dropout) to its input, then normalize each position's vector to zero mean and unit variance and apply a learned gain and bias." remember="Keep x, add the update, normalize.">

$$
\begin{aligned}
\pred{y} &= \operatorname{LN}\big(\data{x} + \operatorname{Dropout}(F(\data{x}))\big) \\
\operatorname{LN}(\data{v}) &= \param{\gamma} \odot \frac{\data{v} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \param{\beta} \\
\mu &= \tfrac{1}{d}\textstyle\sum_j v_j, \qquad \sigma^2 = \tfrac{1}{d}\textstyle\sum_j (v_j - \mu)^2
\end{aligned}
$$

- $F$ = the sub-layer (multi-head attention or feed-forward); LN = LayerNorm
- statistics over the $\hyper{d_{model}}$ features of **one** position; $\param{\gamma}, \param{\beta}$ learned per feature
- dropout $\hyper{P_{drop}} = 0.1$ on the sub-layer output before the addition (base model)

**Example:** notebook encoder block: rows of $\pred{y}$ have mean 0.000 and variance 1.000.

</Formula>

:::danger[Interview Trap: "Post-LN vs Pre-LN"]
The paper puts LayerNorm **after** the residual addition (**Post-LN**): $\operatorname{LayerNorm}(x + \operatorname{Sublayer}(x))$. Most modern large models (GPT-2 onwards) use **Pre-LN**: $x + \operatorname{Sublayer}(\operatorname{LayerNorm}(x))$, plus a final LayerNorm, because it trains more stably in deep stacks and depends less on learning-rate warmup. If asked "where is the LayerNorm in a Transformer?", say which variant you mean.
:::

:::tip[🎯 Interview angle]
**"What does Add & Norm do, and why both parts?"** Add is a residual connection: $x + \operatorname{Sublayer}(x)$ lets each sub-layer learn a correction and gives gradients an identity path, so deep stacks train; it requires every sub-layer to output $d_{model}$. Norm is LayerNorm over each position's features (mean 0, variance 1, then learned $\gamma$, $\beta$), keeping activations stable without depending on the batch. The paper also applies dropout 0.1 to the sub-layer output before adding (Post-LN; modern models often use Pre-LN).
:::

## 7. Feed Forward

The second sub-layer of every encoder (and decoder) layer is **Feed Forward**: a small fully connected network applied to **each position separately and identically** (§3.3). It's two linear transformations with a ReLU between them: $\operatorname{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$. The input and output are $d_{model} = 512$ wide; the hidden layer is $d_{ff} = 2048$, four times wider. The same weights are used at every position within a layer, but each layer has its own. The paper notes another way to see it: two **convolutions with kernel size 1**.

The division of labour with attention is the key idea, and your notebook says it well: **attention is communication across tokens; the feed-forward network is transformation within each token**. Attention mixes information between positions but is mostly a weighted average (linear in the values); the FFN, with its ReLU, adds the non-linear processing that turns gathered information into new features. Applied per position, it has no idea which other tokens exist; it just transforms whatever the attention step wrote into this position's vector.

The FFN holds most of the parameters: per layer, $2 \times 512 \times 2048 + 2048 + 512 = 2{,}099{,}712$ weights and biases, twice the multi-head attention's 1,050,624. Table 3 (C) shows it matters: $d_{ff} = 1024$ scored 25.4 BLEU, 2048 (base) 25.8, and 4096 26.2.

<Formula name="Position-wise feed-forward network (paper eq. 2)" tag="Feed Forward" words="Two linear layers with a ReLU between them, applied to every position's vector separately with the same weights: widen to 2048, keep the positives, project back to 512." remember="Widen, ReLU, narrow, per token.">

$$
\begin{aligned}
\operatorname{FFN}(\data{x}) &= \max(0,\, \data{x}\param{W_1} + \param{b_1})\,\param{W_2} + \param{b_2} \\
\param{W_1} &\in \mathbb{R}^{\hyper{d_{model}} \times \hyper{d_{ff}}}, \quad \param{W_2} \in \mathbb{R}^{\hyper{d_{ff}} \times \hyper{d_{model}}}
\end{aligned}
$$

- base model: $\hyper{d_{model}} = 512$, $\hyper{d_{ff}} = 2048$; same weights at every position, different per layer
- equivalent to two 1 × 1 convolutions over the sequence

**Example:** toy model 64 → 128 → 64 (16,576 parameters per layer); notebook block 4 → 8 → 4.

</Formula>

:::tip[🎯 Interview angle]
**"What is the feed-forward layer for, if attention already mixes tokens?"** Attention moves information between positions (largely a weighted average); the FFN transforms each position's vector on its own, adding the non-linear computation. It's $\max(0, xW_1 + b_1)W_2 + b_2$, 512 → 2048 → 512, shared across positions, and holds about two thirds of each encoder layer's parameters. Bigger $d_{ff}$ helped in the paper (4096: 26.2 vs 2048: 25.8 BLEU).
:::

## 8. N×: Stacking the Encoder

The grey box on the left is **one encoder layer**: Multi-Head Attention → Add & Norm → Feed Forward → Add & Norm. The **N×** beside it means $N$ identical layers are stacked, each with its own weights, the output of one being the input of the next. The paper uses $N = 6$. Because every layer maps $n \times d_{model}$ to $n \times d_{model}$, they stack cleanly (the notebook's last section called this "shape-preserving").

Depth lets representations become more abstract layer by layer: early layers tend to capture local and surface relations, later ones longer-range and more semantic structure (the paper's attention examples come from layer 5 of 6). Table 3 (C) measured depth: $N = 2$ scored 23.7 BLEU, $N = 4$ 25.3, $N = 6$ 25.8, $N = 8$ 25.5.

The output of the **last** encoder layer, $z = (z_1, \ldots, z_n)$, is the encoder's final product: one contextual vector per input token. It's computed **once** per input sentence and then read by **every** decoder layer (the line leaving the top of the encoder in Figure 1). Implementations often call it the "memory".

<Formula name="Encoder layer, stacked N times" tag="N× (encoder)" words="Each encoder layer wraps self-attention and then a feed-forward network in Add & Norm; six of them are applied one after another and the last output is the memory." remember="Attend, add-norm, transform, add-norm; ×6.">

$$
\begin{aligned}
\pred{u} &= \operatorname{LN}(\data{x} + \operatorname{MHA}(\data{x}, \data{x}, \data{x})) \\
\operatorname{Enc}(\data{x}) &= \operatorname{LN}(\pred{u} + \operatorname{FFN}(\pred{u})) \\
\pred{z} &= \operatorname{Enc}_{\hyper{N}} \circ \cdots \circ \operatorname{Enc}_1(\data{x})
\end{aligned}
$$

- LN = LayerNorm, MHA = MultiHead, Enc = EncoderLayer; $\hyper{N} = 6$, each layer with its own weights; all shapes $n \times \hyper{d_{model}}$
- $\pred{z}$ (the memory) is computed once and used by every decoder layer

**Example:** toy model: 2 encoder layers of 33,472 parameters each; memory shape 6 × 64.

</Formula>

:::tip[🎯 Interview angle]
**"What does the encoder output, and how is it used?"** After $N = 6$ identical layers, one contextual $d_{model}$-vector per input token, $z_1 \ldots z_n$. It's computed once per input and serves as the keys and values of the encoder–decoder attention in every decoder layer. The layers can be stacked because each preserves the $n \times d_{model}$ shape.
:::

## 9. Outputs (shifted right)

Now the right half, again from the bottom. The decoder's input is labelled **Outputs (shifted right)**: the target sentence moved one position to the right, with a start token in front. If the target is $(y_1, \ldots, y_m)$, the decoder reads $(\langle s \rangle, y_1, \ldots, y_{m-1})$ and is trained to output $(y_1, \ldots, y_m)$. At position $i$ it has seen the tokens **before** $y_i$ and must predict $y_i$ itself.

Why shift? Because the decoder is a **next-token predictor**. The paper (§3.1) explains that the masking in the decoder, combined with the fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$. During **training** the true previous tokens are fed in (**teacher forcing**), and all $m$ positions are trained **in parallel** in one forward pass: the mask (chapter 11) makes sure no position cheats by looking at its own answer. During **generation** there is no target: the decoder starts with only $\langle s \rangle$, predicts $y_1$, appends it, predicts $y_2$, and so on (chapter 15).

For the toy example the target is [9, 5, 1, 4, 1, 3], so the decoder input is [BOS, 9, 5, 1, 4, 1]: at position 1 it sees only BOS and must output 9; at position 2 it has seen 9 and must output 5; and so on.

![Outputs (shifted right)](/img/deep-learning/diagrams/shifted-right.svg)

*How to read it:* bottom row = what the decoder reads (the target shifted right, starting with BOS); top row = what it must predict at each position. Each red curve shows a target token reappearing as the next position's input. Position $i$ never sees its own target.

:::tip[🎯 Interview angle]
**"What does 'shifted right' mean in the Transformer decoder?"** The decoder input is the target sequence with a start token prepended and the last token dropped, so at position $i$ it reads $y_{i-1}$ and is trained to predict $y_i$. Combined with the causal mask, this lets training compute all positions in parallel with teacher forcing while each prediction only depends on earlier tokens, exactly the conditions it will face when generating.
:::

## 10. Output Embedding and Positional Encoding

The **Output Embedding** does for target tokens what the Input Embedding does for source tokens: each id becomes a $d_{model}$ vector, multiplied by $\sqrt{d_{model}}$. In the paper it's the **same** weight matrix as the Input Embedding (and the pre-softmax Linear), which works because source and target share one BPE vocabulary.

The decoder's **Positional Encoding** is identical to the encoder's: the same sinusoids are added (the ⊕ on the right of Figure 1), so the decoder knows the order of the tokens it has produced. The paper also applies **dropout** ($P_{drop} = 0.1$) to the sums of embeddings and positional encodings in both stacks (§5.4).

:::tip[🎯 Interview angle]
**"Are the encoder and decoder embeddings different?"** In the paper, no: one matrix serves as Input Embedding, Output Embedding and the pre-softmax Linear layer (enabled by a shared BPE vocabulary), always scaled by $\sqrt{d_{model}}$ in the embedding layers. Positional encodings are the same sinusoids on both sides, and dropout 0.1 is applied to embedding + position sums.
:::

## 11. Masked Multi-Head Attention

The first sub-layer of every decoder layer is **Masked Multi-Head Attention**: self-attention over the decoder's own tokens, with one change. A position may only attend to positions **up to and including itself** (§3.2.3). Without that, during parallel training, position 2 could attend to position 3's input, which is the very token position 2 is supposed to predict. The model would learn to copy instead of predict, and fail at generation time when the future doesn't exist yet. The paper calls this preventing leftward information flow, to preserve the **auto-regressive** property.

The mask is implemented inside scaled dot-product attention (the "Mask (opt.)" box of Figure 2): before the softmax, every score for an illegal connection ($j > i$) is set to $-\infty$. Since $e^{-\infty} = 0$, those positions get exactly zero weight, and the softmax redistributes the weight over the allowed (past) positions. The resulting attention matrix is **lower-triangular**, and every row still sums to 1. In the toy model's last decoder layer, row 1 (BOS) has weight 1.00 on itself, row 2 splits 0.76/0.24 over the first two positions, and every entry above the diagonal is exactly 0.

The toy ablation makes the point concrete. Trained the same way but **without** the mask, the model gets **100%** of tokens right during teacher-forced training (it simply reads the answer from the next position), but generating on its own it gets only **53%** of tokens and **0.1%** of sequences right, against 100% with the mask.

<Animation src="transformer/causal-mask" caption="The decoder's 6 by 6 self-attention grid for the input BOS 9 5 1 4 1. Watch every future position receive minus infinity, then the trained model's real weights fill the lower triangle, each row summing to 1 over the visible past." />

<Formula name="Causal (look-ahead) mask" tag="Masked Multi-Head Attention" words="Add minus infinity to every score where the key comes after the query, so after softmax each position only attends to itself and earlier positions." remember="No peeking at the future.">

$$
\begin{aligned}
\operatorname{Attention}(\pred{Q}, \pred{K}, \pred{V}) &= \operatorname{softmax}\!\left(\frac{\pred{Q}\pred{K}^\top}{\sqrt{\hyper{d_k}}} + M\right)\pred{V} \\
M_{ij} &= \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}
\end{aligned}
$$

- $e^{-\infty} = 0$, so future weights are exactly zero; rows still sum to 1; $\pred{A}$ is lower-triangular
- applied inside every head of the decoder's first attention sub-layer

**Example:** toy model, last decoder layer, row 2 = [0.76, 0.24, 0, 0, 0, 0]; row 3 = [0.15, 0.48, 0.37, 0, 0, 0].

</Formula>

:::danger[Interview Trap: "The mask is there so the decoder generates one token at a time"]
At **generation** time there is nothing in the future to hide; the decoder only has its past tokens anyway. The mask matters during **training**, when the whole shifted target is fed in at once: it stops each position from attending to the tokens it must predict. That's what makes parallel training and step-by-step generation consistent.
:::

:::tip[🎯 Interview angle]
**"Why does the decoder's self-attention need a mask, and how is it implemented?"** During training the decoder sees the whole shifted target in parallel; without a mask, position $i$ could attend to later positions and copy the answer, then fail at generation time (our toy: 100% teacher-forced but 0.1% generated sequences without the mask). Scores for $j > i$ are set to $-\infty$ before softmax, so those weights become exactly 0 and the attention matrix is lower-triangular.
:::

## 12. Multi-Head Attention in the Decoder: Encoder–Decoder Attention

The middle sub-layer of each decoder layer is the second **Multi-Head Attention** box, and its three inputs tell you what it does. Two arrows come from the **encoder output** (the line from the left side of Figure 1); one comes from **below**, the decoder's own previous sub-layer. The paper (§3.2.3): in encoder–decoder attention layers, the **queries** come from the previous decoder layer, and the memory **keys and values** come from the output of the encoder. Every decoder position can attend over **all** input positions.

This is the Transformer's version of the classic seq2seq attention of Bahdanau et al. (first page of this section): the decoder, while producing output token $i$, asks "which source tokens matter for this word?" and gathers their encoder vectors. No mask is needed here: the whole source sentence is known. The score matrix is $m \times n$ (target positions by source positions), so it needn't be square.

In the toy model this box does exactly what reversal requires. In the last decoder layer, output position 1 puts weight **1.00** on input position 6, output 2 on input 5, and so on: a perfect **anti-diagonal**. The model learned the alignment "output $i$ = input $7 - i$" entirely from examples. It's the same thing translation models learn when they align "eau" with "water".

<Formula name="Encoder–decoder attention" tag="Multi-Head Attention (decoder)" words="Queries come from the decoder's current state, keys and values from the encoder's final output, so each output position gathers information from the relevant input positions." remember="Decoder asks, encoder answers.">

$$
\begin{aligned}
\operatorname{MHA}(\pred{y}, \pred{z}, \pred{z}) &= \operatorname{Concat}(\pred{\text{head}_1}, \ldots, \pred{\text{head}_h})\param{W^O} \\
\pred{\text{head}_i} &= \operatorname{softmax}\!\left(\frac{\pred{y}\param{W_i^Q}(\pred{z}\param{W_i^K})^\top}{\sqrt{\hyper{d_k}}}\right)\pred{z}\param{W_i^V}
\end{aligned}
$$

- MHA = MultiHead; $\pred{y}$: decoder state after masked attention + Add & Norm ($m \times \hyper{d_{model}}$); $\pred{z}$: encoder output ($n \times \hyper{d_{model}}$)
- scores are $m \times n$; no causal mask (the whole input is known)

**Example:** toy model (reverse 6 digits), last decoder layer: row $i$ puts weight 1.00 on input position $7 - i$, an exact anti-diagonal.

</Formula>

:::tip[🎯 Interview angle]
**"In the decoder's second attention, where do Q, K and V come from?"** Queries from the decoder (the output of the masked self-attention sub-layer), keys and values from the final encoder output, the same memory for every decoder layer. It lets each output position attend to all input positions: the learned alignment, like classic seq2seq attention. The score matrix is target length × source length, and only padding is masked.
:::

## 13. The Rest of the Decoder Layer, and N× Again

After the encoder–decoder attention comes another **Add & Norm**, then a **Feed Forward** network (same form as the encoder's, own weights), then a final **Add & Norm**. So a decoder layer has **three** sub-layers, each wrapped in Add & Norm: masked self-attention, encoder–decoder attention, feed-forward. The encoder layer has two.

The decoder block is also repeated **N×** ($N = 6$), and every decoder layer's middle attention reads the **same** final encoder output. Per layer, the decoder has one more attention block than the encoder: 4,204,032 parameters vs 3,152,384 in the base model (our toy: 50,240 vs 33,472).

| | Encoder layer | Decoder layer |
|---|---|---|
| Sub-layers | 2 | 3 |
| Self-attention | Bidirectional (no mask) | Masked (causal) |
| Encoder–decoder attention | — | Q from decoder, K and V from encoder |
| Feed Forward | 512 → 2048 → 512 | 512 → 2048 → 512 |
| Add & Norm boxes | 2 | 3 |
| Parameters (base) | 3,152,384 | 4,204,032 |

<Formula name="Decoder layer, stacked N times" tag="N× (decoder)" words="Masked self-attention, then attention over the encoder output, then a feed-forward network, each wrapped in Add and Norm; six such layers are stacked." remember="Look back, look across, think.">

$$
\begin{aligned}
\pred{a} &= \operatorname{LN}(\data{y} + \operatorname{MHA}_{mask}(\data{y}, \data{y}, \data{y})) \\
\pred{b} &= \operatorname{LN}(\pred{a} + \operatorname{MHA}(\pred{a}, \pred{z}, \pred{z})) \\
\operatorname{Dec}(\data{y}, \pred{z}) &= \operatorname{LN}(\pred{b} + \operatorname{FFN}(\pred{b}))
\end{aligned}
$$

- LN = LayerNorm, MHA = MultiHead ($\operatorname{MHA}_{mask}$ with the causal mask), Dec = DecoderLayer
- $\data{y}$: previous decoder layer output (or shifted embeddings); $\pred{z}$: final encoder output
- $\hyper{N} = 6$ layers; all shapes $m \times \hyper{d_{model}}$

**Example:** toy model: 2 decoder layers of 50,240 parameters each.

</Formula>

:::tip[🎯 Interview angle]
**"How does a decoder layer differ from an encoder layer?"** It has three sub-layers instead of two: masked self-attention over the outputs so far, encoder–decoder attention (Q from the decoder, K and V from the encoder output), and the feed-forward network, each followed by Add & Norm. Every decoder layer attends to the same final encoder output.
:::

## 14. Linear

Above the decoder stack, **Linear** turns each decoder output vector ($d_{model} = 512$) into one **score per vocabulary token** (about 37,000), the logits. It's a single matrix multiply, and in the paper it **shares its weight matrix with the embeddings** (§3.4): the score for token $t$ is the dot product of the decoder's vector with token $t$'s embedding row. A token scores highly when the decoder's output vector points in the same direction as that token's embedding.

Weight tying saves parameters (one 37,000 × 512 matrix instead of three) and ties input and output meanings together. In the toy model this is literally `logits = y @ embed.weight.T`, which turns a 6 × 64 decoder output into 6 × 11 logits.

<Formula name="Pre-softmax Linear (tied)" tag="Linear" words="Project each decoder output vector onto every vocabulary token's embedding to get one score per token." remember="Score = similarity to each token's embedding.">

$$
\begin{aligned}
\pred{\text{logits}} &= \pred{y_{dec}}\,\param{E}^\top \in \mathbb{R}^{m \times V} \\
\param{E} &\in \mathbb{R}^{V \times \hyper{d_{model}}} \text{ (the shared embedding matrix)}
\end{aligned}
$$

- one score per vocabulary token per output position
- base: 512 → 37,000; toy: 64 → 11

**Example:** toy model: 6 × 64 decoder output → 6 × 11 logits.

</Formula>

:::tip[🎯 Interview angle]
**"What is the Linear layer at the top, and what is weight tying?"** It maps each $d_{model}$-dimensional decoder output to $V$ logits, one per vocabulary token. The paper ties its weight matrix to the input and output embedding matrices, so the logit for token $t$ is the dot product with $t$'s embedding: fewer parameters, and consistent token representations at input and output.
:::

## 15. Softmax and Output Probabilities

**Softmax** turns each position's logits into a probability distribution over the vocabulary: positive, summing to 1. These are the **Output Probabilities**: at position $i$, the model's distribution for the next token $y_i$ given the input sentence and the outputs before $i$.

**In training**, the loss is the cross-entropy between these probabilities and the true next tokens, at all positions at once. The paper uses **label smoothing** with $\epsilon_{ls} = 0.1$ (§5.4): instead of a target of 1 for the true token and 0 for the rest, it spreads $\epsilon$ over the vocabulary. In the paper's words, this hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU. The toy model shows it: even though it's 100% accurate, it puts only about **0.92** on each correct token. With 11 classes, the smoothed target is $1 - 0.1 + 0.1/11 = 0.909$.

**In generation**, the decoder runs **auto-regressively**: encode the input once; start the decoder with $\langle s \rangle$; take the output probabilities at the last position; choose a token; append it; repeat until an end token or a length limit. The simplest choice is **greedy** (take the most probable token). The paper used **beam search** with beam size 4 and length penalty $\alpha = 0.6$, a maximum output length of input length + 50, and stopped early when possible (§6.1). The toy model decodes greedily: from BOS it predicts 9 (p = 0.918), then 5, 1, 4, 1, 3, each step attending (weight 1.00) to the mirrored input position.

<Animation src="transformer/decoding" caption="The trained toy model reversing 3 1 4 1 5 9. Each step the newest output position attends to one input digit through encoder-decoder attention, predicts the next digit with probability about 0.92, and appends it to the decoder input." />

<Formula name="Output distribution and training loss" tag="Softmax → Output Probabilities" words="Softmax turns each position's scores into next-token probabilities; training maximizes the probability of the true next token, with label smoothing spreading a little probability to every token." remember="Predict next, all positions at once.">

$$
\begin{aligned}
P(\data{y_i} = t \mid \data{x}, \data{y_{<i}}) &= \operatorname{softmax}(\pred{\text{logits}_i})_t \\
\loss{L} &= -\sum_i \sum_t q_i(t) \log P(t \mid \data{x}, \data{y_{<i}}) \\
q_i(t) &= (1 - \hyper{\epsilon_{ls}})\,[t = \data{y_i}] + \hyper{\epsilon_{ls}}/V
\end{aligned}
$$

- $\hyper{\epsilon_{ls}} = 0.1$; without smoothing $q_i$ is one-hot (plain cross-entropy)
- generation: greedy or beam search (paper: beam 4, length penalty 0.6)

**Example:** toy model, 11 tokens: smoothed target 0.909; the trained model gives the correct digit about 0.92.

</Formula>

:::tip[🎯 Interview angle]
**"How does the Transformer generate a translation?"** Encode the source once. Start the decoder with a start token, compute the softmax at the last position, pick a token (greedy, or beam search: the paper used beam 4, length penalty 0.6), append it, and repeat until an end token or input length + 50. Training instead feeds the whole shifted target with the causal mask and computes cross-entropy (with label smoothing 0.1) at all positions in parallel.
:::

## 16. Why Self-Attention (Paper §4)

Section 4 of the paper justifies the design by comparing self-attention, recurrent and convolutional layers on three criteria: the **total computation per layer**; how much can be **parallelized**, measured as the minimum number of **sequential operations**; and the **path length** between long-range dependencies, the number of steps a signal must travel between any two positions. Shorter paths make long-range dependencies easier to learn.

| Layer type | Complexity per layer | Sequential operations | Maximum path length |
|---|---|---|---|
| **Self-Attention** | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| **Recurrent** | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| **Convolutional** | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k(n))$ |
| **Self-Attention (restricted)** | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n/r)$ |

Self-attention connects every pair of positions with a constant number of sequential operations, while a recurrent layer needs $O(n)$. It is also **cheaper** than a recurrent layer when the sequence length $n$ is smaller than the representation size $d$, which is typical for sentences (word-piece sentences are much shorter than 512). For very long sequences the paper suggests **restricted** self-attention over a neighbourhood of size $r$, which costs $O(r \cdot n \cdot d)$ but lengthens paths to $O(n/r)$. A single convolution with kernel $k < n$ doesn't connect all pairs; it takes $O(n/k)$ stacked layers (or $O(\log_k n)$ with dilation). As a side benefit, self-attention can be more interpretable: individual heads appear to learn syntactic and semantic tasks (the appendix figures).

:::tip[🎯 Interview angle]
**"Why did the authors choose self-attention? Give the complexities."** Per layer: self-attention $O(n^2 d)$ with $O(1)$ sequential steps and $O(1)$ path length; recurrent $O(n d^2)$, $O(n)$, $O(n)$; convolution $O(k n d^2)$, $O(1)$, $O(\log_k n)$. Self-attention is the most parallel and has the shortest paths, and it's cheaper than recurrence whenever $n < d$, typical for sentences. The price is quadratic cost in $n$; restricted attention ($O(r n d)$, path $O(n/r)$) was proposed for long inputs.
:::

## 17. Training (Paper §5)

**Data and batching (§5.1).** WMT 2014 English–German: about 4.5 million sentence pairs, byte-pair encoded with a shared vocabulary of about 37,000 tokens. English–French: 36 million sentences, 32,000 word-pieces. Pairs were batched by approximate length, about 25,000 source and 25,000 target tokens per batch.

**Hardware and schedule (§5.2).** One machine with 8 NVIDIA P100 GPUs. Base model: about 0.4 s per step, 100,000 steps, **12 hours**. Big model: 1.0 s per step, 300,000 steps, **3.5 days**.

**Optimizer and learning rate (§5.3).** Adam with $\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$ (a larger $\beta_2$ and a smaller $\epsilon$ than Adam's defaults of 0.999 and $10^{-8}$). The learning rate follows a **warmup** schedule: it increases linearly for the first `warmup_steps` = 4,000 steps, then decreases proportionally to the inverse square root of the step number. With $d_{model} = 512$ it peaks at 0.000699 at step 4,000 and falls to 0.000349 by step 16,000 and 0.000140 at step 100,000. Warmup matters for Post-LN Transformers: early in training, when the Adam statistics are unreliable and the network is untrained, large updates destabilize training. The toy model uses the same formula with $d_{model} = 64$ and 400 warmup steps.

**Regularization (§5.4).** **Residual dropout**: dropout on each sub-layer's output before it is added and normalized, and on the embedding + positional-encoding sums ($P_{drop} = 0.1$ for base; the big English–German model used 0.3). **Label smoothing** $\epsilon_{ls} = 0.1$. Table 3 (D) shows both help: no dropout scored 24.6 BLEU vs 25.8; no label smoothing reached a better perplexity (4.67 vs 4.92) but a lower BLEU (25.3).

<Formula name="Learning-rate schedule (paper eq. 3)" tag="Warmup then decay" words="Rise linearly for warmup_steps steps, then decay with the inverse square root of the step number, all scaled by d_model to the minus one half." remember="Ramp up, then 1 over root step.">

$$
\begin{aligned}
\hyper{lrate} &= \hyper{d_{model}}^{-0.5} \cdot \min(a, b) \\
a &= step^{-0.5}, \qquad b = step \cdot warmup\_steps^{-1.5}
\end{aligned}
$$

- warmup_steps = 4000; the two terms meet at step = warmup_steps
- peak $= \hyper{d_{model}}^{-0.5} \cdot warmup\_steps^{-0.5}$

**Example:** $\hyper{d_{model}} = 512$: 0.000175 at step 1,000, 0.000699 at 4,000 (peak), 0.000349 at 16,000, 0.000140 at 100,000.

</Formula>

**Try it yourself.** Move the step through warmup and decay, and change $d_{model}$ and warmup to see how the peak moves.

<LearningRateSchedule />

:::tip[🎯 Interview angle]
**"How was the original Transformer trained?"** WMT14 En–De (4.5M pairs, ~37K shared BPE), batches of ~25K source + 25K target tokens, 8 P100 GPUs: base 100K steps in 12 hours, big 300K steps in 3.5 days. Adam ($\beta_2 = 0.98$, $\epsilon = 10^{-9}$) with a warmup learning rate (linear for 4,000 steps, then $\propto 1/\sqrt{step}$), residual dropout 0.1, label smoothing 0.1, and checkpoint averaging at the end.
:::

## 18. Results and Model Variations (Paper §6)

**Translation (§6.1, Table 2).** On WMT 2014 English–German, the big Transformer scored **28.4 BLEU**, more than 2.0 BLEU above the best previously reported models **including ensembles**; even the base model (27.3) beat all previous models and ensembles. The training cost (estimated in floating-point operations) was a fraction of the competitors': $3.3 \times 10^{18}$ FLOPs for base and $2.3 \times 10^{19}$ for big. English–French: the abstract reports **41.8 BLEU** for the big model while §6.1's text says 41.0 (the arXiv v7 is inconsistent here; Table 2 lists 41.8), a new single-model state of the art at less than a quarter of the previous best model's training cost. The final models averaged the last 5 (base) or 20 (big) checkpoints, written 10 minutes apart.

| Model | EN–DE BLEU | EN–FR BLEU | Training cost (FLOPs) |
|---|---|---|---|
| GNMT + RL | 24.6 | 39.92 | $2.3 \times 10^{19}$ / $1.4 \times 10^{20}$ |
| ConvS2S | 25.16 | 40.46 | $9.6 \times 10^{18}$ / $1.5 \times 10^{20}$ |
| ConvS2S Ensemble | 26.36 | 41.29 | $7.7 \times 10^{19}$ / $1.2 \times 10^{21}$ |
| **Transformer (base)** | **27.3** | **38.1** | $3.3 \times 10^{18}$ |
| **Transformer (big)** | **28.4** | **41.8** | $2.3 \times 10^{19}$ |

**Model variations (§6.2, Table 3).** The ablations on English–German (dev set) are a goldmine of interview facts. (A) **Heads**, at constant computation: 1 head 24.9, 4 heads 25.5, 8 heads 25.8, 16 heads 25.8, 32 heads 25.4. Single-head is 0.9 worse, and too many heads also hurts. (B) **Smaller $d_k$** hurts (16: 25.1, 32: 25.4), suggesting compatibility is hard to compute and a better compatibility function than the dot product might help. (C) **Bigger is better**: $d_{model}$ 1024 → 26.0, $d_{ff}$ 4096 → 26.2, while $N = 2$ drops to 23.7. (D) **Dropout** matters a lot (0.0 → 24.6) and label smoothing trades perplexity for BLEU. (E) **Learned positional embeddings** give nearly identical results to sinusoids (25.7 vs 25.8). The big model (213M parameters) reached 26.4.

**Beyond translation (§6.3).** A 4-layer Transformer with $d_{model} = 1024$, with almost no task-specific tuning, scored 91.3 F1 on English constituency parsing trained only on the 40K-sentence WSJ set (beating the BerkeleyParser, which RNN seq2seq models had failed to do), and 92.7 semi-supervised. It was an early sign that the architecture generalizes.

**Conclusion (§7).** The authors expected attention-based models to extend to other modalities (images, audio, video) and to local, restricted attention for large inputs. That's what happened: the same blocks became BERT (encoder only), GPT (decoder only), vision Transformers and today's large language models.

:::tip[🎯 Interview angle]
**"What results did the paper report, and which ablations matter?"** 28.4 BLEU on WMT14 En–De (more than 2 BLEU over previous best, including ensembles) and 41.8 on En–Fr (41.0 in §6.1's text), at a fraction of the training cost (big model 3.5 days on 8 P100s). Ablations: single-head attention is 0.9 BLEU worse than 8 heads, too many heads also hurt, smaller $d_k$ hurts, bigger $d_{model}$/$d_{ff}$ help, dropout is crucial, label smoothing trades perplexity for BLEU, and learned positional embeddings match sinusoids.
:::

## 19. Putting It Together: a Working Transformer, Box by Box

The implementation section below writes a Transformer whose classes are named after Figure 1's boxes (`PositionalEncoding`, `AddNorm`, `FeedForward`, `EncoderLayer`, `DecoderLayer`, `Transformer`) and trains it like the paper (Adam with $\beta_2 = 0.98$, $\epsilon = 10^{-9}$, the warmup schedule, label smoothing 0.1, residual dropout 0.1, tied embeddings scaled by $\sqrt{d_{model}}$). The task is to output the input digits in reverse. It needs every part of the figure: positions (which digit was where), encoder–decoder attention (fetching the mirrored input), and the causal mask (generating left to right).

With $d_{model} = 64$, 4 heads, $d_{ff} = 128$, $N = 2$ (168,128 parameters), 3,000 steps of batches of 128 take about a minute and a half on a 2-core CPU. On 1,000 unseen sequences, greedy decoding reverses **100%** exactly. Removing single boxes shows what each one is for:

| Variant | Teacher-forced token accuracy | Generated token accuracy | Exact sequences |
|---|---|---|---|
| Full model (Figure 1) | 100% | 100% | **100%** |
| Without Positional Encoding | 50.9% | 25.2% | 0.5% |
| Without the decoder mask | 100% | 53.3% | 0.1% |

The tensor shapes follow the figure exactly. Inputs 1 × 6 ids → Input Embedding 1 × 6 × 64 (×$\sqrt{64} = 8$) → + Positional Encoding 1 × 6 × 64 → 2 encoder layers → memory 1 × 6 × 64. Outputs (shifted right) [BOS, 9, 5, 1, 4, 1] → 1 × 6 × 64 → 2 decoder layers → Linear 1 × 6 × 11 → Softmax 1 × 6 × 11 (each row sums to 1) → argmax [9, 5, 1, 4, 1, 3].

:::tip[🎯 Interview angle]
**"Could you implement a Transformer from scratch? What are the pieces?"** Yes: sinusoidal PositionalEncoding added to $\sqrt{d_{model}}$-scaled embeddings; an AddNorm wrapper (LayerNorm of $x$ + dropout(sublayer)); a FeedForward (Linear–ReLU–Linear); EncoderLayer = self-attention + FFN; DecoderLayer = masked self-attention + cross-attention + FFN; stack $N$ of each; tied output Linear; train with cross-entropy (label smoothing) under a causal mask and warmup schedule; decode greedily or with beam search.
:::

## ⚙️ Going Deeper (Transformer Internals)

**Three uses of one attention block.** Encoder self-attention (no mask), decoder masked self-attention (causal mask) and encoder–decoder attention (Q from decoder, K/V from encoder) are the same `MultiHead` function with different inputs and masks. Implementations pass a `key_padding_mask` (ignore pad tokens) and an `attn_mask` (causal) separately.

**Encoder-only, decoder-only, encoder–decoder.** Keep only the left half and train it to fill in masked words: **BERT**, good for understanding and classification. Keep only the right half without the middle attention (masked self-attention + FFN) and train next-token prediction: **GPT** and most modern LLMs. Keep both: the original Transformer, **T5** and BART, for transduction tasks.

**Inference cost and the KV cache.** At generation step $t$, the decoder only needs the new position's query; keys and values of earlier positions don't change (because of the mask), so implementations cache them. Each step then costs $O(t \cdot d)$ attention instead of recomputing everything. The encoder runs once per input.

**Where the parameters are.** Base model per layer: multi-head attention 1,050,624 (4 × 512² + biases), FFN 2,099,712, each LayerNorm 1,024. Encoder layer 3,152,384, decoder layer 4,204,032. Six of each plus a shared 37,000 × 512 embedding gives about 63.1M by this count (the paper reports 65M, and 213M for the big model where the same count gives 214.2M; the exact vocabulary size and minor details aren't fully specified).

**Post-LN warmup and stability.** With LayerNorm after the residual, the gradients near the output are large at initialization, which is one reason the warmup schedule is needed. Pre-LN moves the normalization inside the residual branch, keeps an identity path from input to output, and often trains without warmup; it needs a final LayerNorm before the output layer.

**Positional schemes since 2017.** Learned absolute embeddings (BERT, GPT-2), relative position biases (Transformer-XL, T5), rotary position embeddings (RoPE: rotate $q$ and $k$ by position-dependent angles, a direct descendant of the paper's rotation property) and ALiBi (linear distance penalties on the scores). The motivation is the paper's own hope: attention by relative position and extrapolation to longer inputs.

**Why $\sqrt{d_{model}}$ and weight tying interact.** With tied weights the embedding matrix also produces the logits, so its scale affects both. Our toy model's first loss was 45.1 with unit-variance embeddings (logits far too large) and 4.99 with variance $1/d_{model}$; the $\sqrt{d_{model}}$ multiplier then restores unit-scale inputs.

## 💻 PyTorch Implementation

:::tip[Run it yourself]
Verified on Python 3.13 and PyTorch 2.15 (CPU); outputs copied from real runs. Blocks 1–5 are one file, `toy_transformer.py`; blocks 6–7 are self-contained. Training takes about a minute and a half per run on 2 CPU cores.
:::

**Note on the reference materials.** Notebook 16 implements only the encoder block (multi-head attention, Add & Norm, FFN) in NumPy with random weights; the decoder side, positional encodings, masking, the output head and training come from the paper. The code below fills in all of Figure 1 and trains it.

### 1. Positional Encoding, Add & Norm, Feed Forward

```python title="toy_transformer.py (part 1)" showLineNumbers
import math, time, torch, torch.nn as nn

VOCAB, BOS = 11, 10                       # digits 0-9 plus a start token; shared by inputs and outputs
LEN, D_MODEL, HEADS, N_LAYERS, D_FF, P_DROP = 6, 64, 4, 2, 128, 0.1


class PositionalEncoding(nn.Module):      # Figure 1: "Positional Encoding"
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pos = torch.arange(max_len)[:, None]
        i2 = torch.arange(0, d_model, 2)
        angle = pos / 10000 ** (i2 / d_model)
        pe = torch.zeros(max_len, d_model)
        # highlight-next-line
        pe[:, 0::2], pe[:, 1::2] = torch.sin(angle), torch.cos(angle)
        self.register_buffer("pe", pe)    # fixed, not a parameter

    def forward(self, x):                 # the circled + in Figure 1
        return x + self.pe[: x.size(1)]


class AddNorm(nn.Module):                 # Figure 1: "Add & Norm" = LayerNorm(x + Dropout(Sublayer(x)))
    def __init__(self, d_model, p):
        super().__init__()
        self.norm, self.drop = nn.LayerNorm(d_model), nn.Dropout(p)

    def forward(self, x, sublayer_out):
        # highlight-next-line
        return self.norm(x + self.drop(sublayer_out))


class FeedForward(nn.Module):             # Figure 1: "Feed Forward" = max(0, xW1 + b1)W2 + b2
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)
```

### 2. The encoder layer and the decoder layer

```python title="toy_transformer.py (part 2)" showLineNumbers
class EncoderLayer(nn.Module):            # left grey box, repeated N times
    def __init__(self):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(D_MODEL, HEADS, dropout=0.0, batch_first=True)
        self.add_norm1, self.ff, self.add_norm2 = AddNorm(D_MODEL, P_DROP), FeedForward(D_MODEL, D_FF), AddNorm(D_MODEL, P_DROP)

    def forward(self, x):
        x = self.add_norm1(x, self.self_attn(x, x, x, need_weights=False)[0])   # Multi-Head Attention -> Add & Norm
        return self.add_norm2(x, self.ff(x))                                      # Feed Forward -> Add & Norm


class DecoderLayer(nn.Module):            # right grey box, repeated N times
    def __init__(self):
        super().__init__()
        self.masked_attn = nn.MultiheadAttention(D_MODEL, HEADS, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(D_MODEL, HEADS, batch_first=True)
        self.add_norm1, self.add_norm2, self.add_norm3 = (AddNorm(D_MODEL, P_DROP) for _ in range(3))
        self.ff = FeedForward(D_MODEL, D_FF)
        self.last_cross = self.last_self = None

    def forward(self, y, memory, causal_mask):
        a, self.last_self = self.masked_attn(y, y, y, attn_mask=causal_mask)
        y = self.add_norm1(y, a)                                                    # Masked MHA -> Add & Norm
        # highlight-next-line
        z, w = self.cross_attn(y, memory, memory, average_attn_weights=True)       # Q from decoder, K = V = encoder
        self.last_cross = w
        y = self.add_norm2(y, z)                                                    # -> Add & Norm
        return self.add_norm3(y, self.ff(y))                                        # Feed Forward -> Add & Norm
```

### 3. The whole Figure 1

```python title="toy_transformer.py (part 3)" showLineNumbers
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)          # Input Embedding AND Output Embedding (shared, as in §3.4)
        nn.init.normal_(self.embed.weight, std=D_MODEL ** -0.5)   # x sqrt(d_model) then gives unit-scale inputs
        self.use_pe, self.use_mask = True, True             # switches for the ablations
        self.pos, self.drop = PositionalEncoding(D_MODEL), nn.Dropout(P_DROP)
        self.encoder = nn.ModuleList(EncoderLayer() for _ in range(N_LAYERS))
        self.decoder = nn.ModuleList(DecoderLayer() for _ in range(N_LAYERS))

    def embed_tokens(self, t):            # embeddings x sqrt(d_model), + positional encoding, dropout
        e = self.embed(t) * math.sqrt(D_MODEL)
        return self.drop(self.pos(e) if self.use_pe else e)

    def encode(self, src):
        x = self.embed_tokens(src)
        for layer in self.encoder:
            x = layer(x)
        return x                          # the encoder output ("memory") read by every decoder layer

    def decode(self, tgt_in, memory):
        n = tgt_in.size(1)
        # highlight-next-line
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)   # True = may NOT attend (future)
        if not self.use_mask:
            causal = None
        y = self.embed_tokens(tgt_in)
        for layer in self.decoder:
            y = layer(y, memory, causal)
        return y @ self.embed.weight.T    # Linear: the pre-softmax projection shares the embedding matrix

    def forward(self, src, tgt_in):
        return self.decode(tgt_in, self.encode(src))       # logits; Softmax is applied in the loss / at decoding
```

### 4. Data (shifted right), the paper's learning rate, greedy decoding

```python title="toy_transformer.py (part 4)" showLineNumbers
def make_batch(b):
    src = torch.randint(0, 10, (b, LEN))
    tgt = src.flip(1)                                       # the "translation": reverse the digits
    # highlight-next-line
    tgt_in = torch.cat([torch.full((b, 1), BOS), tgt[:, :-1]], 1)   # Outputs (shifted right)
    return src, tgt_in, tgt


def lrate(step, d_model=D_MODEL, warmup=400):               # paper eq. (3)
    step = max(step, 1)
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


@torch.no_grad()
def greedy(model, src):
    memory = model.encode(src)                              # encoder runs once
    out = torch.full((src.size(0), 1), BOS)
    for _ in range(LEN):                                    # auto-regressive: one token per step
        nxt = model.decode(out, memory)[:, -1].argmax(-1, keepdim=True)
        out = torch.cat([out, nxt], 1)
    return out[:, 1:]
```

### 5. Train like the paper, evaluate, ablate

```python title="toy_transformer.py (part 5)" showLineNumbers
def train(use_pe=True, use_mask=True, steps=3000, seed=0, log=True):
    torch.manual_seed(seed)
    model = Transformer(); model.use_pe, model.use_mask = use_pe, use_mask
    # highlight-start
    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lrate(s + 1))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    # highlight-end
    for step in range(1, steps + 1):
        src, tgt_in, tgt = make_batch(128)
        loss = loss_fn(model(src, tgt_in).reshape(-1, VOCAB), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if log and step in (1, 100, 400, 1000, 2000, 3000):
            print(f"step {step:5d}  lr {sched.get_last_lr()[0]:.6f}  loss {loss.item():.4f}")
    model.eval()
    return model


def evaluate(model):
    torch.manual_seed(1)
    src, tgt_in, tgt = make_batch(1000)
    with torch.no_grad():
        tf = (model(src, tgt_in).argmax(-1) == tgt).float().mean().item()      # teacher forcing
    pred = greedy(model, src)
    return tf, (pred == tgt).float().mean().item(), (pred == tgt).all(1).float().mean().item()


model = train()
# step     1  lr 0.000031  loss 4.9860
# step   100  lr 0.001578  loss 1.8151
# step   400  lr 0.006242  loss 0.5701
# step  1000  lr 0.003951  loss 0.5295
# step  2000  lr 0.002794  loss 0.5163
# step  3000  lr 0.002282  loss 0.5156
print(sum(p.numel() for p in model.parameters()))          # 168128
print(evaluate(model))                                       # (1.0, 1.0, 1.0)
print(greedy(model, torch.tensor([[3, 1, 4, 1, 5, 9]])).tolist())   # [[9, 5, 1, 4, 1, 3]]
model.decode(torch.tensor([[BOS, 9, 5, 1, 4, 1]]), model.encode(torch.tensor([[3, 1, 4, 1, 5, 9]])))
torch.set_printoptions(precision=2, sci_mode=False)
print(model.decoder[-1].last_cross[0])                       # encoder-decoder attention: anti-diagonal
# tensor([[0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
#         [0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
#         [0.00, 0.00, 0.00, 1.00, 0.00, 0.00],
#         [0.00, 0.00, 1.00, 0.00, 0.00, 0.00],
#         [0.00, 1.00, 0.00, 0.00, 0.00, 0.00],
#         [1.00, 0.00, 0.00, 0.00, 0.00, 0.00]], grad_fn=<SelectBackward0>)
print(model.decoder[-1].last_self[0][:3])                    # masked self-attention: zeros above the diagonal
# tensor([[1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
#         [0.76, 0.24, 0.00, 0.00, 0.00, 0.00],
#         [0.15, 0.48, 0.37, 0.00, 0.00, 0.00]], grad_fn=<SliceBackward0>)

# error-start
print(evaluate(train(use_pe=False, log=False)))             # (0.5091666579246521, 0.2515000104904175, 0.004999999888241291)
print(evaluate(train(use_mask=False, log=False)))           # (1.0, 0.5325000286102295, 0.0010000000474974513)
# error-end
```

The ablations print (teacher-forced token accuracy, generated token accuracy, exact-sequence accuracy). Without positional encoding the model can't tell positions apart; without the mask it learns to copy the next input during training and falls apart when it has to generate.

### 6. Checking the paper's positional-encoding claims

```python title="positional_encoding_checks.py" showLineNumbers
import numpy as np
np.set_printoptions(precision=4, suppress=True)

def pe(pos, d):
    i2 = np.arange(0, d, 2); ang = pos / 10000 ** (i2 / d)
    out = np.zeros(d); out[0::2] = np.sin(ang); out[1::2] = np.cos(ang); return out

d = 512
print(pe(0, d)[:6], pe(1, d)[:6])          # [0. 1. 0. 1. 0. 1.] [0.8415 0.5403 0.8219 0.5697 0.802  0.5974]
P = np.array([pe(p, d) for p in range(50)])
w, k = 1 / 10000 ** (np.arange(0, d, 2) / d), 5
S, C = P[:, 0::2], P[:, 1::2]
rot = np.zeros_like(P)
# highlight-next-line
rot[:, 0::2], rot[:, 1::2] = S * np.cos(w * k) + C * np.sin(w * k), C * np.cos(w * k) - S * np.sin(w * k)
print(np.abs(rot[:45] - P[k:50]).max())    # 6.161737786669619e-15   PE(pos+k) = R_k PE(pos)
print(np.round((P[10] @ P.T)[10:16], 2))   # [256.   249.1  231.73 211.75 196.69 189.6 ]
print(np.round((P[20] @ P.T)[20:26], 2))   # [256.   249.1  231.73 211.75 196.69 189.6 ]   depends only on k
print(np.round(np.linalg.norm(P[[0, 7, 49]], axis=1), 3))   # [16. 16. 16.]   sqrt(d/2)
```

### 7. The learning-rate schedule and the base model's size

```python title="paper_numbers.py" showLineNumbers
def lr(step, d_model=512, warmup=4000):
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)

for s in (1000, 4000, 16000, 100000):
    print(s, round(lr(s), 6))                # 1000 0.000175 / 4000 0.000699 / 16000 0.000349 / 100000 0.00014

dm, dff, V = 512, 2048, 37000
mha = 4 * dm * dm + 4 * dm                   # W_Q, W_K, W_V, W_O + biases
ffn = dm * dff + dff + dff * dm + dm
enc, dec = mha + ffn + 2 * 2 * dm, 2 * mha + ffn + 3 * 2 * dm
print(mha, ffn, enc, dec)                    # 1050624 2099712 3152384 4204032
print(6 * enc + 6 * dec + V * dm)            # 63082496   (paper, Table 3: 65M)
```

## 🚨 Interview Gotchas & Edge Cases

:::warning[The Most Common Question]
**"Walk me through the Transformer architecture (Figure 1)."**

Encoder: token ids → Input Embedding (×$\sqrt{d_{model}}$) + sinusoidal Positional Encoding → $N = 6$ layers of [Multi-Head self-Attention → Add & Norm → Feed Forward (512 → 2048 → 512, ReLU) → Add & Norm], where Add & Norm is $\operatorname{LayerNorm}(x + \operatorname{Dropout}(\operatorname{Sublayer}(x)))$. Decoder: target shifted right → Output Embedding + Positional Encoding → $N = 6$ layers of [**Masked** Multi-Head self-Attention → Add & Norm → Multi-Head Attention with **Q from the decoder and K, V from the encoder output** → Add & Norm → Feed Forward → Add & Norm] → Linear (tied to the embeddings) → Softmax → next-token probabilities. $d_{model} = 512$, $h = 8$, $d_k = 64$. Training is parallel with teacher forcing and the causal mask; generation is auto-regressive.
:::

- **Positional encodings are added, not concatenated,** and fixed (sinusoidal) in the paper; learned ones scored the same.
- **Self-attention alone is order-blind;** toy model without PE: 0.5% exact reversal.
- **The causal mask matters in training, not generation;** without it, 100% teacher-forced but 0.1% generated.
- **Encoder–decoder attention:** Q from the decoder, K and V from the **encoder output** (the same memory for all decoder layers), no causal mask.
- **Shifted right:** decoder input $[\langle s \rangle, y_1..y_{m-1}]$, target $[y_1..y_m]$.
- **Post-LN in the paper;** Pre-LN in most modern LLMs.
- **Dropout is applied before the residual addition,** and to embedding + PE sums.
- **Weight tying:** one matrix for input embedding, output embedding and pre-softmax Linear; embeddings scaled by $\sqrt{d_{model}}$.
- **Label smoothing makes the model less confident on purpose** (toy: 0.92 on correct tokens; perplexity worse, BLEU better).
- **Warmup:** learning rate rises for 4,000 steps, peaks at 0.000699 ($d_{model} = 512$), then decays as $1/\sqrt{step}$.
- **Cost:** self-attention $O(n^2 d)$ per layer; cheaper than recurrence when $n < d$.
- **BLEU numbers:** 28.4 En–De; En–Fr 41.8 (abstract, Table 2) vs 41.0 (§6.1 text).

## ⚡ Essential Interview Recall

A one-screen revision list. Every point is explained in the chapters above.

**Figure 1, encoder (bottom → top)**
- Inputs (BPE, ~37K shared vocab) → Input Embedding ×$\sqrt{d_{model}}$ → ⊕ Positional Encoding.
- N = 6 × [Multi-Head Attention (self, unmasked) → Add & Norm → Feed Forward → Add & Norm] → memory $z$.

**Figure 1, decoder (bottom → top)**
- Outputs (shifted right) → Output Embedding (shared) → ⊕ Positional Encoding.
- N = 6 × [Masked Multi-Head Attention → Add & Norm → Multi-Head Attention (Q = decoder, K = V = encoder) → Add & Norm → Feed Forward → Add & Norm].
- Linear (tied) → Softmax → Output Probabilities.

**Formulas**
- $PE_{(pos,2i)} = \sin(pos/10000^{2i/d})$, $PE_{(pos,2i+1)} = \cos(\cdot)$; $PE_{pos+k} = R_k PE_{pos}$.
- Add & Norm: $\operatorname{LayerNorm}(x + \operatorname{Dropout}(\operatorname{Sublayer}(x)))$.
- $\operatorname{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$, 512 → 2048 → 512.
- Mask: $+M$, $M_{ij} = -\infty$ for $j > i$.
- $lrate = d_{model}^{-0.5}\min(step^{-0.5}, step \cdot 4000^{-1.5})$; peak 0.000699.

**Numbers**
- Base: N 6, $d_{model}$ 512, $d_{ff}$ 2048, h 8, $d_k$ 64, dropout 0.1, 65M params, 12 h on 8 P100s. Big: 1024 / 4096 / 16 heads, 213M, 3.5 days.
- BLEU: En–De 28.4 (big), 27.3 (base); En–Fr 41.8 (41.0 in §6.1 text).
- Table 3: h = 1 24.9 vs h = 8 25.8; N = 2 23.7; no dropout 24.6; learned PE 25.7.
- Table 1: self-attention $O(n^2 d)$ / $O(1)$ / $O(1)$; recurrent $O(n d^2)$ / $O(n)$ / $O(n)$.

**Toy model (reverse 6 digits)**
- 168,128 params; 100% exact; cross-attention anti-diagonal (1.00); masked self-attention lower-triangular.
- No PE: 0.5% exact; no mask: 100% teacher-forced, 0.1% generated.

## 📐 Formula Sheet

Every equation on this page in one place.

| Concept | Formula | In plain words |
|---|---|---|
| Embedding | $\data{e_i} = \param{E}[\data{x_i}]\sqrt{\hyper{d_{model}}}$ | Look up and scale |
| Positional encoding | $PE_{(pos,2i)} = \sin(pos/10000^{2i/\hyper{d_{model}}})$, $PE_{(pos,2i+1)} = \cos(\cdot)$ | Waves of geometric frequencies |
| Shift property | $PE_{pos+k} = R_k\, PE_{pos}$ (rotation per sine–cosine pair) | Shifting = rotating |
| Encoder input | $\data{x} = \data{e} + PE$ | Add, don't concatenate |
| Scaled dot-product attention | $\operatorname{softmax}(\pred{Q}\pred{K}^\top/\sqrt{\hyper{d_k}})\pred{V}$ | Match, scale, normalize, mix |
| Multi-head attention | $\operatorname{Concat}(\pred{\text{head}_1}..\pred{\text{head}_h})\param{W^O}$ | h heads, concat, mix |
| Causal mask | $\operatorname{softmax}(\pred{Q}\pred{K}^\top/\sqrt{\hyper{d_k}} + M)\pred{V}$, $M_{ij} = -\infty$ for $j > i$ | No peeking ahead |
| Encoder–decoder attention | $\operatorname{MultiHead}(\pred{y}, \pred{z}, \pred{z})$ | Decoder asks, encoder answers |
| Add & Norm | $\operatorname{LayerNorm}(\data{x} + \operatorname{Dropout}(\operatorname{Sublayer}(\data{x})))$ | Residual, then normalize |
| LayerNorm | $\param{\gamma} \odot (\data{v} - \mu)/\sqrt{\sigma^2 + \epsilon} + \param{\beta}$ | Per-position standardize |
| Feed Forward | $\max(0, \data{x}\param{W_1} + \param{b_1})\param{W_2} + \param{b_2}$ | Widen, ReLU, narrow |
| Encoder layer | $\operatorname{LN}(\pred{u} + \operatorname{FFN}(\pred{u}))$, $\pred{u} = \operatorname{LN}(\data{x} + \operatorname{MHA}(\data{x},\data{x},\data{x}))$ | Two sub-layers |
| Decoder layer | masked MHA → cross MHA($\pred{a}, \pred{z}, \pred{z}$) → FFN, each with Add & Norm | Three sub-layers |
| Output layer | $\operatorname{softmax}(\pred{y_{dec}}\param{E}^\top)$ | Tied Linear, then Softmax |
| Label-smoothed target | $q(t) = (1 - \hyper{\epsilon_{ls}})[t = \data{y}] + \hyper{\epsilon_{ls}}/V$ | Spread a little probability |
| Learning rate | $\hyper{d_{model}}^{-0.5}\min(step^{-0.5}, step \cdot warmup^{-1.5})$ | Warm up, then 1/√step |

## Interview Questions on the Paper

<details>
<summary><strong>Q1. What problem was the Transformer designed to solve, and what was the key idea?</strong></summary>

Sequence transduction (machine translation) with less sequential computation. Recurrent models process tokens one at a time, which prevents parallelization within a sequence. The Transformer relies entirely on attention to relate positions, so all positions are processed in parallel and any two positions are one step apart.
</details>

<details>
<summary><strong>Q2. Name every box in Figure 1, in order.</strong></summary>

Encoder: Inputs → Input Embedding → Positional Encoding (⊕) → [Multi-Head Attention → Add & Norm → Feed Forward → Add & Norm] × N. Decoder: Outputs (shifted right) → Output Embedding → Positional Encoding (⊕) → [Masked Multi-Head Attention → Add & Norm → Multi-Head Attention → Add & Norm → Feed Forward → Add & Norm] × N → Linear → Softmax → Output Probabilities.
</details>

<details>
<summary><strong>Q3. What are the three ways multi-head attention is used?</strong></summary>

Encoder self-attention (Q, K, V from the previous encoder layer, unmasked), decoder masked self-attention (Q, K, V from the previous decoder layer, future masked) and encoder–decoder attention (Q from the decoder, K and V from the encoder output).
</details>

<details>
<summary><strong>Q4. Why divide by √d_k?</strong></summary>

If query and key components are independent with mean 0 and variance 1, $q \cdot k$ has variance $d_k$. Large dot products push softmax into regions with extremely small gradients (the paper's footnote 4); scaling by $1/\sqrt{d_k}$ counteracts this.
</details>

<details>
<summary><strong>Q5. Why are sinusoidal positional encodings used rather than learned ones?</strong></summary>

They performed about the same (25.7 vs 25.8 BLEU), but sinusoids may let the model extrapolate to sequence lengths longer than those seen in training, and since $PE_{pos+k}$ is a linear function of $PE_{pos}$, they may make attending by relative position easy.
</details>

<details>
<summary><strong>Q6. What does the mask in the decoder do, and why is the output shifted right?</strong></summary>

The mask sets attention scores for future positions to $-\infty$, so position $i$ attends only to positions up to $i$. With outputs shifted right by one, the prediction for position $i$ can depend only on known outputs at positions less than $i$, which preserves the auto-regressive property while training all positions in parallel.
</details>

<details>
<summary><strong>Q7. What is in each sub-layer's residual block, exactly?</strong></summary>

$\operatorname{LayerNorm}(x + \operatorname{Dropout}(\operatorname{Sublayer}(x)))$: dropout on the sub-layer output, a residual addition, then layer normalization (Post-LN). All sub-layers and embeddings output $d_{model} = 512$ to make the residuals possible.
</details>

<details>
<summary><strong>Q8. What are d_model, d_ff, h, d_k and N in the base and big models?</strong></summary>

Base: 512, 2048, 8, 64, 6 (65M parameters). Big: 1024, 4096, 16, 64, 6 (213M parameters), with dropout 0.3 instead of 0.1 for English–German.
</details>

<details>
<summary><strong>Q9. Describe the optimizer and learning-rate schedule.</strong></summary>

Adam with $\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$; $lrate = d_{model}^{-0.5}\min(step^{-0.5}, step \cdot warmup^{-1.5})$ with 4,000 warmup steps: linear increase, then decay with the inverse square root of the step (peak 0.000699 for $d_{model} = 512$).
</details>

<details>
<summary><strong>Q10. What regularization was used?</strong></summary>

Residual dropout ($P_{drop} = 0.1$) on each sub-layer output before the residual addition and on embedding + positional-encoding sums, plus label smoothing $\epsilon_{ls} = 0.1$, which worsened perplexity but improved accuracy and BLEU.
</details>

<details>
<summary><strong>Q11. What did the ablations (Table 3) show?</strong></summary>

Single-head attention is 0.9 BLEU worse than the best, and too many heads also hurts; reducing $d_k$ hurts; bigger models are better; dropout is very helpful; learned positional embeddings give nearly identical results to sinusoids.
</details>

<details>
<summary><strong>Q12. Compare self-attention with recurrent and convolutional layers.</strong></summary>

Per layer: self-attention $O(n^2 d)$, $O(1)$ sequential operations, $O(1)$ maximum path length; recurrent $O(n d^2)$, $O(n)$, $O(n)$; convolutional $O(k n d^2)$, $O(1)$, $O(\log_k n)$. Self-attention is faster than recurrence when $n < d$.
</details>

<details>
<summary><strong>Q13. What results did the paper report?</strong></summary>

WMT14 En–De: 28.4 BLEU (big), over 2 BLEU above the previous best including ensembles; base 27.3 already beat all previous models. En–Fr: 41.8 (abstract and Table 2; 41.0 in §6.1's text), a new single-model state of the art at under a quarter of the previous best's training cost. English constituency parsing: 91.3 F1 (WSJ only), 92.7 (semi-supervised).
</details>

<details>
<summary><strong>Q14. How do BERT and GPT relate to Figure 1?</strong></summary>

BERT is the encoder stack (left half), pre-trained with masked-token prediction; GPT is the decoder stack without the encoder–decoder attention (masked self-attention + FFN), trained on next-token prediction. Both inherit embeddings, positional information, multi-head attention, FFN and residual + LayerNorm blocks (GPT-2 onwards moved to Pre-LN).
</details>
