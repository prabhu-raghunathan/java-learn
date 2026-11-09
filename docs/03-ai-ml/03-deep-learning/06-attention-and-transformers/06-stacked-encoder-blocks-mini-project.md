---
title: "Stacked Encoder Blocks: a Mini Sentiment Project"
sidebar_position: 6
---

import {TinySentimentLive, EncoderStackParams} from '@site/src/components/Widgets';

:::tip[TL;DR (30-Second Interview Pitch)]
An encoder block (multi-head self-attention → Add & Norm → feed-forward → Add & Norm) maps a `[batch, seq_len, d_model]` tensor to a tensor of **exactly the same shape**, so blocks can be **stacked**: $X^{(l)} = \operatorname{Block}_l(X^{(l-1)})$. Each block has its own weights and gives every token **one more round** of "look at the other tokens, then transform yourself"; depth changes the *numbers*, never the shape. This page builds the notebook's tiny sentiment classifier (token + learned position embeddings → $N$ encoder blocks in an `nn.ModuleList` → **masked mean pooling** over real tokens → linear head) with 1 block (9,922 parameters) and 3 blocks (27,010). Both reach 100% training accuracy on the 18 sentences, which proves nothing. On 16 held-out sentences over 20 seeds, depth gave 86.3% vs 84.1%, a difference smaller than the seed-to-seed spread. Depth pays off only with enough data, which is exactly why BERT-style encoders stack 12+ blocks **and** pre-train them on billions of words.
:::

<FormulaKey />

:::info[Where this page sits]
**Before:** [BERT: Pre-training and Fine-tuning, Figure by Figure](./05-bert-pretraining-and-fine-tuning.md) showed what a 12-block encoder can do once it is pre-trained.
**This page:** notebook 18. The file is called *Hugging_Face_Pipeline_Complete*, but its content is **"Mini NLP Project With Stacked Transformer Encoder Blocks"**: it rebuilds the from-scratch sentiment classifier with 1 and 3 encoder blocks and inspects what depth does. Its two Hugging Face preview cells failed in the notebook (`ModuleNotFoundError`), so they move to the next page, where Hugging Face is the topic.
**Next:** notebook 19, Hugging Face pipelines and what they hide (the next sub-section of Deep Learning).
:::

## 🧭 Start Here

The previous pages built one encoder block box by box and then read about BERT, which stacks twelve of them. This page sits in between: it takes the one-block sentiment classifier you could write by hand and asks the simplest possible question about depth. What changes when the same block is repeated three times? The answer has two halves. Mechanically, almost nothing changes: every block takes a `[batch, seq_len, d_model]` tensor and returns one of the same shape, so stacking is just a loop over an `nn.ModuleList`. In terms of what the network can represent, a lot changes: each extra block lets every token look at every other token *again*, using what it learned in the previous round.

The project is deliberately tiny so that every number can be checked: 18 movie-review phrases (9 positive, 9 negative), a whitespace tokenizer with a 35-entry vocabulary, sequences padded to 6 tokens, $d_{model} = 32$, 4 heads, a feed-forward width of 64. Two models with identical settings except the number of blocks (1 vs 3) are trained for 60 epochs with Adam. Everything on this page was re-run on CPU with the notebook's seeds and reproduces the notebook's printed numbers exactly; the widget further down runs those same trained weights in your browser.

The page follows the notebook's order and fixes one thing it got wrong. The notebook compares the two models by **training accuracy**, which is 100% for both and says nothing about generalization. Chapter 7 adds a held-out set built from new combinations of the training words and repeats training over 20 seeds. The honest result (depth did not measurably help with 18 examples) is the most useful lesson on the page, because it explains why real encoders are pre-trained.

| New concept | One-line meaning |
|---|---|
| **Encoder stack** | $N$ encoder blocks applied in sequence, each with its own weights |
| **Shape invariance** | Every block maps `[B, T, D]` to `[B, T, D]`, so blocks compose freely |
| **`nn.ModuleList`** | A list that registers its blocks as sub-modules (parameters, `.to()`, `state_dict`) |
| **Attention mask** | 1 for real tokens, 0 for `<PAD>`; padded keys get $-10^9$ before the softmax |
| **Hidden states** | The `[B, T, D]` tensor after the embeddings and after each block |
| **Masked mean pooling** | Average of the last hidden states over real tokens only: one vector per sentence |
| **Classification head** | A linear layer from the pooled vector to one logit per class |
| **Representation depth** | How many rounds of mixing and transforming a token's vector has been through |

:::analogy[Drafts of an essay]
One encoder block is one editing pass over an essay: every sentence is reread in light of the others, then polished on its own. Three blocks are three passes. The essay keeps the same number of sentences and the same length every time (shape invariance), but each pass can fix things the previous one could only notice. With a two-line essay, though, a second and third pass have little left to fix, and that's this page's tiny dataset.
:::

## 1. The Mini Project: Data, Vocabulary, Padding and the Attention Mask

The dataset is 18 short phrases with binary labels: "i love this movie", "amazing acting and story" and seven more are positive (label 1); "i hate this movie", "terrible acting and story" and seven more are negative (label 0). Two pairs are deliberately tricky: "not bad movie" is labelled positive and "not good movie" negative, so the model can't just count good and bad words; it has to let "not" change the meaning of the next word.

Tokenization is the simplest possible: lower-case and split on spaces. The vocabulary is built from the training words in order of first appearance, after two special entries: `<PAD>` with id 0 and `<UNK>` with id 1, giving 35 entries in total. `encode_text` maps each word to its id (unknown words to `<UNK>`), cuts the sequence at `max_len = 6`, and pads it with `<PAD>` up to 6. It also builds the **attention mask**: 1 for every real token, 0 for every padding position. "i love this movie" becomes ids `[2, 3, 4, 5, 0, 0]` with mask `[1, 1, 1, 1, 0, 0]`; "not good movie" becomes `[25, 19, 5, 0, 0, 0]` with mask `[1, 1, 1, 0, 0, 0]`; "this film rocks" becomes `[4, 15, 1, 0, 0, 0]` because "rocks" was never seen.

Padding exists only so that sentences of different lengths can be stacked into one rectangular batch tensor. The padded positions carry no meaning, and the mask is how the model is told to ignore them in two places: inside every attention layer (no token may attend to a pad) and in the pooling at the end (pads are left out of the sentence average). A `TinySentimentDataset` returns `(input_ids, attention_mask, label)` tensors and a `DataLoader` batches them 4 at a time with shuffling, so one batch is `[4, 6]` ids, `[4, 6]` mask and `[4]` labels.

| Text | `input_ids` | `attention_mask` | Note |
|---|---|---|---|
| i love this movie | `[2, 3, 4, 5, 0, 0]` | `[1, 1, 1, 1, 0, 0]` | 2 pads |
| not good movie | `[25, 19, 5, 0, 0, 0]` | `[1, 1, 1, 0, 0, 0]` | 3 pads |
| this film rocks | `[4, 15, 1, 0, 0, 0]` | `[1, 1, 1, 0, 0, 0]` | "rocks" → `<UNK>` (1) |

:::danger[Interview Trap: padding id vs attention mask]
`padding_idx=PAD_ID` in `nn.Embedding` only keeps the `<PAD>` embedding at zero and stops its gradient. It does **not** stop other tokens from attending to the pad positions, and the position embedding is still added there, so pad positions hold non-zero vectors. Only the attention mask (and masked pooling) keeps them out of the computation.
:::

:::tip[🎯 Interview angle]
**"Why do Transformers need an attention mask when we already pad?"** Padding makes a rectangular batch; the mask tells the model which positions are real. Without it, real tokens would attend to padded positions and the sentence representation would change with the amount of padding. The mask is applied to the attention scores (padded keys get a huge negative number before the softmax) and to the pooling (only real tokens are averaged).
:::

## 2. One Encoder Block, Redefined

The notebook re-implements the four components so the page runs on its own. **`MultiHeadSelfAttention`** projects the input with $W_q$, $W_k$, $W_v$ (each $D \times D$), splits the result into 4 heads of 8 dimensions, computes scaled dot-product scores per head, masks padded keys, takes the softmax, mixes the values, concatenates the heads and applies the output projection. **`AddNorm`** adds the sub-layer's output to its input and layer-normalizes (post-LN, as in the original Transformer). **`PositionWiseFeedForward`** is Linear(32 → 64), ReLU, Linear(64 → 32), applied to each position independently. **`TransformerEncoderBlock`** chains them: attention → Add & Norm → feed-forward → Add & Norm. This is exactly one grey box of Transformer Figure 1, covered in detail in [Multi-Head Attention](./03-multi-head-attention.md) and [The Transformer: Figure 1 Box by Box](./04-transformer-architecture-figure-1.md).

The mask is applied to the **keys**: `attention_mask[:, None, None, :]` reshapes the `[B, T]` mask to `[B, 1, 1, T]`, which broadcasts over heads and query positions, and every score whose key is a pad becomes $-10^9$. After the softmax those weights are exactly 0: over all 18 training sentences, every layer and every head of the trained 3-block model, the largest weight on a pad key is 0. Query rows *at* pad positions are still computed, but nothing downstream uses them because pooling ignores them too.

<Formula name="Masked multi-head self-attention" tag="One block, sub-layer 1" words="Each head scores every query token against every key token, adds minus a billion where the key is padding, turns each row into weights with softmax and mixes the value vectors. The heads are concatenated and projected back to d_model." remember="Pads get minus infinity, weight zero.">

$$
\begin{aligned}
\pred{S_h} &= \frac{(\data{X}\param{W_q})_h (\data{X}\param{W_k})_h^\top}{\sqrt{\hyper{d_h}}} + \data{M} \\
\data{M}_{ij} &= \begin{cases} 0 & \text{key } j \text{ is real} \\ -10^9 & \text{key } j \text{ is padding} \end{cases} \\
\pred{H_h} &= \operatorname{softmax}(\pred{S_h})\,(\data{X}\param{W_v})_h \\
\operatorname{MHSA}(\data{X}) &= [\pred{H_1}; \ldots; \pred{H_4}]\,\param{W_o}
\end{aligned}
$$

- $\data{X}$: `[T, D]` token vectors; $(\cdot)_h$: the 8 columns that belong to head $h$; $\hyper{d_h} = D / \text{heads} = 8$.
- $\data{M}$: the mask term, added to every row of scores.
- $\param{W_q}, \param{W_k}, \param{W_v}, \param{W_o}$: $32 \times 32$ projections with biases.

**Example:** in the trained 3-block model, layer 1, head 1, the query "not" puts weights 0.203, 0.001 and 0.795 on "not", "good" and "movie", and exactly 0 on the three pads.

</Formula>

<Formula name="Encoder block" tag="Two sub-layers with Add & Norm" words="Attention lets tokens exchange information, the feed-forward network transforms each token alone, and each sub-layer is wrapped in a residual connection followed by LayerNorm." remember="Mix, add, norm; transform, add, norm.">

$$
\begin{aligned}
\pred{U} &= \operatorname{LayerNorm}\big(\data{X} + \operatorname{MHSA}(\data{X})\big) \\
\operatorname{FFN}(\pred{U}) &= \operatorname{ReLU}(\pred{U}\param{W_1} + \param{b_1})\param{W_2} + \param{b_2} \\
\operatorname{Block}(\data{X}) &= \operatorname{LayerNorm}\big(\pred{U} + \operatorname{FFN}(\pred{U})\big)
\end{aligned}
$$

- $\param{W_1}$: $32 \times 64$, $\param{W_2}$: $64 \times 32$; the FFN is applied to each of the $T$ positions separately.
- Input and output are both `[B, T, D]`.

**Example:** a `[4, 6, 32]` batch leaves the block as `[4, 6, 32]`, and its attention weights have shape `[4, 4, 6, 6]` (batch, heads, query, key).

</Formula>

:::tip[🎯 Interview angle]
**"Walk me through one encoder block's forward pass, with shapes."** Input `[B, T, D]`. Q, K, V projections keep `[B, T, D]`, reshaped to `[B, h, T, D/h]`; scores `[B, h, T, T]` scaled by $\sqrt{D/h}$, padded keys masked, softmax over the last axis, times V gives `[B, h, T, D/h]`, merged back to `[B, T, D]` and projected. Residual + LayerNorm, then a per-position FFN $D \to d_{ff} \to D$, residual + LayerNorm. Output `[B, T, D]`.
:::

## 3. Why One Block Is Not Enough: Depth as Repeated Refinement

A single block already does useful work: every token's vector is updated with information from every other token (attention), then reshaped by a small MLP (FFN). That is enough to classify most of the 18 training phrases. What one block can't do is **build on its own result**. In "not good movie", block 1 can let "good" absorb something from "not", but the "not"-modified "good" only exists *after* block 1. A second block can use that modified vector as input: tokens now attend to each other's already-contextualized representations, and relations of relations become expressible.

Stacking writes this down as a recursion. The embedding output is $X^{(0)}$, and block $l$ turns $X^{(l-1)}$ into $X^{(l)}$. Each block has its own parameters, so block 2 doesn't have to repeat block 1; it can learn a different mixing pattern. The final hidden state $X^{(N)}$ is what the classifier reads.

<Formula name="Encoder stack recursion" tag="Depth" words="The output of each block is the input of the next; all blocks keep the shape, and each has its own weights." remember="Same shape, new numbers, N times.">

$$
\begin{aligned}
\data{X}^{(0)} &= \param{E_{tok}}[\data{ids}] + \param{E_{pos}}[0..T-1] \\
\pred{X}^{(l)} &= \operatorname{Block}_l\big(\pred{X}^{(l-1)}\big), \quad l = 1, \ldots, \hyper{N}
\end{aligned}
$$

- $\param{E_{tok}}$: $35 \times 32$ token embeddings; $\param{E_{pos}}$: $6 \times 32$ learned position embeddings.
- $\hyper{N}$: number of blocks (1 or 3 here; 12 in BERT-base, 6 in DistilBERT).

**Example:** for "not good movie" in the 3-block model, the mean norm of the real tokens' vectors goes 8.73 → 5.62 → 5.74 → 6.01, and the mean absolute change per block is 0.85, 0.59, 0.63, while the shape stays `[1, 6, 32]` throughout.

</Formula>

![Mini sentiment project: embeddings, stacked encoder blocks, masked mean pooling, classifier](/img/deep-learning/diagrams/stacked-encoder-classifier.svg)

*How to read it:* tokens enter on the left (pads in grey), become embedding + position vectors, pass through the green stack of encoder blocks (1 or 3 in this project), are averaged over real tokens only, and a linear layer gives two logits. The red dashed line is the attention mask, used by every block and by the pooling.

:::analogy[A relay of interpreters]
Picture a line of interpreters passing a sentence along. Each one hears the whole sentence, discusses it with the others, and hands on a slightly better-understood version in the same format. The first notices that "not" sits next to "good"; the second already receives a "good" that carries that note and can work out the phrase means negative. More interpreters help only if there is something left to work out.
:::

:::tip[🎯 Interview angle]
**"Why stack encoder blocks instead of using one wide block?"** Each block can only combine what its input already contains; stacking lets later blocks attend over representations that are already contextualized, so they can model higher-order relations (phrases, negation scope, sentence-level features). Width adds capacity within one round of mixing, depth adds rounds. Every block keeps the `[B, T, D]` shape, so depth is just a loop.
:::

## 4. Building the Stack with `nn.ModuleList`

The stack is a class with one attribute: `self.layers = nn.ModuleList([TransformerEncoderBlock(...) for _ in range(num_layers)])`, and a forward pass that loops `x = layer(x, attention_mask=attention_mask)`. `nn.ModuleList` matters because it **registers** each block as a sub-module: their parameters appear in `model.parameters()` (so the optimizer updates them), they move with `model.to(device)`, switch with `model.train()`/`model.eval()`, and are saved in `state_dict()`. A plain Python list of blocks would hold the same objects, but PyTorch wouldn't see them: the optimizer would silently skip their weights. `nn.Sequential` would also register them, but it can't pass the extra `attention_mask` argument to each block, which is why `ModuleList` plus an explicit loop is the standard pattern.

The stack can optionally return everything it computed, like Hugging Face's `output_hidden_states=True` and `output_attentions=True`: `hidden_states` is a list of $N + 1$ tensors (the input plus one per block), `attentions` a list of $N$ tensors of shape `[B, heads, T, T]`. With a `[4, 6, 32]` input and 3 blocks, every hidden state is `[4, 6, 32]` and every attention tensor `[4, 4, 6, 6]`. Shape mismatches inside a stack only happen if blocks are built with different `d_model`: block 1 would output 32 features and block 2's `nn.Linear(64, 64)` would fail with a matrix-multiplication shape error on its first projection.

| Choice | Registers parameters? | Can pass `attention_mask` to each block? | Use it? |
|---|---|---|---|
| Python `list` | ❌ no (optimizer, `.to()`, `state_dict` all miss them) | ✅ | Never for layers |
| `nn.Sequential` | ✅ | ❌ (single-argument forward) | When blocks take one input |
| `nn.ModuleList` + loop | ✅ | ✅ | **The standard encoder-stack pattern** |

:::danger[Interview Trap: "stacking blocks makes the sequence shorter or the vectors bigger"]
Neither. Every block maps `[B, T, D]` to `[B, T, D]`. Sequence length only changes if you add pooling or downsampling explicitly, and the width only changes if you add a projection. Depth changes the *content* of the tensor, not its shape.
:::

:::tip[🎯 Interview angle]
**"Why `nn.ModuleList` and not a Python list for the layers?"** A Python list isn't registered as a sub-module, so its layers' parameters are invisible to `model.parameters()`, the optimizer, `.to(device)`, `.eval()` and `state_dict()`. `nn.ModuleList` registers them while still letting you write a custom loop, which an encoder needs to pass the attention mask to every block (`nn.Sequential` can't).
:::

## 5. The Full Classifier: Embeddings, Stack, Masked Mean Pooling, Head

`StackedTransformerClassifier` puts the pieces together. Token ids are embedded with `nn.Embedding(35, 32, padding_idx=0)`, a **learned** position embedding `nn.Embedding(6, 32)` is added (like BERT, unlike the sinusoids of the original Transformer), the result goes through the encoder stack, and the last hidden state `[B, 6, 32]` must become one vector per sentence for the classifier. The notebook uses **masked mean pooling**: multiply each token vector by its mask value (0 or 1), sum over the sequence, and divide by the number of real tokens (clamped at $10^{-9}$ so an all-padding row can't divide by zero). A final `nn.Linear(32, 2)` gives one logit per class, and cross-entropy compares them with the label.

Why not just average all six positions? Because the pads would then contribute to the sentence vector, and the same sentence would get a different representation depending on how much padding it received (in a batch with longer sentences, more padding). Why not use one special token like BERT's [CLS]? That also works, but only when the model is trained to put the sentence summary there; with a tiny model trained from scratch, averaging every real token is the simpler, more robust choice. Sentence-embedding models (Sentence-BERT and the MiniLM model used later in the RAG notebooks) also use mean pooling.

<Formula name="Masked mean pooling" tag="Sentence vector" words="Average the final token vectors, counting only the positions where the mask is 1." remember="Sum the real ones, divide by how many.">

$$
\begin{aligned}
\pred{h_{sent}} &= \frac{\sum_{t=1}^{T} \data{m_t}\, \pred{X}^{(N)}_t}{\max\big(\sum_{t=1}^{T} \data{m_t},\; 10^{-9}\big)} \\
\pred{z} &= \pred{h_{sent}}\param{W_c}^\top + \param{b_c}
\end{aligned}
$$

- $\data{m_t} \in \lbrace 0, 1 \rbrace$: attention mask; $\pred{X}^{(N)}_t$: last hidden state of token $t$.
- $\param{W_c}$: $2 \times 32$ classifier; $\pred{z}$: two logits (NEGATIVE, POSITIVE).

**Example:** "not good movie" (3 real tokens): first dimension $(-1.827 - 1.908 - 1.573)/3 = -1.769$; a plain mean over all 6 positions would give $-1.764$ because it also averages the three pad vectors.

</Formula>

![Masked mean pooling: average only the real tokens](/img/deep-learning/diagrams/masked-mean-pooling.svg)

*How to read it:* each column is one position's final 32-dimensional vector (first 4 dimensions shown, real numbers from the trained 3-block model). The mask zeroes the three pad columns before the sum, and the sum is divided by 3, not 6.

<Animation src="stacked-encoder/masked-pooling" caption="Pad columns still hold vectors after the encoder. Watch them fade out of the average: the sentence vector is the mean of the three real columns only." />

| Pooling | How | Pros | Cons |
|---|---|---|---|
| **Masked mean** (this notebook) | Average real tokens' final vectors | Uses every token; robust from scratch; independent of padding | Can blur a decisive word among many neutral ones |
| Plain mean | Average all positions | One line | Depends on how much padding a sentence got ❌ |
| [CLS] token | Read the vector of a special first token | What BERT is pre-trained for; one attention-learned summary | Needs training for that role; no [CLS] in this vocabulary |
| Max pooling | Element-wise maximum over real tokens | Keeps the strongest feature | Needs masking too (pads set to $-\infty$) |

:::tip[🎯 Interview angle]
**"How do you turn token representations into a sentence representation, and why mask the pooling?"** Either read a dedicated [CLS] vector (BERT) or pool the token vectors; the notebook uses masked mean pooling: sum the final vectors where the mask is 1 and divide by the number of real tokens. Unmasked averaging lets padding leak in, so the same sentence would get different vectors in different batches.
:::

## 6. Counting the Parameters

The parameter counts printed by the notebook (9,922 for 1 block, 27,010 for 3) can be derived exactly. Embeddings: $35 \times 32 + 6 \times 32 = 1{,}312$. One block: four $32 \times 32$ projections with biases ($4 \times 1{,}056 = 4{,}224$), the FFN ($32 \times 64 + 64 + 64 \times 32 + 32 = 4{,}192$) and two LayerNorms with $\gamma$ and $\beta$ ($4 \times 32 = 128$), so $8{,}544$. Head: $32 \times 2 + 2 = 66$. With one block, $1{,}312 + 8{,}544 + 66 = 9{,}922$; each extra block adds another $8{,}544$, giving $27{,}010$ for three. The number of heads doesn't appear anywhere: four heads of 8 dimensions use the same $32 \times 32$ matrices as one head of 32.

<Formula name="Parameters of the stacked classifier" tag="Model size" words="Embeddings plus N identical blocks plus the head; each block has four attention projections, a two-layer feed-forward network and two LayerNorms." remember="Each block costs the same.">

$$
\begin{aligned}
\text{block} &= 4(\hyper{D}^2 + \hyper{D}) + (2\hyper{D}\hyper{d_{ff}} + \hyper{d_{ff}} + \hyper{D}) + 4\hyper{D} \\
\text{total} &= (V + T)\hyper{D} + \hyper{N} \cdot \text{block} + 2\hyper{D} + 2
\end{aligned}
$$

- $V = 35$ vocabulary entries, $T = 6$ positions, $\hyper{D} = 32$, $\hyper{d_{ff}} = 64$, 2 classes.

**Example:** block $= 4{,}224 + 4{,}192 + 128 = 8{,}544$; total $= 1{,}312 + 8{,}544\hyper{N} + 66$: 9,922 for $N = 1$, 27,010 for $N = 3$.

</Formula>

<EncoderStackParams />

:::tip[🎯 Interview angle]
**"How do parameters scale with depth, width and heads in an encoder?"** Linearly with depth (each block adds the same count), quadratically with width ($\approx 4D^2$ attention $+ 2 D d_{ff}$ FFN per block, so $12D^2$ when $d_{ff} = 4D$) and not at all with the number of heads, which only splits $D$. For the notebook's model: 8,544 per block, 9,922 total with one block and 27,010 with three.
:::

## 7. Training 1 Block vs 3 Blocks, and Why Training Accuracy Proves Nothing

Both models are created with `torch.manual_seed(7)`, so they start from the same embedding and first-block weights, and are trained identically: Adam, learning rate 3e-3, batch 4, 60 epochs, cross-entropy. The single-block model's average loss goes 0.8294 → 0.2002 (epoch 10) → 0.0004 (epoch 60); the stacked model's goes 0.8185 → 0.0062 → 0.0005. Both reach **100% training accuracy**. The notebook's comparison table stops there, with final losses of 0.000441 and 0.000492.

That comparison can't tell the models apart. With 18 training examples and 9,922 or 27,010 parameters, both models can memorize the training set, so 100% is guaranteed and uninformative. The notebook's own extra test sentences hint at the problem: on "this movie is terrible" the single-block model says NEGATIVE (0.9598) but the stacked model says **POSITIVE (0.9180)**. "terrible" appeared only once in training ("terrible acting and story"), and the stacked model learned something else from "this movie is …".

To measure generalization, we added a **held-out set**: 16 new sentences built only from training words but in combinations the model never saw ("i love this film", "what a bad movie", "not good film", "awful acting", …), 8 positive and 8 negative. Then we retrained both architectures from 20 different seeds (same recipe) and scored each on the held-out set:

| Model | Parameters | Training accuracy | Held-out accuracy, 20 seeds (mean ± sd) | Worst / best seed |
|---|---|---|---|---|
| 1 encoder block | 9,922 | 100% | **84.1% ± 6.4** | 75.0% / 93.8% |
| 3 encoder blocks | 27,010 | 100% | **86.3% ± 7.6** | 68.8% / 100% |

The 2.2-point difference is far smaller than the 6–8 point spread between seeds: on this data, three blocks are **not measurably better** than one. That is the expected result, not a failure. Depth adds capacity, and capacity only pays off when there's enough data to use it; with 18 phrases there is almost nothing for extra rounds of refinement to discover. BERT is the other end of the spectrum: 12 blocks *and* 3.3 billion words of pre-training, which is why the paper's ablation (Table 6) finds that bigger is better even on small downstream tasks. The notebook's models trained on the notebook's seeds run live in the widget below.

<TinySentimentLive />

:::danger[Interview Trap: "the deeper model reached 100% training accuracy, so it is better"]
Training accuracy measures memorization. With fewer examples than parameters, any reasonable model fits the training set. Compare on held-out data, and over several seeds: here the spread between seeds (±6–8 points) is bigger than the difference between 1 and 3 blocks (2.2 points).
:::

:::analogy[Two students and one practice sheet]
Two students memorize the same 18-question practice sheet and both score 100% on it. That says nothing about who understands the subject. Give them 16 new questions using the same words and they score about the same, and which one wins depends mostly on the day (the seed). The student with the bigger notebook (more blocks) only pulls ahead after reading the whole textbook, and that is what pre-training is.
:::

:::tip[🎯 Interview angle]
**"Your 3-layer model and 1-layer model both get 100% training accuracy. Which is better?"** Neither conclusion follows: with tiny data both can memorize. I'd evaluate on a held-out set and over several seeds; on this project, 1 block scored 84.1% ± 6.4 and 3 blocks 86.3% ± 7.6 on 16 held-out sentences over 20 seeds, which is within noise. Depth helps when there's enough data, which is why large encoders are pre-trained.
:::

## 8. Watching Representations Change Layer by Layer

The notebook's `inspect_hidden_state_changes` runs "not good movie" through the trained 3-block model with `return_hidden_states=True` and reports two numbers per stage, computed on the three real tokens only: the mean norm of the token vectors, and the mean absolute change from the previous stage. The shape stays `(1, 6, 32)` everywhere; the numbers do not.

| Stage | Shape | Mean token norm | Mean absolute change from previous |
|---|---|---|---|
| Embedding output | (1, 6, 32) | 8.7262 | – |
| After block 1 | (1, 6, 32) | 5.6187 | 0.8485 |
| After block 2 | (1, 6, 32) | 5.7441 | 0.5855 |
| After block 3 | (1, 6, 32) | 6.0104 | 0.6335 |

The norms tell a story of their own. The embedding output is the sum of two vectors drawn from $\mathcal{N}(0, 1)$ at initialization, so its norm is near $\sqrt{2 \times 32} = 8$ (8.73 after training). Every block ends with a LayerNorm, which rescales each token vector to mean 0 and variance 1 across its 32 features before applying the learned $\gamma$ and $\beta$; with $\gamma = 1$ and $\beta = 0$ that gives a norm of exactly $\sqrt{32} = 5.66$, and the trained $\gamma, \beta$ move it to 5.62–6.01. The changes show that every block keeps rewriting the representation, even though the shape never changes. If all three hidden states were *identical*, something would be wrong: the blocks were not called, were zero-initialized, or the forward returns its input.

<Animation src="stacked-encoder/depth-refinement" caption="The shape label never changes while the numbers do. Notice the norm drop to about the square root of 32 after the first block: that is the LayerNorm at the end of every block." />

:::tip[🎯 Interview angle]
**"Why are intermediate hidden states useful?"** They show what each layer does to the representation: how much it changes, where information appears (probing), and whether a layer is dead or exploding. They are also features in their own right; BERT's feature-based experiments concatenate the last four layers. `output_hidden_states=True` returns them in Hugging Face, with the embedding output first.
:::

## 9. Attention Across Layers

Each block has its own $W_q$, $W_k$, $W_v$, so each can learn a different mixing pattern. For "not good movie", head 1 of the trained 3-block model looks like this (rows are queries, columns keys, real tokens only; the pad columns are exactly 0):

| Query \ key | Layer 1: not | good | movie | Layer 2: not | good | movie | Layer 3: not | good | movie |
|---|---|---|---|---|---|---|---|---|---|
| **not** | 0.203 | 0.001 | 0.795 | 0.230 | 0.290 | 0.480 | 0.233 | 0.430 | 0.337 |
| **good** | 0.055 | 0.836 | 0.109 | 0.290 | 0.296 | 0.414 | 0.230 | 0.441 | 0.330 |
| **movie** | 0.005 | 0.994 | 0.001 | 0.382 | 0.218 | 0.400 | 0.230 | 0.430 | 0.340 |

Layer 1 is sharp: "good" and "movie" both look almost entirely at "good", and "not" mostly at "movie". Layers 2 and 3 spread their attention much more evenly, and in layer 3 all three rows are nearly identical, as if every token were collecting the same sentence-level summary before pooling. The notebook's teaching interpretation (block 1 for local word relations, block 2 for phrase meaning, block 3 for a task-friendly representation) is a useful picture but, as it says itself, not a rule; the widget above lets you browse all 3 layers × 4 heads for any sentence.

<Animation src="stacked-encoder/attention-by-layer" caption="Layer by layer, head 1 of the trained model. Layer 1 concentrates on single tokens, later layers average more broadly." />

:::danger[Interview Trap: "the attention map explains the decision"]
Attention weights show where information was *mixed*, not why the output came out as it did. The value vectors, the FFNs, the residual paths and the other heads all shape the prediction, and different attention patterns can lead to the same output. Treat attention maps as a diagnostic (for example, to catch attention on padding, which means the mask is broken), not as an explanation.
:::

:::tip[🎯 Interview angle]
**"Why can different layers have different attention patterns, and are attention maps reliable explanations?"** Each layer has its own projection weights and receives a different input (the previous layer's output), so it can learn a different pattern. Attention maps are useful diagnostics (padding leaks, degenerate heads) but not faithful explanations; attribution methods or ablations are needed to claim why a model decided something.
:::

## 10. From This Mini Model to BERT-style Encoders

The notebook ends where BERT begins. Our model and a BERT-style encoder share the same skeleton: tokenize → ids → token + position embeddings → many encoder blocks → contextual token representations → a task head. What differs is scale, and above all **where the weights come from**. Our 27,010 weights start random and see 18 phrases. BERT-base's 110 million start from pre-training on 3.3 billion words with the masked-LM and next-sentence tasks, and only then meet a downstream task. That is the step from this page to the next one: instead of training an encoder from scratch, download a pre-trained one with Hugging Face, and let a `pipeline()` hide the tokenization, padding, mask, forward pass and softmax we wrote by hand here.

| | This notebook's model | BERT-base | DistilBERT (next notebooks) |
|---|---|---|---|
| Tokenizer | Whitespace, 35 entries, `<UNK>` | WordPiece, 30,522 | WordPiece (cased QnA model: 28,996) |
| Special tokens | `<PAD>`, `<UNK>` | [CLS], [SEP], [PAD], [MASK], [UNK] | Same as BERT |
| Max length | 6 | 512 | 512 |
| Embeddings | token + learned position | token + learned position + segment | token + position (no segments) |
| Blocks × width × heads | 1 or 3 × 32 × 4 | 12 × 768 × 12 | 6 × 768 × 12 |
| FFN | 64, ReLU | 3072, GELU | 3072, GELU |
| Sentence vector | masked mean pooling | [CLS] vector $C$ (+ pooler) | [CLS] vector |
| Parameters | 27,010 | 110M | 66M (65,192,450 with the QnA head, notebook 20) |
| Training | 18 phrases, from scratch | pre-trained (MLM + NSP), then fine-tuned | distilled from BERT, then fine-tuned |

:::tip[🎯 Interview angle]
**"How does a small from-scratch Transformer classifier relate to BERT?"** Same architecture family: embeddings plus a stack of encoder blocks producing contextual token vectors, then a task head. BERT differs in scale (12 × 768 vs 3 × 32), in its WordPiece tokenizer and special tokens, in using the [CLS] vector, and mainly in pre-training on billions of words, which is what makes depth pay off and lets a small labeled dataset suffice.
:::

## ⚙️ Going Deeper (Encoder Stack Internals)

**Why LayerNorm fixes the norm.** LayerNorm standardizes each token vector across its $D$ features: $\hat x = (x - \mu)/\sqrt{\sigma^2 + \epsilon}$, so $\sum_k \hat x_k^2 = D \cdot \sigma^2/(\sigma^2 + \epsilon) \approx D$ and $\lVert \hat x \rVert \approx \sqrt D$. Every post-LN block therefore hands the next one vectors of norm about $\sqrt{D}$ (5.66 for $D = 32$), scaled only by the learned $\gamma$. That is the reason deep post-LN stacks don't blow up in the forward direction, and why the embedding output (8.73) is the only stage with a different scale.

**Masking with $-10^9$ vs $-\infty$.** The notebook uses `masked_fill(~mask, -1e9)`. In float32 that's safely below any real score and $e^{-10^9}$ underflows to exactly 0. In float16 the largest representable magnitude is 65,504, and `masked_fill(..., -1e9)` on a half-precision tensor fails with `RuntimeError: value cannot be converted to type c10::Half without overflow`; library code uses `torch.finfo(scores.dtype).min` (−65,504 in float16) instead. A row whose keys are *all* masked (an empty sentence) would get a uniform softmax over pads with $-10^9$, and NaN with $-\infty$; real implementations guard against all-padding inputs.

**Pre-LN vs post-LN when stacking deep.** This notebook (like the 2017 Transformer and BERT) uses post-LN, $\operatorname{LN}(x + F(x))$. With many layers, post-LN needs learning-rate warmup to train stably; most modern deep models use pre-LN, $x + F(\operatorname{LN}(x))$, which keeps an identity path through the whole stack. At 3 layers the difference doesn't matter.

**Reproducibility depends on everything that touches the RNG.** The shuffled `DataLoader` draws from PyTorch's global generator each time it is iterated, so *evaluating* the single-block model on `train_loader` before training the stacked one changes the stacked model's batch order. Re-running the notebook in the same order reproduces its numbers exactly; evaluating at a different point does not (the stacked model then predicts "this movie is terrible" as POSITIVE 0.9993 instead of 0.9180). Use a non-shuffled loader or a separate `torch.Generator` for evaluation.

**Cost of depth.** Compute and activation memory grow linearly with the number of blocks; attention cost per block is $O(T^2 D)$ and FFN cost $O(T D d_{ff})$. For $T = 6$ both are trivial; for BERT's $T = 512$ the attention matrices dominate memory (chapter 7 of the BERT page).

**What the hidden-states API mirrors.** Hugging Face models return `last_hidden_state` `[B, T, D]`, `hidden_states` (a tuple of $N + 1$ tensors, embeddings first) and `attentions` ($N$ tensors of `[B, heads, T, T]`), exactly the dictionary the notebook's stack builds by hand.

## 💻 PyTorch Implementation

:::tip[Run it yourself]
Verified on Python 3.13 and PyTorch 2.15 (CPU); every output below is copied from a real run. Blocks 1–5 are one file, `stacked_encoder.py`, taken from the notebook's cells (only the imports were trimmed and the prints moved out); blocks 6–8 import it. Training both models takes a few seconds; the 20-seed comparison about a minute. With the notebook's seeds and order, the weights come out bit-for-bit identical to the notebook's (and to the widget above).
:::

**What was corrected from the notebook.** (1) The notebook compares the models only on training accuracy; block 8 adds a held-out set and a 20-seed comparison. (2) Its last two cells (the Hugging Face `pipeline` and `AutoModelForSequenceClassification` preview) failed with `ModuleNotFoundError` because of the local environment; they belong to the next notebook and are covered there. (3) The order of the evaluation call matters for reproducing its numbers (block 7 explains why).

### 1. Data, vocabulary, padding and the attention mask

Whitespace tokens, `<PAD>` = 0 and `<UNK>` = 1, truncate and pad to 6, mask 1 for real tokens.

```python title="stacked_encoder.py (part 1)" showLineNumbers
import math, random
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

raw_data = [
    ("i love this movie", 1),
    ("this movie is great", 1),
    ("amazing acting and story", 1),
    ("what a wonderful film", 1),
    ("i enjoyed the movie", 1),
    ("this was a good experience", 1),
    ("happy and satisfied", 1),
    ("excellent direction", 1),
    ("not bad movie", 1),

    ("i hate this movie", 0),
    ("this movie is bad", 0),
    ("terrible acting and story", 0),
    ("what a boring film", 0),
    ("i disliked the movie", 0),
    ("this was a poor experience", 0),
    ("sad and disappointed", 0),
    ("awful direction", 0),
    ("not good movie", 0),
]

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def tokenize(text):
    return text.lower().split()

counter = Counter()
for text, label in raw_data:
    counter.update(tokenize(text))

vocab = {
    PAD_TOKEN: 0,
    UNK_TOKEN: 1,
}

for word, freq in counter.items():
    if word not in vocab:
        vocab[word] = len(vocab)

idx_to_word = {idx: word for word, idx in vocab.items()}

PAD_ID = vocab[PAD_TOKEN]
UNK_ID = vocab[UNK_TOKEN]

max_len = 6

def encode_text(text, vocab=vocab, max_len=max_len):
    tokens = tokenize(text)
    token_ids = [vocab.get(token, UNK_ID) for token in tokens]

    token_ids = token_ids[:max_len]
    attention_mask = [1] * len(token_ids)

    pad_length = max_len - len(token_ids)
    token_ids = token_ids + [PAD_ID] * pad_length
    # highlight-next-line
    attention_mask = attention_mask + [0] * pad_length

    return token_ids, attention_mask

class TinySentimentDataset(Dataset):
    def __init__(self, raw_data):
        self.raw_data = raw_data

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        text, label = self.raw_data[idx]
        input_ids, attention_mask = encode_text(text)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

dataset = TinySentimentDataset(raw_data)

train_loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)
```

### 2. Encoder components: masked multi-head attention, Add & Norm, FFN, block

The mask is broadcast to `[B, 1, 1, T]` and fills padded keys with −1e9 before the softmax.

```python title="stacked_encoder.py (part 2)" showLineNumbers
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.last_attention_weights = None

    def forward(self, x, attention_mask=None):
        """
        x shape:
            [batch_size, seq_len, d_model]

        attention_mask shape:
            [batch_size, seq_len]
            1 = real token
            0 = padding token
        """

        batch_size, seq_len, d_model = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].bool()
            # highlight-next-line
            scores = scores.masked_fill(~key_mask, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)

        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, d_model)

        output = self.out_proj(context)

        self.last_attention_weights = attention_weights.detach()

        return output


class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_output):
        # highlight-next-line
        return self.layer_norm(x + self.dropout(sublayer_output))


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()

        self.self_attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        self.add_norm_1 = AddNorm(d_model=d_model, dropout=dropout)

        self.feed_forward = PositionWiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )

        self.add_norm_2 = AddNorm(d_model=d_model, dropout=dropout)

    def forward(self, x, attention_mask=None):
        attention_output = self.self_attention(x, attention_mask=attention_mask)
        x = self.add_norm_1(x, attention_output)

        ffn_output = self.feed_forward(x)
        x = self.add_norm_2(x, ffn_output)

        return x
```

### 3. The encoder stack

`nn.ModuleList` registers the blocks; the loop passes the mask to each one and can collect every hidden state and attention map.

```python title="stacked_encoder.py (part 3)" showLineNumbers
class TransformerEncoderStack(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.0):
        super().__init__()

        # highlight-next-line
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x,
        attention_mask=None,
        return_hidden_states=False,
        return_attentions=False
    ):
        hidden_states = []
        attentions = []

        if return_hidden_states:
            hidden_states.append(x)

        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)

            if return_hidden_states:
                hidden_states.append(x)

            if return_attentions:
                attentions.append(layer.self_attention.last_attention_weights)

        if return_hidden_states or return_attentions:
            output = {
                "last_hidden_state": x
            }

            if return_hidden_states:
                output["hidden_states"] = hidden_states

            if return_attentions:
                output["attentions"] = attentions

            return output

        return x
```

### 4. The classifier with masked mean pooling

Token + learned position embeddings, the stack, pooling over real tokens only, and a linear head.

```python title="stacked_encoder.py (part 4)" showLineNumbers
class StackedTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        d_model=32,
        num_heads=4,
        d_ff=64,
        num_layers=2,
        num_classes=2,
        dropout=0.0,
        pad_id=0
    ):
        super().__init__()

        self.max_len = max_len
        self.pad_id = pad_id

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_id
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=max_len,
            embedding_dim=d_model
        )

        self.encoder = TransformerEncoderStack(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout
        )

        self.classifier = nn.Linear(d_model, num_classes)

    def masked_mean_pooling(self, token_representations, attention_mask):
        """
        token_representations shape:
            [batch_size, seq_len, d_model]

        attention_mask shape:
            [batch_size, seq_len]
        """

        mask = attention_mask.unsqueeze(-1).type_as(token_representations)

        # highlight-next-line
        summed = (token_representations * mask).sum(dim=1)
        # highlight-next-line
        denominator = mask.sum(dim=1).clamp(min=1e-9)

        pooled = summed / denominator

        return pooled

    def forward(
        self,
        input_ids,
        attention_mask,
        return_hidden_states=False,
        return_attentions=False
    ):
        batch_size, seq_len = input_ids.shape

        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)

        encoder_outputs = self.encoder(
            x,
            attention_mask=attention_mask,
            return_hidden_states=return_hidden_states,
            return_attentions=return_attentions
        )

        if isinstance(encoder_outputs, dict):
            last_hidden_state = encoder_outputs["last_hidden_state"]
        else:
            last_hidden_state = encoder_outputs

        pooled_output = self.masked_mean_pooling(
            last_hidden_state,
            attention_mask
        )

        logits = self.classifier(pooled_output)

        if return_hidden_states or return_attentions:
            encoder_outputs["pooled_output"] = pooled_output
            encoder_outputs["logits"] = logits
            return encoder_outputs

        return logits
```

### 5. Training and evaluation utilities

Adam at 3e-3, cross-entropy, the average loss printed every 10 epochs; accuracy over a loader.

```python title="stacked_encoder.py (part 5)" showLineNumbers
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, train_loader, epochs=60, lr=3e-3, device=device):
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f}")

    return losses


@torch.no_grad()
def evaluate_accuracy(model, data_loader, device=device):
    model.eval()
    model = model.to(device)

    correct = 0
    total = 0

    for input_ids, attention_mask, labels in data_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        logits = model(input_ids, attention_mask)
        predictions = torch.argmax(logits, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.numel()

    return correct / total
```

### 6. Shapes through the stack

Every hidden state is `[B, T, D]` and every attention tensor `[B, heads, T, T]`, whatever the depth.

```python title="shapes.py" showLineNumbers
from stacked_encoder import *

print(len(vocab), dataset[0][0].tolist(), dataset[0][1].tolist())    # 35 [2, 3, 4, 5, 0, 0] [1, 1, 1, 1, 0, 0]
print(encode_text("not good movie"), encode_text("this film rocks"))
# ([25, 19, 5, 0, 0, 0], [1, 1, 1, 0, 0, 0]) ([4, 15, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0])   "rocks" -> <UNK> = 1

batch_ids, batch_mask, batch_labels = next(iter(train_loader))
x = torch.randn(4, max_len, 32)                                         # any [B, T, D] input
stack = TransformerEncoderStack(d_model=32, num_heads=4, d_ff=64, num_layers=3)
out = stack(x, attention_mask=batch_mask, return_hidden_states=True, return_attentions=True)
print([tuple(h.shape) for h in out["hidden_states"]])     # [(4, 6, 32), (4, 6, 32), (4, 6, 32), (4, 6, 32)]
print([tuple(a.shape) for a in out["attentions"]])        # [(4, 4, 6, 6), (4, 4, 6, 6), (4, 4, 6, 6)]
```

### 7. Train 1 block vs 3 blocks, predict, inspect the layers

Same seed for both models, the notebook's training recipe, then the notebook's test sentences, hidden-state norms and changes, and layer-1 attention for "not good movie".

```python title="train_compare.py" showLineNumbers
from stacked_encoder import *

config = dict(vocab_size=len(vocab), max_len=max_len, d_model=32, num_heads=4, d_ff=64,
              num_classes=2, dropout=0.0, pad_id=PAD_ID)
torch.manual_seed(7); single = StackedTransformerClassifier(**config, num_layers=1)
torch.manual_seed(7); stacked = StackedTransformerClassifier(**config, num_layers=3)
print(count_parameters(single), count_parameters(stacked))

single_losses = train_model(single, train_loader, epochs=60, lr=3e-3)
# highlight-next-line
single_acc = evaluate_accuracy(single, train_loader)     # iterating the shuffled loader uses the RNG, so the order matters
stacked_losses = train_model(stacked, train_loader, epochs=60, lr=3e-3)
print(single_acc, evaluate_accuracy(stacked, train_loader))

@torch.no_grad()
def predict(model, sentence):
    model.eval()
    ids, mask = encode_text(sentence)
    p = F.softmax(model(torch.tensor([ids]), torch.tensor([mask])), dim=-1)[0]
    return ("POSITIVE" if p[1] > p[0] else "NEGATIVE", round(p.max().item(), 4))

for s in ["i love this movie", "this movie is terrible", "not good movie", "not bad movie"]:
    print(f"{s:24s}", predict(single, s), predict(stacked, s))

with torch.no_grad():
    ids, mask = encode_text("not good movie")
    out = stacked(torch.tensor([ids]), torch.tensor([mask]), return_hidden_states=True, return_attentions=True)
hs = out["hidden_states"]
print([round(h[0, :3].norm(dim=-1).mean().item(), 4) for h in hs])
print([round((hs[i][0, :3] - hs[i - 1][0, :3]).abs().mean().item(), 4) for i in range(1, 4)])
print(out["attentions"][0][0, 0, :3, :3].numpy().round(3))
print(out["attentions"][0][0, 0, :3, 3:].abs().max().item())
torch.save({"single": single.state_dict(), "stacked": stacked.state_dict()}, "tiny_models.pt")

# Output:
# 9922 27010
# Epoch 01 | Loss: 0.8294
# Epoch 10 | Loss: 0.2002
# Epoch 20 | Loss: 0.0025
# Epoch 30 | Loss: 0.0010
# Epoch 40 | Loss: 0.0007
# Epoch 50 | Loss: 0.0006
# Epoch 60 | Loss: 0.0004
# Epoch 01 | Loss: 0.8185
# Epoch 10 | Loss: 0.0062
# Epoch 20 | Loss: 0.0016
# Epoch 30 | Loss: 0.0011
# Epoch 40 | Loss: 0.0008
# Epoch 50 | Loss: 0.0006
# Epoch 60 | Loss: 0.0005
# 1.0 1.0
# i love this movie        ('POSITIVE', 0.9996) ('POSITIVE', 0.9996)
# this movie is terrible   ('NEGATIVE', 0.9598) ('POSITIVE', 0.918)
# not good movie           ('NEGATIVE', 0.9997) ('NEGATIVE', 0.9995)
# not bad movie            ('POSITIVE', 0.9995) ('POSITIVE', 0.9996)
# [8.7262, 5.6187, 5.7441, 6.0104]
# [0.8485, 0.5855, 0.6335]
# [[0.203 0.001 0.795]
#  [0.055 0.836 0.109]
#  [0.005 0.994 0.001]]
# 0.0
```

The single-block model's log and every number after it match the notebook. The stacked model gets "this movie is terrible" wrong with 0.918 confidence; the last line confirms padded keys receive exactly zero attention.

### 8. The fix: held-out sentences and 20 seeds

New combinations of training words, scored for both depths over 20 seeds.

```python title="depth_vs_data.py" showLineNumbers
import io, contextlib, statistics
from stacked_encoder import *

# new combinations of training words: every word is in the vocabulary, no sentence is in raw_data
heldout = [("i love this film", 1), ("what a great movie", 1), ("this film is great", 1), ("i enjoyed the story", 1),
           ("good acting and direction", 1), ("a wonderful experience", 1), ("not bad film", 1), ("excellent acting", 1),
           ("i hate this film", 0), ("what a bad movie", 0), ("this film is terrible", 0), ("i disliked the story", 0),
           ("poor acting and direction", 0), ("a boring experience", 0), ("not good film", 0), ("awful acting", 0)]

@torch.no_grad()
def accuracy(model, data):
    model.eval()
    hits = 0
    for text, label in data:
        ids, mask = encode_text(text)
        hits += model(torch.tensor([ids]), torch.tensor([mask])).argmax(-1).item() == label
    return hits / len(data)

config = dict(vocab_size=len(vocab), max_len=max_len, d_model=32, num_heads=4, d_ff=64,
              num_classes=2, dropout=0.0, pad_id=PAD_ID)
scores = {1: [], 3: []}
for seed in range(20):
    for layers in (1, 3):
        torch.manual_seed(seed)
        model = StackedTransformerClassifier(**config, num_layers=layers)
        with contextlib.redirect_stdout(io.StringIO()):            # silence the per-epoch log
            train_model(model, train_loader, epochs=60, lr=3e-3)
        scores[layers].append(accuracy(model, heldout))
for layers, acc in scores.items():
    print(layers, round(statistics.mean(acc), 3), round(statistics.pstdev(acc), 3), min(acc), max(acc))
# 1 0.841 0.064 0.75 0.9375
# 3 0.863 0.076 0.6875 1.0
```

## 🚨 Interview Gotchas & Edge Cases

:::warning[The Most Common Question]
**"What happens to the tensor as it goes through a stack of encoder blocks, and why stack them at all?"**

The shape never changes: `[B, T, D]` in, `[B, T, D]` out of every block, with attention maps of `[B, heads, T, T]` along the way. What changes is the content: each block lets every token attend to every other token's *current* representation and then transforms it, so stacking gives repeated, increasingly contextual refinement (one block = one round of mixing). Each block has its own weights, so layers can learn different patterns. Depth adds capacity linearly (8,544 parameters per block in the notebook's model), and it only pays off with enough data: on 18 sentences, 3 blocks were within noise of 1 block on held-out data, which is why deep encoders like BERT are pre-trained.
:::

- **Training accuracy is not evidence:** both models hit 100%; held-out over 20 seeds: 84.1% ± 6.4 (1 block) vs 86.3% ± 7.6 (3 blocks).
- **`nn.ModuleList`, not a Python list:** otherwise the blocks' parameters aren't registered and the optimizer never updates them.
- **`nn.Sequential` can't pass the mask** to each block; use `ModuleList` + loop.
- **`padding_idx` is not masking:** it only zeroes the `<PAD>` embedding; the attention mask and masked pooling do the actual ignoring.
- **Mask the keys, then the pooling:** padded keys get −1e9 before softmax; pooled mean divides by the number of real tokens.
- **Plain mean pooling leaks padding** into the sentence vector (−1.764 vs −1.769 in dimension 1 for "not good movie").
- **−1e9 in float16** raises an overflow error; use `torch.finfo(dtype).min`.
- **Identical hidden states across layers** mean the blocks weren't applied or are degenerate.
- **Attention on pads** means the mask is missing or wrongly shaped (it should broadcast as `[B, 1, 1, T]`).
- **Heads don't add parameters;** depth and width do.
- **Unknown words become `<UNK>`:** "hello world movie" is `<UNK> <UNK> movie`, and both models call it POSITIVE with over 0.998, confidently, from "movie" alone.
- **Reproducibility:** iterating a shuffled `DataLoader` (even for evaluation) consumes random numbers and changes later training.
- **Post-LN keeps token norms near $\sqrt{D}$** after every block (5.62–6.01 for $D = 32$).

## ⚡ Essential Interview Recall

A one-screen revision list. Every point is explained in the chapters above.

**The project**
- 18 phrases (9 positive, 9 negative), whitespace tokens, vocab 35 incl. `<PAD>` 0 and `<UNK>` 1, max_len 6, batch 4.
- $D$ = 32, 4 heads (8 each), $d_{ff}$ = 64, ReLU, post-LN, dropout 0, Adam 3e-3, 60 epochs.

**Stacking**
- $X^{(l)} = \operatorname{Block}_l(X^{(l-1)})$; shape `[B, T, D]` everywhere; attentions `[B, h, T, T]`.
- `nn.ModuleList` + loop; return `hidden_states` ($N+1$) and `attentions` ($N$) like Hugging Face.
- Depth = rounds of mixing + transforming; each block has its own weights.

**Classifier**
- token emb + learned position emb → stack → masked mean pooling → Linear(32, 2) → cross-entropy.
- Masked mean: $\sum m_t X_t / \max(\sum m_t, 10^{-9})$.

**Numbers**
- Parameters: 1,312 embeddings + 8,544 per block + 66 head → 9,922 (1 block), 27,010 (3 blocks).
- Loss at epoch 60: 0.0004 (1 block), 0.0005 (3 blocks); both 100% training accuracy.
- Held-out, 20 seeds: 84.1% ± 6.4 vs 86.3% ± 7.6: depth not measurably better on tiny data.
- "not good movie": mean norm 8.73 → 5.62 → 5.74 → 6.01; change 0.85, 0.59, 0.63.
- Layer 1 attention sharp ("movie" → "good" 0.994), layers 2–3 diffuse; pads get 0.

**Bridge**
- Same skeleton as BERT; BERT = 12 × 768 × 12, WordPiece, [CLS], pre-trained on 3.3B words.

:::speak[Why stack Transformer encoder blocks?]
An encoder block takes a sequence of token vectors and returns a sequence of the same shape: attention lets every token gather information from the others, and the feed-forward network transforms each token on its own, each wrapped in a residual connection and LayerNorm. Because input and output shapes match, I can stack blocks in an nn.ModuleList and loop over them, passing the attention mask to each. Every block has its own weights and works on the previous block's already-contextualized vectors, so depth gives repeated refinement: later layers can model relations between phrases, not just words. The cost is linear in depth; in the small sentiment model I built, each block added 8,544 parameters. But depth only helps with enough data: with 18 training sentences, one and three blocks both memorized the training set, and on held-out sentences over 20 seeds the difference was within noise. That's exactly why BERT-style encoders are deep and pre-trained on billions of words.
:::

## 📐 Formula Sheet

| Concept | Formula | In plain words |
|---|---|---|
| Embedding input | $\data{X}^{(0)} = \param{E_{tok}}[\data{ids}] + \param{E_{pos}}[0..T-1]$ | Token vector plus learned position vector |
| Masked scores | $\pred{S_h} = (\data{X}\param{W_q})_h(\data{X}\param{W_k})_h^\top / \sqrt{\hyper{d_h}} + \data{M}$, $\data{M}_{ij} = -10^9$ for pad keys | Scaled dot products, pads pushed to −∞ |
| Multi-head output | $[\operatorname{softmax}(\pred{S_1})V_1; \ldots; \operatorname{softmax}(\pred{S_h})V_h]\param{W_o}$ | Mix values per head, concatenate, project |
| Encoder block | $\pred{U} = \operatorname{LN}(\data{X} + \operatorname{MHSA}(\data{X}))$; $\operatorname{LN}(\pred{U} + \operatorname{FFN}(\pred{U}))$ | Mix, add, norm; transform, add, norm |
| Stack | $\pred{X}^{(l)} = \operatorname{Block}_l(\pred{X}^{(l-1)})$, $l = 1..\hyper{N}$ | Same shape, repeated refinement |
| Masked mean pooling | $\pred{h_{sent}} = \sum_t \data{m_t}\pred{X}^{(N)}_t / \max(\sum_t \data{m_t}, 10^{-9})$ | Average real tokens only |
| Classifier | $\pred{z} = \pred{h_{sent}}\param{W_c}^\top + \param{b_c}$, $\loss{L} = -\log\operatorname{softmax}(\pred{z})_{\data{y}}$ | Two logits, cross-entropy |
| Parameters per block | $4(\hyper{D}^2 + \hyper{D}) + 2\hyper{D}\hyper{d_{ff}} + \hyper{d_{ff}} + \hyper{D} + 4\hyper{D}$ | 8,544 for D 32, d_ff 64 |
| Total parameters | $(V + T)\hyper{D} + \hyper{N}\cdot\text{block} + 2\hyper{D} + 2$ | 9,922 (N = 1), 27,010 (N = 3) |
| LayerNorm norm | $\lVert \operatorname{LN}(x) \rVert \approx \sqrt{\hyper{D}}$ (with $\gamma = 1$, $\beta = 0$) | Why norms sit near 5.66 after each block |

## Concept-Check Answers (from the notebook)

### Encoder stack

<details>
<summary><strong>Q1. What is the role of `nn.ModuleList` in the encoder stack?</strong></summary>

It holds the encoder blocks *and registers them as sub-modules*, so their parameters are part of `model.parameters()` (trained by the optimizer), move with `.to(device)`, follow `.train()`/`.eval()` and are saved in `state_dict()`. Unlike `nn.Sequential`, it doesn't impose a forward pass, so the stack can loop over the blocks and pass the attention mask to each.
</details>

<details>
<summary><strong>Q2. If we pass an input of shape [4, 8, 32] through 3 encoder blocks, what is the final output shape?</strong></summary>

`[4, 8, 32]`. Every block preserves `[batch, seq_len, d_model]`; the attention maps inside each block are `[4, heads, 8, 8]`.
</details>

<details>
<summary><strong>Q3. What error may happen if block 1 outputs d_model = 32 but block 2 expects d_model = 64?</strong></summary>

Block 2's first projection (`nn.Linear(64, 64)`) receives 32 features and raises a matrix-multiplication shape error (`mat1 and mat2 shapes cannot be multiplied`). All blocks in a stack must share `d_model`, or a projection must be inserted between them.
</details>

<details>
<summary><strong>Q4. Why does stacking not automatically mean that the sequence length changes?</strong></summary>

Self-attention produces one output per query position and the FFN works per position, so each block returns exactly $T$ vectors of size $D$. Nothing in a block pools or downsamples; the length only changes if you add such an operation explicitly.
</details>

### Training and comparison

<details>
<summary><strong>Q1. Why does the stacked model have more parameters than the single-block model?</strong></summary>

Each block has its own attention projections, FFN and LayerNorms (8,544 parameters here) and they are not shared. Three blocks add two more of them: 9,922 + 2 × 8,544 = 27,010. Embeddings and the head are the same in both.
</details>

<details>
<summary><strong>Q2. Does increasing the number of encoder blocks change the shape of the final token representations?</strong></summary>

No. The last hidden state is `[B, T, D]` for any depth (here `[4, 6, 32]`), so the pooling and the classifier are identical for 1 and 3 blocks.
</details>

<details>
<summary><strong>Q3. If the stacked model trains slower, is that expected? Why?</strong></summary>

Yes, per step: each forward and backward pass goes through three blocks instead of one, so compute and activation memory roughly triple for the block part. (Per *epoch of convergence* it may still reach low loss quickly: here its loss was 0.0062 at epoch 10 vs 0.2002 for one block.) Deeper post-LN stacks can also be harder to optimize, which is why big models use warmup or pre-LN.
</details>

<details>
<summary><strong>Q4. Why is training accuracy alone not enough to prove that the stacked model is better?</strong></summary>

Both models can memorize 18 sentences, so both reach 100% and the metric can't distinguish them. You need held-out data and repeated runs: over 20 seeds on 16 held-out sentences, 1 block scored 84.1% ± 6.4 and 3 blocks 86.3% ± 7.6, a difference within the seed-to-seed noise.
</details>

### Layer-wise representations

<details>
<summary><strong>Q1. Why do hidden states change after every encoder block?</strong></summary>

Each block applies attention (each token becomes a weighted mix of all tokens' value vectors), a residual connection, LayerNorm, a per-token FFN and another residual + LayerNorm, all with that block's own learned weights. For "not good movie" the mean absolute change was 0.85, 0.59 and 0.63 for blocks 1, 2 and 3.
</details>

<details>
<summary><strong>Q2. If the embedding output has shape [1, 8, 32], what is the shape after block 3?</strong></summary>

`[1, 8, 32]`, the same as the input.
</details>

<details>
<summary><strong>Q3. If hidden states after every layer are exactly identical, what might be wrong?</strong></summary>

The blocks were probably not applied (the loop doesn't reassign `x = layer(x, ...)`, or the stack returns its input), or the sub-layers output zeros (zero-initialized output projections, or dead ReLUs everywhere) so that only the residual passes through. Check the forward loop, that `hidden_states` appends the new `x`, and the sub-layer outputs' magnitudes.
</details>

<details>
<summary><strong>Q4. Why are intermediate hidden states useful for analysis?</strong></summary>

They show what each layer contributes (how much it changes the representation, where information emerges via probing), help debug dead or exploding layers, and can be used as features: BERT's feature-based NER results come from concatenating the last four layers' hidden states.
</details>

### Attention across layers

<details>
<summary><strong>Q1. Why can different layers have different attention patterns?</strong></summary>

Each layer has separate trainable parameters, and each receives a different input (the previous layer's output). Therefore each layer can learn a different way of mixing token information; in the trained model, layer 1 is sharp and layers 2 and 3 are diffuse.
</details>

<details>
<summary><strong>Q2. If attention has shape [batch_size, num_heads, seq_len, seq_len], what does the last dimension represent?</strong></summary>

The key positions being attended to by a query token. Each row (fixed batch, head and query) is a probability distribution over the key positions and sums to 1.
</details>

<details>
<summary><strong>Q3. If attention is strongly focused on PAD tokens, what part of the model should we inspect?</strong></summary>

The attention mask: it may be missing, inverted (0 and 1 swapped), not broadcast to `[B, 1, 1, T]`, or not passed to every block. Padded keys should receive zero weight; in the trained model, the maximum weight on any pad key is 0.
</details>

<details>
<summary><strong>Q4. Are attention maps always reliable explanations of model decisions?</strong></summary>

No. Attention maps are useful diagnostic tools, but they show where information was mixed, not why the output was produced; values, FFNs, residual paths and other heads all contribute. They are not complete explanations of model reasoning.
</details>
