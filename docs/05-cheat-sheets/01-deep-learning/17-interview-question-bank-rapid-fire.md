---
title: "Cheat Sheet: Module III Question Bank, Rapid Fire"
sidebar_position: 17
---

:::tip[What this sheet covers]
One line per question for all 51 questions of the Module III revision question bank, for a final pass before the interview. Full answers, follow-up nuances and verified code: [Deep Learning Interview Question Bank](../../03-ai-ml/03-deep-learning/09-interview-question-bank.md).
:::

## 1. NLP Warm-Up (Q1–Q8)

| # | Question | One-line answer |
|---|---|---|
| 1 | Why preprocess text? | Raw text is noisy and models need numbers; but it's task-dependent ("not good" → "good" if *not* is dropped) |
| 2 | Tokenisation vs vectorisation | Text → tokens vs tokens → numbers |
| 3 | Stemming vs lemmatisation | Rule chopping (*studi*) vs dictionary form (*study*); speed vs accuracy |
| 4 | Stopword removal risk | Negations are stopwords; removing them flips sentiment |
| 5 | BoW vs TF-IDF | Counts vs counts × rarity; both ignore order ("dog bites man" = "man bites dog") |
| 6 | Embeddings vs one-hot | Dense, small, similar words close vs 50,000-long, orthogonal (king · queen = 0) |
| 7 | Sentiment pipeline | Collect → preprocess → vectorise → split → train → evaluate → same pipeline at inference |
| 8 | Why sequence models | Order carries meaning; BoW can't tell "dog chased cat" from "cat chased dog" |

## 2. PyTorch (Q9–Q20)

| # | Question | One-line answer |
|---|---|---|
| 9 | Tensor | N-d array with shape, dtype, device, GPU support and autograd |
| 10 | `requires_grad=True` | Record operations for gradients; $x^2$ at 2 → `tensor(4.)` |
| 11 | Autograd | Dynamic graph built in the forward pass; `backward()` applies the chain rule |
| 12 | `loss.backward()` | Computes gradients only; `optimizer.step()` updates |
| 13 | `zero_grad()` | Gradients accumulate (9, 18, 27…); clear each step unless accumulating on purpose |
| 14 | Step order | zero_grad → forward → loss → backward → step |
| 15 | `nn.Module` | Registers parameters and sub-modules; `__init__` = layers, `forward` = flow |
| 16 | `.item()`, `.detach()`, `.cpu()`, `.numpy()` | Scalar / cut graph / to CPU / to NumPy; safe chain `.detach().cpu().numpy()` |
| 17 | Broadcasting | Align shapes from the right; (3, 4) + (4,) → (3, 4); (N,) + (N, 1) → (N, N) silently |
| 18 | Batch dimension | Samples per forward pass; (32, 20, 100) = 32 sentences × 20 tokens × 100 dims |
| 19 | Embedding layer | Lookup table: (B, T) ids → (B, T, d); `Embedding(10000, 128)` = 1,280,000 params |
| 20 | `eval()` vs `no_grad()` | Layer behaviour (dropout, batch norm) vs no graph; use both for testing |

## 3. RNN, LSTM, Generation (Q21–Q30)

| # | Question | One-line answer |
|---|---|---|
| 21 | Hidden state | Running summary $\pred{h_t} = \tanh(\param{W_x}\data{x_t} + \param{W_h}\pred{h_{t-1}} + \param{b})$; lossy |
| 22 | Vanishing gradients | Repeated factors below 1 shrink gradients over time; long-range learning fails |
| 23 | Why LSTM | Gated, additively updated cell state keeps memory and gradients over long spans |
| 24 | Cell vs hidden state | Cell = internal long-term memory; hidden = exposed output |
| 25 | Forget gate | $\sigma$ dial between 0 (erase) and 1 (keep) per memory component |
| 26 | Input gate + candidate | Candidate drafts new content (tanh); input gate decides how much is written |
| 27 | Cell update | $\pred{C_t} = \pred{f_t} \odot \pred{C_{t-1}} + \pred{i_t} \odot \pred{\tilde C_t}$; additive path helps gradients |
| 28 | Output gate | $\pred{h_t} = \pred{o_t} \odot \tanh(\pred{C_t})$: expose selected memory |
| 29 | Text generation | Predict distribution → pick token → append → repeat (autoregressive) |
| 30 | Temperature | $\operatorname{softmax}(\pred{z}/\hyper{T})$: low = sharp, repetitive; high = flat, random |

## 4. Attention (Q31–Q40)

| # | Question | One-line answer |
|---|---|---|
| 31 | Why attention | Fixed-size encoder vector is a bottleneck; attend to all encoder states per output step |
| 32 | Q, K, V | What I look for / what I offer / what I pass on |
| 33 | Scaled dot product | $\operatorname{softmax}(\pred{Q}\pred{K}^\top/\sqrt{\hyper{d_k}})\pred{V}$; scaling stops softmax saturating |
| 34 | Score shape | (batch, seq_len, seq_len), e.g. (4, 6, 6) |
| 35 | Softmax axis | Over keys (`dim=-1`); each query's weights sum to 1 |
| 36 | Attention heatmap | Rows = queries, columns = keys; a diagnostic, not an explanation |
| 37 | Cross vs self-attention | Cross: Q from decoder, K/V from encoder; self: all from one sequence |
| 38 | Positional information | Attention is order-blind (permute in → permute out); add positional encodings |
| 39 | Parallelism trade-off | All positions at once vs $O(n^2)$ memory: 64 MiB per head at 4,096 tokens |
| 40 | Self-attention in plain words | Each word compares itself with all others, weights them, blends their information |

## 5. Vision and Generative (Q41–Q47)

| # | Question | One-line answer |
|---|---|---|
| 41 | Classification / detection / segmentation | Image label / class + box per object / class per pixel |
| 42 | Bounding box | Axis-aligned rectangle: xyxy or cxcywh; where (regression) + what (classification) |
| 43 | Semantic vs instance | All dogs = "dog" vs dog 1, dog 2, dog 3 |
| 44 | Autoencoder | Encoder → latent code → decoder; bottleneck forces compact features |
| 45 | Reconstruction loss | $\lVert \data{x} - \pred{\hat x} \rVert^2$; high error flags anomalies |
| 46 | GAN | Generator fakes, discriminator judges; trained in competition |
| 47 | Mode collapse | Limited variety (only some digits); realistic samples hide it |

## 6. Pressure Round (Q48–Q51)

| # | Question | One-line answer |
|---|---|---|
| 48 | "I know the attention formula" | Also shapes, projections, softmax axis, masks, heads, positions, FFN, residuals, LayerNorm, objectives |
| 49 | Shape reasoning | Most bugs are shape bugs; some are silent (broadcasting, batch_first) |
| 50 | Learning vs interview-ready | Explain, follow up, shapes, code, debug, compare, apply |
| 51 | What next | Practise explaining, coding, debugging and connecting ideas; print shapes everywhere |
