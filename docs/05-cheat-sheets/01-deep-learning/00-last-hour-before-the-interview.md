---
title: "Deep Learning: The Last Hour Before the Interview"
sidebar_position: 0
---

:::tip[How to use this sheet]
Read it top to bottom in the hour before an interview. **Answer each question out loud first**, then open it to check yourself. Everything here is a summary of the full notes (linked in each section), with the same numbers and notation. The topic sheets that follow go deeper: [PyTorch and neural networks](./01-pytorch-and-neural-networks.md), [CNNs and transfer learning](./02-cnns-and-transfer-learning.md), [NLP foundations](./03-nlp-foundations.md), [RNNs](./04-rnns.md), [LSTM and GRU](./05-lstm-and-gru.md).
:::

## 🎤 The 20 Questions You Are Most Likely to Get

<details>
<summary><strong>1. Walk me through one PyTorch training step. Why <code>zero_grad()</code>?</strong></summary>

`optimizer.zero_grad()` → `preds = model(X)` (forward, records the graph) → `loss = criterion(preds, y)` (a scalar) → `loss.backward()` (chain rule fills every `.grad`, **no weight changes**) → `optimizer.step()` ($\theta \leftarrow \theta - \eta\nabla L$, the **only** place weights change). `backward()` **adds** into `.grad`, so without `zero_grad()` every step uses the sum of all past gradients and the loss oscillates or diverges. Mnemonic: **zero → forward → loss → backward → step**.
</details>

<details>
<summary><strong>2. Explain backpropagation.</strong></summary>

The forward pass computes the loss as a chain of simple operations and records them. Each operation knows its **local derivative** (add → 1, multiply → the other input, ReLU → 1 or 0, sigmoid → $\sigma(1-\sigma)$). The backward pass starts from $\partial L/\partial L = 1$ and, at each node, multiplies the **upstream gradient by the local derivative** (chain rule); where a value feeds several operations, the gradients are **summed**. One reverse pass gives $\partial L/\partial w$ for every weight. Example: $f = (a+b)ab + ab$ at $a = 2, b = 3$ gives $\partial f/\partial a = 24$, $\partial f/\partial b = 18$.
</details>

<details>
<summary><strong>3. Why do neural networks need activation functions?</strong></summary>

Without them, stacked linear layers collapse into one: $W_2(W_1x + b_1) + b_2 = W'x + b'$, which can never fit a curve like $x^2$. The activation also **gates the gradient**: sigmoid's slope is at most 0.25 (vanishing gradients in deep stacks), tanh's at most 1, ReLU's exactly 1 for $z > 0$ and 0 otherwise (dead ReLUs). Default: ReLU in hidden layers, no activation on a regression output.
</details>

<details>
<summary><strong>4. Which loss for which problem, and why not softmax before <code>CrossEntropyLoss</code>?</strong></summary>

Regression → identity output + **MSE**. Binary → one logit + **`BCEWithLogitsLoss`** (sigmoid inside, stable). Multi-class → $C$ logits + **`CrossEntropyLoss`** (= log-softmax + NLL) with **integer** labels. Both take **raw logits**: applying softmax/sigmoid first applies it twice, which gives a wrong loss and weak gradients. The gradient of softmax + cross-entropy w.r.t. the logits is simply $p - y$. A sane starting loss is about $\ln C$ (2.30 for 10 classes).
</details>

<details>
<summary><strong>5. What is the learning rate, and what happens if it is too high or too low?</strong></summary>

The step size of $W \leftarrow W - \eta\,\partial L/\partial W$. On $L(w) = (w-3)^2$ each step multiplies the distance to the minimum by $(1 - 2\eta)$: $\eta = 0.05$ → 0.9 (slow), $0.4$ → 0.2 (fast), $1.05$ → $-1.1$ (diverges). Too low: slow, nearly linear loss decline. Too high: oscillation, loss → `inf`/`nan`, and with ReLU, dead neurons. Adam's sensible default is 1e-3.
</details>

<details>
<summary><strong>6. Why CNNs instead of fully connected layers for images?</strong></summary>

An FC layer from a 32×32×3 image to 100 neurons needs **307,300** parameters, grows with $H \cdot W$, and flattening destroys pixel adjacency. A conv layer has **local connectivity** and **weight sharing**: $C_{out}(C_{in}F^2 + 1)$ parameters (`Conv2d(3, 5, 3)` = 140), independent of image size, and the same detector works at every position (translation equivariance). Stacked Conv → ReLU → Pool blocks learn edges → textures → parts → objects.
</details>

<details>
<summary><strong>7. Output size and parameter count of a conv layer?</strong></summary>

Output $= \lfloor (W - F + 2P)/S \rfloor + 1$ per spatial dimension (always floor). Parameters $= C_{out}(C_{in}F^2 + 1)$; stride, padding and image size don't appear. $F = 3, P = 1, S = 1$ keeps the size; stride 2 or 2×2 max-pooling halves it. Pooling never changes the channel count and has zero parameters.
</details>

<details>
<summary><strong>8. Where are the parameters of a CNN classifier, and how do you regularize it?</strong></summary>

In the **first fully connected layer**: in the notes' SmallCNN, `fc1` holds 131,136 of 136,874 parameters (96%); in SimpleCNN 97.8%. Regularize with **dropout** (≈ 0.5) around the FC layers, **data augmentation** on the training set only, **weight decay** (AdamW), **early stopping**, or replace flatten + big FC with **global average pooling**. Remember `model.train()` / `model.eval()`.
</details>

<details>
<summary><strong>9. You have 5,000 labelled images. How do you build a classifier?</strong></summary>

Start **pretrained** (e.g. ResNet18 with ImageNet weights), replace the head with `Linear(512, num_classes)`, match the preprocessing (RGB, 224×224, ImageNet mean/std). **Feature extraction** first: freeze the backbone and train only the head (1,026 trainable parameters for 2 classes). Then **fine-tune** the last block (`layer4`) with a ~10× smaller learning rate (1e-4), AdamW, a scheduler and early stopping. In the notes this took cats vs dogs from ~81% (scratch) to ~98% (feature extraction) and 98.7% (fine-tuning).
</details>

<details>
<summary><strong>10. What is a residual connection and why does it help?</strong></summary>

$y = F(x) + x$: the block learns only a correction to its input. If it has nothing to add it can learn $F \approx 0$, so extra layers can't make the network worse, and on the backward pass $\partial y/\partial x = \partial F/\partial x + I$, so the gradient always keeps an identity path and doesn't vanish through deep stacks. That let ResNets train 18–152 layers.
</details>

<details>
<summary><strong>11. How do you turn text into something a network can use?</strong></summary>

Preprocess (tokenize, lowercase, optionally remove stop words **but keep negations**, lemmatize; do less for Transformers) → vocabulary with `<PAD>` and `<UNK>` → **not** raw indices (fake order) and not one-hot (huge, every pair has cosine 0) → **embeddings**: an $|V| \times d$ table of dense learned vectors (`nn.Embedding`), trained or initialized from Word2Vec/GloVe. They work because of the **distributional hypothesis**: words in similar contexts get similar vectors.
</details>

<details>
<summary><strong>12. Skip-gram vs CBOW?</strong></summary>

Both are Word2Vec: generate (centre, context) pairs with a window (4 words, window 1 → 6 pairs). **Skip-gram** predicts the context from the centre word: one example per pair, better for **small data and rare words**, slower. **CBOW** averages the context to predict the centre word: faster, better for **frequent words**. Score with $P(o \mid c) = \text{softmax}(u_o \cdot v_c)$; in practice negative sampling. Needs a large corpus.
</details>

<details>
<summary><strong>13. Build a sentiment classifier for reviews.</strong></summary>

Stratified split first → one shared preprocessing function (keep negations) → **TF-IDF** (with bigrams for "not good") **fitted on training data only**, inside a `Pipeline` → Naive Bayes and a linear SVM as baselines → cross-validation for model selection, test set once, per-class precision/recall/F1 → then pretrained embeddings or a fine-tuned Transformer, compared against the baseline.
</details>

<details>
<summary><strong>14. Why do we need RNNs, and how do they work?</strong></summary>

Sequences are ordered, variable-length and need memory of the past; ANNs/CNNs assume fixed-size independent inputs. An RNN reads one element per step and keeps a hidden state $h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$, $h_0 = 0$, with the **same weights at every step** (any length, fixed parameter count). Patterns: many-to-one (sentiment), one-to-many (captioning), many-to-many aligned (tagging) or encoder-decoder (translation).
</details>

<details>
<summary><strong>15. How is an RNN trained, and why do its gradients vanish or explode?</strong></summary>

**BPTT**: unroll the sequence, backprop through the unrolled graph; a shared weight's gradient is the **sum over steps**. Each step back multiplies the gradient by $\text{diag}(1 - h_t^2)\,W_h$, so over $k$ steps it behaves like a power: $0.5^{15} = 0.000031$ (vanishes), $1.5^{15} = 437.89$ (explodes). Exploding → **gradient clipping**; vanishing → architecture: **LSTM/GRU**, attention.
</details>

<details>
<summary><strong>16. How does an LSTM solve the vanishing gradient problem?</strong></summary>

It adds a **cell state** updated **additively**: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t$. Along that path $\partial c_t/\partial c_{t-1} \approx \text{diag}(f_t)$: no weight matrix, no tanh slope, so over $k$ steps the gradient is scaled by the product of forget gates. Gates are **learned**: initialize the forget bias to 1+, because at default init $f \approx 0.5$ and the gradient vanishes like an RNN's (measured 49 steps back: RNN $3.2 \times 10^{-14}$, default LSTM $1.8 \times 10^{-11}$, forget bias +3 $5.3 \times 10^{-2}$). It **mitigates**, not eliminates; clipping is still used.
</details>

<details>
<summary><strong>17. Cell state vs hidden state; what does each gate do?</strong></summary>

$c_t$ = internal long-term memory (additive, unbounded); $h_t = o_t \odot \tanh(c_t)$ = the exposed, bounded view used for outputs and fed to all gates next step. **Forget** $f_t$: how much old memory to keep. **Input** $i_t$: how much new content to write. **Candidate** $\tilde c_t$ (tanh): what to write. **Output** $o_t$: how much memory to reveal. All gates are sigmoid vectors, one soft decision per dimension. "$c$ stores, $h$ reveals."
</details>

<details>
<summary><strong>18. LSTM vs GRU?</strong></summary>

GRU has no separate cell and 2 gates: update $z_t$ interpolates, $h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde h_t$ (notebook/PyTorch convention), and reset $r_t$ filters the past before the candidate. Parameters 1 : 3 : 4 (RNN : GRU : LSTM). They perform similarly on most tasks; GRU is lighter and faster, LSTM gives finer memory control. Try both if it matters.
</details>

<details>
<summary><strong>19. How does an LSTM generate text, and what does temperature do?</strong></summary>

Train next-token prediction (targets = inputs shifted by one, cross-entropy at every position). Generate autoregressively: warm up on a prompt, then repeatedly take the **last** position's logits, pick a token, append it and feed **only that token** back with the carried $(h, c)$. Greedy is deterministic and repetitive. Temperature: $p_i \propto e^{z_i/T}$; $T < 1$ sharper (→ greedy), $T > 1$ flatter (→ uniform), ranking unchanged, **inference only**.
</details>

<details>
<summary><strong>20. Why did Transformers replace RNNs/LSTMs for most NLP?</strong></summary>

Recurrence is **sequential** (no parallelism over time), compresses the past into a **fixed-size state**, and reaches distant tokens only through a **long chain** of updates. Self-attention connects every position to every other directly in one layer and trains in parallel. Costs: memory quadratic in context length and no built-in order (positional encodings). RNNs remain useful for streaming/on-device use (constant memory per step).
</details>

## 🔭 Attention and Transformers: Extra Questions

The full sheet is [Attention and Why Transformers](./06-attention-and-transformers.md).

<details>
<summary><strong>21. What is attention, in one breath?</strong></summary>

Score every token for the current need, softmax the scores into weights ($\alpha_i \ge 0$, $\sum \alpha_i = 1$), output the weighted sum of value vectors $c = \sum_i \alpha_i v_i$. "The movie was not good": weights [0.071, 0.086, 0.064, 0.428, 0.351], context [0.381, 0.472] vs the plain average [0.28, 0.24]. It fixed seq2seq's single fixed context $c = h_T$ by giving every output step its own $c_t$ (Bahdanau, 2014).
</details>

<details>
<summary><strong>22. Query, Key, Value, and how to read an attention matrix?</strong></summary>

$q = xW_Q$ (what I look for), $k = xW_K$ (what I offer for matching), $v = xW_V$ (what I pass on); $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$. In the $n \times n$ matrix each **row** is one token's distribution over all tokens (sums to 1), each **column** is how much a token is attended to. New vectors $= AV$. Positional information is needed because self-attention is order-blind.
</details>

## 🧮 Self-Attention Mechanics: Extra Questions

The full sheet is [Scaled Dot-Product Self-Attention](./07-full-self-attention.md).

<details>
<summary><strong>23. Write scaled dot-product attention with shapes.</strong></summary>

$\operatorname{softmax}(QK^\top/\sqrt{d_k})V$ with $Q = XW_Q$, $K = XW_K$, $V = XW_V$. $X$: $n \times d_{model}$ → $Q, K$: $n \times d_k$ → $QK^\top$: $n \times n$ → row-wise softmax $A$ (rows sum to 1) → $Z = AV$: $n \times d_v$. Softmax over keys (`dim=-1`); masks as $-\infty$ before softmax.
</details>

<details>
<summary><strong>24. Why √d_k, and not d_k?</strong></summary>

$\operatorname{Var}(q \cdot k) \approx d_k$, so scores grow like $\sqrt{d_k}$ (std 1.38 at $d_k = 2$, 11.02 at 128); large scores saturate softmax (near one-hot, vanishing gradients). Dividing by $\sqrt{d_k}$ restores variance 1; dividing by $d_k$ leaves $1/d_k$ and near-uniform attention.
</details>

## 🏗️ Multi-Head Attention and the Transformer: Extra Questions

Full sheets: [Multi-Head Attention](./08-multi-head-attention.md) · [The Transformer (Figure 1)](./09-transformer-architecture.md).

<details>
<summary><strong>25. Why multi-head attention, and does it add parameters?</strong></summary>

One head gives one softmax distribution per token and must average several relations; $h$ heads with their own projections into $d_k = d_{model}/h$ attend to different relations, then Concat + $W^O$ mix them. Same $4d_{model}^2$ weights as one full-width head. Paper: $h = 1$ 24.9 vs $h = 8$ 25.8 BLEU.
</details>

<details>
<summary><strong>26. Walk through Figure 1 of "Attention Is All You Need".</strong></summary>

Encoder: Input Embedding (×√d_model) ⊕ Positional Encoding → 6 × [Multi-Head Attention → Add & Norm → Feed Forward → Add & Norm]. Decoder: Outputs (shifted right) → Output Embedding ⊕ PE → 6 × [Masked Multi-Head Attention → Add & Norm → Multi-Head Attention (Q = decoder, K = V = encoder) → Add & Norm → Feed Forward → Add & Norm] → Linear → Softmax → Output Probabilities. $d_{model}$ 512, $h$ 8, $d_{ff}$ 2048; 28.4 BLEU En–De.
</details>

## 🧠 BERT: Extra Questions

Full sheet: [BERT](./10-bert.md).

<details>
<summary><strong>27. Explain BERT's pre-training: Masked LM and NSP.</strong></summary>

Encoder-only Transformer (base: L 12, H 768, A 12, 110M) with bidirectional attention. MLM: choose 15% of WordPiece tokens; 80% → [MASK], 10% → random token, 10% unchanged; predict the originals (loss only there). [MASK] never appears in fine-tuning, hence the mix. NSP: B is the real next span 50% of the time, classified from the [CLS] vector $C$ (97–98% accuracy; later dropped by RoBERTa). Input = token + segment + learned position embeddings.
</details>

<details>
<summary><strong>28. How do you fine-tune BERT for classification and for SQuAD?</strong></summary>

Classification: $\operatorname{softmax}(CW^\top)$ with $W \in \mathbb{R}^{K \times H}$ on the [CLS] vector. SQuAD: start/end vectors $S$, $E$; $P^{start}_i = \operatorname{softmax}(S \cdot T_i)$; answer = best $S \cdot T_i + E \cdot T_j$ with $j \ge i$; SQuAD 2.0 answers only if that beats $S \cdot C + E \cdot C + \tau$. All weights are updated: lr 2e-5 to 5e-5, batch 16/32, 2–4 epochs.
</details>

## 🧱 Stacked Encoder Blocks: Extra Questions

Full sheet: [Stacked Encoder Blocks](./11-stacked-encoder-blocks.md).

<details>
<summary><strong>29. Why stack encoder blocks, and what happens to the shapes?</strong></summary>

Every block maps `[B, T, D]` to `[B, T, D]`, so blocks stack in an `nn.ModuleList` loop (a plain list wouldn't register parameters; `nn.Sequential` can't pass the mask). Each block is another round of attention + FFN on already-contextualized vectors, with its own weights. Depth needs data: in the notebook's 18-sentence project, 1 vs 3 blocks scored 84.1% vs 86.3% held-out over 20 seeds (within noise), both 100% on training data.
</details>

## 🤗 Hugging Face Pipelines: Extra Questions

Full sheet: [Hugging Face Pipelines](./12-hugging-face-pipelines.md).

<details>
<summary><strong>30. What does a Hugging Face pipeline hide, and why open it?</strong></summary>

Preprocess (the checkpoint's tokenizer: WordPiece ids with [CLS]/[SEP], attention mask, padding, truncation) → forward (`AutoModelFor…` in eval mode under `no_grad`; logits `[B, num_labels]`) → postprocess (softmax, argmax, `id2label`). The sentiment default is DistilBERT fine-tuned on SST-2 (66,955,010 parameters): pin model and revision. Open it for batching, devices, logits/hidden states, custom thresholds and fine-tuning. The score is a softmax probability (two classes: sigmoid of the margin), not a guarantee: "It was fine, I guess." → POSITIVE 0.99976.
</details>

## 📄 Extractive Document QnA: Extra Questions

Full sheet: [Extractive Document QnA](./13-extractive-document-qna.md).

<details>
<summary><strong>31. How do you answer questions over a long PDF with an extractive QnA model, and where does it break?</strong></summary>

Extract text page by page (keep page numbers), cut overlapping chunks that fit the model (180 words, 40 overlap → 78 chunks for the BERT paper), run `AutoModelForQuestionAnswering` on each: per-token start/end logits from a `Linear(768, 2)`, best context-only span by $s_i + e_j$, sliced with `offset_mapping`. Rank, abstain below a threshold, return page + chunk + preview. It breaks because logits aren't comparable across chunks (right 20.123 vs wrong 20.049) and a SQuAD v1.1 reader never abstains ("Which city was BERT invented in?" → "Transformers", 0.9964). Fix: retrieve relevant chunks first (RAG), SQuAD 2.0 reader, calibrated thresholds.
</details>

## 🔎 Retriever + Reader with FAISS: Extra Questions

Full sheet: [Retriever + Reader with FAISS](./14-retriever-reader-faiss.md).

<details>
<summary><strong>32. How does semantic retrieval with embeddings and FAISS work, and how do you know it works?</strong></summary>

Embed every chunk once with a bi-encoder (`all-MiniLM-L6-v2`: mean pooling + L2 normalize → 384-d unit vectors), store them in a FAISS index, and per question embed it, take the top-k by inner product (= cosine for unit vectors; squared L2 = 2 − 2cos gives the same order), map rows back to chunk metadata and read only those k chunks (1.16 s vs 18.7 s per question on CPU for 5 of 78 chunks). Flat indexes are exact scans, no faster than NumPy; HNSW/IVF are the speed-up. Measure recall@k on labelled questions first: on the BERT paper it was 0.67 at k = 5, and every answer the pipeline lost to brute force (6 vs 8 of 12) was a retrieval miss.
</details>

## 🧾 Generative RAG End to End: Extra Questions

Full sheet: [Generative RAG End to End](./15-generative-rag.md).

<details>
<summary><strong>33. Walk me through a RAG system over a PDF, and how you make its answers trustworthy.</strong></summary>

Offline: read the PDF page by page (PyMuPDF), clean with targeted rules (don't delete document words such as `[CLS]` in an NLP paper), chunk by whole sentences with page metadata, embed with a sentence-transformer into a FAISS inner-product index, and build BM25 statistics. Per question: refuse if the best cosine is below a threshold (FIFA question 0.1408), retrieve with hybrid search fused by RRF (recall@1 0.75 vs 0.50 for either alone on the BERT paper), route exact facts to an extractive reader and explanations to FLAN-T5 with a token-budgeted prompt of the most relevant sentences (no metadata inside; 1 → 5 of 12 facts right), decode with beam search, and refuse if the answer's words aren't in the context ("22,500M" support 0.0). Evaluate by stage: 2 of 12 failures were retrieval misses, and generator and reader each got 5 of 12 but different ones (7 by at least one).
</details>

## 🖼️ Detection, Segmentation, Autoencoders and GANs: Extra Questions

Full sheet: [Detection, Segmentation, Autoencoders and GANs](./16-detection-segmentation-autoencoders-gans.md).

<details>
<summary><strong>34. Classification vs detection vs segmentation, and how is detection evaluated?</strong></summary>

Classification gives one label per image; detection gives class + score + box per object; segmentation gives a class per pixel (semantic), plus a separate mask per object (instance). Boxes are compared with IoU (overlap ÷ union, a match at ≥ 0.5); NMS keeps the best box and deletes overlapping duplicates; mAP is the area under the precision-recall curve averaged over classes (COCO: also over IoU 0.5–0.95). A real DETR run found the coffee cup at 0.9975 and a false "carrot" at 0.8612 on a cat's nose, so thresholds are chosen on validation data.
</details>

<details>
<summary><strong>35. Autoencoder vs GAN, and what is mode collapse?</strong></summary>

An autoencoder compresses its input into a small latent code and rebuilds it, trained only on reconstruction loss; it is used for compression, denoising and anomaly detection (on digits: test MSE 0.0145 with 8 numbers; unseen 9s flagged with AUC 0.8836). A GAN trains a generator (noise → fake) against a discriminator (real or fake); balance is D ≈ 0.5 and D loss ≈ 2 ln 2. Mode collapse is a generator producing only a few kinds of output: with a 100× faster discriminator it covered 2 of 8 modes while 93.4% of samples looked real.
</details>

## 🔢 Numbers Worth Quoting

| Fact | Number | From |
|---|---|---|
| Sigmoid's maximum slope | 0.25 (at $z = 0$); 5 layers → ≤ 0.001 | [Calculus](../../03-ai-ml/03-deep-learning/01-pytorch-and-neural-networks/02-calculus-behind-pytorch.md) |
| Chain-rule example $f = (a+b)ab + ab$, $a=2, b=3$ | $f = 36$, $\partial f/\partial a = 24$, $\partial f/\partial b = 18$ | Calculus |
| GD on $(w-3)^2$: factor $1 - 2\eta$ | 0.05 → 0.9, 0.4 → 0.2, 1.05 → −1.1 (diverges) | [ANN](../../03-ai-ml/03-deep-learning/01-pytorch-and-neural-networks/03-artificial-neural-networks.md) |
| FC layer on 32×32×3 → 100 neurons | 307,300 params | [CNN fundamentals](../../03-ai-ml/03-deep-learning/02-cnns/01-cnn-fundamentals.md) |
| `Conv2d(3, 5, 3)` | 140 params | CNN fundamentals |
| SmallCNN | 136,874 params, `fc1` 96% | [CNN architecture](../../03-ai-ml/03-deep-learning/02-cnns/02-cnn-architecture-training-alexnet.md) |
| Starting cross-entropy, 10 classes | $\ln 10 = 2.30$ | CNN architecture |
| AlexNet (2012) | top-5 error ~26% → ~15%, ~60 M params, ~94% in FC | CNN architecture |
| Cats vs dogs | scratch ~81% → feature extraction 98.19% → fine-tune 98.74% | [Transfer learning](../../03-ai-ml/03-deep-learning/02-cnns/04-transfer-learning-resnet18.md) |
| ResNet18 | 11,689,512 params; new head trains 1,026; + `layer4` 8,394,754 | Transfer learning |
| Word2Vec pairs, "i love deep learning" | window 1 → 6, window 2 → 10 | [NLP foundations](../../03-ai-ml/03-deep-learning/03-nlp-foundations/01-nlp-foundations-words-to-vectors.md) |
| TF-IDF cosine | "good movie" vs "not good movie" 0.744; "man bites dog" vs "dog bites man" 1.0 | [LSTM and GRU](../../03-ai-ml/03-deep-learning/05-lstm-and-gru/01-lstm-and-gru-gated-memory.md) |
| Repeated multiplication | $0.5^{15} = 0.000031$, $1.5^{15} = 437.89$, $0.95^{20} = 0.358$ | [RNNs in practice](../../03-ai-ml/03-deep-learning/04-rnns/03-rnns-in-practice-training-and-limits.md) |
| Gradient 49 steps back (untrained) | RNN $3.2 \times 10^{-14}$, LSTM $1.8 \times 10^{-11}$, LSTM forget bias +3 $5.3 \times 10^{-2}$ | LSTM and GRU |
| Params, $D = 300$, $H = 512$ | RNN 416,768 · GRU 1,250,304 · LSTM 1,667,072 | LSTM and GRU |
| Temperature on logits [2.5, 1.2, 0.5, −0.3] | top $p$: 0.912 (T 0.5), 0.681 (T 1), 0.468 (T 2) | [LSTM text generation](../../03-ai-ml/03-deep-learning/05-lstm-and-gru/02-lstm-text-generation-temperature-sampling.md) |
| Attention, "The movie was not good" | weights 0.428 (not), 0.351 (good); context [0.381, 0.472] vs average [0.28, 0.24] | [From LSTM to Attention](../../03-ai-ml/03-deep-learning/06-attention-and-transformers/01-from-lstm-to-attention-why-transformers.md) |
| Per-layer cost | self-attention $O(n^2 d)$, path $O(1)$; recurrent $O(n d^2)$, path $O(n)$ | From LSTM to Attention |
| √d_k scaling | std of q·k: 1.38 (d_k 2) → 11.02 (d_k 128) raw, about 1 scaled; scores [10, 20, 5, 15, 2] → max weight 0.993 | [Full Self-Attention](../../03-ai-ml/03-deep-learning/06-attention-and-transformers/02-full-self-attention-scaled-dot-product.md) |
| Multi-head | 4·d_model² weights for any h; Table 3: h = 1 24.9, h = 8 25.8, h = 32 25.4 BLEU | [Multi-Head Attention](../../03-ai-ml/03-deep-learning/06-attention-and-transformers/03-multi-head-attention.md) |
| Transformer base | N 6, d_model 512, d_ff 2048, h 8, 65M params; 28.4 BLEU En–De (big); lr peak 0.000699 at step 4,000 | [Transformer: Figure 1](../../03-ai-ml/03-deep-learning/06-attention-and-transformers/04-transformer-architecture-figure-1.md) |
| BERT | base L 12 / H 768 / A 12 / 110M, large 24 / 1024 / 16 / 340M; 15% chosen → 12% [MASK] / 1.5% random / 1.5% same; SQuAD v1.1 F1 93.2, GLUE 80.5 | [BERT](../../03-ai-ml/03-deep-learning/06-attention-and-transformers/05-bert-pretraining-and-fine-tuning.md) |
| Stacked encoder mini project | 8,544 params per block (D 32, d_ff 64): 9,922 vs 27,010; norms ≈ √D after each LayerNorm; masked mean pooling | [Stacked encoder blocks](../../03-ai-ml/03-deep-learning/06-attention-and-transformers/06-stacked-encoder-blocks-mini-project.md) |
| Hugging Face sentiment pipeline | DistilBERT SST-2: 66,955,010 params, logits [−4.2341, 4.6189] → 0.99986; [CLS] 101, [SEP] 102; 512-token limit | [Hugging Face pipelines](../../03-ai-ml/03-deep-learning/07-hugging-face-qna-and-rag/01-hugging-face-pipelines.md) |
| Extractive document QnA | DistilBERT-SQuAD 65,192,450 params, head 1,538; BERT paper → 78 chunks (180/40 words); right 20.123 vs wrong 20.049; pages 1–4 = 4,556 tokens, 8% seen at 384 | [Extractive document QnA](../../03-ai-ml/03-deep-learning/07-hugging-face-qna-and-rag/02-extractive-document-qna.md) |
| Retriever + reader (FAISS) | MiniLM 384-d unit vectors, 256-token limit (48 of 78 chunks truncated); 1.16 s vs 18.7 s per question (k = 5, CPU); recall@5 0.67; brute force 8 vs top-5 6 of 12; flat FAISS ≈ NumPy, HNSW 0.17 ms at recall 0.96 | [Retriever + reader](../../03-ai-ml/03-deep-learning/07-hugging-face-qna-and-rag/03-retriever-reader-with-faiss.md) |
| Generative RAG (FLAN-T5) | 54–55 sentence chunks, ~1.7 tokens/word; hybrid recall@1 0.75 vs 0.50; notebook prompt 1 of 12 facts, sentence-selected 5, extractive 5, either 7; refuse below cosine 0.2 or support 0.5 | [Generative RAG](../../03-ai-ml/03-deep-learning/07-hugging-face-qna-and-rag/04-generative-rag-end-to-end.md) |
| Detection and segmentation | DETR cup 0.9975, false carrot 0.8612; IoU (0,0,4,4) vs (2,2,6,6) = 0.143; NMS keeps 3 of 8 toy boxes; two cats: semantic 78.3% "cat" vs 37.7% + 37.9% | [Detection and segmentation](../../03-ai-ml/03-deep-learning/08-vision-tasks-and-generative-models/01-object-detection-and-segmentation.md) |
| Autoencoders and GANs | 8-number code MSE 0.0145 (PCA 0.0246, mean 0.0740); anomaly AUC 0.8836; balanced GAN D 0.511 / 0.457; collapsed 2 of 8 modes | [Autoencoders and GANs](../../03-ai-ml/03-deep-learning/08-vision-tasks-and-generative-models/02-autoencoders-and-gans.md) |

## 🎯 Pick the Output Layer and Loss

| Task | Output layer | Loss (PyTorch) | Targets |
|---|---|---|---|
| Regression | `Linear(h, 1)`, no activation | `MSELoss` (or `L1Loss`/Huber) | float, same shape as preds |
| Binary classification | `Linear(h, 1)` → raw logit | `BCEWithLogitsLoss` | float 0/1, shape `(N, 1)` |
| Multi-class | `Linear(h, C)` → raw logits | `CrossEntropyLoss` | int64 class indices, shape `(N,)` |
| Next-token (sequence) | `Linear(H, V)` at every step | `CrossEntropyLoss` on `(B·T, V)` | int64, `(B·T,)` |

## 📏 Shapes at a Glance

| Model | Flow |
|---|---|
| ANN (2 layers) | $(N, d) \to (N, h) \to (N, o)$; `nn.Linear(in, out).weight` is `(out, in)` |
| CNN | `(N, C, H, W)` (PyTorch is channels-first) → conv/pool → `(N, C', H', W')` → flatten `(N, C'·H'·W')` → `(N, classes)` |
| RNN / LSTM (`batch_first=True`) | ids `(B, T)` → embedding `(B, T, E)` → `out` `(B, T, H)`, `h_n` (and `c_n`) `(layers·dirs, B, H)` **always** |
| Language model loss | logits `(B, T, V)` → `(B·T, V)`, targets `(B·T,)` |

## 🩺 When Training Misbehaves: Check in This Order

| Symptom | Likely cause | First fix |
|---|---|---|
| Loss → `inf`/`nan`, or spikes | Learning rate too high, unscaled inputs, exploding gradients | Lower LR (Adam 1e-3), normalize inputs, clip gradients |
| Loss oscillates | Learning rate slightly too high, missing `zero_grad()` | Lower LR, check the loop |
| Loss stuck at $\ln C$ | Nothing learned: wrong labels, LR far off, dead network | Overfit one small batch first |
| Loss plateaus early, gradient norm ≈ 0 | Vanishing gradients, dead ReLUs, saturated sigmoids | ReLU, better init, residuals/LSTM, forget-gate bias |
| Train ↓ but validation ↑ | Overfitting | More data/augmentation, dropout, weight decay, early stopping, pretrained model |
| Wrong loss but no error | Shape broadcast `(N,1)` vs `(N,)`, softmax applied twice | Match shapes, pass raw logits |
| `Expected all tensors to be on the same device` | Model and batch on different devices | `model.to(device)` and every `x.to(device)` |

## 🚨 The Ten Traps Interviewers Love

- `backward()` **computes**, `step()` **updates**; gradients **accumulate** until `zero_grad()`.
- `model.eval()` ≠ `torch.no_grad()`: one changes dropout/BatchNorm behaviour, the other stops graph building. Inference uses **both**.
- Softmax before `CrossEntropyLoss`, or sigmoid before `BCEWithLogitsLoss`, applies it **twice**.
- `nn.Linear(d, h).weight` is `(h, d)`; conv weights are `(C_out, C_in, kH, kW)`.
- Pooling never changes channels and has **zero** parameters; stride and padding never change a conv's parameter count.
- Fit the TF-IDF vectorizer (or any preprocessing statistic) on **training data only**; split first.
- Standard stop-word lists delete **negations**: "good" and "not good" become identical.
- `h_n` ignores `batch_first`; for padded batches `out[:, -1]` is a padding position (pack, or index by length).
- Clipping fixes **exploding** gradients, not vanishing ones; clip **after** `backward()`, **before** `step()`.
- LSTMs **mitigate** vanishing gradients only when forget gates are near 1; initialize the forget bias to 1+.
