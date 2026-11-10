---
title: "Deep Learning Interview Question Bank (Module III Revision)"
sidebar_position: 10
---

:::tip[TL;DR (30-Second Interview Pitch)]
All **51 interview-style questions** from the Module III revision lecture (NLP basics, PyTorch, RNNs and LSTMs, attention, detection and segmentation, autoencoders and GANs, and a final pressure round), each with a model answer, the follow-up nuance an interviewer is waiting for, and links to the pages that teach it. Every number and every error message was checked by running the code (PyTorch 2.15). It ends with an 8-drill **Mock Interview Lab** of coding and debugging tasks, the hands-on follow-up the course itself recommends.
:::

<FormulaKey />

:::info[Where this page sits]
**Before:** the topic pages of this section, from [PyTorch Fundamentals](./01-pytorch-and-neural-networks/01-pytorch-fundamentals.md) to [Autoencoders and GANs](./08-vision-tasks-and-generative-models/02-autoencoders-and-gans.md).
**This page:** the course's *Module III Revision Lecture: Interview-Style Technical Question Bank*, answered in full and corrected where needed, with each answer pointing to the page that explains it.
**Next:** the one-screen revision sheets in the Cheat Sheets tab, including [the last hour before the interview](../../05-cheat-sheets/01-deep-learning/00-last-hour-before-the-interview.md).
:::

## 🧭 Start Here

This page simulates the interview itself rather than recapping theory. The revision lecture it comes from was designed to leave students with a balanced feeling: *I know many of these concepts, but I need more practice explaining, reasoning, debugging and connecting them under pressure.* So each question is presented as an interviewer would ask it, with its sub-prompt, and the answer is hidden until you click. Try answering out loud first, then compare.

The questions run in the lecture's six rounds, from comfortable to hard: a **warm-up** on NLP preprocessing and the sentiment pipeline; a **PyTorch reality check** on tensors, autograd and the training loop; **RNNs, LSTMs and text generation**; the deepest round, **attention and self-attention**; an awareness round on **detection, segmentation, autoencoders and GANs**; and a **final pressure round** about maturity and preparation. Each answer has three layers: the core answer you should be able to say in under a minute, **the nuance** (the follow-up the instructor notes flag, which is what separates a good answer from an average one), and **Read more** links to the full pages on this site.

Where the course's suggested answer was incomplete or imprecise, the answer here is corrected and the correction is checked by a run: for example, Bag of Words and TF-IDF give **identical** vectors for "dog bites man" and "man bites dog" (not merely similar ones), and `model.eval()` alone leaves `requires_grad=True` on outputs. The **Mock Interview Lab** at the end turns the lecture's "future improvement" list into eight runnable drills.

| Round | Questions | Focus | Course time |
|---|---|---|---|
| 1. Confidence warm-up | Q1–Q8 | Preprocessing, BoW vs TF-IDF, embeddings, sentiment pipeline | 10–15 min |
| 2. PyTorch reality check | Q9–Q20 | Tensors, autograd, training loop, conversions, shapes | 20–25 min |
| 3. RNN, LSTM, text generation | Q21–Q30 | Hidden state, vanishing gradients, gates, temperature | 25–30 min |
| 4. Attention and self-attention | Q31–Q40 | Q/K/V, scaling, shapes, softmax axis, position, cost | 30–35 min |
| 5. CV, autoencoders, GANs | Q41–Q47 | Detection, segmentation, reconstruction, mode collapse | 15–20 min |
| 6. Final pressure round | Q48–Q51 | Completeness, shapes, readiness, preparation plan | 5–10 min |

:::analogy[Driving test, not the theory exam]
Knowing the highway code is the theory exam. The interview is the driving test: the examiner asks you to explain, then changes the conditions (a follow-up), then watches you handle something going wrong (a debugging question). This page trains the driving test.
:::


## 1. Phase 1: Confidence Warm-Up (NLP Basics)

The first round starts with text preprocessing and the sentiment pipeline. These are the questions candidates feel safe on, which is exactly why interviewers use them: a confident one-line answer is easy, but a *good* answer gives an example, names the trade-off and says when the textbook step is wrong. Each answer below follows that shape: the definition, a concrete example, then the nuance an interviewer is waiting for.

<details>
<summary><strong>Q1. Why do we need text preprocessing in NLP?</strong> What problems can raw text create for a machine learning model?</summary>

Raw text is noisy and inconsistent, and models can't use strings directly. The same word can appear as *Good*, *good* and *GOOD*, which a model would treat as three different tokens; text also carries punctuation, special characters, spelling variants and symbols that add vocabulary without adding meaning. Preprocessing normalises this (lowercasing, tokenisation, punctuation removal, stopword handling, stemming or lemmatisation) and ends in **vectorisation**, turning text into numbers. The point of all of it is to prepare text for a numerical representation the model can learn from.

**The nuance:** preprocessing is task-dependent, and too much of it hurts. Removing *not* from "This movie is not good" turns it into "This movie is good", reversing the sentiment. Transformer models go further and skip most classic preprocessing, because their subword tokenizers and contextual embeddings handle case and rare words themselves.

**Read more:** [nlp foundations words to vectors](./03-nlp-foundations/01-nlp-foundations-words-to-vectors.md) · [nlp in practice preprocessing word2vec sentiment](./03-nlp-foundations/02-nlp-in-practice-preprocessing-word2vec-sentiment.md)

</details>

<details>
<summary><strong>Q2. What is the difference between tokenization and vectorization?</strong> Use the sentence: I love machine learning.</summary>

**Tokenisation** splits text into units: words, subwords or characters. "I love machine learning" becomes `["I", "love", "machine", "learning"]` with a word tokenizer. **Vectorisation** turns those tokens into numbers: a count or presence vector over a vocabulary (Bag of Words, TF-IDF), or dense embeddings. Tokenisation converts text into tokens; vectorisation converts tokens into numbers.

**The nuance:** tokenising isn't the end of the job, because models need numbers, not strings. In practice the two steps meet in the middle: a tokenizer maps each token to an integer **id** from the vocabulary, and an embedding layer maps each id to a vector (Q19).

**Read more:** [nlp foundations words to vectors](./03-nlp-foundations/01-nlp-foundations-words-to-vectors.md)

</details>

<details>
<summary><strong>Q3. What is the difference between stemming and lemmatization?</strong> Both reduce words to simpler forms. Why do we need both?</summary>

**Stemming** chops suffixes with rules and may produce non-words: the Porter stemmer turns *studies* and *studying* into **studi**, and merges unrelated words (*university* and *universe* both become **univers**). **Lemmatisation** maps a word to its dictionary form using vocabulary and part of speech: *studies* → **study**. Stemming is fast and crude; lemmatisation is more accurate and heavier, since it needs a dictionary and often a part-of-speech tag.

**The nuance:** don't say lemmatisation is always better. The right choice depends on the task, the accuracy and speed needed, the language and how complex a pipeline you can afford; for search over huge corpora, a stemmer's speed often wins, and for Transformer models neither is used.

**Read more:** [nlp foundations words to vectors](./03-nlp-foundations/01-nlp-foundations-words-to-vectors.md)

</details>

<details>
<summary><strong>Q4. Why can stopword removal be risky in sentiment analysis?</strong> Consider: The movie is not good.</summary>

Standard stopword lists include negations and intensifiers (*not, no, never, hardly, very*) because they are frequent. In sentiment analysis those words carry the meaning: removing *not* turns "The movie is not good" into "The movie is good", flipping the label. So stopword removal should never be applied blindly; for sentiment, keep negations (or the whole list) and let the model decide what matters.

**The nuance, in the course's words:** preprocessing is not a ritual, it is a design decision. Removing stopwords helps keyword-style tasks (topic classification, search) and can hurt tasks where small function words change meaning.

**Read more:** [nlp in practice preprocessing word2vec sentiment](./03-nlp-foundations/02-nlp-in-practice-preprocessing-word2vec-sentiment.md)

</details>

<details>
<summary><strong>Q5. What is the difference between Bag of Words and TF-IDF?</strong> Both convert text into numerical vectors. What changes?</summary>

**Bag of Words** represents a document by word counts (or presence) over the vocabulary; it says how often each word appears but not whether that word is common everywhere. **TF-IDF** (term frequency × inverse document frequency) re-weights those counts: a word gets a high weight if it is frequent in *this* document but rare across the corpus, and words that appear in nearly every document get pushed toward zero. Bag of Words measures frequency; TF-IDF measures importance relative to the corpus, which highlights discriminative terms.

**The nuance:** neither understands word order. "dog bites man" and "man bites dog" give **identical** vectors: scikit-learn's `CountVectorizer` produces `[1, 1, 1]` over `bites, dog, man` for both, and their TF-IDF rows are equal too.

**Read more:** [nlp foundations words to vectors](./03-nlp-foundations/01-nlp-foundations-words-to-vectors.md)

</details>

<details>
<summary><strong>Q6. Why are word embeddings better than one-hot vectors?</strong> Suppose the vocabulary size is 50,000.</summary>

A one-hot vector has one position per vocabulary word: with 50,000 words, each word is a 50,000-long vector with a single 1, which is sparse, high-dimensional (200 KB per word as float32) and grows with the vocabulary. Worse, every pair of one-hot vectors is orthogonal: the dot product of *king* and *queen* is exactly 0, so the representation says nothing about meaning. **Word embeddings** are dense, low-dimensional vectors (typically 50–1,000 numbers) learned so that related words point in similar directions, which captures semantic similarity and generalises to words seen in similar contexts.

**The nuance:** Word2Vec-style embeddings are **static**, one vector per word whatever the context. Transformers produce **contextual** embeddings, where *bank* gets different vectors in "river bank" and "bank account".

**Read more:** [nlp foundations words to vectors](./03-nlp-foundations/01-nlp-foundations-words-to-vectors.md) · [nlp in practice preprocessing word2vec sentiment](./03-nlp-foundations/02-nlp-in-practice-preprocessing-word2vec-sentiment.md)

</details>

<details>
<summary><strong>Q7. What are the main stages of a sentiment classification pipeline?</strong> Explain the path from raw text to prediction.</summary>

Collect labelled text → clean and preprocess → convert text to numerical features → split into training and test sets → train a classifier → evaluate it → and at inference, run new text through the **same** preprocessing and vectorisation before predicting. For example: raw review → preprocessing → TF-IDF → logistic regression → sentiment.

**The nuance:** consistency between training and inference is the part that breaks in production. If inference lowercases differently, uses a different tokenizer, or refits the vectoriser on new data, the model receives inputs in a format it never learned and predictions become unreliable. Save the fitted vectoriser (or the whole pipeline object) with the model, and fit it on the training split only to avoid leakage.

**Read more:** [nlp in practice preprocessing word2vec sentiment](./03-nlp-foundations/02-nlp-in-practice-preprocessing-word2vec-sentiment.md)

</details>

<details>
<summary><strong>Q8. Why do we need sequence models for text?</strong> Compare: The dog chased the cat. / The cat chased the dog.</summary>

Because word order carries meaning. Both sentences contain the same words, so Bag of Words and TF-IDF give them identical vectors, yet the meanings are opposite. **Sequence models** such as RNNs and LSTMs read tokens in order and carry a hidden state from one step to the next, so the representation depends on order and context.

**The nuance (the bridge to the next rounds):** once you accept that order matters, you need models that process sequences, not word counts. Order-aware alternatives include n-gram features, RNN/LSTM/GRU, 1-D convolutions, and Transformers, which add positional information to otherwise order-blind attention (Q38).

**Read more:** [sequence data and recurrence](./04-rnns/01-sequence-data-and-recurrence.md) · [lstm and gru gated memory](./05-lstm-and-gru/01-lstm-and-gru-gated-memory.md)

</details>

:::tip[🎯 Interview angle]
**"Why can't you just remove stopwords and punctuation for every NLP task?"** Because preprocessing is a design decision tied to the task. Negations like *not* carry sentiment, punctuation and case can matter ("US" vs "us"), and Transformer tokenizers need the raw text. Choose each step by asking what information the task needs, and apply exactly the same steps at inference.
:::


## 2. Phase 2: PyTorch Reality Check

This round checks whether you understand what PyTorch does during training rather than what the code looks like: tensors, `requires_grad`, autograd, the five-step training loop, tensor conversion and shapes. Interviewers like these because they separate people who have trained models from people who have read about it; every snippet below was run and the outputs are copied from the run.

<details>
<summary><strong>Q9. What is a tensor in PyTorch?</strong> How is it different from a Python list or NumPy array?</summary>

A tensor is a multi-dimensional array of numbers with a **shape**, a **dtype** and a **device**. It represents everything in deep learning: a scalar loss, a vector of biases, a weight matrix, a batch of images $(N, C, H, W)$ or a batch of token sequences $(N, T)$. Compared with a Python list, it stores one contiguous block of numbers of one type and runs vectorised maths in optimised C/CUDA. Compared with a NumPy array, it adds the two things deep learning needs: **GPU acceleration** (`.to("cuda")`) and **automatic differentiation** (autograd records operations to compute gradients).

**The nuance:** "a tensor is like an array" is the weak answer. The strong one mentions shape, dtype, device, autograd, and that tensors carry inputs, weights, activations and gradients alike.

**Read more:** [pytorch fundamentals](./01-pytorch-and-neural-networks/01-pytorch-fundamentals.md)

</details>

<details>
<summary><strong>Q10. What does requires_grad=True mean?</strong> Why do we write x = torch.tensor(2.0, requires_grad=True)?</summary>

It tells autograd to record every operation involving that tensor so gradients with respect to it can be computed later. For $y = x^2$, $dy/dx = 2x$, so at $x = 2$ the gradient is 4.

```python showLineNumbers
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)                # tensor(4.)
```

**The nuance:** model parameters (`nn.Parameter`) require gradients by default because they're what training updates. Input data usually doesn't; input gradients are needed only for special cases such as saliency maps, adversarial examples or optimising an input.

**Read more:** [pytorch fundamentals](./01-pytorch-and-neural-networks/01-pytorch-fundamentals.md) · [calculus behind pytorch](./01-pytorch-and-neural-networks/02-calculus-behind-pytorch.md)

</details>

<details>
<summary><strong>Q11. What is autograd in PyTorch?</strong> How does PyTorch know which gradients to compute?</summary>

Autograd is PyTorch's automatic differentiation engine. During the forward pass it builds a **dynamic computation graph**: every operation on tensors that require gradients is recorded with a function that knows its local derivative. `loss.backward()` walks that graph backwards from the loss and applies the **chain rule**, accumulating $\partial \loss{L} / \partial \param{w}$ into each leaf tensor's `.grad`. For $y = x^2 + 2x + 1$, $dy/dx = 2x + 2$, and autograd returns `tensor(6.)` at $x = 2$.

**The nuance:** "dynamic" means the graph is rebuilt on every forward pass, so ordinary Python control flow (loops, `if` statements, variable-length sequences) just works, and the graph is freed after `backward()` unless you ask to keep it.

**Read more:** [pytorch fundamentals](./01-pytorch-and-neural-networks/01-pytorch-fundamentals.md) · [calculus behind pytorch](./01-pytorch-and-neural-networks/02-calculus-behind-pytorch.md)

</details>

<details>
<summary><strong>Q12. What is the role of loss.backward()?</strong> Does it update model weights?</summary>

It computes gradients; it does **not** update weights. `loss.backward()` fills each trainable parameter's `.grad` with $\partial \loss{L} / \partial \param{w}$. The update happens in `optimizer.step()`, which reads those `.grad` values and changes the weights (for plain SGD, $\param{w} \leftarrow \param{w} - \hyper{\eta}\, \partial \loss{L} / \partial \param{w}$).

**The nuance:** this is one of the most common mistakes in interviews. Say it in two lines: `backward()` computes gradients, `step()` updates weights.

**Read more:** [pytorch fundamentals](./01-pytorch-and-neural-networks/01-pytorch-fundamentals.md)

</details>

<details>
<summary><strong>Q13. Why do we call optimizer.zero_grad()?</strong> What happens if we forget it?</summary>

Because PyTorch **accumulates** gradients: each `backward()` *adds* to `.grad` instead of overwriting it. Without clearing, the gradient used for this batch is the sum of all previous batches' gradients. With $\loss{L} = (3w)^2/2$, whose gradient is $9w = 9$ at $w = 1$, three `backward()` calls without zeroing leave `w.grad` at 9, then 18, then 27. Fitting $y = 3x + 1$ with SGD for 50 steps reaches weight 3.0 and bias 1.0 with `zero_grad()`, but only weight 1.31 and bias −0.41 (loss 1.59) without it.

**The nuance:** accumulation is deliberate, not a bug. **Gradient accumulation** uses it to simulate a larger batch: call `backward()` on several small batches (dividing each loss by the number of batches) and `step()` + `zero_grad()` once. `zero_grad(set_to_none=True)`, the default in recent PyTorch, sets `.grad` to `None` instead of zeros, saving memory.

**Read more:** [pytorch fundamentals](./01-pytorch-and-neural-networks/01-pytorch-fundamentals.md)

</details>

<details>
<summary><strong>Q14. What is the correct order of a PyTorch training step?</strong> Arrange: loss.backward(), optimizer.step(), optimizer.zero_grad(), outputs = model(inputs), loss = loss_fn(outputs, targets).</summary>

Clear old gradients → forward pass → compute the loss → backward pass → update parameters.

```python showLineNumbers
optimizer.zero_grad()                 # 1. clear old gradients
outputs = model(inputs)               # 2. forward pass
loss = loss_fn(outputs, targets)      # 3. compute loss
loss.backward()                       # 4. compute gradients
optimizer.step()                      # 5. update parameters
```

**Memory hook:** zero grad → forward → loss → backward → step. `zero_grad()` can also go right after `step()`; what matters is that it happens once between consecutive `backward()` calls.

**Read more:** [pytorch fundamentals](./01-pytorch-and-neural-networks/01-pytorch-fundamentals.md)

</details>

<details>
<summary><strong>Q15. What is the purpose of nn.Module?</strong> Why do we create model classes using nn.Module?</summary>

`nn.Module` is the base class for every layer and model. Subclassing it gives you automatic **parameter registration** (any `nn.Parameter` or sub-module assigned in `__init__` shows up in `model.parameters()`, so the optimizer finds it), nested **sub-modules**, **device moves** (`model.to("cuda")` moves everything), **saving and loading** (`state_dict()`), **train/eval modes** (`model.train()`, `model.eval()`) and hooks. You define the layers in `__init__` and the computation in `forward`, and call the model as `model(x)`, which runs hooks and then `forward`.

**The nuance:** `__init__` says *what layers exist*; `forward` says *how data flows*. Layers stored in a plain Python list aren't registered (use `nn.ModuleList`), so the optimizer silently never trains them.

**Read more:** [pytorch fundamentals](./01-pytorch-and-neural-networks/01-pytorch-fundamentals.md) · [artificial neural networks](./01-pytorch-and-neural-networks/03-artificial-neural-networks.md)

</details>

<details>
<summary><strong>Q16. What is the difference between .item(), .detach(), .cpu() and .numpy()?</strong> Why can arr = x.numpy() fail?</summary>

`.item()` turns a **one-element** tensor into a Python number (`tensor([3.5]).item()` is `3.5`, a `float`; on a 2-element tensor it raises `RuntimeError: a Tensor with 2 elements cannot be converted to Scalar`). `.detach()` returns a tensor that shares the data but is cut out of the autograd graph. `.cpu()` copies a tensor from the GPU to the CPU. `.numpy()` returns a NumPy array sharing memory with a CPU tensor. `x.numpy()` fails if `x` still requires gradients or lives on the GPU; for a tensor with `requires_grad=True` PyTorch says exactly: `RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.`

**The nuance:** remember the safe chain `x.detach().cpu().numpy()` for plotting or NumPy analysis, and use `.item()` for logging a scalar loss; logging `loss` itself keeps the whole graph alive and leaks memory.

**Read more:** [pytorch fundamentals](./01-pytorch-and-neural-networks/01-pytorch-fundamentals.md)

</details>

<details>
<summary><strong>Q17. What is broadcasting in PyTorch?</strong> What happens with A = torch.randn(3, 4), b = torch.randn(4), C = A + b?</summary>

Broadcasting lets element-wise operations combine tensors of different but compatible shapes without copying data. Shapes are compared from the **last** dimension backwards; two dimensions are compatible if they're equal or one of them is 1 (a missing dimension counts as 1). Here `b` has shape (4,), treated as (1, 4) and stretched across the 3 rows, so `C` has shape (3, 4): `b` is added to every row. This is exactly how a bias is added in $Y = XW + b$: one bias vector, broadcast over the batch.

**The nuance:** know the failure and the silent bug. `A + torch.randn(3)` fails with `RuntimeError: The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 1`; to add one number per **row**, reshape to (3, 1). And `(N,) + (N, 1)` broadcasts silently to `(N, N)`, a classic source of wrong losses when predictions and targets have mismatched shapes.

**Read more:** [pytorch fundamentals](./01-pytorch-and-neural-networks/01-pytorch-fundamentals.md)

</details>

<details>
<summary><strong>Q18. What is the meaning of the batch dimension?</strong> Explain (batch_size, sequence_length, embedding_dim).</summary>

The batch dimension counts how many independent samples go through the model in one forward pass. For NLP, `(32, 20, 100)` means 32 sentences, each 20 tokens long (after padding or truncation), each token a 100-dimensional vector. Batching makes the hardware efficient (one big matrix multiply instead of 32 small ones) and gives a less noisy gradient estimate than a single example.

**The nuance:** many bugs come from confusing batch size with sequence length. PyTorch's `nn.RNN` / `nn.LSTM` default to `(seq_len, batch, features)` unless you pass `batch_first=True`; feed a `(32, 20, 8)` batch to a default `nn.LSTM` and it silently treats 32 as the sequence length and 20 as the batch, so the final hidden state comes back as `(1, 20, 16)` instead of `(1, 32, 16)`.

**Read more:** [rnns in practice training and limits](./04-rnns/03-rnns-in-practice-training-and-limits.md) · [lstm and gru gated memory](./05-lstm-and-gru/01-lstm-and-gru-gated-memory.md)

</details>

<details>
<summary><strong>Q19. What is an embedding layer in PyTorch?</strong> How are token IDs converted into vectors?</summary>

`nn.Embedding(vocab_size, embedding_dim)` is a trainable lookup table: a weight matrix with one row per token id. Given integer ids of shape (batch, seq_len), it returns those rows, shape (batch, seq_len, embedding_dim). `nn.Embedding(10000, 128)` has 10,000 × 128 = 1,280,000 parameters and maps a `(32, 20)` batch of ids to `(32, 20, 128)`. It is equivalent to multiplying a one-hot vector by the weight matrix, without ever building the one-hot vector.

**The nuance:** embeddings can be learned from scratch, initialised from pretrained vectors (Word2Vec, GloVe) and fine-tuned, or frozen. Ids must be integers in `[0, vocab_size)`; id 10000 in a 10,000-row table raises `IndexError: index out of range in self`. `padding_idx` keeps the padding row at zero and out of training.

**Read more:** [pytorch fundamentals](./01-pytorch-and-neural-networks/01-pytorch-fundamentals.md) · [nlp in practice preprocessing word2vec sentiment](./03-nlp-foundations/02-nlp-in-practice-preprocessing-word2vec-sentiment.md)

</details>

<details>
<summary><strong>Q20. What is the difference between model.eval() and torch.no_grad()?</strong> Why do we use both during testing?</summary>

`model.eval()` switches layers whose behaviour differs between training and inference: **dropout** stops dropping (in training, `Dropout(0.5)` on a row of six 1s gave `[0, 0, 2, 0, 0, 0]`; in eval mode it returns the input unchanged) and **batch normalisation** uses its stored running statistics instead of the batch's. It does **not** turn off gradient tracking: an output computed in eval mode still has `requires_grad=True`. `torch.no_grad()` disables gradient tracking, so no graph is built (outputs have `requires_grad=False`), which saves memory and time. For evaluation you want both: correct layer behaviour **and** no graph.

**The nuance:** `model.eval()` → changes layer behaviour; `torch.no_grad()` → disables gradient tracking. `torch.inference_mode()` is a stricter, slightly faster `no_grad` for pure inference. Forgetting to call `model.train()` again before the next epoch is a common bug.

**Read more:** [pytorch fundamentals](./01-pytorch-and-neural-networks/01-pytorch-fundamentals.md)

</details>

:::tip[🎯 Interview angle]
**"Walk me through one PyTorch training step and what each line does to the tensors."** `zero_grad()` clears `.grad`; the forward pass builds the autograd graph; the loss is a scalar at its root; `backward()` applies the chain rule and fills each parameter's `.grad`; `step()` updates the weights from those gradients. For evaluation, `model.eval()` changes dropout and batch-norm behaviour and `torch.no_grad()` stops graph building.
:::


## 3. Phase 3: RNN, LSTM and Text Generation

This round checks sequence modelling: what the hidden state is, why vanilla RNNs forget, how the LSTM's gates and cell state fix it, and how a trained model generates text. Expect follow-ups on the equations, so each answer below states the formula with the page's colour code and says what each term does.

<details>
<summary><strong>Q21. What is the hidden state in an RNN?</strong> Why is it needed?</summary>

The hidden state $\pred{h_t}$ is a vector that summarises everything the RNN has read up to step $t$. At each step the RNN combines the new input with the previous hidden state, $\pred{h_t} = \tanh(\param{W_x}\data{x_t} + \param{W_h}\pred{h_{t-1}} + \param{b})$, using the **same** weights at every step. That shared, recurrent update is what lets one network process sequences of any length and carry information from earlier tokens to later ones; the last hidden state is often used as a summary of the whole sequence.

**The nuance:** the hidden state is a fixed-size, lossy **compression**: it can't store all previous words perfectly, and in practice information from far back fades (Q22).

**Read more:** [sequence data and recurrence](./04-rnns/01-sequence-data-and-recurrence.md) · [rnn architecture forward pass bptt](./04-rnns/02-rnn-architecture-forward-pass-bptt.md)

</details>

<details>
<summary><strong>Q22. What is the vanishing gradient problem in RNNs?</strong> Why does it affect long sequences?</summary>

Training an RNN backpropagates through time: the gradient that reaches step $t-k$ is a product of $k$ Jacobians, each involving $\param{W_h}$ and the derivative of tanh (at most 1). If those factors are typically smaller than 1, the product shrinks exponentially with distance, so early time steps receive almost no learning signal and the network can't learn long-range dependencies (connecting a pronoun to a noun 30 words back). If the factors are larger than 1 the gradient **explodes** instead, which is handled by gradient clipping.

**The nuance:** vanishing gradients don't mean the network can't *see* old inputs in the forward pass; it means training can't *learn* to use them. LSTM and GRU address it with an additive memory path (Q27); Transformers avoid the long chain entirely.

**Read more:** [rnns in practice training and limits](./04-rnns/03-rnns-in-practice-training-and-limits.md) · [lstm and gru gated memory](./05-lstm-and-gru/01-lstm-and-gru-gated-memory.md)

</details>

<details>
<summary><strong>Q23. Why was LSTM introduced?</strong> What problem does it solve compared to a vanilla RNN?</summary>

To learn long-term dependencies that vanilla RNNs lose to vanishing gradients. An LSTM adds a separate **cell state** $\pred{C_t}$ that runs through time with only element-wise, gated changes, and three **gates** (forget, input, output) plus a **candidate memory** that decide what to erase, what to write and what to expose. Because the cell state is updated **additively**, gradients can flow back along it over many steps without being repeatedly squashed.

**The nuance:** LSTM improves long-sequence learning but doesn't solve long context completely; it still processes tokens one at a time and compresses history into fixed-size vectors. Transformers became popular partly because they avoid strict sequential processing.

**Read more:** [lstm and gru gated memory](./05-lstm-and-gru/01-lstm-and-gru-gated-memory.md)

</details>

<details>
<summary><strong>Q24. What is the difference between cell state and hidden state in LSTM?</strong> Which one is long-term memory?</summary>

The **cell state** $\pred{C_t}$ is the internal long-term memory: a conveyor belt carried across time steps with controlled, gated updates. The **hidden state** $\pred{h_t} = \pred{o_t} \odot \tanh(\pred{C_t})$ is the exposed, filtered view of that memory at step $t$: it is the layer's output, used for predictions and passed to the next layer and the next step's gates. Cell state = long-term memory; hidden state = what the cell chooses to show now.

**The nuance:** in PyTorch, `nn.LSTM` returns `output, (h_n, c_n)`; `output` holds the hidden state at every step, and the prediction head almost always uses the hidden state, not the cell state.

**Read more:** [lstm and gru gated memory](./05-lstm-and-gru/01-lstm-and-gru-gated-memory.md)

</details>

<details>
<summary><strong>Q25. Explain the forget gate in LSTM.</strong> Why does it use sigmoid?</summary>

The forget gate decides how much of each component of the old cell state to keep: $\pred{f_t} = \sigma(\param{W_f}[\pred{h_{t-1}}, \data{x_t}] + \param{b_f})$, applied as $\pred{f_t} \odot \pred{C_{t-1}}$. The sigmoid outputs values between 0 and 1, so each component acts like a dial: near 1 keeps the memory, near 0 erases it, and values in between keep part of it.

**The nuance:** it's a **soft**, learned filter, not a binary keep/delete, and it depends on the current input and previous hidden state, so the network learns *when* to forget (for example, at the start of a new sentence). Initialising the forget-gate bias to a positive value (around 1) is a known trick that keeps memory by default early in training.

**Read more:** [lstm and gru gated memory](./05-lstm-and-gru/01-lstm-and-gru-gated-memory.md)

</details>

<details>
<summary><strong>Q26. What is the role of the input gate and candidate memory?</strong> Why do we need both?</summary>

The **candidate memory** $\pred{\tilde C_t} = \tanh(\param{W_C}[\pred{h_{t-1}}, \data{x_t}] + \param{b_C})$ proposes new content, values between −1 and 1. The **input gate** $\pred{i_t} = \sigma(\param{W_i}[\pred{h_{t-1}}, \data{x_t}] + \param{b_i})$ decides how much of each proposed component is actually written, so the new information added is $\pred{i_t} \odot \pred{\tilde C_t}$. Splitting them separates *what* could be written (candidate) from *whether* to write it (gate).

**The analogy from the course:** the candidate memory writes a draft; the input gate decides how much of the draft enters memory.

**Read more:** [lstm and gru gated memory](./05-lstm-and-gru/01-lstm-and-gru-gated-memory.md)

</details>

<details>
<summary><strong>Q27. What is the complete LSTM cell state update equation?</strong> Explain each term.</summary>

$\pred{C_t} = \pred{f_t} \odot \pred{C_{t-1}} + \pred{i_t} \odot \pred{\tilde C_t}$. The first term is the **retained old memory** (old cell state scaled by the forget gate); the second is the **selected new memory** (candidate scaled by the input gate). With one component where $\pred{f_t} = 0.9$, $\pred{C_{t-1}} = 1.5$, $\pred{i_t} = 0.2$ and $\pred{\tilde C_t} = -0.8$: $\pred{C_t} = 1.35 - 0.16 = 1.19$, and with output gate 0.7 the hidden state is $0.7 \cdot \tanh(1.19) = 0.5814$.

**The nuance:** the update is **additive**. Along the cell path, $\partial \pred{C_t} / \partial \pred{C_{t-1}} = \pred{f_t}$ (element-wise), so with forget gates near 1 the gradient passes through many steps without the repeated weight-matrix multiplication that makes vanilla RNN gradients vanish.

**Read more:** [lstm and gru gated memory](./05-lstm-and-gru/01-lstm-and-gru-gated-memory.md)

</details>

<details>
<summary><strong>Q28. What is the output gate in LSTM?</strong> How is the hidden state produced?</summary>

The output gate $\pred{o_t} = \sigma(\param{W_o}[\pred{h_{t-1}}, \data{x_t}] + \param{b_o})$ decides how much of the updated memory to expose. The hidden state is $\pred{h_t} = \pred{o_t} \odot \tanh(\pred{C_t})$: tanh squashes the cell state into −1 to 1, and the gate scales each component down to what should be visible now.

**The nuance:** the cell state stores memory; the hidden state exposes selected memory. The cell can hold information that isn't relevant to the current output (the gate hides it) and release it later.

**Read more:** [lstm and gru gated memory](./05-lstm-and-gru/01-lstm-and-gru-gated-memory.md)

</details>

<details>
<summary><strong>Q29. How does an LSTM text generator generate the next word?</strong> What happens after one word is predicted?</summary>

Feed a seed sequence through the LSTM; the final hidden state goes through a linear layer and softmax to give a probability distribution over the vocabulary for the next token. Choose a token from that distribution (argmax, random sampling, or temperature / top-k / top-p sampling), **append it to the input**, and repeat, so each generated token becomes part of the context for the next. This step-by-step loop is **autoregressive generation**, the same idea GPT-style models use.

**The nuance:** always taking the argmax (greedy decoding) is deterministic and tends to repeat itself; sampling adds variety. In practice you carry the hidden state forward instead of re-reading the whole sequence each step.

**Read more:** [lstm text generation temperature sampling](./05-lstm-and-gru/02-lstm-text-generation-temperature-sampling.md)

</details>

<details>
<summary><strong>Q30. What is temperature in text generation?</strong> What happens when temperature is high or low?</summary>

Temperature $\hyper{T}$ rescales the logits before softmax: $\pred{p_i} = \operatorname{softmax}(\pred{z_i} / \hyper{T})$. Low temperature sharpens the distribution toward the most likely token; high temperature flattens it. For logits (2.0, 1.0, 0.1): at $\hyper{T} = 0.5$ the probabilities are (0.8638, 0.1169, 0.0193); at $\hyper{T} = 1$ (0.659, 0.2424, 0.0986); at $\hyper{T} = 2$ (0.5017, 0.3043, 0.194). As $\hyper{T} \to 0$ sampling becomes greedy.

**The nuance:** very high temperature gives diverse but meaningless text; very low gives safe but repetitive text. Temperature doesn't change the ranking of tokens, only how concentrated the distribution is; top-k and top-p cut off the unlikely tail instead.

**Read more:** [lstm text generation temperature sampling](./05-lstm-and-gru/02-lstm-text-generation-temperature-sampling.md)

</details>

:::tip[🎯 Interview angle]
**"Why does an LSTM remember longer than a vanilla RNN?"** Its cell state is updated additively, $\pred{C_t} = \pred{f_t} \odot \pred{C_{t-1}} + \pred{i_t} \odot \pred{\tilde C_t}$, so the gradient along the cell path is scaled by the forget gate rather than repeatedly multiplied by a weight matrix and a tanh derivative. Gates learn what to keep, write and expose, and the hidden state is the filtered view used for predictions.
:::


## 4. Phase 4: Attention and Self-Attention

The deepest round. Expect to be tested on intuition (why attention, what Q, K and V are), the formula, **tensor shapes**, the softmax dimension, positional information and the cost of attention. Shape questions are where most candidates stumble, so the answers below give shapes from a real run.

<details>
<summary><strong>Q31. Why was attention introduced in sequence models?</strong> What problem did it solve in encoder-decoder models?</summary>

In a basic RNN encoder-decoder (for translation), the encoder squeezes the whole input sentence into one fixed-size vector, its final hidden state, and the decoder must generate the entire output from it. For long inputs that vector becomes a **bottleneck** and early words are lost. Attention lets the decoder look at **all** encoder hidden states at every output step and take a weighted average of them, with weights computed from how relevant each input position is to the word being generated now.

**The nuance:** different output words depend on different input words (when translating the verb, look at the source verb), so a single fixed vector is the wrong design; attention also made models more interpretable through their alignment weights.

**Read more:** [from lstm to attention why transformers](./06-attention-and-transformers/01-from-lstm-to-attention-why-transformers.md)

</details>

<details>
<summary><strong>Q32. What are Query, Key and Value in attention?</strong> Explain intuitively.</summary>

Each token is projected into three vectors with learned matrices: $\pred{Q} = \data{X}\param{W_Q}$, $\pred{K} = \data{X}\param{W_K}$, $\pred{V} = \data{X}\param{W_V}$. The **query** represents what a position is looking for; the **key** represents what each position offers for matching; the **value** is the information each position actually passes on. The query is compared with every key (dot product) to get scores, softmax turns the scores into weights, and the output is the weighted sum of the values.

**The analogy:** the query asks a question, keys are the labels on the folders, and values are what's inside the folders. Separate K and V let a token be *found* for one reason and *contribute* something different.

**Read more:** [full self attention scaled dot product](./06-attention-and-transformers/02-full-self-attention-scaled-dot-product.md)

</details>

<details>
<summary><strong>Q33. Explain the scaled dot-product attention formula.</strong> Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V</summary>

Step by step: $\pred{Q}\pred{K}^\top$ computes the similarity of every query with every key; dividing by $\sqrt{\hyper{d_k}}$ keeps the scores at a stable scale; softmax over the keys turns each row into weights that sum to 1; multiplying by $\pred{V}$ gives each position a weighted, context-aware combination of the values.

**Why the scaling matters:** for random vectors with unit-variance components, the dot product's standard deviation grows like $\sqrt{\hyper{d_k}}$: measured over 10,000 random pairs it was 2.78 for $\hyper{d_k} = 8$, 7.96 for 64 and 22.63 for 512, and about 1 after dividing by $\sqrt{\hyper{d_k}}$. Unscaled scores that large push softmax into a near one-hot output (the largest weight was 1.0 for six random scores at $\hyper{d_k} = 512$, versus 0.5926 after scaling), where its gradients vanish and training stalls.

**Read more:** [full self attention scaled dot product](./06-attention-and-transformers/02-full-self-attention-scaled-dot-product.md)

</details>

<details>
<summary><strong>Q34. If input embeddings have shape (batch, seq_len, emb_dim), what is the shape of the self-attention score matrix?</strong> Assume each token attends to every other token.</summary>

**(batch, seq_len, seq_len)**: one score for every (query position, key position) pair in each sequence. With batch 4, sequence length 6 and embedding size 8, a real run gives scores and weights of shape (4, 6, 6) and an output of shape (4, 6, 8); each row of the weight matrix sums to 1. One row shows how much one query token attends to all key tokens.

**The nuance:** with multi-head attention the scores are (batch, heads, seq_len, seq_len), and the output goes back to (batch, seq_len, emb_dim) after the heads are concatenated and projected. The seq_len × seq_len size is where the quadratic cost comes from (Q39).

**Read more:** [full self attention scaled dot product](./06-attention-and-transformers/02-full-self-attention-scaled-dot-product.md) · [multi head attention](./06-attention-and-transformers/03-multi-head-attention.md)

</details>

<details>
<summary><strong>Q35. Why is softmax used in attention?</strong> Along which dimension is it applied?</summary>

Softmax turns raw scores into positive weights that sum to 1, so each output is a proper weighted average of the value vectors, and it emphasises the highest-scoring positions while staying differentiable. It is applied along the **key** dimension, the last axis of the (batch, query, key) score tensor (`dim=-1`), so each query gets a probability distribution over the tokens it attends to.

**The nuance:** applied along the wrong axis it still runs without error, but the weights no longer mean "how much query $i$ attends to each key". In a real run, softmax over the query axis made each **column** sum to 1 while the rows summed to between 0.912 and 1.066. Check `P.sum(-1)` is all ones.

**Read more:** [full self attention scaled dot product](./06-attention-and-transformers/02-full-self-attention-scaled-dot-product.md)

</details>

<details>
<summary><strong>Q36. What is attention visualization?</strong> What does an attention heatmap show?</summary>

It plots the attention weight matrix as a heatmap. For a sentence of $n$ tokens the self-attention matrix is $n \times n$: rows are usually query tokens, columns are key tokens, and a bright cell means that query attends strongly to that key. It is used to inspect what a model links together (a pronoun attending to its noun) and to debug masks.

**The nuance:** attention weights are not a faithful explanation of model reasoning. Each layer and head has its own pattern, values are mixed afterwards by feed-forward layers and residual connections, and different weight patterns can give the same output, so treat heatmaps as a diagnostic, not a proof.

**Read more:** [full self attention scaled dot product](./06-attention-and-transformers/02-full-self-attention-scaled-dot-product.md) · [multi head attention](./06-attention-and-transformers/03-multi-head-attention.md)

</details>

<details>
<summary><strong>Q37. What is the difference between encoder-decoder attention and self-attention?</strong> Where do Q, K and V come from?</summary>

In **self-attention**, $\pred{Q}$, $\pred{K}$ and $\pred{V}$ all come from the same sequence, so its tokens attend to each other (every Transformer encoder layer, and the decoder's masked self-attention). In **encoder-decoder (cross-) attention**, $\pred{Q}$ comes from the **decoder** and $\pred{K}$, $\pred{V}$ come from the **encoder** output, so each output position can look at the whole input sentence.

**The nuance:** a Transformer decoder layer has both: masked self-attention over the words generated so far (the causal mask stops it seeing the future), then cross-attention to the encoder. Decoder-only models such as GPT keep only the masked self-attention.

**Read more:** [transformer architecture figure 1](./06-attention-and-transformers/04-transformer-architecture-figure-1.md) · [generative rag end to end](./07-hugging-face-qna-and-rag/04-generative-rag-end-to-end.md)

</details>

<details>
<summary><strong>Q38. Why do Transformers need positional information?</strong> Why is self-attention alone not enough?</summary>

Self-attention treats its input as a set: shuffle the input tokens and the outputs are simply shuffled the same way. A real check confirms it: reversing the order of 6 input vectors and running self-attention gives exactly the reversed outputs. So without extra information "dog bites man" and "man bites dog" get the same representations up to order. **Positional encodings** (fixed sinusoidal, as in the original Transformer) or **learned positional embeddings** (BERT, GPT-2) are added to the token embeddings so each vector also carries where it is.

**The nuance:** modern models often use relative schemes such as rotary embeddings (RoPE) or ALiBi, which encode distances between tokens and extrapolate better to longer sequences.

**Read more:** [transformer architecture figure 1](./06-attention-and-transformers/04-transformer-architecture-figure-1.md)

</details>

<details>
<summary><strong>Q39. Why is self-attention more parallelizable than RNNs?</strong> What is the trade-off?</summary>

An RNN must compute $\pred{h_t}$ after $\pred{h_{t-1}}$, so its $n$ steps run one after another. Self-attention computes all pairwise interactions with a few matrix multiplications, so every position is processed at the same time, which suits GPUs, and any two tokens are one step apart instead of $n$. The trade-off is **quadratic** cost in sequence length: the score matrix has $n^2$ entries, $O(n^2 d)$ time and $O(n^2)$ memory. In float32 one head's scores for one sequence take 1 MiB at $n = 512$, 64 MiB at $n = 4{,}096$ and 4 GiB at $n = 32{,}768$.

**The strong line:** Transformers gain parallelism but pay quadratic attention cost. FlashAttention avoids storing the full matrix; sparse, sliding-window and linear attention variants reduce the cost itself.

**Read more:** [from lstm to attention why transformers](./06-attention-and-transformers/01-from-lstm-to-attention-why-transformers.md) · [full self attention scaled dot product](./06-attention-and-transformers/02-full-self-attention-scaled-dot-product.md)

</details>

<details>
<summary><strong>Q40. Explain self-attention without using the word attention.</strong> A pressure question.</summary>

Each word builds a new version of itself by looking at every other word in the sentence. It scores how relevant each word is to it, turns those scores into percentages that add up to 100, and takes the matching percentage of each word's information. The result is a context-aware representation: *bank* in "river bank" ends up mixed with *river*, and in "bank account" with *account*.

**The nuance:** this tests whether you understand the mechanism or only the vocabulary. A good plain-language answer covers the three steps (compare, normalise into weights, blend) and the outcome (context-aware vectors), and can be followed by the formula when asked.

**Read more:** [full self attention scaled dot product](./06-attention-and-transformers/02-full-self-attention-scaled-dot-product.md)

</details>

:::tip[🎯 Interview angle]
**"Walk me through self-attention with shapes."** Input (B, T, d) is projected to Q, K, V; scores $\pred{Q}\pred{K}^\top / \sqrt{\hyper{d_k}}$ are (B, T, T); softmax over the last axis gives weights that sum to 1 per query; weights × V gives (B, T, d). Scaling keeps softmax out of saturation, positional encodings supply order, and the T × T matrix makes the cost quadratic in sequence length.
:::


## 5. Phase 5: Detection, Segmentation, Autoencoders and GANs

In the course these were awareness-level questions, because the topics were taught by another faculty. Interviewers still ask them, so this site now has two full pages for them: [Object Detection and Segmentation](./08-vision-tasks-and-generative-models/01-object-detection-and-segmentation.md) and [Autoencoders and GANs](./08-vision-tasks-and-generative-models/02-autoencoders-and-gans.md), both checked with real models. The answers below are the interview-length versions.

<details>
<summary><strong>Q41. What is the difference between image classification, object detection and image segmentation?</strong> Give one example of each.</summary>

**Classification** predicts one label for the whole image: "this image contains a dog". **Object detection** predicts each object's class, a confidence score and its location as a bounding box: "a dog at this box". **Segmentation** predicts a class for every pixel: "these exact pixels belong to the dog". Classification is image-level, detection is object-level with location, and segmentation is pixel-level, the most detailed spatial output.

**The nuance:** detection outputs a variable-length list, which is why detectors need score thresholds and non-maximum suppression; on a real DETR run, a 0.5 threshold turned 100 query outputs into 3 detections for a coffee photo.

**Read more:** [object detection and segmentation](./08-vision-tasks-and-generative-models/01-object-detection-and-segmentation.md)

</details>

<details>
<summary><strong>Q42. What is a bounding box in object detection?</strong> What are common ways to represent it?</summary>

The smallest axis-aligned rectangle around an object. It is written as corners $(x_{min}, y_{min}, x_{max}, y_{max})$ or as centre and size $(c_x, c_y, w, h)$, in pixels or normalised to 0–1; COCO files use $(x_{min}, y_{min}, w, h)$. A detector answers two questions per object, where (the box, a regression) and what (class probabilities, a classification), and boxes are compared with IoU (overlap area ÷ union area). DETR's coffee-cup box is (172.4, 17.7, 410.8, 287.1) in corners and (291.6, 152.4, 238.4, 269.4) in centre format.

**Read more:** [object detection and segmentation](./08-vision-tasks-and-generative-models/01-object-detection-and-segmentation.md)

</details>

<details>
<summary><strong>Q43. What is the difference between semantic segmentation and instance segmentation?</strong> Use the example of multiple dogs in one image.</summary>

**Semantic** segmentation labels every pixel with its class but doesn't separate objects: with three dogs, all dog pixels are just "dog". **Instance** segmentation gives each object its own mask: dog 1, dog 2, dog 3. Instance segmentation is harder because it needs pixel-level classification and object-level separation (usually a detector plus a mask head, as in Mask R-CNN). On a two-cats image, a real Mask2Former model's semantic output was one "cat" region covering 78.3% of the pixels, and its panoptic output split it into cat 1 (37.7%) and cat 2 (37.9%).

**Read more:** [object detection and segmentation](./08-vision-tasks-and-generative-models/01-object-detection-and-segmentation.md)

</details>

<details>
<summary><strong>Q44. What is an autoencoder?</strong> What are its two main parts?</summary>

A network trained to reconstruct its own input. The **encoder** compresses the input into a small latent code; the **decoder** rebuilds the input from that code. No labels are needed, and the useful product is the code. The **bottleneck** stops the model from simply copying the input and forces it to learn compact features: on 8 × 8 handwritten digits, an 8-number code rebuilds unseen test digits with MSE 0.0145, against 0.0246 for 8-component PCA.

**Read more:** [autoencoders and gans](./08-vision-tasks-and-generative-models/02-autoencoders-and-gans.md)

</details>

<details>
<summary><strong>Q45. What is reconstruction loss?</strong> How is it used in autoencoders?</summary>

It measures how different the reconstruction $\pred{\hat x}$ is from the input $\data{x}$, commonly the mean squared error $\lVert \data{x} - \pred{\hat x} \rVert^2$ (averaged over pixels), or binary cross-entropy for 0–1 data. Training minimises it; a low value means good reconstruction. Trained on normal data only, an autoencoder rebuilds unusual inputs badly, so the reconstruction error becomes an **anomaly score**: trained on digits 0–8, its error on unseen 9s averaged 0.0309 against 0.0143 for normal digits (ROC AUC 0.8836).

**Read more:** [autoencoders and gans](./08-vision-tasks-and-generative-models/02-autoencoders-and-gans.md)

</details>

<details>
<summary><strong>Q46. What is the basic idea of a GAN?</strong> What are the generator and discriminator?</summary>

Two networks trained in competition. The **generator** turns random noise into fake samples that should look real; it never sees real data and learns only through the discriminator's gradient. The **discriminator** classifies samples as real or fake. The discriminator is trained to catch fakes, the generator to fool it. At the ideal equilibrium the generator matches the data distribution and the discriminator outputs 0.5 for everything, with a loss of $2\ln 2 = 1.386$; a balanced toy run reached 0.511 on real, 0.457 on fake and a loss of 1.343. Training is difficult because the two must stay balanced.

**Read more:** [autoencoders and gans](./08-vision-tasks-and-generative-models/02-autoencoders-and-gans.md)

</details>

<details>
<summary><strong>Q47. What is mode collapse in GANs?</strong> Why is it a problem?</summary>

The generator produces only a few kinds of output instead of the full variety of the data; trained on digits, it might draw only one or two digit types. It's a problem because the model hasn't learned the data distribution, and it is easy to miss because each sample can look realistic. In a toy run where the discriminator learned 100 times faster than the generator, the generator covered only 2 of 8 modes while 93.4% of its samples sat on a real mode. Mode collapse is one reason GAN training is considered unstable; remedies include balanced learning rates, minibatch discrimination, WGAN-GP and spectral normalisation.

**Read more:** [autoencoders and gans](./08-vision-tasks-and-generative-models/02-autoencoders-and-gans.md)

</details>

:::tip[🎯 Interview angle]
**"How would you explain detection, segmentation, autoencoders and GANs in one minute?"** Detection outputs class + score + box per object, judged by IoU and mAP; segmentation labels every pixel, instance segmentation also separates objects. An autoencoder compresses and rebuilds its input through a bottleneck (reconstruction loss), useful for denoising and anomaly detection; a GAN trains a generator against a discriminator and can suffer mode collapse.
:::


## 6. Phase 6: Final Interview Pressure Round

The last round is not about one topic. These questions test maturity: whether you can see a model as a system, reason about shapes, tell knowing a concept from being able to use it under pressure, and plan your own preparation. There's no single correct wording, so the answers below show what a strong answer covers.

<details>
<summary><strong>Q48. A student says: I know Transformers because I know the attention formula. What is missing?</strong> Explain what else is needed.</summary>

A formula is not the model; it's one part of it. Knowing Transformers also means knowing: **tensor shapes** through every layer; the **Q, K, V projections** and where they come from (self- vs cross-attention); the **softmax dimension** and **masks** (padding and causal); **multi-head attention** and why heads are split and concatenated; **positional information**; the **feed-forward** sublayer; **residual connections** and **layer normalisation** (and their pre-/post-norm placement); the **training objectives** (masked language modelling for BERT, next-token prediction for GPT); and **implementation details** such as scaling by $\sqrt{\hyper{d_k}}$, the quadratic cost and the KV cache at inference.

**The strong line from the course:** a formula is not the model; it is only one part of the model.

**Read more:** [transformer architecture figure 1](./06-attention-and-transformers/04-transformer-architecture-figure-1.md) · [multi head attention](./06-attention-and-transformers/03-multi-head-attention.md) · [stacked encoder blocks mini project](./06-attention-and-transformers/06-stacked-encoder-blocks-mini-project.md) · [bert pretraining and fine tuning](./06-attention-and-transformers/05-bert-pretraining-and-fine-tuning.md)

</details>

<details>
<summary><strong>Q49. Why is shape reasoning important in deep learning interviews?</strong> Give examples from this module.</summary>

Because most implementation bugs are shape bugs, and shapes are how you check that a model does what you think. Reasoning about shapes verifies matrix multiplications, batching, sequence handling, model outputs and loss inputs. Examples from the module: embedding input (batch, seq_len) → output (batch, seq_len, emb_dim); attention scores (batch, seq_len, seq_len); LSTM output (batch, seq_len, hidden) with `batch_first=True` and $h_n$ (layers, batch, hidden); detector outputs (batch, queries, classes + 1) and (batch, queries, 4); segmentation logits (batch, classes, H, W).

**The nuance:** interviewers test whether you can read an error such as `mat1 and mat2 shapes cannot be multiplied (32x20 and 10x1)` and immediately say "the layer expects 10 input features but got 20", and whether you notice the bugs that *don't* raise errors, such as broadcasting `(N,)` with `(N, 1)` into `(N, N)` or a default `nn.LSTM` reading the batch as the time axis.

**Read more:** [pytorch fundamentals](./01-pytorch-and-neural-networks/01-pytorch-fundamentals.md) · [full self attention scaled dot product](./06-attention-and-transformers/02-full-self-attention-scaled-dot-product.md) · [lstm and gru gated memory](./05-lstm-and-gru/01-lstm-and-gru-gated-memory.md) · [object detection and segmentation](./08-vision-tasks-and-generative-models/01-object-detection-and-segmentation.md)

</details>

<details>
<summary><strong>Q50. What is the difference between learning a concept and being interview-ready in it?</strong> Give a mature answer.</summary>

Learning a concept means understanding its basic idea. Being interview-ready means you can **explain** it clearly in two or three minutes, **answer follow-ups**, **reason about shapes**, **write and read the code**, **debug** mistakes, **compare** it with alternatives and **apply** it to a practical scenario. For self-attention that means more than the formula: Q, K and V, the shape flow, the softmax dimension, the scaling, why positional information is needed and what it costs.

**The closing message from the course:** if some questions made you uncomfortable, that isn't failure; it means your real interview preparation has started.

</details>

<details>
<summary><strong>Q51. What should students do after this revision lecture to become stronger?</strong> Give a practical preparation plan.</summary>

Practise in four directions. **Explanation:** explain each concept in 2–3 minutes, out loud. **Coding:** implement small PyTorch models and training loops from memory. **Debugging:** deliberately break and fix shape errors, gradient errors and tensor conversions. **Connections:** compare related ideas (RNN vs LSTM, TF-IDF vs embeddings, attention vs self-attention, autoencoder vs GAN). Interview readiness comes from repeated explanation, implementation and debugging.

**The course's suggested assignment:** build one small sentiment classifier, print tensor shapes at every step, explain the training loop in writing, and compute one toy attention matrix by hand. The Mock Interview Lab below does exactly that, with verified outputs.

</details>

:::tip[🎯 Interview angle]
**"What does being interview-ready in a topic mean to you?"** Being able to explain it simply, handle follow-ups, reason through shapes, write and debug the code, compare it with alternatives and apply it to a real problem, not just recite its formula.
:::


## 7. Mock Interview Lab: Coding and Debugging Drills

The revision notes end by naming their own limitation: the questions test explanations, not **coding fluency**, and the suggested follow-up is a hands-on lab of coding and debugging tasks. These are those tasks. Try each one before reading the solution; every output is copied from a real run (PyTorch 2.15, CPU).

:::tip[Run it yourself]
Every block runs on its own after `import math, torch, torch.nn as nn` and `torch.manual_seed(0)`.
:::

### Drill 1. Fix the training loop

The loop below fits $y = 3x + 1$ but trains badly. Find the bug.

```python title="❌ Gradients pile up across steps" showLineNumbers
torch.manual_seed(0); X = torch.randn(64, 1); y = 3 * X + 1
m = nn.Linear(1, 1); opt = torch.optim.SGD(m.parameters(), lr=0.1)
for step in range(50):
    # error-next-line
    loss = nn.functional.mse_loss(m(X), y)      # ❌ no opt.zero_grad(): every backward() adds to .grad
    loss.backward()
    opt.step()
print(round(loss.item(), 6), [round(p.item(), 4) for p in m.parameters()])   # 1.59239 [1.3092, -0.4147]
```

```python title="✅ Clear gradients every step" showLineNumbers
for step in range(50):
    # success-next-line
    opt.zero_grad()                             # ✅ clear, then forward, loss, backward, step
    loss = nn.functional.mse_loss(m(X), y)
    loss.backward()
    opt.step()
# (fresh model, same seed) -> 0.0 [3.0, 1.0]
```

### Drill 2. Read a shape error

```python title="❌ Linear layer fed the wrong feature size" showLineNumbers
# error-next-line
nn.Linear(10, 1)(torch.randn(32, 20))
# RuntimeError: addmm: mat1 and mat2 shapes cannot be multiplied (32x20 and 10x1)
```

The batch is 32 samples with **20** features, but the layer was built for **10** (its weight is used as a 10 × 1 matrix). Fix the layer (`nn.Linear(20, 1)`) or the data, whichever is wrong; never "fix" it by reshaping data until the error disappears.

### Drill 3. Implement softmax safely

```python showLineNumbers
z = torch.tensor([1000., 1001.])
# error-next-line
print(torch.exp(z) / torch.exp(z).sum())       # ❌ tensor([nan, nan])   exp(1000) overflows to inf
s = torch.exp(z - z.max())                     # ✅ subtracting the max changes nothing mathematically
print(s / s.sum(), torch.softmax(z, 0))        # tensor([0.2689, 0.7311]) tensor([0.2689, 0.7311])
```

### Drill 4. Compute attention weights by hand

With $\pred{Q} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$, $\pred{K} = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}$, $\pred{V} = \begin{bmatrix} 10 & 0 \\ 0 & 10 \end{bmatrix}$ and $\hyper{d_k} = 2$: the scores $\pred{Q}\pred{K}^\top / \sqrt{2}$ are $\begin{bmatrix} 0.7071 & 0.7071 \\ 0 & 0.7071 \end{bmatrix}$. Row 1 has equal scores, so its weights are (0.5, 0.5) and its output (5, 5). Row 2 gives weights (0.3302, 0.6698) and output (3.3024, 6.6976).

```python showLineNumbers
Q = torch.tensor([[1., 0.], [0., 1.]]); K = torch.tensor([[1., 0.], [1., 1.]]); V = torch.tensor([[10., 0.], [0., 10.]])
P = torch.softmax(Q @ K.T / math.sqrt(2), dim=-1)   # softmax over the keys
print(P, P @ V)
# tensor([[0.5000, 0.5000], [0.3302, 0.6698]]) tensor([[5.0000, 5.0000], [3.3024, 6.6976]])
```

### Drill 5. Convert a tensor to NumPy

```python showLineNumbers
x = torch.tensor([1.0, 2.0], requires_grad=True)
# error-next-line
x.numpy()           # ❌ RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
# success-next-line
x.detach().cpu().numpy()                       # ✅ array([1., 2.], dtype=float32)
```

### Drill 6. Feed CrossEntropyLoss correctly

`nn.CrossEntropyLoss` takes **raw logits** of shape (N, C) and **class indices** of shape (N,).

```python showLineNumbers
logits = torch.randn(4, 3); t = torch.tensor([0, 2, 1, 2])
print(round(nn.CrossEntropyLoss()(logits, t).item(), 4))       # 1.6286
# error-next-line
nn.CrossEntropyLoss()(logits, t.view(4, 1))   # ❌ RuntimeError: 0D or 1D target tensor expected, multi-target not supported
# error-next-line
nn.CrossEntropyLoss()(torch.softmax(logits, 1), t)   # ❌ runs without error, but applies softmax twice
```

The second bug is the dangerous one: `CrossEntropyLoss` already applies log-softmax, so softmax-ing first squashes the logits and slows learning without any error message.

### Drill 7. Check an LSTM's batch dimension

```python showLineNumbers
x = torch.randn(32, 20, 8)                     # (batch=32, seq_len=20, features=8)
out, (h, c) = nn.LSTM(8, 16)(x)
# error-next-line
print(out.shape, h.shape)                      # ❌ (32, 20, 16) (1, 20, 16)   read as seq_len=32, batch=20
out, (h, c) = nn.LSTM(8, 16, batch_first=True)(x)
# success-next-line
print(out.shape, h.shape)                      # ✅ (32, 20, 16) (1, 32, 16)   one final state per sentence
```

No error is raised in either case; only checking the final hidden state's shape reveals the bug.

### Drill 8. A mini text classifier, shapes printed at every step

Eight labelled phrases, a 13-word vocabulary, an embedding layer, a masked average and a linear layer.

```python title="tiny_classifier.py" showLineNumbers
texts = ["good movie", "great acting", "loved it", "not good", "bad plot", "boring and bad", "really great film", "not great"]
labels = torch.tensor([1, 1, 1, 0, 0, 0, 1, 0])
vocab = {w: i + 1 for i, w in enumerate(sorted({w for t in texts for w in t.split()}))}   # 0 = padding
ids = [[vocab[w] for w in t.split()] for t in texts]
L = max(map(len, ids))
X = torch.tensor([s + [0] * (L - len(s)) for s in ids])
print(len(vocab), X.shape)                                # 13 torch.Size([8, 3])

class TinyClassifier(nn.Module):
    def __init__(self, vocab_size, dim=8):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.out = nn.Linear(dim, 2)
    def forward(self, x):
        e = self.emb(x)                                   # (batch, seq_len, dim)
        mask = (x != 0).unsqueeze(-1).float()             # (batch, seq_len, 1)
        # highlight-next-line
        mean = (e * mask).sum(1) / mask.sum(1)            # (batch, dim): average of real tokens only
        return self.out(mean)                             # (batch, 2) logits

torch.manual_seed(0)
model = TinyClassifier(len(vocab) + 1)
opt = torch.optim.Adam(model.parameters(), lr=0.05)
for step in range(100):
    opt.zero_grad()
    logits = model(X)
    loss = nn.functional.cross_entropy(logits, labels)
    loss.backward()
    opt.step()
print(model.emb(X).shape, logits.shape, round(loss.item(), 4))   # torch.Size([8, 3, 8]) torch.Size([8, 2]) 0.0
print((logits.argmax(1) == labels).float().mean().item())        # 1.0
model.eval()
with torch.no_grad():
    # error-next-line
    print(torch.softmax(model(torch.tensor([[vocab["not"], vocab["bad"]]])), 1))   # ❌ [[1., 0.]]: "not bad" called negative
```

It fits all eight training phrases (loss rounds to 0.0, accuracy 1.0) and then calls "not bad" negative with probability 1.0. Averaging word vectors throws away order and composition: *not* and *bad* both learned negative meanings, and their average can't express that together they're positive. That is Q4 and Q8 in one result, and the reason sequence models and Transformers exist.


## ⚙️ Going Deeper (What Interviewers Probe Next)

**Preprocessing for Transformers.** Subword tokenizers (WordPiece for BERT, BPE for GPT) split rare words into known pieces, so lowercasing, stemming and stopword removal are usually skipped; the tokenizer used for fine-tuning must be the one the model was pretrained with.

**Autograd details.** Only leaf tensors with `requires_grad=True` keep `.grad`; intermediate results need `retain_grad()`. Calling `backward()` twice on the same graph fails unless `retain_graph=True`. In-place operations on tensors autograd saved for the backward pass raise errors at `backward()` time.

**Gradient clipping.** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)` rescales the whole gradient when its norm exceeds `max_norm`; it is standard for RNNs (exploding gradients) and Transformers.

**Masks in attention.** A **padding mask** sets the scores of padding keys to −∞ before softmax so they get zero weight; a **causal mask** does the same for future positions in a decoder. Both are applied to the (B, T, T) score tensor.

**KV cache.** During generation a decoder stores the keys and values of previous tokens, so each new token costs one row of attention instead of recomputing the whole T × T matrix.

**Positional encodings.** Sinusoidal encodings need no training and extend to any length; learned embeddings are limited to the trained length; rotary embeddings rotate Q and K so their dot product depends on relative distance.

## 🚨 Interview Gotchas & Edge Cases

:::warning[The Most Common Question]
**"What's the difference between `loss.backward()` and `optimizer.step()`, and why do we call `zero_grad()`?"** `backward()` computes gradients into `.grad`; `step()` updates the weights; `zero_grad()` clears `.grad` because PyTorch accumulates gradients across `backward()` calls. Forgetting it in Drill 1 left the model at weight 1.31 instead of 3.0.
:::

- **Stopword removal can flip sentiment** ("not good" → "good").
- **BoW and TF-IDF are order-blind**: "dog bites man" and "man bites dog" are identical vectors.
- **One-hot vectors are orthogonal**: king · queen = 0.
- **`model.eval()` doesn't disable gradients**; `torch.no_grad()` does.
- **`.numpy()` fails on tensors that require grad** (and on GPU tensors); use `.detach().cpu().numpy()`.
- **Broadcasting `(N,)` with `(N, 1)` silently gives `(N, N)`.**
- **`nn.LSTM` is not batch-first by default**; check `h_n`'s shape.
- **`CrossEntropyLoss` wants raw logits and class indices**; softmax first is a silent bug.
- **Softmax over the wrong axis runs without error**; check that rows sum to 1.
- **Unscaled attention scores saturate softmax** (max weight 1.0 at $\hyper{d_k}$ = 512).
- **Self-attention is permutation-equivariant**; it needs positional information.
- **Attention heatmaps aren't explanations.**
- **Self-attention memory is quadratic**: 64 MiB per head per sequence at 4,096 tokens in float32.
- **Mode collapse hides behind realistic samples**; check diversity.

## ⚡ Essential Interview Recall

A one-screen revision list. Every point is explained in the answers above.

**NLP basics**
- Preprocessing is task-dependent; keep negations for sentiment; same pipeline at train and inference time.
- Tokenisation → tokens; vectorisation → numbers. Stemming: *studies* → *studi*; lemmatisation: → *study*.
- BoW = counts; TF-IDF = counts × rarity; both ignore order. Embeddings: dense, similar words close; Transformers: contextual.

**PyTorch**
- Tensor = shape + dtype + device + autograd. `requires_grad` records operations; $x^2$ at 2 → grad 4.
- zero_grad → forward → loss → backward → step; `backward()` computes, `step()` updates; gradients accumulate.
- `__init__` = layers, `forward` = data flow. `.detach().cpu().numpy()`; `.item()` for one-element tensors.
- Broadcasting aligns from the right; `nn.Embedding(V, d)`: (B, T) → (B, T, d), V × d parameters.
- `eval()` = layer behaviour; `no_grad()` = no graph.

**RNN and LSTM**
- $\pred{h_t} = \tanh(\param{W_x}\data{x_t} + \param{W_h}\pred{h_{t-1}} + \param{b})$; vanishing gradients from repeated products.
- $\pred{C_t} = \pred{f_t} \odot \pred{C_{t-1}} + \pred{i_t} \odot \pred{\tilde C_t}$; $\pred{h_t} = \pred{o_t} \odot \tanh(\pred{C_t})$; sigmoid gates are soft dials.
- Generation: predict, sample, append, repeat; temperature $\operatorname{softmax}(\pred{z}/\hyper{T})$ sharpens (low) or flattens (high).

**Attention**
- Attention fixes the fixed-vector bottleneck; Q = what I look for, K = what I offer, V = what I pass on.
- $\operatorname{softmax}(\pred{Q}\pred{K}^\top / \sqrt{\hyper{d_k}})\pred{V}$; scores (B, T, T); softmax over keys.
- Cross-attention: Q from decoder, K/V from encoder. Positional encodings give order. Parallel but $O(T^2)$.

**Vision and generative**
- Classification / detection / segmentation = image / object / pixel. Boxes: xyxy or cxcywh. Semantic vs instance.
- Autoencoder: encoder + decoder + bottleneck; reconstruction loss; anomaly detection by error.
- GAN: generator vs discriminator; balance at D ≈ 0.5; mode collapse = low diversity.

**Interview readiness**
- Explain, follow up, shapes, code, debug, compare, apply. Practise out loud, implement, break and fix.

## 📐 Formula Sheet

| Concept | Formula | In plain words |
|---|---|---|
| RNN hidden state | $\pred{h_t} = \tanh(\param{W_x}\data{x_t} + \param{W_h}\pred{h_{t-1}} + \param{b})$ | Mix the new input with the old summary |
| Forget gate | $\pred{f_t} = \sigma(\param{W_f}[\pred{h_{t-1}}, \data{x_t}] + \param{b_f})$ | How much old memory to keep |
| Input gate | $\pred{i_t} = \sigma(\param{W_i}[\pred{h_{t-1}}, \data{x_t}] + \param{b_i})$ | How much new content to write |
| Candidate memory | $\pred{\tilde C_t} = \tanh(\param{W_C}[\pred{h_{t-1}}, \data{x_t}] + \param{b_C})$ | The proposed new content |
| Cell update | $\pred{C_t} = \pred{f_t} \odot \pred{C_{t-1}} + \pred{i_t} \odot \pred{\tilde C_t}$ | Keep some old, add some new |
| Output gate, hidden | $\pred{o_t} = \sigma(\param{W_o}[\pred{h_{t-1}}, \data{x_t}] + \param{b_o})$, $\ \pred{h_t} = \pred{o_t} \odot \tanh(\pred{C_t})$ | Expose a filtered view of memory |
| Temperature | $\pred{p_i} = \operatorname{softmax}(\pred{z_i} / \hyper{T})$ | Low T sharp, high T flat |
| Q, K, V | $\pred{Q} = \data{X}\param{W_Q}$, $\pred{K} = \data{X}\param{W_K}$, $\pred{V} = \data{X}\param{W_V}$ | Three learned views of each token |
| Scaled dot-product attention | $\operatorname{softmax}(\pred{Q}\pred{K}^\top / \sqrt{\hyper{d_k}})\, \pred{V}$ | Compare, normalise, blend |
| SGD update | $\param{w} \leftarrow \param{w} - \hyper{\eta}\, \partial \loss{L} / \partial \param{w}$ | What `optimizer.step()` does |
| Reconstruction loss | $\lVert \data{x} - \pred{\hat x} \rVert^2$ | Autoencoder training signal |
| GAN objective | $\min_{\param{G}} \max_{\param{D}} \mathbb{E}[\log \pred{D}(\data{x})] + \mathbb{E}[\log(1 - \pred{D}(\param{G}(\data{z})))]$ | D catches fakes, G fools D |
