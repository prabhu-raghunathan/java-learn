---
title: "Cheat Sheet: LSTM, GRU and Text Generation"
sidebar_position: 5
---

:::tip[What this sheet covers]
Why vanilla RNN memory is fragile, the LSTM cell gate by gate, why its cell path keeps gradients alive (with measurements), GRU, choosing between them, and character-level text generation with greedy decoding and temperature. Full notes: [LSTM and GRU: Gated Memory](../../03-ai-ml/03-deep-learning/05-lstm-and-gru/01-lstm-and-gru-gated-memory.md) · [LSTM in Practice: Text Generation and Temperature](../../03-ai-ml/03-deep-learning/05-lstm-and-gru/02-lstm-text-generation-temperature-sampling.md).
:::

![The LSTM cell](/img/deep-learning/diagrams/lstm-cell.svg)

## 🎤 Answer These Out Loud

<details>
<summary><strong>How does an LSTM solve the vanishing gradient problem?</strong></summary>

A vanilla RNN multiplies the gradient by $\text{diag}(1 - \tanh^2(a_t))W_h$ every step back. An LSTM adds a **cell state** updated **additively**, $c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t$, so along that path $\partial c_t/\partial c_{t-1} \approx \text{diag}(f_t)$: no weight matrix, no tanh slope. Over $k$ steps the gradient is scaled by the product of forget gates, which the network can keep near 1. Caveats: at default init $f \approx 0.5$, so set the **forget bias to 1+**; very long ranges still fade; exploding gradients still need clipping. "Mitigates", not "solves".
</details>

<details>
<summary><strong>Explain each LSTM gate.</strong></summary>

All gates are sigmoid layers on $x_t$ and $h_{t-1}$, giving one soft decision per memory dimension. **Forget** $f_t$: fraction of old memory to keep (near 1 keeps). **Input** $i_t$: how much new content to write. **Candidate** $\tilde c_t$ (tanh, signed): what to write ("the draft"). **Output** $o_t$: how much of $\tanh(c_t)$ to reveal as $h_t$. The input gate and candidate split *how much* from *what*.
</details>

<details>
<summary><strong>Cell state vs hidden state?</strong></summary>

$c_t$: internal long-term memory, additive, unbounded, goes only to the next step's cell update. $h_t = o_t \odot \tanh(c_t)$: bounded, exposed view used for outputs and fed to all four gates next step. A dimension can be stored but hidden ($o = 0.002$ → $h = 0.000$ while $c = 0.205$ in the notebook's cell). In PyTorch: `out, (h_n, c_n) = lstm(x)`.
</details>

<details>
<summary><strong>Why tanh for the candidate and sigmoid for the gates? Why tanh on $c_t$?</strong></summary>

A gate is a **fraction** (0 to 1); content needs a **sign** (memory must move up or down), so the candidate uses tanh. The cell state is unbounded (additions accumulate), so $\tanh(c_t)$ bounds what the next layer and the gates see.
</details>

<details>
<summary><strong>GRU vs LSTM?</strong></summary>

GRU: no separate cell, two gates. Update $z_t$: $h_t = z_t \odot h_{t-1} + (1 - z_t)\odot\tilde h_t$ (notebook/PyTorch convention; some sources swap $z$ and $1 - z$). Reset $r_t$ filters $h_{t-1}$ before the candidate $\tilde h_t = \tanh(W_h x_t + U_h(r_t \odot h_{t-1}) + b_h)$ (PyTorch applies $r_t$ after the matmul). 3 weight sets vs 4, faster, usually similar accuracy; LSTM gives finer memory control. Try both.
</details>

<details>
<summary><strong>What can't LSTMs fix?</strong></summary>

They are still **sequential** (no parallelism over time), squeeze the past into a **fixed-size state**, reach distant tokens only through a **long chain** of updates, and still struggle with very long sequences. Self-attention addresses all four, which is why Transformers replaced them for most NLP; LSTMs remain useful for streaming/on-device (constant memory per step).
</details>

<details>
<summary><strong>How do you train a character-level language model?</strong></summary>

Sorted vocabulary → ids → sliding windows of length $T$; **target = input shifted by one** ("hello" → "ello "). Model: Embedding → LSTM → Linear, logits $(B, T, V)$; flatten to $(BT, V)$ with targets $(BT)$ for `CrossEntropyLoss` on raw logits. Report perplexity $e^{L}$. The notebook's 2,935-parameter model: loss 2.514 → 0.117, 98.7% next-char accuracy at positions 1–19 but 37% at position 0 (one character of context: a newline is followed by d/l/g/s equally often).
</details>

<details>
<summary><strong>How does generation work, and what does temperature do?</strong></summary>

Warm up on the seed once, then loop: take the **last** position's logits, convert to probabilities, choose a character, append it, and feed **only that character** back with the carried $(h, c)$: $O(n)$ instead of $O(n^2)$ re-reading. Greedy (argmax) is deterministic and repetitive (it replays the corpus). Temperature: $p_i \propto e^{z_i/T}$, $T < 1$ sharper, $T > 1$ flatter, $T \to 0$ greedy, $T \to \infty$ uniform, ranking unchanged, **inference only**. Production adds top-k/top-p and a repetition penalty.
</details>

<details>
<summary><strong>What was the bug in the notebook's generator?</strong></summary>

After the warm-up it fed the last seed character **again**, so the first prediction was conditioned on text that never existed ("deep" followed by two spaces: p(l) 0.997 → 0.956, p(o) 0.03). The warm-up's own last logits already hold the prediction for the next character; use them, then feed only new characters. The same off-by-one appears with Transformer KV-cache prefill.
</details>

## 📐 Formulas

| Concept | Formula | In plain words |
|---|---|---|
| Vanilla RNN Jacobian | $\partial\pred{h_t}/\partial\pred{h_{t-1}} = \text{diag}(1 - \tanh^2(\pred{a_t}))\param{W_h}$ | The factor that vanishes |
| Forget gate | $\pred{f_t} = \sigma(\param{W_f}\data{x_t} + \param{U_f}\pred{h_{t-1}} + \param{b_f})$ | How much old memory to keep |
| Input gate | $\pred{i_t} = \sigma(\param{W_i}\data{x_t} + \param{U_i}\pred{h_{t-1}} + \param{b_i})$ | How much new content to write |
| Candidate | $\pred{\tilde c_t} = \tanh(\param{W_c}\data{x_t} + \param{U_c}\pred{h_{t-1}} + \param{b_c})$ | What to write |
| Cell update | $\pred{c_t} = \pred{f_t}\odot\pred{c_{t-1}} + \pred{i_t}\odot\pred{\tilde c_t}$ | Keep some old, add some new |
| Output gate / hidden | $\pred{o_t} = \sigma(\ldots)$, $\pred{h_t} = \pred{o_t}\odot\tanh(\pred{c_t})$ | Reveal a bounded view |
| Cell-path gradient | $\partial\pred{c_t}/\partial\pred{c_{t-k}} \approx \text{diag}(\prod_j\pred{f_j})$ | Scaled only by forget gates |
| GRU gates | $\pred{z_t}, \pred{r_t} = \sigma(\param{W}\data{x_t} + \param{U}\pred{h_{t-1}} + \param{b})$ | Keep vs filter the past |
| GRU candidate | $\pred{\tilde h_t} = \tanh(\param{W_h}\data{x_t} + \param{U_h}(\pred{r_t}\odot\pred{h_{t-1}}) + \param{b_h})$ | Proposal from input and filtered past |
| GRU update | $\pred{h_t} = \pred{z_t}\odot\pred{h_{t-1}} + (1 - \pred{z_t})\odot\pred{\tilde h_t}$ | One dial between old and new |
| Parameters (PyTorch) | RNN $HD + H^2 + 2H$; GRU ×3; LSTM ×4 | One weight set per gate/candidate |
| Language model | $\pred{p_t} = \text{softmax}(\pred{z_t}) \approx P(\data{x_{t+1}}\mid\data{x_{\le t}})$ | Next-character distribution |
| Sequence cross-entropy | $\loss{L} = -\frac{1}{BT}\sum_{b,t}\log\pred{p_{b,t}}[\data{y_{b,t}}]$ | Average surprise |
| Perplexity | $e^{\loss{L}}$ (uniform over $V$ = $V$) | Effective number of choices |
| Temperature | $\pred{p_i}(\hyper{T}) = e^{\pred{z_i}/\hyper{T}}/\sum_j e^{\pred{z_j}/\hyper{T}}$ | Below 1 sharper, above 1 flatter |

## 🔢 Numbers to Quote

| Fact | Number |
|---|---|
| tanh slope | 0.420 at 1, 0.0707 at 2, 0.00987 at 3 |
| Repeated factors | $0.8^{30} = 0.00124$, $1.2^{30} = 237.4$ |
| TF-IDF cosine | "good movie" vs "not good movie" 0.744; "man bites dog" vs "dog bites man" 1.0 |
| Notebook cell demo | $c_{t-1} = [0.9, 0.2, -0.8, 0.5]$ → $c_t = [0.9, 0.32, 0.24, -0.6]$ (kept / blended / rewritten / replaced) |
| Gradient 49 steps back (untrained, T = 50) | RNN $3.2 \times 10^{-14}$; GRU $1.1 \times 10^{-11}$; LSTM $1.8 \times 10^{-11}$; forget bias +1 $1.6 \times 10^{-5}$; +3 $5.3 \times 10^{-2}$ |
| Mean forget gate | default init ≈ 0.5; bias +3 ≈ 0.95 |
| Non-diagonal part of $\partial c_t/\partial c_{t-1}$ | 23% (f ≈ 0.49), 12% (f ≈ 0.95) |
| Parameters, $D = 300$, $H = 512$ | 416,768 / 1,250,304 / 1,667,072 (RNN / GRU / LSTM) |
| Char model | 23 chars, 239-char corpus, 219 windows of 20, 2,935 params (368 + 2,176 + 391) |
| Training | loss 2.5136 → 0.1169 (perplexity 12.35 → 1.12); 55 batches per epoch |
| Position 0 vs rest | 37.0% vs 98.7% accuracy; loss 1.555 vs 0.031 |
| Temperature, logits [2.5, 1.2, 0.5, −0.3] | top p 0.912 (0.5), 0.681 (1), 0.468 (2), 0.290 (10) |
| Invented words | 0.2% (T 0.5), 2.2% (1.0), 21% (1.5), 55% (2.0), 91% (3.0) |

## 📏 Shapes (`batch_first=True`)

| Stage | Shape |
|---|---|
| Input ids | $(B, T)$ |
| Embedding | $(B, T, E)$ |
| LSTM `out` | $(B, T, H)$ |
| `h_n`, `c_n` | $(\text{layers}\times\text{dirs}, B, H)$, whatever `batch_first` says |
| Logits | $(B, T, V)$ → flatten $(BT, V)$, targets $(BT)$ |
| PyTorch LSTM weights | `weight_ih_l0` $(4H, D)$, `weight_hh_l0` $(4H, H)$, gate order **i, f, g, o**; GRU order **r, z, n** |

## ⚖️ RNN vs LSTM vs GRU

| | Vanilla RNN | LSTM | GRU |
|---|---|---|---|
| State | $h_t$ | $h_t$ and $c_t$ | $h_t$ |
| Gates | None | Forget, input, output | Update, reset |
| Memory update | Full rewrite through tanh | Additive: $f\odot c + i\odot\tilde c$ | Interpolation: $z\odot h + (1-z)\odot\tilde h$ |
| Parameters | 1× | 4× | 3× |
| Long dependencies | Weak | Strong (relative) | Strong (relative) |
| Speed | Fastest | Slowest | In between |

## 💻 Code You Should Be Able to Write

```python title="forget_gate_bias.py" showLineNumbers
import torch.nn as nn

H = 32
lstm = nn.LSTM(8, H, batch_first=True)
# highlight-next-line
lstm.bias_ih_l0.data[H:2 * H] += 1.0      # forget-gate slice (order i, f, g, o): start out remembering
```

```python title="generate.py" showLineNumbers
def generate_text(model, start_text, length=100, temperature=1.0, greedy=False):
    input_ids = torch.tensor([[char2idx[ch] for ch in start_text]])
    generated = start_text
    with torch.no_grad():
        logits, hidden = model(input_ids)                   # warm-up: reads the whole seed once
        for _ in range(length):
            # highlight-next-line
            last = logits[0, -1]                            # the prediction AFTER the last seed char
            if greedy:
                next_idx = last.argmax().item()
            else:
                next_idx = torch.multinomial(F.softmax(last / temperature, dim=0), 1).item()
            generated += idx2char[next_idx]
            logits, hidden = model(torch.tensor([[next_idx]]), hidden)   # feed only the new char
    return generated
```

The second block is the corrected generator from the notes (it assumes the notebook's `model`, `char2idx`, `idx2char`, `torch` and `F` are defined).

## 🚨 Traps

- The forget gate is misnamed: $f \approx 1$ **keeps**.
- GRU update-gate convention varies between sources: say which one you use.
- Setting the "forget bias" on the wrong slice biases another gate (PyTorch order i, f, g, o).
- PyTorch's two bias vectors make counts $4H$ (LSTM) / $3H$ (GRU) larger than textbook formulas.
- For padded batches `out[:, -1]` is a pad position: pack or index by length.
- The cell state is unbounded; don't feed $c_t$ where $(-1, 1)$ is expected.
- `dropout=` in `nn.LSTM` only works with `num_layers > 1`.
- Bidirectional LSTMs can't generate (they need the future).
- `CrossEntropyLoss` wants $(N, C)$ or $(B, C, T)$: never pass $(B, T, V)$ as is, never softmax first.
- Position 0 of each window has no context when every window starts from a zero state: stateful training (carry and `detach()` the state) or longer windows.
- Divide the epoch loss by the number of batches, not `num_samples / batch_size`.
- 98.7% training accuracy on a memorized four-sentence corpus says nothing about generalization.
- Temperature 0 is a division by zero: use `argmax` for greedy decoding; seed `torch.multinomial` for repeatable samples.
- Teacher forcing in training vs own predictions at generation = **exposure bias**.
