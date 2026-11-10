---
title: "Cheat Sheet: LLM APIs and Prompt Engineering"
sidebar_position: 1
---

:::tip[What this sheet covers]
Calling LLM APIs and engineering prompts on one screen: the anatomy of a call, messages and chat templates, tokens, sampling knobs, latency and streaming, cost and prompt caching, the prompt-engineering ladder, chain of thought, structured outputs, and robust SDK use. Every number comes from the runs on the full page with Qwen2.5-0.5B-Instruct on a CPU. Full notes: [Working with LLM APIs and Prompt Engineering](../../03-ai-ml/04-nlp-llms-and-projects/01-llm-apps-and-agents/01-llm-apis-and-prompt-engineering.md).
:::

## 🎤 Answer These Out Loud

<details>
<summary><strong>What happens when you call an LLM API?</strong></summary>

A stateless HTTPS POST: model name, role-tagged messages and settings go in. The provider renders the messages with the model's chat template, prefills the prompt in parallel, then decodes one token at a time until a stop condition. It returns the text, a usage block (input and output tokens) and a stop reason. Because it's stateless, the client re-sends the whole history every turn: 35 prompt tokens on turn 1 became 78 on turn 2 in our run.
</details>

<details>
<summary><strong>What is a chat template and why can it fail silently?</strong></summary>

A model-specific recipe that serialises the message list into one token string with special tokens marking roles and turn ends (Qwen: `<|im_start|>role ... <|im_end|>`), ending with a generation prompt that cues the reply. The model was fine-tuned on that exact format. A wrong template or a missing generation prompt raises no error; the output just gets worse. Print the rendered prompt once (`tokenize=False`).
</details>

<details>
<summary><strong>Why does the same text cost different amounts on different models?</strong></summary>

Billing and limits are in tokens, and each model family has its own tokenizer. The same Hindi sentence was 64 tokens for Qwen2.5 and 23 for OpenAI's o200k; a phone number was 10 tokens in Qwen (one per digit) and 4 in o200k (digits in threes). Count with the target model's own tokenizer, never a word count or an unofficial port.
</details>

<details>
<summary><strong>Explain temperature, top-k, top-p and min-p.</strong></summary>

Temperature divides logits before softmax: below 1 sharper, above 1 flatter; the ranking never changes. Top-k keeps k tokens; top-p keeps the smallest set reaching probability p, so it adapts (1 token for a yes/no question, 268 for an open naming question at p = 0.9); min-p keeps tokens above a fraction of the top probability (1 vs 7 at 0.1). Use greedy for extraction and code, about 0.7 with top-p 0.9 for assistants. Newer reasoning models restrict these knobs: Claude API 2026-10-21 removed all three.
</details>

<details>
<summary><strong>How do you reduce latency?</strong></summary>

Total = TTFT + TPOT × (n − 1). TTFT grows with prompt length (prefill: 213 ms at 28 tokens, 6.3 s at 2,227), so trim prompts and cache the static prefix (6,163 → 270 ms). The decode part grows with output length, so cap and shorten answers, use a faster model, and run independent calls in parallel. Stream, so the user waits for TTFT instead of the whole answer (about 24× sooner in our run).
</details>

<details>
<summary><strong>How do you cut cost?</strong></summary>

Measure tokens per request and split input from output, because output is priced higher (44% of the bill at 300 of 1,800 tokens in the example). Shorten outputs (100 instead of 300 tokens: 29.6% less), send only relevant context, make the static prefix cacheable (1,200 cached tokens: 40.0% less), route easy requests to a cheaper model, and use batch APIs for offline work.
</details>

<details>
<summary><strong>How do you do prompt engineering properly?</strong></summary>

Treat prompts as versioned code and measure them on a labelled set, changing one thing at a time. On 24 support tickets with a 0.5B model: vague prompt 0 valid labels; clear instructions 12 correct; few-shot 14 (for twice the prompt tokens); one-line definitions per label 17; definitions plus scoring only the 4 allowed labels 20, all valid.
</details>

<details>
<summary><strong>Chain of thought: what does it buy and cost?</strong></summary>

Asking for step-by-step reasoning before the answer took 10 arithmetic word problems from 1 to 9 correct, but output grew from 8.4 to 193.3 tokens per question (about 23× latency and cost). The model used the requested "Answer:" line in only 2 of 10 replies (it wrote `\boxed{}`), so parse robustly. Self-consistency majority-votes several sampled chains at roughly K times the cost. Reasoning models do this internally and expose an effort setting.
</details>

<details>
<summary><strong>How do you get reliable JSON?</strong></summary>

Pydantic schema plus the provider's strict mode (OpenAI `text_format`, Anthropic `output_config.format` via `messages.parse`, Gemini `response_schema`) or constrained decoding on open models. Check the stop reason, validate, repair predictable wrappers. Prompt only: 20/24 valid (2 code fences, 2 invented enums); strip fences: 22/24; constrained: 24/24 valid. But valid ≠ correct: constrained got 10 to 13 of 24 intents right vs 14 unconstrained.
</details>

<details>
<summary><strong>How do you call an LLM API robustly?</strong></summary>

Official SDK with your own timeout (the default read timeout is 600 s); its retries (2 by default, 0.5 s doubling to 8 s with jitter, on 408/409/429/5xx, honouring Retry-After); no retries for 400/401/403. Budget tokens per minute as well as requests. Check stop reasons, validate outputs, log usage, make side effects idempotent (a retry is a new generation), fall back to another model or provider.
</details>

## 📐 Formulas

| Concept | Formula |
|---|---|
| Autoregressive generation | $P(\pred{y} \mid \data{x}) = \prod_{t} P_{\param{\theta}}(\pred{y_t} \mid \data{x}, \pred{y_{\lt t}})$ |
| Conversation input tokens | $\data{N_{in}}(T) = T\hyper{S} + \hyper{m}\,T(T+1)/2$ (10 turns, $\hyper{S}$ = 500, $\hyper{m}$ = 100: 10,500) |
| Context budget | $\data{n_{in}} + \pred{n_{out}} \le \hyper{L_{ctx}}$ |
| Temperature | $\pred{p_i} = e^{\pred{z_i}/\hyper{T}} / \sum_j e^{\pred{z_j}/\hyper{T}}$ (logits [1, 2]: T 1 → 0.7311, T 0.5 → 0.8808) |
| Top-p | smallest top set with $\sum \pred{p_i} \ge \hyper{p}$, renormalised |
| Min-p | keep $\pred{p_i} \ge \hyper{p_{min}} \max_j \pred{p_j}$ |
| Latency | $\pred{T_{total}} = \hyper{\text{TTFT}} + \hyper{\text{TPOT}}(\pred{n} - 1)$ (0.213 + 59 × 0.0824 ≈ 5.08 s) |
| Cost | $\loss{C} = [(\data{n_{in}} - \data{n_{cache}})\hyper{r_{in}} + \data{n_{cache}}\hyper{r_{cache}} + \pred{n_{out}}\hyper{r_{out}}]/10^6$ |
| Self-consistency | $\pred{\hat a} = \arg\max_a \sum_k \mathbb{1}[\pred{a_k} = a]$ |
| Constrained decoding | $\pred{\tilde z_i} = -\infty$ for tokens the schema forbids |
| Backoff | $\hyper{d_k} = \min(0.5 \cdot 2^k, 8)(1 - 0.25\,\data{u})$ seconds |

## 🔢 Numbers to Quote

| What | Number |
|---|---|
| Rendered 2-message request (Qwen2.5 template) | 35 tokens; turn 2 with history: 78 |
| Hindi sentence: Qwen vs o200k | 64 vs 23 tokens |
| JSON vs YAML, same data (Qwen) | 19 vs 13 tokens |
| First token, yes/no question | Yes 0.9852, No 0.0141 → top-p 0.9 keeps 1 |
| First token, open question | Green 0.1668 → top-p 0.9 keeps 268 (T 0.5: 4, T 1.5: 6,574) |
| Qwen2.5 default generation config | do_sample True, T 0.7, top-p 0.8, top-k 20, repetition penalty 1.1 |
| TTFT / TPOT (28-token prompt, CPU) | 213 ms / 82.4 ms; total 5.08 s for 60 tokens |
| TTFT for a 2,227-token prompt | 6,311 ms; with a cached 2,218-token prefix 270 ms |
| KV cache, Qwen2.5-0.5B, float32 | 24,576 bytes per token |
| Cost example ($1 / $4 / $0.10 per 1M) | $0.0027 per request, $27 per day at 10k requests |
| Prompt ladder (24 tickets) | 0 valid → 12 → 14 → 17 → 20 correct |
| Chain of thought (10 problems) | 1 → 9 correct; 8.4 → 193.3 output tokens |
| JSON (24 tickets) | prompt 20 valid, fence repair 22, constrained 24 (10 to 13 correct) |
| SDK retry defaults | 2 retries, 0.5 s doubling, 8 s cap, 0–25% jitter |

## ⚖️ Quick Comparisons

| | OpenAI Responses | Anthropic Messages | Gemini |
|---|---|---|---|
| Call | `client.responses.create` | `client.messages.create` | `client.models.generate_content` |
| Instructions | `instructions=` | `system=` | `config.system_instruction` |
| Output cap | `max_output_tokens` | `max_tokens` (required) | `config.max_output_tokens` |
| Structured output | `responses.parse(text_format=Model)` | `messages.parse(output_format=Model)` | `response_schema=Model` |
| Streaming events | `response.output_text.delta` | `content_block_delta` / `text_delta` | `generate_content_stream` chunks |

| Technique | Fixes | Costs |
|---|---|---|
| Clear instructions + allowed outputs | Format, ambiguity | A few prompt tokens |
| Few-shot examples | Unusual formats, edge cases | Prompt length (54 → 105 tokens) |
| Label definitions | Meaning of categories | Prompt length; the model may echo definition words |
| Chain of thought | Multi-step reasoning | Output tokens (about 23×) |
| Constrained decoding / strict schema | Invalid structure | Can cost accuracy on small models |
| Validate and retry | Occasional slips (strong models) | Extra calls; can make weak models worse |

| Decoding | Behaviour | Use |
|---|---|---|
| Greedy / T = 0 | Same top token every time | Extraction, classification, code |
| T 0.7 + top-p 0.9 | Varied but sensible | Assistants, summaries |
| T above 1, no top-p | Long tail: other scripts, nonsense | Almost never |

## 💻 Code You Should Be Able to Write

```python title="Render a chat template and count its tokens" showLineNumbers
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
messages = [
    {"role": "system", "content": "You are a support assistant for an Indian telecom. Answer in one sentence."},
    {"role": "user", "content": "My data stopped working after recharge."},
]
ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_dict=True)["input_ids"]
print(len(ids))                                     # 35
```

```python title="Validate structured output with Pydantic" showLineNumbers
from typing import Literal
from pydantic import BaseModel, Field, ValidationError

class Ticket(BaseModel):
    intent: Literal["billing", "network", "plan_change", "account"]
    urgency: Literal["low", "medium", "high"]
    refund_requested: bool
    summary: str = Field(max_length=80)

try:
    Ticket.model_validate_json('{"intent": "refund", "urgency": "high", "refund_requested": true, "summary": "x"}')
except ValidationError as e:
    print(e.errors()[0]["type"], e.errors()[0]["loc"])   # literal_error ('intent',)
```

```python title="Cost and latency in two lines" showLineNumbers
cost = lambda n_in, n_out, n_cached=0: ((n_in - n_cached) * 1.00 + n_cached * 0.10 + n_out * 4.00) / 1e6
latency = lambda ttft, tpot, n: ttft + tpot * (n - 1)
print(round(cost(1500, 300), 5), round(cost(1500, 300, 1200), 5), round(latency(0.213, 0.0824, 60), 3))   # 0.0027 0.00162 5.075
```

```python title="Backoff with jitter" showLineNumbers
import random
random.seed(7)
delay = lambda k: min(0.5 * 2 ** k, 8.0) * (1 - 0.25 * random.random())
print([round(delay(k), 2) for k in range(2)], [min(0.5 * 2 ** k, 8.0) for k in range(6)])   # [0.46, 0.96] [0.5, 1.0, 2.0, 4.0, 8.0, 8.0]
```

## 🚨 Traps

- The model doesn't remember the conversation; your code re-sends it, so input grows quadratically with turns.
- A wrong chat template raises no error. Qwen's template also inserts its own system prompt if you send none.
- `generate()` in Hugging Face follows the model's `generation_config`: Qwen2.5 samples by default. Pass `do_sample=False`.
- Temperature 0 isn't bit-exact on hosted GPUs; newer models may reject temperature entirely (Claude API 2026-10-21, GPT-5 reasoning models).
- Total latency uses n − 1 decode steps; streaming improves perceived latency, not total time.
- A `max_tokens` cut-off is a normal response with stop reason `max_tokens` / `length`, and truncated JSON is invalid.
- Prompt caching needs a byte-identical prefix: static content first, timestamps and user data last.
- Models don't always follow format instructions; parse robustly or constrain.
- Structured outputs guarantee shape, not truth.
- A retry is a new generation with new charges; make side effects idempotent and don't retry 4xx bugs.
- The system prompt is neither secret nor a security boundary.
