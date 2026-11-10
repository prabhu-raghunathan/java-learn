---
title: "AI Engineering: The Last Hour Before the Interview"
sidebar_position: 0
---

:::tip[How to use this sheet]
Read it top to bottom in the hour before an AI Engineer interview. **Answer each question out loud first**, then open it to check yourself. Everything here summarises the full notes, with the same numbers and notation, and grows as the AI Engineering track grows. Topic sheets: [LLM APIs and Prompt Engineering](./01-llm-apis-and-prompt-engineering.md). For transformers, embeddings, FAISS and RAG basics, use the [Deep Learning last-hour sheet](../01-deep-learning/00-last-hour-before-the-interview.md).
Also: [Tool Calling and the ReAct Agent Loop](./02-tool-calling-and-react.md).
:::

## 🎤 The Questions You Are Most Likely to Get

### LLM APIs and prompt engineering

<details>
<summary><strong>1. What actually happens when you call an LLM API?</strong></summary>

A stateless HTTPS POST: model + role-tagged messages + settings in; the provider applies the chat template, prefills the prompt, decodes one token at a time, and returns text + usage (input/output tokens) + stop reason. Because it's stateless, history is re-sent every turn (35 → 78 prompt tokens on turn 2 in our run), so long chats get slower and costlier: total input $T\hyper{S} + \hyper{m}T(T+1)/2$.
</details>

<details>
<summary><strong>2. Temperature, top-p, min-p: explain and set them.</strong></summary>

$\pred{p_i} \propto e^{\pred{z_i}/\hyper{T}}$: below 1 sharper, above 1 flatter, order unchanged. Top-p keeps the smallest set reaching mass p, so it adapts: 1 token for a yes/no question, 268 for an open question at p = 0.9. Min-p keeps tokens above a fraction of the top one. Greedy for extraction and code, about 0.7 + top-p 0.9 for assistants. Newer reasoning models restrict them: Claude API 2026-10-21 removed temperature, top_p and top_k.
</details>

<details>
<summary><strong>3. How do you reduce latency and cost?</strong></summary>

Latency = TTFT + TPOT × (n − 1) (0.213 + 59 × 0.0824 ≈ 5.08 s). Trim and cache the prompt prefix (TTFT 6,163 → 270 ms with a cached 2,218-token prefix), cap output, stream (first token about 24× sooner), parallelise. Cost is linear in tokens with output priced higher: shorter answers saved 29.6% and caching 1,200 tokens saved 40.0% in the worked example; route easy requests to cheaper models.
</details>

<details>
<summary><strong>4. How do you do prompt engineering properly?</strong></summary>

Versioned prompts measured on a labelled set, one change at a time. 24 tickets, 0.5B model: vague 0 valid → clear 12 correct → few-shot 14 → label definitions 17 → definitions + scoring the 4 allowed labels 20 (all valid). Chain of thought: 1 → 9 of 10 correct for 8.4 → 193.3 output tokens.
</details>

<details>
<summary><strong>5. How do you get reliable structured output?</strong></summary>

Pydantic schema + strict structured outputs (OpenAI `text_format`, Anthropic `output_config.format`, Gemini `response_schema`) or constrained decoding; check the stop reason, validate, repair wrappers. Prompt only: 20/24 valid; strip code fences: 22/24; constrained: 24/24. Valid isn't correct: 10 to 13 of 24 intents right constrained vs 14 unconstrained.
</details>

<details>
<summary><strong>6. How do you call LLM APIs robustly in production?</strong></summary>

Own timeouts (SDK default read timeout 600 s); SDK retries (2, 0.5 s doubling to 8 s with jitter, on 408/409/429/5xx, honouring Retry-After); no retries on 400/401/403; budget tokens per minute; check stop reasons; idempotent side effects; fallback models; log tokens, latency and failure rates; pin SDK and model versions (the Anthropic SDK 2.0 upgrade turned `temperature=` into a TypeError).
</details>

<details>
<summary><strong>7. What is prompt injection and how do you defend against it?</strong></summary>

Untrusted text (from users, documents, web pages, emails or tool outputs) carrying instructions the model follows. Prompts alone can't stop it: no secrets in prompts, label untrusted content as data, least-privilege tools validated in code, confirmation for side effects, output filtering.
</details>

### Tool calling and agents

<details>
<summary><strong>8. How does function calling work, and who executes the function?</strong></summary>

Tool schemas (name, description, JSON Schema) go with the request; the model returns a structured call; your code validates, authorises, approves writes, executes and returns the result linked by the call ID (`call_id`, `tool_call_id`, `tool_use_id`). The model never executes anything. Five schemas cost 584 tokens on every call in our run.
</details>

<details>
<summary><strong>9. How do agents fail, and how do you guard them?</strong></summary>

Invented tools, invalid or wrong arguments (a required budget became an invented ₹100), under-calling (how-to questions answered from memory), and skipped steps with invented answers (₹100 and 10 GB that no tool returned). Guards: allowlist + Pydantic validation, approval + idempotency for writes, max steps + repeated-call stop, grounding checks, plan-then-execute for multi-part requests. Errors compound: $0.95^{10} = 0.5987$.
</details>

<details>
<summary><strong>10. ReAct vs native function calling?</strong></summary>

ReAct writes Thought / Action / Observation as text; stop at "Observation:" and insert the real result. Native calling returns typed calls. On our two-part question the native loop called one tool (grounding 0/3) while ReAct called both (3/3), at the cost of more output tokens and regex parsing.
</details>

## 🔢 Numbers Worth Quoting

| Topic | Number |
|---|---|
| Same Hindi sentence | 64 Qwen tokens vs 23 o200k tokens |
| Top-p 0.9 nucleus | 1 token (yes/no) vs 268 (open question) |
| TTFT / TPOT on CPU | 213 ms / 82.4 ms; 6.3 s TTFT at 2,227 prompt tokens |
| Prefix cache | TTFT 6,163 → 270 ms, same first token |
| Prompt ladder | 0 → 12 → 14 → 17 → 20 of 24 |
| Chain of thought | 1 → 9 of 10, 23× output tokens |
| JSON validity | 20 → 22 → 24 of 24; correct 14 vs 10 to 13 |
| Tool selection (24 queries) | 0.5B 18, 1.5B 21 |
| Agent prompt growth | 674 → 756 tokens per tool round |
| Compound reliability | 0.95^10 = 0.5987, 0.95^100 = 0.0059 |

## 🚨 Traps

- "The model remembers the conversation": no, the app re-sends it.
- Temperature 0 isn't bit-exact on hosted GPUs, and some models no longer accept it.
- Hugging Face `generate()` follows the model's `generation_config`; Qwen2.5 samples by default.
- Streaming improves perceived latency, not total latency.
- Structured outputs guarantee shape, not truth.
- A retry is a new generation with new charges.
- The model proposes tool calls; your code executes them.
- A loop that ends without errors can still invent facts: check grounding.
- Forcing tool use everywhere turns small talk into tool calls.
