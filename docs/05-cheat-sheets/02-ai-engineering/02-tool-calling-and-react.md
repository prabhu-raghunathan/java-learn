---
title: "Cheat Sheet: Tool Calling and the ReAct Agent Loop"
sidebar_position: 2
---

:::tip[What this sheet covers]
Tool calling and agents on one screen: what the model does and what your code does, tool schemas and their token cost, provider message shapes, the agent loop and its guards, tool-selection failures, safe execution, grounding, plan-then-execute, ReAct and compound reliability. Every number comes from the runs on the full page (Qwen2.5-1.5B and 0.5B, five telecom tools, CPU, greedy). Full notes: [Tool Calling and the ReAct Agent Loop](../../03-ai-ml/04-nlp-llms-and-projects/01-llm-apps-and-agents/02-tool-calling-and-react-agent-loop.md).
:::

## 🎤 Answer These Out Loud

<details>
<summary><strong>What is function calling, and who executes the function?</strong></summary>

You send tool definitions (name, description, JSON Schema) with the request. When a tool is needed the model returns a structured call (name plus JSON arguments) instead of text. Your code validates, authorises, executes and returns the result linked by the call ID, and the model continues. The model never executes anything; it's structured output pointed at a function, so arguments are untrusted input.
</details>

<details>
<summary><strong>Walk me through an agent loop.</strong></summary>

Call the model with messages and tools. If the reply has tool calls: check the name is allowed, validate arguments, authorise, ask approval for write actions, execute with a timeout, then append the call and its result (same ID) and call again. A reply with no tool calls is the answer. Guard with max steps, repeated-call detection and budgets. Each round re-sends everything: 674 → 756 prompt tokens in one round of our run.
</details>

<details>
<summary><strong>How do you evaluate and improve tool selection?</strong></summary>

A labelled set with the expected tool and arguments; measure right tool, valid parameters, correct values, and over- and under-calling. On 24 queries a 1.5B model scored 21 (it answered how-to questions from memory) and a 0.5B model 18 (including an invented `activate_roaming` tool). A better description and a system rule didn't change the 21; forcing a call fixed those three but turned small talk into tool calls. So route by intent first, then force tools for the intents that need them.
</details>

<details>
<summary><strong>How do you execute tool calls safely?</strong></summary>

Allowlist names; validate with the same Pydantic schema you advertised; authorise for the end user; cap time and output size; return errors as tool results (Anthropic `is_error`). Write actions need human approval and idempotency keys, because a retry is a new call. Design tools the model can use: a required budget made the model invent ₹100 and find nothing, while optional filters sorted cheapest first led it straight to P299.
</details>

<details>
<summary><strong>Why can an agent finish without errors and still be wrong?</strong></summary>

The loop ends when the model stops calling tools, not when the task is done. Asked about an outage and remaining data, our agent called only `check_outage` and invented ₹100 and 10 GB. A numeric grounding check (every answer number must appear in the question or tool results) flagged 100 and 10, plus a 6 PM-vs-18:00 false positive. Plan-then-execute and ReAct both fetched both facts.
</details>

<details>
<summary><strong>What is ReAct?</strong></summary>

The model writes Thought, then Action (tool plus JSON), then your code stops generation at "Observation:", runs the action and inserts the real result; repeat until Final Answer. The explicit thought made our 1.5B model fetch both facts (prompts 362 → 447 → 517 tokens). Costs: output tokens and fragile parsing. It needs a stop string and a repeated-action guard; our first prompt, whose example contradicted the schema, looped on the same failing call.
</details>

<details>
<summary><strong>Why do long agent chains fail, and when would you not use an agent?</strong></summary>

Errors compound: at 95% per step, $0.95^{10} = 0.5987$ and $0.95^{100} = 0.0059$. Keep chains short, validate between steps, plan then execute for known request types. If the steps are known, a fixed workflow (route, call the API, phrase the answer) is cheaper, faster and testable; use an agent only where the path depends on intermediate results.
</details>

## 📐 Formulas

| Concept | Formula |
|---|---|
| Prompt with tools | $\data{n_{in}} = \data{n_{msgs}} + \sum_j \hyper{n_{schema}^{(j)}} + \hyper{n_{tool\,prompt}}$ (68 → 652 tokens) |
| Prompt at step k | $\data{P_k} = \data{P} + \data{o}(k-1)$ |
| Total agent input | $\hyper{K}\data{P} + \data{o}\hyper{K}(\hyper{K}-1)/2$ (674, +82, 5 calls → 4,190) |
| Compound reliability | $\pred{P_{task}} = \pred{p}^{\hyper{K}}$ |
| Numeric grounding | $\pred{G} = \lvert \pred{\mathcal{N}_{ans}} \cap \data{\mathcal{N}_{src}} \rvert / \lvert \pred{\mathcal{N}_{ans}} \rvert$ |

## 🔢 Numbers to Quote

| What | Number |
|---|---|
| Token cost of five tool schemas (Qwen template) | 584 tokens per call (68 → 652) |
| Tool selection, 24 queries | 0.5B: 18; 1.5B: 21 (exact arguments on every correct pick) |
| Levers on 1.5B under-calling | description 21, system rule 21, forced call 22 (small talk broken) |
| Native loop prompt growth | 674 → 756 tokens per tool round (+82) |
| ReAct prompts | 362 → 447 → 517 tokens |
| Invented values on the two-part question | ₹100 and 10 GB (real: ₹42.5, 1.2 GB) |
| Grounding rate | native 0/3, plan-then-execute 2/3, ReAct 3/3 |
| Approved recharge flow | list_plans → P299 → approval → done (921 → 1,010 tokens) |
| $0.95^{K}$ | 0.7738 (5), 0.5987 (10), 0.0059 (100) |
| $0.99^{K}$ | 0.9044 (10), 0.366 (100) |

## ⚖️ Quick Comparisons

| | OpenAI Responses | Chat Completions | Anthropic | Gemini |
|---|---|---|---|---|
| Tool definition | flat `name`, `parameters`, `strict` | nested under `function` | `input_schema` | function declarations / Python functions |
| Model's call | `function_call` item, `call_id`, string `arguments` | `tool_calls[i]`, `id` | `tool_use` block, `id`, object `input` | `function_call` part |
| Result | `function_call_output` + `call_id` | `role: "tool"` + `tool_call_id` | `tool_result` in a **user** message + `tool_use_id`, `is_error` | `function_response` part |
| Force / forbid | `required`, named, `none` | same | `any`, `tool`, `none` | `ANY`, `NONE`, `VALIDATED` |

| | Reactive loop | Plan → validate → execute | ReAct |
|---|---|---|---|
| Our two-part question | 1 tool, invented values | both tools, grounded | both tools, grounded |
| Strength | Adapts to results | Whole request considered; plan checkable | Visible reasoning per step |
| Cost | Fewer tokens | One planning call | Extra output tokens, regex parsing |

| Guard | Stops |
|---|---|
| Allowlist + schema validation | Invented tools, malformed arguments |
| Approval + idempotency key | Unwanted or duplicated writes |
| Max steps + repeated-call stop | Runaway loops |
| Grounding check | Invented facts |
| Tool output treated as data | Indirect prompt injection |

## 💻 Code You Should Be Able to Write

```python title="A Pydantic model as a tool schema and validator" showLineNumbers
from pydantic import BaseModel, Field, ValidationError

class GetBalance(BaseModel):
    """Get the prepaid balance (rupees) and mobile data left (GB) for a mobile number."""
    phone_number: str = Field(pattern=r"^\d{10}$", description="10-digit mobile number")

print(GetBalance.model_json_schema()["required"])                 # ['phone_number']
try:
    GetBalance.model_validate({"phone_number": "98765 43210"})
except ValidationError as e:
    err = e.errors()[0]
    print(f"invalid argument {err['loc'][0]}: {err['msg']}")      # invalid argument phone_number: String should match pattern '^\d{10}$'
```

```python title="Grounding check and agent arithmetic" showLineNumbers
import re
def ungrounded_numbers(answer, sources):
    known = {float(x) for x in re.findall(r"\d+(?:\.\d+)?", sources)}
    return [x for x in re.findall(r"\d+(?:\.\d+)?", answer) if float(x) not in known]

print(ungrounded_numbers("Your balance is 100 and 10 GB remain.", '{"outage": "Fibre cut; back by 18:00."}'))   # ['100', '10']
print(round(0.95 ** 10, 4), 5 * 674 + 82 * 5 * 4 // 2)                                                           # 0.5987 4190
```

## 🚨 Traps

- The model proposes calls; your code executes them. Never say "the LLM calls the API".
- Tool definitions are sent on every call (584 tokens for five tools here).
- Unpaired call IDs are 400 errors; Anthropic tool results go in a user message, first.
- Forcing tool use everywhere turns small talk into tool calls.
- Required parameters the user rarely gives get invented values.
- A loop that ends cleanly can still be wrong; check grounding.
- ReAct without a stop at "Observation:" invents tool results.
- A retry of a write action is a second write unless you use idempotency keys.
- Tool results can carry injected instructions; permissions belong in code.
