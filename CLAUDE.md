# Project Context
This is an interview preparation wiki built on Docusaurus v3. It covers deep-dive Java, JVM internals, multithreading, Spring Boot, databases, system design, and AI/ML (Python for data exploration, classical machine learning, deep learning, and applied NLP/LLM projects). More subjects will be added over time, each as its own top-level numbered folder under `docs/`.

# Note Enrichment Protocol
When I provide raw, handwritten, or lecture notes, DO NOT just format them blindly. 
1. **Cross-Reference:** Check my notes against the documents located in the `_reference_materials/` folder.
2. **Correct & Clarify:** Silently fix technical inaccuracies.
3. **Enrich:** Expand brief notes using the reference materials to ensure they meet senior-level interview standards.

# Formatting & UI Constraints (STRICT)
- **NO WALLS OF TEXT.** The UI must be punchy, highly visual, and optimized for instant recall.
- Use **Markdown Tables** whenever comparing two concepts.
- Use **Docusaurus Admonitions** (`:::tip`, `:::danger`, `:::warning`, `:::info`) to break up the visual monotony and highlight interview traps.
- **Admonition title syntax (this Docusaurus version requires brackets):** `:::tip[Your Title Here]` — NOT `:::tip Your Title Here`. The space-separated legacy form silently fails to render (renders as literal `:::` text) on Docusaurus 3.10+, which uses `remark-directive` under the hood. Bare admonitions with no title (`:::danger` / `:::` with no title) don't need brackets.
- **Bold** key architectural terms and JVM mechanics for scannability.
- **Math (KaTeX)** is available for formulas — use `$...$` inline (e.g. `$y = mx + b$`) and `$$...$$` for block equations. Use this for ML/DL content (loss functions, gradients, regression formulas) instead of describing math in prose.
- **Images and Mermaid diagrams are click-to-zoom** by default (`docusaurus-plugin-image-zoom`) — no extra markup needed.
- **Mermaid rendering:** `src/theme/Mermaid/index.js` is a swizzled copy of the theme component that **queues diagram renders one at a time**. The stock component races (several diagrams per page + the dark-mode re-render) and leaves diagrams blank. **Do not delete it.** All diagram types (flowchart, decision diamonds, mindmap) render with it. Layout tips: if a `subgraph` sets its own `direction`, don't draw edges that cross between subgraphs (Mermaid ignores the direction and squashes the layout); split into separate diagrams instead. Prefer `flowchart TB` for chains of 4+ nodes, because long `LR` chains render too small to read.
- The site has **local full-text search** built in — no action needed when writing pages, just keep headings and titles descriptive since search ranks on them.

# Page Structure Specification
Every generated `.md` page MUST strictly follow this layout:

---
title: "<Concept Name>"
sidebar_position: <number>
---

:::tip[TL;DR (30-Second Interview Pitch)]
A concise 2-3 sentence definition explaining what the concept is and why it matters in production.
:::

## ⚡ Essential Interview Recall
A highly refined, scannable bulleted list of the absolute must-remember facts, mechanics, and keywords for this topic. No fluff, no Q&A—just the raw, distilled data points required to pass a senior-level interview.

## The Mental Model
Always include a visual diagram using Mermaid.js (` ```mermaid `) for flowcharts, lifecycles, or mindmaps.

## The Paradigm / Core Concepts
Use a Markdown Table here if comparing approaches. Otherwise, use a concise bulleted list.

## ⚙️ Under the Hood (JVM Mechanics)
Bullet points breaking down the internal mechanics. 
- Emphasize memory layout, bytecode instructions, and performance tradeoffs.
- If there is a common misconception, wrap it in a `:::danger[Interview Trap]` admonition block.

## 💻 Java Implementation
Clean, compilable code demonstrating the concept, following the **Code Block Conventions** below.

# Code Block Conventions (STRICT)
- **Never one giant block.** Split the implementation into one small block per idea, each under its own `### ` sub-heading with a one-line lead-in.
- **Full classes / files** get a filename title and line numbers: ```` ```java title="Vehicle.java" showLineNumbers ````.
- **Short snippets** (statement-level, runnable in `jshell`) get `showLineNumbers` and no title. Pages made mostly of snippets start the section with a `:::tip[Run it yourself]` note pointing at `jshell`.
- **Wrong-vs-right demos** go in their own block titled with ❌, e.g. ```` ```java title="❌ IS-A only goes one way" ````. Show the broken line as **real code** (not commented out) so it is coloured.
- **Line markers** (configured in `docusaurus.config.js` → `prism.magicComments`, styled in `src/css/custom.css`). The marker comment goes on its own line *above* the target line(s):
  - `// highlight-next-line` or `// highlight-start` … `// highlight-end` → **amber**: the line you must look at.
  - `// error-next-line` or `// error-start` … `// error-end` → **red**: won't compile, throws, or gives a surprising/wrong result.
  - `// success-next-line` or `// success-start` … `// success-end` → **green**: the correct fix.
  - In bash/python blocks use `#` instead of `//`.
- Prefix inline comments on red/green lines with ❌ / ✅. Keep trailing comments **column-aligned** within a block.
- Put expected output in trailing comments (`// 14, not 20`), not in separate print-out blocks.
- Pipelines and step sequences go in a **numbered table**, not in a wall of code comments.
- Languages available for highlighting: `java`, `python`, `bash`, `json`, `yaml`, `sql`, `properties`, `groovy`, `kotlin`, `text` (add more in `prism.additionalLanguages`).

## 🚨 Interview Gotchas & Edge Cases
:::warning[The Most Common Question]
Highlight the single most frequently asked deep-dive question regarding this topic here.
:::
- Follow with bullet points covering edge cases, performance pitfalls, or default configuration traps.

# Execution Guardrails
- Write files directly to disk in the `docs/` folder, maintaining a clean numeric hierarchy (e.g., `docs/01-core-java/`).
- Ensure all Mermaid syntax, code blocks, and Admonition blocks (`:::`) are properly closed.