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
- **Bold** key architectural terms and JVM mechanics for scannability.

# Page Structure Specification
Every generated `.md` page MUST strictly follow this layout:

---
title: "<Concept Name>"
sidebar_position: <number>
---

:::tip TL;DR (30-Second Interview Pitch)
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
- If there is a common misconception, wrap it in a `:::danger Interview Trap` admonition block.

## 💻 Java Implementation
Clean, compilable code snippet demonstrating the concept.
- Use `// highlight-next-line` to draw attention to critical lines.

## 🚨 Interview Gotchas & Edge Cases
:::warning The Most Common Question
Highlight the single most frequently asked deep-dive question regarding this topic here.
:::
- Follow with bullet points covering edge cases, performance pitfalls, or default configuration traps.

# Execution Guardrails
- Write files directly to disk in the `docs/` folder, maintaining a clean numeric hierarchy (e.g., `docs/01-core-java/`).
- Ensure all Mermaid syntax, code blocks, and Admonition blocks (`:::`) are properly closed.