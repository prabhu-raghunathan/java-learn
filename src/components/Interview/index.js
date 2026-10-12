/**
 * Interview-prep building blocks for the Java / Spring pages.
 * All registered globally in src/theme/MDXComponents.js (no imports needed).
 *
 *  <PriorityKey />                       legend for the 🔴 🟠 🟡 heading badges (once, after the TL;DR)
 *  <Since v="17" />                      inline "Java 17+" version pill
 *
 *  <Predict n="1" title="Integer cache" output="true false">
 *
 *  ```java
 *  ...question code...
 *  ```
 *
 *  Explanation markdown (hidden until the reader clicks Reveal).
 *
 *  </Predict>
 *
 *  <RuleCard name="equals/hashCode contract" tag="Object" why="..." remember="...">
 *
 *  The rule in one or two lines, then an optional small code block.
 *
 *  </RuleCard>
 *
 *  <Ladder title="Garbage collection">
 *  <Rung q="What is GC?">answer markdown</Rung>
 *  <Rung q="Which generations?">answer markdown</Rung>
 *  </Ladder>
 *
 * Props are plain text: no double quotes, curly braces or < inside them.
 * Blank lines around inner markdown are REQUIRED so MDX parses it.
 */
import React, {useState} from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

/* ------------------------------------------------------------------ */
/* Priority legend + version pill                                      */
/* ------------------------------------------------------------------ */

export function PriorityKey() {
  return (
    <div className={styles.key}>
      <span className={styles.keyTitle}>Section badges</span>
      <span className={styles.keyItem}>
        🔴 <span>asked in almost every round</span>
      </span>
      <span className={styles.keyItem}>
        🟠 <span>frequently asked</span>
      </span>
      <span className={styles.keyItem}>
        🟡 <span>senior-level differentiator</span>
      </span>
      <span className={styles.keyItem}>
        <Since v="17" /> <span>needs that Java version or newer</span>
      </span>
    </div>
  );
}

export function Since({v}) {
  return (
    <span className={styles.since} title={`Available since Java ${v}`}>
      Java {v}+
    </span>
  );
}

/* ------------------------------------------------------------------ */
/* Predict the output                                                  */
/* ------------------------------------------------------------------ */

export function Predict({n, title, output, children}) {
  const [open, setOpen] = useState(false);
  const parts = React.Children.toArray(children);
  const question = parts[0];
  const explanation = parts.slice(1);
  return (
    <div className={clsx(styles.predict, open && styles.predictOpen)}>
      <div className={styles.predictHeader}>
        <span className={styles.predictIcon} aria-hidden="true">?</span>
        <span className={styles.predictTitle}>
          {n && <span className={styles.predictNum}>#{n}</span>}
          {title}
        </span>
        <span className={styles.predictTag}>Predict the output</span>
      </div>
      <div className={styles.predictBody}>{question}</div>
      {!open ? (
        <button type="button" className={styles.revealBtn} onClick={() => setOpen(true)}>
          Guess first, then reveal ↓
        </button>
      ) : (
        <div className={styles.answer}>
          {output && (
            <div className={styles.output}>
              <span className={styles.outputLabel}>Output</span>
              <code>{output}</code>
            </div>
          )}
          <div className={styles.why}>
            <span className={styles.whyLabel}>Why</span>
            {explanation}
          </div>
          <button type="button" className={styles.hideBtn} onClick={() => setOpen(false)}>
            Hide answer
          </button>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Rule card (the Java counterpart of the Formula card)                */
/* ------------------------------------------------------------------ */

export function RuleCard({name, tag, why, remember, children}) {
  return (
    <div className={styles.rule}>
      <div className={styles.ruleHeader}>
        <span className={styles.ruleIcon} aria-hidden="true">§</span>
        <span className={styles.ruleName}>{name}</span>
        {tag && <span className={styles.ruleTag}>{tag}</span>}
      </div>
      <div className={styles.ruleBody}>{children}</div>
      {why && (
        <div className={styles.ruleWhy}>
          <span className={styles.ruleWhyLabel}>Why it exists</span>
          {why}
        </div>
      )}
      {remember && (
        <div className={styles.remember}>
          <span aria-hidden="true">🧠</span>
          <span>
            <strong>Remember it as:</strong> {remember}
          </span>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Follow-up ladder                                                    */
/* ------------------------------------------------------------------ */

export function Ladder({title, children}) {
  return (
    <div className={styles.ladder}>
      <div className={styles.ladderHeader}>
        <span aria-hidden="true">🪜</span>
        <span>
          Follow-up ladder{title ? `: ${title}` : ''}
        </span>
        <span className={styles.ladderHint}>each rung is what the interviewer asks next</span>
      </div>
      <ol className={styles.rungs}>{children}</ol>
    </div>
  );
}

export function Rung({q, children}) {
  return (
    <li className={styles.rung}>
      <details>
        <summary>{q}</summary>
        <div className={styles.rungBody}>{children}</div>
      </details>
    </li>
  );
}
