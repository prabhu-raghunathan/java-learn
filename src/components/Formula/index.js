/**
 * <Formula name="Mean Squared Error" tag="Regression loss"
 *          words="Average of the squared misses."
 *          remember="Square the miss, then average it.">
 *
 * $$ ... $$                      ← the equation (use \data \param \pred \loss \hyper colours)
 *
 * - $\pred{\hat y}$ … prediction ← optional symbol legend (markdown list)
 *
 * </Formula>
 *
 * Blank lines around the inner markdown are REQUIRED so MDX parses the math.
 * Registered globally in src/theme/MDXComponents.js.
 */
import React from 'react';
import styles from './styles.module.css';

export default function Formula({name, tag, words, remember, children}) {
  return (
    <div className={styles.card}>
      <div className={styles.header}>
        <span className={styles.icon} aria-hidden="true">∑</span>
        <span className={styles.name}>{name}</span>
        {tag && <span className={styles.tag}>{tag}</span>}
      </div>
      <div className={styles.body}>{children}</div>
      {words && (
        <div className={styles.words}>
          <span className={styles.wordsLabel}>In words</span>
          {words}
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

/** Colour key for the symbol roles; put once near the top of each AI/ML page. */
export function FormulaKey() {
  const roles = [
    ['eq-data', 'x, X, y', 'data & targets'],
    ['eq-param', 'W, b, θ', 'learnable parameters'],
    ['eq-pred', 'z, a, ŷ, p', 'outputs & activations'],
    ['eq-loss', 'L, ∇, δ', 'loss & gradients'],
    ['eq-hyper', 'η, p, S, F', 'hyperparameters'],
  ];
  return (
    <div className={styles.key}>
      <span className={styles.keyTitle}>Formula colours</span>
      {roles.map(([cls, sym, meaning]) => (
        <span key={cls} className={styles.keyItem}>
          <span className={`${cls} ${styles.keySym}`}>{sym}</span>
          <span className={styles.keyMeaning}>{meaning}</span>
        </span>
      ))}
    </div>
  );
}
