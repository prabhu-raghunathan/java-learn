/**
 * Adds a custom :::analogy admonition (everyday picture for a hard idea).
 * Usage:  :::analogy[Autograd is a shop receipt]  ...  :::
 * The keyword is enabled in docusaurus.config.js → docs.admonitions.keywords.
 */
import React from 'react';
import clsx from 'clsx';
import DefaultAdmonitionTypes from '@theme-original/Admonition/Types';
import AdmonitionLayout from '@theme/Admonition/Layout';

function AnalogyAdmonition(props) {
  return (
    <AdmonitionLayout
      icon={<span style={{fontSize: '1.1em'}}>💡</span>}
      title="Analogy"
      {...props}
      className={clsx('alert alert--analogy', props.className)}>
      {props.children}
    </AdmonitionLayout>
  );
}

export default {
  ...DefaultAdmonitionTypes,
  analogy: AnalogyAdmonition,
};
