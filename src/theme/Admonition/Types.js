/**
 * Custom admonitions (keywords enabled in docusaurus.config.js → docs.admonitions.keywords):
 *   :::analogy[Autograd is a shop receipt]   everyday picture for a hard idea (💡)
 *   :::speak[Why is String immutable?]       the 60-second spoken interview answer (🎤)
 *   :::incident[Service OOMs every 6 hours]  production / debugging scenario (🚑)
 */
import React from 'react';
import clsx from 'clsx';
import DefaultAdmonitionTypes from '@theme-original/Admonition/Types';
import AdmonitionLayout from '@theme/Admonition/Layout';

function make(icon, defaultTitle, cls) {
  return function CustomAdmonition(props) {
    return (
      <AdmonitionLayout
        icon={<span style={{fontSize: '1.1em'}}>{icon}</span>}
        title={defaultTitle}
        {...props}
        className={clsx('alert', cls, props.className)}>
        {props.children}
      </AdmonitionLayout>
    );
  };
}

export default {
  ...DefaultAdmonitionTypes,
  analogy: make('💡', 'Analogy', 'alert--analogy'),
  speak: make('🎤', 'Say it in 60 seconds', 'alert--speak'),
  incident: make('🚑', 'Production scenario', 'alert--incident'),
};
