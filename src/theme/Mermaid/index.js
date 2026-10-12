/**
 * Swizzled @theme/Mermaid (wraps @docusaurus/theme-mermaid 3.10.x).
 *
 * WHY: Mermaid is one global, mutable singleton. The stock component calls
 * mermaid.initialize() + mermaid.render() from every diagram at the same time,
 * and in dark mode each diagram renders TWICE (light first, then dark after the
 * colour mode kicks in). Those overlapping renders race each other and some
 * diagrams end up blank (seen reliably on pages with 2+ diagrams in dark mode).
 *
 * FIX: run every render through one promise queue (strictly one at a time),
 * give each render a unique id, and ignore results that became stale
 * (e.g. the light-theme render that finished after dark mode took over).
 */
import React, {useEffect, useRef, useState} from 'react';
import ErrorBoundary from '@docusaurus/ErrorBoundary';
import {ErrorBoundaryErrorMessageFallback} from '@docusaurus/theme-common';
import {
  MermaidContainerClassName,
  useMermaidConfig,
} from '@docusaurus/theme-mermaid/client';
import styles from './styles.module.css';

let renderQueue = Promise.resolve();
let renderCounter = 0;

function renderSerially(text, config) {
  const job = renderQueue.then(async () => {
    const mermaid = (await import('mermaid')).default;
    mermaid.initialize(config);
    const id = `mermaid-svg-${++renderCounter}`;
    try {
      return await mermaid.render(id, text);
    } catch (error) {
      // Mermaid leaves an error SVG in the DOM on failure; clean it up.
      document.querySelector(`#d${id}`)?.remove();
      throw error;
    }
  });
  renderQueue = job.catch(() => {}); // keep the queue alive after a failure
  return job;
}

function MermaidRenderer({value}) {
  const config = useMermaidConfig();
  const [result, setResult] = useState(null);
  const ref = useRef(null);

  useEffect(() => {
    let stale = false;
    renderSerially(value, config).then(
      (rendered) => {
        if (!stale) setResult(rendered);
      },
      (error) => {
        if (!stale) {
          // Re-throw during render so the ErrorBoundary shows the message
          setResult(() => {
            throw error;
          });
        }
      },
    );
    return () => {
      stale = true;
    };
  }, [value, config]);

  useEffect(() => {
    if (result && ref.current) {
      result.bindFunctions?.(ref.current);
    }
  }, [result]);

  if (result === null) {
    return null;
  }
  return (
    <div
      ref={ref}
      className={`${MermaidContainerClassName} ${styles.container}`}
      // eslint-disable-next-line react/no-danger
      dangerouslySetInnerHTML={{__html: result.svg}}
    />
  );
}

export default function Mermaid(props) {
  return (
    <ErrorBoundary
      fallback={(params) => <ErrorBoundaryErrorMessageFallback {...params} />}>
      <MermaidRenderer {...props} />
    </ErrorBoundary>
  );
}
