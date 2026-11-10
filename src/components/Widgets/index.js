/**
 * Interactive "Try it" widgets for the Deep Learning pages. Each page imports only what it uses:
 *
 *   import {TemperatureSampler} from '@site/src/components/Widgets';
 *   <TemperatureSampler />
 *
 * All maths lives in ./math.js (unit-tested by ./math.test.mjs against the pages' verified numbers).
 * Widgets render on the server too (no window access at import time), and work in light and dark mode.
 */
import React, {useMemo, useState} from 'react';
import * as M from './math';
import styles from './styles.module.css';
import useBaseUrl from '@docusaurus/useBaseUrl';
import katex from 'katex';
import {useEffect} from 'react';

// Same colour macros as docusaurus.config.js, so widget formulas match the page equations.
const KATEX_MACROS = {
  '\\data': '\\htmlClass{eq-data}{#1}', '\\param': '\\htmlClass{eq-param}{#1}', '\\pred': '\\htmlClass{eq-pred}{#1}',
  '\\loss': '\\htmlClass{eq-loss}{#1}', '\\hyper': '\\htmlClass{eq-hyper}{#1}',
};
/** Typeset a LaTeX string with KaTeX (display style by default). */
function Tex({children, display = true}) {
  const html = katex.renderToString(children, {displayMode: display, throwOnError: false, macros: {...KATEX_MACROS},
    trust: (ctx) => ctx.command === '\\htmlClass', strict: 'ignore'});
  return <span style={display ? {display: 'block', overflowX: 'auto', overflowY: 'hidden'} : undefined} dangerouslySetInnerHTML={{__html: html}} />;
}

/* =================================================================================== */
/* Shared building blocks                                                              */
/* =================================================================================== */

function Widget({title, hint, children}) {
  return (
    <div className={styles.widget}>
      <div className={styles.header}>
        <span className={styles.badge}>Try it</span>
        <span className={styles.title}>{title}</span>
      </div>
      <div className={styles.body}>{children}</div>
      {hint && <div className={styles.hint}>{hint}</div>}
    </div>
  );
}

function Slider({label, value, min, max, step, onChange, fmt = (v) => v, color}) {
  return (
    <label className={styles.slider}>
      <span className={styles.sliderLabel} style={color ? {color: `var(--eq-${color})`} : undefined}>{label}</span>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))} />
      <span className={styles.sliderValue}>{fmt(value)}</span>
    </label>
  );
}

function Readout({items}) {
  return (
    <div className={styles.readout}>
      {items.map(([k, v, color], i) => (
        <div key={i} className={styles.readoutItem}>
          <span className={styles.readoutKey} style={color ? {color: `var(--eq-${color})`} : undefined}>{k}</span>
          <span className={styles.readoutVal}>{v}</span>
        </div>
      ))}
    </div>
  );
}

function Select({label, value, options, onChange}) {
  return (
    <label className={styles.select}>
      <span>{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
      </select>
    </label>
  );
}

const f3 = (v) => (Number.isFinite(v) ? (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : v.toFixed(3)) : '∞');
const f2 = (v) => v.toFixed(2);

/** Bars with value labels (used by softmax, LSTM and decay charts). */
function Bars({values, labels, max, highlight = -1, colors, height = 170, fmt = (v) => v.toFixed(3)}) {
  const W = 800, H = height, top = 18, bottom = labels ? 22 : 6;
  const bw = W / values.length;
  const m = max ?? Math.max(...values.map((v) => Math.abs(v)), 1e-9);
  return (
    <svg className={styles.bars} viewBox={`0 0 ${W} ${H}`}>
      <line x1="0" y1={H - bottom} x2={W} y2={H - bottom} className={styles.axis} />
      {values.map((v, i) => {
        const h = Math.max((Math.abs(v) / m) * (H - top - bottom), 1);
        const fill = colors ? colors[i] : i === highlight ? 'var(--w-hl)' : 'var(--w-bar)';
        const x = i * bw + bw * 0.15;
        return (
          <g key={i}>
            <rect x={x} y={H - bottom - h} width={bw * 0.7} height={h} fill={fill} rx="3" opacity={v < 0 ? 0.55 : 1} />
            {fmt(v) !== '' && <text x={i * bw + bw / 2} y={H - bottom - h - 5} className={styles.bText} textAnchor="middle">{fmt(v)}</text>}
            {labels && <text x={i * bw + bw / 2} y={H - 6} className={styles.bLabel} textAnchor="middle">{labels[i]}</text>}
          </g>
        );
      })}
    </svg>
  );
}

/** A line plot of one or more functions on [x0, x1]; marks optional points. */
function Plot({fns, x0, x1, y0, y1, points = [], path, height = 220, xLabel, vline}) {
  const W = 400, H = 220, pad = 28;
  const X = (x) => pad + ((x - x0) / (x1 - x0)) * (W - 2 * pad);
  const Y = (y) => H - pad - ((Math.min(Math.max(y, y0), y1) - y0) / (y1 - y0)) * (H - 2 * pad);
  const n = 160;
  return (
    <svg className={styles.plot} viewBox={`0 0 ${W} ${H}`} style={{maxHeight: height}}>
      <line x1={pad} y1={Y(0)} x2={W - pad} y2={Y(0)} className={styles.axis} />
      <line x1={X(0)} y1={pad / 2} x2={X(0)} y2={H - pad} className={styles.axis} />
      {vline !== undefined && <line x1={X(vline)} y1={pad / 2} x2={X(vline)} y2={H - pad} className={styles.vline} />}
      {fns.map(({f, color, dash}, k) => {
        const d = Array.from({length: n + 1}, (_, i) => {
          const x = x0 + ((x1 - x0) * i) / n;
          return `${i ? 'L' : 'M'}${X(x).toFixed(1)},${Y(f(x)).toFixed(1)}`;
        }).join(' ');
        return <path key={k} d={d} fill="none" stroke={color} strokeWidth="2.4" strokeDasharray={dash} />;
      })}
      {path && (
        <polyline points={path.map(([x, y]) => `${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join(' ')}
          fill="none" className={styles.pathLine} />
      )}
      {points.map(([x, y, cls], i) => (
        <circle key={i} cx={X(x)} cy={Y(y)} r={i === points.length - 1 ? 5 : 3.5} className={cls || styles.point} />
      ))}
      {xLabel && <text x={W - pad + 14} y={Y(0) + 4} className={styles.axisText} textAnchor="start">{xLabel}</text>}
      <text x={pad - 4} y={Y(y1) + 4} className={styles.axisText} textAnchor="end">{y1}</text>
      <text x={pad - 4} y={Y(y0) + 4} className={styles.axisText} textAnchor="end">{y0}</text>
      <text x={X(x0)} y={H - 10} className={styles.axisText} textAnchor="middle">{x0}</text>
      <text x={X(x1)} y={H - 10} className={styles.axisText} textAnchor="middle">{x1}</text>
    </svg>
  );
}

/* =================================================================================== */
/* PyTorch / calculus / ANN                                                            */
/* =================================================================================== */

/** Gradient descent on L(w) = (w − 3)², the pages' running example. */
export function GradientDescentPlayground() {
  const [lr, setLr] = useState(0.1);
  const [steps, setSteps] = useState(10);
  const [w0, setW0] = useState(0);
  const {ws, factor, regime} = useMemo(() => M.gradientDescent({w0, lr, steps}), [w0, lr, steps]);
  const L = (w) => (w - 3) ** 2;
  const shown = ws.filter((w) => Math.abs(w) < 1e6);
  return (
    <Widget title="Gradient descent on L(w) = (w − 3)²"
      hint="Each step: w ← w − η · 2(w − 3). The distance to the minimum is multiplied by (1 − 2η): try η = 0.05, 0.4, 1.0 and 1.05.">
      <div className={styles.controls}>
        <Slider label="η (learning rate)" color="hyper" value={lr} min={0.01} max={1.2} step={0.01} onChange={setLr} fmt={f2} />
        <Slider label="steps" value={steps} min={1} max={25} step={1} onChange={setSteps} />
        <Slider label="start w₀" color="param" value={w0} min={-2} max={8} step={0.5} onChange={setW0} fmt={(v) => v.toFixed(1)} />
      </div>
      <Plot fns={[{f: L, color: 'var(--eq-loss)'}]} x0={-4} x1={10} y0={0} y1={50} vline={3}
        path={shown.map((w) => [w, L(w)])} points={shown.map((w) => [w, L(w)])} xLabel="w" />
      <Readout items={[
        ['factor 1 − 2η', f3(factor), 'hyper'],
        ['w after last step', f3(ws[ws.length - 1]), 'param'],
        ['loss', f3(L(ws[ws.length - 1])), 'loss'],
        ['regime', regime],
      ]} />
      <div className={styles.seq}>w: {ws.slice(0, 9).map((w) => f3(w)).join(' → ')}{ws.length > 9 ? ' → …' : ''}</div>
    </Widget>
  );
}

/** Activation functions and their slopes. */
export function ActivationExplorer({initial = 'sigmoid'}) {
  const [act, setAct] = useState(initial);
  const [z, setZ] = useState(1);
  const A = M.ACTIVATIONS[act];
  return (
    <Widget title="Activation functions and their slopes"
      hint="The slope is what backprop multiplies by. Sigmoid never passes more than 0.25 of the gradient; ReLU passes all of it for z > 0 and none for z ≤ 0 (dead ReLU).">
      <div className={styles.controls}>
        <Select label="activation" value={act} onChange={setAct}
          options={Object.entries(M.ACTIVATIONS).map(([k, v]) => [k, v.label])} />
        <Slider label="z (pre-activation)" color="pred" value={z} min={-6} max={6} step={0.1} onChange={setZ} fmt={(v) => v.toFixed(1)} />
      </div>
      <Plot fns={[{f: A.f, color: 'var(--eq-pred)'}, {f: A.df, color: 'var(--eq-loss)', dash: '5 4'}]}
        x0={-6} x1={6} y0={act === 'tanh' ? -1 : 0} y1={act === 'relu' ? 6 : 1}
        points={[[z, A.f(z)], [z, A.df(z), styles.pointGrad]]} vline={z} xLabel="z" />
      <Readout items={[
        [`${A.label}(z)`, f3(A.f(z)), 'pred'],
        ['slope (dashed)', f3(A.df(z)), 'loss'],
        ['output range', A.range],
        ['largest slope', A.maxSlope],
      ]} />
    </Widget>
  );
}

/** The calculus page's graph: c = a + b, d = ab, e = cd, f = e + d. */
export function ChainRuleExplorer() {
  const [a, setA] = useState(2);
  const [b, setB] = useState(3);
  const r = M.chainRule(a, b);
  return (
    <Widget title="Chain rule on c = a + b, d = a·b, e = c·d, f = e + d"
      hint="d feeds both e and f, so its gradient is the SUM of both paths: ∂f/∂d = c + 1. Then ∂f/∂a = ∂f/∂c · 1 + ∂f/∂d · b.">
      <div className={styles.controls}>
        <Slider label="a" color="data" value={a} min={-5} max={5} step={0.5} onChange={setA} fmt={(v) => v.toFixed(1)} />
        <Slider label="b" color="data" value={b} min={-5} max={5} step={0.5} onChange={setB} fmt={(v) => v.toFixed(1)} />
      </div>
      <div className={styles.twoCol}>
        <Readout items={[['c = a + b', f3(r.c), 'pred'], ['d = a · b', f3(r.d), 'pred'], ['e = c · d', f3(r.e), 'pred'], ['f = e + d', f3(r.f), 'loss']]} />
        <Readout items={[
          ['∂f/∂e', f3(r.df_de), 'loss'], ['∂f/∂c = d', f3(r.df_dc), 'loss'], ['∂f/∂d = c + 1', f3(r.df_dd), 'loss'],
          ['∂f/∂a = d·1 + (c+1)·b', f3(r.df_da), 'loss'], ['∂f/∂b = d·1 + (c+1)·a', f3(r.df_db), 'loss'],
        ]} />
      </div>
    </Widget>
  );
}

/** NumPy / PyTorch broadcasting rules. */
export function BroadcastingChecker() {
  const [sa, setSa] = useState('(3, 1)');
  const [sb, setSb] = useState('(1, 4)');
  const r = M.broadcast(M.parseShape(sa), M.parseShape(sb));
  return (
    <Widget title="Will these shapes broadcast?"
      hint="Align shapes from the RIGHT. Each pair of dimensions must be equal, or one of them must be 1 (or missing). Try (2, 3) with (3), then (2, 3) with (2).">
      <div className={styles.controls}>
        <label className={styles.textIn}>A <input value={sa} onChange={(e) => setSa(e.target.value)} /></label>
        <label className={styles.textIn}>B <input value={sb} onChange={(e) => setSb(e.target.value)} /></label>
      </div>
      <table className={styles.table}>
        <thead><tr><th>A</th><th>B</th><th>result</th><th>rule</th></tr></thead>
        <tbody>{r.rows.map((row, i) => (
          <tr key={i} className={row.r === null ? styles.bad : undefined}>
            <td>{row.a ?? '–'}</td><td>{row.b ?? '–'}</td><td>{row.r ?? '✗'}</td><td>{row.why}</td>
          </tr>
        ))}</tbody>
      </table>
      <div className={r.ok ? styles.good : styles.badText}>
        {r.ok ? `✅ result shape (${r.shape.join(', ')})` : '❌ RuntimeError: the shapes are not broadcastable'}
      </div>
    </Widget>
  );
}

/* =================================================================================== */
/* CNNs                                                                                */
/* =================================================================================== */

export function ConvShapeCalculator() {
  const [n, setN] = useState(32);
  const [k, setK] = useState(3);
  const [s, setS] = useState(1);
  const [p, setP] = useState(1);
  const [cin, setCin] = useState(3);
  const [cout, setCout] = useState(8);
  const o = M.convOut(n, k, s, p);
  return (
    <Widget title="Convolution output size and parameter count"
      hint="out = ⌊(n − F + 2P) / S⌋ + 1 and params = C_out · (C_in · F² + 1). With S = 1, P = (F − 1)/2 keeps the size ('same' padding). The parameter count never depends on n.">
      <div className={styles.controls}>
        <Slider label="input size n" color="data" value={n} min={4} max={256} step={1} onChange={setN} />
        <Slider label="kernel F" color="hyper" value={k} min={1} max={11} step={1} onChange={setK} />
        <Slider label="stride S" color="hyper" value={s} min={1} max={4} step={1} onChange={setS} />
        <Slider label="padding P" color="hyper" value={p} min={0} max={5} step={1} onChange={setP} />
        <Slider label="C_in" color="data" value={cin} min={1} max={256} step={1} onChange={setCin} />
        <Slider label="C_out" color="hyper" value={cout} min={1} max={512} step={1} onChange={setCout} />
      </div>
      <Readout items={[
        ['output', o.size >= 1 ? `(${cout}, ${o.size}, ${o.size})` : 'invalid (≤ 0)', 'pred'],
        ['(n − F + 2P) / S', o.raw - 1 === Math.floor(o.raw - 1) ? String(o.raw - 1) : `${(o.raw - 1).toFixed(2)} → floored`],
        ['parameters', M.convParams(cin, cout, k).toLocaleString('en-US'), 'param'],
        ['same size?', o.size === n ? 'yes' : 'no'],
      ]} />
      {!o.exact && o.size >= 1 && <div className={styles.note}>The kernel doesn't tile the input exactly: the last {(n - k + 2 * p) % s} row(s)/column(s) are skipped.</div>}
    </Widget>
  );
}

export function CnnShapeTracer({preset = 'SmallCNN'}) {
  const P = M.CNN_PRESETS[preset];
  const [size, setSize] = useState(P.input[1]);
  const t = M.traceCnn([P.input[0], size, size], P.layers);
  const max = Math.max(...t.rows.map((r) => r.params));
  return (
    <Widget title={`${preset}: shapes and parameters layer by layer`}
      hint="Change the image size: the conv parameters stay the same, but the flatten size and the first fully connected layer grow with the square of the image side.">
      <div className={styles.controls}>
        <Select label="input image" value={String(size)} onChange={(v) => setSize(parseInt(v, 10))}
          options={P.sizes.map((s) => [String(s), `${P.input[0]} × ${s} × ${s}`])} />
      </div>
      {t.error ? <div className={styles.badText}>{t.error}</div> : (
        <table className={styles.table}>
          <thead><tr><th>layer</th><th>output shape</th><th>params</th><th>share</th></tr></thead>
          <tbody>{t.rows.map((r, i) => (
            <tr key={i} className={r.params === max && max > 0 ? styles.hlRow : undefined}>
              <td>{r.name}</td><td><code>{r.shape}</code></td><td>{r.params ? r.params.toLocaleString('en-US') : '—'}</td>
              <td>{r.params ? `${((100 * r.params) / t.total).toFixed((100 * r.params) / t.total < 0.1 ? 3 : 1)}%` : ''}</td>
            </tr>
          ))}</tbody>
        </table>
      )}
      {!t.error && <Readout items={[['total parameters', t.total.toLocaleString('en-US'), 'param'], ['preset from', P.note]]} />}
    </Widget>
  );
}

/** Softmax with optional temperature, sampling and cross-entropy. */
export function SoftmaxPlayground({mode = 'temperature', labels}) {
  const isT = mode === 'temperature';
  const names = labels ? labels.split(',').map((s) => s.trim()) : (isT ? ['A', 'B', 'C', 'D'] : ['cat', 'dog', 'car', 'ship']);
  const [z, setZ] = useState(isT ? [2.5, 1.2, 0.5, -0.3] : [2.0, 1.0, 0.1, -1.0]);
  const [T, setT] = useState(1);
  const [y, setY] = useState(0);
  const [seed, setSeed] = useState(0);
  const p = M.softmax(z, isT ? T : 1);
  const counts = useMemo(() => (seed ? M.sampleCounts(p, 100, M.rng(seed)) : null), [seed, p.join()]); // eslint-disable-line
  const setZi = (i) => (v) => setZ(z.map((x, j) => (j === i ? v : x)));
  return (
    <Widget title={isT ? 'Temperature sampling: softmax(z / T)' : 'Softmax and cross-entropy'}
      hint={isT ? 'T < 1 sharpens towards the top logit (T → 0 is greedy); T > 1 flattens towards uniform. The order of the bars never changes.'
        : 'Cross-entropy is −log of the probability given to the TRUE class. It is small when that bar is tall, and grows without limit as it shrinks.'}>
      <div className={styles.controls}>
        {z.map((v, i) => (
          <Slider key={i} label={`logit ${names[i]}`} color="pred" value={v} min={-4} max={6} step={0.1} onChange={setZi(i)} fmt={(x) => x.toFixed(1)} />
        ))}
        {isT && <Slider label="temperature T" color="hyper" value={T} min={0.1} max={5} step={0.05} onChange={setT} fmt={f2} />}
        {!isT && <Select label="true class" value={String(y)} onChange={(v) => setY(parseInt(v, 10))} options={names.map((nm, i) => [String(i), nm])} />}
      </div>
      <Bars values={p} labels={names} max={1} highlight={isT ? p.indexOf(Math.max(...p)) : y} fmt={(v) => v.toFixed(3)} />
      <Readout items={isT ? [
        ['top probability', f3(Math.max(...p)), 'pred'],
        ['entropy', `${f3(M.entropy(p))} nats (uniform = ${f3(Math.log(z.length))})`],
      ] : [
        [`p(true = ${names[y]})`, f3(p[y]), 'pred'],
        ['cross-entropy −log p', f3(M.crossEntropy(p, y)), 'loss'],
        ['uniform guess', f3(Math.log(z.length)), 'loss'],
      ]} />
      {isT && (
        <div className={styles.row}>
          <button className={styles.button} onClick={() => setSeed(seed + 1)}>Sample 100 tokens</button>
          {counts && <span className={styles.seq}>{names.map((nm, i) => `${nm}: ${counts[i]}`).join(' · ')}</span>}
        </div>
      )}
    </Widget>
  );
}

/* =================================================================================== */
/* NLP                                                                                 */
/* =================================================================================== */

export function TfidfPlayground({docs = 'good movie\nnot good movie\nman bites dog\ndog bites man'}) {
  const [text, setText] = useState(docs);
  const ds = text.split('\n').map((s) => s.trim()).filter(Boolean);
  const r = useMemo(() => M.tfidf(ds), [text]); // eslint-disable-line
  return (
    <Widget title="TF-IDF and cosine similarity (scikit-learn defaults)"
      hint="Same rules as TfidfVectorizer(): lower-cased, words of 2+ characters, idf = ln((1 + n) / (1 + df)) + 1, rows scaled to length 1. Word order never matters: reorder a sentence and its row doesn't change.">
      <textarea className={styles.textarea} rows={Math.min(Math.max(ds.length, 3), 7)} value={text}
        onChange={(e) => setText(e.target.value)} />
      {ds.length > 0 && r.vocab.length > 0 && (
        <>
          <div className={styles.scroll}>
            <table className={styles.table}>
              <thead><tr><th>document</th>{r.vocab.map((w) => <th key={w}>{w}</th>)}</tr></thead>
              <tbody>{r.rows.map((row, i) => (
                <tr key={i}><td>{ds[i]}</td>{row.map((v, j) => <td key={j} className={v ? undefined : styles.zero}>{v ? v.toFixed(3) : '0'}</td>)}</tr>
              ))}</tbody>
            </table>
          </div>
          <div className={styles.scroll}>
            <table className={styles.table}>
              <thead><tr><th>cosine</th>{ds.map((_, j) => <th key={j}>#{j + 1}</th>)}</tr></thead>
              <tbody>{r.rows.map((a, i) => (
                <tr key={i}><td>#{i + 1}</td>{r.rows.map((b, j) => {
                  const c = M.cosine(a, b);
                  return <td key={j} className={i !== j && c > 0.999 ? styles.hlCell : undefined}>{c.toFixed(3)}</td>;
                })}</tr>
              ))}</tbody>
            </table>
          </div>
        </>
      )}
    </Widget>
  );
}

export function ContextWindowExplorer({sentence = 'i love deep learning'}) {
  const [s, setS] = useState(sentence);
  const [w, setW] = useState(1);
  const [i, setI] = useState(1);
  const {words, pairs} = M.contextPairs(s, w);
  const c = Math.min(i, Math.max(words.length - 1, 0));
  return (
    <Widget title="Word2Vec training pairs from a context window"
      hint="Each word is paired with every neighbour up to w positions away. Skip-gram learns centre → neighbour; CBOW averages the neighbours to predict the centre.">
      <div className={styles.controls}>
        <label className={styles.textIn}>sentence <input value={s} onChange={(e) => setS(e.target.value)} /></label>
        <Slider label="window w" color="hyper" value={w} min={1} max={4} step={1} onChange={setW} />
      </div>
      <div className={styles.words}>
        {words.map((wd, j) => (
          <button key={j} onClick={() => setI(j)}
            className={j === c ? styles.wordCentre : Math.abs(j - c) <= w ? styles.wordCtx : styles.word}>{wd}</button>
        ))}
      </div>
      <Readout items={[
        ['centre (click a word)', words[c] ?? '–', 'data'],
        ['its context', words.filter((_, j) => j !== c && Math.abs(j - c) <= w).join(', ') || '–', 'pred'],
        ['total (centre, context) pairs', String(pairs.length)],
      ]} />
      <div className={styles.seq}>{pairs.map(([a, b]) => `(${a}, ${b})`).join('  ')}</div>
    </Widget>
  );
}

/* =================================================================================== */
/* RNN / LSTM / GRU                                                                    */
/* =================================================================================== */

export function GradientDecayExplorer({gamma = 0.5, steps = 15, label = 'per-step factor γ'}) {
  const [g, setG] = useState(gamma);
  const [T, setT] = useState(steps);
  const [z, setZ] = useState(0);
  const vals = M.repeatedFactor(g, T);
  const last = vals[vals.length - 1];
  const logs = vals.map((v) => Math.log10(Math.max(v, 1e-30)));
  const lo = Math.min(...logs, -1), hi = Math.max(...logs, 0);
  return (
    <Widget title="How a gradient changes as it travels back through time"
      hint="Backprop through time multiplies the gradient by roughly the same factor once per step. Below 1 it vanishes, above 1 it explodes. A tanh slope 1 − tanh²(z) is part of that factor, and it is small when z is large.">
      <div className={styles.controls}>
        <Slider label={label} color="loss" value={g} min={0.1} max={1.6} step={0.01} onChange={setG} fmt={f2} />
        <Slider label="steps back" value={T} min={1} max={50} step={1} onChange={setT} />
        <Slider label="tanh input z" color="pred" value={z} min={-4} max={4} step={0.1} onChange={setZ} fmt={(v) => v.toFixed(1)} />
      </div>
      <Bars values={logs.map((v) => v - lo + 0.3)} max={Math.max(hi - lo + 0.3, 1)}
        colors={vals.map((v) => (v > 1 ? 'var(--eq-loss)' : 'var(--w-bar)'))} fmt={() => ''} height={150} />
      <Readout items={[
        [`γ^${T}`, f3(last), 'loss'],
        ['verdict', last < 1e-3 ? 'vanished: early steps learn nothing' : last > 1e3 ? 'exploded: clip the gradient' : 'still usable'],
        [`tanh slope at z = ${z.toFixed(1)}`, f3(M.tanhSlope(z)), 'pred'],
      ]} />
      <div className={styles.note}>Bars: gradient size on a log scale, one bar per step back from the loss (red = above 1, growing).</div>
    </Widget>
  );
}

const LSTM_PRESETS = {
  dim1: {cPrev: 0.9, f: 1, i: 0, cTilde: 0.3, o: 1, label: 'dim 1: keep'},
  dim2: {cPrev: 0.2, f: 0.9, i: 0.2, cTilde: 0.7, o: 1, label: 'dim 2: mostly keep'},
  dim3: {cPrev: -0.8, f: 0.1, i: 0.8, cTilde: 0.4, o: 1, label: 'dim 3: rewrite'},
  dim4: {cPrev: 0.5, f: 0, i: 1, cTilde: -0.6, o: 1, label: 'dim 4: replace'},
};

export function LstmCellPlayground() {
  const [s, setS] = useState(LSTM_PRESETS.dim2);
  const [k, setK] = useState(20);
  const set = (key) => (v) => setS({...s, [key]: v});
  const r = M.lstmStep(s);
  return (
    <Widget title="One LSTM memory dimension, gate by gate"
      hint="cₜ = f·cₜ₋₁ + i·c̃ₜ and hₜ = o·tanh(cₜ). The presets are the notebook's four demo dimensions. The last line shows how much memory survives k steps with the same forget gate: f^k.">
      <div className={styles.controls}>
        <Select label="preset" value="" onChange={(v) => v && setS(LSTM_PRESETS[v])}
          options={[['', 'choose a notebook dimension…'], ...Object.entries(LSTM_PRESETS).map(([key, v]) => [key, v.label])]} />
        <Slider label="cₜ₋₁ (old memory)" color="pred" value={s.cPrev} min={-1.5} max={1.5} step={0.05} onChange={set('cPrev')} fmt={f2} />
        <Slider label="f (forget gate)" color="pred" value={s.f} min={0} max={1} step={0.01} onChange={set('f')} fmt={f2} />
        <Slider label="i (input gate)" color="pred" value={s.i} min={0} max={1} step={0.01} onChange={set('i')} fmt={f2} />
        <Slider label="c̃ₜ (candidate)" color="pred" value={s.cTilde} min={-1} max={1} step={0.05} onChange={set('cTilde')} fmt={f2} />
        <Slider label="o (output gate)" color="pred" value={s.o} min={0} max={1} step={0.01} onChange={set('o')} fmt={f2} />
      </div>
      <Bars values={[s.cPrev, r.kept, r.written, r.c, r.h]} labels={['cₜ₋₁', 'f·cₜ₋₁', 'i·c̃ₜ', 'cₜ', 'hₜ']}
        colors={['var(--w-bar)', 'var(--w-bar)', 'var(--w-bar)', 'var(--w-hl)', 'var(--eq-pred)']}
        fmt={(v) => v.toFixed(2)} max={Math.max(1, ...[s.cPrev, r.kept, r.written, r.c].map(Math.abs))} />
      <Readout items={[
        ['kept f · cₜ₋₁', f3(r.kept), 'pred'], ['written i · c̃ₜ', f3(r.written), 'pred'],
        ['new memory cₜ', f3(r.c), 'pred'], ['shown hₜ = o · tanh(cₜ)', f3(r.h), 'pred'],
      ]} />
      <div className={styles.controls}>
        <Slider label="steps k" value={k} min={1} max={100} step={1} onChange={setK} />
      </div>
      <div className={styles.seq}>With f = {f2(s.f)} held for {k} steps, a memory keeps f^{k} = {f3(s.f ** k)} of its size (the gradient along the cell path scales the same way).</div>
    </Widget>
  );
}

const GRU_PRESETS = {
  dim1: {hPrev: 0.9, z: 0.95, hTilde: 0.1, label: 'dim 1: z = 0.95'},
  dim2: {hPrev: 0.2, z: 0.7, hTilde: 0.8, label: 'dim 2: z = 0.70'},
  dim3: {hPrev: -0.7, z: 0.2, hTilde: 0.4, label: 'dim 3: z = 0.20'},
  dim4: {hPrev: 0.5, z: 0.05, hTilde: -0.6, label: 'dim 4: z = 0.05'},
};

export function GruBlendPlayground() {
  const [s, setS] = useState(GRU_PRESETS.dim1);
  const set = (key) => (v) => setS({...s, [key]: v});
  const r = M.gruStep(s);
  const X = (v) => `${((v + 1) / 2) * 100}%`;
  return (
    <Widget title="GRU update: one dial between old state and candidate"
      hint="hₜ = z·hₜ₋₁ + (1 − z)·h̃ₜ (notebook / PyTorch convention): z near 1 keeps the old value, z near 0 takes the candidate. Presets are the notebook's four demo dimensions.">
      <div className={styles.controls}>
        <Select label="preset" value="" onChange={(v) => v && setS(GRU_PRESETS[v])}
          options={[['', 'choose a notebook dimension…'], ...Object.entries(GRU_PRESETS).map(([key, v]) => [key, v.label])]} />
        <Slider label="hₜ₋₁ (old)" color="data" value={s.hPrev} min={-1} max={1} step={0.05} onChange={set('hPrev')} fmt={f2} />
        <Slider label="h̃ₜ (candidate)" color="hyper" value={s.hTilde} min={-1} max={1} step={0.05} onChange={set('hTilde')} fmt={f2} />
        <Slider label="z (update gate)" color="pred" value={s.z} min={0} max={1} step={0.01} onChange={set('z')} fmt={f2} />
      </div>
      <div className={styles.track}>
        <span className={styles.trackOld} style={{left: X(s.hPrev)}} title="old" />
        <span className={styles.trackCand} style={{left: X(s.hTilde)}} title="candidate" />
        <span className={styles.trackNew} style={{left: X(r.h)}} title="new" />
        <span className={styles.trackTick} style={{left: '0%'}}>−1</span>
        <span className={styles.trackTick} style={{left: '50%'}}>0</span>
        <span className={styles.trackTick} style={{left: '100%'}}>1</span>
      </div>
      <Readout items={[
        ['z · hₜ₋₁', f3(r.kept), 'data'], ['(1 − z) · h̃ₜ', f3(r.added), 'hyper'], ['new hₜ', f3(r.h), 'pred'],
      ]} />
      <div className={styles.note}>Blue = old state, violet = candidate, green = new state.</div>
    </Widget>
  );
}

/* =================================================================================== */
/* Attention (notebook 14)                                                             */
/* =================================================================================== */

const vec2 = (v) => `[${v.map((x) => x.toFixed(3)).join(', ')}]`;

/** Raw relevance scores → softmax weights → weighted context vector. */
export function AttentionWeightsPlayground() {
  const [scores, setScores] = useState(M.ATT_SCORES);
  const [k, setK] = useState(1);
  const w = M.softmax(scores.map((s) => s * k), 1);
  const ctx = M.weightedSum(w, M.ATT_VALUES);
  const avg = M.weightedSum(M.ATT_TOKENS.map(() => 1 / M.ATT_TOKENS.length), M.ATT_VALUES);
  const setI = (i) => (v) => setScores(scores.map((x, j) => (j === i ? v : x)));
  return (
    <Widget title="Soft alignment: scores → softmax weights → context vector"
      hint="αᵢ = softmax(eᵢ) and c = Σ αᵢ vᵢ. Raise a token's score and its share grows at the expense of all the others; the weights always sum to 1. The sharpness slider multiplies every score: large values approach hard selection.">
      <div className={styles.controls}>
        {M.ATT_TOKENS.map((t, i) => (
          <Slider key={t} label={`score “${t}”`} color="pred" value={scores[i]} min={-3} max={5} step={0.1} onChange={setI(i)} fmt={(x) => x.toFixed(1)} />
        ))}
        <Slider label="sharpness ×" color="hyper" value={k} min={0.1} max={6} step={0.1} onChange={setK} fmt={(x) => x.toFixed(1)} />
      </div>
      <Bars values={w} labels={M.ATT_TOKENS} max={1} highlight={w.indexOf(Math.max(...w))} fmt={(v) => v.toFixed(3)} />
      <Readout items={[
        ['sum of weights', w.reduce((a, b) => a + b, 0).toFixed(3)],
        ['context c = Σ αᵢ vᵢ', vec2(ctx), 'pred'],
        ['plain average (no attention)', vec2(avg)],
      ]} />
      <div className={styles.note}>Value vectors (notebook toy, [positive-like, negation-like]): {M.ATT_TOKENS.map((t, i) => `${t} ${vec2(M.ATT_VALUES[i])}`).join(' · ')}</div>
    </Widget>
  );
}

/** Token-to-token attention matrix; click a row to see what that token attends to. */
export function SelfAttentionMatrix() {
  const [row, setRow] = useState(4);
  const [equal, setEqual] = useState(false);
  const n = M.ATT_TOKENS.length;
  const weights = equal ? M.ATT_TOKENS.map(() => 1 / n) : M.ATT_MATRIX[row];
  const out = M.weightedSum(weights, M.ATT_VALUES);
  return (
    <Widget title="Self-attention matrix: every token looks at every token"
      hint="Row = the token asking (query), column = the token being looked at. Each row is a softmax distribution, so it sums to 1; columns don't have to. The new vector for a token is its row times the value vectors.">
      <div className={styles.scroll}>
        <table className={styles.table}>
          <thead><tr><th>query ↓ / attends to →</th>{M.ATT_TOKENS.map((t) => <th key={t}>{t}</th>)}<th>row sum</th></tr></thead>
          <tbody>{M.ATT_MATRIX.map((r, i) => (
            <tr key={i} onClick={() => { setRow(i); setEqual(false); }} style={{cursor: 'pointer'}} className={i === row && !equal ? styles.hlRow : undefined}>
              <td><strong>{M.ATT_TOKENS[i]}</strong></td>
              {r.map((v, j) => (
                <td key={j} style={{background: `rgba(56, 217, 169, ${(v * 0.85).toFixed(2)})`}}>{v.toFixed(2)}</td>
              ))}
              <td>{r.reduce((a, b) => a + b, 0).toFixed(2)}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
      <div className={styles.row}>
        <button className={styles.button} onClick={() => setEqual(!equal)}>{equal ? 'Back to learned-style attention' : 'Compare: equal attention (0.20 each)'}</button>
      </div>
      <Bars values={weights} labels={M.ATT_TOKENS} max={1} highlight={weights.indexOf(Math.max(...weights))} fmt={(v) => v.toFixed(2)} height={140} />
      <Readout items={[
        [`“${M.ATT_TOKENS[row]}” before`, vec2(M.ATT_VALUES[row]), 'data'],
        [equal ? 'after equal attention' : `“${M.ATT_TOKENS[row]}” after attention`, vec2(out), 'pred'],
      ]} />
      <div className={styles.note}>Click a row. “good” starts as [1.000, 0.100] (mostly positive); after attending to “not” it becomes [0.275, 0.580]: the negation feature now dominates.</div>
    </Widget>
  );
}

/* ------------------------------------------------------------------ */
/* Full self-attention (notebook 15)                                   */
/* ------------------------------------------------------------------ */
const SA_STAGES = [
  {key: 'X', label: 'X', shape: '5 × 4', what: 'Input embeddings: one row per token, no interaction yet.'},
  {key: 'Q', label: 'Q = XW_Q', shape: '5 × 3', what: 'Queries: what each token is looking for.'},
  {key: 'K', label: 'K = XW_K', shape: '5 × 3', what: 'Keys: what each token offers for matching.'},
  {key: 'V', label: 'V = XW_V', shape: '5 × 3', what: 'Values: the information each token passes on.'},
  {key: 'S', label: 'S = QKᵀ', shape: '5 × 5', what: 'Raw scores: unbounded, can be negative, rows do not sum to 1.'},
  {key: 'Ss', label: 'S / √d_k', shape: '5 × 5', what: 'Scaled scores (d_k = 3, so divide by 1.732): same ranking, smaller magnitude.'},
  {key: 'A', label: 'A = softmax(·)', shape: '5 × 5', what: 'Attention weights: row-wise softmax, every row sums to 1.'},
  {key: 'Z', label: 'Z = AV', shape: '5 × 3', what: 'Contextual outputs: each row is a weighted mix of all value rows.'},
];

/** Step through every matrix of scaled dot-product attention; click a cell to see how it is computed. */
export function AttentionPipelineExplorer() {
  const st = useMemo(() => M.selfAttentionStages(), []);
  const [stage, setStage] = useState(4);
  const [cell, setCell] = useState([4, 3]);
  const s = SA_STAGES[stage];
  const mat = st[s.key];
  const square = mat[0].length === 5;
  const [i, j] = [Math.min(cell[0], 4), Math.min(cell[1], mat[0].length - 1)];
  const T = M.SA_TOKENS;
  const fmt = (v) => (Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(1) : v.toFixed(3));
  const maxAbs = Math.max(...mat.flat().map(Math.abs));
  const explain = (() => {
    if (s.key === 'S') {
      const terms = st.Q[i].map((q, l) => `${q.toFixed(3)}×${st.K[j][l].toFixed(3)}`);
      return `S[${T[i]}, ${T[j]}] = q_${T[i]} · k_${T[j]} = ${terms.join(' + ')} = ${st.S[i][j].toFixed(4)}`;
    }
    if (s.key === 'Ss') return `${st.S[i][j].toFixed(4)} / √3 = ${st.Ss[i][j].toFixed(4)}`;
    if (s.key === 'A') {
      const den = st.Ss[i].reduce((a, v) => a + Math.exp(v), 0);
      return `A[${T[i]}, ${T[j]}] = exp(${st.Ss[i][j].toFixed(3)}) / Σ exp(row ${T[i]}) = ${Math.exp(st.Ss[i][j]).toFixed(4)} / ${den.toFixed(4)} = ${fmt(st.A[i][j])}`;
    }
    if (s.key === 'Z') {
      const terms = st.A[i].map((a, m) => `${fmt(a)}×${st.V[m][j].toFixed(3)}`);
      return `z_${T[i]}[${j}] = Σ A[${T[i]}, m] · V[m, ${j}] = ${terms.join(' + ')} = ${st.Z[i][j].toFixed(4)}`;
    }
    if (s.key === 'X') return `x_${T[i]}[${j}] = ${st.X[i][j].toFixed(4)} (random embedding, np.random.seed(42))`;
    const W = {Q: M.SA_WQ, K: M.SA_WK, V: M.SA_WV}[s.key];
    const terms = st.X[i].map((x, l) => `${x.toFixed(3)}×${W[l][j].toFixed(3)}`);
    return `${s.key}[${T[i]}, ${j}] = x_${T[i]} · column ${j} of W_${s.key} = ${terms.join(' + ')} = ${mat[i][j].toFixed(4)}`;
  })();
  return (
    <Widget title="Scaled dot-product attention, one matrix at a time"
      hint="These are the notebook's real matrices (np.random.seed(42), d_model = 4, d_k = 3). Pick a stage, then click any cell to see the exact arithmetic behind it. The weights are random and untrained, so the pattern has no linguistic meaning; the mechanism is what matters.">
      <div className={styles.row} style={{flexWrap: 'wrap', gap: '0.4rem'}}>
        {SA_STAGES.map((x, k) => (
          <button key={x.key} className={styles.button} onClick={() => setStage(k)}
            style={k === stage ? {outline: '2px solid var(--ifm-color-primary)', fontWeight: 700} : {opacity: 0.75}}>{x.label}</button>
        ))}
      </div>
      <div className={styles.note}><strong>{s.label}</strong> · shape {s.shape} · {s.what}</div>
      <div className={styles.scroll}>
        <table className={styles.table}>
          <thead><tr><th>{square ? 'query ↓ / key →' : 'token ↓ / feature →'}</th>{mat[0].map((_, c) => <th key={c}>{square ? T[c] : c}</th>)}{square && <th>row sum</th>}</tr></thead>
          <tbody>{mat.map((r, ri) => (
            <tr key={ri}>
              <td><strong>{T[ri]}</strong></td>
              {r.map((v, ci) => {
                const a = s.key === 'A' ? v : Math.abs(v) / maxAbs;
                const neg = v < 0 && s.key !== 'A';
                return (
                  <td key={ci} onClick={() => setCell([ri, ci])}
                    style={{cursor: 'pointer', background: neg ? `rgba(240, 101, 149, ${(a * 0.6).toFixed(2)})` : `rgba(56, 217, 169, ${(a * 0.75).toFixed(2)})`,
                      outline: ri === i && ci === j ? '2px solid #fab005' : 'none'}}>{fmt(v)}</td>
                );
              })}
              {square && <td>{r.reduce((x, y) => x + y, 0).toFixed(3)}</td>}
            </tr>
          ))}</tbody>
        </table>
      </div>
      <div className={styles.note} style={{fontFamily: 'var(--ifm-font-family-monospace)', fontSize: '0.8rem'}}>{explain}</div>
    </Widget>
  );
}

/** Same normalized similarities, bigger d_k: raw softmax saturates, scaled softmax does not. */
export function ScalingSharpnessPlayground() {
  const [logDk, setLogDk] = useState(Math.log2(3));
  const dk = Math.round(2 ** logDk);
  const raw = M.SA_GOOD_UNIT.map((v) => v * Math.sqrt(dk));
  const pRaw = M.softmax(raw);
  const pScaled = M.softmax(raw.map((v) => v / Math.sqrt(dk)));
  const T = M.SA_TOKENS;
  return (
    <Widget title="Why divide by √d_k: softmax sharpness as d_k grows"
      hint="The row is the notebook's 'good' scores. Dot products of d_k-dimensional vectors have a standard deviation that grows like √d_k, so the same relative similarities arrive as bigger raw numbers. Softmax on raw scores then sharpens, squeezing the small weights to zero; dividing by √d_k keeps the distribution the same at every size.">
      <div className={styles.controls}>
        <Slider label="d_k (query/key size)" color="hyper" value={logDk} min={0} max={9} step={0.05} onChange={setLogDk} fmt={() => String(dk)} />
      </div>
      <div className={styles.note}>raw scores q·k for “good” at d_k = {dk}: [{raw.map((v) => v.toFixed(2)).join(', ')}] (std grows like √d_k = {Math.sqrt(dk).toFixed(2)})</div>
      <div className={styles.note}><strong>softmax(raw scores)</strong></div>
      <Bars values={pRaw} labels={T} max={1} highlight={pRaw.indexOf(Math.max(...pRaw))} fmt={(v) => v.toFixed(3)} height={130} />
      <div className={styles.note}><strong>softmax(raw / √d_k)</strong></div>
      <Bars values={pScaled} labels={T} max={1} highlight={pScaled.indexOf(Math.max(...pScaled))} fmt={(v) => v.toFixed(3)} height={130} />
      <Readout items={[
        ['largest weight, raw', Math.max(...pRaw).toFixed(4), 'loss'],
        ['largest weight, scaled', Math.max(...pScaled).toFixed(4), 'pred'],
        ['tokens with weight above 0.01 (raw / scaled)', `${pRaw.filter((p) => p > 0.01).length} / ${pScaled.filter((p) => p > 0.01).length}`],
      ]} />
      <div className={styles.note}>At d_k = 3 the raw row is exactly the notebook's S[good] = [−1.605, −1.493, 3.856, −8.661, 3.918]. Push d_k up (the original Transformer uses d_k = 64): the raw weights for “The” and “movie” fall from about 0.02 to 0.000, and “good” pulls away from “was” (0.571 vs 0.429 at 64, 0.691 vs 0.309 at 512), even though the underlying similarities never changed. The scaled row stays at 0.020, 0.021, 0.471, 0.000, 0.488 at every d_k.</div>
    </Widget>
  );
}

/* ------------------------------------------------------------------ */
/* Multi-head attention (notebook 16)                                  */
/* ------------------------------------------------------------------ */
function MiniHeat({M: mat, labels, row, onRow, color = '56, 217, 169'}) {
  return (
    <div className={styles.scroll}>
      <table className={styles.table}>
        <thead><tr><th>q ↓ / k →</th>{labels.map((t) => <th key={t}>{t}</th>)}</tr></thead>
        <tbody>{mat.map((r, i) => (
          <tr key={i} onClick={() => onRow(i)} style={{cursor: 'pointer'}} className={i === row ? styles.hlRow : undefined}>
            <td><strong>{labels[i]}</strong></td>
            {r.map((v, j) => <td key={j} style={{background: `rgba(${color}, ${(v * 0.85).toFixed(2)})`}}>{v.toFixed(2)}</td>)}
          </tr>
        ))}</tbody>
      </table>
    </div>
  );
}

/** Two heads on the same sentence: different attention, then concat, then W_O mixes them. */
export function MultiHeadExplorer() {
  const st = useMemo(() => M.multiHeadStages(), []);
  const [row, setRow] = useState(4);
  const T = M.SA_TOKENS;
  const v = (a, d = 3) => `[${a.map((x) => x.toFixed(d)).join(', ')}]`;
  return (
    <Widget title="Multi-head attention: two heads, one sentence"
      hint="Notebook 16's real numbers: X is 5 × 4, each head projects to d_k = 2 with its own W_Q, W_K, W_V, attends, and outputs 5 × 2. The two outputs are placed side by side (5 × 4) and W_O (4 × 4) mixes them. Weights are random and untrained: compare the two patterns, don't read meaning into them.">
      <div className={styles.row} style={{gap: '1rem', flexWrap: 'wrap', alignItems: 'flex-start'}}>
        <div style={{flex: '1 1 300px'}}>
          <div className={styles.note}><strong>Head 1</strong>: A⁽¹⁾ = softmax(Q⁽¹⁾K⁽¹⁾ᵀ / √2)</div>
          <MiniHeat M={st.heads[0].A} labels={T} row={row} onRow={setRow} />
        </div>
        <div style={{flex: '1 1 300px'}}>
          <div className={styles.note}><strong>Head 2</strong>: A⁽²⁾ = softmax(Q⁽²⁾K⁽²⁾ᵀ / √2)</div>
          <MiniHeat M={st.heads[1].A} labels={T} row={row} onRow={setRow} color="250, 176, 5" />
        </div>
      </div>
      <div className={styles.note}>Click a row. For “{T[row]}”: head 1 looks most at “{T[st.heads[0].A[row].indexOf(Math.max(...st.heads[0].A[row]))]}”, head 2 most at “{T[st.heads[1].A[row].indexOf(Math.max(...st.heads[1].A[row]))]}”.</div>
      <Readout items={[
        [`head₁ output for “${T[row]}” (5 × 2 row)`, v(st.heads[0].out[row]), 'pred'],
        [`head₂ output for “${T[row]}”`, v(st.heads[1].out[row]), 'pred'],
        ['Concat → H row (side by side, not mixed)', v(st.H[row]), 'pred'],
        ['× W_O → final Z row (mixed, d_model = 4)', v(st.Z[row]), 'pred'],
      ]} />
    </Widget>
  );
}

const HEAD_PRESETS = [
  ['Notebook toy', 4, 2], ['Transformer base', 512, 8], ['Transformer big', 1024, 16], ['BERT-base', 768, 12],
];
const TABLE3_A = {1: [512, 24.9, 5.29], 4: [128, 25.5, 5.0], 8: [64, 25.8, 4.92], 16: [32, 25.8, 4.91], 32: [16, 25.4, 5.01]};

/** d_model and h → d_k, parameter count, and the paper's Table 3 (A) result when it applies. */
export function HeadSplitCalculator() {
  const [dModel, setDModel] = useState(512);
  const [logH, setLogH] = useState(3);
  const h = 2 ** logH;
  const s = M.headSplit(dModel, h);
  const t3 = dModel === 512 ? TABLE3_A[h] : null;
  const fmt = (n) => n.toLocaleString('en-US');
  return (
    <Widget title="Splitting d_model into h heads"
      hint="d_k = d_model / h. Each head gets a smaller slice, so the total projection size (and compute) stays the same as one full-width head: 4·d_model² weights without biases. More heads = more, smaller attention patterns; the paper found 8 heads of 64 best for d_model = 512.">
      <div className={styles.row} style={{flexWrap: 'wrap', gap: '0.4rem'}}>
        {HEAD_PRESETS.map(([name, d, hh]) => (
          <button key={name} className={styles.button} onClick={() => { setDModel(d); setLogH(Math.log2(hh)); }}
            style={dModel === d && h === hh ? {outline: '2px solid var(--ifm-color-primary)', fontWeight: 700} : {opacity: 0.75}}>{name}</button>
        ))}
      </div>
      <div className={styles.controls}>
        <Slider label="d_model" color="hyper" value={dModel} min={4} max={1024} step={4} onChange={setDModel} fmt={(x) => String(x)} />
        <Slider label="h (heads)" color="hyper" value={logH} min={0} max={5} step={1} onChange={setLogH} fmt={() => String(h)} />
      </div>
      {!s.valid && <div className={styles.note} style={{color: '#f06595'}}>d_model = {dModel} is not divisible by h = {h}: implementations reject this (d_k must be a whole number).</div>}
      <Readout items={[
        ['d_k = d_v = d_model / h', s.valid ? String(s.dk) : (dModel / h).toFixed(2), 'hyper'],
        ['W_Q, W_K, W_V for all heads', s.valid ? fmt(s.proj) : '—', 'param'],
        ['W_O (h·d_v × d_model)', s.valid ? fmt(s.wo) : '—', 'param'],
        ['total MHA weights', s.valid ? fmt(s.total) : '—', 'param'],
        ['one full-width head would use', fmt(s.singleFull)],
      ]} />
      {t3 && <div className={styles.note}>Paper, Table 3 row (A), d_model = 512, compute held constant: h = {h}, d_k = {t3[0]} → dev BLEU <strong>{t3[1]}</strong>, perplexity {t3[2]}. (h = 1: 24.9 · h = 4: 25.5 · h = 8: 25.8 · h = 16: 25.8 · h = 32: 25.4)</div>}
    </Widget>
  );
}

/* ------------------------------------------------------------------ */
/* The Transformer, paper Figure 1 (clickable)                         */
/* ------------------------------------------------------------------ */
// Box positions in the 920 × 1120 viewBox of /img/deep-learning/diagrams/transformer-figure-1.svg
const FIG1_W = 920, FIG1_H = 1120;
const FIG1_BOXES = [
  {k: 'outprob', x: 650, y: 73, w: 170, h: 52, title: 'Output Probabilities', sec: '§3.4, §6.1',
    what: 'A probability for every vocabulary token: the model’s prediction for the next output token at each position. Training maximises the probability of the true next token (cross-entropy with label smoothing 0.1); generation picks from it step by step (the paper used beam search, beam size 4, length penalty 0.6).',
    f: String.raw`\begin{aligned} &P(\data{y_i} = t \mid \data{x},\, \data{y_{<i}}) \\ &\quad = \operatorname{softmax}(\pred{\text{logits}_i})_t \end{aligned}`, base: 'n_out × 37,000 (BPE vocabulary)', toy: '6 × 11'},
  {k: 'softmax', x: 650, y: 175, w: 160, h: 30, title: 'Softmax', sec: '§3.4',
    what: 'Turns the Linear layer’s scores (logits) into probabilities that are positive and sum to 1 over the vocabulary, separately at each position.',
    f: String.raw`\operatorname{softmax}(\pred{z})_i = \dfrac{e^{\pred{z_i}}}{\sum_j e^{\pred{z_j}}}`, base: 'n_out × 37,000', toy: '6 × 11'},
  {k: 'linear', x: 650, y: 245, w: 160, h: 30, title: 'Linear', sec: '§3.4',
    what: 'Projects each decoder output vector (d_model wide) to one score per vocabulary token. The paper shares this weight matrix with the two embedding layers (weight tying).',
    f: String.raw`\begin{gathered} \pred{\text{logits}} = \pred{y}\,\param{E}^\top \\ \param{E}\text{: shared embedding matrix} \end{gathered}`, base: '512 → 37,000', toy: '64 → 11'},
  {k: 'dec_an3', x: 650, y: 365, w: 190, h: 28, title: 'Add & Norm (decoder, after Feed Forward)', sec: '§3.1, §5.4',
    what: 'Residual connection plus layer normalization around the decoder’s feed-forward sub-layer: the sub-layer output (after dropout) is added to its input, then each position is normalized.',
    f: String.raw`\begin{aligned} \pred{y} &= \operatorname{LayerNorm}\big(\data{x} + \operatorname{Dropout}(\pred{s})\big) \\ \pred{s} &= \operatorname{FFN}(\data{x}) \end{aligned}`, base: 'n_out × 512', toy: '6 × 64'},
  {k: 'dec_ff', x: 650, y: 445, w: 190, h: 58, title: 'Feed Forward (decoder)', sec: '§3.3',
    what: 'The same position-wise two-layer network as in the encoder (different weights): applied to each position separately and identically.',
    f: String.raw`\operatorname{FFN}(\data{x}) = \max(0,\, \data{x}\param{W_1} + \param{b_1})\,\param{W_2} + \param{b_2}`, base: '512 → 2048 → 512', toy: '64 → 128 → 64'},
  {k: 'dec_an2', x: 650, y: 525, w: 190, h: 28, title: 'Add & Norm (decoder, after encoder–decoder attention)', sec: '§3.1',
    what: 'Residual connection plus layer normalization around the encoder–decoder attention sub-layer.',
    f: String.raw`\begin{aligned} \pred{y} &= \operatorname{LayerNorm}\big(\data{x} + \operatorname{Dropout}(\pred{s})\big) \\ \pred{s} &= \operatorname{MultiHead}(\data{x}, \pred{z}, \pred{z}) \end{aligned}`, base: 'n_out × 512', toy: '6 × 64'},
  {k: 'dec_cross', x: 650, y: 590, w: 190, h: 58, title: 'Multi-Head Attention (encoder–decoder)', sec: '§3.2.3',
    what: 'Queries come from the decoder (the arrow from below); keys and values come from the encoder output (the two arrows from the left). Every output position can look at every input position: this is where the translation alignment lives.',
    f: String.raw`\begin{gathered} \operatorname{MultiHead}(\pred{Q}, \pred{K}, \pred{V}) \\ \pred{Q}\text{: decoder}, \quad \pred{K} = \pred{V}\text{: encoder output} \end{gathered}`, base: 'scores n_out × n_in, 8 heads', toy: '6 × 6, 4 heads (learns an anti-diagonal for reversal)'},
  {k: 'dec_an1', x: 650, y: 730, w: 190, h: 28, title: 'Add & Norm (decoder, after masked attention)', sec: '§3.1',
    what: 'Residual connection plus layer normalization around the masked self-attention sub-layer.',
    f: String.raw`\begin{aligned} \pred{y} &= \operatorname{LayerNorm}\big(\data{x} + \operatorname{Dropout}(\pred{s})\big) \\ \pred{s} &= \operatorname{MaskedMultiHead}(\data{x}) \end{aligned}`, base: 'n_out × 512', toy: '6 × 64'},
  {k: 'dec_masked', x: 650, y: 800, w: 190, h: 74, title: 'Masked Multi-Head Attention', sec: '§3.1, §3.2.3',
    what: 'Self-attention over the output tokens, with every future position masked to −∞ before the softmax, so position i only sees positions ≤ i. Together with the right shift, this keeps the model auto-regressive and lets training predict all positions at once.',
    f: String.raw`\begin{gathered} \operatorname{softmax}\!\Big(\dfrac{\pred{Q}\pred{K}^\top}{\sqrt{\hyper{d_k}}} + M\Big)\pred{V} \\ M_{ij} = -\infty \text{ for } j > i \end{gathered}`, base: 'n_out × n_out, 8 heads', toy: '6 × 6 lower-triangular, 4 heads'},
  {k: 'outputs', x: 650, y: 1091, w: 170, h: 48, title: 'Outputs (shifted right)', sec: '§3, §3.1',
    what: 'The target sentence moved one position to the right with a start token in front, so that the decoder input at position i is the true token i−1 and the prediction at position i is token i (teacher forcing during training; its own previous predictions at generation time).',
    f: String.raw`\begin{aligned} \text{input: } &[\langle s\rangle, \data{y_1}, \ldots, \data{y_{m-1}}] \\ \text{predict: } &[\data{y_1}, \ldots, \data{y_m}] \end{aligned}`, base: 'n_out token ids', toy: '[10, 9, 5, 1, 4, 1] for target [9, 5, 1, 4, 1, 3]'},
  {k: 'dec_emb', x: 650, y: 995, w: 190, h: 52, title: 'Output Embedding', sec: '§3.4',
    what: 'Learned lookup table turning each output token id into a d_model vector, multiplied by √d_model. Shared with the input embedding and the pre-softmax Linear layer.',
    f: String.raw`\param{E}[\data{\text{token}}] \cdot \sqrt{\hyper{d_{model}}}`, base: '37,000 × 512 (shared)', toy: '11 × 64 (shared)'},
  {k: 'dec_pe', x: 726, y: 925, w: 64, h: 44, title: 'Positional Encoding (decoder)', sec: '§3.5',
    what: 'The same sinusoidal position vectors as the encoder, added (the ⊕) to the output embeddings so the decoder knows token order.',
    f: String.raw`\begin{aligned} PE_{(pos,2i)} &= \sin\!\big(pos/10000^{2i/\hyper{d_{model}}}\big) \\ PE_{(pos,2i+1)} &= \cos\!\big(pos/10000^{2i/\hyper{d_{model}}}\big) \end{aligned}`, base: 'n_out × 512', toy: '6 × 64'},
  {k: 'enc_an2', x: 270, y: 580, w: 190, h: 28, title: 'Add & Norm (encoder, after Feed Forward)', sec: '§3.1',
    what: 'Residual connection plus layer normalization around the encoder’s feed-forward sub-layer. The output of the last encoder layer is the memory read by every decoder layer.',
    f: String.raw`\begin{aligned} \pred{y} &= \operatorname{LayerNorm}\big(\data{x} + \operatorname{Dropout}(\pred{s})\big) \\ \pred{s} &= \operatorname{FFN}(\data{x}) \end{aligned}`, base: 'n_in × 512', toy: '6 × 64'},
  {k: 'enc_ff', x: 270, y: 660, w: 190, h: 58, title: 'Feed Forward (encoder)', sec: '§3.3',
    what: 'Two linear layers with a ReLU, applied to each position separately and identically (like two convolutions with kernel size 1). Attention mixes across positions; this transforms within each position.',
    f: String.raw`\operatorname{FFN}(\data{x}) = \max(0,\, \data{x}\param{W_1} + \param{b_1})\,\param{W_2} + \param{b_2}`, base: '512 → 2048 → 512', toy: '64 → 128 → 64'},
  {k: 'enc_an1', x: 270, y: 745, w: 190, h: 28, title: 'Add & Norm (encoder, after attention)', sec: '§3.1, §5.4',
    what: 'Add: the residual connection (the line going around the attention box) adds the sub-layer’s input back to its output. Norm: layer normalization of each position’s vector. Dropout 0.1 is applied to the sub-layer output before the addition.',
    f: String.raw`\begin{aligned} \pred{y} &= \operatorname{LayerNorm}\big(\data{x} + \operatorname{Dropout}(\pred{s})\big) \\ \pred{s} &= \operatorname{Sublayer}(\data{x}) \end{aligned}`, base: 'n_in × 512', toy: '6 × 64'},
  {k: 'enc_mha', x: 270, y: 805, w: 190, h: 58, title: 'Multi-Head Attention (encoder)', sec: '§3.2.2, §3.2.3',
    what: 'Encoder self-attention: the three arrows are Q, K and V, all coming from the same place (the previous layer), so every input position attends to every input position. 8 heads of size 64.',
    f: String.raw`\begin{aligned} &\operatorname{Concat}(\pred{\text{head}_1}, \ldots, \pred{\text{head}_h})\,\param{W^O} \\ &\pred{\text{head}_i} = \operatorname{Attention}(\data{X}\param{W_i^Q}, \data{X}\param{W_i^K}, \data{X}\param{W_i^V}) \end{aligned}`, base: 'n_in × 512, 8 heads × 64', toy: '6 × 64, 4 heads × 16'},
  {k: 'enc_nx', x: 95, y: 700, w: 70, h: 40, title: 'N× (encoder)', sec: '§3.1',
    what: 'The grey box is one encoder layer; N identical layers (with separate weights) are stacked, each feeding the next. N = 6 in the paper.',
    f: String.raw`\data{x} \leftarrow \operatorname{EncoderLayer}_l(\data{x}), \quad l = 1, \ldots, \hyper{N}`, base: 'N = 6', toy: 'N = 2'},
  {k: 'dec_nx', x: 812, y: 600, w: 70, h: 40, title: 'N× (decoder)', sec: '§3.1',
    what: 'N identical decoder layers are stacked; every one of them reads the same final encoder output in its encoder–decoder attention. N = 6 in the paper.',
    f: String.raw`\data{y} \leftarrow \operatorname{DecoderLayer}_l(\data{y}, \pred{z}), \quad l = 1, \ldots, \hyper{N}`, base: 'N = 6', toy: 'N = 2'},
  {k: 'enc_pe', x: 196, y: 925, w: 64, h: 44, title: 'Positional Encoding (encoder)', sec: '§3.5',
    what: 'Attention has no notion of order, so a position vector of the same size d_model is added (the ⊕) to every input embedding: sines and cosines of geometrically spaced frequencies (wavelengths from 2π to 10000·2π).',
    f: String.raw`\begin{aligned} PE_{(pos,2i)} &= \sin\!\big(pos/10000^{2i/\hyper{d_{model}}}\big) \\ PE_{(pos,2i+1)} &= \cos\!\big(pos/10000^{2i/\hyper{d_{model}}}\big) \end{aligned}`, base: 'n_in × 512', toy: '6 × 64'},
  {k: 'enc_emb', x: 270, y: 995, w: 190, h: 52, title: 'Input Embedding', sec: '§3.4',
    what: 'Learned lookup table turning each input token id into a d_model vector, multiplied by √d_model. The paper shares it with the Output Embedding and the pre-softmax Linear layer.',
    f: String.raw`\param{E}[\data{\text{token}}] \cdot \sqrt{\hyper{d_{model}}}`, base: '37,000 × 512 (shared)', toy: '11 × 64 (shared)'},
  {k: 'inputs', x: 270, y: 1090, w: 120, h: 30, title: 'Inputs', sec: '§3, §5.1',
    what: 'The source sentence as token ids (byte-pair encoding in the paper). The encoder reads all of them at once, not one by one.',
    f: String.raw`(\data{x_1}, \ldots, \data{x_n})`, base: 'n_in token ids, ≈25,000 source tokens per batch', toy: '[3, 1, 4, 1, 5, 9]'},
];

/** Click any box of Figure 1 to see what it does, its formula, shapes and paper section. */
export function Figure1Explorer() {
  const src = useBaseUrl('/img/deep-learning/diagrams/transformer-figure-1.svg');
  const [sel, setSel] = useState('enc_mha');
  const b = FIG1_BOXES.find((x) => x.k === sel);
  return (
    <Widget title="Figure 1, clickable: what every box does"
      hint="Click a box in the figure (or a name below). Shapes are given for the paper's base model (d_model = 512, h = 8, d_ff = 2048, N = 6) and for this page's toy model (d_model = 64, h = 4, d_ff = 128, N = 2, 6 tokens).">
      <div className={styles.row} style={{gap: '1rem', flexWrap: 'wrap', alignItems: 'flex-start'}}>
        <div style={{position: 'relative', flex: '1 1 300px', maxWidth: 440}}>
          <img src={src} alt="Transformer architecture, paper Figure 1, clickable" style={{width: '100%', display: 'block', borderRadius: 8}} />
          {FIG1_BOXES.map((x) => (
            <button key={x.k} onClick={() => setSel(x.k)} aria-label={x.title} title={x.title}
              style={{position: 'absolute', left: `${((x.x - x.w / 2) / FIG1_W) * 100}%`, top: `${((x.y - x.h / 2) / FIG1_H) * 100}%`,
                width: `${(x.w / FIG1_W) * 100}%`, height: `${(x.h / FIG1_H) * 100}%`, background: x.k === sel ? 'rgba(250, 176, 5, 0.28)' : 'transparent',
                border: x.k === sel ? '2px solid #fab005' : '1px dashed rgba(127,127,127,0.0)', borderRadius: 6, cursor: 'pointer', padding: 0}} />
          ))}
        </div>
        <div style={{flex: '1 1 280px', minWidth: 0}}>
          <div className={styles.note}><strong style={{fontSize: '1.05rem'}}>{b.title}</strong> · paper {b.sec}</div>
          <div className={styles.note}>{b.what}</div>
          <div className={styles.note} style={{padding: '0.4rem 0.6rem', borderRadius: 8, background: 'var(--ifm-color-emphasis-100)'}}>
            <div style={{fontSize: '0.75rem', fontWeight: 700, color: 'var(--eq-pred)'}}>formula</div>
            <Tex>{b.f}</Tex>
          </div>
          <Readout items={[
            ['paper base model', b.base, 'hyper'],
            ['toy model on this page', b.toy, 'data'],
          ]} />
          <div className={styles.row} style={{flexWrap: 'wrap', gap: '0.3rem'}}>
            {FIG1_BOXES.slice().reverse().map((x) => (
              <button key={x.k} className={styles.button} onClick={() => setSel(x.k)}
                style={{fontSize: '0.72rem', padding: '0.15rem 0.45rem', ...(x.k === sel ? {outline: '2px solid var(--ifm-color-primary)'} : {opacity: 0.7})}}>{x.title.replace(/ \((encoder|decoder)(?:, ([^)]*))?\)/, (m, side, rest) => ` · ${side.slice(0, 3)}${rest ? ', ' + rest : ''}`)}</button>
            ))}
          </div>
        </div>
      </div>
    </Widget>
  );
}

/** Sinusoidal positional encoding: the vector for one position, the waves, and similarity by distance. */
export function PositionalEncodingExplorer() {
  const [d, setD] = useState(512);
  const [pos, setPos] = useState(10);
  const pe = M.positionalEncoding(pos, d);
  const shown = pe.slice(0, Math.min(d, 16));
  const sim = (k) => M.dot(M.positionalEncoding(pos, d), M.positionalEncoding(pos + k, d)) / (d / 2);
  const wave = (i2) => (p) => Math.sin(p / 10000 ** (i2 / d));
  const pairs = [0, Math.round(d / 8) * 2, Math.round(d / 4) * 2].filter((v, i, a) => a.indexOf(v) === i && v < d);
  return (
    <Widget title="Positional Encoding: a unique, smoothly varying vector per position"
      hint="Each pair of dimensions is a sine and a cosine of the same angle (formula above). Low dimensions oscillate fast, high dimensions slowly (wavelengths 2π … 10000·2π). Every PE vector has the same length √(d/2), and the dot product of two positions depends only on how far apart they are.">
      <Tex>{String.raw`\begin{aligned} PE_{(pos,2i)} &= \sin\!\big(pos/10000^{2i/\hyper{d_{model}}}\big) \\ PE_{(pos,2i+1)} &= \cos\!\big(pos/10000^{2i/\hyper{d_{model}}}\big) \end{aligned}`}</Tex>
      <div className={styles.row} style={{flexWrap: 'wrap', gap: '0.4rem'}}>
        {[4, 16, 64, 512].map((v) => (
          <button key={v} className={styles.button} onClick={() => setD(v)}
            style={v === d ? {outline: '2px solid var(--ifm-color-primary)', fontWeight: 700} : {opacity: 0.75}}>d_model = {v}</button>
        ))}
      </div>
      <div className={styles.controls}>
        <Slider label="position pos" color="data" value={pos} min={0} max={50} step={1} onChange={setPos} fmt={(x) => String(x)} />
      </div>
      <div className={styles.note}><strong>PE({pos})</strong>, first {shown.length} of {d} dimensions (even = sin, odd = cos; faded bars are negative)</div>
      <Bars values={shown} labels={shown.map((_, i) => String(i))} max={1} fmt={(v) => v.toFixed(1)} height={140} />
      <div className={styles.row} style={{gap: '1rem', flexWrap: 'wrap'}}>
        <div style={{flex: '1 1 260px'}}>
          <div className={styles.note}>Dimensions {pairs.join(', ')} as a function of position (vertical line = pos)</div>
          <Plot fns={pairs.map((i2, k) => ({f: wave(i2), color: ['var(--eq-data)', 'var(--eq-pred)', 'var(--eq-hyper)'][k]}))}
            x0={0} x1={50} y0={-1} y1={1} vline={pos} xLabel="pos" />
        </div>
        <div style={{flex: '1 1 260px'}}>
          <div className={styles.note}>Similarity PE(pos)·PE(pos + k) / (d/2) for offsets k = −20 … 20</div>
          <Plot fns={[{f: sim, color: 'var(--eq-loss)'}]} x0={-20} x1={20} y0={-1} y1={1} vline={0} xLabel="k" />
        </div>
      </div>
      <Readout items={[
        ['|PE(pos)|² (always d/2)', M.dot(pe, pe).toFixed(3), 'data'],
        ['PE(pos)·PE(pos+1)', M.dot(pe, M.positionalEncoding(pos + 1, d)).toFixed(3), 'pred'],
        ['PE(pos)·PE(pos+5)', M.dot(pe, M.positionalEncoding(pos + 5, d)).toFixed(3), 'pred'],
      ]} />
      <div className={styles.note}>Move pos: the two dot products don't change (for d_model = 512 they are 249.102 and 189.6 at every position). That's the shift property behind the paper's claim that PE(pos + k) is a linear function of PE(pos).</div>
    </Widget>
  );
}

/** The paper's learning-rate schedule: linear warmup, then decay with 1/√step. */
export function LearningRateSchedule() {
  const [dModel, setDModel] = useState(512);
  const [warm, setWarm] = useState(4000);
  const [step, setStep] = useState(4000);
  const lr = (s) => M.transformerLR(s, dModel, warm);
  const peak = lr(warm);
  return (
    <Widget title="Learning rate with warmup (paper eq. 3)"
      hint="It rises linearly for warmup_steps, peaks at d_model to the power −0.5 times warmup_steps to the power −0.5, then decays like 1/√step. The paper used warmup_steps = 4000 and trained the base model for 100,000 steps.">
      <Tex>{String.raw`\begin{aligned} \hyper{lrate} &= \hyper{d_{model}}^{-0.5} \cdot \min\!\big(step^{-0.5},\; step \cdot warmup\_steps^{-1.5}\big) \\ \text{peak} &= \hyper{d_{model}}^{-0.5} \cdot warmup\_steps^{-0.5} \;\text{ at step} = warmup\_steps \end{aligned}`}</Tex>
      <div className={styles.controls}>
        <Slider label="d_model" color="hyper" value={dModel} min={64} max={1024} step={64} onChange={setDModel} fmt={(x) => String(x)} />
        <Slider label="warmup_steps" color="hyper" value={warm} min={500} max={16000} step={500} onChange={setWarm} fmt={(x) => String(x)} />
        <Slider label="step" color="data" value={step} min={1} max={100000} step={100} onChange={setStep} fmt={(x) => x.toLocaleString('en-US')} />
      </div>
      <Plot fns={[{f: (s) => lr(s) * 1000, color: 'var(--eq-hyper)'}]} x0={0} x1={100000} y0={0} y1={Math.ceil(peak * 1200 * 10) / 10}
        vline={step} xLabel="step" points={[[step, lr(step) * 1000]]} />
      <Readout items={[
        ['learning rate at this step', lr(step).toFixed(6), 'hyper'],
        ['peak (at step = warmup_steps)', peak.toFixed(6), 'hyper'],
        ['phase', step < warm ? 'warmup: rising linearly' : 'decay: ∝ 1/√step'],
      ]} />
      <div className={styles.note}>The y-axis is lrate × 1000. Base model: 0.000699 at step 4,000, 0.000349 at 16,000, 0.000140 at 100,000.</div>
    </Widget>
  );
}

/* =================================================================================== */
/* BERT (Devlin et al. 2019)                                                           */
/* =================================================================================== */

const BERT_CELL = {padding: '0.15rem 0.35rem', textAlign: 'center', fontFamily: 'var(--ifm-font-family-monospace)', fontSize: '0.8rem', whiteSpace: 'nowrap'};
const BERT_BG = {
  tok: 'rgba(214, 90, 104, 0.16)', emb: 'rgba(214, 178, 40, 0.18)', A: 'rgba(80, 160, 70, 0.22)', B: 'rgba(60, 130, 200, 0.2)',
  pos: 'rgba(140, 140, 140, 0.16)', mask: 'rgba(214, 60, 60, 0.28)', random: 'rgba(245, 159, 0, 0.3)', same: 'rgba(60, 130, 200, 0.28)',
};

/** Figure 2 live: type sentence A and B, see WordPiece tokens, segment ids and positions. */
export function BertInputExplorer() {
  const [a, setA] = useState('my dog is cute');
  const [b, setB] = useState('he likes playing');
  const [pair, setPair] = useState(true);
  const ta = M.bertTokenize(a), tb = M.bertTokenize(b);
  const {tokens, segments, positions} = M.bertPack(ta, pair ? tb : null);
  const rows = [
    ['Input', tokens, () => BERT_BG.tok],
    ['Token Embeddings', tokens.map((t) => <>E<sub>{t}</sub></>), () => BERT_BG.emb],
    ['Segment Embeddings', segments.map((s) => <>E<sub>{s ? 'B' : 'A'}</sub></>), (i) => (segments[i] ? BERT_BG.B : BERT_BG.A)],
    ['Position Embeddings', positions.map((p) => <>E<sub>{p}</sub></>), () => BERT_BG.pos],
  ];
  return (
    <Widget title="BERT input representation (paper Figure 2), live"
      hint="Tokenizer: lower-case, split on spaces and punctuation, then greedy longest-match-first WordPiece over a toy vocabulary of 148 pieces (real BERT: 30,000). A word that isn't in the vocabulary is split into pieces; continuation pieces start with ##.">
      <Tex>{String.raw`\pred{x_i} = \param{E_{token}}[\data{t_i}] + \param{E_{segment}}[\data{s_i}] + \param{E_{position}}[i] \;\in \mathbb{R}^{\hyper{H}}`}</Tex>
      <div className={styles.controls}>
        <label className={styles.textIn}>sentence A <input value={a} onChange={(e) => setA(e.target.value)} /></label>
        <label className={styles.textIn}>sentence B <input value={b} disabled={!pair} onChange={(e) => setB(e.target.value)} /></label>
      </div>
      <div className={styles.row} style={{gap: '0.4rem', flexWrap: 'wrap'}}>
        {[[true, 'sentence pair (NSP, QA, NLI)'], [false, 'single sentence (sentiment, tagging)']].map(([v, l]) => (
          <button key={l} className={styles.button} onClick={() => setPair(v)}
            style={v === pair ? {outline: '2px solid var(--ifm-color-primary)', fontWeight: 700} : {opacity: 0.75}}>{l}</button>
        ))}
        <button className={styles.button} onClick={() => { setA('penguin are flightless birds'); setB('the man went to the store'); setPair(true); }}>try: flightless</button>
      </div>
      <div className={styles.scroll}>
        <table className={styles.table}>
          <tbody>
            {rows.map(([name, vals, bg]) => (
              <tr key={name}>
                <th style={{whiteSpace: 'nowrap', fontSize: '0.8rem'}}>{name}</th>
                {vals.map((v, i) => <td key={i} style={{...BERT_CELL, background: bg(i)}}>{v}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Readout items={[
        ['sequence length n', String(tokens.length), 'data'],
        ['WordPiece tokens in A / B', `${ta.length} / ${pair ? tb.length : 0}`, 'data'],
        ['words split into pieces', String(tokens.filter((t) => t.startsWith('##')).length), 'pred'],
        ['[CLS] final state C feeds', pair ? 'NSP or a pair classifier' : 'a sentence classifier', 'pred'],
      ]} />
      <div className={styles.note}>Every column is one token: its three embeddings are added into one H-dimensional vector (768 in BERT-base), then LayerNorm and dropout. Maximum n is 512 (the size of the position table).</div>
    </Widget>
  );
}

/** Masked LM corruption: 15% chosen, then 80% [MASK] / 10% random / 10% unchanged. */
export function MlmMaskingPlayground() {
  const [text, setText] = useState('the man went to the store he bought a gallon of milk');
  const [rate, setRate] = useState(0.15);
  const [pMask, setPMask] = useState(0.8);
  const [pRand, setPRand] = useState(0.1);
  const [seed, setSeed] = useState(1);
  const toks = M.bertTokenize(text);
  const {tokens} = M.bertPack(toks);
  const rec = M.bertMask(tokens, M.mulberry32(seed), rate, pMask, pRand);
  const stats = useMemo(() => {
    const rng = M.mulberry32(12345);
    const sent = Array.from({length: 100}, (_, i) => 'w' + i);
    const c = {none: 0, mask: 0, random: 0, same: 0};
    for (let r = 0; r < 400; r++) for (const t of M.bertMask(sent, rng, rate, pMask, pRand)) c[t.kind]++;
    return c;
  }, [rate, pMask, pRand]);
  const N = 40000;
  const pSame = Math.max(0, 1 - pMask - pRand);
  return (
    <Widget title="Masked LM: what the encoder sees, and what it must predict"
      hint="The paper: choose 15% of WordPiece positions; replace the chosen token with [MASK] 80% of the time, a random token 10%, and leave it unchanged 10%. The loss is computed only at the chosen positions. Resample to see different draws.">
      <div className={styles.controls}>
        <label className={styles.textIn}>sentence <input value={text} onChange={(e) => setText(e.target.value)} /></label>
        <Slider label="masking rate" color="hyper" value={rate} min={0.05} max={0.5} step={0.05} onChange={setRate} fmt={(v) => `${Math.round(v * 100)}%`} />
        <Slider label="P([MASK])" color="hyper" value={pMask} min={0} max={1} step={0.1} onChange={(v) => { setPMask(v); setPRand(Math.min(pRand, +(1 - v).toFixed(1))); }} fmt={(v) => `${Math.round(v * 100)}%`} />
        <Slider label="P(random)" color="hyper" value={pRand} min={0} max={+(1 - pMask).toFixed(1)} step={0.1} onChange={setPRand} fmt={(v) => `${Math.round(v * 100)}%`} />
      </div>
      <div className={styles.row} style={{gap: '0.4rem', flexWrap: 'wrap'}}>
        <button className={styles.button} onClick={() => setSeed(seed + 1)}>resample</button>
        <button className={styles.button} onClick={() => { setRate(0.15); setPMask(0.8); setPRand(0.1); }}>paper setting</button>
        <button className={styles.button} onClick={() => { setRate(0.15); setPMask(1); setPRand(0); }}>100% [MASK] (Table 8 row 2)</button>
      </div>
      <div className={styles.scroll}>
        <table className={styles.table}>
          <tbody>
            <tr><th style={{fontSize: '0.8rem'}}>original</th>{rec.map((r, i) => <td key={i} style={{...BERT_CELL, background: r.kind === 'none' ? undefined : 'rgba(245, 159, 0, 0.12)'}}>{r.orig}</td>)}</tr>
            <tr><th style={{fontSize: '0.8rem'}}>encoder input</th>{rec.map((r, i) => <td key={i} style={{...BERT_CELL, background: BERT_BG[r.kind]}}>{r.shown}</td>)}</tr>
            <tr><th style={{fontSize: '0.8rem'}}>MLM label</th>{rec.map((r, i) => <td key={i} style={{...BERT_CELL, color: r.kind === 'none' ? 'var(--ifm-color-emphasis-500)' : 'var(--eq-loss)'}}>{r.kind === 'none' ? '–' : r.orig}</td>)}</tr>
          </tbody>
        </table>
      </div>
      <Readout items={[
        ['chosen positions here', `${rec.filter((r) => r.kind !== 'none').length} of ${tokens.length - 2}`, 'data'],
        ['over 40,000 simulated tokens: [MASK]', `${(100 * stats.mask / N).toFixed(1)}%`, 'loss'],
        ['random', `${(100 * stats.random / N).toFixed(1)}%`, 'hyper'],
        ['unchanged but predicted', `${(100 * stats.same / N).toFixed(1)}%`, 'pred'],
      ]} />
      <div className={styles.note}>Expected shares of all tokens: [MASK] {(100 * rate * pMask).toFixed(1)}%, random {(100 * rate * pRand).toFixed(1)}%, unchanged-but-predicted {(100 * rate * pSame).toFixed(1)}%. With the paper's setting that is 12% / 1.5% / 1.5%. Colours: red = [MASK], amber = random token, blue = left unchanged but still graded.</div>
    </Widget>
  );
}

/** SQuAD head on real scores from the page's tiny BERT: best span with j ≥ i, plus the SQuAD 2.0 no-answer rule. */
export function SpanPredictor() {
  const [k, setK] = useState(0);
  const [maxLen, setMaxLen] = useState(10);
  const [tau, setTau] = useState(0);
  const ex = M.BERT_SPAN_EXAMPLES[k];
  const para = ex.tokens.map((t, i) => i).filter((i) => i >= 4 && ex.tokens[i] !== '[SEP]');
  const {best, top} = M.bestSpan(ex.start, ex.end, para, maxLen, 5);
  const sNull = ex.start[0] + ex.end[0];
  const answer = best.score > sNull + tau;
  const softmax = (v) => { const m = Math.max(...v), e = v.map((x) => Math.exp(x - m)), s = e.reduce((a, b) => a + b, 0); return e.map((x) => x / s); };
  const ps = softmax(para.map((i) => ex.start[i])), pe = softmax(para.map((i) => ex.end[i]));
  const span = (s) => ex.tokens.slice(s.i, s.j + 1).join(' ');
  return (
    <Widget title="SQuAD span prediction with a start vector S and an end vector E"
      hint="Scores are the real S·T_i and E·T_j of this page's fine-tuned tiny BERT (question = two numbers, answer = the part of the progression between them). Shrink the maximum answer length or raise τ to see how decoding changes.">
      <Tex>{String.raw`\begin{aligned} \pred{P^{start}_i} &= \frac{e^{\param{S}\cdot\pred{T_i}}}{\sum_j e^{\param{S}\cdot\pred{T_j}}} \quad (\text{end: same with } \param{E}) \\ \pred{\hat s_{i,j}} &= \param{S}\cdot\pred{T_i} + \param{E}\cdot\pred{T_j}\quad (j \ge i) \\ \text{SQuAD 2.0: answer iff } & \max_{j\ge i}\pred{\hat s_{i,j}} > \param{S}\cdot\pred{C} + \param{E}\cdot\pred{C} + \hyper{\tau} \end{aligned}`}</Tex>
      <div className={styles.row} style={{gap: '0.4rem', flexWrap: 'wrap'}}>
        {M.BERT_SPAN_EXAMPLES.map((e, i) => (
          <button key={i} className={styles.button} onClick={() => setK(i)}
            style={i === k ? {outline: '2px solid var(--ifm-color-primary)', fontWeight: 700} : {opacity: 0.75}}>example {i + 1}: question “{e.tokens.slice(1, 3).join(' ')}”</button>
        ))}
      </div>
      <div className={styles.controls}>
        <Slider label="max answer length" color="hyper" value={maxLen} min={1} max={10} step={1} onChange={setMaxLen} fmt={(v) => String(v)} />
        <Slider label="threshold τ" color="hyper" value={tau} min={-5} max={25} step={0.5} onChange={setTau} fmt={(v) => v.toFixed(1)} />
      </div>
      <div className={styles.scroll}>
        <table className={styles.table}>
          <tbody>
            <tr><th style={{fontSize: '0.8rem'}}>paragraph</th>{para.map((i) => (
              <td key={i} style={{...BERT_CELL, background: answer && i >= best.i && i <= best.j ? 'rgba(245, 159, 0, 0.3)' : BERT_BG.tok}}>{ex.tokens[i]}</td>))}</tr>
            <tr><th style={{fontSize: '0.8rem', color: 'var(--eq-pred)'}}>S·T_i</th>{para.map((i) => <td key={i} style={BERT_CELL}>{ex.start[i].toFixed(2)}</td>)}</tr>
            <tr><th style={{fontSize: '0.8rem', color: 'var(--eq-pred)'}}>P start</th>{para.map((i, n) => <td key={i} style={BERT_CELL}>{ps[n].toFixed(3)}</td>)}</tr>
            <tr><th style={{fontSize: '0.8rem', color: 'var(--eq-loss)'}}>E·T_j</th>{para.map((i) => <td key={i} style={BERT_CELL}>{ex.end[i].toFixed(2)}</td>)}</tr>
            <tr><th style={{fontSize: '0.8rem', color: 'var(--eq-loss)'}}>P end</th>{para.map((i, n) => <td key={i} style={BERT_CELL}>{pe[n].toFixed(3)}</td>)}</tr>
          </tbody>
        </table>
      </div>
      <Readout items={[
        ['best span (j ≥ i)', `“${span(best)}”  score ${best.score.toFixed(2)}`, 'pred'],
        ['gold answer', `“${ex.tokens.slice(ex.gold[0], ex.gold[1] + 1).join(' ')}”`, 'data'],
        ['s_null = S·C + E·C', sNull.toFixed(2), 'loss'],
        ['SQuAD 2.0 decision', answer ? `answer (${best.score.toFixed(2)} > ${(sNull + tau).toFixed(2)})` : `no answer (${best.score.toFixed(2)} ≤ ${(sNull + tau).toFixed(2)})`, 'hyper'],
      ]} />
      <div className={styles.note}>Top spans: {top.map((s) => `“${span(s)}” ${s.score.toFixed(2)}`).join(' · ')}</div>
    </Widget>
  );
}

const BERT_PRESETS = {base: {L: 12, H: 768, A: 12, V: 30522, P: 512}, large: {L: 24, H: 1024, A: 16, V: 30522, P: 512}, tiny: {L: 4, H: 64, A: 4, V: 27, P: 15}};
/** Where BERT's parameters live, and what sequence length does to attention cost. */
export function BertSizeCalculator() {
  const [cfg, setCfg] = useState(BERT_PRESETS.base);
  const [n, setN] = useState(128);
  const p = M.bertParams({L: cfg.L, H: cfg.H, V: cfg.V, P: cfg.P});
  const fmt = (v) => v.toLocaleString('en-US');
  const mill = (v) => `${(v / 1e6).toFixed(1)}M`;
  const valid = cfg.H % cfg.A === 0;
  return (
    <Widget title="BERT size calculator (L layers, hidden size H, A heads)"
      hint="Counts match HuggingFace BertModel with the released 30,522-piece vocabulary. Per layer: attention 4H²+4H, feed-forward H→4H→H 8H²+5H, two LayerNorms 4H; 514 = 512 positions + 2 segments. The number of heads A does not change the parameter count: the heads split H into A slices of H/A.">
      <Tex>{String.raw`\begin{aligned} \text{embeddings} &= (\hyper{V} + 514)\,\hyper{H} + 2\hyper{H} \\ \text{one layer} &= (4\hyper{H}^2 + 4\hyper{H}) + (8\hyper{H}^2 + 5\hyper{H}) + 4\hyper{H} \\ &= 12\hyper{H}^2 + 13\hyper{H} \end{aligned}`}</Tex>
      <div className={styles.row} style={{gap: '0.4rem', flexWrap: 'wrap'}}>
        {Object.entries(BERT_PRESETS).map(([k, v]) => (
          <button key={k} className={styles.button} onClick={() => setCfg(v)}
            style={v.L === cfg.L && v.H === cfg.H && v.A === cfg.A ? {outline: '2px solid var(--ifm-color-primary)', fontWeight: 700} : {opacity: 0.75}}>
            {k === 'tiny' ? 'this page’s tiny BERT' : `BERT-${k}`}</button>
        ))}
      </div>
      <div className={styles.controls}>
        <Slider label="layers L" color="hyper" value={cfg.L} min={1} max={48} step={1} onChange={(v) => setCfg({...cfg, L: v})} fmt={(v) => String(v)} />
        <Slider label="hidden H" color="hyper" value={cfg.H} min={64} max={2048} step={64} onChange={(v) => setCfg({...cfg, H: v})} fmt={(v) => String(v)} />
        <Slider label="heads A" color="hyper" value={cfg.A} min={1} max={32} step={1} onChange={(v) => setCfg({...cfg, A: v})} fmt={(v) => String(v)} />
        <Slider label="sequence length n" color="data" value={n} min={32} max={512} step={32} onChange={setN} fmt={(v) => String(v)} />
      </div>
      <Readout items={[
        ['total parameters', `${fmt(p.total)} (${mill(p.total)})`, 'param'],
        ['embeddings', `${fmt(p.emb)} (${(100 * p.emb / p.total).toFixed(1)}%)`, 'param'],
        [`encoder (${cfg.L} × ${fmt(p.layer)})`, `${fmt(p.encoder)} (${(100 * p.encoder / p.total).toFixed(1)}%)`, 'param'],
        ['pooler (dense + tanh on C)', fmt(p.pooler), 'param'],
        ['vocabulary V, positions', `${fmt(cfg.V)}, ${cfg.P}`, 'hyper'],
        ['head size H / A', valid ? String(cfg.H / cfg.A) : 'H must be divisible by A', 'hyper'],
        ['attention scores per layer (A·n²)', fmt(cfg.A * n * n), 'pred'],
      ]} />
      <div className={styles.note}>The paper rounds to 110M (base) and 340M (large). The tiny preset uses this page's 27-token vocabulary and 15 positions: embeddings + encoder = 202,880, as printed in block 5 (its pooler lives in the pre-training heads). Attention scores grow with n²: 512 tokens cost 16× the score entries of 128, which is why BERT pre-trained 90% of steps at length 128 and only the last 10% at 512.</div>
    </Widget>
  );
}

/* =================================================================================== */
/* Stacked encoder blocks: the mini sentiment project (notebook 18)                    */
/* =================================================================================== */

const TINY_PRESETS = ['not good movie', 'not bad movie', 'this movie is terrible', 'i love this film', 'what a boring film', 'hello world movie'];

/** The notebook's two trained classifiers (1 block and 3 blocks), running in the browser on their exported weights. */
export function TinySentimentLive() {
  const url = useBaseUrl('/data/deep-learning/stacked_encoder_sentiment.json');
  const [W, setW] = useState(null);
  const [err, setErr] = useState(null);
  const [text, setText] = useState('not good movie');
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  useEffect(() => {
    let alive = true;
    fetch(url).then((r) => r.json()).then((j) => alive && setW(j)).catch((e) => alive && setErr(String(e)));
    return () => { alive = false; };
  }, [url]);
  const title = "Live: the notebook's 1-block and 3-block sentiment models";
  if (err) return <Widget title={title}><div className={styles.note}>Could not load the model weights ({err}).</div></Widget>;
  if (!W) return <Widget title={title}><div className={styles.note}>Loading the trained weights (about 370 KB)…</div></Widget>;
  const enc = M.encodeTiny(text, W.vocab, W.maxLen);
  const inv = Object.fromEntries(Object.entries(W.vocab).map(([k, v]) => [v, k]));
  const toks = enc.ids.map((id) => inv[id]);
  const one = M.tinyEncoderForward(W.models.single, enc.ids, enc.mask, W.heads);
  const three = M.tinyEncoderForward(W.models.stacked, enc.ids, enc.mask, W.heads);
  const n = enc.mask.reduce((a, b) => a + b, 0);
  const A = n ? three.attentions[layer][head].slice(0, n).map((r) => r.slice(0, n)) : [];
  const norm = (h) => h.slice(0, n).reduce((s, x) => s + Math.sqrt(x.reduce((a, v) => a + v * v, 0)), 0) / Math.max(n, 1);
  const label = (p) => (p[1] > p[0] ? `POSITIVE ${p[1].toFixed(4)}` : `NEGATIVE ${p[0].toFixed(4)}`);
  const sel = (on) => (on ? {outline: '2px solid var(--ifm-color-primary)', fontWeight: 700} : {opacity: 0.75});
  return (
    <Widget title={title}
      hint="These are the two models trained in this page's notebook run (same seeds, so the same weights as the notebook), running in your browser. Only the 35 vocabulary entries are known; any other word becomes <UNK>, and anything past 6 words is cut off.">
      <div className={styles.controls}>
        <label className={styles.textIn}>sentence <input value={text} onChange={(e) => setText(e.target.value)} /></label>
      </div>
      <div className={styles.row} style={{gap: '0.4rem', flexWrap: 'wrap'}}>
        {TINY_PRESETS.map((p) => <button key={p} className={styles.button} onClick={() => setText(p)}>{p}</button>)}
      </div>
      <div className={styles.scroll}>
        <table className={styles.table}>
          <tbody>
            <tr><th style={{fontSize: '0.8rem'}}>token</th>{toks.map((t, i) => <td key={i} style={{...BERT_CELL, opacity: enc.mask[i] ? 1 : 0.45}}>{t}</td>)}</tr>
            <tr><th style={{fontSize: '0.8rem'}}>input_id</th>{enc.ids.map((v, i) => <td key={i} style={BERT_CELL}>{v}</td>)}</tr>
            <tr><th style={{fontSize: '0.8rem'}}>attention_mask</th>{enc.mask.map((v, i) => <td key={i} style={{...BERT_CELL, background: v ? 'rgba(80, 160, 70, 0.22)' : 'rgba(140, 140, 140, 0.16)'}}>{v}</td>)}</tr>
          </tbody>
        </table>
      </div>
      <Readout items={[
        ['1 encoder block (9,922 params)', label(one.probs), 'pred'],
        ['3 encoder blocks (27,010 params)', label(three.probs), 'pred'],
        ['unknown words → <UNK>', enc.unknown.length ? enc.unknown.join(', ') : 'none', 'data'],
        ['3-block model, mean token norm: embedding → block 1 → 2 → 3', three.hidden.map((h) => norm(h).toFixed(2)).join(' → '), 'hyper'],
      ]} />
      {enc.truncated && <div className={styles.note}>More than 6 words: everything after the sixth word was truncated, exactly as encode_text does.</div>}
      <div className={styles.row} style={{gap: '0.4rem', flexWrap: 'wrap'}}>
        {[0, 1, 2].map((l) => <button key={l} className={styles.button} onClick={() => setLayer(l)} style={sel(l === layer)}>layer {l + 1}</button>)}
        {[0, 1, 2, 3].map((h) => <button key={h} className={styles.button} onClick={() => setHead(h)} style={sel(h === head)}>head {h + 1}</button>)}
      </div>
      <div className={styles.note}>3-block model, layer {layer + 1}, head {head + 1}: each row is a query token, each column a key token. Padded keys get exactly 0 and are not shown.</div>
      <div className={styles.scroll}>
        <table className={styles.table}>
          <thead><tr><th>q ↓ / k →</th>{toks.slice(0, n).map((t, j) => <th key={j}>{t}</th>)}</tr></thead>
          <tbody>{A.map((r, i) => (
            <tr key={i}><td><strong>{toks[i]}</strong></td>
              {r.map((v, j) => <td key={j} style={{background: `rgba(56, 217, 169, ${(v * 0.85).toFixed(2)})`}}>{v.toFixed(3)}</td>)}</tr>
          ))}</tbody>
        </table>
      </div>
    </Widget>
  );
}

/** Where the parameters of the notebook's classifier live, and how they grow with depth. */
export function EncoderStackParams() {
  const [L, setL] = useState(3);
  const [D, setD] = useState(32);
  const [dff, setDff] = useState(64);
  const [heads, setHeads] = useState(4);
  const p = M.tinyEncoderParams({L, D, dff});
  const fmt = (v) => v.toLocaleString('en-US');
  return (
    <Widget title="Parameter counter for the stacked encoder classifier"
      hint="Defaults are the notebook's model: 35-entry vocabulary, 6 positions, d_model 32, 4 heads, d_ff 64, 2 classes. Each extra block adds the same parameters (8,544 at the defaults); the number of heads changes nothing.">
      <Tex>{String.raw`\begin{aligned} \text{block} &= 4(\hyper{D}^2 + \hyper{D}) + (2\hyper{D}\hyper{d_{ff}} + \hyper{d_{ff}} + \hyper{D}) + 4\hyper{D} \\ \text{total} &= (V + T)\,\hyper{D} + \hyper{L}\cdot\text{block} + 2\hyper{D} + 2 \end{aligned}`}</Tex>
      <div className={styles.controls}>
        <Slider label="blocks L" color="hyper" value={L} min={1} max={12} step={1} onChange={setL} fmt={(v) => String(v)} />
        <Slider label="d_model D" color="hyper" value={D} min={8} max={128} step={8} onChange={setD} fmt={(v) => String(v)} />
        <Slider label="d_ff" color="hyper" value={dff} min={16} max={512} step={16} onChange={setDff} fmt={(v) => String(v)} />
        <Slider label="heads" color="hyper" value={heads} min={1} max={8} step={1} onChange={setHeads} fmt={(v) => String(v)} />
      </div>
      <Readout items={[
        ['total trainable parameters', fmt(p.total), 'param'],
        ['embeddings (token + position)', fmt(p.emb), 'param'],
        [`per block (× ${L})`, `${fmt(p.block)} = attention ${fmt(p.attn)} + FFN ${fmt(p.ffn)} + LayerNorm ${fmt(p.ln)}`, 'param'],
        ['classification head', fmt(p.head), 'param'],
        ['head_dim = D / heads', D % heads === 0 ? String(D / heads) : 'D must be divisible by heads (the notebook asserts this)', 'hyper'],
        ['output shape after every block', `[batch, 6, ${D}]`, 'pred'],
      ]} />
    </Widget>
  );
}

/* =================================================================================== */
/* Hugging Face pipelines and what they hide (notebook 19)                             */
/* =================================================================================== */

const TOK_PRESETS = [
  ['The movie was surprisingly good.', null],
  ['Transformers tokenization is unbelievably subword-based!', null],
  ["Café résumé naïve — it's 2026, isn't it?", null],
  ['What is BERT?', 'BERT is an encoder-only Transformer model.'],
];

/** The real bert-base-uncased / DistilBERT tokenizer, re-implemented in the browser and tested against Hugging Face. */
export function BertTokenizerLive() {
  const url = useBaseUrl('/data/deep-learning/bert-base-uncased-vocab.txt');
  const [vocab, setVocab] = useState(null);
  const [err, setErr] = useState(null);
  const [text, setText] = useState('The movie was surprisingly good.');
  const [pair, setPair] = useState('');
  const [maxLen, setMaxLen] = useState(512);
  const [padTo, setPadTo] = useState(0);
  useEffect(() => {
    let alive = true;
    fetch(url).then((r) => r.text()).then((t) => alive && setVocab(t.split('\n').slice(0, 30522))).catch((e) => alive && setErr(String(e)));
    return () => { alive = false; };
  }, [url]);
  const title = 'Live: the DistilBERT / bert-base-uncased tokenizer';
  if (err) return <Widget title={title}><div className={styles.note}>Could not load the vocabulary ({err}).</div></Widget>;
  if (!vocab) return <Widget title={title}><div className={styles.note}>Loading the 30,522-entry WordPiece vocabulary (about 230 KB)…</div></Widget>;
  const enc = M.bertEncode(text, vocab, {pair: pair.trim() ? pair : null, maxLength: maxLen, padTo});
  const words = M.bertBasicTokenize(text + (pair.trim() ? ' ' + pair : '')).length;
  const rows = [
    ['token', enc.tokens, (t) => (t.startsWith('[') ? BERT_BG.pos : t.startsWith('##') ? BERT_BG.emb : BERT_BG.tok)],
    ['input_ids', enc.ids, () => undefined],
    ['token_type_ids', enc.typeIds, (_, i) => (enc.typeIds[i] ? BERT_BG.B : BERT_BG.A)],
    ['attention_mask', enc.mask, (_, i) => (enc.mask[i] ? 'rgba(80, 160, 70, 0.22)' : 'rgba(140, 140, 140, 0.16)')],
  ];
  return (
    <Widget title={title}
      hint="Same output as AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english'): lower-case, accents stripped, punctuation split off, then greedy WordPiece over the real 30,522-entry vocabulary (checked against Hugging Face on this page's test sentences). Grey = special tokens, yellow = ## continuation pieces.">
      <div className={styles.controls}>
        <label className={styles.textIn}>text <input value={text} onChange={(e) => setText(e.target.value)} /></label>
        <label className={styles.textIn}>text pair (optional) <input value={pair} onChange={(e) => setPair(e.target.value)} /></label>
        <Slider label="max_length (truncation)" color="hyper" value={maxLen} min={4} max={512} step={1} onChange={setMaxLen} fmt={(v) => String(v)} />
        <Slider label="pad to (0 = no padding)" color="hyper" value={padTo} min={0} max={32} step={1} onChange={setPadTo} fmt={(v) => String(v)} />
      </div>
      <div className={styles.row} style={{gap: '0.4rem', flexWrap: 'wrap'}}>
        {TOK_PRESETS.map(([a, b]) => <button key={a} className={styles.button} onClick={() => { setText(a); setPair(b || ''); }}>{b ? 'question + context pair' : a.slice(0, 32)}</button>)}
      </div>
      <div className={styles.scroll}>
        <table className={styles.table}>
          <tbody>
            {rows.map(([name, vals, bg]) => (
              <tr key={name}>
                <th style={{whiteSpace: 'nowrap', fontSize: '0.8rem'}}>{name}</th>
                {vals.map((v, i) => <td key={i} style={{...BERT_CELL, background: bg(enc.tokens[i], i)}}>{v}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Readout items={[
        ['words after basic tokenization', String(words), 'data'],
        ['tokens incl. specials and padding', String(enc.tokens.length), 'data'],
        ['continuation pieces (##)', String(enc.tokens.filter((t) => t.startsWith('##')).length), 'pred'],
        ['truncated?', enc.truncated ? 'yes: the model would never see the rest' : 'no', 'hyper'],
      ]} />
      <div className={styles.note}>The model receives input_ids and attention_mask. DistilBERT has no segment embeddings, so it ignores token_type_ids (the tokenizer still returns them because it is a BertTokenizer).</div>
    </Widget>
  );
}

const SST2_LOGITS = [
  ['I love this movie.', -4.3135, 4.6626], ['The story was not bad.', -3.3528, 3.5306], ['The movie was not terrible.', -2.0694, 2.0548],
  ['I expected more from this film.', 2.5908, -2.148], ['The movie was okay, but not amazing.', 2.9823, -2.4684], ['It was fine, I guess.', -3.9914, 4.3276],
];

/** Logits → softmax → confidence: with two classes, confidence is the sigmoid of the logit margin. */
export function LogitsToConfidence() {
  const [z0, setZ0] = useState(-4.2341);
  const [z1, setZ1] = useState(4.6189);
  const p1 = M.twoClassProb(z0, z1), margin = z1 - z0;
  const label = p1 >= 0.5 ? 'POSITIVE' : 'NEGATIVE';
  return (
    <Widget title="From logits to the pipeline's score"
      hint="The SST-2 model outputs two logits, [NEGATIVE, POSITIVE]. Softmax turns them into probabilities; with two classes that is the sigmoid of the margin, so the score saturates fast: a margin of 4.6 already gives 0.99, a margin of 9.2 gives 0.9999. Presets are real logits from this page's runs.">
      <Tex>{String.raw`\pred{p}(\text{POS}) = \frac{e^{\pred{z_1}}}{e^{\pred{z_0}} + e^{\pred{z_1}}} = \sigma(\pred{z_1} - \pred{z_0}), \qquad \text{score} = \max(\pred{p}, 1 - \pred{p})`}</Tex>
      <div className={styles.row} style={{gap: '0.4rem', flexWrap: 'wrap'}}>
        {SST2_LOGITS.map(([s, a, b]) => <button key={s} className={styles.button} onClick={() => { setZ0(a); setZ1(b); }}>{s}</button>)}
      </div>
      <div className={styles.controls}>
        <Slider label="z0 (NEGATIVE logit)" color="pred" value={z0} min={-8} max={8} step={0.01} onChange={setZ0} fmt={(v) => v.toFixed(2)} />
        <Slider label="z1 (POSITIVE logit)" color="pred" value={z1} min={-8} max={8} step={0.01} onChange={setZ1} fmt={(v) => v.toFixed(2)} />
      </div>
      <Plot fns={[{f: (m) => 1 / (1 + Math.exp(-m)), color: 'var(--eq-pred)'}]} x0={-10} x1={10} y0={0} y1={1}
        vline={margin} xLabel="z1 − z0" points={[[Math.max(-10, Math.min(10, margin)), p1]]} />
      <Readout items={[
        ['margin z1 − z0', margin.toFixed(4), 'pred'],
        ['softmax [NEGATIVE, POSITIVE]', `[${(1 - p1).toFixed(6)}, ${p1.toFixed(6)}]`, 'pred'],
        ['pipeline output', `{label: ${label}, score: ${Math.max(p1, 1 - p1).toFixed(6)}}`, 'data'],
      ]} />
      <div className={styles.note}>Defaults are the notebook's logits for "The movie was surprisingly good.": [−4.2341, 4.6189] → 0.99986. A near-certain score only says the margin is large; "It was fine, I guess." gets 0.99976 POSITIVE with a margin of 8.32.</div>
    </Widget>
  );
}

/** The notebook's overlapping word chunker over the real BERT-paper page lengths. */
export function ChunkPlanner() {
  const [size, setSize] = useState(180);
  const [overlap, setOverlap] = useState(40);
  const [minWords, setMinWords] = useState(30);
  const [page, setPage] = useState(1);
  const [maxLen, setMaxLen] = useState(384);
  const W = M.BERT_PDF_PAGE_WORDS[page - 1];
  const plan = M.planChunks(W, size, overlap, minWords);
  const total = M.BERT_PDF_PAGE_WORDS.reduce((n, w) => n + M.planChunks(w, size, overlap, minWords).filter((c) => c.kept).length, 0);
  const estTokens = Math.round(1.74 * size) + 11;
  const cols = ['var(--eq-data)', 'var(--eq-pred)', 'var(--eq-param)', 'var(--eq-hyper)'];
  return (
    <Widget title="Plan the chunks: size, overlap and the token budget"
      hint="Page lengths are the real word counts of the 16 BERT-paper pages after extraction. Chunks step by size − overlap words; a final window shorter than the minimum is dropped. Token estimate: 1.74 tokens per word (this PDF's average) plus 11 for the question and special tokens.">
      <div className={styles.controls}>
        <Slider label="chunk_size (words)" color="hyper" value={size} min={40} max={400} step={10} onChange={setSize} fmt={(v) => String(v)} />
        <Slider label="overlap (words)" color="hyper" value={overlap} min={0} max={200} step={5} onChange={setOverlap} fmt={(v) => String(v)} />
        <Slider label="min_chunk_words" color="hyper" value={minWords} min={0} max={100} step={5} onChange={setMinWords} fmt={(v) => String(v)} />
        <Slider label="page to draw" color="data" value={page} min={1} max={16} step={1} onChange={setPage} fmt={(v) => `${v} (${M.BERT_PDF_PAGE_WORDS[v - 1]} words)`} />
        <Slider label="model max_length" color="hyper" value={maxLen} min={128} max={512} step={64} onChange={setMaxLen} fmt={(v) => String(v)} />
      </div>
      {overlap >= size && <div className={styles.note}>overlap must be smaller than chunk_size (the notebook raises ValueError).</div>}
      {overlap < size && (
        <svg viewBox={`0 0 600 ${30 + 22 * plan.length}`} style={{width: '100%', maxHeight: 260}}>
          <rect x={10} y={6} width={580} height={14} fill="var(--ifm-color-emphasis-200)" rx={3} />
          <text x={300} y={17} fontSize={10} textAnchor="middle" fill="var(--ifm-font-color-base)">page {page}: words 0 to {W}</text>
          {plan.map((c, k) => (
            <g key={k}>
              <rect x={10 + 580 * c.start / W} y={28 + 22 * k} width={580 * (c.end - c.start) / W} height={16} rx={3}
                fill={cols[k % 4]} opacity={c.kept ? 0.55 : 0.15} stroke={cols[k % 4]} />
              <text x={14 + 580 * c.start / W} y={40 + 22 * k} fontSize={10} fill="var(--ifm-font-color-base)">
                {`chunk ${k}: ${c.start}–${c.end} (${c.end - c.start} words)${c.kept ? '' : ' dropped'}`}
              </text>
            </g>
          ))}
        </svg>
      )}
      <Readout items={[
        ['chunks on this page', overlap < size ? String(plan.filter((c) => c.kept).length) : '–', 'data'],
        ['chunks in the whole PDF', overlap < size ? String(total) : '–', 'data'],
        ['words read twice per boundary', String(overlap), 'hyper'],
        ['≈ tokens per question + chunk', `${estTokens} ${estTokens > maxLen ? `> ${maxLen}: the end of each chunk is truncated` : `≤ ${maxLen}: fits`}`, estTokens > maxLen ? 'loss' : 'pred'],
      ]} />
      <div className={styles.note}>Defaults are the notebook's: 180 / 40 / 30 give 78 chunks. Every extra chunk is one more forward pass per question in the brute-force system.</div>
    </Widget>
  );
}

/** Rank the real per-chunk answers for eight questions three ways, then apply the filter and threshold. */
export function DocQaRanker() {
  const url = useBaseUrl('/data/deep-learning/docqa_candidates.json');
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [qi, setQi] = useState('0');
  const [key, setKey] = useState('ns');
  const [filter, setFilter] = useState(true);
  const [thr, setThr] = useState(5.0);
  useEffect(() => {
    let alive = true;
    fetch(url).then((r) => r.json()).then((d) => alive && setData(d)).catch((e) => alive && setErr(String(e)));
    return () => { alive = false; };
  }, [url]);
  const title = 'Brute-force document QnA: rank, filter, threshold';
  if (err) return <Widget title={title}><div className={styles.note}>Could not load the candidates ({err}).</div></Widget>;
  if (!data) return <Widget title={title}><div className={styles.note}>Loading 8 × 78 real candidate answers…</div></Widget>;
  const q = data.questions[Number(qi)];
  const range = {ns: [0, 25, 0.5], m: [0, 30, 0.5], pr: [0, 1, 0.01]}[key];
  const t = Math.min(Math.max(thr, range[0]), range[1]);
  const res = M.rankCandidates(q.rows, {key, filter, threshold: t});
  const keyName = {ns: 'span score s_i + e_j (notebook)', m: 'margin over the [CLS] score', pr: 'probability among valid spans'}[key];
  return (
    <Widget title={title}
      hint="Every row is a real result: distilbert-base-cased-distilled-squad run on each of the 78 chunks of the BERT paper. The notebook ranks by raw span score; the other two keys come from the context-only span search on this page. Change the key and watch rank 1 move: none of the three scores is calibrated across chunks.">
      <div className={styles.controls}>
        <Select label="question" value={qi} onChange={setQi} options={data.questions.map((x, i) => [String(i), x.q])} />
        <Select label="rank by" value={key} onChange={(v) => { setKey(v); setThr(v === 'pr' ? 0.5 : 5.0); }}
          options={[['ns', 'span score (notebook)'], ['m', 'margin over [CLS] score'], ['pr', 'span probability']]} />
        <label className={styles.select}><span>quality filter</span>
          <input type="checkbox" checked={filter} onChange={(e) => setFilter(e.target.checked)} /></label>
        <Slider label="threshold on the rank-1 score" color="hyper" value={t} min={range[0]} max={range[1]} step={range[2]} onChange={setThr} fmt={(v) => v.toFixed(2)} />
      </div>
      <div className={styles.scroll}>
        <table className={styles.table}>
          <thead><tr><th>rank</th><th>answer</th><th>{key === 'ns' ? 'span score' : key === 'm' ? 'margin' : 'span prob.'}</th><th>notebook conf.</th><th>page</th><th>chunk</th></tr></thead>
          <tbody>
            {res.top.map((r, i) => (
              <tr key={r.c} className={i === 0 ? styles.hlRow : undefined}>
                <td>{i + 1}</td><td>{r.answer || <em>(empty)</em>}</td><td>{key === 'pr' ? r.score.toFixed(4) : r.score.toFixed(3)}</td>
                <td>{r.nc.toFixed(4)}</td><td>{r.p}</td><td>{r.c}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Readout items={[
        ['ranking key', keyName, 'pred'],
        ['system output', res.answered ? `"${res.best.answer}" (page ${res.best.p})` : 'No reliable answer found in the document.', res.answered ? 'data' : 'loss'],
      ]} />
      <div className={styles.note}>Try "Which city was BERT invented in?": the filtered top answer is "Transformers" with a span score of 6.34 (above 5.0) and a notebook confidence of 0.9964. A threshold cannot tell a confident wrong span from a right one.</div>
    </Widget>
  );
}

/** Retriever + reader on the real 12-question evaluation: choose chunking, k and α, see what gets read and picked. */
export function RetrieverReaderLab() {
  const url = useBaseUrl('/data/deep-learning/rr_eval.json');
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [qi, setQi] = useState('3');
  const [variant, setVariant] = useState('notebook');
  const [k, setK] = useState(5);
  const [alpha, setAlpha] = useState(0.4);
  useEffect(() => {
    let alive = true;
    fetch(url).then((r) => r.json()).then((d) => alive && setData(d)).catch((e) => alive && setErr(String(e)));
    return () => { alive = false; };
  }, [url]);
  const title = 'Retriever + reader: what k and α actually do';
  if (err) return <Widget title={title}><div className={styles.note}>Could not load the evaluation ({err}).</div></Widget>;
  if (!data) return <Widget title={title}><div className={styles.note}>Loading the 12-question evaluation…</div></Widget>;
  const V = data.variants[variant];
  const Q = V.q[Number(qi)];
  const res = M.combineCandidates(Q.cands, k, alpha);
  const total = V.q.filter((q) => { const b = M.combineCandidates(q.cands, k, alpha).best; return b && b.ok; }).length;
  const recall = V.q.filter((q) => q.goldRank !== null && q.goldRank <= k).length;
  const bf = data.bruteforce.filter((b) => b.ok).length;
  return (
    <Widget title={title}
      hint="Real runs: all-MiniLM-L6-v2 retrieves, distilbert-base-cased-distilled-squad reads each retrieved chunk, and the notebook's rule picks the answer: min-max both scores over the valid candidates, final = α × retrieval + (1 − α) × reader. Green rows contain the right answer. Accuracy and recall are over all 12 questions for the current settings.">
      <div className={styles.controls}>
        <Select label="question" value={qi} onChange={setQi} options={data.questions.map((q, i) => [String(i), q])} />
        <Select label="chunks" value={variant} onChange={setVariant}
          options={[['notebook', `notebook: 180 words, overlap 40 (${data.variants.notebook.n} chunks)`], ['small', `120 words, overlap 30, min 40 (${data.variants.small.n} chunks)`]]} />
        <Slider label="top-k chunks read" color="hyper" value={k} min={1} max={10} step={1} onChange={setK} fmt={(v) => String(v)} />
        <Slider label="α (weight on retrieval)" color="hyper" value={alpha} min={0} max={1} step={0.05} onChange={setAlpha} fmt={(v) => v.toFixed(2)} />
      </div>
      <div className={styles.scroll}>
        <table className={styles.table}>
          <thead><tr><th>ret. rank</th><th>chunk (page)</th><th>cosine</th><th>reader span</th><th>span score</th><th>norm. ret.</th><th>norm. reader</th><th>final</th></tr></thead>
          <tbody>
            {Q.cands.slice(0, k).map((c) => {
              const r = res.rows.find((x) => x.r === c.r);
              const isBest = res.best && res.best.r === c.r;
              return (
                <tr key={c.r} className={isBest ? styles.hlRow : undefined} style={c.ok ? {background: 'rgba(80, 160, 70, 0.18)'} : undefined}>
                  <td>{c.r}</td><td>{c.c} ({c.p}){c.w < 60 ? `, ${c.w} words` : ''}</td><td>{c.cos.toFixed(4)}</td>
                  <td>{c.a || <em>(none)</em>}</td><td>{c.s === null ? '–' : c.s.toFixed(2)}</td>
                  <td>{r ? r.nr.toFixed(3) : '–'}</td><td>{r ? r.ns.toFixed(3) : '–'}</td><td>{r ? r.f.toFixed(3) : '–'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <Readout items={[
        ['picked answer', res.best ? `"${res.best.a}" ${res.best.ok ? '✓ right' : '✗ wrong'}` : 'none', res.best && res.best.ok ? 'data' : 'loss'],
        ['first chunk containing the answer', Q.goldRank ? `retrieval rank ${Q.goldRank}` : 'not in the top 10', 'pred'],
        ['recall@k over 12 questions', `${recall} / 12`, 'pred'],
        ['answers right over 12 questions', `${total} / 12 (brute force over all chunks: ${bf} / 12)`, 'hyper'],
      ]} />
      <div className={styles.note}>Try α = 0 and α = 1 at k = 5: ranking by retrieval alone (α = 1) is right only 3 times, because the most similar chunk is often not the one that answers. Raising k raises recall but hands the reader more distractors.</div>
    </Widget>
  );
}

/** Cosine vs dot product vs L2 distance: identical rankings for unit vectors, different ones when lengths differ. */
export function CosineVsL2() {
  const [aDeg, setADeg] = useState(20);
  const [aLen, setALen] = useState(3);
  const [bDeg, setBDeg] = useState(40);
  const [bLen, setBLen] = useState(1);
  const [unit, setUnit] = useState(false);
  const q = M.vec2(0);
  const A = M.vec2(aDeg, unit ? 1 : aLen), B = M.vec2(bDeg, unit ? 1 : bLen);
  const ma = M.vecMetrics(q, A), mb = M.vecMetrics(q, B);
  const win = (x, y, higher) => ((higher ? x > y : x < y) ? 'A' : 'B');
  const S = 50, ox = 200, oy = 190;
  const pt = (v) => [ox + S * v[0], oy - S * v[1]];
  return (
    <Widget title="Cosine, dot product and L2: when do they agree?"
      hint="Query q points along the x-axis with length 1. Two documents A and B have an angle to q and a length. With unit vectors (tick the box) all three metrics rank the same way, because squared L2 = 2 − 2·cosine. all-MiniLM-L6-v2 already outputs unit vectors, so the notebook's L2 and inner-product indexes return the same chunks.">
      <div className={styles.controls}>
        <Slider label="angle of A (degrees)" color="data" value={aDeg} min={0} max={180} step={1} onChange={setADeg} fmt={(v) => String(v)} />
        <Slider label="length of A" color="data" value={aLen} min={0.2} max={3.5} step={0.1} onChange={setALen} fmt={(v) => v.toFixed(1)} />
        <Slider label="angle of B (degrees)" color="pred" value={bDeg} min={0} max={180} step={1} onChange={setBDeg} fmt={(v) => String(v)} />
        <Slider label="length of B" color="pred" value={bLen} min={0.2} max={3.5} step={0.1} onChange={setBLen} fmt={(v) => v.toFixed(1)} />
        <label className={styles.select}><span>normalize to unit length</span>
          <input type="checkbox" checked={unit} onChange={(e) => setUnit(e.target.checked)} /></label>
      </div>
      <svg viewBox="0 0 400 210" style={{width: '100%', maxHeight: 230}}>
        <line x1={0} y1={oy} x2={400} y2={oy} stroke="var(--ifm-color-emphasis-300)" />
        <circle cx={ox} cy={oy} r={S} fill="none" stroke="var(--ifm-color-emphasis-300)" strokeDasharray="3 3" />
        {[[q, 'var(--eq-hyper)', 'q'], [A, 'var(--eq-data)', 'A'], [B, 'var(--eq-pred)', 'B']].map(([v, c, n]) => {
          const [x, y] = pt(v);
          return <g key={n}><line x1={ox} y1={oy} x2={x} y2={y} stroke={c} strokeWidth={2.5} /><circle cx={x} cy={y} r={4} fill={c} /><text x={x + 5} y={y - 5} fontSize={12} fill={c}>{n}</text></g>;
        })}
      </svg>
      <div className={styles.scroll}>
        <table className={styles.table}>
          <thead><tr><th>metric</th><th>q vs A</th><th>q vs B</th><th>closer</th></tr></thead>
          <tbody>
            <tr><td>cosine (higher = closer)</td><td>{ma.cos.toFixed(4)}</td><td>{mb.cos.toFixed(4)}</td><td>{win(ma.cos, mb.cos, true)}</td></tr>
            <tr><td>dot product / inner product (higher)</td><td>{ma.dot.toFixed(4)}</td><td>{mb.dot.toFixed(4)}</td><td>{win(ma.dot, mb.dot, true)}</td></tr>
            <tr><td>squared L2 distance (lower)</td><td>{ma.l2sq.toFixed(4)}</td><td>{mb.l2sq.toFixed(4)}</td><td>{win(ma.l2sq, mb.l2sq, false)}</td></tr>
            <tr><td>2 − 2 · cosine</td><td>{(2 - 2 * ma.cos).toFixed(4)}</td><td>{(2 - 2 * mb.cos).toFixed(4)}</td><td>{win(2 - 2 * ma.cos, 2 - 2 * mb.cos, false)}</td></tr>
          </tbody>
        </table>
      </div>
      <div className={styles.note}>{unit ? 'Unit vectors: squared L2 equals 2 − 2·cosine exactly, so every metric picks the same document.' : 'Unnormalized: the long vector A wins on cosine and dot product but loses on L2 distance, because L2 also measures length. Tick the box to see them agree.'}</div>
    </Widget>
  );
}

/** The whole RAG system's decision on 19 real questions: change the route and the two refusal thresholds. */
export function RagSystemLab() {
  const url = useBaseUrl('/data/deep-learning/rag_eval.json');
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [qi, setQi] = useState('4');
  const [mode, setMode] = useState('auto');
  const [minCos, setMinCos] = useState(0.2);
  const [minSupport, setMinSupport] = useState(0.5);
  useEffect(() => {
    let alive = true;
    fetch(url).then((r) => r.json()).then((d) => alive && setData(d)).catch((e) => alive && setErr(String(e)));
    return () => { alive = false; };
  }, [url]);
  const title = 'The whole RAG system: route, answer or refuse';
  if (err) return <Widget title={title}><div className={styles.note}>Could not load the runs ({err}).</div></Widget>;
  if (!data) return <Widget title={title}><div className={styles.note}>Loading 19 real question runs…</div></Widget>;
  const Q = data.questions[Number(qi)];
  const opts = {mode, minCos, minSupport};
  const d = M.ragDecide(Q, opts);
  const facts = data.questions.filter((q) => q.type === 'fact');
  const right = facts.filter((q) => M.ragDecide(q, opts).ok).length;
  const refusedFacts = facts.filter((q) => M.ragDecide(q, opts).route === 'refused').length;
  const outside = data.questions.filter((q) => q.type === 'outside');
  const outsideRefused = outside.filter((q) => M.ragDecide(q, opts).route === 'refused').length;
  const autoRoute = M.ragRoute(Q.q);
  const verdict = d.route === 'refused' ? (Q.type === 'outside' ? '✓ refused' : 'refused') : Q.type === 'outside' ? '✗ answered a question the paper cannot answer' : (d.ok === true ? '✓ right' : d.ok === false ? '✗ wrong' : 'explanation: judge it yourself');
  return (
    <Widget title={title}
      hint="Every answer here was produced by the page's pipeline on the BERT paper (hybrid search, DistilBERT-SQuAD extractive reader, FLAN-T5-base with sentence-selected prompts). The widget only replays the final decision: refuse if the best cosine is too low, pick a route, and for generated answers refuse if too few answer words appear in the context.">
      <div className={styles.controls}>
        <Select label="question" value={qi} onChange={setQi}
          options={data.questions.map((q, i) => [String(i), `${q.type === 'fact' ? 'fact' : q.type === 'explain' ? 'explain' : 'outside'} · ${q.q}`])} />
        <Select label="route" value={mode} onChange={setMode}
          options={[['auto', `auto (router says: ${autoRoute})`], ['extractive', 'always extractive reader'], ['generative', 'always FLAN-T5 generator']]} />
        <Slider label="refuse if best cosine below" color="hyper" value={minCos} min={0} max={0.6} step={0.01} onChange={setMinCos} fmt={(v) => v.toFixed(2)} />
        <Slider label="refuse generated answer if support below" color="hyper" value={minSupport} min={0} max={1} step={0.05} onChange={setMinSupport} fmt={(v) => v.toFixed(2)} />
      </div>
      <div className={styles.scroll}>
        <table className={styles.table}>
          <thead><tr><th>signal</th><th>value</th></tr></thead>
          <tbody>
            <tr><td>best cosine (is the document about this at all?)</td><td>{Q.topCos.toFixed(4)}</td></tr>
            <tr><td>extractive reader's span</td><td>{Q.ext} <em>(score {Q.extScore})</em></td></tr>
            <tr><td>FLAN-T5's answer</td><td>{Q.gen}</td></tr>
            <tr><td>support: share of its content words found in the context</td><td>{Q.genSupport.toFixed(2)}</td></tr>
            {Q.type === 'fact' && <tr><td>answer chunk in the top 5 / sentence in the prompt</td><td>{Q.goldTop5 ? 'yes' : 'no'} / {Q.goldPrompt ? 'yes' : 'no'}</td></tr>}
          </tbody>
        </table>
      </div>
      <Readout items={[
        ['system says', d.answer, d.route === 'refused' ? 'loss' : 'data'],
        ['path', `${d.route} · ${d.why}`, 'pred'],
        ['verdict', verdict, d.ok ? 'data' : 'loss'],
        ['12 factual questions right', `${right} / 12 (${refusedFacts} refused)`, 'hyper'],
        ['out-of-document questions refused', `${outsideRefused} / ${outside.length}`, 'hyper'],
      ]} />
      <div className={styles.note}>Try "What GPU was used to train BERT?" (the paper trained on TPUs and names no GPU): the extractive route answers BooksCorpus with confidence, while the generator says the document does not contain it. Then force every question to the generator: still 5 of 12 facts right, but different ones.</div>
    </Widget>
  );
}

/** The notebook's balanced prompt builder: how much of each retrieved chunk actually reaches FLAN-T5. */
export function PromptBudgetLab() {
  const [maxInput, setMaxInput] = useState(512);
  const [maxContext, setMaxContext] = useState(360);
  const [perPassage, setPerPassage] = useState(100);
  const chunkTokens = [459, 340, 384, 368, 394];
  const ids = [6, 9, 39, 10, 7];
  const b = M.balancedBudget(chunkTokens, {maxInput, maxContext, perPassage});
  const W = 400, scale = W / 522;
  return (
    <Widget title="Token budget: how much of each chunk the generator sees"
      hint="The notebook's build_balanced_token_aware_rag_prompt: 67 instruction tokens, a 2-token answer cue and a 12-token safety margin come first; each retrieved passage is then cut to a per-passage cap. Chunk sizes are the real FLAN-T5 token counts of the top-5 chunks for 'What is the full form of BERT?'. Counts are approximate: trimming and re-encoding can move a passage by a few tokens.">
      <div className={styles.controls}>
        <Slider label="model max input tokens" color="hyper" value={maxInput} min={256} max={1024} step={32} onChange={setMaxInput} fmt={(v) => String(v)} />
        <Slider label="max context tokens" color="hyper" value={maxContext} min={60} max={900} step={10} onChange={setMaxContext} fmt={(v) => String(v)} />
        <Slider label="cap per passage" color="hyper" value={perPassage} min={20} max={500} step={10} onChange={setPerPassage} fmt={(v) => String(v)} />
      </div>
      <svg viewBox={`0 0 600 ${40 + 34 * chunkTokens.length}`} style={{width: '100%', maxHeight: 260}}>
        {chunkTokens.map((t, i) => {
          const p = b.passages.find((x) => x.i === i);
          const kept = p ? p.kept : 0;
          return (
            <g key={i}>
              <text x={0} y={30 + 34 * i} fontSize={11} fill="var(--ifm-font-color-base)">{`chunk ${ids[i]}`}</text>
              <rect x={60} y={18 + 34 * i} width={t * scale} height={18} rx={3} fill="var(--ifm-color-emphasis-200)" />
              <rect x={60} y={18 + 34 * i} width={kept * scale} height={18} rx={3} fill="var(--eq-data)" opacity={0.75} />
              <text x={66 + t * scale} y={31 + 34 * i} fontSize={10} fill="var(--ifm-font-color-base)">{p ? `${kept} of ${t} kept (${Math.round((100 * kept) / t)}%)` : `0 of ${t}: no room`}</text>
            </g>
          );
        })}
      </svg>
      <Readout items={[
        ['room for context', `min(${maxInput} − 67 − 2 − 12, ${maxContext}) = ${b.available} tokens`, 'hyper'],
        ['passages that fit', `${b.passages.length} of ${chunkTokens.length}`, 'data'],
        ['prompt tokens', `${b.total} ${b.total > maxInput ? '> limit: the end would be cut' : `≤ ${maxInput}`}`, b.total > maxInput ? 'loss' : 'pred'],
        ['retrieved tokens never shown to the model', `${chunkTokens.reduce((s, t) => s + t, 0) - b.passages.reduce((s, p) => s + p.kept, 0)} of ${chunkTokens.reduce((s, t) => s + t, 0)}`, 'loss'],
      ]} />
      <div className={styles.note}>With the notebook's settings (512 / 360 / 100), 3 passages fit and each keeps only its first 100 tokens: about a quarter of each chunk. If the answer sentence sits later in the chunk, the generator never sees it, which is why this page picks the most relevant sentences instead of the first tokens.</div>
    </Widget>
  );
}

/** IoU of two boxes you move, then non-maximum suppression on 8 candidate boxes (same rule as torchvision.ops.nms). */
export function BoxNmsLab() {
  const [dx, setDx] = useState(20);
  const [dy, setDy] = useState(20);
  const [size, setSize] = useState(40);
  const [iouThr, setIouThr] = useState(0.5);
  const [scoreThr, setScoreThr] = useState(0);
  const A = [10, 10, 50, 50];
  const B = [10 + dx, 10 + dy, 10 + dx + size, 10 + dy + size];
  const iou = M.boxIou(A, B);
  const iw = Math.max(0, Math.min(A[2], B[2]) - Math.max(A[0], B[0])), ih = Math.max(0, Math.min(A[3], B[3]) - Math.max(A[1], B[1]));
  const boxes = [[10, 20, 46, 80], [12, 18, 48, 78], [8, 24, 44, 84], [14, 22, 50, 82], [56, 30, 92, 86], [58, 28, 94, 84], [54, 34, 90, 88], [30, 40, 70, 70]];
  const scores = [0.95, 0.9, 0.8, 0.6, 0.92, 0.85, 0.55, 0.4];
  const keep = M.nmsKeep(boxes, scores, iouThr, scoreThr);
  const s = 2.2;
  return (
    <Widget title="IoU and non-maximum suppression"
      hint="Left: box A is fixed at (10, 10, 50, 50); move and resize box B to see how IoU = overlap / union reacts. Right: 8 candidate boxes a detector might output for two cats. NMS keeps the best-scoring box, deletes every box whose IoU with it is above the threshold, and repeats; the score threshold drops weak boxes first.">
      <div className={styles.controls}>
        <Slider label="B shift right" color="data" value={dx} min={-40} max={60} step={1} onChange={setDx} fmt={(v) => String(v)} />
        <Slider label="B shift down" color="data" value={dy} min={-40} max={60} step={1} onChange={setDy} fmt={(v) => String(v)} />
        <Slider label="B size" color="data" value={size} min={10} max={80} step={1} onChange={setSize} fmt={(v) => String(v)} />
        <Slider label="NMS IoU threshold" color="hyper" value={iouThr} min={0.1} max={0.95} step={0.05} onChange={setIouThr} fmt={(v) => v.toFixed(2)} />
        <Slider label="score threshold" color="hyper" value={scoreThr} min={0} max={0.95} step={0.05} onChange={setScoreThr} fmt={(v) => v.toFixed(2)} />
      </div>
      <svg viewBox="0 0 520 250" style={{width: '100%', maxHeight: 280}}>
        <g transform="translate(15,15)">
          <rect x={0} y={0} width={220} height={220} fill="none" stroke="var(--ifm-color-emphasis-300)" />
          <g transform="translate(60,60) scale(1.1)">
            <rect x={A[0]} y={A[1]} width={40} height={40} fill="var(--eq-data)" fillOpacity={0.18} stroke="var(--eq-data)" strokeWidth={1.5} />
            <rect x={B[0]} y={B[1]} width={size} height={size} fill="var(--eq-pred)" fillOpacity={0.18} stroke="var(--eq-pred)" strokeWidth={1.5} />
            {iw * ih > 0 && <rect x={Math.max(A[0], B[0])} y={Math.max(A[1], B[1])} width={iw} height={ih} fill="var(--eq-hyper)" fillOpacity={0.5} />}
          </g>
          <text x={6} y={14} fontSize={11} fill="var(--ifm-font-color-base)">A (blue), B (green), overlap (shaded)</text>
        </g>
        <g transform="translate(270,15)">
          <rect x={0} y={0} width={220} height={220} fill="none" stroke="var(--ifm-color-emphasis-300)" />
          {boxes.map((b, i) => {
            const k = keep.includes(i);
            return (
              <g key={i}>
                <rect x={b[0] * s} y={b[1] * s} width={(b[2] - b[0]) * s} height={(b[3] - b[1]) * s} fill="none"
                  stroke={k ? 'var(--eq-pred)' : 'var(--eq-loss)'} strokeOpacity={k ? 1 : 0.35} strokeWidth={k ? 2.5 : 1} strokeDasharray={k ? undefined : '3 3'} />
                {k && <text x={b[0] * s + 3} y={b[1] * s + 12} fontSize={10} fill="var(--eq-pred)">{`box ${i} · ${scores[i].toFixed(2)}`}</text>}
              </g>
            );
          })}
        </g>
      </svg>
      <Readout items={[
        ['IoU(A, B)', `${(iw * ih).toFixed(0)} / ${(1600 + size * size - iw * ih).toFixed(0)} = ${iou.toFixed(3)}`, 'hyper'],
        ['counts as a match at 0.5?', iou >= 0.5 ? 'yes' : 'no', iou >= 0.5 ? 'data' : 'loss'],
        ['boxes kept by NMS', keep.length ? keep.join(', ') : 'none', 'pred'],
        ['boxes removed', `${8 - keep.length} of 8`, 'loss'],
      ]} />
      <div className={styles.note}>At the default 0.5, NMS keeps boxes 0, 4 and 7. Box 7 (score 0.40) barely overlaps anything, so NMS cannot remove it; raise the score threshold above 0.40 to drop it. At 0.9, nothing overlaps enough and all 8 boxes survive.</div>
    </Widget>
  );
}

/** Autoencoder anomaly detection on real reconstruction errors: choose the alarm threshold. */
export function AnomalyThresholdLab() {
  const url = useBaseUrl('/data/deep-learning/ae_anomaly.json');
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [thr, setThr] = useState(0.031);
  useEffect(() => {
    let alive = true;
    fetch(url).then((r) => r.json()).then((d) => alive && setData(d)).catch((e) => alive && setErr(String(e)));
    return () => { alive = false; };
  }, [url]);
  const title = 'Anomaly detection with reconstruction error';
  if (err) return <Widget title={title}><div className={styles.note}>Could not load the errors ({err}).</div></Widget>;
  if (!data) return <Widget title={title}><div className={styles.note}>Loading 450 reconstruction errors…</div></Widget>;
  const st = M.thresholdStats(data.err, data.label, thr);
  const maxE = 0.09, nb = 36, W = 500, H = 150;
  const bins = (lab) => {
    const c = new Array(nb).fill(0);
    data.err.forEach((e, i) => { if (data.label[i] === lab) c[Math.min(nb - 1, Math.floor((e / maxE) * nb))]++; });
    const tot = c.reduce((a, b) => a + b, 0);
    return c.map((v) => v / tot);
  };
  const n0 = bins(0), n1 = bins(1), top = Math.max(...n0, ...n1);
  const bw = W / nb, X = (e) => (e / maxE) * W;
  return (
    <Widget title={title}
      hint="An 8-number autoencoder was trained only on digits 0 to 8. On 450 test digits, each image's reconstruction error (mean squared error per pixel) is its anomaly score; digit 9 was never seen in training. Move the alarm threshold: everything to the right of it is flagged.">
      <div className={styles.controls}>
        <Slider label="alarm threshold (MSE)" color="hyper" value={thr} min={0.005} max={0.08} step={0.001} onChange={setThr} fmt={(v) => v.toFixed(3)} />
      </div>
      <svg viewBox={`0 0 ${W} ${H + 24}`} style={{width: '100%', maxHeight: 220}}>
        {n0.map((v, i) => <rect key={`a${i}`} x={i * bw} y={H - (v / top) * H} width={bw - 1} height={(v / top) * H} fill="var(--eq-data)" fillOpacity={0.55} />)}
        {n1.map((v, i) => <rect key={`b${i}`} x={i * bw + bw * 0.25} y={H - (v / top) * H} width={bw * 0.5} height={(v / top) * H} fill="var(--eq-loss)" fillOpacity={0.8} />)}
        <line x1={X(thr)} y1={0} x2={X(thr)} y2={H} stroke="var(--eq-hyper)" strokeWidth={2} strokeDasharray="4 3" />
        <line x1={0} y1={H} x2={W} y2={H} stroke="var(--ifm-color-emphasis-400)" />
        <text x={2} y={H + 16} fontSize={11} fill="var(--ifm-font-color-base)">0</text>
        <text x={W - 2} y={H + 16} fontSize={11} textAnchor="end" fill="var(--ifm-font-color-base)">0.09</text>
        <text x={W / 2} y={H + 16} fontSize={11} textAnchor="middle" fill="var(--ifm-font-color-base)">reconstruction error · blue: digits 0 to 8 (405) · red: digit 9 (45)</text>
      </svg>
      <Readout items={[
        ['9s caught (recall)', `${st.tp} of ${st.pos} = ${(100 * st.recall).toFixed(1)}%`, 'data'],
        ['normal digits flagged (false alarms)', `${st.fp} of ${st.neg} = ${(100 * st.falseAlarm).toFixed(1)}%`, 'loss'],
        ['precision of an alarm', st.tp + st.fp ? `${(100 * st.precision).toFixed(1)}%` : 'no alarms', 'pred'],
        ['ROC AUC (threshold-free)', data.auc.toFixed(4), 'hyper'],
      ]} />
      <div className={styles.note}>At 0.031 (the 95th percentile of normal errors) the alarm catches 17 of 45 nines while flagging 21 normal digits. Many 9s are rebuilt almost as well as normal digits (the two histograms overlap), so a low false-alarm rate costs recall: the threshold is a business choice, not a model property.</div>
    </Widget>
  );
}

/* =================================================================================== */
/* AI Engineering: LLM APIs and prompt engineering                                     */
/* =================================================================================== */

export function SamplingFilterPlayground({preset = 'open'}) {
  const [which, setWhich] = useState(preset);
  const [T, setT] = useState(1);
  const [topK, setTopK] = useState(0);
  const [topP, setTopP] = useState(0.9);
  const [minP, setMinP] = useState(0);
  const d = M.LLM_NEXT_TOKEN[which];
  const r = M.samplingFilter(d.logits, {T, topK, topP, minP});
  const names = d.tokens.map((t) => JSON.stringify(t).slice(1, -1));
  const colors = r.keep.map((k, i) => (!k ? 'var(--ifm-color-emphasis-300)' : i === 0 ? 'var(--w-hl)' : 'var(--w-bar)'));
  return (
    <Widget title="Sampling filters on real next-token logits (Qwen2.5-0.5B-Instruct, top 10 candidates)"
      hint="Hugging Face order: divide by T, then top-k, then top-p, then min-p; the survivors are renormalised and one is drawn at random. Grey bars can never be chosen. Probabilities are over these 10 candidates only, so they differ from full-vocabulary numbers.">
      <div className={styles.controls}>
        <Select label="prompt" value={which} onChange={setWhich}
          options={[['open', 'open: name a coffee shop in Pune'], ['closed', 'closed: is Pune in Maharashtra? yes/no']]} />
        <Slider label="temperature T" color="hyper" value={T} min={0.1} max={3} step={0.05} onChange={setT} fmt={f2} />
        <Slider label="top-k (0 = off)" color="hyper" value={topK} min={0} max={10} step={1} onChange={setTopK} fmt={(v) => String(v)} />
        <Slider label="top-p (1 = off)" color="hyper" value={topP} min={0.05} max={1} step={0.01} onChange={setTopP} fmt={f2} />
        <Slider label="min-p (0 = off)" color="hyper" value={minP} min={0} max={0.5} step={0.01} onChange={setMinP} fmt={f2} />
      </div>
      <Bars values={r.probs} labels={names} max={1} colors={colors} fmt={(v) => (v > 0 ? v.toFixed(3) : '')} />
      <Readout items={[
        ['tokens kept', `${r.kept} of 10`, 'pred'],
        ['top token before filtering', `${names[0]}: ${f3(r.raw[0])}`],
        ['top token after filtering', f3(Math.max(...r.probs)), 'pred'],
      ]} />
    </Widget>
  );
}

export function LlmCostLatencyCalculator() {
  const [nIn, setIn] = useState(1500);
  const [nOut, setOut] = useState(300);
  const [nCached, setCached] = useState(0);
  const [inRate, setInRate] = useState(1);
  const [outRate, setOutRate] = useState(4);
  const [cachedRate, setCachedRate] = useState(0.1);
  const [perDay, setPerDay] = useState(10000);
  const [ttft, setTtft] = useState(0.213);
  const [tpot, setTpot] = useState(0.0824);
  const c = Math.min(nCached, nIn);
  const cost = M.llmRequestCost({nIn, nOut, nCached: c, inRate, outRate, cachedRate});
  const lat = M.llmLatency(ttft, tpot, nOut);
  const outShare = (nOut * outRate) / ((nIn - c) * inRate + c * cachedRate + nOut * outRate);
  const usd = (v) => `$${v < 0.01 ? v.toFixed(5) : v.toFixed(2)}`;
  return (
    <Widget title="LLM request cost and latency (illustrative rates, not a price list)"
      hint="Cost = (uncached input × input rate + cached input × cached rate + output × output rate) / 1M. Latency without streaming = TTFT + TPOT × (n − 1). Defaults: the page's worked example and the TTFT/TPOT measured for Qwen2.5-0.5B on CPU.">
      <div className={styles.controls}>
        <Slider label="input tokens" color="data" value={nIn} min={0} max={20000} step={50} onChange={setIn} fmt={(v) => v.toLocaleString('en-IN')} />
        <Slider label="cached input tokens" color="data" value={nCached} min={0} max={20000} step={50} onChange={setCached} fmt={(v) => Math.min(v, nIn).toLocaleString('en-IN')} />
        <Slider label="output tokens n" color="pred" value={nOut} min={1} max={4000} step={1} onChange={setOut} fmt={(v) => v.toLocaleString('en-IN')} />
        <Slider label="input $ / 1M" color="hyper" value={inRate} min={0.05} max={15} step={0.05} onChange={setInRate} fmt={f2} />
        <Slider label="cached input $ / 1M" color="hyper" value={cachedRate} min={0} max={15} step={0.01} onChange={setCachedRate} fmt={f2} />
        <Slider label="output $ / 1M" color="hyper" value={outRate} min={0.05} max={60} step={0.05} onChange={setOutRate} fmt={f2} />
        <Slider label="requests per day" color="data" value={perDay} min={100} max={1000000} step={100} onChange={setPerDay} fmt={(v) => v.toLocaleString('en-IN')} />
        <Slider label="TTFT (s)" color="hyper" value={ttft} min={0.05} max={10} step={0.001} onChange={setTtft} fmt={(v) => v.toFixed(3)} />
        <Slider label="TPOT (s per token)" color="hyper" value={tpot} min={0.005} max={0.2} step={0.0001} onChange={setTpot} fmt={(v) => v.toFixed(4)} />
      </div>
      <Readout items={[
        ['cost per request', usd(cost), 'loss'],
        ['cost per day', usd(cost * perDay), 'loss'],
        ['cost per 30 days', usd(cost * perDay * 30), 'loss'],
        ['output share of the bill', `${(100 * outShare).toFixed(1)}%`],
        ['time to first token', `${ttft.toFixed(3)} s`, 'pred'],
        ['full answer (no streaming)', `${lat.toFixed(2)} s`, 'pred'],
      ]} />
    </Widget>
  );
}

export function AgentLoopBudget() {
  const [p, setP] = useState(0.95);
  const [K, setK] = useState(5);
  const [P, setPrompt] = useState(674);
  const [o, setO] = useState(82);
  const [inRate, setInRate] = useState(1);
  const succ = M.agentSuccess(p, K);
  const total = M.agentInputTokens(P, o, K);
  const last = M.agentStepPrompt(P, o, K);
  const steps = Array.from({length: Math.min(K, 12)}, (_, i) => M.agentStepPrompt(P, o, i + 1));
  return (
    <Widget title="Agent loop budget: reliability and context growth"
      hint="Each tool round is one more model call that re-sends everything so far. Defaults: the page's measured loop (674 prompt tokens on call 1, +82 tokens per round) and the book's 0.95 per-step accuracy. Rates are illustrative dollars per 1M input tokens.">
      <div className={styles.controls}>
        <Slider label="per-step success p" color="pred" value={p} min={0.5} max={1} step={0.005} onChange={setP} fmt={(v) => v.toFixed(3)} />
        <Slider label="model calls K" color="hyper" value={K} min={1} max={100} step={1} onChange={setK} fmt={(v) => String(v)} />
        <Slider label="call-1 prompt P (tokens)" color="data" value={P} min={50} max={20000} step={1} onChange={setPrompt} fmt={(v) => v.toLocaleString('en-IN')} />
        <Slider label="tokens added per round o" color="data" value={o} min={0} max={5000} step={1} onChange={setO} fmt={(v) => v.toLocaleString('en-IN')} />
        <Slider label="input $ / 1M" color="hyper" value={inRate} min={0.05} max={15} step={0.05} onChange={setInRate} fmt={f2} />
      </div>
      <Bars values={steps} labels={steps.map((_, i) => `call ${i + 1}`)} max={1.15 * Math.max(...steps)} fmt={(v) => v.toLocaleString('en-IN')} />
      <Readout items={[
        ['P(all K steps right) = p^K', succ < 0.0001 ? succ.toExponential(2) : succ.toFixed(4), 'pred'],
        ['prompt of the last call', `${last.toLocaleString('en-IN')} tokens`, 'data'],
        ['input tokens over the task', total.toLocaleString('en-IN'), 'data'],
        ['input cost of the task', `$${((total * inRate) / 1e6).toFixed(5)}`, 'loss'],
      ]} />
    </Widget>
  );
}
