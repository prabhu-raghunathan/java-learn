/**
 * Pure maths behind the interactive widgets (no React), so it can be unit-tested with node:
 *   node src/components/Widgets/math.test.mjs
 * Every default value matches a verified number on the page that embeds the widget.
 */

// ---------- activations ------------------------------------------------------------
export const sigmoid = (z) => 1 / (1 + Math.exp(-z));
export const ACTIVATIONS = {
  relu: {label: 'ReLU', f: (z) => Math.max(0, z), df: (z) => (z > 0 ? 1 : 0), range: '[0, ∞)', maxSlope: '1 (for z > 0)'},
  sigmoid: {label: 'Sigmoid', f: sigmoid, df: (z) => sigmoid(z) * (1 - sigmoid(z)), range: '(0, 1)', maxSlope: '0.25 (at z = 0)'},
  tanh: {label: 'Tanh', f: Math.tanh, df: (z) => 1 - Math.tanh(z) ** 2, range: '(−1, 1)', maxSlope: '1 (at z = 0)'},
};

// ---------- gradient descent on L(w) = (w - target)^2 ----------------------------
export function gradientDescent({w0 = 0, lr = 0.1, steps = 10, target = 3}) {
  const ws = [w0];
  for (let k = 0; k < steps; k++) {
    const w = ws[ws.length - 1];
    ws.push(w - lr * 2 * (w - target));
  }
  const factor = 1 - 2 * lr;          // distance to the minimum is multiplied by this each step
  let regime;
  if (Math.abs(factor) < 1e-9) regime = 'one step: lands exactly on the minimum';
  else if (Math.abs(factor) < 1) regime = factor > 0 ? 'converges smoothly' : 'converges, but zig-zags across the minimum';
  else if (Math.abs(Math.abs(factor) - 1) < 1e-9) regime = 'bounces forever (never converges)';
  else regime = 'diverges: every step lands further away';
  return {ws, factor, regime};
}

// ---------- chain rule example: c = a + b, d = a b, e = c d, f = e + d ------------
export function chainRule(a, b) {
  const c = a + b, d = a * b, e = c * d, f = e + d;
  const df_de = 1, df_dc = d, df_dd = c + 1;          // d feeds e AND f: its gradients add
  const df_da = df_dc * 1 + df_dd * b;                // via c, via d
  const df_db = df_dc * 1 + df_dd * a;
  return {c, d, e, f, df_de, df_dc, df_dd, df_da, df_db};
}

// ---------- broadcasting ------------------------------------------------------------
export function parseShape(s) {
  const nums = (s.match(/-?\d+/g) || []).map(Number);
  return nums;
}

export function broadcast(a, b) {
  const n = Math.max(a.length, b.length);
  const A = Array(n - a.length).fill(null).concat(a);
  const B = Array(n - b.length).fill(null).concat(b);
  const out = [], rows = [];
  let ok = true;
  for (let i = 0; i < n; i++) {
    const x = A[i], y = B[i];
    let r, why;
    if (x === null) { r = y; why = 'missing on the left → treated as 1, stretched'; }
    else if (y === null) { r = x; why = 'missing on the right → treated as 1, stretched'; }
    else if (x === y) { r = x; why = 'equal'; }
    else if (x === 1) { r = y; why = 'left is 1 → stretched'; }
    else if (y === 1) { r = x; why = 'right is 1 → stretched'; }
    else { r = null; why = `${x} vs ${y}: neither is 1 → error`; ok = false; }
    out.push(r);
    rows.push({a: x, b: y, r, why});
  }
  return {ok, shape: ok ? out : null, rows};
}

// ---------- convolution / pooling output size --------------------------------------
export function convOut(n, k, s, p) {
  const raw = (n - k + 2 * p) / s + 1;
  return {size: Math.floor(raw), exact: Number.isInteger((n - k + 2 * p) / s), raw};
}

export function convParams(cin, cout, k) {
  return cout * (cin * k * k + 1);
}

/** Layer specs: conv{cout,k,s,p}, pool{k,s}, flatten, fc{out}. Returns per-layer shapes and params. */
export function traceCnn(input, layers) {
  let [c, h, w] = input;
  let flat = null;
  const rows = [{name: 'Input', shape: `(${c}, ${h}, ${w})`, params: 0}];
  let total = 0, error = null;
  for (const L of layers) {
    let params = 0, shape;
    if (L.type === 'conv') {
      params = convParams(c, L.cout, L.k);
      h = convOut(h, L.k, L.s ?? 1, L.p ?? 0).size;
      w = convOut(w, L.k, L.s ?? 1, L.p ?? 0).size;
      c = L.cout;
      shape = `(${c}, ${h}, ${w})`;
    } else if (L.type === 'pool') {
      h = convOut(h, L.k, L.s ?? L.k, 0).size;
      w = convOut(w, L.k, L.s ?? L.k, 0).size;
      shape = `(${c}, ${h}, ${w})`;
    } else if (L.type === 'flatten') {
      flat = c * h * w;
      shape = `(${flat})`;
    } else if (L.type === 'fc') {
      params = L.out * (flat + 1);
      flat = L.out;
      shape = `(${flat})`;
    }
    if (h < 1 || w < 1) error = `${L.name} would produce a size below 1: the input is too small for this stack.`;
    total += params;
    rows.push({name: L.name, shape, params});
  }
  return {rows, total, error};
}

export const CNN_PRESETS = {
  SmallCNN: {
    note: 'notebook 4 (CNN 2)', input: [3, 32, 32], sizes: [16, 32, 48, 64, 96, 128],
    layers: [
      {name: 'conv1 3→16, k3 p1 + ReLU', type: 'conv', cout: 16, k: 3, p: 1},
      {name: 'MaxPool 2', type: 'pool', k: 2},
      {name: 'conv2 16→32, k3 p1 + ReLU', type: 'conv', cout: 32, k: 3, p: 1},
      {name: 'MaxPool 2', type: 'pool', k: 2},
      {name: 'Flatten', type: 'flatten'},
      {name: 'fc1 → 64 + ReLU', type: 'fc', out: 64},
      {name: 'fc2 → 10', type: 'fc', out: 10},
    ],
  },
  SimpleCNN: {
    note: 'notebook 5 (cats vs dogs)', input: [3, 128, 128], sizes: [32, 64, 96, 128, 160, 224],
    layers: [
      {name: 'conv1 3→32, k3 p1 + ReLU', type: 'conv', cout: 32, k: 3, p: 1},
      {name: 'MaxPool 2', type: 'pool', k: 2},
      {name: 'conv2 32→64 + ReLU', type: 'conv', cout: 64, k: 3, p: 1},
      {name: 'MaxPool 2', type: 'pool', k: 2},
      {name: 'conv3 64→128 + ReLU', type: 'conv', cout: 128, k: 3, p: 1},
      {name: 'MaxPool 2', type: 'pool', k: 2},
      {name: 'Flatten', type: 'flatten'},
      {name: 'fc1 → 128 + ReLU + Dropout', type: 'fc', out: 128},
      {name: 'fc2 → 2', type: 'fc', out: 2},
    ],
  },
};

// ---------- softmax / temperature / cross-entropy -------------------------------------
export function softmax(logits, T = 1) {
  const z = logits.map((v) => v / T);
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export const entropy = (p) => -p.reduce((a, q) => a + (q > 0 ? q * Math.log(q) : 0), 0);
export const crossEntropy = (p, y) => -Math.log(Math.max(p[y], 1e-300));

/** Seeded RNG (mulberry32) so a "sample" button is reproducible within a session. */
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function sampleCounts(p, n, rand) {
  const counts = p.map(() => 0);
  for (let k = 0; k < n; k++) {
    let u = rand(), i = 0;
    while (i < p.length - 1 && u >= p[i]) { u -= p[i]; i++; }
    counts[i]++;
  }
  return counts;
}

// ---------- TF-IDF exactly like scikit-learn's TfidfVectorizer defaults -----------------
// lowercase, token_pattern (?u)\b\w\w+\b, smooth_idf: idf = ln((1 + n) / (1 + df)) + 1, l2 norm
export function tokenize(doc) {
  return (doc.toLowerCase().match(/\b\w\w+\b/gu) || []);
}

export function tfidf(docs) {
  const toks = docs.map(tokenize);
  const vocab = [...new Set(toks.flat())].sort();
  const n = docs.length;
  const df = vocab.map((w) => toks.filter((t) => t.includes(w)).length);
  const idf = df.map((d) => Math.log((1 + n) / (1 + d)) + 1);
  const rows = toks.map((t) => {
    const v = vocab.map((w, j) => t.filter((x) => x === w).length * idf[j]);
    const norm = Math.sqrt(v.reduce((a, x) => a + x * x, 0)) || 1;
    return v.map((x) => x / norm);
  });
  return {vocab, df, idf, rows};
}

export const cosine = (a, b) => {
  const dot = a.reduce((s, x, i) => s + x * b[i], 0);
  const na = Math.sqrt(a.reduce((s, x) => s + x * x, 0));
  const nb = Math.sqrt(b.reduce((s, x) => s + x * x, 0));
  return na && nb ? dot / (na * nb) : 0;
};

// ---------- Word2Vec context-window pairs ----------------------------------------------
export function contextPairs(sentence, w) {
  const words = sentence.trim().split(/\s+/).filter(Boolean);
  const pairs = [];
  words.forEach((c, i) => {
    for (let j = Math.max(0, i - w); j <= Math.min(words.length - 1, i + w); j++) {
      if (j !== i) pairs.push([c, words[j]]);
    }
  });
  return {words, pairs};
}

// ---------- gradient through time -------------------------------------------------------
export const repeatedFactor = (gamma, steps) => Array.from({length: steps + 1}, (_, k) => gamma ** k);
export const tanhSlope = (z) => 1 - Math.tanh(z) ** 2;

// ---------- LSTM / GRU single dimension ---------------------------------------------------
export function lstmStep({cPrev, f, i, cTilde, o}) {
  const kept = f * cPrev, written = i * cTilde, c = kept + written;
  return {kept, written, c, tanhC: Math.tanh(c), h: o * Math.tanh(c)};
}

export const gruStep = ({hPrev, z, hTilde}) => ({
  kept: z * hPrev, added: (1 - z) * hTilde, h: z * hPrev + (1 - z) * hTilde,
});

// ---------- attention (notebook 14 toy example) -----------------------------------------
export const ATT_TOKENS = ['The', 'movie', 'was', 'not', 'good'];
export const ATT_SCORES = [0.2, 0.4, 0.1, 2.0, 1.8];
// 2-D toy value vectors: [positive-like, negation-like]
export const ATT_VALUES = [[0.1, 0.0], [0.2, 0.1], [0.1, 0.0], [0.0, 1.0], [1.0, 0.1]];
// hand-made self-attention matrix: row = query token, column = attended token, rows sum to 1
export const ATT_MATRIX = [
  [0.60, 0.20, 0.10, 0.05, 0.05],
  [0.10, 0.55, 0.15, 0.10, 0.10],
  [0.10, 0.20, 0.45, 0.15, 0.10],
  [0.05, 0.05, 0.10, 0.30, 0.50],
  [0.05, 0.05, 0.10, 0.55, 0.25],
];

/** Weighted sum of value vectors: c = Σ α_i v_i. */
export function weightedSum(weights, values) {
  return values[0].map((_, d) => values.reduce((s, v, i) => s + weights[i] * v[d], 0));
}

// --- Full self-attention (notebook 15): np.random.seed(42), X = randn(5, 4), then W_Q, W_K, W_V = randn(4, 3) ---
export const SA_TOKENS = ['The', 'movie', 'was', 'not', 'good'];
export const SA_X = [[0.49671415, -0.1382643, 0.64768854, 1.52302986], [-0.23415337, -0.23413696, 1.57921282, 0.76743473], [-0.46947439, 0.54256004, -0.46341769, -0.46572975], [0.24196227, -1.91328024, -1.72491783, -0.56228753], [-1.01283112, 0.31424733, -0.90802408, -1.4123037]];
export const SA_WQ = [[1.46564877, -0.2257763, 0.0675282], [-1.42474819, -0.54438272, 0.11092259], [-1.15099358, 0.37569802, -0.60063869], [-0.29169375, -0.60170661, 1.85227818]];
export const SA_WK = [[-0.01349722, -1.05771093, 0.82254491], [-1.22084365, 0.2088636, -1.95967012], [-1.32818605, 0.19686124, 0.73846658], [0.17136828, -0.11564828, -0.3011037]];
export const SA_WV = [[-1.47852199, -0.71984421, -0.46063877], [1.05712223, 0.34361829, -1.76304016], [0.32408397, -0.38508228, -0.676922], [0.61167629, 1.03099952, 0.93128012]];
// The notebook's scaled score row for "good" (S[good] / sqrt(3)); times sqrt(d_k) it gives the raw row at any d_k.
export const SA_GOOD_UNIT = [-0.92662514, -0.86183467, 2.22631523, -5.00070212, 2.26186867];

export const matmul = (A, B) => A.map((r) => B[0].map((_, j) => r.reduce((s, a, k) => s + a * B[k][j], 0)));
export const transpose = (A) => A[0].map((_, j) => A.map((r) => r[j]));

/** Every stage of scaled dot-product self-attention, exactly as the notebook computes it. */
export function selfAttentionStages(X = SA_X, WQ = SA_WQ, WK = SA_WK, WV = SA_WV) {
  const Q = matmul(X, WQ), K = matmul(X, WK), V = matmul(X, WV);
  const dk = Q[0].length;
  const S = matmul(Q, transpose(K));
  const Ss = S.map((r) => r.map((v) => v / Math.sqrt(dk)));
  const A = Ss.map((r) => softmax(r));
  const Z = matmul(A, V);
  return {X, Q, K, V, S, Ss, A, Z, dk};
}

// --- Multi-head attention (notebook 16): np.random.seed(42), X = randn(5, 4); per head W_Q, W_K, W_V = randn(4, 2); then W_O = randn(4, 4) ---
// (X is the same matrix as SA_X above: both notebooks start from seed 42.)
export const MHA_WQ = [[[1.46564877, -0.2257763], [0.0675282, -1.42474819], [-0.54438272, 0.11092259], [-1.15099358, 0.37569802]], [[-1.47852199, -0.71984421], [-0.46063877, 1.05712223], [0.34361829, -1.76304016], [0.32408397, -0.38508228]]];
export const MHA_WK = [[[-0.60063869, -0.29169375], [-0.60170661, 1.85227818], [-0.01349722, -1.05771093], [0.82254491, -1.22084365]], [[-0.676922, 0.61167629], [1.03099952, 0.93128012], [-0.83921752, -0.30921238], [0.33126343, 0.97554513]]];
export const MHA_WV = [[[0.2088636, -1.95967012], [-1.32818605, 0.19686124], [0.73846658, 0.17136828], [-0.11564828, -0.3011037]], [[-0.47917424, -0.18565898], [-1.10633497, -1.19620662], [0.81252582, 1.35624003], [-0.07201012, 1.0035329]]];
export const MHA_WO = [[0.36163603, -0.64511975, 0.36139561, 1.53803657], [-0.03582604, 1.56464366, -2.6197451, 0.8219025], [0.08704707, -0.29900735, 0.09176078, -1.98756891], [-0.21967189, 0.35711257, 1.47789404, -0.51827022]];

/** Multi-head attention exactly as notebook 16 computes it: per-head attention, concat, then W_O. */
export function multiHeadStages(X = SA_X, WQ = MHA_WQ, WK = MHA_WK, WV = MHA_WV, WO = MHA_WO) {
  const heads = WQ.map((_, r) => {
    const Q = matmul(X, WQ[r]), K = matmul(X, WK[r]), V = matmul(X, WV[r]);
    const dk = Q[0].length;
    const A = matmul(Q, transpose(K)).map((row) => softmax(row.map((v) => v / Math.sqrt(dk))));
    return {Q, K, V, A, out: matmul(A, V)};
  });
  const H = X.map((_, i) => heads.flatMap((hd) => hd.out[i]));
  const Z = matmul(H, WO);
  return {heads, H, Z};
}

/** Head-split arithmetic for a Transformer configuration (no biases). */
export function headSplit(dModel, h) {
  const dk = dModel / h;
  const perHead = 3 * dModel * dk;
  return {dk, valid: Number.isInteger(dk), perHead, proj: h * perHead, wo: h * dk * dModel, total: h * perHead + h * dk * dModel, singleFull: 4 * dModel * dModel};
}

// --- The Transformer (paper §3.5, §5.3) ---
/** Sinusoidal positional encoding: PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(same). */
export function positionalEncoding(pos, d) {
  const out = new Array(d);
  for (let i2 = 0; i2 < d; i2 += 2) {
    const ang = pos / 10000 ** (i2 / d);
    out[i2] = Math.sin(ang);
    if (i2 + 1 < d) out[i2 + 1] = Math.cos(ang);
  }
  return out;
}
export const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
/** Paper eq. (3): lrate = d_model^-0.5 · min(step^-0.5, step · warmup^-1.5). */
export function transformerLR(step, dModel = 512, warmup = 4000) {
  const s = Math.max(step, 1);
  return dModel ** -0.5 * Math.min(s ** -0.5, s * warmup ** -1.5);
}

// --- BERT (Devlin et al. 2019) ---
/** A toy WordPiece vocabulary (real BERT: 30,000 pieces). Single letters and ##letters guarantee no [UNK] for a-z words. */
const BERT_WORDS = ('the a an man went to store he she bought gallon of milk my dog is cute likes like play penguin are flight ' +
  'birds bird hairy apple i love deep learn bert model read book run walk jump talk good bad movie not bank river money and was ' +
  'it this that in on with for you we they transform pre train token sentence word question answer paris capital france what ' +
  'where who city sat cat mat new york').split(' ');
const BERT_SUFFIXES = 'ing ed s es er ers ly less ness ful able ation ize ization ment'.split(' ');
const LETTERS = 'abcdefghijklmnopqrstuvwxyz'.split('');
export const BERT_TOY_VOCAB = [...new Set([...BERT_WORDS, ...BERT_SUFFIXES.map((s) => '##' + s), ...LETTERS,
  ...LETTERS.map((c) => '##' + c), '.', ',', '?', '!', "'", '-'])];
/** Greedy longest-match-first WordPiece for one word (the algorithm BERT uses). */
export function wordpiece(word, vocab = BERT_TOY_VOCAB) {
  const V = vocab instanceof Set ? vocab : new Set(vocab);
  const out = [];
  let start = 0;
  while (start < word.length) {
    let end = word.length, piece = null;
    while (start < end) {
      const sub = (start > 0 ? '##' : '') + word.slice(start, end);
      if (V.has(sub)) { piece = sub; break; }
      end--;
    }
    if (piece === null) return ['[UNK]'];
    out.push(piece);
    start = end;
  }
  return out;
}
/** Lower-case, split on whitespace and punctuation, then WordPiece each word. */
export function bertTokenize(text, vocab = BERT_TOY_VOCAB) {
  const V = new Set(vocab);
  const words = text.toLowerCase().match(/[a-z0-9]+|[^\sa-z0-9]/g) || [];
  return words.flatMap((w) => wordpiece(w, V));
}
/** Figure 2 packing: [CLS] A [SEP] (B [SEP]); segment 0 for A, 1 for B; positions 0..n-1. */
export function bertPack(a, b = null) {
  const tokens = ['[CLS]', ...a, '[SEP]'];
  const segments = tokens.map(() => 0);
  if (b) {
    tokens.push(...b, '[SEP]');
    segments.push(...b.map(() => 1), 1);
  }
  return {tokens, segments, positions: tokens.map((_, i) => i)};
}
/** Small seeded PRNG (mulberry32) so the masking demo is reproducible. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
/** BERT masking: choose round(rate·n) real tokens (at least 1); each chosen one becomes [MASK] (pMask),
 * a random vocabulary token (pRand) or stays unchanged (the rest). Returns one record per token. */
export function bertMask(tokens, rng, rate = 0.15, pMask = 0.8, pRand = 0.1, vocab = BERT_WORDS) {
  const cand = tokens.map((t, i) => i).filter((i) => tokens[i] !== '[CLS]' && tokens[i] !== '[SEP]');
  const k = cand.length ? Math.max(1, Math.round(rate * cand.length)) : 0;
  for (let i = cand.length - 1; i > 0; i--) {         // Fisher-Yates shuffle, then take the first k
    const j = Math.floor(rng() * (i + 1));
    [cand[i], cand[j]] = [cand[j], cand[i]];
  }
  const chosen = new Set(cand.slice(0, k));
  return tokens.map((t, i) => {
    if (!chosen.has(i)) return {orig: t, shown: t, kind: 'none'};
    const r = rng();
    if (r < pMask) return {orig: t, shown: '[MASK]', kind: 'mask'};
    if (r < pMask + pRand) return {orig: t, shown: vocab[Math.floor(rng() * vocab.length)], kind: 'random'};
    return {orig: t, shown: t, kind: 'same'};
  });
}
/** BERT parameter count (matches HuggingFace BertModel): embeddings, L encoder layers, pooler. */
export function bertParams({L, H, V = 30522, P = 512, S = 2, ffn = 4 * H}) {
  const emb = (V + P + S) * H + 2 * H;                  // token + position + segment tables, LayerNorm
  const attn = 4 * H * H + 4 * H;                       // Q, K, V, output projections with biases
  const ff = 2 * H * ffn + ffn + H;                     // H -> 4H -> H with biases
  const layer = attn + ff + 4 * H;                      // + two LayerNorms
  const pooler = H * H + H;
  return {emb, attn, ff, layer, encoder: L * layer, pooler, total: emb + L * layer + pooler};
}
/** Real start/end scores (S·T_i, E·T_j) from the BERT page's fine-tuned tiny model. */
export const BERT_SPAN_EXAMPLES = [
  {tokens: ['[CLS]', '11', '20', '[SEP]', '19', '22', '2', '5', '8', '11', '14', '17', '20', '0', '[SEP]'], gold: [9, 12],
    start: [-4.644, -2.961, -2.713, -3.875, -4.803, -3.506, -4.91, -3.077, -3.823, 6.066, -5.438, -2.716, -3.626, -2.414, -1.54],
    end: [-0.933, -4.611, 4.607, -1.538, -2.946, -4.397, -1.229, -4.587, -3.804, -3.049, -0.915, -5.007, 6.237, -5.11, -5.646]},
  {tokens: ['[CLS]', '3', '13', '[SEP]', '14', '19', '1', '6', '11', '16', '21', '3', '8', '13', '[SEP]'], gold: [11, 13],
    start: [-2.192, -2.618, -5.51, -0.811, -1.786, 3.594, -3.533, -2.633, -2.276, -2.618, -1.879, 4.726, -1.143, -4.109, -1.198],
    end: [-4.91, -4.122, -1.111, -5.54, -5.418, -4.106, -4.011, -5.144, -5.386, -5.23, -5.627, -4.23, -5.935, 6.136, -5.935]},
  {tokens: ['[CLS]', '9', '13', '[SEP]', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '[SEP]'], gold: [4, 8],
    start: [-1.986, -0.22, -6.175, 1.21, 5.716, -0.36, -1.917, -3.772, -3.465, -2.464, -1.047, -0.802, -1.179, -1.334, -0.274],
    end: [-5.13, -5.926, 2.863, -5.934, -4.217, -6.201, -5.476, -4.052, 6.199, -4.829, -5.968, -6.07, -6.006, -5.638, -6.218]},
];
/** argmax over paragraph positions i ≤ j (j − i < maxLen) of start[i] + end[j]; also the top-k spans. */
export function bestSpan(start, end, allowed, maxLen = Infinity, k = 5) {
  const spans = [];
  for (const i of allowed) for (const j of allowed) {
    if (j >= i && j - i < maxLen) spans.push({i, j, score: start[i] + end[j]});
  }
  spans.sort((a, b) => b.score - a.score);
  return {best: spans[0], top: spans.slice(0, k)};
}

// --- Stacked encoder blocks: the mini sentiment project (notebook 18) ---
/** Notebook 18's encoder: whitespace tokens, <UNK> = 1, pad with 0 to maxLen, mask 1 = real token. */
export function encodeTiny(text, vocab, maxLen = 6) {
  const words = text.toLowerCase().split(/\s+/).filter(Boolean);
  const ids = words.map((w) => (w in vocab ? vocab[w] : 1)).slice(0, maxLen);
  const mask = ids.map(() => 1);
  while (ids.length < maxLen) { ids.push(0); mask.push(0); }
  return {words, ids, mask, unknown: words.filter((w) => !(w in vocab)), truncated: words.length > maxLen};
}
const linear = (x, W, b) => W.map((row, o) => row.reduce((s, w, i) => s + w * x[i], b[o]));
function layerNorm(x, g, b, eps = 1e-5) {
  const n = x.length, mu = x.reduce((s, v) => s + v, 0) / n;
  const varr = x.reduce((s, v) => s + (v - mu) ** 2, 0) / n;
  return x.map((v, i) => ((v - mu) / Math.sqrt(varr + eps)) * g[i] + b[i]);
}
/** Forward pass of the notebook's StackedTransformerClassifier (eval mode, no dropout), from exported weights.
 * Returns probabilities, every hidden state, and every layer's attention [head][query][key]. */
export function tinyEncoderForward(model, ids, mask, heads = 4) {
  const T = ids.length, D = model.tok[0].length, dh = D / heads;
  let X = ids.map((id, t) => model.tok[id].map((v, i) => v + model.pos[t][i]));
  const hidden = [X], attentions = [];
  for (const L of model.layers) {
    const Q = X.map((x) => linear(x, L.Wq, L.bq)), K = X.map((x) => linear(x, L.Wk, L.bk)), Vv = X.map((x) => linear(x, L.Wv, L.bv));
    const ctx = X.map(() => new Array(D).fill(0)), att = [];
    for (let h = 0; h < heads; h++) {
      const A = [];
      for (let i = 0; i < T; i++) {
        const s = [];
        for (let j = 0; j < T; j++) {
          let d = 0;
          for (let k = h * dh; k < (h + 1) * dh; k++) d += Q[i][k] * K[j][k];
          s.push(mask[j] ? d / Math.sqrt(dh) : -1e9);       // padded keys masked, as in the notebook
        }
        const m = Math.max(...s), e = s.map((v) => Math.exp(v - m)), z = e.reduce((a, b) => a + b, 0);
        const w = e.map((v) => v / z);
        A.push(w);
        for (let j = 0; j < T; j++) for (let k = h * dh; k < (h + 1) * dh; k++) ctx[i][k] += w[j] * Vv[j][k];
      }
      att.push(A);
    }
    attentions.push(att);
    const attnOut = ctx.map((c) => linear(c, L.Wo, L.bo));
    X = X.map((x, t) => layerNorm(x.map((v, i) => v + attnOut[t][i]), L.ln1g, L.ln1b));
    const ff = X.map((x) => linear(linear(x, L.W1, L.b1).map((v) => Math.max(0, v)), L.W2, L.b2));
    X = X.map((x, t) => layerNorm(x.map((v, i) => v + ff[t][i]), L.ln2g, L.ln2b));
    hidden.push(X);
  }
  const n = mask.reduce((a, b) => a + b, 0);
  const pooled = new Array(D).fill(0).map((_, k) => X.reduce((s, x, t) => s + (mask[t] ? x[k] : 0), 0) / Math.max(n, 1e-9));
  const logits = linear(pooled, model.clsW, model.clsb);
  const m = Math.max(...logits), e = logits.map((v) => Math.exp(v - m)), z = e.reduce((a, b) => a + b, 0);
  return {probs: e.map((v) => v / z), logits, pooled, hidden, attentions};
}
/** Trainable parameters of the notebook's classifier (biases included, LayerNorm γ and β). */
export function tinyEncoderParams({V = 35, T = 6, D = 32, dff = 64, L = 1, C = 2}) {
  const emb = V * D + T * D, attn = 4 * (D * D + D), ffn = D * dff + dff + dff * D + D, ln = 4 * D;
  return {emb, attn, ffn, ln, block: attn + ffn + ln, head: D * C + C, total: emb + L * (attn + ffn + ln) + D * C + C};
}

// --- Hugging Face pipelines and what they hide (notebook 19) ---
/** BERT's BasicTokenizer (uncased): clean, lower-case, strip accents, split on whitespace and punctuation. */
export function bertBasicTokenize(text) {
  const cleaned = text.replace(/[\u0000�]|\p{Cc}/gu, (c) => (/\s/.test(c) ? ' ' : ''));
  const lowered = cleaned.toLowerCase().normalize('NFD').replace(/\p{Mn}/gu, '');
  const out = [];
  for (const word of lowered.split(/\s+/).filter(Boolean)) {
    let cur = '';
    for (const ch of word) {
      const cp = ch.codePointAt(0);
      const ascii = (cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126);
      if (ascii || /\p{P}/u.test(ch)) { if (cur) out.push(cur); out.push(ch); cur = ''; } else cur += ch;
    }
    if (cur) out.push(cur);
  }
  return out;
}
/** Full uncased BERT/DistilBERT encoding: WordPiece over a real vocabulary list (index = id), [CLS] … [SEP] (… [SEP]),
 * token_type_ids, attention_mask, optional truncation and padding to maxLength. */
export function bertEncode(text, vocabList, {pair = null, maxLength = 512, padTo = 0} = {}) {
  const V = new Set(vocabList), id = (t) => vocabList.indexOf(t);
  const pieces = (s) => bertBasicTokenize(s).flatMap((w) => (w.length > 100 ? ['[UNK]'] : wordpiece(w, V)));
  let a = pieces(text), b = pair === null ? null : pieces(pair);
  const budget = maxLength - (b ? 3 : 2);
  while (a.length + (b ? b.length : 0) > budget) { if (b && b.length > a.length) b.pop(); else a.pop(); }   // longest-first
  const tokens = ['[CLS]', ...a, '[SEP]', ...(b ? [...b, '[SEP]'] : [])];
  const typeIds = tokens.map((_, i) => (b && i >= a.length + 2 ? 1 : 0));
  const mask = tokens.map(() => 1);
  while (tokens.length < padTo) { tokens.push('[PAD]'); typeIds.push(0); mask.push(0); }
  const index = new Map(vocabList.map((t, i) => [t, i]));
  return {tokens, ids: tokens.map((t) => (index.has(t) ? index.get(t) : id('[UNK]'))), typeIds, mask, truncated: a.length + (b ? b.length : 0) < pieces(text).length + (b ? pieces(pair).length : 0)};
}
/** Two-class softmax: the probability of class 1 is the sigmoid of the logit margin z1 − z0. */
export const twoClassProb = (z0, z1) => 1 / (1 + Math.exp(-(z1 - z0)));
