// Checks the widget maths against numbers verified on the pages. Run: node src/components/Widgets/math.test.mjs
import * as M from './math.js';

let fails = 0;
const close = (name, got, want, tol = 1e-4) => {
  const ok = Array.isArray(want)
    ? want.every((w, i) => Math.abs(got[i] - w) <= tol)
    : Math.abs(got - want) <= tol;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}: ${JSON.stringify(got)}${ok ? '' : ' want ' + JSON.stringify(want)}`);
  if (!ok) fails++;
};
const eq = (name, got, want) => close(name, got === want ? 1 : 0, 1);

// calculus page: w from 0, lr 0.1 -> 0.6, 1.08, 1.464, 1.771, 2.017, 2.214 ; lr 1.1 -> 6.6, -1.32, 8.184, -3.22
close('GD lr 0.1', M.gradientDescent({lr: 0.1, steps: 6}).ws.slice(1), [0.6, 1.08, 1.464, 1.7712, 2.01696, 2.213568]);
close('GD lr 1.1', M.gradientDescent({lr: 1.1, steps: 4}).ws.slice(1), [6.6, -1.32, 8.184, -3.2208]);
close('GD factor 0.05/0.4/1.05', [0.05, 0.4, 1.05].map((lr) => M.gradientDescent({lr}).factor), [0.9, 0.2, -1.1]);

// chain rule: a=2, b=3 -> c 5 d 6 e 30 f 36, df/da 24, df/db 18
const cr = M.chainRule(2, 3);
close('chain rule values', [cr.c, cr.d, cr.e, cr.f, cr.df_da, cr.df_db], [5, 6, 30, 36, 24, 18]);

// activations: sigmoid'(0)=0.25, tanh'(1)=0.420, tanh'(2)=0.0707, sigmoid'(4)=0.0177
close('activation slopes', [M.ACTIVATIONS.sigmoid.df(0), M.ACTIVATIONS.tanh.df(1), M.ACTIVATIONS.tanh.df(2), M.ACTIVATIONS.sigmoid.df(4)],
  [0.25, 0.41997, 0.07065, 0.01766]);

// broadcasting
eq('broadcast (3,1)+(1,4)', JSON.stringify(M.broadcast([3, 1], [1, 4]).shape), '[3,4]');
eq('broadcast (2,3)+(3)', JSON.stringify(M.broadcast([2, 3], [3]).shape), '[2,3]');
eq('broadcast (2,3)+(2)', M.broadcast([2, 3], [2]).ok, false);

// conv output size + CNN traces (page tables)
close('conv 32 k3 p1 s1', M.convOut(32, 3, 1, 1).size, 32);
close('conv params 3->16 k3', M.convParams(3, 16, 3), 448);
const small = M.traceCnn(M.CNN_PRESETS.SmallCNN.input, M.CNN_PRESETS.SmallCNN.layers);
close('SmallCNN params', small.rows.map((r) => r.params), [0, 448, 0, 4640, 0, 0, 131136, 650]);
close('SmallCNN total', small.total, 136874);
const simple = M.traceCnn(M.CNN_PRESETS.SimpleCNN.input, M.CNN_PRESETS.SimpleCNN.layers);
close('SimpleCNN total', simple.total, 4287938);
eq('SimpleCNN flatten', simple.rows[7].shape, '(32768)');

// softmax / temperature (LSTM page 2)
close('softmax T=0.5', M.softmax([2.5, 1.2, 0.5, -0.3], 0.5), [0.9122, 0.0678, 0.0167, 0.0034]);
close('softmax T=1', M.softmax([2.5, 1.2, 0.5, -0.3], 1), [0.6809, 0.1856, 0.0921, 0.0414]);
close('softmax T=2', M.softmax([2.5, 1.2, 0.5, -0.3], 2), [0.4681, 0.2443, 0.1722, 0.1154]);
close('softmax T=10', M.softmax([2.5, 1.2, 0.5, -0.3], 10), [0.2896, 0.2543, 0.2371, 0.2189]);

// TF-IDF (LSTM page 1, sklearn defaults)
const tf = M.tfidf(['good movie', 'not good movie', 'man bites dog', 'dog bites man']);
eq('tfidf vocab', tf.vocab.join(','), 'bites,dog,good,man,movie,not');
close('tfidf row 2', tf.rows[1], [0, 0, 0.526405, 0, 0.526405, 0.667679]);
close('cos good/not good', M.cosine(tf.rows[0], tf.rows[1]), 0.7444);
close('cos man/dog', M.cosine(tf.rows[2], tf.rows[3]), 1.0);

// context window (NLP page): 6 pairs at w=1, 10 at w=2
close('pairs w=1/w=2', [1, 2].map((w) => M.contextPairs('i love deep learning', w).pairs.length), [6, 10]);

// repeated multiplication (RNN page): 0.5^15, 1.5^15 ; forget gate 0.95^20
close('0.5^15', M.repeatedFactor(0.5, 15)[15], 0.000030518, 1e-8);
close('1.5^15', M.repeatedFactor(1.5, 15)[15], 437.89, 0.01);
close('0.95^20', M.repeatedFactor(0.95, 20)[20], 0.3585, 1e-3);

// LSTM / GRU demos (LSTM page 1)
close('lstm dims', [[0.9, 1, 0, 0.3], [0.2, 0.9, 0.2, 0.7], [-0.8, 0.1, 0.8, 0.4], [0.5, 0, 1, -0.6]]
  .map(([cPrev, f, i, cTilde]) => M.lstmStep({cPrev, f, i, cTilde, o: 1}).c), [0.9, 0.32, 0.24, -0.6]);
close('gru dims', [[0.9, 0.95, 0.1], [0.2, 0.7, 0.8], [-0.7, 0.2, 0.4], [0.5, 0.05, -0.6]]
  .map(([hPrev, z, hTilde]) => M.gruStep({hPrev, z, hTilde}).h), [0.86, 0.38, 0.18, -0.545]);


// attention (notebook 14): weights, context vector, matrix rows
const aw = M.softmax(M.ATT_SCORES, 1);
close('attention weights', aw, [0.0708, 0.0864, 0.0640, 0.4282, 0.3506]);
close('context vector', M.weightedSum(aw, M.ATT_VALUES), [0.3813, 0.4719]);
close('uniform average', M.weightedSum([0.2, 0.2, 0.2, 0.2, 0.2], M.ATT_VALUES), [0.28, 0.24]);
close('row sums', M.ATT_MATRIX.map((r) => r.reduce((a, b) => a + b, 0)), [1, 1, 1, 1, 1]);
close('good row -> new vector', M.weightedSum(M.ATT_MATRIX[4], M.ATT_VALUES), [0.275, 0.58]);
close('not row -> new vector', M.weightedSum(M.ATT_MATRIX[3], M.ATT_VALUES), [0.525, 0.355]);

// --- notebook 15: full self-attention ---
{
  const st = M.selfAttentionStages();
  close('S[good][not]', st.S[4][3], -8.66147015, 1e-6);
  close('S[not][not]', st.S[3][3], 22.4746107, 1e-5);
  close('S_scaled[good][good]', st.Ss[4][4], 2.26186867, 1e-6);
  close('A[good][was]', st.A[4][2], 0.470536813, 1e-6);
  close('A[good][not]', st.A[4][3], 0.000341932081, 1e-8);
  close('A[not][not]', st.A[3][3], 0.999970812, 1e-7);
  close('A[The][not]', st.A[0][3], 0.807538887, 1e-6);
  close('Z[good][0]', st.Z[4][0], 0.746486816, 1e-6); close('Z[good][1]', st.Z[4][1], -0.0025840467, 1e-7); close('Z[good][2]', st.Z[4][2], -0.764146038, 1e-6);
  close('V[good][0]', st.V[4][0], 0.67154219, 1e-6);
  st.A.forEach((r, i) => close(`A row ${i} sums to 1`, r.reduce((a, b) => a + b, 0), 1, 1e-12));
  const g = M.softmax(M.SA_GOOD_UNIT.map((v) => v * Math.sqrt(3)));
  close('raw softmax good row at d_k=3', g[4], 0.513150525, 1e-6);
  close('scaled softmax good row', M.softmax(M.SA_GOOD_UNIT)[4], 0.48756696, 1e-6);
}


// --- notebook 16: multi-head attention ---
{
  const mh = M.multiHeadStages();
  close('A1[good]', mh.heads[0].A[4], [0.44401522, 0.42344427, 0.01025913, 0.11836134, 0.00392004], 1e-6);
  close('A2[good]', mh.heads[1].A[4], [0.92252552, 0.02059308, 0.05336374, 0.00069109, 0.00282657], 1e-6);
  close('head1[The]', mh.heads[0].out[0], [-1.06313467, 1.91112855], 1e-6);
  close('head2[The]', mh.heads[1].out[0], [0.17528449, -1.36668523], 1e-6);
  close('H[good]', mh.H[4], [0.97685597, -0.50250718, 0.29949111, 2.25755955], 1e-6);
  close('Z[The]', mh.Z[0], [-0.13745564, 3.13561251, -7.39461358, 0.29554359], 1e-6);
  close('Z[movie]', mh.Z[1], [0.02077774, 3.11960883, -9.2089391, 1.74194754], 1e-6);
  const b = M.headSplit(512, 8);
  close('base: d_k, total params', [b.dk, b.total, b.singleFull], [64, 1048576, 1048576], 0);
  close('toy: total params', M.headSplit(4, 2).total, 64, 0);
}

// --- the Transformer: positional encoding and learning-rate schedule ---
{
  close('PE(1, 512)[0..5]', M.positionalEncoding(1, 512).slice(0, 6), [0.8415, 0.5403, 0.8219, 0.5697, 0.802, 0.5974], 1e-4);
  close('PE(10).PE(11)', M.dot(M.positionalEncoding(10, 512), M.positionalEncoding(11, 512)), 249.1, 0.05);
  close('PE(20).PE(21)', M.dot(M.positionalEncoding(20, 512), M.positionalEncoding(21, 512)), 249.1, 0.05);
  close('|PE|^2 = d/2', M.dot(M.positionalEncoding(7, 512), M.positionalEncoding(7, 512)), 256, 1e-9);
  close('lr(4000) peak', M.transformerLR(4000), 0.000699, 1e-6);
  close('lr(100000)', M.transformerLR(100000), 0.000140, 1e-6);
}
// --- BERT: WordPiece, packing, masking statistics, parameter counts, span scoring ---
{
  eq('wordpiece playing', M.wordpiece('playing').join(' '), 'play ##ing');
  eq('wordpiece flightless (paper A.1)', M.wordpiece('flightless').join(' '), 'flight ##less');
  eq('tokenize Figure 2 pair', M.bertPack(M.bertTokenize('my dog is cute'), M.bertTokenize('he likes playing')).tokens.join(' '),
    '[CLS] my dog is cute [SEP] he likes play ##ing [SEP]');
  close('Figure 2 segments', M.bertPack(['my', 'dog'], ['he']).segments, [0, 0, 0, 0, 1, 1], 0);
  const rng = M.mulberry32(0);
  let n = 0, ch = 0, mk = 0, rd = 0, sm = 0;
  const sent = Array.from({length: 100}, (_, i) => 'w' + i);
  for (let r = 0; r < 2000; r++) {
    for (const t of M.bertMask(sent, rng)) { n++; if (t.kind !== 'none') ch++; if (t.kind === 'mask') mk++; if (t.kind === 'random') rd++; if (t.kind === 'same') sm++; }
  }
  close('chosen share = 15%', ch / n, 0.15, 1e-9);
  close('mask / random / same shares of chosen', [mk / ch, rd / ch, sm / ch], [0.8, 0.1, 0.1], 0.01);
  const base = M.bertParams({L: 12, H: 768}), large = M.bertParams({L: 24, H: 1024});
  close('BERT-base params (HF BertModel)', [base.total, base.emb, base.layer, base.pooler], [109482240, 23837184, 7087872, 590592], 0);
  close('BERT-large params (HF BertModel)', large.total, 335141888, 0);
  close('BERT-base, vocab 30000', M.bertParams({L: 12, H: 768, V: 30000}).total, 109081344, 0);
  const tiny = M.bertParams({L: 4, H: 64, V: 27, P: 15});
  close('tiny BERT embeddings + encoder (page block 5)', tiny.emb + tiny.encoder, 202880, 0);
  const ex = M.BERT_SPAN_EXAMPLES[0];
  const sp = M.bestSpan(ex.start, ex.end, [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]).best;
  close('span example 1: i, j, score', [sp.i, sp.j, sp.score], [9, 12, 12.303], 1e-9);
  close('span example 1: s_null = S.C + E.C', ex.start[0] + ex.end[0], -5.577, 1e-9);
}
// --- stacked encoder (notebook 18): JS forward pass vs PyTorch, parameter counts ---
{
  const {readFileSync} = await import('node:fs');
  const {fileURLToPath} = await import('node:url');
  const W = JSON.parse(readFileSync(fileURLToPath(new URL('../../../static/data/deep-learning/stacked_encoder_sentiment.json', import.meta.url)), 'utf8'));
  const sents = ['this movie is terrible', 'not good movie', 'i love this movie', 'hello world movie'];
  const want = {single: [[0.959757, 0.040243], [0.999651, 0.000349], [0.000431, 0.999569], [0.001228, 0.998772]],
    stacked: [[0.08201, 0.91799], [0.999467, 0.000533], [0.000396, 0.999604], [0.000671, 0.999329]]};
  for (const name of ['single', 'stacked']) sents.forEach((s, k) => {
    const e = M.encodeTiny(s, W.vocab, W.maxLen);
    close(`${name} "${s}"`, M.tinyEncoderForward(W.models[name], e.ids, e.mask).probs, want[name][k], 1e-4);
  });
  close('encodeTiny "not good movie"', M.encodeTiny('not good movie', W.vocab).ids.concat(M.encodeTiny('not good movie', W.vocab).mask), [25, 19, 5, 0, 0, 0, 1, 1, 1, 0, 0, 0], 0);
  close('params 1 block / 3 blocks / per block', [M.tinyEncoderParams({L: 1}).total, M.tinyEncoderParams({L: 3}).total, M.tinyEncoderParams({}).block], [9922, 27010, 8544], 0);
  const f = M.tinyEncoderForward(W.models.stacked, ...Object.values((({ids, mask}) => ({ids, mask}))(M.encodeTiny('not good movie', W.vocab))));
  close('stacked layer-1 head-1 attention row "not"', f.attentions[0][0][0].slice(0, 3), [0.203, 0.001, 0.795], 1e-3);
}
// --- notebook 19: JS BERT tokenizer vs Hugging Face's (bert-base-uncased vocabulary), two-class softmax ---
{
  const {readFileSync} = await import('node:fs');
  const {fileURLToPath} = await import('node:url');
  const vocab = readFileSync(fileURLToPath(new URL('../../../static/data/deep-learning/bert-base-uncased-vocab.txt', import.meta.url)), 'utf8').split('\n').slice(0, 30522);
  close('vocab size', vocab.length, 30522, 0);
  const cases = [
    ["The movie was surprisingly good.", [101, 1996, 3185, 2001, 10889, 2204, 1012, 102]],
    ["Transformers tokenization is unbelievably subword-based!", [101, 19081, 19204, 3989, 2003, 4895, 8671, 2666, 3567, 6321, 4942, 18351, 1011, 2241, 999, 102]],
    ["Caf\u00e9 r\u00e9sum\u00e9 na\u00efve \u2014 it's 2026, isn't it?", [101, 7668, 13746, 15743, 1517, 2009, 1005, 1055, 16798, 2575, 1010, 3475, 1005, 1056, 2009, 1029, 102]],
    ["I expected more from this film.", [101, 1045, 3517, 2062, 2013, 2023, 2143, 1012, 102]],
    ["hello   WORLD...xyzzyqwerty", [101, 7592, 2088, 1012, 1012, 1012, 1060, 2100, 28753, 4160, 13777, 3723, 102]]];
  for (const [text, ids] of cases) close(`ids "${text}"`, M.bertEncode(text, vocab).ids, ids, 0);
  const pairEnc = M.bertEncode('What is BERT?', vocab, {pair: 'BERT is an encoder.', padTo: 16});
  close('pair ids (Hugging Face)', pairEnc.ids, [101, 2054, 2003, 14324, 1029, 102, 14324, 2003, 2019, 4372, 16044, 2099, 1012, 102, 0, 0], 0);
  close('pair token_type_ids', pairEnc.typeIds, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], 0);
  close('pair attention_mask', pairEnc.mask, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], 0);
  close('truncation to 512', M.bertEncode(Array(200).fill('the movie was good').join(' '), vocab).ids.length, 512, 0);
  close('P(POSITIVE) = sigmoid(margin)', M.twoClassProb(-4.2341, 4.6189), 0.99986, 1e-5);
}
console.log(fails ? `\n${fails} FAILED` : '\nall passed');
process.exit(fails ? 1 : 0);
