// LLMClear's shared models. Everything here is real, tiny and computed in your browser, with no
// libraries and no downloads:
//  - CORPUS: a short original text about the monsoon and chai (written for this box, about 2,100
//    characters), plus HINDI: an original Hindi paragraph on the same subject.
//  - A character n-gram model (Shannon 1948/1951): counts of which character follows each context of
//    0–4 characters, add-alpha smoothing, and sampling with temperature and top-k.
//  - A byte-level BPE tokenizer (Sennrich et al. 2016; the byte-level form used by GPT-2, Radford et al.
//    2019): 256 byte tokens, then the most frequent adjacent pair is merged again and again. Token ids
//    0–255 are bytes and 256+ are merges, as in GPT-2. Devanagari letters take 3 bytes each in UTF-8
//    (Unicode block U+0900–U+097F), so a tokenizer trained on English text needs about 3 tokens per
//    Hindi letter. Petrov et al., "Language Model Tokenizers Introduce Unfairness Between Languages",
//    NeurIPS 2023, measure the same effect in real tokenizers.
//  - Word vectors from co-occurrence: PPMI with context-distribution smoothing 0.75 (Levy, Goldberg &
//    Dagan 2015), compared with cosine similarity, squashed to 3D with PCA only for the picture.
//  - A word trigram model (with back-off to bigrams) for the hallucination, memory and bias demos.
//  - Scaled dot-product attention (Vaswani et al., "Attention Is All You Need", 2017) on hand-set toy
//    embeddings, with a real causal mask and softmax.
//  - Board and tile helpers for drawing text in 3D.
import { THREE, M, box, canvasTexture } from './kit.js';

// ---------------------------------------------------------------- the training text (original)
export const CORPUS = `the monsoon reaches kerala in early june. dark clouds roll in from the sea, and the first rain falls on the hot red earth. the air smells of wet soil. children run outside and float paper boats in the lanes.
amma puts a pot of water on the stove and makes chai. she adds tea leaves, milk, sugar, crushed ginger and a little cardamom. the chai boils and rises, and she pours it into small glasses. we sit by the window and watch the rain. the rain is loud on the tin roof. the chai is hot and sweet. i hold the glass with both hands and drink it slowly.
the monsoon moves north in june and july. the monsoon reaches mumbai in june, and the monsoon reaches delhi in late june. farmers wait for the rain, because rice needs a lot of water. when the rain is late, the fields stay dry and the farmers worry. when the rain is good, the fields turn green and the farmers smile.
at the station, the chai seller makes chai in a big pot. he pours hot chai into clay cups. travellers drink hot chai and eat hot pakoras while the train waits in the rain. a cup of chai costs ten rupees. a cup of coffee costs twenty rupees. some people drink coffee, some people drink lemon tea, and most people drink chai.
in winter the cold wind comes to delhi. thick fog covers delhi in december, and the trains run late. people drink hot chai and hot coffee to stay warm. in summer people drink cold lassi and cold lemon water, because the sun is hot and the wind is dry.
one evening the storm was strong. the wind was loud and the rain was heavy. the river rose and the lights went out. the doctor came to the village in the storm. the doctor said he would stay the night. the nurse came with him, and the nurse said she would help. the teacher opened the school for the families, and the teacher said she had blankets. the engineer checked the bridge, and the engineer said he would fix the lights.
in the morning the clouds were gone. the sky was blue and the fields were wet. amma made chai again, with more ginger this time. the children drank warm milk and the farmers drank strong tea. the rain will come again tomorrow, and there will be chai again.`.replace(/\s+/g, ' ').trim();

// The same kind of text in Hindi (original), used to train a second tokenizer.
export const HINDI = `मानसून जून में केरल पहुँचता है। काले बादल समुद्र से आते हैं और गरम धरती पर पहली बारिश गिरती है। बच्चे बाहर दौड़ते हैं और गलियों में कागज़ की नाव चलाते हैं। अम्मा चूल्हे पर पानी रखती है और चाय बनाती है। वह चाय में पत्ती, दूध, चीनी, अदरक और थोड़ी इलायची डालती है। हम खिड़की के पास बैठकर बारिश देखते हैं। टीन की छत पर बारिश का शोर है। चाय गरम और मीठी है। स्टेशन पर चाय वाला बड़े बर्तन में चाय बनाता है और मिट्टी के कुल्हड़ में गरम चाय डालता है। सर्दी में दिल्ली में घना कोहरा होता है और लोग गरम चाय पीते हैं।`;

// ---------------------------------------------------------------- random numbers you can replay
export function rng(seed) {
  let a = Math.imul((seed >>> 0) ^ 0x9e3779b9, 0x85ebca6b) ^ 0x27d4eb2f;
  const next = () => { a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  next(); next(); next();                                    // let nearby seeds drift apart
  return next;
}

// ---------------------------------------------------------------- character n-gram model
export const VOCAB = [...new Set(CORPUS)].sort();          // ' ', ',', '.', a–z
export const V = VOCAB.length;
const IDX = new Map(VOCAB.map((c, i) => [c, i]));
export const MAXCTX = 4;
export const ALPHA = 0.01;                                   // add-alpha smoothing: unseen letters keep a sliver of chance
const TABLES = [];                                           // TABLES[n]: Map(context of n chars → Float64Array counts)
for (let n = 0; n <= MAXCTX; n++) {
  const T = new Map();
  for (let i = n; i < CORPUS.length; i++) {
    const ctx = CORPUS.slice(i - n, i);
    let row = T.get(ctx); if (!row) { row = new Float64Array(V); T.set(ctx, row); }
    row[IDX.get(CORPUS[i])]++;
  }
  TABLES.push(T);
}
export const showChar = (c) => (c === ' ' ? '␣' : c);
// Probabilities of the next character after `text`, using up to n characters of context. If that exact
// context never appeared in the training text, back off to a shorter one (and say so).
export function nextChar(text, n) {
  const clean = [...text.toLowerCase()].filter((c) => IDX.has(c)).join('');
  for (let k = Math.min(n, clean.length); k >= 0; k--) {
    const ctx = clean.slice(clean.length - k);
    const row = TABLES[k].get(ctx);
    if (!row) continue;
    let tot = 0; for (let i = 0; i < V; i++) tot += row[i];
    const p = new Float64Array(V);
    for (let i = 0; i < V; i++) p[i] = (row[i] + ALPHA) / (tot + ALPHA * V);
    return { p, ctx, k, asked: Math.min(n, clean.length), seen: tot, counts: row };
  }
}
// How many table cells a model with n characters of context has, and how many are non-zero.
export function tableSize(n) { let used = 0; TABLES[n].forEach((r) => r.forEach((c) => { if (c) used++; })); return { cells: Math.pow(V, n + 1), used, contexts: TABLES[n].size }; }

// Temperature and top-k, applied to any probability list.
export function shape(p, T, k) {
  const n = p.length, q = new Float64Array(n);
  const order = [...p.keys()].sort((a, b) => p[b] - p[a]);
  const keep = new Set(order.slice(0, Math.max(1, Math.min(k || n, n))));
  if (T < 0.08) { q[order[0]] = 1; return q; }            // temperature near 0: always the top choice
  let s = 0;
  for (let i = 0; i < n; i++) if (keep.has(i)) { q[i] = Math.pow(p[i], 1 / T); s += q[i]; }
  for (let i = 0; i < n; i++) q[i] /= s;
  return q;
}
export function pick(q, r) { let u = r(), last = 0; for (let i = 0; i < q.length; i++) { if (q[i] <= 0) continue; last = i; u -= q[i]; if (u <= 0) return i; } return last; }
export function entropyBits(q) { let h = 0; for (const x of q) if (x > 0) h -= x * Math.log2(x); return h; }

// ---------------------------------------------------------------- byte-level BPE tokenizer
const te = new TextEncoder();
const td = new TextDecoder('utf-8', { fatal: true });
// Split into word-ish pieces first, the GPT-2 way: a word keeps the space in front of it.
export const pretok = (text) => text.match(/ ?[^\s.,।!?]+|[.,।!?]| +/gu) || [];
function trainBPE(text, max) {
  const types = new Map();
  for (const w of pretok(text)) types.set(w, (types.get(w) || 0) + 1);
  const words = [...types].map(([w, c]) => ({ ids: [...te.encode(w)], c }));
  const bytes = []; for (let i = 0; i < 256; i++) bytes.push([i]);
  const merges = [];
  for (let m = 0; m < max; m++) {
    const pairs = new Map();
    for (const w of words) for (let i = 0; i < w.ids.length - 1; i++) { const k = w.ids[i] * 100000 + w.ids[i + 1]; pairs.set(k, (pairs.get(k) || 0) + w.c); }
    let best = -1, bc = 1;
    for (const [k, c] of pairs) if (c > bc || (c === bc && best >= 0 && k < best)) { best = k; bc = c; }
    if (best < 0 || bc < 2) break;
    const a = Math.floor(best / 100000), b = best % 100000, id = 256 + merges.length;
    merges.push({ a, b, id, count: bc });
    bytes.push([...bytes[a], ...bytes[b]]);
    for (const w of words) {
      const out = [];
      for (let i = 0; i < w.ids.length; i++) { if (i < w.ids.length - 1 && w.ids[i] === a && w.ids[i + 1] === b) { out.push(id); i++; } else out.push(w.ids[i]); }
      w.ids = out;
    }
  }
  const rank = new Map(merges.map((m, i) => [m.a * 100000 + m.b, i]));
  return { merges, bytes, rank };
}
export const MAXMERGES = 400;
export const TOKENIZERS = {
  en: { name: 'English text only', bpe: trainBPE(CORPUS, MAXMERGES) },
  mix: { name: 'English + Hindi text', bpe: trainBPE(CORPUS + ' ' + HINDI, MAXMERGES) },
};
// Encode with only the first n merges learned so far.
export function encode(text, bpe, n) {
  const out = [];
  for (const w of pretok(text)) {
    let ids = [...te.encode(w)];
    for (;;) {
      let bi = -1, br = Infinity;
      for (let i = 0; i < ids.length - 1; i++) { const r = bpe.rank.get(ids[i] * 100000 + ids[i + 1]); if (r !== undefined && r < n && r < br) { br = r; bi = i; } }
      if (bi < 0) break;
      ids.splice(bi, 2, 256 + br);
    }
    out.push(...ids);
  }
  return out;
}
// What a token looks like: its text, or its bytes in hex if it is only part of a letter.
export function tokenText(bpe, id) {
  const b = bpe.bytes[id];
  try { return td.decode(new Uint8Array(b)).replace(/ /g, '·'); } catch { return b.map((x) => x.toString(16).toUpperCase().padStart(2, '0')).join(' '); }
}
export const isPartial = (bpe, id) => { try { td.decode(new Uint8Array(bpe.bytes[id])); return false; } catch { return true; } };
export const utf8Len = (s) => te.encode(s).length;

// ---------------------------------------------------------------- words, sentences
export const SENTENCES = CORPUS.split('.').map((s) => s.trim()).filter(Boolean).map((s) => s.replace(/,/g, '').split(' '));
export const WORDS = SENTENCES.flat();

// ---------------------------------------------------------------- word vectors from co-occurrence
export const EMB_WORDS = ['chai', 'coffee', 'tea', 'lassi', 'milk', 'water', 'rain', 'wind', 'fog', 'storm', 'clouds', 'sun',
  'delhi', 'mumbai', 'kerala', 'june', 'july', 'december', 'hot', 'cold', 'amma', 'farmers', 'children', 'people',
  'doctor', 'nurse', 'teacher', 'engineer', 'he', 'she'];
export const EMB_GROUP = { chai: 'drink', coffee: 'drink', tea: 'drink', lassi: 'drink', milk: 'drink', water: 'drink', rain: 'weather', wind: 'weather', fog: 'weather', storm: 'weather', clouds: 'weather', sun: 'weather', delhi: 'place', mumbai: 'place', kerala: 'place', june: 'time', july: 'time', december: 'time', hot: 'describe', cold: 'describe', amma: 'person', farmers: 'person', children: 'person', people: 'person', doctor: 'person', nurse: 'person', teacher: 'person', engineer: 'person', he: 'pronoun', she: 'pronoun' };
export const GROUP_COL = { drink: 0xffb547, weather: 0x8ef0ff, place: 0xc49bff, time: 0x7be08c, describe: 0xff7a59, person: 0xf5a3d0, pronoun: 0xe8eef8 };
export const EMB_WINDOW = 3;
export const EMB = (() => {
  const freq = new Map(); WORDS.forEach((w) => freq.set(w, (freq.get(w) || 0) + 1));
  const ctxWords = [...freq.keys()].filter((w) => freq.get(w) >= 2);
  const ci = new Map(ctxWords.map((w, i) => [w, i]));
  const C = EMB_WORDS.map(() => new Float64Array(ctxWords.length));
  const wi = new Map(EMB_WORDS.map((w, i) => [w, i]));
  let pairs = 0;
  for (const s of SENTENCES) for (let i = 0; i < s.length; i++) {
    const r = wi.get(s[i]); if (r === undefined) continue;
    for (let j = Math.max(0, i - EMB_WINDOW); j <= Math.min(s.length - 1, i + EMB_WINDOW); j++) { if (j === i) continue; const c = ci.get(s[j]); if (c !== undefined) { C[r][c]++; pairs++; } }
  }
  // PPMI with context smoothing
  const rowTot = C.map((r) => r.reduce((a, b) => a + b, 0)), all = rowTot.reduce((a, b) => a + b, 0);
  const colTot = new Float64Array(ctxWords.length); C.forEach((r) => r.forEach((v, j) => (colTot[j] += v)));
  const colS = colTot.map((v) => Math.pow(v, 0.75)), colSum = colS.reduce((a, b) => a + b, 0);
  const P = C.map((r, i) => r.map((v, j) => (v && rowTot[i] ? Math.max(0, Math.log((v / all) / ((rowTot[i] / all) * (colS[j] / colSum)))) : 0)));
  const unit = P.map((r) => { const n = Math.hypot(...r) || 1; return r.map((v) => v / n); });
  const cos = (a, b) => { let s = 0; for (let k = 0; k < a.length; k++) s += a[k] * b[k]; return s; };
  const sim = unit.map((a) => unit.map((b) => cos(a, b)));
  // PCA to 3D, only for the picture: eigenvectors of the centred Gram matrix by power iteration.
  const n = unit.length, d = unit[0].length, mean = new Float64Array(d);
  unit.forEach((r) => r.forEach((v, j) => (mean[j] += v / n)));
  const X = unit.map((r) => r.map((v, j) => v - mean[j]));
  let G = X.map((a) => X.map((b) => cos(a, b)));
  const coords = EMB_WORDS.map(() => [0, 0, 0]), lams = [];
  for (let k = 0; k < 3; k++) {
    let v = new Float64Array(n).map((_, i) => Math.sin(i * 1.7 + k * 3.1) + 1.1);
    let lam = 0;
    for (let it = 0; it < 300; it++) {
      const w = G.map((row) => cos(row, v)); lam = Math.hypot(...w); v = w.map((x) => x / (lam || 1));
    }
    lams.push(lam);
    for (let i = 0; i < n; i++) coords[i][k] = v[i] * Math.sqrt(lam);
    G = G.map((row, i) => row.map((g, j) => g - lam * v[i] * v[j]));
  }
  const tr = X.reduce((a, r) => a + cos(r, r), 0);
  return { words: EMB_WORDS, dims: ctxWords.length, pairs, sim, coords, kept: (lams[0] + lams[1] + lams[2]) / tr };
})();
export function neighbours(w, k = 3) {
  const i = EMB.words.indexOf(w); if (i < 0) return [];
  return EMB.words.map((x, j) => ({ w: x, s: EMB.sim[i][j] })).filter((o) => o.w !== w).sort((a, b) => b.s - a.s).slice(0, k);
}
// A clearly made-up set for vector arithmetic: three hand-picked meanings, one per axis.
export const TOY = {
  axes: ['royal', 'female', 'young'],
  words: { king: [1, 0, 0], queen: [1, 1, 0], man: [0, 0, 0], woman: [0, 1, 0], prince: [1, 0, 1], princess: [1, 1, 1], boy: [0, 0, 1], girl: [0, 1, 1] },
};

// ---------------------------------------------------------------- word trigram model (chapter 6)
const W2 = new Map(), W1 = new Map();
for (const s of SENTENCES) {
  const t = ['<s>', ...s, '.'];
  for (let i = 1; i < t.length; i++) {
    const k1 = t[i - 1]; if (!W1.has(k1)) W1.set(k1, new Map()); W1.get(k1).set(t[i], (W1.get(k1).get(t[i]) || 0) + 1);
    if (i >= 2) { const k2 = t[i - 2] + ' ' + t[i - 1]; if (!W2.has(k2)) W2.set(k2, new Map()); W2.get(k2).set(t[i], (W2.get(k2).get(t[i]) || 0) + 1); }
  }
}
export function nextWord(words) {
  const a = words.at(-2) ?? '<s>', b = words.at(-1) ?? '<s>';
  let m = W2.get(a + ' ' + b), used = 2;
  if (!m) { m = W1.get(b); used = 1; }
  if (!m) return { list: [{ w: '.', p: 1 }], used: 0 };
  const tot = [...m.values()].reduce((x, y) => x + y, 0);
  return { list: [...m].map(([w, c]) => ({ w, p: c / tot, c })).sort((x, y) => y.p - x.p), used };
}
export function generateWords(prompt, seed, max = 14) {
  const r = rng(seed), words = prompt.split(' '), steps = [];
  for (let i = 0; i < max; i++) {
    const { list } = nextWord(words);
    let u = r(), k = 0; for (; k < list.length - 1; k++) { u -= list[k].p; if (u <= 0) break; }
    steps.push({ w: list[k].w, p: list[k].p, n: list.length });
    if (list[k].w === '.') break;
    words.push(list[k].w);
  }
  return { words, steps, text: words.join(' ') };
}
export const inText = (sentence) => (' ' + CORPUS.replace(/[.,]/g, ' ').replace(/\s+/g, ' ') + ' ').includes(' ' + sentence.trim() + ' ');
// IMD normal monsoon onset dates (revised 2020): Kerala 1 June, Mumbai 11 June, Delhi 27 June.
export const ONSET = { kerala: ['june', 'early'], mumbai: ['june'], delhi: ['june', 'late'] };

// ---------------------------------------------------------------- attention on toy embeddings (chapter 4)
// Features: person, thing (drink), action (verb), little word, pronoun, describing word. Plus position
// as sin/cos pairs at two frequencies. Real models learn hundreds of such directions; these are hand-set.
export const SENT = ['amma', 'poured', 'the', 'chai', 'because', 'it', 'was', 'hot'];
const FEAT = {
  amma: [1, 0, 0, 0, 0, 0], poured: [0, 0, 1, 0, 0, 0], the: [0, 0, 0, 1, 0, 0], chai: [0, 1, 0, 0, 0, 0],
  because: [0, 0, 0, 1, 0, 0], it: [0, 0, 0, 0, 1, 0], was: [0, 0, 0, 1, 0, 0], hot: [0, 0, 0, 0, 0, 1],
};
export const FEAT_NAMES = ['person', 'thing', 'action', 'little word', 'pronoun', 'describing'];
const OM = [0.9, 1.9, 0.35];                                // position frequencies (radians per word)
export const embedTok = (w, i) => [...FEAT[w], ...OM.flatMap((o) => [Math.cos(o * i), Math.sin(o * i)])];
const DM = 12;
// Each head is a pair of 6×12 matrices (W_Q, W_K), written as a function for readability.
export const HEADS = {
  refer: { name: 'Who is "it"?', what: 'Pronouns look for the thing they stand for.',
    q: (e) => [5.5 * e[4] + 4.5 * e[5], 0, 0, 0, 0, 0], k: (e) => [e[1] + 0.2 * e[0], 0, 0, 0, 0, 0] },
  prev: { name: 'Previous word', what: 'Each word looks one step back, using position alone.',
    // Rotating a position vector by one step is a linear map; this is how a head can learn "one back".
    q: (e) => { const out = []; OM.forEach((o, f) => { const c = e[6 + 2 * f], s = e[7 + 2 * f]; out.push(4 * (c * Math.cos(o) + s * Math.sin(o)), 4 * (s * Math.cos(o) - c * Math.sin(o))); }); return out; },
    k: (e) => e.slice(6, 12) },
  doer: { name: 'Who did it?', what: 'Action words look for the person doing them.',
    q: (e) => [5 * e[2], 0, 0, 0, 0, 0], k: (e) => [e[0] + 0.15 * e[1], 0, 0, 0, 0, 0] },
};
export const D_HEAD = 6;
export function attention(head, causal = true) {
  const H = HEADS[head], E = SENT.map((w, i) => embedTok(w, i));
  const Q = E.map(H.q), K = E.map(H.k);
  const scores = Q.map((q, i) => K.map((k, j) => (causal && j > i ? -Infinity : q.reduce((a, x, t) => a + x * k[t], 0) / Math.sqrt(D_HEAD))));
  const W = scores.map((row) => { const m = Math.max(...row); const ex = row.map((x) => (x === -Infinity ? 0 : Math.exp(x - m))); const s = ex.reduce((a, b) => a + b, 0); return ex.map((x) => x / s); });
  // Output = weighted mix of value vectors (here V = the word's features).
  const out = W.map((row) => FEAT_NAMES.map((_, f) => row.reduce((a, w, j) => a + w * E[j][f], 0)));
  return { E, Q, K, scores, W, out };
}
export { DM };

// ---------------------------------------------------------------- drawing helpers
export const COL = { white: '#e8eef8', soft: 'rgba(255,255,255,.55)', hot: '#ffd166', teal: '#b8f2e6', cyan: '#8ef0ff', orange: '#ffb547', red: '#ff7a59', pink: '#ff5a8a', green: '#7be08c', violet: '#c49bff' };
export function panel(g, w, h) { g.clearRect(0, 0, w, h); g.fillStyle = 'rgba(9,11,16,.92)'; g.fillRect(0, 0, w, h); g.strokeStyle = 'rgba(184,242,230,.25)'; g.lineWidth = 3; g.strokeRect(1.5, 1.5, w - 3, h - 3); }
export function txt(g, s, x, y, font = '24px sans-serif', col = COL.white, align = 'left') { g.font = font; g.fillStyle = col; g.textAlign = align; g.fillText(s, x, y); g.textAlign = 'left'; }
export const F = (px, bold = false, mono = false) => `${bold ? 'bold ' : ''}${px}px ${mono ? '"Geist Mono", ui-monospace, Menlo, monospace' : 'Geist, "Noto Sans Devanagari", "Kohinoor Devanagari", sans-serif'}`;
export function wrap(g, s, x, y, maxW, lh, font, col) {
  g.font = font; g.fillStyle = col; const words = s.split(' '); let line = '', yy = y;
  for (const w of words) { const t = line ? line + ' ' + w : w; if (g.measureText(t).width > maxW && line) { g.fillText(line, x, yy); line = w; yy += lh; } else line = t; }
  if (line) g.fillText(line, x, yy);
  return yy + lh;
}
export function boardMesh(tex, w, h) { return new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ map: tex, transparent: true, toneMapped: false, side: THREE.DoubleSide })); }
export function board(pxW, pxH, w, h, draw) { const b = canvasTexture(pxW, pxH, draw); const m = boardMesh(b.tex, w, h); return { ...b, mesh: m }; }
export const inReel = () => document.body.classList.contains('gb-reel');
export function fitNarrow(stage, minor = [], y0 = -0.12) {
  const narrow = stage.host.clientWidth < 560;
  minor.forEach((l) => { if (l) l.visible = !narrow; });
  const y = narrow && !inReel() ? y0 : 0;
  if (!stage.shift || stage.shift[1] !== y) stage.setShift(0, y);
  return narrow;
}
// A small glowing tile with text on its face (a token, a letter, a word). redraw(text, colour) to change it.
export function tile(text, w = 0.5, h = 0.5, color = COL.teal, px = 128) {
  const pw = Math.round(px * Math.max(1, w / h)), ph = px;
  const c = canvasTexture(pw, ph, (g, W, H, t = text, col = color, sub = '') => {
    g.clearRect(0, 0, W, H);
    g.fillStyle = 'rgba(12,16,24,.94)'; roundRect(g, 3, 3, W - 6, H - 6, 16); g.fill();
    g.strokeStyle = col; g.lineWidth = 5; roundRect(g, 3, 3, W - 6, H - 6, 16); g.stroke();
    let size = sub ? H * 0.46 : H * 0.58; g.font = F(size, true); while (g.measureText(t).width > W - 18 && size > 12) { size -= 2; g.font = F(size, true); }
    g.fillStyle = COL.white; g.textAlign = 'center'; g.textBaseline = 'middle'; g.fillText(t, W / 2, sub ? H * 0.4 : H / 2 + 2);
    if (sub) { g.font = F(H * 0.2, false, true); g.fillStyle = col; g.fillText(sub, W / 2, H * 0.8); }
    g.textAlign = 'left'; g.textBaseline = 'alphabetic';
  });
  const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ map: c.tex, transparent: true, toneMapped: false, side: THREE.DoubleSide }));
  m.redraw = c.redraw;
  return m;
}
export function roundRect(g, x, y, w, h, r) { g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r); g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath(); }
// A soft base plate for a scene.
export function plate(w, d, color = 0x1a2030) { const p = box(w, 0.08, d, M.matte(color)); p.position.y = 0.04; return p; }
export const hex = (n) => '#' + n.toString(16).padStart(6, '0');
export const fmtBig = (x) => {
  if (x >= 1e12) return (x / 1e12).toFixed(x < 1e13 ? 1 : 0) + ' trillion';
  if (x >= 1e9) return (x / 1e9).toFixed(x < 1e10 ? 1 : 0) + ' billion';
  if (x >= 1e6) return (x / 1e6).toFixed(x < 1e7 ? 1 : 0) + ' million';
  return Math.round(x).toLocaleString('en');
};
export const sci = (x) => { const e = Math.floor(Math.log10(x)); return `${(x / 10 ** e).toFixed(1)} × 10^${e}`; };
// Frame the model: on wide screens nudge the picture right (x, a fraction of the width) so the readout
// in the top-left corner doesn't cover it; on phones hide minor labels and nudge it down instead.
export function fit(stage, minor = [], { x = 0, y = 0, ny = -0.12 } = {}) {
  const narrow = stage.host.clientWidth < 560, reel = inReel();
  minor.forEach((l) => { if (l) l.visible = !narrow; });
  const sx = narrow || reel ? 0 : x, sy = reel ? 0 : narrow ? ny : y;
  if (!stage.shift || stage.shift[0] !== sx || stage.shift[1] !== sy) stage.setShift(sx, sy);
  return narrow;
}
// Tall video frames (the 9:16 reel): chapters re-stack their parts upwards, and this
// re-aims the camera at the taller stack. dy raises the camera target; zoom scales its distance.
export const isTall = (stage) => inReel() && stage.host.clientWidth / Math.max(1, stage.host.clientHeight) < 0.8;
export function tallView(stage, dy, zoom = 1, dx = 0) {
  const tall = isTall(stage);
  if (tall && stage.home && stage.home !== stage._tallHome) {
    const { pos, target } = stage.home;
    const t = [target[0] + dx, target[1] + dy, target[2]];
    stage.setView(t.map((v, i) => v + (pos[i] - target[i]) * zoom), t, 1.0);
    stage._tallHome = stage.home;
  }
  return tall;
}
