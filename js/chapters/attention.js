// Chapter 4: attention and the transformer. Real scaled dot-product attention (Vaswani et al., "Attention
// Is All You Need", NeurIPS 2017): score = q·k / √d, a causal mask (a word may only look back, as in
// GPT-style decoders), softmax, then a weighted mix of value vectors. The embeddings and the query/key
// maps are hand-set toys (llm.js HEADS), chosen so each head does one readable job; in a real model all
// of these numbers are learned. GPT-3 175B: 96 layers, 96 heads per layer, 12,288-number vectors, 2,048
// token context (Brown et al. 2020, table 2.1).
import { THREE, M, box, approach, clamp } from '../kit.js';
import { SENT, HEADS, FEAT_NAMES, D_HEAD, attention, board, panel, txt, F, COL, tile, fit, inReel, roundRect, tallView } from '../llm.js';

const N = SENT.length, SPX = 0.84;
const STACK = [
  { name: 'Next-token probabilities', col: 0xffd166 },
  { name: 'Unembed: scores for every token', col: 0xffb547 },
  { name: 'Block N … (GPT-3 has 96)', col: 0x3a4658, ghost: true },
  { name: 'Feed-forward network (MLP)', col: 0x7be08c },
  { name: 'Attention: words look at words', col: 0xb8f2e6, hot: true },
  { name: 'Block 1: attention + MLP', col: 0x3a4658, ghost: true },
  { name: 'Embeddings + positions', col: 0xc49bff },
  { name: 'Token IDs', col: 0x8fa3b8 },
];

export default {
  id: 'attention',
  short: 'Attention',
  title: 'Attention: every word looks at the others',
  subtitle: 'The transformer lets each token pull in meaning from the tokens before it.',
  view: { pos: [-0.7, 4.8, 8.4], target: [-0.5, 1.45, 0.7] },
  learn: `<p>Read: "amma poured the chai because <b>it</b> was hot." What was hot? You knew at once: the chai. To get that, "it" has to <b>look back</b> at the other words and pick the right one. That is <b>attention</b>, the key idea of the <b>transformer</b>, from the 2017 paper "Attention Is All You Need" by a team at Google.</p>
    <p>Each token makes three vectors from its embedding. A <b>query</b>: "what am I looking for?" A <b>key</b>: "what do I contain?" A <b>value</b>: "what I will pass on." Every query is compared with every key (a dot product), the scores are turned into shares that add to 100% (a <b>softmax</b>), and the token takes that mix of values. Now "it" carries a bit of "chai" inside it.</p>
    <p>This is a real attention computation on 8 words, with toy embeddings and hand-set weights so each <b>head</b> does one clear job. In a real model all those weights are learned, and each layer runs many heads side by side: GPT-3 has <b>96 heads in each of 96 layers</b>. GPT-style models also use a <b>causal mask</b>: a word may look only at words before it, never ahead, because it is trying to predict what comes next.</p>
    <p>Attention is one half of a <b>transformer block</b>. The other half is a small <b>neural network</b> (see NeuralNetClear for how those learn) that works on each token alone. Stack dozens of blocks, and at the top turn the last token's vector into a score for every token in the vocabulary. That is the next-token probability from chapter 1.</p>
    <p class="tip"><b>Try it:</b> click "it" (or pick it below) with the "Who is it?" head. Then click "hot". Switch to the "Previous word" head and turn the causal mask off to see words look ahead.</p>`,
  terms: [
    { t: 'Attention', d: 'Each token scores every other token and takes a weighted mix of their information.' },
    { t: 'Query, key, value', d: 'Three vectors per token: what it seeks, what it offers to be found by, and what it passes on.' },
    { t: 'Softmax', d: 'Turns any list of scores into positive shares that add up to 1.' },
    { t: 'Head', d: 'One set of query/key/value weights. Many heads run side by side, each on its own pattern.' },
    { t: 'Causal mask', d: 'Blocks each token from looking at tokens after it.' },
    { t: 'Transformer block', d: 'Attention followed by a small feed-forward network, stacked many times.' },
  ],
  defaults: { head: 'refer', focus: 5, causal: true },
  controls: [
    { key: 'head', type: 'seg', label: 'Attention head', options: Object.entries(HEADS).map(([v, h]) => ({ v, label: h.name })), fmt: (v) => HEADS[v].what },
    { key: 'focus', type: 'seg', label: 'Word that is looking (or click a word)', options: SENT.map((w, i) => ({ v: i, label: w })) },
    { key: 'causal', type: 'toggle', label: 'Causal mask (only look back)' },
  ],
  quiz: [
    { q: 'In attention, what does a query do?', options: ['Stores the word', 'Says what this token is looking for, to be matched against keys', 'Counts letters', 'Picks the next word directly'], answer: 1, why: 'Each query is compared with every key; better matches get more of the attention.' },
    { q: 'What does the causal mask stop?', options: ['Looking at earlier words', 'Looking at later words', 'Using numbers', 'Training'], answer: 1, why: 'A model that predicts the next token must not peek at tokens that come after.' },
    { q: 'Where does the transformer come from?', options: ['A 1966 chatbot', 'The 2017 paper "Attention Is All You Need"', 'Shannon in 1948', 'The first calculator'], answer: 1, why: 'A Google team introduced it in 2017. Nearly every modern LLM is built on it.' },
  ],
  reel: [
    { ms: 5600, caption: 'Attention: to understand "it", the model looks back and finds "chai".', set: { head: 'refer', focus: 5, causal: true }, spin: 0, view: { pos: [-0.6, 5.0, 9.6], target: [-0.6, 1.9, 0.7] } },
    { ms: 5000, caption: 'Stack attention and small neural networks dozens of times, and you have a transformer.', set: { head: 'prev', focus: 7, causal: true }, spin: 0, view: { pos: [1.6, 4.6, 9.4], target: [0.2, 2.1, 0.3] } },
  ],

  build({ stage }) {
    const root = new THREE.Group(); stage.root.add(root);
    const x0 = -1.9 - ((N - 1) * SPX) / 2;
    // the sentence: word tiles standing on a strip
    const strip = box(N * SPX + 0.3, 0.1, 0.9, M.matte(0x1a2030)); strip.position.set(x0 + ((N - 1) * SPX) / 2 + 0, 0.05, 0.3); root.add(strip);
    const words = SENT.map((w, i) => {
      const t = tile(w, 0.78, 0.44, COL.teal, 96); t.position.set(x0 + i * SPX, 0.45, 0.3); t.userData.idx = i; root.add(t);
      const bar = box(0.46, 1, 0.12, M.plastic(0xffd166, { emissive: 0x4a3200 })); bar.position.set(x0 + i * SPX, 0.12, 0.8); root.add(bar);
      return { t, bar };
    });
    stage.pickables = words.map((w) => w.t);
    // arcs from the focus word to every other word, thickness = attention weight
    const arcs = SENT.map(() => {
      const geo = new THREE.TubeGeometry(new THREE.CatmullRomCurve3([new THREE.Vector3(), new THREE.Vector3(0, 1, 0), new THREE.Vector3(1, 0, 0)]), 40, 0.03, 8);
      const m = new THREE.Mesh(geo, M.glow(0xffd166, { transparent: true, opacity: 0.9 })); root.add(m); return m;
    });
    // the transformer stack
    const tower = new THREE.Group(); tower.position.set(2.2, 0, -0.2); root.add(tower);
    const slabs = STACK.map((b, i) => {
      const y = 0.3 + (STACK.length - 1 - i) * 0.52;
      const m = box(2.3, 0.34, 1.0, b.ghost ? M.ghost(0x8ea0b8, 0.18) : M.plastic(b.col, { transparent: true, opacity: 0.85, emissive: b.col, emissiveIntensity: 0.15 }));
      m.position.y = y; tower.add(m);
      const t = tile(b.name, 2.2, 0.26, b.ghost ? COL.soft : '#' + b.col.toString(16).padStart(6, '0'), 64); t.position.set(0, y, 0.51); tower.add(t);
      return m;
    });
    const hot = slabs[STACK.findIndex((b) => b.hot)];
    const bd = board(1100, 520, 3.8, 1.8, (g, w, h, s) => s && drawBoard(g, w, h, s)); bd.mesh.position.set(-1.9, 0.42, 2.45); bd.mesh.rotation.x = -1.2; root.add(bd.mesh);
    const L = { stack: stage.label('Inside one transformer', [2.2, 4.6, -0.2], root) };
    let st = null, res = null, key = '', grow = 0, arcsDone = false;

    function drawBoard(g, w, h, s) {
      panel(g, w, h);
      const f = s.focus, W = res.W[f];
      txt(g, `"${SENT[f]}" looks at…   head: ${HEADS[s.head].name}`, 26, 52, F(32, true), COL.white);
      const order = W.map((v, j) => [v, j]).filter(([v]) => v > 0).sort((a, b) => b[0] - a[0]).slice(0, 5);
      order.forEach(([v, j], k) => {
        const y = 108 + k * 62;
        txt(g, SENT[j], 26, y, F(32, true), j === f ? COL.soft : COL.teal);
        g.fillStyle = 'rgba(255,209,102,.85)'; g.fillRect(220, y - 28, Math.max(3, v * 620), 32);
        txt(g, `${Math.round(v * 100)}%`, 232 + v * 620, y, F(28, true), COL.hot);
        const sc = res.scores[f][j];
        txt(g, `score ${sc.toFixed(2)}`, w - 26, y, F(24, false, true), COL.soft, 'right');
      });
      txt(g, `score = query · key / √${D_HEAD}, then softmax. Hand-set toy weights.`, 26, h - 26, F(24), COL.soft);
    }

    return {
      pick(o) { if (o.userData.idx !== undefined && st) { st.focus = o.userData.idx; key = ''; } },
      update(dt, s, time) {
        dt = Math.max(0, dt); st = s;
        fit(stage, [L.stack], { x: 0.03, y: -0.08, ny: -0.16 });
        const cx = x0 + ((N - 1) * SPX) / 2, tall = tallView(stage, 2.2, 1.0, cx - (stage.home?.target[0] ?? 0));
        if (tall) { tower.position.set(cx, 3.5, -0.7); L.stack.position.set(tower.position.x, 7.9, -0.7); }
        else { tower.position.set(2.2, 0, -0.2); L.stack.position.set(2.2, 4.6, -0.2); }
        const k = `${s.head}|${s.focus}|${s.causal}`;
        if (k !== key) { key = k; res = attention(s.head, s.causal); bd.redraw(s); grow = 0; arcsDone = false; }
        grow = Math.min(1, grow + dt * 1.8);
        const f = s.focus, W = res.W[f];
        words.forEach((w, i) => {
          const on = i === f;
          w.t.scale.setScalar(on ? 1.15 : 1); w.t.position.y = on ? 0.55 : 0.45;
          const h = Math.max(0.01, W[i] * 1.6 * grow); w.bar.scale.y = approach(w.bar.scale.y, h, 10, dt); w.bar.position.y = 0.1 + w.bar.scale.y / 2;
          w.bar.visible = W[i] > 0.004;
        });
        if (!arcsDone) arcs.forEach((a, j) => {
          const v = W[j];
          a.visible = v > 0.01 && j !== f;
          if (!a.visible) return;
          const A = new THREE.Vector3(x0 + f * SPX, 0.75, 0.3), B = new THREE.Vector3(x0 + j * SPX, 0.75, 0.3);
          const mid = A.clone().add(B).multiplyScalar(0.5); mid.y += 0.5 + 0.35 * Math.abs(f - j);
          a.geometry.dispose();
          const pts = new THREE.QuadraticBezierCurve3(A, mid, B);
          a.geometry = new THREE.TubeGeometry(pts, 40, 0.012 + 0.07 * v * grow, 8);
          a.material.opacity = clamp(0.25 + v * 1.2, 0.25, 1);
        });
        if (grow >= 1) arcsDone = true;
        hot.material.emissiveIntensity = 0.25 + 0.2 * Math.sin(time * 4);
      },
      readout(s) {
        if (!res) return '';
        const f = s.focus, W = res.W[f], j = W.reduce((b, v, i) => (i !== f && v > W[b] ? i : b), f === 0 ? 1 : 0);
        const o = res.out[f], top = o.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 2);
        return `<div class="big">"${SENT[f]}" → "${SENT[j]}" ${Math.round(W[j] * 100)}%</div>
          <div class="row"><span>Head</span><b>${HEADS[s.head].name}</b></div>
          <div class="row"><span>New meaning of "${SENT[f]}"</span><b>${top.map(([v, i]) => `${FEAT_NAMES[i]} ${v.toFixed(2)}`).join(', ')}</b></div>
          <small>${f === 0 && s.causal ? 'The first word can only look at itself.' : HEADS[s.head].what}</small>`;
      },
    };
  },
};
