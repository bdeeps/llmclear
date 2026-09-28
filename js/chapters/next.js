// Chapter 1: predict the next letter. A real character n-gram model, trained when the page loads on
// the box's own short text (llm.js CORPUS). For the last n letters it looks up how often each letter
// came next, turns the counts into probabilities (add-alpha smoothing), then reshapes them with
// temperature and top-k and draws one at random. This is Shannon's 1948 idea ("A Mathematical Theory
// of Communication", section 3: letter and word approximations of English), and the same loop an LLM
// runs, only an LLM uses a neural network over tokens instead of a count table over letters.
import { THREE, M, box, approach } from '../kit.js';
import {
  CORPUS, VOCAB, V, nextChar, shape, pick, rng, tableSize, showChar, entropyBits,
  board, panel, txt, F, COL, tile, plate, fit, inReel, tallView,
} from '../llm.js';

const PROMPTS = { monsoon: 'the monsoon ', chai: 'amma makes ', rain: 'the rain ', blank: '' };
const TABLES_INFO = (n) => { const t = tableSize(n); return `${t.contexts.toLocaleString('en')} contexts seen in ${CORPUS.length.toLocaleString('en')} letters`; };
const SP = 0.25, BW = 0.19, HMAX = 3.2;

export default {
  id: 'next',
  short: 'Next letter',
  title: 'Predict the next letter',
  subtitle: 'A language model is a machine for guessing what comes next.',
  view: { pos: [0, 4.6, 7.9], target: [0, 2.3, 0.3] },
  onChange(s, key) { if (key === 'prompt' || key === 'ctx') s.gen = (s.gen || 0) + 1; },
  learn: `<p>Type "the monsoon reaches ker" and you can guess the next letter. So can a machine. A <b>language model</b> is a machine that, given some text, gives a <b>probability</b> for every possible next piece. Pick one, add it to the text, and ask again. Do that a few thousand times and you have written a page.</p>
    <p>This one is real and tiny. When the page loaded, it read a short text about the monsoon and chai (about <b>2,100 letters</b>, written for this box) and simply <b>counted</b>: after "th", how often came "e"? how often "a"? Those counts become the bars. The engineer Claude Shannon played this game in <b>1948</b>, building fake English from letter counts.</p>
    <p>The <b>context</b> is how many letters back it looks. With 1 letter it writes gibberish. With 3 or 4 it writes real-looking words, but it starts to <b>copy</b> its training text word for word, because it has seen so little. Big LLMs look back thousands of tokens and learn patterns instead of storing counts.</p>
    <p><b>Temperature</b> changes how it picks. Near 0 it always takes the tallest bar: safe, but it loops. At 1 it follows the true odds. Above 1 the odds flatten and it gets wild. <b>Top-k</b> throws away all but the k best choices first. Chatbots use the same dials.</p>
    <p class="tip"><b>Try it:</b> press "Next letter" a few times and watch the bars change with every letter. Then turn on auto-write, slide the temperature from 0 to 2, and try 1 letter of context against 4.</p>`,
  terms: [
    { t: 'Language model', d: 'A program that gives the probability of each possible next piece of text.' },
    { t: 'Context', d: 'The text the model looks at before guessing. Here, the last few letters.' },
    { t: 'n-gram', d: 'A run of n letters or words. A trigram model looks at 2 letters to guess the 3rd.' },
    { t: 'Sampling', d: 'Picking the next piece at random, following the probabilities.' },
    { t: 'Temperature', d: 'A dial that sharpens (low) or flattens (high) the probabilities before picking.' },
    { t: 'Top-k', d: 'Keep only the k most likely choices, then pick among them.' },
  ],
  defaults: { prompt: 'monsoon', ctx: 3, temp: 0.8, topk: 0, auto: false, rate: 6, seed: 7, gen: 0 },
  controls: [
    { key: 'prompt', type: 'seg', label: 'Start with', options: Object.entries(PROMPTS).map(([v, t]) => ({ v, label: t.trim() ? `"${t.trim()}…"` : 'nothing' })) },
    { key: 'go', type: 'buttons', label: 'Write', items: [
      { label: 'Next letter', act: (s, i) => i.step?.(1) },
      { label: 'Next 20', act: (s, i) => i.step?.(20) },
      { label: 'Start again', act: (s) => { s.gen++; s.seed++; } },
    ] },
    { key: 'auto', type: 'toggle', label: 'Auto-write' },
    { key: 'ctx', type: 'seg', label: 'Context: letters it looks back', options: [0, 1, 2, 3, 4].map((v) => ({ v, label: String(v) })), fmt: (v) => ['unigram', 'bigram', 'trigram', '4-gram', '5-gram'][v] },
    { key: 'temp', type: 'range', label: 'Temperature', min: 0, max: 2, step: 0.01, ends: ['always the top pick', 'wild'], fmt: (v) => (v < 0.08 ? '0 (greedy)' : v.toFixed(2)) },
    { key: 'topk', type: 'seg', label: 'Top-k', options: [{ v: 1, label: '1' }, { v: 3, label: '3' }, { v: 5, label: '5' }, { v: 10, label: '10' }, { v: 0, label: 'all' }] },
  ],
  quiz: [
    { q: 'What does a language model actually output?', options: ['One fixed answer', 'A probability for every possible next piece', 'A search result', 'The rules of grammar'], answer: 1, why: 'It scores every option for the next piece. Then one is picked, added, and the loop repeats.' },
    { q: 'You set the temperature near 0. What happens?', options: ['It picks at random from everything', 'It always takes the most likely choice, and often repeats itself', 'It stops writing', 'It writes in Hindi'], answer: 1, why: 'Low temperature sharpens the odds until only the top choice is left. Safe, but it can loop.' },
    { q: 'Why does the 5-gram model copy its training text?', options: ['It is broken', 'Its long contexts were seen only once or twice, so only one letter ever followed', 'Copying is its goal', 'The text is too long'], answer: 1, why: 'With little text, a long context matches just one place in it, so the counts point to exactly what came next there.' },
  ],
  reel: [
    { ms: 5600, caption: 'A language model does one thing: it guesses the next piece of text, again and again.', set: { prompt: 'monsoon', ctx: 3, temp: 0.8, topk: 0, auto: true, rate: 5, seed: 11, gen: 101 }, spin: 0, view: { pos: [0, 4.4, 8.4], target: [0, 2.4, 0.3] } },
    { ms: 5000, caption: 'Temperature decides how daring it is: low plays safe, high gets wild.', set: { prompt: 'chai', ctx: 2, topk: 0, auto: true, rate: 9, seed: 4, gen: 102 }, anim: { temp: [0.1, 1.9] }, spin: 0, view: { pos: [0, 4.4, 8.4], target: [0, 2.4, 0.3] } },
  ],

  build({ stage }) {
    const top = new THREE.Group(); stage.root.add(top);
    const root = new THREE.Group(); root.position.y = 1.95; top.add(root);   // the bars stand on a shelf above the text board
    root.add(plate(7.6, 1.2));
    const leg = box(7.6, 1.95, 0.25, M.matte(0x151a24)); leg.position.set(0, -0.975, -0.3); root.add(leg);
    const x0 = -((V - 1) * SP) / 2;
    const bars = [], ghosts = [], tiles = [];
    const ghostMat = M.ghost(0xb8f2e6, 0.13);
    for (let i = 0; i < V; i++) {
      const b = box(BW, 1, BW, M.plastic(0x5fd3c0, { emissive: 0x0b3a33 })); b.position.set(x0 + i * SP, 0, 0); root.add(b); bars.push(b);
      const g = box(BW + 0.05, 1, BW + 0.05, ghostMat); g.castShadow = false; g.position.copy(b.position); root.add(g); ghosts.push(g);
      const t = tile(showChar(VOCAB[i]), 0.22, 0.22, COL.teal, 64); t.position.set(x0 + i * SP, 0.1, 0.4); t.rotation.x = -0.9; root.add(t); tiles.push(t);
    }
    // the text board
    const bd = board(1600, 400, 7.2, 1.8, (g, w, h, st) => st && drawText(g, w, h, st)); bd.mesh.position.set(0, 0.98, 0.72); bd.mesh.rotation.x = -0.12; top.add(bd.mesh);
    const L = {
      bars: stage.label('Solid bar: odds it picks from · faint: raw counts', [-2.0, 1.2, 0], root),
    };
    let blink = -1, text = '', gen = -1, r = rng(1), acc = 0, flash = -1, flashT = 0, st = null, dirty = true, seedUsed = -1, cur = null;

    function reset(s) { text = PROMPTS[s.prompt]; gen = s.gen; r = rng(s.seed * 7919 + s.gen); seedUsed = s.seed; acc = 0; dirty = true; if (!inReel() && text) step(24); }
    function step(n) {
      if (!st) return;
      n = Math.min(n, 200);
      for (let k = 0; k < n; k++) {
        const { p } = nextChar(text, st.ctx), q = shape(p, st.temp, st.topk), i = pick(q, r);
        text += VOCAB[i]; flash = i; flashT = 0.45;
        if (text.length > 420) text = text.slice(text.length - 300);
      }
      dirty = true;
    }
    function drawText(g, w, h, s) {
      panel(g, w, h);
      const res = nextChar(text, s.ctx);
      txt(g, 'THE TEXT SO FAR  (␣ = space)', 28, 50, F(26, true), COL.soft);
      txt(g, `looking back ${res.k} letter${res.k === 1 ? '' : 's'}: "${res.ctx.replace(/ /g, '␣')}"`, w - 28, 50, F(26, true), COL.hot, 'right');
      if (res.k < res.asked) txt(g, 'That longer context never appeared, so it backs off to a shorter one.', 28, h - 22, F(22), COL.soft);
      // the text, last ~3 lines, with the context highlighted and the newest letter marked
      g.font = F(44, false, true); const cw = g.measureText('m').width, perLine = Math.floor((w - 56) / cw);
      const shown = text.slice(-perLine * 4), lines = [];
      for (let i = 0; i < shown.length; i += perLine) lines.push(shown.slice(i, i + perLine));
      if (!lines.length) lines.push('');
      const start = shown.length - res.k, y0 = 118;
      let idx = 0;
      lines.forEach((ln, li) => {
        for (let c = 0; c < ln.length; c++, idx++) {
          const x = 28 + c * cw, y = y0 + li * 64;
          if (idx >= start) { g.fillStyle = 'rgba(255,209,102,.22)'; g.fillRect(x - 1, y - 42, cw + 2, 56); }
          const isNew = idx === shown.length - 1 && flashT > 0;
          txt(g, ln[c], x, y, F(44, isNew, true), isNew ? COL.hot : idx < shown.length - (text.length - PROMPTS[s.prompt].length) ? COL.soft : COL.white);
        }
      });
      const li = lines.length - 1, cx = 28 + (lines[li].length) * cw, cy = y0 + li * 64;
      if (Math.floor(performance.now() / 400) % 2 || inReel()) { g.fillStyle = COL.teal; g.fillRect(cx + 2, cy - 40, 4, 52); }
    }

    return {
      step,
      update(dt, s, time) {
        dt = Math.max(0, dt); st = s;
        const narrow = fit(stage, [L.bars], { x: 0.02, y: -0.06, ny: -0.16 });
        const tall = tallView(stage, 3.0, 0.95), hm = tall ? HMAX * 2.5 : HMAX;
        if (gen !== s.gen || seedUsed !== s.seed) reset(s);
        if (s.auto) { acc += dt * s.rate; const n = Math.floor(acc); if (n > 0) { acc -= n; step(n); } }
        flashT = Math.max(0, flashT - dt);
        const res = nextChar(text, s.ctx), q = shape(res.p, s.temp, s.topk); cur = { res, q };
        for (let i = 0; i < V; i++) {
          const hq = Math.max(0.002, q[i] * hm), hp = Math.max(0.002, res.p[i] * hm);
          const b = bars[i]; b.scale.y = approach(b.scale.y, hq, 14, dt); b.position.y = 0.08 + b.scale.y / 2;
          const gh = ghosts[i]; gh.scale.y = approach(gh.scale.y, hp, 14, dt); gh.position.y = 0.08 + gh.scale.y / 2;
          const on = i === flash && flashT > 0;
          b.material.color.setHex(on ? 0xffd166 : q[i] > 0 ? 0x5fd3c0 : 0x39424f);
          b.material.emissive.setHex(on ? 0x6a4a00 : 0x0b3a33);
        }
        
        if (dirty || Math.floor(time * 2.5) !== blink) { blink = Math.floor(time * 2.5); dirty = false; bd.redraw(s); }
      },
      readout(s) {
        if (!cur) return '';
        const { res, q } = cur, top = [...q.keys()].sort((a, b) => q[b] - q[a]).slice(0, 3);
        return `<div class="big">"${res.ctx.replace(/ /g, '␣')}" → ${top.map((i) => `${showChar(VOCAB[i])} ${(q[i] * 100).toFixed(0)}%`).join(', ')}</div>
          <div class="row"><span>That context in the text</span><b>${res.seen.toLocaleString('en')} times</b></div>
          <div class="row"><span>Uncertainty</span><b>${entropyBits(q).toFixed(2)} bits</b></div>
          <small>Count table: ${TABLES_INFO(s.ctx)}</small>`;
      },
    };
  },
};
