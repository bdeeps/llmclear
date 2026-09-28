// Chapter 6: using LLMs well. A real word-level trigram model (with back-off to bigrams), trained on the
// same short text (llm.js), writes four answers to a prompt. Each word shows the probability it had
// when picked. Then each sentence is checked two ways: is it copied word for word from the training
// text, and (for monsoon and fog claims) does it match the facts?
// Facts used for the check: IMD's normal monsoon onset dates, revised in 2020: Kerala 1 June, Mumbai
// 11 June, Delhi 27 June (India Meteorological Department). Delhi's dense fog season is winter,
// mainly December–January (IMD). The bias prompts show a pattern copied from the text: every doctor and
// engineer in it is "he", every nurse and teacher "she" (written that way on purpose to mirror skews
// found in web text; Bolukbasi et al. 2016 showed the same in word2vec).
// Memorisation: Carlini et al. 2021, "Extracting Training Data from Large Language Models".
import { THREE, M, box, approach } from '../kit.js';
import { generateWords, nextWord, inText, board, panel, txt, wrap, F, COL, tile, fit, inReel, tallView } from '../llm.js';

const PROMPTS = {
  monsoon: { text: 'the monsoon reaches', kind: 'fact', label: 'the monsoon reaches…' },
  fog: { text: 'thick fog covers', kind: 'fact', label: 'thick fog covers…' },
  doctor: { text: 'the doctor said', kind: 'bias', label: 'the doctor said…' },
  nurse: { text: 'the nurse said', kind: 'bias', label: 'the nurse said…' },
  cup: { text: 'a cup of chai costs', kind: 'memory', label: 'a cup of chai costs…' },
};
const NOTE = {
  fact: ['Fluent is not the same as true.', 'It stitches together pieces that often follow each other. "Delhi in" was followed by "december" once (about fog), so it can tell you the monsoon reaches Delhi in December, just as confidently. Big LLMs do this too: it is called hallucination. Check facts against a trusted source.'],
  bias: ['It repeats the patterns in its text.', 'Every doctor in its text was "he" and every nurse "she", so that is all it can say. Real models learn skews from the web in the same way, which is why builders test and adjust for bias, and why you should notice it.'],
  memory: ['It can repeat its training text word for word.', 'This line was seen once, so it comes back exactly. Big models can sometimes repeat rare text they were trained on, which is a privacy worry if that text held personal details. Don\'t paste private data into chatbots you don\'t trust.'],
};
const N_ANS = 4;

function check(words) {
  const s = words.join(' ');
  const month = (after) => { const i = words.indexOf(after); if (i < 0) return null; const rest = words.slice(i + 1); const m = rest.find((w) => ['june', 'july', 'december'].includes(w)); return m ? { m, late: rest[0] === 'late', early: rest[0] === 'early' } : null; };
  if (words.includes('monsoon') && words.includes('reaches')) {
    const place = ['kerala', 'mumbai', 'delhi'].find((p) => words[words.indexOf('reaches') + 1] === p);
    if (place && words[words.indexOf(place) + 1] === 'in') {
      const m = month(place);
      if (m) return m.m === 'june' ? { ok: true, why: `True: IMD's normal onset for ${place[0].toUpperCase() + place.slice(1)} is in June.` } : { ok: false, why: `False: the monsoon reaches ${place[0].toUpperCase() + place.slice(1)} in June, not ${m.m[0].toUpperCase() + m.m.slice(1)}.` };
    }
  }
  if (words.includes('fog') && words.includes('delhi')) {
    const m = month('delhi');
    if (m) return m.m === 'december' ? { ok: true, why: 'True: Delhi\'s thick fog comes in winter.' } : { ok: false, why: `False: Delhi has no thick fog in ${m.m[0].toUpperCase() + m.m.slice(1)}. Fog is a winter thing.` };
  }
  return null;
}

export default {
  id: 'use',
  short: 'Use them well',
  title: 'Use them well: fluent is not the same as true',
  subtitle: 'What LLMs do well, where they go wrong, and how to think about them.',
  view: { pos: [0.3, 3.2, 8.4], target: [0.3, 1.75, 0] },
  learn: `<p>Here our tiny model writes whole sentences, one word at a time, with the chance of each word shown under it. Every answer sounds sure. Some are copied from its text, some are new, and some new ones are <b>false</b>. Nothing inside the model marks the difference.</p>
    <p>That is the big lesson for large models too. They predict likely text, so they can produce a smooth, confident answer that is simply wrong: a made-up date, a quote nobody said, a book that doesn't exist. This is called <b>hallucination</b>. They also absorb <b>bias</b> from their training text, and they can sometimes <b>repeat</b> rare text they saw, which matters for <b>privacy</b>.</p>
    <p><b>What they do well:</b> explain an idea in simpler words, draft and edit writing, summarise a long text you give them, translate, brainstorm, and help with code. <b>What they do badly:</b> exact facts, fresh news, sums done "in the head", and anything where being wrong is costly, unless they can check a source or run a tool.</p>
    <p><b>Use them well.</b> Give context and say what you want: "explain the monsoon to a 12-year-old in 5 lines". Ask for sources, then open them. Check numbers, names, dates and medicines against trusted sites. Don't paste passwords, Aadhaar numbers or private data. Treat the answer as a helpful first draft, not the final word.</p>
    <p><b>In India</b>, groups are building models for Indian languages: Sarvam AI (chosen in 2025 under the government's IndiaAI Mission to build a national model), Ola's Krutrim (2023) and the government-funded BharatGen (2024), alongside the Bhashini translation platform. In 2023 NPCI announced <b>Hello! UPI</b>, voice payments in Hindi and English, with language models developed with AI4Bharat at IIT Madras and the Bhashini programme, on a payment system that handled around 20 billion payments a month in 2025. How well such tools serve every language is still being worked out.</p>
    <p><b>How to think about AI:</b> an LLM is neither a mind nor a trick. It is a very large, very well-trained next-token predictor. That turns out to be remarkably useful and also unreliable in predictable ways. Use it like a clever helper whose work you check. Your own brain (see BrainClear) learns from far less text and knows when it doesn't know. The model mostly doesn't.</p>
    <p class="tip"><b>Try it:</b> pick "the monsoon reaches…" and press "Write again" a few times. Watch for a false answer said with high confidence. Then try the doctor and the nurse.</p>`,
  terms: [
    { t: 'Hallucination', d: 'A fluent, confident answer that is false or made up.' },
    { t: 'Bias', d: 'Unfair or one-sided patterns a model learns from its training text.' },
    { t: 'Memorisation', d: 'Repeating training text word for word, a risk for private or copyrighted text.' },
    { t: 'Prompt', d: 'The text you give the model. Clear context and a clear request get better answers.' },
    { t: 'Grounding', d: 'Giving the model trusted sources to answer from, and checking its claims against them.' },
  ],
  defaults: { prompt: 'monsoon', seed: 3 },
  controls: [
    { key: 'prompt', type: 'seg', label: 'Prompt', options: Object.entries(PROMPTS).map(([v, p]) => ({ v, label: p.label })) },
    { key: 'go', type: 'buttons', label: 'Answers', items: [{ label: 'Write again', act: (s) => { s.seed += N_ANS; } }] },
  ],
  quiz: [
    { q: 'An LLM gives a confident, fluent answer. What does that tell you?', options: ['It must be true', 'Nothing about truth: it predicts likely text', 'It looked it up', 'It is copying a textbook'], answer: 1, why: 'Confidence comes from how likely the words are, not from checking facts.' },
    { q: 'Why did our model say every doctor was "he"?', options: ['Doctors are always men', 'That was the only pattern in its training text', 'It chose at random', 'It was told to'], answer: 1, why: 'Models repeat the patterns in their data, including unfair ones.' },
    { q: 'Which is the best habit when using a chatbot?', options: ['Paste your passwords so it can help', 'Trust numbers and dates it gives', 'Ask for sources and check important facts', 'Never give it any context'], answer: 2, why: 'Treat the answer as a first draft. Check what matters, and keep private data out.' },
  ],
  reel: [
    { ms: 6000, caption: 'Fluent is not the same as true: models can state false things with full confidence.', set: { prompt: 'monsoon', seed: 3 }, spin: 0, view: { pos: [0.3, 2.9, 8.2], target: [0.3, 2.1, 0] } },
  ],

  build({ stage }) {
    const root = new THREE.Group(); stage.root.add(root);
    const wall = box(9.2, 3.1, 0.08, M.matte(0x121722)); wall.position.set(0, 2.05, -0.15); root.add(wall);
    const bd = board(1500, 380, 6.8, 1.72, (g, w, h, s) => s && drawBoard(g, w, h, s)); bd.mesh.position.set(0, 0.5, 1.25); bd.mesh.rotation.x = -1.1; root.add(bd.mesh);
    const L = { conf: stage.label('Under each word: how likely it was', [2.8, 3.75, 0], root) };
    let tiles = [], key = '', answers = [], tall = false;

    function rebuild(s) {
      tiles.forEach((t) => { root.remove(t); t.geometry.dispose(); t.material.map.dispose(); t.material.dispose(); }); tiles = [];
      const P = PROMPTS[s.prompt];
      answers = [];
      let y0 = tall ? 7.2 : 3.1;
      for (let a = 0; a < N_ANS; a++) {
        const g = generateWords(P.text, s.seed * 131 + a * 17);
        const cont = g.words.slice(P.text.split(' ').length), probs = g.steps.map((x) => x.p);
        const full = g.words.join(' '), copied = inText(full), fact = check(g.words);
        answers.push({ full, copied, fact, conf: probs.length ? probs.reduce((x, y) => x * y, 1) : 1 });
        const ws = cont.map((w) => 0.3 + 0.1 * w.length);
        let k = 1, x = tall ? -2.45 : -4.3, yy = y0;
        if (!tall) { const tot = ws.reduce((p, q) => p + q + 0.06, 0.32); k = Math.min(1, 7.9 / tot); }
        const vy = yy;
        const put = (m, w) => { if (tall && x + w > 2.45) { x = -2.45; yy -= 0.6; } m.position.set(x + w / 2, yy, 0); root.add(m); tiles.push(m); x += w + 0.06 * k; };
        cont.forEach((w, i) => {
          const tw = ws[i] * k, col = probs[i] >= 0.99 ? '#b8f2e6' : probs[i] >= 0.5 ? '#8ef0ff' : '#c49bff';
          const m = tile(w, tw, 0.5, col, 96); m.redraw(w, col, `${Math.round(probs[i] * 100)}%`); put(m, tw);
        });
        const dot = tile('.', 0.26 * k, 0.5, '#8fa3b8', 96); put(dot, 0.26 * k);
        y0 = yy - (tall ? 0.85 : 0.7);
        const verdict = fact ? (fact.ok ? ['TRUE', COL.green] : ['FALSE', COL.pink]) : copied ? ['COPIED', COL.teal] : ['NEW', COL.violet];
        const v = tile(verdict[0], 0.95, 0.44, verdict[1], 96); v.position.set(tall ? 3.0 : 4.1, vy, 0.02); root.add(v); tiles.push(v);
      }
      bd.redraw(s);
    }
    function drawBoard(g, w, h, s) {
      panel(g, w, h);
      const P = PROMPTS[s.prompt], [head, body] = NOTE[P.kind];
      txt(g, `Prompt: "${P.text}…"`, 30, 56, F(34, true), COL.soft);
      txt(g, head, w - 30, 56, F(34, true), COL.hot, 'right');
      const firstFact = answers.find((a) => a.fact);
      let y = wrap(g, body, 30, 108, w - 60, 38, F(28), COL.white);
      if (firstFact) wrap(g, 'Fact check: ' + firstFact.fact.why + ' (IMD)', 30, y + 6, w - 60, 36, F(28, true), firstFact.fact.ok ? COL.green : COL.pink);
    }

    return {
      update(dt, s) {
        dt = Math.max(0, dt);
        fit(stage, [L.conf], { x: 0.05, y: -0.07, ny: -0.16 });
        tall = tallView(stage, 2.4, 0.95);
        wall.scale.set(tall ? 0.72 : 1, tall ? 2.35 : 1, 1); wall.position.set(tall ? 0.3 : 0, tall ? 4.1 : 2.05, -0.15);
        L.conf.position.y = tall ? 7.9 : 3.75;
        const k = `${s.prompt}|${s.seed}|${tall}`;
        if (k !== key) { key = k; rebuild(s); }
      },
      readout(s) {
        const P = PROMPTS[s.prompt], nx = nextWord(P.text.split(' ')).list.slice(0, 3);
        const copied = answers.filter((a) => a.copied).length, facts = answers.filter((a) => a.fact), wrong = facts.filter((a) => !a.fact.ok).length;
        return `<div class="big">${facts.length ? `${wrong} of ${facts.length} claims false` : `${copied} of ${N_ANS} copied`}</div>
          <div class="row"><span>Next word odds</span><b>${nx.map((o) => `${o.w} ${Math.round(o.p * 100)}%`).join(', ')}</b></div>
          <div class="row"><span>Answers copied from its text</span><b>${copied} of ${N_ANS}</b></div>
          <small>A word trigram model trained on this box's text.</small>`;
      },
    };
  },
};
