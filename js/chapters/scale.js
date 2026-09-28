// Chapter 5: scale and training. Towers show reported parameter counts on a log scale. Sources:
//  GPT-1 117M (Radford et al. 2018); BERT-Large 340M (Devlin et al. 2018); GPT-2 1.5B (Radford et al.
//  2019); T5 11B (Raffel et al. 2019); GPT-3 175B (Brown et al. 2020); Gopher 280B (Rae et al. 2021);
//  PaLM 540B (Chowdhery et al. 2022); Chinchilla 70B on 1.4T tokens (Hoffmann et al. 2022); LLaMA 65B
//  (Touvron et al. 2023); GPT-4: size not disclosed (OpenAI technical report, 2023); Llama 3.1 405B on
//  over 15T tokens, 30.84M H100 GPU-hours (Meta model card, July 2024); Sarvam-1 2B (Sarvam AI, Oct 2024);
//  DeepSeek-V3 671B total, 37B active per token (DeepSeek-AI, Dec 2024).
// Compute: training FLOPs ≈ 6 × parameters × tokens (Kaplan et al. 2020). Check: GPT-3 6 × 175e9 × 300e9
//  = 3.15e23, matching the paper's 3.14e23. Throughput assumption: an H100 does about 989 TFLOP/s dense
//  BF16 at peak; big runs reach roughly 40% of that (Llama 3 paper reports 38–43% MFU), so ~400 TFLOP/s.
//  With that, Llama 3.1 405B (6 × 405e9 × 15.6e12 = 3.8e25) comes out near 26M GPU-hours against 30.84M
//  reported. GPU energy: 700 W per H100 (its TDP); the whole data centre uses more.
// India comparison: per-capita electricity consumption about 1,395 kWh in 2023-24 (Ministry of Power /
//  PIB), so 1,287 MWh ≈ 920 people's yearly use.
// Energy reported: GPT-3 training about 1,287 MWh (Patterson et al. 2021, estimate). A median Gemini Apps
//  text prompt about 0.24 Wh (Google, August 2025, company figure).
import { THREE, M, box, approach } from '../kit.js';
import { board, panel, txt, wrap, F, COL, tile, fit, inReel, fmtBig, sci, tableSize, V, tallView } from '../llm.js';

const MODELS = [
  { name: 'Our 4-gram', year: 'ch. 1', p: 0, ours: true },
  { name: 'GPT-1', year: 2018, p: 117e6 },
  { name: 'BERT-Large', year: 2018, p: 340e6 },
  { name: 'GPT-2', year: 2019, p: 1.5e9 },
  { name: 'T5', year: 2019, p: 11e9 },
  { name: 'GPT-3', year: 2020, p: 175e9 },
  { name: 'Gopher', year: 2021, p: 280e9 },
  { name: 'PaLM', year: 2022, p: 540e9 },
  { name: 'Chinchilla', year: 2022, p: 70e9 },
  { name: 'LLaMA', year: 2023, p: 65e9 },
  { name: 'GPT-4', year: 2023, p: null },
  { name: 'Llama 3.1', year: 2024, p: 405e9 },
  { name: 'Sarvam-1', year: 2024, p: 2e9 },
  { name: 'DeepSeek-V3', year: 2024, p: 671e9, note: '37B active' },
  { name: 'Your model', year: 'slider', p: 0, yours: true },
];
MODELS[0].p = tableSize(3).contexts * V;                  // the chapter-1 table at 3 letters of context: contexts × letters (about 21,000)
const HY = (p) => Math.max(0.05, 0.42 * (Math.log10(p) - 3));   // 1 thousand at the floor, +0.42 per ×10
const SPX = 0.56;
const STAGES = {
  pre: { name: '1. Pretraining', short: 'Pretrain', what: 'Read a huge pile of text and learn to predict the next token. The result can continue any text, but it is not yet an assistant.',
    reply: 'What is the capital of Nepal? What is the capital of Bhutan? What is the capital of' },
  sft: { name: '2. Instruction tuning', short: 'Fine-tune', what: 'Train further on many written examples of good questions and answers, so it learns to reply instead of just continuing.',
    reply: 'The capital of India is New Delhi.' },
  rlhf: { name: '3. Feedback (RLHF)', short: 'RLHF', what: 'People compare pairs of answers and pick the better one. A reward model learns their taste, and the LLM is nudged towards answers people prefer.',
    reply: 'A: "New Delhi. It became the capital in 1931."  B: "Delhi, obviously."  People prefer A, so the model is nudged that way.' },
};

export default {
  id: 'scale',
  short: 'Scale & training',
  title: 'Scale: more numbers, more text, more power',
  subtitle: 'The same next-token idea, grown a million times bigger, then taught to be helpful.',
  view: { pos: [0.8, 3.4, 10.0], target: [0.6, 1.8, 0.4] },
  learn: `<p>Our chapter 1 model is a table of about <b>21,000</b> numbers. Big LLMs have <b>billions</b> of numbers, called <b>parameters</b>: the weights inside all those attention layers and neural networks. GPT-2 (2019) had 1.5 billion. GPT-3 (2020) had 175 billion. Some newer companies no longer say. The towers use a <b>log scale</b>: each step up is ten times more.</p>
    <p><b>1. Pretraining.</b> The model reads a huge pile of text, from books, websites and code, trillions of tokens, and is trained on one task: guess the next token. Each wrong guess nudges every parameter a little (the learning is the same as in NeuralNetClear, just vastly bigger). After that it can continue any text, but it is not yet an assistant.</p>
    <p><b>2. Instruction tuning.</b> It is trained further on written examples of questions and good answers. <b>3. Feedback.</b> People compare pairs of answers and pick the better one, and the model is nudged towards what they prefer. This is <b>RLHF</b>, reinforcement learning from human feedback, or a similar method.</p>
    <p><b>Why bigger helps:</b> researchers found that the prediction error falls smoothly and predictably as parameters, data and computing grow together. A 2022 study (Chinchilla) found many models had been trained on too little text: roughly <b>20 tokens per parameter</b> works better for a fixed budget.</p>
    <p><b>The costs are real.</b> Training GPT-3 was estimated at about <b>1,300 MWh</b> of electricity, about the yearly electricity use of 900 people in India. Meta reported <b>30.8 million GPU-hours</b> for its largest Llama 3.1 model. Each answer costs energy too: Google reported <b>0.24 Wh</b> for a median text prompt to its Gemini app in 2025, about one second of a 1,000-watt microwave. Estimates vary a lot between models and studies.</p>
    <p class="tip"><b>Try it:</b> set your own model's size and training text with the sliders and read the compute, GPU time and energy. Then step through the three training stages on the board.</p>`,
  terms: [
    { t: 'Parameter', d: 'One adjustable number inside the model. Big LLMs have billions.' },
    { t: 'Pretraining', d: 'The first, huge training run: predict the next token on trillions of tokens of text.' },
    { t: 'Fine-tuning', d: 'Extra training on a smaller, chosen set of examples to shape behaviour.' },
    { t: 'RLHF', d: 'Reinforcement learning from human feedback: learning from people\'s choices between answers.' },
    { t: 'FLOP', d: 'One floating-point operation, such as one multiplication. Training takes about 6 per parameter per token.' },
    { t: 'GPU', d: 'A chip with thousands of small cores, good at the matrix sums that neural networks need.' },
    { t: 'Log scale', d: 'An axis where each equal step means ten times more.' },
  ],
  defaults: { N: 7e9, D: 2e12, stage: 'pre' },
  controls: [
    { key: 'N', type: 'log', label: 'Your model: parameters', min: 1e8, max: 2e12, fmt: (v) => fmtBig(v) },
    { key: 'D', type: 'log', label: 'Training text: tokens', min: 1e10, max: 3e13, fmt: (v) => fmtBig(v) + ' tokens' },
    { key: 'stage', type: 'seg', label: 'Training stage', options: Object.entries(STAGES).map(([v, o]) => ({ v, label: o.short })), fmt: (v) => STAGES[v].name },
  ],
  quiz: [
    { q: 'What is a parameter?', options: ['A setting you choose in the app', 'One learned number inside the model', 'A token', 'A training document'], answer: 1, why: 'Parameters are the weights learned in training. GPT-3 had 175 billion.' },
    { q: 'What does pretraining teach a model?', options: ['To follow instructions politely', 'To predict the next token in huge amounts of text', 'To search the internet', 'Grammar rules written by linguists'], answer: 1, why: 'Instruction tuning and feedback come afterwards to turn it into a helpful assistant.' },
    { q: 'About how many operations does training take, per parameter per token?', options: ['1', 'About 6', 'About 1,000', 'A million'], answer: 1, why: 'Roughly 2 for the forward pass and 4 for learning, so compute ≈ 6 × parameters × tokens.' },
  ],
  reel: [
    { ms: 5800, caption: 'Scale: from thousands of numbers to hundreds of billions, trained on trillions of tokens.', set: { stage: 'pre', D: 2e12 }, anim: { N: [1e8, 1.5e12, true] }, spin: 0, view: { pos: [0.6, 2.9, 9.0], target: [0.6, 2.0, 0] } },
  ],

  build({ stage }) {
    const root = new THREE.Group(); stage.root.add(root);
    const n = MODELS.length, x0 = -((n - 1) * SPX) / 2;
    // back wall with the log grid (a second, taller one for tall frames, where the towers stretch ×2)
    const drawWall = (g, w, h, f) => {
      g.clearRect(0, 0, w, h); g.fillStyle = 'rgba(12,15,22,.9)'; g.fillRect(0, 0, w, h);
      const pxu = h / (5.0 * f);
      [[1e3, 'a thousand'], [1e6, 'a million'], [1e9, 'a billion'], [1e12, 'a trillion']].forEach(([v, s]) => {
        const y = h - (HY(v) * f + 0.1) * pxu;
        g.strokeStyle = 'rgba(184,242,230,.35)'; g.lineWidth = 2; g.setLineDash([10, 8]); g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke(); g.setLineDash([]);
        txt(g, s, w - 16, y - 10, F(30, true), COL.teal, 'right');
      });
      txt(g, 'PARAMETERS (log scale, reported)', w - 16, 44, F(28, true), COL.soft, 'right');
    };
    const wall = board(1800, 900, 10.0, 5.0, (g, w, h) => drawWall(g, w, h, 1));
    wall.mesh.position.set(0.55, 2.5 - 0.1, -0.7); root.add(wall.mesh);
    const wallT = board(1800, 1800, 10.0, 10.0, (g, w, h) => drawWall(g, w, h, 2));
    wallT.mesh.position.set(0.55, 5.0 - 0.1, -0.7); root.add(wallT.mesh); wallT.mesh.visible = false;
    const floor = box(n * SPX + 0.6, 0.1, 2.2, M.matte(0x1a2030)); floor.position.set(0, 0.05, 0.3); root.add(floor);
    const towers = MODELS.map((m, i) => {
      const col = m.ours ? 0xb8f2e6 : m.yours ? 0xffd166 : m.p === null ? 0x8ea0b8 : m.name === 'Sarvam-1' ? 0x7be08c : 0xc49bff;
      const mat = m.p === null ? M.ghost(col, 0.25) : M.plastic(col, { emissive: col, emissiveIntensity: 0.12 });
      const t = box(0.4, 1, 0.4, mat); t.position.set(x0 + i * SPX, 0, 0); root.add(t);
      const h = m.p === null ? 0.3 : HY(m.p);   // size not disclosed: a short ghost stub, not a guess
      const nm = tile(m.name, 1.05, 0.34, '#' + col.toString(16).padStart(6, '0'), 80); nm.redraw(m.name, '#' + col.toString(16).padStart(6, '0'), String(m.year));
      nm.position.set(x0 + i * SPX, 0.16, i % 2 ? 1.25 : 0.62); nm.rotation.x = -1.2; root.add(nm);
      const val = tile(m.p === null ? '? not disclosed' : fmtShort(m.p), m.p === null ? 1.0 : 0.62, 0.22, COL.white, 64); root.add(val);
      return { t, val, m, h };
    });
    const place = (f) => towers.forEach((o) => { if (o.m.yours) return; const h = o.m.p === null ? o.h : o.h * f; o.t.scale.y = h; o.t.position.y = 0.1 + h / 2; o.val.position.set(o.t.position.x, 0.1 + h + 0.2, 0); });
    place(1);
    const yours = towers.at(-1);
    const bd = board(1500, 420, 6.6, 1.85, (g, w, h, s) => s && drawStage(g, w, h, s)); bd.mesh.position.set(0.3, 0.3, 2.85); bd.mesh.rotation.x = -1.25; root.add(bd.mesh);
    const L = { wall: stage.label('Each step up the wall: 10 times more', [3.4, 4.75, -0.6], root) };
    let key = '', vKey = '', f = 1;

    function drawStage(g, w, h, s) {
      panel(g, w, h);
      const S = STAGES[s.stage];
      txt(g, S.name, 30, 58, F(40, true), COL.hot);
      wrap(g, S.what, 30, 108, w - 60, 38, F(28), COL.white);
      txt(g, 'Prompt: "What is the capital of India?"', 30, 250, F(28, true), COL.soft);
      wrap(g, 'Reply: ' + S.reply, 30, 300, w - 60, 40, F(30, true), COL.teal);
      txt(g, 'Illustrative example', w - 30, h - 18, F(22), COL.soft, 'right');
    }

    return {
      update(dt, s) {
        dt = Math.max(0, dt);
        fit(stage, [L.wall], { x: 0.06, y: -0.03, ny: -0.16 });
        if (s.stage !== key) { key = s.stage; bd.redraw(s); }
        const tall = tallView(stage, 2.6, 0.95), nf = tall ? 2 : 1;
        if (nf !== f) { f = nf; place(f); wall.mesh.visible = f === 1; wallT.mesh.visible = f === 2; L.wall.position.y = f === 1 ? 4.75 : 9.6; }
        const h = HY(s.N) * f;
        yours.t.scale.y = approach(yours.t.scale.y, h, 8, dt); yours.t.position.y = 0.1 + yours.t.scale.y / 2;
        yours.val.position.y = 0.1 + yours.t.scale.y + 0.2;
        const vk = fmtShort(s.N); if (vk !== vKey) { vKey = vk; yours.val.redraw(vk, COL.hot); }
      },
      readout(s) {
        const C = 6 * s.N * s.D, gpuh = C / 4e14 / 3600, mwh = (gpuh * 0.7) / 1000;
        return `<div class="big">${fmtBig(s.N)} parameters</div>
          <div class="row"><span>Compute, 6 × N × D</span><b>${sci(C)} FLOPs</b></div>
          <div class="row"><span>H100 GPU-hours</span><b>${gpuh >= 1e6 ? (gpuh / 1e6).toFixed(1) + ' million' : Math.round(gpuh).toLocaleString('en')}</b></div>
          <div class="row"><span>GPU electricity</span><b>${mwh >= 10 ? Math.round(mwh).toLocaleString('en') : mwh.toFixed(1)} MWh</b></div>
          <small>${Math.round(s.D / s.N)} tokens per parameter (Chinchilla: ~20). Rough: ~400 TFLOP/s and 700 W per GPU, GPUs only.</small>`;
      },
    };
  },
};
function fmtShort(p) { if (p >= 1e12) return (p / 1e12).toFixed(1) + 'T'; if (p >= 1e9) return (p / 1e9 >= 10 ? Math.round(p / 1e9) : (p / 1e9).toFixed(1)) + 'B'; if (p >= 1e6) return Math.round(p / 1e6) + 'M'; if (p >= 1e3) return Math.round(p / 1e3) + 'K'; return String(Math.round(p)); }
