// Chapter 2: tokens. A real byte-level BPE tokenizer (Sennrich, Haddow & Birch 2016; byte-level as in
// GPT-2) is trained when the page loads, twice: once on the English text only, once on English + Hindi.
// Training starts from the 256 possible bytes and keeps merging the most frequent neighbouring pair
// (ids 256, 257, …), stopping when no pair repeats. The slider replays the merges one by one.
// UTF-8: a–z are 1 byte each, Devanagari letters and signs (U+0900–U+097F) are 3 bytes each.
// Petrov et al. (NeurIPS 2023) found the same text can take many times more tokens in some languages
// than in English with common tokenizers; the exact ratio for Hindi depends on the tokenizer.
import { THREE, M, box, approach, clamp } from '../kit.js';
import {
  TOKENIZERS, encode, tokenText, isPartial, utf8Len, pretok,
  board, panel, txt, F, COL, tile, fit, inReel, hex, tallView,
} from '../llm.js';

const SAMPLES = {
  en: { name: 'English', text: 'the rain is loud on the tin roof, and the chai is hot and sweet.' },
  hi: { name: 'Hindi', text: 'टीन की छत पर बारिश का शोर है, और चाय गरम और मीठी है।' },
  yours: { name: 'Yours', text: '' },
};
let yours = 'Shall we get chai? चलो चाय पीते हैं।';
const ROWW = 7.4, TH = 0.56, GAP = 0.07;
const colFor = (id, partial) => (partial ? '#ff7a59' : id < 256 ? '#8fa3b8' : ['#b8f2e6', '#ffd166', '#c49bff', '#8ef0ff', '#7be08c', '#f5a3d0', '#ffb547'][id % 7]);

export default {
  id: 'tokens',
  short: 'Tokens',
  title: 'Text becomes tokens',
  subtitle: 'Before a model reads a word, the word is chopped into numbered pieces.',
  view: { pos: [0, 2.4, 7.6], target: [0, 1.6, 0] },
  onChange(s, key) { const max = TOKENIZERS[s.tok].bpe.merges.length; if (s.merges > max) s.merges = max; },
  learn: `<p>An LLM never sees letters or words. It sees <b>tokens</b>: numbered pieces of text. "chai" might be one token, "monsoon" two or three, a space-plus-word often one. A big model knows somewhere between tens of thousands and a few hundred thousand tokens.</p>
    <p>Where do the pieces come from? A method called <b>byte-pair encoding</b> (BPE). Start with the <b>256 bytes</b> that computers store text in. Find the pair of neighbours that appears most often in the training text, like "t" + "h", and give it a new number. Repeat. After hundreds or thousands of merges, common words become single tokens and rare words are built from parts. This one is trained live on this box's text: watch it merge.</p>
    <p>Now try <b>Hindi</b>. In UTF-8, each Devanagari letter or sign takes <b>3 bytes</b>, against 1 for an English letter. If a tokenizer learned its merges mostly from English, it knows few Hindi pieces, so Hindi falls apart into bytes: about 3 tokens per letter here. Researchers have found the same effect in real tokenizers: some languages need <b>several times more tokens</b> than English for the same meaning, which costs more and fits less text in the model's window. Train on some Hindi too and the gap shrinks. Newer tokenizers and Indian models are built with this in mind.</p>
    <p><b>Count your own:</b></p>
    <textarea id="tokIn" rows="3" spellcheck="false" style="width:100%;box-sizing:border-box;background:#0d1118;color:#e8eef8;border:1px solid rgba(184,242,230,.35);border-radius:10px;padding:10px;font-size:15px;line-height:1.4;font-family:inherit;resize:vertical">${yours}</textarea>
    <p id="tokOut" style="font-size:14px;color:#b8f2e6;margin-top:6px"></p>
    <p class="tip"><b>Try it:</b> press "Back to bytes", then "Learn 20 merges" a few times and watch the pieces fuse. Switch the sentence to Hindi, then switch the tokenizer to "English + Hindi" and see the token count fall.</p>`,
  terms: [
    { t: 'Token', d: 'A numbered piece of text: a word, part of a word, a letter or a byte.' },
    { t: 'Token ID', d: 'The number a token is stored as. The model only ever sees these numbers.' },
    { t: 'Byte', d: 'Eight bits. UTF-8 stores an English letter in 1 byte and a Hindi letter in 3.' },
    { t: 'BPE', d: 'Byte-pair encoding: build tokens by merging the most common neighbouring pair, again and again.' },
    { t: 'Vocabulary', d: 'The full list of tokens a model knows: here 256 bytes plus every merge.' },
    { t: 'Context window', d: 'The most tokens a model can read at once. More tokens per word means less text fits.' },
  ],
  defaults: { tok: 'en', sample: 'en', merges: 60, auto: false },
  controls: [
    { key: 'sample', type: 'seg', label: 'Sentence', options: Object.entries(SAMPLES).map(([v, o]) => ({ v, label: o.name })) },
    { key: 'tok', type: 'seg', label: 'Tokenizer trained on', options: Object.entries(TOKENIZERS).map(([v, o]) => ({ v, label: o.name })) },
    { key: 'merges', type: 'range', label: 'Merges learned', min: 0, max: 306, step: 1, ends: ['raw bytes', 'all it found'], fmt: (v, s) => `${Math.round(v)} of ${TOKENIZERS[s.tok].bpe.merges.length}` },
    { key: 'go', type: 'buttons', label: 'Train', items: [
      { label: 'Learn 1 merge', act: (s) => { s.merges = Math.min(TOKENIZERS[s.tok].bpe.merges.length, Math.round(s.merges) + 1); } },
      { label: 'Learn 20 merges', act: (s) => { s.merges = Math.min(TOKENIZERS[s.tok].bpe.merges.length, Math.round(s.merges) + 20); } },
      { label: 'Back to bytes', act: (s) => { s.merges = 0; } },
    ] },
    { key: 'auto', type: 'toggle', label: 'Keep merging (about 10 a second)' },
  ],
  quiz: [
    { q: 'What does an LLM actually read?', options: ['Letters', 'Whole sentences', 'Token IDs: numbers for pieces of text', 'Pictures of words'], answer: 2, why: 'Text is chopped into tokens and each token is a number. The model works only with those numbers.' },
    { q: 'How does BPE build its tokens?', options: ['From a dictionary', 'By merging the most common neighbouring pair, again and again', 'One token per letter', 'At random'], answer: 1, why: 'It starts from bytes and keeps giving the most frequent pair a new number.' },
    { q: 'Why can Hindi take more tokens than English?', options: ['Hindi words are longer', 'Each Devanagari letter is 3 bytes, and English-heavy tokenizers learned few Hindi merges', 'Hindi has no spaces', 'Models cannot read Hindi'], answer: 1, why: 'Without learned Hindi pieces, text falls back to bytes. Training the tokenizer on Hindi too brings the count down.' },
  ],
  reel: [
    { ms: 5400, caption: 'Models read tokens, not letters. Merging common pairs turns bytes into word pieces.', set: { tok: 'en', sample: 'en', auto: false }, anim: { merges: [0, 231] }, spin: 0, view: { pos: [0, 2.4, 7.4], target: [0, 1.9, 0] } },
    { ms: 5200, caption: 'Hindi letters take 3 bytes each, so English-trained tokenizers often need many more tokens.', set: { tok: 'en', sample: 'hi', merges: 231, auto: false }, spin: 0, view: { pos: [0, 2.4, 7.8], target: [0, 1.8, 0] } },
  ],

  build({ stage }) {
    const root = new THREE.Group(); stage.root.add(root);
    const wall = box(ROWW + 0.6, 2.9, 0.08, M.matte(0x121722)); wall.position.set(0, 1.95, -0.12); root.add(wall);
    const bd = board(1500, 300, 7.4, 1.48, (g, w, h, s) => s && drawBoard(g, w, h, s)); bd.mesh.position.set(0, 0.5, 1.0); bd.mesh.rotation.x = -1.0; root.add(bd.mesh);
    const L = { wall: stage.label('Each tile: one token, with its ID number', [2.4, 3.6, 0], root) };
    let tiles = [], key = '', st = null, acc = 0, outKey = '', info = null, rowW = ROWW, topY = 3.05, maxH = 2.6, wasTall = false;

    // layout: wrap tokens into rows on the wall
    function layout(toks) {
      const items = toks.map((t) => ({ ...t, w: clamp(0.2 + 0.11 * [...t.text].length, 0.3, 2.2) }));
      const rows = [[]]; let x = 0;
      for (const it of items) { if (x + it.w > rowW && rows.at(-1).length) { rows.push([]); x = 0; } rows.at(-1).push(it); x += it.w + GAP; }
      const n = rows.length, th = n > 4 ? Math.min(TH, maxH / n - GAP) : TH;
      rows.forEach((r, ri) => {
        const W = r.reduce((a, it) => a + it.w + GAP, -GAP); let xx = -W / 2;
        r.forEach((it) => { it.x = xx + it.w / 2; it.y = topY - th / 2 - ri * (th + GAP) - (n <= 4 ? (4 - n) * 0.3 : 0); it.h = th; xx += it.w + GAP; });
      });
      return items;
    }
    function rebuild(s) {
      const bpe = TOKENIZERS[s.tok].bpe, text = s.sample === 'yours' ? yours.slice(0, 140) : SAMPLES[s.sample].text;
      const n = Math.round(s.merges), ids = encode(text, bpe, n);
      let off = 0;
      const toks = ids.map((id) => { const t = { id, text: tokenText(bpe, id), partial: isPartial(bpe, id), start: off, len: bpe.bytes[id].length }; off += t.len; return t; });
      const items = layout(toks);
      const old = tiles; tiles = [];
      for (const it of items) {
        // reuse a tile if the very same token sat at the same place before; otherwise fuse from the old pieces
        const same = old.find((o) => o.start === it.start && o.id === it.id && !o.used);
        if (same) { same.used = true; Object.assign(same, { tx: it.x, ty: it.y, h: it.h, w: it.w }); tiles.push(same); continue; }
        const parts = old.filter((o) => o.start >= it.start && o.start < it.start + it.len && !o.used);
        const m = tile(it.text, it.w, it.h, colFor(it.id, it.partial), 96);
        m.redraw(it.text, colFor(it.id, it.partial), String(it.id));
        const px = parts.length ? parts.reduce((a, o) => a + o.m.position.x, 0) / parts.length : it.x;
        const py = parts.length ? parts.reduce((a, o) => a + o.m.position.y, 0) / parts.length : it.y + 0.6;
        m.position.set(px, py, 0);
        const fresh = parts.length > 1 && it.id >= 256;
        root.add(m);
        tiles.push({ m, id: it.id, start: it.start, len: it.len, tx: it.x, ty: it.y, h: it.h, w: it.w, pop: fresh ? 1 : 0.4 });
      }
      old.filter((o) => !o.used).forEach((o) => { root.remove(o.m); o.m.geometry.dispose(); o.m.material.map.dispose(); o.m.material.dispose(); });
      tiles.forEach((t) => { t.used = false; });
      info = { chars: [...text].length, bytes: utf8Len(text), tokens: ids.length, n, max: bpe.merges.length, last: n ? bpe.merges[n - 1] : null, bpe, partial: toks.filter((t) => t.partial).length };
      bd.redraw(s);
    }
    function drawBoard(g, w, h, s) {
      panel(g, w, h);
      if (!info) return;
      txt(g, `${info.chars} letters  →  ${info.bytes} bytes  →  `, 30, 64, F(40, true), COL.white);
      g.font = F(40, true); const x2 = 30 + g.measureText(`${info.chars} letters  →  ${info.bytes} bytes  →  `).width;
      txt(g, `${info.tokens} tokens`, x2, 64, F(48, true), COL.hot);
      txt(g, `Tokenizer: ${TOKENIZERS[s.tok].name}. Vocabulary: 256 bytes + ${info.n} merges = ${256 + info.n} tokens.`, 30, 124, F(28), COL.soft);
      if (info.last) {
        const a = tokenText(info.bpe, info.last.a), b = tokenText(info.bpe, info.last.b), c = tokenText(info.bpe, info.last.id);
        txt(g, `Newest merge #${info.last.id}:  "${a}" + "${b}"  →  "${c}"   (a pair seen ${info.last.count} times)`, 30, 184, F(32, true), COL.teal);
      } else txt(g, 'No merges yet: every byte is its own token.', 30, 184, F(32, true), COL.teal);
      if (info.partial) txt(g, `${info.partial} tokens are only pieces of a letter (shown as hex bytes, in orange).`, 30, 240, F(28), COL.red);
      else txt(g, '· marks a space that belongs to the next word.', 30, 240, F(28), COL.soft);
    }
    const onInput = (e) => { if (e.target?.id === 'tokIn') { yours = e.target.value; if (st?.sample === 'yours') key = ''; outKey = ''; } };
    document.addEventListener('input', onInput);

    return {
      update(dt, s) {
        dt = Math.max(0, dt); st = s;
        fit(stage, [L.wall], { x: 0.04, y: -0.1, ny: -0.16 });
        const tall = tallView(stage, 1.9, 0.9);
        if (tall !== wasTall) {
          wasTall = tall; rowW = tall ? 4.6 : ROWW; topY = tall ? 6.5 : 3.05; maxH = tall ? 5.8 : 2.6; key = '';
          wall.scale.set(tall ? 5.2 / (ROWW + 0.6) : 1, tall ? 6.3 / 2.9 : 1, 1); wall.position.y = tall ? 3.55 : 1.95;
        }
        const max = TOKENIZERS[s.tok].bpe.merges.length;
        if (s.auto) { acc += dt * 10; const k = Math.floor(acc); if (k) { acc -= k; s.merges = s.merges >= max ? 0 : Math.min(max, Math.round(s.merges) + k); } }
        const k = `${s.tok}|${s.sample}|${Math.round(s.merges)}|${s.sample === 'yours' ? yours : ''}`;
        if (k !== key) { key = k; rebuild(s); }
        for (const t of tiles) {
          t.m.position.x = approach(t.m.position.x, t.tx, 9, dt); t.m.position.y = approach(t.m.position.y, t.ty, 9, dt);
          t.pop = Math.max(0, t.pop - dt * 1.6);
          t.m.scale.setScalar(1 + 0.35 * t.pop); t.m.position.z = 0.25 * t.pop;
        }
        // the token counter in the side panel
        const out = document.getElementById('tokOut');
        const ok = `${yours}|${s.tok}|${Math.round(s.merges)}`;
        if (out && ok !== outKey) {
          outKey = ok;
          const en = TOKENIZERS.en.bpe, mx = TOKENIZERS.mix.bpe;
          out.innerHTML = `${[...yours].length} letters · ${utf8Len(yours)} bytes · <b>${encode(yours, en, en.merges.length).length}</b> tokens (English-trained) · <b>${encode(yours, mx, mx.merges.length).length}</b> tokens (English + Hindi)`;
        }
      },
      readout(s) {
        if (!info) return '';
        const per = info.tokens / Math.max(1, pretok((s.sample === 'yours' ? yours : SAMPLES[s.sample].text)).filter((w) => /[^\s.,।!?]/u.test(w)).length);
        return `<div class="big">${info.tokens} tokens</div>
          <div class="row"><span>Letters · bytes</span><b>${info.chars} · ${info.bytes}</b></div>
          <div class="row"><span>Tokens per word</span><b>${per.toFixed(1)}</b></div>
          <div class="row"><span>Merges learned</span><b>${info.n} of ${info.max}</b></div>
          <small>Training stops when no pair appears twice.</small>`;
      },
      dispose() { document.removeEventListener('input', onInput); },
    };
  },
};
