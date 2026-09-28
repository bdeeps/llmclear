// Chapter 3: embeddings. Two sets of word vectors:
//  - "Learned from our text": real vectors built when the page loads from which words appear near which
//    (±3 words) in the box's text: positive pointwise mutual information (PPMI) with context smoothing
//    0.75 (Levy, Goldberg & Dagan 2015). One dimension per context word (about 70). Similarity is the
//    cosine in that full space; the 3D picture is a PCA squash that keeps only part of the spread.
//  - "Illustrative": eight hand-placed words on three named axes, so vector arithmetic is easy to see.
//    word2vec (Mikolov et al. 2013) made "king − man + woman ≈ queen" famous; later work (e.g. Nissim,
//    van Noord & van der Goot 2020, "Fair is Better than Sensational") showed such analogies work less
//    cleanly than headlines suggest, partly because the input words are excluded from the answer.
import { THREE, M, sphere, beam, arrow, approach } from '../kit.js';
import { EMB, EMB_GROUP, GROUP_COL, TOY, neighbours, board, panel, txt, F, COL, tile, fit, inReel, hex, tallView } from '../llm.js';

const R = 2.3, CY = 2.3;
const PICKS = ['chai', 'rain', 'delhi', 'june', 'doctor', 'nurse'];
const ANALOGY = {
  kmw: { a: 'king', b: 'man', c: 'woman', name: 'king − man + woman' },
  pbg: { a: 'prince', b: 'boy', c: 'girl', name: 'prince − boy + girl' },
  gwy: { a: 'woman', b: 'girl', c: 'boy', name: 'woman − girl + boy' },
};
const GROUP_NAME = { drink: 'drinks', weather: 'weather', place: 'places', time: 'months', describe: 'hot/cold', person: 'people', pronoun: 'he/she' };

// learned coordinates, scaled into the scene: each axis is stretched by a robust scale and softly
// squashed (tanh) so one far-out word doesn't crush the rest. The picture distorts distances anyway;
// the neighbours listed come from the full space.
const LC = (() => {
  const sc = [0, 1, 2].map((k) => { const a = EMB.coords.map((c) => Math.abs(c[k])).sort((x, y) => x - y); return a[Math.floor(a.length * 0.8)] || 1; });
  const f = (v, k) => Math.tanh(v / (0.9 * sc[k]));
  return EMB.coords.map(([x, y, z]) => [f(x, 0) * R, CY + f(y, 1) * R * 0.85, f(z, 2) * R]);
})();
const TC = Object.fromEntries(Object.entries(TOY.words).map(([w, [x, y, z]]) => [w, [(x - 0.5) * 3.2, CY + (y - 0.5) * 3.2, (z - 0.5) * 3.2]]));

export default {
  id: 'embed',
  short: 'Embeddings',
  title: 'Words as points in space',
  subtitle: 'Each token becomes a long list of numbers. Similar meanings end up close together.',
  view: { pos: [1.2, 3.7, 8.2], target: [0.2, 2.0, 0.4] },
  learn: `<p>A token ID like 283 says nothing about meaning. So the first thing a model does is swap each ID for an <b>embedding</b>: a long list of numbers, a point in a space with hundreds or thousands of directions. GPT-3's embeddings had <b>12,288</b> numbers each.</p>
    <p>Nobody writes these numbers by hand. They are learned from one clue: <b>words that appear in similar company tend to mean similar things</b>. "Chai" and "coffee" both sit next to "hot", "cup" and "drink". So they end up as neighbours.</p>
    <p>The points here are real and tiny: built from this box's short text by counting which words appear within 3 words of each other (a method called PPMI). Pick a word to see its nearest neighbours. With so little text the result is rough, but you will see <b>drinks</b>, <b>places</b> and <b>people</b> drift into groups. We squash about 70 directions into the 3 you can see, so trust the listed neighbours more than the picture.</p>
    <p>In the <b>illustrative</b> set, eight words sit on three hand-chosen axes. Here "king − man + woman" lands exactly on "queen". In real models such <b>vector arithmetic</b> works only roughly and not always: a famous result, but less tidy than headlines suggest.</p>
    <p>Embeddings also soak up <b>bias</b>. In our text the doctor and the engineer are "he", the nurse and the teacher "she", so doctor sits closer to engineer, and nurse to teacher. Real models learn the same kinds of patterns from the web.</p>
    <p class="tip"><b>Try it:</b> click any word, or pick one below, and read its neighbours. Then switch to the illustrative set and run the three sums.</p>`,
  terms: [
    { t: 'Embedding', d: 'A list of numbers that stands for a token: its position in meaning-space.' },
    { t: 'Dimension', d: 'One number in the list, one direction in the space. Real models use thousands.' },
    { t: 'Cosine similarity', d: 'How closely two vectors point the same way: 1 is identical, 0 unrelated.' },
    { t: 'Co-occurrence', d: 'Two words turning up near each other in text.' },
    { t: 'Vector arithmetic', d: 'Adding and subtracting embeddings, as in king − man + woman.' },
  ],
  defaults: { set: 'learned', word: 'chai', analogy: 'kmw', spin: true },
  controls: [
    { key: 'set', type: 'seg', label: 'Which vectors', options: [{ v: 'learned', label: 'Learned from our text' }, { v: 'toy', label: 'Illustrative (hand-made)' }] },
    { key: 'word', type: 'seg', label: 'Pick a word (or click one)', options: PICKS.map((w) => ({ v: w, label: w })) },
    { key: 'analogy', type: 'seg', label: 'Vector sum (illustrative set)', options: Object.entries(ANALOGY).map(([v, a]) => ({ v, label: a.name })) },
    { key: 'spin', type: 'toggle', label: 'Turn slowly' },
  ],
  quiz: [
    { q: 'What is an embedding?', options: ['A picture of a word', 'A list of numbers that places a token in meaning-space', 'A dictionary definition', 'The token ID'], answer: 1, why: 'Each token ID is swapped for a learned vector. Similar tokens get nearby vectors.' },
    { q: 'How do embeddings learn that "chai" is like "coffee"?', options: ['Someone labels them', 'They appear in similar company in text', 'They have similar spellings', 'They are the same length'], answer: 1, why: 'Words used in similar contexts get similar vectors: "you shall know a word by the company it keeps".' },
    { q: 'Does "king − man + woman = queen" always work in real models?', options: ['Yes, exactly', 'Only roughly, and not always', 'Never', 'Only in Hindi'], answer: 1, why: 'It is a real effect but messy. Here it is exact only because the illustrative set was placed by hand.' },
  ],
  reel: [
    { ms: 5600, caption: 'Each word becomes a point in space. Words used in similar ways land close together.', set: { set: 'learned', word: 'chai', spin: false }, spin: 0.8, view: { pos: [2.4, 3.4, 7.2], target: [0, 2.2, 0] } },
  ],

  build({ stage }) {
    const root = new THREE.Group(); stage.root.add(root);
    // frame: a faint cube and three axes
    const frame = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(2 * R + 0.4, 2 * R * 0.85 + 0.4, 2 * R + 0.4)), new THREE.LineBasicMaterial({ color: 0x3a4658, transparent: true, opacity: 0.6 }));
    frame.position.y = CY; root.add(frame);
    const pts = {}, tags = {};
    const add = (w, col, key) => {
      const m = sphere(0.12, M.plastic(col, { emissive: col, emissiveIntensity: 0.25 })); m.userData.word = w; m.userData.set = key; root.add(m);
      const t = tile(w, 0.26 + 0.07 * w.length, 0.24, hex(col), 72); root.add(t);
      return { m, t };
    };
    EMB.words.forEach((w, i) => { pts['L:' + w] = add(w, GROUP_COL[EMB_GROUP[w]], 'learned'); pts['L:' + w].home = LC[i]; });
    Object.keys(TOY.words).forEach((w) => { const c = TOY.words[w][1] ? 0xf5a3d0 : 0x8ef0ff; pts['T:' + w] = add(w, c, 'toy'); pts['T:' + w].home = TC[w]; });
    stage.pickables = Object.values(pts).map((p) => p.m);
    const axisTiles = TOY.axes.map((a, k) => { const t = tile(a + ' →', 1.0, 0.3, COL.hot, 72); root.add(t); return t; });
    axisTiles[0].position.set(0, CY - 1.95, 1.9); axisTiles[1].position.set(-2.1, CY, 1.9); axisTiles[2].position.set(-2.1, CY - 1.95, 0);
    // neighbour lines and analogy arrows
    const nb = [0, 1, 2].map(() => { const b = beam([0, 0, 0], [0, 1, 0], 0.025, M.glow(0xffd166)); root.add(b); return b; });
    const arrows = [arrow(0xff7a59, 1, 0.22, 0.035), arrow(0x7be08c, 1, 0.22, 0.035)]; arrows.forEach((a) => root.add(a));
    const ghost = sphere(0.2, M.ghost(0xffd166, 0.5)); root.add(ghost);
    const bd = board(1400, 300, 6.4, 1.37, (g, w, h, s) => s && drawBoard(g, w, h, s)); stage.root.add(bd.mesh);
    let st = null, key = '', res = null;
    const up = new THREE.Vector3(0, 1, 0);
    const setBeam = (b, A, B, r) => { const a = new THREE.Vector3(...A), c = new THREE.Vector3(...B); b.position.copy(a).add(c).multiplyScalar(0.5); b.scale.set(r, a.distanceTo(c), r); b.quaternion.setFromUnitVectors(up, c.clone().sub(a).normalize()); };
    const setArrow = (ar, A, B) => { const a = new THREE.Vector3(...A), c = new THREE.Vector3(...B); ar.position.copy(a); ar.quaternion.setFromUnitVectors(up, c.clone().sub(a).normalize()); ar.set(a.distanceTo(c)); };

    function compute(s) {
      if (s.set === 'learned') return { kind: 'learned', word: s.word, list: neighbours(s.word, 3) };
      const A = ANALOGY[s.analogy], v = TOY.words[A.a].map((x, k) => x - TOY.words[A.b][k] + TOY.words[A.c][k]);
      const ranked = Object.entries(TOY.words).filter(([w]) => ![A.a, A.b, A.c].includes(w)).map(([w, u]) => ({ w, d: Math.hypot(...u.map((x, k) => x - v[k])) })).sort((a, b) => a.d - b.d);
      return { kind: 'toy', A, v, best: ranked[0] };
    }
    function drawBoard(g, w, h, s) {
      panel(g, w, h);
      if (!res) return;
      if (res.kind === 'learned') {
        txt(g, `Nearest to "${res.word}"  (cosine similarity, all ${EMB.dims} dimensions)`, 30, 60, F(34, true), COL.white);
        res.list.forEach((o, i) => {
          const y = 120 + i * 58, x = 30;
          txt(g, o.w, x, y, F(34, true), hex(GROUP_COL[EMB_GROUP[o.w]]));
          g.fillStyle = 'rgba(255,209,102,.8)'; g.fillRect(260, y - 26, Math.max(4, o.s * 800), 28);
          txt(g, o.s.toFixed(2), 270 + Math.max(4, o.s * 800), y, F(30), COL.hot);
        });
      } else {
        txt(g, `${res.A.name}  =  (${res.v.join(', ')})`, 30, 70, F(40, true), COL.white);
        txt(g, `Nearest word, leaving out the three inputs:  ${res.best.w}`, 30, 140, F(36, true), COL.hot);
        txt(g, `Axes: ${TOY.axes.join(', ')}. Hand-placed, so the sum is exact. Real models are messier.`, 30, 210, F(28), COL.soft);
      }
    }

    return {
      pick(o) { const w = o.userData.word; if (!w || !st) return; if (o.userData.set === 'learned') { st.set = 'learned'; st.word = w; } else { st.set = 'toy'; } key = ''; },
      update(dt, s, time) {
        dt = Math.max(0, dt); st = s;
        fit(stage, [], { x: 0.06, y: -0.06, ny: -0.12 });
        const tall = tallView(stage, 2.0, 0.88);
        root.position.y = tall ? 2.3 : 0;
        bd.mesh.scale.setScalar(tall ? 0.72 : 1);
        if (tall) { bd.mesh.position.set(0, 1.1, 2.2); bd.mesh.rotation.x = -0.3; } else { bd.mesh.position.set(0, 0.5, 2.7); bd.mesh.rotation.x = -1.05; }
        stage.controls.autoRotate = s.spin && !inReel(); stage.controls.autoRotateSpeed = 0.6;
        const k = `${s.set}|${s.word}|${s.analogy}`;
        if (k !== key) { key = k; res = compute(s); bd.redraw(s); }
        const learned = s.set === 'learned';
        for (const [id, p] of Object.entries(pts)) {
          const on = id.startsWith('L:') === learned;
          p.m.visible = p.t.visible = on;
          if (!on) continue;
          p.m.position.set(...p.home);
          const w = id.slice(2);
          const hi = learned ? (w === s.word ? 1 : res.list?.some((o) => o.w === w) ? 0.6 : 0) : [res.A.a, res.A.b, res.A.c, res.best.w].includes(w) ? 1 : 0;
          p.m.scale.setScalar(1 + hi * 0.6);
          p.t.position.set(p.home[0], p.home[1] + 0.26, p.home[2]);
          p.t.quaternion.copy(stage.camera.quaternion);
          p.t.material.opacity = learned ? (hi ? 1 : 0.55) : 1;
        }
        frame.scale.set(1, learned ? 1 : 0.9, 1);
        axisTiles.forEach((t) => { t.visible = !learned; t.quaternion.copy(stage.camera.quaternion); });
        nb.forEach((b, i) => {
          b.visible = learned && !!res.list?.[i];
          if (b.visible) { const o = res.list[i]; setBeam(b, pts['L:' + s.word].home, pts['L:' + o.w].home, 0.4 + o.s * 2.5); }
        });
        // analogy: arrow from A by −B, then +C, landing on the answer
        arrows.forEach((a) => (a.visible = !learned)); ghost.visible = !learned;
        if (!learned) {
          const A = res.A, P = (w) => TC[w], a = P(A.a), mid = a.map((x, k) => x - P(A.b)[k] + TC.man[k]);
          const end = mid.map((x, k) => x + P(A.c)[k] - TC.man[k]);
          setArrow(arrows[0], a, mid); setArrow(arrows[1], mid, end);
          ghost.position.set(...end); ghost.scale.setScalar(1 + 0.15 * Math.sin(time * 4));
        }
      },
      readout(s) {
        if (!res) return '';
        if (res.kind === 'learned') return `<div class="big">"${res.word}" is near ${res.list.map((o) => o.w).join(', ')}</div>
          <div class="row"><span>Dimensions (context words)</span><b>${EMB.dims}</b></div>
          <div class="row"><span>Word pairs counted</span><b>${EMB.pairs}</b></div>
          <div class="row"><span>Spread shown in 3D</span><b>${Math.round(EMB.kept * 100)}%</b></div>
          <small>Colours: ${Object.entries(GROUP_NAME).map(([g, n]) => `<span style="color:${hex(GROUP_COL[g])}">${n}</span>`).join(' · ')}</small>`;
        return `<div class="big">${res.A.name} ≈ ${res.best.w}</div>
          <div class="row"><span>Result vector</span><b>(${res.v.join(', ')})</b></div>
          <small>Axes: ${TOY.axes.join(', ')}. Illustrative, hand-placed words.</small>`;
      },
    };
  },
};
