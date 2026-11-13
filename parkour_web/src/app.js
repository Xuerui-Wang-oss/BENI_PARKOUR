// The page: a three-step course editor (how many obstacles, which ones, how large and how far apart), then the run,
// seen from behind the robot (or, with C, from its own camera) with the depth image the policy gets in a corner.

import { View } from './view/scene.js';
import {
  COUNT_RANGE, GAP_RANGE, GOAL_MARGIN, KINDS, KIND_INFO, VALUE_RANGE, buildCourse, defaultItem, formatValue, randomItems,
} from './course.js';

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, children = []) => {
  const node = Object.assign(document.createElement(tag), props);
  for (const child of [].concat(children)) node.append(child);
  return node;
};

const OUTCOMES = {
  goal_reached: 'Goal reached', fell: 'Fell over', in_pit: 'Fell into a pit', stuck: 'Got stuck',
  body_contact: 'Body on the ground', out_of_time: 'Out of time', time_out: 'Out of time', out_of_lane: 'Left the lane',
};
const MODES = { drive: 'Driving', crouch: 'Crouching', prepare: 'Preparing to jump', jump: 'Jumping' };

const state = {
  step: 1, // 1 count, 2 place, 3 sizes, 'run'
  items: Array(5).fill(null), // One entry per slot: {kind, value, gap} or null.
  selected: 0, // Slot shown in step 3.
  picked: null, // Kind picked from the palette by a click, to place with a click on a slot.
  hover: -1, // Slot under a drag.
  error: '',
  failure: '', // An error of the simulation worker.
  loaded: 0, // Fraction of the downloads done.
  ready: false,
  threads: 1,
  camera: 'chase', // 'chase' (behind the robot) or 'first' (the robot's own camera).
  run: 0, // Number of the current run; frames of an earlier run are ignored.
  waiting: false, // The robot waits on the grid for the start lights.
  lightTimers: [], // Timers of the start lights.
  paused: false,
  result: null,
};
let course = buildCourse(state.items);

const view = new View($('view'));
if (import.meta.env.DEV) window.view = view; // For inspection from the console of the dev server.
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });

// ---- Simulation worker ------------------------------------------------------------------------------------------

worker.onmessage = ({ data }) => {
  if (data.type === 'progress') {
    state.loaded = data.total ? data.loaded / data.total : 0;
    updateLoading();
  } else if (data.type === 'ready') {
    state.ready = true;
    state.threads = data.threads;
    renderPanel();
  } else if (data.type === 'frame' && data.run === state.run) {
    frames.push({ time: data.time, bodies: data.bodies });
    if (frames.length > 80) frames.shift();
    updateHud(data.status);
    if (data.depth) drawDepth(data.depth);
  } else if (data.type === 'done' && data.run === state.run) {
    state.result = data;
    renderResult();
    if (data.outcome.includes('goal_reached')) fireworksUntil = performance.now() + 6000;
  } else if (data.type === 'error') {
    console.error(data.message);
    state.failure = `The simulator stopped: ${data.message.split('\n')[0]}`;
    renderPanel();
  }
};
worker.postMessage({ type: 'init' });
view.loadRobot(import.meta.env.BASE_URL);

// ---- Course editing ---------------------------------------------------------------------------------------------

function setCourse() {
  course = buildCourse(state.items);
  view.setCourse(course);
  updateSlots();
}

function updateSlots() {
  const pad = 0.25;
  view.setSlots(course.slots.map((slot, i) => ({
    x0: slot.x0 - pad, x1: slot.x1 + pad, filled: Boolean(slot.item),
    active: state.step === 3 ? i === state.selected : i === state.hover,
  })));
}

function frameCamera() {
  if (state.step === 3) {
    const slot = course.slots[state.selected];
    view.frameLane(slot.x0 - 2.2, slot.x1 + 2.2);
  } else {
    view.frameLane(-1, course.goalX + 1);
  }
}

function setCount(count) {
  state.items = Array.from({ length: count }, (_, i) => state.items[i] ?? null);
  state.selected = Math.min(state.selected, count - 1);
  refresh();
}

function place(slot, kind) {
  const gap = state.items[slot]?.gap ?? 3;
  state.items[slot] = { ...defaultItem(kind), gap };
  state.error = '';
  refresh();
}

function goTo(step) {
  course = buildCourse(state.items);
  if (step === 3 && !course.complete) {
    const empty = state.items.filter((item) => !item).length;
    state.error = `${empty} slot${empty > 1 ? 's are' : ' is'} still empty.`;
    renderPanel();
    return;
  }
  state.step = step;
  state.error = '';
  state.picked = null;
  refresh();
}

function refresh() {
  setCourse();
  frameCamera();
  renderPanel();
}

/** The slot whose frame is nearest to the point (clientX, clientY), or -1 if none is near. */
function slotAt(clientX, clientY) {
  const box = $('view').getBoundingClientRect();
  let best = -1;
  let bestDistance = 90; // Pixels.
  course.slots.forEach((slot, i) => {
    const p = view.project((slot.x0 + slot.x1) / 2, 0, 0.35);
    const distance = Math.hypot(p.x - (clientX - box.left), p.y - (clientY - box.top));
    if (distance < bestDistance) {
      best = i;
      bestDistance = distance;
    }
  });
  return best;
}

const stage = $('stage');
stage.addEventListener('dragover', (event) => {
  if (state.step !== 2) return;
  event.preventDefault();
  const slot = slotAt(event.clientX, event.clientY);
  if (slot !== state.hover) {
    state.hover = slot;
    updateSlots();
  }
});
stage.addEventListener('dragleave', () => {
  state.hover = -1;
  updateSlots();
});
stage.addEventListener('drop', (event) => {
  event.preventDefault();
  const slot = slotAt(event.clientX, event.clientY);
  const kind = event.dataTransfer.getData('text/plain');
  state.hover = -1;
  if (state.step === 2 && slot >= 0 && KINDS.includes(kind)) place(slot, kind);
  else updateSlots();
});
stage.addEventListener('click', (event) => {
  const slot = slotAt(event.clientX, event.clientY);
  if (slot < 0) return;
  if (state.step === 2) {
    if (state.picked) place(slot, state.picked);
    else if (state.items[slot]) {
      state.items[slot] = null; // A click on a filled slot empties it.
      refresh();
    }
  } else if (state.step === 3) {
    state.selected = slot;
    refresh();
  }
});

// ---- Panel ------------------------------------------------------------------------------------------------------

function loadingNote() {
  if (state.failure) return el('div', { className: 'note bad', textContent: state.failure });
  if (state.ready) return el('div', { className: 'loading', textContent: 'Simulator ready' });
  return el('div', { className: 'loading', id: 'loading' }, [
    el('span', { textContent: `Loading the simulator and the policies… ${Math.round(100 * state.loaded)}%` }),
    el('div', { className: 'bar' }, el('i', { style: `width:${100 * state.loaded}%` })),
  ]);
}

function updateLoading() {
  const node = $('loading');
  if (node) node.replaceWith(loadingNote());
}

function stepsHeader(step) {
  const names = ['Count', 'Place', 'Tune'];
  return el('div', { className: 'steps' }, names.flatMap((name, i) => [
    el(i + 1 === step ? 'b' : 'span', { textContent: `${i + 1} ${name}` }), i < 2 ? el('span', { textContent: '›' }) : '',
  ]));
}

const button = (text, onclick, className = '') => el('button', { textContent: text, onclick, className });

function renderPanel() {
  const panel = $('panel');
  panel.replaceChildren();
  const add = (...nodes) => panel.append(...nodes);
  if (state.step === 'run') {
    add(el('h2', { textContent: 'Running' }),
      el('p', { textContent: 'The robot gets no map of your course. A learned policy reads the depth image and decides how fast to drive, how low to crouch and when to jump.' }),
      el('p', { textContent: 'It crosses about three of four courses built at random from these obstacles, so a fall here and there is expected.' }),
      el('div', { className: 'grow' }),
      el('div', { className: 'row' }, [button(state.paused ? 'Resume' : 'Pause', togglePause), button('Restart', startRun)]),
      button('Edit course', () => stopRun(3), 'ghost'));
    return;
  }
  add(stepsHeader(state.step));
  if (state.step === 1) {
    add(el('h2', { textContent: 'How many obstacles?' }),
      el('p', { textContent: 'The lane gets one empty slot per obstacle.' }),
      el('div', { className: 'row count' }, [4, 5, 6].filter((n) => n >= COUNT_RANGE[0] && n <= COUNT_RANGE[1])
        .map((n) => button(String(n), () => setCount(n), n === state.items.length ? 'on' : ''))),
      el('p', { textContent: 'Or let chance build the whole course:' }),
      button('Random course', () => {
        state.items = randomItems(state.items.length);
        state.selected = 0;
        goTo(3);
      }),
      el('div', { className: 'grow' }), loadingNote(),
      button('Next', () => goTo(2), 'primary'));
  } else if (state.step === 2) {
    add(el('h2', { textContent: 'Place the obstacles' }),
      el('p', { textContent: 'Drag an obstacle onto a slot of the lane, or click one here and then a slot. Click a filled slot to empty it.' }));
    for (const kind of KINDS) {
      const info = KIND_INFO[kind];
      const [low, high] = VALUE_RANGE[kind];
      const chip = el('div', { className: `chip${state.picked === kind ? ' on' : ''}`, draggable: true }, [
        el('span', { className: 'swatch', style: `background:${info.color}` }),
        el('span', { className: 'name', textContent: info.label }),
        el('span', { className: 'range', textContent: `${info.size.toLowerCase()} ${formatValue(kind, low)} – ${formatValue(kind, high)}` }),
      ]);
      chip.addEventListener('dragstart', (event) => event.dataTransfer.setData('text/plain', kind));
      chip.addEventListener('click', () => {
        state.picked = state.picked === kind ? null : kind;
        renderPanel();
      });
      add(chip);
    }
    add(el('div', { className: 'note bad', textContent: state.error }), el('div', { className: 'grow' }), loadingNote(),
      el('div', { className: 'row' }, [button('Back', () => goTo(1)), button('Next', () => goTo(3), 'primary')]));
  } else {
    add(el('h2', { textContent: 'Sizes and gaps' }),
      el('p', { textContent: 'Every slider covers the range the policy was trained on.' }));
    state.items.forEach((item, i) => {
      const info = KIND_INFO[item.kind];
      const on = i === state.selected;
      const card = el('div', { className: `item${on ? ' on' : ''}` }, el('div', { className: 'head' }, [
        el('span', { className: 'swatch', style: `background:${info.color}` }),
        el('span', { textContent: `${i + 1}  ${info.label}` }),
        el('span', { className: 'value', textContent: formatValue(item.kind, item.value) }),
      ]));
      card.addEventListener('click', () => {
        if (state.selected === i) return;
        state.selected = i;
        refresh();
      });
      if (on) {
        const slider = (label, value, [low, high], step, format, set) => {
          const text = el('span', { textContent: format(value) });
          const input = el('input', { type: 'range', min: low, max: high, step, value });
          input.addEventListener('input', () => {
            set(Number(input.value));
            text.textContent = format(Number(input.value));
            card.querySelector('.value').textContent = formatValue(item.kind, item.value);
            setCourse();
            frameCamera();
            updateLength();
          });
          card.append(el('label', {}, [el('span', { textContent: label }), text]), input);
        };
        slider(info.size, item.value, VALUE_RANGE[item.kind], info.step, (v) => formatValue(item.kind, v), (v) => { item.value = v; });
        if (i) slider('Gap to the obstacle before', item.gap, GAP_RANGE, 0.1, (v) => `${v.toFixed(1)} m`, (v) => { item.gap = v; });
      }
      add(card);
    });
    add(el('div', { className: 'note', id: 'length' }), el('div', { className: 'note bad', textContent: state.error }),
      el('div', { className: 'grow' }), loadingNote(),
      el('div', { className: 'row' }, [button('Back', () => goTo(2)),
        button(state.ready ? 'Start' : 'Loading…', () => state.ready && startRun(), 'primary')]));
    updateLength();
  }
}

function updateLength() {
  const node = $('length');
  if (!node) return;
  node.className = `note${course.beyondTraining ? ' warn' : ''}`;
  node.textContent = `Start to goal: ${course.goalX.toFixed(1)} m (goal ${GOAL_MARGIN} m past the last obstacle).`
    + (course.beyondTraining ? ' This is longer than any course the policy trained on; it still runs.' : '');
}

// ---- Run --------------------------------------------------------------------------------------------------------

function startRun() {
  state.step = 'run';
  state.result = null;
  state.paused = false;
  state.run += 1;
  frames.length = 0;
  playTime = null;
  fireworksUntil = 0;
  setCourse();
  worker.postMessage({
    type: 'start', run: state.run, hold: true,
    course: { obstacles: course.obstacles, length: course.length, goalX: course.goalX, spawn: course.spawn },
  });
  view.setMode(state.camera);
  startSequence();
  depthContext.clearRect(0, 0, 80, 60); // Until the policy has looked at its first image.
  for (const id of ['hud', 'gauges', 'depth-box', 'keys']) $(id).hidden = false;
  $('result').hidden = true;
  $('labels').replaceChildren();
  // Obstacles and goal along the progress bar.
  const span = course.goalX + 1;
  $('track-marks').replaceChildren(
    ...course.slots.map((slot) => el('i', { style: `left:${100 * slot.x0 / span}%;width:${Math.max(100 * (slot.x1 - slot.x0) / span, 0.6)}%;background:${KIND_INFO[slot.item.kind].color}` })),
    el('i', { className: 'goal', style: `left:${100 * course.goalX / span}%` }));
  renderPanel();
}

/**
 * The start of a race: the robot waits on the grid while the five red lights come on one by one, and goes when
 * they all go out.
 */
function startSequence() {
  for (const timer of state.lightTimers) clearTimeout(timer);
  const run = state.run;
  const at = (ms, action) => state.lightTimers.push(setTimeout(() => state.run === run && action(), ms));
  state.lightTimers = [];
  state.waiting = true;
  view.setLights(0);
  for (let count = 1; count <= 5; count++) at(180 * count, () => view.setLights(count));
  at(1400, () => {
    view.setLights(0);
    state.waiting = false;
    worker.postMessage({ type: 'pause', paused: state.paused });
  });
}

function stopRun(step) {
  worker.postMessage({ type: 'stop' });
  state.run += 1;
  state.waiting = false;
  frames.length = 0;
  fireworksUntil = 0;
  state.step = step;
  view.setMode('editor');
  view.setLights(5);
  view.robot.visible = false;
  for (const id of ['hud', 'gauges', 'depth-box', 'keys', 'result']) $(id).hidden = true;
  refresh();
}

function togglePause() {
  state.paused = !state.paused;
  // While the start lights are counting, the robot waits anyway; the choice takes effect when they go out.
  if (!state.waiting) worker.postMessage({ type: 'pause', paused: state.paused });
  renderPanel();
}

function toggleCamera() {
  state.camera = state.camera === 'first' ? 'chase' : 'first';
  view.setMode(state.camera);
}

// ---- Smooth playback --------------------------------------------------------------------------------------------
//
// The worker sends a frame after every control step, but not at even intervals: every fifth step waits for the depth
// encoder. Drawing the newest frame would make the robot stutter, so the page draws the simulation a little in the
// past, at a time that advances evenly, and blends the two frames around that time.

const frames = []; // {time, bodies} of the current run, oldest first.
const PLAYBACK_DELAY = 0.08; // Seconds behind the newest frame.
let playTime = null; // Simulated time being drawn.
let lastWall = performance.now();

function blendBodies(a, b, t) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i += 7) {
    for (let k = 0; k < 3; k++) out[i + k] = a[i + k] + t * (b[i + k] - a[i + k]);
    // Quaternions: blend along the shorter way and renormalise.
    let dot = 0;
    for (let k = 3; k < 7; k++) dot += a[i + k] * b[i + k];
    const sign = dot < 0 ? -1 : 1;
    let norm = 0;
    for (let k = 3; k < 7; k++) {
      out[i + k] = a[i + k] + t * (sign * b[i + k] - a[i + k]);
      norm += out[i + k] ** 2;
    }
    norm = Math.sqrt(norm);
    for (let k = 3; k < 7; k++) out[i + k] /= norm;
  }
  return out;
}

function playback() {
  const now = performance.now();
  const dt = Math.min((now - lastWall) / 1000, 0.1);
  lastWall = now;
  if (!frames.length) return;
  const newest = frames.at(-1).time;
  const target = Math.max(newest - PLAYBACK_DELAY, frames[0].time);
  if (playTime === null || Math.abs(target - playTime) > 0.4) {
    playTime = target;
  } else {
    // Advance at the speed of the wall clock, a little faster or slower to stay PLAYBACK_DELAY behind.
    const speed = Math.min(Math.max(1 + 8 * (target - playTime), 0), 2);
    playTime = Math.min(playTime + dt * speed, newest);
  }
  let i = frames.length - 1;
  while (i > 0 && frames[i - 1].time >= playTime) i -= 1;
  const after = frames[i];
  const before = frames[Math.max(i - 1, 0)];
  const span = after.time - before.time;
  const t = span > 0 ? Math.min(Math.max((playTime - before.time) / span, 0), 1) : 1;
  view.setFrame(blendBodies(before.bodies, after.bodies, t));
}

// ---- Fireworks --------------------------------------------------------------------------------------------------

const fireworksCanvas = $('fireworks');
const fireworksContext = fireworksCanvas.getContext('2d');
let fireworksUntil = 0; // New bursts start until this time (ms).
let nextBurst = 0;
let sparks = [];

function drawFireworks(dt) {
  const now = performance.now();
  if (now >= fireworksUntil && !sparks.length) {
    if (fireworksCanvas.width) fireworksCanvas.width = 0; // Nothing to draw: clear.
    return;
  }
  const { clientWidth: width, clientHeight: height } = stage;
  if (fireworksCanvas.width !== width || fireworksCanvas.height !== height) {
    fireworksCanvas.width = width;
    fireworksCanvas.height = height;
  }
  if (now < fireworksUntil && now >= nextBurst) {
    nextBurst = now + 220 + 320 * Math.random();
    const x = width * (0.12 + 0.76 * Math.random());
    const y = height * (0.12 + 0.42 * Math.random());
    const hue = 360 * Math.random();
    for (let i = 0; i < 90; i++) {
      const angle = 2 * Math.PI * Math.random();
      const speed = 60 + 220 * Math.random();
      sparks.push({ x, y, vx: speed * Math.cos(angle), vy: speed * Math.sin(angle), life: 1, fade: 0.6 + 0.5 * Math.random(), hue: hue + 40 * Math.random() });
    }
  }
  fireworksContext.clearRect(0, 0, width, height);
  fireworksContext.globalCompositeOperation = 'lighter';
  for (const spark of sparks) {
    spark.x += spark.vx * dt;
    spark.y += spark.vy * dt;
    spark.vy += 160 * dt; // Gravity, in pixels.
    spark.vx *= 1 - 1.2 * dt;
    spark.life -= spark.fade * dt;
    fireworksContext.fillStyle = `hsla(${spark.hue}, 95%, 62%, ${Math.max(spark.life, 0)})`;
    fireworksContext.beginPath();
    fireworksContext.arc(spark.x, spark.y, 3.4, 0, 2 * Math.PI);
    fireworksContext.fill();
  }
  sparks = sparks.filter((spark) => spark.life > 0);
}

let hudTime = 0;
/** Show the status of a frame; the text is rewritten ten times a second at most. */
function updateHud(status) {
  const span = course.goalX + 1;
  $('track-robot').style.left = `${100 * Math.min(Math.max(status.x / span, 0), 1)}%`;
  // The gauges: the latest action of the high-level policy. Speed runs from 0 to 1.5 m/s, the body height from
  // the lowest crouch (6.7 cm) to standing (11 cm); the lamp is amber while the policy presses the jump trigger
  // and lit while the jump policy drives.
  const { policy } = status;
  const jumping = status.mode === 'jump' || status.mode === 'prepare';
  $('gauge-speed').style.width = `${100 * Math.min(Math.max(policy.speed / 1.5, 0), 1)}%`;
  $('gauge-height').style.width = `${100 * Math.min(Math.max((policy.height - 0.067) / (0.1101 - 0.067), 0), 1)}%`;
  $('gauge-jump').className = `lamp${jumping ? ' jumping' : policy.trigger || status.trigger ? ' pressed' : ''}`;
  if (performance.now() - hudTime < 100) return;
  hudTime = performance.now();
  $('value-speed').textContent = `${policy.speed.toFixed(2)} m/s`;
  $('value-height').textContent = `${(100 * policy.height).toFixed(1)} cm`;
  $('value-jump').textContent = jumping ? 'jumping' : policy.trigger || status.trigger ? 'trigger' : 'no';
  const slow = status.realTime < 0.9;
  $('readouts').replaceChildren(
    el('span', {}, [el('b', { textContent: MODES[status.mode] })]),
    el('span', { innerHTML: `Obstacles <b>${status.passed} / ${course.obstacles.length}</b>` }),
    el('span', { innerHTML: `Time <b>${status.time.toFixed(1)} s</b>` }),
    slow ? el('span', { className: 'slow', innerHTML: `Running at <b>${status.realTime.toFixed(2)}×</b> real time` }) : '');
}

const depthContext = $('depth').getContext('2d');
const depthPixels = depthContext.createImageData(80, 60);
/** Draw the depth image: near is bright, far is dark, no return (sky) is black. */
function drawDepth(depth) {
  const far = Math.log1p(9);
  for (let i = 0; i < depth.length; i++) {
    const near = depth[i] > 0 ? 1 - Math.min(Math.log1p(depth[i]) / far, 1) : 0;
    const lit = depth[i] > 0 ? 0.18 + 0.82 * near : 0;
    depthPixels.data.set([255 * lit * (0.55 + 0.45 * near), 255 * lit, 255 * lit * (0.75 - 0.45 * near), 255], 4 * i);
  }
  depthContext.putImageData(depthPixels, 0, 0);
}

function renderResult() {
  const { outcome, passed, time } = state.result;
  const good = outcome.includes('goal_reached');
  const title = OUTCOMES[outcome.split('+').find((name) => name in OUTCOMES)] ?? 'Episode over';
  const box = $('result');
  box.replaceChildren(
    el('h2', { className: good ? 'good' : '', textContent: good ? 'Good!' : title }),
    el('div', { className: 'stats', textContent: `${good ? 'Goal reached · ' : ''}${passed} of ${course.obstacles.length} obstacles passed in ${time.toFixed(1)} s` }),
    el('div', { className: 'row' }, [button('Run again', startRun, 'primary'), button('Edit course', () => stopRun(3))]),
    button('Random course', () => {
      state.items = randomItems(state.items.length);
      state.selected = 0;
      startRun();
    }, 'ghost'));
  box.hidden = false;
}

window.addEventListener('keydown', (event) => {
  if (state.step !== 'run' || event.target instanceof HTMLInputElement) return;
  if (event.code === 'KeyC') toggleCamera();
  else if (event.code === 'Space') {
    event.preventDefault();
    togglePause();
  } else if (event.code === 'KeyR') startRun();
  else if (event.code === 'KeyE') stopRun(3);
});

// ---- Frame loop -------------------------------------------------------------------------------------------------

/** Labels over the slots while editing: number and content. */
function updateLabels() {
  const labels = $('labels');
  if (state.step === 'run') return;
  course.slots.forEach((slot, i) => {
    const node = labels.children[i] ?? labels.appendChild(el('div'));
    const p = view.project((slot.x0 + slot.x1) / 2, 0, 0.85);
    const on = state.step === 3 && i === state.selected;
    node.className = `label${slot.item ? '' : ' empty'}${on ? ' on' : ''}`;
    node.textContent = slot.item ? `${i + 1} · ${KIND_INFO[slot.item.kind].label} ${formatValue(slot.item.kind, slot.item.value)}` : `${i + 1} · empty`;
    node.style.left = `${p.x}px`;
    node.style.top = `${p.y}px`;
    node.style.display = p.visible ? '' : 'none';
  });
  while (labels.children.length > course.slots.length) labels.lastChild.remove();
}

// The view refits itself when the stage changes size; the editor camera then frames the lane anew.
view.onResize = () => {
  if (state.step !== 'run') frameCamera();
};

let lastLoop = performance.now();
function loop() {
  const now = performance.now();
  const dt = Math.min((now - lastLoop) / 1000, 0.1);
  lastLoop = now;
  updateLabels();
  if (state.step === 'run') playback();
  view.render();
  drawFireworks(dt);
  requestAnimationFrame(loop);
}

setCourse();
view.frameLane(-1, course.goalX + 1, true);
renderPanel();
loop();
