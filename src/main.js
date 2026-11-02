import loadMujoco from '@mujoco/mujoco';
import * as ort from 'onnxruntime-web/wasm';
import { TERRAINS, describe } from './terrain.js';
import { RaiwaySim } from './sim.js';
import { Viewer } from './render.js';
import { Joystick } from './joystick.js';
import './style.css';

const ASSETS = import.meta.env.BASE_URL + 'model/';
const MESHES = ['raiway_TORSO_new', 'raiway_L_THIGH', 'raiway_L_SHANK', 'raiway_L_WHEEL', 'raiway_R_THIGH', 'raiway_R_SHANK', 'raiway_R_WHEEL'];
const ACCEL = { vx: 2.0, wz: 4.0 };
const $ = (id) => document.getElementById(id);

async function fetchBytes(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

function status(text) {
  $('loading').textContent = text;
}

async function main() {
  status('Loading MuJoCo…');
  const mj = await loadMujoco();
  status('Loading robot and policy…');
  ort.env.wasm.numThreads = 1;
  const [manifest, xml, ...meshes] = await Promise.all([
    fetch(ASSETS + 'policies.json').then((r) => r.json()),
    fetchBytes(ASSETS + 'raiway.xml'),
    ...MESHES.map((n) => fetchBytes(`${ASSETS}meshes/${n}.STL`)),
  ]);
  const policies = new Map();
  const loadPolicy = (name) => {
    if (!policies.has(name)) {
      const dir = `${ASSETS}policies/${name}/`;
      policies.set(name, Promise.all([fetch(dir + 'config.json').then((r) => r.json()), fetchBytes(dir + 'policy.onnx')])
        .then(async ([cfg, onnx]) => ({ name, cfg, session: await ort.InferenceSession.create(onnx, { executionProviders: ['wasm'] }) }))
        .catch((e) => { policies.delete(name); throw e; }));
    }
    return policies.get(name);
  };
  for (const p of manifest.policies) $('policy').add(new Option(p.label, p.name, false, p.name === manifest.default));
  let { cfg, session } = await loadPolicy(manifest.default);
  const vfs = new mj.MjVFS();
  vfs.addBuffer('raiway.xml', xml);
  MESHES.forEach((n, i) => vfs.addBuffer(`meshes/${n}.STL`, meshes[i]));

  const terrains = {};
  const level = () => Number($('level').value);
  const getTerrain = (name) => (terrains[`${name}:${level()}`] ??= TERRAINS[name](level()));
  const sim = new RaiwaySim(mj, vfs, cfg, session, ort, getTerrain($('terrain').value));
  sim.reset();
  const viewer = new Viewer($('view'), mj, sim);
  viewer.setTerrain(sim.terrain);
  $('loading').remove();
  $('checkpoint').textContent = cfg.checkpoint;

  const JOINT_NAMES = ['L hip', 'L knee', 'L wheel', 'R hip', 'R knee', 'R wheel'];
  const JOINT_COLORS = ['#2463eb', '#d9482b', '#1f9d55', '#2463eb', '#d9482b', '#1f9d55'];
  const jointCells = JOINT_NAMES.map((name, j) => {
    const tr = $('joints').tBodies[0].insertRow();
    const label = tr.insertCell();
    label.textContent = name;
    label.style.borderLeft = `3px ${j < 3 ? 'solid' : 'dashed'} ${JOINT_COLORS[j]}`;
    return [tr.insertCell(), tr.insertCell()];
  });
  const updateJoints = () => {
    const qd = sim.data.qvel, tau = sim.data.qfrc_applied;
    jointCells.forEach(([vel, trq], j) => {
      vel.textContent = qd[6 + j].toFixed(1);
      trq.textContent = tau[6 + j].toFixed(0);
      trq.style.setProperty('--w', `${(100 * Math.abs(tau[6 + j]) / cfg.torque_limit).toFixed(0)}%`);
    });
  };

  const PLOT_SECONDS = 5;
  const histLen = Math.round(PLOT_SECONDS / cfg.control_dt);
  const hist = { vel: JOINT_NAMES.map(() => new Float32Array(histLen)), trq: JOINT_NAMES.map(() => new Float32Array(histLen)), head: 0, count: 0 };
  const recordJoints = () => {
    const qd = sim.data.qvel, tau = sim.data.qfrc_applied;
    for (let j = 0; j < 6; j++) {
      hist.vel[j][hist.head] = qd[6 + j];
      hist.trq[j][hist.head] = tau[6 + j];
    }
    hist.head = (hist.head + 1) % histLen;
    hist.count = Math.min(hist.count + 1, histLen);
  };
  const clearJoints = () => { hist.head = hist.count = 0; };
  const drawPlot = (canvas, rows, yMax, unit) => {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
    const g = canvas.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    g.strokeStyle = 'rgba(20, 30, 45, 0.25)';
    g.lineWidth = 1;
    g.beginPath(); g.moveTo(0, h / 2); g.lineTo(w, h / 2); g.stroke();
    g.fillStyle = '#5a6675';
    g.font = '10px system-ui, sans-serif';
    g.textBaseline = 'top';
    g.fillText(`±${yMax} ${unit}`, 2, 1);
    g.textAlign = 'right';
    g.fillText(`${PLOT_SECONDS} s`, w - 2, 1);
    g.textAlign = 'left';
    const first = (hist.head - hist.count + histLen) % histLen;
    rows.forEach((row, j) => {
      g.strokeStyle = JOINT_COLORS[j];
      g.setLineDash(j < 3 ? [] : [3, 2]);
      g.beginPath();
      for (let k = 0; k < hist.count; k++) {
        const x = (w * (histLen - hist.count + k)) / (histLen - 1);
        const y = h / 2 - (h / 2) * Math.min(Math.max(row[(first + k) % histLen] / yMax, -1), 1);
        if (k === 0) g.moveTo(x, y); else g.lineTo(x, y);
      }
      g.stroke();
    });
    g.setLineDash([]);
  };
  const drawJoints = () => {
    let vMax = 1;
    for (const row of hist.vel) for (let k = 0; k < hist.count; k++) vMax = Math.max(vMax, Math.abs(row[k]));
    drawPlot($('plot-vel'), hist.vel, Math.ceil(vMax), 'rad/s');
    drawPlot($('plot-trq'), hist.trq, cfg.torque_limit, 'N·m');
  };

  const joystick = new Joystick($('joystick'));
  const keys = new Set();
  const state = { cmd: [0, 0], fallTimer: null, paused: false, speed: 1, policy: null, active: manifest.default };

  const showLevel = () => {
    const name = $('terrain').value;
    $('level').disabled = name === 'flat';
    $('level-v').textContent = level().toFixed(2);
    $('level-d').textContent = describe(name, level());
  };
  const setTerrain = () => {
    showLevel();
    sim.setTerrain(getTerrain($('terrain').value));
    viewer.setTerrain(sim.terrain);
    reset();
  };
  showLevel();
  const reset = () => {
    sim.reset();
    clearJoints();
    state.cmd = [0, 0];
    state.fallTimer = null;
    $('banner').hidden = true;
  };

  $('policy').addEventListener('change', async (e) => {
    const name = e.target.value;
    try {
      const next = await loadPolicy(name);
      if ($('policy').value === name) state.policy = next;
    } catch (err) {
      console.error(err);
      if ($('policy').value === name) $('policy').value = state.active;
    }
  });
  $('terrain').addEventListener('change', setTerrain);
  $('level').addEventListener('input', showLevel);
  $('level').addEventListener('change', setTerrain);
  $('reset').addEventListener('click', reset);
  $('pause').addEventListener('click', () => {
    state.paused = !state.paused;
    $('pause').textContent = state.paused ? 'Resume' : 'Pause';
  });
  viewer.scan.visible = $('scan').checked;
  $('scan').addEventListener('change', (e) => { viewer.scan.visible = e.target.checked; });
  $('follow').addEventListener('change', (e) => { viewer.follow = e.target.checked; });
  for (const id of ['vmax', 'wmax', 'slow']) {
    const input = $(id), out = $(id + '-v');
    const sync = () => {
      out.textContent = Number(input.value).toFixed(2);
      if (id === 'slow') state.speed = Number(input.value);
    };
    input.addEventListener('input', sync);
    sync();
  }
  const DRIVE_KEYS = new Set(['w', 'a', 's', 'd', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright']);
  window.addEventListener('keydown', (e) => {
    const key = e.key.toLowerCase();
    if (DRIVE_KEYS.has(key)) {
      e.preventDefault();
      document.activeElement?.blur();
    }
    keys.add(key);
    if (key === 'r') reset();
  }, { capture: true });
  window.addEventListener('blur', () => keys.clear());
  window.addEventListener('keyup', (e) => keys.delete(e.key.toLowerCase()));

  const velocityTarget = () => {
    const vmax = Number($('vmax').value), wmax = Number($('wmax').value);
    const axis = (pos, neg) => (pos.some((k) => keys.has(k)) ? 1 : 0) - (neg.some((k) => keys.has(k)) ? 1 : 0);
    const fwd = axis(['w', 'arrowup'], ['s', 'arrowdown']) || -joystick.y;
    const turn = axis(['a', 'arrowleft'], ['d', 'arrowright']) || -joystick.x;
    return [fwd * vmax, turn * wmax];
  };

  let last = performance.now(), acc = 0, busy = false, simClock = 0, wallClock = 0;
  const loop = async (now) => {
    requestAnimationFrame(loop);
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    if (busy) return;
    busy = true;
    if (state.policy) {
      ({ cfg, session } = state.policy);
      state.active = state.policy.name;
      state.policy = null;
      sim.setPolicy(cfg, session);
      viewer.buildScan();
      viewer.scan.visible = $('scan').checked;
      $('checkpoint').textContent = cfg.checkpoint;
      reset();
    }
    if (!state.paused) acc += dt * state.speed;
    const ctrl = cfg.control_dt;
    let n = 0;
    while (acc >= ctrl && n < 6) {
      acc -= ctrl;
      n++;
      if (state.fallTimer !== null) {
        state.fallTimer -= ctrl;
        if (state.fallTimer <= 0) reset();
        continue;
      }
      const target = velocityTarget();
      state.cmd = state.cmd.map((c, i) => {
        const a = (i === 0 ? ACCEL.vx : ACCEL.wz) * ctrl;
        return c + Math.min(Math.max(target[i] - c, -a), a);
      });
      const fell = await sim.step(state.cmd[0], state.cmd[1]);
      recordJoints();
      simClock += ctrl;
      const q = sim.data.qpos;
      if (fell || !sim.terrain.inside(q[0], q[1], 0.6)) {
        state.fallTimer = 1.0;
        $('banner').textContent = fell ? 'Fell — resetting' : 'Edge of map — resetting';
        $('banner').hidden = false;
      }
    }
    if (acc > ctrl * 6) acc = 0;
    viewer.render();
    const { lin, ang } = sim.bodyVelocity();
    $('hud-cmd').textContent = `${state.cmd[0].toFixed(2)} m/s  ${state.cmd[1].toFixed(2)} rad/s`;
    $('hud-act').textContent = `${lin[0].toFixed(2)} m/s  ${ang[2].toFixed(2)} rad/s`;
    updateJoints();
    drawJoints();
    wallClock += dt;
    if (wallClock > 1) {
      $('hud-rt').textContent = `${(simClock / wallClock).toFixed(2)}×`;
      simClock = wallClock = 0;
    }
    busy = false;
  };
  requestAnimationFrame(loop);
}

main().catch((e) => {
  console.error(e);
  status('Failed to load: ' + e.message);
});
