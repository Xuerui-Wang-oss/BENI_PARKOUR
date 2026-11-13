// M0 check: the MuJoCo WASM build loads in Node, compiles an XML string, steps, and casts rays.
//   node tools/check_mujoco.mjs
import load from '@mujoco/mujoco';

const mj = await load();
console.log('MuJoCo', mj.mj_versionString());

const xml = `
<mujoco>
  <option timestep="0.002" integrator="implicitfast" cone="pyramidal"/>
  <worldbody>
    <geom name="floor" type="box" pos="0 0 -0.5" size="5 5 0.5" group="0"/>
    <geom name="cube" type="box" pos="2 0 0.05" size="0.5 1 0.05" group="0"/>
    <body name="ball" pos="0 0 0.3">
      <freejoint/>
      <geom name="ball" type="sphere" size="0.05" group="3"/>
    </body>
  </worldbody>
</mujoco>`;

let t = performance.now();
const model = mj.MjModel.from_xml_string(xml);
const data = new mj.MjData(model);
console.log(`compile: ${(performance.now() - t).toFixed(1)} ms, ngeom ${model.ngeom}, nq ${model.nq}`);

t = performance.now();
for (let i = 0; i < 1000; i++) mj.mj_step(model, data);
console.log(`1000 steps: ${(performance.now() - t).toFixed(1)} ms, ball z ${data.qpos[2].toFixed(4)} (rests at 0.05)`);

// One ray and a fan of rays from above the floor, looking along +x and down; terrain group 0 only.
const group = [1, 0, 0, 0, 0, 0];
const geomid = new mj.IntBuffer(1);
const single = mj.mj_ray(model, data, [0, 0, 0.1], [1, 0, 0], group, true, -1, geomid, null);
console.log(`mj_ray along +x from z=0.1: distance ${single.toFixed(4)} (cube face at 1.5), geom ${geomid.GetView()[0]}`);

const n = 4800;
const vec = [];
for (let i = 0; i < n; i++) {
  const pitch = -0.6 + (1.2 * i) / (n - 1);
  vec.push(Math.cos(pitch), 0, Math.sin(pitch));
}
const ids = new mj.IntBuffer(n);
const dist = new mj.DoubleBuffer(n);
t = performance.now();
for (let k = 0; k < 20; k++) mj.mj_multiRay(model, data, [0, 0, 0.1], vec, group, true, -1, ids, dist, null, n, 1e10);
const d = dist.GetView();
let hits = 0;
for (let i = 0; i < n; i++) if (d[i] >= 0) hits++;
console.log(`mj_multiRay ${n} rays: ${((performance.now() - t) / 20).toFixed(2)} ms per image, ${hits} hits, ` +
  `first ${d[0].toFixed(4)}, last ${d[n - 1].toFixed(4)} (-1 = no hit)`);
