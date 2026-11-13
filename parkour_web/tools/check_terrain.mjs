// Check the JavaScript lane builder against the lanes that mjlab compiled (tools/ref/env2_lanes.json, written by
// export_robot.py): the same boxes, in the same order, for all 80 env2 lanes.
//   node tools/check_terrain.mjs
import fs from 'node:fs';
import { laneBoxes, placeObstacles, obstacleLength } from '../src/sim/terrain.js';

const lanes = JSON.parse(fs.readFileSync(new URL('./ref/env2_lanes.json', import.meta.url)));
let worst = 0;
let boxes = 0;
for (const lane of lanes) {
  const obstacles = lane.obstacles.map((o) => ({ kind: o.kind, value: o.value, xStart: o.x_start, xEnd: o.x_end }));
  const built = laneBoxes(obstacles);
  if (built.length !== lane.geoms.length) {
    throw new Error(`row ${lane.row} column ${lane.column}: ${built.length} boxes, reference ${lane.geoms.length}`);
  }
  built.forEach((box, i) => {
    const ref = lane.geoms[i];
    for (const key of ['pos', 'size', 'quat']) {
      box[key].forEach((v, k) => { worst = Math.max(worst, Math.abs(v - ref[key][k])); });
    }
  });
  boxes += built.length;

  // placeObstacles reproduces the positions from kinds, values and gaps.
  const items = obstacles.map((o, i) => ({ kind: o.kind, value: o.value, gap: i ? o.xStart - obstacles[i - 1].xEnd : 0 }));
  const placed = placeObstacles(items, obstacles[0].xStart - 1);
  placed.forEach((o, i) => {
    worst = Math.max(worst, Math.abs(o.xStart - obstacles[i].xStart), Math.abs(o.xEnd - obstacles[i].xEnd),
      Math.abs(obstacleLength(o.kind, o.value) - (obstacles[i].xEnd - obstacles[i].xStart)));
  });
}
console.log(`${lanes.length} lanes, ${boxes} boxes: largest difference ${worst.toExponential(2)} m`);
if (worst > 1e-6) throw new Error('the lane builder differs from the reference');
console.log('lane builder matches the reference');
