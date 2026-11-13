// Parkour lane geometry: a port of python/parkour_rl/parkour_terrain.py (the env2 lanes).
//
// A lane is a straight corridor along +x, 2 m wide, with a wall on each side. Lane x runs from 0 at the start of
// the lane; the lane origin, where positions are measured from, is at lane x = SPAWN_X on the centre line, ground
// level. An obstacle is {kind, value, xStart, xEnd} with xStart/xEnd in lane x. `value` is the pit width, the cube
// or thin_cube height, the hollow clearance (m), or the ramp angle (degrees).

export const KINDS = ['pit', 'cube', 'thin_cube', 'hollow', 'ramp_cube'];
// Range of `value` that env2 trained (python/parkour_rl/env2.py, ENV2_VALUES).
export const VALUE_RANGE = {
  pit: [0.15, 0.35], cube: [0.06, 0.12], thin_cube: [0.06, 0.12], hollow: [0.2, 0.26], ramp_cube: [8, 20],
};
export const COUNT_RANGE = [4, 6]; // Obstacles of a lane.
export const GAP_RANGE = [2, 4]; // End of an obstacle to the start of the next (m).
export const RUN_UP_RANGE = [2, 3]; // Lane origin to the first obstacle (m).
export const GOAL_MARGIN_RANGE = [2, 4]; // Last obstacle to the goal (m).

export const LANE_LENGTH = 37;
export const LANE_WIDTH = 2;
export const SPAWN_X = 1;
// A training lane ends at most this far along lane x (LANE_LENGTH - GOAL_MARGIN_RANGE[1] - 0.5).
export const TRAINED_END = 32.5;
const GROUND_THICKNESS = 0.5;
const WALL_HEIGHT = 0.5;
const WALL_THICKNESS = 0.05;
const CUBE_LENGTH = 1;
const THIN_CUBE_LENGTH = 0.1;
export const PIT_DEPTH = 0.25;
const HOLLOW_LENGTH = 0.3;
const HOLLOW_THICKNESS = 0.1;
export const RAMP_CUBE_HEIGHT = 0.15;
const RAMP_THICKNESS = 0.02;
// Ground boxes: at most MAX_GROUND_BOX long, their centres kept out of the approach to the obstacles the robot jumps.
const MAX_GROUND_BOX = 4;
const MIN_GROUND_BOX = 0.5;
const GROUND_GRID = 0.1;
const JUMP_KINDS = ['pit', 'cube', 'thin_cube'];
const APPROACH = 1.3;
const CENTRE_RADIUS = 0.3;

const radians = (degrees) => (degrees * Math.PI) / 180;

/** Length of an obstacle along x (m). */
export function obstacleLength(kind, value) {
  if (kind === 'pit') return value;
  if (kind === 'hollow') return HOLLOW_LENGTH;
  if (kind === 'ramp_cube') return RAMP_CUBE_HEIGHT / Math.tan(radians(value)) + CUBE_LENGTH;
  if (kind === 'thin_cube') return THIN_CUBE_LENGTH;
  return CUBE_LENGTH;
}

/**
 * Place obstacles along the lane.
 * @param items [{kind, value, gap}] in lane order; `gap` is the distance (m) from the end of the obstacle before
 *   it and is ignored for the first one.
 * @param runUp distance (m) from the lane origin to the first obstacle.
 * @returns [{kind, value, xStart, xEnd}] in lane x.
 */
export function placeObstacles(items, runUp) {
  let x = SPAWN_X + runUp;
  return items.map(({ kind, value, gap }, i) => {
    if (i) x += gap;
    const obstacle = { kind, value, xStart: x, xEnd: x + obstacleLength(kind, value) };
    x = obstacle.xEnd;
    return obstacle;
  });
}

/**
 * Edges of the ground boxes that cover [xMin, xMax], first xMin and last xMax. Each box is MIN_GROUND_BOX to
 * MAX_GROUND_BOX long; among such splits the one with the fewest box centres inside `zones` (open intervals
 * [low, high] of lane x) is chosen, then the one with the fewest boxes. Candidate edges lie GROUND_GRID apart.
 */
export function groundEdges(xMin, xMax, zones) {
  const slack = 1e-9;
  const gridSteps = Math.trunc((xMax - xMin - slack) / GROUND_GRID);
  const xs = [];
  for (let k = 0; k <= gridSteps; k++) xs.push(xMin + k * GROUND_GRID);
  xs.push(xMax);
  // best.get(i) is the best way to cover [xMin, xs[i]]: [box centres inside a zone, boxes, previous edge index].
  const best = new Map([[0, [0, 0, null]]]);
  for (let i = 1; i < xs.length; i++) {
    let pick = null;
    for (const [j, [inside, boxes]] of best) {
      const length = xs[i] - xs[j];
      if (length < MIN_GROUND_BOX - slack || length > MAX_GROUND_BOX + slack) continue;
      const centre = (xs[i] + xs[j]) / 2;
      const option = [inside + (zones.some(([low, high]) => low < centre && centre < high) ? 1 : 0), boxes + 1, j];
      if (pick === null || option[0] < pick[0] || (option[0] === pick[0] && option[1] < pick[1])) pick = option;
    }
    if (pick) best.set(i, pick);
  }
  if (!best.has(xs.length - 1)) throw new Error(`no ground boxes fit between ${xMin} and ${xMax}`);
  const edges = [];
  for (let i = xs.length - 1; i !== null; i = best.get(i)[2]) edges.push(xs[i]);
  return edges.reverse();
}

/**
 * The boxes of a lane, in the order parkour_terrain.py adds them.
 * @param obstacles [{kind, value, xStart, xEnd}] in lane x, in lane order.
 * @param length lane length (m); the ground and the walls run from lane x = 0 to `length`.
 * @returns [{role, index, pos, size, quat}]: centre, half sizes and orientation (w, x, y, z) relative to the lane
 *   origin. `role` is 'ground', 'pit', 'cube', 'thin_cube', 'hollow', 'ramp', 'plateau' or 'wall'; `index` is the
 *   obstacle the box belongs to, -1 for ground and walls.
 */
export function laneBoxes(obstacles, length = LANE_LENGTH) {
  const zones = obstacles.filter((o) => JUMP_KINDS.includes(o.kind))
    .map((o) => [o.xStart - APPROACH, o.xStart + CENTRE_RADIUS]);
  const boxes = [];
  const box = (role, index, xMin, xMax, zMin, zMax) => boxes.push({
    role, index,
    pos: [(xMin + xMax) / 2 - SPAWN_X, 0, (zMin + zMax) / 2],
    size: [(xMax - xMin) / 2, LANE_WIDTH / 2, (zMax - zMin) / 2],
    quat: [1, 0, 0, 0],
  });
  const ground = (xMin, xMax) => {
    const edges = groundEdges(xMin, xMax, zones);
    for (let i = 0; i + 1 < edges.length; i++) box('ground', -1, edges[i], edges[i + 1], -GROUND_THICKNESS, 0);
  };
  const wall = (yMin) => boxes.push({
    role: 'wall', index: -1,
    pos: [length / 2 - SPAWN_X, yMin + WALL_THICKNESS / 2 - LANE_WIDTH / 2, (WALL_HEIGHT - GROUND_THICKNESS) / 2],
    size: [length / 2, WALL_THICKNESS / 2, (WALL_HEIGHT + GROUND_THICKNESS) / 2],
    quat: [1, 0, 0, 0],
  });

  let groundX = 0;
  obstacles.forEach(({ kind, value, xStart, xEnd }, index) => {
    if (kind === 'pit') {
      ground(groundX, xStart);
      box('pit', index, xStart, xEnd, -GROUND_THICKNESS, -PIT_DEPTH);
      groundX = xEnd;
    } else if (kind === 'cube' || kind === 'thin_cube') {
      box(kind, index, xStart, xEnd, 0, value);
    } else if (kind === 'hollow') {
      box('hollow', index, xStart, xEnd, value, value + HOLLOW_THICKNESS);
    } else {
      // A box rotated about y whose top face rises from (xStart, 0) to the plateau height, then the plateau.
      const theta = radians(value);
      const run = RAMP_CUBE_HEIGHT / Math.tan(theta);
      const slope = RAMP_CUBE_HEIGHT / Math.sin(theta);
      const normal = [-Math.sin(theta), 0, Math.cos(theta)];
      boxes.push({
        role: 'ramp', index,
        pos: [xStart + run / 2 - (RAMP_THICKNESS / 2) * normal[0] - SPAWN_X, 0,
          RAMP_CUBE_HEIGHT / 2 - (RAMP_THICKNESS / 2) * normal[2]],
        size: [slope / 2, LANE_WIDTH / 2, RAMP_THICKNESS / 2],
        quat: [Math.cos(theta / 2), 0, -Math.sin(theta / 2), 0],
      });
      box('plateau', index, xEnd - CUBE_LENGTH, xEnd, 0, RAMP_CUBE_HEIGHT);
    }
  });
  ground(groundX, length);
  wall(0);
  wall(LANE_WIDTH - WALL_THICKNESS);
  return boxes;
}

/** The boxes as MJCF geoms with the contact parameters of the training terrain (geom group 0). */
export function boxesXml(boxes, offset = [0, 0, 0]) {
  return boxes.map(({ pos, size, quat }) => `<geom type="box" size="${size.join(' ')}" `
    + `pos="${pos.map((v, i) => v + offset[i]).join(' ')}" quat="${quat.join(' ')}" group="0" contype="1" `
    + 'conaffinity="1" condim="3" priority="0" friction="1 0.005 0.0001" solref="0.02 1" '
    + 'solimp="0.9 0.95 0.001 0.5 2" solmix="1" margin="0" gap="0" mass="0"/>').join('');
}
