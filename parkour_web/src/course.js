// A user-built course: 4 to 6 obstacles, each with a kind, a size and the gap to the obstacle before it.

import {
  COUNT_RANGE, GAP_RANGE, KINDS, LANE_LENGTH, SPAWN_X, TRAINED_END, VALUE_RANGE, obstacleLength,
} from './sim/terrain.js';

export { COUNT_RANGE, GAP_RANGE, KINDS, VALUE_RANGE };

export const RUN_UP = 2.5; // Spawn to the first obstacle (m); training drew 2-3 m.
export const GOAL_MARGIN = 3; // Last obstacle to the goal (m); training drew 2-4 m.
// Flat lane past the goal (m). In training another lane begins where a lane ends, so the robot always saw ground
// and walls beyond its goal.
const TAIL = 8;

// How each kind is shown: its name, what its size means, the slider step, and its colour in the scene.
export const KIND_INFO = {
  pit: { label: 'Pit', size: 'Width', unit: 'm', step: 0.01, color: '#232b38' },
  cube: { label: 'Cube', size: 'Height', unit: 'm', step: 0.01, color: '#f2a65a' },
  thin_cube: { label: 'Bar', size: 'Height', unit: 'm', step: 0.01, color: '#e02d2d' },
  hollow: { label: 'Tunnel', size: 'Clearance', unit: 'm', step: 0.01, color: '#5fb3d9' },
  ramp_cube: { label: 'Ramp', size: 'Slope', unit: '°', step: 1, color: '#7fcf95' },
};

export const formatValue = (kind, value) => (KIND_INFO[kind].unit === '°' ? `${Math.round(value)}°` : `${value.toFixed(2)} m`);

/** An obstacle of `kind` in the middle of its trained size range, 3 m after the one before it. */
export function defaultItem(kind) {
  const [low, high] = VALUE_RANGE[kind];
  const { step } = KIND_INFO[kind];
  return { kind, value: Math.round((low + high) / 2 / step) * step, gap: 3 };
}

/** `count` obstacles drawn as the training lanes were: kind, size and gap uniform in their ranges. */
export function randomItems(count, random = Math.random) {
  return Array.from({ length: count }, () => {
    const kind = KINDS[Math.floor(random() * KINDS.length)];
    const [low, high] = VALUE_RANGE[kind];
    const { step } = KIND_INFO[kind];
    return {
      kind,
      value: Math.round((low + random() * (high - low)) / step) * step,
      gap: Math.round((GAP_RANGE[0] + random() * (GAP_RANGE[1] - GAP_RANGE[0])) * 10) / 10,
    };
  });
}

const EMPTY_SLOT_LENGTH = 1; // Room an empty slot takes along the lane while the course is being built (m).

/**
 * The course of `items`: one entry per slot, {kind, value, gap} or null for a slot that is still empty.
 * @returns {slots, obstacles, goalX, length, spawn, complete, beyondTraining}. `slots` are the stretches
 *   [{x0, x1, item}] of the slots and `goalX` and `spawn` the goal and the start, all relative to the lane origin,
 *   where the robot starts; `obstacles` are the filled slots as `laneBoxes` takes them (lane x, which starts 1 m
 *   before the origin); `beyondTraining` tells that the last obstacle ends farther along the lane than any
 *   training lane did.
 */
export function buildCourse(items) {
  let x = SPAWN_X + RUN_UP;
  const slots = [];
  const obstacles = [];
  items.forEach((item, i) => {
    if (i) x += item ? item.gap : 3;
    const length = item ? obstacleLength(item.kind, item.value) : EMPTY_SLOT_LENGTH;
    slots.push({ x0: x - SPAWN_X, x1: x + length - SPAWN_X, item });
    if (item) obstacles.push({ kind: item.kind, value: item.value, xStart: x, xEnd: x + length });
    x += length;
  });
  return {
    slots,
    obstacles,
    goalX: x - SPAWN_X + GOAL_MARGIN,
    length: Math.max(LANE_LENGTH, x + GOAL_MARGIN + TAIL),
    spawn: { x: 0, y: 0 },
    complete: items.every(Boolean),
    beyondTraining: x > TRAINED_END,
  };
}
