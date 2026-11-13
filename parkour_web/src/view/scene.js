// The 3D view of the page (three.js): a toy grand-prix circuit with the lane as its start-finish straight, the robot,
// the slot frames of the editor, and the cameras. The world is z-up, as in MuJoCo; positions are relative to the
// lane origin, where the robot starts. Everything here is only drawn: the physics and the robot's depth camera see
// the lane boxes alone.

import * as THREE from 'three';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { BODIES } from '../protocol.js';
import { KIND_INFO } from '../course.js';
import { laneBoxes } from '../sim/terrain.js';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

const COLORS = {
  sky: '#a6d4f2', grassA: '#5fa85a', grassB: '#559d52', track: '#3a3d43', pitFloor: '#15171b', kerbA: '#e0241b',
  kerbB: '#f5f5f5', line: '#f4f6f8', concrete: '#cfd3d8', dark: '#1d2026', steel: '#8b929b', gravel: '#dccfa9',
  red: '#e0241b', blue: '#2f6fd6', lightsOn: '#ff2318', lightsOff: '#3b1512', slot: '#ffffff', slotFilled: '#d4ff4f',
};
const CROWD = ['#e0241b', '#f5f5f5', '#2f6fd6', '#f2c230', '#21a179', '#ff7a1a', '#222831'];
// Banners over and beside the track: text, background, text colour.
const BANNERS = [['BENI GRAND PRIX', '#e0241b', '#ffffff'], ['PARKOUR', '#f5f5f5', '#e0241b'], ['OPENBENI', '#1d2026', '#d4ff4f']];
const FIRST_PERSON_FOV = 75;
const CHASE_FOV = 55;
const EDITOR_FOV = 40;
const WHEEL_RADIUS = 0.038;
const WHEEL_WIDTH = 0.021;
const SLOT_HEIGHT = 0.7;
// Half width of the road between the walls, whose inner faces are 0.95 m from the centre line. The physics boxes of
// the ground and of the obstacles are as wide as the outer faces of the walls (1 m); drawn that wide, their sides
// would lie in the same plane as the wall faces and flicker through them. Drawn boxes end just inside the walls.
const ROAD_HALF_WIDTH = 0.955;
const MARKING_WIDTH = 1.86; // Width of what is painted on the road or framed above it, clear of the walls.
// Near plane of each camera (m). The depth buffer resolves about distance^2 / (near * 16 million): with the 1 cm
// that the robot's view needs, markings a few millimetres above the road flicker from the 25 m of the editor view.
const NEAR = { first: 0.01, chase: 0.05, editor: 1 };
// The robot's depth camera in the frame of its base body (assets/beni_base/mjcf/beni.xml): the first-person view.
const EYE_POSITION = new THREE.Vector3(0.07, 0.03, -0.016);
const EYE_QUATERNION = new THREE.Quaternion(0.5, -0.5, -0.5, 0.5); // MuJoCo (w x y z) = (0.5, 0.5, -0.5, -0.5).
// Orientation of an upright sign (a plane with its text along the plane's x) that is read by someone looking along
// +x (the robot), and of one read from the -y side (the editor camera).
const FACING_BACK = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(
  new THREE.Vector3(0, -1, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(-1, 0, 0)));
const FACING_SIDE = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(
  new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, -1, 0)));

function canvasTexture(width, height, draw) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  draw(canvas.getContext('2d'), width, height);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.anisotropy = 4;
  return texture;
}

/** Two blocks side by side in colours `a` and `b`: kerbs, mown grass. */
const blocks = (a, b) => canvasTexture(64, 8, (c, w, h) => {
  c.fillStyle = b;
  c.fillRect(0, 0, w, h);
  c.fillStyle = a;
  c.fillRect(0, 0, w / 2, h);
});

const checker = () => canvasTexture(64, 64, (c, w, h) => {
  c.fillStyle = '#ffffff';
  c.fillRect(0, 0, w, h);
  c.fillStyle = COLORS.dark;
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) if ((i + j) % 2) c.fillRect(i * w / 4, j * h / 4, w / 4, h / 4);
});

const banner = ([text, background, color]) => canvasTexture(512, 64, (c, w, h) => {
  c.fillStyle = background;
  c.fillRect(0, 0, w, h);
  c.fillStyle = color;
  c.font = 'italic 900 40px system-ui, sans-serif';
  c.textAlign = 'center';
  c.textBaseline = 'middle';
  c.fillText(text, w / 2, h / 2 + 2);
});

/** A small repeatable random generator, so the decorations stand in the same places on every visit. */
function repeatable(seed) {
  let state = seed;
  return () => {
    state = (state * 16807) % 2147483647;
    return state / 2147483647;
  };
}

export class View {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(COLORS.sky);
    this.scene.fog = new THREE.Fog(COLORS.sky, 45, 130);
    this.camera = new THREE.PerspectiveCamera(EDITOR_FOV, 1, NEAR.editor, 300);

    this.scene.add(new THREE.HemisphereLight('#ffffff', '#7fae7a', 1.5));
    this.sun = new THREE.DirectionalLight('#fff6e6', 2.5);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    Object.assign(this.sun.shadow.camera, { left: -10, right: 10, top: 10, bottom: -10, near: 1, far: 50 });
    this.scene.add(this.sun, this.sun.target);

    this.lane = new THREE.Group();
    this.slots = new THREE.Group();
    this.robot = new THREE.Group();
    this.robot.visible = false;
    this.scene.add(this.lane, this.slots, this.robot);
    this.bodies = BODIES.map(() => {
      const group = new THREE.Group();
      this.robot.add(group);
      return group;
    });
    // The five start lights, one material per pod: red while the course is being built, coming on one by one
    // before a run and out while the robot runs.
    this.startLights = Array.from({ length: 5 }, () => new THREE.MeshBasicMaterial({ color: COLORS.lightsOn }));

    this.mode = 'editor'; // 'editor', 'chase' (behind the robot) or 'first' (the robot's own camera).
    this.editorPosition = new THREE.Vector3(18, -24, 14); // Where the editor camera is heading, and its target.
    this.editorTarget = new THREE.Vector3(18, 0, 0);
    this.lookAt = new THREE.Vector3(18, 0, 0);
    this.snap = true; // The next chase frame jumps to its place instead of gliding there.
    this.lastTime = performance.now();
    this.onResize = null; // Called after the picture was fitted to a new size.
    this.resize();
  }

  /** Load the robot's visual meshes (public/robot/visual) and build its wheels. */
  async loadRobot(base) {
    const meshes = await (await fetch(`${base}robot/visual/meshes.json`)).json();
    const texture = await new THREE.TextureLoader().loadAsync(`${base}robot/visual/beni_white.png`);
    texture.colorSpace = THREE.SRGBColorSpace;
    const material = new THREE.MeshStandardMaterial({ map: texture, roughness: 0.55, metalness: 0.05 });
    const loader = new OBJLoader();
    await Promise.all(meshes.map(async ({ body, file, refpos, refquat }) => {
      const object = await loader.loadAsync(`${base}robot/visual/${file}`);
      object.traverse((child) => {
        if (!child.isMesh) return;
        child.material = material;
        child.castShadow = true;
      });
      // The mesh vertices are given relative to a reference frame: undo it to get the body frame.
      const [w, x, y, z] = refquat;
      object.quaternion.set(-x, -y, -z, w);
      object.position.set(-refpos[0], -refpos[1], -refpos[2]).applyQuaternion(object.quaternion);
      this.bodies[BODIES.indexOf(body)].add(object);
    }));
    // The wheels: a tyre around the body's y axis, with three pale spokes that show the rotation.
    const tyre = new THREE.MeshStandardMaterial({ color: '#2b2732', roughness: 0.9 });
    const hub = new THREE.MeshStandardMaterial({ color: '#f4eef7', roughness: 0.5 });
    const spoke = new THREE.MeshStandardMaterial({ color: '#cfc6d6', roughness: 0.6 });
    for (const [body, side] of [['LW_Link', 1], ['RW_Link', -1]]) {
      const wheel = new THREE.Group();
      wheel.position.y = side * 0.013;
      const rim = new THREE.Mesh(new THREE.CylinderGeometry(WHEEL_RADIUS, WHEEL_RADIUS, WHEEL_WIDTH, 32), tyre);
      const cap = new THREE.Mesh(new THREE.CylinderGeometry(WHEEL_RADIUS * 0.68, WHEEL_RADIUS * 0.68, WHEEL_WIDTH + 0.002, 24), hub);
      rim.castShadow = true;
      wheel.add(rim, cap);
      for (let i = 0; i < 3; i++) {
        const bar = new THREE.Mesh(new THREE.BoxGeometry(WHEEL_RADIUS * 1.2, WHEEL_WIDTH + 0.004, 0.006), spoke);
        bar.rotation.y = (i * Math.PI) / 3;
        wheel.add(bar);
      }
      this.bodies[BODIES.indexOf(body)].add(wheel);
    }
  }

  /** Build the lane of `course` (from `buildCourse`) and the circuit around it. */
  setCourse(course) {
    this.lane.clear();
    this.course = course;
    const lambert = (color) => new THREE.MeshLambertMaterial({ color });
    const add = (geometry, material, x, y, z, shadow = true) => {
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(x, y, z);
      mesh.castShadow = shadow;
      mesh.receiveShadow = true;
      this.lane.add(mesh);
      return mesh;
    };
    const box = (sx, sy, sz, material, x, y, z, shadow) => add(new THREE.BoxGeometry(sx, sy, sz), material, x, y, z, shadow);
    const start = -1; // Lane x where the lane begins.
    const end = course.length - 1;

    // The lane itself: the boxes of the physics. The walls are painted like kerbs.
    const kerb = blocks(COLORS.kerbA, COLORS.kerbB);
    kerb.repeat.set(course.length, 1);
    const materials = {
      ground: lambert(COLORS.track), pit: lambert(COLORS.pitFloor), wall: new THREE.MeshLambertMaterial({ map: kerb }),
      plateau: lambert(KIND_INFO.ramp_cube.color), ramp: lambert(KIND_INFO.ramp_cube.color),
    };
    for (const kind of ['cube', 'thin_cube', 'hollow']) materials[kind] = lambert(KIND_INFO[kind].color);
    for (const { role, pos, size, quat } of laneBoxes(course.obstacles, course.length)) {
      const width = role === 'wall' ? 2 * size[1] : 2 * ROAD_HALF_WIDTH;
      const mesh = box(2 * size[0], width, 2 * size[2], materials[role], pos[0], pos[1], pos[2],
        role !== 'ground' && role !== 'pit');
      mesh.quaternion.set(quat[1], quat[2], quat[3], quat[0]);
    }

    // Mown grass around the lane, with a hole where the lane is (a pit of the course must stay open).
    const outline = new THREE.Shape([[-400, -400], [400, -400], [400, 400], [-400, 400]].map(([x, y]) => new THREE.Vector2(x, y)));
    outline.holes.push(new THREE.Path([[start, -1], [end, -1], [end, 1], [start, 1]].map(([x, y]) => new THREE.Vector2(x, y))));
    const lawn = blocks(COLORS.grassA, COLORS.grassB);
    lawn.repeat.set(1 / 5, 1);
    const grass = add(new THREE.ShapeGeometry(outline), new THREE.MeshLambertMaterial({ map: lawn }), 0, 0, -0.01, false);
    grass.receiveShadow = true;

    // Paint lies 5 mm above the road and is pulled towards the camera in the depth test, so it never flickers
    // against the road under it.
    const decal = (parameters) => new THREE.MeshBasicMaterial({ ...parameters, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2 });
    const paint = (x, y, length, width, material) => {
      const mesh = add(new THREE.PlaneGeometry(length, width), material, x, y, 0.005, false);
      mesh.receiveShadow = false;
    };
    const white = decal({ color: COLORS.line });
    // White lines along both edges of the road, broken where an obstacle is.
    const breaks = course.slots.filter((slot) => slot.item).map((slot) => [slot.x0 - 0.03, slot.x1 + 0.03]);
    let from = start;
    for (const [x0, x1] of [...breaks, [end, end]]) {
      if (x0 > from) for (const y of [-0.87, 0.87]) paint((from + x0) / 2, y, x0 - from, 0.035, white);
      from = x1;
    }
    // The grid box of the start, and the chequered finish line.
    paint(0.2, 0, 0.04, 0.7, white);
    for (const y of [-0.35, 0.35]) paint(0.06, y, 0.3, 0.04, white);
    const chequered = checker();
    chequered.repeat.set(1, 6);
    paint(course.goalX, 0, 0.5, MARKING_WIDTH, decal({ map: chequered }));

    const dark = lambert(COLORS.dark);
    const steel = lambert(COLORS.steel);
    /** A gantry over the lane at lane x: two posts and a beam, `height` to the middle of the beam. */
    const gantry = (x, height, material = dark) => {
      for (const y of [-1.1, 1.1]) box(0.08, 0.08, height + 0.1, material, x, y, (height + 0.1) / 2);
      return box(0.08, 2.28, 0.1, material, x, 0, height + 0.15);
    };
    /** An upright sign: `texture` on a board `width` x `height`, its centre at (x, y, z), turned by `facing`. */
    const sign = (texture, width, height, x, y, z, facing) => {
      const mesh = add(new THREE.PlaneGeometry(width, height), new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide }), x, y, z, false);
      mesh.quaternion.copy(facing);
      return mesh;
    };

    // The start lights: five pods of two lamps each under a gantry ahead of the grid, low and far enough to be
    // in the picture of the camera behind the robot. The pods come on from left to right as the robot sees them.
    const lights = 1.7;
    gantry(lights, 1.0);
    for (let i = 0; i < 5; i++) {
      const y = 0.72 - 0.36 * i;
      box(0.06, 0.26, 0.24, dark, lights, y, 0.98);
      for (const z of [0.92, 1.04]) {
        const lamp = add(new THREE.CircleGeometry(0.045, 20), this.startLights[i], lights - 0.035, y, z, false);
        lamp.quaternion.copy(FACING_BACK);
      }
    }
    // The finish: a chequered beam and two flags.
    gantry(course.goalX, 1.2);
    const beam = checker();
    beam.repeat.set(8, 1);
    sign(beam, 2.2, 0.3, course.goalX - 0.045, 0, 1.2, FACING_BACK);
    // Banner bridges over the lane, read by the robot as it drives.
    const banners = BANNERS.map(banner);
    for (let i = 0, x = 6; x < end - 2; x += 8, i++) {
      if (Math.abs(x - course.goalX) < 2.5) continue;
      gantry(x, 1.35, steel);
      box(0.05, 2.2, 0.3, dark, x, 0, 1.33);
      sign(banners[i % banners.length], 2.2, 0.28, x - 0.03, 0, 1.33, FACING_BACK);
    }

    const random = repeatable(23);
    const concrete = lambert(COLORS.concrete);
    const crowdGeometry = new THREE.BoxGeometry(0.13, 0.13, 0.2);
    const crowdMaterial = new THREE.MeshLambertMaterial({ color: '#ffffff' });
    const color = new THREE.Color();
    const matrix = new THREE.Matrix4();
    /**
     * A grandstand `length` long with its front edge at (x, y): five steps rising away from the track (`side` is
     * +1 on the +y side of the lane, -1 on the other), spectators, a roof, and a banner that faces -y, where the
     * editor camera stands: on the front edge of the roof for a stand beyond the lane, on the back wall for one
     * on the near side.
     */
    const grandstand = (x, y, side, length, roofColor, text) => {
      const rows = 5;
      const people = new THREE.InstancedMesh(crowdGeometry, crowdMaterial, rows * Math.floor(length / 0.2));
      let count = 0;
      for (let row = 0; row < rows; row++) {
        const top = 0.3 * (row + 1);
        box(length, 0.5, top, concrete, x, y + side * (0.25 + 0.5 * row), top / 2);
        for (let k = 0; k < Math.floor(length / 0.2); k++) {
          if (random() < 0.12) continue; // An empty seat.
          matrix.setPosition(x - length / 2 + 0.1 + 0.2 * k, y + side * (0.25 + 0.5 * row), top + 0.1);
          people.setMatrixAt(count, matrix);
          people.setColorAt(count, color.set(CROWD[Math.floor(random() * CROWD.length)]));
          count += 1;
        }
      }
      people.count = count;
      people.instanceColor.needsUpdate = true;
      this.lane.add(people);
      const depth = 0.5 * rows;
      box(length, 0.08, 2.5, concrete, x, y + side * (depth + 0.04), 1.25); // Back wall.
      box(length + 0.3, depth + 0.5, 0.08, lambert(roofColor), x, y + side * (depth / 2 + 0.1), 2.5); // Roof.
      if (side > 0) sign(text, Math.min(length, 5), 0.34, x, y - 0.16, 2.33, FACING_SIDE);
      else sign(text, Math.min(length - 0.4, 5), 0.6, x, y - depth - 0.09, 1.6, FACING_SIDE);
    };
    const stand = 7;
    for (let i = 0, x = 5.5; x < end - 3; x += stand + 2.5, i++) {
      grandstand(x, 2.6, 1, stand, i % 2 ? COLORS.blue : COLORS.red, banners[i % banners.length]);
      // The editor camera looks down from the near side: a stand there has to be far enough from the lane for its
      // roof not to hide the lane (its 2.5 m hide about 3.6 m of ground behind it).
      grandstand(x + 4, -5.2, -1, stand - 2, i % 2 ? COLORS.red : COLORS.blue, banners[(i + 1) % banners.length]);
    }

    // The pit building beside the grid: garages with dark doors, a red band, and a terrace rail.
    box(4.6, 1.5, 1.1, lambert('#eef1f4'), -1.2, 3.4, 0.55);
    box(4.7, 1.6, 0.12, lambert(COLORS.red), -1.2, 3.4, 1.16);
    for (let k = 0; k < 4; k++) {
      const door = add(new THREE.PlaneGeometry(0.85, 0.75), new THREE.MeshBasicMaterial({ color: COLORS.dark, side: THREE.DoubleSide }), -2.85 + 1.1 * k, 2.645, 0.4, false);
      door.quaternion.copy(FACING_SIDE);
    }
    sign(banner(['PIT LANE', '#1d2026', '#ffffff']), 2.4, 0.26, -1.2, 2.64, 0.95, FACING_SIDE);

    // Gravel traps with tyre walls on the near side, between the lane and the small stands.
    const gravel = decal({ color: COLORS.gravel });
    const tyres = new THREE.CylinderGeometry(0.15, 0.15, 0.13, 14).rotateX(Math.PI / 2);
    const stacks = [];
    for (let x = 1; x < end - 4; x += 9.5) {
      const patch = add(new THREE.PlaneGeometry(4.5, 1.7), gravel, x, -2.05, -0.004, false);
      patch.receiveShadow = true;
      for (let k = 0; k < 13; k++) for (let level = 0; level < 3; level++) stacks.push([x - 2.1 + 0.35 * k, -3.05, 0.065 + 0.13 * level, (k + level) % 2]);
    }
    const tyreWall = new THREE.InstancedMesh(tyres, new THREE.MeshLambertMaterial({ color: '#ffffff' }), stacks.length);
    stacks.forEach(([x, y, z, odd], i) => {
      tyreWall.setMatrixAt(i, matrix.setPosition(x, y, z));
      tyreWall.setColorAt(i, color.set(odd ? COLORS.kerbA : COLORS.kerbB));
    });
    tyreWall.castShadow = true;
    this.lane.add(tyreWall);

    // Flags along the far side, chequered ones at the finish.
    const flagColors = [COLORS.red, COLORS.blue, '#f2c230', '#21a179'].map((c) => new THREE.MeshBasicMaterial({ color: c, side: THREE.DoubleSide }));
    const chequeredFlag = new THREE.MeshBasicMaterial({ map: checker(), side: THREE.DoubleSide });
    const flag = (x, y, material) => {
      box(0.03, 0.03, 1.9, steel, x, y, 0.95, false);
      sign(null, 0.5, 0.32, x + 0.27, y, 1.72, FACING_SIDE).material = material;
    };
    for (let i = 0, x = 1.5; x < end - 1; x += 2.2, i++) if (Math.abs(x - course.goalX) > 1) flag(x, 1.25, flagColors[i % flagColors.length]);
    for (const y of [-1.25, 1.25]) for (const dx of [-0.5, 0.5]) flag(course.goalX + dx, y, chequeredFlag);

    // Trees behind everything.
    const count = 60;
    const crowns = new THREE.InstancedMesh(new THREE.ConeGeometry(0.9, 2.4, 9).rotateX(Math.PI / 2),
      new THREE.MeshLambertMaterial({ color: '#ffffff' }), count);
    const trunks = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.12, 0.14, 0.8, 7).rotateX(Math.PI / 2), lambert('#6b4a2f'), count);
    for (let i = 0; i < count; i++) {
      const x = start - 4 + (end - start + 8) * random();
      // Only beyond the lane: on the near side they would stand between the editor camera and the lane.
      const y = 8 + 18 * random();
      const scale = 0.7 + 0.7 * random();
      crowns.setMatrixAt(i, matrix.makeScale(scale, scale, scale).setPosition(x, y, 0.7 * scale + 1.2 * scale));
      crowns.setColorAt(i, color.set(random() < 0.5 ? '#2f7d43' : '#3b8f4c'));
      trunks.setMatrixAt(i, matrix.makeScale(scale, scale, scale).setPosition(x, y, 0.4 * scale));
    }
    crowns.castShadow = true;
    this.lane.add(crowns, trunks);
  }

  /** Light the first `count` of the five start lights; the others are out. */
  setLights(count) {
    this.startLights.forEach((material, i) => material.color.set(i < count ? COLORS.lightsOn : COLORS.lightsOff));
  }

  /**
   * Draw the slot frames of the editor: one dashed box per slot.
   * @param slots [{x0, x1, filled, active}] along the lane (relative to the lane origin).
   */
  setSlots(slots) {
    this.slots.clear();
    for (const { x0, x1, filled, active } of slots) {
      const geometry = new THREE.BoxGeometry(x1 - x0, MARKING_WIDTH - 0.06, SLOT_HEIGHT);
      const color = filled ? COLORS.slotFilled : COLORS.slot;
      const frame = new THREE.LineSegments(new THREE.EdgesGeometry(geometry),
        new THREE.LineDashedMaterial({ color, dashSize: 0.12, gapSize: 0.08 }));
      frame.computeLineDistances();
      frame.position.set((x0 + x1) / 2, 0, SLOT_HEIGHT / 2);
      this.slots.add(frame);
      if (active) {
        const fill = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.22, depthWrite: false }));
        fill.position.copy(frame.position);
        this.slots.add(fill);
      }
    }
  }

  /** The body poses to draw: 7 numbers per body of BODIES (position, then quaternion w x y z). */
  setFrame(bodies) {
    this.bodies.forEach((group, i) => {
      group.position.set(bodies[7 * i], bodies[7 * i + 1], bodies[7 * i + 2]);
      group.quaternion.set(bodies[7 * i + 4], bodies[7 * i + 5], bodies[7 * i + 6], bodies[7 * i + 3]);
    });
    this.robot.visible = true;
  }

  /** 'editor', 'chase' (behind the robot) or 'first' (the robot's own camera). */
  setMode(mode) {
    this.mode = mode;
    this.snap = true;
    this.slots.visible = mode === 'editor';
    this.camera.fov = { first: FIRST_PERSON_FOV, chase: CHASE_FOV, editor: EDITOR_FOV }[mode];
    this.camera.near = NEAR[mode];
    this.camera.updateProjectionMatrix();
  }

  /** Point the editor camera at the stretch [x0, x1] of the lane, seen from the side and above. */
  frameLane(x0, x1, immediate = false) {
    const centre = (x0 + x1) / 2;
    const halfWidth = Math.max((x1 - x0) / 2, 1.2) * 1.12;
    // Distance at which the stretch fills the width of the view. The canvas can have no size yet while the page
    // loads; the aspect is then bounded so that the distance stays finite.
    const horizontal = 2 * Math.atan(Math.tan((EDITOR_FOV * Math.PI) / 360) * Math.max(this.camera.aspect, 0.2));
    const distance = halfWidth / Math.tan(horizontal / 2);
    this.editorTarget.set(centre, 0, 0);
    this.editorPosition.set(centre, -distance * 0.82, distance * 0.57);
    if (immediate) this.snap = true;
  }

  /**
   * Fit the picture to the size of the canvas's container. `render` calls this whenever that size has changed, and
   * then `onResize`, so the view also recovers when the page was laid out late (inside a frame that had no size
   * while it loaded, for one).
   */
  resize() {
    const { clientWidth, clientHeight } = this.canvas.parentElement;
    this.width = clientWidth;
    this.height = clientHeight;
    if (!clientWidth || !clientHeight) return; // Nothing to draw into yet.
    this.renderer.setSize(clientWidth, clientHeight, false);
    this.camera.aspect = clientWidth / clientHeight;
    this.camera.updateProjectionMatrix();
  }

  /** Position of the lane point (x, y, z) on the canvas, in CSS pixels. */
  project(x, y, z) {
    const point = new THREE.Vector3(x, y, z).project(this.camera);
    return {
      x: (point.x + 1) / 2 * this.canvas.clientWidth,
      y: (1 - point.y) / 2 * this.canvas.clientHeight,
      visible: point.z < 1,
    };
  }

  render() {
    const { clientWidth, clientHeight } = this.canvas.parentElement;
    if (clientWidth !== this.width || clientHeight !== this.height) {
      this.resize();
      this.onResize?.();
    }
    if (!this.width || !this.height) return;
    const now = performance.now();
    const blend = this.snap ? 1 : 1 - Math.exp(-(now - this.lastTime) / 160); // Smoothing of the camera moves.
    this.lastTime = now;
    const base = this.bodies[0];
    const riding = this.robot.visible && this.mode !== 'editor';
    if (riding && this.mode === 'first') {
      this.camera.position.copy(EYE_POSITION).applyQuaternion(base.quaternion).add(base.position);
      this.camera.quaternion.copy(base.quaternion).multiply(EYE_QUATERNION);
    } else if (riding) {
      // Behind the robot along the lane, which runs along +x. The camera keeps its height and follows the robot's
      // sideways drift only in part, so the picture stays calm while the robot leans, bounces and jumps.
      const wanted = new THREE.Vector3(base.position.x - 0.95, 0.6 * base.position.y, 0.46);
      const target = new THREE.Vector3(base.position.x + 0.7, 0.8 * base.position.y, 0.1);
      this.camera.position.lerp(wanted, blend);
      this.camera.position.x = wanted.x; // No lag along the lane: the robot stays at the same place in the picture.
      this.lookAt.lerp(target, blend);
      this.lookAt.x = target.x;
      this.camera.lookAt(this.lookAt);
      this.snap = false;
    } else {
      this.camera.position.lerp(this.editorPosition, blend);
      this.lookAt.lerp(this.editorTarget, blend);
      this.camera.lookAt(this.lookAt);
      this.snap = false;
    }
    // The robot hides its own body from its first-person view.
    for (const group of this.bodies) group.visible = this.mode !== 'first';
    const focus = this.mode === 'editor' ? this.lookAt : base.position;
    this.sun.position.set(focus.x - 5, focus.y - 7, 16);
    this.sun.target.position.copy(focus);
    this.renderer.render(this.scene, this.camera);
  }
}
