// Copy the robot's visual meshes and texture into public/, with the transform of each mesh into its body frame
// (from assets/beni_base/mjcf/beni.xml).
// Runs before `npm run dev` and `npm run build`. The networks (public/models) and the physics model
// (public/robot/beni.xml) are written by the Python tools in tools/.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const assets = path.resolve(root, '../assets/beni_base');
const copy = (from, to) => {
  fs.mkdirSync(path.dirname(to), { recursive: true });
  fs.copyFileSync(from, to);
};

// The wheels are drawn as cylinders: their meshes are 8 MB each.
const BODY_OF_MESH = { BASE_VIS: 'base_link', HEAD_VIS: 'head', L1_VIS: 'L1_Link', L2_VIS: 'L2_Link', R1_VIS: 'R1_Link', R2_VIS: 'R2_Link' };
const xml = fs.readFileSync(path.join(assets, 'mjcf/beni.xml'), 'utf8');
const numbers = (text) => text.trim().split(/\s+/).map(Number);
const meshes = [];
for (const [, name, file, refpos, refquat] of xml.matchAll(
  /<mesh name="(\w+)" file="([^"]+)"[^>]*?refpos="([^"]+)"\s+refquat="([^"]+)"/gs)) {
  if (!(name in BODY_OF_MESH)) continue;
  copy(path.join(assets, 'meshes', file), path.join(root, 'public/robot/visual', file));
  meshes.push({ body: BODY_OF_MESH[name], file, refpos: numbers(refpos), refquat: numbers(refquat) });
}
if (meshes.length !== Object.keys(BODY_OF_MESH).length) throw new Error(`found ${meshes.length} visual meshes`);
copy(path.join(assets, 'textures/beni_white.png'), path.join(root, 'public/robot/visual/beni_white.png'));
fs.writeFileSync(path.join(root, 'public/robot/visual/meshes.json'), JSON.stringify(meshes, null, 1));
console.log(`prepared public/robot/visual (${meshes.length} meshes)`);
