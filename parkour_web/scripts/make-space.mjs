// Put the built page into space/, the folder of the Hugging Face Space (a static Space: Hugging Face only serves
// the files). space/ keeps README.md (the Space's settings), LICENSE and NOTICE.md; everything else in it is
// replaced by the contents of dist/. Run after `vite build` (npm run space does both).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const space = path.join(root, 'space');
const KEEP = new Set(['README.md', 'LICENSE', 'NOTICE.md']);
for (const name of fs.readdirSync(space)) {
  if (!KEEP.has(name)) fs.rmSync(path.join(space, name), { recursive: true, force: true });
}
fs.cpSync(path.join(root, 'dist'), space, { recursive: true });
const files = fs.readdirSync(space, { recursive: true }).filter((name) => fs.statSync(path.join(space, name)).isFile());
const bytes = files.reduce((sum, name) => sum + fs.statSync(path.join(space, name)).size, 0);
console.log(`space/: ${files.length} files, ${(bytes / 1e6).toFixed(1)} MB`);
