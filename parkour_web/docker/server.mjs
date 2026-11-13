// Static file server for the built page: serves the folder dist/ next to this file. Node only, no dependencies.
//   PORT=7860 node server.mjs
// Everything runs in the visitor's browser; this server only hands out files. The Hugging Face Space does not use
// it (a static Space serves the files itself); it is for hosting the page in a container (see Dockerfile).
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), 'dist');
const port = Number(process.env.PORT ?? 7860);
const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css',
  '.json': 'application/json', '.wasm': 'application/wasm', '.onnx': 'application/octet-stream',
  '.xml': 'application/xml', '.obj': 'text/plain', '.png': 'image/png', '.svg': 'image/svg+xml',
};

http.createServer((request, response) => {
  const url = new URL(request.url, 'http://localhost');
  let file = path.normalize(path.join(root, decodeURIComponent(url.pathname)));
  if (!file.startsWith(root)) {
    response.writeHead(403).end();
    return;
  }
  if (fs.existsSync(file) && fs.statSync(file).isDirectory()) file = path.join(file, 'index.html');
  if (!fs.existsSync(file)) {
    response.writeHead(404, { 'Content-Type': 'text/plain' }).end('Not found');
    return;
  }
  response.writeHead(200, {
    'Content-Type': TYPES[path.extname(file)] ?? 'application/octet-stream',
    'Content-Length': fs.statSync(file).size,
    // The files under assets/ carry a hash of their content in their name.
    'Cache-Control': url.pathname.startsWith('/assets/') ? 'public, max-age=31536000, immutable' : 'no-cache',
    // Cross-origin isolation: lets WebAssembly use threads where the page asks for them.
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
    'Cross-Origin-Resource-Policy': 'same-origin',
  });
  if (request.method === 'HEAD') response.end();
  else fs.createReadStream(file).pipe(response);
}).listen(port, '0.0.0.0', () => console.log(`BENI Parkour on port ${port}`));
