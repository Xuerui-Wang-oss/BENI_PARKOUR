import { defineConfig } from 'vite';

// Cross-origin isolation lets WebAssembly use threads (SharedArrayBuffer), which speeds up the depth encoder.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  server: { headers: isolation },
  preview: { headers: isolation },
  worker: { format: 'es' },
  // Both packages find their .wasm file next to their own script, which pre-bundling would move.
  optimizeDeps: { exclude: ['@mujoco/mujoco', 'onnxruntime-web'] },
  build: { target: 'es2022' },
});
