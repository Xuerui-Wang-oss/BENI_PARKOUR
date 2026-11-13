# Third-party components

This repository contains or builds on the following work.

| Component | What is here | License |
| --- | --- | --- |
| [OpenBeni](https://github.com/Mondo-Robotics/open-beni), © 2026 Open Beni | the source this repository starts from: robot model, meshes and textures, the low-level training code, the browser game in `web/`, and the released policy that serves as the jump policy (`python/parkour_rl/low_level_policy/model_jump.pt`, `parkour_web/public/models/jump.onnx`) | MIT ([LICENSE](LICENSE)) |
| [DeFM](https://github.com/leggedrobotics/defm), © 2026 ETH Zurich, Manthan Patel | the depth encoder: RegNetY-400MF weights (`python/parkour_rl/depth_model/defm_regnet_y_400mf.pth`, and converted to ONNX in `parkour_web/public/models/defm.onnx`) and its preprocessing and model code, copied into `python/parkour_rl/depth.py` | Apache-2.0 ([python/parkour_rl/depth_model/LICENSE](python/parkour_rl/depth_model/LICENSE)) |

The web demo in `parkour_web/` also uses [MuJoCo](https://github.com/google-deepmind/mujoco) as WebAssembly
(Apache-2.0), [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) (MIT) and
[three.js](https://github.com/mrdoob/three.js) (MIT). They are installed by `npm ci` and are not part of this
repository; `parkour_web/space/NOTICE.md` lists them for the published page, which bundles them.
