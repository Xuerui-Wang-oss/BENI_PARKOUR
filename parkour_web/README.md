# BENI Parkour in the browser

A web page where a visitor builds a course of 4 to 6 obstacles and watches the BENI robot cross it, driven by the
high-level parkour policy of `python/parkour_rl` (env2, `model_1000.pt`) through the frozen jump and crawl policies.
Everything runs in the visitor's browser: MuJoCo (WebAssembly) for the physics, ray casting for the depth camera,
ONNX Runtime Web for the four networks, three.js for the picture. A server only hands out files.

The simulation code is a port of the Python hierarchy and is checked against it step by step (see Checks).

## Layout

    src/sim/        the simulation, without any browser API (also runs in Node)
      terrain.js      lane geometry                 (python/parkour_rl/parkour_terrain.py)
      robot.js        MuJoCo model, motors, sensors, low-level observation
      jump.js         jump command state machine    (python/beni_rl/jump.py)
      hierarchy.js    commands, trigger latch, terminations, high-level observation
                                                    (python/parkour_rl/hierarchy.py, env.py, goal.py, reward.py)
      depth.js        depth camera by ray casting
      policies.js     the ONNX networks
      episode.js      one episode: perception at 20 Hz, control at 100 Hz
    src/worker.js   the simulation in a Web Worker, paced to real time
    src/app.js      the page: three-step course editor, run view, result
    src/view/       the three.js scene
    public/models   the four ONNX networks          (written by tools/export_onnx.py)
    public/robot    physics model and visual meshes (tools/export_robot.py, scripts/prepare-assets.mjs)
    tools/          exporters (Python) and checks (Node)
    space/          the Hugging Face Space (static): settings, licence, and the built page
    docker/         Dockerfile and file server for hosting the page in a container

## Setup

Node 22 from the conda environment `beni-web`:

    export PATH=$HOME/miniconda3/envs/beni-web/bin:$PATH
    npm ci

The Python tools run in the training environment (`beni-mjlab`), from `open-beni/python`:

    PYTHONPATH=. python ../parkour_web/tools/export_robot.py       # public/robot/beni.xml, tools/ref/env2_lanes.json
    PYTHONPATH=. python ../parkour_web/tools/export_onnx.py        # public/models/*.onnx, tools/ref/onnx_vectors.json
    PYTHONPATH=. python ../parkour_web/tools/record_reference.py   # tools/ref/rollouts: 8 recorded Python episodes

`export_onnx.py` takes another high-level checkpoint as its argument.

## Run

    npm run dev        # http://localhost:5173
    npm run build      # dist/
    npm run space      # build, and copy the built page into space/ for the Hugging Face Space

## Checks

Each check compares one part of the JavaScript code with the Python original and prints the largest differences.

    node tools/check_mujoco.mjs       # the MuJoCo build loads, compiles, steps and casts rays
    node tools/check_terrain.mjs      # lane builder: all 80 env2 lanes, box by box
    node tools/check_onnx.mjs         # the networks against PyTorch on test vectors, and their speed
    node tools/check_robot.mjs        # observation, motor torques, one physics step, contact sensors
    node tools/check_hierarchy.mjs    # the whole control logic on recorded episodes, every control step
    node tools/check_perception.mjs   # depth image, depth encoder, policies on recorded inputs
    node tools/run_episodes.mjs 2     # closed loop on the 80 env2 lanes, the counts of parkour_rl.evaluate

The last four need the recorded episodes of `record_reference.py`.

## Deploy

The page is published as a static Hugging Face Space, https://huggingface.co/spaces/XueruiWang/beni-parkour
(Hugging Face only serves the files; Docker Spaces need a paid plan). `space/` is the folder of that Space: its
`README.md` (the Space's settings), `LICENSE`, `NOTICE.md` and, after `npm run space`, the built page. To publish
a new version, upload the folder as it is; `--delete` removes the scripts of the previous build, whose names
carry a hash of their content. The `hf` command is in the conda environment `beni-hf`.

    npm run space
    hf upload XueruiWang/beni-parkour space . --repo-type space --delete "assets/*"

`docker/` holds a `Dockerfile` and a dependency-free file server for hosting the page in a container instead.
