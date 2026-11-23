# Results: a model in the seat of the high-level policy

Runs of 2026-11-22 on two sets: the 16 held-out full courses of env2, and the 25 single-obstacle lanes of env1 (5
obstacle kinds x 5 rows; row 0 of every kind is a flat lane with no obstacle, rows 1 to 4 get harder). One attempt
per lane or course unless a table says otherwise, so most numbers below are single samples.

Each model gets the robot's own 80x60 depth frame and the 24 body numbers the PPO policy gets, and returns speed,
body height and the jump step. Nothing else reaches it at run time. Two things to keep in mind when comparing with
the trained policy: the depth table in the request is the same frame written as numbers, and the instructions hold
hand-written rules, including the distance at which to jump at each obstacle.

## Full courses: the 16 held-out test courses

Run on env2 row 4, the 16 courses the trained policy never saw in training, each with 4 to 6 obstacles in a row
(`ENV2_COURSES.md` lists them). One attempt per course.

| Driver | Goal reached | Obstacles cleared, of those it got to | Calls | Cost |
| --- | --- | --- | --- | --- |
| Scripted driver that reads the simulator (not a fair controller) | 15 of 16 | not counted | none | none |
| Trained PPO policy, one attempt per course (repository README) | 11 of 16 | not reported | none | none |
| Trained PPO policy, 5 attempts per course with a random start and goal (README) | 54 of 80, 68% | 88% of episodes cleared every obstacle | none | none |
| Claude Opus 5.5 | 10 of 16 | 62 of 68 | 1119 | $12.36 |
| Jev (TypeSafe) | 4 of 16 | 34 of 46 | 1688 | $0.25 |

The PPO rows are the repository's own figures; they were not re-run here.

| Course | Obstacles | Opus 5.5 | Jev |
| --- | --- | --- | --- |
| 0:4 | bar, pit, pit, ramp, box, tunnel | fell at obstacle 3 (pit) | in pit at obstacle 2 (pit) |
| 1:4 | bar, pit, tunnel, ramp, box, box | fell at obstacle 3 (tunnel) | in pit at obstacle 2 (pit) |
| 2:4 | bar, tunnel, tunnel, ramp, tunnel, ramp | goal | goal |
| 3:4 | bar, ramp, ramp, box, ramp, ramp | goal | fell at obstacle 2 (ramp) |
| 4:4 | box, box, tunnel, pit, bar | goal | body contact at obstacle 2 (box) |
| 5:4 | pit, bar, tunnel, bar, box, bar | in pit at obstacle 1 (pit) | fell at obstacle 3 (tunnel) |
| 6:4 | bar, bar, pit, pit | goal | stuck at obstacle 1 (bar) |
| 7:4 | pit, tunnel, pit, ramp | goal | goal |
| 8:4 | pit, tunnel, tunnel, bar | goal | body contact at obstacle 4 (bar) |
| 9:4 | bar, tunnel, pit, box, bar, bar | goal | body contact at obstacle 2 (tunnel) |
| 10:4 | box, tunnel, box, ramp, pit, ramp | in pit at obstacle 5 (pit) | body contact at obstacle 3 (box) |
| 11:4 | pit, box, tunnel, box | goal | body contact at obstacle 2 (box) |
| 12:4 | ramp, pit, box, ramp | goal | goal |
| 13:4 | ramp, box, ramp, bar, bar, bar | body contact at obstacle 6 (bar) | body contact at obstacle 2 (box) |
| 14:4 | pit, ramp, tunnel, box, ramp | goal | goal |
| 15:4 | tunnel, bar, pit, pit | body contact at obstacle 2 (bar) | body contact at obstacle 2 (bar) |

Opus read the scene correctly on all six courses it failed; the failures are in jump timing and landing:

- 5:4 and 10:4, pit: when the pit first showed, about 0.75 m away, it took the last visible floor row as the edge,
  which is about 0.10 m short of the real edge, scheduled the jump from that and jumped about 0.10 m early.
- 0:4, pit: jumped on time (0.42 m) and toppled on landing.
- 1:4: cleared the pit, landed badly and rolled over before the tunnel.
- 13:4 and 15:4, bar: recognised the bar, jumped at 0.55 to 0.67 m and came down on it.

Jev failed at a box on four courses, at a bar on three, at a pit on two, at a tunnel on two and at a ramp on one.

### Instructions v2 on the same 16 courses

Everything else in this file uses the first version of the instructions (`--prompt v1`). Version 2 (`--prompt v2`,
the default for the Claude models on the command line; the website still uses v1) changes the points the v1
failures showed: jump distances at the middle of the windows that scripted triggers cleared, one rule for box and
bar, when to schedule a trigger and when to wait for the next frame, where a pit's edge lies between two table
rows, how the end wall differs from an obstacle, and that the goal distance is capped. `prompt.py` lists them.

One attempt per course, same seed as the v1 runs:

| Driver | Goal reached, v1 | Goal reached, v2 | Obstacles cleared with v2, of those it got to | Calls | Cost |
| --- | --- | --- | --- | --- | --- |
| Claude Opus 5.5 | 10 of 16 | 14 of 16 | 74 of 76 | 1313 | $14.07 |
| Claude Haiku 5.5 | not run | 3 of 16 | 34 of 46 | 735 | $0.27 |
| Jev, with `--jev oneface` | 4 of 16 | 0 of 16 | 5 of 21 | 680 | $0.12 |

| Course | Opus, v1 | Opus, v2 | Haiku, v2 | Jev, v2 |
| --- | --- | --- | --- | --- |
| 0:4 | fell at obstacle 3 (pit) | goal | fell at obstacle 3 (pit) | in pit at obstacle 2 (pit) |
| 1:4 | fell at obstacle 3 (tunnel) | body contact at obstacle 5 (box) | body contact at obstacle 1 (bar) | body contact at obstacle 1 (bar) |
| 2:4 | goal | goal | goal | body contact at obstacle 1 (bar) |
| 3:4 | goal | goal | body contact at obstacle 4 (box) | fell at obstacle 1 (bar) |
| 4:4 | goal | goal | in pit at obstacle 4 (pit) | body contact at obstacle 1 (box) |
| 5:4 | in pit at obstacle 1 (pit) | in pit at obstacle 1 (pit) | fell at obstacle 2 (bar) | in pit at obstacle 1 (pit) |
| 6:4 | goal | goal | stuck at obstacle 2 (bar) | stuck at obstacle 1 (bar) |
| 7:4 | goal | goal | in pit at obstacle 1 (pit) | in pit at obstacle 1 (pit) |
| 8:4 | goal | goal | in pit at obstacle 1 (pit) | in pit at obstacle 1 (pit) |
| 9:4 | goal | goal | fell at obstacle 2 (tunnel) | fell at obstacle 1 (bar) |
| 10:4 | in pit at obstacle 5 (pit) | goal | body contact at obstacle 3 (box) | body contact at obstacle 1 (box) |
| 11:4 | goal | goal | stuck after the last obstacle | in pit at obstacle 1 (pit) |
| 12:4 | goal | goal | goal | body contact at obstacle 3 (box) |
| 13:4 | body contact at obstacle 6 (bar) | goal | goal | body contact at obstacle 2 (box) |
| 14:4 | goal | goal | in pit at obstacle 1 (pit) | in pit at obstacle 1 (pit) |
| 15:4 | body contact at obstacle 2 (bar) | goal | body contact at obstacle 2 (bar) | body contact at obstacle 2 (bar) |

- Opus kept every course it had passed and passed four of the six it had failed. Read this with care: v2 was
  written from Opus's failures on these same 16 courses, so the gain is flattering, and each number is one attempt.
  Course 5:4 shows the spread: a separate single run of Opus with v2 on it reached the goal
  (`runs/opus_v2_5-4/`), and in this run, from another start position, it dropped into the first pit.
- Jev got worse. With v2 it fired at pits 0.63 to 0.74 m before the edge (window 0.40 to 0.55 m) and at boxes and
  bars mostly 1.2 to 1.9 m away; with v1 its pit jumps came at 0.38 to 0.47 m. v2 adds a paragraph of distance
  arithmetic and gives the distances as ranges, which suits a model that reasons and not one that does not. Jev
  stays on v1.
- Haiku fails across all obstacle kinds, as on the single lanes.

Runs: `runs/v2_opus_env2_test/`, `runs/v2_haiku_env2_test/`, `runs/v2_jev_env2_test/`, each with `videos/` (16
clips, start to end of the episode).

## Score on the 25 single-obstacle lanes

| Driver | Goal reached | Calls | Cost | Median answer time | Asked every |
| --- | --- | --- | --- | --- | --- |
| Claude Opus 5.5 | 24 of 25 | 625 | $6.49 | 2.3 s | 0.25 s |
| Jev (TypeSafe) | 20 of 25 | 1472 | $0.22 | 0.15 s | 0.10 s |
| Claude Sonnet 5.5 | 19 of 25 | 606 | $3.17 | 2.2 s | 0.25 s |
| Scripted driver that reads the simulator (not a fair controller) | 25 of 25 | none | none | none | 0.25 s |

The simulation pauses while a model answers, so the answer time does not affect the score.

## Lane by lane

`goal` means the robot reached the goal. Anything else is how the episode ended.

| Lane | Opus 5.5 | Sonnet 5.5 | Jev |
| --- | --- | --- | --- |
| pit r0 (flat) | goal | goal | goal |
| pit r1 | goal | goal | goal |
| pit r2 | goal | goal | goal |
| pit r3 | goal | goal | in pit |
| pit r4 | in pit | goal | goal |
| ramp r0 (flat) | goal | goal | goal |
| ramp r1 | goal | goal | goal |
| ramp r2 | goal | goal | goal |
| ramp r3 | goal | goal | goal |
| ramp r4 | goal | goal | goal |
| box r0 (flat) | goal | goal | goal |
| box r1 | goal | goal | goal |
| box r2 | goal | goal | goal |
| box r3 | goal | body contact | fell |
| box r4 | goal | body contact | goal |
| tunnel r0 (flat) | goal | goal | goal |
| tunnel r1 | goal | goal | goal |
| tunnel r2 | goal | goal | goal |
| tunnel r3 | goal | body contact | goal |
| tunnel r4 | goal | body contact | goal |
| bar r0 (flat) | goal | goal | goal |
| bar r1 | goal | goal | goal |
| bar r2 | goal | body contact | body contact |
| bar r3 | goal | goal | body contact |
| bar r4 | goal | stuck | body contact |

In the code the kinds are named `pit`, `ramp_cube` (ramp), `cube` (box), `hollow` (tunnel) and `thin_cube` (bar).

## Why the runs failed

Read from the logged answers next to the simulator's ground truth.

Opus 5.5, 1 failure:

- pit r4: saw the pit at 0.75 m and scheduled the jump for about 0.55 m, 0.10 m before the tuned point of 0.45 m,
  and landed in the pit. Sonnet jumped at the same distance on this lane and cleared it.

Sonnet 5.5, 6 failures, five of them in reading the scene:

- tunnel r3 and r4: saw the slab's returns but called them "side walls" or "the end wall" and never crouched.
- box r4 and bar r4: called the obstacle "the end wall" and drove into it without a jump.
- box r3: saw the obstacle but scheduled the jump about 0.5 m too early.
- bar r2: took the bar for a box, jumped about 0.2 m early and hung on it.

Jev, 5 failures: three of the four bars (it does not pick the 1.2 m/s a bar needs and fires early), one box, one
pit.

## Jev: which setup is best

Nine setups, each on the 20 lanes with an obstacle x 3 attempts = 60 runs, 12 per obstacle kind. Each setup was run
once, so differences of a few runs are within the run-to-run variation.

| Setup | Goal reached | Pits | Boxes | Bars | Tunnels | Ramps |
| --- | --- | --- | --- | --- | --- | --- |
| Current: camera table as text, "now or none" jump question, asked every 0.10 s | 48 | 11 | 9 | 4 | 12 | 12 |
| Speed question explains each speed (`--jev barspeed`) | 45 | 8 | 8 | 9 | 11 | 9 |
| Obstacle question splits box and bar (`--jev split`) | 45 | 8 | 8 | 5 | 12 | 12 |
| Asked every 0.05 s (`--period 1`) | 41 | 7 | 6 | 4 | 12 | 12 |
| Floor table + wait/jump answers (`--table floor --jev ranges`) | 39 | 10 | 2 | 3 | 12 | 12 |
| Depth picture added (`--view depth`) | 39 | 8 | 7 | 4 | 12 | 8 |
| Floor table + wait/jump answers + speed explanations | 35 | 8 | 2 | 1 | 12 | 12 |
| Floor-relative table (`--table floor`) | 34 | 6 | 1 | 3 | 12 | 12 |
| Worked examples (`--jev examples`) | 34 | 3 | 3 | 4 | 12 | 12 |

Nothing beat the current setup on the 60 runs. Jev triggers a jump when it first sees an obstacle, not at a distance, so the
clearer floor table made it fire too early, and the worked examples made it fire late. The speed explanations did
not make it choose 1.2 m/s at bars (it stayed at 1.0 m/s on 97% of those calls).

### One jump rule for boxes and bars

A scripted sweep (no model; `cube` and `thin_cube` rows 1 to 4, trigger raised with the face 0.40 to 1.00 m from the
robot's root, one run each) showed that at 1.0 m/s a box and a bar of every height are cleared when the trigger comes
between 0.55 and 0.70 m. At 1.2 m/s the common window is 0.65 to 0.85 m. So a bar needs no speed of its own, and
Jev does not have to tell a bar from a box, which it cannot: within 1 m of a bar its obstacle answer was "box or bar"
on 1 of 167 calls.

`--jev oneface` puts that one rule in the jump question (a low face 0.61 m ahead of the camera or nearer). On the
box and bar lanes only, rows 1 to 4 x 3 attempts = 12 runs per kind:

| Setup | Boxes: cleared / goal | Bars: cleared / goal |
| --- | --- | --- |
| Current | 9 / 9 | 5 / 4 |
| `--jev oneface` | 8 / 7 | 11 / 8 |

Bars improved clearly; boxes are within the run-to-run variation. Its remaining box failures are jumps fired at the
first sight of a tall box, 1.5 m away. It has not been run on pits, tunnels and ramps, so the website still uses the
current setup. (A first version also reworded the speed question and made Jev drive past the goal on 5 of 24 runs;
that wording was taken back.)

## Where the data is

Under `python/runs/`, which git ignores:

| Run | Folder |
| --- | --- |
| Opus 5.5, 25 lanes | `runs/opus_env1_all25/` |
| Sonnet 5.5, 25 lanes | `runs/sonnet_env1_all25/` |
| Jev, 25 lanes | `runs/jev_env1_v2_p2/` |
| Opus 5.5, 16 full courses | `runs/opus_env2_test/` |
| Jev, 16 full courses | `runs/jev_env2_test/` |
| Jev setups, round 1 | `runs/bench/` (`base`, `floor`, `split`, `examples`, `every_step`, `picture`) |
| Jev setups, round 2 | `runs/bench2/` (`oneface_faces_v2` is the one-rule test; `oneface_faces` its first version) |

The six other folders in `runs/bench/` (`floor_ranges`, `floor_examples`, `base_ranges`, `base_barspeed`,
`floor_split_ranges_barspeed`, `floor_all`) are invalid: the TypeSafe account ran out of credits during them.

Each run folder holds `summary.json`, `episodes.json` (the outcome per lane) and `decisions.jsonl` (every call: the
text sent, the answer, the tokens, and the simulator's ground truth, which was logged but never sent). The Opus and
Sonnet folders also hold `frames/` (every depth picture sent). The five 25-lane and full-course folders of Opus,
Sonnet and Jev listed first, except `jev_env1_v2_p2`, hold `videos/`: one clip per lane or course, from the first
step to the end of the episode, with the robot from outside beside what the model was shown and what it answered.

## How to repeat

```
cd python
uv run python -m parkour_llm.run env1 --lanes all --model claude-opus-5-5 --max-calls 900 --out runs/NAME
uv run python -m parkour_llm.run env1 --lanes all --model claude-sonnet-5-5 --max-calls 900 --out runs/NAME
uv run python -m parkour_llm.run env1 --lanes all --model jev-latest --period 2 --out runs/NAME
uv run python -m parkour_llm.run env2 --lanes test --model claude-opus-5-5 --max-calls 1700 --out runs/NAME
uv run python -m parkour_llm.run env2 --lanes test --model jev-latest --period 2 --out runs/NAME
uv run python -m parkour_llm.run env1 --replay runs/NAME --videos --no-frames
```

The last line makes the videos of a finished run. It replays the logged decisions without calling a model, and the
seeded simulation gives the same episodes again.
