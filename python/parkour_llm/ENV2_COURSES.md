# The 80 full courses (env2)

The courses are not stored as files. `make_env2_terrain_cfg(seed=0)` in `python/parkour_rl/env2.py` generates them
with the lane builder in `python/parkour_rl/parkour_terrain.py` every time the simulation starts, and the fixed seed
gives the same 80 each time. This list was read from that generator on 2026-11-22.

A course is named `lane:row`. Rows 0 to 3 are the 64 training courses; row 4 is the 16 held-out test courses. Each
obstacle is given with its size and, in brackets, where it starts along the lane in metres. The robot starts near
1 m, and the goal is drawn 2 to 4 m past the last obstacle on every attempt.

To drive them: `--lanes all` (all 80), `--lanes test` (row 4) or `--lanes 7:4` (one course) with
`python -m parkour_llm.run env2`.

## Row 0: training courses

| Course | Obstacles in order |
| --- | --- |
| 0:0 | ramp 18 deg [3.8]; tunnel 0.26 m clearance [7.3]; box 0.10 m tall [11.3]; box 0.10 m tall [14.4]; pit 0.26 m long [18.8]; pit 0.34 m long [21.4] |
| 1:0 | pit 0.28 m long [3.5]; box 0.12 m tall [6.6]; ramp 13 deg [10.9]; pit 0.34 m long [16.5] |
| 2:0 | ramp 15 deg [3.4]; bar 0.10 m tall [8.8]; ramp 18 deg [11.0]; box 0.07 m tall [16.1] |
| 3:0 | tunnel 0.24 m clearance [3.7]; pit 0.34 m long [6.4]; ramp 9 deg [9.5]; tunnel 0.23 m clearance [13.6] |
| 4:0 | bar 0.12 m tall [3.9]; pit 0.34 m long [7.0]; bar 0.09 m tall [11.0]; pit 0.30 m long [14.5] |
| 5:0 | ramp 17 deg [3.6]; box 0.10 m tall [7.9]; ramp 16 deg [12.8]; tunnel 0.24 m clearance [18.2]; tunnel 0.20 m clearance [20.8]; pit 0.20 m long [24.3] |
| 6:0 | tunnel 0.25 m clearance [3.4]; tunnel 0.25 m clearance [6.7]; ramp 19 deg [9.5]; bar 0.10 m tall [14.5] |
| 7:0 | box 0.06 m tall [3.4]; box 0.11 m tall [7.5]; pit 0.31 m long [11.6]; tunnel 0.22 m clearance [14.6]; tunnel 0.22 m clearance [18.2]; pit 0.20 m long [21.8] |
| 8:0 | tunnel 0.23 m clearance [3.3]; box 0.09 m tall [6.6]; ramp 10 deg [10.8]; pit 0.31 m long [16.1]; tunnel 0.25 m clearance [18.7] |
| 9:0 | pit 0.29 m long [3.7]; tunnel 0.22 m clearance [7.3]; box 0.11 m tall [11.0]; ramp 11 deg [15.1]; bar 0.06 m tall [19.1]; tunnel 0.24 m clearance [22.6] |
| 10:0 | ramp 14 deg [3.2]; box 0.12 m tall [8.0]; bar 0.06 m tall [12.6]; tunnel 0.20 m clearance [15.3]; box 0.06 m tall [18.1]; bar 0.10 m tall [22.6] |
| 11:0 | ramp 15 deg [3.1]; pit 0.28 m long [8.4]; box 0.07 m tall [11.2]; ramp 8 deg [14.7]; bar 0.12 m tall [20.2]; box 0.11 m tall [23.9] |
| 12:0 | pit 0.26 m long [4.0]; tunnel 0.24 m clearance [6.3]; box 0.07 m tall [9.3]; box 0.09 m tall [13.0] |
| 13:0 | bar 0.09 m tall [3.0]; tunnel 0.25 m clearance [6.4]; tunnel 0.25 m clearance [9.6]; pit 0.30 m long [13.7]; ramp 11 deg [17.7] |
| 14:0 | ramp 9 deg [3.5]; tunnel 0.25 m clearance [8.7]; ramp 12 deg [11.1]; bar 0.09 m tall [16.4] |
| 15:0 | pit 0.15 m long [3.6]; pit 0.32 m long [7.5]; box 0.09 m tall [10.3]; box 0.09 m tall [13.9] |

## Row 1: training courses

| Course | Obstacles in order |
| --- | --- |
| 0:1 | pit 0.21 m long [3.7]; bar 0.09 m tall [7.2]; pit 0.16 m long [10.5]; tunnel 0.21 m clearance [13.4] |
| 1:1 | bar 0.09 m tall [3.7]; pit 0.26 m long [7.7]; tunnel 0.25 m clearance [10.2]; bar 0.09 m tall [13.9]; ramp 17 deg [17.9] |
| 2:1 | tunnel 0.20 m clearance [3.9]; bar 0.10 m tall [6.3]; bar 0.08 m tall [9.2]; pit 0.29 m long [12.1]; pit 0.34 m long [15.4]; pit 0.17 m long [19.7] |
| 3:1 | ramp 14 deg [3.1]; box 0.12 m tall [8.4]; pit 0.30 m long [12.7]; box 0.12 m tall [16.0]; box 0.07 m tall [19.3]; bar 0.06 m tall [23.7] |
| 4:1 | bar 0.07 m tall [3.4]; box 0.08 m tall [6.7]; box 0.06 m tall [9.9]; ramp 14 deg [14.8]; tunnel 0.23 m clearance [19.8] |
| 5:1 | ramp 19 deg [3.3]; bar 0.08 m tall [6.8]; pit 0.20 m long [9.0]; pit 0.31 m long [12.3]; box 0.10 m tall [15.0]; pit 0.24 m long [20.0] |
| 6:1 | pit 0.28 m long [3.4]; box 0.11 m tall [6.9]; tunnel 0.21 m clearance [11.3]; tunnel 0.24 m clearance [14.8] |
| 7:1 | tunnel 0.24 m clearance [3.5]; tunnel 0.24 m clearance [7.1]; pit 0.22 m long [10.6]; box 0.08 m tall [13.6] |
| 8:1 | pit 0.26 m long [3.3]; box 0.11 m tall [6.0]; box 0.09 m tall [10.3]; ramp 13 deg [13.9] |
| 9:1 | pit 0.17 m long [3.1]; tunnel 0.23 m clearance [6.8]; pit 0.28 m long [10.4]; bar 0.08 m tall [13.6]; tunnel 0.24 m clearance [16.0]; ramp 11 deg [19.6] |
| 10:1 | tunnel 0.24 m clearance [3.2]; tunnel 0.25 m clearance [6.4]; pit 0.33 m long [10.4]; ramp 14 deg [14.1]; pit 0.18 m long [18.2] |
| 11:1 | bar 0.08 m tall [3.9]; bar 0.10 m tall [6.6]; pit 0.32 m long [10.3]; tunnel 0.22 m clearance [12.8]; ramp 10 deg [16.9]; bar 0.12 m tall [22.4] |
| 12:1 | bar 0.09 m tall [3.2]; tunnel 0.25 m clearance [7.0]; pit 0.27 m long [10.6]; bar 0.09 m tall [13.3] |
| 13:1 | tunnel 0.24 m clearance [3.4]; pit 0.32 m long [7.3]; box 0.10 m tall [10.7]; ramp 14 deg [13.8]; ramp 14 deg [18.3]; bar 0.11 m tall [22.5] |
| 14:1 | ramp 13 deg [3.0]; pit 0.25 m long [8.6]; bar 0.10 m tall [11.9]; ramp 12 deg [15.5] |
| 15:1 | pit 0.16 m long [3.1]; box 0.09 m tall [5.5]; box 0.07 m tall [10.4]; ramp 9 deg [14.8]; pit 0.33 m long [20.2] |

## Row 2: training courses

| Course | Obstacles in order |
| --- | --- |
| 0:2 | bar 0.10 m tall [3.5]; bar 0.10 m tall [6.2]; bar 0.08 m tall [9.3]; box 0.07 m tall [13.2]; ramp 17 deg [18.1] |
| 1:2 | bar 0.12 m tall [3.2]; bar 0.07 m tall [6.9]; pit 0.34 m long [10.9]; pit 0.33 m long [13.8]; bar 0.11 m tall [17.2]; pit 0.25 m long [20.1] |
| 2:2 | ramp 18 deg [3.3]; ramp 19 deg [7.1]; pit 0.25 m long [12.3]; box 0.08 m tall [16.2]; bar 0.12 m tall [20.6] |
| 3:2 | box 0.09 m tall [3.5]; box 0.11 m tall [8.5]; pit 0.17 m long [11.8]; ramp 15 deg [15.9]; pit 0.19 m long [21.1]; tunnel 0.25 m clearance [24.2] |
| 4:2 | box 0.06 m tall [3.8]; bar 0.10 m tall [8.8]; tunnel 0.23 m clearance [11.6]; bar 0.11 m tall [14.2]; ramp 14 deg [17.1] |
| 5:2 | bar 0.07 m tall [3.5]; pit 0.30 m long [6.5]; ramp 16 deg [8.8]; tunnel 0.24 m clearance [12.8]; tunnel 0.20 m clearance [17.0]; box 0.08 m tall [19.6] |
| 6:2 | bar 0.08 m tall [4.0]; ramp 11 deg [7.4]; ramp 16 deg [12.3]; tunnel 0.24 m clearance [17.5]; tunnel 0.21 m clearance [20.7] |
| 7:2 | tunnel 0.25 m clearance [3.0]; tunnel 0.20 m clearance [6.3]; bar 0.11 m tall [8.9]; pit 0.34 m long [13.0]; tunnel 0.22 m clearance [17.1] |
| 8:2 | bar 0.09 m tall [3.1]; pit 0.28 m long [5.5]; box 0.11 m tall [7.9]; tunnel 0.21 m clearance [12.0] |
| 9:2 | box 0.08 m tall [3.0]; ramp 19 deg [6.4]; bar 0.10 m tall [10.3]; pit 0.20 m long [13.9]; pit 0.34 m long [17.1] |
| 10:2 | ramp 13 deg [3.4]; box 0.11 m tall [8.0]; tunnel 0.20 m clearance [11.4]; box 0.10 m tall [15.6] |
| 11:2 | ramp 12 deg [3.0]; pit 0.30 m long [6.9]; ramp 15 deg [9.6]; box 0.07 m tall [13.8]; tunnel 0.24 m clearance [16.8] |
| 12:2 | pit 0.23 m long [3.2]; tunnel 0.25 m clearance [7.0]; pit 0.33 m long [9.3]; pit 0.20 m long [12.8]; bar 0.06 m tall [15.3] |
| 13:2 | bar 0.09 m tall [3.3]; tunnel 0.23 m clearance [7.0]; pit 0.30 m long [10.8]; bar 0.12 m tall [14.2]; tunnel 0.23 m clearance [16.6]; pit 0.33 m long [19.7] |
| 14:2 | tunnel 0.21 m clearance [3.9]; bar 0.10 m tall [7.5]; bar 0.07 m tall [10.1]; bar 0.11 m tall [13.3]; tunnel 0.24 m clearance [16.9] |
| 15:2 | tunnel 0.23 m clearance [3.5]; pit 0.23 m long [7.0]; pit 0.18 m long [10.2]; bar 0.07 m tall [13.6] |

## Row 3: training courses

| Course | Obstacles in order |
| --- | --- |
| 0:3 | box 0.10 m tall [3.6]; ramp 12 deg [6.8]; tunnel 0.22 m clearance [12.2]; box 0.11 m tall [16.0]; box 0.07 m tall [19.5] |
| 1:3 | pit 0.27 m long [3.9]; bar 0.06 m tall [6.3]; ramp 17 deg [10.1]; pit 0.15 m long [13.8]; bar 0.10 m tall [16.6]; ramp 14 deg [19.6] |
| 2:3 | tunnel 0.24 m clearance [3.7]; bar 0.11 m tall [7.4]; tunnel 0.21 m clearance [9.8]; bar 0.07 m tall [12.9] |
| 3:3 | bar 0.06 m tall [3.5]; tunnel 0.25 m clearance [6.4]; ramp 13 deg [10.5]; ramp 12 deg [14.3]; ramp 20 deg [19.5]; tunnel 0.25 m clearance [24.4] |
| 4:3 | ramp 10 deg [3.6]; ramp 17 deg [8.4]; ramp 11 deg [13.7]; tunnel 0.21 m clearance [17.6]; ramp 18 deg [21.3] |
| 5:3 | bar 0.09 m tall [3.7]; pit 0.32 m long [7.6]; bar 0.10 m tall [11.6]; ramp 18 deg [14.0]; box 0.10 m tall [19.0] |
| 6:3 | ramp 17 deg [3.6]; tunnel 0.25 m clearance [8.9]; ramp 16 deg [12.0]; box 0.08 m tall [15.9]; box 0.10 m tall [20.6] |
| 7:3 | bar 0.11 m tall [3.8]; pit 0.28 m long [6.9]; ramp 11 deg [9.5]; pit 0.34 m long [13.9]; tunnel 0.23 m clearance [17.4] |
| 8:3 | tunnel 0.22 m clearance [3.9]; pit 0.24 m long [7.1]; pit 0.19 m long [10.1]; bar 0.08 m tall [13.5]; box 0.07 m tall [16.9]; pit 0.21 m long [21.2] |
| 9:3 | bar 0.07 m tall [3.8]; tunnel 0.21 m clearance [6.3]; box 0.11 m tall [9.2]; box 0.09 m tall [12.9]; tunnel 0.25 m clearance [16.9]; ramp 15 deg [20.2] |
| 10:3 | box 0.11 m tall [3.9]; pit 0.24 m long [8.8]; bar 0.11 m tall [12.6]; tunnel 0.23 m clearance [15.0]; ramp 19 deg [17.5]; tunnel 0.24 m clearance [22.2] |
| 11:3 | tunnel 0.22 m clearance [4.0]; bar 0.12 m tall [7.0]; bar 0.08 m tall [9.3]; box 0.09 m tall [12.7]; pit 0.21 m long [17.1] |
| 12:3 | ramp 13 deg [3.8]; ramp 9 deg [7.5]; tunnel 0.25 m clearance [12.8]; box 0.11 m tall [16.6]; tunnel 0.24 m clearance [20.3]; ramp 14 deg [24.3] |
| 13:3 | box 0.08 m tall [3.0]; bar 0.09 m tall [6.7]; bar 0.09 m tall [8.8]; pit 0.22 m long [11.9]; ramp 13 deg [15.3]; ramp 15 deg [19.1] |
| 14:3 | ramp 19 deg [3.8]; bar 0.10 m tall [9.0]; box 0.08 m tall [11.3]; box 0.12 m tall [14.9] |
| 15:3 | tunnel 0.21 m clearance [3.2]; bar 0.08 m tall [5.7]; tunnel 0.23 m clearance [8.4]; tunnel 0.25 m clearance [12.1]; box 0.09 m tall [15.6]; box 0.06 m tall [19.9] |

## Row 4: held-out test courses

| Course | Obstacles in order |
| --- | --- |
| 0:4 | bar 0.09 m tall [3.1]; pit 0.31 m long [6.3]; pit 0.20 m long [9.3]; ramp 9 deg [12.8]; box 0.08 m tall [17.2]; tunnel 0.21 m clearance [22.1] |
| 1:4 | bar 0.11 m tall [3.8]; pit 0.20 m long [7.0]; tunnel 0.21 m clearance [9.8]; ramp 11 deg [12.9]; box 0.10 m tall [18.3]; box 0.09 m tall [22.6] |
| 2:4 | bar 0.09 m tall [3.8]; tunnel 0.23 m clearance [7.8]; tunnel 0.20 m clearance [11.3]; ramp 20 deg [14.2]; tunnel 0.23 m clearance [18.0]; ramp 8 deg [21.7] |
| 3:4 | bar 0.07 m tall [3.0]; ramp 12 deg [6.6]; ramp 13 deg [12.0]; box 0.09 m tall [15.9]; ramp 9 deg [20.3]; ramp 13 deg [25.9] |
| 4:4 | box 0.08 m tall [3.0]; box 0.08 m tall [6.3]; tunnel 0.26 m clearance [11.1]; pit 0.19 m long [15.0]; bar 0.09 m tall [18.3] |
| 5:4 | pit 0.34 m long [3.8]; bar 0.07 m tall [6.5]; tunnel 0.20 m clearance [9.8]; bar 0.08 m tall [14.0]; box 0.12 m tall [17.5]; bar 0.08 m tall [22.5] |
| 6:4 | bar 0.07 m tall [3.2]; bar 0.10 m tall [5.5]; pit 0.31 m long [8.6]; pit 0.27 m long [12.6] |
| 7:4 | pit 0.30 m long [3.5]; tunnel 0.24 m clearance [7.8]; pit 0.21 m long [11.6]; ramp 17 deg [15.5] |
| 8:4 | pit 0.31 m long [3.3]; tunnel 0.23 m clearance [7.5]; tunnel 0.21 m clearance [10.7]; bar 0.07 m tall [14.8] |
| 9:4 | bar 0.07 m tall [3.8]; tunnel 0.20 m clearance [6.8]; pit 0.21 m long [9.4]; box 0.08 m tall [12.3]; bar 0.12 m tall [16.1]; bar 0.06 m tall [18.7] |
| 10:4 | box 0.11 m tall [3.7]; tunnel 0.21 m clearance [7.9]; box 0.09 m tall [12.1]; ramp 13 deg [16.0]; pit 0.31 m long [21.6]; ramp 14 deg [24.7] |
| 11:4 | pit 0.23 m long [3.6]; box 0.09 m tall [6.2]; tunnel 0.26 m clearance [10.6]; box 0.11 m tall [14.4] |
| 12:4 | ramp 12 deg [3.6]; pit 0.28 m long [8.3]; box 0.07 m tall [11.0]; ramp 11 deg [15.8] |
| 13:4 | ramp 13 deg [3.7]; box 0.11 m tall [7.3]; ramp 11 deg [10.7]; bar 0.10 m tall [15.0]; bar 0.09 m tall [18.5]; bar 0.09 m tall [22.0] |
| 14:4 | pit 0.29 m long [3.0]; ramp 18 deg [6.4]; tunnel 0.24 m clearance [11.6]; box 0.08 m tall [14.1]; ramp 15 deg [17.8] |
| 15:4 | tunnel 0.23 m clearance [3.4]; bar 0.09 m tall [7.4]; pit 0.20 m long [10.1]; pit 0.32 m long [14.1] |
