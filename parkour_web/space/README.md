---
title: BENI Parkour
emoji: 🤖
colorFrom: blue
colorTo: red
sdk: static
app_file: index.html
license: mit
short_description: Build an obstacle course; a learned policy drives BENI
---

# BENI Parkour

Build a course of 4 to 6 obstacles (pits, cubes, bars, tunnels, ramps) and watch the BENI robot cross it.

Everything runs in your browser: the MuJoCo physics (WebAssembly), the depth camera, and four neural networks
(ONNX Runtime Web). A high-level policy reads the robot's depth image 20 times a second and decides how fast to
drive, how low to crouch and when to jump; two frozen low-level policies turn that into motor commands. The robot
gets no map of the course. It crosses about three of four courses built at random from these obstacles.

The robot and its low-level skills come from [OpenBeni](https://github.com/Mondo-Robotics/open-beni). The high-level
parkour policy was trained in simulation on top of them. See `NOTICE.md` for the third-party components.
