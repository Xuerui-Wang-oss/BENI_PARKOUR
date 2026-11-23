"""Two pictures: where the 68% comes from, and the 16 test courses drawn from the side."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon, Rectangle

courses = json.load(open(sys.argv[1])); out = sys.argv[2]
SURFACE, INK, MUTED, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c9c8c0"
COLOR = {"pit": "#2a78d6", "cube": "#eb6834", "thin_cube": "#1baf7a", "hollow": "#eda100", "ramp_cube": "#e87ba4"}
NAME = {"pit": "pit", "cube": "box", "thin_cube": "bar", "hollow": "tunnel", "ramp_cube": "ramp"}
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 11, "text.color": INK, "savefig.facecolor": SURFACE})

# ---------------------------------------------------------------- picture 1: where the 68% comes from
fig, ax = plt.subplots(figsize=(14, 6.4)); fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
ax.set_xlim(0, 41.5); ax.set_ylim(-1.2, 11.4); ax.axis("off")
ax.text(0, 10.9, "Where the 68% comes from", fontsize=20, fontweight="bold", va="top")
ax.text(0, 9.75, "There are 80 different courses. The 68% was measured on only 16 of them, each driven 5 times.",
        fontsize=12.5, color=MUTED, va="top")
# Left: the 80 courses, 16 lanes x 5 rows.
x0, y0, s, g = 2.4, 2.0, 0.86, 0.14
ax.text(x0, 8.55, "The 80 courses", fontsize=14, fontweight="bold")
ax.text(x0, 8.0, "one square = one course (4 to 6 obstacles in a row)", fontsize=10.5, color=MUTED)
for row in range(5):
    y = y0 + (4 - row) * (s + g)
    test = row == 4
    for lane in range(16):
        ax.add_patch(Rectangle((x0 + lane * (s + g), y), s, s, facecolor=COLOR["pit"] if test else NEUTRAL, edgecolor="none"))
    ax.text(x0 - 0.3, y + s / 2, f"row {row}", ha="right", va="center", fontsize=10.5, color=MUTED)
for lane in (0, 5, 10, 15):
    ax.text(x0 + lane * (s + g) + s / 2, y0 + 5 * (s + g) + 0.12, f"lane {lane}", ha="center", fontsize=9.5, color=MUTED)
top, bottom = y0 + 5 * (s + g) - g, y0 + (s + g)
ax.plot([x0 + 16 * (s + g) + 0.15] * 2, [bottom, top], color=MUTED, lw=1.2)
ax.text(x0 + 16 * (s + g) + 0.4, (top + bottom) / 2, "64 training\ncourses", va="center", fontsize=11.5)
ax.text(x0, y0 - 0.55, "16 test courses (row 4): never seen in training", fontsize=11.5, fontweight="bold", va="top")
# Arrow.
ax.add_patch(FancyArrowPatch((x0 + 16 * (s + g) + 0.3, y0 + s / 2), (25.3, y0 + s / 2 + 1.4), arrowstyle="-|>", mutation_scale=18,
                             color=INK, lw=1.4, connectionstyle="arc3,rad=-0.25"))
ax.text(21.2, y0 - 0.25, "each driven 5 times,\nwith a random start\nand goal", fontsize=10.5, color=MUTED, ha="center", va="top")
# Right: the 80 runs, 54 reached the goal.
rx, ry, rw, rh, px, py = 26.0, 2.0, 0.74, 0.51, 0.92, 0.70
ax.text(rx, 8.55, "The 80 test runs", fontsize=14, fontweight="bold")
ax.text(rx, 8.0, "one circle = one run; 16 test courses x 5 attempts", fontsize=10.5, color=MUTED)
for i in range(80):
    col, r = i % 16, i // 16
    cx, cy = rx + col * px + rw / 2, ry + (4 - r) * py + rh / 2 + 1.2
    ok = i < 54
    ax.add_patch(Ellipse((cx, cy), rw, rh, facecolor=COLOR["pit"] if ok else SURFACE, edgecolor=COLOR["pit"] if ok else MUTED, lw=1.3))
ax.text(rx, ry + 0.25, "54 reached the goal", fontsize=12.5, fontweight="bold", va="top")
ax.text(rx + 7.0, ry + 0.25, "(filled)    26 did not (hollow)", fontsize=11.5, color=MUTED, va="top")
ax.text(rx, ry - 0.55, "54 of 80 = 68%", fontsize=21, fontweight="bold", va="top")
ax.text(0, -0.95, "The circles show the total only: the repository reports 54 of 80, not which run on which course succeeded. "
        "Source: BENI_PARKOUR README, results table.", fontsize=9.5, color=MUTED)
fig.savefig(f"{out}/where_the_68_percent_comes_from.png", dpi=170, bbox_inches="tight", pad_inches=0.3)

# ---------------------------------------------------------------- picture 2: the 16 test courses from the side
V = 2.6  # Drawn height per metre of real height: the courses are 20-30 m long and the obstacles 0.1-0.3 m tall.
PITCH = 2.4
TOP = 7.2
results = {name: {e["lane"]: e for e in json.load(open(path))} for name, path in
           (("Opus 5.5", "runs/opus_env2_test/episodes.json"), ("Jev", "runs/jev_env2_test/episodes.json"))}
end = max(o["x_end"] for lane in range(16) for o in courses[f"{lane}:4"]) + 4
fig, ax = plt.subplots(figsize=(17, 14.4)); fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
ax.set_xlim(-3.2, end + 16.5); ax.set_ylim(-16 * PITCH - 2.4, TOP); ax.axis("off")
ax.text(-3.2, TOP - .1, "The 16 test courses, seen from the side", fontsize=21, fontweight="bold", va="top")
ax.text(-3.2, TOP - 1.75, "These are the courses behind the 68%. The robot drives left to right. Heights are stretched and short "
        "obstacles widened so they can be seen.", fontsize=12.5, color=MUTED, va="top")
# Legend: one sample of each kind, drawn as in the courses.
def draw(kind, x0, x1, value, base):
    c = COLOR[kind]
    if kind == "pit":
        w = max(x1 - x0, .45); ax.add_patch(Rectangle((x0, base - .45), w, .45, facecolor=c, edgecolor="none")); return x0 + w / 2
    if kind == "cube":
        ax.add_patch(Rectangle((x0, base), x1 - x0, value * V, facecolor=c, edgecolor="none")); return (x0 + x1) / 2
    if kind == "thin_cube":
        ax.add_patch(Rectangle((x0, base), .32, value * V, facecolor=c, edgecolor="none")); return x0 + .16
    if kind == "hollow":
        ax.add_patch(Rectangle((x0, base + value * V), .5, .1 * V, facecolor=c, edgecolor="none")); return x0 + .25
    foot = x1 - 1.0  # ramp_cube: a slope up to a 1 m platform, 0.15 m high.
    ax.add_patch(Polygon([(x0, base), (foot, base + .15 * V), (x1, base + .15 * V), (x1, base)], facecolor=c, edgecolor="none"))
    return (x0 + x1) / 2
lx = -3.2
LEGEND = TOP - 4.3
for kind, text, value, length in (("pit", "pit: jump the gap", .3, .3), ("cube", "box: jump onto it", .1, 1.),
                                  ("thin_cube", "bar: jump over it", .1, .1), ("hollow", "tunnel: crouch under", .23, .3),
                                  ("ramp_cube", "ramp: drive up", 14, 1.6)):
    wide = kind in ("cube", "ramp_cube")
    if kind == "pit":
        ax.plot([lx, lx + .8], [LEGEND] * 2, color=INK, lw=1.2); ax.plot([lx + 1.25, lx + 2.0], [LEGEND] * 2, color=INK, lw=1.2)
    else:
        ax.plot([lx, lx + 2.0], [LEGEND] * 2, color=INK, lw=1.2, zorder=3)
    draw(kind, lx + (.2 if wide else .8), lx + (.2 if wide else .8) + length, value, LEGEND)
    ax.text(lx + 2.35, LEGEND + .2, text, fontsize=11.5, va="center")
    lx += 9.3
for lane in range(16):
    base = -(lane + 1) * PITCH
    obstacles = courses[f"{lane}:4"]
    last = obstacles[-1]["x_end"]
    ax.add_patch(Rectangle((last + 2, base - .12), 2, .5, facecolor=GRID, edgecolor="none"))
    # Floor, broken at the pits.
    x = 0.
    for o in obstacles:
        if o["kind"] == "pit":
            ax.plot([x, o["x_start"]], [base, base], color=INK, lw=1.2, solid_capstyle="butt"); x = o["x_start"] + max(o["x_end"] - o["x_start"], .45)
    ax.plot([x, last + 4], [base, base], color=INK, lw=1.2, solid_capstyle="butt")
    for o in obstacles:
        cx = draw(o["kind"], o["x_start"], o["x_end"], o["value"], base)
        ax.text(cx, base + 1.0, NAME[o["kind"]], ha="center", va="bottom", fontsize=9.5, color=MUTED)
    ax.plot([1], [base + .16], marker=">", color=INK, ms=7)
    ax.text(-0.35, base + .2, f"{lane}:4", ha="right", va="center", fontsize=12, fontweight="bold")
    for i, (name, table) in enumerate(results.items()):
        e = table[f"lane {lane} r4"]; ok = e["outcome"] == "goal_reached"
        text = "reached the goal" if ok else f"failed at obstacle {e['passed'] + 1} ({NAME[e['kinds'][e['passed']]]})"
        ax.text(end + .8 + i * 8.0, base + .2, text, va="center", fontsize=10.5, color=INK if ok else MUTED,
                fontweight="bold" if ok else "normal")
for i, name in enumerate(results):
    ax.text(end + .8 + i * 8.0, -PITCH + 1.1, name, fontsize=12.5, fontweight="bold")
ax.text(end + .8, -PITCH + 1.95, "Our runs, one attempt per course, 2026-11-22", fontsize=10, color=MUTED)
ax.plot([1], [-16 * PITCH - 2.1], marker=">", color=INK, ms=7); ax.text(1.5, -16 * PITCH - 2.1, "start", va="center", fontsize=10.5, color=MUTED)
ax.add_patch(Rectangle((5.2, -16 * PITCH - 2.35), 2, .5, facecolor=GRID, edgecolor="none"))
ax.text(7.5, -16 * PITCH - 2.1, "the goal is placed somewhere in this band, 2 to 4 m past the last obstacle, anew on every attempt",
        va="center", fontsize=10.5, color=MUTED)
for m in range(0, int(end) + 1, 5):
    ax.text(m, -16 * PITCH - 1.35, f"{m} m", ha="center", fontsize=9.5, color=MUTED)
    ax.plot([m, m], [-16 * PITCH - .8, -16 * PITCH - .62], color=MUTED, lw=1)
fig.savefig(f"{out}/the_16_test_courses.png", dpi=150, bbox_inches="tight", pad_inches=0.3)
