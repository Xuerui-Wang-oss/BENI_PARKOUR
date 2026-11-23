"""Draw the control stack with an LLM in the seat of the high-level policy: ``llm_architecture.png`` and ``.svg``.

    python parkour_llm/figures/make_architecture.py

The layout follows the diagram of the trained stack (depth image and body state on the left, the high-level policy
in the middle, the frozen crawl and jump policies and the joint actions on the right) so the two can be shown side
by side. The depth picture is a frame that was sent to the model in a run.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

HERE = Path(__file__).parent
FRAME = HERE.parents[1] / "runs/v1_opus_ppo_starts/frames/env07_ep1_call024_depth.png"
W, H = 2420, 860  # Drawing units; y runs down, as in the diagram this one mirrors.

INK, MUTED = "#1f2328", "#57606a"
FROZEN = dict(face="#e8eefc", edge="#1f2328")
LLM = dict(face="#e2f3e5", edge="#2f8f4e")
TEXT = dict(face="#ffffff", edge="#8c959f")
PLAIN = dict(face="#f3f4f6", edge="#8c959f")
ACTIONS = dict(face="#e3e3e3", edge="#8c959f")
SNOW, PEN = "#3d7edb", "#2f8f4e"


def box(ax, x0, y0, x1, y1, face, edge, width=1.6):
    ax.add_patch(FancyBboxPatch((x0 + 18, y0 + 18), x1 - x0 - 36, y1 - y0 - 36, boxstyle="round,pad=18,rounding_size=22",
                                facecolor=face, edgecolor=edge, linewidth=width, zorder=2))


def arrow(ax, start, end, label=None, below=None, side=None):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, linewidth=1.5, color=INK,
                                 shrinkA=0, shrinkB=0, zorder=3))
    x, y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
    if label:
        ax.text(x, y - 12, label, ha="center", va="bottom", fontsize=12.5, color=INK)
    if below:
        ax.text(x, y + 12, below, ha="center", va="top", fontsize=10.5, color=MUTED)
    if side:
        ax.text(x + side[0], y, side[1], ha="left" if side[0] > 0 else "right", va="center", fontsize=11, color=MUTED)


def main() -> None:
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    # Inputs: the same two as the trained policy gets.
    ax.text(347, 268, "Depth image", ha="center", va="bottom", fontsize=14, color=INK)
    ax.imshow(Image.open(FRAME), extent=(245, 450, 430, 276), zorder=2, interpolation="nearest")
    ax.add_patch(plt.Rectangle((245, 276), 205, 154, fill=False, edgecolor="#4a7a2a", linewidth=2, zorder=3))
    ax.text(347, 444, "80 × 60 pixels, front camera", ha="center", va="top", fontsize=11.5, color=MUTED)

    ax.text(347, 528, "Body state", ha="center", va="bottom", fontsize=14, color=INK)
    box(ax, 222, 540, 474, 648, **TEXT)
    ax.text(348, 594, "angular velocity, gravity,\njoints, last action,\njump status, goal distance", ha="center",
            va="center", fontsize=11.5, style="italic", color="#3a3f45", linespacing=1.35)
    ax.text(347, 662, "24 numbers", ha="center", va="top", fontsize=11.5, color=MUTED)

    # Where the DeFM encoder stood: plain code that writes the frame as a picture and a table.
    box(ax, 542, 303, 812, 403, **PLAIN)
    ax.text(677, 337, "Picture and table", ha="center", va="center", fontsize=15, weight="bold", color=INK)
    ax.text(677, 373, "plain code, no learning", ha="center", va="center", fontsize=10.5, color=MUTED)
    arrow(ax, (450, 353), (542, 353))
    arrow(ax, (812, 353), (980, 353), "picture + table")
    arrow(ax, (474, 594), (980, 594), "as text")

    # The high-level policy: an LLM, prompted and not trained.
    box(ax, 980, 303, 1260, 645, **LLM, width=2)
    ax.text(1236, 330, "✎", ha="center", va="center", fontsize=22, color=PEN)
    ax.text(1120, 430, "High-level policy", ha="center", va="center", fontsize=20, weight="bold", color=INK)
    ax.text(1120, 498, "LLM (Claude Opus)\none call every 0.25 s\nprompted, not trained", ha="center", va="center",
            fontsize=17, color=INK, linespacing=1.4)

    box(ax, 900, 78, 1340, 198, **TEXT)
    ax.text(1120, 114, "Instructions", ha="center", va="center", fontsize=15, weight="bold", color=INK)
    ax.text(1120, 156, "task rules, obstacle types, jump distances\nfixed text, the same on every call",
            ha="center", va="center", fontsize=11.5, color=MUTED, linespacing=1.35)
    arrow(ax, (1120, 198), (1120, 303))

    box(ax, 900, 730, 1340, 830, **TEXT)
    ax.text(1120, 762, "Memory", ha="center", va="center", fontsize=15, weight="bold", color=INK)
    ax.text(1120, 798, "its own last 3 decisions and a short note", ha="center", va="center", fontsize=11.5,
            color=MUTED)
    arrow(ax, (1190, 645), (1190, 730), side=(14, "writes"))
    arrow(ax, (1050, 730), (1050, 645), side=(-14, "reads on the next call"))

    # Outputs and everything after them: unchanged.
    arrow(ax, (1260, 353), (1478, 353), "speed, body height", "held for 0.25 s")
    arrow(ax, (1260, 594), (1478, 594), "jump trigger", "on the step it chooses")
    for y0, name in ((303, "Crawl policy"), (544, "Jump policy")):
        box(ax, 1478, y0, 1748, y0 + 100, **FROZEN)
        ax.text(1503, y0 + 24, "❄", ha="center", va="center", fontsize=19, color=SNOW)
        ax.text(1613, y0 + 36, name, ha="center", va="center", fontsize=15, weight="bold", color=INK)
        ax.text(1613, y0 + 68, "100 Hz", ha="center", va="center", fontsize=12, color=INK)
    arrow(ax, (1748, 353), (1916, 353), "otherwise")
    arrow(ax, (1748, 594), (1916, 594), "during a jump")
    box(ax, 1916, 303, 2128, 645, **ACTIONS)
    ax.text(2022, 458, "Joint actions", ha="center", va="center", fontsize=15, weight="bold", color=INK)
    ax.text(2022, 490, "to the robot", ha="center", va="center", fontsize=12, color=INK)

    # Legend and the Jev variant.
    ax.text(236, 772, "❄", ha="center", va="center", fontsize=18, color=SNOW)
    ax.text(262, 772, "frozen", ha="left", va="center", fontsize=12.5, color=INK)
    ax.text(388, 772, "✎", ha="center", va="center", fontsize=18, color=PEN)
    ax.text(414, 772, "prompted, no training", ha="left", va="center", fontsize=12.5, color=INK)
    ax.text(222, 818, "Jev in place of Claude: the table only, no picture, one call every 0.10 s.", ha="left",
            va="center", fontsize=11.5, color=MUTED)

    for suffix in ("png", "svg"):
        fig.savefig(HERE / f"llm_architecture.{suffix}", dpi=200, facecolor="white")


if __name__ == "__main__":
    main()
