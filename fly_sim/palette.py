"""Figure colors, from the data-visualization reference palette (light mode).

Series order is fixed: the four readouts always take these slots, in this order, so a color
means the same thing in every figure. Aqua and yellow fall below 3:1 on the surface, so every
chart that uses them carries visible labels.
"""

SERIES = {  # readout -> (categorical slot, label)
    "fly": ("#2a78d6", "Fly optic lobe"),
    "pixels": ("#eb6834", "Raw hex pixels"),
    "rewired": ("#1baf7a", "Degree-preserving rewire"),
    "random": ("#eda100", "Random network"),
}
SURFACE, GRID, AXIS, MUTED, SECONDARY, PRIMARY = "#fcfcfb", "#e1e0d9", "#c3c2b7", "#898781", "#52514e", "#0b0b0b"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
FONTS = ["Helvetica Neue", "Arial", "DejaVu Sans"]


def style(plt):
    plt.rcParams.update({"font.family": FONTS, "font.size": 10})


def clean(ax, spines=("top", "right", "left")):
    """Recessive axes: a horizontal grid under the marks and no box around the plot."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in spines:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0)
