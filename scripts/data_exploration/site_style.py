"""House chart style for the site (validated categorical order from the dataviz palette).

    import site_style; site_style.apply()
"""
import matplotlib as mpl

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f2a3a2", "#e66767", "#b8302f"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def apply():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "lines.linewidth": 2, "lines.markersize": 6, "legend.frameon": False,
        "font.family": "DejaVu Sans", "font.size": 10, "figure.dpi": 110, "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("site_div", DIVERGING)


def sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("site_seq", BLUE_RAMP)
