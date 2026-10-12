"""
Shared look & feel for every Manim scene on the site.

Import with:  from style import *
Every scene should subclass `WikiScene` so backgrounds, fonts and colours stay
consistent across pages (and match the site's dark theme / code-block colours).
"""
from manim import *

# ---- palette (matches src/css/custom.css + the Dracula code theme) ----------
BG        = "#1b1b1d"   # Docusaurus dark background
FG        = "#e3e3e3"   # main text
MUTED     = "#8a8f98"   # secondary text, grid lines
ACCENT    = "#25c2a0"   # site primary colour (teal)
HIGHLIGHT = "#f1fa8c"   # amber/yellow: "look here" (same as highlighted code lines)
ERROR     = "#ff5555"   # red: wrong / exploding / dropped
SUCCESS   = "#50fa7b"   # green: correct / kept
INFO      = "#8be9fd"   # cyan: data / forward flow
GRAD      = "#ff79c6"   # pink: gradients / backward flow
PURPLE    = "#bd93f9"

FONT = "Inter"
MONO = "DejaVu Sans Mono"

config.background_color = BG


def label(text, size=26, color=FG, weight=NORMAL, mono=False):
    """Plain text in the site font."""
    return Text(text, font=MONO if mono else FONT, font_size=size, color=color, weight=weight)


MAX_TEXT_WIDTH = 13.0   # frame is ~14.2 units wide; keep a margin


def _fit(mob, max_width=MAX_TEXT_WIDTH):
    if mob.width > max_width:
        mob.scale_to_fit_width(max_width)
    return mob


def title(text, size=34):
    """Scene title pinned to the top edge (auto-shrinks to fit the frame)."""
    return _fit(label(text, size=size, weight=BOLD)).to_edge(UP, buff=0.4)


def math(tex, size=40, color=FG):
    """LaTeX formula (same notation as the KaTeX on the pages)."""
    return MathTex(tex, font_size=size, color=color)


def grid_of_numbers(values, cell=0.6, size=22, stroke=MUTED, fill=None, fill_opacity=0.0, fmt="{:g}"):
    """A 2-D grid of squares with numbers inside. values: list of lists.
    Returns a VGroup of rows, each row a VGroup of VGroup(square, number)."""
    rows = VGroup()
    for i, row in enumerate(values):
        r = VGroup()
        for j, v in enumerate(row):
            sq = Square(cell, stroke_width=1.5, stroke_color=stroke)
            if fill is not None:
                sq.set_fill(fill, opacity=fill_opacity)
            txt = label(fmt.format(v) if v is not None else "", size=size, mono=True)
            r.add(VGroup(sq, txt).move_to([j * cell, -i * cell, 0]))
        rows.add(r)
    return rows


def caption_bar(text, size=24, color=MUTED):
    """One-line explanation pinned to the bottom edge."""
    return _fit(label(text, size=size, color=color)).to_edge(DOWN, buff=0.45)


class WikiScene(Scene):
    """Base class: consistent background; subclasses implement construct()."""

    def setup(self):
        self.camera.background_color = BG
