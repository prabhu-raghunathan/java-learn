"""Tiny SVG toolkit for the site's block diagrams (deep-learning pages).

Visual language (same on every diagram):
  * green rounded rectangle  = a cell / module that is repeated or reused
  * yellow box               = a learned layer (sigma, tanh, Linear, Conv, ...)
  * pink circle / ellipse    = a pointwise operation (x, +, tanh, 1-, mean, ...)
  * blue circle              = an input (x_t, a word, an image)
  * purple circle            = an output (h_t, a prediction)
  * thick black line + arrow = a vector flowing forward
  * red dashed line + arrow  = a gradient flowing backward
Everything sits on a white card so the diagram stays readable in dark mode.

Labels accept a tiny markup: x_t, h_{t-1}, W^T, c~_t (c with a tilde), so symbols match the pages.
"""
import functools
import math
import os
import re

# Arial-compatible metrics (Arial, Helvetica and Liberation Sans share widths), used to centre text
_FONT_FILES = {
    "normal": ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
               "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial.ttf", "C:/Windows/Fonts/arial.ttf"],
    "bold": ["/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
             "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf", "C:/Windows/Fonts/arialbd.ttf"],
}


@functools.lru_cache(maxsize=None)
def _font(weight, size):
    try:
        from PIL import ImageFont
        for f in _FONT_FILES["bold" if weight == "bold" else "normal"]:
            if os.path.exists(f):
                return ImageFont.truetype(f, int(round(size * 10)))   # measure at 10x for precision
    except ImportError:
        pass
    return None


def measure(text, size, weight="normal"):
    f = _font(weight, size)
    if f is None:
        return len(text) * 0.55 * size
    return f.getlength(text) / 10.0

FONT = "Helvetica, Arial, Liberation Sans, sans-serif"
INK = "#1f1f1f"
CELL_FILL, CELL_STROKE = "#e2f0d5", "#7fae5b"
LAYER_FILL, LAYER_STROKE = "#fbe9a6", "#9c8526"
OP_FILL, OP_STROKE = "#f7d0cf", "#b5605c"
IN_FILL, IN_STROKE = "#a8cbee", "#3b78bd"
OUT_FILL, OUT_STROKE = "#dcb6ee", "#8a4fae"
GRAD = "#d23c3c"
MUTED = "#8a8a8a"
FROZEN_FILL, FROZEN_STROKE = "#e3e3e3", "#9a9a9a"
NOTE_FILL, NOTE_STROKE = "#f4f7fb", "#c9d3df"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_TOKEN = re.compile(r"(_\{[^}]*\}|\^\{[^}]*\}|_[A-Za-z0-9]|\^[A-Za-z0-9]|~|[^_^~]+)")


def rich(s):
    """Markup -> list of (text, shift) where shift is 'sub', 'sup' or ''."""
    out = []
    s = s.replace("\\_", "\u2063")         # "\_" = a literal underscore (e.g. zero\_grad)
    for tok in _TOKEN.findall(s):
        if tok == "~":                         # tilde on the previous character
            if out:
                t, sh = out[-1]
                out[-1] = (t + "̃", sh)
            continue
        if tok.startswith("_"):
            out.append((tok[2:-1] if tok.startswith("_{") else tok[1:], "sub"))
        elif tok.startswith("^"):
            out.append((tok[2:-1] if tok.startswith("^{") else tok[1:], "sup"))
        else:
            out.append((tok, ""))
    return [(t.replace("\u2063", "_"), sh) for t, sh in out]


class Diagram:
    def __init__(self, w, h, title=None, card=True):
        self.w, self.h = w, h
        self.parts = []
        self.card = card
        self.title = title
        self._group_stack = []

    # ---- grouping (used to dim everything except a highlighted path) -----------
    def group(self, opacity=1.0):
        d = self

        class _G:
            def __enter__(self_):
                d.parts.append(f'<g opacity="{opacity}">')

            def __exit__(self_, *a):
                d.parts.append("</g>")
        return _G()

    def fit_from(self, start, x_end, margin=30):
        """Scale everything drawn since parts[start] so it fits horizontally."""
        avail = self.w - 2 * margin
        used = x_end - margin
        if used <= avail:
            return
        k = avail / used
        self.parts.insert(start, f'<g transform="translate({margin * (1 - k):.1f},{self.h * (1 - k) / 2:.1f}) scale({k:.4f})">')
        self.parts.append("</g>")

    # ---- text --------------------------------------------------------------------
    def text(self, x, y, s, size=18, anchor="middle", color=INK, weight="normal", italic=False, family=FONT):
        """Text with sub/superscripts and tildes. Widths are measured (Arial metrics) and the text is
        always emitted left-anchored, so centring is exact in every renderer. Tildes are drawn as a
        small '~' placed over the previous character."""
        style = ' font-style="italic"' if italic else ""
        shift = {"": 0.0, "sub": 0.32 * size, "sup": -0.42 * size}
        runs, tildes, xpos = [], [], 0.0      # runs: (text, shift, size, x)
        for t, sh in rich(s):
            fs = size * 0.72 if sh else size
            pieces = t.split("\u0303")
            for k, piece in enumerate(pieces):
                if k > 0 and runs:
                    last_t, last_sh, last_fs, last_x = runs[-1]
                    ch = last_t[-1]
                    cw = measure(ch, last_fs, weight)
                    tildes.append((last_x + measure(last_t, last_fs, weight) - cw / 2, shift[last_sh] - 0.62 * last_fs, last_fs))
                if piece:
                    if runs and sh != runs[-1][1]:
                        xpos += 0.06 * size          # breathing room around a sub/superscript
                    runs.append((piece, sh, fs, xpos))
                    xpos += measure(piece, fs, weight)
        total = xpos
        x0 = {"start": x, "middle": x - total / 2, "end": x - total}[anchor]
        for t, sh, fs, rx in runs:
            body = esc(t)
            for sym in "⊙∇⇒①②③④⑤":
                body = body.replace(sym, f'<tspan font-family="DejaVu Sans, Apple Symbols, Segoe UI Symbol, Cambria Math, sans-serif">{sym}</tspan>')
            self.parts.append(
                f'<text xml:space="preserve" style="white-space:pre" x="{x0 + rx:.1f}" y="{y + shift[sh]:.1f}" font-family="{family}" font-size="{fs:.1f}" '
                f'fill="{color}" dominant-baseline="central" font-weight="{weight}"{style}>{body}</text>')
        for tx, dy, fs in tildes:
            self.parts.append(
                f'<text x="{x0 + tx:.1f}" y="{y + dy:.1f}" font-family="{family}" font-size="{fs * 0.9:.1f}" '
                f'fill="{color}" text-anchor="middle" dominant-baseline="central">~</text>')

    # ---- shapes ------------------------------------------------------------------
    def rect(self, x, y, w, h, fill, stroke, rx=8, sw=2.2, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="{sw}"{d}/>')

    def cell(self, x, y, w, h, label=None):
        self.rect(x, y, w, h, CELL_FILL, CELL_STROKE, rx=22, sw=2.6)
        if label:
            self.text(x + 14, y + 18, label, 15, "start", "#4d7a32", "bold")

    def layer(self, cx, cy, label, w=None, h=34, size=18, fill=LAYER_FILL, stroke=LAYER_STROKE, color=INK):
        w = w or max(44, 12 * len(label) + 18)
        self.rect(cx - w / 2, cy - h / 2, w, h, fill, stroke, rx=4, sw=2)
        self.text(cx, cy, label, size, color=color)
        return (cx, cy, w, h)

    def op(self, cx, cy, label, r=14, size=17, w=None):
        if w:   # ellipse (e.g. 'tanh')
            self.parts.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{w / 2}" ry="{r}" fill="{OP_FILL}" '
                              f'stroke="{OP_STROKE}" stroke-width="2"/>')
        else:
            self.parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{OP_FILL}" stroke="{OP_STROKE}" '
                              f'stroke-width="2"/>')
        self.text(cx, cy, label, size)

    def node(self, cx, cy, label, kind="in", r=26, size=22):
        fill, stroke = (IN_FILL, IN_STROKE) if kind == "in" else (OUT_FILL, OUT_STROKE)
        self.parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2.4"/>')
        self.text(cx, cy, label, size)

    def note(self, x, y, w, h):
        self.rect(x, y, w, h, NOTE_FILL, NOTE_STROKE, rx=10, sw=1.5)

    # ---- lines -------------------------------------------------------------------
    def line(self, pts, color=INK, width=3.2, arrow=True, dash=None, radius=12, head=11):
        """Polyline with rounded corners and a filled arrowhead at the end."""
        pts = [tuple(map(float, p)) for p in pts]
        if arrow:   # shorten the last segment so the line ends under the head
            (x1, y1), (x2, y2) = pts[-2], pts[-1]
            L = math.hypot(x2 - x1, y2 - y1)
            ux, uy = (x2 - x1) / L, (y2 - y1) / L
            tip = (x2, y2)
            pts[-1] = (x2 - ux * head * 0.8, y2 - uy * head * 0.8)
        d = f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"
        for i in range(1, len(pts) - 1):
            (ax, ay), (bx, by), (cx, cy) = pts[i - 1], pts[i], pts[i + 1]
            l1, l2 = math.hypot(bx - ax, by - ay), math.hypot(cx - bx, cy - by)
            r = min(radius, l1 / 2, l2 / 2)
            p1 = (bx - (bx - ax) / l1 * r, by - (by - ay) / l1 * r)
            p2 = (bx + (cx - bx) / l2 * r, by + (cy - by) / l2 * r)
            d += f" L{p1[0]:.1f},{p1[1]:.1f} Q{bx:.1f},{by:.1f} {p2[0]:.1f},{p2[1]:.1f}"
        d += f" L{pts[-1][0]:.1f},{pts[-1][1]:.1f}"
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
                          f'stroke-linecap="round" stroke-linejoin="round"{da}/>')
        if arrow:
            bx, by = tip
            px, py = -uy, ux
            hw = head * 0.55
            p = [(bx, by), (bx - ux * head + px * hw, by - uy * head + py * hw),
                 (bx - ux * head - px * hw, by - uy * head - py * hw)]
            self.parts.append('<polygon points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in p) +
                              f'" fill="{color}"/>')

    def curve(self, p0, c1, c2, p1, color=INK, width=3.2, dash=None, head=11):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        ux, uy = p1[0] - c2[0], p1[1] - c2[1]
        L = math.hypot(ux, uy)
        ux, uy = ux / L, uy / L
        end = (p1[0] - ux * head * 0.8, p1[1] - uy * head * 0.8)
        self.parts.append(f'<path d="M{p0[0]},{p0[1]} C{c1[0]},{c1[1]} {c2[0]},{c2[1]} {end[0]:.1f},{end[1]:.1f}" '
                          f'fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round"{da}/>')
        px, py = -uy, ux
        hw = head * 0.55
        p = [p1, (p1[0] - ux * head + px * hw, p1[1] - uy * head + py * hw),
             (p1[0] - ux * head - px * hw, p1[1] - uy * head - py * hw)]
        self.parts.append('<polygon points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in p) + f'" fill="{color}"/>')

    def dot(self, x, y, r=4.5, color=INK):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"/>')

    # ---- output ------------------------------------------------------------------
    def svg(self):
        bg = (f'<rect x="1" y="1" width="{self.w - 2}" height="{self.h - 2}" rx="14" fill="#ffffff" '
              f'stroke="#d9d9d9" stroke-width="1.5"/>') if self.card else ""
        title = f"<title>{esc(self.title)}</title>" if self.title else ""
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
                f'width="{self.w}" height="{self.h}" role="img">{title}{bg}' + "".join(self.parts) + "</svg>")

    def save(self, path):
        with open(path, "w") as f:
            f.write(self.svg())


def legend(d, x, y, items=("layer", "op", "flow", "copy", "concat"), grad=False):
    """A one-line legend (Colah-style) at (x, y)."""
    cx = x
    for it in items:
        if it == "layer":
            d.layer(cx + 22, y, "σ", w=40, h=28, size=16)
            d.text(cx + 50, y, "learned layer", 14, "start", MUTED)
            cx += 160
        elif it == "op":
            d.op(cx + 14, y, "×", r=12, size=15)
            d.text(cx + 34, y, "pointwise op", 14, "start", MUTED)
            cx += 150
        elif it == "flow":
            d.line([(cx, y), (cx + 46, y)], width=2.6, head=9)
            d.text(cx + 56, y, "vector flow", 14, "start", MUTED)
            cx += 150
        elif it == "copy":
            d.line([(cx, y), (cx + 22, y), (cx + 46, y - 10)], width=2.4, head=8, radius=4)
            d.line([(cx + 22, y), (cx + 46, y + 10)], width=2.4, head=8, radius=4)
            d.text(cx + 56, y, "copy", 14, "start", MUTED)
            cx += 110
        elif it == "concat":
            d.line([(cx, y - 10), (cx + 22, y)], width=2.4, arrow=False)
            d.line([(cx, y + 10), (cx + 22, y), (cx + 46, y)], width=2.4, head=8, radius=4)
            d.text(cx + 56, y, "concatenate", 14, "start", MUTED)
            cx += 150
    if grad:
        d.line([(cx, y), (cx + 46, y)], color=GRAD, width=2.6, head=9, dash="7 5")
        d.text(cx + 56, y, "gradient (backward)", 14, "start", MUTED)
