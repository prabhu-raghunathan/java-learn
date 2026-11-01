"""Animations for the Recurrent Neural Network pages (keys rnn/...).
Every number shown is computed (NumPy / PyTorch) and matches the verified outputs quoted on the pages.
Text-only labels (no MathTex), so no LaTeX/dvisvgm is needed to render these scenes."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from style import *

PARAM = "#ffb86c"   # orange: learnable parameters (matches \param on the pages)


def sym(base, sub, size=28, color=FG):
    """A symbol with a subscript, e.g. sym('h', 't-1') → h with a small t-1."""
    b = label(base, size, color)
    s = label(sub, int(size * 0.62), color).next_to(b, RIGHT, buff=0.03).shift(DOWN * size * 0.006)
    return VGroup(b, s)


def chip(text, color=FG, size=22, fill_opacity=0.15, w=None, h=0.55):
    t = label(text, size, FG, mono=True)
    width = w if w is not None else t.width + 0.36
    box = RoundedRectangle(corner_radius=0.08, width=width, height=h,
                           stroke_color=color, stroke_width=2).set_fill(color, fill_opacity)
    return VGroup(box, t.move_to(box))


def vec_column(values, color=FG, size=20, w=1.15, h=0.42):
    """A column vector drawn as stacked number cells."""
    cells = VGroup(*[chip(f"{v:+.3f}", color, size, 0.12, w=w, h=h) for v in values])
    return cells.arrange(DOWN, buff=0.04)


def cell_box(text="tanh", color=ACCENT, w=1.3, h=0.9):
    box = RoundedRectangle(corner_radius=0.15, width=w, height=h, stroke_color=color,
                           stroke_width=3).set_fill(color, 0.12)
    return VGroup(box, label(text, 24, FG).move_to(box))


# ---------------------------------------------------------------------------
# Page 1: the hidden state carries memory forward (toy weights from notebook 12)
# ---------------------------------------------------------------------------
TOY_X = [(1.0, 0.0), (0.5, 1.0), (0.0, 1.0), (1.0, 1.0)]
TOY_H = [(0.462, 0.291, -0.537), (0.355, 0.733, -0.498),
         (0.114, 0.769, -0.398), (0.464, 0.878, -0.771)]


class HiddenStateFlow(WikiScene):
    def construct(self):
        self.add(title("The hidden state carries memory from step to step"))
        xs = [-3.9, -1.2, 1.5, 4.2]
        y_cell, y_in, y_h = -0.2, -2.05, 1.75

        h0 = vec_column([0, 0, 0], MUTED).move_to([-6.1, y_h, 0])
        h0_lbl = sym("h", "0", 24, MUTED).next_to(h0, UP, buff=0.12)
        self.play(FadeIn(h0), FadeIn(h0_lbl), run_time=0.5)
        prev = h0
        for t, (x, xv, hv) in enumerate(zip(xs, TOY_X, TOY_H), start=1):
            box = cell_box().move_to([x, y_cell, 0])
            xin = chip(f"[{xv[0]:g}, {xv[1]:g}]", INFO, 22).move_to([x, y_in, 0])
            xl = sym("x", str(t), 24, INFO).next_to(xin, DOWN, buff=0.1)
            up = Arrow(xin.get_top(), box.get_bottom(), buff=0.08, color=INFO, stroke_width=4,
                       max_tip_length_to_length_ratio=0.2)
            side = Arrow(prev.get_right(), box.get_left(), buff=0.1, color=SUCCESS, stroke_width=4,
                         max_tip_length_to_length_ratio=0.15)
            self.play(FadeIn(xin), FadeIn(xl), FadeIn(box), run_time=0.45)
            self.play(GrowArrow(up), GrowArrow(side), run_time=0.55)
            self.play(box[0].animate.set_fill(HIGHLIGHT, 0.4), run_time=0.25)
            hcol = vec_column(hv, SUCCESS).move_to([x, y_h, 0])
            hl = sym("h", str(t), 24, SUCCESS).next_to(hcol, UP, buff=0.12)
            out = Arrow(box.get_top(), hcol.get_bottom(), buff=0.06, color=SUCCESS, stroke_width=4,
                        max_tip_length_to_length_ratio=0.25)
            self.play(box[0].animate.set_fill(ACCENT, 0.12), GrowArrow(out),
                      FadeIn(hcol, shift=UP * 0.2), FadeIn(hl), run_time=0.55)
            prev = hcol
        final = SurroundingRectangle(prev, color=HIGHLIGHT, buff=0.08, stroke_width=3)
        self.play(Create(final), run_time=0.4)
        self.play(FadeIn(caption_bar("Same weights at every step: each new h mixes the current x with the previous h")))
        self.wait(2.6)


# ---------------------------------------------------------------------------
# Page 1: padding and masking a batch of variable-length sentences
# ---------------------------------------------------------------------------
PAD_IDS = [[5, 8, 2], [5, 8, 4, 7, 3, 10, 9], [4, 7, 6, 11]]
PAD_TEXT = ["i love ai", "i love deep learning and neural networks", "deep learning is powerful"]


class PaddingMask(WikiScene):
    def construct(self):
        self.add(title("Batching sentences of different lengths: pad, then mask"))
        cw, x0 = 0.78, -2.6
        rows_y = [1.55, 0.55, -0.45]
        sents = VGroup()
        real_cells = []
        for r, (ids, txt, y) in enumerate(zip(PAD_IDS, PAD_TEXT, rows_y)):
            s = label(f'"{txt}"', 17, MUTED).move_to([0, y + 0.42, 0])
            s.align_to([x0 - cw / 2, 0, 0], LEFT)
            sents.add(s)
            row = VGroup(*[chip(str(v), INFO, 22, 0.18, w=cw - 0.08) for v in ids])
            for j, c in enumerate(row):
                c.move_to([x0 + j * cw, y, 0])
            real_cells.append(row)
            ln = label(f"len {len(ids)}", 20, FG).move_to([-4.9, y, 0])
            self.play(FadeIn(s), FadeIn(row, shift=RIGHT * 0.2), FadeIn(ln), run_time=0.6)
        self.wait(0.4)
        T = max(len(i) for i in PAD_IDS)
        pads = VGroup()
        for r, (ids, y) in enumerate(zip(PAD_IDS, rows_y)):
            for j in range(len(ids), T):
                pads.add(chip("0", MUTED, 22, 0.05, w=cw - 0.08).move_to([x0 + j * cw, y, 0]))
        note = label("<PAD> = 0 fills short rows: batch tensor shape (3, 7)", 21, FG).move_to([0.4, -1.45, 0])
        self.play(LaggedStart(*[FadeIn(p, shift=LEFT * 0.15) for p in pads], lag_ratio=0.08), run_time=1.2)
        self.play(FadeIn(note), run_time=0.4)
        self.wait(0.6)
        anims = []
        for row in real_cells:
            anims += [c[0].animate.set_stroke(SUCCESS).set_fill(SUCCESS, 0.25) for c in row]
        anims += [p[0].animate.set_stroke(ERROR).set_fill(ERROR, 0.22) for p in pads]
        mask = label("mask = (ids != 0): green counts, red is ignored", 21, HIGHLIGHT).move_to([0.4, -2.05, 0])
        self.play(*anims, FadeIn(mask), run_time=0.9)
        self.wait(0.4)
        self.play(FadeIn(caption_bar("Pass lengths [3, 7, 4] too, so the RNN stops at each row's real last word")))
        self.wait(2.6)


# ---------------------------------------------------------------------------
# Page 2: unrolling one cell into T steps with shared weights
# ---------------------------------------------------------------------------
class UnrolledSharedWeights(WikiScene):
    def construct(self):
        self.add(title("Unrolling: one cell, reused at every time step"))
        # folded view
        box = cell_box("RNN cell", ACCENT, 1.8, 1.0).move_to([-4.6, 0.2, 0])
        xin = sym("x", "t", 26, INFO).move_to([-4.6, -1.9, 0])
        hout = sym("h", "t", 26, SUCCESS).move_to([-4.6, 2.2, 0])
        a_in = Arrow(xin.get_top(), box.get_bottom(), buff=0.1, color=INFO, stroke_width=4)
        a_out = Arrow(box.get_top(), hout.get_bottom(), buff=0.1, color=SUCCESS, stroke_width=4)
        loop = CurvedArrow(box.get_right() + UP * 0.25, box.get_right() + DOWN * 0.25, angle=-4.2,
                           color=SUCCESS, stroke_width=4)
        wh = sym("W", "h", 24, PARAM).next_to(loop, RIGHT, buff=0.1)
        wx = sym("W", "x", 24, PARAM).next_to(a_in, LEFT, buff=0.12)
        folded = VGroup(box, xin, hout, a_in, a_out, loop, wh, wx)
        self.play(FadeIn(folded), run_time=0.8)
        fl = label("folded (the loop)", 20, MUTED).next_to(xin, DOWN, buff=0.25)
        self.play(FadeIn(fl), run_time=0.3)
        self.wait(0.6)

        self.play(VGroup(folded, fl).animate.scale(0.62).move_to([-5.45, 0.1, 0]), run_time=0.7)
        xs = [-1.6, 0.7, 3.0, 5.3]
        copies, whs, wxs = VGroup(), [], []
        h0 = sym("h", "0", 24, MUTED).move_to([-3.25, 0.2, 0])
        self.play(FadeIn(h0), run_time=0.3)
        prev = h0
        for t, x in enumerate(xs, start=1):
            b = cell_box("tanh", ACCENT, 1.25, 0.9).move_to([x, 0.2, 0])
            xi = sym("x", str(t), 24, INFO).move_to([x, -1.9, 0])
            ho = sym("h", str(t), 24, SUCCESS).move_to([x, 2.2, 0])
            ai = Arrow(xi.get_top(), b.get_bottom(), buff=0.1, color=INFO, stroke_width=3.5)
            ao = Arrow(b.get_top(), ho.get_bottom(), buff=0.1, color=SUCCESS, stroke_width=3.5)
            ah = Arrow(prev.get_right(), b.get_left(), buff=0.1, color=SUCCESS, stroke_width=3.5,
                       max_tip_length_to_length_ratio=0.2)
            whl = sym("W", "h", 20, PARAM).next_to(ah, UP, buff=0.05)
            wxl = sym("W", "x", 20, PARAM).next_to(ai, RIGHT, buff=0.08)
            grp = VGroup(b, xi, ho, ai, ao, ah, whl, wxl)
            self.play(TransformFromCopy(box, b), FadeIn(VGroup(xi, ho, ai, ao, ah, whl, wxl)), run_time=0.6)
            copies.add(grp)
            whs.append(whl)
            wxs.append(wxl)
            prev = b
        self.wait(0.3)
        rects = [SurroundingRectangle(m, color=HIGHLIGHT, buff=0.06, stroke_width=3) for m in whs + wxs]
        same = label("the same matrices at every step", 22, HIGHLIGHT).move_to([2.2, -2.85, 0])
        self.play(*[Create(r) for r in rects], FadeIn(same), run_time=0.8)
        self.play(FadeIn(caption_bar("4 steps or 400: still one Wx, one Wh and one b, so the parameter count never grows")))
        self.wait(2.6)


# ---------------------------------------------------------------------------
# Page 3: repeated multiplication vanishes or explodes (0.5^k and 1.5^k)
# ---------------------------------------------------------------------------
class VanishingExploding(WikiScene):
    def construct(self):
        self.add(title("Backprop through time multiplies one factor per step"))
        ax = Axes(x_range=[0, 15, 1], y_range=[-5, 3, 1], x_length=8.6, y_length=4.5,
                  axis_config={"color": MUTED, "stroke_width": 2, "include_ticks": True},
                  tips=False).move_to([-0.9, 0.2, 0])
        self.play(Create(ax), run_time=0.6)
        for e in range(-5, 4):
            txt = "1" if e == 0 else f"1e{e}"
            self.add(label(txt, 16, MUTED, mono=True).next_to(ax.c2p(0, e), LEFT, buff=0.12))
        for k in (0, 5, 10, 15):
            self.add(label(str(k), 16, MUTED, mono=True).next_to(ax.c2p(k, -5), DOWN, buff=0.12))
        self.add(label("steps back in time", 18, MUTED).next_to(ax.c2p(7.5, -5), DOWN, buff=0.4))
        self.add(label("gradient (log scale)", 18, MUTED).rotate(PI / 2).next_to(ax.c2p(0, -1), LEFT, buff=0.85))
        one = DashedLine(ax.c2p(0, 0), ax.c2p(15, 0), color=MUTED, stroke_width=1.5)
        self.add(one)

        ks = np.arange(0, 16)
        van = [ax.c2p(k, np.log10(0.5 ** k)) for k in ks]
        exp = [ax.c2p(k, np.log10(1.5 ** k)) for k in ks]
        lv = VMobject(color=INFO, stroke_width=4).set_points_as_corners(van)
        le = VMobject(color=ERROR, stroke_width=4).set_points_as_corners(exp)
        dv, de = Dot(van[0], color=INFO), Dot(exp[0], color=ERROR)
        rv = label("x 0.5 per step", 22, INFO).move_to([5.0, 0.4, 0])
        re_ = label("x 1.5 per step", 22, ERROR).move_to([5.0, 2.35, 0])
        val_v = label("1.000000", 22, INFO, mono=True).next_to(rv, DOWN, buff=0.15)
        val_e = label("1.0000", 22, ERROR, mono=True).next_to(re_, DOWN, buff=0.15)
        self.play(FadeIn(rv), FadeIn(re_), FadeIn(val_v), FadeIn(val_e), FadeIn(dv), FadeIn(de), run_time=0.4)
        tr = ValueTracker(0)

        def upd_v(m):
            k = tr.get_value()
            m.move_to(ax.c2p(k, np.log10(0.5 ** k)))

        def upd_e(m):
            k = tr.get_value()
            m.move_to(ax.c2p(k, np.log10(1.5 ** k)))

        dv.add_updater(upd_v)
        de.add_updater(upd_e)
        val_v.add_updater(lambda m: m.become(label(f"{0.5 ** round(tr.get_value()):.6f}", 22, INFO, mono=True).next_to(rv, DOWN, buff=0.15)))
        val_e.add_updater(lambda m: m.become(label(f"{1.5 ** round(tr.get_value()):.4f}", 22, ERROR, mono=True).next_to(re_, DOWN, buff=0.15)))
        self.add(lv, le)
        lv.set_points_as_corners(van[:1] * 2)
        le.set_points_as_corners(exp[:1] * 2)

        def grow(line, pts):
            def f(m):
                k = tr.get_value()
                n = int(np.floor(k))
                frac = k - n
                p = pts[:n + 1]
                if n < 15:
                    p = p + [pts[n] + (pts[n + 1] - pts[n]) * frac]
                if len(p) < 2:
                    p = p * 2
                m.set_points_as_corners(p)
            return f

        lv.add_updater(grow(lv, van))
        le.add_updater(grow(le, exp))
        self.play(tr.animate.set_value(15), run_time=6, rate_func=linear)
        for m in (dv, de, val_v, val_e, lv, le):
            m.clear_updaters()
        self.play(FadeIn(caption_bar("After 15 steps: 0.5^15 = 0.000031 (vanished), 1.5^15 = 437.89 (exploded)")))
        self.wait(2.6)


# ---------------------------------------------------------------------------
# Page 3: multiplicative memory vs gated (additive) memory over 20 steps
# ---------------------------------------------------------------------------
class GatedMemory(WikiScene):
    def construct(self):
        self.add(title("Why gates help: what survives 20 steps?"))
        n = 20
        x0, dx = -5.7, 0.43
        lanes = [("plain RNN path: x 0.5 per step", 0.5, ERROR, 1.1),
                 ("gated cell path: forget gate f = 0.95", 0.95, SUCCESS, -1.5)]
        hmax = 1.9
        finals = []
        for name, f, col, y in lanes:
            self.add(label(name, 22, col).move_to([0, y + 1.25, 0]))
            base = Line([x0 - 0.3, y - 0.9, 0], [x0 + dx * (n - 1) + 0.3, y - 0.9, 0], color=MUTED, stroke_width=1.5)
            self.add(base)
            bars = VGroup()
            for k in range(n):
                v = f ** k
                hgt = max(hmax * v, 0.012)
                bars.add(Rectangle(width=dx * 0.72, height=hgt, stroke_width=0)
                         .set_fill(col, 0.85).move_to([x0 + dx * k, y - 0.9 + hgt / 2, 0]))
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.12), run_time=2.6)
            fv = f ** (n - 1) * f
            txt = label(f"after 20 steps\n{fv:.3g}", 22, col, mono=True).move_to([4.6, y - 0.3, 0])
            finals.append(txt)
            self.play(FadeIn(txt), run_time=0.4)
        self.add(label("step 1", 16, MUTED).move_to([x0, -2.65, 0]),
                 label("step 20", 16, MUTED).move_to([x0 + dx * (n - 1), -2.65, 0]))
        self.play(FadeIn(caption_bar("LSTM/GRU keep an additive memory path whose per-step factor is a learned gate near 1")))
        self.wait(2.6)


# ---------------------------------------------------------------------------
# Page 2: BPTT. Forward sweep, then the gradient flows back and shrinks.
# Norms ||dL/dh_t|| from PyTorch: T = 20, H = 16, W_hh = 0.5 x orthogonal,
# loss = sum(h_20). Verified on the page (scale 0.5 row of the table).
# ---------------------------------------------------------------------------
BPTT_NORMS = [6.69e-06, 1.34e-05, 2.69e-05, 5.39e-05, 0.000108, 0.000217, 0.000434, 0.000869,
              0.00177, 0.00356, 0.00721, 0.0145, 0.0294, 0.0595, 0.121, 0.243, 0.492, 0.987, 1.99, 4.0]


class BpttGradientFlow(WikiScene):
    def construct(self):
        self.add(title("Backpropagation through time: the gradient walks back step by step"))
        n = len(BPTT_NORMS)
        x0, dx, y = -5.9, 0.6, 1.2
        nodes = VGroup(*[Circle(0.2, color=MUTED, stroke_width=2).move_to([x0 + dx * k, y, 0]) for k in range(n)])
        arrows = VGroup(*[Line(nodes[k].get_right(), nodes[k + 1].get_left(), color=MUTED, stroke_width=2)
                          for k in range(n - 1)])
        self.add(label("h1", 18, MUTED).next_to(nodes[0], UP, buff=0.12),
                 label("h20", 18, MUTED).next_to(nodes[-1], UP, buff=0.12))
        self.play(FadeIn(nodes), FadeIn(arrows), run_time=0.5)
        loss = chip("L", GRAD, 26, 0.25).move_to([x0 + dx * n + 0.35, y, 0])
        fw = label("forward: h1 -> h20", 22, INFO).move_to([-3.2, 2.35, 0])
        self.play(FadeIn(fw), LaggedStart(*[n_.animate.set_stroke(INFO).set_fill(INFO, 0.35) for n_ in nodes],
                                         lag_ratio=0.15), run_time=2.0)
        self.play(FadeIn(loss), run_time=0.4)

        base_y = -2.3
        self.add(Line([x0 - 0.3, base_y, 0], [x0 + dx * (n - 1) + 0.3, base_y, 0], color=MUTED, stroke_width=1.5))
        bw = label("backward: gradient norm at each h (log-height bars)", 22, GRAD).move_to([0.6, 0.35, 0])
        self.play(FadeIn(bw), run_time=0.4)
        bars = []
        for k in reversed(range(n)):
            v = BPTT_NORMS[k]
            hgt = 0.12 + (np.log10(v) + 6) / 7 * 2.0       # 1e-6 -> 0.12, 10 -> 2.4
            bar = Rectangle(width=dx * 0.7, height=hgt, stroke_width=0).set_fill(GRAD, 0.85)
            bar.move_to([x0 + dx * k, base_y + hgt / 2, 0])
            bars.append(bar)
        anims = []
        for i, k in enumerate(reversed(range(n))):
            anims.append(AnimationGroup(nodes[k].animate.set_stroke(GRAD).set_fill(GRAD, 0.35),
                                        GrowFromEdge(bars[i], DOWN)))
        self.play(LaggedStart(*anims, lag_ratio=0.35), run_time=4.0)
        for k, txt in ((19, "4.0"), (14, "0.121"), (9, "0.00356"), (0, "0.0000067")):
            b = bars[n - 1 - k]
            self.add(label(txt, 16, FG, mono=True).next_to(b, UP, buff=0.08))
        self.wait(0.3)
        self.play(FadeIn(caption_bar("With Wh = 0.5 x orthogonal the gradient halves at every step: h1 gets 600,000x less signal than h20")))
        self.wait(2.6)
