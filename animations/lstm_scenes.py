"""Animations for the LSTM and GRU pages (keys lstm/...).
Every number shown is computed (NumPy / PyTorch) and matches the verified outputs quoted on the pages.
Text-only labels (no MathTex), so no LaTeX/dvisvgm is needed to render these scenes."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from style import *
from rnn_scenes import sym, chip, cell_box, PARAM


def num_column(values, color=FG, fmt="{:+.2f}", size=20, w=1.1, h=0.36):
    """A column vector drawn as stacked number cells."""
    cells = VGroup(*[chip(fmt.format(v), color, size, 0.12, w=w, h=h) for v in values])
    return cells.arrange(DOWN, buff=0.04)


def op(text, size=34, color=MUTED):
    return label(text, size, color, weight=BOLD)


# ---------------------------------------------------------------------------
# Page 1: the cell-state update on the notebook's 4-dimension demo
# c_t = f * c_{t-1} + i * c~_t   (values from the notebook's interpretation cell)
# ---------------------------------------------------------------------------
C_PREV = [0.9, 0.2, -0.8, 0.5]
F_GATE = [1.0, 0.9, 0.1, 0.0]
I_GATE = [0.0, 0.2, 0.8, 1.0]
C_CAND = [0.3, 0.7, 0.4, -0.6]
KEEP = [f * c for f, c in zip(F_GATE, C_PREV)]           # [0.9, 0.18, -0.08, 0.0]
WRITE = [i * c for i, c in zip(I_GATE, C_CAND)]          # [0.0, 0.14, 0.32, -0.6]
C_NEW = [k + w for k, w in zip(KEEP, WRITE)]             # [0.9, 0.32, 0.24, -0.6]
NOTES = ["kept as is", "mostly kept, a little added", "mostly rewritten", "fully replaced"]


class CellUpdate(WikiScene):
    def construct(self):
        self.add(title("The LSTM cell update: keep some old memory, write some new"))
        top_y, bot_y = 1.2, -1.3
        xs = [-5.4, -3.35, -1.3]

        def row(vals_a, name_a, col_a, vals_b, name_b, col_b, y, res_vals, res_name, res_col):
            a = num_column(vals_a, col_a).move_to([xs[0], y, 0])
            b = num_column(vals_b, col_b).move_to([xs[1], y, 0])
            la = sym(*name_a, 24, col_a).next_to(a, UP, buff=0.14)
            lb = sym(*name_b, 24, col_b).next_to(b, UP, buff=0.14)
            times = op("x").move_to([(xs[0] + xs[1]) / 2, y, 0])
            eq = op("=").move_to([(xs[1] + xs[2]) / 2, y, 0])
            r = num_column(res_vals, res_col).move_to([xs[2], y, 0])
            lr = label(res_name, 20, res_col).next_to(r, UP, buff=0.14)
            self.play(FadeIn(a), FadeIn(la), run_time=0.45)
            self.play(FadeIn(times), FadeIn(b), FadeIn(lb), run_time=0.45)
            self.play(FadeIn(eq), TransformFromCopy(VGroup(a, b), r), FadeIn(lr), run_time=0.8)
            return r

        keep = row(C_PREV, ("c", "t-1"), INFO, F_GATE, ("f", "t"), PURPLE, top_y,
                   KEEP, "kept memory", ACCENT)
        self.wait(0.3)
        write = row(I_GATE, ("i", "t"), PURPLE, C_CAND, ("c~", "t"), PARAM, bot_y,
                    WRITE, "new memory", ACCENT)
        self.wait(0.3)

        plus = op("+", 40, FG).move_to([0.3, (top_y + bot_y) / 2, 0])
        c_new = num_column(C_NEW, SUCCESS).move_to([1.75, (top_y + bot_y) / 2, 0])
        lc = sym("c", "t", 26, SUCCESS).next_to(c_new, UP, buff=0.14)
        self.play(FadeIn(plus), run_time=0.3)
        self.play(TransformFromCopy(VGroup(keep, write), c_new), FadeIn(lc), run_time=1.0)
        frame = SurroundingRectangle(c_new, color=HIGHLIGHT, buff=0.08, stroke_width=3)
        self.play(Create(frame), run_time=0.4)

        notes = VGroup()
        for cell, txt in zip(c_new, NOTES):
            notes.add(label(txt, 19, FG).next_to(cell, RIGHT, buff=0.35))
        self.play(LaggedStart(*[FadeIn(n, shift=RIGHT * 0.15) for n in notes], lag_ratio=0.3), run_time=1.4)
        self.play(FadeIn(caption_bar("Each dimension gets its own gate values: f near 1 keeps, i near 1 writes")))
        self.wait(2.6)


# ---------------------------------------------------------------------------
# Page 1: measured gradient flow through 50 steps (untrained nn.RNN / nn.LSTM)
# ||d sum(h_50) / d x_k|| for each input, T = 50, D = 8, H = 32, torch.manual_seed(0) weights,
# torch.manual_seed(1) input. Index = steps back from the last step. Verified on the page.
# ---------------------------------------------------------------------------
G_RNN = [1.88, 0.898, 0.548, 0.208, 0.155, 0.0942, 0.0235, 0.0314, 0.00739, 0.00473, 0.00318, 0.00176,
         0.000663, 0.000806, 0.000262, 7.33e-05, 8.1e-05, 2.14e-05, 1.36e-05, 8.28e-06, 5.64e-06, 3.22e-06,
         1.62e-06, 7.26e-07, 3.29e-07, 1.37e-07, 9.21e-08, 4.95e-08, 2.41e-08, 1.7e-08, 7.29e-09, 3.1e-09,
         1.33e-09, 9.01e-10, 3.59e-10, 1.63e-10, 1.34e-10, 5.27e-11, 4.79e-11, 1.65e-11, 9.57e-12, 5.25e-12,
         1.68e-12, 1.06e-12, 5.09e-13, 4.45e-13, 2.04e-13, 1.2e-13, 7.05e-14, 3.16e-14]
G_LSTM = [0.496, 0.233, 0.113, 0.0526, 0.0298, 0.0201, 0.00921, 0.00538, 0.00278, 0.0018, 0.00107, 0.000657,
          0.000411, 0.000293, 0.00015, 0.000124, 5.18e-05, 3.9e-05, 3.32e-05, 1.92e-05, 1.16e-05, 6.93e-06,
          4.78e-06, 2.99e-06, 2.17e-06, 1.09e-06, 6.47e-07, 4.44e-07, 3.34e-07, 1.97e-07, 9.12e-08, 6.88e-08,
          2.67e-08, 1.92e-08, 1.02e-08, 6.15e-09, 4.3e-09, 2.86e-09, 1.37e-09, 9.01e-10, 6.77e-10, 4.67e-10,
          2.55e-10, 1.97e-10, 1.3e-10, 8.16e-11, 5.54e-11, 3.79e-11, 2.44e-11, 1.79e-11]
G_LSTM_FB3 = [0.195, 0.103, 0.0959, 0.0859, 0.0732, 0.0511, 0.0537, 0.0463, 0.0446, 0.0445, 0.045, 0.0421,
              0.0423, 0.0407, 0.0422, 0.0404, 0.0386, 0.0465, 0.0478, 0.0434, 0.0397, 0.0396, 0.043, 0.0432,
              0.0448, 0.0381, 0.0437, 0.0468, 0.0364, 0.0401, 0.0349, 0.0455, 0.0391, 0.0372, 0.0311, 0.0348,
              0.0283, 0.0322, 0.0262, 0.0256, 0.0244, 0.0257, 0.0374, 0.0311, 0.0352, 0.0346, 0.0365, 0.0418,
              0.0429, 0.0526]


class GradientHighway(WikiScene):
    def construct(self):
        self.add(title("Measured: how much gradient reaches an input k steps back?"))
        ax = Axes(x_range=[0, 49, 10], y_range=[0, 15, 1], x_length=9.0, y_length=4.6,
                  axis_config={"color": MUTED, "stroke_width": 2}, tips=False).move_to([-0.6, 0.15, 0])
        self.play(Create(ax), run_time=0.6)

        def y(v):                      # log10 shifted so that 1e-14 sits on the x-axis
            return np.log10(v) + 14

        for e in (0, -4, -8, -12):
            self.add(label("1" if e == 0 else f"1e{e}", 16, MUTED, mono=True).next_to(ax.c2p(0, e + 14), LEFT, buff=0.12))
        for k in (0, 10, 20, 30, 40, 49):
            self.add(label(str(k), 16, MUTED, mono=True).next_to(ax.c2p(k, 0), DOWN, buff=0.12))
        self.add(label("steps back from the last time step", 18, MUTED).next_to(ax.c2p(24.5, 0), DOWN, buff=0.45))
        self.add(label("gradient norm (log)", 18, MUTED).rotate(PI / 2).next_to(ax.c2p(0, 7.5), LEFT, buff=0.85))

        series = [("vanilla RNN", G_RNN, ERROR),
                  ("LSTM, default init (mean f = 0.51)", G_LSTM, PARAM),
                  ("LSTM, forget bias +3 (mean f = 0.95)", G_LSTM_FB3, SUCCESS)]
        legend_y = 3.4
        for name, vals, col in series:
            pts = [ax.c2p(k, y(v)) for k, v in enumerate(vals)]
            line = VMobject(stroke_color=col, stroke_width=4).set_points_as_corners(pts)
            self.play(Create(line), run_time=1.8, rate_func=linear)
            end = label(f"{vals[-1]:.1e}", 18, col, mono=True).next_to(pts[-1], RIGHT, buff=0.15)
            key = VGroup(Line(ORIGIN, RIGHT * 0.45, color=col, stroke_width=5),
                         label(name, 18, col)).arrange(RIGHT, buff=0.15)
            key.move_to(ax.c2p(2, legend_y), aligned_edge=LEFT)
            legend_y -= 1.0
            self.play(FadeIn(end), FadeIn(key), run_time=0.4)
            self.wait(0.3)
        self.play(FadeIn(caption_bar("Same 50 inputs, untrained models: the cell path only helps when the forget gate is near 1")))
        self.wait(2.6)


# ---------------------------------------------------------------------------
# Page 1: GRU update as interpolation (notebook demo values)
# h_t = z * h_{t-1} + (1 - z) * h~_t
# ---------------------------------------------------------------------------
GH_PREV = [0.9, 0.2, -0.7, 0.5]
GZ = [0.95, 0.70, 0.20, 0.05]
GH_CAND = [0.1, 0.8, 0.4, -0.6]
GH_NEW = [z * a + (1 - z) * b for z, a, b in zip(GZ, GH_PREV, GH_CAND)]   # [0.86, 0.38, 0.18, -0.545]


class GruBlend(WikiScene):
    def construct(self):
        self.add(title("GRU update: slide between the old state and the candidate"))
        ys = [1.75, 0.6, -0.55, -1.7]
        x_left, x_right = -3.4, 3.4

        def xpos(v):
            return x_left + (v + 1) / 2 * (x_right - x_left)

        lines = VGroup()
        for k, y in enumerate(ys):
            ln = NumberLine(x_range=[-1, 1, 0.5], length=x_right - x_left, color=MUTED,
                            include_numbers=False, stroke_width=2).move_to([0, y, 0])
            lines.add(ln)
            self.add(label(f"dim {k + 1}", 18, MUTED).next_to(ln, LEFT, buff=0.3))
        for v in (-1, 0, 1):
            self.add(label(f"{v:g}", 15, MUTED, mono=True).move_to([xpos(v), ys[-1] - 0.38, 0]))
        self.play(Create(lines), run_time=0.6)

        old_dots, cand_dots = VGroup(), VGroup()
        for y, a, b in zip(ys, GH_PREV, GH_CAND):
            old_dots.add(Dot([xpos(a), y, 0], radius=0.1, color=INFO))
            cand_dots.add(Dot([xpos(b), y, 0], radius=0.1, color=PURPLE))
        key_old = VGroup(Dot(radius=0.08, color=INFO), label("old state h(t-1)", 18, INFO)).arrange(RIGHT, buff=0.12)
        key_new = VGroup(Dot(radius=0.08, color=PURPLE), label("candidate h~(t)", 18, PURPLE)).arrange(RIGHT, buff=0.12)
        keys = VGroup(key_old, key_new).arrange(RIGHT, buff=0.6).move_to([0, 2.6, 0])
        self.play(FadeIn(old_dots), FadeIn(key_old), run_time=0.5)
        self.play(FadeIn(cand_dots), FadeIn(key_new), run_time=0.5)
        self.wait(0.3)

        zs = VGroup(*[label(f"z = {z:.2f}", 19, HIGHLIGHT, mono=True).move_to([4.55, y, 0]) for z, y in zip(GZ, ys)])
        self.play(FadeIn(zs), run_time=0.5)
        movers = VGroup(*[Dot([xpos(a), y, 0], radius=0.13, color=SUCCESS) for a, y in zip(GH_PREV, ys)])
        self.play(FadeIn(movers), run_time=0.3)
        self.play(*[m.animate.move_to([xpos(v), y, 0]) for m, v, y in zip(movers, GH_NEW, ys)], run_time=1.6)
        vals = VGroup(*[label(f"h = {v:+.3f}", 19, SUCCESS, mono=True).move_to([4.55, y - 0.32, 0])
                        for v, y in zip(GH_NEW, ys)])
        self.play(FadeIn(vals), run_time=0.5)
        self.play(FadeIn(caption_bar("z near 1 keeps the old value (dim 1); z near 0 takes the candidate (dim 4)")))
        self.wait(2.6)


# ---------------------------------------------------------------------------
# Page 2: autoregressive generation, greedy, from the trained character LSTM
# Seed "deep " (state warmed up once), then each sampled char is fed back.
# Top-1 probabilities from the page's verified run.
# ---------------------------------------------------------------------------
GEN_STEPS = [("l", 0.997), ("e", 1.000), ("a", 0.967), ("r", 0.994),
             ("n", 0.997), ("i", 0.996), ("n", 0.999), ("g", 0.999)]


class AutoregressiveLoop(WikiScene):
    def construct(self):
        self.add(title("Generation: predict one character, feed it back, repeat"))
        model = cell_box("LSTM + Linear", ACCENT, 3.0, 1.0).move_to([0, 0.3, 0])
        state = label("carries (h, c)", 18, MUTED).next_to(model, DOWN, buff=0.15)
        loop = CurvedArrow(model.get_right() + UP * 0.25, model.get_right() + DOWN * 0.25, angle=-4.0,
                           color=SUCCESS, stroke_width=4)
        self.play(FadeIn(model), FadeIn(state), Create(loop), run_time=0.6)

        text = label('"deep "', 30, INFO, mono=True).move_to([0, 2.25, 0])
        warm = label("1. warm up: run the seed once", 20, FG).move_to([0, -1.25, 0])
        self.play(FadeIn(text), FadeIn(warm), run_time=0.5)
        down = Arrow(text.get_bottom(), model.get_top(), buff=0.1, color=INFO, stroke_width=4)
        self.play(GrowArrow(down), model[0].animate.set_fill(HIGHLIGHT, 0.35), run_time=0.6)
        self.play(model[0].animate.set_fill(ACCENT, 0.12), FadeOut(down), run_time=0.3)
        rule = label("2. take the last logits, softmax, pick a character, append it, feed only it back", 20, FG)
        rule.move_to([0, -1.25, 0])
        self.play(Transform(warm, rule), run_time=0.5)

        generated = "deep "
        for k, (ch, p) in enumerate(GEN_STEPS):
            fast = k >= 3
            pred = chip(f"'{ch}'  p = {p:.3f}", HIGHLIGHT, 20, 0.2).move_to([-4.0, 0.3, 0])
            out_arrow = Arrow(model.get_left(), pred.get_right(), buff=0.1, color=HIGHLIGHT,
                              stroke_width=3)
            self.play(FadeIn(pred), GrowArrow(out_arrow), model[0].animate.set_fill(HIGHLIGHT, 0.35),
                      run_time=0.25 if fast else 0.5)
            generated += ch
            new_text = label(f'"{generated}"', 30, INFO, mono=True).move_to(text)
            back = CurvedArrow(pred.get_top(), text.get_left() + LEFT * 0.1, angle=-1.0, color=INFO,
                               stroke_width=3)
            self.play(Create(back), run_time=0.2 if fast else 0.45)
            self.play(Transform(text, new_text), FadeOut(back), FadeOut(pred), FadeOut(out_arrow),
                      model[0].animate.set_fill(ACCENT, 0.12), run_time=0.25 if fast else 0.45)
        self.play(FadeIn(caption_bar("Each new character is one more step for the same LSTM: the state carries the prefix")))
        self.wait(2.6)


# ---------------------------------------------------------------------------
# Page 2: temperature reshapes the softmax (notebook's logits [2.5, 1.2, 0.5, -0.3])
# ---------------------------------------------------------------------------
LOGITS = np.array([2.5, 1.2, 0.5, -0.3])
TEMPS = [1.0, 0.5, 1.5, 2.0]


def softmax_t(t):
    e = np.exp(LOGITS / t)
    return e / e.sum()


class TemperatureBars(WikiScene):
    def construct(self):
        self.add(title("Temperature: divide the logits by T, then softmax"))
        base_y, hmax, w, gap = -2.0, 3.6, 1.2, 0.55
        xs = [(-1.5 * (w + gap)) + k * (w + gap) - 1.6 for k in range(4)]
        self.add(Line([xs[0] - 0.9, base_y, 0], [xs[-1] + 0.9, base_y, 0], color=MUTED, stroke_width=2))
        for x, z in zip(xs, LOGITS):
            self.add(label(f"logit {z:+.1f}", 17, MUTED, mono=True).move_to([x, base_y - 0.35, 0]))

        def bars_for(t):
            p = softmax_t(t)
            g = VGroup()
            for x, v in zip(xs, p):
                h = max(hmax * v, 0.02)
                r = Rectangle(width=w, height=h, stroke_width=0).set_fill(INFO if v < p.max() else HIGHLIGHT, 0.85)
                r.move_to([x, base_y + h / 2, 0])
                num = label(f"{v:.4f}", 19, FG, mono=True).next_to(r, UP, buff=0.1)
                g.add(VGroup(r, num))
            return g

        tl = label("T = 1.0", 30, HIGHLIGHT, mono=True).move_to([4.6, 1.3, 0])
        note = label("plain softmax", 20, FG).next_to(tl, DOWN, buff=0.25)
        bars = bars_for(1.0)
        self.play(FadeIn(bars), FadeIn(tl), FadeIn(note), run_time=0.8)
        self.wait(1.0)
        notes = {0.5: "sharper: safer, repetitive", 1.5: "flatter: more variety", 2.0: "flatter still: riskier"}
        for t in TEMPS[1:]:
            new_tl = label(f"T = {t:.1f}", 30, HIGHLIGHT, mono=True).move_to(tl)
            new_note = label(notes[t], 20, FG).next_to(new_tl, DOWN, buff=0.25)
            self.play(Transform(bars, bars_for(t)), Transform(tl, new_tl), Transform(note, new_note), run_time=1.1)
            self.wait(1.0)
        self.play(FadeIn(caption_bar("Same model, same logits: T only changes how we sample, never the training")))
        self.wait(2.6)
