"""Animations for 01-calculus-behind-pytorch (keys calculus/...).
Every number shown is computed here with numpy or matches the page's verified PyTorch output."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from style import *


def plane(x0, y0, sx, sy):
    """Return a function mapping data coordinates to scene points."""
    return lambda x, y: np.array([x0 + x * sx, y0 + y * sy, 0.0])


class SecantToTangent(WikiScene):
    def construct(self):
        head = title("A derivative is the slope the secant settles on")
        self.add(head)
        P = plane(-6.2, -2.35, 1.55, 0.23)
        f = lambda x: x ** 2
        axes = VGroup(Line(P(0, 0), P(4.7, 0), color=MUTED), Line(P(0, 0), P(0, 21), color=MUTED))
        xt = VGroup(*[label(str(v), 18, MUTED).move_to(P(v, 0) + DOWN * 0.28) for v in range(1, 5)])
        yt = VGroup(*[label(str(v), 16, MUTED).next_to(P(0, v), LEFT, buff=0.1) for v in (5, 10, 15, 20)])
        curve = ParametricFunction(lambda t: P(t, f(t)), t_range=[0, 4.55], color=INFO, stroke_width=4)
        clbl = label("f(x) = x²", 22, INFO).move_to(P(1.2, 14))
        self.play(Create(axes), FadeIn(xt), FadeIn(yt), Create(curve), FadeIn(clbl), run_time=1.2)

        p_dot = Dot(P(3, 9), color=HIGHLIGHT, radius=0.09)
        p_lbl = label("(3, 9)", 18, HIGHLIGHT).next_to(p_dot, LEFT, buff=0.15)
        self.play(FadeIn(p_dot), FadeIn(p_lbl), run_time=0.4)

        def sec_line(m):
            return Line(P(1.8, 9 + m * (1.8 - 3)), P(4.5, 9 + m * (4.5 - 3)), color=ERROR, stroke_width=3)

        panel_x = RIGHT * 3.9
        rows = VGroup()
        line = q_dot = None
        for i, h in enumerate([1.0, 0.5, 0.1]):
            m = (f(3 + h) - f(3)) / h
            new_line = sec_line(m)
            new_q = Dot(P(3 + h, f(3 + h)), color=ERROR, radius=0.08)
            txt = label(f"h = {h:g}:  ({f(3 + h):g} − 9) / {h:g} = {m:.1f}", 22, FG, mono=True)
            txt.move_to(panel_x + UP * (1.6 - 0.65 * i))
            rows.add(txt)
            if line is None:
                self.play(Create(new_line), FadeIn(new_q), FadeIn(txt), run_time=0.9)
                line, q_dot = new_line, new_q
            else:
                self.play(Transform(line, new_line), Transform(q_dot, new_q), FadeIn(txt), run_time=0.9)
            self.wait(0.5)

        tangent = Line(P(1.8, 9 + 6 * (1.8 - 3)), P(4.5, 9 + 6 * 1.5), color=HIGHLIGHT, stroke_width=4)
        final = label("h → 0:  slope = 2x = 6", 26, HIGHLIGHT, weight=BOLD).move_to(panel_x + DOWN * 0.45)
        self.play(Transform(line, tangent), FadeOut(q_dot), FadeIn(final), run_time=1)
        self.play(FadeIn(caption_bar("Near x = 3, f behaves like a line of slope 6: nudge x by 0.001 and f moves by about 0.006")))
        self.wait(2.5)


class GraphBackprop(WikiScene):
    def construct(self):
        head = title("Backprop by hand:  a = 2, b = 3  →  f")
        self.add(head)
        spec = {
            "a": ("a", LEFT * 5.4 + UP * 1.25), "b": ("b", LEFT * 5.4 + DOWN * 1.75),
            "c": ("c = a + b", LEFT * 2.2 + UP * 1.25), "d": ("d = a · b", LEFT * 2.2 + DOWN * 1.75),
            "e": ("e = c · d", RIGHT * 1.2 + UP * 1.25), "f": ("f = e + d", RIGHT * 4.6 + DOWN * 0.25),
        }
        nodes = {}
        for k, (txt, pos) in spec.items():
            w = 1.0 if k in "ab" else 1.85
            box = RoundedRectangle(corner_radius=0.15, width=w, height=0.72, stroke_color=FG, stroke_width=2.5)
            box.set_fill(BG, 1).move_to(pos)
            nodes[k] = VGroup(box, label(txt, 21, FG, mono=True).move_to(box))
        edges_spec = [("a", "c"), ("b", "c"), ("a", "d"), ("b", "d"), ("c", "e"), ("d", "e"), ("e", "f"), ("d", "f")]
        edges = {}
        for u, v in edges_spec:
            edges[(u, v)] = Arrow(nodes[u][0].get_center(), nodes[v][0].get_center(), buff=0.5,
                                  stroke_width=2.5, color=MUTED, max_tip_length_to_length_ratio=0.08)
        self.play(*[GrowArrow(e) for e in edges.values()], *[FadeIn(n) for n in nodes.values()], run_time=1)

        # forward values (cyan, above each box)
        vals = {"a": "2", "b": "3", "c": "5", "d": "6", "e": "30", "f": "36"}
        fwd = {}
        for k in "abcdef":
            t = label(f"= {vals[k]}", 22, INFO, mono=True).next_to(nodes[k], UP, buff=0.1)
            t.add_background_rectangle(color=BG, opacity=1, buff=0.05)
            fwd[k] = t
        tag = label("forward: compute values", 22, INFO).to_edge(DOWN, buff=1.0)
        self.play(FadeIn(tag), FadeIn(fwd["a"]), FadeIn(fwd["b"]), run_time=0.5)
        for k in "cdef":
            self.play(FadeIn(fwd[k], shift=UP * 0.1), nodes[k][0].animate.set_stroke(INFO), run_time=0.45)
        self.wait(0.4)

        # backward gradients (pink, below each box)
        tag2 = label("backward: upstream × local, add where paths meet", 22, GRAD).move_to(tag)
        self.play(Transform(tag, tag2), run_time=0.5)
        grads = [("f", "∂f/∂f = 1", []),
                 ("e", "∂f/∂e = 1", [("e", "f")]),
                 ("c", "∂f/∂c = 1·d = 6", [("c", "e")]),
                 ("d", "∂f/∂d = 1 + 1·c = 6", [("d", "f"), ("d", "e")]),
                 ("a", "6·1 + 6·b = 24", [("a", "c"), ("a", "d")]),
                 ("b", "6·1 + 6·a = 18", [("b", "c"), ("b", "d")])]
        for k, txt, es in grads:
            g = label(txt, 20, GRAD, mono=True).next_to(nodes[k], DOWN, buff=0.1)
            g.add_background_rectangle(color=BG, opacity=1, buff=0.05)
            anims = [FadeIn(g, shift=DOWN * 0.1), nodes[k][0].animate.set_stroke(GRAD)]
            anims += [edges[e].animate.set_color(GRAD) for e in es]
            self.play(*anims, run_time=0.7 if k != "d" else 1.0)
            if k == "d":
                self.wait(0.6)
        final = label("a.grad = 24    b.grad = 18", 26, SUCCESS, weight=BOLD).move_to(tag)
        self.play(Transform(tag, final), run_time=0.6)
        self.play(FadeIn(caption_bar("d feeds two nodes, so its two path contributions add: 1 + 5 = 6")))
        self.wait(2.5)


class GradientDescent(WikiScene):
    def construct(self):
        head = title("Gradient descent on L(w) = (w − 3)²,  η = 0.1")
        self.add(head)
        P = plane(-5.9, -2.6, 0.95, 0.36)
        L = lambda w: (w - 3) ** 2
        axes = VGroup(Line(P(-0.5, 0), P(6.5, 0), color=MUTED), Line(P(-0.5, 0), P(-0.5, 13), color=MUTED))
        ticks = VGroup(*[label(str(v), 18, MUTED).move_to(P(v, 0) + DOWN * 0.28) for v in range(0, 7)])
        wl = label("w", 20, MUTED).next_to(axes[0], RIGHT, buff=0.1)
        curve = ParametricFunction(lambda t: P(t, L(t)), t_range=[-0.5, 6.5], color=INFO, stroke_width=4)
        minm = DashedLine(P(3, 0), P(3, 2.2), color=SUCCESS)
        minl = label("minimum w = 3", 18, SUCCESS).next_to(P(3, 2.2), UP, buff=0.08)
        self.play(Create(axes), FadeIn(ticks), FadeIn(wl), Create(curve), Create(minm), FadeIn(minl), run_time=1.2)

        rule = label("w ← w − 0.1 · 2(w − 3)", 24, HIGHLIGHT, mono=True).move_to(RIGHT * 3.9 + UP * 2.0)
        self.play(FadeIn(rule), run_time=0.4)

        w = 0.0
        ball = Dot(P(w, L(w)), radius=0.13, color=HIGHLIGHT)
        info = label("", 22).move_to(RIGHT * 3.9 + UP * 1.2)
        self.play(FadeIn(ball), run_time=0.3)
        trail = VGroup()
        for step in range(1, 9):
            slope = 2 * (w - 3)
            new_w = w - 0.1 * slope
            tan = Line(P(w - 0.6, L(w) - 0.6 * slope), P(w + 0.6, L(w) + 0.6 * slope), color=ERROR, stroke_width=3)
            txt = label(f"step {step}: slope {slope:+.3f} → w = {new_w:.4f}", 20, FG, mono=True).move_to(RIGHT * 3.9 + UP * 1.2)
            self.play(Create(tan), Transform(info, txt), run_time=0.35 if step > 3 else 0.6)
            ghost = Dot(P(w, L(w)), radius=0.06, color=MUTED)
            trail.add(ghost)
            self.add(ghost)
            self.play(ball.animate.move_to(P(new_w, L(new_w))), FadeOut(tan), run_time=0.45 if step > 3 else 0.7)
            w = new_w
        self.play(FadeIn(caption_bar("Big steps where the slope is steep, small steps near the bottom: they shrink by 20% each time")))
        self.wait(2.5)


class SigmoidDerivative(WikiScene):
    def construct(self):
        head = title("Sigmoid's slope is at most 0.25")
        self.add(head)
        sig = lambda z: 1 / (1 + np.exp(-z))
        dsig = lambda z: sig(z) * (1 - sig(z))
        Ptop = plane(0.55, 0.55, 0.55, 1.8)       # σ in [0, 1]
        Pbot = plane(0.55, -2.45, 0.55, 7.0)      # σ' in [0, 0.25]
        top_ax = VGroup(Line(Ptop(-6, 0), Ptop(6, 0), color=MUTED), Line(Ptop(0, 0), Ptop(0, 1.05), color=MUTED))
        bot_ax = VGroup(Line(Pbot(-6, 0), Pbot(6, 0), color=MUTED), Line(Pbot(0, 0), Pbot(0, 0.27), color=MUTED))
        s_curve = ParametricFunction(lambda t: Ptop(t, sig(t)), t_range=[-6, 6], color=INFO, stroke_width=4)
        d_curve = ParametricFunction(lambda t: Pbot(t, dsig(t)), t_range=[-6, 6], color=GRAD, stroke_width=4)
        s_lbl = label("σ(z)", 22, INFO).next_to(Ptop(6, 1), RIGHT, buff=0.15)
        d_lbl = label("σ′(z) = σ(1 − σ)", 22, GRAD).next_to(Pbot(6, 0.05), RIGHT, buff=0.15)
        one = label("1", 16, MUTED).next_to(Ptop(0, 1), LEFT, buff=0.08)
        q = label("0.25", 16, MUTED).next_to(Pbot(0, 0.25), LEFT, buff=0.08)
        ticks = VGroup(*[label(str(v), 16, MUTED).move_to(Pbot(v, 0) + DOWN * 0.25) for v in (-4, -2, 0, 2, 4)])
        self.play(Create(top_ax), Create(bot_ax), Create(s_curve), Create(d_curve), FadeIn(s_lbl), FadeIn(d_lbl),
                  FadeIn(one), FadeIn(q), FadeIn(ticks), run_time=1.3)

        z = ValueTracker(-5.0)
        dot_t = always_redraw(lambda: Dot(Ptop(z.get_value(), sig(z.get_value())), color=HIGHLIGHT, radius=0.08))
        dot_b = always_redraw(lambda: Dot(Pbot(z.get_value(), dsig(z.get_value())), color=HIGHLIGHT, radius=0.08))
        tan = always_redraw(lambda: Line(
            Ptop(z.get_value() - 1.0, sig(z.get_value()) - dsig(z.get_value())),
            Ptop(z.get_value() + 1.0, sig(z.get_value()) + dsig(z.get_value())), color=HIGHLIGHT, stroke_width=3))
        readout = always_redraw(lambda: label(f"z = {z.get_value():+.1f}\nslope = {dsig(z.get_value()):.4f}",
                                              22, HIGHLIGHT, mono=True).move_to(LEFT * 4.7 + UP * 1.6))
        self.add(dot_t, dot_b, tan, readout)
        self.play(z.animate.set_value(0.0), run_time=3, rate_func=linear)
        self.wait(0.6)
        self.play(z.animate.set_value(5.0), run_time=2.5, rate_func=linear)
        chain = label(f"5 sigmoid layers:\n0.25⁵ = {0.25 ** 5:.4f}", 24, ERROR, weight=BOLD).move_to(LEFT * 4.85 + DOWN * 0.6)
        self.play(FadeIn(chain), run_time=0.6)
        self.play(FadeIn(caption_bar("The chain rule multiplies one of these slopes per layer: vanishing gradients")))
        self.wait(2.5)
