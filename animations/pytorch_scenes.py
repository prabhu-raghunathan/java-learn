"""Animations for 01-pytorch-fundamentals.md"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from style import *


# ----------------------------------------------------------------------------
# 1. Broadcasting (4,3) + (3,) and an incompatible (4,3) + (4,)
# ----------------------------------------------------------------------------
class Broadcasting(WikiScene):
    def construct(self):
        head = title("Broadcasting: (4, 3) + (3,) → (4, 3), with no copying")
        self.add(head)
        cell = 0.62
        A = grid_of_numbers([[1, 1, 1]] * 4, cell=cell, size=24, stroke=INFO, fill=INFO, fill_opacity=0.12)
        A.move_to(LEFT * 4.6 + DOWN * 0.3)
        b = grid_of_numbers([[10, 20, 30]], cell=cell, size=24, stroke=HIGHLIGHT, fill=HIGHLIGHT, fill_opacity=0.15)
        b.next_to(A, RIGHT, buff=1.4).align_to(A, UP)
        plus = label("+", 40).move_to((A.get_right() + b.get_left()) / 2)
        a_lbl = label("A  (4, 3)", 22, INFO).next_to(A, UP, buff=0.2)
        b_lbl = label("b  (3,)", 22, HIGHLIGHT).next_to(b, UP, buff=0.2)
        self.play(FadeIn(A), FadeIn(b), FadeIn(plus), FadeIn(a_lbl), FadeIn(b_lbl), run_time=1)

        rule = VGroup(
            label("align shapes from the RIGHT:", 22, FG),
            label("A : (4, 3)", 24, INFO, mono=True),
            label("b :    (3)", 24, HIGHLIGHT, mono=True),
            label("missing dim → treated as 1", 20, MUTED),
            label("b : (1, 3) → stretched to (4, 3)", 22, HIGHLIGHT, mono=True),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to(RIGHT * 3.6 + UP * 0.6)
        for line in rule:
            self.play(FadeIn(line, shift=RIGHT * 0.2), run_time=0.45)

        ghosts = VGroup()
        for r in range(1, 4):
            g = b.copy().set_opacity(0.35).shift(DOWN * cell * r)
            for cellg in g[0]:
                cellg[0].set_stroke(opacity=0.5)
            ghosts.add(g)
        stride_note = label("virtual rows: stride 0, same memory", 19, MUTED).next_to(b, DOWN, buff=cell * 3 + 0.25)
        self.play(LaggedStart(*[TransformFromCopy(b, g) for g in ghosts], lag_ratio=0.25), FadeIn(stride_note),
                  run_time=1.3)
        self.wait(0.5)

        res = grid_of_numbers([[11, 21, 31]] * 4, cell=cell, size=24, stroke=SUCCESS, fill=SUCCESS, fill_opacity=0.15)
        res.next_to(b, RIGHT, buff=1.4).align_to(A, UP)
        eq = label("=", 40).move_to((b.get_right() + res.get_left()) / 2 + DOWN * cell * 1.5)
        plus.generate_target().move_to((A.get_right() + b.get_left()) / 2 + DOWN * cell * 1.5)
        self.play(FadeOut(rule), MoveToTarget(plus), run_time=0.5)
        r_lbl = label("(4, 3)", 22, SUCCESS).next_to(res, UP, buff=0.2)
        self.play(FadeIn(eq), FadeIn(res), FadeIn(r_lbl), run_time=0.9)
        everything = VGroup(A, a_lbl, plus, b, b_lbl, ghosts, stride_note, eq, res, r_lbl)
        self.play(everything.animate.move_to([0, everything.get_center()[1] + 0.2, 0]), run_time=0.7)
        self.wait(1)

        bad = VGroup(
            label("(4, 3) + (4,)", 26, FG, mono=True),
            label("compare last dims: 3 vs 4, neither is 1", 22, ERROR),
            label("RuntimeError: size mismatch", 22, ERROR, mono=True),
        ).arrange(DOWN, buff=0.2).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(bad[0]), run_time=0.4)
        self.play(FadeIn(bad[1]), FadeIn(bad[2]), run_time=0.7)
        self.wait(2.5)


# ----------------------------------------------------------------------------
# 2. Autograd: forward values, then the chain rule backwards (x used twice)
# ----------------------------------------------------------------------------
class AutogradBackward(WikiScene):
    def construct(self):
        head = title("Autograd: y = x² + 3x + 1 at x = 2  →  x.grad = 7")
        self.add(head)

        def node(text, color, pos, w=1.7):
            box = RoundedRectangle(corner_radius=0.15, width=w, height=0.75, color=color, stroke_width=3)
            box.set_fill(color, 0.12).move_to(pos)
            return VGroup(box, label(text, 22, FG, mono=True).move_to(box))

        X = node("x", INFO, [-5.2, 0, 0], w=1.2)
        SQ = node("pow: x²", ACCENT, [-2.4, 1.3, 0])
        MUL = node("mul: 3·x", ACCENT, [-2.4, -1.3, 0])
        ADD = node("add", ACCENT, [0.6, 0, 0], w=1.2)
        ADD1 = node("add: +1", ACCENT, [3.0, 0, 0])
        Y = node("y", PURPLE, [5.5, 0, 0], w=1.2)
        edges = [(X, SQ), (X, MUL), (SQ, ADD), (MUL, ADD), (ADD, ADD1), (ADD1, Y)]
        arrows = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=0.1, color=MUTED, stroke_width=3,
                                max_tip_length_to_length_ratio=0.12) for a, b in edges])
        leaf = label("leaf · requires_grad=True", 18, MUTED).next_to(X, DOWN, buff=0.2)
        self.play(*[FadeIn(n) for n in (X, SQ, MUL, ADD, ADD1, Y)], Create(arrows), FadeIn(leaf), run_time=1.2)

        phase = label("FORWARD: record ops, compute values", 24, INFO).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(phase))
        vals = [(X, "2"), (SQ, "4"), (MUL, "6"), (ADD, "10"), (ADD1, "11"), (Y, "11")]
        val_mobs = {}
        for i, (n, v) in enumerate(vals):
            t = label(v, 24, INFO, weight=BOLD).next_to(n, UP, buff=0.12)
            val_mobs[id(n)] = t
            anims = [FadeIn(t, shift=DOWN * 0.1)]
            if i in (1, 2):
                anims.append(ShowPassingFlash(arrows[i - 1].copy().set_color(INFO).set_stroke(width=6), time_width=0.7))
            elif i >= 3:
                anims.append(ShowPassingFlash(arrows[i if i < 5 else 5].copy().set_color(INFO).set_stroke(width=6),
                                              time_width=0.7))
            self.play(*anims, run_time=0.5)
        self.wait(0.5)

        self.play(Transform(phase, label("BACKWARD: chain rule from y, local gradient × upstream", 24, GRAD)
                            .to_edge(DOWN, buff=0.5)), *[FadeOut(t) for t in val_mobs.values()], run_time=0.6)

        def grad_tag(n, text, direction=DOWN):
            return label(text, 20, GRAD, weight=BOLD).next_to(n, direction, buff=0.15)

        steps = [
            (Y, "∂y/∂y = 1", None),
            (ADD1, "1 × 1 = 1", 5),
            (ADD, "1 × 1 = 1", 4),
        ]
        tags = []
        for n, txt, ai in steps:
            anims = [FadeIn(grad_tag(n, txt))]
            if ai is not None:
                rev = Arrow(arrows[ai].get_end(), arrows[ai].get_start(), buff=0, color=GRAD, stroke_width=6)
                anims.append(ShowPassingFlash(rev, time_width=0.8))
            self.play(*anims, run_time=0.7)

        sq_t = grad_tag(SQ, "local 2x = 4  →  4", UP)
        mul_t = grad_tag(MUL, "local 3  →  3", DOWN)
        self.play(ShowPassingFlash(Line(ADD.get_left(), SQ.get_right()).set_color(GRAD).set_stroke(width=6), time_width=0.8),
                  ShowPassingFlash(Line(ADD.get_left(), MUL.get_right()).set_color(GRAD).set_stroke(width=6), time_width=0.8),
                  FadeIn(sq_t), FadeIn(mul_t), run_time=0.9)
        self.play(ShowPassingFlash(Line(SQ.get_left(), X.get_right()).set_color(GRAD).set_stroke(width=6), time_width=0.8),
                  ShowPassingFlash(Line(MUL.get_left(), X.get_right()).set_color(GRAD).set_stroke(width=6), time_width=0.8),
                  run_time=0.8)
        acc = label("x is used twice → its gradients ADD:  4 + 3 = 7", 24, HIGHLIGHT).to_edge(DOWN, buff=0.5)
        res = label("x.grad = 7", 28, HIGHLIGHT, weight=BOLD, mono=True).next_to(X, DOWN, buff=0.25)
        self.play(FadeOut(phase), FadeIn(acc), FadeOut(leaf), run_time=0.7)
        self.play(TransformFromCopy(acc, res), run_time=0.8)
        check = math(r"\tfrac{dy}{dx} = 2x + 3 = 7\ \checkmark", 30, SUCCESS).next_to(res, DOWN, buff=0.25)
        self.play(Write(check))
        self.wait(2.5)


# ----------------------------------------------------------------------------
# 3. zero_grad(): gradients accumulate unless cleared
# ----------------------------------------------------------------------------
class ZeroGradAccumulation(WikiScene):
    def construct(self):
        head = title("Why optimizer.zero_grad()? PyTorch ADDS new gradients into .grad")
        self.add(head)

        loop = VGroup(*[label(t, 22, FG, mono=True) for t in [
            "optimizer.zero_grad()", "pred = model(X)", "loss = loss_fn(pred, y)",
            "loss.backward()", "optimizer.step()",
        ]]).arrange(DOWN, aligned_edge=LEFT, buff=0.32).move_to(LEFT * 4.3 + DOWN * 0.2)
        cursor = SurroundingRectangle(loop[0], color=HIGHLIGHT, buff=0.1)
        it_lbl = label("iteration 1", 24, HIGHLIGHT).next_to(loop, UP, buff=0.4)
        self.play(FadeIn(loop), Create(cursor), FadeIn(it_lbl))

        base_y = -2.3
        scale = 0.42
        cols = {"with": (0.5, SUCCESS, "with zero_grad()"), "without": (4.7, ERROR, "WITHOUT zero_grad()")}
        bars = {}
        for key, (x, col, name) in cols.items():
            axis = Line([x - 1.0, base_y, 0], [x + 1.0, base_y, 0], color=MUTED)
            nm = label(name, 22, col, weight=BOLD).move_to([x, 2.35, 0])
            bar = Rectangle(width=1.0, height=0.001, stroke_width=0).set_fill(col, 0.8)
            bar.move_to([x, base_y, 0], aligned_edge=DOWN)
            val = label(".grad = 0", 22, FG, mono=True).next_to(axis, DOWN, buff=0.2)
            self.play(Create(axis), FadeIn(nm), FadeIn(val), run_time=0.4)
            bars[key] = dict(x=x, col=col, bar=bar, val=val, total=0.0)
            self.add(bar)

        grads = [4.0, 3.0, 2.5]   # gradient computed by backward() in each iteration
        for it, g in enumerate(grads, start=1):
            self.play(Transform(it_lbl, label(f"iteration {it}", 24, HIGHLIGHT).move_to(it_lbl)), run_time=0.3)
            # zero_grad (only affects the "with" column)
            self.play(cursor.animate.become(SurroundingRectangle(loop[0], color=HIGHLIGHT, buff=0.1)), run_time=0.3)
            w = bars["with"]
            w["total"] = 0.0
            self.play(w["bar"].animate.stretch_to_fit_height(0.001).move_to([w["x"], bars["with"]["bar"].get_bottom()[1], 0], aligned_edge=DOWN),
                      Transform(w["val"], label(".grad = 0", 22, FG, mono=True).move_to(w["val"])), run_time=0.4)
            for line in (1, 2):
                self.play(cursor.animate.become(SurroundingRectangle(loop[line], color=HIGHLIGHT, buff=0.1)), run_time=0.25)
            self.play(cursor.animate.become(SurroundingRectangle(loop[3], color=GRAD, buff=0.1)), run_time=0.3)
            anims = []
            for key in ("with", "without"):
                d = bars[key]
                d["total"] += g
                h = d["total"] * scale
                target = Rectangle(width=1.0, height=h, stroke_width=0).set_fill(d["col"], 0.8)
                target.move_to([d["x"], base_y, 0], aligned_edge=DOWN)
                anims.append(Transform(d["bar"], target))
                expr = f".grad = {d['total']:g}" if key == "with" or it == 1 else \
                    f".grad = {' + '.join(f'{x:g}' for x in grads[:it])} = {d['total']:g}"
                anims.append(Transform(d["val"], label(expr, 20, FG, mono=True).move_to([d["x"], d["val"].get_center()[1], 0])))
            self.play(*anims, run_time=0.8)
            self.play(cursor.animate.become(SurroundingRectangle(loop[4], color=HIGHLIGHT, buff=0.1)), run_time=0.3)
            self.wait(0.4)

        self.play(FadeIn(caption_bar("step() uses .grad: without zero_grad() stale gradients pile up → bigger and bigger updates",
                                     color=FG)))
        self.wait(3)
