"""Animations for 02-artificial-neural-networks.md"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from style import *


# ----------------------------------------------------------------------------
# 1. Forward pass, then backprop (with one dead ReLU blocking its gradient)
# ----------------------------------------------------------------------------
class ForwardBackward(WikiScene):
    def construct(self):
        head = title("Forward pass computes ŷ · backward pass sends the error back")
        self.add(head)

        xs = {"in": -5.2, "hid": -1.6, "out": 2.0, "loss": 5.0}
        in_ys = [0.5, -1.1]
        hid_ys = [1.35, 0.25, -0.85, -1.95]
        dead = 2  # this hidden unit has z < 0 → ReLU outputs 0

        ins = VGroup(*[Circle(0.32, color=INFO, stroke_width=3).set_fill(INFO, 0.25).move_to([xs["in"], y, 0])
                       for y in in_ys])
        hid = VGroup(*[Circle(0.32, color=ACCENT, stroke_width=3).set_fill(ACCENT, 0.25).move_to([xs["hid"], y, 0])
                       for y in hid_ys])
        out = Circle(0.36, color=PURPLE, stroke_width=3).set_fill(PURPLE, 0.25).move_to([xs["out"], -0.3, 0])
        loss = RoundedRectangle(corner_radius=0.15, width=1.4, height=0.8, color=HIGHLIGHT).move_to([xs["loss"], -0.3, 0])
        loss_t = label("Loss", 24, HIGHLIGHT).move_to(loss)

        e1 = VGroup(*[Line(a.get_right(), b.get_left(), stroke_width=2, color=MUTED) for a in ins for b in hid])
        e2 = VGroup(*[Line(h.get_right(), out.get_left(), stroke_width=2, color=MUTED) for h in hid])
        e3 = Line(out.get_right(), loss.get_left(), stroke_width=2, color=MUTED)

        layer_lbls = VGroup(
            label("X  (N, 2)", 20, MUTED).next_to(ins, DOWN, buff=0.5),
            label("hidden  (N, 4)", 20, MUTED).next_to(hid, DOWN, buff=0.25),
            label("ŷ  (N, 1)", 20, MUTED).next_to(out, DOWN, buff=0.35),
        )
        self.play(FadeIn(ins), FadeIn(hid), FadeIn(out), Create(loss), FadeIn(loss_t),
                  Create(e1), Create(e2), Create(e3), FadeIn(layer_lbls), run_time=1.2)

        # ---------------- forward ----------------
        phase = label("FORWARD", 28, INFO, weight=BOLD).move_to([-5.2, 2.3, 0])
        self.play(FadeIn(phase))
        f1 = math(r"Z^{(1)} = XW_1^\top + b_1", 30, INFO).next_to(hid, UP, buff=0.25)
        self.play(ShowPassingFlash(e1.copy().set_color(INFO).set_stroke(width=5), time_width=0.6),
                  FadeIn(f1), run_time=1.1)
        relu_t = math(r"\rightarrow\ A^{(1)} = \text{ReLU}(Z^{(1)})", 30, ACCENT).next_to(f1, RIGHT, buff=0.2)
        dead_mark = label("unit 3: z < 0 → ReLU outputs 0", 19, ERROR).move_to([2.6, -1.75, 0])
        dead_ptr = Arrow(dead_mark.get_left(), hid[dead].get_right(), buff=0.12, color=ERROR, stroke_width=2.5,
                         max_tip_length_to_length_ratio=0.12)
        dead_mark = VGroup(dead_mark, dead_ptr)
        self.play(FadeIn(relu_t), hid[dead].animate.set_fill(GREY_E, 0.9).set_stroke(GREY_D), FadeIn(dead_mark),
                  run_time=0.9)
        alive_e2 = VGroup(*[e2[i] for i in range(4) if i != dead])
        f2 = math(r"\hat y = A^{(1)}W_2^\top + b_2", 30, PURPLE).next_to(out, UP, buff=0.35)
        self.play(ShowPassingFlash(alive_e2.copy().set_color(INFO).set_stroke(width=5), time_width=0.6),
                  FadeIn(f2), run_time=1.0)
        f3 = math(r"L = \tfrac{1}{N}\sum(\hat y - y)^2", 30, HIGHLIGHT).next_to(loss, UP, buff=0.35)
        self.play(ShowPassingFlash(e3.copy().set_color(INFO).set_stroke(width=5), time_width=0.6),
                  FadeIn(f3), run_time=0.8)
        self.wait(0.6)

        # ---------------- backward ----------------
        self.play(FadeOut(f1), FadeOut(relu_t), FadeOut(f2), FadeOut(f3),
                  Transform(phase, label("BACKWARD", 28, GRAD, weight=BOLD).move_to(phase)), run_time=0.6)
        b1 = math(r"\delta_2 = \tfrac{2}{N}(\hat y - y)", 30, GRAD).next_to(out, UP, buff=0.35)
        self.play(ShowPassingFlash(Line(loss.get_left(), out.get_right()).set_color(GRAD).set_stroke(width=6),
                                   time_width=0.8), FadeIn(b1), run_time=0.9)
        b2 = math(r"\nabla W_2 = \delta_2^\top A^{(1)}", 30, GRAD).next_to(b1, RIGHT, buff=0.35)
        back_e2 = VGroup(*[Line(out.get_left(), hid[i].get_right()) for i in range(4) if i != dead])
        self.play(ShowPassingFlash(back_e2.set_color(GRAD).set_stroke(width=6), time_width=0.7), FadeIn(b2),
                  run_time=1.0)
        b3 = math(r"\delta_1 = (\delta_2 W_2) \odot \text{ReLU}'(Z^{(1)})", 30, GRAD).next_to(hid, UP, buff=0.25)
        block = Cross(hid[dead], stroke_color=ERROR, stroke_width=5).scale(0.8)
        self.play(FadeIn(b3), Create(block), run_time=0.9)
        dead_note = label("ReLU′ = 0 → its gradient stops here", 19, ERROR).next_to(dead_mark[0], DOWN, buff=0.15)
        self.play(FadeIn(dead_note), run_time=0.5)
        b4 = math(r"\nabla W_1 = \delta_1^\top X", 30, GRAD).next_to(ins, UP, buff=0.4)
        back_e1 = VGroup(*[Line(hid[i].get_left(), a.get_right()) for i in range(4) if i != dead for a in ins])
        self.play(ShowPassingFlash(back_e1.set_color(GRAD).set_stroke(width=6), time_width=0.7), FadeIn(b4),
                  run_time=1.1)
        self.wait(0.5)

        # ---------------- update ----------------
        upd = math(r"W \leftarrow W - \eta\,\nabla_W L", 36, HIGHLIGHT).to_edge(DOWN, buff=0.5)
        live = VGroup(*[e1[k] for k in range(len(e1)) if k % 4 != dead], *alive_e2, e3)
        self.play(Write(upd), live.animate.set_color(HIGHLIGHT).set_stroke(width=3), run_time=1.0)
        self.play(live.animate.set_color(MUTED).set_stroke(width=2), run_time=0.6)
        self.wait(2.5)


# ----------------------------------------------------------------------------
# 2. Each ReLU unit adds one kink → piecewise-linear fit of x²
# ----------------------------------------------------------------------------
class ReluKinks(WikiScene):
    def construct(self):
        head = title("Each hidden ReLU adds one kink: piecewise-linear fit of y = x²")
        self.add(head)
        ax = Axes(x_range=[-2, 2, 1], y_range=[-0.5, 4.5, 1], x_length=7.5, y_length=4.6,
                  axis_config={"color": MUTED, "include_numbers": True, "font_size": 22},
                  tips=False).shift(LEFT * 1.9 + DOWN * 0.5)
        target = ax.plot(lambda x: x ** 2, color=FG, stroke_width=3).set_stroke(opacity=0.5)
        target_lbl = label("target  y = x²", 20, FG).next_to(ax, UP, buff=0.1).align_to(ax, LEFT)
        self.play(Create(ax), Create(target), FadeIn(target_lbl), run_time=1.2)

        xs = np.linspace(-2, 2, 401)
        ys = xs ** 2
        kink_order = [0.0, -1.0, 1.0, -1.5, 1.5, -0.5, 0.5]

        def fit(kinks):
            cols = [np.ones_like(xs), xs] + [np.maximum(0, xs - k) for k in kinks]
            A = np.stack(cols, 1)
            coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
            pred = A @ coef
            return pred, float(np.mean((pred - ys) ** 2))

        def curve(pred):
            pts = [ax.c2p(x, y) for x, y in zip(xs, pred)]
            return VMobject(color=HIGHLIGHT, stroke_width=5).set_points_as_corners(pts)

        pred, mse = fit([])
        fitted = curve(pred)
        info_x = 4.4
        n_lbl = label("hidden ReLUs: 0", 26, FG).move_to([info_x, 1.6, 0])
        m_lbl = label(f"MSE = {mse:.3f}", 26, HIGHLIGHT).move_to([info_x, 0.9, 0])
        note = label("no ReLU = straight line\n(can't bend)", 20, MUTED).move_to([info_x, -0.2, 0])
        self.play(Create(fitted), FadeIn(n_lbl), FadeIn(m_lbl), FadeIn(note), run_time=1.2)
        self.wait(0.8)
        self.play(FadeOut(note))

        dots = VGroup()
        for m in range(1, len(kink_order) + 1):
            kinks = kink_order[:m]
            pred, mse = fit(kinks)
            new = curve(pred)
            k = kinks[-1]
            dot = Dot(ax.c2p(k, float(np.interp(k, xs, pred))), color=GRAD, radius=0.07)
            moves = [d.animate.move_to(ax.c2p(kk, float(np.interp(kk, xs, pred))))
                     for d, kk in zip(dots, kinks[:-1])]
            dots.add(dot)
            self.play(
                Transform(fitted, new),
                FadeIn(dot, scale=1.6),
                *moves,
                Transform(n_lbl, label(f"hidden ReLUs: {m}", 26, FG).move_to(n_lbl)),
                Transform(m_lbl, label(f"MSE = {mse:.4f}", 26, HIGHLIGHT).move_to(m_lbl)),
                run_time=0.9,
            )
            self.wait(0.25)

        formula = math(r"\hat y = b + w x + \sum_j w_j\,\text{ReLU}(x - k_j)", 30, FG).move_to([info_x, -0.6, 0])
        dots_lbl = label("● = kink (one per unit)", 20, GRAD).next_to(formula, DOWN, buff=0.3)
        self.play(Write(formula), FadeIn(dots_lbl), run_time=1.2)
        self.play(FadeIn(caption_bar("More hidden units → more kinks → closer fit (best least-squares fit shown at each step)")))
        self.wait(2.5)


# ----------------------------------------------------------------------------
# 3. Gradient descent with three learning rates on L(w) = (w - 3)^2
# ----------------------------------------------------------------------------
class LearningRate(WikiScene):
    def construct(self):
        head = title("Same loss, three learning rates: w ← w − η · dL/dw")
        self.add(head)
        sub = math(r"L(w) = (w-3)^2,\quad \tfrac{dL}{dw} = 2(w-3),\quad w_0 = 0", 30, MUTED).next_to(head, DOWN, buff=0.2)
        self.play(FadeIn(sub))

        configs = [
            (0.05, "η = 0.05  too small", INFO, [-1, 7], [0, 16]),
            (0.4, "η = 0.4  good", SUCCESS, [-1, 7], [0, 16]),
            (1.05, "η = 1.05  too large", ERROR, [-6, 12], [0, 90]),
        ]
        steps = 10
        panels = []
        for idx, (eta, name, col, xr, yr) in enumerate(configs):
            ax = Axes(x_range=[xr[0], xr[1], 3], y_range=[yr[0], yr[1], yr[1] // 2], x_length=3.8, y_length=3.0,
                      axis_config={"color": MUTED, "include_numbers": False}, tips=False)
            ax.move_to([-4.6 + idx * 4.6, -0.3, 0])
            # draw only the part of the parabola inside the panel's y-range
            curve = ax.plot(lambda w: (w - 3) ** 2,
                            x_range=[max(xr[0], 3 - np.sqrt(yr[1])), min(xr[1], 3 + np.sqrt(yr[1]))],
                            color=FG, stroke_width=2).set_stroke(opacity=0.6)
            name_lbl = label(name, 22, col, weight=BOLD).next_to(ax, UP, buff=0.25)
            ws = [0.0]
            for _ in range(steps):
                w = ws[-1]
                ws.append(w - eta * 2 * (w - 3))
            dot = Dot(ax.c2p(ws[0], (ws[0] - 3) ** 2), color=col, radius=0.09)
            loss_lbl = label(f"step 0 · L = {(ws[0] - 3) ** 2:.2f}", 20, FG, mono=True).next_to(ax, DOWN, buff=0.3)
            panels.append(dict(ax=ax, ws=ws, dot=dot, col=col, loss=loss_lbl))
            self.play(Create(ax), Create(curve), FadeIn(name_lbl), FadeIn(dot), FadeIn(loss_lbl), run_time=0.7)

        for t in range(1, steps + 1):
            anims = []
            for p in panels:
                ax, ws, col = p["ax"], p["ws"], p["col"]
                a = ax.c2p(ws[t - 1], (ws[t - 1] - 3) ** 2)
                b = ax.c2p(ws[t], (ws[t] - 3) ** 2)
                seg = Line(a, b, color=col, stroke_width=2.5).set_opacity(0.8)
                self.add(seg)
                anims.append(p["dot"].animate.move_to(b))
                anims.append(Transform(p["loss"], label(f"step {t} · L = {(ws[t] - 3) ** 2:.2f}", 20, FG, mono=True)
                                       .move_to(p["loss"])))
            self.play(*anims, run_time=0.45 if t < 4 else 0.3)

        verdicts = [
            ("still far from w = 3", INFO),
            ("converged in ~4 steps", SUCCESS),
            ("overshoots more each step → diverges", ERROR),
        ]
        for p, (txt, col) in zip(panels, verdicts):
            self.play(FadeIn(label(txt, 19, col).next_to(p["loss"], DOWN, buff=0.15)), run_time=0.4)
        self.play(FadeIn(caption_bar("Each step multiplies the distance to the minimum by (1 − 2η): 0.9 · 0.2 · −1.1")))
        self.wait(3)
