"""Animations for the CNN pages (03-cnn-fundamentals, 04-cnn-architecture-training-alexnet)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from style import *


def cell_center(grid, i, j):
    return grid[i][j].get_center()


# ----------------------------------------------------------------------------
# 1. Convolution: a vertical-edge kernel slides over an image with an edge
# ----------------------------------------------------------------------------
class ConvolutionSlide(WikiScene):
    def construct(self):
        img = np.array([[10, 10, 10, 0, 0, 0]] * 5)
        k = np.array([[1, 0, -1]] * 3)
        H, W = img.shape
        cell = 0.55

        head = title("Convolution: one 3×3 kernel slides over the image")
        inp = grid_of_numbers(img.tolist(), cell=cell, size=20)
        for i in range(H):
            for j in range(W):
                if img[i, j] > 0:
                    inp[i][j][0].set_fill(FG, opacity=0.18)
        inp.move_to(LEFT * 3.9 + UP * 0.8)
        inp_lbl = label("input 5×6  (bright | dark)", 22, MUTED).next_to(inp, UP, buff=0.2)

        ker = grid_of_numbers(k.tolist(), cell=cell, size=20, stroke=HIGHLIGHT)
        ker.move_to(DOWN * 1.7 + LEFT * 3.9)
        ker_lbl = label("kernel: vertical-edge detector", 20, HIGHLIGHT).next_to(ker, DOWN, buff=0.15)

        oh, ow = H - 2, W - 2
        out = grid_of_numbers([[None] * ow for _ in range(oh)], cell=cell, size=20)
        out.move_to(RIGHT * 3.9 + UP * 0.8)
        out_lbl = label("feature map 3×4", 22, MUTED).next_to(out, UP, buff=0.2)

        self.play(FadeIn(head), FadeIn(inp), FadeIn(inp_lbl), FadeIn(ker), FadeIn(ker_lbl),
                  FadeIn(out), FadeIn(out_lbl), run_time=1.2)

        win = Square(cell * 3, stroke_color=HIGHLIGHT, stroke_width=5)
        win.move_to(cell_center(inp, 1, 1))
        calc = label("", 24).move_to(RIGHT * 2.6 + DOWN * 1.4)
        self.play(Create(win), run_time=0.5)

        for i in range(oh):
            for j in range(ow):
                self.play(win.animate.move_to(cell_center(inp, i + 1, j + 1)),
                          run_time=0.55 if (i, j) in [(0, 0), (0, 1)] else 0.3)
                patch = img[i:i + 3, j:j + 3]
                left, right = int(patch[:, 0].sum()), int(patch[:, 2].sum())
                val = int((patch * k).sum())
                new_calc = label(f"left − right = {left} − {right} = {val}", 26,
                                 HIGHLIGHT if val else FG).move_to(calc)
                slot = out[i][j]
                num = label(str(val), 20, BG if val else FG, mono=True).move_to(slot)
                self.play(
                    Transform(calc, new_calc),
                    slot[0].animate.set_fill(HIGHLIGHT if val else MUTED, opacity=0.9 if val else 0.15),
                    FadeIn(num),
                    run_time=0.8 if (i, j) in [(0, 0), (0, 1)] else 0.3,
                )
                if (i, j) == (0, 1):
                    self.wait(0.6)

        self.play(FadeOut(win), FadeOut(calc))
        self.play(FadeIn(caption_bar("Output fires (30) only where the window straddles the edge · size = (5−3+1) × (6−3+1)")))
        self.wait(2)


# ----------------------------------------------------------------------------
# 2. Stride & padding: three configurations side by side
# ----------------------------------------------------------------------------
class StridePadding(WikiScene):
    def build_panel(self, n, pad, cell):
        size = n + 2 * pad
        g = VGroup()
        for i in range(size):
            for j in range(size):
                is_pad = pad and (i < pad or j < pad or i >= size - pad or j >= size - pad)
                sq = Square(cell, stroke_width=1.2,
                            stroke_color=MUTED if not is_pad else GREY_D)
                if is_pad:
                    sq.set_stroke(opacity=0.8).set_fill(GREY_E, opacity=0.6)
                    g.add(VGroup(sq, label("0", 14, MUTED, mono=True)).move_to([j * cell, -i * cell, 0]))
                else:
                    sq.set_fill(INFO, opacity=0.12)
                    g.add(VGroup(sq).move_to([j * cell, -i * cell, 0]))
        return g, size

    def construct(self):
        head = title("Stride & padding decide how many positions the kernel visits")
        self.add(head)
        cell = 0.36
        configs = [
            ("S = 1, P = 0", 1, 0),
            ("S = 2, P = 0", 2, 0),
            ("S = 1, P = 1  (\"same\")", 1, 1),
        ]
        xs = [-4.6, 0, 4.6]
        n, F = 5, 3
        for (name, S, P), x in zip(configs, xs):
            g, size = self.build_panel(n, P, cell)
            g.move_to([x, 0, 0]).align_to(np.array([0, 2.3, 0]), UP)
            name_lbl = label(name, 24, FG).move_to([x, 2.7, 0])
            self.play(FadeIn(g), FadeIn(name_lbl), run_time=0.6)

            out_n = (n - F + 2 * P) // S + 1
            oc = 0.3
            out = VGroup(*[Square(oc, stroke_width=1.2, stroke_color=MUTED).move_to([j * oc, -i * oc, 0])
                           for i in range(out_n) for j in range(out_n)])
            out.move_to([x, -1.05, 0]).align_to(np.array([0, -0.45, 0]), UP)
            arrow = Arrow(g.get_bottom() + DOWN * 0.05, out.get_top() + UP * 0.05, buff=0.1,
                          color=MUTED, stroke_width=3, max_tip_length_to_length_ratio=0.15)
            self.play(FadeIn(out), GrowArrow(arrow), run_time=0.4)

            win = Square(cell * F, stroke_color=HIGHLIGHT, stroke_width=4)

            def pos(r, c):
                top_left = g[0].get_center()
                return top_left + np.array([(c * S + 1) * cell, -(r * S + 1) * cell, 0])

            win.move_to(pos(0, 0))
            self.play(Create(win), run_time=0.3)
            for r in range(out_n):
                for c in range(out_n):
                    self.play(win.animate.move_to(pos(r, c)), run_time=0.18)
                    self.play(out[r * out_n + c].animate.set_fill(HIGHLIGHT, opacity=0.75), run_time=0.12)
            self.play(FadeOut(win), run_time=0.2)
            formula = math(rf"\lfloor ({n} - {F} + 2\cdot{P}) / {S} \rfloor + 1 = {out_n}",
                           size=34, color=HIGHLIGHT).move_to([x, -2.75, 0])
            self.play(Write(formula), run_time=0.6)

        self.play(FadeIn(caption_bar("Bigger stride → fewer positions (downsampling) · padding P = (F−1)/2 keeps the size")))
        self.wait(2.5)


# ----------------------------------------------------------------------------
# 3. One filter spans ALL input channels → ONE feature map
# ----------------------------------------------------------------------------
class MultiChannelFilter(WikiScene):
    def construct(self):
        rng = np.random.default_rng(3)
        head = title("One filter spans all input channels → one feature map")
        self.add(head)
        cell = 0.42
        colors = [ERROR, SUCCESS, INFO]
        names = ["R", "G", "B"]
        chans = [rng.integers(0, 3, (4, 4)) for _ in range(3)]
        kers = [rng.integers(-1, 2, (3, 3)) for _ in range(3)]
        bias = 1

        stack = VGroup()
        for c in range(3):
            g = grid_of_numbers(chans[c].tolist(), cell=cell, size=16, stroke=colors[c],
                                fill=colors[c], fill_opacity=0.08)
            g.move_to(LEFT * 4.6 + UP * (1.6 - c * 1.95))
            tag = label(names[c], 22, colors[c], weight=BOLD).next_to(g, LEFT, buff=0.2)
            stack.add(VGroup(g, tag))
        in_lbl = label("input (3, 4, 4)", 22, MUTED).next_to(stack, DOWN, buff=0.25)

        kstack = VGroup()
        for c in range(3):
            g = grid_of_numbers(kers[c].tolist(), cell=cell, size=16, stroke=colors[c],
                                fmt="{:+d}")
            g.move_to(LEFT * 0.9 + UP * (1.6 - c * 1.95))
            kstack.add(g)
        k_lbl = label("ONE filter (3, 3, 3)", 22, HIGHLIGHT).next_to(kstack, DOWN, buff=0.25)

        out = grid_of_numbers([[None] * 2 for _ in range(2)], cell=0.6, size=20)
        out.move_to(RIGHT * 4.8 + UP * 0.3)
        out_lbl = label("feature map (1, 2, 2)", 22, MUTED).next_to(out, UP, buff=0.25)

        self.play(FadeIn(stack), FadeIn(in_lbl), FadeIn(kstack), FadeIn(k_lbl),
                  FadeIn(out), FadeIn(out_lbl), run_time=1.2)

        partial_area = RIGHT * 2.0 + DOWN * 1.6
        for (r, cc) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            wins = VGroup()
            partials = []
            for c in range(3):
                g = stack[c][0]
                center = g[r + 1][cc + 1].get_center()
                wins.add(Square(cell * 3, stroke_color=HIGHLIGHT, stroke_width=4).move_to(center))
                partials.append(int((chans[c][r:r + 3, cc:cc + 3] * kers[c]).sum()))
            total = sum(partials) + bias
            first = (r, cc) == (0, 0)
            self.play(*[Create(w) for w in wins], run_time=0.5 if first else 0.25)
            txt = label(f"R: {partials[0]:+d}   G: {partials[1]:+d}   B: {partials[2]:+d}   bias: +{bias}",
                        22, FG).move_to(partial_area)
            res = label(f"Σ = {total}", 28, HIGHLIGHT, weight=BOLD).next_to(txt, DOWN, buff=0.25)
            self.play(FadeIn(txt), run_time=0.6 if first else 0.3)
            self.play(FadeIn(res), run_time=0.5 if first else 0.25)
            slot = out[r][cc]
            num = label(str(total), 20, BG, mono=True).move_to(slot)
            self.play(TransformFromCopy(res, num), slot[0].animate.set_fill(HIGHLIGHT, opacity=0.9),
                      run_time=0.6 if first else 0.35)
            if first:
                self.wait(0.8)
            self.play(FadeOut(wins), FadeOut(txt), FadeOut(res), run_time=0.25)

        more = label("K filters → K feature maps\n→ output depth K", 24, ACCENT).move_to(RIGHT * 4.0 + DOWN * 1.3)
        params = math(r"\text{params} = K(C_{in}F^2 + 1) = 28", size=32).next_to(more, DOWN, buff=0.35)
        self.play(FadeIn(more), Write(params))
        self.wait(2.5)


# ----------------------------------------------------------------------------
# 4. Max pooling + translation tolerance
# ----------------------------------------------------------------------------
class MaxPooling(WikiScene):
    def construct(self):
        head = title("Max pooling 2×2, stride 2: keep the strongest response")
        self.add(head)
        fm = np.array([[1, 5, 0, 2], [2, 3, 1, 4], [6, 0, 3, 1], [1, 2, 8, 2]])
        cell = 0.7
        g = grid_of_numbers(fm.tolist(), cell=cell, size=24)
        g.move_to(LEFT * 3.3 + DOWN * 0.2)
        out = grid_of_numbers([[None, None], [None, None]], cell=cell, size=24)
        out.move_to(RIGHT * 3.3 + DOWN * 0.2)
        self.play(FadeIn(g), FadeIn(label("feature map (4×4)", 22, MUTED).next_to(g, UP, buff=0.25)),
                  FadeIn(out), FadeIn(label("pooled (2×2)", 22, MUTED).next_to(out, UP, buff=0.25)))
        block_colors = [INFO, PURPLE, ACCENT, GRAD]
        for b, (r, c) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            col = block_colors[b]
            cells = [g[2 * r + di][2 * c + dj] for di in range(2) for dj in range(2)]
            box = SurroundingRectangle(VGroup(*cells), buff=0.02, color=col, stroke_width=4)
            vals = fm[2 * r:2 * r + 2, 2 * c:2 * c + 2]
            mi, mj = np.unravel_index(vals.argmax(), vals.shape)
            winner = g[2 * r + mi][2 * c + mj]
            self.play(Create(box), *[cl[0].animate.set_fill(col, opacity=0.18) for cl in cells], run_time=0.45)
            self.play(winner[0].animate.set_fill(col, opacity=0.85), Indicate(winner[1], color=BG), run_time=0.45)
            num = label(str(vals.max()), 24, BG, mono=True).move_to(out[r][c])
            self.play(TransformFromCopy(winner[1], num), out[r][c][0].animate.set_fill(col, opacity=0.85),
                      run_time=0.55)
        info = label("channels unchanged · H, W halved · 0 parameters", 24, FG).to_edge(DOWN, buff=1.0)
        self.play(FadeIn(info))
        self.wait(1)

        # translation tolerance: move the 8 one pixel left, inside the same window
        note = label("move the 8 one pixel right (still the same window) → output unchanged", 24, HIGHLIGHT)
        note.to_edge(DOWN, buff=0.45)
        self.play(FadeIn(note))
        src = g[3][2][1]
        dst = g[3][3][1]
        new8 = label("8", 24, mono=True).move_to(dst)
        new2 = label("2", 24, mono=True).move_to(src)
        self.play(Transform(src, new2), Transform(dst, new8),
                  g[3][2][0].animate.set_fill(GRAD, opacity=0.18),
                  g[3][3][0].animate.set_fill(GRAD, opacity=0.85), run_time=0.8)
        self.play(Indicate(out[1][1], color=HIGHLIGHT, scale_factor=1.15), run_time=0.9)
        self.wait(2)


# ----------------------------------------------------------------------------
# 5. SmallCNN shape funnel with parameter budget
# ----------------------------------------------------------------------------
class ShapeFunnel(WikiScene):
    def construct(self):
        head = title("SmallCNN: space shrinks, depth grows, then flatten → logits")
        self.add(head)

        def block(ch, hw, color, w=None, h=None):
            # height ∝ spatial size, width ∝ √channels (keeps 3 vs 32 channels readable)
            h = h if h is not None else 0.7 + 2.9 * hw / 32
            w = w if w is not None else 0.25 + 0.17 * np.sqrt(ch)
            return Rectangle(width=w, height=h, stroke_color=color, stroke_width=2).set_fill(color, opacity=0.25)

        # (shape label, mobject, op on the incoming arrow, params text, params colour)
        stages = [
            ("(3,32,32)",  block(3, 32, INFO),     None,               None,       None),
            ("(16,32,32)", block(16, 32, ACCENT),  "conv1\n+ReLU",   "448",      HIGHLIGHT),
            ("(16,16,16)", block(16, 16, ACCENT),  "pool",             "0",        MUTED),
            ("(32,16,16)", block(32, 16, PURPLE),  "conv2\n+ReLU",   "4,640",    HIGHLIGHT),
            ("(32,8,8)",   block(32, 8, PURPLE),   "pool",             "0",        MUTED),
            ("2048",       block(0, 0, HIGHLIGHT, w=0.2, h=3.4), "flatten", "0",  MUTED),
            ("64",         block(0, 0, ERROR, w=0.2, h=1.3),     "fc1",     "131,136", ERROR),
            ("10 logits",  block(0, 0, SUCCESS, w=0.2, h=0.55),  "fc2",     "650",  HIGHLIGHT),
        ]
        gap = 0.95
        total = sum(st[1].width for st in stages) + gap * (len(stages) - 1)
        x = -total / 2
        y = 0.55
        prev = None
        for shape, rect, op, params, pcol in stages:
            rect.move_to([x + rect.width / 2, y, 0])
            s_lbl = label(shape, 17, FG, mono=True).next_to(rect, DOWN, buff=0.18)
            anims = [FadeIn(rect, shift=RIGHT * 0.2), FadeIn(s_lbl)]
            if prev is not None:
                big = op == "fc1"
                arr = Arrow(prev.get_right(), rect.get_left(), buff=0.06,
                            color=ERROR if big else MUTED, stroke_width=5 if big else 3,
                            max_tip_length_to_length_ratio=0.18)
                arr.move_to([arr.get_center()[0], y, 0])
                op_lbl = label(op, 16, ERROR if big else MUTED).next_to(arr, UP, buff=0.12)
                anims += [GrowArrow(arr), FadeIn(op_lbl)]
            if params is not None:
                p_lbl = label(f"{params} params", 15, pcol, weight=BOLD if pcol == ERROR else NORMAL)
                p_lbl.next_to(s_lbl, DOWN, buff=0.1)
                anims.append(FadeIn(p_lbl))
            self.play(*anims, run_time=0.75 if op in ("flatten", "fc1") else 0.6)
            if op == "fc1":
                self.play(Indicate(rect, color=ERROR, scale_factor=1.3), run_time=0.8)
            prev = rect
            x = rect.get_right()[0] + gap

        self.play(FadeIn(caption_bar("Conv layers ≈ 5 K params · fc1 alone = 131 K of 137 K (96%) → that's where dropout goes",
                                     size=24, color=FG)))
        self.wait(3)


# ----------------------------------------------------------------------------
# 6. Dropout: random masks in train mode, identity in eval mode
# ----------------------------------------------------------------------------
class DropoutTrainEval(WikiScene):
    def construct(self):
        head = title("Dropout (p = 0.5): random masks in training, identity at test")
        self.add(head)
        n_in, n_out = 8, 3
        ys_in = np.linspace(2.3, -2.3, n_in)
        ys_out = np.linspace(1.2, -1.2, n_out)
        x_in, x_out = -1.2, 2.6
        neurons = VGroup(*[Circle(0.26, color=INFO, stroke_width=3).set_fill(INFO, 0.35).move_to([x_in, y, 0])
                           for y in ys_in])
        outs = VGroup(*[Circle(0.26, color=ACCENT, stroke_width=3).set_fill(ACCENT, 0.3).move_to([x_out, y, 0])
                        for y in ys_out])
        edges = VGroup(*[Line(a.get_right(), b.get_left(), stroke_width=1.5, color=MUTED, stroke_opacity=0.6)
                         for a in neurons for b in outs])
        vals = VGroup(*[label("1.0", 18, FG, mono=True).next_to(nn_, LEFT, buff=0.2) for nn_ in neurons])
        in_lbl = label("activations h", 20, MUTED).next_to(neurons, UP, buff=0.2)
        self.play(FadeIn(neurons), FadeIn(outs), Create(edges), FadeIn(vals), FadeIn(in_lbl), run_time=1)

        mode = label("model.train()", 30, HIGHLIGHT, mono=True).move_to(LEFT * 4.8 + UP * 1.5)
        rule = label("zero with prob. p,\nsurvivors × 1/(1−p) = ×2", 22, FG).next_to(mode, DOWN, buff=0.35)
        self.play(FadeIn(mode), FadeIn(rule))

        rng = np.random.default_rng(7)
        masks = [rng.random(n_in) > 0.5 for _ in range(3)]
        for step, keep in enumerate(masks):
            anims = []
            for idx in range(n_in):
                kept = keep[idx]
                anims.append(neurons[idx].animate.set_fill(INFO if kept else GREY_E, 0.35 if kept else 0.9)
                             .set_stroke(INFO if kept else GREY_D))
                anims.append(Transform(vals[idx], label("2.0" if kept else "0.0", 18,
                                                        SUCCESS if kept else ERROR, mono=True)
                                       .next_to(neurons[idx], LEFT, buff=0.2)))
                for o in range(n_out):
                    anims.append(edges[idx * n_out + o].animate.set_stroke(opacity=0.6 if kept else 0.05))
            step_lbl = label(f"forward pass {step + 1}: new random mask", 20, MUTED).move_to(DOWN * 3.0)
            self.play(*anims, FadeIn(step_lbl), run_time=0.8)
            self.wait(0.6)
            self.play(FadeOut(step_lbl), run_time=0.2)

        mode2 = label("model.eval()", 30, SUCCESS, mono=True).move_to(mode)
        rule2 = label("every unit kept, no scaling:\ndeterministic output", 22, FG).next_to(mode2, DOWN, buff=0.35)
        anims = [Transform(mode, mode2), Transform(rule, rule2)]
        for idx in range(n_in):
            anims.append(neurons[idx].animate.set_fill(INFO, 0.35).set_stroke(INFO))
            anims.append(Transform(vals[idx], label("1.0", 18, FG, mono=True).next_to(neurons[idx], LEFT, buff=0.2)))
        anims.append(edges.animate.set_stroke(opacity=0.6))
        self.play(*anims, run_time=1)
        self.play(FadeIn(caption_bar("Same expected value in both modes: 0.5 × 2.0 = 1.0, so nothing to rescale at test time")))
        self.wait(2.5)
