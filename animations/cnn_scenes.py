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


# ============================================================================
# CNN in Practice: Cats vs Dogs (05-cnn-practical-cats-vs-dogs)
# ============================================================================
CAT_C, DOG_C = INFO, PURPLE
SCALE = 12.0 / 25000          # scene units per image, shared by every bar


def class_bar(n_cat, n_dog, height=0.55):
    """A horizontal bar split into a cat part and a dog part, widths proportional to counts."""
    cat = Rectangle(width=n_cat * SCALE, height=height, stroke_width=0).set_fill(CAT_C, 0.85)
    dog = Rectangle(width=n_dog * SCALE, height=height, stroke_width=0).set_fill(DOG_C, 0.85)
    dog.next_to(cat, RIGHT, buff=0)
    return VGroup(cat, dog)


class StratifiedSplit(WikiScene):
    """25,000 files → balanced 12,000 → stratified 8,400 / 1,800 / 1,800 (numbers from sklearn)."""

    def construct(self):
        head = title("Balanced sample, then two stratified splits")
        self.add(head)
        legend = VGroup(
            Square(0.25, stroke_width=0).set_fill(CAT_C, 0.85), label("cat (0)", 20, FG),
            Square(0.25, stroke_width=0).set_fill(DOG_C, 0.85), label("dog (1)", 20, FG),
        ).arrange(RIGHT, buff=0.15)
        legend[2].shift(RIGHT * 0.3); legend[3].shift(RIGHT * 0.3)
        legend.next_to(head, DOWN, buff=0.25)
        self.add(legend)

        # Row 1: everything
        all_bar = class_bar(12500, 12500).move_to(UP * 1.35)
        all_lbl = label("all files: 25,000  =  12,500 cats + 12,500 dogs", 22, FG).next_to(all_bar, DOWN, buff=0.15)
        self.play(FadeIn(all_bar), FadeIn(all_lbl), run_time=1)
        self.wait(0.6)

        # Row 2: balanced 6,000 + 6,000
        bal_bar = class_bar(6000, 6000).move_to(UP * 0.0)
        bal_lbl = label("df.sample(6,000) per class  →  12,000 (50 / 50)", 22, FG).next_to(bal_bar, DOWN, buff=0.15)
        self.play(TransformFromCopy(all_bar, bal_bar), FadeIn(bal_lbl), run_time=1.2)
        self.wait(0.6)

        # Row 3: 70 / 15 / 15, each stratified
        tr = class_bar(4200, 4200); va = class_bar(900, 900); te = class_bar(900, 900)
        row = VGroup(tr, va, te).arrange(RIGHT, buff=1.5).move_to(DOWN * 1.55)
        names = [("train 70%", "8,400", "4,200 | 4,200"), ("val 15%", "1,800", "900 | 900"),
                 ("test 15%", "1,800", "900 | 900")]
        lbls = VGroup()
        for bar, (nm, total, split) in zip(row, names):
            lbls.add(VGroup(label(nm, 20, FG), label(total, 20, FG, mono=True), label(split, 18, MUTED, mono=True))
                     .arrange(DOWN, buff=0.06).next_to(bar, DOWN, buff=0.15))
        step1 = label("train_test_split(test_size=0.3, stratify=label)", 19, HIGHLIGHT, mono=True)
        step1.next_to(bal_lbl, DOWN, buff=0.12)
        self.play(FadeIn(step1), run_time=0.5)
        self.play(TransformFromCopy(bal_bar, tr), run_time=1)
        self.play(FadeIn(lbls[0]), run_time=0.4)
        step2 = label("then split the 30% in half, stratified again", 19, HIGHLIGHT, mono=True).move_to(step1)
        self.play(Transform(step1, step2), run_time=0.5)
        self.play(TransformFromCopy(bal_bar, VGroup(va, te)), run_time=1)
        self.play(FadeIn(lbls[1]), FadeIn(lbls[2]), run_time=0.5)
        self.wait(0.5)
        self.play(FadeIn(caption_bar("Every subset keeps exactly 50% cats: no split is starved of a class")))
        self.wait(2.5)


class PixelToTensor(WikiScene):
    """One stored pixel to model input: BGR→RGB, ÷255, then HWC → CHW regrouping."""

    def construct(self):
        head = title("From a stored pixel to a model-ready tensor")
        self.add(head)

        def cells(values, colors, size=24, w=1.5):
            g = VGroup()
            for v, c in zip(values, colors):
                box = Rectangle(width=w, height=0.75, stroke_color=c, stroke_width=3).set_fill(c, 0.18)
                g.add(VGroup(box, label(v, size, FG, mono=True).move_to(box)))
            return g.arrange(RIGHT, buff=0.12)

        R, G, B = ERROR, SUCCESS, INFO
        # Stage 1: OpenCV gives B, G, R
        bgr = cells(["87", "164", "203"], [B, G, R]).move_to(UP * 1.5 + LEFT * 2.6)
        tags = VGroup(*[label(t, 20, c).next_to(bgr[i], UP, buff=0.12) for i, (t, c) in enumerate([("B", B), ("G", G), ("R", R)])])
        note = label("cv2.imread  →  uint8, BGR order", 22, MUTED).next_to(bgr, RIGHT, buff=0.5)
        self.play(FadeIn(bgr), FadeIn(tags), FadeIn(note), run_time=0.8)
        self.wait(0.6)

        # Stage 2: swap to R, G, B
        note2 = label("cvtColor(BGR2RGB)  →  203, 164, 87", 22, HIGHLIGHT).move_to(note, aligned_edge=LEFT)
        p0, p2 = bgr[0].get_center(), bgr[2].get_center()
        t0, t2 = tags[0].get_center(), tags[2].get_center()
        self.play(bgr[0].animate.move_to(p2), bgr[2].animate.move_to(p0),
                  tags[0].animate.move_to(t2), tags[2].animate.move_to(t0),
                  Transform(note, note2), run_time=1.2)
        self.wait(0.5)

        # Stage 3: divide by 255 (values from numpy: 0.796, 0.643, 0.341)
        scaled = cells(["0.796", "0.643", "0.341"], [R, G, B], size=22).move_to(DOWN * 0.1 + LEFT * 2.6)
        note3 = label("÷ 255  →  float in [0, 1]", 22, HIGHLIGHT).next_to(scaled, RIGHT, buff=0.5)
        self.play(FadeIn(scaled, shift=DOWN * 0.3), FadeIn(note3), run_time=1)
        self.wait(0.6)

        # Stage 4: HWC → CHW for a 2×2 image (symbolic values)
        self.play(*[FadeOut(m) for m in (bgr, tags, note, scaled, note3)], run_time=0.5)
        hwc_title = label("(H, W, C): channels interleaved, pixel by pixel", 22, FG).move_to(UP * 1.75)
        names = []
        for p in range(1, 5):
            names += [(f"R{p}", R), (f"G{p}", G), (f"B{p}", B)]
        strip = VGroup(*[VGroup(Rectangle(width=0.82, height=0.6, stroke_color=c, stroke_width=2.5).set_fill(c, 0.18),
                                label(n, 18, FG, mono=True)) for n, c in names])
        for item in strip:
            item[1].move_to(item[0])
        strip.arrange(RIGHT, buff=0.06).move_to(UP * 0.95)
        self.play(FadeIn(hwc_title), FadeIn(strip), run_time=0.8)
        self.wait(0.4)
        ghost = strip.copy().set_opacity(0.35)          # keep the "before" layout visible
        self.add(ghost)

        chw_title = label("np.transpose(img, (2, 0, 1))  →  (C, H, W): one plane per colour", 22, HIGHLIGHT).move_to(DOWN * 0.15)
        targets = []
        plane_centers = [LEFT * 4 + DOWN * 1.65, DOWN * 1.65, RIGHT * 4 + DOWN * 1.65]
        for idx in range(12):
            ch, p = idx % 3, idx // 3
            r, c = p // 2, p % 2
            targets.append(plane_centers[ch] + RIGHT * (c - 0.5) * 0.9 + DOWN * (r - 0.5) * 0.68)
        plane_lbls = VGroup(*[label(t, 20, col).move_to(pc + UP * 1.0) for t, col, pc in
                              zip(["R plane", "G plane", "B plane"], [R, G, B], plane_centers)])
        self.play(FadeIn(chw_title), run_time=0.5)
        self.play(*[strip[i].animate.move_to(targets[i]) for i in range(12)], FadeIn(plane_lbls), run_time=1.6)
        self.play(FadeIn(caption_bar("Conv2d reads whole colour planes: (3, H, W) per image, (B, 3, H, W) per batch")))
        self.wait(2.5)


class ConfusionMetrics(WikiScene):
    """Example 1,800-image test set: TN 780, FP 120, FN 90, TP 810 (metrics from sklearn)."""

    def construct(self):
        head = title("Confusion matrix → precision, recall, F1   (positive class = dog)")
        self.add(head)
        cw, chh = 2.0, 1.1
        origin = LEFT * 3.6 + UP * 0.35
        vals = [[("TN", 780, SUCCESS), ("FP", 120, ERROR)], [("FN", 90, ERROR), ("TP", 810, SUCCESS)]]
        cells = [[None, None], [None, None]]
        grid = VGroup()
        for i in range(2):
            for j in range(2):
                box = Rectangle(width=cw, height=chh, stroke_color=MUTED, stroke_width=2)
                box.move_to(origin + RIGHT * j * cw + DOWN * i * chh)
                cells[i][j] = box
                grid.add(box)
        col_h = VGroup(label("pred cat", 20, MUTED).next_to(cells[0][0], UP, buff=0.12),
                       label("pred dog", 20, MUTED).next_to(cells[0][1], UP, buff=0.12))
        row_h = VGroup(label("actual cat", 20, MUTED).next_to(cells[0][0], LEFT, buff=0.15),
                       label("actual dog", 20, MUTED).next_to(cells[1][0], LEFT, buff=0.15))
        self.play(Create(grid), FadeIn(col_h), FadeIn(row_h), run_time=0.8)

        for i, j in [(0, 0), (1, 1), (0, 1), (1, 0)]:
            tag, v, col = vals[i][j]
            txt = VGroup(label(tag, 18, col), label(str(v), 30, FG, mono=True)).arrange(DOWN, buff=0.05).move_to(cells[i][j])
            self.play(cells[i][j].animate.set_fill(col, 0.15), FadeIn(txt), run_time=0.45)
        self.wait(0.4)

        rx = RIGHT * 3.2
        acc = label("accuracy = (780 + 810) / 1800 = 0.883", 22, FG).move_to(rx + UP * 1.6)
        self.play(FadeIn(acc), run_time=0.6)

        col_box = SurroundingRectangle(VGroup(cells[0][1], cells[1][1]), color=HIGHLIGHT, buff=0.05, stroke_width=4)
        prec = label("precision = TP / (TP + FP) = 810 / 930 = 0.871", 22, HIGHLIGHT).move_to(rx + UP * 0.7)
        prec_note = label("of everything called dog, how much was dog?", 18, MUTED).next_to(prec, DOWN, buff=0.08)
        self.play(Create(col_box), FadeIn(prec), FadeIn(prec_note), run_time=0.9)
        self.wait(0.8)

        row_box = SurroundingRectangle(VGroup(cells[1][0], cells[1][1]), color=INFO, buff=0.05, stroke_width=4)
        rec = label("recall = TP / (TP + FN) = 810 / 900 = 0.900", 22, INFO).move_to(rx + DOWN * 0.45)
        rec_note = label("of all real dogs, how many were found?", 18, MUTED).next_to(rec, DOWN, buff=0.08)
        self.play(ReplacementTransform(col_box, row_box), FadeIn(rec), FadeIn(rec_note), run_time=0.9)
        self.wait(0.8)

        f1 = label("F1 = 2PR / (P + R) = 0.885", 26, SUCCESS, weight=BOLD).move_to(rx + DOWN * 1.6)
        self.play(FadeOut(row_box), FadeIn(f1, scale=1.1), run_time=0.8)
        self.play(FadeIn(caption_bar("Precision is hurt by false alarms (FP), recall by misses (FN)")))
        self.wait(2.5)
