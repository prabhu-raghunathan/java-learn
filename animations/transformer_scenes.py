"""Animations for the Figure 1 Transformer page (keys transformer/...).
PositionalEncodingWaves computes the paper's sinusoids directly. CausalMask and AutoregressiveDecoding load
data/toy_transformer_decode.npz, exported from the page's toy Transformer (trained to reverse 6 digits,
torch.manual_seed(0), 3000 steps): real attention weights, predictions and probabilities."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from style import *

HERE = os.path.dirname(os.path.abspath(__file__))


def pe(pos, d):
    out = np.zeros(d)
    i2 = np.arange(0, d, 2)
    ang = pos / 10000 ** (i2 / d)
    out[0::2], out[1::2] = np.sin(ang), np.cos(ang)
    return out


class PositionalEncodingWaves(WikiScene):
    def construct(self):
        self.add(title("Positional Encoding: each dimension is a sine wave of a different frequency"))
        d = 8
        ax = Axes(x_range=[0, 20, 5], y_range=[-1.2, 1.2, 1], x_length=8.2, y_length=3.6, tips=False,
                  axis_config={"color": MUTED, "stroke_width": 2}).to_edge(LEFT, buff=0.6).shift(DOWN * 0.35)
        xl = label("pos", 20, MUTED).next_to(ax.x_axis, RIGHT, buff=0.15)
        self.play(Create(ax), FadeIn(xl), run_time=0.7)
        cols = [INFO, HIGHLIGHT, SUCCESS, GRAD]
        curves, names = VGroup(), VGroup()
        for k, i2 in enumerate(range(0, d, 2)):
            w = 1 / 10000 ** (i2 / d)
            c = ax.plot(lambda p, w=w: np.sin(w * p), x_range=[0, 20, 0.05], color=cols[k], stroke_width=3)
            curves.add(c)
            names.add(label(f"dim {i2}: sin(pos / 10000^({i2}/{d}))", 17, cols[k]))
        names.arrange(DOWN, aligned_edge=LEFT, buff=0.12).to_corner(UR, buff=0.5).shift(DOWN * 0.8)
        for c, n in zip(curves, names):
            self.play(Create(c), FadeIn(n), run_time=0.7)
        cap = caption_bar("wavelengths grow geometrically: fast dims tell nearby positions apart, slow dims track coarse position")
        self.play(FadeIn(cap), run_time=0.4)
        self.t = ValueTracker(0)
        line = always_redraw(lambda: ax.get_vertical_line(ax.c2p(self.t.get_value(), 1.2), color=FG, stroke_width=2))
        readout = always_redraw(lambda: label(
            "PE(" + f"{self.t.get_value():.0f}" + ") = [" + ", ".join(f"{v:.2f}" for v in pe(round(self.t.get_value()), d)) + "]",
            20, FG, mono=True).to_corner(DR, buff=0.5).shift(UP * 0.9))
        self.add(line, readout)
        for target in (1, 2, 3, 5, 8):
            self.play(self.t.animate.set_value(target), run_time=0.7)
            self.wait(0.25)
        self.wait(2.5)


class CausalMask(WikiScene):
    def construct(self):
        dd = np.load(os.path.join(HERE, "data", "toy_transformer_decode.npz"))
        A = dd["self_att"]
        self.add(title("Masked Multi-Head Attention: position i may only look at positions ≤ i"))
        cell = 0.62
        toks = ["BOS", "9", "5", "1", "4", "1"]
        grid = VGroup()
        for i in range(6):
            for j in range(6):
                sq = Square(cell, stroke_color=MUTED, stroke_width=1.5).set_fill(BG, 1)
                grid.add(sq.move_to([j * cell, -i * cell, 0]))
        grid.move_to([-2.6, -0.4, 0])
        rl = VGroup(*[label(t, 18, INFO).next_to(grid[6 * i], LEFT, buff=0.15) for i, t in enumerate(toks)])
        cl = VGroup(*[label(t, 18, INFO).next_to(grid[j], UP, buff=0.12) for j, t in enumerate(toks)])
        self.play(FadeIn(grid), FadeIn(rl), FadeIn(cl), run_time=0.8)
        cap = caption_bar("decoder input = target shifted right; rows = query position, columns = key position")
        self.play(FadeIn(cap), run_time=0.4)
        self.wait(0.6)
        masks = VGroup()
        for i in range(6):
            for j in range(i + 1, 6):
                m = VGroup(grid[6 * i + j].copy().set_fill(GRAD, 0.35), label("−∞", 18, GRAD).move_to(grid[6 * i + j]))
                masks.add(m)
        self.play(LaggedStart(*[FadeIn(m) for m in masks], lag_ratio=0.06), run_time=1.6)
        note = label("added to the scores before softmax: e^(−∞) = 0", 20, GRAD).move_to([3.4, 1.2, 0])
        self.play(FadeIn(note), run_time=0.5)
        self.wait(0.8)
        vals = VGroup()
        for i in range(6):
            for j in range(i + 1):
                sq = grid[6 * i + j]
                vals.add(VGroup(sq.copy().set_fill(ACCENT, 0.15 + 0.8 * float(A[i, j])), label(f"{A[i, j]:.2f}", 17, FG).move_to(sq)))
        self.play(LaggedStart(*[FadeIn(v) for v in vals], lag_ratio=0.04), run_time=1.6)
        note2 = label("trained toy model, last decoder layer:", 19, FG).move_to([3.4, 0.2, 0])
        note3 = label("every row sums to 1 over the visible past", 19, FG).next_to(note2, DOWN, buff=0.15)
        self.play(FadeIn(note2), FadeIn(note3), run_time=0.6)
        self.wait(3.0)


class AutoregressiveDecoding(WikiScene):
    def construct(self):
        dd = np.load(os.path.join(HERE, "data", "toy_transformer_decode.npz"))
        src, outs, probs, rows = dd["src"], dd["outputs"], dd["probs"], dd["cross_rows"]
        self.add(title("Generating one token at a time: reverse 3 1 4 1 5 9"))
        bw = 0.85
        def boxes(vals, y, color, x0):
            g = VGroup()
            for k, v in enumerate(vals):
                b = VGroup(RoundedRectangle(corner_radius=0.1, width=bw, height=0.7, stroke_color=color, stroke_width=2.5).set_fill(color, 0.15),
                           label(str(v), 24, FG)).move_to([x0 + k * (bw + 0.12), y, 0])
                g.add(b)
            return g
        enc = boxes([int(v) for v in src], 1.3, INFO, -2.6)
        el = label("Inputs → encoder", 20, INFO).next_to(enc, LEFT, buff=0.3)
        self.play(FadeIn(enc), FadeIn(el), run_time=0.7)
        dec_slots_y = -1.6
        dl = label("Outputs so far →", 20, HIGHLIGHT).move_to([-4.6, dec_slots_y, 0])
        cur = boxes(["BOS"], dec_slots_y, HIGHLIGHT, -2.6)
        cur[0][1].scale(0.75)
        self.play(FadeIn(dl), FadeIn(cur), run_time=0.6)
        pred_l = None
        for t in range(6):
            j = int(np.argmax(rows[t]))
            q = cur[-1]
            arrow = Arrow(q.get_top(), enc[j].get_bottom(), buff=0.08, color=GRAD, stroke_width=4)
            w = label(f"attends to input {j + 1} (weight {rows[t][j]:.2f})", 19, GRAD).move_to([3.9, 0.15, 0])
            nxt = int(outs[t + 1])
            p = label(f"predict {nxt}  (p = {probs[t]:.3f})", 22, SUCCESS).move_to([3.9, -0.35, 0])
            anims = [GrowArrow(arrow), FadeIn(w), FadeIn(p)]
            if pred_l is not None:
                anims += [FadeOut(x) for x in pred_l]
            self.play(*anims, run_time=0.55)
            new = boxes([nxt], dec_slots_y, HIGHLIGHT, -2.6 + (t + 1) * (bw + 0.12))
            self.play(FadeIn(new, shift=UP * 0.2), run_time=0.45)
            cur.add(new[0])
            pred_l = (arrow, w, p)
        self.play(*[FadeOut(x) for x in pred_l], run_time=0.3)
        res = label("output 9 5 1 4 1 3: exactly reversed (100% of 1,000 test sequences)", 21, SUCCESS).to_edge(DOWN, buff=1.0)
        cap = caption_bar("each prediction is appended and fed back (the right shift); the encoder runs only once")
        self.play(FadeIn(res), FadeIn(cap), run_time=0.6)
        self.wait(3.0)
