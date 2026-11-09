"""Animations for the stacked-encoder mini project page (keys stacked-encoder/...).
All numbers come from data/stacked_encoder_tiny.json, exported from the notebook's 3-block model
(torch.manual_seed(7), 60 epochs, reproduced exactly on CPU): hidden-state norms, the last hidden
state of "not good movie", its masked and plain mean, and layer-by-layer attention (head 1)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from style import *

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "data", "stacked_encoder_tiny.json")))


def box(text, color, w=1.6, h=0.9, size=22, fill=0.18):
    r = RoundedRectangle(corner_radius=0.1, width=w, height=h, stroke_color=color, stroke_width=2.5).set_fill(color, fill)
    return VGroup(r, label(text, size, FG).move_to(r))


class DepthRefinement(WikiScene):
    def construct(self):
        self.add(title("Stacking blocks keeps the shape and changes the numbers"))
        names = ["Embedding", "Block 1", "Block 2", "Block 3"]
        cols = [INFO, ACCENT, ACCENT, ACCENT]
        xs = [-4.8, -1.6, 1.6, 4.8]
        boxes = VGroup(*[box(n, c, w=1.9, size=20).move_to([x, 1.2, 0]) for n, c, x in zip(names, cols, xs)])
        arrows = VGroup(*[Arrow([xs[k] + 0.97, 1.2, 0], [xs[k + 1] - 0.97, 1.2, 0], buff=0, color=FG, stroke_width=4)
                          for k in range(3)])
        self.play(FadeIn(boxes[0]), run_time=0.5)
        shape0 = label("[1, 6, 32]", 22, MUTED, mono=True).next_to(boxes[0], DOWN, buff=0.3)
        norm0 = label(f"mean norm {DATA['norms'][0]:.2f}", 20, HIGHLIGHT).next_to(shape0, DOWN, buff=0.2)
        self.play(FadeIn(shape0), FadeIn(norm0), run_time=0.5)
        cap = caption_bar("sentence: not good movie (3 real tokens + 3 pads), the notebook's trained 3-block model")
        self.play(FadeIn(cap), run_time=0.4)
        shapes, norms, changes = [shape0], [norm0], []
        for k in range(1, 4):
            self.play(GrowArrow(arrows[k - 1]), FadeIn(boxes[k]), run_time=0.6)
            s = label("[1, 6, 32]", 22, MUTED, mono=True).next_to(boxes[k], DOWN, buff=0.3)
            nn_ = label(f"mean norm {DATA['norms'][k]:.2f}", 20, HIGHLIGHT).next_to(s, DOWN, buff=0.2)
            ch = label(f"avg |change| {DATA['changes'][k - 1]:.2f}", 18, GRAD).next_to(arrows[k - 1], UP, buff=0.25)
            self.play(FadeIn(s), FadeIn(nn_), FadeIn(ch), run_time=0.6)
            shapes.append(s); norms.append(nn_); changes.append(ch)
            self.wait(0.3)
        note = label("after LayerNorm with gamma = 1, a 32-dim vector has norm close to sqrt(32) = 5.66", 20, FG).move_to([0, -1.6, 0])
        self.play(FadeIn(note), run_time=0.5)
        cap2 = caption_bar("same shape after every block; each block re-mixes tokens (attention) and re-transforms them (FFN)")
        self.play(Transform(cap, cap2), run_time=0.4)
        self.wait(3)


class MaskedPooling(WikiScene):
    def construct(self):
        self.add(title("Masked mean pooling: one sentence vector from the real tokens only"))
        toks = ["not", "good", "movie", "PAD", "PAD", "PAD"]
        last = DATA["last"]
        xs = [-5.4 + 1.25 * k for k in range(6)]
        cols = VGroup()
        for k, (t, x) in enumerate(zip(toks, xs)):
            real = t != "PAD"
            head = label(t, 20, FG if real else MUTED).move_to([x, 2.4, 0])
            cells = VGroup(*[box(f"{last[k][r]:.2f}", SUCCESS if real else MUTED, w=1.05, h=0.5, size=17, fill=0.15).move_to([x, 1.75 - 0.55 * r, 0])
                             for r in range(4)])
            m = box("1" if real else "0", ERROR if real else MUTED, w=0.6, h=0.45, size=18).move_to([x, -0.9, 0])
            cols.add(VGroup(head, cells, m))
        lab_m = label("mask", 18, MUTED).next_to(cols[0][2], LEFT, buff=0.3)
        lab_d = label("first 4 of 32 dims shown", 16, MUTED).move_to([xs[1], -0.42, 0])
        self.play(LaggedStart(*[FadeIn(c) for c in cols], lag_ratio=0.1), FadeIn(lab_m), FadeIn(lab_d), run_time=1.4)
        cap = caption_bar("last hidden state of the trained 3-block model; padded positions still hold vectors")
        self.play(FadeIn(cap), run_time=0.4)
        self.wait(0.6)
        self.play(*[cols[k].animate.set_opacity(0.2) for k in range(3, 6)], run_time=0.8)
        res_x = 3.6
        hdr = label("masked mean = sum of 3 real / 3", 20, HIGHLIGHT).move_to([res_x + 0.9, 2.75, 0])
        pooled = VGroup(*[box(f"{DATA['pooled'][r]:.3f}", HIGHLIGHT, w=1.2, h=0.5, size=17).move_to([res_x + 0.3, 1.75 - 0.55 * r, 0]) for r in range(4)])
        plain = VGroup(*[label(f"{DATA['plain'][r]:.3f}", 17, MUTED).move_to([res_x + 1.9, 1.75 - 0.55 * r, 0]) for r in range(4)])
        plain_h = label("plain mean", 16, MUTED).move_to([res_x + 1.9, 2.25, 0])
        self.play(FadeIn(hdr), *[TransformFromCopy(VGroup(*[cols[k][1][r] for k in range(3)]), pooled[r]) for r in range(4)], run_time=1.4)
        self.play(FadeIn(plain), FadeIn(plain_h), run_time=0.6)
        cap2 = caption_bar("the plain mean would also average the PAD rows: the result would depend on how much padding a sentence got")
        self.play(Transform(cap, cap2), run_time=0.4)
        self.wait(3)


class AttentionByLayer(WikiScene):
    def construct(self):
        self.add(title("Each block learns its own token-mixing pattern (head 1, not good movie)"))
        toks = ["not", "good", "movie"]
        cell = 0.72
        grids = VGroup()
        for l, A in enumerate(DATA["attn"]):
            g = VGroup()
            for i in range(3):
                for j in range(3):
                    v = A[i][j]
                    sq = Square(cell, stroke_color=MUTED, stroke_width=1.5).set_fill(ACCENT, 0.1 + 0.85 * v)
                    sq.move_to([j * cell, -i * cell, 0])
                    g.add(VGroup(sq, label(f"{v:.2f}", 16, FG).move_to(sq)))
            g.move_to([-4.3 + 4.3 * l, -0.2, 0])
            rl = VGroup(*[label(t, 16, INFO).next_to(g[3 * i], LEFT, buff=0.12) for i, t in enumerate(toks)])
            cl = VGroup(*[label(t, 16, INFO).next_to(g[j], UP, buff=0.1) for j, t in enumerate(toks)])
            name = label(f"Layer {l + 1}", 22, FG).next_to(cl, UP, buff=0.3)
            grids.add(VGroup(g, rl, cl, name))
        cap = caption_bar("rows = query token, columns = key token; padded keys get weight 0 and are hidden")
        self.play(FadeIn(cap), run_time=0.4)
        for gr in grids:
            self.play(FadeIn(gr), run_time=0.9)
            self.wait(0.5)
        cap2 = caption_bar("layer 1 is sharp (good and movie look at good, not looks at movie); layers 2 and 3 spread attention out")
        self.play(Transform(cap, cap2), run_time=0.4)
        self.wait(1.5)
        cap3 = caption_bar("attention maps are a diagnostic, not an explanation of why the model decided NEGATIVE")
        self.play(Transform(cap, cap3), run_time=0.4)
        self.wait(3)
