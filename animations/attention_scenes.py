"""Animations for the attention page (keys attention/...).
Every number shown is computed with NumPy below and matches the verified outputs quoted on the page
(notebook 14 toy example: tokens The movie was not good). Text-only labels, no LaTeX needed."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from style import *
from rnn_scenes import chip

TOKENS = ["The", "movie", "was", "not", "good"]
SCORES = np.array([0.2, 0.4, 0.1, 2.0, 1.8])
VALUES = np.array([[0.1, 0.0], [0.2, 0.1], [0.1, 0.0], [0.0, 1.0], [1.0, 0.1]])
WEIGHTS = np.exp(SCORES - SCORES.max()) / np.exp(SCORES - SCORES.max()).sum()   # [0.0708 0.0864 0.0640 0.4282 0.3506]
CONTEXT = WEIGHTS @ VALUES                                                        # [0.3813 0.4719]
MEAN = VALUES.mean(axis=0)                                                        # [0.28 0.24]
MATRIX = np.array([[0.60, 0.20, 0.10, 0.05, 0.05], [0.10, 0.55, 0.15, 0.10, 0.10],
                   [0.10, 0.20, 0.45, 0.15, 0.10], [0.05, 0.05, 0.10, 0.30, 0.50],
                   [0.05, 0.05, 0.10, 0.55, 0.25]])
NEW = MATRIX @ VALUES                                                              # good -> [0.275 0.58]


def vec(v):
    return f"[{v[0]:.2f}, {v[1]:.2f}]"


def vec3(v):
    return f"[{v[0]:.3f}, {v[1]:.3f}]"


# ---------------------------------------------------------------------------
# 1. Soft alignment: scores -> softmax weights -> weighted context vector
# ---------------------------------------------------------------------------
class SoftAlignment(WikiScene):
    def construct(self):
        self.add(title("Soft alignment: scores become weights, weights mix the values"))
        xs = [-4.8 + 2.4 * i for i in range(5)]
        base_y = -1.2
        toks = VGroup(*[label(t, 26, INFO).move_to([x, base_y - 0.45, 0]) for t, x in zip(TOKENS, xs)])
        scores = VGroup(*[label(f"score {s:.1f}", 20, MUTED).move_to([x, 2.55, 0]) for s, x in zip(SCORES, xs)])
        self.play(FadeIn(toks), run_time=0.6)
        self.play(FadeIn(scores), run_time=0.6)
        cap = caption_bar("softmax turns scores into positive weights that sum to 1")
        self.play(FadeIn(cap), run_time=0.4)
        bars, nums = VGroup(), VGroup()
        for w, x in zip(WEIGHTS, xs):
            h = 2.7 * w / WEIGHTS.max()
            b = Rectangle(width=1.1, height=h, stroke_width=0).set_fill(HIGHLIGHT if w > 0.3 else ACCENT, 0.85)
            b.move_to([x, base_y + h / 2, 0])
            bars.add(b)
            nums.add(label(f"{w:.3f}", 22, FG).next_to(b, UP, buff=0.1))
        self.play(*[GrowFromEdge(b, DOWN) for b in bars], run_time=1.2)
        self.play(FadeIn(nums), run_time=0.5)
        self.wait(0.8)
        self.play(FadeOut(cap), run_time=0.3)
        res = VGroup(
            label(f"weighted context = {vec(CONTEXT)}", 26, HIGHLIGHT, weight=BOLD),
            label(f"plain average      = {vec(MEAN)}", 24, MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        res.to_edge(DOWN, buff=0.35)
        self.play(*[b.animate.set_fill(opacity=0.35) for b, w in zip(bars, WEIGHTS) if w < 0.3], run_time=0.6)
        self.play(FadeIn(res[0], shift=UP * 0.2), run_time=0.8)
        self.play(FadeIn(res[1]), run_time=0.6)
        self.wait(2.5)


# ---------------------------------------------------------------------------
# 2. Path length: n-1 sequential hops (recurrence) vs one hop (self-attention)
# ---------------------------------------------------------------------------
class PathLength(WikiScene):
    def construct(self):
        self.add(title("Path length: recurrence hops step by step, attention connects directly"))
        xs = [-4.8 + 2.4 * i for i in range(5)]

        def row(y, color):
            return VGroup(*[VGroup(Circle(0.45, stroke_color=color, stroke_width=3).set_fill(color, 0.15),
                                   label(t, 20, FG)).move_to([x, y, 0]) for t, x in zip(TOKENS, xs)])

        top = row(1.4, INFO)
        tl = label("RNN / LSTM", 22, MUTED).next_to(top, UP, buff=0.25).to_edge(LEFT, buff=0.6)
        arrows = VGroup(*[Arrow(top[i].get_right(), top[i + 1].get_left(), buff=0.08, color=MUTED, stroke_width=3)
                          for i in range(4)])
        self.play(FadeIn(top), FadeIn(tl), Create(arrows), run_time=0.9)
        dot = Dot(top[0].get_center(), color=HIGHLIGHT, radius=0.13)
        counter = label("steps: 0", 24, HIGHLIGHT).next_to(top, DOWN, buff=0.3)
        self.play(FadeIn(dot), FadeIn(counter), run_time=0.3)
        for k in range(1, 5):
            self.play(dot.animate.move_to(top[k].get_center()),
                      Transform(counter, label(f"steps: {k}", 24, HIGHLIGHT).move_to(counter)), run_time=0.55)
        self.wait(0.3)

        bot = row(-1.9, ACCENT)
        bl = label("Self-attention", 22, MUTED).next_to(bot, UP, buff=1.25).to_edge(LEFT, buff=0.6)
        arcs = VGroup()
        for i in range(5):
            for j in range(i + 1, 5):
                arcs.add(ArcBetweenPoints(bot[i].get_top(), bot[j].get_top(), angle=-PI / 2.6,
                                          stroke_color=MUTED, stroke_width=1.6, stroke_opacity=0.6))
        self.play(FadeIn(bot), FadeIn(bl), run_time=0.5)
        self.play(Create(arcs), run_time=1.0)
        direct = ArcBetweenPoints(bot[0].get_top(), bot[4].get_top(), angle=-PI / 2.6,
                                  stroke_color=HIGHLIGHT, stroke_width=5)
        dot2 = Dot(bot[0].get_center(), color=HIGHLIGHT, radius=0.13)
        self.play(FadeIn(dot2), Create(direct), run_time=0.6)
        self.play(MoveAlongPath(dot2, direct), run_time=0.8)
        self.play(dot2.animate.move_to(bot[4].get_center()), run_time=0.2)
        c2 = label("steps: 1   (every pair computed at once)", 24, HIGHLIGHT).next_to(bot, DOWN, buff=0.3)
        self.play(FadeIn(c2), run_time=0.5)
        self.wait(2.5)


# ---------------------------------------------------------------------------
# 3. Self-attention matrix: one row = one token's weights over all tokens
# ---------------------------------------------------------------------------
class HeatmapRow(WikiScene):
    def construct(self):
        self.add(title("Self-attention matrix: each row mixes all value vectors"))
        cell = 0.95
        grid = VGroup()
        for i in range(5):
            for j in range(5):
                w = MATRIX[i, j]
                sq = Square(cell, stroke_color=BG, stroke_width=2).set_fill(ACCENT, 0.12 + 1.4 * w)
                t = label(f"{w:.2f}", 21, FG)
                grid.add(VGroup(sq, t).move_to([j * cell, -i * cell, 0]))
        grid.move_to([-2.6, -0.3, 0])
        cols = VGroup(*[label(t, 19, MUTED).next_to(grid[j], UP, buff=0.12) for j, t in enumerate(TOKENS)])
        rows = VGroup(*[label(t, 21, INFO).next_to(grid[5 * i], LEFT, buff=0.18) for i, t in enumerate(TOKENS)])
        self.play(FadeIn(grid), FadeIn(cols), FadeIn(rows), run_time=1.0)
        cap = caption_bar("rows = the token that asks · columns = the tokens it attends to · each row sums to 1")
        self.play(FadeIn(cap), run_time=0.4)
        self.wait(0.6)

        vals = VGroup(*[label(f"v {t:<6} = {vec(v)}", 20, FG, mono=True) for t, v in zip(TOKENS, VALUES)])
        vals.arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([3.5, 0.6, 0])
        self.play(FadeIn(vals), run_time=0.6)

        prev = None
        for i, color in ((4, HIGHLIGHT), (3, GRAD)):
            box = SurroundingRectangle(VGroup(*grid[5 * i:5 * i + 5]), color=color, buff=0.04, stroke_width=4)
            out = label(f"new {TOKENS[i]} = {vec3(NEW[i])}", 24, color, weight=BOLD).move_to([3.5, -1.6, 0])
            anims = [Create(box), FadeIn(out, shift=UP * 0.2)]
            if prev:
                anims += [FadeOut(prev[0]), FadeOut(prev[1])]
            self.play(*anims, run_time=0.9)
            self.wait(1.4)
            prev = (box, out)
        self.wait(2.0)
