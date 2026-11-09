"""Animations for the full self-attention page (keys self-attention/...).
Every number is computed below with NumPy, reproducing notebook 15 (np.random.seed(42) setup)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from style import *

TOKENS = ["The", "movie", "was", "not", "good"]


def notebook_state():
    np.random.seed(42)
    X = np.random.randn(5, 4)
    W_Q, W_K, W_V = (np.random.randn(4, 3) for _ in range(3))
    dims, raw, scaled = [2, 4, 8, 16, 32, 64, 128], [], []
    for d in dims:                                   # notebook cell 43, same random stream
        r = np.array([np.dot(np.random.randn(d), np.random.randn(d)) for _ in range(1000)])
        raw.append(r.std()); scaled.append((r / np.sqrt(d)).std())
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    S = Q @ K.T / np.sqrt(3)
    A = np.exp(S - S.max(1, keepdims=True)); A /= A.sum(1, keepdims=True)
    return dims, np.array(raw), np.array(scaled), A, V


DIMS, RAW_STD, SCALED_STD, A, V = notebook_state()   # raw std [1.38 ... 11.02], scaled std ~1


def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()


# ---------------------------------------------------------------------------
class ScalingVariance(WikiScene):
    def construct(self):
        self.add(title("Dot products grow like √d_k; dividing by √d_k keeps them at 1"))
        base, h_unit, bw = -2.4, 0.36, 0.55
        xs = [-4.8 + 1.6 * i for i in range(7)]
        axis = Line([-5.6, base, 0], [5.6, base, 0], color=MUTED, stroke_width=2)
        dl = VGroup(*[label(f"d_k={d}", 18, MUTED).move_to([x, base - 0.35, 0]) for d, x in zip(DIMS, xs)])
        self.play(Create(axis), FadeIn(dl), run_time=0.6)
        cap = caption_bar("std of q·k over 1000 random pairs, unit-variance entries")
        self.play(FadeIn(cap), run_time=0.3)
        raw = VGroup(); rl = VGroup()
        for x, s in zip(xs, RAW_STD):
            b = Rectangle(width=bw, height=s * h_unit, stroke_width=0).set_fill(GRAD, 0.85)
            b.move_to([x - 0.3, base + s * h_unit / 2, 0]); raw.add(b)
            rl.add(label(f"{s:.2f}", 17, GRAD).next_to(b, UP, buff=0.06))
        self.play(*[GrowFromEdge(b, DOWN) for b in raw], run_time=1.6)
        self.play(FadeIn(rl), run_time=0.4)
        leg1 = label("raw q·k  (theory √d_k)", 20, GRAD).to_corner(UL, buff=0.6).shift(DOWN * 0.9)
        self.play(FadeIn(leg1), run_time=0.3)
        self.wait(1.2)
        sc = VGroup(); sl = VGroup()
        for x, s in zip(xs, SCALED_STD):
            b = Rectangle(width=bw, height=s * h_unit, stroke_width=0).set_fill(SUCCESS, 0.9)
            b.move_to([x + 0.3, base + s * h_unit / 2, 0]); sc.add(b)
            sl.add(label(f"{s:.2f}", 17, SUCCESS).next_to(b, UP, buff=0.06))
        leg2 = label("q·k / √d_k", 20, SUCCESS).next_to(leg1, DOWN, aligned_edge=LEFT, buff=0.15)
        self.play(*[GrowFromEdge(b, DOWN) for b in sc], FadeIn(leg2), run_time=1.2)
        self.play(FadeIn(sl), run_time=0.4)
        self.wait(3.0)


# ---------------------------------------------------------------------------
class SoftmaxSharpness(WikiScene):
    def construct(self):
        self.add(title("Same ranking, bigger numbers: softmax sharpens toward one-hot"))
        sets = [("small", [1.0, 1.2, 0.8, 1.1, 0.9]), ("medium", [2.0, 4.0, 1.0, 3.0, 0.5]),
                ("large", [10.0, 20.0, 5.0, 15.0, 2.0])]
        xs = [-4.0 + 2.0 * i for i in range(5)]
        base, H = -2.2, 3.4
        axis = Line([-5.2, base, 0], [5.2, base, 0], color=MUTED, stroke_width=2)
        tl = VGroup(*[label(t, 20, INFO).move_to([x, base - 0.35, 0]) for t, x in zip(TOKENS, xs)])
        self.add(axis, tl)

        def bars_for(p):
            g = VGroup()
            for x, v in zip(xs, p):
                h = max(H * v, 0.02)
                b = Rectangle(width=1.0, height=h, stroke_width=0).set_fill(HIGHLIGHT if v == p.max() else ACCENT, 0.85)
                b.move_to([x, base + h / 2, 0])
                g.add(VGroup(b, label(f"{v:.3f}", 19, FG).next_to(b, UP, buff=0.08)))
            return g

        cur = None; head = None
        for name, s in sets:
            p = softmax(np.array(s))
            new = bars_for(p)
            h2 = label(f"{name} scores {s}  →  max weight {p.max():.3f}", 22, FG).to_edge(UP, buff=1.2)
            if cur is None:
                self.play(FadeIn(new), FadeIn(h2), run_time=0.9)
            else:
                self.play(Transform(cur, new), Transform(head, h2), run_time=1.1)
            if cur is None:
                cur, head = new, h2
            self.wait(1.2)
        cap = caption_bar("a near one-hot row ignores context and passes almost no gradient to the other tokens")
        self.play(FadeIn(cap), run_time=0.4)
        self.wait(2.5)


# ---------------------------------------------------------------------------
class ZEqualsAV(WikiScene):
    def construct(self):
        self.add(title("z_good = Σ A[good, j] · v_j: one row of A mixes all value vectors"))
        w = A[4]; contrib = w[:, None] * V; z = contrib.sum(0)
        rows = VGroup()
        for t, a, v, c in zip(TOKENS, w, V, contrib):
            r = VGroup(
                label(f"{a:.3f}", 26, HIGHLIGHT if a > 0.3 else MUTED, mono=True),
                label("×", 26, MUTED),
                label(f"v_{t:<5} [{v[0]:6.3f} {v[1]:6.3f} {v[2]:6.3f}]", 25, INFO, mono=True),
                label("=", 26, MUTED),
                label(f"[{c[0]:6.3f} {c[1]:6.3f} {c[2]:6.3f}]", 25, FG, mono=True),
            ).arrange(RIGHT, buff=0.25)
            rows.add(r)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([0, 0.55, 0])
        for k in range(5):
            r = rows[k]
            self.play(FadeIn(r[0]), FadeIn(r[1]), FadeIn(r[2]), run_time=0.45)
        self.wait(0.9)
        self.play(*[FadeIn(r[3]) for r in rows], *[FadeIn(r[4], shift=LEFT * 0.2) for r in rows], run_time=0.9)
        big = [r for r, a in zip(rows, w) if a > 0.3]
        self.play(*[Indicate(r[4], color=HIGHLIGHT, scale_factor=1.05) for r in big], run_time=0.8)
        line = Line(rows.get_corner(DL) + DOWN * 0.15, rows.get_corner(DR) + DOWN * 0.15, color=MUTED)
        total = label(f"z_good = [{z[0]:.3f}, {z[1]:.3f}, {z[2]:.3f}]", 28, HIGHLIGHT, weight=BOLD).next_to(line, DOWN, buff=0.3)
        self.play(Create(line), FadeIn(total, shift=UP * 0.2), run_time=0.8)
        cap = caption_bar("“was” and “good” supply 96% of the weight; “not” only 0.03% (random, untrained W)")
        self.play(FadeIn(cap), run_time=0.4)
        self.wait(3.0)
