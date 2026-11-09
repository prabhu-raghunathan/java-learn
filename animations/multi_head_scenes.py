"""Animations for the multi-head attention page (keys multi-head/...).
MultiHeadSplit recomputes notebook 16 with NumPy (seed 42). HeadSpecialization loads
data/multi_head_toy_attention.npz: batch-averaged attention of the toy model trained on the page
(implementation block 'Trained heads specialize', torch.manual_seed(0), 1500 Adam steps)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from style import *

TOKENS = ["The", "movie", "was", "not", "good"]
HERE = os.path.dirname(os.path.abspath(__file__))


def notebook16():
    np.random.seed(42)
    X = np.random.randn(5, 4)
    Ws = [[np.random.randn(4, 2) for _ in range(3)] for _ in range(2)]
    W_O = np.random.randn(4, 4)
    As, outs = [], []
    for WQ, WK, WV in Ws:
        S = (X @ WQ) @ (X @ WK).T / np.sqrt(2)
        A = np.exp(S - S.max(1, keepdims=True)); A /= A.sum(1, keepdims=True)
        As.append(A); outs.append(A @ (X @ WV))
    H = np.concatenate(outs, 1)
    return As, outs, H, H @ W_O


AS, OUTS, H, Z = notebook16()


def heat(M, cell, color, labels=None, fs=15):
    g = VGroup()
    n, m = M.shape
    for i in range(n):
        for j in range(m):
            sq = Square(cell, stroke_color=BG, stroke_width=1.5).set_fill(color, 0.08 + 0.92 * float(M[i, j]))
            g.add(sq.move_to([j * cell, -i * cell, 0]))
    if labels is not None:
        for i in range(n):
            for j in range(m):
                g.add(label(f"{M[i, j]:.2f}", fs, FG).move_to(g[i * m + j]))
    return g


def matrix_text(M, fs=18, color=FG):
    rows = VGroup(*[label("[" + "  ".join(f"{v:6.2f}" for v in r) + " ]", fs, color, mono=True) for r in M])
    return rows.arrange(DOWN, aligned_edge=LEFT, buff=0.08)


class MultiHeadSplit(WikiScene):
    def construct(self):
        self.add(title("Two heads attend separately, then Concat and W_O merge them"))
        h1 = heat(AS[0], 0.46, INFO, labels=True, fs=14).move_to([-4.1, 0.85, 0])
        h2 = heat(AS[1], 0.46, HIGHLIGHT, labels=True, fs=14).move_to([-4.1, -1.85, 0])
        l1 = label("head 1: A(1)", 20, INFO).next_to(h1, UP, buff=0.12)
        l2 = label("head 2: A(2)", 20, HIGHLIGHT).next_to(h2, UP, buff=0.12)
        cap = caption_bar("same X (5 × 4), different W_Q, W_K, W_V per head → different attention patterns")
        self.play(FadeIn(h1), FadeIn(l1), run_time=0.8)
        self.play(FadeIn(h2), FadeIn(l2), FadeIn(cap), run_time=0.8)
        self.wait(1.0)
        o1 = matrix_text(OUTS[0], 17, INFO).move_to([-0.3, 0.85, 0])
        o2 = matrix_text(OUTS[1], 17, HIGHLIGHT).move_to([-0.3, -1.85, 0])
        t1 = label("head1 = A(1)V(1)  (5 × 2)", 17, INFO).next_to(o1, UP, buff=0.12)
        t2 = label("head2 = A(2)V(2)  (5 × 2)", 17, HIGHLIGHT).next_to(o2, UP, buff=0.12)
        self.play(FadeIn(o1, shift=RIGHT * 0.3), FadeIn(t1), FadeIn(o2, shift=RIGHT * 0.3), FadeIn(t2), run_time=1.0)
        self.wait(0.8)
        Hc = VGroup(*[VGroup(label("[" + "  ".join(f"{v:6.2f}" for v in OUTS[0][i]), 17, INFO, mono=True),
                             label("  ".join(f"{v:6.2f}" for v in OUTS[1][i]) + " ]", 17, HIGHLIGHT, mono=True)).arrange(RIGHT, buff=0.4)
                      for i in range(5)]).arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to([3.9, 0.85, 0])
        th = label("Concat → H  (5 × 4)", 18, FG).next_to(Hc, UP, buff=0.12)
        self.play(TransformFromCopy(VGroup(o1, o2), Hc), FadeIn(th), run_time=1.2)
        self.wait(0.6)
        zt = matrix_text(Z, 17, SUCCESS).move_to([3.9, -1.85, 0])
        tz = label("Z = H W_O  (5 × 4, heads mixed)", 18, SUCCESS).next_to(zt, UP, buff=0.12)
        self.play(FadeOut(cap), run_time=0.2)
        cap2 = caption_bar("Concat only places heads side by side; the learned W_O mixes them into d_model features")
        self.play(FadeIn(zt, shift=DOWN * 0.2), FadeIn(tz), FadeIn(cap2), run_time=1.0)
        self.wait(3.0)


class HeadSpecialization(WikiScene):
    def construct(self):
        d = np.load(os.path.join(HERE, "data", "multi_head_toy_attention.npz"))
        self.add(title("Trained heads specialize: predict the previous AND the first token"))
        cell = 0.36
        hs = heat(d["single"], cell, ACCENT).move_to([-4.4, -0.35, 0])
        h1 = heat(d["head1"], cell, INFO).move_to([0.0, -0.35, 0])
        h2 = heat(d["head2"], cell, HIGHLIGHT).move_to([4.4, -0.35, 0])
        ls = label("1 head", 22, ACCENT).next_to(hs, UP, buff=0.2)
        l1 = label("2 heads: head 1", 22, INFO).next_to(h1, UP, buff=0.2)
        l2 = label("2 heads: head 2", 22, HIGHLIGHT).next_to(h2, UP, buff=0.2)
        ax = label("rows = query position, columns = key position (8 tokens)", 18, MUTED).to_edge(UP, buff=1.15)
        self.play(FadeIn(ax), run_time=0.4)
        self.play(FadeIn(hs), FadeIn(ls), run_time=1.0)
        n1 = label("splits ≈0.47 first / ≈0.51 previous", 18, ACCENT).next_to(hs, DOWN, buff=0.25)
        self.play(FadeIn(n1), run_time=0.5)
        self.wait(1.2)
        self.play(FadeIn(h1), FadeIn(l1), run_time=1.0)
        n2 = label("previous token: 0.996", 18, INFO).next_to(h1, DOWN, buff=0.25)
        self.play(FadeIn(n2), run_time=0.4)
        self.play(FadeIn(h2), FadeIn(l2), run_time=1.0)
        n3 = label("first token: 0.999", 18, HIGHLIGHT).next_to(h2, DOWN, buff=0.25)
        self.play(FadeIn(n3), run_time=0.4)
        cap = caption_bar("one softmax must average two jobs; two heads each do one job cleanly")
        self.play(FadeIn(cap), run_time=0.4)
        self.wait(3.0)
