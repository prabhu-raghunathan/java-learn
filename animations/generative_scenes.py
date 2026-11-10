"""Animations for the autoencoders and GANs page (keys generative/...).
Digits and reconstructions come from the page's PyTorch autoencoders trained on scikit-learn's 8x8 digits;
GAN points are real samples from the page's 2-D GAN runs (healthy and mode-collapsed) at the logged steps."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from style import *

HERE = os.path.dirname(os.path.abspath(__file__))


def digit(vals, cell=0.26, color=FG):
    g = VGroup()
    for k, v in enumerate(vals):
        r, c = divmod(k, 8)
        sq = Square(side_length=cell, stroke_width=0.4, stroke_color=MUTED).set_fill(color, float(v))
        sq.move_to([(c - 3.5) * cell, (3.5 - r) * cell, 0])
        g.add(sq)
    return g


class BottleneckSqueeze(WikiScene):
    """An autoencoder squeezes 64 pixels into k numbers and rebuilds them; a smaller k loses more detail."""
    def construct(self):
        A = json.load(open(os.path.join(HERE, "data", "gen_autoencoder.json")))
        self.add(title("Autoencoder: 64 pixels squeezed into k numbers, then rebuilt"))
        x = digit(A["input"][0]).move_to([-4.6, 0.2, 0])
        lx = label("input: 64 pixels", 20, INFO).next_to(x, DOWN, buff=0.3)
        cap = caption_bar("the encoder compresses, the decoder rebuilds; the only signal is how close the rebuild is")
        self.play(FadeIn(x), FadeIn(lx), FadeIn(cap), run_time=0.7)
        enc = Polygon([-3.2, 1.3, 0], [-1.2, 0.5, 0], [-1.2, -0.1, 0], [-3.2, -0.9, 0], stroke_color=INFO, stroke_width=2).set_fill(INFO, 0.15)
        dec = Polygon([1.2, 0.5, 0], [3.2, 1.3, 0], [3.2, -0.9, 0], [1.2, -0.1, 0], stroke_color=PURPLE, stroke_width=2).set_fill(PURPLE, 0.15)
        self.play(FadeIn(enc), FadeIn(dec), FadeIn(label("encoder", 18, INFO).move_to([-2.2, 1.6, 0])),
                  FadeIn(label("decoder", 18, PURPLE).move_to([2.2, 1.6, 0])), run_time=0.6)
        prev = None
        for k in (16, 8, 2):
            z = VGroup(*[Square(side_length=0.2, stroke_width=1, stroke_color=SUCCESS).set_fill(SUCCESS, 0.6) for _ in range(k)])
            z.arrange_in_grid(rows=max(1, k // 4) if k > 2 else 1, buff=0.04).move_to([0, 0.2, 0])
            zl = label(f"latent z: {k} numbers", 18, SUCCESS).move_to([0, -1.35, 0])
            out = digit(A[str(k)][0], color=FG).move_to([4.6, 0.2, 0])
            ol = label(f"rebuilt: test MSE {A['mse' + str(k)]}", 20, HIGHLIGHT).next_to(out, DOWN, buff=0.3)
            grp = VGroup(z, zl, out, ol)
            if prev is None:
                self.play(FadeIn(grp), run_time=0.8)
            else:
                self.play(ReplacementTransform(prev, grp), run_time=0.9)
            prev = grp
            self.play(Transform(cap, caption_bar(f"k = {k}: average squared error per pixel on 450 test digits is {A['mse' + str(k)]}")), run_time=0.4)
            self.wait(1.3)
        self.play(Transform(cap, caption_bar("a tighter bottleneck keeps only the gist: predicting the mean image scores 0.0740")), run_time=0.5)
        self.wait(2.5)


class GanModeCollapse(WikiScene):
    """Two GANs on the same 8-mode ring: one learns every mode, the other collapses onto two."""
    def construct(self):
        G = json.load(open(os.path.join(HERE, "data", "gen_gan_ring.json")))
        self.add(title("GAN training: covering all 8 modes vs mode collapse"))
        centers = []
        import math
        for k in range(8):
            a = 2 * math.pi * k / 8
            centers.append((2 * math.cos(a), 2 * math.sin(a)))
        panels = {}
        for name, cx, col in (("healthy", -3.4, INFO), ("collapsed", 3.4, ERROR)):
            ax = VGroup(*[Circle(radius=0.17, stroke_color=SUCCESS, stroke_width=2).move_to([cx + 0.72 * x, -0.15 + 0.72 * y, 0]) for x, y in centers])
            head = label("balanced learning rates" if name == "healthy" else "discriminator 100x faster than generator", 19, col).move_to([cx, 2.55, 0])
            self.add(ax, head)
            panels[name] = (cx, col)
        cap = caption_bar("green rings: the 8 real modes. Dots: 300 generator samples at each training step")
        self.add(cap)
        prev = None
        for s in G["steps"]:
            grp = VGroup()
            for name, (cx, col) in panels.items():
                pts = G[name][str(s)]["pts"]
                dots = VGroup(*[Dot([cx + 0.72 * max(-2.9, min(2.9, p[0])), -0.15 + 0.72 * max(-2.9, min(2.9, p[1])), 0], radius=0.025, color=col) for p in pts])
                info = label(f"step {s}: {G[name][str(s)]['modes']} of 8 modes", 20, FG).move_to([cx, -2.75, 0])
                grp.add(dots, info)
            if prev is None:
                self.play(FadeIn(grp), run_time=0.7)
            else:
                self.play(ReplacementTransform(prev, grp), run_time=0.9)
            prev = grp
            self.wait(0.6)
        self.play(Transform(cap, caption_bar("collapse: 93% of samples sit on real modes, so quality looks fine, but 6 of 8 modes are never produced")), run_time=0.5)
        self.wait(2.5)
