"""Animations for the BERT page (keys bert/...).
InputRepresentation draws the paper's Figure 2 example. MaskingProcedure uses a numpy simulation of the
15% / 80-10-10 rule over 1,000,000 tokens (seed 0). LeftVsBidirectional and SpanScoring load
data/bert_tiny.json, exported from the page's tiny BERT (H=64, L=4, A=4, torch.manual_seed(0)):
real per-position accuracies and real start/end scores."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from style import *

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "data", "bert_tiny.json")))


def chip(text, color, w=0.95, h=0.5, size=17, fill_opacity=0.18):
    box = RoundedRectangle(corner_radius=0.08, width=w, height=h, stroke_color=color, stroke_width=2)
    box.set_fill(color, fill_opacity)
    return VGroup(box, label(text, size, FG).move_to(box))


class InputRepresentation(WikiScene):
    def construct(self):
        self.add(title("BERT input = token embedding + segment embedding + position embedding"))
        toks = ["[CLS]", "my", "dog", "is", "cute", "[SEP]", "he", "likes", "play", "##ing", "[SEP]"]
        n = len(toks)
        xs = [-5.0 + 1.07 * k for k in range(n)]
        y_in, y_tok, y_seg, y_pos = 2.0, 0.95, -0.15, -1.25
        names = VGroup(label("Input", 18, MUTED).move_to([-6.35, y_in, 0]),
                       label("Token", 18, MUTED).move_to([-6.35, y_tok, 0]),
                       label("Segment", 18, MUTED).move_to([-6.35, y_seg, 0]),
                       label("Position", 18, MUTED).move_to([-6.35, y_pos, 0]))
        inp = VGroup(*[chip(t, ERROR, size=15).move_to([x, y_in, 0]) for t, x in zip(toks, xs)])
        self.play(FadeIn(names[0]), LaggedStart(*[FadeIn(c) for c in inp], lag_ratio=0.06), run_time=1.2)
        cap = caption_bar("sentence pair packed into one sequence: [CLS] sentence A [SEP] sentence B [SEP]")
        self.play(FadeIn(cap), run_time=0.4)
        self.wait(0.5)
        tok = VGroup(*[chip("E", HIGHLIGHT, size=17).move_to([x, y_tok, 0]) for x in xs])
        for c, t in zip(tok, toks):
            c[1].become(label("E(" + t.replace("[", "").replace("]", "").replace("##", "") + ")", 13, FG).move_to(c[0]))
        self.play(FadeIn(names[1]), *[TransformFromCopy(a, b) for a, b in zip(inp, tok)], run_time=1.2)
        plus1 = VGroup(*[label("+", 22, FG).move_to([x, (y_tok + y_seg) / 2, 0]) for x in xs])
        seg = VGroup(*[chip("E(A)" if k <= 5 else "E(B)", SUCCESS if k <= 5 else INFO, size=17).move_to([x, y_seg, 0])
                       for k, x in enumerate(xs)])
        self.play(FadeIn(names[2]), FadeIn(plus1), LaggedStart(*[FadeIn(c) for c in seg], lag_ratio=0.05), run_time=1.2)
        cap2 = caption_bar("segment embedding: one learned vector for sentence A, another for sentence B")
        self.play(Transform(cap, cap2), run_time=0.4)
        self.wait(0.6)
        plus2 = VGroup(*[label("+", 22, FG).move_to([x, (y_seg + y_pos) / 2, 0]) for x in xs])
        pos = VGroup(*[chip(f"E({k})", MUTED, size=17).move_to([x, y_pos, 0]) for k, x in enumerate(xs)])
        self.play(FadeIn(names[3]), FadeIn(plus2), LaggedStart(*[FadeIn(c) for c in pos], lag_ratio=0.05), run_time=1.2)
        cap3 = caption_bar("position embedding: learned, one per index 0..511 (not the sinusoids of the original Transformer)")
        self.play(Transform(cap, cap3), run_time=0.4)
        self.wait(0.6)
        sums = VGroup(*[chip("x", PURPLE, w=0.95, h=0.55, size=17).move_to([x, -2.35, 0]) for x in xs])
        for k, c in enumerate(sums):
            c[1].become(label(f"x{k}", 16, FG).move_to(c[0]))
        anims = [ReplacementTransform(VGroup(tok[k].copy(), seg[k].copy(), pos[k].copy()), sums[k]) for k in range(n)]
        self.play(LaggedStart(*anims, lag_ratio=0.05), run_time=1.6)
        cap4 = caption_bar("sum each column, then LayerNorm and dropout: one H-dimensional vector per token enters layer 1")
        self.play(Transform(cap, cap4), run_time=0.4)
        self.wait(3)


class MaskingProcedure(WikiScene):
    def construct(self):
        sim = DATA["mask_sim"]
        self.add(title("Masked LM: choose 15% of positions, then 80% [MASK], 10% random, 10% unchanged"))
        words = ["the", "man", "went", "to", "the", "store", "he", "bought", "a", "gallon", "of", "milk"]
        xs = [-5.6 + 1.02 * k for k in range(len(words))]
        row = VGroup(*[chip(w, INFO, w=0.95, size=16).move_to([x, 2.0, 0]) for w, x in zip(words, xs)])
        self.play(LaggedStart(*[FadeIn(c) for c in row], lag_ratio=0.05), run_time=1.0)
        cap = caption_bar("15% of 12 tokens is about 2: here 'store' and 'milk' are chosen for prediction")
        self.play(FadeIn(cap), run_time=0.4)
        chosen = [5, 11]
        self.play(*[row[k][0].animate.set_stroke(HIGHLIGHT, 4) for k in chosen], run_time=0.6)
        new1 = chip("[MASK]", GRAD, w=0.95, size=14).move_to(row[5])
        new2 = chip("milk", HIGHLIGHT, w=0.95, size=16).move_to(row[11])
        t1 = label("80%: replaced by [MASK]", 17, GRAD).next_to(row[5], DOWN, buff=0.25)
        t2 = label("10%: kept as is", 17, HIGHLIGHT).next_to(row[11], DOWN, buff=0.25).shift(LEFT * 0.6)
        self.play(Transform(row[5], new1), FadeIn(t1), run_time=0.7)
        self.play(Transform(row[11], new2), FadeIn(t2), run_time=0.7)
        lab = label("loss only at chosen positions: predict 'store' and 'milk' from both sides", 19, FG).move_to([0, 0.55, 0])
        self.play(FadeIn(lab), run_time=0.5)
        self.wait(0.8)
        # simulation bar: 1,000,000 tokens
        W = 11.0
        x0 = -W / 2
        y = -1.2
        segs = [("not chosen 85.0%", 1 - sim["chosen"], MUTED), ("[MASK] 12.0%", sim["mask"], GRAD),
                ("random 1.5%", sim["random"], ERROR), ("same 1.5%", sim["same"], HIGHLIGHT)]
        bars, texts = VGroup(), VGroup()
        cx = x0
        for k, (name, frac, col) in enumerate(segs):
            w = W * frac
            r = Rectangle(width=w, height=0.6, stroke_width=1.5, stroke_color=col).set_fill(col, 0.45)
            r.move_to([cx + w / 2, y, 0])
            bars.add(r)
            cx += w
        texts.add(label(segs[0][0], 18, FG).move_to(bars[0]))
        right = VGroup(*[label(s[0], 17, s[2]) for s in segs[1:]]).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        right.next_to(bars[-1], DOWN, buff=0.3).align_to(bars[1], LEFT).shift(LEFT * 1.6)
        hdr = label("numpy simulation over 1,000,000 tokens (seed 0)", 18, MUTED).next_to(bars, UP, buff=0.2)
        self.play(FadeOut(cap), FadeIn(hdr), LaggedStart(*[GrowFromEdge(b, LEFT) for b in bars], lag_ratio=0.3), run_time=1.4)
        self.play(FadeIn(texts), FadeIn(right), run_time=0.6)
        cap2 = caption_bar("of the chosen 15%: 80.1% / 10.0% / 10.0%, so random replacement touches only 1.5% of all tokens")
        self.play(FadeIn(cap2), run_time=0.4)
        self.wait(3)


class LeftVsBidirectional(WikiScene):
    def construct(self):
        bi, ltr = DATA["pos_acc_bidir"], DATA["pos_acc_ltr"]
        self.add(title("Why bidirectional? Masked-token accuracy by position in sentence A"))
        ax = Axes(x_range=[0.4, 6.6, 1], y_range=[0, 1.05, 0.25], x_length=9.5, y_length=4.0, tips=False,
                  axis_config={"color": MUTED, "stroke_width": 2}).shift(DOWN * 0.4 + LEFT * 0.3)
        ylab = VGroup(*[label(f"{int(v * 100)}%", 16, MUTED).next_to(ax.c2p(0.4, v), LEFT, buff=0.12) for v in (0, 0.5, 1.0)])
        xlab = VGroup(*[label(f"pos {k}", 16, MUTED).next_to(ax.c2p(k, 0), DOWN, buff=0.15) for k in range(1, 7)])
        self.play(Create(ax), FadeIn(ylab), FadeIn(xlab), run_time=0.8)
        cap = caption_bar("same tiny encoder, same masked-LM data; one token masked; sentence = arithmetic progression mod 23")
        self.play(FadeIn(cap), run_time=0.4)
        def bars(vals, dx, col):
            g = VGroup()
            for k, v in enumerate(vals, start=1):
                h = max(ax.c2p(0, v)[1] - ax.c2p(0, 0)[1], 0.02)
                r = Rectangle(width=0.55, height=h, stroke_width=1.5, stroke_color=col).set_fill(col, 0.55)
                r.move_to(ax.c2p(k + dx, 0), aligned_edge=DOWN)
                g.add(r)
            return g
        b1 = bars(ltr, -0.17, GRAD)
        leg1 = label("left-to-right (causal mask)", 18, GRAD).to_corner(UR, buff=0.5).shift(DOWN * 0.9)
        self.play(FadeIn(leg1), LaggedStart(*[GrowFromEdge(b, DOWN) for b in b1], lag_ratio=0.15), run_time=1.4)
        n1 = label(f"{ltr[0] * 100:.1f}%", 16, GRAD).next_to(b1[0], UP, buff=0.08)
        n2 = label(f"{ltr[1] * 100:.1f}%", 16, GRAD).next_to(b1[1], UP, buff=0.08)
        self.play(FadeIn(n1), FadeIn(n2), run_time=0.4)
        cap2 = caption_bar("pos 1 sees only [CLS]: chance is 1/23 = 4.3%; pos 2 sees one value: the step (1..5) is a 1-in-5 guess")
        self.play(Transform(cap, cap2), run_time=0.4)
        self.wait(1.5)
        b2 = bars(bi, 0.17, SUCCESS)
        leg2 = label("bidirectional (BERT)", 18, SUCCESS).next_to(leg1, DOWN, aligned_edge=LEFT, buff=0.12)
        self.play(FadeIn(leg2), LaggedStart(*[GrowFromEdge(b, DOWN) for b in b2], lag_ratio=0.15), run_time=1.4)
        cap3 = caption_bar("with right context every position is recoverable: 100% at all six positions")
        self.play(Transform(cap, cap3), run_time=0.4)
        self.wait(3)


class SpanScoring(WikiScene):
    def construct(self):
        sp = DATA["span"]
        toks, st, en = sp["tokens"], sp["start"], sp["end"]
        para = list(range(4, 14))
        self.add(title("SQuAD head: score every start with S, every end with E, take the best span with j ≥ i"))
        q = label("question:  " + " ".join(toks[1:3]), 22, INFO).move_to([-3.6, 2.75, 0])
        self.play(FadeIn(q), run_time=0.5)
        xs = [-4.95 + 1.1 * k for k in range(len(para))]
        row = VGroup(*[chip(toks[p], INFO, w=0.9, size=18).move_to([x, -2.15, 0]) for p, x in zip(para, xs)])
        self.play(LaggedStart(*[FadeIn(c) for c in row], lag_ratio=0.05), run_time=0.9)
        base_s, base_e = 1.2, -0.8
        k = 0.15
        def bar(v, x, base, col):
            h = max(abs(v) * k, 0.02)
            r = Rectangle(width=0.36, height=h, stroke_width=1.2, stroke_color=col).set_fill(col, 0.6)
            if v >= 0:
                r.move_to([x, base, 0], aligned_edge=DOWN)
            else:
                r.move_to([x, base, 0], aligned_edge=UP)
            return r
        ls = label("S·Tᵢ (start)", 18, SUCCESS).move_to([5.6, base_s + 0.6, 0])
        le = label("E·Tⱼ (end)", 18, GRAD).move_to([5.6, base_e + 0.6, 0])
        axs = VGroup(Line([-5.5, base_s, 0], [5.2, base_s, 0], color=MUTED, stroke_width=1.5),
                     Line([-5.5, base_e, 0], [5.2, base_e, 0], color=MUTED, stroke_width=1.5))
        bs = VGroup(*[bar(st[p], x - 0.2, base_s, SUCCESS) for p, x in zip(para, xs)])
        be = VGroup(*[bar(en[p], x + 0.2, base_e, GRAD) for p, x in zip(para, xs)])
        self.play(FadeIn(axs), FadeIn(ls), LaggedStart(*[GrowFromEdge(b, DOWN) for b in bs], lag_ratio=0.05), run_time=1.2)
        self.play(FadeIn(le), LaggedStart(*[GrowFromEdge(b, DOWN) for b in be], lag_ratio=0.05), run_time=1.2)
        cap = caption_bar("real scores from the fine-tuned tiny BERT: one start bar and one end bar stand out")
        self.play(FadeIn(cap), run_time=0.4)
        self.wait(0.8)
        i, j = sp["pred"]
        ii, jj = para.index(i), para.index(j)
        v1 = label(f"{st[i]:.2f}", 17, SUCCESS).next_to(bs[ii], UP, buff=0.08)
        v2 = label(f"{en[j]:.2f}", 17, GRAD).next_to(be[jj], UP, buff=0.08)
        self.play(FadeIn(v1), FadeIn(v2), run_time=0.5)
        box = SurroundingRectangle(VGroup(row[ii], row[jj]), color=HIGHLIGHT, buff=0.1, stroke_width=4)
        self.play(Create(box), run_time=0.7)
        res = label(f"best span: {' '.join(toks[i:j + 1])}   score = {st[i]:.2f} + {en[j]:.2f} = {st[i] + en[j]:.2f}", 22, HIGHLIGHT)
        res.move_to([1.6, 2.75, 0])
        cap2 = caption_bar("only S and E (2 × H numbers) are new; every T_i comes from the pre-trained encoder")
        self.play(FadeIn(res), Transform(cap, cap2), run_time=0.6)
        self.wait(3)
