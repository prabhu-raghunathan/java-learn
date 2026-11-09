"""Animations for the Hugging Face pipelines page (keys hf/...).
Every number is from a real run of distilbert-base-uncased-finetuned-sst-2-english (revision 714eb0f),
the model pipeline("sentiment-analysis") loads by default, reproduced on CPU with transformers 5.22."""
import math as pymath
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from style import *


def chip(text, color, w=1.35, h=0.55, size=18, fill=0.18):
    r = RoundedRectangle(corner_radius=0.08, width=w, height=h, stroke_color=color, stroke_width=2).set_fill(color, fill)
    return VGroup(r, label(text, size, FG).move_to(r))


class PipelineSteps(WikiScene):
    def construct(self):
        self.add(title("What pipeline(\"sentiment-analysis\") does, step by step"))
        sent = label("\"The movie was surprisingly good.\"", 24, INFO).move_to([0, 2.55, 0])
        self.play(FadeIn(sent), run_time=0.5)
        toks = ["[CLS]", "the", "movie", "was", "surprisingly", "good", ".", "[SEP]"]
        ids = [101, 1996, 3185, 2001, 10889, 2204, 1012, 102]
        ws = [1.1, 0.8, 1.1, 0.8, 1.95, 0.95, 0.55, 1.1]
        xs, x = [], -6.15
        for w in ws:
            xs.append(x + w / 2); x += w + 0.12
        trow = VGroup(*[chip(t, MUTED if t.startswith("[") else INFO, w=w, size=17).move_to([cx, 1.5, 0]) for t, w, cx in zip(toks, ws, xs)])
        irow = VGroup(*[chip(str(i), HIGHLIGHT, w=w, size=17, fill=0.08).move_to([cx, 0.6, 0]) for i, w, cx in zip(ids, ws, xs)])
        l1 = label("1  tokenizer", 18, MUTED).next_to(trow, LEFT, buff=0.2).shift(LEFT * 0.0)
        cap = caption_bar("1. tokenizer: lower-case, WordPiece, add [CLS] and [SEP], map each token to its vocabulary id")
        self.play(FadeIn(cap), LaggedStart(*[TransformFromCopy(sent, t) for t in trow], lag_ratio=0.06), run_time=1.3)
        self.play(LaggedStart(*[TransformFromCopy(t, i) for t, i in zip(trow, irow)], lag_ratio=0.06), run_time=1.1)
        self.wait(0.4)
        model = chip("DistilBERT: 6 blocks, head on [CLS]", ACCENT, w=6.2, h=0.7, size=20).move_to([0, -0.55, 0])
        cap2 = caption_bar("2. forward pass: ids and attention mask through the encoder; the [CLS] vector goes to a 768 to 2 head")
        self.play(Transform(cap, cap2), GrowFromCenter(model), run_time=0.9)
        logit = label("logits  [-4.2341,  4.6189]", 22, PURPLE).move_to([-3.1, -1.65, 0])
        self.play(FadeIn(logit, shift=DOWN * 0.3), run_time=0.7)
        cap3 = caption_bar("3. postprocess: softmax, argmax, then config.id2label turns index 1 into POSITIVE")
        probs = label("softmax  [0.000143,  0.999857]", 22, SUCCESS).move_to([3.1, -1.65, 0])
        arrow = Arrow(logit.get_right(), probs.get_left(), buff=0.15, color=FG, stroke_width=3)
        self.play(Transform(cap, cap3), GrowArrow(arrow), FadeIn(probs), run_time=0.9)
        out = label("{'label': 'POSITIVE', 'score': 0.99986}", 24, HIGHLIGHT, mono=True).move_to([0, -2.45, 0])
        self.play(FadeIn(out, shift=UP * 0.2), run_time=0.7)
        self.wait(3)


class ConfidenceSaturation(WikiScene):
    def construct(self):
        self.add(title("Two-class confidence is the sigmoid of the logit margin"))
        ax = Axes(x_range=[-10, 10, 2], y_range=[0, 1, 0.25], x_length=7.6, y_length=4.2, tips=False,
                  axis_config={"color": MUTED, "stroke_width": 2}).shift(DOWN * 0.45 + LEFT * 2.6)
        xl = label("margin z1 - z0", 18, MUTED).next_to(ax.c2p(10, 0), UP + LEFT, buff=0.15)
        yl = VGroup(*[label(f"{v:g}", 16, MUTED).next_to(ax.c2p(-10, v), LEFT, buff=0.1) for v in (0, 0.5, 1)])
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=0.8)
        curve = ax.plot(lambda m: 1 / (1 + pymath.exp(-m)), x_range=[-10, 10, 0.05], color=PURPLE, stroke_width=4)
        self.play(Create(curve), run_time=1.0)
        cap = caption_bar("softmax over [z0, z1] gives P(POSITIVE) = 1 / (1 + exp(-(z1 - z0)))")
        self.play(FadeIn(cap), run_time=0.4)
        pts = [("not terrible", 4.1242, INFO), ("expected more", -4.7388, ERROR), ("fine, I guess", 8.319, HIGHLIGHT),
               ("love this movie", 8.9761, SUCCESS)]
        legend = VGroup()
        for name, m, col in pts:
            p = 1 / (1 + pymath.exp(-m))
            dot = Dot(ax.c2p(m, p), color=col, radius=0.09)
            legend.add(label(f"{name}: margin {m:+.2f}, P(POS) {p:.4f}", 17, col))
            self.play(FadeIn(dot, scale=0.5), run_time=0.45)
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.18).to_edge(RIGHT, buff=0.35).shift(DOWN * 0.4)
        self.play(FadeIn(legend), run_time=0.6)
        cap2 = caption_bar("beyond a margin of about 5 every score looks near-certain: high confidence is not correctness")
        self.play(Transform(cap, cap2), run_time=0.4)
        self.wait(3)
