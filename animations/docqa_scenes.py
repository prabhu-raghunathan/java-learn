"""Animations for the extractive document QnA page (keys docqa/...).
Every number comes from a real run of distilbert-base-cased-distilled-squad on the BERT paper PDF
(17 Bert_Model.pdf, 16 pages), reproduced on CPU with transformers 5.22.2 (see data/docqa_candidates.json)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from style import *

HERE = os.path.dirname(os.path.abspath(__file__))


def chip(text, color, w=1.35, h=0.55, size=18, fill=0.18):
    r = RoundedRectangle(corner_radius=0.08, width=w, height=h, stroke_color=color, stroke_width=2).set_fill(color, fill)
    return VGroup(r, label(text, size, FG).move_to(r))


class ChunkWindow(WikiScene):
    """Page 1 has 584 words; 180-word windows step by 140 words, so neighbours share 40 words."""
    def construct(self):
        self.add(title("Overlapping word chunks: size 180, overlap 40, step 140"))
        W, x0, x1 = 584, -6.2, 6.2
        sx = lambda w: x0 + (x1 - x0) * w / W
        bar = Rectangle(width=x1 - x0, height=0.5, stroke_color=MUTED, stroke_width=2).set_fill(MUTED, 0.12).move_to([0, 1.9, 0])
        blabel = label("page 1: 584 words", 20, MUTED).next_to(bar, UP, buff=0.15)
        ticks = VGroup(*[label(str(w), 15, MUTED).move_to([sx(w), 1.45, 0]) for w in (0, 140, 280, 420, 584)])
        self.play(FadeIn(bar), FadeIn(blabel), FadeIn(ticks), run_time=0.8)
        ans = Rectangle(width=sx(61) - sx(55), height=0.5, stroke_width=0).set_fill(HIGHLIGHT, 0.85).move_to([(sx(55) + sx(61)) / 2, 1.9, 0])
        alab = label("answer at words 55 to 60", 16, HIGHLIGHT).next_to(ans, UP, buff=0.12).shift(RIGHT * 1.2)
        self.play(FadeIn(ans), FadeIn(alab), run_time=0.6)
        cap = caption_bar("each chunk starts where the previous one ended, minus 40 words")
        self.play(FadeIn(cap), run_time=0.4)
        spans = [(0, 180), (140, 320), (280, 460), (420, 584)]
        cols = [INFO, ACCENT, PURPLE, GRAD]
        boxes = VGroup()
        for k, ((a, b), col) in enumerate(zip(spans, cols)):
            y = 0.75 - 0.72 * k
            r = Rectangle(width=sx(b) - sx(a), height=0.5, stroke_color=col, stroke_width=2.5).set_fill(col, 0.22).move_to([(sx(a) + sx(b)) / 2, y, 0])
            t = label(f"chunk {k}: words {a} to {b}  ({b - a} words)", 16, FG).move_to(r)
            boxes.add(VGroup(r, t))
            self.play(GrowFromEdge(r, LEFT), FadeIn(t), run_time=0.7)
            if k > 0:
                ov = Rectangle(width=sx(a + 40) - sx(a), height=0.5 + 0.72, stroke_width=0).set_fill(HIGHLIGHT, 0.25).move_to([(sx(a) + sx(a + 40)) / 2, y + 0.36, 0])
                self.play(FadeIn(ov), run_time=0.35)
        cap2 = caption_bar("the last window is shorter (164 words); the 16 pages give 78 chunks in total")
        self.play(Transform(cap, cap2), run_time=0.4)
        note = label("40 shared words: a sentence cut at one boundary is whole in the neighbour", 18, HIGHLIGHT).move_to([0, -2.35, 0])
        self.play(FadeIn(note), run_time=0.6)
        self.wait(3)


class ChunkRanking(WikiScene):
    """Best span score of each of the 78 chunks for one question, then the top-5 after ranking."""
    def construct(self):
        data = json.load(open(os.path.join(HERE, "data", "docqa_candidates.json")))
        rows = [r for r in data["questions"][0]["rows"]]
        self.add(title("\"What does BERT stand for?\" asked of all 78 chunks"))
        ax = Axes(x_range=[0, 78, 10], y_range=[-10, 25, 5], x_length=12.0, y_length=4.0, tips=False,
                  axis_config={"color": MUTED, "stroke_width": 2}).shift(DOWN * 0.35)
        ylab = VGroup(*[label(str(v), 14, MUTED).next_to(ax.c2p(0, v), LEFT, buff=0.1) for v in (-10, 0, 10, 20)])
        xl = label("chunk id", 16, MUTED).next_to(ax.c2p(78, -10), DOWN, buff=0.15).shift(LEFT * 0.4)
        yl = label("best span score", 16, MUTED).next_to(ax.c2p(0, 25), RIGHT, buff=0.15)
        self.play(Create(ax), FadeIn(ylab), FadeIn(xl), FadeIn(yl), run_time=0.8)
        cap = caption_bar("one forward pass per chunk; each chunk returns its own best span and its raw score s_i + e_j")
        self.play(FadeIn(cap), run_time=0.4)
        bars = VGroup()
        for r in rows:
            v = r["ns"] if r["ok"] else 0.0
            col = SUCCESS if r["c"] == 0 else (ERROR if r["c"] == 27 else INFO)
            b = Rectangle(width=0.1, height=max(abs(ax.c2p(0, v)[1] - ax.c2p(0, 0)[1]), 0.01), stroke_width=0).set_fill(col, 0.9)
            b.move_to(ax.c2p(r["c"] + 0.5, v / 2))
            bars.add(b)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.02), run_time=2.2)
        thr = DashedLine(ax.c2p(0, 5), ax.c2p(78, 5), color=HIGHLIGHT, stroke_width=2.5)
        tl = label("threshold 5.0", 15, HIGHLIGHT).next_to(ax.c2p(78, 5), UP, buff=0.08).shift(LEFT * 0.7)
        self.play(Create(thr), FadeIn(tl), run_time=0.6)
        top = sorted(rows, key=lambda r: -r["ns"])[:2]
        t1 = label(f"chunk 0, page 1: {top[0]['na'][:26]}... {top[0]['ns']:.3f}", 17, SUCCESS).move_to([2.6, 2.2, 0])
        t2 = label(f"chunk 27, page 6: \"{top[1]['na']}\"  {top[1]['ns']:.3f}", 17, ERROR).next_to(t1, DOWN, buff=0.12, aligned_edge=LEFT)
        self.play(FadeIn(t1), run_time=0.5)
        self.play(FadeIn(t2), run_time=0.5)
        n_above = sum(1 for r in rows if r["ok"] and r["ns"] >= 5)
        cap2 = caption_bar(f"rank 1 wins by 0.074 over a wrong span; {n_above} of 78 chunks clear the 5.0 threshold")
        self.play(Transform(cap, cap2), run_time=0.4)
        self.wait(3)


class OffsetsToLabels(WikiScene):
    """answer_start (a character index) becomes start/end token labels through offset_mapping."""
    def construct(self):
        self.add(title("Fine-tuning labels: answer_start 16 becomes tokens 15 to 27"))
        ctx = "BERT stands for Bidirectional Encoder Representations from Transformers."
        text = label(ctx, 22, FG, mono=True).move_to([0, 2.35, 0])
        self.play(FadeIn(text), run_time=0.6)
        cw = text.width / len(ctx)
        cx = lambda k: text.get_left()[0] + cw * (k + 0.5)
        mark = Rectangle(width=cw * (71 - 16), height=0.5, stroke_color=HIGHLIGHT, stroke_width=2.5).move_to([(cx(16) + cx(70)) / 2, 2.35, 0])
        c16 = label("char 16", 16, HIGHLIGHT).next_to(mark, UP, buff=0.08).align_to(mark, LEFT)
        c71 = label("char 71 (exclusive)", 16, HIGHLIGHT).next_to(mark, UP, buff=0.08).align_to(mark, RIGHT)
        cap = caption_bar("the dataset stores the answer text and its character start: answer_start = 16")
        self.play(Create(mark), FadeIn(c16), FadeIn(c71), FadeIn(cap), run_time=0.9)
        toks = [("B", 0, 1), ("##ER", 1, 3), ("##T", 3, 4), ("stands", 5, 11), ("for", 12, 15), ("B", 16, 17), ("##id", 17, 19),
                ("##ire", 19, 22), ("##ctional", 22, 29), ("En", 30, 32), ("##code", 32, 36), ("##r", 36, 37), ("Rep", 38, 41),
                ("##res", 41, 44), ("##entation", 44, 52), ("##s", 52, 53), ("from", 54, 58), ("Transformers", 59, 71), (".", 71, 72)]
        row = VGroup()
        for k, (t, a, b) in enumerate(toks):
            inside = 16 <= a and b <= 71
            ch = chip(t, SUCCESS if inside else MUTED, w=max(0.42, 0.13 * len(t) + 0.25), h=0.48, size=14, fill=0.2 if inside else 0.08)
            row.add(ch)
        row.arrange(RIGHT, buff=0.06).move_to([0, 0.75, 0])
        if row.width > 13.4:
            row.scale_to_fit_width(13.4)
        idx = VGroup(*[label(str(10 + k), 13, MUTED).next_to(c, DOWN, buff=0.08) for k, c in enumerate(row)])
        offs = VGroup(*[label(f"{a}-{b}", 11, MUTED).next_to(i, DOWN, buff=0.05) for i, (t, a, b) in zip(idx, toks)])
        cap2 = caption_bar("tokenize with return_offsets_mapping=True: every context token knows its character span")
        self.play(Transform(cap, cap2), LaggedStart(*[FadeIn(c, shift=DOWN * 0.2) for c in row], lag_ratio=0.04), run_time=1.4)
        self.play(FadeIn(idx), FadeIn(offs), run_time=0.6)
        s_arrow = Arrow([cx(16), 2.05, 0], row[5].get_top(), buff=0.08, color=HIGHLIGHT, stroke_width=3)
        e_arrow = Arrow([cx(70), 2.05, 0], row[17].get_top(), buff=0.08, color=HIGHLIGHT, stroke_width=3)
        cap3 = caption_bar("start token: offset contains char 16 → 15;  end token: offset contains char 71 → 27")
        self.play(Transform(cap, cap3), GrowArrow(s_arrow), GrowArrow(e_arrow), run_time=0.9)
        res = label("start_positions = 15,  end_positions = 27,  loss on this example 0.8377", 20, SUCCESS).move_to([0, -1.6, 0])
        self.play(FadeIn(res, shift=UP * 0.2), run_time=0.6)
        self.wait(3)
