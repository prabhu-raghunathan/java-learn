"""Animations for the generative RAG page (keys rag/...).
Every number comes from a real run of the page's pipeline on the BERT paper PDF: all-MiniLM-L6-v2 + BM25 retrieval,
google/flan-t5-base generation and distilbert-base-cased-distilled-squad extraction, CPU, transformers 5.22.2."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from style import *

HERE = os.path.dirname(os.path.abspath(__file__))


def chip(text, color, w=1.35, h=0.5, size=17, fill=0.18):
    r = RoundedRectangle(corner_radius=0.08, width=w, height=h, stroke_color=color, stroke_width=2).set_fill(color, fill)
    return VGroup(r, label(text, size, FG).move_to(r))


class TokenByToken(WikiScene):
    """Greedy decoding: at each step the decoder scores every vocabulary token and takes the most likely one."""
    def construct(self):
        G = json.load(open(os.path.join(HERE, "data", "rag_greedy.json")))
        steps = G["steps"][:10]
        self.add(title("The decoder writes one token at a time (greedy decoding)"))
        q = label("Q: " + G["q"], 18, HIGHLIGHT).move_to([0, 2.7, 0])
        self.add(q)
        cap = caption_bar("each step: a probability for all 32,128 tokens; greedy keeps the single most likely one")
        self.play(FadeIn(cap), run_time=0.4)
        answer = None
        prev = None
        for k, s in enumerate(steps):
            bars = VGroup()
            for j, (tok, p) in enumerate(s["top"]):
                t = tok.replace("▁", "_")
                col = SUCCESS if j == 0 else INFO
                bar = Rectangle(width=max(4.2 * p, 0.04), height=0.36, stroke_width=0).set_fill(col, 0.85)
                bar.move_to([-1.0 + max(4.2 * p, 0.04) / 2, 1.55 - 0.48 * j, 0])
                lt = label(f"{t}", 17, FG).next_to([-1.0, 1.55 - 0.48 * j, 0], LEFT, buff=0.2)
                lp = label(f"{p:.3f}", 16, col).next_to(bar, RIGHT, buff=0.15)
                bars.add(VGroup(bar, lt, lp))
            head = label(f"step {k + 1}: top 3 candidates", 17, MUTED).move_to([0.6, 2.15, 0])
            grp = VGroup(head, bars)
            if prev is None:
                self.play(FadeIn(grp), run_time=0.45)
            else:
                self.play(ReplacementTransform(prev, grp), run_time=0.35)
            prev = grp
            text_so_far = "".join(x["tok"] for x in steps[:k + 1]).replace("\u2581", " ").replace("</s>", "").strip()
            line = label("answer so far:  " + text_so_far, 22, SUCCESS).move_to([0, -0.35, 0]).align_to([-6.3, 0, 0], LEFT)
            if answer is None:
                self.play(FadeIn(line), run_time=0.25)
            else:
                self.play(ReplacementTransform(answer, line), run_time=0.25)
            answer = line
        cap2 = caption_bar("step 1 was a near tie: We 0.134 vs To 0.113 vs Pre 0.106. Greedy commits to We and never looks back")
        self.play(Transform(cap, cap2), run_time=0.4)
        full = label("greedy: " + G["text"][:92] + "...", 16, FG).move_to([0, -1.35, 0])
        beam = label("beam search (4 beams): " + G["beam_text"][:78] + "...", 16, HIGHLIGHT).move_to([0, -1.8, 0])
        sc = label(f"average log-probability per token: greedy -0.1395, beam -0.0966 (higher is better)", 16, MUTED).move_to([0, -2.25, 0])
        for m in (full, beam, sc):
            if m.width > 13.2:
                m.scale_to_fit_width(13.2)
        self.play(FadeIn(full), run_time=0.5)
        self.play(FadeIn(beam), FadeIn(sc), run_time=0.6)
        self.wait(3)


class HybridFusion(WikiScene):
    """Dense and BM25 rankings disagree; reciprocal rank fusion lifts the chunks both lists rank well."""
    def construct(self):
        D = json.load(open(os.path.join(HERE, "data", "rag_fusion.json")))
        rows = D["rows"]
        self.add(title("Hybrid search: fuse the meaning ranking with the keyword ranking"))
        self.add(label("Q: " + D["q"], 19, HIGHLIGHT).move_to([0, 2.7, 0]))
        head = label("chunk  page   dense rank (cosine)   BM25 rank (score)   1/(61+r_d) + 1/(61+r_b)   hybrid rank", 15, MUTED, mono=True)
        lines = VGroup(head)
        for r in sorted(rows, key=lambda r: r["dense_rank"]):
            col = SUCCESS if r["gold"] else FG
            lines.add(label(f"{r['chunk']:>5}  {r['page']:>4}   {r['dense_rank']:>4}  ({r['cos']:.3f})        {r['bm25_rank']:>4}  ({r['bm25']:>6.3f})      {r['rrf']:.5f}             {r['hybrid_rank']:>3}", 15, col, mono=True))
        lines.arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([0, 0.15, 0])
        if lines.width > 13.2:
            lines.scale_to_fit_width(13.2)
        cap = caption_bar("dense search alone puts the three answer chunks (green) at ranks 4, 6 and 22")
        self.play(FadeIn(lines[0]), LaggedStart(*[FadeIn(l, shift=RIGHT * 0.2) for l in lines[1:]], lag_ratio=0.12), FadeIn(cap), run_time=1.8)
        self.wait(0.8)
        cap2 = caption_bar("BM25 matches the rare words english and wikipedia exactly: the answer chunks are its ranks 1, 2, 3")
        self.play(Transform(cap, cap2), run_time=0.4)
        self.wait(1.0)
        rrf = label("reciprocal rank fusion: add 1/(60 + rank) from each list, then sort", 19, HIGHLIGHT).move_to([0, -2.1, 0])
        cap3 = caption_bar("after fusion the answer chunks are hybrid ranks 1, 2 and 4; chunks only one list liked fall back")
        self.play(FadeIn(rrf), Transform(cap, cap3), run_time=0.6)
        self.wait(3)


class RouteAndRefuse(WikiScene):
    """Three questions, three paths through the system: extract, generate, or refuse."""
    def construct(self):
        self.add(title("One system, three paths: extract, generate or refuse"))
        stages = ["hybrid search", "cosine >= 0.2?", "question type", "answer", "check"]
        xs = [-5.0, -2.4, 0.2, 3.0, 5.6]
        boxes = VGroup(*[chip(s, MUTED, w=2.2, h=0.6, size=17, fill=0.08).move_to([x, 1.6, 0]) for s, x in zip(stages, xs)])
        arrows = VGroup(*[Arrow([xs[i] + 1.1, 1.6, 0], [xs[i + 1] - 1.1, 1.6, 0], buff=0.05, color=MUTED, stroke_width=2) for i in range(4)])
        self.play(FadeIn(boxes), FadeIn(arrows), run_time=0.7)
        cases = [("Which corpus is used for pre-training besides English Wikipedia?", ["5 chunks", "0.517 yes", "exact fact", "extract", "span 23.91"], "BooksCorpus  (page 5)", SUCCESS),
                 ("What is next sentence prediction and why is it used?", ["5 chunks", "0.615 yes", "explain", "FLAN-T5", "support 1.00"], "To train a model that understands sentence relationships, we pre-train for a binarized next sentence prediction task ...", SUCCESS),
                 ("Who won the 2018 FIFA World Cup?", ["5 chunks", "0.141 no", "", "", ""], "The provided document does not contain enough information to answer this question.", ERROR)]
        cap = caption_bar("same pipeline for every question; the checks decide which path it takes")
        self.play(FadeIn(cap), run_time=0.3)
        prev = None
        for k, (q, vals, ans, col) in enumerate(cases):
            ql = label(q, 19, HIGHLIGHT).move_to([0, 2.55, 0])
            vs = VGroup(*[label(v, 16, col if v else MUTED).move_to([x, 0.95, 0]) for v, x in zip(vals, xs)])
            al = label(ans, 18, col).move_to([0, -0.5 - 0.0, 0])
            if al.width > 13:
                al.scale_to_fit_width(13)
            grp = VGroup(ql, vs, al)
            if prev:
                self.play(FadeOut(prev), run_time=0.3)
            self.play(FadeIn(ql), run_time=0.3)
            self.play(LaggedStart(*[FadeIn(v, shift=DOWN * 0.15) for v in vs], lag_ratio=0.25), run_time=1.0)
            self.play(FadeIn(al, shift=UP * 0.15), run_time=0.4)
            self.wait(0.9)
            prev = grp
        cap2 = caption_bar("refusing is an answer too: below cosine 0.2 nothing in the paper is about the question")
        self.play(Transform(cap, cap2), run_time=0.4)
        self.wait(3)
