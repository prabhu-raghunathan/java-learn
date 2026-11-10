"""Animations for the retriever + reader page (keys rr/...).
Every number comes from a real run of sentence-transformers/all-MiniLM-L6-v2 (retriever) and
distilbert-base-cased-distilled-squad (reader) on the BERT paper PDF, notebook 21's code, CPU, transformers 5.22.2."""
import json
import math as pymath
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from style import *

HERE = os.path.dirname(os.path.abspath(__file__))


class SemanticNeighbours(WikiScene):
    """Real cosines of five sentences to one query, drawn as angles from the query direction."""
    def construct(self):
        self.add(title("Meaning as direction: cosine to the query"))
        q = "What are the disadvantages of the method?"
        cands = [("The method has some weaknesses when applied to noisy data.", 0.582322, SUCCESS),
                 ("The limitations of the proposed approach are discussed in this section.", 0.425186, SUCCESS),
                 ("The paper introduces a new training strategy.", 0.227246, INFO),
                 ("The model achieves strong performance on several benchmark datasets.", 0.126243, INFO),
                 ("The cat is sleeping on the sofa.", 0.006088, ERROR)]
        O = [-3.6, -2.2, 0]
        R = 3.4
        qa = Arrow(O, [O[0] + R, O[1], 0], buff=0, color=HIGHLIGHT, stroke_width=6)
        ql = label("query: " + q, 18, HIGHLIGHT).next_to(qa, DOWN, buff=0.15).align_to(qa, LEFT)
        arc = Arc(radius=R, start_angle=0, angle=PI / 2, arc_center=O, color=MUTED, stroke_width=1.5)
        self.play(Create(arc), GrowArrow(qa), FadeIn(ql), run_time=0.9)
        cap = caption_bar("angle = arccos(cosine) between each 384-d sentence vector and the query vector")
        self.play(FadeIn(cap), run_time=0.4)
        rows = VGroup()
        for k, (t, c, col) in enumerate(cands):
            ang = pymath.acos(c)
            end = [O[0] + R * pymath.cos(ang), O[1] + R * pymath.sin(ang), 0]
            a = Arrow(O, end, buff=0, color=col, stroke_width=4)
            row = label(f"{c:.3f}  ({pymath.degrees(ang):.1f} deg)  {t}", 16, col)
            rows.add(row)
            self.play(GrowArrow(a), run_time=0.5)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([3.0, 0.6, 0])
        if rows.width > 7.6:
            rows.scale_to_fit_width(7.6)
        rows.to_edge(RIGHT, buff=0.3)
        self.play(FadeIn(rows), run_time=0.6)
        cap2 = caption_bar("no shared keyword, yet weaknesses and limitations point closest; the cat is at about 90 degrees")
        self.play(Transform(cap, cap2), run_time=0.4)
        self.wait(3)


class RetrieveThenRead(WikiScene):
    """78 cosines, top-5 retrieved, the reader's spans, then the combined score picks BooksCorpus."""
    def construct(self):
        D = json.load(open(os.path.join(HERE, "data", "rr_retrieve_read.json")))
        sims, cands = D["sims"], D["cands"]
        self.add(title("Retrieve 5 of 78 chunks, then read only those"))
        ql = label("Q: " + D["q"], 18, HIGHLIGHT).move_to([0, 2.75, 0])
        self.add(ql)
        ax = Axes(x_range=[0, 78, 10], y_range=[0, 0.6, 0.2], x_length=12.0, y_length=2.0, tips=False,
                  axis_config={"color": MUTED, "stroke_width": 2}).move_to([0, 1.1, 0])
        yl = VGroup(*[label(f"{v:.1f}", 13, MUTED).next_to(ax.c2p(0, v), LEFT, buff=0.08) for v in (0, 0.2, 0.4)])
        self.play(Create(ax), FadeIn(yl), run_time=0.6)
        cap = caption_bar("step 1: one matrix product gives the cosine of the question with every chunk")
        self.play(FadeIn(cap), run_time=0.3)
        top = [c["c"] for c in cands]
        bars = VGroup()
        for i, s in enumerate(sims):
            h = max(abs(ax.c2p(0, s)[1] - ax.c2p(0, 0)[1]), 0.01)
            b = Rectangle(width=0.1, height=h, stroke_width=0).set_fill(INFO, 0.75)
            b.move_to(ax.c2p(i + 0.5, max(s, 0) / 2))
            bars.add(b)
        self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.01), run_time=1.4)
        cap2 = caption_bar("step 2: keep the top 5 (yellow); the other 73 chunks are never read")
        self.play(Transform(cap, cap2), *[bars[i].animate.set_fill(HIGHLIGHT, 1) for i in top],
                  *[bars[i].animate.set_fill(MUTED, 0.3) for i in range(78) if i not in top], run_time=0.8)
        # step 3: table of reader outputs
        lo_r, hi_r = min(c["cos"] for c in cands), max(c["cos"] for c in cands)
        lo_s, hi_s = min(c["s"] for c in cands), max(c["s"] for c in cands)
        rows = VGroup()
        head = label("rank  chunk  cosine   reader span                   span score   final (α = 0.4)", 15, MUTED, mono=True)
        rows.add(head)
        finals = []
        for c in cands:
            f = 0.4 * (c["cos"] - lo_r) / (hi_r - lo_r + 1e-10) + 0.6 * (c["s"] - lo_s) / (hi_s - lo_s + 1e-10)
            finals.append(f)
        best = max(range(5), key=lambda i: finals[i])
        for i, c in enumerate(cands):
            a = c["a"][:26]
            col = SUCCESS if i == best else FG
            rows.add(label(f"{c['r']:>4}  {c['c']:>5}  {c['cos']:.4f}   {a:<28}  {c['s']:>8.2f}   {finals[i]:.3f}", 15, col, mono=True))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([0, -1.35, 0])
        if rows.width > 13:
            rows.scale_to_fit_width(13)
        cap3 = caption_bar("step 3: the reader extracts one span per retrieved chunk; min-max both scores, mix with α = 0.4")
        self.play(Transform(cap, cap3), FadeIn(rows[0]), LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in rows[1:]], lag_ratio=0.15), run_time=1.6)
        box = SurroundingRectangle(rows[best + 1], color=SUCCESS, buff=0.06)
        cap4 = caption_bar("retrieval rank 4 wins: BooksCorpus, page 5. Retrieval alone (rank 1) would have read the wrong chunk")
        self.play(Create(box), Transform(cap, cap4), run_time=0.7)
        self.wait(3)


class MinMaxStretch(WikiScene):
    """Notebook's main-idea question: five cosines within 0.022 of each other become 0 to 1."""
    def construct(self):
        self.add(title("Min-max normalization stretches tiny gaps to the full 0 to 1 range"))
        ret = [0.2365582, 0.22704573, 0.22055396, 0.2158128, 0.21425034]
        names = ["chunk 51", "chunk 57", "chunk 71", "chunk 53", "chunk 54"]
        lo, hi = min(ret), max(ret)
        norm = [(r - lo) / (hi - lo + 1e-10) for r in ret]
        ax1 = Axes(x_range=[0, 5, 1], y_range=[0, 1, 0.25], x_length=5.2, y_length=3.4, tips=False,
                   axis_config={"color": MUTED, "stroke_width": 2}).move_to([-3.4, -0.2, 0])
        ax2 = Axes(x_range=[0, 5, 1], y_range=[0, 1, 0.25], x_length=5.2, y_length=3.4, tips=False,
                   axis_config={"color": MUTED, "stroke_width": 2}).move_to([3.4, -0.2, 0])
        t1 = label("raw cosine (true scale 0 to 1)", 18, INFO).next_to(ax1, UP, buff=0.2)
        t2 = label("after min-max within the top 5", 18, ERROR).next_to(ax2, UP, buff=0.2)
        for ax in (ax1, ax2):
            ax.add(VGroup(*[label(f"{v:g}", 13, MUTED).next_to(ax.c2p(0, v), LEFT, buff=0.08) for v in (0, 0.5, 1)]))
        self.play(Create(ax1), Create(ax2), FadeIn(t1), FadeIn(t2), run_time=0.8)
        cap = caption_bar("question: What is the main idea of this document?  top-5 cosines 0.2366 down to 0.2143")
        self.play(FadeIn(cap), run_time=0.3)
        b1, b2 = VGroup(), VGroup()
        for i, (r, n) in enumerate(zip(ret, norm)):
            h1 = max(ax1.c2p(0, r)[1] - ax1.c2p(0, 0)[1], 0.01)
            r1 = Rectangle(width=0.6, height=h1, stroke_width=0).set_fill(INFO, 0.85).move_to(ax1.c2p(i + 0.5, r / 2))
            l1 = label(f"{r:.4f}", 13, FG).next_to(r1, UP, buff=0.06)
            b1.add(VGroup(r1, l1))
            h2 = max(ax2.c2p(0, n)[1] - ax2.c2p(0, 0)[1], 0.01)
            r2 = Rectangle(width=0.6, height=h2, stroke_width=0).set_fill(ERROR, 0.85).move_to(ax2.c2p(i + 0.5, n / 2))
            l2 = label(f"{n:.3f}", 13, FG).next_to(r2, UP, buff=0.06)
            b2.add(VGroup(r2, l2))
        self.play(LaggedStart(*[FadeIn(b, shift=UP * 0.2) for b in b1], lag_ratio=0.12), run_time=1.0)
        self.play(*[TransformFromCopy(a, b) for a, b in zip(b1, b2)], run_time=1.2)
        cap2 = caption_bar("a 0.022 gap becomes a 1.0 gap: rank 5 gets 0 retrieval credit, even though all five are equally weak")
        self.play(Transform(cap, cap2), run_time=0.4)
        note = label("the top-ranked candidate always gets 1 and the last always 0, whatever the absolute relevance", 17, HIGHLIGHT).move_to([0, -2.45, 0])
        self.play(FadeIn(note), run_time=0.5)
        self.wait(3)
