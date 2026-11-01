"""Animations for the NLP foundations pages (keys nlp/...).
Every value shown matches the pages' verified outputs (NLTK 3.10, scikit-learn 1.9)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from style import *


def chip(text, color=FG, size=22, fill_opacity=0.15, pad=0.18):
    t = label(text, size, FG, mono=True)
    box = RoundedRectangle(corner_radius=0.1, width=t.width + 2 * pad, height=0.55,
                           stroke_color=color, stroke_width=2).set_fill(color, fill_opacity)
    return VGroup(box, t.move_to(box))


def chip_row(words, color=INFO, size=22, buff=0.15):
    return VGroup(*[chip(w, color, size) for w in words]).arrange(RIGHT, buff=buff)


class PreprocessingPipeline(WikiScene):
    def construct(self):
        head = title("One sentence through the preprocessing pipeline")
        self.add(head)
        raw = label("I am learning machine learning and loving it", 26, FG).move_to(UP * 2.35)
        self.play(FadeIn(raw), run_time=0.6)

        def stage_label(t, y, col=MUTED):
            return label(t, 20, col).move_to(LEFT * 5.2 + UP * y)

        toks = ["i", "am", "learning", "machine", "learning", "and", "loving", "it"]
        row = chip_row(toks, INFO, 21).move_to(RIGHT * 0.9 + UP * 1.3)
        s1 = stage_label("lowercase +\ntokenize", 1.3)
        self.play(FadeIn(s1), FadeIn(row, shift=DOWN * 0.3), run_time=1)
        self.wait(0.4)

        stops = [0, 1, 5, 7]
        s2 = stage_label("remove stop\nwords", 0.15)
        self.play(FadeIn(s2), *[row[i][0].animate.set_stroke(ERROR).set_fill(ERROR, 0.35) for i in stops], run_time=0.7)
        kept_row = chip_row(["learning", "machine", "learning", "loving"], INFO, 21).move_to(RIGHT * 0.9 + UP * 0.15)
        kept = list(kept_row)
        self.play(*[ReplacementTransform(row[i].copy(), kept_row[k]) for k, i in enumerate([2, 3, 4, 6])], run_time=0.9)
        n_lbl = label("4 of 8 tokens kept", 18, MUTED).next_to(kept_row, RIGHT, buff=0.3)
        self.play(FadeIn(n_lbl), run_time=0.3)
        self.wait(0.4)

        stem = chip_row(["learn", "machin", "learn", "love"], PURPLE, 21).move_to(RIGHT * 0.9 + DOWN * 1.05)
        stem[1][0].set_stroke(ERROR).set_fill(ERROR, 0.3)
        s3 = stage_label("Porter\nstemming", -1.05, PURPLE)
        self.play(FadeIn(s3), *[ReplacementTransform(kept[k].copy(), stem[k]) for k in range(4)], run_time=0.9)
        warn = label("not a word", 18, ERROR).next_to(stem, RIGHT, buff=0.3)
        self.play(FadeIn(warn), run_time=0.3)

        lem = chip_row(["learn", "machine", "learn", "love"], SUCCESS, 21).move_to(RIGHT * 0.9 + DOWN * 2.15)
        s4 = stage_label("POS-aware\nlemmatization", -2.15, SUCCESS)
        self.play(FadeIn(s4), *[ReplacementTransform(kept[k].copy(), lem[k]) for k in range(4)], run_time=0.9)
        ok = label("real words", 18, SUCCESS).next_to(lem, RIGHT, buff=0.3)
        self.play(FadeIn(ok), run_time=0.3)
        self.play(FadeIn(caption_bar("Stemming chops suffixes by rule; lemmatization looks up the dictionary form")))
        self.wait(2.5)


class ContextWindow(WikiScene):
    def construct(self):
        head = title("Word2Vec training pairs: window size 1")
        self.add(head)
        words = ["i", "love", "deep", "learning"]
        row = chip_row(words, INFO, 30, buff=0.35).move_to(UP * 1.5 + LEFT * 2.6)
        self.play(FadeIn(row), run_time=0.6)

        pairs_title = label("(centre, context) pairs", 22, MUTED).move_to(RIGHT * 3.9 + UP * 2.3)
        self.play(FadeIn(pairs_title), run_time=0.3)
        listed = VGroup()
        count = label("pairs: 0", 26, FG, mono=True).move_to(LEFT * 2.6 + DOWN * 0.4)
        self.add(count)
        win = None
        n = 0
        for i, w in enumerate(words):
            lo, hi = max(0, i - 1), min(len(words) - 1, i + 1)
            new_win = SurroundingRectangle(VGroup(*row[lo:hi + 1]), color=HIGHLIGHT, buff=0.12, stroke_width=4)
            centre = row[i][0]
            anims = [centre.animate.set_fill(HIGHLIGHT, 0.45)]
            if win is None:
                win = new_win
                anims.append(Create(win))
            else:
                anims.append(Transform(win, new_win))
                anims.append(row[i - 1][0].animate.set_fill(INFO, 0.15))
            self.play(*anims, run_time=0.6)
            for j in (i - 1, i + 1):
                if 0 <= j < len(words):
                    n += 1
                    p = label(f"({w}, {words[j]})", 22, FG, mono=True)
                    p.move_to(RIGHT * 3.9 + UP * (1.7 - 0.55 * (n - 1)))
                    arrow = Arrow(row[i].get_bottom(), row[j].get_bottom(), buff=0.05, color=GRAD,
                                  stroke_width=3, path_arc=-1.2, max_tip_length_to_length_ratio=0.25)
                    self.play(Create(arrow), FadeIn(p), Transform(count, label(f"pairs: {n}", 26, FG, mono=True).move_to(count)), run_time=0.5)
                    self.play(FadeOut(arrow), run_time=0.2)
                    listed.add(p)
        self.play(FadeOut(win), row[3][0].animate.set_fill(INFO, 0.15), run_time=0.4)
        note = label("skip-gram: centre → context\nCBOW: context → centre", 22, HIGHLIGHT).move_to(LEFT * 2.6 + DOWN * 1.6)
        self.play(FadeIn(note), run_time=0.5)
        self.play(FadeIn(caption_bar("4 words, window 1: edge words get 1 neighbour, middle words 2, so 6 pairs (window 2 gives 10)")))
        self.wait(2.5)


class CooccurrenceCoffeeTea(WikiScene):
    def construct(self):
        head = title("Co-occurrence rows: words described by their neighbours")
        self.add(head)
        corpus = ["i drink coffee every morning", "i drink tea every morning",
                  "she likes hot coffee with milk", "she likes hot tea with milk",
                  "he drives his car to work", "he parks his car near work"]
        sent = VGroup(*[label(s, 19, MUTED, mono=True) for s in corpus]).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        sent.move_to(LEFT * 3.7 + UP * 0.55)
        self.play(FadeIn(sent), run_time=0.6)

        cols = ["drink", "every", "hot", "with", "his", "to", "near"]
        rows = ["coffee", "tea", "car"]
        cw, ch = 0.82, 0.55
        x0, y0 = 1.25, 1.5
        hdr = VGroup(*[label(c, 16, MUTED).move_to([x0 + k * cw, y0 + 0.45, 0]) for k, c in enumerate(cols)])
        rlab = VGroup(*[label(r, 20, FG).move_to([x0 - 1.0, y0 - i * ch, 0]) for i, r in enumerate(rows)])
        cells = [[Square(0.48, stroke_color=MUTED, stroke_width=1.2).move_to([x0 + k * cw, y0 - i * ch, 0]) for k in range(7)] for i in range(3)]
        nums = [[label("0", 18, MUTED, mono=True).move_to(cells[i][k]) for k in range(7)] for i in range(3)]
        self.play(FadeIn(hdr), FadeIn(rlab), *[FadeIn(c) for r in cells for c in r], *[FadeIn(n) for r in nums for n in r], run_time=0.8)

        counts = np.zeros((3, 7), int)
        for si, s in enumerate(corpus):
            ws = s.split()
            hi = SurroundingRectangle(sent[si], color=HIGHLIGHT, buff=0.06)
            self.play(Create(hi), run_time=0.25)
            anims = []
            for i, w in enumerate(ws):
                if w in rows:
                    r = rows.index(w)
                    for j in (i - 1, i + 1):
                        if 0 <= j < len(ws) and ws[j] in cols:
                            k = cols.index(ws[j])
                            counts[r, k] += 1
                            col = INFO if r < 2 else PURPLE
                            new = label(str(counts[r, k]), 18, FG, mono=True).move_to(cells[r][k])
                            anims += [Transform(nums[r][k], new), cells[r][k].animate.set_fill(col, 0.35)]
            self.play(*anims, run_time=0.45)
            self.play(FadeOut(hi), run_time=0.15)

        cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
        r1 = label(f"cos(coffee, tea) = {cos(counts[0], counts[1]):.1f}", 24, SUCCESS, mono=True).move_to(RIGHT * 3.6 + DOWN * 0.9)
        r2 = label(f"cos(coffee, car) = {cos(counts[0], counts[2]):.1f}", 24, ERROR, mono=True).next_to(r1, DOWN, buff=0.25)
        self.play(FadeIn(r1), FadeIn(r2), run_time=0.7)
        self.play(FadeIn(caption_bar("Same neighbours, same row: coffee and tea come out identical without ever sharing a sentence")))
        self.wait(2.5)


class TfidfWeights(WikiScene):
    def construct(self):
        head = title("TF-IDF for review 1: order arrive two day late box damage")
        self.add(head)
        words = ["order", "arrive", "two", "day", "late", "box", "damage"]
        dfs = [10, 6, 2, 5, 3, 2, 2]
        N = 80
        idf = np.array([np.log((1 + N) / (1 + d)) + 1 for d in dfs])
        norm = float(np.linalg.norm(idf))
        tfidf = idf / norm
        xs = {"word": -5.6, "df": -3.9, "idf": -2.4}
        hdr = VGroup(label("word", 20, MUTED).move_to([xs["word"], 2.3, 0]), label("df", 20, MUTED).move_to([xs["df"], 2.3, 0]),
                     label("idf", 20, MUTED).move_to([xs["idf"], 2.3, 0]), label("tf-idf after ÷ norm", 20, MUTED).move_to([2.0, 2.3, 0]))
        self.play(FadeIn(hdr), run_time=0.4)
        ys = [1.75 - 0.6 * i for i in range(7)]
        wl = VGroup(*[label(w, 22, FG, mono=True).move_to([xs["word"], y, 0]) for w, y in zip(words, ys)])
        dl = VGroup(*[label(str(d), 22, INFO, mono=True).move_to([xs["df"], y, 0]) for d, y in zip(dfs, ys)])
        self.play(FadeIn(wl), FadeIn(dl), run_time=0.6)
        il = VGroup(*[label(f"{v:.4f}", 22, HIGHLIGHT, mono=True).move_to([xs["idf"], y, 0]) for v, y in zip(idf, ys)])
        formula = label("idf = ln((1 + 80) / (1 + df)) + 1", 22, HIGHLIGHT, mono=True).move_to(DOWN * 2.55 + LEFT * 2.6)
        self.play(FadeIn(formula), LaggedStart(*[FadeIn(m) for m in il], lag_ratio=0.12), run_time=1.2)
        self.wait(0.4)

        bar_x0, scale = -1.1, 9.0
        bars_raw = VGroup(*[Rectangle(width=v / norm * scale * 0 + v * 0.55, height=0.36, stroke_width=0).set_fill(MUTED, 0.6)
                            .move_to([bar_x0, y, 0], aligned_edge=LEFT) for v, y in zip(idf, ys)])
        self.play(LaggedStart(*[GrowFromEdge(b, LEFT) for b in bars_raw], lag_ratio=0.08), run_time=0.9)
        norm_lbl = label(f"vector length = {norm:.4f}", 22, FG, mono=True).move_to(DOWN * 2.55 + RIGHT * 3.4)
        self.play(FadeIn(norm_lbl), run_time=0.5)
        bars = VGroup(*[Rectangle(width=v * scale, height=0.36, stroke_width=0)
                        .set_fill(SUCCESS if v > 0.4 else (ERROR if v < 0.3 else INFO), 0.8)
                        .move_to([bar_x0, y, 0], aligned_edge=LEFT) for v, y in zip(tfidf, ys)])
        vals = VGroup(*[label(f"{v:.4f}", 20, FG, mono=True).next_to(b, RIGHT, buff=0.15) for v, b in zip(tfidf, bars)])
        self.play(*[Transform(a, b) for a, b in zip(bars_raw, bars)], run_time=1.1)
        self.play(FadeIn(vals), run_time=0.5)
        self.play(FadeIn(caption_bar("Rare words (df 2) score highest; order, found in 10 of 80 reviews, scores lowest")))
        self.wait(2.5)


class SplitVariance(WikiScene):
    NB = [0.9375, 0.75, 0.75, 0.8125, 0.6875, 0.6875, 0.8125, 0.75, 0.8125, 0.5625,
          0.5625, 0.75, 0.6875, 0.8125, 0.6875, 0.8125, 0.8125, 0.875, 0.8125, 0.5]
    SVM = [0.9375, 0.75, 0.5625, 0.6875, 0.6875, 0.625, 0.6875, 0.75, 0.625, 0.5625,
           0.5625, 0.75, 0.625, 0.625, 0.4375, 0.75, 0.75, 0.75, 0.875, 0.375]

    def construct(self):
        head = title("One 16-review test set is a lottery: 20 random splits")
        self.add(head)
        X = lambda a: -4.0 + (a - 0.3) / 0.7 * 9.6
        axis = Line([X(0.3), -2.2, 0], [X(1.0), -2.2, 0], color=MUTED)
        ticks = VGroup(*[VGroup(Line([X(a), -2.25, 0], [X(a), -2.15, 0], color=MUTED),
                                label(f"{int(a * 100)}%", 18, MUTED).move_to([X(a), -2.55, 0])) for a in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)])
        rows = {"Naive Bayes": (0.9, self.NB, INFO, 0.728), "Linear SVM": (-0.9, self.SVM, PURPLE, 0.674)}
        self.play(Create(axis), FadeIn(ticks), *[FadeIn(label(n, 22, c).move_to([-5.6, y + 0.05, 0])) for n, (y, _, c, _) in rows.items()], run_time=0.8)
        chance = DashedLine([X(0.5), -2.2, 0], [X(0.5), 2.0, 0], color=MUTED, stroke_width=2)
        ch_l = label("coin flip", 16, MUTED).next_to(chance, UP, buff=0.05)
        self.play(Create(chance), FadeIn(ch_l), run_time=0.4)

        for name, (y, accs, col, cvm) in rows.items():
            seen = {}
            dots = []
            for a in accs:
                k = seen.get(a, 0)
                seen[a] = k + 1
                dots.append(Dot([X(a), y - 0.45 + 0.17 * k, 0], radius=0.075, color=col))
            self.play(LaggedStart(*[FadeIn(d, scale=1.6) for d in dots], lag_ratio=0.12), run_time=1.8)
            rng = label(f"range {min(accs) * 100:.1f}% to {max(accs) * 100:.2f}%", 16, col).move_to([-5.6, y - 0.4, 0])
            self.play(FadeIn(rng), run_time=0.3)

        for name, (y, accs, col, cvm) in rows.items():
            m = Line([X(cvm), y - 0.6, 0], [X(cvm), y + 0.55, 0], color=SUCCESS, stroke_width=5)
            ml = label(f"CV mean {cvm * 100:.1f}%", 18, SUCCESS).next_to(m, DOWN, buff=0.05)
            ml.add_background_rectangle(color=BG, opacity=1, buff=0.04)
            self.play(Create(m), FadeIn(ml), run_time=0.5)
        nb_x = Dot([X(0.625), 0.9 - 0.45, 0], radius=0.13, color=ERROR).set_fill(opacity=0)
        sv_x = Dot([X(0.4375), -0.9 - 0.45, 0], radius=0.13, color=ERROR).set_fill(opacity=0)
        ring = VGroup(Circle(0.16, color=ERROR, stroke_width=4).move_to(nb_x), Circle(0.16, color=ERROR, stroke_width=4).move_to(sv_x))
        rl = label("circled: the notebook split, 62.5% vs 43.75%", 18, ERROR).move_to([X(0.82), 2.15, 0])
        self.play(Create(ring), FadeIn(rl), run_time=0.6)
        self.play(FadeIn(caption_bar("Same data, same models: the split alone moves accuracy by over 50 points")))
        self.wait(2.5)
