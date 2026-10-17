"""Animations for docs/03-ai-ml/01-python-for-data-exploration/*.md

Every number shown comes from the verified runs quoted on the pages
(NumPy 2.5.4, pandas 3.0.6, Matplotlib 3.11.2, Seaborn 0.13.2) or is computed
here with numpy from the same ShopStream data.
Render with:  python animations/render.py numpy/<name>   (keys in MANIFEST)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "data_exploration"))

from style import *
from python_scenes import PyScene, box, retext, arrow


def cell(text, color=FG, w=1.0, h=0.6, size=22, fill=0.12, mono=True):
    return box(text, color, width=w, height=h, size=size, mono=mono, fill_opacity=fill)


def row_of(texts, color=FG, w=1.0, h=0.6, size=22, buff=0.0):
    return VGroup(*[cell(t, color, w, h, size) for t in texts]).arrange(RIGHT, buff=buff)


def col_of(texts, color=FG, w=1.0, h=0.6, size=22, buff=0.0):
    return VGroup(*[cell(t, color, w, h, size) for t in texts]).arrange(DOWN, buff=buff)


def shopstream_clean():
    from shopstream import make_shopstream
    from clean import clean_orders
    raw, customers = make_shopstream()
    return clean_orders(raw, customers)


# ============================================================================
# NumPy page 1
# ============================================================================
class ListVsArray(PyScene):
    def construct(self):
        self.add(title("Python list vs NumPy array: where the numbers live"))
        lt = label("list of floats", size=24, color=INFO).move_to([-3.6, 2.0, 0])
        at = label("np.ndarray float64", size=24, color=SUCCESS).move_to([3.4, 2.0, 0])
        self.play(FadeIn(lt), FadeIn(at), run_time=0.4)

        ptrs = col_of(["ptr"] * 4, INFO, w=1.0, h=0.55, size=18).move_to([-5.2, 0.0, 0])
        objs = VGroup(*[cell(f"float {v}", PURPLE, w=1.7, h=0.55, size=18) for v in ["0.0", "1.0", "2.0", "3.0"]])
        spots = [[-2.6, 1.1, 0], [-1.4, 0.2, 0], [-3.0, -0.8, 0], [-1.6, -1.6, 0]]
        for o, p in zip(objs, spots):
            o.move_to(p)
        self.play(FadeIn(ptrs), run_time=0.4)
        arrs = VGroup(*[arrow(pt.get_right(), o.get_left(), INFO, width=2.5) for pt, o in zip(ptrs, objs)])
        self.play(LaggedStart(*[AnimationGroup(GrowArrow(a), FadeIn(o)) for a, o in zip(arrs, objs)], lag_ratio=0.25), run_time=1.2)
        self.say("Each element is a separate 24-byte object; the list stores pointers to them")
        self.wait(0.6)

        buf = row_of(["0.0", "1.0", "2.0", "3.0"], SUCCESS, w=1.0, h=0.7, size=20).move_to([3.4, 0.6, 0])
        bytes_lbl = label("one buffer, 8 bytes per number, side by side", size=18, color=MUTED).next_to(buf, DOWN, buff=0.2)
        self.play(FadeIn(buf), FadeIn(bytes_lbl), run_time=0.6)
        self.say("The array stores the raw numbers contiguously with one dtype")
        self.wait(0.5)

        hop = Dot(color=HIGHLIGHT, radius=0.09).move_to(ptrs[0])
        self.play(FadeIn(hop), run_time=0.2)
        for o, pt in zip(objs, ptrs):
            self.play(hop.animate.move_to(pt), run_time=0.15)
            self.play(hop.animate.move_to(o), run_time=0.25)
        sweep = Rectangle(width=0.15, height=0.8, color=HIGHLIGHT, fill_opacity=0.6, stroke_width=0).move_to(buf[0].get_left())
        self.play(FadeIn(sweep), run_time=0.15)
        self.play(sweep.animate.move_to(buf[-1].get_right()), run_time=0.6, rate_func=linear)
        self.say("Loop: hop object to object, type-check each. Vectorized: one C loop sweeps the buffer")
        self.wait(0.4)

        stats = VGroup(
            label("1M floats    list: ~32.4 MB    array: 8 MB", size=22, mono=True),
            label("sum of x*2   loop: 25.2 ms     vectorized: 0.99 ms", size=22, mono=True, color=HIGHLIGHT),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).move_to([0, -2.3, 0])
        self.play(FadeOut(self._cap), FadeIn(stats), run_time=0.6)
        self._cap = None
        self.wait(2.5)


class StridesViews(PyScene):
    def construct(self):
        self.add(title("One buffer, many views: shape + strides"))
        buf = row_of([str(i) for i in range(12)], INFO, w=0.8, h=0.6, size=20).move_to([0, 2.0, 0])
        bl = label("data buffer: 12 x float64 (8 bytes each), never moves", size=18, color=MUTED).next_to(buf, DOWN, buff=0.15)
        self.play(FadeIn(buf), FadeIn(bl), run_time=0.6)

        def grid(vals, color):
            return VGroup(*[row_of([str(v) for v in r], color, w=0.7, h=0.5, size=18) for r in vals]).arrange(DOWN, buff=0)

        m = np.arange(12).reshape(3, 4)
        views = [
            ("m = a.reshape(3, 4)", "strides (32, 8)", m, SUCCESS),
            ("m.T", "strides (8, 32)", m.T, PURPLE),
            ("m[:, ::2]", "strides (32, 16)", m[:, ::2], HIGHLIGHT),
        ]
        shown = None
        for i, (code, strides, vals, color) in enumerate(views):
            self.code(code, pos=UP * 0.6)
            g = grid(vals, color).move_to([-1.0, -1.4, 0])
            s = label(strides, size=24, mono=True, color=color).next_to(g, RIGHT, buff=0.8)
            if shown is None:
                self.play(FadeIn(g), FadeIn(s), run_time=0.6)
            else:
                self.play(FadeOut(shown), FadeIn(g), FadeIn(s), run_time=0.6)
            shown = VGroup(g, s)
            hl = [buf[int(v)][0].copy().set_stroke(color, 5) for v in vals[0]]
            self.play(*[Create(h) for h in hl], run_time=0.5)
            self.play(*[FadeOut(h) for h in hl], run_time=0.3)
            self.say(["Next element in a row: +8 bytes; next row: +32 bytes",
                      "Transpose swaps the strides: same buffer, no copy",
                      "Step 2 along rows: stride 16; still a view (np.shares_memory is True)"][i])
            self.wait(0.6)

        self.code("m[[0, 2]]   (fancy indexing)", pos=UP * 0.6, color=ERROR)
        new_buf = row_of(["0", "1", "2", "3", "8", "9", "10", "11"], ERROR, w=0.7, h=0.5, size=18).move_to([0, -1.4, 0])
        nl = label("NEW buffer: elements gathered and copied", size=20, color=ERROR).next_to(new_buf, DOWN, buff=0.2)
        self.play(FadeOut(shown), FadeIn(new_buf), FadeIn(nl), run_time=0.6)
        self.say("Arbitrary selections can't be described by strides, so NumPy copies")
        self.wait(2.5)


# ============================================================================
# NumPy page 2
# ============================================================================
class Broadcasting(PyScene):
    def construct(self):
        self.add(title("Broadcasting: (3, 1) + (1, 4) → (3, 4)"))
        w, h = 0.8, 0.6
        col = col_of(["0", "1", "2"], INFO, w=w, h=h).move_to([-4.6, 0.2, 0])
        row = row_of(["0", "1", "2", "3"], PURPLE, w=w, h=h).move_to([-1.0, 1.6, 0])
        cl = label("(3, 1)", size=22, color=INFO, mono=True).next_to(col, DOWN, buff=0.2)
        rl = label("(1, 4)", size=22, color=PURPLE, mono=True).next_to(row, UP, buff=0.15)
        self.play(FadeIn(col), FadeIn(row), FadeIn(cl), FadeIn(rl), run_time=0.6)
        self.say("Align shapes from the right: each pair must be equal, or one of them 1")
        self.wait(0.6)

        origin = np.array([-2.2, 0.2, 0])
        col_st = VGroup(*[VGroup(*[cell(str(i), INFO, w, h, fill=0.05) for _ in range(4)]).arrange(RIGHT, buff=0) for i in range(3)]).arrange(DOWN, buff=0).move_to(origin)
        self.play(TransformFromCopy(col, col_st), run_time=0.9)
        self.say("The column is stretched across 4 columns (stride 0: no copy)")
        self.wait(0.4)
        row_st = VGroup(*[VGroup(*[cell(str(j), PURPLE, w, h, fill=0.05) for j in range(4)]).arrange(RIGHT, buff=0) for _ in range(3)]).arrange(DOWN, buff=0).move_to(origin + RIGHT * 3.9)
        self.play(TransformFromCopy(row, row_st), run_time=0.9)
        self.say("The row is stretched down 3 rows, also virtual")
        self.wait(0.4)

        plus = label("+", size=36).move_to(origin + RIGHT * 1.95)
        self.play(FadeIn(plus), run_time=0.3)
        res = VGroup(*[row_of([str(i + j) for j in range(4)], SUCCESS, w=w, h=h) for i in range(3)]).arrange(DOWN, buff=0)
        res.move_to(origin + RIGHT * 3.9 + DOWN * 2.3)
        eq = label("=", size=36).next_to(res, LEFT, buff=0.3)
        self.play(FadeIn(eq), LaggedStart(*[FadeIn(c) for r in res for c in r], lag_ratio=0.06), run_time=1.2)
        self.say("Added cell by cell: [[0 1 2 3] [1 2 3 4] [2 3 4 5]]", color=SUCCESS)
        self.wait(2.5)


class AxisAggregation(PyScene):
    def construct(self):
        self.add(title("axis = the dimension that disappears"))
        vals = [[3, 7, 2], [9, 1, 5]]
        w, h = 0.9, 0.7
        M = VGroup(*[row_of([str(v) for v in r], INFO, w=w, h=h, size=24) for r in vals]).arrange(DOWN, buff=0).move_to([0, 1.1, 0])
        ml = label("M  shape (2, 3)", size=22, mono=True, color=MUTED).next_to(M, UP, buff=0.2)
        self.play(FadeIn(M), FadeIn(ml), run_time=0.6)
        self.wait(0.4)

        self.code("M.sum(axis=0)   → shape (3,)", pos=[-3.5, -0.9, 0])
        top = M[0].copy()
        bottom = M[1].copy()
        self.play(top.animate.move_to([-3.5, -1.9, 0]), bottom.animate.move_to([-3.5, -1.9, 0]), run_time=0.9)
        r0 = row_of(["12", "8", "7"], SUCCESS, w=w, h=h, size=24).move_to([-3.5, -1.9, 0])
        self.play(FadeOut(top), FadeOut(bottom), FadeIn(r0), run_time=0.5)
        self.say("axis=0: rows collapse together, one total per column")
        self.wait(0.8)

        c1 = label("M.sum(axis=1)   → shape (2,)", size=26, mono=True, color=HIGHLIGHT).move_to([3.5, -0.9, 0])
        self.play(FadeIn(c1), run_time=0.3)
        parts = VGroup(*[c.copy() for r in M for c in r])
        targets = [[3.5, -1.7, 0]] * 3 + [[3.5, -2.4, 0]] * 3
        self.play(*[p.animate.move_to(t) for p, t in zip(parts, targets)], run_time=0.9)
        r1 = col_of(["12", "15"], PURPLE, w=w, h=h, size=24).move_to([3.5, -2.05, 0])
        self.play(FadeOut(parts), FadeIn(r1), run_time=0.5)
        self.say("axis=1: columns collapse within each row, one total per row")
        self.wait(0.6)
        self.say("keepdims=True keeps that axis as size 1: shape (2, 1), ready to broadcast back", color=HIGHLIGHT)
        self.wait(2.5)


# ============================================================================
# pandas page 3
# ============================================================================
class IndexAlignment(PyScene):
    def construct(self):
        self.add(title("Series arithmetic aligns on labels"))
        names_a, vals_a = ["Alex", "Jordan", "Morgan"], ["10", "20", "30"]
        names_b, vals_b = ["Jordan", "Morgan", "Riley"], ["1", "2", "3"]

        def series(names, vals, color, y):
            g = VGroup(*[VGroup(cell(n, MUTED, w=1.6, h=0.55, size=18, fill=0.0), cell(v, color, w=0.9, h=0.55, size=20)).arrange(RIGHT, buff=0) for n, v in zip(names, vals)]).arrange(DOWN, buff=0)
            return g.move_to([-3.8, y, 0])

        a = series(names_a, vals_a, INFO, 1.0)
        b = series(names_b, vals_b, PURPLE, -1.4)
        self.play(FadeIn(a), FadeIn(b), FadeIn(label("a", size=26, color=INFO).next_to(a, LEFT)), FadeIn(label("b", size=26, color=PURPLE).next_to(b, LEFT)), run_time=0.6)
        self.code("a + b", pos=[3.2, 1.7, 0])
        self.say("Step 1: take the union of labels and match them by name, not position")
        self.wait(0.5)

        union = ["Alex", "Jordan", "Morgan", "Riley"]
        res_vals = ["NaN", "21.0", "32.0", "NaN"]
        res = VGroup(*[VGroup(cell(n, MUTED, w=1.6, h=0.55, size=18, fill=0.0), cell("", FG, w=1.1, h=0.55, size=20)).arrange(RIGHT, buff=0) for n in union]).arrange(DOWN, buff=0).move_to([3.2, -0.2, 0])
        self.play(FadeIn(res), run_time=0.5)
        links = []
        for i, n in enumerate(union):
            for g, names in [(a, names_a), (b, names_b)]:
                if n in names:
                    j = names.index(n)
                    links.append(arrow(g[j].get_right(), res[i].get_left(), MUTED, width=1.5))
        self.play(LaggedStart(*[GrowArrow(l) for l in links], lag_ratio=0.1), run_time=1.0)
        for i, v in enumerate(res_vals):
            col = ERROR if v == "NaN" else SUCCESS
            self.play(retext(res[i][1], v, color=col, size=20), run_time=0.25)
        self.say("Jordan and Morgan exist on both sides; Alex and Riley only on one → NaN")
        self.wait(0.8)

        self.play(FadeOut(VGroup(*links)), run_time=0.3)
        self.code("a.add(b, fill_value=0)", pos=[3.2, 1.7, 0])
        for i, v in enumerate(["10.0", "21.0", "32.0", "3.0"]):
            self.play(retext(res[i][1], v, color=SUCCESS, size=20), run_time=0.25)
        self.say("fill_value=0 treats a missing side as 0 before adding", color=SUCCESS)
        self.wait(2.5)


class LocVsIloc(PyScene):
    def construct(self):
        self.add(title("loc counts labels, iloc counts positions"))
        labels_ = ["5", "2", "0", "4", "3", "1"]
        vals = list("ABCDEF")
        rows = VGroup(*[VGroup(cell(str(p), MUTED, w=0.9, h=0.55, size=18, fill=0.0),
                               cell(l, PURPLE, w=0.9, h=0.55, size=20),
                               cell(v, INFO, w=0.9, h=0.55, size=20)).arrange(RIGHT, buff=0)
                        for p, l, v in zip(range(6), labels_, vals)]).arrange(DOWN, buff=0).move_to([-3.5, -0.3, 0])
        hdr = VGroup(label("position", size=16, color=MUTED), label("index", size=16, color=PURPLE), label("v", size=16, color=INFO))
        for hd, c in zip(hdr, rows[0]):
            hd.next_to(c, UP, buff=0.15)
        self.play(FadeIn(rows), FadeIn(hdr), run_time=0.6)
        self.say("After a shuffle, labels no longer match positions")
        self.wait(0.6)

        self.code("df.loc[2:4]", pos=[3.2, 1.6, 0], color=PURPLE)
        box_loc = SurroundingRectangle(VGroup(rows[1], rows[3]), color=PURPLE, buff=0.05, stroke_width=4)
        self.play(Create(box_loc), run_time=0.6)
        r1 = label("labels 2 → 4, END INCLUDED:  B C D", size=22, mono=True, color=PURPLE).move_to([3.2, 0.9, 0])
        self.play(FadeIn(r1), run_time=0.4)
        self.say("loc walks from label 2 to label 4 in index order and includes label 4")
        self.wait(0.8)

        self.code("df.iloc[2:4]", pos=[3.2, -0.3, 0], color=INFO)
        box_iloc = SurroundingRectangle(VGroup(rows[2], rows[3]), color=INFO, buff=0.12, stroke_width=4)
        self.play(Create(box_iloc), run_time=0.6)
        r2 = label("positions 2, 3, END EXCLUDED:  C D", size=22, mono=True, color=INFO).move_to([3.2, -1.0, 0])
        self.play(FadeIn(r2), run_time=0.4)
        self.say("iloc counts positions like a Python list and stops before 4")
        self.wait(2.5)


# ============================================================================
# pandas page 4
# ============================================================================
class FfillBfill(PyScene):
    def construct(self):
        self.add(title("ffill must stop at group boundaries"))
        ids = ["S1"] * 4 + ["S2"] * 4
        vals = ["20.1", "NaN", "NaN", "23.0", "NaN", "31.0", "NaN", "33.0"]

        def column(values, x, color_fn):
            return VGroup(*[cell(v, color_fn(v), w=1.3, h=0.5, size=18) for v in values]).arrange(DOWN, buff=0).move_to([x, -0.4, 0])

        idc = VGroup(*[cell(i, PURPLE if i == "S1" else HIGHLIGHT, w=0.9, h=0.5, size=18) for i in ids]).arrange(DOWN, buff=0).move_to([-5.2, -0.4, 0])
        orig = column(vals, -3.6, lambda v: MUTED if v == "NaN" else INFO)
        hdr = VGroup(label("sensor_id", size=16, color=MUTED).next_to(idc, UP, buff=0.15), label("sensor_val", size=16, color=MUTED).next_to(orig, UP, buff=0.15))
        self.play(FadeIn(idc), FadeIn(orig), FadeIn(hdr), run_time=0.6)
        line = DashedLine([-5.8, -0.4, 0], [5.8, -0.4, 0], color=MUTED)
        self.play(Create(line), run_time=0.4)

        plain = ["20.1", "20.1", "20.1", "23.0", "23.0", "31.0", "31.0", "33.0"]
        pc = column(plain, -0.6, lambda v: INFO)
        pc[4].set_color(ERROR)
        h1 = label(".ffill()", size=20, mono=True, color=ERROR).next_to(pc, UP, buff=0.15)
        self.play(FadeIn(h1), run_time=0.3)
        self.play(LaggedStart(*[FadeIn(c) for c in pc], lag_ratio=0.12), run_time=1.2)
        leak = arrow(pc[3].get_right() + RIGHT * 0.1, pc[4].get_right() + RIGHT * 0.1, ERROR)
        self.play(GrowArrow(leak), Circumscribe(pc[4], color=ERROR), run_time=0.8)
        self.say("Plain ffill carries S1's last reading (23.0) into S2's first gap: leakage", color=ERROR)
        self.wait(0.8)

        grouped = ["20.1", "20.1", "20.1", "23.0", "NaN", "31.0", "31.0", "33.0"]
        gc = column(grouped, 2.4, lambda v: MUTED if v == "NaN" else SUCCESS)
        h2 = label("groupby(id).ffill()", size=20, mono=True, color=SUCCESS).next_to(gc, UP, buff=0.15)
        self.play(FadeIn(h2), LaggedStart(*[FadeIn(c) for c in gc], lag_ratio=0.12), run_time=1.3)
        self.say("Grouped ffill restarts per sensor: S2's first gap stays empty", color=SUCCESS)
        self.wait(0.8)

        back = ["20.1", "23.0", "23.0", "23.0", "31.0", "31.0", "33.0", "33.0"]
        bc = column(back, 5.0, lambda v: HIGHLIGHT)
        h3 = label(".bfill()", size=20, mono=True, color=HIGHLIGHT).next_to(bc, UP, buff=0.15)
        self.play(FadeIn(h3), LaggedStart(*[FadeIn(c) for c in reversed(bc)], lag_ratio=0.12), run_time=1.2)
        self.say("bfill copies the NEXT value backwards: it uses the future")
        self.wait(2.5)


class CategoryCodes(PyScene):
    def construct(self):
        self.add(title("category dtype: labels stored once, codes per row"))
        labels_ = ["Fashion", "Books", "Beauty", "Electronics", "Fashion", "Home", "Books", "Electronics"]
        codes = [3, 1, 0, 2, 3, 4, 1, 2]
        colors = {"Beauty": INFO, "Books": PURPLE, "Electronics": HIGHLIGHT, "Fashion": SUCCESS, "Home": GRAD}
        col = VGroup(*[cell(l, colors[l], w=2.3, h=0.5, size=18) for l in labels_]).arrange(DOWN, buff=0).move_to([-4.3, -0.3, 0])
        cl = label("str column: one string per row", size=18, color=MUTED).next_to(col, UP, buff=0.15)
        self.play(FadeIn(col), FadeIn(cl), run_time=0.6)
        self.wait(0.5)

        cats = ["Beauty", "Books", "Electronics", "Fashion", "Home"]
        table = VGroup(*[VGroup(cell(str(i), MUTED, w=0.6, h=0.5, size=18, fill=0.0), cell(c, colors[c], w=2.3, h=0.5, size=18)).arrange(RIGHT, buff=0) for i, c in enumerate(cats)]).arrange(DOWN, buff=0).move_to([3.6, 0.6, 0])
        tl = label(".cat.categories (stored once)", size=18, color=MUTED).next_to(table, UP, buff=0.15)
        self.play(FadeIn(table), FadeIn(tl), run_time=0.6)
        self.say("Step 1: the distinct labels go into a small lookup table")
        self.wait(0.6)

        codes_col = VGroup(*[cell(str(c), colors[cats[c]], w=0.8, h=0.5, size=20) for c in codes]).arrange(DOWN, buff=0).move_to([-0.6, -0.3, 0])
        kl = label(".cat.codes  int8", size=18, color=MUTED).next_to(codes_col, UP, buff=0.15)
        self.play(FadeIn(kl), *[TransformFromCopy(a, b) for a, b in zip(col, codes_col)], run_time=1.2)
        self.say("Step 2: each row keeps only a 1-byte code pointing into the table")
        self.wait(0.6)
        self.play(FadeOut(col), FadeOut(cl), run_time=0.4)
        mem = VGroup(label("ShopStream category column", size=20, color=MUTED),
                     label("str:       76,114 bytes", size=22, mono=True, color=FG),
                     label("category:   5,226 bytes", size=22, mono=True, color=SUCCESS)).arrange(DOWN, buff=0.12, aligned_edge=LEFT).move_to([3.6, -1.7, 0])
        self.play(FadeIn(mem), run_time=0.5)
        self.say("Comparisons and groupby work on the integer codes: smaller and faster")
        self.wait(2.5)


# ============================================================================
# pandas page 5
# ============================================================================
class SplitApplyCombine(PyScene):
    def construct(self):
        self.add(title("groupby: split → apply → combine"))
        cats = ["Beauty", "Books", "Electronics", "Fashion", "Home"]
        colors = [INFO, PURPLE, HIGHLIGHT, SUCCESS, GRAD]
        order = [2, 0, 3, 1, 2, 4, 3, 0, 2, 4, 1, 3, 2, 4, 0]
        rows = VGroup(*[Rectangle(width=2.2, height=0.28, stroke_width=0, fill_color=colors[k], fill_opacity=0.85) for k in order]).arrange(DOWN, buff=0.04).move_to([-5.0, -0.2, 0])
        rl = label("orders (4,946 rows)", size=18, color=MUTED).next_to(rows, UP, buff=0.15)
        self.play(FadeIn(rows), FadeIn(rl), run_time=0.6)
        self.say("Each row belongs to one category")
        self.wait(0.4)

        piles_x = [-2.8, -1.5, -0.2, 1.1, 2.4]
        self.say("Split: rows are grouped into one pile per key")
        counts = [0] * 5
        anims = []
        for r, k in zip(rows, order):
            target = np.array([piles_x[k], -2.0 + counts[k] * 0.34, 0])
            counts[k] += 1
            anims.append(r.animate.set_width(1.1).move_to(target))
        self.play(LaggedStart(*anims, lag_ratio=0.05), run_time=1.6)
        names = VGroup(*[label(c, size=16, color=colors[i]).move_to([piles_x[i], -2.5, 0]) for i, c in enumerate(cats)])
        self.play(FadeIn(names), run_time=0.4)
        self.wait(0.3)

        self.code("['revenue'].sum()", pos=[-0.2, 1.4, 0])
        totals = ["1.03", "0.49", "20.98", "3.62", "4.00"]
        sums = VGroup(*[cell(t, colors[i], w=1.1, h=0.5, size=18).move_to([piles_x[i], 0.4, 0]) for i, t in enumerate(totals)])
        self.play(LaggedStart(*[FadeIn(s, shift=UP * 0.3) for s in sums], lag_ratio=0.15), run_time=1.0)
        self.say("Apply: each pile is reduced to one number (₹ millions)")
        self.wait(0.5)

        table = VGroup(*[VGroup(cell(c, colors[i], w=1.9, h=0.48, size=17), cell(totals[i], colors[i], w=1.2, h=0.48, size=17)).arrange(RIGHT, buff=0) for i, c in enumerate(cats)]).arrange(DOWN, buff=0).move_to([5.2, -0.4, 0])
        self.play(*[TransformFromCopy(s, table[i]) for i, s in enumerate(sums)], run_time=1.0)
        self.say("Combine: the results are stacked into a 5-row table indexed by category", color=SUCCESS)
        self.wait(2.5)


class AggVsTransform(PyScene):
    def construct(self):
        self.add(title("agg collapses, transform broadcasts back"))
        gs = ["a", "b", "a", "b", "b"]
        vs = ["10", "5", "20", "15", "40"]
        col = {"a": INFO, "b": PURPLE}
        data = VGroup(*[VGroup(cell(g, col[g], w=0.8, h=0.55, size=20), cell(v, col[g], w=1.0, h=0.55, size=20)).arrange(RIGHT, buff=0) for g, v in zip(gs, vs)]).arrange(DOWN, buff=0).move_to([-4.6, -0.3, 0])
        dl = label("g   x", size=18, mono=True, color=MUTED).next_to(data, UP, buff=0.15)
        self.play(FadeIn(data), FadeIn(dl), run_time=0.6)
        means = label("mean(a) = (10 + 20) / 2 = 15    mean(b) = (5 + 15 + 40) / 3 = 20", size=20, mono=True, color=MUTED).move_to([1.4, 2.1, 0])
        self.play(FadeIn(means), run_time=0.5)
        self.wait(0.4)

        agg = VGroup(*[VGroup(cell(g, col[g], w=0.8, h=0.55, size=20), cell(m, SUCCESS, w=1.0, h=0.55, size=20)).arrange(RIGHT, buff=0) for g, m in [("a", "15"), ("b", "20")]]).arrange(DOWN, buff=0).move_to([-1.0, -0.3, 0])
        al = label(".groupby('g')['x'].mean()", size=18, mono=True, color=SUCCESS).next_to(agg, UP, buff=0.5)
        self.play(FadeIn(al), FadeIn(agg, shift=RIGHT * 0.3), run_time=0.6)
        self.say("agg: one row per group (2 rows), indexed by the group key")
        self.wait(0.6)

        tr_vals = ["15", "20", "15", "20", "20"]
        tr = VGroup(*[cell(t, HIGHLIGHT, w=1.0, h=0.55, size=20) for t in tr_vals]).arrange(DOWN, buff=0).move_to([3.8, -0.3, 0])
        tl = label(".transform('mean')", size=18, mono=True, color=HIGHLIGHT).next_to(tr, UP, buff=0.2)
        self.play(FadeIn(tl), run_time=0.3)
        links = VGroup(*[arrow(agg[0 if g == "a" else 1].get_right(), tr[i].get_left(), col[g], width=1.5) for i, g in enumerate(gs)])
        self.play(LaggedStart(*[GrowArrow(l) for l in links], lag_ratio=0.1), LaggedStart(*[FadeIn(t) for t in tr], lag_ratio=0.1), run_time=1.2)
        self.say("transform: the group's value is written next to EVERY original row (5 rows)")
        self.wait(0.6)
        ratio = label("x / mean → 0.67  0.25  1.33  0.75  2.00", size=20, mono=True, color=HIGHLIGHT).move_to([1.4, -2.4, 0])
        self.play(FadeOut(links), FadeIn(ratio), run_time=0.5)
        self.say("Same length as the input, so row-wise arithmetic just works", color=SUCCESS)
        self.wait(2.5)


# ============================================================================
# pandas page 6
# ============================================================================
class MeltPivot(PyScene):
    def construct(self):
        self.add(title("melt: wide → long   ·   pivot_table: long → wide"))
        cats = ["Beauty", "Books", "Electronics"]
        jan = ["0.13", "0.10", "3.43"]
        feb = ["0.14", "0.07", "3.58"]
        hdr = row_of(["category", "Jan", "Feb"], MUTED, w=1.7, h=0.5, size=17)
        body = VGroup(*[row_of([c, j, f], INFO, w=1.7, h=0.5, size=17) for c, j, f in zip(cats, jan, feb)]).arrange(DOWN, buff=0)
        wide = VGroup(hdr, body).arrange(DOWN, buff=0).move_to([-3.6, 0.6, 0])
        body[0][2].set_color(PURPLE); body[1][2].set_color(PURPLE); body[2][2].set_color(PURPLE)
        wl = label("wide: 3 rows x 2 month columns (₹M)", size=17, color=MUTED).next_to(wide, UP, buff=0.15)
        self.play(FadeIn(wide), FadeIn(wl), run_time=0.6)
        self.code("wide.melt(id_vars='category', var_name='month')", pos=[0, -2.0, 0])
        self.wait(0.4)

        lhdr = row_of(["category", "month", "value"], MUTED, w=1.5, h=0.45, size=16)
        long_rows = [(c, "Jan", v) for c, v in zip(cats, jan)] + [(c, "Feb", v) for c, v in zip(cats, feb)]
        lbody = VGroup(*[row_of(list(r), INFO if r[1] == "Jan" else PURPLE, w=1.5, h=0.45, size=16) for r in long_rows]).arrange(DOWN, buff=0)
        longt = VGroup(lhdr, lbody).arrange(DOWN, buff=0).move_to([3.4, 0.4, 0])
        self.play(FadeIn(lhdr), run_time=0.3)
        self.play(Circumscribe(VGroup(*[body[i][1] for i in range(3)]), color=INFO), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(lbody[i], shift=RIGHT * 0.4) for i in range(3)], lag_ratio=0.2), run_time=0.8)
        self.say("The Jan column is peeled off: one row per (category, Jan)")
        self.wait(0.3)
        self.play(Circumscribe(VGroup(*[body[i][2] for i in range(3)]), color=PURPLE), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(lbody[3 + i], shift=RIGHT * 0.4) for i in range(3)], lag_ratio=0.2), run_time=0.8)
        self.say("Then Feb is stacked underneath: 3 x 2 = 6 long rows")
        self.wait(0.6)

        pv = label("long.pivot_table(index='category', columns='month', aggfunc='sum')", size=19, mono=True, color=HIGHLIGHT).move_to([0, -2.0, 0])
        self.play(FadeOut(self._code, run_time=0.15), FadeIn(pv), run_time=0.3)
        self._code = pv
        self.play(Indicate(wide, color=HIGHLIGHT, scale_factor=1.03), run_time=0.8)
        self.say("pivot_table goes back: rows by category, one column per month", color=SUCCESS)
        self.wait(2.5)


class MergeJoins(PyScene):
    def construct(self):
        self.add(title("merge on customer_id: inner, left, and a duplicate key"))
        orders = [("O1", "C0001"), ("O2", "C0002"), ("O3", "C9999"), ("O4", "C0001")]
        ocol = {"C0001": INFO, "C0002": PURPLE, "C9999": ERROR}
        otab = VGroup(*[row_of([o, c], ocol[c], w=1.2, h=0.5, size=17) for o, c in orders]).arrange(DOWN, buff=0).move_to([-4.8, 0.5, 0])
        ctab = VGroup(*[row_of([c, t], ocol[c], w=1.2, h=0.5, size=17) for c, t in [("C0001", "Silver"), ("C0002", "Bronze")]]).arrange(DOWN, buff=0).move_to([-1.6, 0.8, 0])
        self.play(FadeIn(otab), FadeIn(ctab), FadeIn(label("orders", size=18, color=MUTED).next_to(otab, UP, buff=0.15)),
                  FadeIn(label("customers", size=18, color=MUTED).next_to(ctab, UP, buff=0.15)), run_time=0.6)
        links = VGroup(*[arrow(otab[i].get_right(), ctab[0 if c == "C0001" else 1].get_left(), ocol[c], width=1.5) for i, (_, c) in enumerate(orders) if c != "C9999"])
        self.play(LaggedStart(*[GrowArrow(l) for l in links], lag_ratio=0.15), run_time=0.8)
        self.say("C9999 has no partner in customers")
        self.wait(0.4)

        def result(rows, x):
            return VGroup(*[row_of(list(r), ocol.get(r[1], FG), w=1.05, h=0.45, size=15) for r in rows]).arrange(DOWN, buff=0).move_to([x, 0.3, 0])

        inner = result([("O1", "C0001", "Silver"), ("O2", "C0002", "Bronze"), ("O4", "C0001", "Silver")], 1.7)
        il = label("how='inner': 3 rows", size=18, color=MUTED).next_to(inner, UP, buff=0.15)
        self.play(FadeIn(inner), FadeIn(il), run_time=0.6)
        left = result([("O1", "C0001", "Silver"), ("O2", "C0002", "Bronze"), ("O3", "C9999", "NaN"), ("O4", "C0001", "Silver")], 5.0)
        ll = label("how='left': 4 rows", size=18, color=MUTED).next_to(left, UP, buff=0.15)
        self.play(FadeIn(left), FadeIn(ll), run_time=0.6)
        self.say("inner drops the orphan; left keeps it with NaN (ShopStream: 4,846 vs 4,946 rows)")
        self.wait(1.0)

        dup = row_of(["C0001", "Silver"], ERROR, w=1.2, h=0.5, size=17).next_to(ctab, DOWN, buff=0)
        self.play(FadeIn(dup, shift=DOWN * 0.2), FadeOut(inner), FadeOut(il), FadeOut(left), FadeOut(ll), run_time=0.6)
        exploded = result([("O1", "C0001", "Silver"), ("O1", "C0001", "Silver"), ("O2", "C0002", "Bronze"),
                           ("O3", "C9999", "NaN"), ("O4", "C0001", "Silver"), ("O4", "C0001", "Silver")], 3.3)
        for i in [1, 5]:
            exploded[i].set_color(ERROR)
        self.play(FadeIn(exploded), run_time=0.6)
        self.say("A duplicated customer doubles its orders: 4 → 6 rows. validate='m:1' raises instead", color=ERROR)
        self.wait(2.5)


# ============================================================================
# Matplotlib page 7
# ============================================================================
class MplAnatomy(PyScene):
    def construct(self):
        self.add(title("A Matplotlib chart, layer by layer"))
        fig = Rectangle(width=10, height=5.6, color=MUTED, stroke_width=2).move_to([0, -0.4, 0])
        fl = label("Figure", size=20, color=MUTED).next_to(fig, UP, buff=0.05).align_to(fig, LEFT)
        self.play(Create(fig), FadeIn(fl), run_time=0.6)
        self.say("fig, ax = plt.subplots()  creates the Figure (the canvas)...")
        ax_rect = Rectangle(width=7.6, height=3.8, color=INFO, stroke_width=2).move_to([0.3, -0.5, 0])
        al = label("Axes", size=20, color=INFO).next_to(ax_rect, UP, buff=0.05).align_to(ax_rect, RIGHT)
        self.play(Create(ax_rect), FadeIn(al), run_time=0.6)
        self.say("...and one Axes: the plot area with its own coordinate system")
        self.wait(0.3)

        x0, y0, _ = ax_rect.get_corner(DL)
        x1, y1, _ = ax_rect.get_corner(UR)
        xaxis = Line([x0, y0, 0], [x1, y0, 0], color=FG, stroke_width=3)
        yaxis = Line([x0, y0, 0], [x0, y1, 0], color=FG, stroke_width=3)
        xt = VGroup(*[Line([x0 + k * (x1 - x0) / 6, y0, 0], [x0 + k * (x1 - x0) / 6, y0 - 0.15, 0], color=FG) for k in range(1, 6)])
        yt = VGroup(*[Line([x0, y0 + k * (y1 - y0) / 4, 0], [x0 - 0.15, y0 + k * (y1 - y0) / 4, 0], color=FG) for k in range(1, 4)])
        self.play(Create(xaxis), Create(yaxis), Create(xt), Create(yt), run_time=0.8)
        self.say("Two Axis objects: spines, ticks (Locator) and tick labels (Formatter)")
        self.wait(0.3)

        ys = [0.45, 0.55, 0.42, 0.6, 0.7, 0.5, 0.8, 0.65, 0.75]
        pts = [[x0 + (i + 0.5) * (x1 - x0) / len(ys), y0 + v * (y1 - y0), 0] for i, v in enumerate(ys)]
        line = VMobject(color=INFO, stroke_width=4).set_points_as_corners(pts)
        self.play(Create(line), run_time=1.0)
        self.say("ax.plot(...) adds a Line2D artist")
        t = label("ax.set_title(...)", size=22, weight=BOLD).move_to([0.3, y1 + 0.35, 0])
        xl = label("ax.set_xlabel(...)", size=18, color=MUTED).move_to([0.3, y0 - 0.55, 0])
        yl = label("ax.set_ylabel(...)", size=18, color=MUTED).rotate(PI / 2).move_to([x0 - 0.6, (y0 + y1) / 2, 0])
        leg = VGroup(Line(ORIGIN, RIGHT * 0.5, color=INFO, stroke_width=4), label("label=...", size=16)).arrange(RIGHT, buff=0.15).move_to([x1 - 1.2, y0 + 0.5, 0])
        self.play(FadeIn(t), FadeIn(xl), FadeIn(yl), run_time=0.7)
        self.play(FadeIn(leg), run_time=0.4)
        self.say("Title, labels and legend are Text artists owned by the Axes", color=SUCCESS)
        self.wait(2.5)


class HistogramBinning(PyScene):
    def construct(self):
        o = shopstream_clean()
        p = o["unit_price"].to_numpy()
        eq_counts, _ = np.histogram(p, bins=12)
        log_counts, _ = np.histogram(p, bins=np.logspace(np.log10(p.min()), np.log10(p.max()), 12 + 1))
        self.add(title("Same prices, different bins, different story"))

        def bars(counts, x_center, color, caption):
            mx = counts.max()
            g = VGroup(*[Rectangle(width=0.38, height=max(0.02, 3.2 * c / mx), stroke_width=0, fill_color=color, fill_opacity=0.9) for c in counts]).arrange(RIGHT, buff=0.04, aligned_edge=DOWN)
            g.move_to([x_center, -2.0, 0], aligned_edge=DOWN)
            base = Line(g.get_corner(DL) + LEFT * 0.1, g.get_corner(DR) + RIGHT * 0.1, color=MUTED)
            cap = label(caption, size=18, color=MUTED).next_to(base, DOWN, buff=0.15)
            return g, base, cap

        g1, b1, c1 = bars(eq_counts, -3.3, INFO, "12 equal-width bins (₹)")
        self.play(Create(b1), FadeIn(c1), run_time=0.4)
        self.play(LaggedStart(*[GrowFromEdge(r, DOWN) for r in g1], lag_ratio=0.08), run_time=1.2)
        share = eq_counts[0] / eq_counts.sum()
        self.say(f"Equal widths: the first bar alone holds {share:.0%} of orders ({eq_counts[0]:,})")
        self.wait(2.0)

        g2, b2, c2 = bars(log_counts, 3.3, SUCCESS, "12 log-spaced bins (log axis)")
        self.play(Create(b2), FadeIn(c2), run_time=0.4)
        self.play(LaggedStart(*[GrowFromEdge(r, DOWN) for r in g2], lag_ratio=0.08), run_time=1.2)
        self.say("Log-spaced bins reveal the shape: everyday items and an Electronics hump", color=SUCCESS)
        self.wait(2.5)


# ============================================================================
# Seaborn page 8
# ============================================================================
class KdeBumps(PyScene):
    def construct(self):
        self.add(title("A KDE is a bump per point, summed"))
        pts = np.array([2, 3, 3, 4, 4, 4, 6, 9])           # 8 delivery-day values
        axes = Axes(x_range=[0, 12, 1], y_range=[0, 0.8, 0.1], x_length=10, y_length=4,
                    axis_config={"color": MUTED, "include_ticks": True}).move_to([0, -0.5, 0])
        self.play(Create(axes), run_time=0.6)
        dots = VGroup(*[Dot(axes.c2p(x, 0), color=HIGHLIGHT, radius=0.08).shift(UP * 0.12 * (list(pts[:i]).count(x))) for i, x in enumerate(pts)])
        self.play(LaggedStart(*[FadeIn(d, shift=DOWN * 0.3) for d in dots], lag_ratio=0.1), run_time=0.8)
        self.say("8 observations of delivery days")
        self.wait(0.3)

        def gauss(x, mu, h):
            return np.exp(-0.5 * ((x - mu) / h) ** 2) / (h * np.sqrt(2 * np.pi))

        def kde_curve(h, color):
            return axes.plot(lambda x: sum(gauss(x, m, h) for m in pts) / len(pts), x_range=[0, 12, 0.02], color=color, stroke_width=4)

        h = 0.8
        bumps = VGroup(*[axes.plot(lambda x, m=m: gauss(x, m, h) / len(pts), x_range=[0, 12, 0.02], color=INFO, stroke_width=2) for m in pts])
        self.play(LaggedStart(*[Create(b) for b in bumps], lag_ratio=0.1), run_time=1.2)
        self.say("Each point gets a small bell curve of width h = 0.8 (area 1/8 each)")
        self.wait(0.5)
        total = kde_curve(h, SUCCESS)
        self.play(ReplacementTransform(bumps.copy(), total), run_time=1.0)
        self.say("Adding them up gives the density estimate (total area = 1)", color=SUCCESS)
        self.wait(0.6)

        narrow = kde_curve(0.2, ERROR)
        self.play(Transform(total, narrow), FadeOut(bumps), run_time=0.8)
        self.say("Small bandwidth (0.2): a spike at every value", color=ERROR)
        self.wait(0.6)
        wide = kde_curve(2.5, PURPLE)
        self.play(Transform(total, wide), run_time=0.8)
        self.say("Large bandwidth (2.5): real structure is blurred away", color=PURPLE)
        self.wait(2.5)


class BoxplotQuartiles(PyScene):
    def construct(self):
        o = shopstream_clean()
        all_books = o.loc[o["category"] == "Books", "unit_price"]
        books = np.append(all_books.sample(15, random_state=7).to_numpy(), all_books.quantile(0.995))   # 15 random + one real high price
        vals = np.sort(books.round(0))
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        iqr = q3 - q1
        lo_w = vals[vals >= q1 - 1.5 * iqr].min()
        hi_w = vals[vals <= q3 + 1.5 * iqr].max()
        self.add(title("Reading a box plot: 16 Books prices"))

        lo, hi = 0, float(np.ceil(vals.max() / 250) * 250)
        nl = NumberLine(x_range=[lo, hi, 250], length=11, color=MUTED, include_numbers=False).move_to([0, -2.2, 0])
        ticks = VGroup(*[label(f"{int(v)}", size=14, color=MUTED).next_to(nl.n2p(v), DOWN, buff=0.15) for v in np.arange(lo, hi + 1, 250)])
        self.play(Create(nl), FadeIn(ticks), run_time=0.5)
        dots = VGroup(*[Dot(nl.n2p(v) + UP * 0.35, color=INFO, radius=0.07) for v in vals])
        self.play(LaggedStart(*[FadeIn(d, shift=DOWN * 0.2) for d in dots], lag_ratio=0.06), run_time=1.0)
        self.say("Sorted prices (₹) of 16 real Books orders: 15 random plus one from the top 0.5%")
        self.wait(0.4)

        y = 0.6
        box_ = Rectangle(width=nl.n2p(q3)[0] - nl.n2p(q1)[0], height=1.0, color=SUCCESS, stroke_width=3).move_to([(nl.n2p(q1)[0] + nl.n2p(q3)[0]) / 2, y, 0])
        med_l = Line([nl.n2p(med)[0], y - 0.5, 0], [nl.n2p(med)[0], y + 0.5, 0], color=HIGHLIGHT, stroke_width=5)
        self.play(Create(box_), run_time=0.7)
        self.say(f"Box = middle half: Q1 ₹{q1:,.0f} to Q3 ₹{q3:,.0f} (IQR ₹{iqr:,.0f})")
        self.wait(0.6)
        self.play(Create(med_l), run_time=0.4)
        self.say(f"Line = median ₹{med:,.0f}", color=HIGHLIGHT)
        self.wait(0.5)
        wl = Line([nl.n2p(lo_w)[0], y, 0], box_.get_left(), color=SUCCESS, stroke_width=3)
        wr = Line(box_.get_right(), [nl.n2p(hi_w)[0], y, 0], color=SUCCESS, stroke_width=3)
        self.play(Create(wl), Create(wr), run_time=0.6)
        self.say("Whiskers reach the furthest points within 1.5 x IQR of the box")
        self.wait(0.5)
        out = vals[vals > q3 + 1.5 * iqr]
        flags = VGroup(*[Circle(radius=0.14, color=ERROR, stroke_width=3).move_to([nl.n2p(v)[0], y, 0]) for v in out])
        self.play(FadeIn(flags), *[dots[list(vals).index(v)].animate.set_color(ERROR) for v in out], run_time=0.6)
        self.say(f"Beyond the fence (₹{q3 + 1.5 * iqr:,.0f}): drawn as separate outlier dots", color=ERROR)
        self.wait(2.5)


# ============================================================================
# Capstone page 9
# ============================================================================
class CleaningFunnel(PyScene):
    def construct(self):
        self.add(title("The cleaning log: every lost row has a reason"))
        steps = [("raw rows", 5020, None), ("drop exact duplicates", 5000, 20),
                 ("drop impossible quantity", 4958, 42), ("drop extreme prices (per category)", 4934, 24)]
        scale = 9.0 / 5020
        bars = VGroup()
        for i, (name, n, removed) in enumerate(steps):
            y = 1.6 - i * 1.15
            bar = Rectangle(width=n * scale, height=0.6, stroke_width=0, fill_color=INFO if i < 3 else SUCCESS, fill_opacity=0.85)
            bar.move_to([-4.6, y, 0], aligned_edge=LEFT)
            nm = label(name, size=18).next_to(bar, UP, buff=0.08).align_to(bar, LEFT)
            cnt = label(f"{n:,}", size=20, mono=True).move_to(bar)
            grp = VGroup(bar, nm, cnt)
            bars.add(grp)
            self.play(GrowFromEdge(bar, LEFT), FadeIn(nm), FadeIn(cnt), run_time=0.6)
            if removed:
                gone = label(f"-{removed}", size=22, mono=True, color=ERROR).next_to(bar, RIGHT, buff=0.3)
                self.play(FadeIn(gone, shift=RIGHT * 0.2), run_time=0.4)
            self.wait(0.7)
        self.say("5,020 → 5,000 → 4,958 → 4,934: 86 rows removed, each by a named rule", color=SUCCESS)
        self.wait(2.5)


class GridspecAssembly(PyScene):
    def construct(self):
        self.add(title("GridSpec(3, 4): slice a grid into panels"))
        W, H = 2.6, [0.8, 1.8, 1.8]
        x0, ytop = -5.2, 2.1
        cells = {}
        g = VGroup()
        y = ytop
        for r in range(3):
            for c in range(4):
                rect = Rectangle(width=W, height=H[r], color=MUTED, stroke_width=1.5).move_to([x0 + c * W + W / 2, y - H[r] / 2, 0])
                cells[(r, c)] = rect
                g.add(rect)
            y -= H[r]
        self.play(Create(g), run_time=0.8)
        self.say("height_ratios=[0.35, 1, 1]: a short KPI row and two chart rows")
        self.wait(0.5)

        def panel(r, c0, c1, text, color):
            left = cells[(r, c0)].get_left()[0]
            right = cells[(r, c1 - 1)].get_right()[0]
            cy = cells[(r, c0)].get_center()[1]
            under = Rectangle(width=right - left - 0.04, height=H[r] - 0.04, stroke_width=0,
                              fill_color=BG, fill_opacity=1).move_to([(left + right) / 2, cy, 0])   # hides the cell lines
            rect = Rectangle(width=right - left - 0.08, height=H[r] - 0.08, color=color, stroke_width=3,
                             fill_color=color, fill_opacity=0.12).move_to([(left + right) / 2, cy, 0])
            t = label(text, size=16, mono=True).move_to(rect)
            if t.width > rect.width - 0.15:
                t.scale_to_fit_width(rect.width - 0.15)
            return VGroup(under, rect, t)

        kpis = VGroup(*[panel(0, c, c + 1, f"gs[0, {c}] KPI", HIGHLIGHT) for c in range(4)])
        self.play(LaggedStart(*[FadeIn(k) for k in kpis], lag_ratio=0.1), run_time=0.7)
        self.say("Row 0: four single-cell Axes with ax.axis('off') and ax.text")
        self.wait(0.4)
        p1 = panel(1, 0, 2, "gs[1, :2]  ax.plot", INFO)
        p2 = panel(1, 2, 4, "gs[1, 2:]  ax.barh", INFO)
        self.play(FadeIn(p1), FadeIn(p2), run_time=0.7)
        self.say("Slices merge cells: gs[1, :2] is one Axes two columns wide (Matplotlib)")
        self.wait(0.5)
        p3 = panel(2, 0, 2, "gs[2, :2]  sns.lineplot(ax=ax)", PURPLE)
        p4 = panel(2, 2, 3, "sns.boxplot", PURPLE)
        p5 = panel(2, 3, 4, "sns.barplot", PURPLE)
        self.play(FadeIn(p3), FadeIn(p4), FadeIn(p5), run_time=0.7)
        self.say("Seaborn axes-level functions draw into the Axes you pass with ax=", color=SUCCESS)
        self.wait(2.5)
