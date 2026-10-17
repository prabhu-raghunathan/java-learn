"""Animations for docs/03-ai-ml/00-python-fundamentals/*.md

Every number shown comes from the verified runs quoted on the pages (CPython 3.14.6).
Render with:  python animations/render.py python/<name>   (keys in MANIFEST)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from style import *


# ----------------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------------
def box(text, color=FG, width=None, height=0.75, size=24, mono=True, fill_opacity=0.12):
    t = label(text, size=size, mono=mono)
    w = width if width is not None else max(t.width + 0.5, 1.0)
    r = RoundedRectangle(corner_radius=0.12, width=w, height=height, stroke_color=color, stroke_width=2.5)
    r.set_fill(color, opacity=fill_opacity)
    t.move_to(r)
    return VGroup(r, t)


def retext(b, text, color=None, size=24, mono=True):
    """Animation replacing the text inside a box() group."""
    new = label(text, size=size, mono=mono, color=color or FG).move_to(b[0])
    if new.width > b[0].width - 0.2:
        new.scale_to_fit_width(b[0].width - 0.2)
    return Transform(b[1], new)


def code_block(lines, size=23, buff=0.16):
    """Monospace lines that keep their leading indentation (Text strips spaces)."""
    cw = label("M" * 10, size=size, mono=True).width / 10
    g = VGroup(*[label(t.strip() or " ", size=size, mono=True) for t in lines]).arrange(DOWN, aligned_edge=LEFT, buff=buff)
    for mob, t in zip(g, lines):
        mob.shift(RIGHT * cw * (len(t) - len(t.lstrip(" "))))
    return g


def arrow(a, b, color=FG, buff=0.1, width=4):
    return Arrow(a, b, color=color, buff=buff, stroke_width=width, max_tip_length_to_length_ratio=0.12)


class PyScene(WikiScene):
    """WikiScene plus a caption that can be swapped in place."""

    def say(self, text, size=24, color=FG):
        new = caption_bar(text, size=size, color=color)
        old = getattr(self, "_cap", None)
        if old is None:
            self.play(FadeIn(new), run_time=0.35)
        else:
            self.play(FadeOut(old, run_time=0.2), FadeIn(new), run_time=0.35)
        self._cap = new

    def code(self, text, pos=UP * 2.55, color=HIGHLIGHT):
        """Show the statement being executed (swapped in place)."""
        new = label(text, size=26, mono=True, color=color).move_to(pos)
        old = getattr(self, "_code", None)
        if old is None:
            self.play(FadeIn(new), run_time=0.3)
        else:
            self.play(FadeOut(old, run_time=0.15), FadeIn(new), run_time=0.3)
        self._code = new


# ----------------------------------------------------------------------------
# 01  names point at objects
# ----------------------------------------------------------------------------
class NamesObjects(PyScene):
    def construct(self):
        self.add(title("Names are labels: assignment never copies"))
        a = box("a", INFO, width=1.0).move_to([-4.8, 0.7, 0])
        b = box("b", PURPLE, width=1.0).move_to([-4.8, -1.0, 0])
        lst = box("[1, 2]", SUCCESS, width=3.4, height=0.9, size=28).move_to([1.8, 0.7, 0])
        id_lbl = label("list object  id 0x...a0", size=18, color=MUTED).next_to(lst, UP, buff=0.12)

        self.code("a = [1, 2]")
        self.play(FadeIn(a), FadeIn(lst), FadeIn(id_lbl), run_time=0.6)
        a_ar = arrow(a.get_right(), lst.get_left(), INFO)
        self.play(GrowArrow(a_ar), run_time=0.5)
        self.say("a is a name bound to a list object")
        self.wait(0.6)

        self.code("b = a")
        self.play(FadeIn(b), run_time=0.4)
        b_ar = arrow(b.get_right(), lst.get_left() + DOWN * 0.2, PURPLE)
        self.play(GrowArrow(b_ar), run_time=0.5)
        self.say("b = a binds a SECOND name to the SAME object (nothing is copied)")
        self.wait(0.8)

        self.code("b.append(3)")
        self.play(retext(lst, "[1, 2, 3]", size=28), run_time=0.5)
        self.play(Circumscribe(lst, color=HIGHLIGHT), run_time=0.7)
        self.say("Mutation through b is visible through a: print(a) → [1, 2, 3]")
        self.wait(1.0)

        self.code("b = b + [4]")
        new = box("[1, 2, 3, 4]", HIGHLIGHT, width=3.4, height=0.9, size=28).move_to([1.8, -1.6, 0])
        new_lbl = label("NEW list object", size=18, color=MUTED).next_to(new, DOWN, buff=0.12)
        self.play(FadeIn(new, shift=DOWN * 0.2), FadeIn(new_lbl), run_time=0.6)
        nb = arrow(b.get_right(), new.get_left(), PURPLE)
        self.play(Transform(b_ar, nb), run_time=0.7)
        self.say("+ builds a new list and only b is rebound; a still points at the old one")
        self.wait(0.8)
        verdict = label("a → [1, 2, 3]     b → [1, 2, 3, 4]     a is b → False", size=24, mono=True,
                        color=SUCCESS).move_to(DOWN * 2.75)
        self.play(FadeIn(verdict), run_time=0.5)
        self.wait(2.5)


# ----------------------------------------------------------------------------
# 01  reference counting
# ----------------------------------------------------------------------------
class RefCount(PyScene):
    def construct(self):
        self.add(title("Reference counting: freed the moment the count hits 0"))
        obj = box("[0.5, 0.25]", SUCCESS, width=3.2, height=1.0, size=28).move_to([2.6, 0.2, 0])
        badge = box("refcount 1", HIGHLIGHT, width=2.6, height=0.6, size=22).next_to(obj, UP, buff=0.25)

        def name_box(text, y, color):
            return box(text, color, width=2.6, height=0.65, size=22).move_to([-4.2, y, 0])

        data = name_box("data", 1.5, INFO)
        alias = name_box("alias", 0.3, PURPLE)
        batch = name_box('batch["x"]', -0.9, GRAD)

        self.code("data = [0.5, 0.25]", pos=UP * 2.6)
        self.play(FadeIn(obj), FadeIn(data), FadeIn(badge), run_time=0.6)
        ar_data = arrow(data.get_right(), obj.get_left() + UP * 0.25, INFO)
        self.play(GrowArrow(ar_data), run_time=0.4)
        self.wait(0.4)

        self.code("alias = data")
        self.play(FadeIn(alias), run_time=0.3)
        ar_alias = arrow(alias.get_right(), obj.get_left(), PURPLE)
        self.play(GrowArrow(ar_alias), retext(badge, "refcount 2", size=22), run_time=0.5)
        self.wait(0.4)

        self.code('batch = {"x": data}')
        self.play(FadeIn(batch), run_time=0.3)
        ar_batch = arrow(batch.get_right(), obj.get_left() + DOWN * 0.25, GRAD)
        self.play(GrowArrow(ar_batch), retext(badge, "refcount 3", size=22), run_time=0.5)
        self.say("Every name or container slot that refers to it adds 1")
        self.wait(0.8)

        self.code("del alias")
        self.play(FadeOut(ar_alias), FadeOut(alias), retext(badge, "refcount 2", size=22), run_time=0.6)
        self.code("batch.clear()")
        self.play(FadeOut(ar_batch), FadeOut(batch), retext(badge, "refcount 1", size=22), run_time=0.6)
        self.say("del and clear() remove references: the count goes down")
        self.wait(0.6)

        self.code("del data")
        self.play(FadeOut(ar_data), FadeOut(data), retext(badge, "refcount 0", color=ERROR, size=22), run_time=0.6)
        self.play(obj[0].animate.set_stroke(ERROR).set_fill(ERROR, opacity=0.15), run_time=0.4)
        self.play(FadeOut(obj, scale=0.5), run_time=0.7)
        freed = label("freed immediately (no GC pass needed)", size=26, color=SUCCESS).move_to([2.6, 0.2, 0])
        self.play(FadeIn(freed), run_time=0.4)
        self.say("Reference cycles never reach 0: the cyclic GC handles those")
        self.wait(2.5)


# ----------------------------------------------------------------------------
# 02  short-circuit evaluation returns an operand
# ----------------------------------------------------------------------------
class ShortCircuit(PyScene):
    def chain(self, items, y, op):
        parts = VGroup()
        for i, t in enumerate(items):
            parts.add(box(t, MUTED, height=0.8, size=26))
            if i < len(items) - 1:
                parts.add(label(op, size=26, color=MUTED, mono=True))
        parts.arrange(RIGHT, buff=0.22).move_to([0, y, 0])
        return parts

    def construct(self):
        self.add(title("and / or stop early and return an OPERAND"))
        items = ['0', '""', '[]', '"x"', '1 / 0']
        row = self.chain(items, 1.3, "or")
        self.play(FadeIn(row), run_time=0.6)
        self.say("a or b: return the first TRUTHY value")
        boxes = [row[i] for i in range(0, len(row), 2)]
        verdicts = ["falsy", "falsy", "falsy", "truthy"]
        for bx, v in zip(boxes[:4], verdicts):
            color = ERROR if v == "falsy" else SUCCESS
            tag = label(v, size=20, color=color).next_to(bx, DOWN, buff=0.15)
            self.play(bx[0].animate.set_stroke(color).set_fill(color, opacity=0.18), FadeIn(tag), run_time=0.45)
        skip = label("never evaluated (no ZeroDivisionError)", size=20, color=MUTED).next_to(boxes[4], DOWN, buff=0.6).align_to(boxes[4], RIGHT)
        self.play(boxes[4].animate.set_opacity(0.3), FadeIn(skip), run_time=0.5)
        res = label('→  "x"', size=30, mono=True, color=SUCCESS).next_to(row, RIGHT, buff=0.3)
        if res.get_right()[0] > 6.8:
            res.next_to(row, UP, buff=0.35).align_to(row, RIGHT)
        self.play(FadeIn(res), run_time=0.4)
        self.wait(0.8)

        row2 = self.chain(["1", "2", "0", "3"], -1.3, "and")
        self.play(FadeIn(row2), run_time=0.5)
        self.say("a and b: return the first FALSY value (or the last one)")
        boxes2 = [row2[i] for i in range(0, len(row2), 2)]
        for bx, v in zip(boxes2[:3], ["truthy", "truthy", "falsy"]):
            color = ERROR if v == "falsy" else SUCCESS
            tag = label(v, size=20, color=color).next_to(bx, DOWN, buff=0.15)
            self.play(bx[0].animate.set_stroke(color).set_fill(color, opacity=0.18), FadeIn(tag), run_time=0.45)
        self.play(boxes2[3].animate.set_opacity(0.3), run_time=0.4)
        res2 = label("→  0", size=30, mono=True, color=ERROR).next_to(row2, RIGHT, buff=0.4)
        self.play(FadeIn(res2), run_time=0.4)
        self.say("So x or default replaces EVERY falsy x, including a deliberate 0")
        self.wait(2.5)


# ----------------------------------------------------------------------------
# 02  floor division vs truncation
# ----------------------------------------------------------------------------
class FloorDivision(PyScene):
    def construct(self):
        self.add(title("-7 // 2 == -4, while int(-7 / 2) == -3"))
        nl = NumberLine(x_range=[-5, 1, 1], length=11, include_numbers=False, color=MUTED).move_to(DOWN * 0.2)
        nums = VGroup(*[label(str(v), size=26, mono=True, color=MUTED).next_to(nl.n2p(v), DOWN, buff=0.25)
                        for v in range(-5, 2)])
        self.play(Create(nl), FadeIn(nums), run_time=0.8)
        p = Dot(nl.n2p(-3.5), color=HIGHLIGHT, radius=0.11)
        p_lbl = label("-7 / 2 = -3.5", size=24, mono=True, color=HIGHLIGHT).next_to(p, UP, buff=1.05)
        self.play(FadeIn(p, scale=1.5), FadeIn(p_lbl), run_time=0.6)
        self.say("True division gives -3.5: now round it to a whole number")
        self.wait(0.6)

        floor_dot = Dot(nl.n2p(-4), color=SUCCESS, radius=0.11)
        trunc_dot = Dot(nl.n2p(-3), color=GRAD, radius=0.11)
        f_ar = CurvedArrow(nl.n2p(-3.5) + UP * 0.18, nl.n2p(-4) + UP * 0.18, color=SUCCESS, angle=TAU / 4)
        t_ar = CurvedArrow(nl.n2p(-3.5) + UP * 0.18, nl.n2p(-3) + UP * 0.18, color=GRAD, angle=-TAU / 4)
        f_lbl = label("// floors toward -inf → -4", size=22, color=SUCCESS).move_to(nl.n2p(-4.2) + DOWN * 1.6)
        t_lbl = label("int() / Java truncate → -3", size=22, color=GRAD).move_to(nl.n2p(-1.6) + DOWN * 1.6)
        self.play(Create(f_ar), FadeIn(floor_dot), FadeIn(f_lbl), run_time=0.8)
        self.play(Create(t_ar), FadeIn(trunc_dot), FadeIn(t_lbl), run_time=0.8)
        self.say("Python floors (down); Java and int() truncate (toward zero)")
        self.wait(1.0)

        eq = VGroup(
            label("a == (a // b) * b + a % b", size=26, mono=True, color=FG),
            label("-7 == (-4) * 2 + 1", size=26, mono=True, color=SUCCESS),
            label("so  -7 % 2 == 1   (sign of the divisor)", size=24, color=SUCCESS),
        ).arrange(DOWN, buff=0.2).move_to(UP * 1.9)
        self.play(FadeOut(p_lbl), run_time=0.3)
        for line in eq:
            self.play(FadeIn(line, shift=DOWN * 0.15), run_time=0.45)
        self.say("The remainder fills the gap from -8 back up to -7")
        self.wait(2.5)


# ----------------------------------------------------------------------------
# 03  list over-allocation
# ----------------------------------------------------------------------------
class ListGrowth(PyScene):
    CAP_AT = {1: 4, 5: 8, 9: 16}   # measured: len at resize -> new capacity (sys.getsizeof)

    def slots(self, cap, y, filled):
        cell = 0.62
        g = VGroup()
        for i in range(cap):
            sq = Square(cell, stroke_color=MUTED if i >= filled else INFO, stroke_width=2)
            if i < filled:
                sq.set_fill(INFO, opacity=0.2)
            g.add(VGroup(sq, label(str(i) if i < filled else "", size=20, mono=True)))
        g.arrange(RIGHT, buff=0.06).move_to([0, y, 0])
        return g

    def construct(self):
        self.add(title("list.append: spare slots make it amortized O(1)"))
        stat = label("len 0 · capacity 0", size=26, mono=True, color=HIGHLIGHT).move_to(UP * 2.4)
        self.add(stat)
        arr = VGroup()
        cap = 0
        for n in range(1, 11):
            if n in self.CAP_AT:
                new_cap = self.CAP_AT[n]
                new = self.slots(new_cap, -0.2, n - 1)
                note = "full: allocate a bigger block, copy the pointers" if cap else "first append: allocate 4 slots"
                self.say(f"{note} (capacity {cap} → {new_cap})")
                if len(arr):
                    old = arr
                    self.play(old.animate.shift(UP * 1.1).set_opacity(0.45), FadeIn(new), run_time=0.6)
                    copies = [TransformFromCopy(old[i], new[i]) for i in range(n - 1)]
                    self.play(*copies, run_time=0.7)
                    self.play(FadeOut(old), run_time=0.3)
                else:
                    self.play(FadeIn(new), run_time=0.5)
                arr, cap = new, new_cap
            sq, txt = arr[n - 1]
            self.play(sq.animate.set_stroke(INFO).set_fill(INFO, opacity=0.2),
                      Transform(txt, label(str(n - 1), size=20, mono=True).move_to(sq)),
                      Transform(stat, label(f"len {n} · capacity {cap}", size=26, mono=True, color=HIGHLIGHT).move_to(UP * 2.4)),
                      run_time=0.28)
            if n == 3:
                self.say("Appending into a free slot: no copying at all")
        self.say("Measured capacities: 4, 8, 16, 24, 32, 40, 52, 64, 76 (about +12.5%)")
        self.wait(2.5)


# ----------------------------------------------------------------------------
# 03  shallow vs deep copy
# ----------------------------------------------------------------------------
class ShallowDeepCopy(PyScene):
    def outer(self, name, color, x):
        o = VGroup(*[Square(0.7, stroke_color=color, stroke_width=2.5).set_fill(color, opacity=0.12) for _ in range(2)])
        o.arrange(RIGHT, buff=0).move_to([x, 1.3, 0])
        nm = label(name, size=24, color=color, weight=BOLD).next_to(o, UP, buff=0.18)
        dots = VGroup(*[Dot(sq.get_center(), radius=0.07, color=color) for sq in o])
        return VGroup(o, nm, dots)

    def construct(self):
        self.add(title("Shallow copy shares the inner lists; deepcopy doesn't"))
        orig = self.outer("matrix", INFO, -4.4)
        in1 = box("[1, 2]", SUCCESS, width=1.9, size=24).move_to([-5.5, -1.2, 0])
        in2 = box("[3, 4]", SUCCESS, width=1.9, size=24).move_to([-3.3, -1.2, 0])
        a1 = arrow(orig[2][0].get_center(), in1.get_top(), INFO, buff=0.05)
        a2 = arrow(orig[2][1].get_center(), in2.get_top(), INFO, buff=0.05)
        self.play(FadeIn(orig), FadeIn(in1), FadeIn(in2), GrowArrow(a1), GrowArrow(a2), run_time=0.8)
        self.say("matrix = [[1, 2], [3, 4]]: an outer list of two pointers")
        self.wait(0.6)

        sh = self.outer("shallow = matrix.copy()", PURPLE, -0.6)
        s1 = arrow(sh[2][0].get_center(), in1.get_right() + UP * 0.1, PURPLE, buff=0.05)
        s2 = arrow(sh[2][1].get_center(), in2.get_right(), PURPLE, buff=0.05)
        self.play(FadeIn(sh), run_time=0.5)
        self.play(GrowArrow(s1), GrowArrow(s2), run_time=0.6)
        self.say("A shallow copy is a NEW outer list holding the SAME inner lists")
        self.wait(0.8)

        self.say("shallow[0][0] = 99 writes into the shared inner list")
        self.play(retext(in1, "[99, 2]", color=ERROR, size=24), run_time=0.5)
        self.play(Circumscribe(in1, color=ERROR), run_time=0.7)
        warn = label("matrix changed too!", size=22, color=ERROR).next_to(VGroup(in1, in2), DOWN, buff=0.25)
        self.play(FadeIn(warn), run_time=0.4)
        self.wait(0.8)

        dp = self.outer("deep = copy.deepcopy(...)", SUCCESS, 4.3)
        d1 = box("[1, 2]", SUCCESS, width=1.9, size=24).move_to([3.3, -1.2, 0])
        d2 = box("[3, 4]", SUCCESS, width=1.9, size=24).move_to([5.5, -1.2, 0])
        e1 = arrow(dp[2][0].get_center(), d1.get_top(), SUCCESS, buff=0.05)
        e2 = arrow(dp[2][1].get_center(), d2.get_top(), SUCCESS, buff=0.05)
        self.play(FadeIn(dp), FadeIn(d1), FadeIn(d2), GrowArrow(e1), GrowArrow(e2), run_time=0.8)
        self.say("deepcopy (taken before the write) copies the inner lists too: fully independent")
        self.wait(2.5)


# ----------------------------------------------------------------------------
# 04  compact dict: index table + entries, with probing
# ----------------------------------------------------------------------------
class DictHashTable(PyScene):
    # probe sequences computed with CPython's rule i = (5*i + perturb + 1) & mask, mask = 7
    PROBES = [(3, [3]), (11, [3, 0]), (19, [3, 0, 1]), (5, [5])]

    def construct(self):
        self.add(title("dict insert: hash → slot, probe on collision, append entry"))
        cell_h = 0.58
        slots = VGroup()
        for i in range(8):
            r = Rectangle(width=1.5, height=cell_h, stroke_color=MUTED, stroke_width=2)
            idx = label(str(i), size=20, mono=True, color=MUTED)
            slots.add(VGroup(r, idx))
        slots.arrange(DOWN, buff=0).move_to([-3.6, -0.7, 0])
        for s in slots:
            s[1].next_to(s[0], LEFT, buff=0.2)
        idx_title = label("indices (8 slots)", size=22, color=INFO).next_to(slots, UP, buff=0.2)
        ent_title = label("entries (insertion order)", size=22, color=SUCCESS).move_to([2.6, idx_title.get_y(), 0])
        self.play(FadeIn(slots), FadeIn(idx_title), FadeIn(ent_title), run_time=0.7)

        entries = VGroup()
        for n, (key, probes) in enumerate(self.PROBES):
            self.code(f"d[{key}] = ...    hash({key}) & 7 = {key & 7}", pos=UP * 2.55)
            for j, sl in enumerate(probes):
                cell = slots[sl][0]
                last = j == len(probes) - 1
                color = SUCCESS if last else ERROR
                self.play(cell.animate.set_stroke(color, width=4).set_fill(color, opacity=0.25), run_time=0.35)
                if not last:
                    self.say(f"slot {sl} taken → next probe: i = (5·i + perturb + 1) & 7 = {probes[j + 1]}")
                    self.play(cell.animate.set_stroke(MUTED, width=2).set_fill(INFO, opacity=0.12), run_time=0.25)
            val = label(str(n), size=22, mono=True, color=INFO).move_to(slots[probes[-1]][0])
            row = box(f"{n}: hash {key}, key {key}", SUCCESS, width=4.2, height=0.6, size=22)
            row.move_to([2.6, slots[0].get_y() - n * 0.75, 0])
            link = arrow(slots[probes[-1]][0].get_right(), row.get_left(), INFO, buff=0.08, width=3)
            self.play(FadeIn(val), FadeIn(row, shift=LEFT * 0.2), GrowArrow(link), run_time=0.6)
            self.play(slots[probes[-1]][0].animate.set_stroke(INFO, width=2).set_fill(INFO, opacity=0.12), run_time=0.2)
            entries.add(row, link, val)
            if n == 0:
                self.say("No collision: slot 3 stores the index of entry 0")
            if n == 3:
                self.say("Iteration walks the ENTRIES array: 3, 11, 19, 5 (insertion order)")
        self.wait(2.5)


# ----------------------------------------------------------------------------
# 04  membership: list scan vs set hash jump
# ----------------------------------------------------------------------------
class SetVsListLookup(PyScene):
    VALUES = [12, 7, 45, 2, 88, 21, 64, 5, 30, 99]

    def construct(self):
        self.add(title("99 in list scans; 99 in set jumps straight to its slot"))
        lst = VGroup(*[box(str(v), MUTED, width=0.95, height=0.7, size=24) for v in self.VALUES])
        lst.arrange(RIGHT, buff=0.12).move_to(UP * 1.25)
        l_lbl = label("list: compare one by one", size=22, color=INFO).next_to(lst, UP, buff=0.2)
        self.play(FadeIn(lst), FadeIn(l_lbl), run_time=0.6)
        count = label("comparisons: 0", size=24, mono=True, color=HIGHLIGHT).next_to(lst, DOWN, buff=0.3)
        self.add(count)
        for i, bx in enumerate(lst):
            hit = i == len(lst) - 1
            color = SUCCESS if hit else ERROR
            self.play(bx[0].animate.set_stroke(color).set_fill(color, opacity=0.2),
                      Transform(count, label(f"comparisons: {i + 1}", size=24, mono=True, color=HIGHLIGHT).move_to(count)),
                      run_time=0.22)
        self.say("O(n): the work grows with the length of the list")
        self.wait(0.5)

        cells = VGroup()
        for i in range(32):
            sq = Square(0.36, stroke_color=MUTED, stroke_width=1.5)
            cells.add(sq)
        cells.arrange(RIGHT, buff=0.0).move_to(DOWN * 1.35)
        for v in self.VALUES:
            sl = v & 31
            cells[sl].set_fill(INFO, opacity=0.25).set_stroke(INFO)
        s_lbl = label("set: 32-slot hash table holding the same 10 values", size=22, color=INFO).next_to(cells, UP, buff=0.2)
        self.play(FadeIn(cells), FadeIn(s_lbl), run_time=0.6)
        calc = label("hash(99) & 31 = 3", size=26, mono=True, color=HIGHLIGHT).next_to(cells, DOWN, buff=0.75)
        calc.align_to(cells[3], LEFT).shift(LEFT * 0.2)
        self.play(FadeIn(calc), run_time=0.4)
        pointer = arrow(calc.get_corner(UL) + RIGHT * 0.35, cells[3].get_bottom(), HIGHLIGHT, buff=0.05)
        self.play(GrowArrow(pointer), cells[3].animate.set_stroke(SUCCESS, width=4).set_fill(SUCCESS, opacity=0.5), run_time=0.6)
        one = label("comparisons: 1", size=24, mono=True, color=SUCCESS).next_to(calc, RIGHT, buff=0.8)
        self.play(FadeIn(one), run_time=0.3)
        self.say("O(1) on average: one hash, one jump, whatever the size")
        self.wait(2.5)


# ----------------------------------------------------------------------------
# 05  mutable default argument
# ----------------------------------------------------------------------------
class MutableDefault(PyScene):
    def construct(self):
        self.add(title("def add_metric(val, history=[]): the default is made ONCE"))
        fn = box("function add_metric", PURPLE, width=4.0, height=0.8, size=24).move_to([-3.6, 0.9, 0])
        dflt = label("__defaults__", size=20, mono=True, color=MUTED).next_to(fn, DOWN, buff=0.25)
        lst = box("[]", HIGHLIGHT, width=4.0, height=0.9, size=28).move_to([3.0, 0.9, 0])
        made = label("created when def ran", size=20, color=MUTED).next_to(lst, UP, buff=0.15)
        self.code("def add_metric(val, history=[]): ...", pos=UP * 2.5)
        self.play(FadeIn(fn), FadeIn(dflt), run_time=0.4)
        link = arrow(dflt.get_right(), lst.get_left(), PURPLE)
        self.play(FadeIn(lst), FadeIn(made), GrowArrow(link), run_time=0.6)
        self.say("Defaults are evaluated once, when def runs, and stored on the function")
        self.wait(0.6)

        calls = [("add_metric(0.8)", "[0.8]"), ("add_metric(0.9)", "[0.8, 0.9]"), ("add_metric(0.7)", "[0.8, 0.9, 0.7]")]
        outs = VGroup()
        for i, (call, result) in enumerate(calls):
            self.code(call, pos=UP * 2.5)
            self.play(retext(lst, result, size=28), run_time=0.5)
            self.play(Circumscribe(lst, color=HIGHLIGHT), run_time=0.5)
            out = label(f"{call}  →  {result}", size=24, mono=True,
                        color=SUCCESS if i == 0 else ERROR).move_to([0, -0.8 - i * 0.6, 0])
            self.play(FadeIn(out, shift=UP * 0.1), run_time=0.35)
            outs.add(out)
            if i == 1:
                self.say("The second call sees the first call's data: same list object")
        fix = label("Fix: history=None, then  if history is None: history = []", size=22, mono=True,
                    color=SUCCESS).move_to(DOWN * 2.5)
        self.play(FadeIn(fix), run_time=0.4)
        self.say("Every call that omits history shares (and grows) that one list")
        self.wait(2.5)


# ----------------------------------------------------------------------------
# 05  late-binding closures
# ----------------------------------------------------------------------------
class LateBinding(PyScene):
    def construct(self):
        self.add(title("[lambda x: x + i for i in range(3)] → all read the LAST i"))
        cell = box("i = 0", HIGHLIGHT, width=2.6, height=0.9, size=28).move_to([1.4, 0.7, 0])
        cell_lbl = label("one shared cell", size=20, color=MUTED).next_to(cell, UP, buff=0.15)
        self.play(FadeIn(cell), FadeIn(cell_lbl), run_time=0.4)
        fns, links = VGroup(), VGroup()
        for i in range(3):
            self.play(retext(cell, f"i = {i}", size=28), run_time=0.35)
            f = box(f"f{i} = lambda x: x + i", INFO, width=5.0, height=0.7, size=22).move_to([-4.0, 1.6 - i * 0.9, 0])
            ln = arrow(f.get_right(), cell.get_left(), INFO, buff=0.08, width=3)
            self.play(FadeIn(f, shift=RIGHT * 0.2), GrowArrow(ln), run_time=0.45)
            fns.add(f); links.add(ln)
        self.say("Each lambda stores a reference to the variable, not its value")
        self.play(retext(cell, "i = 2 (loop over)", color=HIGHLIGHT, size=24), run_time=0.4)
        self.wait(0.5)

        results = VGroup()
        for i in range(3):
            self.play(Indicate(links[i], color=HIGHLIGHT), run_time=0.35)
            r = label(f"f{i}(10) → 12", size=26, mono=True, color=ERROR).move_to([5.2, fns[i].get_y(), 0])
            results.add(r)
            self.play(FadeIn(r), run_time=0.3)
        self.say("Called after the loop, every lambda reads i == 2 → [12, 12, 12]")
        self.wait(0.8)

        self.play(FadeOut(links), FadeOut(cell), FadeOut(cell_lbl), run_time=0.4)
        anims = []
        for i in range(3):
            anims.append(retext(fns[i], f"f{i} = lambda x, i={i}: x + i", color=SUCCESS, size=22))
            anims.append(fns[i][0].animate.set_stroke(SUCCESS).set_fill(SUCCESS, opacity=0.12))
            anims.append(Transform(results[i], label(f"f{i}(10) → {10 + i}", size=26, mono=True, color=SUCCESS).move_to(results[i])))
        self.play(*anims, run_time=0.8)
        self.say("Fix: a default argument copies the CURRENT i at creation → [10, 11, 12]")
        self.wait(2.5)


# ----------------------------------------------------------------------------
# 06  generator pause / resume
# ----------------------------------------------------------------------------
class GeneratorPause(PyScene):
    LINES = ["def count_up(n):", "    print('started')", "    i = 1", "    while i <= n:",
             "        yield i", "        i += 1", "    print('finished')"]

    def construct(self):
        self.add(title("A generator runs to the next yield, then freezes"))
        code = code_block(self.LINES)
        code.move_to([-3.3, 0.1, 0])
        panel = SurroundingRectangle(code, color=MUTED, buff=0.3, corner_radius=0.12)
        self.play(FadeIn(panel), FadeIn(code), run_time=0.6)
        state = box("frame: not started", PURPLE, width=4.6, height=0.7, size=22).move_to([3.6, 1.9, 0])
        log = VGroup()

        def log_line(text, color=FG):
            t = label(text, size=23, mono=True, color=color)
            t.move_to([3.6, 1.15 - len(log) * 0.45, 0])
            log.add(t)
            return FadeIn(t, shift=UP * 0.1)

        ptr = Triangle(color=HIGHLIGHT, fill_opacity=1).scale(0.12).rotate(-PI / 2)

        def point(i):
            return ptr.animate.next_to(code[i], LEFT, buff=0.2)

        self.play(FadeIn(state), log_line("g = count_up(2)"), run_time=0.5)
        self.say("Calling it only creates the generator: nothing has run yet")
        self.wait(0.6)

        ptr.next_to(code[0], LEFT, buff=0.2)
        self.play(FadeIn(ptr), log_line("next(g)", HIGHLIGHT), run_time=0.4)
        for i in [1, 2, 3, 4]:
            self.play(point(i), run_time=0.3)
            if i == 1:
                self.play(log_line("  started", MUTED), run_time=0.25)
        self.play(retext(state, "paused at yield · i = 1", color=HIGHLIGHT, size=22), log_line("→ 1", SUCCESS), run_time=0.5)
        self.say("yield hands out 1 and FREEZES: locals and position are kept")
        self.wait(0.6)

        self.play(log_line("next(g)", HIGHLIGHT), run_time=0.3)
        for i in [5, 3, 4]:
            self.play(point(i), run_time=0.3)
        self.play(retext(state, "paused at yield · i = 2", color=HIGHLIGHT, size=22), log_line("→ 2", SUCCESS), run_time=0.5)
        self.say("Resumes right after the yield, loops, and pauses again")
        self.wait(0.5)

        self.play(log_line("next(g)", HIGHLIGHT), run_time=0.3)
        for i in [5, 3, 6]:
            self.play(point(i), run_time=0.3)
        self.play(log_line("  finished", MUTED), retext(state, "frame finished", color=ERROR, size=22), run_time=0.4)
        self.play(log_line("→ StopIteration", ERROR), run_time=0.4)
        self.say("Falling off the end raises StopIteration: a for loop stops there")
        self.wait(2.5)


# ----------------------------------------------------------------------------
# 06  decorator wrapping
# ----------------------------------------------------------------------------
class DecoratorWrap(PyScene):
    def construct(self):
        self.add(title("@timer means: train_step = timer(train_step)"))
        orig = box("original train_step(batch)", INFO, width=5.8, height=0.9, size=24).move_to([0, 0.2, 0])
        tag = box("name: train_step", HIGHLIGHT, width=3.4, height=0.6, size=22).move_to([-4.8, 2.0, 0])
        tag_ar = arrow(tag.get_bottom(), orig.get_left() + UP * 0.2, HIGHLIGHT)
        self.play(FadeIn(orig), FadeIn(tag), GrowArrow(tag_ar), run_time=0.6)
        self.say("def creates the function; the name points at it")
        self.wait(0.5)

        self.code("timer(train_step) builds a wrapper around it", pos=UP * 2.55, color=PURPLE)
        outer = RoundedRectangle(corner_radius=0.2, width=7.4, height=3.6, stroke_color=PURPLE, stroke_width=3)
        outer.set_fill(PURPLE, opacity=0.06).move_to([0, 0.0, 0])
        before = label("start = perf_counter()", size=22, mono=True, color=PURPLE).move_to([0, 1.35, 0])
        after = label("print(elapsed); return result", size=22, mono=True, color=PURPLE).move_to([0, -1.25, 0])
        wl = label("wrapper(*args, **kwargs)", size=20, mono=True, color=PURPLE).next_to(outer, DOWN, buff=0.12)
        self.play(Create(outer), FadeIn(before), FadeIn(after), FadeIn(wl), run_time=0.8)
        new_ar = arrow(tag.get_bottom(), outer.get_left() + UP * 1.2, HIGHLIGHT)
        self.play(Transform(tag_ar, new_ar), run_time=0.6)
        self.say("The name train_step is rebound to the WRAPPER")
        self.wait(0.6)

        self.code("train_step(100)", pos=UP * 2.55)
        steps = [(before, "1. wrapper runs code BEFORE"), (orig, "2. wrapper calls the original"),
                 (after, "3. wrapper runs code AFTER and returns the result")]
        for mob, text in steps:
            self.say(text)
            self.play(Indicate(mob, color=HIGHLIGHT, scale_factor=1.08), run_time=0.7)
        res = label("[TIMER] train_step executed in 0.011524 seconds", size=22, mono=True, color=SUCCESS).move_to(DOWN * 2.6)
        self.play(FadeIn(res), run_time=0.4)
        self.say("functools.wraps copies the name and docstring onto the wrapper")
        self.wait(2.5)


# ----------------------------------------------------------------------------
# 07  attribute lookup and shadowing with +=
# ----------------------------------------------------------------------------
class AttributeLookup(PyScene):
    def dict_panel(self, name, x, color):
        frame = RoundedRectangle(corner_radius=0.15, width=4.4, height=2.4, stroke_color=color, stroke_width=2.5)
        frame.set_fill(color, opacity=0.05).move_to([x, 0.2, 0])
        head = label(name, size=24, color=color, weight=BOLD).next_to(frame, UP, buff=0.15)
        return VGroup(frame, head)

    def construct(self):
        self.add(title("c.count += 1 reads the CLASS, writes the INSTANCE"))
        inst = self.dict_panel("c.__dict__ (instance)", -3.4, INFO)
        cls = self.dict_panel("Counter.__dict__ (class)", 3.4, PURPLE)
        empty = label("(empty)", size=22, color=MUTED).move_to(inst[0])
        cls_val = label("count = 0", size=28, mono=True).move_to(cls[0])
        self.play(FadeIn(inst), FadeIn(cls), FadeIn(empty), FadeIn(cls_val), run_time=0.6)

        self.code("c.count          # read")
        self.play(Circumscribe(inst[0], color=ERROR), run_time=0.6)
        miss = label("not here", size=22, color=ERROR).next_to(inst[0], DOWN, buff=0.2)
        self.play(FadeIn(miss), run_time=0.3)
        hop = arrow(inst[0].get_right(), cls[0].get_left(), HIGHLIGHT)
        self.play(GrowArrow(hop), Circumscribe(cls[0], color=SUCCESS), run_time=0.7)
        self.say("Not on the instance → fall back to the class → 0")
        self.wait(0.5)

        self.code("c.count += 1     # c.count = c.count + 1")
        self.play(FadeOut(miss), FadeOut(hop), run_time=0.3)
        self.say("The read finds the class's 0; the WRITE goes to the instance")
        inst_val = label("count = 1", size=28, mono=True, color=HIGHLIGHT).move_to(inst[0])
        self.play(FadeOut(empty), TransformFromCopy(cls_val, inst_val), run_time=0.8)
        self.wait(0.5)

        self.code("c.count += 1")
        self.play(Transform(inst_val, label("count = 2", size=28, mono=True, color=HIGHLIGHT).move_to(inst[0])), run_time=0.5)
        self.say("From now on the instance attribute SHADOWS the class attribute")
        self.wait(0.5)
        verdict = label("c.count → 2      Counter.count → 0", size=26, mono=True, color=SUCCESS).move_to(DOWN * 2.2)
        self.play(FadeIn(verdict), Indicate(cls_val, color=PURPLE), run_time=0.7)
        self.wait(2.5)


# ----------------------------------------------------------------------------
# 07  C3 linearization for D(B, C)
# ----------------------------------------------------------------------------
class MroC3(PyScene):
    def lst(self, items, y, color=FG):
        g = VGroup(*[box(t, color, width=1.6, height=0.65, size=24) for t in items]).arrange(RIGHT, buff=0.12)
        g.move_to([0, y, 0])
        return g

    def construct(self):
        self.add(title("MRO of class D(B, C): the C3 merge"))
        self.code("L[D] = D + merge(L[B], L[C], [B, C])", pos=UP * 2.55, color=FG)
        rows = [["B", "A", "object"], ["C", "A", "object"], ["B", "C"]]
        names = ["L[B]", "L[C]", "bases"]
        groups = []
        for i, (r, n) in enumerate(zip(rows, names)):
            g = self.lst(r, 1.35 - i * 0.95)
            g.align_to(LEFT * 5.0, LEFT)
            tag = label(n, size=22, color=MUTED).next_to(g, LEFT, buff=0.3)
            groups.append([g, tag, list(r)])
            self.play(FadeIn(g), FadeIn(tag), run_time=0.35)
        result = VGroup(box("D", SUCCESS, width=1.6, height=0.65, size=24))
        result.move_to([-3.4, -1.9, 0])
        r_tag = label("MRO", size=22, color=SUCCESS).next_to(result, LEFT, buff=0.3)
        self.play(FadeIn(result), FadeIn(r_tag), run_time=0.4)
        self.say("Rule: take the first head that is NOT in the tail of any list")

        def take(name):
            anims = []
            for g, tag, items in groups:
                if items and items[0] == name:
                    idx = len(g) - len(items)
                    anims.append(g[idx].animate.set_opacity(0.15))
                    items.pop(0)
            new = box(name, SUCCESS, width=1.6, height=0.65, size=24)
            new.next_to(result, RIGHT, buff=0.12)
            result.add(new)
            anims.append(FadeIn(new, shift=DOWN * 0.2))
            return anims

        def head_box(row_i):
            g, tag, items = groups[row_i]
            return g[len(g) - len(items)]

        # step 1: B
        self.play(Circumscribe(head_box(0), color=SUCCESS), run_time=0.5)
        self.play(*take("B"), run_time=0.6)
        # step 2: candidate A is rejected (it is in the tail of L[C])
        self.play(Circumscribe(head_box(0), color=ERROR), run_time=0.5)
        self.say("Head A is rejected: it still appears in the TAIL of L[C] = [C, A, object]")
        self.play(Indicate(groups[1][0][1], color=ERROR), run_time=0.6)
        self.say("So try the next list's head: C is not in any tail → take C")
        self.play(Circumscribe(head_box(1), color=SUCCESS), run_time=0.5)
        self.play(*take("C"), run_time=0.6)
        self.say("Now A is free to go, then object")
        self.play(*take("A"), run_time=0.6)
        self.play(*take("object"), run_time=0.6)
        self.say("D.__mro__ = D, B, C, A, object: children before parents, bases in order")
        self.wait(2.5)


# ----------------------------------------------------------------------------
# 08  try / except / else / finally paths
# ----------------------------------------------------------------------------
class TryFlow(PyScene):
    def construct(self):
        self.add(title("Which blocks run? try · except · else · finally"))
        t = box("try:  r = 10 / x", INFO, width=4.6, height=0.8, size=24).move_to([0, 1.55, 0])
        e = box("except ZeroDivisionError", ERROR, width=5.2, height=0.8, size=22).move_to([-3.6, 0.0, 0])
        el = box("else:  return 'from else'", SUCCESS, width=5.2, height=0.8, size=22).move_to([3.6, 0.0, 0])
        f = box("finally:  always runs", HIGHLIGHT, width=4.6, height=0.8, size=22).move_to([0, -1.55, 0])
        self.play(*[FadeIn(m) for m in (t, e, el, f)], run_time=0.6)
        ar_te = arrow(t.get_left() + DOWN * 0.2, e.get_top(), ERROR)
        ar_tel = arrow(t.get_right() + DOWN * 0.2, el.get_top(), SUCCESS)
        ar_ef = arrow(e.get_bottom(), f.get_left() + UP * 0.2, ERROR)
        ar_elf = arrow(el.get_bottom(), f.get_right() + UP * 0.2, SUCCESS)
        self.play(*[Create(a.set_opacity(0.25)) for a in (ar_te, ar_tel, ar_ef, ar_elf)], run_time=0.5)

        def run_path(x, path, out, color):
            self.code(f"flow({x})")
            for mob in path:
                self.play(mob.animate.set_opacity(1).set_stroke(width=6) if isinstance(mob, Arrow) else
                          Indicate(mob, color=color, scale_factor=1.06), run_time=0.45)
            res = label(out, size=24, mono=True, color=color)
            return res

        self.say("x = 2: no exception → try, else, finally")
        r1 = run_path(2, [t, ar_tel, el, ar_elf, f], "try; else; finally; from else", SUCCESS)
        r1.move_to(DOWN * 2.45)
        self.play(FadeIn(r1), run_time=0.4)
        self.wait(0.6)
        self.play(*[a.animate.set_opacity(0.25).set_stroke(width=4) for a in (ar_tel, ar_elf)], run_time=0.3)

        self.say("x = 0: division raises → except, else is SKIPPED, finally still runs")
        r2 = run_path(0, [t, ar_te, e, ar_ef, f], "try; except; finally; from except", ERROR)
        self.play(r1.animate.shift(UP * 0.0).set_opacity(0.5), run_time=0.2)
        r2.move_to(DOWN * 2.45)
        self.play(FadeOut(r1), FadeIn(r2), run_time=0.4)
        self.say("finally runs even when try or except has already decided to return")
        self.wait(2.5)


# ----------------------------------------------------------------------------
# 08  exception propagation up the call stack
# ----------------------------------------------------------------------------
class ExceptionPropagation(PyScene):
    def construct(self):
        self.add(title("An exception unwinds the stack until a handler matches"))
        frames = ["main()", "train()   except JSONDecodeError", "load_batch()   finally: close file", "parse_record()"]
        colors = [MUTED, PURPLE, INFO, ERROR]
        stack = VGroup()
        for i, (name, c) in enumerate(zip(frames, colors)):
            b = box(name, c, width=7.2, height=0.8, size=22, mono=True)
            b.move_to([-1.2, -1.8 + i * 1.0, 0])
            stack.add(b)
        st_lbl = label("call stack (top = most recent)", size=20, color=MUTED).next_to(stack, UP, buff=0.2)
        for b in stack:
            self.play(FadeIn(b, shift=UP * 0.2), run_time=0.3)
        self.play(FadeIn(st_lbl), run_time=0.3)

        exc = box("JSONDecodeError", ERROR, width=3.2, height=0.7, size=22).next_to(stack[3], RIGHT, buff=0.4)
        self.play(FadeIn(exc, scale=1.3), Flash(stack[3], color=ERROR), run_time=0.6)
        self.say("parse_record raises: it has no handler, so its frame is discarded")
        self.play(stack[3].animate.set_opacity(0.2), exc.animate.next_to(stack[2], RIGHT, buff=0.4), run_time=0.7)

        self.say("load_batch has no matching except, but its finally runs on the way out")
        fin = label("finally ran: file closed", size=20, color=INFO).next_to(stack[2], DOWN, buff=0.08).align_to(stack[2], RIGHT)
        self.play(Indicate(stack[2], color=INFO), FadeIn(fin), run_time=0.6)
        self.play(stack[2].animate.set_opacity(0.2), FadeOut(fin), exc.animate.next_to(stack[1], RIGHT, buff=0.4), run_time=0.7)

        self.say("train() has except JSONDecodeError: handled, the program continues")
        self.play(stack[1][0].animate.set_stroke(SUCCESS, width=4).set_fill(SUCCESS, opacity=0.2),
                  exc[0].animate.set_stroke(SUCCESS).set_fill(SUCCESS, opacity=0.2), run_time=0.6)
        ok = label("handled → log, skip the record, keep training", size=22, color=SUCCESS).move_to(DOWN * 2.55)
        self.play(FadeIn(ok), run_time=0.4)
        self.wait(2.5)


# ----------------------------------------------------------------------------
# 09  GIL: threads take turns, processes run in parallel (timings measured on the page)
# ----------------------------------------------------------------------------
class GilThreadsVsProcesses(PyScene):
    SCALE = 9.0   # screen units per second of wall time

    def lane(self, y, name, color):
        lbl = label(name, size=22, color=color).move_to([-5.6, y, 0])
        base = Line([-4.4, y, 0], [-4.4 + 1.0 * self.SCALE, y, 0], color=MUTED, stroke_width=1).set_opacity(0.4)
        return VGroup(lbl, base)

    def seg(self, y, t0, t1, color, h=0.42):
        return Rectangle(width=max((t1 - t0) * self.SCALE, 0.01), height=h, stroke_width=0, fill_color=color,
                         fill_opacity=0.85).move_to([-4.4 + (t0 + t1) / 2 * self.SCALE, y, 0])

    def construct(self):
        self.add(title("CPU-bound Python: 2 threads vs 2 processes (2 cores)"))
        axis = NumberLine(x_range=[0, 1.0, 0.25], length=self.SCALE, include_numbers=False, color=MUTED)
        axis.move_to([-4.4 + self.SCALE / 2, -2.25, 0])
        ticks = VGroup(*[label(f"{v:.2f}s", size=18, mono=True, color=MUTED).next_to(axis.n2p(v), DOWN, buff=0.15)
                         for v in [0, 0.25, 0.5, 0.75, 1.0]])
        self.play(Create(axis), FadeIn(ticks), run_time=0.5)

        lanes = {"seq": 2.1, "t1": 0.95, "t2": 0.45, "p1": -0.65, "p2": -1.15}
        self.play(FadeIn(self.lane(lanes["seq"], "sequential", FG)), run_time=0.3)
        self.play(GrowFromEdge(self.seg(lanes["seq"], 0, 0.345, FG), LEFT), run_time=0.6)
        self.play(GrowFromEdge(self.seg(lanes["seq"], 0.345, 0.69, MUTED), LEFT), run_time=0.6)
        self.add(label("0.69 s", size=22, mono=True).move_to([-4.4 + 0.69 * self.SCALE + 0.65, lanes["seq"], 0]))
        self.say("Sequential: two runs of the loop, one after the other")

        self.play(FadeIn(self.lane(lanes["t1"], "thread 1", INFO)), FadeIn(self.lane(lanes["t2"], "thread 2", GRAD)), run_time=0.3)
        gil = box("GIL", HIGHLIGHT, width=0.9, height=0.42, size=18).move_to([-4.4, 1.45, 0])
        self.play(FadeIn(gil), run_time=0.2)
        self.say("Threads: only the GIL holder runs Python, so they take turns (and pay for switching)")
        n, total = 12, 0.86
        step = total / n
        segs = []
        for k in range(n):
            y = lanes["t1"] if k % 2 == 0 else lanes["t2"]
            c = INFO if k % 2 == 0 else GRAD
            s = self.seg(y, k * step, (k + 1) * step - 0.004, c)
            segs.append(s)
            self.play(GrowFromEdge(s, LEFT), gil.animate.move_to([-4.4 + (k + 0.5) * step * self.SCALE, 1.45, 0]),
                      run_time=0.16)
        self.add(label("0.86 s", size=22, mono=True, color=ERROR).move_to([-4.4 + total * self.SCALE + 0.55, (lanes["t1"] + lanes["t2"]) / 2, 0]))
        self.wait(0.4)

        self.play(FadeIn(self.lane(lanes["p1"], "process 1", INFO)), FadeIn(self.lane(lanes["p2"], "process 2", GRAD)), run_time=0.3)
        self.say("Processes: each has its own interpreter and GIL → both cores busy at once")
        self.play(GrowFromEdge(self.seg(lanes["p1"], 0, 0.48, INFO), LEFT),
                  GrowFromEdge(self.seg(lanes["p2"], 0, 0.48, GRAD), LEFT), run_time=1.0)
        self.add(label("0.48 s", size=22, mono=True, color=SUCCESS).move_to([-4.4 + 0.48 * self.SCALE + 0.55, (lanes["p1"] + lanes["p2"]) / 2, 0]))
        self.say("Measured: 0.69 s sequential · 0.86 s threads · 0.48 s processes")
        self.wait(2.5)


# ----------------------------------------------------------------------------
# 09  asyncio: overlapping waits on one thread
# ----------------------------------------------------------------------------
class AsyncioEventLoop(PyScene):
    SCALE = 2.6   # screen units per second

    def construct(self):
        self.add(title("asyncio: 8 LLM calls of 0.5 s on ONE thread"))
        x0 = -5.0
        axis = NumberLine(x_range=[0, 4, 0.5], length=4 * self.SCALE, include_numbers=False, color=MUTED)
        axis.move_to([x0 + 2 * self.SCALE, -2.4, 0])
        ticks = VGroup(*[label(f"{v}s", size=18, mono=True, color=MUTED).next_to(axis.n2p(v), DOWN, buff=0.12)
                         for v in [0, 1, 2, 3, 4]])
        self.play(Create(axis), FadeIn(ticks), run_time=0.5)

        seq_lbl = label("sequential", size=22, color=MUTED).move_to([x0 - 0.9, 1.9, 0])
        self.play(FadeIn(seq_lbl), run_time=0.2)
        self.say("Blocking calls one after another: the thread mostly WAITS")
        seq = VGroup()
        for k in range(8):
            r = Rectangle(width=0.5 * self.SCALE - 0.03, height=0.32, stroke_width=0, fill_color=MUTED, fill_opacity=0.8)
            r.move_to([x0 + (k + 0.5) * 0.5 * self.SCALE, 1.9, 0])
            seq.add(r)
        self.play(LaggedStart(*[GrowFromEdge(r, LEFT) for r in seq], lag_ratio=0.6), run_time=2.4)
        self.add(label("4.00 s", size=22, mono=True, color=ERROR).next_to(seq, RIGHT, buff=0.2))

        loop_lbl = label("event loop", size=22, color=HIGHLIGHT).move_to([x0 - 0.9, 1.0, 0])
        self.play(FadeIn(loop_lbl), run_time=0.2)
        self.say("asyncio: start a call, hit await, start the next one immediately")
        starts = VGroup()
        bars = VGroup()
        for k in range(8):
            tick = Rectangle(width=0.06, height=0.32, stroke_width=0, fill_color=HIGHLIGHT, fill_opacity=1)
            tick.move_to([x0 + 0.03 + k * 0.012, 1.0, 0])
            starts.add(tick)
            y = 0.35 - k * 0.3
            bar = Rectangle(width=0.5 * self.SCALE, height=0.2, stroke_width=0, fill_color=INFO, fill_opacity=0.75)
            bar.move_to([x0 + 0.25 * self.SCALE + k * 0.012, y, 0])
            bars.add(bar)
        self.play(FadeIn(starts), LaggedStart(*[GrowFromEdge(b, LEFT) for b in bars], lag_ratio=0.08), run_time=1.4)
        names = label("8 coroutines waiting on the network at the same time", size=20, color=INFO).next_to(bars, RIGHT, buff=0.3)
        self.play(FadeIn(names), run_time=0.3)
        self.add(label("0.50 s", size=22, mono=True, color=SUCCESS).next_to(bars[0], RIGHT, buff=0.15).shift(UP * 0.35))
        self.say("Measured: 4.00 s sequential vs 0.50 s with asyncio.gather")
        self.wait(2.5)
