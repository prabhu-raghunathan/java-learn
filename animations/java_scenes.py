"""
Manim scenes for the Core Java pages (memory, GC, bytecode).

Render with:  python animations/render.py jvm/gc-generations   (see MANIFEST in render.py)
"""
from style import *


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def region(width, height, center, name, color):
    box = RoundedRectangle(corner_radius=0.15, width=width, height=height,
                           stroke_color=color, stroke_width=2.5)
    box.set_fill(color, opacity=0.06).move_to(center)
    lab = label(name, size=22, color=color, weight=BOLD).next_to(box, UP, buff=0.1)
    return VGroup(box, lab)


def obj(age, color=INFO):
    c = Circle(radius=0.22, stroke_color=color, stroke_width=2.5).set_fill(color, opacity=0.22)
    t = label(str(age), size=18, mono=True)
    return VGroup(c, t)


def recolor(o, color):
    return [o[0].animate.set_stroke(color).set_fill(color, opacity=0.25)]


def with_age(o, age):
    """Animation that swaps the age digit."""
    new_t = label(str(age), size=18, mono=True).move_to(o[1])
    return Transform(o[1], new_t)


# ---------------------------------------------------------------------------
# 1. Generational GC: Eden → survivors → old
# ---------------------------------------------------------------------------
class GcGenerations(WikiScene):
    THRESHOLD = 3

    def construct(self):
        self.add(title("Generational GC: minor collections and promotion"))

        eden = region(5.6, 2.3, [-2.8, 0.55, 0], "Eden", INFO)
        s0 = region(2.4, 2.3, [1.55, 0.55, 0], "Survivor S0", PURPLE)
        s1 = region(2.4, 2.3, [4.25, 0.55, 0], "Survivor S1", PURPLE)
        old = region(11.05, 1.25, [-0.075, -2.0, 0], "Old generation", HIGHLIGHT)
        self.play(*[FadeIn(r) for r in (eden, s0, s1, old)], run_time=0.8)

        eden_slots = [[-5.05 + 0.85 * c, 1.2 - 0.65 * r, 0] for r in range(3) for c in range(6)]

        def surv_slots(cx):
            return [[cx - 0.55 + 1.1 * c, 1.2 - 0.65 * r, 0] for r in range(3) for c in range(2)]

        old_slots = [[-5.0 + 0.62 * i, -2.05, 0] for i in range(17)]
        old_used = 0
        cap = None

        def caption(text):
            nonlocal cap
            new = caption_bar(text, size=24, color=FG)
            if cap is None:
                self.play(FadeIn(new), run_time=0.4)
            else:
                self.play(Transform(cap, new), run_time=0.4)
                return
            cap = new

        def spawn(n):
            objs = [{"m": obj(0).move_to(eden_slots[i]), "age": 0} for i in range(n)]
            self.play(LaggedStart(*[GrowFromCenter(o["m"]) for o in objs], lag_ratio=0.08), run_time=1.3)
            return objs

        def minor_gc(eden_objs, eden_live, from_objs, from_live, to_cx, label_text):
            """eden_live/from_live: indexes of live objects. Returns the new survivor list."""
            nonlocal old_used
            caption(label_text)
            live = [eden_objs[i] for i in eden_live] + [from_objs[i] for i in from_live]
            dead = [o for i, o in enumerate(eden_objs) if i not in eden_live] + \
                   [o for i, o in enumerate(from_objs) if i not in from_live]
            anims = []
            for o in live:
                anims += recolor(o["m"], SUCCESS)
            for o in dead:
                anims += recolor(o["m"], MUTED)
            self.play(*anims, run_time=0.8)
            self.wait(0.4)

            slots = surv_slots(to_cx)
            kept, moves = [], []
            for o in live:
                o["age"] += 1
                if o["age"] >= self.THRESHOLD:
                    target = obj(o["age"], HIGHLIGHT).move_to(old_slots[old_used])
                    old_used += 1
                else:
                    target = obj(o["age"], SUCCESS).move_to(slots[len(kept)])
                    kept.append(o)
                moves.append(Transform(o["m"], target))
            self.play(*moves, *[FadeOut(o["m"], scale=0.4) for o in dead], run_time=1.4)
            # survivors settle back to the normal colour
            self.play(*[a for o in kept for a in recolor(o["m"], INFO)], run_time=0.3)
            return kept

        # ---- round 1 ----
        caption("New objects are bump-allocated in Eden (age 0)")
        e1 = spawn(12)
        self.wait(0.3)
        surv = minor_gc(e1, [1, 6, 10], [], [], 1.55,
                           "Eden full: minor GC copies ONLY the live objects into S0, age +1")
        self.wait(0.6)

        # ---- round 2 ----
        caption("Eden fills up again")
        e2 = spawn(12)
        surv = minor_gc(e2, [0, 4, 9], surv, [0, 2], 4.25,
                           "Live objects from Eden AND S0 go to S1; S0 and S1 swap roles")
        self.wait(0.6)

        # ---- round 3 ----
        caption("And again")
        e3 = spawn(10)
        surv = minor_gc(e3, [3, 8], surv, [0, 1, 3, 4], 1.55,
                                  "Age reached the threshold: promoted to the old generation")
        self.wait(0.8)
        caption("Demo threshold is 3; HotSpot allows up to 15. Cost = survivors, not garbage")
        self.wait(3)


# ---------------------------------------------------------------------------
# 2. Object initialization order with inheritance
# ---------------------------------------------------------------------------
class InitOrder(WikiScene):
    def construct(self):
        self.add(title("new Child(): the initialization order"))

        def panel(cx, name, color, rows):
            box = RoundedRectangle(corner_radius=0.18, width=5.9, height=3.7,
                                   stroke_color=color, stroke_width=2.5).set_fill(color, opacity=0.05)
            box.move_to([cx, 0.25, 0])
            head = label(name, size=24, color=color, weight=BOLD).next_to(box, UP, buff=0.1)
            items = VGroup()
            for i, text in enumerate(rows):
                r = RoundedRectangle(corner_radius=0.1, width=4.7, height=0.78,
                                     stroke_color=MUTED, stroke_width=1.5)
                r.move_to([cx + 0.35, 1.3 - 1.05 * i, 0])
                t = label(text, size=19, mono=True).move_to(r)
                if t.width > 4.4:
                    t.scale_to_fit_width(4.4)
                items.add(VGroup(r, t))
            return VGroup(box, head), items

        p_panel, p_rows = panel(-3.2, "class Parent", PURPLE,
                                ["static { ... }", "{ instance block }", "Parent() { describe(); }"])
        c_panel, c_rows = panel(3.2, "class Child extends Parent", INFO,
                                ["static { ... }", "name = \"set\"; { block }", "Child() { super(); ... }"])
        self.play(FadeIn(p_panel), FadeIn(c_panel), FadeIn(p_rows), FadeIn(c_rows), run_time=0.8)

        cap = caption_bar("Step through what the JVM runs", size=24, color=FG)
        self.play(FadeIn(cap), run_time=0.3)

        def say(text):
            self.play(Transform(cap, caption_bar(text, size=24, color=FG)), run_time=0.35)

        badges = VGroup()

        def step(n, row, text, hold=0.7):
            say(text)
            b = VGroup(Circle(radius=0.22, stroke_color=HIGHLIGHT, stroke_width=2.5)
                       .set_fill(HIGHLIGHT, opacity=0.25),
                       label(str(n), size=18, mono=True, weight=BOLD))
            b.next_to(row, LEFT, buff=0.12)
            badges.add(b)
            self.play(row[0].animate.set_stroke(HIGHLIGHT, width=3).set_fill(HIGHLIGHT, opacity=0.15),
                      GrowFromCenter(b), run_time=0.5)
            self.wait(hold)
            self.play(row[0].animate.set_stroke(SUCCESS, width=2).set_fill(SUCCESS, opacity=0.08), run_time=0.3)

        step(1, p_rows[0], "Class initialization first, parent before child: static blocks")
        step(2, c_rows[0], "Then the child's static block. Each runs ONCE per class")
        step(3, p_rows[1], "Per object: super() reaches Parent, its instance block runs")
        step(4, p_rows[2], "Parent's constructor body runs and calls describe()", hold=0.3)

        # the trap: overridden method sees an uninitialized child field
        warn = label("Child.describe() runs here and sees name = null", size=22, color=ERROR, weight=BOLD)
        warn.move_to([0, -2.35, 0])
        arrow = Arrow(p_rows[2].get_right(), c_rows[1].get_left() + DOWN * 0.15,
                      color=ERROR, stroke_width=4, buff=0.15)
        self.play(GrowArrow(arrow), FadeIn(warn), run_time=0.6)
        say("Trap: dynamic dispatch reaches Child before Child's fields are set")
        self.wait(1.4)
        self.play(FadeOut(arrow), FadeOut(warn), run_time=0.4)

        step(5, c_rows[1], "Only now: Child's field initializers and instance block")
        step(6, c_rows[2], "Finally the rest of Child's constructor body")
        self.wait(0.8)

        # second object
        say("A second new Child(): statics are already done, steps 3 to 6 repeat")
        self.play(FadeOut(badges),
                  *[r[0].animate.set_stroke(MUTED, width=1.5).set_fill(MUTED, opacity=0) for r in
                    list(p_rows) + list(c_rows)],
                  run_time=0.5)
        self.play(p_rows[0].animate.set_opacity(0.3), c_rows[0].animate.set_opacity(0.3), run_time=0.4)
        for row in (p_rows[1], p_rows[2], c_rows[1], c_rows[2]):
            self.play(row[0].animate.set_stroke(SUCCESS, width=2.5).set_fill(SUCCESS, opacity=0.12),
                      run_time=0.35)
        self.wait(3)


# ---------------------------------------------------------------------------
# 3. Pass-by-value: references are copied
# ---------------------------------------------------------------------------
class PassByValue(WikiScene):
    def construct(self):
        self.add(title("Java passes a COPY of the reference"))

        stack = RoundedRectangle(corner_radius=0.15, width=5.2, height=5.0, stroke_color=MUTED, stroke_width=2)
        stack.move_to([-3.7, -0.2, 0])
        heap = RoundedRectangle(corner_radius=0.15, width=5.6, height=5.0, stroke_color=MUTED, stroke_width=2)
        heap.move_to([3.4, -0.2, 0])
        st_lab = label("JVM stack (frames)", size=22, color=MUTED, weight=BOLD).next_to(stack, UP, buff=0.08)
        hp_lab = label("Heap (objects)", size=22, color=MUTED, weight=BOLD).next_to(heap, UP, buff=0.08)
        self.play(FadeIn(stack), FadeIn(heap), FadeIn(st_lab), FadeIn(hp_lab), run_time=0.6)

        def box_obj(v, y):
            r = RoundedRectangle(corner_radius=0.12, width=2.4, height=0.95, stroke_color=SUCCESS, stroke_width=2.5)
            r.set_fill(SUCCESS, opacity=0.1).move_to([3.6, y, 0])
            t = label(f"Box  v = {v}", size=22, mono=True).move_to(r)
            return VGroup(r, t)

        box1 = box_obj(1, 0.9)
        box2 = box_obj(2, -1.5)

        def frame(name, y, color, vars_):
            f = RoundedRectangle(corner_radius=0.12, width=4.6, height=1.7, stroke_color=color, stroke_width=2.5)
            f.set_fill(color, opacity=0.07).move_to([-3.7, y, 0])
            n = label(name, size=19, color=color, weight=BOLD).move_to(f.get_corner(UL) + [0.95, -0.25, 0])
            cells = {}
            grp = VGroup(f, n)
            for i, v in enumerate(vars_):
                lab = label(v, size=22, mono=True).move_to([-5.0 + 1.9 * i, y - 0.25, 0])
                cell = Square(0.5, stroke_color=color, stroke_width=2).next_to(lab, RIGHT, buff=0.15)
                dot = Dot(cell.get_center(), radius=0.07, color=color)
                cells[v] = dot
                grp.add(lab, cell, dot)
            return grp, cells

        def ref(dot, target, color):
            return Arrow(dot.get_center(), target.get_left(), color=color, buff=0.08,
                         stroke_width=4, max_tip_length_to_length_ratio=0.08)

        cap = caption_bar("main(): a and b hold references to two objects", size=24, color=FG)
        main_f, main_c = frame("main()", -1.65, INFO, ["a", "b"])
        a_arrow = ref(main_c["a"], box1, INFO)
        b_arrow = ref(main_c["b"], box2, INFO)
        self.play(FadeIn(main_f), FadeIn(box1), FadeIn(box2), FadeIn(cap), run_time=0.6)
        self.play(GrowArrow(a_arrow), GrowArrow(b_arrow), run_time=0.7)
        self.wait(0.8)

        def say(text):
            self.play(Transform(cap, caption_bar(text, size=24, color=FG)), run_time=0.35)

        # ---- swap ----
        say("swap(a, b): the new frame gets COPIES of both references")
        sw_f, sw_c = frame("swap(a, b)", 0.75, PURPLE, ["a", "b"])
        sa = ref(sw_c["a"], box1, PURPLE)
        sb = ref(sw_c["b"], box2, PURPLE)
        self.play(FadeIn(sw_f, shift=DOWN * 0.3), run_time=0.5)
        self.play(GrowArrow(sa), GrowArrow(sb), run_time=0.6)
        self.wait(0.6)
        say("swap() exchanges ITS copies...")
        self.play(Transform(sa, ref(sw_c["a"], box2, PURPLE)), Transform(sb, ref(sw_c["b"], box1, PURPLE)), run_time=0.9)
        self.wait(0.8)
        say("...then its frame is popped. main's a and b never moved: prints 1 2")
        self.play(FadeOut(VGroup(sw_f, sa, sb), shift=UP * 0.3), run_time=0.6)
        self.play(Indicate(a_arrow, color=HIGHLIGHT), Indicate(b_arrow, color=HIGHLIGHT), run_time=0.9)
        self.wait(0.6)

        # ---- mutate ----
        say("mutate(a): a copy of the reference, pointing at the SAME object")
        mu_f, mu_c = frame("mutate(a)", 0.75, PURPLE, ["a"])
        ma = ref(mu_c["a"], box1, PURPLE)
        self.play(FadeIn(mu_f, shift=DOWN * 0.3), run_time=0.5)
        self.play(GrowArrow(ma), run_time=0.5)
        say("a.v = 99 changes the shared object")
        new_t = label("Box  v = 99", size=22, mono=True, color=HIGHLIGHT).move_to(box1[1])
        self.play(Transform(box1[1], new_t), box1[0].animate.set_stroke(HIGHLIGHT), run_time=0.7)
        self.wait(0.5)
        self.play(FadeOut(VGroup(mu_f, ma), shift=UP * 0.3), run_time=0.5)
        say("Frame gone, change stays: main sees v = 99 through its own reference")
        self.play(Indicate(a_arrow, color=HIGHLIGHT), run_time=0.9)
        self.wait(3)


# ---------------------------------------------------------------------------
# 4. i = i++ at the bytecode level
# ---------------------------------------------------------------------------
class PostfixTrap(WikiScene):
    def construct(self):
        self.add(title("int i = 5;  i = i++;   →   i is still 5"))

        code_lines = ["iconst_5", "istore_0", "iload_0", "iinc 0, 1", "istore_0"]
        notes = ["push 5", "i = 5", "push the OLD i", "i = i + 1 (in place)", "pop into i"]
        listing = VGroup()
        for k, (c, n) in enumerate(zip(code_lines, notes)):
            row = VGroup(label(c, size=24, mono=True), label(n, size=18, color=MUTED))
            row[1].next_to(row[0], RIGHT, buff=0.35)
            listing.add(row)
        listing.arrange(DOWN, aligned_edge=LEFT, buff=0.38).move_to([-3.6, 0.1, 0])
        code_box = SurroundingRectangle(listing, color=MUTED, buff=0.3, corner_radius=0.12)
        code_lab = label("bytecode (javap -c)", size=20, color=MUTED, weight=BOLD).next_to(code_box, UP, buff=0.08)

        # local variable slot
        lv_cell = Rectangle(width=1.4, height=0.9, stroke_color=INFO, stroke_width=2.5).move_to([1.6, 0.9, 0])
        lv_lab = label("local slot 0  (i)", size=20, color=INFO, weight=BOLD).next_to(lv_cell, UP, buff=0.12)
        lv_val = label("–", size=30, mono=True).move_to(lv_cell)

        # operand stack
        os_box = Rectangle(width=1.6, height=2.6, stroke_color=PURPLE, stroke_width=2.5).move_to([4.7, 0.2, 0])
        os_lab = label("operand stack", size=20, color=PURPLE, weight=BOLD).next_to(os_box, UP, buff=0.12)
        slot_pos = os_box.get_bottom() + UP * 0.5

        self.play(FadeIn(code_box), FadeIn(code_lab), FadeIn(listing),
                  FadeIn(lv_cell), FadeIn(lv_lab), FadeIn(lv_val), FadeIn(os_box), FadeIn(os_lab), run_time=0.8)

        cap = caption_bar("Each line runs in order; watch the slot and the stack", size=24, color=FG)
        self.play(FadeIn(cap), run_time=0.3)
        pointer = Triangle(color=HIGHLIGHT, fill_opacity=1).scale(0.13).rotate(-PI / 2)

        def point_to(k):
            target = listing[k][0].get_left() + LEFT * 0.3
            if pointer not in self.mobjects:
                pointer.move_to(target)
                self.play(FadeIn(pointer), run_time=0.2)
            else:
                self.play(pointer.animate.move_to(target), run_time=0.3)
            self.play(listing[k][0].animate.set_color(HIGHLIGHT), run_time=0.2)

        def say(t):
            self.play(Transform(cap, caption_bar(t, size=24, color=FG)), run_time=0.3)

        def chip(v, color=PURPLE):
            r = RoundedRectangle(corner_radius=0.08, width=1.2, height=0.6, stroke_color=color, stroke_width=2)
            r.set_fill(color, opacity=0.2)
            return VGroup(r, label(str(v), size=26, mono=True)).move_to(slot_pos)

        # iconst_5
        point_to(0)
        five = chip(5)
        self.play(FadeIn(five, shift=DOWN * 0.4), run_time=0.5)
        # istore_0
        point_to(1)
        self.play(five.animate.move_to(lv_cell), run_time=0.6)
        self.remove(five)
        self.play(Transform(lv_val, label("5", size=30, mono=True).move_to(lv_cell)), run_time=0.2)
        say("i = 5")
        self.wait(0.4)
        # iload_0
        point_to(2)
        say("i++ first PUSHES the old value of i (5)...")
        old = label("5", size=30, mono=True).move_to(lv_cell)
        self.add(old)
        old_chip = chip(5)
        self.play(Transform(old, old_chip), run_time=0.6)
        self.wait(0.4)
        # iinc
        point_to(3)
        say("...then increments the local slot directly: i is 6 for a moment")
        self.play(Transform(lv_val, label("6", size=30, mono=True, color=SUCCESS).move_to(lv_cell)),
                  Flash(lv_cell, color=SUCCESS), run_time=0.7)
        self.wait(0.8)
        # istore_0
        point_to(4)
        say("The assignment pops the OLD 5 back into i, overwriting the 6")
        self.play(old.animate.move_to(lv_cell), run_time=0.7)
        self.remove(old)
        self.play(Transform(lv_val, label("5", size=30, mono=True, color=ERROR).move_to(lv_cell)),
                  lv_cell.animate.set_stroke(ERROR), run_time=0.4)
        self.wait(0.6)
        say("Result: i = i++ never changes i. Use i++ or i = i + 1 on its own")
        self.wait(3)


# ---------------------------------------------------------------------------
# Hashing helpers (scenes 5-9)
# ---------------------------------------------------------------------------
def hcell(w, h, color=MUTED, fill=None, op=0.0):
    r = Rectangle(width=w, height=h, stroke_color=color, stroke_width=2)
    if fill is not None:
        r.set_fill(fill, opacity=op)
    return r


def table_row(n, w, h, center, idx_size=14, color=MUTED):
    """n adjacent cells with their index under each one. Returns (cells, idx_labels)."""
    cells = VGroup(*[hcell(w, h, color) for _ in range(n)]).arrange(RIGHT, buff=0).move_to(center)
    idx = VGroup(*[label(str(i), size=idx_size, color=MUTED, mono=True).next_to(cells[i], UP, buff=0.06)
                   for i in range(n)])
    return cells, idx


def node(txt, color=INFO, w=0.7, h=0.42, size=18):
    r = RoundedRectangle(corner_radius=0.08, width=w, height=h, stroke_color=color, stroke_width=2)
    r.set_fill(color, opacity=0.18)
    return VGroup(r, label(txt, size=size, mono=True))


class HashSay:
    """Mixin: cross-fading caption line."""
    def say(self, text, color=FG):
        new = caption_bar(text, size=24, color=color)
        if getattr(self, "_cap", None) is None:
            self._cap = new
            self.play(FadeIn(new), run_time=0.3)
        else:
            self.play(Transform(self._cap, new), run_time=0.35)


# ---------------------------------------------------------------------------
# 5. String.hashCode(): h = 31*h + c   ("cat" -> 98262, verified on JDK 21)
# ---------------------------------------------------------------------------
class StringHashCode(HashSay, WikiScene):
    def construct(self):
        self.add(title('"cat".hashCode()  =  fold h = 31 · h + c over the chars'))
        chars = [("c", 99), ("a", 97), ("t", 116)]
        boxes = VGroup()
        for ch, code in chars:
            sq = Square(1.0, stroke_color=INFO, stroke_width=2.5).set_fill(INFO, opacity=0.1)
            t = label(f"'{ch}'", size=32, mono=True)
            v = label(str(code), size=20, color=MUTED, mono=True).next_to(sq, DOWN, buff=0.12)
            boxes.add(VGroup(VGroup(sq, t), v))
        boxes.arrange(RIGHT, buff=0.5).move_to([0, 1.6, 0])
        self.play(FadeIn(boxes), run_time=0.6)

        hreg = Rectangle(width=2.6, height=0.9, stroke_color=PURPLE, stroke_width=2.5).move_to([4.6, -0.6, 0])
        hlab = label("h", size=26, color=PURPLE, weight=BOLD).next_to(hreg, UP, buff=0.12)
        hval = label("0", size=32, mono=True).move_to(hreg)
        self.play(FadeIn(hreg), FadeIn(hlab), FadeIn(hval), run_time=0.4)
        self.say("Start with h = 0, then visit the characters left to right")

        steps = [("31 × 0 + 99", "99"), ("31 × 99 + 97", "3166"), ("31 × 3166 + 116", "98262")]
        lines = VGroup()
        for k, (expr, res) in enumerate(steps):
            self.play(boxes[k][0][0].animate.set_stroke(HIGHLIGHT).set_fill(HIGHLIGHT, opacity=0.2), run_time=0.3)
            ln = VGroup(label(f"h = {expr}", size=26, mono=True),
                        label(f"= {res}", size=26, mono=True, color=SUCCESS))
            ln[1].next_to(ln[0], RIGHT, buff=0.25)
            lines.add(ln)
            lines.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([-2.3, -0.9, 0])
            self.play(FadeIn(ln, shift=RIGHT * 0.2), run_time=0.5)
            self.play(Transform(hval, label(res, size=32, mono=True, color=SUCCESS).move_to(hreg)), run_time=0.4)
            self.play(boxes[k][0][0].animate.set_stroke(INFO).set_fill(INFO, opacity=0.1), run_time=0.2)
            self.wait(0.5)
        self.say("Same as 99·31² + 97·31 + 116: every char and its position count")
        self.wait(1.4)
        cache = label("cached in the String's hash field: next call is O(1)", size=22, color=HIGHLIGHT)
        cache.move_to([0, -2.55, 0])
        self.play(FadeIn(cache), run_time=0.4)
        self.say("Different order, different hash: \"act\" and \"cat\" do not collide", color=FG)
        self.wait(3)


# ---------------------------------------------------------------------------
# 6. Key -> hashCode -> spread -> index -> bucket (numbers verified on JDK 21)
# ---------------------------------------------------------------------------
class KeyToBucket(HashSay, WikiScene):
    def construct(self):
        self.add(title("From key to bucket in a HashMap with 16 buckets"))
        heads = ["key", "hashCode()", "h ^ (h >>> 16)", "& 15  →  index"]
        xs = [-5.0, -2.2, 1.0, 4.4]
        head_row = VGroup(*[label(h, size=22, color=MUTED, mono=True).move_to([x, 2.55, 0]) for h, x in zip(heads, xs)])
        arrows = VGroup(*[Arrow([xs[i] + 0.9, 1.85, 0], [xs[i + 1] - 1.0, 1.85, 0], buff=0, color=MUTED,
                                stroke_width=2, max_tip_length_to_length_ratio=0.15) for i in range(3)])
        self.play(FadeIn(head_row), FadeIn(arrows), run_time=0.5)

        cells, idx = table_row(16, 0.78, 0.6, [0, -0.35, 0])
        tlab = label("table[16]", size=20, color=MUTED, weight=BOLD).next_to(cells, LEFT, buff=0.15).shift(UP * 0.4)
        tlab.to_edge(LEFT, buff=0.2)
        self.play(FadeIn(cells), FadeIn(idx), run_time=0.5)

        keys = [('"cat"', "98262", "98263", 7, INFO), ('"dog"', "99644", "99645", 13, INFO),
                ('"Aa"', "2112", "2112", 0, INFO), ('"BB"', "2112", "2112", 0, GRAD)]
        placed = {}
        for name, hc, sp, ix, col in keys:
            vals = [name, hc, sp, str(ix)]
            row = VGroup(*[label(v, size=26, mono=True, color=(HIGHLIGHT if j == 3 else FG)).move_to([x, 1.85, 0])
                           for j, (v, x) in enumerate(zip(vals, xs))])
            self.say(f"{name}: hash, mix the high bits in, keep the low 4 bits")
            for j in range(4):
                self.play(FadeIn(row[j], shift=RIGHT * 0.15), run_time=0.3)
            nd = node(name, color=col, w=0.74)
            nd.move_to(row[3])
            self.play(FadeIn(nd), run_time=0.2)
            target_y = cells[ix].get_center()[1] - 0.75 * placed.get(ix, 0)
            target = [cells[ix].get_center()[0], cells[ix].get_center()[1] if placed.get(ix, 0) == 0 else target_y, 0]
            if placed.get(ix, 0) == 0:
                self.play(nd.animate.move_to(target), cells[ix].animate.set_stroke(col), run_time=0.7)
            else:
                self.say("Same index as \"Aa\": equals() says different key, so it is chained", color=HIGHLIGHT)
                self.play(nd.animate.move_to(target), run_time=0.7)
                link = Arrow(cells[ix].get_bottom() + UP * 0.05, nd.get_top(), buff=0.02, color=GRAD,
                             stroke_width=3, max_tip_length_to_length_ratio=0.35)
                self.play(GrowArrow(link), run_time=0.3)
                self.wait(1.0)
            placed[ix] = placed.get(ix, 0) + 1
            self.play(FadeOut(row), run_time=0.25)
        self.say("get() repeats the same steps, then walks only that one bucket")
        self.wait(3)


# ---------------------------------------------------------------------------
# 7. Resize 16 -> 32: each node stays at j or moves to j + 16 (lo/hi split)
# ---------------------------------------------------------------------------
class ResizeSplit(HashSay, WikiScene):
    def construct(self):
        self.add(title("Resize 16 → 32: one bit decides, stay at j or move to j + 16"))
        old, old_idx = table_row(16, 0.8, 0.5, [0, 1.75, 0])
        olab = label("old[16]", size=18, color=MUTED, weight=BOLD).next_to(old, UP, buff=0.32).align_to(old, LEFT)
        new, new_idx = table_row(32, 0.4, 0.5, [0, -1.2, 0], idx_size=11)
        nlab = label("new[32]", size=18, color=MUTED, weight=BOLD).next_to(new, UP, buff=0.32).align_to(new, LEFT)
        self.play(FadeIn(old), FadeIn(old_idx), FadeIn(olab), run_time=0.5)

        # keys 1,17,33 -> bucket 1; 5,21 -> bucket 5 (insertion order)
        chains = {1: [1, 17, 33], 5: [5, 21]}
        nodes = {}
        for b, ks in chains.items():
            for d, k in enumerate(ks):
                nd = node(str(k), w=0.62, h=0.36, size=16)
                nd.move_to(old[b].get_center() + DOWN * (0.0 if d == 0 else 0.42 * d))
                if d > 0:
                    nd.shift(DOWN * 0.0)
                nodes[k] = nd
        stack = VGroup(*nodes.values())
        self.say("Keys 1, 17, 33 share bucket 1 and keys 5, 21 share bucket 5 (index = hash & 15)")
        self.play(LaggedStart(*[FadeIn(n) for n in stack], lag_ratio=0.15), run_time=1.0)
        self.wait(0.8)
        self.play(FadeIn(new), FadeIn(new_idx), FadeIn(nlab), run_time=0.5)
        self.say("New index = hash & 31. Only bit 16 is new: check hash & 16")

        placed = {}
        for k in [1, 17, 33, 5, 21]:
            j = 1 if k in (1, 17, 33) else 5
            hi = k & 16
            dest = j + (16 if hi else 0)
            col = HIGHLIGHT if hi else SUCCESS
            msg = f"{k} & 16 = {hi}  →  " + (f"move to {j} + 16 = {dest}" if hi else f"stay at {j}")
            tag = label(msg, size=22, mono=True, color=col).move_to([0, -2.55, 0])
            self.play(FadeIn(tag), nodes[k][0].animate.set_stroke(col), run_time=0.3)
            depth = placed.get(dest, 0)
            target = new[dest].get_center() + DOWN * (0.45 * depth)
            self.play(nodes[k].animate.scale(0.62).move_to(target), run_time=0.6)
            placed[dest] = depth + 1
            self.play(FadeOut(tag), run_time=0.2)
        self.say("Relative order is kept, and no hashCode() is called again", color=SUCCESS)
        self.wait(3)


# ---------------------------------------------------------------------------
# 8. Treeify: 9th colliding key resizes (table < 64), 11th turns the bin into a tree
#    (BadKey with hashCode 42; capacities measured with reflection on JDK 21)
# ---------------------------------------------------------------------------
class Treeify(HashSay, WikiScene):
    def construct(self):
        self.add(title("Every key has hashCode 42: when does the bucket become a tree?"))
        cap_lab = label("table length", size=20, color=MUTED).move_to([-4.8, 2.1, 0])
        cap_val = label("16", size=40, mono=True, color=INFO).next_to(cap_lab, DOWN, buff=0.15)
        bin_lab = label("bucket", size=20, color=MUTED).move_to([-1.6, 2.1, 0])
        bin_val = label("42 & 15 = 10", size=26, mono=True).next_to(bin_lab, DOWN, buff=0.22)
        cnt_lab = label("entries in that bucket", size=20, color=MUTED).move_to([2.6, 2.1, 0])
        cnt_val = label("0", size=40, mono=True, color=GRAD).next_to(cnt_lab, DOWN, buff=0.15)
        self.play(FadeIn(VGroup(cap_lab, cap_val, bin_lab, bin_val, cnt_lab, cnt_val)), run_time=0.5)

        chain = VGroup()
        y = -0.3
        caps = {9: "32", 10: "64"}
        bins = {9: "42 & 31 = 10", 10: "42 & 63 = 42"}
        for i in range(1, 12):
            c = Circle(radius=0.27, stroke_color=INFO, stroke_width=2.5).set_fill(INFO, opacity=0.2)
            nd = VGroup(c, label(str(i), size=18, mono=True))
            nd.move_to([-6.0 + 1.0 * (i - 1) * 1.17, y, 0])
            arrows = []
            if i > 1:
                arrows.append(Arrow(chain[-1].get_right(), nd.get_left(), buff=0.03, color=MUTED, stroke_width=2,
                                    max_tip_length_to_length_ratio=0.3))
            self.play(FadeIn(nd, shift=DOWN * 0.2), *[GrowArrow(a) for a in arrows],
                      Transform(cnt_val, label(str(i), size=40, mono=True, color=GRAD).move_to(cnt_val)),
                      run_time=0.32)
            for a in arrows:
                chain.add(a)
            chain.add(nd)
            if i == 8:
                self.say("8 entries: still a plain linked list")
                self.wait(0.6)
            if i in caps:
                txt = ("9th entry: table is only 16 (< 64), so it RESIZES instead of treeifying"
                       if i == 9 else "10th entry: 32 is still < 64, resize again")
                self.say(txt, color=HIGHLIGHT)
                self.play(Transform(cap_val, label(caps[i], size=40, mono=True, color=HIGHLIGHT).move_to(cap_val)),
                          Transform(bin_val, label(bins[i], size=26, mono=True).move_to(bin_val)), run_time=0.5)
                self.wait(1.0)
        self.say("11th entry: table is 64 and the list is long, so the bin becomes a red-black tree", color=SUCCESS)
        self.wait(0.6)

        # balanced tree of 11 nodes (in-order = 1..11)
        pos = {6: (0, 0.5), 3: (-3.2, -0.3), 9: (3.2, -0.3), 2: (-4.6, -1.1), 5: (-1.8, -1.1),
               8: (1.8, -1.1), 10: (4.6, -1.1), 1: (-5.3, -1.9), 4: (-2.5, -1.9), 7: (1.1, -1.9), 11: (5.3, -1.9)}
        reds = {3, 9, 1, 4, 7, 11}
        tree_nodes = {}
        for k, (x, yy) in pos.items():
            col = ERROR if k in reds else FG
            c = Circle(radius=0.27, stroke_color=col, stroke_width=3).set_fill(col, opacity=0.18)
            tree_nodes[k] = VGroup(c, label(str(k), size=18, mono=True)).move_to([x, yy, 0])
        edges = VGroup()
        for p, kids in {6: [3, 9], 3: [2, 5], 9: [8, 10], 2: [1], 5: [4], 8: [7], 10: [11]}.items():
            for kd in kids:
                edges.add(Line(tree_nodes[p].get_center(), tree_nodes[kd].get_center(), color=MUTED,
                               stroke_width=2, buff=0.28))
        list_nodes = [m for m in chain if isinstance(m, VGroup)]
        self.play(FadeOut(VGroup(*[m for m in chain if not isinstance(m, VGroup)])), run_time=0.3)
        self.play(*[Transform(list_nodes[k - 1], tree_nodes[k]) for k in range(1, 12)], run_time=1.2)
        self.play(Create(edges), run_time=0.6)
        self.say("Lookups in this bin: O(log n) if keys are Comparable, else O(n) at worst", color=SUCCESS)
        self.wait(3)


# ---------------------------------------------------------------------------
# 9. ConcurrentHashMap put: CAS into an empty bin, lock only the bin head otherwise
# ---------------------------------------------------------------------------
class ChmPut(HashSay, WikiScene):
    def construct(self):
        self.add(title("ConcurrentHashMap (Java 8+): lock one bin, never the whole map"))
        cells, idx = table_row(8, 1.3, 0.8, [0, 0.9, 0])
        self.play(FadeIn(cells), FadeIn(idx), run_time=0.4)
        existing = node("k5", color=MUTED, w=1.0, h=0.5).move_to(cells[5])
        existing6 = node("k6", color=MUTED, w=1.0, h=0.5).move_to(cells[6])
        self.play(FadeIn(existing), FadeIn(existing6), run_time=0.3)

        def thread(name, color, x):
            r = RoundedRectangle(corner_radius=0.1, width=1.9, height=0.6, stroke_color=color, stroke_width=2.5)
            r.set_fill(color, opacity=0.15)
            return VGroup(r, label(name, size=20, color=color, weight=BOLD)).move_to([x, -1.5, 0])

        ta = thread("Thread A", INFO, -4.6)
        tb = thread("Thread B", GRAD, 0.9)
        tc = thread("Thread C", PURPLE, 3.5)
        self.play(FadeIn(ta), FadeIn(tb), FadeIn(tc), run_time=0.4)

        # A: CAS into empty bin 1
        self.say("A: bin 1 is empty, so one CAS (compare-and-swap) installs the node: no lock")
        na = node("kA", color=INFO, w=1.0, h=0.5).move_to(ta.get_top() + UP * 0.4)
        cas = label("CAS null → node", size=18, color=INFO, mono=True).next_to(cells[1], DOWN, buff=1.0)
        self.play(FadeIn(na), FadeIn(cas), run_time=0.3)
        self.play(na.animate.move_to(cells[1]), cells[1].animate.set_stroke(INFO), run_time=0.7)
        self.wait(0.6)

        # B and C at the same time on bins 5 and 6
        self.say("B and C: bins 5 and 6 are occupied, each does synchronized on its own bin head")
        lock_b = label("locked", size=16, color=GRAD, mono=True).next_to(cells[5], UP, buff=0.4)
        lock_c = label("locked", size=16, color=PURPLE, mono=True).next_to(cells[6], UP, buff=0.4)
        self.play(cells[5].animate.set_stroke(GRAD, width=5), cells[6].animate.set_stroke(PURPLE, width=5),
                  FadeIn(lock_b), FadeIn(lock_c), run_time=0.5)
        nb = node("kB", color=GRAD, w=1.0, h=0.5).move_to(tb.get_top() + UP * 0.4)
        nc = node("kC", color=PURPLE, w=1.0, h=0.5).move_to(tc.get_top() + UP * 0.4)
        self.play(FadeIn(nb), FadeIn(nc), run_time=0.3)
        self.play(nb.animate.move_to(cells[5].get_center() + DOWN * 0.95 + LEFT * 0.0).scale(0.9),
                  nc.animate.move_to(cells[6].get_center() + DOWN * 0.95 + RIGHT * 0.0).scale(0.9), run_time=0.8)
        self.say("Both writes ran in parallel: different bins never block each other", color=SUCCESS)
        self.wait(1.0)

        # reader
        reader = thread("get(k5)", SUCCESS, -1.8)
        self.play(FadeIn(reader), run_time=0.3)
        arr = Arrow(reader.get_top(), cells[5].get_bottom() + LEFT * 0.3, buff=0.1, color=SUCCESS, stroke_width=3)
        self.say("Readers take no lock at all: volatile reads of the table and node fields")
        self.play(GrowArrow(arr), run_time=0.5)
        self.play(FadeOut(lock_b), FadeOut(lock_c), cells[5].animate.set_stroke(MUTED, width=2),
                  cells[6].animate.set_stroke(MUTED, width=2), run_time=0.4)
        self.say("Writers on the SAME bin wait; everyone else keeps going", color=FG)
        self.wait(3)
