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
