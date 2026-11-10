"""Animations for the object detection and segmentation page (keys vision/...).
Box coordinates, scores and IoU values come from torchvision.ops (nms, box_iou); the masks are the real
Mask2Former (facebook/mask2former-swin-small-coco-panoptic) outputs on the two-cats image, downsampled to a grid."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from style import *

HERE = os.path.dirname(os.path.abspath(__file__))


class IouThenNms(WikiScene):
    """Non-maximum suppression: keep the best box, delete boxes that overlap it too much, repeat."""
    def construct(self):
        T = json.load(open(os.path.join(HERE, "data", "vision_nms_toy.json")))
        B, S, I = T["boxes"], T["scores"], T["iou"]
        self.add(title("Non-maximum suppression: one box per object"))
        frame = Rectangle(width=6.6, height=4.4, stroke_color=MUTED, stroke_width=1.5).move_to([-2.6, -0.2, 0])
        self.add(frame)
        ox, oy = frame.get_left()[0], frame.get_top()[1]
        sx, sy = 6.6 / 100, 4.4 / 100

        def box(i, color, width=3):
            x0, y0, x1, y1 = B[i]
            r = Rectangle(width=(x1 - x0) * sx, height=(y1 - y0) * sy, stroke_color=color, stroke_width=width)
            r.move_to([ox + (x0 + x1) / 2 * sx, oy - (y0 + y1) / 2 * sy, 0])
            return r
        boxes = [box(i, INFO, 2.5) for i in range(len(B))]
        cap = caption_bar("a detector outputs 8 candidate boxes for 2 objects: duplicates of the same cat")
        self.play(FadeIn(cap), *[Create(b) for b in boxes], run_time=1.2)
        rows = VGroup()
        order = sorted(range(len(B)), key=lambda i: -S[i])
        for r, i in enumerate(order):
            rows.add(label(f"box {i}   score {S[i]:.2f}", 18, FG, mono=True).move_to([4.2, 1.7 - 0.45 * r, 0]))
        self.play(FadeIn(rows), run_time=0.6)
        self.wait(0.4)
        alive = set(range(len(B)))
        kept = []
        for i in order:
            if i not in alive:
                continue
            kept.append(i)
            row = rows[order.index(i)]
            self.play(boxes[i].animate.set_stroke(SUCCESS, 5), row.animate.set_color(SUCCESS), run_time=0.45)
            sup = [j for j in alive if j != i and I[i][j] > 0.5]
            if sup:
                txt = ", ".join(f"box {j}: {I[i][j]:.2f}" for j in sorted(sup))
                self.play(Transform(cap, caption_bar(f"keep box {i}; IoU above 0.5 with it, so delete: {txt}")), run_time=0.4)
                self.play(*[boxes[j].animate.set_stroke(ERROR, 2, opacity=0.35) for j in sup],
                          *[rows[order.index(j)].animate.set_color(ERROR).set_opacity(0.45) for j in sup], run_time=0.7)
            else:
                self.play(Transform(cap, caption_bar(f"keep box {i}: no remaining box overlaps it by more than 0.5")), run_time=0.4)
            alive -= set(sup) | {i}
            self.wait(0.5)
        self.play(Transform(cap, caption_bar("kept boxes 0, 4 and 7; box 7 (score 0.40) overlaps nothing much, so only a score threshold removes it")), run_time=0.5)
        self.play(boxes[7].animate.set_stroke(HIGHLIGHT, 5), run_time=0.4)
        self.wait(2.5)


class SemanticVsInstance(WikiScene):
    """Semantic segmentation labels pixels by class; instance segmentation also separates the objects."""
    def construct(self):
        D = json.load(open(os.path.join(HERE, "data", "vision_two_cats.json")))
        sem, pan = D["sem"], D["pan"]
        rows, cols = len(sem), len(sem[0])
        cell = 0.34
        self.add(title("Semantic vs instance segmentation (two cats, real model output)"))
        grid = VGroup()
        for r in range(rows):
            for c in range(cols):
                sq = Square(side_length=cell * 0.92, stroke_width=0).set_fill(MUTED, 0.25)
                sq.move_to([(c - (cols - 1) / 2) * cell, 1.6 - r * cell, 0])
                grid.add(sq)
        cap = caption_bar("each square is one block of pixels; a segmentation model gives every pixel a label")
        self.play(FadeIn(grid), FadeIn(cap), run_time=0.8)
        self.wait(0.4)
        anims = []
        for r in range(rows):
            for c in range(cols):
                if sem[r][c] == D["cat_id"]:
                    anims.append(grid[r * cols + c].animate.set_fill(INFO, 0.9))
        self.play(Transform(cap, caption_bar("semantic: 78.3% of pixels are 'cat', but which cat? It cannot say")), run_time=0.4)
        self.play(LaggedStart(*anims, lag_ratio=0.004), run_time=1.6)
        leg = label("cat", 22, INFO).move_to([0, -2.72, 0])
        self.play(FadeIn(leg), run_time=0.3)
        self.wait(1.2)
        anims = []
        for r in range(rows):
            for c in range(cols):
                v = pan[r][c]
                if v == 1:
                    anims.append(grid[r * cols + c].animate.set_fill(INFO, 0.9))
                elif v == 3:
                    anims.append(grid[r * cols + c].animate.set_fill(GRAD, 0.9))
                else:
                    anims.append(grid[r * cols + c].animate.set_fill(MUTED, 0.25))
        self.play(Transform(cap, caption_bar("instance (panoptic): the same pixels split into cat 1 and cat 2")), run_time=0.4)
        leg2 = VGroup(label("cat 1: 37.7% of pixels", 22, INFO), label("cat 2: 37.9% of pixels", 22, GRAD)).arrange(RIGHT, buff=1.0).move_to([0, -2.72, 0])
        self.play(*anims, ReplacementTransform(leg, leg2), run_time=1.4)
        self.wait(3)
