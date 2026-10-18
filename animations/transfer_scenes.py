"""Animations for the transfer-learning page (06-transfer-learning-resnet18)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from style import *

FROZEN = "#5a6f8a"   # muted steel blue for frozen blocks


class FreezeVsFinetune(WikiScene):
    """Backward pass reaches only trainable layers. Counts verified with torchvision resnet18."""

    def construct(self):
        head = title("Gradients only flow into trainable layers")
        self.add(head)
        names = ["stem", "layer1", "layer2", "layer3", "layer4", "fc"]
        blocks = VGroup()
        for n in names:
            box = RoundedRectangle(corner_radius=0.12, width=1.4, height=1.1, stroke_color=FROZEN, stroke_width=3)
            box.set_fill(FROZEN, 0.25)
            blocks.add(VGroup(box, label(n, 22, FG, mono=True).move_to(box)))
        blocks.arrange(RIGHT, buff=0.25).move_to(UP * 0.55 + RIGHT * 0.15)
        loss = Circle(0.5, color=ERROR, stroke_width=3).set_fill(ERROR, 0.2).next_to(blocks, RIGHT, buff=0.4)
        loss_t = label("loss", 22, FG).move_to(loss)
        img = label("image", 22, MUTED).next_to(blocks, LEFT, buff=0.25)
        state = VGroup(*[label("frozen", 17, MUTED).next_to(b, DOWN, buff=0.12) for b in blocks])
        self.play(FadeIn(blocks), FadeIn(loss), FadeIn(loss_t), FadeIn(img), FadeIn(state), run_time=1)

        fwd = Arrow(blocks.get_left() + UP * 0.85, loss.get_right() + UP * 0.85 + RIGHT * 0.0, color=INFO, buff=0, stroke_width=4)
        fwd_t = label("forward: every layer runs", 20, INFO).next_to(fwd, UP, buff=0.08)
        self.play(GrowArrow(fwd), FadeIn(fwd_t), run_time=0.9)

        def make_trainable(i):
            return [blocks[i][0].animate.set_stroke(SUCCESS).set_fill(SUCCESS, 0.25),
                    Transform(state[i], label("trainable", 17, SUCCESS).next_to(blocks[i], DOWN, buff=0.12))]

        mode = label("Feature extraction: freeze all, train fc", 26, HIGHLIGHT).move_to(DOWN * 1.35)
        cnt = label("trainable: 1,026 of 11,177,538 (0.009%)", 24, FG, mono=True).next_to(mode, DOWN, buff=0.2)
        self.play(FadeIn(mode), *make_trainable(5), run_time=0.8)
        back = Arrow(loss.get_bottom() + DOWN * 0.35 + RIGHT * 0.2, blocks[5].get_bottom() + DOWN * 0.55 + LEFT * 0.4,
                     color=GRAD, buff=0, stroke_width=5)
        stop = label("✕", 30, ERROR).next_to(back.get_end(), LEFT, buff=0.08)
        self.play(GrowArrow(back), FadeIn(cnt), run_time=0.9)
        self.play(FadeIn(stop), run_time=0.3)
        self.wait(1.2)

        mode2 = label("Fine-tuning: also unfreeze layer4 (small LR)", 26, HIGHLIGHT).move_to(mode)
        cnt2 = label("trainable: 8,394,754 of 11,177,538 (75.1%)", 24, FG, mono=True).move_to(cnt)
        back2 = Arrow(back.get_start(), blocks[4].get_bottom() + DOWN * 0.55 + LEFT * 0.4, color=GRAD, buff=0, stroke_width=5)
        stop2 = label("✕", 30, ERROR).next_to(back2.get_end(), LEFT, buff=0.08)
        self.play(Transform(mode, mode2), *make_trainable(4), run_time=0.8)
        self.play(Transform(back, back2), Transform(stop, stop2), Transform(cnt, cnt2), run_time=1)
        self.play(FadeIn(caption_bar("Backward stops at the first frozen layer: stem to layer3 keep their ImageNet weights")))
        self.wait(2.5)


class EarlyStoppingScene(WikiScene):
    """The notebook's fine-tuning run: val loss per epoch, ReduceLROnPlateau(factor 0.5, patience 1),
    EarlyStopping(patience 3, min_delta 0.0005). Trace reproduced with torch.optim (see the page)."""

    def construct(self):
        head = title("ReduceLROnPlateau + early stopping on the fine-tuning run")
        self.add(head)
        vl = [0.0415, 0.0382, 0.0549, 0.0793, 0.0341, 0.0415, 0.0437, 0.0424]
        lrs = ["1e-4", "1e-4", "1e-4", "5e-5", "5e-5", "5e-5", "2.5e-5", "2.5e-5"]
        bad = [0, 0, 1, 2, 0, 1, 2, 3]

        x0, y0, xw, yh = -5.8, -2.2, 7.6, 3.6
        ymin, ymax = 0.03, 0.085
        def pt(e, v):
            return np.array([x0 + (e - 0.5) / 8 * xw, y0 + (v - ymin) / (ymax - ymin) * yh, 0])

        axes = VGroup(Line([x0, y0, 0], [x0 + xw, y0, 0], color=MUTED), Line([x0, y0, 0], [x0, y0 + yh, 0], color=MUTED))
        ticks = VGroup(*[label(str(e), 18, MUTED).move_to(pt(e, ymin) + DOWN * 0.3) for e in range(1, 9)])
        yt = VGroup(*[label(f"{v:.2f}", 16, MUTED).next_to([x0, pt(1, v)[1], 0], LEFT, buff=0.12) for v in (0.04, 0.06, 0.08)])
        xl = label("epoch", 18, MUTED).next_to(ticks, DOWN, buff=0.12)
        yl = label("val loss", 18, MUTED).next_to(axes[1], UP, buff=0.12)
        self.play(Create(axes), FadeIn(ticks), FadeIn(yt), FadeIn(xl), FadeIn(yl), run_time=0.8)

        px = RIGHT * 4.75
        lr_t = label("LR: 1e-4", 26, FG, mono=True).move_to(px + UP * 1.4)
        bad_t = label("bad epochs: 0", 26, FG, mono=True).move_to(px + UP * 0.6)
        best_t = label("best: 0.0415 (ep 1)", 22, MUTED, mono=True).move_to(px + DOWN * 0.15)
        self.play(FadeIn(lr_t), FadeIn(bad_t), FadeIn(best_t), run_time=0.5)

        dots, prev = VGroup(), None
        best_marker = None
        for e in range(1, 9):
            v = vl[e - 1]
            is_best = bad[e - 1] == 0
            col = SUCCESS if is_best else (ERROR if bad[e - 1] else FG)
            d = Dot(pt(e, v), radius=0.09, color=col)
            anims = [FadeIn(d, scale=1.5)]
            if prev is not None:
                anims.append(Create(Line(prev, pt(e, v), color=INFO, stroke_width=3)))
            anims.append(Transform(bad_t, label(f"bad epochs: {bad[e - 1]}", 26, ERROR if bad[e - 1] else FG, mono=True).move_to(bad_t)))
            if is_best:
                anims.append(Transform(best_t, label(f"best: {v:.4f} (ep {e})", 22, SUCCESS, mono=True).move_to(best_t)))
                new_marker = SurroundingRectangle(d, color=SUCCESS, buff=0.08)
                anims.append(Create(new_marker) if best_marker is None else Transform(best_marker, new_marker))
                if best_marker is None:
                    best_marker = new_marker
            if e > 1 and lrs[e - 1] != lrs[e - 2]:
                anims.append(Transform(lr_t, label(f"LR: {lrs[e - 1]}", 26, HIGHLIGHT, mono=True).move_to(lr_t)))
                note = label("2 bad in a row: LR × 0.5", 18, HIGHLIGHT).next_to(d, UP, buff=0.15)
                anims.append(FadeIn(note))
            self.play(*anims, run_time=0.75)
            self.wait(0.25)
            prev = pt(e, v)

        stop = label("patience 3 reached: stop,\nrestore the best checkpoint", 20, ERROR).move_to(px + DOWN * 1.1)
        self.play(FadeIn(stop), Indicate(best_marker, color=SUCCESS), run_time=1)
        self.play(FadeIn(caption_bar("Val losses from the notebook log; LR and stop decisions reproduced with torch.optim")))
        self.wait(2.5)
