"""Animations for the AI Engineering page "Working with LLM APIs and Prompt Engineering" (keys llm-api/...).
Every number comes from a real run of Qwen/Qwen2.5-0.5B-Instruct on CPU (transformers 5.22.2, torch 2.15, 2 threads):
the chat template and its 35-token count, the next-token distributions (data/llm_api_sampling.json), and the measured
streaming and prefix-cache timings quoted on the page."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from style import *

HERE = os.path.dirname(os.path.abspath(__file__))


def card(text, color, w, h, size=17, fill=0.16):
    r = RoundedRectangle(corner_radius=0.1, width=w, height=h, stroke_color=color, stroke_width=2).set_fill(color, fill)
    t = label(text, size, FG)
    if t.width > w - 0.3:
        t.scale_to_fit_width(w - 0.3)
    return VGroup(r, t.move_to(r))


class ChatTemplateAssembly(WikiScene):
    """A list of role-tagged messages is rendered by the model's chat template into ONE string of tokens."""
    def construct(self):
        self.add(title("One chat request becomes one token string"))
        sys_card = card("system: You are a support assistant for an Indian telecom. Answer in one sentence.", PURPLE, 12.6, 0.62, 18)
        usr_card = card("user: My data stopped working after recharge.", INFO, 12.6, 0.62, 18)
        msgs = VGroup(sys_card, usr_card).arrange(DOWN, buff=0.18).move_to([0, 1.9, 0])
        head = label("messages = [ ... ]   (what your code sends)", 17, MUTED).next_to(msgs, UP, buff=0.14)
        self.play(FadeIn(head), FadeIn(msgs, shift=DOWN * 0.2), run_time=0.9)
        arrow = Arrow([-4.2, 1.1, 0], [-4.2, 0.55, 0], color=MUTED, stroke_width=3, buff=0)
        tmpl = label("tokenizer.apply_chat_template(messages, add_generation_prompt=True)", 16, HIGHLIGHT, mono=True).next_to(arrow, RIGHT, buff=0.2)
        if tmpl.get_right()[0] > 6.8:
            tmpl.scale_to_fit_width(6.8 - tmpl.get_left()[0])
        self.play(GrowArrow(arrow), FadeIn(tmpl), run_time=0.7)
        lines = [
            [("<|im_start|>", HIGHLIGHT), ("system", PURPLE)],
            [("You are a support assistant for an Indian telecom. Answer in one sentence.", FG), ("<|im_end|>", HIGHLIGHT)],
            [("<|im_start|>", HIGHLIGHT), ("user", INFO)],
            [("My data stopped working after recharge.", FG), ("<|im_end|>", HIGHLIGHT)],
            [("<|im_start|>", HIGHLIGHT), ("assistant", SUCCESS)],
        ]
        rendered = VGroup()
        for i, parts in enumerate(lines):
            row = VGroup(*[label(t, 17, c, mono=True) for t, c in parts]).arrange(RIGHT, buff=0.05)
            if row.width > 12.8:
                row.scale_to_fit_width(12.8)
            row.move_to([0, 0.25 - 0.42 * i, 0]).align_to([-6.4, 0, 0], LEFT)
            rendered.add(row)
        for row in rendered:
            self.play(FadeIn(row, shift=RIGHT * 0.2), run_time=0.45)
        gen = label("<- the model continues from here, and stops when it writes <|im_end|>", 16, SUCCESS)
        gen.next_to(rendered[-1], RIGHT, buff=0.3)
        if gen.get_right()[0] > 6.9:
            gen.scale_to_fit_width(6.9 - gen.get_left()[0])
        self.play(FadeIn(gen), run_time=0.5)
        cap = caption_bar("Qwen2.5-0.5B-Instruct: this request is 35 prompt tokens. Each <|...|> marker is ONE special token")
        self.play(FadeIn(cap), run_time=0.5)
        self.wait(1.2)
        cap2 = caption_bar("Leave out the system message and Qwen's template inserts its own: You are Qwen, created by Alibaba Cloud...")
        self.play(Transform(cap, cap2), run_time=0.5)
        self.wait(3)


class SamplingFilters(WikiScene):
    """Temperature reshapes the next-token distribution; top-p keeps a set whose size adapts to the model's certainty."""
    def construct(self):
        D = json.load(open(os.path.join(HERE, "data", "llm_api_sampling.json")))
        op, cl = D["open"], D["closed"]
        self.add(title("Temperature reshapes, top-p truncates"))
        q = label("Q: " + op["q"], 18, HIGHLIGHT).move_to([0, 2.75, 0])
        self.add(q)
        x0, scale, n = -2.2, 7.0, 8

        def bars(probs, tokens, keep_count=None):
            g = VGroup()
            for j in range(n):
                p = probs[j]
                w = max(scale * p, 0.04)
                kept = keep_count is None or j < keep_count
                col = SUCCESS if j == 0 else (INFO if kept else MUTED)
                y = 2.05 - 0.5 * j
                bar = Rectangle(width=w, height=0.36, stroke_width=0).set_fill(col, 0.85 if kept else 0.3)
                bar.move_to([x0 + w / 2, y, 0])
                tl = label(tokens[j].replace(" ", "\u00b7"), 18, FG, mono=True).next_to([x0, y, 0], LEFT, buff=0.2)
                pl = label(f"{p:.4f}", 17, col).next_to(bar, RIGHT, buff=0.15)
                g.add(VGroup(bar, tl, pl))
            return g

        info = None
        cur = None
        for T in ["0.5", "1.0", "1.5"]:
            s = op["T"][T]
            b = bars(s["probs"], op["tokens"])
            t = VGroup(
                label(f"temperature T = {T}", 22, PURPLE),
                label(f"top-p 0.9 would keep {s['top_p_0.9']:,} tokens", 22, SUCCESS),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([0, -2.15, 0])
            if cur is None:
                self.play(FadeIn(b), FadeIn(t), run_time=0.8)
            else:
                self.play(ReplacementTransform(cur, b), ReplacementTransform(info, t), run_time=1.0)
            cur, info = b, t
            self.wait(1.3)
        cap = caption_bar("open question: many good next tokens, so the nucleus is wide and grows fast with T")
        self.play(FadeIn(cap), run_time=0.4)
        self.wait(1.2)
        q2 = label("Q: " + cl["q"], 18, HIGHLIGHT).move_to(q)
        s = cl["T"]["1.0"]
        b = bars(s["probs"], cl["tokens"], keep_count=s["top_p_0.9"])
        t = VGroup(
            label("temperature T = 1.0", 22, PURPLE),
            label(f"top-p 0.9 keeps {s['top_p_0.9']} token: Yes", 22, SUCCESS),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to([0, -2.15, 0])
        cap2 = caption_bar("closed question: one token holds 0.9852 of the mass, so the same top-p 0.9 keeps exactly 1 (top-k would keep k)")
        self.play(Transform(q, q2), ReplacementTransform(cur, b), ReplacementTransform(info, t), Transform(cap, cap2), run_time=1.0)
        self.wait(3)


class StreamingTimeline(WikiScene):
    """Streaming does not make generation faster; it shows the first token after TTFT instead of after the whole answer."""
    def construct(self):
        TTFT, TPOT, N, TOTAL = 0.213, 0.0824, 60, 5.08
        self.add(title("Streaming: same total time, first words 24x sooner"))
        ax = NumberLine(x_range=[0, 5.5, 0.5], length=10.2, include_numbers=True, font_size=20, color=MUTED,
                        numbers_to_include=[0, 1, 2, 3, 4, 5]).move_to([1.2, -0.6, 0])
        unit = label("time (s)", 16, MUTED).next_to(ax, DOWN, buff=0.45).align_to(ax, RIGHT)
        self.play(Create(ax), FadeIn(unit), run_time=0.7)
        y_stream, y_block = 1.55, 0.35
        l1 = label("streaming", 19, SUCCESS).move_to([0, y_stream, 0]).align_to([-4.25, 0, 0], RIGHT)
        l2 = label("non-streaming", 19, ERROR).move_to([0, y_block, 0]).align_to([-4.25, 0, 0], RIGHT)
        self.play(FadeIn(l1), FadeIn(l2), run_time=0.4)
        x = lambda t: ax.n2p(t)[0]
        prefill = Rectangle(width=x(TTFT) - x(0), height=0.36, stroke_width=0).set_fill(PURPLE, 0.7)
        prefill.move_to([(x(0) + x(TTFT)) / 2, y_stream, 0])
        ttft_l = label("TTFT 213 ms (prefill)", 16, PURPLE).next_to(prefill, UP, buff=0.12).align_to(prefill, LEFT)
        self.play(FadeIn(prefill), FadeIn(ttft_l), run_time=0.6)
        ticks = VGroup(*[Line([x(TTFT + k * TPOT), y_stream - 0.18, 0], [x(TTFT + k * TPOT), y_stream + 0.18, 0],
                              color=SUCCESS, stroke_width=2) for k in range(N)])
        self.play(LaggedStart(*[FadeIn(t) for t in ticks], lag_ratio=0.08), run_time=3.5)
        tpot_l = label("one token every 82.4 ms (decode), 60 tokens", 16, SUCCESS).next_to(ticks, UP, buff=0.12).shift(RIGHT * 0.6)
        self.play(FadeIn(tpot_l), run_time=0.4)
        wait_bar = Rectangle(width=x(TOTAL) - x(0), height=0.36, stroke_width=0).set_fill(MUTED, 0.25)
        wait_bar.move_to([(x(0) + x(TOTAL)) / 2, y_block, 0])
        done = Rectangle(width=0.12, height=0.36, stroke_width=0).set_fill(ERROR, 0.9).move_to([x(TOTAL), y_block, 0])
        nothing = label("user sees nothing ...", 16, MUTED).move_to(wait_bar)
        self.play(FadeIn(wait_bar), FadeIn(nothing), run_time=0.6)
        self.play(FadeIn(done), run_time=0.4)
        f = math(r"T_{total} = 0.213 + 59 \times 0.0824 \approx 5.08\ \text{s}", 34, HIGHLIGHT).move_to([0, -1.75, 0])
        self.play(Write(f), run_time=1.0)
        cap = caption_bar("measured on CPU: 28-token prompt; a 2,227-token prompt raised TTFT to 6.3 s, a cached prefix cut it to 0.27 s")
        self.play(FadeIn(cap), run_time=0.5)
        self.wait(3)
