"""Animations for the AI Engineering page "Tool Calling and the ReAct Agent Loop" (keys agent/...).
Every number and string comes from a real run of Qwen/Qwen2.5-1.5B-Instruct (bfloat16, CPU, greedy) on the page's
telecom tools (data/agent_tool_loop.json): the native tool loop, the ReAct transcript and the compound-reliability values."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from style import *

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "data", "agent_tool_loop.json"), encoding="utf-8"))


def box(text, color, w, h, size=18, fill=0.16):
    r = RoundedRectangle(corner_radius=0.12, width=w, height=h, stroke_color=color, stroke_width=2).set_fill(color, fill)
    t = label(text, size, FG)
    if t.width > w - 0.3:
        t.scale_to_fit_width(w - 0.3)
    return VGroup(r, t.move_to(r))


def fit(m, max_w):
    if m.width > max_w:
        m.scale_to_fit_width(max_w)
    return m


class ToolCallLoop(WikiScene):
    """The model only proposes a call; your code runs it and sends the result back; the loop ends when no call is made."""
    def construct(self):
        L = D["loop"]
        self.add(title("The tool-calling loop: the model asks, your code acts"))
        model_box = box("LLM", PURPLE, 2.6, 1.0, 24).move_to([-4.6, 0.9, 0])
        app_box = box("your code (agent loop)", INFO, 3.6, 1.0, 20).move_to([0.2, 0.9, 0])
        tool_box = box("check_outage()  (backend)", SUCCESS, 3.4, 1.0, 18).move_to([4.7, 0.9, 0])
        self.play(FadeIn(model_box), FadeIn(app_box), FadeIn(tool_box), run_time=0.8)
        q = fit(label("user: " + L["question"], 17, HIGHLIGHT), 12.8).move_to([0, 2.45, 0])
        self.play(FadeIn(q, shift=DOWN * 0.2), run_time=0.6)
        tok_lbl = label(f"step 1 prompt: {L['prompt_tokens'][0]} tokens (5 tool schemas included)", 17, MUTED).move_to([0, -0.85, 0])
        self.play(FadeIn(tok_lbl), run_time=0.4)
        a1 = Arrow(model_box.get_right(), app_box.get_left(), color=PURPLE, buff=0.1, stroke_width=4)
        call = fit(label("tool_call: " + L["call"], 17, PURPLE, mono=True), 9.0).move_to([0, -0.35, 0])
        self.play(GrowArrow(a1), FadeIn(call), run_time=0.8)
        a2 = Arrow(app_box.get_right(), tool_box.get_left(), color=INFO, buff=0.1, stroke_width=4)
        self.play(GrowArrow(a2), run_time=0.5)
        res = fit(label("result: " + L["result"], 16, SUCCESS), 12.6).move_to([0, -1.35, 0])
        self.play(FadeIn(res, shift=UP * 0.2), run_time=0.6)
        back = CurvedArrow(app_box.get_bottom() + LEFT * 0.6, model_box.get_bottom() + RIGHT * 0.2, color=SUCCESS, angle=-0.6)
        tok2 = label(f"step 2 prompt: {L['prompt_tokens'][1]} tokens (history + result appended)", 17, MUTED).move_to(tok_lbl)
        self.play(Create(back), Transform(tok_lbl, tok2), run_time=0.9)
        ans = fit(label("final answer: " + L["answer_bad"], 17, FG), 12.6).move_to([0, -1.9, 0])
        self.play(FadeIn(ans), run_time=0.6)
        self.wait(0.8)
        bad = label("100 and 10 appear in no tool result: get_balance was never called", 18, ERROR).move_to([0, -2.45, 0])
        self.play(FadeIn(bad), Indicate(ans, color=ERROR), run_time=0.9)
        cap = caption_bar("the loop ends when the model stops calling tools, even if it skipped one; check answers against tool results")
        self.play(FadeIn(cap), run_time=0.4)
        self.wait(3)


class ReactTrace(WikiScene):
    """ReAct writes its reasoning as text; your code runs each Action and pastes the Observation back."""
    def construct(self):
        steps, counts = D["react_steps"], D["react_prompt_tokens"]
        self.add(title("ReAct: Thought, Action, Observation, repeat"))
        y = 2.35
        counter = label(f"prompt tokens: {counts[0]}", 19, HIGHLIGHT).move_to([4.9, 3.05 - 0.55, 0])
        self.play(FadeIn(counter), run_time=0.4)
        colors = {"Thought": PURPLE, "Action": INFO, "Observation": SUCCESS, "Final Answer": HIGHLIGHT}
        shown = VGroup()
        for i, st in enumerate(steps):
            if i > 0:
                new = label(f"prompt tokens: {counts[i]}", 19, HIGHLIGHT).move_to(counter)
                self.play(Transform(counter, new), run_time=0.4)
            for line in st:
                key = line.split(":")[0]
                col = colors.get(key, FG)
                who = "  <- written by your code" if key == "Observation" else ""
                import textwrap
                parts = textwrap.wrap(line + who, 92 if key in ("Action", "Observation") else 110)
                t = VGroup(*[fit(label(x, 15, col, mono=key in ("Action", "Observation")), 13.2) for x in parts]).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
                t.align_to([-6.6, 0, 0], LEFT).align_to([0, y, 0], UP)
                shown.add(t)
                self.play(FadeIn(t, shift=RIGHT * 0.2), run_time=0.55)
                y = t.get_bottom()[1] - 0.14
            y -= 0.16
        cap = caption_bar("stop generation at Observation: so the model cannot invent tool results; each round re-sends the whole transcript")
        self.play(FadeIn(cap), run_time=0.4)
        self.wait(3)


class CompoundReliability(WikiScene):
    """Per-step reliability multiplies: 95 percent per step is a coin flip after about 14 steps."""
    def construct(self):
        self.add(title("Small per-step errors compound over many steps"))
        ax = Axes(x_range=[0, 100, 10], y_range=[0, 1, 0.2], x_length=10.5, y_length=4.6,
                  axis_config={"color": MUTED, "include_numbers": True, "font_size": 20}).move_to([0.3, -0.25, 0])
        xl = label("steps K", 17, MUTED).next_to(ax, DOWN, buff=0.15).align_to(ax, RIGHT)
        yl = label("P(all steps right)", 17, MUTED).next_to(ax, UP, buff=0.1).align_to(ax, LEFT)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), run_time=0.8)
        c95 = ax.plot(lambda k: 0.95 ** k, x_range=[0, 100], color=ERROR, stroke_width=4)
        c99 = ax.plot(lambda k: 0.99 ** k, x_range=[0, 100], color=SUCCESS, stroke_width=4)
        l95 = label("p = 0.95 per step", 18, ERROR).move_to(ax.c2p(36, 0.36))
        l99 = label("p = 0.99 per step", 18, SUCCESS).move_to(ax.c2p(72, 0.62)).shift(UP * 0.55)
        self.play(Create(c95), FadeIn(l95), run_time=1.6)
        self.play(Create(c99), FadeIn(l99), run_time=1.6)
        c = D["compound"]
        dot = Dot(ax.c2p(10, c["0.95"][2]), color=ERROR, radius=0.09)
        t10 = label(f"K = 10: {c['0.95'][2]:.4f}", 18, ERROR).next_to(dot, RIGHT, buff=0.15)
        self.play(FadeIn(dot), FadeIn(t10), run_time=0.6)
        d2 = Dot(ax.c2p(100, c["0.95"][3]), color=ERROR, radius=0.09)
        t100 = label(f"K = 100: {c['0.95'][3]:.4f}", 18, ERROR).next_to(d2, UP, buff=0.2).shift(LEFT * 0.9)
        self.play(FadeIn(d2), FadeIn(t100), run_time=0.6)
        d3 = Dot(ax.c2p(100, c["0.99"][3]), color=SUCCESS, radius=0.09)
        t3 = label(f"K = 100: {c['0.99'][3]:.3f}", 18, SUCCESS).next_to(d3, UP, buff=0.2).shift(LEFT * 0.9)
        self.play(FadeIn(d3), FadeIn(t3), run_time=0.6)
        cap = caption_bar("P = p to the power K: fewer steps, better tools and checks between steps matter more than a smarter prompt")
        self.play(FadeIn(cap), run_time=0.4)
        self.wait(3)
