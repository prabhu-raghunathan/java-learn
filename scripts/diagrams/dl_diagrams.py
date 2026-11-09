"""Block diagrams for the Deep Learning pages (docs/03-ai-ml/03-deep-learning/).

    python scripts/diagrams/dl_diagrams.py            # write every SVG
    python scripts/diagrams/dl_diagrams.py lstm-cell  # write one

Output: static/img/deep-learning/diagrams/<name>.svg, embedded in pages as
    ![alt text](/img/deep-learning/diagrams/<name>.svg)
Symbols follow each page's notation; any number drawn comes from that page's verified results.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blockdiag import *  # noqa: E402,F403

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "static", "img", "deep-learning", "diagrams")
DIAGRAMS = {}


def diagram(name):
    def reg(fn):
        DIAGRAMS[name] = fn
        return fn
    return reg


DIM = 0.16   # opacity of the parts that are not highlighted


# =============================================================================
# LSTM cell (Colah-style), with optional highlighted step
# =============================================================================
def lstm_cell(d, ox=0, oy=0, hl=None, labels=True):
    """Draw an LSTM cell. hl: None (all), 'forget', 'input', 'cell', 'output', 'cellstate'."""
    X = lambda v: ox + v   # noqa: E731
    Y = lambda v: oy + v   # noqa: E731
    top, bot = Y(120), Y(310)                 # cell-state line, input bundle line
    gx = {"f": X(250), "i": X(340), "c": X(420), "o": X(510)}
    gy = Y(250)
    on = lambda *parts: 1.0 if hl is None or hl in parts else DIM  # noqa: E731

    d.cell(X(120), Y(70), 540, 290)

    # --- input bundle: h_{t-1} from the left, x_t from below -------------------------
    with d.group(on("forget", "input", "output")):
        d.line([(X(30), bot), (gx["o"], bot)], arrow=False)
        d.line([(X(185), Y(420)), (X(185), bot)], arrow=False)
    for k, step in (("f", "forget"), ("i", "input"), ("c", "input"), ("o", "output")):
        with d.group(on(step)):
            d.line([(gx[k], bot), (gx[k], gy + 17)])
    with d.group(on("forget", "input", "output")):
        d.node(X(185), Y(440), "x_t", "in")
    with d.group(on("forget", "input", "output")):
        d.text(X(40), bot - 20, "h_{t-1}", 20, "start")

    # --- cell-state line ------------------------------------------------------------
    with d.group(on("forget", "cell", "cellstate")):
        d.line([(X(30), top), (gx["f"] - 15, top)], arrow=False)
        d.text(X(40), top - 22, "c_{t-1}", 20, "start")
    with d.group(on("cell", "cellstate", "output")):
        d.line([(gx["f"] + 15, top), (gx["c"] - 15, top)], arrow=False)
        d.line([(gx["c"] + 15, top), (X(735), top)])
        d.text(X(705), top - 22, "c_t", 20, "start")

    # --- forget gate ------------------------------------------------------------------
    with d.group(on("forget")):
        d.layer(gx["f"], gy, "σ")
        d.line([(gx["f"], gy - 17), (gx["f"], top + 16)])
        if labels:
            d.text(gx["f"] + 10, Y(185), "f_t", 18, "start")
    with d.group(on("forget", "cell", "cellstate")):
        d.op(gx["f"], top, "×")

    # --- input gate + candidate ------------------------------------------------------
    with d.group(on("input")):
        d.layer(gx["i"], gy, "σ")
        d.layer(gx["c"], gy, "tanh", w=58)
        d.line([(gx["i"], gy - 17), (gx["i"], Y(185)), (gx["c"] - 16, Y(185))])
        d.line([(gx["c"], gy - 17), (gx["c"], Y(185) + 16)])
        if labels:
            d.text(gx["i"] - 10, Y(210), "i_t", 18, "end")
            d.text(gx["c"] + 34, Y(212), "c~_t", 18, "start")
    with d.group(on("input", "cell")):
        d.op(gx["c"], Y(185), "×")
        d.line([(gx["c"], Y(185) - 16), (gx["c"], top + 16)])
    with d.group(on("cell", "cellstate")):
        d.op(gx["c"], top, "+")

    # --- output gate --------------------------------------------------------------------
    hx = X(585)
    with d.group(on("output")):
        d.layer(gx["o"], gy, "σ")
        d.line([(gx["o"], gy - 17), (gx["o"], Y(205)), (hx - 16, Y(205))])
        d.line([(hx, top), (hx, Y(160) - 14)])
        d.op(hx, Y(160), "tanh", w=54)
        d.line([(hx, Y(160) + 14), (hx, Y(205) - 16)])
        d.op(hx, Y(205), "×")
        if labels:
            d.text(gx["o"] + 10, Y(186), "o_t", 18, "start")
        # h_t: to the next step (right) and up out of the cell
        d.line([(hx, Y(205) + 16), (hx, bot), (X(735), bot)])
        d.line([(X(632), bot), (X(632), Y(48))], radius=10)
        d.node(X(632), Y(28), "h_t", "out", r=22, size=20)
        d.text(X(705), bot - 20, "h_t", 20, "start")


EQ = {
    "forget": ["f_t = σ(W_f x_t + U_f h_{t-1} + b_f)"],
    "input": ["i_t = σ(W_i x_t + U_i h_{t-1} + b_i)", "c~_t = tanh(W_c x_t + U_c h_{t-1} + b_c)"],
    "cell": ["c_t = f_t ⊙ c_{t-1} + i_t ⊙ c~_t"],
    "output": ["o_t = σ(W_o x_t + U_o h_{t-1} + b_o)", "h_t = o_t ⊙ tanh(c_t)"],
    "cellstate": ["c_t = f_t ⊙ c_{t-1} + i_t ⊙ c~_t", "∂c_t / ∂c_{t-1} ≈ diag(f_t)"],
}
NOTE = {
    "forget": ["The forget gate looks at h_{t-1} and x_t", "and outputs a number in (0, 1) for every",
               "memory dimension: 1 keeps, 0 erases."],
    "input": ["σ decides HOW MUCH to write (0 to 1);", "tanh proposes WHAT to write (−1 to 1).",
              "Their product is the new content."],
    "cell": ["Scale the old memory by f_t, add the", "gated new content. Only × and + touch",
             "the cell-state line: no weight matrix."],
    "output": ["σ picks which memory dimensions to show;", "tanh squashes c_t into (−1, 1); the product",
               "is h_t, sent up as output and right to t+1."],
    "cellstate": ["The top line is the conveyor belt: memory", "and gradient pass straight along it,",
                  "scaled only by the forget gate."],
}
TITLE = {"forget": "Step 1 · Forget gate: what to keep", "input": "Step 2 · Input gate and candidate: what to write",
         "cell": "Step 3 · Cell update: keep some old, add some new",
         "output": "Step 4 · Output gate: what to reveal", "cellstate": "The cell state: an additive memory path"}


@diagram("lstm-cell")
def _lstm_cell():
    d = Diagram(780, 560, "LSTM cell")
    lstm_cell(d, 10, 20)
    legend(d, 40, 528)
    return d


def _lstm_step(step):
    d = Diagram(1200, 500, TITLE[step])
    lstm_cell(d, 0, 10, hl=step)
    d.note(760, 60, 420, 330)
    d.text(970, 92, TITLE[step], 16, weight="bold")
    y = 145
    for eq in EQ[step]:
        d.text(782, y, eq, 19, "start")
        y += 40
    y += 10
    for ln in NOTE[step]:
        d.text(782, y, ln, 15, "start", MUTED)
        y += 24
    return d


for _s in ("forget", "input", "cell", "output", "cellstate"):
    diagram(f"lstm-step-{_s}")(lambda s=_s: _lstm_step(s))


# =============================================================================
# GRU cell (notebook / PyTorch convention: h_t = z ⊙ h_{t-1} + (1 − z) ⊙ h~_t)
# =============================================================================
@diagram("gru-cell")
def _gru():
    d = Diagram(1180, 560, "GRU cell")
    ox, oy = 10, 20
    X = lambda v: ox + v   # noqa: E731
    Y = lambda v: oy + v   # noqa: E731
    top, bun = Y(110), Y(340)
    d.cell(X(110), Y(60), 560, 330)
    xr, xz, xc = X(220), X(310), X(500)
    # top line: h_{t-1} -> (x z) -> (+) -> h_t
    d.text(X(30), top - 22, "h_{t-1}", 20, "start")
    d.line([(X(30), top), (xz - 15, top)], arrow=False)
    d.op(xz, top, "×")
    d.line([(xz + 15, top), (xc - 15, top)], arrow=False)
    d.op(xc, top, "+")
    d.line([(xc + 15, top), (X(750), top)])
    d.text(X(715), top - 22, "h_t", 20, "start")
    d.line([(X(640), top), (X(640), Y(38))], radius=8)
    d.node(X(640), Y(20), "h_t", "out", r=20, size=18)
    # h_{t-1} drops into the gate bundle and to the reset multiply
    d.line([(X(140), top), (X(140), bun), (xz, bun)], arrow=False)
    d.line([(X(140), Y(200)), (xr - 15, Y(200))])
    d.dot(X(140), top)
    d.dot(X(140), Y(200))
    # x_t
    d.node(X(185), Y(470), "x_t", "in")
    d.line([(X(185), Y(444)), (X(185), bun)], arrow=False)
    d.line([(X(185), Y(375)), (xc, Y(375)), (xc, Y(296) + 2)])
    d.dot(X(185), Y(375))
    # reset gate
    d.layer(xr, Y(280), "σ")
    d.line([(xr, bun), (xr, Y(280) + 17)])
    d.line([(xr, Y(280) - 17), (xr, Y(200) + 15)])
    d.op(xr, Y(200), "×")
    d.text(xr + 10, Y(245), "r_t", 18, "start")
    d.line([(xr + 15, Y(200)), (X(400), Y(200)), (X(400), Y(279)), (xc - 30, Y(279))])
    d.text(X(410), Y(232), "r_t ⊙ h_{t-1}", 15, "start", MUTED)
    # update gate
    d.layer(xz, Y(280), "σ")
    d.line([(xz, bun), (xz, Y(280) + 17)])
    d.line([(xz, Y(280) - 17), (xz, top + 15)])
    d.text(xz + 10, Y(245), "z_t", 18, "start")
    d.dot(xz, Y(150))
    d.line([(xz, Y(150)), (X(420) - 15, Y(150))])
    d.op(X(420), Y(150), "1−", size=15)
    d.line([(X(420) + 15, Y(150)), (xc - 15, Y(150))])
    # candidate
    d.layer(xc, Y(280), "tanh", w=58)
    d.line([(xc, Y(280) - 17), (xc, Y(150) + 15)])
    d.text(xc + 36, Y(230), "h~_t", 18, "start")
    d.op(xc, Y(150), "×")
    d.line([(xc, Y(150) - 15), (xc, top + 15)])
    # equations
    d.note(790, 70, 370, 330)
    d.text(975, 100, "GRU: two gates, one state", 17, weight="bold")
    eqs = ["z_t = σ(W_z x_t + U_z h_{t-1} + b_z)", "r_t = σ(W_r x_t + U_r h_{t-1} + b_r)",
           "h~_t = tanh(W_h x_t + U_h (r_t ⊙ h_{t-1}) + b_h)", "h_t = z_t ⊙ h_{t-1} + (1 − z_t) ⊙ h~_t"]
    y = 145
    for e in eqs:
        d.text(805, y, e, 16.5, "start")
        y += 40
    for ln in ["Top line: z_t keeps old state (× on the left),", "1 − z_t lets the candidate in (× on the right).",
               "r_t filters h_{t-1} before it shapes h~_t."]:
        y += 4
        d.text(805, y + 6, ln, 14.5, "start", MUTED)
        y += 22
    legend(d, 40, 528, items=("layer", "op", "flow", "copy"))
    return d



# =============================================================================
# Vanilla RNN
# =============================================================================
@diagram("rnn-cell")
def _rnn_cell():
    d = Diagram(980, 470, "Vanilla RNN cell")
    ox, oy = 90, 60
    d.cell(ox, oy + 40, 330, 210)
    by, ty, cx = oy + 210, oy + 110, ox + 190
    d.text(ox - 70, by - 22, "h_{t-1}", 20, "start")
    d.line([(ox - 70, by), (cx, by), (cx, oy + 175 + 17)], radius=14)
    d.node(ox + 90, oy + 330, "x_t", "in")
    d.line([(ox + 90, oy + 304), (ox + 90, by)], arrow=False)
    d.layer(cx, oy + 175, "tanh", w=62)
    d.line([(cx, oy + 175 - 17), (cx, ty), (ox + 400, ty)], radius=14)
    d.text(ox + 375, ty - 22, "h_t", 19, "start")
    d.text(ox + 375, ty + 24, "to step t+1", 13, "start", MUTED)
    d.line([(ox + 280, ty), (ox + 280, oy - 2)], radius=8)
    d.dot(ox + 280, ty)
    d.node(ox + 280, oy - 24, "h_t", "out", r=22, size=19)
    d.text(cx + 40, oy + 150, "a_t", 16, "start", MUTED)
    d.note(560, 60, 400, 280)
    d.text(760, 92, "One step of a vanilla RNN", 17, weight="bold")
    d.text(578, 140, "h_t = tanh(W_x x_t + W_h h_{t-1} + b)", 19, "start")
    d.text(578, 180, "y_t = W_y h_t + b_y   (if an output is needed)", 16, "start")
    for i, ln in enumerate(["h_{t-1} and x_t merge (concatenate) and pass", "through ONE tanh layer: a_t is the sum inside.",
                            "The same h_t goes up (output) and right", "(memory for the next step). There is no",
                            "separate memory line: every step rewrites h."]):
        d.text(578, 225 + 24 * i, ln, 15, "start", MUTED)
    legend(d, 40, 438, items=("layer", "flow", "copy", "concat"))
    return d


@diagram("rnn-unrolled")
def _rnn_unrolled():
    d = Diagram(1180, 460, "RNN unrolled through time")
    # folded cell with loop
    fx, fy = 50, 120
    d.cell(fx, fy, 120, 120)
    d.layer(fx + 60, fy + 60, "tanh", w=56)
    d.node(fx + 60, fy + 230, "x_t", "in", r=22, size=19)
    d.line([(fx + 60, fy + 208), (fx + 60, fy + 77)])
    d.line([(fx + 60, fy + 43), (fx + 60, fy - 40)])
    d.node(fx + 60, fy - 62, "h_t", "out", r=22, size=19)
    d.curve((fx + 120, fy + 40), (fx + 175, fy + 10), (fx + 175, fy + 120), (fx + 122, fy + 92))
    d.text(fx + 182, fy + 66, "W_h", 15, "start", MUTED)
    d.text(285, fy + 60, "=", 40, weight="bold")
    words = ["I", "love", "deep", "learning"]
    x0, step, cw = 360, 205, 130
    d.text(x0 - 40, fy + 32, "h_0 = 0", 15, "middle", MUTED)
    for t in range(4):
        cx = x0 + t * step
        d.cell(cx, fy, cw, 120)
        d.layer(cx + cw / 2, fy + 60, "tanh", w=56)
        d.node(cx + cw / 2, fy + 230, f"x_{t + 1}", "in", r=22, size=19)
        d.text(cx + cw / 2, fy + 272, f"“{words[t]}”", 15, color=MUTED)
        d.line([(cx + cw / 2, fy + 208), (cx + cw / 2, fy + 77)])
        d.line([(cx + cw / 2, fy + 43), (cx + cw / 2, fy - 40)])
        d.node(cx + cw / 2, fy - 62, f"h_{t + 1}", "out", r=22, size=19)
        src = x0 - 40 if t == 0 else cx - step + cw
        d.line([(src, fy + 60), (cx + cw / 2 - 30, fy + 60)] if t == 0 else
               [(src - cw / 2 + 28, fy + 60), (cx + cw / 2 - 30, fy + 60)])
        d.text(cx - 38, fy + 42, "W_h", 14, "middle", MUTED) if t else None
    d.text(590, 432, "the same W_x, W_h, b in every copy:  h_t = tanh(W_x x_t + W_h h_{t-1} + b)", 16, color=MUTED)
    return d


@diagram("rnn-patterns")
def _rnn_patterns():
    d = Diagram(1180, 360, "RNN input-output patterns")
    panels = [("one-to-many", "image captioning", [1, 0, 0], [0, 1, 1, 1][:3]),
              ("many-to-one", "sentiment", [1, 1, 1], [0, 0, 1]),
              ("many-to-many (aligned)", "tagging, next char", [1, 1, 1], [1, 1, 1]),
              ("encoder–decoder", "translation", [1, 1, 0, 0], [0, 0, 1, 1])]
    px = 20
    for name, ex, ins, outs in panels:
        n = len(ins)
        cw, gap = 44, 26
        width = n * cw + (n - 1) * gap
        pw = max(width + 40, 250)
        x0 = px + (pw - width) / 2
        d.text(px + pw / 2, 36, name, 17, weight="bold")
        d.text(px + pw / 2, 60, ex, 14, color=MUTED)
        for k in range(n):
            cx = x0 + k * (cw + gap)
            d.rect(cx, 150, cw, 56, "#e2f0d5", "#7fae5b", rx=10, sw=2.2)
            if k:
                d.line([(cx - gap, 178), (cx, 178)], head=9, width=2.6)
            if ins[k]:
                d.parts.append(f'<circle cx="{cx + cw / 2}" cy="{270}" r="15" fill="#a8cbee" stroke="#3b78bd" stroke-width="2"/>')
                d.line([(cx + cw / 2, 255), (cx + cw / 2, 208)], head=9, width=2.6)
            if outs[k]:
                d.parts.append(f'<circle cx="{cx + cw / 2}" cy="{98}" r="15" fill="#dcb6ee" stroke="#8a4fae" stroke-width="2"/>')
                d.line([(cx + cw / 2, 148), (cx + cw / 2, 115)], head=9, width=2.6)
        px += pw + 22
    d.parts.append('<circle cx="380" cy="330" r="10" fill="#a8cbee" stroke="#3b78bd" stroke-width="2"/>')
    d.text(396, 330, "input", 14, "start", MUTED)
    d.rect(470, 320, 26, 20, "#e2f0d5", "#7fae5b", rx=5, sw=2)
    d.text(504, 330, "RNN step (same weights)", 14, "start", MUTED)
    d.parts.append('<circle cx="700" cy="330" r="10" fill="#dcb6ee" stroke="#8a4fae" stroke-width="2"/>')
    d.text(716, 330, "output", 14, "start", MUTED)
    return d


@diagram("rnn-bptt")
def _rnn_bptt():
    d = Diagram(1180, 470, "Backpropagation through time")
    x0, step, cw, fy = 110, 250, 130, 150
    d.text(x0 - 60, fy + 60, "h_0", 18)
    for t in range(4):
        cx = x0 + t * step
        d.cell(cx, fy, cw, 120)
        d.layer(cx + cw / 2, fy + 60, "tanh", w=56)
        d.node(cx + cw / 2, fy + 220, f"x_{t + 1}", "in", r=22, size=19)
        d.line([(cx + cw / 2, fy + 198), (cx + cw / 2, fy + 77)])
        d.line([(cx + cw / 2, fy + 43), (cx + cw / 2, fy - 50)])
        d.op(cx + cw / 2, fy - 66, f"L_{t + 1}", r=17, size=15)
        start = x0 - 40 if t == 0 else cx - step + cw / 2 + 30
        d.line([(start, fy + 60), (cx + cw / 2 - 30, fy + 60)])
        if t:
            # backward: from cell t to cell t-1, below the forward arrow
            d.line([(cx + 10, fy + 100), (cx - step + cw - 10, fy + 100)], color=GRAD, width=2.8, dash="8 5")
            d.text(cx - (step - cw) / 2, fy + 125, "× diag(1 − h_t^2) W_h", 14, color=GRAD)
        d.line([(cx + cw / 2 + 22, fy - 52), (cx + cw - 12, fy - 10), (cx + cw - 12, fy + 30)],
               color=GRAD, width=2.6, dash="8 5", radius=8)
    d.text(590, 40, "forward: black, left to right · backward: red, right to left, one factor per step", 16, color=MUTED)
    d.text(590, 440, "∂L/∂W_h = Σ_t (contribution of step t): the shared weight collects a gradient from every step", 16, color=MUTED)
    return d


@diagram("rnn-many-to-one-model")
def _rnn_m2o():
    d = Diagram(1180, 470, "Many-to-one RNN for sentiment")
    words = ["good", "bad", "good"]
    x0, step, cw, fy = 150, 230, 130, 150
    d.text(x0 - 75, fy + 60, "h_0 = 0", 16, color=MUTED)
    for t, w in enumerate(words):
        cx = x0 + t * step
        d.cell(cx, fy, cw, 120)
        d.layer(cx + cw / 2, fy + 60, "RNN", w=60)
        d.layer(cx + cw / 2, fy + 185, "Embedding", w=110, size=15)
        d.line([(cx + cw / 2, fy + 168), (cx + cw / 2, fy + 77)])
        d.node(cx + cw / 2, fy + 250, "", "in", r=16)
        d.text(cx + cw / 2, fy + 284, f"“{w}”", 15, color=MUTED)
        d.line([(cx + cw / 2, fy + 234), (cx + cw / 2, fy + 202)])
        src = x0 - 40 if t == 0 else cx - step + cw / 2 + 30
        d.line([(src, fy + 60), (cx + cw / 2 - 30, fy + 60)])
        d.text(cx + cw / 2 + 30, fy + 40, f"h_{t + 1}", 15, "start", MUTED)
    lx = x0 + 2 * step + cw / 2 + 30
    d.line([(lx, fy + 60), (lx + 120, fy + 60)])
    d.text(lx + 50, fy + 40, "h_T", 17)
    d.layer(lx + 175, fy + 60, "Linear", w=96)
    d.line([(lx + 223, fy + 60), (lx + 290, fy + 60)])
    d.node(lx + 316, fy + 60, "z", "out", r=24)
    d.text(lx + 316, fy + 104, "logit", 14, color=MUTED)
    d.line([(lx + 316, fy + 36), (lx + 316, fy - 20)])
    d.layer(lx + 316, fy - 40, "BCEWithLogitsLoss", w=180, size=15)
    d.text(590, 40, "Read the whole sequence, decide ONCE from the final hidden state:  y = W_{hy} h_T + b_y", 16, color=MUTED)
    return d


# =============================================================================
# Character-level LSTM (page 2 of LSTM)
# =============================================================================
@diagram("char-lstm-model")
def _char_model():
    d = Diagram(1180, 520, "Character-level LSTM, unrolled")
    ins, outs = ["\\n", "d", "e", "e"], ["d", "e", "e", "p"]
    x0, step, cw, fy = 120, 245, 150, 190
    d.text(x0 - 60, fy + 50, "h_0, c_0", 15, color=MUTED)
    for t in range(4):
        cx = x0 + t * step
        mid = cx + cw / 2
        d.cell(cx, fy, cw, 100)
        d.text(mid, fy + 50, "LSTM", 20, weight="bold", color="#4d7a32")
        d.layer(mid, fy + 160, "Embedding", w=116, size=15)
        d.line([(mid, fy + 143), (mid, fy + 102)])
        d.node(mid, fy + 245, ins[t], "in", r=22, size=19)
        d.line([(mid, fy + 223), (mid, fy + 177)])
        d.layer(mid, fy - 50, "Linear", w=90, size=16)
        d.line([(mid, fy - 2), (mid, fy - 33)])
        d.node(mid, fy - 135, outs[t], "out", r=22, size=19)
        d.line([(mid, fy - 67), (mid, fy - 113)])
        d.text(mid + 12, fy - 90, "logits z_t", 13, "start", MUTED)
        src = x0 - 40 if t == 0 else cx - step + cw + 2
        d.line([(src, fy + 35), (cx - 2, fy + 35)])
        d.line([(src, fy + 70), (cx - 2, fy + 70)])
    d.text(x0 + 4 * step - 60, fy + 30, "c_t", 14, "start", MUTED)
    d.text(x0 + 4 * step - 60, fy + 75, "h_t", 14, "start", MUTED)
    d.text(590, 480, "input window (B, T) → Embedding (B, T, E) → LSTM (B, T, H) → Linear (B, T, V);  target = input shifted by one", 15, color=MUTED)
    d.text(1010, fy - 135, "targets", 15, "start", MUTED)
    d.text(1010, fy + 245, "inputs", 15, "start", MUTED)
    return d


@diagram("char-lstm-generation")
def _char_gen():
    d = Diagram(1180, 560, "Autoregressive generation with temperature")
    outs = ["l", "e", "a", "r"]
    x0, step, cw, fy = 160, 245, 150, 260
    d.text(x0 - 85, fy + 52, "state after", 14, color=MUTED)
    d.text(x0 - 85, fy + 72, "“deep ”", 14, color=MUTED)
    for t in range(4):
        cx = x0 + t * step
        mid = cx + cw / 2
        d.cell(cx, fy, cw, 100)
        d.text(mid, fy + 50, "LSTM", 20, weight="bold", color="#4d7a32")
        d.layer(mid, fy - 45, "softmax(z / T)", w=130, size=15)
        d.line([(mid, fy - 2), (mid, fy - 28)])
        d.op(mid, fy - 115, "sample", r=17, w=78, size=14)
        d.line([(mid, fy - 62), (mid, fy - 98)])
        d.node(mid, fy - 190, outs[t], "out", r=22, size=19)
        d.line([(mid, fy - 132), (mid, fy - 168)])
        if t < 3:
            nx = mid + step
            d.line([(mid + 22, fy - 190), (mid + 95, fy - 190), (mid + 95, fy + 175), (nx, fy + 175), (nx, fy + 138)],
                   color=IN_STROKE, width=2.8, radius=16)
        if t == 0:
            d.node(mid, fy + 160, "␣", "in", r=22, size=19)
            d.line([(mid, fy + 138), (mid, fy + 102)])
            d.text(mid, fy + 200, "last seed char", 13, color=MUTED)
        else:
            d.line([(mid, fy + 138), (mid, fy + 102)])
            d.parts.append(f'<circle cx="{mid}" cy="{fy + 160}" r="18" fill="#a8cbee" stroke="#3b78bd" stroke-width="2"/>')
            d.text(mid, fy + 160, outs[t - 1], 17)
        src = x0 - 30 if t == 0 else cx - step + cw + 2
        d.line([(src, fy + 50), (cx - 2, fy + 50)])
    d.text(590, 510, "pick a character, feed ONLY it back with the carried (h, c);  T < 1 sharper, T > 1 flatter, T → 0 = greedy", 15, color=MUTED)
    return d


# =============================================================================
# Neurons, graphs, training loop, MLP
# =============================================================================
@diagram("neuron")
def _neuron():
    d = Diagram(1060, 400, "A single neuron")
    for i, (lab, w) in enumerate([("x_1", "w_1"), ("x_2", "w_2"), ("x_3", "w_3")]):
        y = 90 + i * 100
        d.node(90, y, lab, "in", r=26)
        d.line([(116, y), (330, 190)], radius=0)
        d.text(210, y + (190 - y) * 0.45 - 16, w, 17, color="#9c8526", weight="bold")
    d.node(250, 340, "b", "in", r=20, size=18)
    d.line([(266, 328), (334, 210)])
    d.op(360, 190, "Σ", r=28, size=24)
    d.line([(388, 190), (470, 190)])
    d.text(430, 166, "z", 18)
    d.layer(520, 190, "φ", w=64, h=44, size=24)
    d.line([(552, 190), (640, 190)])
    d.node(670, 190, "a", "out", r=28, size=22)
    d.text(330, 138, "weigh + sum + shift", 14, "middle", MUTED)
    d.text(520, 235, "bend (ReLU / σ / tanh)", 14, color=MUTED)
    d.note(730, 70, 310, 250)
    d.text(885, 100, "Weigh, sum, shift, bend", 17, weight="bold")
    d.text(748, 150, "z = w^T x + b = Σ_j w_j x_j + b", 17, "start")
    d.text(748, 190, "a = φ(z)", 19, "start")
    for i, ln in enumerate(["w, b are learned; φ is chosen.", "Without φ the neuron is just", "linear regression."]):
        d.text(748, 235 + 24 * i, ln, 15, "start", MUTED)
    return d


@diagram("sigmoid-neuron-backprop")
def _sig_neuron():
    d = Diagram(1180, 380, "Sigmoid neuron: forward and backward")
    d.node(80, 150, "x = 1", "in", r=34, size=16)
    d.text(80, 205, "w = 0.5, b = 0", 14, color=MUTED)
    d.line([(114, 150), (220, 150)])
    d.op(260, 150, "wx + b", r=24, w=96, size=15)
    d.line([(308, 150), (420, 150)])
    d.text(364, 128, "z = 0.5", 16)
    d.layer(470, 150, "σ", w=60, h=40, size=22)
    d.line([(500, 150), (640, 150)])
    d.text(570, 128, "a = 0.6225", 16)
    d.op(700, 150, "(a − y)²", r=24, w=104, size=15)
    d.line([(752, 150), (850, 150)])
    d.node(890, 150, "L", "out", r=30)
    d.text(890, 200, "0.1425", 15)
    d.text(700, 200, "y = 1", 14, color=MUTED)
    # backward
    d.line([(860, 250), (740, 250)], color=GRAD, width=2.8, dash="8 5")
    d.text(800, 275, "∂L/∂a = 2(a − y) = −0.7551", 14, color=GRAD)
    d.line([(640, 250), (500, 250)], color=GRAD, width=2.8, dash="8 5")
    d.text(570, 275, "× σ(1 − σ) = 0.2350", 14, color=GRAD)
    d.line([(430, 250), (300, 250)], color=GRAD, width=2.8, dash="8 5")
    d.text(365, 275, "∂L/∂z = −0.1774", 14, color=GRAD)
    d.line([(230, 250), (110, 250)], color=GRAD, width=2.8, dash="8 5")
    d.text(170, 275, "∂L/∂w = ∂L/∂b = −0.1774", 14, color=GRAD)
    d.text(590, 340, "forward (black) computes and stores values; backward (red) multiplies one local slope per link", 15, color=MUTED)
    return d


@diagram("comp-graph")
def _comp_graph():
    d = Diagram(1080, 460, "Computation graph f = (a + b)(ab) + ab")
    d.node(90, 130, "a = 2", "in", r=34, size=16)
    d.node(90, 330, "b = 3", "in", r=34, size=16)
    d.op(330, 130, "+", r=22, size=22)
    d.text(330, 92, "c = 5", 16)
    d.op(330, 330, "×", r=22, size=22)
    d.text(330, 378, "d = 6", 16)
    d.op(580, 200, "×", r=22, size=22)
    d.text(580, 162, "e = c · d = 30", 16)
    d.op(820, 290, "+", r=22, size=22)
    d.node(960, 290, "f", "out", r=30)
    d.text(960, 340, "f = 36", 16)
    d.line([(124, 130), (308, 130)])
    d.line([(120, 150), (310, 316)], radius=0)
    d.line([(118, 310), (310, 145)], radius=0)
    d.line([(124, 330), (308, 330)])
    d.line([(352, 135), (560, 192)], radius=0)
    d.line([(352, 325), (560, 210)], radius=0)
    d.line([(602, 207), (800, 282)], radius=0)
    d.line([(352, 335), (798, 296)], radius=0)
    d.line([(842, 290), (930, 290)])
    # gradients (red)
    d.text(735, 222, "∂f/∂e = 1", 15, color=GRAD)
    d.text(470, 140, "∂f/∂c = d = 6", 15, color=GRAD)
    d.text(600, 340, "∂f/∂d = c + 1 = 6", 15, color=GRAD)
    d.text(200, 60, "∂f/∂a = 6·1 + 6·3 = 24", 16, color=GRAD, weight="bold")
    d.text(200, 410, "∂f/∂b = 6·1 + 6·2 = 18", 16, color=GRAD, weight="bold")
    d.text(540, 440, "forward: values (black) · backward: gradients (red); a node used twice SUMS its gradients", 15, color=MUTED)
    return d


@diagram("training-loop")
def _train_loop():
    d = Diagram(1180, 380, "The PyTorch training loop")
    xs = [110, 330, 560, 790, 1020]
    labels = [("① zero\\_grad()", "clear old .grad"), ("② model(X)", "forward pass"),
              ("③ criterion(preds, y)", "loss: one number"), ("④ loss.backward()", "fill every .grad"),
              ("⑤ optimizer.step()", "θ ← θ − η ∇L")]
    for x, (a, b) in zip(xs, labels):
        d.layer(x, 160, a, w=200, h=50, size=16)
        d.text(x, 205, b, 14, color=MUTED)
    for i in range(4):
        d.line([(xs[i] + 100, 160), (xs[i + 1] - 100, 160)])
    d.line([(1020, 135), (1020, 80), (110, 80), (110, 135)], radius=18)
    d.text(565, 60, "next batch", 15, color=MUTED)
    d.node(400, 290, "X", "in", r=22)
    d.line([(400, 268), (400, 186)])
    d.node(640, 290, "y", "in", r=22)
    d.line([(640, 268), (640, 186)])
    d.text(590, 350, "backward() ADDS into .grad, so zero\\_grad() must run once per update, never between backward() and step()", 15, color=MUTED)
    return d


def _mlp(d, backward):
    y = 170
    d.node(80, y, "X", "in", r=30)
    d.text(80, y + 50, "(N, d)", 14, color=MUTED)
    d.line([(110, y), (200, y)])
    d.layer(270, y, "Linear W_1, b_1", w=140, size=15)
    d.line([(340, y), (410, y)])
    d.text(375, y - 22, "Z^{(1)}", 16)
    d.layer(460, y, "ReLU", w=80)
    d.line([(500, y), (580, y)])
    d.text(540, y - 22, "A^{(1)}", 16)
    d.text(540, y + 30, "(N, h)", 14, color=MUTED)
    d.layer(650, y, "Linear W_2, b_2", w=140, size=15)
    d.line([(720, y), (800, y)])
    d.node(830, y, "ŷ", "out", r=28)
    d.text(830, y + 48, "(N, o)", 14, color=MUTED)
    d.line([(858, y), (930, y)])
    d.op(970, y, "MSE", r=24, w=72, size=15)
    d.node(970, y - 100, "y", "in", r=20)
    d.line([(970, y - 80), (970, y - 26)])
    d.line([(1006, y), (1060, y)])
    d.node(1090, y, "L", "out", r=26)
    if backward:
        by = y + 110
        d.line([(1080, by), (850, by)], color=GRAD, width=2.8, dash="8 5")
        d.text(965, by + 24, "δ_2 = 2(ŷ − y)/N", 14, color=GRAD)
        d.line([(800, by), (590, by)], color=GRAD, width=2.8, dash="8 5")
        d.text(695, by + 24, "∂L/∂W_2 = δ_2^T A^{(1)}", 14, color=GRAD)
        d.line([(560, by), (390, by)], color=GRAD, width=2.8, dash="8 5")
        d.text(475, by + 24, "δ_1 = δ_2 W_2 ⊙ φ'(Z^{(1)})", 14, color=GRAD)
        d.line([(360, by), (140, by)], color=GRAD, width=2.8, dash="8 5")
        d.text(250, by + 24, "∂L/∂W_1 = δ_1^T X", 14, color=GRAD)
        d.text(590, by + 70, "weight gradient = error × what flowed in · previous error = error × weights × activation slope", 15, color=MUTED)
    else:
        d.text(590, y + 110, "linear → bend → linear:   ŷ = ReLU(X W_1^T + b_1) W_2^T + b_2", 16, color=MUTED)


@diagram("mlp-forward")
def _mlp_f():
    d = Diagram(1180, 320, "Two-layer network, forward pass")
    _mlp(d, False)
    return d


@diagram("mlp-backprop")
def _mlp_b():
    d = Diagram(1180, 400, "Two-layer network, forward and backward")
    _mlp(d, True)
    return d


# =============================================================================
# CNNs
# =============================================================================
def fmaps(d, x, y, w, h, n=4, fill=CELL_FILL, stroke=CELL_STROKE, off=7):
    """A stack of feature maps drawn as offset sheets; (x, y) = centre of the front sheet."""
    for k in range(n - 1, -1, -1):
        d.rect(x - w / 2 + k * off, y - h / 2 - k * off, w, h, fill, stroke, rx=3, sw=1.6)


def vec(d, x, y, h, w=22, fill=OUT_FILL, stroke=OUT_STROKE):
    d.rect(x - w / 2, y - h / 2, w, h, fill, stroke, rx=4, sw=1.8)


@diagram("cnn-block")
def _cnn_block():
    d = Diagram(1180, 360, "Conv → ReLU → Pool")
    y = 170
    fmaps(d, 110, y, 120, 120, 3, IN_FILL, IN_STROKE, 9)
    d.text(125, y + 100, "(1, 3, 32, 32)", 15)
    d.text(125, y + 122, "input image", 14, color=MUTED)
    d.line([(195, y), (275, y)])
    d.layer(335, y, "Conv2d 3→8", w=120, size=15)
    d.text(335, y + 34, "k=3, p=1", 13, color=MUTED)
    d.line([(395, y), (450, y)])
    fmaps(d, 520, y, 120, 120, 6, CELL_FILL, CELL_STROKE, 7)
    d.text(540, y + 100, "(1, 8, 32, 32)", 15)
    d.text(540, y + 122, "8 feature maps", 14, color=MUTED)
    d.line([(620, y), (680, y)])
    d.layer(720, y, "ReLU", w=72)
    d.line([(756, y), (800, y)])
    d.layer(850, y, "MaxPool 2×2", w=110, size=15)
    d.line([(905, y), (960, y)])
    fmaps(d, 1020, y + 15, 60, 60, 6, CELL_FILL, CELL_STROKE, 7)
    d.text(1040, y + 100, "(1, 8, 16, 16)", 15)
    d.text(1040, y + 122, "size halved", 14, color=MUTED)
    d.text(590, 40, "the block every classic CNN repeats: channels go up, height and width go down", 16, color=MUTED)
    return d


def cnn_arch(d, layers, y=190):
    """layers: list of ('maps'|'layer'|'vec'|'logits', ...)"""
    x = 60
    prev = None
    start = len(d.parts)
    for item in layers:
        kind = item[0]
        if prev is not None:
            d.line([(prev, y), (x - 6, y)], head=9, width=2.8)
        if kind == "maps":
            _, w, n, lab, fill = item
            fmaps(d, x + w / 2, y, w, w, n, *(fill if fill else (CELL_FILL, CELL_STROKE)), 6)
            d.text(x + w / 2 + (n - 1) * 3, y + w / 2 + 26, lab, 13)
            prev = x + w + (n - 1) * 6 + 4
            x = prev + 34
        elif kind == "layer":
            _, lab, sub = item
            w = 10 * len(lab) + 22
            d.layer(x + w / 2, y, lab, w=w, size=14, h=32)
            if sub:
                d.text(x + w / 2, y + 32, sub, 12, color=MUTED)
            prev = x + w
            x = prev + 30
        elif kind == "vec":
            _, h, lab = item
            vec(d, x + 11, y, h, fill=CELL_FILL, stroke=CELL_STROKE)
            d.text(x + 11, y + h / 2 + 18, lab, 13)
            prev = x + 22
            x = prev + 30
        elif kind == "logits":
            _, h, lab = item
            vec(d, x + 11, y, h)
            d.text(x + 11, y + h / 2 + 18, lab, 13)
            prev = x + 22
            x = prev + 30
    d.fit_from(start, prev + 10)
    return x


@diagram("smallcnn-architecture")
def _smallcnn():
    d = Diagram(1180, 360, "SmallCNN architecture")
    cnn_arch(d, [("maps", 90, 3, "(3, 32, 32)", (IN_FILL, IN_STROKE)),
                 ("layer", "Conv 3→16", "+ ReLU"),
                 ("maps", 90, 5, "(16, 32, 32)", None),
                 ("layer", "Pool", None),
                 ("maps", 48, 5, "(16, 16, 16)", None),
                 ("layer", "Conv 16→32", "+ ReLU"),
                 ("maps", 48, 7, "(32, 16, 16)", None),
                 ("layer", "Pool", None),
                 ("maps", 26, 7, "(32, 8, 8)", None),
                 ("layer", "Flatten", None),
                 ("vec", 170, "2048"),
                 ("layer", "FC + ReLU", "131,136 params"),
                 ("vec", 70, "64"),
                 ("layer", "FC", None),
                 ("logits", 44, "10 logits")])
    d.rect(50, 50, 590, 34, "#f1f8ea", CELL_STROKE, rx=8, sw=1.5)
    d.text(345, 67, "feature extractor: 5,088 params (conv1 448 + conv2 4,640)", 14, color="#4d7a32")
    d.rect(735, 50, 420, 34, "#fbf3d6", LAYER_STROKE, rx=8, sw=1.5)
    d.text(945, 67, "classifier head: fc1 alone holds 96% of 136,874", 14, color="#7a6a1c")
    return d


@diagram("simplecnn-architecture")
def _simplecnn():
    d = Diagram(1180, 360, "SimpleCNN (cats vs dogs) architecture")
    cnn_arch(d, [("maps", 96, 3, "(3, 128, 128)", (IN_FILL, IN_STROKE)),
                 ("layer", "Conv 3→32", "+ReLU+Pool"),
                 ("maps", 64, 4, "(32, 64, 64)", None),
                 ("layer", "Conv 32→64", "+ReLU+Pool"),
                 ("maps", 40, 6, "(64, 32, 32)", None),
                 ("layer", "Conv 64→128", "+ReLU+Pool"),
                 ("maps", 24, 8, "(128, 16, 16)", None),
                 ("layer", "Flatten", None),
                 ("vec", 180, "32,768"),
                 ("layer", "FC", "+ReLU+Drop 0.3"),
                 ("vec", 60, "128"),
                 ("layer", "FC", None),
                 ("logits", 30, "2 logits")])
    d.rect(50, 50, 640, 34, "#f1f8ea", CELL_STROKE, rx=8, sw=1.5)
    d.text(370, 67, "3 conv blocks: 93,248 params", 14, color="#4d7a32")
    d.rect(740, 50, 415, 34, "#fbf3d6", LAYER_STROKE, rx=8, sw=1.5)
    d.text(947, 67, "fc1: 4,194,432 params = 97.8% of 4,287,938", 14, color="#7a6a1c")
    return d


@diagram("residual-block")
def _residual():
    d = Diagram(1180, 380, "Residual (skip) connection")
    y = 210
    d.node(70, y, "x", "in", r=28)
    d.line([(98, y), (170, y)])
    xs = [("Conv 3×3", 220), ("BN", 330), ("ReLU", 420), ("Conv 3×3", 530), ("BN", 640)]
    prev = 170
    for lab, cx in xs:
        w = 92 if "Conv" in lab else 60
        if prev != 170:
            d.line([(prev, y), (cx - w / 2, y)], head=9)
        d.layer(cx, y, lab, w=w, size=15)
        prev = cx + w / 2
    d.rect(165, y - 50, 510, 100, "none", CELL_STROKE, rx=16, sw=2, dash="7 5")
    d.text(420, y + 68, "F(x; W): the learned residual", 14, color="#4d7a32")
    d.line([(670, y), (746, y)])
    d.op(770, y, "+", r=22, size=22)
    d.line([(130, y), (130, 90), (770, 90), (770, y - 24)], radius=20)
    d.dot(130, y)
    d.text(450, 70, "identity shortcut: x skips the block", 15, color=MUTED)
    d.line([(792, y), (850, y)])
    d.layer(895, y, "ReLU", w=72)
    d.line([(931, y), (1000, y)])
    d.node(1030, y, "y", "out", r=28)
    d.text(590, 330, "y = F(x; W) + x    so    ∂y/∂x = ∂F/∂x + I   (the gradient always keeps an identity path)", 17)
    return d


@diagram("transfer-learning")
def _transfer():
    d = Diagram(1180, 430, "Feature extraction vs fine-tuning on ResNet18")
    stages = ["stem", "layer1", "layer2", "layer3", "layer4", "avgpool"]
    rows = [("Feature extraction", 150, set(), "trainable: 1,026 params (fc only)"),
            ("Fine-tuning", 300, {"layer4"}, "trainable: 8,394,754 params (layer4 + fc)")]
    for title, y, train, note in rows:
        d.text(40, y - 62, title, 17, "start", weight="bold")
        d.text(40, y - 38, note, 14, "start", MUTED)
        d.node(80, y, "img", "in", r=26, size=15)
        x = 120
        for s in stages:
            w = 118
            if x > 120:
                pass
            d.line([(x, y), (x + 22, y)], head=8, width=2.6)
            x += 26
            frozen = s not in train and s != "avgpool"
            if s == "avgpool":
                d.op(x + w / 2, y, "avg pool", r=18, w=100, size=14)
            elif frozen:
                d.layer(x + w / 2, y, s, w=w, size=15, fill=FROZEN_FILL, stroke=FROZEN_STROKE, color="#666")
                d.text(x + w / 2, y + 30, "frozen", 12, color=MUTED)
            else:
                d.layer(x + w / 2, y, s, w=w, size=15)
                d.text(x + w / 2, y + 30, "trainable, lr 1e-4", 12, color="#7a6a1c")
            x += w
        d.line([(x, y), (x + 22, y)], head=8, width=2.6)
        x += 26
        d.text(x - 14, y - 18, "512", 12, color=MUTED)
        d.layer(x + 60, y, "new fc 512→2", w=120, size=14)
        d.text(x + 60, y + 30, "trainable", 12, color="#7a6a1c")
    d.layer(380, 400, "frozen", w=80, size=13, fill=FROZEN_FILL, stroke=FROZEN_STROKE, color="#666")
    d.text(430, 400, "requires\\_grad = False: no gradient, no change", 13, "start", MUTED)
    d.layer(800, 400, "trainable", w=90, size=13)
    return d


# =============================================================================
# NLP
# =============================================================================
@diagram("nlp-preprocessing-pipeline")
def _nlp_pipe():
    d = Diagram(1180, 330, "Text preprocessing pipeline")
    steps = [("lowercase +", "tokenize"), ("remove", "stop words"), ("stem or", "lemmatize")]
    examples = ["['i', 'am', 'learning', 'machine', …]", "['learning', 'machine', 'learning', 'loving']",
                "['learn', 'machine', 'learn', 'love']"]
    d.node(90, 140, "text", "in", r=34, size=16)
    d.text(90, 200, "“I am learning machine", 13, color=MUTED)
    d.text(90, 218, "learning and loving it”", 13, color=MUTED)
    x = 124
    for (a, b), ex in zip(steps, examples):
        d.line([(x, 140), (x + 60, 140)])
        cx = x + 140
        d.rect(cx - 80, 110, 160, 60, LAYER_FILL, LAYER_STROKE, rx=4, sw=2)
        d.text(cx, 128, a, 15)
        d.text(cx, 150, b, 15)
        d.text(cx + 20, 210 + (0 if x == 124 else 0), ex, 12, color=MUTED)
        x = cx + 80
    d.line([(x, 140), (x + 60, 140)])
    d.node(x + 94, 140, "tokens", "out", r=34, size=14)
    d.text(590, 290, "each step shrinks the vocabulary and throws information away (e.g. 'not' is a stop word)", 15, color=MUTED)
    return d


@diagram("word2vec")
def _w2v():
    d = Diagram(1180, 420, "Word2Vec: skip-gram vs CBOW")
    # skip-gram
    d.text(290, 40, "Skip-gram: centre word → context", 17, weight="bold")
    d.node(80, 210, "love", "in", r=32, size=15)
    d.line([(112, 210), (170, 210)])
    d.layer(230, 210, "v_c", w=100, size=18)
    d.text(230, 245, "input vector", 12, color=MUTED)
    d.line([(280, 210), (330, 210)])
    d.layer(400, 210, "softmax(u_o·v_c)", w=150, size=14)
    d.line([(475, 200), (520, 140)])
    d.line([(475, 220), (520, 280)])
    d.node(548, 120, "i", "out", r=26, size=16)
    d.node(548, 300, "deep", "out", r=26, size=13)
    # CBOW
    d.text(890, 40, "CBOW: context → centre word", 17, weight="bold")
    d.node(660, 120, "i", "in", r=26, size=16)
    d.node(660, 300, "deep", "in", r=26, size=13)
    d.layer(750, 120, "v", w=60, size=18)
    d.layer(750, 300, "v", w=60, size=18)
    d.line([(686, 120), (718, 120)])
    d.line([(686, 300), (718, 300)])
    d.line([(780, 120), (835, 195)], radius=0)
    d.line([(780, 300), (835, 225)], radius=0)
    d.op(860, 210, "mean", r=20, w=72, size=14)
    d.line([(896, 210), (930, 210)])
    d.layer(1000, 210, "softmax(u·v̄)", w=130, size=14)
    d.line([(1065, 210), (1092, 210)])
    d.node(1122, 210, "love", "out", r=28, size=14)
    d.text(590, 385, "window w = 1 on “i love deep learning” · keep the input vectors v as the embeddings", 15, color=MUTED)
    return d


@diagram("sentiment-pipeline")
def _sent():
    d = Diagram(1180, 300, "Sentiment classification pipeline")
    d.node(80, 130, "review", "in", r=36, size=14)
    items = [("clean +", "preprocess"), ("TF-IDF", "vectorizer"), ("Naive Bayes", "or Linear SVM")]
    subs = ["tokens", "sparse vector, L2-normalized", "fit on train split only"]
    x = 116
    for (a, b), sub in zip(items, subs):
        d.line([(x, 130), (x + 60, 130)])
        cx = x + 140
        d.rect(cx - 80, 100, 160, 60, LAYER_FILL, LAYER_STROKE, rx=4, sw=2)
        d.text(cx, 118, a, 15)
        d.text(cx, 140, b, 15)
        d.text(cx, 185, sub, 13, color=MUTED)
        x = cx + 80
    d.line([(x, 130), (x + 60, 130)])
    d.node(x + 100, 130, "pos / neg", "out", r=40, size=13)
    d.text(590, 262, "TF-IDF = term frequency × inverse document frequency: frequent here, rare elsewhere", 15, color=MUTED)
    return d



# =============================================================================
# Attention and Transformers (notebook 14)
# =============================================================================
def _small_cell(d, cx, cy, label="RNN", w=96, h=56):
    d.rect(cx - w / 2, cy - h / 2, w, h, CELL_FILL, CELL_STROKE, rx=12, sw=2.2)
    d.text(cx, cy, label, 16, color="#4d7a32", weight="bold")


@diagram("seq2seq-bottleneck")
def _seq2seq_bottleneck():
    d = Diagram(1180, 430, "Encoder-decoder with one fixed context vector")
    y = 200
    d.text(270, 40, "Encoder reads the whole source sentence", 16, weight="bold")
    d.text(920, 40, "Decoder writes the output", 16, weight="bold")
    xs = [90, 220, 350, 480]
    for i, x in enumerate(xs):
        _small_cell(d, x, y, "LSTM")
        d.node(x, y + 120, f"x_{i + 1}", "in", r=22, size=18)
        d.line([(x, y + 98), (x, y + 30)])
        if i:
            d.line([(xs[i - 1] + 48, y), (x - 50, y)])
    d.line([(528, y), (600, y)])
    d.text(564, y - 22, "h_T", 17)
    d.op(640, y, "c", r=26, size=22)
    d.text(640, y + 48, "one vector", 14, color=GRAD, weight="bold")
    d.text(640, y + 68, "for the whole", 14, color=GRAD)
    d.text(640, y + 88, "sentence", 14, color=GRAD)
    ds = [790, 920, 1050]
    for i, x in enumerate(ds):
        _small_cell(d, x, y, "LSTM")
        d.node(x, y - 120, f"y_{i + 1}", "out", r=22, size=18)
        d.line([(x, y - 30), (x, y - 96)])
        if i:
            d.line([(ds[i - 1] + 48, y), (x - 50, y)])
        d.curve((664, y + 14), (700, y + 110), (x - 10, y + 110), (x, y + 30), color=GRAD, width=2.6)
    d.line([(666, y), (740, y)])
    d.text(590, 400, "every output step gets the SAME summary c = h_T:   y_t = g(y_{t-1}, s_{t-1}, c)", 16, color=MUTED)
    return d


@diagram("seq2seq-attention")
def _seq2seq_attention():
    d = Diagram(1180, 470, "Encoder-decoder with attention")
    hy = 330
    xs = [110, 250, 390, 530]
    for i, x in enumerate(xs):
        _small_cell(d, x, hy, "LSTM")
        d.node(x, hy + 100, f"x_{i + 1}", "in", r=20, size=16)
        d.line([(x, hy + 80), (x, hy + 30)])
        if i:
            d.line([(xs[i - 1] + 48, hy), (x - 50, hy)])
        d.text(x - 30, hy - 44, f"h_{i + 1}", 16, "end")
    ws = [0.9, 2.2, 5.0, 1.3]          # line widths only (illustrative thickness, no numbers shown)
    for x, w in zip(xs, ws):
        d.line([(x, hy - 30), (790, 175)], width=w, radius=0, head=9)
    d.op(820, 170, "Σ", r=26, size=22)
    d.text(760, 286, "weights α_{t,i}", 16, color=MUTED)
    d.text(760, 308, "(thicker = more attention)", 13, color=MUTED)
    d.line([(846, 170), (930, 170)])
    d.text(888, 148, "c_t", 17)
    _small_cell(d, 990, 170, "decoder")
    d.line([(990, 140), (990, 90)])
    d.node(990, 66, "y_t", "out", r=22, size=18)
    d.text(990, 262, "s_{t-1}", 16, color=MUTED)
    d.line([(990, 244), (990, 202)], head=8, width=2.4)
    d.note(40, 40, 520, 150)
    d.text(300, 66, "A NEW context for every output step", 16, weight="bold")
    d.text(58, 104, "e_{t,i} = score(s_{t-1}, h_i)", 17, "start")
    d.text(58, 138, "α_{t,i} = softmax_i(e_{t,i})", 17, "start")
    d.text(58, 172, "c_t = Σ_i α_{t,i} h_i", 17, "start")
    return d


@diagram("chain-vs-attention")
def _chain_vs_attention():
    d = Diagram(1180, 470, "Recurrence chain vs self-attention")
    toks = ["The", "movie", "was", "not", "good"]
    xs = [190 + i * 200 for i in range(5)]
    d.text(40, 50, "RNN / LSTM: information travels step by step", 17, "start", weight="bold")
    for i, (x, t) in enumerate(zip(xs, toks)):
        d.node(x, 120, t, "in", r=34, size=14)
        if i:
            d.line([(xs[i - 1] + 34, 120), (x - 36, 120)])
    d.text(1150, 176, "“The” → “good”: 4 sequential steps (n − 1)", 15, "end", GRAD)
    d.text(40, 212, "Self-attention: every token connects to every token", 17, "start", weight="bold")
    yb = 390
    for i, x in enumerate(xs):
        for j in range(i + 1, 5):
            xm = (x + xs[j]) / 2
            h = 26 + 22 * (j - i)
            d.parts.append(f'<path d="M{x},{yb - 34} Q{xm},{yb - 34 - h * 1.8} {xs[j]},{yb - 34}" fill="none" '
                           f'stroke="{INK}" stroke-width="2" opacity="0.55"/>')
    for x, t in zip(xs, toks):
        d.node(x, yb, t, "in", r=34, size=14)
    d.text(1150, 450, "“The” → “good”: 1 step, and all n² pairs are computed in parallel", 15, "end", GRAD)
    return d


@diagram("qkv-retrieval")
def _qkv():
    d = Diagram(1180, 470, "Query, key and value")
    d.text(590, 34, "“The animal didn't cross the road because it was tired”: processing the token “it”", 16, color=MUTED)
    d.node(90, 240, "it", "in", r=30, size=18)
    d.layer(210, 240, "W_Q", w=70)
    d.line([(120, 240), (174, 240)])
    d.line([(246, 240), (300, 240)])
    d.text(273, 218, "q", 20, weight="bold")
    words = ["animal", "road", "tired"]
    ys = [120, 240, 360]
    for w_, y in zip(words, ys):
        d.node(470, y, w_, "in", r=30, size=12)
        d.layer(570, y - 22, "W_K", w=62, h=28, size=14)
        d.layer(570, y + 22, "W_V", w=62, h=28, size=14)
        d.line([(500, y - 10), (538, y - 22)], radius=0, head=8, width=2.4)
        d.line([(500, y + 10), (538, y + 22)], radius=0, head=8, width=2.4)
        d.op(700, y - 22, "q·k", r=18, w=58, size=13)
        d.line([(602, y - 22), (670, y - 22)], head=8, width=2.4)
        d.text(636, y + 2, "v", 15, color=MUTED)
        d.line([(602, y + 22), (900, y + 22)], head=8, width=2.4, color="#6b6b6b")
        d.line([(330, 240), (330, y - 50), (700, y - 50), (700, y - 41)], head=8, width=2.2, radius=10)
        d.line([(730, y - 22), (790, y - 22)], head=8, width=2.4)
    d.layer(830, 240, "softmax", w=86, h=280, size=15)
    d.text(830, 410, "scores → weights α", 13, color=MUTED)
    for y in ys:
        d.op(915, y + 22, "×", r=13, size=15)
        d.line([(873, y - 22), (895, y - 22), (915, y + 6)], head=8, width=2.2, radius=6)
        d.line([(928, y + 22), (1010, 240)], radius=0, head=8, width=2.4)
    d.op(1035, 240, "Σ", r=24, size=20)
    d.line([(1059, 240), (1110, 240)])
    d.node(1140, 240, "out", "out", r=28, size=14)
    d.text(590, 452, "Query = what “it” is looking for · Key = what each word offers for matching · Value = what each word passes on", 15, color=MUTED)
    return d


@diagram("transformer-big-picture")
def _transformer_big():
    d = Diagram(1180, 430, "Transformer block, big picture")
    y = 215
    d.node(70, y, "tokens", "in", r=36, size=13)
    d.line([(106, y), (150, y)])
    d.layer(210, y, "Embedding", w=110, size=15)
    d.line([(265, y), (305, y)])
    d.op(330, y, "+", r=18, size=20)
    d.layer(330, y + 100, "positional", w=104, size=14)
    d.line([(330, y + 83), (330, y + 18)])
    d.line([(348, y), (420, y)])
    d.cell(430, 80, 560, 270)
    d.text(445, 98, "× N layers", 15, "start", "#4d7a32", "bold")
    d.layer(520, y, "Self-attention", w=130, size=15)
    d.op(625, y, "+", r=15, size=17)
    d.layer(700, y, "LayerNorm", w=100, size=14)
    d.layer(820, y, "Feed-forward", w=120, size=15)
    d.op(905, y, "+", r=15, size=17)
    d.layer(955, y, "Norm", w=60, size=14)
    d.line([(585, y), (608, y)], head=8)
    d.line([(640, y), (648, y)], head=6, width=2.4)
    d.line([(750, y), (758, y)], head=6, width=2.4)
    d.line([(880, y), (888, y)], head=6, width=2.4)
    d.line([(920, y), (923, y)], head=5, width=2.2)
    d.line([(440, y), (440, 130), (625, 130), (625, y - 17)], width=2.4, radius=10, head=8)
    d.line([(760, y), (760, 130), (905, 130), (905, y - 17)], width=2.4, radius=10, head=8)
    d.text(532, 118, "residual", 13, color=MUTED)
    d.text(832, 118, "residual", 13, color=MUTED)
    d.text(520, y + 50, "tokens talk to", 13, color=MUTED)
    d.text(520, y + 68, "each other", 13, color=MUTED)
    d.text(820, y + 50, "each token is", 13, color=MUTED)
    d.text(820, y + 68, "refined on its own", 13, color=MUTED)
    d.line([(985, y), (1060, y)])
    d.node(1105, y, "context", "out", r=40, size=12)
    d.text(1105, y + 58, "vectors", 13, color=MUTED)
    d.text(590, 400, "attention = communication across tokens · feed-forward = transformation within each token", 15, color=MUTED)
    return d


# =============================================================================
# Full self-attention (notebook 15)
# =============================================================================
@diagram("cross-vs-self-attention")
def _cross_vs_self():
    d = Diagram(1180, 470, "Encoder-decoder (cross) attention vs self-attention")
    d.text(290, 40, "Cross-attention (encoder–decoder)", 17, weight="bold")
    d.text(890, 40, "Self-attention", 17, weight="bold")
    enc = ["I", "drink", "water"]
    ex = [130, 290, 450]
    for x, t in zip(ex, enc):
        d.node(x, 330, t, "in", r=34, size=13)
    d.text(290, 400, "encoder outputs → keys and values", 14, color=MUTED)
    d.node(290, 130, "eau", "out", r=34, size=14)
    d.text(290, 82, "decoder token → query", 14, color=MUTED)
    for x in ex:
        d.line([(290, 164), (x, 294)], radius=0, head=9, width=2.4)
    d.text(290, 440, "Q from one sequence, K and V from another", 15, color=GRAD)
    toks = ["The", "movie", "was", "not", "good"]
    xs = [700 + i * 95 for i in range(5)]
    for x, t in zip(xs, toks):
        d.node(x, 260, t, "in", r=34, size=12)
    gx = xs[4]
    for x in xs:
        if x == gx:
            d.parts.append(f'<path d="M{gx - 22},{226} C{gx - 40},{150} {gx + 40},{150} {gx + 22},{226}" fill="none" stroke="{GRAD}" stroke-width="2.6"/>')
        else:
            xm = (x + gx) / 2
            d.parts.append(f'<path d="M{gx},{226} Q{xm},{226 - 40 - 0.55 * (gx - x)} {x},{226}" fill="none" stroke="{GRAD}" stroke-width="2.4" opacity="0.85"/>')
    d.text(890, 330, "“good” scores every token, itself included", 14, color=MUTED)
    d.text(890, 356, "(every other token does the same at the same time)", 14, color=MUTED)
    d.text(890, 440, "Q, K and V all come from the same X", 15, color=GRAD)
    return d


@diagram("self-attention-pipeline")
def _sa_pipeline():
    d = Diagram(1180, 470, "Scaled dot-product self-attention, shapes for the notebook example")
    yq, yk, yv = 110, 230, 370
    d.node(70, yk, "X", "in", r=34, size=22)
    d.text(70, yk + 56, "5 × 4", 14, color=MUTED)
    for y, w in ((yq, "W_Q"), (yk, "W_K"), (yv, "W_V")):
        d.line([(104, yk), (190, y)] if y == yk else [(104, yk), (150, yk), (150, y), (190, y)], radius=10, head=9, width=2.6)
        d.layer(230, y, w, w=74)
        d.line([(267, y), (310, y)], head=9, width=2.6)
    for y, lab in ((yq, "Q"), (yk, "K"), (yv, "V")):
        d.node(336, y, lab, "out", r=24, size=19)
        d.text(336, y + 40, "5 × 3", 13, color=MUTED)
    d.op(450, 170, "QK^T", r=22, w=70, size=15)
    d.line([(360, yq), (420, yq), (430, 152)], radius=8, head=8, width=2.4)
    d.line([(360, yk), (420, yk), (430, 188)], radius=8, head=8, width=2.4)
    d.line([(485, 170), (540, 170)], head=9, width=2.6)
    d.node(570, 170, "S", "out", r=26, size=20)
    d.text(570, 214, "5 × 5", 13, color=MUTED)
    d.line([(596, 170), (640, 170)], head=9, width=2.6)
    d.op(682, 170, "÷√d_k", r=22, w=78, size=15)
    d.line([(721, 170), (760, 170)], head=9, width=2.6)
    d.layer(830, 170, "softmax (each row)", w=130, size=14)
    d.line([(895, 170), (945, 170)], head=9, width=2.6)
    d.node(972, 170, "A", "out", r=26, size=20)
    d.text(990, 214, "5 × 5, rows sum to 1", 13, "start", MUTED)
    d.op(972, yv, "×", r=18, size=20)
    d.line([(972, 196), (972, yv - 20)], head=9, width=2.6)
    d.line([(360, yv), (952, yv)], head=9, width=2.6)
    d.line([(990, yv), (1070, yv)], head=9, width=2.6)
    d.node(1105, yv, "Z", "out", r=28, size=21)
    d.text(1105, yv + 46, "5 × 3", 13, color=MUTED)
    d.text(1105, yv + 66, "one context-aware", 13, color=MUTED)
    d.text(1105, yv + 84, "row per token", 13, color=MUTED)
    d.text(640, 446, "Z = softmax(QK^T / √d_k) V        n = 5 tokens, d_{model} = 4, d_k = d_v = 3", 16, color=GRAD)
    return d


@diagram("score-matrix-shapes")
def _score_shapes():
    d = Diagram(1180, 450, "Where one score comes from: S = QK^T")
    c = 44

    def grid(x0, y0, rows, cols, hl_row=None, hl_col=None, hl_cell=None):
        for r in range(rows):
            for k in range(cols):
                on = (r == hl_row) or (k == hl_col) or ((r, k) == hl_cell)
                strong = (r, k) == hl_cell
                fill = "#f6c344" if strong else ("#fde9b0" if on else "#f4f4f4")
                d.rect(x0 + k * c, y0 + r * c, c, c, fill, "#9a9a9a", rx=2, sw=1.2)

    toks = ["The", "movie", "was", "not", "good"]
    qx, y0 = 120, 110
    grid(qx, y0, 5, 3, hl_row=4)
    for r, t in enumerate(toks):
        d.text(qx - 12, y0 + r * c + c / 2, t, 14, "end", MUTED)
    d.text(qx + 1.5 * c, y0 - 26, "Q   (n × d_k)", 16, weight="bold")
    d.text(qx + 1.5 * c, y0 + 5 * c + 30, "row = q_{good}", 14, color=MUTED)
    d.text(330, y0 + 2.5 * c, "×", 30)
    kx = 380
    grid(kx, y0 + c, 3, 5, hl_col=3)
    for k, t in enumerate(toks):
        d.text(kx + k * c + c / 2, y0 + c - 12, t, 12, color=MUTED)
    d.text(kx + 2.5 * c, y0 - 26, "K^T   (d_k × n)", 16, weight="bold")
    d.text(kx + 2.5 * c, y0 + 4 * c + 30, "column = k_{not}", 14, color=MUTED)
    d.text(650, y0 + 2.5 * c, "=", 30)
    sx = 700
    grid(sx, y0, 5, 5, hl_cell=(4, 3))
    for k, t in enumerate(toks):
        d.text(sx + k * c + c / 2, y0 - 10, t, 12, color=MUTED)
    for r, t in enumerate(toks):
        d.text(sx + 5 * c + 12, y0 + r * c + c / 2, t, 13, "start", MUTED)
    d.text(sx + 2.5 * c, y0 - 40, "S   (n × n)", 16, weight="bold")
    d.note(40, 370, 1100, 60)
    d.text(590, 400, "S_{good,not} = q_{good} · k_{not} = Σ_{ℓ} q_{good,ℓ} k_{not,ℓ}   (d_k multiplications, then a sum)", 17)
    return d


# =============================================================================
# Multi-head attention (notebook 16) + paper Figure 2
# =============================================================================
def _stack(d, cx, cy, label, w, h=40, n=3, size=16, fill=LAYER_FILL, stroke=LAYER_STROKE):
    for k in range(n - 1, 0, -1):
        d.rect(cx - w / 2 + 9 * k, cy - h / 2 - 9 * k, w, h, fill, stroke, rx=6, sw=1.6)
    d.layer(cx, cy, label, w=w, h=h, size=size, fill=fill, stroke=stroke)


@diagram("paper-figure2-scaled-dot-product")
def _fig2_left():
    d = Diagram(620, 640, "Scaled Dot-Product Attention (paper Figure 2, left)")
    d.text(310, 34, "Scaled Dot-Product Attention", 19, weight="bold")
    xq, xk, xv = 150, 270, 540
    yb = 590
    for x, lab in ((xq, "Q"), (xk, "K"), (xv, "V")):
        d.node(x, yb, lab, "in", r=24, size=18)
    d.layer(210, 500, "MatMul", w=150, fill=OP_FILL, stroke=OP_STROKE)
    d.line([(xq, yb - 24), (xq, 520)], head=9, width=2.6); d.line([(xk, yb - 24), (xk, 520)], head=9, width=2.6)
    d.layer(210, 420, "Scale", w=150)
    d.layer(210, 340, "Mask (opt.)", w=150, fill="#f3e6fa", stroke="#8a4fae")
    d.layer(210, 260, "SoftMax", w=150, fill=CELL_FILL, stroke=CELL_STROKE)
    d.layer(330, 160, "MatMul", w=150, fill=OP_FILL, stroke=OP_STROKE)
    for y0, y1 in ((480, 440), (400, 360), (320, 280)):
        d.line([(210, y0), (210, y1)], head=9, width=2.6)
    d.line([(210, 240), (210, 200), (290, 200), (290, 180)], head=9, width=2.6, radius=10)
    d.line([(xv, yb - 24), (xv, 200), (370, 200), (370, 180)], head=9, width=2.6, radius=10)
    d.line([(330, 140), (330, 80)], head=9, width=2.6)
    d.text(345, 96, "output", 14, "start", MUTED)
    d.text(300, 500, "QK^T", 15, "start", MUTED); d.text(300, 420, "÷ √d_k", 15, "start", MUTED)
    d.text(300, 340, "−∞ on illegal links", 15, "start", MUTED); d.text(300, 260, "row-wise", 15, "start", MUTED)
    d.text(420, 160, "× V", 15, "start", MUTED)
    return d


@diagram("paper-figure2-multi-head")
def _fig2_right():
    d = Diagram(620, 640, "Multi-Head Attention (paper Figure 2, right)")
    d.text(310, 34, "Multi-Head Attention", 19, weight="bold")
    xv, xk, xq = 150, 290, 430
    yb = 600
    for x, lab in ((xv, "V"), (xk, "K"), (xq, "Q")):
        d.node(x, yb, lab, "in", r=24, size=18)
        _stack(d, x, 500, "Linear", w=100, size=16)
        d.line([(x, yb - 24), (x, 522)], head=9, width=2.6)
        d.line([(x, 478), (x, 390)], head=9, width=2.6)
    _stack(d, 290, 360, "Scaled Dot-Product Attention", w=380, h=56, size=17, fill="#f3e6fa", stroke="#8a4fae")
    d.text(540, 300, "h", 22, weight="bold", italic=True)
    d.line([(290, 332), (290, 262)], head=9, width=2.6)
    d.layer(290, 240, "Concat", w=150, fill=CELL_FILL, stroke=CELL_STROKE)
    d.line([(290, 220), (290, 172)], head=9, width=2.6)
    d.layer(290, 150, "Linear", w=150)
    d.line([(290, 130), (290, 76)], head=9, width=2.6)
    d.text(306, 92, "MultiHead(Q, K, V)", 14, "start", MUTED)
    for x, lab in ((xv, "W_i^V"), (xk, "W_i^K"), (xq, "W_i^Q")):
        d.text(x + 10, 446, lab, 15, "start", MUTED)
    d.text(380, 150, "W^O", 15, "start", MUTED)
    return d


@diagram("mha-toy-shapes")
def _mha_toy():
    d = Diagram(1180, 440, "Multi-head attention shapes in the notebook example (n = 5, d_model = 4, h = 2)")
    d.node(70, 220, "X", "in", r=34, size=22); d.text(70, 276, "5 × 4", 14, color=MUTED)
    for y, r, col in ((120, 1, "#a8cbee"), (320, 2, "#fde9b0")):
        d.line([(104, 220), (140, 220), (140, y), (180, y)], radius=10, head=9, width=2.6)
        d.layer(250, y, f"W_Q^{{({r})}}, W_K^{{({r})}}, W_V^{{({r})}}", w=140, size=14)
        d.text(250, y + 36, "each 4 × 2", 13, color=MUTED)
        d.line([(320, y), (370, y)], head=9, width=2.6)
        d.layer(470, y, f"Attention, head {r}", w=190, size=15, fill="#f3e6fa", stroke="#8a4fae")
        d.text(470, y + 36, f"A^{{({r})}}: 5 × 5", 13, color=MUTED)
        d.line([(565, y), (620, y)], head=9, width=2.6)
        d.node(660, y, f"head_{r}", "out", r=36, size=14); d.text(660, y + 52, "5 × 2", 13, color=MUTED)
        d.line([(696, y), (740, y), (740, 220), (780, 220)], radius=10, head=9, width=2.6)
    d.layer(835, 220, "Concat", w=100, fill=CELL_FILL, stroke=CELL_STROKE); d.text(835, 256, "H: 5 × 4", 13, color=MUTED)
    d.line([(885, 220), (925, 220)], head=9, width=2.6)
    d.layer(970, 220, "× W_O", w=80); d.text(970, 256, "4 × 4", 13, color=MUTED)
    d.line([(1010, 220), (1060, 220)], head=9, width=2.6)
    d.node(1100, 220, "Z", "out", r=32, size=22); d.text(1100, 270, "5 × 4", 14, color=MUTED)
    d.text(590, 420, "each head sees all of X but works in its own 2-dim subspace; h · d_{k} = 2 · 2 = 4 = d_{model}, so Z has X's shape", 15, color=GRAD)
    return d


# =============================================================================
# The Transformer, paper Figure 1 (labels word for word)
# =============================================================================
F1 = {
    "emb": ("#fbe1e1", "#c05e5e"), "mha": ("#fde0b6", "#c98a2d"), "an": ("#eef5c6", "#8aa33a"),
    "ff": ("#c9e6f7", "#3a8cc0"), "lin": ("#dfe1f3", "#6b6fae"), "sm": ("#d4ecd0", "#4f8f4a"),
    "box": ("#f1f1f1", "#8f8f8f"),
}


def _f1_box(d, cx, cy, w, h, lines, kind, size=17, hl=None):
    fill, stroke = F1[kind]
    sw = 2.2
    if hl is not None:
        sw = 4.2 if hl else 1.4
    d.rect(cx - w / 2, cy - h / 2, w, h, fill, stroke, rx=7, sw=sw)
    n = len(lines)
    for k, t in enumerate(lines):
        d.text(cx, cy + (k - (n - 1) / 2) * (size + 3), t, size)


def _f1_pe(d, cx, cy):
    d.parts.append(f'<circle cx="{cx}" cy="{cy}" r="19" fill="white" stroke="{INK}" stroke-width="2.2"/>')
    d.parts.append(f'<path d="M{cx - 13},{cy} C{cx - 8},{cy - 16} {cx - 3},{cy - 16} {cx},{cy} '
                   f'C{cx + 3},{cy + 16} {cx + 8},{cy + 16} {cx + 13},{cy}" fill="none" stroke="{INK}" stroke-width="2"/>')


def _f1_plus(d, cx, cy):
    d.parts.append(f'<circle cx="{cx}" cy="{cy}" r="11" fill="white" stroke="{INK}" stroke-width="2.2"/>')
    d.parts.append(f'<path d="M{cx - 7},{cy} H{cx + 7} M{cx},{cy - 7} V{cy + 7}" stroke="{INK}" stroke-width="2.2"/>')


def figure1(hl=None):
    """hl: name of the box group to emphasise (None = plain figure)."""
    H = lambda key: (None if hl is None else (key == hl))
    d = Diagram(920, 1120, "The Transformer - model architecture (after Vaswani et al. 2017, Figure 1)")
    ex, dx = 270, 650                      # encoder and decoder centre lines
    W = 190
    # ---------------- encoder (left) ----------------
    d.text(ex, 1090, "Inputs", 18)
    d.line([(ex, 1072), (ex, 1022)], head=9, width=2.6)
    _f1_box(d, ex, 995, W, 52, ["Input", "Embedding"], "emb", hl=H("emb"))
    d.line([(ex, 969), (ex, 937)], head=9, width=2.6)
    _f1_plus(d, ex, 925)
    _f1_pe(d, ex - 75, 925)
    d.line([(ex - 56, 925), (ex - 11, 925)], arrow=False, width=2.2)
    d.text(ex - 175, 914, "Positional", 17); d.text(ex - 175, 936, "Encoding", 17)
    gb, gt = 870, 520                      # grey box bottom/top
    fill, stroke = F1["box"]
    d.rect(ex - 135, gt, 270, gb - gt, fill, stroke, rx=14, sw=2.2)
    d.text(ex - 175, 700, "N×", 20)
    d.line([(ex, 914), (ex, 842)], head=9, width=2.6)
    # Multi-Head Attention with its three inputs and residual
    _f1_box(d, ex, 805, W, 58, ["Multi-Head", "Attention"], "mha", hl=H("enc_mha"))
    d.line([(ex, 855), (ex - 60, 855), (ex - 60, 836)], head=8, width=2.2, radius=6)
    d.line([(ex, 855), (ex + 60, 855), (ex + 60, 836)], head=8, width=2.2, radius=6)
    d.line([(ex, 862), (ex - 118, 862), (ex - 118, 745), (ex - 97, 745)], head=8, width=2.2, radius=10)
    d.line([(ex, 776), (ex, 760)], head=8, width=2.4)
    _f1_box(d, ex, 745, W, 28, ["Add & Norm"], "an", size=16, hl=H("enc_an1"))
    d.line([(ex, 731), (ex, 690)], head=9, width=2.6)
    _f1_box(d, ex, 660, W, 58, ["Feed", "Forward"], "ff", hl=H("enc_ff"))
    d.line([(ex, 712), (ex - 118, 712), (ex - 118, 580), (ex - 97, 580)], head=8, width=2.2, radius=10)
    d.line([(ex, 631), (ex, 595)], head=8, width=2.4)
    _f1_box(d, ex, 580, W, 28, ["Add & Norm"], "an", size=16, hl=H("enc_an2"))
    # ---------------- decoder (right) ----------------
    d.text(dx, 1080, "Outputs", 18); d.text(dx, 1102, "(shifted right)", 18)
    d.line([(dx, 1062), (dx, 1022)], head=9, width=2.6)
    _f1_box(d, dx, 995, W, 52, ["Output", "Embedding"], "emb", hl=H("emb"))
    d.line([(dx, 969), (dx, 937)], head=9, width=2.6)
    _f1_plus(d, dx, 925)
    _f1_pe(d, dx + 75, 925)
    d.line([(dx + 56, 925), (dx + 11, 925)], arrow=False, width=2.2)
    d.text(dx + 175, 914, "Positional", 17); d.text(dx + 175, 936, "Encoding", 17)
    gb2, gt2 = 870, 330
    d.rect(dx - 135, gt2, 270, gb2 - gt2, fill, stroke, rx=14, sw=2.2)
    d.text(dx + 172, 600, "N×", 20)
    # encoder output -> decoder's middle Multi-Head Attention (keys and values); drawn over the grey box
    d.line([(ex, 566), (ex, 490), (dx - 165, 490), (dx - 165, 645), (dx - 60, 645), (dx - 60, 620)], head=8, width=2.4, radius=12)
    d.line([(dx - 165, 645), (dx, 645), (dx, 620)], head=8, width=2.4, radius=8)
    d.line([(dx, 914), (dx, 838)], head=9, width=2.6)
    _f1_box(d, dx, 800, W, 74, ["Masked", "Multi-Head", "Attention"], "mha", size=16, hl=H("dec_masked"))
    d.line([(dx, 855), (dx - 60, 855), (dx - 60, 838)], head=8, width=2.2, radius=6)
    d.line([(dx, 855), (dx + 60, 855), (dx + 60, 838)], head=8, width=2.2, radius=6)
    d.line([(dx, 862), (dx + 118, 862), (dx + 118, 730), (dx + 97, 730)], head=8, width=2.2, radius=10)
    d.line([(dx, 763), (dx, 745)], head=8, width=2.4)
    _f1_box(d, dx, 730, W, 28, ["Add & Norm"], "an", size=16, hl=H("dec_an1"))
    d.line([(dx, 716), (dx, 680), (dx + 60, 680), (dx + 60, 618)], head=8, width=2.4, radius=10)
    _f1_box(d, dx, 590, W, 58, ["Multi-Head", "Attention"], "mha", hl=H("dec_cross"))
    d.line([(dx, 697), (dx + 118, 697), (dx + 118, 525), (dx + 97, 525)], head=8, width=2.2, radius=10)
    d.line([(dx, 561), (dx, 540)], head=8, width=2.4)
    _f1_box(d, dx, 525, W, 28, ["Add & Norm"], "an", size=16, hl=H("dec_an2"))
    d.line([(dx, 511), (dx, 475)], head=9, width=2.6)
    _f1_box(d, dx, 445, W, 58, ["Feed", "Forward"], "ff", hl=H("dec_ff"))
    d.line([(dx, 495), (dx + 118, 495), (dx + 118, 365), (dx + 97, 365)], head=8, width=2.2, radius=10)
    d.line([(dx, 416), (dx, 380)], head=8, width=2.4)
    _f1_box(d, dx, 365, W, 28, ["Add & Norm"], "an", size=16, hl=H("dec_an3"))
    # ---------------- top ----------------
    d.line([(dx, 351), (dx, 262)], head=9, width=2.6)
    _f1_box(d, dx, 245, 160, 30, ["Linear"], "lin", size=16, hl=H("linear"))
    d.line([(dx, 230), (dx, 192)], head=9, width=2.6)
    _f1_box(d, dx, 175, 160, 30, ["Softmax"], "sm", size=16, hl=H("softmax"))
    d.line([(dx, 160), (dx, 100)], head=9, width=2.6)
    d.text(dx, 62, "Output", 18); d.text(dx, 84, "Probabilities", 18)
    return d


@diagram("transformer-figure-1")
def _fig1():
    return figure1()


@diagram("add-and-norm")
def _add_norm():
    d = Diagram(1180, 360, "Inside one Add & Norm (post-LN, as in the paper)")
    y = 200
    d.node(70, y, "x", "in", r=28, size=20)
    d.line([(98, y), (170, y)], head=9, width=2.6)
    _f1_box(d, 255, y, 160, 56, ["Sub-layer", "(MHA or FFN)"], "mha", size=15)
    d.line([(335, y), (390, y)], head=9, width=2.6)
    d.layer(450, y, "Dropout", w=110, size=15)
    d.text(450, y + 42, "P_{drop} = 0.1", 13, color=MUTED)
    d.line([(505, y), (585, y)], head=9, width=2.6)
    _f1_plus(d, 600, y)
    d.line([(134, y), (134, 90), (600, 90), (600, y - 13)], head=9, width=2.6, radius=12)
    d.text(370, 72, "residual connection: the input skips the sub-layer", 15, color=GRAD)
    d.line([(612, y), (690, y)], head=9, width=2.6)
    _f1_box(d, 790, y, 190, 56, ["LayerNorm", "(per position)"], "an", size=15)
    d.line([(886, y), (960, y)], head=9, width=2.6)
    d.node(1010, y, "out", "out", r=30, size=16)
    d.text(1010, y + 50, "same shape as x", 13, color=MUTED)
    d.note(40, 290, 1100, 50)
    d.text(590, 316, "out = LayerNorm(x + Dropout(Sublayer(x)));   LayerNorm(v) = γ ⊙ (v − μ) / √(σ² + ε) + β  over the d_{model} features of one position", 16)
    return d


@diagram("shifted-right")
def _shifted():
    d = Diagram(1180, 420, "Outputs (shifted right): what the decoder reads and what it must predict")
    toks_in = ["BOS", "9", "5", "1", "4", "1"]
    toks_out = ["9", "5", "1", "4", "1", "3"]
    xs = [330 + 118 * i for i in range(6)]
    d.text(240, 300, "decoder input", 17, "end", weight="bold"); d.text(240, 322, "(Outputs, shifted right)", 14, "end", MUTED)
    d.text(240, 92, "target", 17, "end", weight="bold"); d.text(240, 114, "(predict at each position)", 14, "end", MUTED)
    for i, (a, b) in enumerate(zip(toks_in, toks_out)):
        d.node(xs[i], 305, a, "in", r=30, size=15 if a == "BOS" else 19)
        d.node(xs[i], 100, b, "out", r=30, size=19)
        d.line([(xs[i], 273), (xs[i], 134)], head=9, width=2.4)
        d.text(xs[i], 360, f"pos {i + 1}", 13, color=MUTED)
    for i in range(5):
        d.curve((xs[i] + 22, 82), (xs[i] + 70, 30), (xs[i + 1] + 20, 230), (xs[i + 1] + 18, 282), color=GRAD, width=1.8)
    d.text(1050, 190, "each target token", 14, "start", GRAD); d.text(1050, 210, "becomes the next", 14, "start", GRAD)
    d.text(1050, 230, "position's input", 14, "start", GRAD)
    d.text(640, 400, "position i sees tokens < i (the mask hides the rest) and is trained to output token i; all 6 positions train in one parallel pass", 14, color=MUTED)
    return d

# =============================================================================
# BERT (notebook 17: Devlin et al. 2019, Figures 1-4)
# =============================================================================
BERT_FILL, BERT_STROKE = "#d6defa", "#5468b8"
TOK_FILL, TOK_STROKE = "#f9d9dc", "#b9646c"
EMB_FILL, EMB_STROKE = "#fbefc4", "#a08a2a"
T_FILL, T_STROKE = "#dcefd6", "#6c9d5a"
SEG_FILL, SEG_STROKE = "#b9dcaa", "#5f8f4b"
POS_FILL, POS_STROKE = "#eeeeee", "#8f8f8f"
HEAD_RED = "#c83232"


def _chip(d, cx, cy, label, fill, stroke, w=64, h=34, size=15):
    d.rect(cx - w / 2, cy - h / 2, w, h, fill, stroke, rx=6, sw=1.8)
    d.text(cx, cy, label, size)


def _bert_panel(d, x0, cols, heads, toks, sentA, sentB, bottom, title, colw=66):
    """One half of BERT Figure 1, outer card starting at x0. heads: {col_index: label}."""
    xs = [x0 + 24 + 42 + colw * k for k in range(len(cols))]
    left, right = xs[0] - 42, xs[-1] + 42
    d.rect(left - 24, 70, right - left + 48, 460, "#f6f6f6", "#bdbdbd", rx=28, sw=1.6)
    d.rect(left, 150, right - left, 170, BERT_FILL, BERT_STROKE, rx=6, sw=2.4)
    for x in xs:
        for x2 in xs:                         # faint all-to-all lines: bidirectional attention
            d.parts.append(f'<line x1="{x}" y1="276" x2="{x2}" y2="196" stroke="#9aa6d6" stroke-width="0.8" opacity="0.55"/>')
    mid = (left + right) / 2
    d.rect(mid - 50, 220, 100, 34, BERT_FILL, BERT_FILL, rx=6, sw=0)
    d.text(mid, 237, "BERT", 26, weight="bold")
    for (e, t), x in zip(cols, xs):
        if e == "...":
            d.text(x, 178, "...", 18); d.text(x, 296, "...", 18); d.text(x, 372, "...", 18)
            continue
        _chip(d, x, 178, t, T_FILL, T_STROKE, w=60)
        _chip(d, x, 296, e, EMB_FILL, EMB_STROKE, w=60)
    for t, x in zip(toks, xs):
        if t != "...":
            _chip(d, x, 372, t, TOK_FILL, TOK_STROKE, w=60, size=13)
            d.line([(x, 353), (x, 316)], head=7, width=2.0)
    for k, lab in heads.items():
        d.line([(xs[k], 158), (xs[k], 118)], color=HEAD_RED, width=3.4, head=11)
        if lab:
            d.text(xs[k], 100, lab, 15)
    sep = toks.index("[SEP]")
    for (a, b, lab) in ((1, sep - 1, sentA), (sep + 1, len(toks) - 1, sentB)):
        xa, xb = xs[a], xs[b]
        d.line([(xa, 398), (xa, 410), (xb, 410), (xb, 398)], arrow=False, width=1.6, radius=3)
        d.line([((xa + xb) / 2, 410), ((xa + xb) / 2, 420)], arrow=False, width=1.6)
        d.text((xa + xb) / 2, 438, lab, 15)
    d.line([(mid, 488), (mid, 460)], color="#2f5fd0", width=4, head=12)
    d.text(mid, 508, bottom, 15)
    d.text(mid, 570, title, 22)
    return xs, left, right


@diagram("bert-figure-1")
def _bert_fig1():
    d = Diagram(1430, 610, "Overall pre-training and fine-tuning procedures for BERT (after Devlin et al. 2019, Figure 1)")
    cols = [("E_{[CLS]}", "C"), ("E_1", "T_1"), ("...", "..."), ("E_N", "T_N"), ("E_{[SEP]}", "T_{[SEP]}"),
            ("E_1'", "T_1'"), ("...", "..."), ("E_M'", "T_M'")]
    toks = ["[CLS]", "Tok 1", "...", "Tok N", "[SEP]", "Tok 1", "...", "TokM"]
    xs, l, r = _bert_panel(d, 26, cols, {0: "NSP", 1: "Mask LM", 7: "Mask LM"}, toks,
                           "Masked Sentence A", "Masked Sentence B", "Unlabeled Sentence A and B Pair", "Pre-training")
    width = r - l + 48
    front = 1430 - 26 - width
    for k, lab in enumerate(["MNLI", "NER"]):
        x = front - 108 + 54 * k
        d.rect(x, 70 - 0, width, 460, "#e9e9e9", "#bdbdbd", rx=28, sw=1.6)
        d.text(x + 9, 92, lab, 14, "start", weight="bold")
    xs2, l2, r2 = _bert_panel(d, front, cols, {5: "", 6: "", 7: ""}, toks, "Question", "Paragraph",
                              "Question Answer Pair", "")
    d.text(l2 + 6, 92, "SQuAD", 14, "start", weight="bold")
    d.text(xs2[6], 100, "Start/End Span", 15)
    d.text(front - 46 + width / 2, 570, "Fine-Tuning", 22)
    for k, y in enumerate((200, 236, 272)):
        d.line([(r + 30, y), (front - 108 + 54 * k - 6, y)], color="#555555", width=2.2, dash="2 6", head=9)
    return d


@diagram("bert-figure-2")
def _bert_fig2():
    d = Diagram(1200, 420, "BERT input representation (after Devlin et al. 2019, Figure 2)")
    toks = ["[CLS]", "my", "dog", "is", "cute", "[SEP]", "he", "likes", "play", "##ing", "[SEP]"]
    xs = [255 + 87 * k for k in range(len(toks))]
    rows = [(70, "Input", None), (145, "Token", "Embeddings"), (240, "Segment", "Embeddings"), (335, "Position", "Embeddings")]
    for y, a, b in rows:
        if b:
            d.text(40, y - 11, a, 16, "start"); d.text(40, y + 11, b, 16, "start")
        else:
            d.text(40, y, a, 16, "start")
    d.parts.append(f'<line x1="40" y1="104" x2="1170" y2="104" stroke="{INK}" stroke-width="2"/>')
    for k, (t, x) in enumerate(zip(toks, xs)):
        _chip(d, x, 70, t, TOK_FILL, TOK_STROKE, w=76, size=14 if t.startswith("[") or t.startswith("#") else 16)
        tl = "E_{" + t + "}"
        _chip(d, x, 145, tl, EMB_FILL, EMB_STROKE, w=76, h=40, size=15)
        seg = "E_A" if k <= 5 else "E_B"
        _chip(d, x, 240, seg, SEG_FILL, SEG_STROKE, w=76, h=40, size=18)
        _chip(d, x, 335, f"E_{{{k}}}", POS_FILL, POS_STROKE, w=76, h=40, size=18)
        for y in (192, 287):
            d.text(x, y, "+", 26, weight="bold")
    d.text(600, 395, "each column is summed: input vector of token i = E_{token} + E_{segment} + E_{position}   (all three are learned, H-dimensional)", 15, color=MUTED)
    return d


def _fig3_model(d, x0, title, kind):
    xs = [x0 + 70, x0 + 165, x0 + 330]
    lab = ["1", "2", "N"]
    yT, y2, y1, yE = 70, 165, 270, 370
    blk = "Lstm" if kind == "elmo" else "Trm"
    def links(ya, yb, allowed):
        for i, xa in enumerate(xs):
            for j, xb in enumerate(xs):
                if allowed(i, j):
                    d.line([(xa, ya - 20), (xb, yb + 20)], width=1.6, head=7, color="#4a4a4a")
    if kind == "elmo":
        xs = [x0 + 40, x0 + 120, x0 + 230, x0 + 310, x0 + 390, x0 + 500]
        for half, (cols, dirn) in enumerate(((xs[:3], 1), (xs[3:], -1))):
            for y in (y1, y2):
                for x in cols:
                    _chip(d, x, y, blk, "#f3d9b5", "#b07a2c", w=60, h=36, size=14)
                for a, b in zip(cols, cols[1:]):
                    if dirn == 1:
                        d.line([(a + 30, y), (b - 30, y)], width=1.8, head=7)
                    else:
                        d.line([(b - 30, y), (a + 30, y)], width=1.8, head=7)
                for x in cols:
                    pass
            for x in cols:
                d.line([(x, y1 - 18), (x, y2 + 18)], width=1.8, head=7)
        d.text((xs[0] + xs[2]) / 2, 408, "left-to-right LSTMs", 13, color=MUTED)
        d.text((xs[3] + xs[5]) / 2, 408, "right-to-left LSTMs", 13, color=MUTED)
        es = [x0 + 150, x0 + 250, x0 + 400]
        for x, l in zip(es, lab):
            _chip(d, x, yE, f"E_{l}", EMB_FILL, EMB_STROKE, w=58)
            _chip(d, x, yT, f"T_{l}", T_FILL, T_STROKE, w=58)
        for k, x in enumerate(es):
            for c in (xs[k], xs[3 + k]):
                d.line([(x, yE - 18), (c, y1 + 18)], width=1.3, head=6, color=MUTED)
                d.line([(c, y2 - 18), (x, yT + 18)], width=1.3, head=6, color=MUTED)
        d.text(x0 + 275, 434, "T_i = [L→R state ; R→L state], used as features", 14, color=MUTED)
        d.text(x0 + 275, 22, title, 20, weight="bold")
        return
    for x, l in zip(xs, lab):
        _chip(d, x, yT, f"T_{l}", T_FILL, T_STROKE, w=60)
        _chip(d, x, y2, blk, "#f3d9b5", "#b07a2c", w=60, h=40)
        _chip(d, x, y1, blk, "#f3d9b5", "#b07a2c", w=60, h=40)
        _chip(d, x, yE, f"E_{l}", EMB_FILL, EMB_STROKE, w=60)
    d.text((xs[1] + xs[2]) / 2, y2, "...", 20); d.text((xs[1] + xs[2]) / 2, y1, "...", 20)
    allowed = (lambda i, j: True) if kind == "bert" else (lambda i, j: j >= i)
    links(yE, y1, allowed); links(y1, y2, allowed)
    for x in xs:
        d.line([(x, y2 - 20), (x, yT + 18)], width=1.8, head=7)
    d.text(x0 + 200, 22, title, 20, weight="bold")
    d.text(x0 + 200, 420, "every token sees left and right context" if kind == "bert" else "token i sees only tokens ≤ i (left context)", 14, color=MUTED)


@diagram("bert-figure-3")
def _bert_fig3():
    d = Diagram(1480, 450, "Differences in pre-training model architectures (after Devlin et al. 2019, Figure 3)")
    _fig3_model(d, 30, "BERT (Ours)", "bert")
    _fig3_model(d, 470, "OpenAI GPT", "gpt")
    _fig3_model(d, 920, "ELMo", "elmo")
    return d


def _fig4_panel(d, x0, y0, caption, toks, outs, headline, pair=True):
    colw = 66
    xs = [x0 + 40 + colw * k for k in range(len(toks))]
    left, right = xs[0] - 38, xs[-1] + 38
    d.rect(left, y0 + 105, right - left, 70, BERT_FILL, BERT_STROKE, rx=6, sw=2.2)
    d.text((left + right) / 2, y0 + 140, "BERT", 22, weight="bold")
    for k, (t, o, x) in enumerate(zip(toks, outs, xs)):
        if t == "...":
            for y in (y0 + 78, y0 + 202, y0 + 252):
                d.text(x, y, "...", 16)
            continue
        e = "E_{" + t + "}" if t.startswith("[") else "E_" + t.replace("Tok ", "")
        if t.startswith("Tok"):
            e = "E_" + t.split()[1]
        _chip(d, x, y0 + 78, o, T_FILL, T_STROKE, w=58, h=32, size=14)
        _chip(d, x, y0 + 202, e, EMB_FILL, EMB_STROKE, w=58, h=32, size=14)
        _chip(d, x, y0 + 252, t, TOK_FILL, TOK_STROKE, w=58, h=30, size=12)
        for ya, yb in ((y0 + 237, y0 + 220), (y0 + 186, y0 + 177), (y0 + 105, y0 + 96)):
            d.line([(x, ya), (x, yb)], width=1.6, head=6, color=MUTED)
    for k, lab in headline.items():
        d.line([(xs[k], y0 + 60), (xs[k], y0 + 30)], color=HEAD_RED, width=3.0, head=10)
        d.text(xs[k], y0 + 14, lab, 14)
    if pair:
        sep = toks.index("[SEP]")
        groups = ((1, sep - 1, pair[0]), (sep + 1, len(toks) - 1, pair[1]))
    else:
        groups = ((1, len(toks) - 1, "Single Sentence"),)
    for a, b, lab in groups:
        d.line([(xs[a], y0 + 272), (xs[a], y0 + 280), (xs[b], y0 + 280), (xs[b], y0 + 272)], arrow=False, width=1.4, radius=3)
        d.text((xs[a] + xs[b]) / 2, y0 + 298, lab, 14)
    d.text((left + right) / 2, y0 + 330, caption, 15, weight="bold")


@diagram("bert-figure-4")
def _bert_fig4():
    d = Diagram(1360, 760, "Illustrations of Fine-tuning BERT on Different Tasks (after Devlin et al. 2019, Figure 4)")
    pair_t = ["[CLS]", "Tok 1", "...", "Tok N", "[SEP]", "Tok 1'", "...", "Tok M'"]
    pair_o = ["C", "T_1", "...", "T_N", "T_{[SEP]}", "T_1'", "...", "T_M'"]
    single_t = ["[CLS]", "Tok 1", "Tok 2", "...", "Tok N"]
    single_o = ["C", "T_1", "T_2", "...", "T_N"]
    _fig4_panel(d, 40, 20, "(a) Sentence Pair Classification Tasks: MNLI, QQP, QNLI, STS-B, MRPC, RTE, SWAG",
                pair_t, pair_o, {0: "Class Label"}, ("Sentence 1", "Sentence 2"))
    _fig4_panel(d, 860, 20, "(b) Single Sentence Classification Tasks: SST-2, CoLA",
                single_t, single_o, {0: "Class Label"}, pair=False)
    _fig4_panel(d, 40, 395, "(c) Question Answering Tasks: SQuAD v1.1",
                pair_t, pair_o, {5: "Start/End Span"}, ("Question", "Paragraph"))
    _fig4_panel(d, 860, 395, "(d) Single Sentence Tagging Tasks: CoNLL-2003 NER",
                single_t, single_o, {1: "O", 2: "B-PER", 4: "O"}, pair=False)
    return d


@diagram("bert-mlm-masking")
def _bert_mlm():
    d = Diagram(1180, 470, "Masked LM: which positions are predicted, and what the model is shown there")
    d.text(590, 40, "unlabeled sentence: my dog is hairy   →   15% of WordPiece positions are chosen at random (here: hairy)", 16)
    d.node(170, 180, "chosen", "in", r=46, size=15)
    d.text(170, 245, "15% of tokens", 14, color=MUTED)
    rows = [(110, "80%", "[MASK]", "my dog is [MASK]", "12% of all tokens"),
            (210, "10%", "random", "my dog is apple", "1.5% of all tokens"),
            (310, "10%", "unchanged", "my dog is hairy", "1.5% of all tokens")]
    for y, p, what, ex, share in rows:
        d.line([(216, 180), (300, 180), (300, y), (380, y)], width=2.4, head=9, radius=10)
        d.text(338, y - 14, p, 15, weight="bold")
        d.layer(470, y, what, w=150, size=16)
        d.line([(545, y), (600, y)], width=2.2, head=8)
        d.rect(600, y - 20, 250, 40, TOK_FILL, TOK_STROKE, rx=6, sw=1.6)
        d.text(725, y, ex, 16)
        d.text(870, y, share, 14, "start", MUTED)
    d.note(60, 370, 1060, 80)
    d.text(590, 392, "the label at every chosen position is the original token (hairy); unchosen positions get no loss", 15)
    d.text(590, 418, "the encoder can't tell which inputs are fake or which will be graded, so it must model every token from both sides", 15, color=MUTED)
    return d


@diagram("bert-span-head")
def _bert_span():
    d = Diagram(1180, 430, "SQuAD fine-tuning: a start vector S and an end vector E are the only new parameters")
    toks = ["[CLS]", "11", "20", "[SEP]", "19", "22", "2", "5", "8", "11", "14", "17", "20", "0", "[SEP]"]
    xs = [70 + 74 * k for k in range(len(toks))]
    for k, (t, x) in enumerate(zip(toks, xs)):
        para = 4 <= k <= 13
        _chip(d, x, 360, t, TOK_FILL if para else "#eeeeee", TOK_STROKE if para else "#9a9a9a", w=62, size=14)
        _chip(d, x, 250, f"T_{{{k}}}", T_FILL, T_STROKE, w=62, size=14)
        d.line([(x, 343), (x, 270)], head=7, width=1.6, color=MUTED)
    d.rect(xs[0] - 34, 280, xs[-1] - xs[0] + 68, 42, BERT_FILL, BERT_STROKE, rx=6, sw=2)
    d.text((xs[0] + xs[-1]) / 2, 301, "BERT (fine-tuned)", 16, weight="bold")
    d.text((xs[1] + xs[2]) / 2, 400, "Question (A)", 14, color=MUTED)
    d.text((xs[4] + xs[13]) / 2, 400, "Paragraph (B): only these can be the answer", 14, color=MUTED)
    d.line([(xs[9], 230), (xs[9], 170)], color=HEAD_RED, width=3, head=10)
    d.line([(xs[12], 230), (xs[12], 170)], color=HEAD_RED, width=3, head=10)
    d.text(xs[9], 150, "start: S·T_9 = 6.07", 15)
    d.text(xs[12] + 20, 150, "end: E·T_{12} = 6.24", 15)
    d.rect(xs[9] - 36, 205, xs[12] - xs[9] + 72, 26, "none", HEAD_RED, rx=6, sw=2.2, dash="6 4")
    d.note(60, 40, 1060, 76)
    d.text(590, 62, "P_{start}(i) = softmax_i(S·T_i),  P_{end}(j) = softmax_j(E·T_j),  predicted span = argmax over j ≥ i of  S·T_i + E·T_j", 16)
    d.text(590, 92, "real output of this page's tiny BERT: question \"11 20\" → answer \"11 14 17 20\" (score 12.30)", 15, color=MUTED)
    return d


@diagram("bert-encoder-only")
def _bert_enc_only():
    d = figure1()
    d.title = "BERT keeps only the left half of Transformer Figure 1"
    d.parts.append('<rect x="420" y="30" width="492" height="1082" rx="16" fill="white" opacity="0.84"/>')
    d.parts.append('<rect x="272" y="476" width="150" height="26" fill="white"/>')
    d.line([(270, 566), (270, 430)], head=10, width=2.8)
    d.text(270, 405, "T_1 ... T_n", 18)
    d.text(270, 383, "one vector per token", 14, color=MUTED)
    d.note(470, 420, 420, 260)
    d.text(680, 455, "BERT = the encoder stack only", 19, weight="bold")
    d.text(680, 490, "no decoder, no cross-attention, no mask", 16)
    d.text(680, 525, "BASE: L = 12, H = 768, A = 12", 16)
    d.text(680, 555, "LARGE: L = 24, H = 1024, A = 16", 16)
    d.text(680, 590, "learned position embeddings (512)", 16)
    d.text(680, 620, "+ segment embeddings (A / B)", 16)
    d.text(680, 650, "GELU in the Feed Forward (4H)", 16)
    return d


# =============================================================================
# Stacked encoder blocks: the mini sentiment project (notebook 18)
# =============================================================================
@diagram("stacked-encoder-classifier")
def _stacked_classifier():
    d = Diagram(1320, 440, "Mini sentiment project: token + position embeddings, N stacked encoder blocks, masked mean pooling, classifier")
    y = 200
    toks = ["not", "good", "movie", "PAD", "PAD", "PAD"]
    for k, t in enumerate(toks):
        cy = 95 + 42 * k
        d.rect(30, cy - 15, 74, 30, OP_FILL if t != "PAD" else FROZEN_FILL, OP_STROKE if t != "PAD" else FROZEN_STROKE, rx=6, sw=1.6)
        d.text(67, cy, t, 14)
    d.text(67, 72, "input\\_ids", 14, color=MUTED)
    d.text(67, 365, "mask 1 1 1 0 0 0", 13, color=MUTED)
    d.line([(108, y), (150, y)], head=9, width=2.6)
    d.layer(215, y - 40, "Token Emb.", w=120, size=15)
    d.layer(215, y + 40, "Position Emb.", w=120, size=15)
    d.op(305, y, "+", r=16, size=19)
    d.line([(275, y - 40), (290, y - 12)], head=7, width=2)
    d.line([(275, y + 40), (290, y + 12)], head=7, width=2)
    d.line([(150, y), (150, y - 40), (155, y - 40)], arrow=False, width=2.2, radius=4)
    d.line([(150, y), (150, y + 40), (155, y + 40)], arrow=False, width=2.2, radius=4)
    d.text(305, y + 90, "[B, 6, 32]", 14, color=MUTED)
    d.cell(345, 90, 470, 220)
    d.text(360, 108, "× N encoder blocks (N = 1 or 3)", 15, "start", "#4d7a32", "bold")
    xs = [430, 580, 730]
    d.line([(321, y), (378, y)], head=9, width=2.6)
    for k, x in enumerate(xs):
        d.layer(x, y, f"Block {k + 1}", w=104, h=56, size=16)
        d.text(x, y + 48, "MHSA + Add&Norm", 12, color=MUTED)
        d.text(x, y + 64, "FFN + Add&Norm", 12, color=MUTED)
        if k < 2:
            d.line([(x + 52, y), (xs[k + 1] - 54, y)], head=9, width=2.6)
    d.text(580, 135, "every block: [B, 6, 32] → [B, 6, 32]", 14, color=MUTED)
    d.line([(782, y), (880, y)], head=9, width=2.6)
    d.op(930, y, "masked mean", r=22, size=13, w=118)
    d.text(930, y - 38, "[B, 32]", 14, color=MUTED)
    d.line([(989, y), (1040, y)], head=9, width=2.6)
    d.layer(1090, y, "Linear 32→2", w=104, size=15)
    d.text(1090, y + 42, "[B, 2] logits", 14, color=MUTED)
    d.line([(1142, y), (1190, y)], head=9, width=2.6)
    d.node(1240, y, "softmax", "out", r=40, size=13)
    d.text(1240, y + 58, "NEGATIVE / POSITIVE", 13, color=MUTED)
    # the mask feeds attention (keys) and pooling
    d.line([(67, 380), (67, 400), (930, 400), (930, y + 24)], color=GRAD, width=2, dash="7 5", head=8, radius=10)
    for x in xs:
        d.line([(x, 400), (x, y + 30 + 44)], color=GRAD, width=1.6, dash="5 5", head=7)
    d.text(500, 420, "attention\\_mask: padded keys get −1e9 before softmax, and are left out of the pooling average", 14, color=GRAD)
    return d


@diagram("masked-mean-pooling")
def _masked_pool():
    d = Diagram(1180, 430, "Masked mean pooling: average only the real tokens")
    toks = ["not", "good", "movie", "PAD", "PAD", "PAD"]
    vals = [[-1.827, -0.619, -1.658, -1.182], [-1.908, -0.945, -1.456, -1.385], [-1.573, -1.026, -1.719, -1.178],
            [-1.919, -1.145, -1.598, -1.069], [-1.543, -1.148, -1.522, -1.42], [-1.814, -1.004, -1.463, -1.457]]
    x0 = 150
    for k, (t, v) in enumerate(zip(toks, vals)):
        x = x0 + 112 * k
        real = t != "PAD"
        d.text(x, 62, t, 15, weight="bold" if real else "normal", color=INK if real else MUTED)
        for r, val in enumerate(v):
            d.rect(x - 46, 80 + 34 * r, 92, 30, CELL_FILL if real else FROZEN_FILL, CELL_STROKE if real else FROZEN_STROKE, rx=4, sw=1.4)
            d.text(x, 95 + 34 * r, f"{val:.3f}", 14, color=INK if real else MUTED)
        d.text(x, 246, "..." , 14, color=MUTED)
        d.rect(x - 22, 268, 44, 30, OP_FILL if real else FROZEN_FILL, OP_STROKE if real else FROZEN_STROKE, rx=4, sw=1.6)
        d.text(x, 283, "1" if real else "0", 16)
    d.text(60, 160, "last hidden", 14, "middle", MUTED)
    d.text(60, 178, "state T_t", 14, "middle", MUTED)
    d.text(60, 283, "mask", 14, "middle", MUTED)
    d.text(60, 246, "(32 dims)", 13, "middle", MUTED)
    d.note(830, 70, 320, 230)
    d.text(990, 98, "pooled = Σ mask_t T_t / Σ mask_t", 15)
    d.text(990, 130, "= (T_{not} + T_{good} + T_{movie}) / 3", 14, color=MUTED)
    pooled = [-1.769, -0.863, -1.611, -1.248]
    plain = [-1.764, -0.981, -1.569, -1.282]
    d.text(925, 165, "masked", 14, weight="bold"); d.text(1060, 165, "plain mean", 14, color=MUTED)
    for r in range(4):
        d.text(925, 192 + 24 * r, f"{pooled[r]:.3f}", 15)
        d.text(1060, 192 + 24 * r, f"{plain[r]:.3f}", 15, color=MUTED)
    d.line([(800, 185), (828, 185)], head=8, width=2.2)
    d.text(590, 345, "real numbers: first 4 of 32 dimensions of the trained 3-block model's last hidden state for \"not good movie\"", 14, color=MUTED)
    d.text(590, 372, "a plain mean would also average the three PAD rows, so the sentence vector would depend on how much padding it got", 14, color=MUTED)
    return d


# =============================================================================
# Hugging Face pipelines and what they hide (notebook 19)
# =============================================================================
@diagram("hf-pipeline-internals")
def _hf_pipeline():
    d = Diagram(1340, 470, "What pipeline(\"sentiment-analysis\") does: preprocess, forward, postprocess")
    d.rect(30, 70, 1280, 300, NOTE_FILL, NOTE_STROKE, rx=18, sw=1.6, dash="8 6")
    d.text(50, 92, "pipeline(\"sentiment-analysis\")", 16, "start", MUTED, "bold")
    y = 215
    d.node(95, y, "text", "in", r=40, size=15)
    d.text(95, y + 62, "\"The movie was", 13, color=MUTED)
    d.text(95, y + 80, "surprisingly good.\"", 13, color=MUTED)
    d.line([(135, y), (190, y)], head=9, width=2.6)
    # stage 1
    d.cell(195, 120, 300, 200, "1 · preprocess")
    d.layer(345, y - 30, "AutoTokenizer", w=170, size=16)
    d.text(345, y + 22, "WordPiece + [CLS] … [SEP]", 13, color=MUTED)
    d.text(345, y + 44, "input\\_ids, attention\\_mask", 13, color=MUTED)
    d.line([(495, y), (550, y)], head=9, width=2.6)
    # stage 2
    d.cell(555, 120, 330, 200, "2 · forward")
    d.layer(720, y - 30, "", w=230, h=54, size=16)
    d.text(720, y - 41, "AutoModelFor", 16)
    d.text(720, y - 19, "SequenceClassification", 15)
    d.text(720, y + 30, "DistilBERT encoder → head on [CLS]", 13, color=MUTED)
    d.text(720, y + 52, "torch.no\\_grad(), model.eval()", 13, color=MUTED)
    d.line([(885, y), (940, y)], head=9, width=2.6)
    # stage 3
    d.cell(945, 120, 230, 200, "3 · postprocess")
    d.op(1000, y - 20, "softmax", r=18, size=13, w=86)
    d.op(1110, y - 20, "argmax", r=18, size=13, w=80)
    d.line([(1043, y - 20), (1068, y - 20)], head=7, width=2)
    d.text(1060, y + 30, "config.id2label", 13, color=MUTED)
    d.line([(1175, y), (1215, y)], head=9, width=2.6)
    d.node(1260, y, "label", "out", r=40, size=14)
    d.text(1255, y + 62, "{label, score}", 13, color=MUTED)
    # tensors between stages
    d.text(522, y - 30, "ids", 13, color=INK)
    d.text(912, y - 30, "logits", 13, color=INK)
    d.text(670, 410, "everything inside the dashed box is what the manual version on this page writes out step by step", 15, color=MUTED)
    return d

@diagram("distilbert-sequence-classification")
def _distilbert_cls():
    d = Diagram(1340, 470, "DistilBertForSequenceClassification (distilbert-base-uncased-finetuned-sst-2-english)")
    y = 230
    toks = ["[CLS]", "the", "movie", "...", "[SEP]"]
    for k, t in enumerate(toks):
        cy = 120 + 55 * k
        d.rect(30, cy - 17, 86, 34, FROZEN_FILL if t.startswith("[") else OP_FILL, FROZEN_STROKE if t.startswith("[") else OP_STROKE, rx=6, sw=1.6)
        d.text(73, cy, t, 14)
    d.line([(120, y), (165, y)], head=9, width=2.6)
    d.layer(250, y - 52, "word emb.", w=150, size=15)
    d.text(250, y - 22, "30,522 × 768", 13, color=MUTED)
    d.layer(250, y + 28, "position emb.", w=150, size=15)
    d.text(250, y + 58, "512 × 768 (learned)", 13, color=MUTED)
    d.text(250, y + 90, "no segment embeddings", 13, color=GRAD)
    d.op(355, y, "+", r=16, size=19)
    d.line([(371, y), (410, y)], head=9, width=2.6)
    d.layer(450, y, "LN", w=56, size=15)
    d.line([(478, y), (515, y)], head=9, width=2.6)
    d.cell(520, 110, 300, 240)
    d.text(535, 128, "× 6 Transformer blocks", 15, "start", "#4d7a32", "bold")
    d.layer(670, y - 52, "Multi-head self-attn", w=230, size=15)
    d.text(670, y - 22, "12 heads × 64, + Add & Norm", 13, color=MUTED)
    d.layer(670, y + 22, "FFN 768 → 3072 → 768", w=230, size=15)
    d.text(670, y + 52, "GELU, + Add & Norm", 13, color=MUTED)
    d.text(670, 330, "66,362,880 parameters (pre-trained)", 13, color=MUTED)
    d.line([(820, y), (870, y)], head=9, width=2.6)
    d.text(845, y - 18, "h[CLS]", 13)
    d.layer(940, y, "pre\\_classifier", w=130, size=14)
    d.text(940, y + 32, "768 × 768", 12, color=MUTED)
    d.line([(1005, y), (1030, y)], head=7, width=2.2)
    d.op(1055, y, "ReLU", r=16, size=12, w=50)
    d.line([(1080, y), (1100, y)], head=7, width=2.2)
    d.op(1130, y, "drop 0.2", r=16, size=11, w=58)
    d.line([(1159, y), (1180, y)], head=7, width=2.2)
    d.layer(1225, y, "classifier", w=86, size=14)
    d.text(1225, y + 32, "768 × 2", 12, color=MUTED)
    d.text(1080, y - 50, "head: 592,130 parameters (fine-tuned on SST-2)", 13, color=MUTED)
    d.line([(1268, y), (1300, y), (1300, 380)], head=9, width=2.6, radius=8)
    d.text(1320, 400, "logits [NEGATIVE, POSITIVE]", 14, "end")
    d.text(1320, 422, "[−4.2341, 4.6189]", 14, "end", MUTED)
    d.text(420, 430, "total 66,955,010 parameters; only the [CLS] position's final vector feeds the head", 14, color=MUTED)
    return d


# -----------------------------------------------------------------------------
def main(names):
    os.makedirs(OUT, exist_ok=True)
    for n in names or DIAGRAMS:
        DIAGRAMS[n]().save(os.path.join(OUT, f"{n}.svg"))
        print("wrote", n)


if __name__ == "__main__":
    main(sys.argv[1:])
