# Animations (Manim)

Short looping animations for the AI/ML pages, written in [Manim](https://www.manim.community/)
and embedded with the `<Animation>` component.

| File | What it holds |
|---|---|
| `style.py` | Shared colours, fonts and helpers. Every scene subclasses `WikiScene` |
| `pytorch_scenes.py`, `ann_scenes.py`, `cnn_scenes.py`, `transfer_scenes.py` | One file per topic, several `Scene` classes each |
| `render.py` | Renders scenes → `static/animations/<topic>/<name>.{mp4,webm,jpg}` |

## Render

```bash
pip install -r animations/requirements.txt     # plus ffmpeg, pango, LaTeX (see render.py)
python animations/render.py --list             # all keys
python animations/render.py cnn/convolution    # one animation
python animations/render.py                    # everything
```

## Embed in a page

```mdx
<Animation src="cnn/convolution" caption="One sentence saying what to watch for." />
```

No import needed: the component is registered in `src/theme/MDXComponents.js`.

## Add a new animation

1. Write a `Scene` (subclass `WikiScene`) in the topic's `*_scenes.py` file (or a new one).
2. Add a `"topic/name": ("file.py", "SceneClass")` entry to `MANIFEST` in `render.py`.
3. `python animations/render.py topic/name`, then check the frames (see CLAUDE.md).
4. Embed it with `<Animation src="topic/name" caption="..." />`.

## Core Java scenes

`java_scenes.py` holds the Java memory/bytecode animations (`jvm/gc-generations`, `oop/init-order`, `variables/pass-by-value`, `operators/postfix-trap`). They use the same `WikiScene` helpers; values shown must match what the page's verified code prints.

## Python Fundamentals scenes

`python_scenes.py` holds the 18 animations for `docs/03-ai-ml/00-python-fundamentals/` (keys `python/...`). They subclass `PyScene` (a `WikiScene` with `say()` / `code()` helpers that cross-fade captions and code lines) and use `box()`, `code_block()` (keeps indentation, which `Text` strips) and `arrow()`. Numbers shown (list capacities, dict probe slots, GIL and asyncio timings) come from the verified runs quoted on the pages. Avoid `NumberLine(include_numbers=True)`: it needs LaTeX/dvisvgm; draw tick labels with `label()` instead.

`data_scenes.py` holds the 18 animations for `docs/03-ai-ml/01-python-for-data-exploration/` (keys `numpy/...`, `pandas/...`, `matplotlib/...`, `seaborn/...`, `capstone/...`). It reuses `PyScene`, `box()`, `retext()` and `arrow()` from `python_scenes.py` and adds `cell()`, `row_of()` and `col_of()` for table-like layouts. `HistogramBinning` and `BoxplotQuartiles` compute their numbers from ShopStream at render time (`scripts/data_exploration/`), so the Manim venv also needs `pandas==3.0.6` and `pyarrow` installed.
