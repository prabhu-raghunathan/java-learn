# Animations (Manim)

Short looping animations for the AI/ML pages, written in [Manim](https://www.manim.community/)
and embedded with the `<Animation>` component.

| File | What it holds |
|---|---|
| `style.py` | Shared colours, fonts and helpers. Every scene subclasses `WikiScene` |
| `pytorch_scenes.py`, `ann_scenes.py`, `cnn_scenes.py` | One file per topic, several `Scene` classes each |
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
