#!/usr/bin/env python3
"""
Render Manim scenes into the site's static folder.

    python animations/render.py                 # render everything in MANIFEST
    python animations/render.py cnn/convolution # render one (or several) keys
    python animations/render.py --list          # show all keys

Each key "topic/name" produces:
    static/animations/topic/name.webm  (VP9 fallback for browsers without H.264)
    static/animations/topic/name.mp4   (H.264, first choice; web-optimised with faststart)
    static/animations/topic/name.jpg   (poster = final frame, shown before playback
                                        and for visitors with reduced-motion enabled)
and is embedded in a page with:
    <Animation src="topic/name" caption="..." />

Requirements: Python 3.10+, `pip install manim`, ffmpeg, Pango/Cairo, and a LaTeX
install with dvisvgm (only for scenes that use MathTex).
macOS: brew install ffmpeg pango && brew install --cask mactex-no-gui && pip install manim
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # animations/
SITE = ROOT.parent                              # project root
OUT = SITE / "static" / "animations"
MEDIA = ROOT / ".media"                         # manim's scratch output (git-ignored)

# key -> (scene file inside animations/, Scene class name)
MANIFEST = {
    # PyTorch Fundamentals
    "pytorch/broadcasting":        ("pytorch_scenes.py", "Broadcasting"),
    "pytorch/autograd-backward":   ("pytorch_scenes.py", "AutogradBackward"),
    "pytorch/zero-grad":           ("pytorch_scenes.py", "ZeroGradAccumulation"),
    # Artificial Neural Networks
    "ann/forward-backward":        ("ann_scenes.py", "ForwardBackward"),
    "ann/relu-kinks":              ("ann_scenes.py", "ReluKinks"),
    "ann/learning-rate":           ("ann_scenes.py", "LearningRate"),
    # CNN Fundamentals
    "cnn/convolution":             ("cnn_scenes.py", "ConvolutionSlide"),
    "cnn/stride-padding":          ("cnn_scenes.py", "StridePadding"),
    "cnn/multichannel-filter":     ("cnn_scenes.py", "MultiChannelFilter"),
    "cnn/max-pooling":             ("cnn_scenes.py", "MaxPooling"),
    # CNN Architecture, Training, Dropout & AlexNet
    "cnn/shape-funnel":            ("cnn_scenes.py", "ShapeFunnel"),
    "cnn/dropout":                 ("cnn_scenes.py", "DropoutTrainEval"),
}

QUALITY = "-qm"   # 1280x720 @ 30 fps: crisp text, small files (~0.2–0.6 MB each)


def render(key: str) -> None:
    scene_file, scene = MANIFEST[key]
    out_name = key.replace("/", "__")
    cmd = [
        sys.executable, "-m", "manim", QUALITY, "--format=mp4",
        "--media_dir", str(MEDIA), "-o", out_name,
        str(ROOT / scene_file), scene,
    ]
    print(f"▶ {key}  ({scene_file}::{scene})")
    subprocess.run(cmd, check=True, cwd=ROOT)

    produced = next(MEDIA.rglob(f"{out_name}.mp4"))
    dest = OUT / f"{key}.mp4"
    dest.parent.mkdir(parents=True, exist_ok=True)

    # H.264 MP4: first choice (Safari, Chrome, Edge, Firefox); faststart = plays while downloading
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(produced), "-c:v", "libx264", "-preset", "slow",
         "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dest)],
        check=True,
    )
    # VP9 WebM: fallback for browsers that can't decode H.264 (e.g. open-source Chromium builds)
    webm = OUT / f"{key}.webm"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(produced), "-c:v", "libvpx-vp9", "-crf", "38",
         "-b:v", "0", "-row-mt", "1", "-deadline", "good", "-cpu-used", "4", "-an", str(webm)],
        check=True,
    )

    # poster = last frame (the "finished" state is the most informative still)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-sseof", "-0.1", "-i", str(dest),
         "-frames:v", "1", "-q:v", "4", str(OUT / f"{key}.jpg")],
        check=True,
    )
    print(f"  ✓ {key}: webm {webm.stat().st_size / 1024:.0f} KB · mp4 {dest.stat().st_size / 1024:.0f} KB")


def main(argv):
    if "--list" in argv:
        print("\n".join(MANIFEST))
        return
    keys = [a for a in argv if not a.startswith("-")] or list(MANIFEST)
    unknown = [k for k in keys if k not in MANIFEST]
    if unknown:
        sys.exit(f"Unknown key(s): {unknown}. Use --list.")
    for key in keys:
        render(key)


if __name__ == "__main__":
    main(sys.argv[1:])
