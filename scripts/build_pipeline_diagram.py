"""Build matching SVG and looping GIF from the actual frame-processing graph."""

from html import escape
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs/media"
W, H = 1440, 960
BG, TEXT, MUTED = "#0b1220", "#e8eef8", "#a7b7cc"
GREEN, BLUE, GOLD = "#67dec0", "#88baff", "#ffc77a"
# Shared layout: the pipeline detects/analyses before tracking and scoring.
NODES = [
    ("input", (40, 140, 290, 230), "Video / camera / API", "Decode every frame", BLUE),
    ("person", (365, 140, 645, 230), "YOLOv8n · ONNX", "CUDA / CPU person boxes", BLUE),
    (
        "crop",
        (725, 140, 1005, 230),
        "Head crop + retries",
        "Upper 40% · padded crop",
        BLUE,
    ),
    (
        "mesh",
        (1080, 140, 1400, 230),
        "MediaPipe Face Landmarker",
        "Current facial landmarks · CPU",
        BLUE,
    ),
    (
        "track",
        (365, 380, 645, 470),
        "IoU association",
        "One-to-one IDs · IoU >= 0.5",
        GREEN,
    ),
    (
        "metrics",
        (1080, 380, 1400, 470),
        "Head pose + eye signals",
        "solvePnP · EAR · iris ratio",
        GREEN,
    ),
    (
        "score",
        (1080, 610, 1400, 700),
        "Weighted attention rule",
        "Current-frame score + label",
        GOLD,
    ),
    (
        "join",
        (365, 610, 645, 700),
        "Tracked observation",
        "ID + face bounds + result",
        GOLD,
    ),
    (
        "json",
        (40, 810, 440, 900),
        "JSON summary + JSONL",
        "Person box retained even without face",
        GREEN,
    ),
    (
        "video",
        (790, 810, 1400, 900),
        "Annotated H.264 video / preview",
        "Face-only boxes · name + attention label",
        BLUE,
    ),
]
EDGES = [
    ([(290, 185), (365, 185)], BLUE),
    ([(645, 185), (725, 185)], BLUE),
    ([(1005, 185), (1080, 185)], BLUE),
    ([(505, 230), (505, 380)], GREEN),
    ([(1240, 230), (1240, 380)], GREEN),
    ([(1240, 470), (1240, 610)], GOLD),
    ([(505, 470), (505, 610)], GREEN),
    ([(1080, 655), (645, 655)], GOLD),
    ([(505, 700), (505, 755), (240, 755), (240, 810)], GREEN),
    ([(505, 755), (1095, 755), (1095, 810)], BLUE),
    # No face still produces an observation and retains the person track.
    ([(1400, 185), (1420, 185), (1420, 540), (505, 540), (505, 610)], MUTED),
]
LABELS = [
    (40, 35, "MULTI-PERSON ATTENTION TRACKING", TEXT, 34),
    (
        40,
        86,
        "Shared CLI and upload API pipeline · session-local spatial IDs",
        MUTED,
        22,
    ),
    (525, 290, "Current person boxes", GREEN, 19),
    (930, 505, "No face: score 0 · no_face · not drawn", MUTED, 19),
    (720, 620, "Fresh measurements", GOLD, 19),
    (
        40,
        927,
        "Spatial tracking is not identity recognition. Visual attention scores are heuristics, not probabilities.",
        MUTED,
        18,
    ),
]


def font(size):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return (
        ImageFont.truetype(str(path), size)
        if path.exists()
        else ImageFont.load_default(size=size)
    )


def arrow(points):
    x, y = points[-1]
    a, b = points[-2]
    angle = math.atan2(y - b, x - a)
    return [
        (x, y),
        (x - 12 * math.cos(angle - 0.5), y - 12 * math.sin(angle - 0.5)),
        (x - 12 * math.cos(angle + 0.5), y - 12 * math.sin(angle + 0.5)),
    ]


def along(points, fraction):
    lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
    distance = fraction * sum(lengths)
    for a, b, length in zip(points, points[1:], lengths):
        if distance <= length:
            return (
                a[0] + (b[0] - a[0]) * distance / length,
                a[1] + (b[1] - a[1]) * distance / length,
            )
        distance -= length
    return points[-1]


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
        '<title id="title">Multi-person attention tracking pipeline</title>',
        '<desc id="desc">Video or camera frames feed person detection, head cropping and MediaPipe. Person boxes drive IoU tracking. Facial signals drive the weighted attention rule. Track IDs and results produce JSON and face-only annotations. Missing faces retain a tracked no_face record.</desc>',
        f'<rect width="100%" height="100%" fill="{BG}"/>',
    ]
    for points, color in EDGES:
        coords = " ".join(f"{x},{y}" for x, y in points)
        tri = " ".join(f"{x:.1f},{y:.1f}" for x, y in arrow(points))
        svg.extend(
            [
                f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3"/>',
                f'<polygon points="{tri}" fill="{color}"/>',
            ]
        )
    for _, box, title, subtitle, color in NODES:
        x1, y1, x2, y2 = box
        svg.append(
            f'<rect x="{x1}" y="{y1}" width="{x2 - x1}" height="{y2 - y1}" rx="14" fill="#172238" stroke="{color}" stroke-width="2"/>'
        )
        for text, size, y in [(title, 21, y1 + 35), (subtitle, 17, y1 + 64)]:
            svg.append(
                f'<text x="{(x1 + x2) / 2}" y="{y}" text-anchor="middle" fill="{TEXT if size == 21 else MUTED}" font-family="DejaVu Sans, sans-serif" font-size="{size}">{escape(text)}</text>'
            )
    for x, y, text, color, size in LABELS:
        svg.append(
            f'<text x="{x}" y="{y + size}" fill="{color}" font-family="DejaVu Sans, sans-serif" font-size="{size}">{escape(text)}</text>'
        )
    svg.append("</svg>")
    (DEST / "pipeline.svg").write_text("\n".join(svg) + "\n")
    base = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(base)
    for points, color in EDGES:
        draw.line(points, fill=color, width=3)
        draw.polygon(arrow(points), fill=color)
    for _, box, title, subtitle, color in NODES:
        draw.rounded_rectangle(box, radius=14, fill="#172238", outline=color, width=2)
        x1, y1, x2, y2 = box
        draw.text(
            ((x1 + x2) / 2, y1 + 27), title, font=font(21), fill=TEXT, anchor="mm"
        )
        draw.text(
            ((x1 + x2) / 2, y1 + 59), subtitle, font=font(17), fill=MUTED, anchor="mm"
        )
    for x, y, text, color, size in LABELS:
        draw.text((x, y), text, font=font(size), fill=color)
    frames = []
    for step in range(40):
        frame = base.copy()
        draw = ImageDraw.Draw(frame)
        for points, color in EDGES:
            x, y = along(points, step / 40)
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)
        frames.append(frame.quantize(colors=128))
    frames[0].save(
        DEST / "pipeline.gif",
        save_all=True,
        append_images=frames[1:],
        duration=70,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print("Created docs/media/pipeline.svg and pipeline.gif")


if __name__ == "__main__":
    main()
