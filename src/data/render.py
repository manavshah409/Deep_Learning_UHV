"""Readable, deterministic annotation drawing with one color per category."""

import colorsys
from PIL import ImageFont


def draw_box(draw, box, label, class_id, nclasses=14):
    rgb = colorsys.hsv_to_rgb(class_id / max(nclasses, 1), 0.8, 1)
    color = tuple(round(c * 255) for c in rgb)
    font = ImageFont.load_default(size=19)
    draw.rectangle(box, outline=color, width=3)
    x, y = box[0], max(0, box[1] - 23)
    bounds = draw.textbbox((x, y), label, font=font)
    draw.rectangle(bounds, fill=color)
    draw.text((x, y), label, font=font, fill="black")
