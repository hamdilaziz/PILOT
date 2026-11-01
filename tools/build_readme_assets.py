from __future__ import annotations

from html import escape
import json
from pathlib import Path
import re

import cairosvg
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets" / "readme"
BRAND = ROOT / "assets" / "brand"
THEMES = {
    "dark": {"bg": "#0B1220", "panel": "#111F35", "ink": "#EEF5FF", "muted": "#A6B8D0", "line": "#263B55", "blue": "#78B4FF", "teal": "#5EEAD4", "purple": "#B4ACFF"},
    "light": {"bg": "#F3F7FC", "panel": "#FFFFFF", "ink": "#122743", "muted": "#526A83", "line": "#D7E3EF", "blue": "#2563EB", "teal": "#087F78", "purple": "#7252C9"},
}
FONT = "DejaVu Sans, Arial, sans-serif"


def text(x, y, value, size=18, color="#EEF5FF", weight=400, **attrs):
    properties = " ".join(f'{key.replace("_", "-")}="{escape(str(value))}"' for key, value in attrs.items())
    return f'<text x="{x}" y="{y}" fill="{color}" font-family="{FONT}" font-size="{size}" font-weight="{weight}" {properties}>{escape(value)}</text>'


def rect(x, y, width, height, fill, stroke="none", radius=12, **attrs):
    properties = " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in attrs.items())
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" fill="{fill}" stroke="{stroke}" {properties}/>'


def svg(width, height, content, title, description):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description"><title id="title">{escape(title)}</title><desc id="description">{escape(description)}</desc>{content}</svg>\n'


def mark(color="#5EEAD4", ink="#EEF5FF", scale=1, x=0, y=0):
    return f'''<g transform="translate({x} {y}) scale({scale})" fill="none" stroke-linecap="round" stroke-linejoin="round">
<path d="M2 20V2H20M44 2H62V20M62 44V62H44M20 62H2V44" stroke="{color}" stroke-width="3"/>
<path d="M23 46V19H34C49 19 49 36 34 36H23" stroke="{ink}" stroke-width="5"/></g>'''


def banner(theme="dark", social=False):
    c = THEMES[theme]
    content = rect(0, 0, 1280, 440, c["bg"], radius=24)
    for x in range(714, 1250, 32):
        for y in range(32, 414, 32):
            content += f'<circle cx="{x}" cy="{y}" r="1" fill="{c["line"]}"/>'
    content += mark(c["teal"], c["ink"], 0.43, 50, 36)
    content += text(91, 56, "DOCUMENT INTELLIGENCE / OPEN RESEARCH", 12, c["muted"], 600, letter_spacing=1.5)
    content += text(48, 177, "PILOT", 108, c["ink"], 700, letter_spacing=-4)
    content += text(54, 238, "Read text. Keep the layout.", 31, c["ink"], 500, letter_spacing=-0.7)
    content += text(54, 287, "A Promptable Interleaved Layout-aware", 19, c["muted"])
    content += text(54, 315, "OCR Transformer", 19, c["muted"])
    for x, width, label, color in ((54, 103, "4 TASKS", c["teal"]), (169, 178, "CNN + mBART", c["blue"]), (359, 139, "IJDAR 2026", c["purple"])):
        content += rect(x, 359, width, 33, c["panel"], c["line"], 8)
        content += text(x + 14, 380, label, 12, color, 600, letter_spacing=0.6)
    content += rect(745, 48, 292, 330, c["panel"], c["line"], 16)
    content += text(771, 81, "DOCUMENT", 11, c["muted"], 600, letter_spacing=1.6)
    content += rect(771, 103, 130, 11, c["ink"], radius=3, opacity=0.9)
    for y, width in ((139, 223), (157, 185), (175, 213), (218, 188), (236, 217), (279, 147), (297, 183), (315, 201)):
        content += rect(772, y, width, 6, c["muted"], radius=2, opacity=0.38)
    for x, y, w, h, color in ((761, 127, 245, 62, c["blue"]), (761, 207, 237, 48, c["teal"]), (761, 268, 222, 66, c["purple"])):
        content += rect(x, y, w, h, "none", color, 3, stroke_width=1.5)
        content += rect(x, y - 3, 6, 6, color, radius=0)
    content += f'<path d="M1008 155H1104V192" fill="none" stroke="{c["teal"]}" stroke-width="2"/>'
    content += f'<circle cx="1008" cy="155" r="4" fill="{c["teal"]}"/>'
    content += rect(890, 195, 332, 183, c["bg"], c["line"], 14)
    content += text(912, 225, "ONE OUTPUT SEQUENCE", 10, c["muted"], 600, letter_spacing=1.3)
    content += text(912, 261, "<ocr_with_boxes>", 18, c["teal"], 500)
    content += text(912, 294, "<x_12><y_4>", 15, c["blue"])
    content += text(912, 320, "Text, with its place.", 16, c["ink"])
    content += text(912, 347, "<x_48><y_9><sep/>", 15, c["purple"])
    if social:
        content = rect(0, 0, 1280, 640, c["bg"], radius=0) + f'<g transform="translate(0 74)">{content}</g>'
        content += text(54, 586, "RESEARCH CODE  /  MODEL WEIGHTS  /  FINE-TUNING", 13, c["muted"], 500, letter_spacing=1.6)
        return svg(1280, 640, content, "PILOT", "Promptable document OCR with text and layout in one generated sequence.")
    return svg(1280, 440, content, "PILOT — Read text. Keep the layout.", "A CNN encoder and mBART decoder support four prompted OCR tasks. The document and token stream are schematic illustrations.")


def architecture(theme):
    c = THEMES[theme]
    content = rect(0, 0, 1280, 270, c["bg"], radius=20)
    content += text(32, 39, "FROM PIXELS TO A STRUCTURED SEQUENCE", 12, c["muted"], 600, letter_spacing=1.4)
    nodes = [(32, 158, "Document", "RGB image", c["ink"]), (252, 247, "CNN encoder", "Visual features + 2D positions", c["blue"]), (561, 239, "mBART decoder", "4 layers · cross-attention", c["teal"]), (862, 386, "Text + layout tokens", "Autoregressive output sequence", c["purple"])]
    for x, width, title, subtitle, color in nodes:
        content += rect(x, 119, width, 96, c["panel"], c["line"], 12)
        content += rect(x + 18, 137, 25, 3, color, radius=1)
        content += text(x + 18, 167, title, 18, c["ink"], 600)
        content += text(x + 18, 193, subtitle, 12, c["muted"])
    for left, right in ((190, 252), (499, 561), (800, 862)):
        content += f'<path d="M{left + 12} 167H{right - 14}m-7-5 7 5-7 5" fill="none" stroke="{c["muted"]}" stroke-width="1.7"/>'
    content += rect(594, 61, 173, 36, c["panel"], c["teal"], 8)
    content += text(617, 85, "Task prompt", 14, c["teal"], 500)
    content += f'<path d="M680 98V115m-4-5 4 5 4-5" fill="none" stroke="{c["teal"]}" stroke-width="1.7"/>'
    return svg(1280, 270, content, "PILOT architecture", "The image passes through the CNN encoder and 2D positional encoding. The task prompt and visual features condition the four-layer mBART decoder, which generates interleaved text and layout tokens.")


def font(size, bold=False, mono=False):
    name = "DejaVuSansMono.ttf" if mono else "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(name, size)


def demo_frame(count, image, result, segments):
    c = THEMES["dark"]
    frame = Image.new("RGB", (1280, 650), c["bg"])
    draw = ImageDraw.Draw(frame)
    draw.text((34, 25), "PILOT / HANDWRITING + LAYOUT", font=font(15, True), fill=c["teal"])
    draw.text((34, 61), "Read the page. Recover its structure.", font=font(27, True), fill=c["ink"])
    draw.rounded_rectangle((34, 127, 618, 590), 14, fill=c["panel"], outline=c["line"])
    draw.text((54, 144), "INPUT + PREDICTED BOXES", font=font(12, True), fill=c["muted"])
    page = image.copy()
    page.thumbnail((548, 381), Image.Resampling.LANCZOS)
    origin = (34 + (584 - page.width) // 2, 189 + (381 - page.height) // 2)
    frame.paste(page, origin)
    sx, sy = page.width / image.width, page.height / image.height
    for i, box in enumerate(result["boxes"][:count]):
        x1, y1, x2, y2 = box
        color = c["teal"] if i == count - 1 else c["blue"]
        draw.rectangle((origin[0] + x1 * sx, origin[1] + y1 * sy, origin[0] + x2 * sx, origin[1] + y2 * sy), outline=color, width=2)
    draw.rounded_rectangle((645, 127, 1246, 590), 14, fill=c["panel"], outline=c["line"])
    draw.text((668, 147), "PROMPT", font=font(12, True), fill=c["muted"])
    draw.text((668, 174), result["prompt"], font=font(22, mono=True), fill=c["teal"])
    draw.line((668, 215, 1222, 215), fill=c["line"], width=1)
    draw.text((668, 235), "GENERATED TEXT", font=font(12, True), fill=c["muted"])
    draw.text((1120, 235), f"{count:02d} / {len(segments):02d}", font=font(12, mono=True), fill=c["blue"])
    lines = result["cleaned_text"].splitlines()
    for i, line in enumerate(lines[:count]):
        y = 266 + i * 29
        draw.ellipse((670, y + 7, 675, y + 12), fill=c["teal"] if i == count - 1 else c["blue"])
        draw.text((689, y), line, font=font(18), fill=c["ink"])
    if count:
        current = segments[count - 1]
        coords = re.findall(r"<[xy]_\d+>", current)
        draw.rounded_rectangle((666, 515, 1223, 565), 7, fill=c["bg"])
        draw.text((680, 531), "BOX  " + "".join(coords), font=font(15, mono=True), fill=c["purple"])
    draw.text((35, 613), "RECORDED PILOT OUTPUT  /  IAM SAMPLE  /  ANIMATED REPLAY, NOT A LATENCY BENCHMARK", font=font(12), fill=c["muted"])
    return frame


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    BRAND.mkdir(parents=True, exist_ok=True)
    for theme in tqdm(THEMES, desc="Render theme assets"):
        for name, content in (("hero", banner(theme)), ("architecture", architecture(theme))):
            path = ASSETS / f"{name}-{theme}.svg"
            path.write_text(content, encoding="utf-8")
            cairosvg.svg2png(bytestring=content.encode(), write_to=str(path.with_suffix(".png")), scale=2)
    icon = svg(68, 68, rect(0, 0, 68, 68, THEMES["dark"]["bg"], radius=14) + mark(scale=0.8, x=8, y=8), "PILOT mark", "A letter P framed by document localization corners.")
    (BRAND / "pilot-mark.svg").write_text(icon, encoding="utf-8")
    cairosvg.svg2png(bytestring=icon.encode(), write_to=str(BRAND / "pilot-mark.png"), output_width=272, output_height=272)
    cairosvg.svg2png(bytestring=banner("dark", social=True).encode(), write_to=str(BRAND / "social-preview.png"))
    result = json.loads((ROOT / "outputs" / "iam_ocr_with_boxes.json").read_text())
    segments = result["raw_prediction"].split("<sep/>")
    assert len(segments) == len(result["boxes"]) == len(result["cleaned_text"].splitlines())
    with Image.open(ROOT / "input_samples" / "iam.png") as source:
        image = source.convert("RGB")
    counts = [len(segments), 0, *range(1, len(segments) + 1)]
    frames = [demo_frame(n, image, result, segments) for n in tqdm(counts, desc="Animate recorded output")]
    frames[0].save(ASSETS / "demo-static.png", optimize=True)
    palette = frames[0].quantize(colors=128)
    indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    indexed[0].save(ASSETS / "demo.gif", save_all=True, append_images=indexed[1:], duration=[1800, 450, *([500] * (len(segments) - 1)), 2400], loop=0, optimize=True, disposal=2)
    for path in sorted((ROOT / "assets").rglob("*")):
        if path.is_file():
            print(f"{path.relative_to(ROOT)}: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
