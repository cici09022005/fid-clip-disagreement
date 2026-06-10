"""Replace 'KMC D-High' with 'KMC D-High' in the qualitative figure."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "ACCESS_latex_template_20240429" / "figures" / "fig6_qualitative_style.png"
OUT_PNG = ROOT / "kmc_lora" / "figures" / "qualitative" / "fig6_qualitative_style.png"
OUT_PDF = OUT_PNG.with_suffix(".pdf")

img = Image.open(SRC).convert("RGB")
draw = ImageDraw.Draw(img)

# Load font matching the original
def load_font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()

font = load_font(18, bold=True)

# The image is ~1168x586.  Each block is roughly half-width, half-height.
# Column headers appear at specific y-offsets within each block.
# I'll find "KMC D-High" text locations by scanning.
# From visual inspection:
# Top-left block (Anime-Student): KMC D-High label ~around x=254, y=28
# Top-right block (WikiArt-Mixed): ~around x=838, y=28
# Bottom-left block (DreamBooth-Mixed): ~around x=254, y=312
# Bottom-right block (DreamBooth-Single): ~around x=838, y=312

w, h = img.size
print(f"Image size: {w}x{h}")

# Strategy: paint white rectangles over "KMC D-High" and redraw "KMC D-High"
# The text "KMC D-High" in the original is about 18px bold.
# Let me locate them more precisely by looking at approximate positions.

# Block layout (from the generate script):
# outer_pad=28, block_gap_x=44, block_gap_y=42
# thumb=170, image_gap=16
# block_w = 3*170 + 2*16 = 542
# canvas_w = 28*2 + 542*2 + 44 = 1184
# dataset_title_h=28, prompt_h=22, method_label_h=24

outer_pad = 28
block_gap_x = 44
block_gap_y = 42
thumb = 170
image_gap = 16
block_w = 3 * thumb + 2 * image_gap  # 542
dataset_title_h = 28
prompt_h = 22
method_label_h = 24

# For each block, method labels start at:
# y = y0 + dataset_title_h + prompt_h
# x = x0 + method_idx * (thumb + image_gap) + centering

# Method index 1 is "KMC D-High" (index 0=Random, 1=KMC, 2=Anti)
method_idx = 1

for row in range(2):
    for col in range(2):
        x0 = outer_pad + col * (block_w + block_gap_x)
        y0 = outer_pad + row * (block_w // 3 * 1.3 + block_gap_y)  # approximate

        # More precise: from the script, block_h = dataset_title_h + prompt_h + method_label_h + thumb
        block_h = dataset_title_h + prompt_h + method_label_h + thumb
        y0_actual = outer_pad + row * (block_h + block_gap_y)

        label_y = y0_actual + dataset_title_h + prompt_h
        x_img = x0 + method_idx * (thumb + image_gap)

        # Center the text in the thumb width
        old_text = "KMC D-High"
        new_text = "KMC D-High"

        bbox_old = draw.textbbox((0, 0), old_text, font=font)
        text_w = bbox_old[2] - bbox_old[0]
        text_h = bbox_old[3] - bbox_old[1]

        text_x = x_img + (thumb - text_w) / 2
        text_y = label_y

        # White out the old text area (with some margin)
        margin = 3
        draw.rectangle(
            [text_x - margin, text_y - margin,
             text_x + text_w + margin, text_y + text_h + margin],
            fill="white"
        )

        # Draw new text
        bbox_new = draw.textbbox((0, 0), new_text, font=font)
        new_text_w = bbox_new[2] - bbox_new[0]
        new_text_x = x_img + (thumb - new_text_w) / 2
        draw.text((new_text_x, text_y), new_text, fill="black", font=font)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
img.save(OUT_PNG)
img.save(OUT_PDF)
print(f"[OK] Saved {OUT_PNG}")
print(f"[OK] Saved {OUT_PDF}")
