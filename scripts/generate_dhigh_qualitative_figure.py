from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
ROBUSTNESS_DIR = ROOT / "kmc_lora" / "results_structured" / "robustness"

BLOCKS = [
    ("anime_student", "Anime-Student", "Brave knight", 0),
    ("wikiart_mixed", "WikiArt-Mixed", "Woman reading", 50),
    ("dreambooth_mixed", "DreamBooth-Mixed", "Cat on sofa", 50),
    ("dreambooth_single", "DreamBooth-Single", "Dog on beach", 0),
]

METHODS = [
    ("Random_D-High", "Random_D-High"),
    ("KMC_D-High", "KMC_D-High"),
    ("Anti_Curriculum", "Anti_Curriculum"),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed from the repeated-run D-High sweep used to draw representative samples.",
    )
    ap.add_argument(
        "--out",
        default=str(ROOT / "ACCESS_latex_template_20240429" / "figures" / "fig6_qualitative_style.png"),
    )
    return ap.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def get_image_path(dataset: str, method_tag: str, seed: int, image_index: int) -> Path:
    return (
        ROBUSTNESS_DIR
        / dataset
        / "multi_seed"
        / f"{method_tag}_seed{seed}"
        / "generated"
        / f"gen_{image_index:04d}.png"
    )


def open_and_resize(path: Path, size: int) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"missing qualitative sample: {path}")
    resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    return Image.open(path).convert("RGB").resize((size, size), resample)


def make_figure(seed: int, out_path: Path) -> None:
    outer_pad = 28
    block_gap_x = 44
    block_gap_y = 42
    image_gap = 16
    thumb = 170
    dataset_title_h = 28
    prompt_h = 22
    method_label_h = 24

    block_w = 3 * thumb + 2 * image_gap
    block_h = dataset_title_h + prompt_h + method_label_h + thumb
    canvas_w = outer_pad * 2 + block_w * 2 + block_gap_x
    canvas_h = outer_pad * 2 + block_h * 2 + block_gap_y + 24

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(28, bold=True)
    subtitle_font = load_font(21, bold=False)
    method_font = load_font(18, bold=True)

    for row in range(2):
        for col in range(2):
            block = BLOCKS[row * 2 + col]
            dataset, dataset_label, prompt_label, image_index = block
            x0 = outer_pad + col * (block_w + block_gap_x)
            y0 = outer_pad + row * (block_h + block_gap_y)

            draw.text((x0, y0), dataset_label, fill="black", font=title_font)
            draw.text((x0, y0 + dataset_title_h), prompt_label, fill="#555555", font=subtitle_font)

            y_img = y0 + dataset_title_h + prompt_h + method_label_h
            for method_idx, (method_tag, method_label) in enumerate(METHODS):
                x_img = x0 + method_idx * (thumb + image_gap)
                w, _ = text_size(draw, method_label, method_font)
                draw.text((x_img + (thumb - w) / 2, y0 + dataset_title_h + prompt_h), method_label, fill="black", font=method_font)
                image_path = get_image_path(dataset, method_tag, seed, image_index)
                image = open_and_resize(image_path, thumb)
                canvas.paste(image, (x_img, y_img))
                draw.rectangle((x_img, y_img, x_img + thumb, y_img + thumb), outline="#9a9a9a", width=1)

    # Subtle separators between dataset blocks.
    draw.line(
        (
            canvas_w // 2,
            outer_pad + 8,
            canvas_w // 2,
            canvas_h - outer_pad - 8,
        ),
        fill="#d6d6d6",
        width=2,
    )
    draw.line(
        (
            outer_pad,
            canvas_h // 2,
            canvas_w - outer_pad,
            canvas_h // 2,
        ),
        fill="#d6d6d6",
        width=2,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


if __name__ == "__main__":
    args = parse_args()
    make_figure(seed=args.seed, out_path=Path(args.out))
