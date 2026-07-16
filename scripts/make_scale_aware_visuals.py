"""
Create qualitative comparison panels for scale-aware experiments.

Outputs:
  <out_root>/analysis/qualitative/*.png
  <out_root>/analysis/qualitative/README.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SPLIT_ORDER = ["D-High", "D-Medium", "D-Low", "D-Sub-50", "D-Sub-25"]
METHOD_ORDER = ["p2-only", "p3-only-long", "p2-p3-replay"]
METHOD_LABELS = {
    "p2-only": "P2 Only",
    "p3-only-long": "P3 Only Long",
    "p2-p3-replay": "P2+P3 Replay",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="kmc_lora/results/scale_aware_paper_v1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--images-per-method", type=int, default=3)
    return ap.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def find_images(gen_dir: Path, count: int) -> list[Path]:
    return sorted(gen_dir.glob("gen_*.png"))[:count]


def read_prompt(gen_dir: Path, image_index: int) -> str:
    meta_path = gen_dir / "generation_meta.json"
    if not meta_path.exists():
        return ""
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    images = payload.get("images", [])
    if 0 <= image_index < len(images):
        return images[image_index].get("prompt", "")
    return ""


def make_panel(images_by_method: dict[str, list[Path]], split_name: str, prompt_text: str, out_path: Path) -> None:
    thumb_w = 256
    thumb_h = 256
    pad = 18
    header_h = 92
    label_h = 30
    cols = max(len(v) for v in images_by_method.values()) if images_by_method else 0
    rows = len(images_by_method)

    canvas_w = pad + cols * (thumb_w + pad)
    canvas_h = header_h + rows * (label_h + thumb_h + pad) + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(28)
    text_font = load_font(18)
    small_font = load_font(15)

    draw.text((pad, 12), f"Qualitative Comparison: {split_name}", fill="black", font=title_font)
    if prompt_text:
        draw.text((pad, 50), f"Prompt: {prompt_text}", fill="black", font=small_font)

    y = header_h
    for method_name, image_paths in images_by_method.items():
        draw.text((pad, y), METHOD_LABELS.get(method_name, method_name), fill="black", font=text_font)
        x = pad
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB").resize((thumb_w, thumb_h))
            canvas.paste(image, (x, y + label_h))
            draw.rectangle([x, y + label_h, x + thumb_w, y + label_h + thumb_h], outline="#888888", width=1)
            x += thumb_w + pad
        y += label_h + thumb_h + pad

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    qual_dir = out_root / "analysis" / "qualitative"
    qual_dir.mkdir(parents=True, exist_ok=True)

    readme_lines = [
        "# Qualitative Comparisons",
        "",
        f"- Source root: `{out_root}`",
        f"- Seed used for panel generation: `{args.seed}`",
        f"- Images per method: `{args.images_per_method}`",
        "",
    ]

    for split_name in SPLIT_ORDER:
        images_by_method: dict[str, list[Path]] = {}
        prompt_text = ""
        for method_name in METHOD_ORDER:
            gen_dir = out_root / f"{method_name}_{split_name}_seed{args.seed}" / "generated"
            if not gen_dir.exists():
                continue
            image_paths = find_images(gen_dir, args.images_per_method)
            if not image_paths:
                continue
            images_by_method[method_name] = image_paths
            if not prompt_text:
                prompt_text = read_prompt(gen_dir, 0)

        if not images_by_method:
            continue

        out_path = qual_dir / f"{split_name}_seed{args.seed}_panel.png"
        make_panel(images_by_method, split_name, prompt_text, out_path)
        readme_lines.append(f"- [{out_path.name}](./{out_path.name})")

    (qual_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")
    print(f"[OK] Wrote qualitative panels to {qual_dir}")


if __name__ == "__main__":
    main()
