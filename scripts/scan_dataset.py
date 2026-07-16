import argparse
import csv
import os
from pathlib import Path
from PIL import Image, ImageSequence

SUPPORTED = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff', '.gif'}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED


def safe_open_image(path: Path):
    try:
        img = Image.open(path)
        return img
    except Exception:
        return None


def convert_gif_first_frame(src: Path, dst: Path):
    img = Image.open(src)
    frame = ImageSequence.Iterator(img).__next__()
    frame = frame.convert('RGB')
    frame.save(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--convert-gif', action='store_true')
    ap.add_argument('--converted-dir', default=None)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_path = Path(args.out)
    converted_dir = Path(args.converted_dir) if args.converted_dir else None
    if args.convert_gif and converted_dir:
        converted_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in dataset_dir.rglob('*'):
        if not path.is_file():
            continue
        if not is_image(path):
            continue

        final_path = path
        if path.suffix.lower() == '.gif' and args.convert_gif and converted_dir:
            conv_name = path.stem + '_gifframe.png'
            conv_path = converted_dir / conv_name
            if not conv_path.exists():
                try:
                    convert_gif_first_frame(path, conv_path)
                except Exception:
                    continue
            final_path = conv_path

        img = safe_open_image(final_path)
        if img is None:
            continue

        width, height = img.size
        mode = img.mode
        rows.append({
            'path': str(final_path),
            'orig_path': str(path),
            'width': width,
            'height': height,
            'mode': mode,
            'format': final_path.suffix.lower().lstrip('.'),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print(f"Images: {len(rows)}")


if __name__ == '__main__':
    main()
