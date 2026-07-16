"""
Evaluate generated images: FID, CLIP-Score.
Outputs results to stdout and optionally to a CSV file.
"""
import argparse, csv, json, os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("DIFFUSERS_OFFLINE", "1")


def load_images(paths, max_images=None):
    images = []
    for p in paths[:max_images]:
        img = Image.open(p).convert('RGB').resize((299, 299))
        arr = np.array(img, dtype=np.uint8)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        images.append(tensor)
    if not images:
        return torch.empty(0, 3, 299, 299, dtype=torch.uint8)
    return torch.stack(images, dim=0)


def compute_clip_score(gen_paths, prompts, max_images=None, device='cpu'):
    """Compute mean CLIP score between generated images and prompts."""
    try:
        import open_clip
        from torchvision import transforms
    except ImportError:
        print("[WARN] open_clip not installed, skipping CLIP score")
        return None

    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='laion2b_s32b_b82k')
        tokenizer = open_clip.get_tokenizer('ViT-L-14')
        model = model.to(device).eval()
    except Exception as exc:
        print(f"[WARN] failed to initialize CLIP model, skipping CLIP score: {exc}")
        return None

    tf = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    paths = gen_paths[:max_images] if max_images else gen_paths
    scores = []
    with torch.no_grad():
        text_tokens = tokenizer(prompts).to(device)
        text_feat = model.encode_text(text_tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        # average text features across prompts
        text_feat = text_feat.mean(dim=0, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        for p in paths:
            img = Image.open(p).convert('RGB')
            img_t = tf(img).unsqueeze(0).to(device)
            img_feat = model.encode_image(img_t)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            score = (img_feat @ text_feat.T).item()
            scores.append(score)

    return float(np.mean(scores)) if scores else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--real-list', required=True,
                    help='txt file with real image paths')
    ap.add_argument('--gen-dir', required=True,
                    help='directory with generated .png images')
    ap.add_argument('--prompts', nargs='*', default=[],
                    help='prompts for CLIP score')
    ap.add_argument('--max-real', type=int, default=500)
    ap.add_argument('--max-gen', type=int, default=500)
    ap.add_argument('--out-csv', default=None,
                    help='append results to this CSV')
    ap.add_argument('--experiment-name', default='',
                    help='experiment label for CSV')
    args = ap.parse_args()

    real_paths = [p.strip() for p in
                  open(args.real_list, 'r', encoding='utf-8') if p.strip()]
    gen_paths = sorted([str(p) for p in Path(args.gen_dir).glob('*.png')])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── FID ──
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    # Process in batches to avoid OOM
    BATCH = 32
    real_subset = real_paths[:args.max_real]
    gen_subset = gen_paths[:args.max_gen]
    for i in range(0, len(real_subset), BATCH):
        batch = load_images(real_subset[i:i+BATCH])
        fid_metric.update(batch.to(device), real=True)
    for i in range(0, len(gen_subset), BATCH):
        batch = load_images(gen_subset[i:i+BATCH])
        fid_metric.update(batch.to(device), real=False)
    fid_score = fid_metric.compute().item()
    print(f"FID: {fid_score:.4f}")

    # ── CLIP Score ──
    clip_score = None
    if args.prompts:
        clip_score = compute_clip_score(gen_paths, args.prompts,
                                        max_images=args.max_gen,
                                        device=device)
        if clip_score is not None:
            print(f"CLIP Score: {clip_score:.4f}")

    results = {
        'experiment': args.experiment_name,
        'fid': round(fid_score, 4),
        'clip_score': round(clip_score, 4) if clip_score is not None else '',
        'num_real': min(len(real_paths), args.max_real),
        'num_gen': min(len(gen_paths), args.max_gen),
        'gen_dir': args.gen_dir,
    }

    # ── Save to JSON in gen_dir ──
    with open(Path(args.gen_dir) / 'eval_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    # ── Append to master CSV ──
    if args.out_csv:
        csv_path = Path(args.out_csv)
        exists = csv_path.exists()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(results.keys()))
            if not exists:
                w.writeheader()
            w.writerow(results)

    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
