"""
Regenerate samples with CORRECT PEFT LoRA loading + compute FID/CLIP.
The old generate_samples.py used pipe.load_lora_weights() which silently
failed with PEFT adapter format → all images were identical.

This script:
  1. Loads SD pipeline once
  2. For each experiment, loads PEFT adapter → merges → generates → restores base
  3. Computes FID + CLIP Score per experiment
  4. Saves eval_metrics.json + eval_all.csv
"""
import argparse, json, csv, gc, os, sys, hashlib
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from peft import PeftModel
from diffusers import StableDiffusionPipeline


# ────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────
EXPERIMENTS = [
    'Random_D-High', 'KMC_D-High',
    'Random_D-Medium', 'KMC_D-Medium',
    'Random_D-Low', 'KMC_D-Low',
    'Random_D-Sub-50', 'KMC_D-Sub-50',
    'Random_D-Sub-25', 'KMC_D-Sub-25',
    'Ablation_NoPhase1', 'Ablation_NoPhase2', 'Ablation_NoPhase3',
    'Ablation_Phase1Only', 'Ablation_Phase3Only',
    'Quality_Filter', 'Anti_Curriculum',
]

# Which final phase to use for each experiment
KMC_SPLITS = ['KMC_D-High', 'KMC_D-Medium', 'KMC_D-Low',
              'KMC_D-Sub-50', 'KMC_D-Sub-25']
ABLATION_PHASE = {
    'Ablation_NoPhase1': 'phase3',
    'Ablation_NoPhase2': 'phase3',
    'Ablation_NoPhase3': 'phase2',
    'Ablation_Phase1Only': 'phase1',
    'Ablation_Phase3Only': 'phase3',
}

DATASET_CONFIGS = {
    'anime_student': {
        'results_dir': 'kmc_lora/results',
        'real_list': 'kmc_lora/artifacts/splits/D-High.txt',
        'prompts': [
            "a hand-drawn anime storyboard of a brave knight",
            "a hand-drawn anime storyboard of a city at night",
            "a hand-drawn anime storyboard of a forest spirit",
        ],
    },
    'wikiart_mixed': {
        'results_dir': 'kmc_lora/results/wikiart_mixed',
        'real_list': 'kmc_lora/artifacts/wikiart_mixed/splits/D-High.txt',
        'prompts': [
            "a painting of a sunset over mountains",
            "a painting of a woman reading a book",
            "a painting of a bustling city street",
            "a painting of flowers in a vase",
        ],
    },
    'dreambooth_mixed': {
        'results_dir': 'kmc_lora/results/dreambooth_mixed',
        'real_list': 'kmc_lora/artifacts/dreambooth_mixed/splits/D-High.txt',
        'prompts': [
            "a photo of a sks dog sitting on a beach",
            "a photo of a sks cat on a sofa",
            "a photo of a sks backpack in a forest",
        ],
    },
    'dreambooth_single': {
        'results_dir': 'kmc_lora/results/dreambooth_single',
        'real_list': 'kmc_lora/artifacts/dreambooth_single/splits/D-High.txt',
        'prompts': [
            "a photo of a sks dog on the beach",
            "a photo of a sks dog in a garden",
            "a photo of a sks dog wearing a hat",
        ],
    },
}


# ────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────
def get_lora_path(results_dir, exp_name):
    rd = Path(results_dir)
    if exp_name in KMC_SPLITS:
        p = rd / exp_name / 'phase3' / 'final'
    elif exp_name in ABLATION_PHASE:
        phase = ABLATION_PHASE[exp_name]
        p = rd / exp_name / phase / 'final'
    else:
        p = rd / exp_name / 'final'
    if p.exists() and (p / 'adapter_config.json').exists():
        return str(p)
    return None


def load_real_images(real_list_path, max_images=500):
    """Load real images as uint8 tensors [N, 3, 299, 299] for FID."""
    paths = [l.strip() for l in
             open(real_list_path, 'r', encoding='utf-8') if l.strip()]
    images = []
    for p in paths[:max_images]:
        try:
            img = Image.open(p).convert('RGB').resize((299, 299))
            arr = np.array(img, dtype=np.uint8)
            images.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
        except Exception as e:
            pass  # skip unreadable
    if not images:
        return torch.empty(0, 3, 299, 299, dtype=torch.uint8)
    return torch.stack(images)


def compute_fid(real_tensors, gen_paths, max_gen=500):
    """Compute FID between real image tensors and generated image paths."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    gen_imgs = []
    for p in gen_paths[:max_gen]:
        img = Image.open(p).convert('RGB').resize((299, 299))
        arr = np.array(img, dtype=np.uint8)
        gen_imgs.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
    if not gen_imgs:
        return float('nan')
    gen_t = torch.stack(gen_imgs)

    fid = FrechetInceptionDistance(feature=2048)
    fid.update(real_tensors, real=True)
    fid.update(gen_t, real=False)
    return fid.compute().item()


def compute_clip_score(gen_paths, prompts, max_images=500, device='cpu'):
    """Compute mean CLIP score."""
    try:
        import open_clip
        from torchvision import transforms
    except ImportError:
        return None

    model, _, _ = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='laion2b_s32b_b82k')
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    model = model.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    paths = gen_paths[:max_images]
    with torch.no_grad():
        text_tokens = tokenizer(prompts).to(device)
        text_feat = model.encode_text(text_tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat.mean(dim=0, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        scores = []
        for p in paths:
            img = Image.open(p).convert('RGB')
            img_t = tf(img).unsqueeze(0).to(device)
            img_feat = model.encode_image(img_t)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            scores.append((img_feat @ text_feat.T).item())

    del model
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    return float(np.mean(scores)) if scores else None


# ────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='*',
                    default=list(DATASET_CONFIGS.keys()))
    ap.add_argument('--base-model', default='runwayml/stable-diffusion-v1-5')
    ap.add_argument('--num-per-prompt', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--guidance-scale', type=float, default=7.5)
    ap.add_argument('--skip-gen', action='store_true',
                    help='Skip generation, only run evaluation')
    ap.add_argument('--force-regen', action='store_true',
                    help='Delete old generated images and regenerate')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"{'='*60}")
    print(f"KMC-LoRA: Regenerate + Evaluate (PEFT-aware)")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Device: {device}  |  Datasets: {args.datasets}")
    print(f"{'='*60}\n")

    # ── Load pipeline once ──
    if not args.skip_gen:
        print("[INIT] Loading SD pipeline...")
        pipe = StableDiffusionPipeline.from_pretrained(
            args.base_model, torch_dtype=torch.float16,
            safety_checker=None, requires_safety_checker=False)
        pipe = pipe.to(device)
        # Save a copy of base UNet state dict for restoration
        base_sd = {k: v.clone().cpu() for k, v in pipe.unet.state_dict().items()}
        print(f"[INIT] Pipeline loaded. Base UNet keys: {len(base_sd)}\n")

    for ds_key in args.datasets:
        cfg = DATASET_CONFIGS[ds_key]
        rd = Path(cfg['results_dir'])
        prompts = cfg['prompts']

        print(f"\n{'='*50}")
        print(f"Dataset: {ds_key}")
        print(f"{'='*50}")

        # ── Phase 1: Generate images ──
        if not args.skip_gen:
            for exp in EXPERIMENTS:
                lora_path = get_lora_path(str(rd), exp)
                if not lora_path:
                    print(f"  [SKIP-GEN] {exp}: no LoRA found")
                    continue

                gen_dir = rd / exp / 'generated'

                # Skip if already has valid (non-duplicate) images
                if not args.force_regen and gen_dir.exists():
                    pngs = list(gen_dir.glob('*.png'))
                    if len(pngs) >= args.num_per_prompt * len(prompts):
                        # Quick check: are they actually from this LoRA?
                        meta_f = gen_dir / 'generation_meta.json'
                        if meta_f.exists():
                            meta = json.load(open(meta_f, 'r', encoding='utf-8'))
                            if meta.get('peft_loaded', False):
                                print(f"  [SKIP-GEN] {exp}: already generated (PEFT-verified)")
                                continue
                        # Old generation (pre-PEFT fix) → must regenerate
                        print(f"  [REGEN] {exp}: old images detected, regenerating...")
                    # else: not enough images

                gen_dir.mkdir(parents=True, exist_ok=True)
                # Remove old images
                for old in gen_dir.glob('*.png'):
                    old.unlink()
                for old in gen_dir.glob('*.json'):
                    old.unlink()

                # Load PEFT adapter
                print(f"  [GEN] {exp}: loading PEFT adapter from {lora_path}")
                try:
                    peft_unet = PeftModel.from_pretrained(pipe.unet, lora_path)
                    peft_unet = peft_unet.to(device)
                    # Merge LoRA into base weights for inference
                    merged_unet = peft_unet.merge_and_unload()
                    pipe.unet = merged_unet
                except Exception as e:
                    print(f"  [FAIL-GEN] {exp}: PEFT load error: {e}")
                    continue

                # Generate
                g = torch.Generator(device=device).manual_seed(args.seed)
                generated = []
                idx = 0
                for p_text in prompts:
                    for i in range(args.num_per_prompt):
                        image = pipe(
                            p_text,
                            num_inference_steps=args.steps,
                            guidance_scale=args.guidance_scale,
                            generator=g
                        ).images[0]
                        fname = f"gen_{idx:04d}.png"
                        image.save(gen_dir / fname)
                        generated.append({
                            'index': idx, 'file': fname, 'prompt': p_text,
                        })
                        idx += 1

                # Save metadata
                meta = {
                    'timestamp': datetime.now().isoformat(),
                    'base_model': args.base_model,
                    'lora_path': str(lora_path),
                    'peft_loaded': True,
                    'total_images': idx,
                    'prompts': prompts,
                    'num_per_prompt': args.num_per_prompt,
                    'seed': args.seed,
                    'inference_steps': args.steps,
                    'guidance_scale': args.guidance_scale,
                    'images': generated,
                }
                with open(gen_dir / 'generation_meta.json', 'w', encoding='utf-8') as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)

                print(f"       → generated {idx} images")

                # ── Restore base UNet ──
                # Remove PEFT config metadata to avoid "Already found peft_config" warning
                for attr in ['peft_config', '_hf_peft_config_loaded']:
                    try:
                        delattr(pipe.unet, attr)
                    except (AttributeError, Exception):
                        pass
                pipe.unet.load_state_dict(
                    {k: v.to(device, dtype=torch.float16) for k, v in base_sd.items()},
                    strict=True)
                gc.collect()
                torch.cuda.empty_cache()

        # ── Verify images differ ──
        print(f"\n  [CHECK] Verifying generated images are unique...")
        hashes = {}
        for exp in EXPERIMENTS:
            gd = rd / exp / 'generated'
            pngs = sorted(gd.glob('*.png'))
            if pngs:
                h = hashlib.md5(open(pngs[0], 'rb').read()).hexdigest()
                hashes[exp] = h
        unique_h = set(hashes.values())
        if len(unique_h) <= 1 and len(hashes) > 1:
            print(f"  [WARN] All images identical! Generation may have failed.")
        else:
            print(f"  [OK] {len(unique_h)} unique image sets out of {len(hashes)} experiments")

        # ── Phase 2: Evaluate (FID + CLIP) ──
        print(f"\n  [EVAL] Loading real images for FID...")
        real_list = cfg['real_list']
        real_tensors = load_real_images(real_list)
        print(f"         Real images loaded: {real_tensors.shape[0]}")

        # Remove old eval CSV
        eval_csv = rd / 'eval_all.csv'
        if eval_csv.exists():
            eval_csv.unlink()

        clip_model_loaded = False  # defer CLIP model loading

        for exp in EXPERIMENTS:
            gen_dir = rd / exp / 'generated'
            gen_pngs = sorted([str(p) for p in gen_dir.glob('*.png')])
            if len(gen_pngs) < 5:
                print(f"  [SKIP-EVAL] {exp}: insufficient images ({len(gen_pngs)})")
                continue

            eval_json = gen_dir / 'eval_metrics.json'
            # Skip if already evaluated with PEFT-verified images
            if eval_json.exists():
                d = json.load(open(eval_json))
                if d.get('peft_verified', False):
                    print(f"  [SKIP-EVAL] {exp}: already evaluated (PEFT-verified)")
                    # Still append to CSV
                    _append_csv(eval_csv, d)
                    continue

            print(f"  [EVAL] {exp} ({len(gen_pngs)} images)...", end='', flush=True)

            # FID
            fid_score = compute_fid(real_tensors, gen_pngs)
            print(f"  FID={fid_score:.2f}", end='', flush=True)

            # CLIP
            clip_score = compute_clip_score(
                gen_pngs, prompts, device=device)
            if clip_score is not None:
                print(f"  CLIP={clip_score:.4f}", end='')
            print("  OK")

            results = {
                'experiment': exp,
                'fid': round(fid_score, 4),
                'clip_score': round(clip_score, 4) if clip_score else '',
                'num_real': real_tensors.shape[0],
                'num_gen': len(gen_pngs),
                'gen_dir': str(gen_dir),
                'peft_verified': True,
            }

            # Save per-experiment JSON
            with open(eval_json, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)

            # Append to master CSV
            _append_csv(eval_csv, results)

            # Free memory
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()

        print(f"\n  Dataset {ds_key} complete!")

    # ── Free pipeline ──
    if not args.skip_gen:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"ALL DONE: {datetime.now().isoformat()}")
    print(f"{'='*60}")


def _append_csv(csv_path, row_dict):
    """Append a dict row to CSV, creating header if needed."""
    fields = ['experiment', 'fid', 'clip_score', 'num_real', 'num_gen', 'gen_dir']
    exists = csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        if not exists:
            w.writeheader()
        w.writerow(row_dict)


if __name__ == '__main__':
    main()
