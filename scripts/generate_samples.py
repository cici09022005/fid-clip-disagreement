"""
Generate sample images from a LoRA-finetuned Stable Diffusion model.
Saves generation metadata alongside images.

Supports PEFT-format LoRA weights (base_model.model.xxx.lora_A/B.weight)
saved by train_lora.py, loaded via direct state_dict injection.
"""
import argparse, gc, json
from datetime import datetime
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, UNet2DConditionModel
from peft import LoraConfig, get_peft_model


def is_sdxl_model(base_model):
    base_lower = str(base_model).lower()
    if 'sdxl' in base_lower or 'stable-diffusion-xl' in base_lower:
        return True
    base_path = Path(base_model)
    return (base_path / 'tokenizer_2').exists() and (base_path / 'text_encoder_2').exists()


def load_pipeline_with_peft_lora(base_model, lora_path):
    """Load SD pipeline and inject PEFT LoRA weights into UNet."""
    pipe_cls = StableDiffusionXLPipeline if is_sdxl_model(base_model) else StableDiffusionPipeline
    pipe = pipe_cls.from_pretrained(base_model, torch_dtype=torch.float16)

    adapter_dir = Path(lora_path)
    adapter_cfg_path = adapter_dir / "adapter_config.json"
    if not adapter_cfg_path.exists():
        raise FileNotFoundError(f"Missing adapter config: {adapter_cfg_path}")
    adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding='utf-8'))

    # Recreate the LoRA wrapper from the saved adapter config instead of
    # assuming a fixed rank across datasets.
    lora_cfg = LoraConfig(
        r=int(adapter_cfg['r']),
        lora_alpha=int(adapter_cfg.get('lora_alpha', adapter_cfg['r'])),
        init_lora_weights=adapter_cfg.get('init_lora_weights', 'gaussian'),
        target_modules=adapter_cfg.get('target_modules', ["to_k", "to_q", "to_v", "to_out.0"]),
        lora_dropout=float(adapter_cfg.get('lora_dropout', 0.05)),
    )
    pipe.unet = get_peft_model(pipe.unet, lora_cfg)

    # Load saved weights
    from safetensors.torch import load_file
    wf = adapter_dir / "adapter_model.safetensors"
    if not wf.exists():
        wf = adapter_dir / "adapter_model.bin"
    if wf.exists():
        if str(wf).endswith('.safetensors'):
            sd = load_file(str(wf))
        else:
            sd = torch.load(str(wf), map_location='cpu', weights_only=True)

        # Handle PEFT version differences: saved keys may use
        # "lora_A.weight" while current model expects "lora_A.default.weight"
        model_keys = set(dict(pipe.unet.named_parameters()).keys())
        needs_default = any('.default.' in k for k in model_keys)
        has_default = any('.default.' in k for k in sd.keys())

        if needs_default and not has_default:
            # Insert ".default." before ".weight" in lora keys
            sd = {k.replace('.lora_A.weight', '.lora_A.default.weight')
                   .replace('.lora_B.weight', '.lora_B.default.weight'): v
                  for k, v in sd.items()}
        elif not needs_default and has_default:
            # Remove ".default." from lora keys
            sd = {k.replace('.lora_A.default.weight', '.lora_A.weight')
                   .replace('.lora_B.default.weight', '.lora_B.weight'): v
                  for k, v in sd.items()}

        result = pipe.unet.load_state_dict(sd, strict=False)
        loaded = len(sd) - len(result.unexpected_keys)
        print(f"[INFO] Loaded PEFT LoRA from {wf} ({loaded}/{len(sd)} keys matched)")
        del sd
        gc.collect()
    else:
        raise FileNotFoundError(f"No adapter file found in {lora_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe.safety_checker = None  # Disable NSFW filter for research evaluation
    pipe = pipe.to(device)
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-model', required=True)
    ap.add_argument('--lora-path', required=True)
    ap.add_argument('--prompts', nargs='+', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--num-per-prompt', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--guidance-scale', type=float, default=7.5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline_with_peft_lora(args.base_model, args.lora_path)

    g = torch.Generator(device=pipe.device).manual_seed(args.seed)

    generated = []
    idx = 0
    for p in args.prompts:
        for i in range(args.num_per_prompt):
            image = pipe(p, num_inference_steps=args.steps,
                         guidance_scale=args.guidance_scale,
                         generator=g).images[0]
            fname = f"gen_{idx:04d}.png"
            image.save(out_dir / fname)
            generated.append({
                'index': idx,
                'file': fname,
                'prompt': p,
                'seed': args.seed,
                'steps': args.steps,
                'guidance_scale': args.guidance_scale,
            })
            idx += 1

    # Save generation metadata
    meta = {
        'timestamp': datetime.now().isoformat(),
        'base_model': args.base_model,
        'lora_path': str(args.lora_path),
        'total_images': idx,
        'prompts': args.prompts,
        'num_per_prompt': args.num_per_prompt,
        'seed': args.seed,
        'inference_steps': args.steps,
        'guidance_scale': args.guidance_scale,
        'images': generated,
    }
    with open(out_dir / 'generation_meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Generated {idx} images → {out_dir}")


if __name__ == '__main__':
    main()
