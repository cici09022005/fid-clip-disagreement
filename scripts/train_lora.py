"""
LoRA fine-tuning for Stable Diffusion 1.5  (diffusers ≥0.30 + PEFT).
Saves per-step loss CSV, training config JSON, validation samples,
and LoRA adapter weights at every checkpoint — all needed for paper.
"""
import argparse, csv, gc, json, math, os, random, time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from accelerate import Accelerator
from diffusers import (AutoencoderKL, UNet2DConditionModel,
                       DDPMScheduler, StableDiffusionPipeline,
                       StableDiffusionXLPipeline)
from transformers import (AutoTokenizer, CLIPTextModel,
                          CLIPTextModelWithProjection)
from peft import LoraConfig, get_peft_model


# ──────────────────── Dataset ────────────────────
class ImagePromptDataset(Dataset):
    def __init__(self, image_list, prompt, size=512):
        self.paths = [p.strip() for p in
                      open(image_list, 'r', encoding='utf-8') if p.strip()]
        self.prompt = prompt
        self.tf = transforms.Compose([
            transforms.Resize(size,
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.tf(img), self.prompt


# ──────────────────── LoRA helpers ────────────────────
def add_lora(unet, rank=16, alpha=16, dropout=0.05):
    cfg = LoraConfig(
        r=rank, lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=dropout,
    )
    unet = get_peft_model(unet, cfg)
    unet.print_trainable_parameters()
    return unet


def save_lora(unet, output_dir, max_retries=5, retry_wait=1.5):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    u = unet.module if hasattr(unet, 'module') else unet

    last_err = None
    for i in range(max_retries):
        try:
            u.save_pretrained(output_dir, safe_serialization=True)
            return 'safetensors'
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            is_lock_error = ('os error 1224' in msg or
                             'i/o error' in msg or
                             'used by another process' in msg)
            if not is_lock_error or i == max_retries - 1:
                break
            print(f"[WARN] LoRA save lock conflict ({i+1}/{max_retries}), "
                  f"retrying in {retry_wait}s...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            time.sleep(retry_wait)

    print(f"[WARN] safetensors save failed after retries: {last_err}")
    print("[WARN] Falling back to adapter_model.bin for this checkpoint.")
    u.save_pretrained(output_dir, safe_serialization=False)
    return 'bin'


# ──────────────────── Logging helpers ────────────────────
def save_training_config(args, output_dir, num_images):
    """Save all hyperparameters as JSON for reproducibility."""
    cfg = {
        'timestamp': datetime.now().isoformat(),
        'base_model': args.base_model,
        'image_list': str(Path(args.image_list).resolve()),
        'num_images': num_images,
        'instance_prompt': args.instance_prompt,
        'resolution': args.resolution,
        'train_batch_size': args.train_batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'effective_batch_size': args.train_batch_size * args.gradient_accumulation_steps,
        'max_train_steps': args.max_train_steps,
        'save_steps': args.save_steps,
        'learning_rate': args.lr,
        'lora_rank': args.lora_rank,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'lora_path': args.lora_path,
        'seed': args.seed,
        'mixed_precision': args.mixed_precision,
        'validation_prompts': args.validation_prompts,
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / 'training_config.json', 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return cfg


class LossLogger:
    """Append-mode CSV logger for per-step training metrics."""
    def __init__(self, output_dir):
        self.path = Path(output_dir) / 'loss_log.csv'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, 'w', newline='', encoding='utf-8')
        self._writer = csv.writer(self._file)
        self._writer.writerow(['step', 'epoch', 'loss', 'lr',
                                'elapsed_sec', 'gpu_mem_MB'])
        self._file.flush()
        self._t0 = time.time()

    def log(self, step, epoch, loss, lr):
        elapsed = time.time() - self._t0
        gpu_mem = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        self._writer.writerow([step, epoch, f'{loss:.6f}', f'{lr:.2e}',
                               f'{elapsed:.1f}', f'{gpu_mem:.0f}'])
        if step % 10 == 0:
            self._file.flush()

    def close(self):
        self._file.close()


def save_training_summary(output_dir, total_steps, total_time, final_loss,
                          peak_gpu_mb):
    """One-line JSON with key training stats."""
    summary = {
        'completed_at': datetime.now().isoformat(),
        'total_steps': total_steps,
        'training_time_sec': round(total_time, 1),
        'training_time_min': round(total_time / 60, 2),
        'final_loss': round(final_loss, 6),
        'peak_gpu_memory_MB': round(peak_gpu_mb, 0),
    }
    with open(Path(output_dir) / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


def validate(pipe, prompts, out_dir, step, seed=42):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g = torch.Generator(device=pipe.device).manual_seed(seed)
    for i, prompt in enumerate(prompts):
        img = pipe(prompt, num_inference_steps=30, guidance_scale=7.5,
                   generator=g).images[0]
        img.save(out_dir / f"step_{step:06d}_p{i+1}.png")


def is_sdxl_model(base_model):
    # Check by model ID string first (handles HuggingFace hub IDs)
    base_lower = str(base_model).lower()
    if 'sdxl' in base_lower or 'stable-diffusion-xl' in base_lower:
        return True
    # Fall back to checking local directory structure
    base_path = Path(base_model)
    return (base_path / 'tokenizer_2').exists() and (base_path / 'text_encoder_2').exists()


def load_text_components(base_model, sdxl):
    tokenizer = AutoTokenizer.from_pretrained(base_model, subfolder='tokenizer', use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder='text_encoder')
    if not sdxl:
        return {
            'tokenizer': tokenizer,
            'text_encoder': text_encoder,
            'tokenizer_2': None,
            'text_encoder_2': None,
        }

    tokenizer_2 = AutoTokenizer.from_pretrained(base_model, subfolder='tokenizer_2', use_fast=False)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        base_model, subfolder='text_encoder_2')
    return {
        'tokenizer': tokenizer,
        'text_encoder': text_encoder,
        'tokenizer_2': tokenizer_2,
        'text_encoder_2': text_encoder_2,
    }


def encode_prompts(prompts, text_components, device, sdxl):
    prompts = list(prompts)
    if not sdxl:
        tokenizer = text_components['tokenizer']
        text_encoder = text_components['text_encoder']
        ids = tokenizer(
            prompts,
            padding='max_length',
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt',
        ).input_ids.to(device)
        return text_encoder(ids)[0], None

    prompt_embeds_list = []
    pooled_prompt_embeds = None
    for tokenizer, text_encoder in [
        (text_components['tokenizer'], text_components['text_encoder']),
        (text_components['tokenizer_2'], text_components['text_encoder_2']),
    ]:
        text_inputs = tokenizer(
            prompts,
            padding='max_length',
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt',
        )
        outputs = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=True)
        prompt_embeds_list.append(outputs.hidden_states[-2])
        pooled_prompt_embeds = outputs[0]

    prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
    return prompt_embeds, pooled_prompt_embeds


def build_sdxl_time_ids(batch_size, resolution, device, dtype):
    add_time_ids = torch.tensor(
        [[resolution, resolution, 0, 0, resolution, resolution]],
        device=device,
        dtype=dtype,
    )
    return add_time_ids.repeat(batch_size, 1)


def make_validation_pipeline(base_model, sdxl, lora_rank, lora_alpha, lora_dropout):
    pipe_cls = StableDiffusionXLPipeline if sdxl else StableDiffusionPipeline
    pipe = pipe_cls.from_pretrained(base_model, torch_dtype=torch.float16)
    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=lora_dropout,
    )
    pipe.unet = get_peft_model(pipe.unet, lora_cfg)
    return pipe


# ──────────────────── Main training loop ────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-model', required=True)
    ap.add_argument('--image-list', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--instance-prompt', required=True)
    ap.add_argument('--resolution', type=int, default=512)
    ap.add_argument('--train-batch-size', type=int, default=2)
    ap.add_argument('--gradient-accumulation-steps', type=int, default=4)
    ap.add_argument('--max-train-steps', type=int, default=1200)
    ap.add_argument('--save-steps', type=int, default=200)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lora-rank', type=int, default=16)
    ap.add_argument('--lora-alpha', type=int, default=16)
    ap.add_argument('--lora-dropout', type=float, default=0.05)
    ap.add_argument('--lora-path', default=None)
    ap.add_argument('--validation-prompts', nargs='*', default=[])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--mixed-precision', default='fp16')
    ap.add_argument('--max-gpu-memory-fraction', type=float, default=1.0,
                    help='Per-process GPU memory cap (0~1), useful when '
                         'sharing one GPU with other programs on Windows')
    args = ap.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if torch.cuda.is_available() and 0 < args.max_gpu_memory_fraction < 1.0:
        try:
            torch.cuda.set_per_process_memory_fraction(
                args.max_gpu_memory_fraction, device=0)
            print("[INFO] Set per-process GPU memory fraction to "
                  f"{args.max_gpu_memory_fraction:.2f}")
        except Exception as e:
            print(f"[WARN] Failed to set GPU memory fraction: {e}")

    # ── Load models ──
    sdxl = is_sdxl_model(args.base_model)
    text_components = load_text_components(args.base_model, sdxl)
    tokenizer = text_components['tokenizer']
    text_encoder = text_components['text_encoder']
    tokenizer_2 = text_components['tokenizer_2']
    text_encoder_2 = text_components['text_encoder_2']
    vae = AutoencoderKL.from_pretrained(args.base_model, subfolder='vae')
    unet = UNet2DConditionModel.from_pretrained(args.base_model, subfolder='unet')
    noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder='scheduler')

    # Enable gradient checkpointing (critical for SDXL on 24GB GPUs)
    if sdxl:
        unet.enable_gradient_checkpointing()
        if hasattr(text_components['text_encoder'], 'gradient_checkpointing_enable'):
            text_components['text_encoder'].gradient_checkpointing_enable()
        if text_components.get('text_encoder_2') and hasattr(text_components['text_encoder_2'], 'gradient_checkpointing_enable'):
            text_components['text_encoder_2'].gradient_checkpointing_enable()
        print("[INFO] Enabled gradient checkpointing for SDXL")

    # ── LoRA ──
    unet = add_lora(unet, rank=args.lora_rank, alpha=args.lora_alpha,
                    dropout=args.lora_dropout)
    if args.lora_path:
        from safetensors.torch import load_file
        wf = Path(args.lora_path) / "adapter_model.safetensors"
        if not wf.exists():
            wf = Path(args.lora_path) / "adapter_model.bin"
        if wf.exists():
            sd = None
            if str(wf).endswith('.safetensors'):
                try:
                    sd = load_file(str(wf))
                except Exception as e:
                    bin_wf = Path(args.lora_path) / "adapter_model.bin"
                    if bin_wf.exists():
                        print(f"[WARN] Failed to load safetensors ({e}), "
                              f"fallback to {bin_wf.name}")
                        sd = torch.load(str(bin_wf), map_location='cpu')
                    else:
                        raise
            else:
                sd = torch.load(str(wf), map_location='cpu')
            # Handle PEFT version differences: adapt key format if needed
            model_keys = set(dict(unet.named_parameters()).keys())
            needs_default = any('.default.' in k for k in model_keys)
            has_default = any('.default.' in k for k in sd.keys())
            if needs_default and not has_default:
                sd = {k.replace('.lora_A.weight', '.lora_A.default.weight')
                       .replace('.lora_B.weight', '.lora_B.default.weight'): v
                      for k, v in sd.items()}
            elif not needs_default and has_default:
                sd = {k.replace('.lora_A.default.weight', '.lora_A.weight')
                       .replace('.lora_B.default.weight', '.lora_B.weight'): v
                      for k, v in sd.items()}
            result = unet.load_state_dict(sd, strict=False)
            loaded = len(sd) - len(result.unexpected_keys)
            del sd
            gc.collect()
            print(f"[INFO] Loaded LoRA weights from {wf} ({loaded}/{256} keys matched)")

    text_encoder.requires_grad_(False)
    if text_encoder_2 is not None:
        text_encoder_2.requires_grad_(False)
    vae.requires_grad_(False)

    dataset = ImagePromptDataset(args.image_list, args.instance_prompt,
                                 size=args.resolution)
    dl_workers = 0 if os.name == 'nt' else 2
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size,
                            shuffle=True, num_workers=dl_workers,
                            pin_memory=True)

    # ── Save config ──
    if accelerator.is_local_main_process:
        save_training_config(args, args.output_dir, len(dataset))
        loss_logger = LossLogger(args.output_dir)

    # ── Optimizer ──
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=1e-2)

    prepare_items = [unet, optimizer, dataloader, text_encoder, vae]
    if text_encoder_2 is not None:
        prepare_items.append(text_encoder_2)
    prepared = accelerator.prepare(*prepare_items)
    if text_encoder_2 is not None:
        unet, optimizer, dataloader, text_encoder, vae, text_encoder_2 = prepared
    else:
        unet, optimizer, dataloader, text_encoder, vae = prepared

    global_step = 0
    epoch = 0
    last_loss = 0.0
    t0 = time.time()
    unet.train()

    progress = tqdm(total=args.max_train_steps,
                    disable=not accelerator.is_local_main_process,
                    desc="Training")

    for epoch in range(1_000_000):
        for images, prompt in dataloader:
            with accelerator.accumulate(unet):
                scaling_factor = getattr(vae.config, 'scaling_factor', 0.18215)
                latents = vae.encode(images).latent_dist.sample() * scaling_factor

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise,
                                                          timesteps)

                prompts = list(prompt) if isinstance(prompt, (list, tuple)) \
                    else [prompt] * bsz
                runtime_text_components = {
                    'tokenizer': tokenizer,
                    'text_encoder': text_encoder,
                    'tokenizer_2': tokenizer_2,
                    'text_encoder_2': text_encoder_2,
                }
                prompt_embeds, pooled_prompt_embeds = encode_prompts(
                    prompts, runtime_text_components, latents.device, sdxl)

                if sdxl:
                    time_ids = build_sdxl_time_ids(
                        batch_size=bsz,
                        resolution=args.resolution,
                        device=latents.device,
                        dtype=prompt_embeds.dtype,
                    )
                    pred = unet(
                        noisy_latents,
                        timesteps,
                        prompt_embeds,
                        added_cond_kwargs={
                            'text_embeds': pooled_prompt_embeds,
                            'time_ids': time_ids,
                        },
                    ).sample
                else:
                    pred = unet(noisy_latents, timesteps, prompt_embeds).sample
                loss = nn.functional.mse_loss(pred, noise, reduction='mean')

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            last_loss = loss.item()
            global_step += 1

            if accelerator.is_local_main_process:
                progress.update(1)
                progress.set_postfix(loss=f"{last_loss:.4f}", epoch=epoch)
                loss_logger.log(global_step, epoch, last_loss, args.lr)

            # ── Checkpoint ──
            if global_step % args.save_steps == 0 and \
               accelerator.is_local_main_process:
                ckpt = Path(args.output_dir) / f"checkpoint-{global_step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                saved_fmt = save_lora(accelerator.unwrap_model(unet), ckpt)
                print(f"\n[CKPT] Saved checkpoint-{global_step} ({saved_fmt})")

                # Validation (guarded against OOM)
                if args.validation_prompts:
                    try:
                        pipe = make_validation_pipeline(
                            base_model=args.base_model,
                            sdxl=sdxl,
                            lora_rank=args.lora_rank,
                            lora_alpha=args.lora_alpha,
                            lora_dropout=args.lora_dropout,
                        )

                        from safetensors.torch import load_file
                        val_wf = ckpt / "adapter_model.safetensors"
                        if not val_wf.exists():
                            val_wf = ckpt / "adapter_model.bin"
                        if str(val_wf).endswith('.safetensors'):
                            val_sd = load_file(str(val_wf))
                        else:
                            val_sd = torch.load(str(val_wf), map_location='cpu')

                        model_keys = set(dict(pipe.unet.named_parameters()).keys())
                        needs_default = any('.default.' in k for k in model_keys)
                        has_default = any('.default.' in k for k in val_sd.keys())
                        if needs_default and not has_default:
                            val_sd = {k.replace('.lora_A.weight', '.lora_A.default.weight')
                                      .replace('.lora_B.weight', '.lora_B.default.weight'): v
                                      for k, v in val_sd.items()}
                        elif not needs_default and has_default:
                            val_sd = {k.replace('.lora_A.default.weight', '.lora_A.weight')
                                      .replace('.lora_B.default.weight', '.lora_B.weight'): v
                                      for k, v in val_sd.items()}

                        pipe.unet.load_state_dict(val_sd, strict=False)
                        pipe = pipe.to(accelerator.device)
                        with torch.no_grad():
                            validate(pipe, args.validation_prompts,
                                     ckpt / 'samples', global_step)
                        del pipe
                        torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"[WARN] Validation failed at step "
                              f"{global_step}: {e}")

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    progress.close()
    total_time = time.time() - t0
    peak_gpu = (torch.cuda.max_memory_allocated() / 1e6
                if torch.cuda.is_available() else 0)

    if accelerator.is_local_main_process:
        # ── Save final adapter ──
        final_dir = Path(args.output_dir) / 'final'
        final_fmt = save_lora(accelerator.unwrap_model(unet), final_dir)
        print(f"[FINAL] Saved LoRA adapter ({final_fmt}) to {final_dir}")

        # ── Save training summary ──
        loss_logger.close()
        save_training_summary(args.output_dir, global_step, total_time,
                              last_loss, peak_gpu)
        print(f"\n[DONE] {global_step} steps in {total_time/60:.1f} min, "
              f"final loss={last_loss:.4f}, peak GPU={peak_gpu:.0f} MB")


if __name__ == '__main__':
    main()
