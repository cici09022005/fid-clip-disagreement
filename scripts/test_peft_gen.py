"""Quick test: verify PEFT loading produces different images."""
import torch, hashlib
from diffusers import StableDiffusionPipeline
from peft import PeftModel

# Load pipeline
print("[1/5] Loading SD pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16,
    safety_checker=None, requires_safety_checker=False)
pipe = pipe.to('cuda')

# Save base state
base_sd = {k: v.clone().cpu() for k, v in pipe.unet.state_dict().items()}
print(f"[2/5] Base UNet keys: {len(base_sd)}")

# Test 1: Random_D-High
print("[3/5] Loading Random_D-High adapter...")
peft_unet = PeftModel.from_pretrained(pipe.unet, 'kmc_lora/results/Random_D-High/final')
peft_unet.print_trainable_parameters()
merged = peft_unet.merge_and_unload()
pipe.unet = merged
g = torch.Generator(device='cuda').manual_seed(42)
img1 = pipe('a test image', num_inference_steps=5, generator=g).images[0]
img1.save('test_random_dh.png')

# Restore base
pipe.unet.load_state_dict(
    {k: v.to('cuda', dtype=torch.float16) for k, v in base_sd.items()},
    strict=True)

# Test 2: KMC_D-High
print("[4/5] Loading KMC_D-High adapter...")
peft_unet2 = PeftModel.from_pretrained(pipe.unet, 'kmc_lora/results/KMC_D-High/phase3/final')
merged2 = peft_unet2.merge_and_unload()
pipe.unet = merged2
g2 = torch.Generator(device='cuda').manual_seed(42)
img2 = pipe('a test image', num_inference_steps=5, generator=g2).images[0]
img2.save('test_kmc_dh.png')

# Compare
h1 = hashlib.md5(open('test_random_dh.png', 'rb').read()).hexdigest()
h2 = hashlib.md5(open('test_kmc_dh.png', 'rb').read()).hexdigest()
print(f"[5/5] Results:")
print(f"  Random_D-High hash: {h1}")
print(f"  KMC_D-High hash:    {h2}")
print(f"  Images different:   {h1 != h2}")
