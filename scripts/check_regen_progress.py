"""Check progress of regen_and_eval."""
import json
from pathlib import Path

exps = ['Random_D-High','KMC_D-High','Random_D-Medium','KMC_D-Medium',
        'Random_D-Low','KMC_D-Low','Random_D-Sub-50','KMC_D-Sub-50',
        'Random_D-Sub-25','KMC_D-Sub-25',
        'Ablation_NoPhase1','Ablation_NoPhase2','Ablation_NoPhase3',
        'Ablation_Phase1Only','Ablation_Phase3Only',
        'Quality_Filter','Anti_Curriculum']

peft_count = 0
eval_count = 0
for e in exps:
    gm = Path(f'kmc_lora/results/{e}/generated/generation_meta.json')
    if gm.exists():
        d = json.load(open(gm, encoding='utf-8'))
        if d.get('peft_loaded'):
            peft_count += 1
    ej = Path(f'kmc_lora/results/{e}/generated/eval_metrics.json')
    if ej.exists():
        d = json.load(open(ej))
        if d.get('peft_verified'):
            eval_count += 1
            fid = d.get('fid', '?')
            clip = d.get('clip_score', '?')
            print(f"  {e}: FID={fid}, CLIP={clip}")

print(f"\nGenerated: {peft_count}/17  |  Evaluated: {eval_count}/17")
