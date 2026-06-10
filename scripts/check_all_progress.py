"""Check progress of regen_and_eval across all datasets."""
import json
from pathlib import Path

EXPS = ['Random_D-High','KMC_D-High','Random_D-Medium','KMC_D-Medium',
        'Random_D-Low','KMC_D-Low','Random_D-Sub-50','KMC_D-Sub-50',
        'Random_D-Sub-25','KMC_D-Sub-25',
        'Ablation_NoPhase1','Ablation_NoPhase2','Ablation_NoPhase3',
        'Ablation_Phase1Only','Ablation_Phase3Only',
        'Quality_Filter','Anti_Curriculum']

DATASETS = {
    'anime_student': 'kmc_lora/results',
    'wikiart_mixed': 'kmc_lora/results/wikiart_mixed',
    'dreambooth_mixed': 'kmc_lora/results/dreambooth_mixed',
    'dreambooth_single': 'kmc_lora/results/dreambooth_single',
}

for ds, rd in DATASETS.items():
    peft_count = 0
    eval_count = 0
    for e in EXPS:
        gm = Path(rd) / e / 'generated' / 'generation_meta.json'
        if gm.exists():
            d = json.load(open(gm, encoding='utf-8'))
            if d.get('peft_loaded'):
                peft_count += 1
        ej = Path(rd) / e / 'generated' / 'eval_metrics.json'
        if ej.exists():
            d = json.load(open(ej))
            if d.get('peft_verified'):
                eval_count += 1
    print(f"{ds:20s}: Gen={peft_count:2d}/17  Eval={eval_count:2d}/17")
