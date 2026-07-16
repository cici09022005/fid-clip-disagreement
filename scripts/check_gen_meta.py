import json
from pathlib import Path

for exp in ['Random_D-High', 'KMC_D-High', 'Random_D-Medium']:
    meta = Path(f'kmc_lora/results/{exp}/generated/generation_meta.json')
    if meta.exists():
        d = json.load(open(meta, encoding='utf-8'))
        bm = d.get('base_model', 'N/A')
        lp = d.get('lora_path', 'N/A')
        ti = d.get('total_images', 0)
        print(f'{exp}: base_model={bm}, lora_path={lp}, images={ti}')
    else:
        print(f'{exp}: no generation_meta.json')
