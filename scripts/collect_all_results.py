"""Collect all FID/CLIP results across all datasets into a summary CSV."""
import json, csv
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

all_rows = []
for ds, rd in DATASETS.items():
    for e in EXPS:
        ej = Path(rd) / e / 'generated' / 'eval_metrics.json'
        # Also read training summary
        ts_path = Path(rd) / e / 'training_summary.json'
        if not ts_path.exists():
            # Try phase3 for KMC
            for phase in ['phase3', 'phase2', 'phase1']:
                ts_path = Path(rd) / e / phase / 'training_summary.json'
                if ts_path.exists():
                    break
        
        final_loss = None
        training_time = None
        if ts_path.exists():
            td = json.load(open(ts_path, encoding='utf-8'))
            final_loss = td.get('final_loss')
            training_time = td.get('elapsed_sec')
        
        fid = None
        clip_score = None
        if ej.exists():
            ed = json.load(open(ej))
            fid = ed.get('fid')
            clip_score = ed.get('clip_score')
        
        row = {
            'dataset': ds,
            'experiment': e,
            'fid': fid,
            'clip_score': clip_score,
            'final_loss': final_loss,
            'training_time_sec': training_time,
        }
        all_rows.append(row)
        
        # Print key comparisons
        if fid is not None:
            loss_str = f"  loss={final_loss:.4f}" if final_loss else ""
            print(f"{ds:20s} {e:25s}  FID={fid:8.2f}  CLIP={clip_score:.4f}{loss_str}")

# Write CSV
out = Path('kmc_lora/results/all_eval_results.csv')
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['dataset','experiment','fid','clip_score','final_loss','training_time_sec'])
    w.writeheader()
    w.writerows(all_rows)
print(f"\nSaved: {out} ({len(all_rows)} rows)")
