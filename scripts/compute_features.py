import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import open_clip


def load_clip(device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='laion2b_s32b_b82k'
    )
    model = model.eval().to(device)
    return model, preprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metadata', required=True)
    ap.add_argument('--out-features', required=True)
    ap.add_argument('--out-paths', required=True)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    df = pd.read_csv(args.metadata)
    paths = df['path'].tolist()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, preprocess = load_clip(device)

    feats = []
    with torch.no_grad():
        for path in tqdm(paths, desc='CLIP ViT-L-14'):
            img = Image.open(path).convert('RGB')
            x = preprocess(img).unsqueeze(0).to(device)
            y = model.encode_image(x)
            y = y.squeeze(0).cpu().numpy()
            feats.append(y)

    feats = np.stack(feats, axis=0)
    Path(args.out_features).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_features, feats)
    with open(args.out_paths, 'w', encoding='utf-8') as f:
        json.dump(paths, f, ensure_ascii=False)


if __name__ == '__main__':
    main()
