import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch


def laplacian_variance(gray: np.ndarray) -> float:
    from scipy.ndimage import convolve
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    out = convolve(gray, kernel, mode='reflect')
    return float(out.var())


def compute_basic_quality(df):
    q_res = []
    q_sharp = []
    for path in df['path']:
        img = Image.open(path).convert('L')
        gray = np.array(img, dtype=np.float32) / 255.0
        q_sharp.append(laplacian_variance(gray))
        w, h = img.size
        q_res.append(min(w, h))

    q_res = np.array(q_res, dtype=np.float32)
    q_sharp = np.array(q_sharp, dtype=np.float32)

    def norm(x):
        if x.max() == x.min():
            return np.zeros_like(x)
        return (x - x.min()) / (x.max() - x.min())

    return norm(q_res), norm(q_sharp)


def load_aesthetics_model(device, weights_path):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion2b_s32b_b82k')
    model = model.to(device).eval()
    # Linear head
    linear = torch.nn.Linear(model.text_projection.shape[1], 1)
    sd = torch.load(weights_path, map_location='cpu')
    linear.load_state_dict(sd)
    linear = linear.to(device).eval()
    return model, preprocess, linear


def compute_aesthetics(df, device, weights_path):
    model, preprocess, linear = load_aesthetics_model(device, weights_path)
    scores = []
    with torch.no_grad():
        for path in df['path']:
            img = Image.open(path).convert('RGB')
            x = preprocess(img).unsqueeze(0).to(device)
            feats = model.encode_image(x)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            score = linear(feats).squeeze().item()
            scores.append(score)
    scores = np.array(scores, dtype=np.float32)
    if scores.max() == scores.min():
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())


def choose_k(features, k_range):
    best_k = None
    best_score = -1
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(features)
        try:
            score = silhouette_score(features, labels)
        except Exception:
            score = -1
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metadata', required=True)
    ap.add_argument('--features', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--k-min', type=int, default=3)
    ap.add_argument('--k-max', type=int, default=15)
    ap.add_argument('--use-aesthetics', action='store_true')
    ap.add_argument('--aesthetics-weights', default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.metadata)
    feats = np.load(args.features)

    k = choose_k(feats, (args.k_min, args.k_max))
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(feats)
    centers = km.cluster_centers_

    # Distances to cluster centers
    dists = np.linalg.norm(feats - centers[labels], axis=1)
    typicality = 1.0 / (1.0 + dists)
    typicality = (typicality - typicality.min()) / (typicality.max() - typicality.min() + 1e-8)

    # Heterogeneity: distance of each cluster center to largest cluster center
    counts = np.bincount(labels)
    largest = counts.argmax()
    center_dists = np.linalg.norm(centers - centers[largest], axis=1)
    heterogeneity = center_dists[labels]
    heterogeneity = (heterogeneity - heterogeneity.min()) / (heterogeneity.max() - heterogeneity.min() + 1e-8)

    # Quality
    q_res, q_sharp = compute_basic_quality(df)
    quality = 0.6 * q_sharp + 0.4 * q_res

    if args.use_aesthetics and args.aesthetics_weights:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        q_aes = compute_aesthetics(df, device, args.aesthetics_weights)
        quality = 0.5 * quality + 0.5 * q_aes

    df['cluster'] = labels
    df['typicality'] = typicality
    df['heterogeneity'] = heterogeneity
    df['quality'] = quality

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Clusters: {k}")


if __name__ == '__main__':
    main()
