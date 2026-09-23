"""
balanced_medical_fid.py -- class-balanced medical FID.

A pooled FID is dominated by the majority classes, so a model can score well while
generating the long tail badly. This computes the per-class one-vs-rest FID
(torchxrayvision DenseNet features, same scaling as medical_FID_offline.py) and
aggregates it 

Decoupled from generation: it scores image sets that already exist (real H5 + CSV
labels, and a self-contained synthetic H5), so it works whatever produced the
synthetic images.

Sample counts: the DenseNet feature is 1024-d, so a per-class FID needs well over
1024 positives per class to be trustworthy.
"""

import os
import argparse
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from scipy import linalg
from tqdm import tqdm

# Reuse the tested loaders + label schema so nothing diverges.
from train_classifier import RealH5Dataset, SynthH5Dataset
from data_preprocessing import LABEL_COLS, PATHOLOGY_COLS
import pandas as pd


# Fetaure extractor and FID
def get_medical_feature_extractor(weights_path, device):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"DenseNet weights not found at {weights_path}.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = torch.load(weights_path, map_location="cpu")
    return model.to(device).eval().requires_grad_(False)


def calculation_FID(real_feats, synth_feats, eps=1e-6):
    mu_r, mu_s = real_feats.mean(0), synth_feats.mean(0)
    sig_r = np.cov(real_feats, rowvar=False)
    sig_s = np.cov(synth_feats, rowvar=False)
    diff = mu_r - mu_s
    covmean, _ = linalg.sqrtm(sig_r.dot(sig_s), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sig_r.shape[0]) * eps
        covmean = linalg.sqrtm((sig_r + offset).dot(sig_s + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sig_r) + np.trace(sig_s) - 2 * np.trace(covmean))


@torch.no_grad()
def extract_all(dataset, extractor, device, batch_size, num_workers):
    """Return (features [N,1024], labels [N,C]). Images are 1-channel in [0,1]."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    feats, labs = [], []
    for img, lab in tqdm(loader, desc="features"):
        img = img.to(device)
        img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)
        img_xrv = img * 2048.0 - 1024.0                       # torchxrayvision range
        f = extractor.features(img_xrv)
        pooled = F.adaptive_avg_pool2d(f, (1, 1)).view(img.size(0), -1)
        feats.append(pooled.cpu().numpy())
        labs.append(lab.numpy())
    return np.vstack(feats), np.vstack(labs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_h5", nargs="+", required=True)
    ap.add_argument("--real_csv", required=True)
    ap.add_argument("--synth_h5", required=True)
    ap.add_argument("--densenet", type=str,
                    default="/home/vault/b143dc/b143dc55/mimic/models/densenet121-res224-mimic_ch.pt")
    ap.add_argument("--reliable_threshold", type=int, default=1000)
    ap.add_argument("--min_fid_samples", type=int, default=2)
    ap.add_argument("--pca_dim", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1-channel [0,1] transform (torchxrayvision DenseNet expects single-channel).
    tf = T.Compose([T.Resize((args.img_size, args.img_size)), T.ToTensor()])

    print("[data] real...", flush=True)
    real_csv = pd.read_csv(args.real_csv)
    real_ds = RealH5Dataset(args.real_h5, real_csv, tf)
    print("[data] synth...", flush=True)
    synth_ds = SynthH5Dataset(args.synth_h5, tf)

    extractor = get_medical_feature_extractor(args.densenet, device)

    print("\n[extract] real features...", flush=True)
    real_f, real_y = extract_all(real_ds, extractor, device, args.batch_size, args.num_workers)
    print("[extract] synth features...", flush=True)
    synth_f, synth_y = extract_all(synth_ds, extractor, device, args.batch_size, args.num_workers)
    print(f"  real {real_f.shape}  synth {synth_f.shape}", flush=True)

    # Optional PCA (fit on real)
    if args.pca_dim and args.pca_dim > 0:
        mu = real_f.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(real_f - mu, full_matrices=False)
        comp = Vt[: args.pca_dim]
        real_f = (real_f - mu) @ comp.T
        synth_f = (synth_f - mu) @ comp.T
        print(f"  PCA-reduced features to {args.pca_dim} dims.", flush=True)

    feat_dim = real_f.shape[1]
    rng = np.random.default_rng(0)

    def rr_floor(feats):
        """Real-vs-real FID: split these features in half. This is the FID a
        perfect generator would score at this sample size / feature extractor."""
        n = len(feats)
        if n < 4:
            return None
        idx = rng.permutation(n)
        h = n // 2
        return calculation_FID(feats[idx[:h]], feats[idx[h:]])

    # per-class one-vs-rest FID for every class (+ real-vs-real floor)
    per_class = {}
    computed = []          # (name, fid, excess, min_n, reliable)
    for c, name in enumerate(LABEL_COLS):
        rf = real_f[real_y[:, c] == 1.0]
        sf = synth_f[synth_y[:, c] == 1.0]
        n_r, n_s = len(rf), len(sf)
        min_n = min(n_r, n_s)
        if min_n < args.min_fid_samples:
            per_class[name] = {"fid": None, "floor": None, "excess": None,
                               "n_real": n_r, "n_synth": n_s, "reliable": False,
                               "note": "no FID (need >=2 both sides)"}
            continue
        fid_c = calculation_FID(rf, sf)
        floor_c = rr_floor(rf)
        excess_c = (fid_c - floor_c) if floor_c is not None else None
        reliable = min_n >= args.reliable_threshold and min_n >= feat_dim
        note = "ok" if reliable else f"LOW-N (< {max(args.reliable_threshold, feat_dim)})"
        per_class[name] = {"fid": fid_c, "floor": floor_c, "excess": excess_c,
                           "n_real": n_r, "n_synth": n_s, "reliable": reliable, "note": note}
        computed.append((name, fid_c, excess_c, min_n, reliable))

    # aggregated
    pooled = calculation_FID(real_f, synth_f)
    pooled_floor = rr_floor(real_f)
    all_fids = [f for _, f, _, _, _ in computed]
    macro_all = float(np.mean(all_fids)) if all_fids else None
    rel_fids = [f for _, f, _, _, rel in computed if rel]
    macro_reliable = float(np.mean(rel_fids)) if rel_fids else None
    excesses = [e for _, _, e, _, _ in computed if e is not None]
    macro_excess = float(np.mean(excesses)) if excesses else None

    # report
    print("\n================= BALANCED MEDICAL FID =================", flush=True)
    print(f"{'class':<26}{'FID':>9}{'floor':>9}{'excess':>9}{'n_real':>8}  note", flush=True)
    for name in LABEL_COLS:
        d = per_class[name]
        is_path = " (path)" if name in PATHOLOGY_COLS else ""
        if d["fid"] is None:
            print(f"{name:<26}{'--':>9}{'--':>9}{'--':>9}{d['n_real']:>8}  {d['note']}{is_path}",
                  flush=True)
        else:
            fl = "--" if d["floor"] is None else f"{d['floor']:.1f}"
            ex = "--" if d["excess"] is None else f"{d['excess']:.1f}"
            print(f"{name:<26}{d['fid']:>9.1f}{fl:>9}{ex:>9}{d['n_real']:>8}  {d['note']}{is_path}",
                  flush=True)

    print("\n-------------------------------------------------------", flush=True)
    if macro_all is not None:
        print(f"  MACRO FID  -- ALL classes    : {macro_all:.3f}   "
              f"[{len(computed)}/{len(LABEL_COLS)}]", flush=True)
    if macro_excess is not None:
        print(f"  MACRO EXCESS over real floor : {macro_excess:.3f}   "
              f"<-- the number that isolates the GENERATOR from sample-size noise", flush=True)
    if macro_reliable is not None:
        print(f"  MACRO FID  -- reliable only  : {macro_reliable:.3f}   "
              f"[{len(rel_fids)} classes n >= {max(args.reliable_threshold, feat_dim)}]",
              flush=True)
    path_all = [f for n, f, _, _, _ in computed if n in PATHOLOGY_COLS]
    if path_all:
        print(f"  MACRO FID  -- pathologies    : {np.mean(path_all):.3f}   "
              f"[{len(path_all)} classes]", flush=True)
    fl = f"{pooled_floor:.3f}" if pooled_floor is not None else "n/a"
    print(f"  pooled FID {pooled:.3f}   |   pooled real-vs-real floor {fl}", flush=True)
    print("=======================================================", flush=True)
    print("[how to read] 'floor' = real-vs-real FID at that class's sample size (what a "
          "PERFECT generator scores here). 'excess' = FID - floor = the real generator gap. "
          "If excess is small, the big FID is mostly sample-size noise, not your images; if "
          "excess is large, the generator genuinely differs from real in feature space.",
          flush=True)
    print("[reminder] FID needs n >> feature_dim (1024) per side to be trustworthy. "
          "Low-N rows above are inflated by sampling noise, not necessarily bad images. "
          "Enlarge the real reference (all shards) and/or use --pca_dim before comparing.",
          flush=True)


if __name__ == "__main__":
    main()