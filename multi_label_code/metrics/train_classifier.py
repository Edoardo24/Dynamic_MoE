"""
train_classifier.py -- downstream utility: does synthetic data improve a
chest-X-ray pathology classifier, and specifically on the long tail?

The three arms (--mix):
  real  : train on real images only                      (the baseline to beat)
  synth : train on synthetic only                        (is synth learnable at all?)
  both  : real + synthetic                               (does synth add value?)
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

import torchvision.transforms as T
from torchvision.models import densenet121

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocessing import LABEL_COLS, PATHOLOGY_COLS

DENSENET_XRV = "/home/vault/b143dc/b143dc55/mimic/models/densenet121-res224-mimic_ch.pt"


# Dataset
class RealH5Dataset(Dataset):
    """Real images from one or more h5 shards, labels joined from the CSV by path."""

    def __init__(self, h5_paths, csv_df, transform):
        self.transform = transform
        self.index = []           # (h5_idx, row_in_h5, label_vec)
        self.h5_paths = list(h5_paths)
        self.handles = None
        lab_by_path = csv_df.set_index("path")[LABEL_COLS]
        for hi, hp in enumerate(self.h5_paths):
            with h5py.File(hp, "r") as f:
                paths = [str(p)[2:-1] for p in f["paths"][:]]
            keep = 0
            for ri, p in enumerate(paths):
                if p in lab_by_path.index:
                    v = lab_by_path.loc[p].to_numpy().astype("float32")
                    self.index.append((hi, ri, (v == 1.0).astype("float32")))
                    keep += 1
            print(f"    {os.path.basename(hp)}: {keep}/{len(paths)} matched", flush=True)
        print(f"  RealH5Dataset: {len(self.index)} images", flush=True)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        if self.handles is None:
            self.handles = [h5py.File(p, "r") for p in self.h5_paths]
        hi, ri, lab = self.index[i]
        arr = self.handles[hi]["images"][ri]
        img = self.transform(Image.fromarray(arr).convert("L"))
        return img, torch.from_numpy(lab)


class SynthH5Dataset(Dataset):
    # Synthetic images + labels

    def __init__(self, h5_path, transform, limit=None):
        self.h5_path = h5_path
        self.transform = transform
        self.h5 = None
        with h5py.File(h5_path, "r") as f:
            self.n = f["images"].shape[0]
            self.labels = f["labels"][:].astype("float32")
            cols = f.attrs.get("label_cols", ",".join(LABEL_COLS)).split(",")
        # if cols != LABEL_COLS:
        #     raise RuntimeError(f"synth label_cols mismatch:\n {cols}\nvs\n {LABEL_COLS}")
        # Check every required label is present.
        missing = [c for c in LABEL_COLS if c not in cols]
        if missing:
            raise RuntimeError(f"Synthetic H5 is missing required labels: {missing}")
        
        # Map/reorder columns to LABEL_COLS order (drops Support Devices).
        self.label_indices = [cols.index(c) for c in LABEL_COLS]
        if limit is not None:
            self.n = min(self.n, limit)
        print(f"  SynthH5Dataset: {self.n} images from {os.path.basename(h5_path)}", flush=True)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, "r")
        arr = self.h5["images"][i]
        img = self.transform(Image.fromarray(arr).convert("L"))
        return img, torch.from_numpy(self.labels[i])


# Model
def build_classifier(n_out, init="scratch", device="cuda"):
    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, n_out)
    return model.to(device)


# Evaluation
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    P, Y = [], []
    for img, lab in tqdm(loader, desc="eval", leave=False):
        img = img.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            logit = model(img)
        P.append(torch.sigmoid(logit.float()).cpu().numpy())
        Y.append(lab.numpy())
    return np.vstack(P), np.vstack(Y)


def tune_thresholds(probs, y):
    """Per-class threshold maximising F1 on the (real) val split."""
    ths = np.full(y.shape[1], 0.5, dtype=np.float64)
    grid = np.linspace(0.02, 0.98, 49)
    for c in range(y.shape[1]):
        if y[:, c].sum() < 5:
            continue
        best, bt = -1.0, 0.5
        for t in grid:
            f = f1_score(y[:, c], (probs[:, c] >= t).astype(int), zero_division=0)
            if f > best:
                best, bt = f, t
        ths[c] = bt
    return ths


def score(probs, y, ths):
    pred = (probs >= ths[None, :]).astype(int)
    out = {"per_class": {}}
    f1s = []
    for c, name in enumerate(LABEL_COLS):
        n_pos = int(y[:, c].sum())
        if n_pos < 5:
            out["per_class"][name] = {"f1": None, "auroc": None, "ap": None,
                                      "n_pos": n_pos, "note": "too few positives"}
            continue
        f1 = float(f1_score(y[:, c], pred[:, c], zero_division=0))
        try:
            auroc = float(roc_auc_score(y[:, c], probs[:, c]))
            ap = float(average_precision_score(y[:, c], probs[:, c]))
        except ValueError:
            auroc = ap = None
        out["per_class"][name] = {"f1": f1, "auroc": auroc, "ap": ap,
                                  "n_pos": n_pos, "threshold": float(ths[c]), "note": "ok"}
        if name in PATHOLOGY_COLS:
            f1s.append(f1)
    out["macro_f1_pathologies"] = float(np.mean(f1s)) if f1s else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", required=True,
                    choices=["real", "synth", "both", "real_dup"],
                    help="real_dup = real duplicated to match 'both' in SIZE and "
                         "GRADIENT STEPS, with zero synthetic. This is the control "
                         "that separates 'synthetic information helps' from 'both "
                         "simply trained 2x longer'. Since the train transform "
                         "applies RandomAffine, the duplicate is a second AUGMENTED "
                         "view -- so both-vs-real_dup asks whether diffusion samples "
                         "beat free classical augmentation at matched compute.")
    ap.add_argument("--real_limit", type=int, default=None)
    ap.add_argument("--real_train_h5", nargs="+", required=True)
    ap.add_argument("--real_val_h5", nargs="+", required=True)
    ap.add_argument("--real_test_h5", nargs="+", required=True)
    ap.add_argument("--synth_h5", default=None)
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--init", default="scratch", choices=["scratch", "xrv"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--synth_limit", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--save_model", default=None,
                    help="path to save the best classifier (needed by vae_ceiling.py)")
    args = ap.parse_args()

    if args.mix in ("synth", "both") and not args.synth_h5:
        raise ValueError("--synth_h5 required for mix=synth|both")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cls] mix={args.mix} seed={args.seed} init={args.init}", flush=True)

    csv_df = pd.read_csv(args.csv_path)
    for c in LABEL_COLS:
        if c not in csv_df.columns:
            raise ValueError(f"CSV missing label column: {c}")

    # grayscale
    norm = T.Normalize([0.5] * 3, [0.25] * 3)
    tf_train = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.RandomHorizontalFlip(p=0.0),          # CXR laterality is meaningful: OFF
        T.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.97, 1.03)),
        T.ToTensor(), T.Lambda(lambda x: x.repeat(3, 1, 1)), norm,
    ])
    tf_eval = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(), T.Lambda(lambda x: x.repeat(3, 1, 1)), norm,
    ])

    print("[data] real train:", flush=True)
    real_train = RealH5Dataset(args.real_train_h5, csv_df, tf_train)
    print("[data] real val:", flush=True)
    val_ds = RealH5Dataset(args.real_val_h5, csv_df, tf_eval)
    print("[data] real test:", flush=True)
    test_ds = RealH5Dataset(args.real_test_h5, csv_df, tf_eval)

    # subsample real to isolate quality from quantity 
    if args.real_limit is not None and args.real_limit < len(real_train.index):
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(len(real_train.index), size=args.real_limit, replace=False)
        real_train.index = [real_train.index[i] for i in sorted(keep)]
        print(f"[data] real subsampled -> {len(real_train.index)} images "
              f"(seed {args.seed}) for a size-matched comparison", flush=True)

    # class frequency in the real training data
    real_freq = {}
    ytr = np.stack([lab for _, _, lab in real_train.index])
    for c, name in enumerate(LABEL_COLS):
        real_freq[name] = {"n_pos": int(ytr[:, c].sum()),
                           "frac": float(ytr[:, c].mean())}

    # assemble the training arm 
    if args.mix == "real":
        train_ds = real_train
    elif args.mix == "real_dup":
        # size- and step-matched to 'both', no synthetic data at all
        train_ds = ConcatDataset([real_train, real_train])
    elif args.mix == "synth":
        print("[data] synth:", flush=True)
        train_ds = SynthH5Dataset(args.synth_h5, tf_train, args.synth_limit)
    else:
        print("[data] synth:", flush=True)
        synth = SynthH5Dataset(args.synth_h5, tf_train, args.synth_limit)
        train_ds = ConcatDataset([real_train, synth])
    print(f"[data] TRAIN SIZE ({args.mix}) = {len(train_ds)}", flush=True)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)

    model = build_classifier(len(LABEL_COLS), args.init, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.BCEWithLogitsLoss()

    best_macro, best_state, best_epoch = -1.0, None, -1
    for epoch in range(args.epochs):
        model.train()
        run, seen = 0.0, 0
        for img, lab in tqdm(train_dl, desc=f"E{epoch+1}[{args.mix}]"):
            img = img.to(device, non_blocking=True); lab = lab.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                loss = crit(model(img), lab)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run += loss.item() * img.size(0); seen += img.size(0)
        sched.step()

        # model selection on the real val split only
        vp, vy = predict(model, val_dl, device)
        ths = tune_thresholds(vp, vy)
        vres = score(vp, vy, ths)
        macro = vres["macro_f1_pathologies"] or 0.0
        print(f"Epoch {epoch+1} | train {run/max(1,seen):.4f} | "
              f"val macroF1(path) {macro:.4f}", flush=True)
        if macro > best_macro:
            best_macro, best_epoch = macro, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_ths = ths.copy()
            print(f"  new best val macroF1 {best_macro:.4f}", flush=True)

    # best-val model, val-tuned thresholds, real test 
    model.load_state_dict(best_state)
    tp, ty = predict(model, test_dl, device)
    test_res = score(tp, ty, best_ths)

    out = {
        "mix": args.mix, "seed": args.seed, "init": args.init,
        "real_limit": args.real_limit,
        "synth_h5": args.synth_h5, "best_epoch": best_epoch,
        "val_macro_f1": best_macro,
        "train_size": len(train_ds),
        "test": test_res,
        "real_train_frequency": real_freq,
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {args.out_json}")
    if args.save_model:
        torch.save({"state_dict": best_state, "thresholds": best_ths.tolist(),
                    "label_cols": LABEL_COLS, "mix": args.mix, "seed": args.seed,
                    "init": args.init, "best_epoch": best_epoch}, args.save_model)
        print(f"Saved classifier -> {args.save_model}")
    print(f"\nTEST macro-F1 (pathologies): {test_res['macro_f1_pathologies']:.4f}")
    print(f"\n{'class':<28}{'n_pos':>8}{'real_frac':>11}{'F1':>9}{'AUROC':>9}")
    for name in LABEL_COLS:
        d = test_res["per_class"][name]
        rf = real_freq[name]["frac"] * 100
        if d["f1"] is None:
            print(f"{name:<28}{d['n_pos']:>8}{rf:>10.2f}%{'n/a':>9}{'n/a':>9}")
        else:
            au = f"{d['auroc']:.3f}" if d["auroc"] is not None else "n/a"
            print(f"{name:<28}{d['n_pos']:>8}{rf:>10.2f}%{d['f1']:>9.3f}{au:>9}")


if __name__ == "__main__":
    main()