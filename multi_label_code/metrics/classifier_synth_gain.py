"""
classifier_synth_gain.py -- does adding synthetic images improve classification?

Downstream-utility test, run as a single controlled comparison rather than one
mode at a time. It trains the same classifier under two conditions and reports the
per-class and tail-focused improvement:

  real : train on real only                         (baseline)
  both : train on real + synthetic                  (does synth add value?)
  [dup]: train on real duplicated to both's size    (control: more steps, no new
         data; optional, --with_control)
"""

import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Reuse tested components so this matches the other classifier runs.
from train_classifier import (RealH5Dataset, SynthH5Dataset, build_classifier,
                              predict, tune_thresholds, score)
from data_preprocessing import LABEL_COLS, PATHOLOGY_COLS


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)
    torch.cuda.manual_seed_all(s)


def train_and_eval(train_ds, val_ds, test_ds, args, device, tag):
    # Train under one condition, select on real val macro-F1, and score the real
    # test set with val-tuned thresholds
    set_seed(args.seed)  # identical init across conditions
    model = build_classifier(len(LABEL_COLS), init=args.init, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda")

    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True)
    vl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    best_macro, best_state, best_ths = -1.0, None, None
    for ep in range(args.epochs):
        model.train()
        for img, lab in tqdm(tl, desc=f"{tag} ep{ep+1}", leave=False):
            img, lab = img.to(device), lab.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = crit(model(img), lab)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        vp, vy = predict(model, vl, device)
        ths = tune_thresholds(vp, vy)
        macro = score(vp, vy, ths)["macro_f1_pathologies"] or -1.0
        print(f"[{tag}] epoch {ep+1}/{args.epochs}: val macro-F1(path) = {macro:.4f}", flush=True)
        if macro > best_macro:
            best_macro = macro
            best_state = copy.deepcopy(model.state_dict())
            best_ths = ths

    model.load_state_dict(best_state)
    tl_test = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)
    tp, ty = predict(model, tl_test, device)
    return score(tp, ty, best_ths), best_macro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_train_h5", nargs="+", required=True)
    ap.add_argument("--real_csv", required=True)
    ap.add_argument("--real_val_h5", nargs="+", required=True)
    ap.add_argument("--real_test_h5", nargs="+", required=True)
    ap.add_argument("--synth_h5", required=True)
    ap.add_argument("--with_control", action="store_true",
                    help="Also train the real-duplicated control (more steps, no new data).")
    ap.add_argument("--init", type=str, default="scratch", choices=["scratch", "xrv"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Same 3-channel transform as train_classifier.py.
    norm = T.Normalize([0.5] * 3, [0.25] * 3)
    tf = T.Compose([T.Resize((args.img_size, args.img_size)), T.ToTensor(),
                    T.Lambda(lambda x: x.repeat(3, 1, 1)), norm])

    csv = pd.read_csv(args.real_csv)
    print("[data] real train / val / test + synth...", flush=True)
    real_train = RealH5Dataset(args.real_train_h5, csv, tf)
    real_val = RealH5Dataset(args.real_val_h5, csv, tf)
    real_test = RealH5Dataset(args.real_test_h5, csv, tf)
    synth = SynthH5Dataset(args.synth_h5, tf)

    # real only
    print("\n########## CONDITION: real ##########", flush=True)
    res_real, val_real = train_and_eval(real_train, real_val, real_test, args, device, "real")

    # real + synth 
    print("\n########## CONDITION: both (real + synth) ##########", flush=True)
    both_train = ConcatDataset([real_train, synth])
    res_both, __= train_and_eval(both_train, real_val, real_test, args, device, "both")

    res_ctrl = None
    if args.with_control:
        print("\n########## CONDITION: real_dup (control) ##########", flush=True)
        reps = max(2, 1 + len(synth) // max(1, len(real_train)))
        dup_train = ConcatDataset([real_train] * reps)
        res_ctrl, __ = train_and_eval(dup_train, real_val, real_test, args, device, "real_dup")

    def f1_of(res, name):
        d = res["per_class"].get(name, {})
        return d.get("f1"), d.get("n_pos")

    rows = []
    for name in LABEL_COLS:
        f_r, n_pos = f1_of(res_real, name)
        f_b, _ = f1_of(res_both, name)
        delta = (f_b - f_r) if (f_r is not None and f_b is not None) else None
        rows.append((name, n_pos if n_pos is not None else 0, f_r, f_b, delta))
    # rarest first, since the claim is that synthetic helps the tail
    rows.sort(key=lambda r: r[1])

    print("\n===================== SYNTHETIC-DATA GAIN (real test) =====================", flush=True)
    print(f"{'class':<28}{'n_pos':>7}{'F1 real':>10}{'F1 both':>10}{'dF1':>9}", flush=True)
    for name, n_pos, f_r, f_b, delta in rows:
        path = "  (path)" if name in PATHOLOGY_COLS else ""
        if delta is None:
            print(f"{name:<28}{n_pos:>7}{'--':>10}{'--':>10}{'--':>9}{path}", flush=True)
        else:
            arrow = "+" if delta >= 0 else ""
            print(f"{name:<28}{n_pos:>7}{f_r:>10.3f}{f_b:>10.3f}{arrow}{delta:>8.3f}{path}",
                  flush=True)

    print("\n---------------------------------------------------------------------------", flush=True)
    mr = res_real["macro_f1_pathologies"]
    mb = res_both["macro_f1_pathologies"]
    print(f"  macro-F1 (pathologies)   real = {mr:.4f}   both = {mb:.4f}   "
          f"delta = {mb - mr:+.4f}", flush=True)
    if res_ctrl is not None:
        mc = res_ctrl["macro_f1_pathologies"]
        print(f"  control (real_dup)       macro = {mc:.4f}   "
              f"(both must beat THIS, not just real, to prove synth carries information)",
              flush=True)

    # tail vs head: does the gain concentrate in rare classes?
    valid = [(n_pos, d) for _, n_pos, _, _, d in rows if d is not None]
    if len(valid) >= 4:
        k = max(1, len(valid) // 3)
        tail = np.mean([d for _, d in valid[:k]])          # rarest third
        head = np.mean([d for _, d in valid[-k:]])         # most common third
        print(f"  mean dF1 -- rarest third: {tail:+.4f}   |   common third: {head:+.4f}", flush=True)
        if tail > head:
            print("  -> gains concentrate in the tail, as the long-tail hypothesis predicts.",
                  flush=True)
    print("===========================================================================", flush=True)


if __name__ == "__main__":
    main()