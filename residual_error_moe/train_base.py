"""
train_base.py -- train the base DiT (no MoE) on the full multi-label MIMIC set.

Run this first. The MoE stages freeze this backbone and point --base_checkpoint
at its best_model.pt.

A few design points worth recording:

  - Patient-level split
  - Class-balanced sampler. Each image is weighted by the inverse sqrt frequency of its 
    rarest present finding
  - Real unconditional embedding. CFG dropout uses the cached "" embedding
    (row 0 of the bank)
  - Per-finding dropout and order shuffling. Findings are dropped with
    probability --finding_dropout and the survivors read from a random cached
    ordering
"""

import argparse
import glob
import json
import os
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, Transformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from data_preprocessing import (
    DEVICE_TOKEN,
    LABEL_COLS,
    PATHOLOGY_COLS,
    SUPPORT_COL,
    tokens_from_labels,
)

LATENT_SCALE = 0.18215


# Prompt bank
class PromptBank:
    """Mapped [P, 77, 4096] fp16 T5 embeddings, plus a subset to rows index."""

    def __init__(self, bank_dir):
        with open(os.path.join(bank_dir, "prompt_index.json")) as f:
            meta = json.load(f)
        self.subset_rows = meta["subset_rows"]
        self.uncond_row = meta["uncond_row"]
        self.n_prompts = len(meta["prompts"])
        self._emb_path = os.path.join(bank_dir, "prompt_embeddings.fp16.npy")
        self._emb = None

    @property
    def emb(self):
        if self._emb is None:
            self._emb = np.load(self._emb_path, mmap_mode="r")
        return self._emb

    def row_for(self, tokens, rng):
        key = "|".join(sorted(tokens))
        rows = self.subset_rows.get(key)
        if not rows:
            return self.uncond_row
        return rows[rng.randrange(len(rows))]

    def get(self, row):
        return torch.from_numpy(np.ascontiguousarray(self.emb[row]))


# Dataset
class CachedLatentDataset(Dataset):
    def __init__(self, files, bank, finding_dropout=0.1, cfg_dropout=0.10,
                 max_drop=2, train=True):
        self.files = files
        self.bank = bank
        self.finding_dropout = finding_dropout
        self.cfg_dropout = cfg_dropout
        self.max_drop = max_drop
        self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = torch.load(self.files[i], weights_only=True, map_location="cpu")

        mu = d["latent_mean"].float()
        if self.train:
            sigma = torch.exp(0.5 * d["latent_logvar"].float())
            z = (mu + sigma * torch.randn_like(mu)) * LATENT_SCALE
        else:
            z = mu * LATENT_SCALE  # deterministic val

        labels = d["labels"]
        tokens = tokens_from_labels(labels.numpy())

        if self.train:
            rng = random.Random(torch.randint(0, 2**31 - 1, (1,)).item())

            # classifier-free guidance
            if rng.random() < self.cfg_dropout:
                row = self.bank.uncond_row
            else:
                # per-finding dropout, capped at max_drop so the bank always has
                # the resulting subset
                keep = [t for t in tokens if rng.random() >= self.finding_dropout]
                if len(tokens) - len(keep) > self.max_drop:
                    keep = rng.sample(tokens, len(tokens) - self.max_drop)
                row = self.bank.row_for(keep, rng)
        else:
            row = self.bank.row_for(tokens, random.Random(0))

        return z, self.bank.get(row), labels


# Splitting and balancing, no need to open all files
def load_manifest(cache_dirs):
    """
    cache_dirs: one or more per-shard cache directories. Composing the subset
    here rather than at cache time keeps the shard partition a training-time
    decision
    """
    if isinstance(cache_dirs, str):
        cache_dirs = [cache_dirs]

    frames = []
    for d in cache_dirs:
        mans = sorted(glob.glob(os.path.join(d, "manifest_*.csv")))
        if not mans:
            raise FileNotFoundError(
                f"No manifest_*.csv in {d}. Re-run extract_features.py --stage latents."
            )
        for m in mans:
            sub = pd.read_csv(m)
            sub["file"] = sub["file"].apply(lambda f: os.path.join(d, f))
            frames.append(sub)

    df = pd.concat(frames, ignore_index=True)

    before = len(df)
    df = df.drop_duplicates(subset="path").reset_index(drop=True)
    if len(df) != before:
        print(f"[WARN] dropped {before - len(df)} duplicate image paths across cache dirs. "
              f"Overlapping shard sets?", flush=True)

    df = df[df["file"].apply(os.path.exists)].reset_index(drop=True)
    print(f"Manifest: {len(df)} cached latents from {len(cache_dirs)} cache dir(s): "
          f"{[os.path.basename(d) for d in cache_dirs]}", flush=True)
    return df


def patient_level_split(df, val_frac=0.05, seed=42):
    """
    Group by group_id so no patient straddles the train/val boundary
    """
    gids = sorted(df["group_id"].unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(gids)

    n_val = max(1, int(len(gids) * val_frac))
    val_gids = set(gids[:n_val])

    is_val = df["group_id"].isin(val_gids)
    train_df = df[~is_val].reset_index(drop=True)
    val_df = df[is_val].reset_index(drop=True)

    print(f"Split: {len(train_df)} train / {len(val_df)} val images "
          f"across {len(gids)} patients ({n_val} held out).", flush=True)
    return train_df, val_df


def balanced_weights(train_df, alpha=0.5):
    """
    w_i = (1 / freq(rarest finding present in image i)) ** alpha
    """
    label_mat = train_df[LABEL_COLS].to_numpy().astype(np.float32)
    freq = np.maximum(label_mat.sum(axis=0), 1.0)

    print("\nTraining-set label frequencies:")
    for c, n in zip(LABEL_COLS, freq):
        print(f"  {c:<28} {int(n):>7}")

    nf_idx = LABEL_COLS.index("No Finding")
    n_path = len(PATHOLOGY_COLS)

    weights = np.empty(len(train_df), dtype=np.float64)
    for i, row in enumerate(label_mat):
        present = [j for j in range(n_path) if row[j] == 1.0]
        rarest = min(freq[j] for j in present) if present else freq[nf_idx]
        weights[i] = (1.0 / rarest) ** alpha

    weights /= weights.mean()
    print(f"\nSampler weights: min={weights.min():.3f} max={weights.max():.3f} "
          f"ratio={weights.max() / weights.min():.1f}x\n", flush=True)
    return torch.from_numpy(weights)


def build_dit():
    return Transformer2DModel(
        sample_size=64, num_layers=12, patch_size=2, attention_head_dim=64,
        num_attention_heads=16, in_channels=4, out_channels=4,
        cross_attention_dim=4096, norm_type="ada_norm_zero",
        num_embeds_ada_norm=1, activation_fn="gelu-approximate",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cached_img_path", type=str, nargs="+", required=True,
                   help="one or more per-shard latent cache dirs, e.g. cache_s0000 cache_s0001")
    p.add_argument("--val_cached_path", type=str, nargs="+", default=None,
                   help="latent cache dir(s) built from MIMIC's official val folder. "
                        "If omitted, a patient-level split is carved out of train.")
    p.add_argument("--bank_dir", type=str, required=True)
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--physical_batch_size", type=int, default=16)
    p.add_argument("--accumulation_steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--balance_alpha", type=float, default=0.5,
                   help="0 = uniform sampling, 0.5 = inverse sqrt freq, 1 = inverse freq")
    p.add_argument("--finding_dropout", type=float, default=0.10)
    p.add_argument("--cfg_dropout", type=float, default=0.10)
    p.add_argument("--max_drop", type=int, default=2, help="must match extract_features")
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training base DiT on {device}", flush=True)

    manifest = load_manifest(args.cached_img_path)

    if args.val_cached_path:
        # MIMIC ships separate train / val / test folders and the official
        # splits are already patient-disjoint
        val_manifest = load_manifest(args.val_cached_path)
        train_df, val_df = manifest, val_manifest
        overlap = set(train_df["path"]) & set(val_df["path"])
        if overlap:
            raise SystemExit(f"train and val caches share {len(overlap)} images.")
        shared_patients = set(train_df["group_id"]) & set(val_df["group_id"])
        if shared_patients:
            print(f"[WARN] {len(shared_patients)} patients appear in BOTH train and val "
                  f"caches. The official MIMIC split should be patient-disjoint -- check "
                  f"how the CSV's Split column was built.", flush=True)
        print(f"Using official val folder: {len(train_df)} train / {len(val_df)} val", flush=True)
    else:
        # No val cache: hold out whole patients from trai
        train_df, val_df = patient_level_split(manifest)

    train_files = train_df["file"].tolist()
    val_files = val_df["file"].tolist()

    bank = PromptBank(args.bank_dir)
    print(f"Prompt bank: {bank.n_prompts} prompts.", flush=True)

    train_ds = CachedLatentDataset(train_files, bank, args.finding_dropout,
                                   args.cfg_dropout, args.max_drop, train=True)
    val_ds = CachedLatentDataset(val_files, bank, 0.0, 0.0, args.max_drop, train=False)

    if args.balance_alpha > 0:
        w = balanced_weights(train_df, alpha=args.balance_alpha)
        sampler = WeightedRandomSampler(w, num_samples=len(train_files), replacement=True)
        train_dl = DataLoader(train_ds, batch_size=args.physical_batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    else:
        train_dl = DataLoader(train_ds, batch_size=args.physical_batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)

    val_dl = DataLoader(val_ds, batch_size=args.physical_batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear",
                              prediction_type="v_prediction")

    model = build_dit().to(device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"DiT parameters: {n_params / 1e6:.1f}M (all trainable)", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.999))

    ema_model = EMAModel(model.parameters(), decay=args.ema_decay, use_ema_warmup=True,
                         model_cls=Transformer2DModel, model_config=model.config)
    ema_model.to(device)

    steps_per_epoch = len(train_dl) // args.accumulation_steps
    lr_sched = get_scheduler("cosine", optimizer=optimizer,
                             num_warmup_steps=args.warmup_steps,
                             num_training_steps=args.epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda")

    start_epoch, best_val = 0, float("inf")
    ckpts = glob.glob(os.path.join(args.checkpoint_path, "dit_epoch_*.pt"))
    if ckpts:
        latest = max(ckpts, key=lambda f: int(re.search(r"dit_epoch_(\d+)", f).group(1)))
        print(f"[Resume] {latest}", flush=True)
        ck = torch.load(latest, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model_state_dict"])
        ema_model.load_state_dict(ck["ema_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        lr_sched.load_state_dict(ck["scheduler_state_dict"])
        scaler.load_state_dict(ck["scaler_state_dict"])
        best_val = ck.get("best_val_loss", float("inf"))
        start_epoch = ck["epoch"]
        model.to(device)
        ema_model.to(device)

    os.makedirs(args.checkpoint_path, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---", flush=True)
        model.train()
        run_loss, n_seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, (latents, text_emb, _labels) in enumerate(
                tqdm(train_dl, desc=f"Train {epoch + 1}")):
            latents = latents.to(device, non_blocking=True)
            text_emb = text_emb.to(device, non_blocking=True)
            bsz = latents.shape[0]

            # CFG dropout is already applied per sample in the Dataset, using
            # the cached "" embedding rather than zeros.
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                                      (bsz,), device=device).long()
            noise = torch.randn_like(latents)
            noisy = scheduler.add_noise(latents, noise, timesteps)
            target = scheduler.get_velocity(latents, noise, timesteps)
            dummy = torch.zeros((bsz,), dtype=torch.long, device=device)

            with torch.amp.autocast("cuda"):
                out = model(hidden_states=noisy, encoder_hidden_states=text_emb,
                            timestep=timesteps, class_labels=dummy).sample
                loss = F.mse_loss(out.float(), target.float())

            scaler.scale(loss / args.accumulation_steps).backward()
            run_loss += loss.item() * bsz
            n_seen += bsz

            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                lr_sched.step()
                optimizer.zero_grad(set_to_none=True)
                ema_model.step(model.parameters())

            if (step + 1) % (100 * args.accumulation_steps) == 0:
                print(f"  step {step + 1} | loss {run_loss / n_seen:.4f} | "
                      f"lr {lr_sched.get_last_lr()[0]:.2e}", flush=True)

        train_loss = run_loss / max(1, n_seen)

        # validation on EMA weights
        ema_model.store(model.parameters())
        ema_model.copy_to(model.parameters())
        model.eval()

        # Fixed noise/timesteps per epoch so val loss is comparable across epochs
        # rather than dominated by the timestep draw.
        val_loss, val_n = 0.0, 0
        g = torch.Generator(device=device).manual_seed(1234)
        with torch.no_grad():
            for latents, text_emb, _labels in tqdm(val_dl, desc=f"Val {epoch + 1}"):
                latents = latents.to(device)
                text_emb = text_emb.to(device)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                                          (bsz,), device=device, generator=g).long()
                noise = torch.randn(latents.shape, device=device, generator=g,
                                    dtype=latents.dtype)
                noisy = scheduler.add_noise(latents, noise, timesteps)
                target = scheduler.get_velocity(latents, noise, timesteps)
                dummy = torch.zeros((bsz,), dtype=torch.long, device=device)
                with torch.amp.autocast("cuda"):
                    out = model(hidden_states=noisy, encoder_hidden_states=text_emb,
                                timestep=timesteps, class_labels=dummy).sample
                val_loss += F.mse_loss(out.float(), target.float()).item() * bsz
                val_n += bsz
        val_loss /= max(1, val_n)

        # Restore the online weights before saving the optimizer state.
        ema_model.restore(model.parameters())

        print(f"Epoch {epoch + 1} | train {train_loss:.4f} | val(EMA) {val_loss:.4f}", flush=True)

        ck = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_sched.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_loss": best_val,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "args": vars(args),
        }
        if (epoch + 1) % 5 == 0:
            torch.save(ck, os.path.join(args.checkpoint_path, f"dit_epoch_{epoch + 1}.pt"))
        if val_loss < best_val:
            best_val = val_loss
            ck["best_val_loss"] = best_val
            torch.save(ck, os.path.join(args.checkpoint_path, "best_model.pt"))
            print(f"  new best val loss {best_val:.4f}", flush=True)


if __name__ == "__main__":
    main()