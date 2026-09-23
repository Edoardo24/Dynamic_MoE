"""
train_moe_stage0.py -- Stage 0: frozen base + one shared LoRA expert, no routing.

does residual-weighted LoRA adapter help the balanced long-tail objective at all, before any routing
is introduced?

Two ideas tested here in isolation from routing:
  1. Residual-weighted loss: weight each token's diffusion loss by how wrong the
     frozen base is on it
  2. No-regression guard: penalise the MoE only when it makes a token the base
     already handled well worse
"""

import argparse
import glob
import math
import os
import random
import re
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, Transformer2DModel
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from data_preprocessing import LABEL_COLS, PATHOLOGY_COLS, tokens_from_labels
from moe_lora import inject_moe_lora, trainable_parameters

# Reuse the base trainer class
from train_base import CachedLatentDataset, PromptBank, build_dit, LATENT_SCALE


def load_manifest(cache_dirs):
    if isinstance(cache_dirs, str):
        cache_dirs = [cache_dirs]
    frames = []
    for d in cache_dirs:
        for man in sorted(glob.glob(os.path.join(d, "manifest_*.csv"))):
            sub = pd.read_csv(man)
            sub["file"] = sub["file"].apply(lambda f: os.path.join(d, f))
            frames.append(sub)
    df = pd.concat(frames, ignore_index=True).drop_duplicates("path").reset_index(drop=True)
    df = df[df["file"].apply(os.path.exists)].reset_index(drop=True)
    print(f"Manifest: {len(df)} latents from {len(cache_dirs)} dir(s).", flush=True)
    return df


def balanced_weights(df, alpha=0.5):
    mat = df[LABEL_COLS].to_numpy().astype(np.float32)
    freq = np.maximum(mat.sum(0), 1.0)
    nf = LABEL_COLS.index("No Finding")
    npath = len(PATHOLOGY_COLS)
    w = np.empty(len(df))
    for i, row in enumerate(mat):
        present = [j for j in range(npath) if row[j] == 1.0]
        rarest = min(freq[j] for j in present) if present else freq[nf]
        w[i] = (1.0 / rarest) ** alpha
    return torch.from_numpy(w / w.mean())


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_frozen_base(checkpoint, device):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = ck.get("ema_state_dict", None)
    model = build_dit().to(device)
    # EMA weights if present 
    sd = ck.get("model_state_dict", ck)
    model.load_state_dict(sd, strict=False)
    if "ema_state_dict" in ck:
        try:
            from diffusers.training_utils import EMAModel
            ema = EMAModel(model.parameters(), model_cls=Transformer2DModel,
                           model_config=model.config)
            ema.load_state_dict(ck["ema_state_dict"])
            ema.copy_to(model.parameters())
            print("  base: applied EMA weights", flush=True)
        except Exception as e:
            print(f"  base: EMA load failed ({e}); using raw weights", flush=True)
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_checkpoint", required=True)
    ap.add_argument("--cached_img_path", nargs="+", required=True)
    ap.add_argument("--val_cached_path", nargs="+", default=None)
    ap.add_argument("--bank_dir", required=True)
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--moe_layers", nargs="+", type=int, default=[11],
                    help="which transformer blocks get the adapter (late = better)")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--physical_batch_size", type=int, default=16)
    ap.add_argument("--accumulation_steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=300)
    ap.add_argument("--balance_alpha", type=float, default=0.5)
    ap.add_argument("--finding_dropout", type=float, default=0.10)
    ap.add_argument("--cfg_dropout", type=float, default=0.10)
    ap.add_argument("--max_drop", type=int, default=2)
    ap.add_argument("--residual_weight_pow", type=float, default=1.0,
                    help="w_i = (||r_i|| / mean||r||) ** this. 0 = uniform (ablation).")
    ap.add_argument("--guard_beta", type=float, default=0.5,
                    help="no-regression guard strength. 0 = off (ablation).")
    ap.add_argument("--guard_quantile", type=float, default=0.5,
                    help="tokens below this residual quantile are 'well-served' -> guarded.")
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_path, exist_ok=True)
    print(f"Stage 0 MoE on {device} | layers={args.moe_layers} rank={args.rank}", flush=True)

    # data
    man = load_manifest(args.cached_img_path)
    bank = PromptBank(args.bank_dir)
    train_ds = CachedLatentDataset(man["file"].tolist(), bank,
                                   args.finding_dropout, args.cfg_dropout,
                                   args.max_drop, train=True)
    if args.val_cached_path:
        vman = load_manifest(args.val_cached_path)
        val_ds = CachedLatentDataset(vman["file"].tolist(), bank, 0.0, 0.0,
                                     args.max_drop, train=False)
    else:
        val_ds = None

    w = balanced_weights(man, args.balance_alpha)
    sampler = WeightedRandomSampler(w, num_samples=len(man), replacement=True)
    train_dl = DataLoader(train_ds, batch_size=args.physical_batch_size, sampler=sampler,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_dl = (DataLoader(val_ds, batch_size=args.physical_batch_size, shuffle=False,
                         num_workers=4, pin_memory=True) if val_ds else None)

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear",
                              prediction_type="v_prediction")

    # frozen base and a second copy that will host the MoE adapters
    base = load_frozen_base(args.base_checkpoint, device)          # residual reference
    moe_model = load_frozen_base(args.base_checkpoint, device)     # same weights, will host adapters
    dim = moe_model.config.cross_attention_dim if False else moe_model.transformer_blocks[0].ff.net[0].proj.in_features
    moe_model, injected = inject_moe_lora(moe_model, args.moe_layers, dim,
                                          rank=args.rank, e_max=8, top_k=1)
    # Stage 0: no routed experts activated, so only the shared expert trains.
    moe_model.to(device)

    params = list(trainable_parameters(injected))
    n_train = sum(p.numel() for p in params)
    print(f"Trainable adapter params: {n_train/1e6:.3f}M "
          f"(base frozen: {sum(p.numel() for p in base.parameters())/1e6:.1f}M)", flush=True)

    # LoRA params get no weight decay (decay drives B toward 0).
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))
    steps_per_epoch = len(train_dl) // args.accumulation_steps
    lr_sched = get_scheduler("cosine", optimizer=opt,
                             num_warmup_steps=args.warmup_steps,
                             num_training_steps=args.epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda")

    def velocity_and_residual(latents, text_emb, timesteps, noise):
        """Return (v_target, base_residual_per_token_norm, noisy, dummy)."""
        noisy = scheduler.add_noise(latents, noise, timesteps)
        v_target = scheduler.get_velocity(latents, noise, timesteps)
        dummy = torch.zeros((latents.shape[0],), dtype=torch.long, device=device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            v_base = base(hidden_states=noisy, encoder_hidden_states=text_emb,
                          timestep=timesteps, class_labels=dummy).sample
        # per-token (per spatial location) residual magnitude, averaged over channels
        r = (v_target - v_base)                      # [B,4,64,64]
        r_norm = r.pow(2).mean(dim=1, keepdim=True)  # [B,1,64,64]
        return v_target, v_base, r_norm, noisy, dummy

    best_val = float("inf")
    for epoch in range(args.epochs):
        moe_model.train()
        run = 0.0; seen = 0
        opt.zero_grad(set_to_none=True)
        for step, (latents, text_emb, _lbl) in enumerate(tqdm(train_dl, desc=f"E{epoch+1}")):
            latents = latents.to(device); text_emb = text_emb.to(device)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, 1000, (bsz,), device=device).long()
            noise = torch.randn_like(latents)

            v_target, v_base, r_norm, noisy, dummy = velocity_and_residual(
                latents, text_emb, timesteps, noise)

            with torch.amp.autocast("cuda"):
                v_moe = moe_model(hidden_states=noisy, encoder_hidden_states=text_emb,
                                  timestep=timesteps, class_labels=dummy).sample
                per_tok = (v_moe.float() - v_target.float()).pow(2).mean(dim=1, keepdim=True)  # [B,1,64,64]

                # residual-weighted main loss
                rw = r_norm.detach()
                w_tok = (rw / (rw.mean() + 1e-8)) ** args.residual_weight_pow
                main = (w_tok * per_tok).mean()

                # no-regression guard on well-served (low-residual) tokens
                if args.guard_beta > 0:
                    thr = torch.quantile(rw.flatten(), args.guard_quantile)
                    well = (rw <= thr).float()
                    base_err = (v_base.float() - v_target.float()).pow(2).mean(dim=1, keepdim=True)
                    worse = F.relu(per_tok - base_err.detach())      # >0 only if MoE worse
                    guard = args.guard_beta * (well * worse).sum() / (well.sum() + 1e-8)
                else:
                    guard = torch.zeros((), device=device)

                loss = main + guard

            scaler.scale(loss / args.accumulation_steps).backward()
            run += loss.item() * bsz; seen += bsz

            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); lr_sched.step()
                opt.zero_grad(set_to_none=True)

            if (step + 1) % (200 * args.accumulation_steps) == 0:
                print(f"  step {step+1} | loss {run/seen:.4f} | main {main.item():.4f} "
                      f"| guard {float(guard):.4f}", flush=True)

        train_loss = run / max(1, seen)

        # validation: plain (unweighted) diffusion loss, comparable to base
        val_loss = float("nan")
        if val_dl:
            moe_model.eval(); vrun = 0.0; vn = 0
            g = torch.Generator(device=device).manual_seed(1234)
            with torch.no_grad():
                for latents, text_emb, _ in val_dl:
                    latents = latents.to(device); text_emb = text_emb.to(device)
                    bsz = latents.shape[0]
                    ts = torch.randint(0, 1000, (bsz,), device=device, generator=g).long()
                    noise = torch.randn(latents.shape, device=device, generator=g, dtype=latents.dtype)
                    noisy = scheduler.add_noise(latents, noise, ts)
                    vt = scheduler.get_velocity(latents, noise, ts)
                    dummy = torch.zeros((bsz,), dtype=torch.long, device=device)
                    with torch.amp.autocast("cuda"):
                        vm = moe_model(hidden_states=noisy, encoder_hidden_states=text_emb,
                                       timestep=ts, class_labels=dummy).sample
                    vrun += F.mse_loss(vm.float(), vt.float()).item() * bsz; vn += bsz
            val_loss = vrun / max(1, vn)

        print(f"Epoch {epoch+1} | train(weighted) {train_loss:.4f} | val(plain) {val_loss:.4f}", flush=True)

        # save adapters only (base is frozen and known)
        adapter_state = {k: v.cpu() for k, v in moe_model.state_dict().items()
                         if any(t in k for t in ["shared_expert", "experts.", "router", "active_mask"])}
        ck = {"epoch": epoch+1, "adapter_state": adapter_state,
              "val_loss": val_loss, "args": vars(args), "moe_layers": args.moe_layers}
        if shutil.disk_usage(args.checkpoint_path).free / 1e9 > 5:
            atomic_save(ck, os.path.join(args.checkpoint_path, "last.pt"))
            if val_loss < best_val:
                best_val = val_loss
                atomic_save(ck, os.path.join(args.checkpoint_path, "best_adapter.pt"))
                print(f"  new best val {best_val:.4f}", flush=True)


if __name__ == "__main__":
    main() 