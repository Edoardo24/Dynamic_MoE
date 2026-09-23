"""
train_moe_stage1.py -- Stage 1: routed residual-guided experts (oracle).

Per-step pipeline (a second forward pass for the exact residual):
  1. forward the frozen base for v_base, and the layer-l hidden state via a hook.
  2. residual r = v_target - v_base -> routing feature = residual direction
  3. forward the MoE model: the router reads that feature and routes each token.
  4. loss = per-token residual-weighted MSE + guard

Stage 2 (online spawn) slots into the two marked hooks
"""

import argparse
import glob
import math
import os
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from data_preprocessing import LABEL_COLS, PATHOLOGY_COLS
from moe_lora import inject_moe_lora, trainable_parameters
from train_base import CachedLatentDataset, PromptBank, build_dit, LATENT_SCALE
from train_moe_stage0 import (
    atomic_save, balanced_weights, load_frozen_base, load_manifest,
)


# Grab the hidden state entering the MoE layer's FeedForward on the frozen base pass.
# That hidden state is the transformer's token representation at layer l, the grid the router operates on.
class FFInputCapture:
    """Forward pre-hook on block.ff that stashes its input [B, T, dim]."""
    def __init__(self):
        self.value = None

    def __call__(self, module, args):
        self.value = args[0].detach()
        return None


def make_routing_feature(h_ff_in, v_residual, patch=2):
    """
    Build the per-token routing feature: the residual direction on the token grid.

    h_ff_in    : [B, T, dim]     token grid entering the FF (T = (64/patch)^2)
    v_residual : [B, 4, 64, 64]   velocity residual in latent-pixel space

    The transformer patchifies the 64x64 latent into T tokens with patch p, so
    token t aggregates a p x p x 4 block of the residual. We pool the residual
    to the token grid, then L2-normalise to keep direction only
    """
    B, C, H, W = v_residual.shape
    gh = H // patch
    T = h_ff_in.shape[1]
    # average-pool residual to the token grid: [B, C, gh, gw]
    pooled = F.avg_pool2d(v_residual, kernel_size=patch, stride=patch)  # [B,4,32,32]
    feat = pooled.flatten(2).permute(0, 2, 1)                           # [B, gh*gw, 4]
    if feat.shape[1] != T:
        # token count mismatch
        feat = F.interpolate(pooled.flatten(2), size=T, mode="linear",
                             align_corners=False).permute(0, 2, 1)
    return F.normalize(feat, dim=-1)                                    # direction only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_checkpoint", required=True)
    ap.add_argument("--cached_img_path", nargs="+", required=True)
    ap.add_argument("--val_cached_path", nargs="+", default=None)
    ap.add_argument("--bank_dir", required=True)
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--moe_layer", type=int, default=11,
                    help="single MoE layer (late = better, per your ablation)")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--n_experts", type=int, default=4,
                    help="routed experts to activate at start (Stage 1 = fixed)")
    ap.add_argument("--e_max", type=int, default=8, help="pre-allocated slots (Stage 2 grows into these)")
    ap.add_argument("--top_k", type=int, default=1)
    ap.add_argument("--patch", type=int, default=2, help="DiT patch size (must match base)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--physical_batch_size", type=int, default=16)
    ap.add_argument("--accumulation_steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--router_lr_mult", type=float, default=0.3,
                    help="router LR = lr * this. Low so assignments don't thrash.")
    ap.add_argument("--warmup_steps", type=int, default=300)
    ap.add_argument("--balance_alpha", type=float, default=0.5)
    ap.add_argument("--finding_dropout", type=float, default=0.10)
    ap.add_argument("--cfg_dropout", type=float, default=0.10)
    ap.add_argument("--max_drop", type=int, default=2)
    # gentle residual weighting (global pow=1.0 was too hot in Stage 0)
    ap.add_argument("--residual_weight_pow", type=float, default=0.5)
    ap.add_argument("--residual_weight_clip", type=float, default=3.0,
                    help="clamp per-token weight to [1/clip, clip]")
    ap.add_argument("--guard_beta", type=float, default=0.5)
    ap.add_argument("--guard_quantile", type=float, default=0.5)
    ap.add_argument("--lb_beta", type=float, default=0.0,
                    help="load-balance aux (OFF by default; supervisor: not needed for long-tail)")
    ap.add_argument("--router_warmup_epochs", type=int, default=1,
                    help="epochs to train experts with FROZEN router before letting it learn")
    ap.add_argument("--kmeans_init", action="store_true",
                    help="init router rows from k-means centroids of residual directions")
    ap.add_argument("--kmeans_init_batches", type=int, default=40)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_path, exist_ok=True)
    print(f"Stage 1 MoE | layer={args.moe_layer} rank={args.rank} "
          f"n_experts={args.n_experts} top_k={args.top_k}", flush=True)

    # data
    man = load_manifest(args.cached_img_path)
    bank = PromptBank(args.bank_dir)
    train_ds = CachedLatentDataset(man["file"].tolist(), bank,
                                   args.finding_dropout, args.cfg_dropout,
                                   args.max_drop, train=True)
    val_ds = (CachedLatentDataset(load_manifest(args.val_cached_path)["file"].tolist(),
                                  bank, 0.0, 0.0, args.max_drop, train=False)
              if args.val_cached_path else None)

    w = balanced_weights(man, args.balance_alpha)
    sampler = WeightedRandomSampler(w, num_samples=len(man), replacement=True)
    train_dl = DataLoader(train_ds, batch_size=args.physical_batch_size, sampler=sampler,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_dl = (DataLoader(val_ds, batch_size=args.physical_batch_size, shuffle=False,
                         num_workers=4, pin_memory=True) if val_ds else None)

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear",
                              prediction_type="v_prediction")

    # frozen base (residual reference) and MoE model
    base = load_frozen_base(args.base_checkpoint, device)
    moe_model = load_frozen_base(args.base_checkpoint, device)
    dim = moe_model.transformer_blocks[0].ff.net[0].proj.in_features
    moe_model, injected = inject_moe_lora(moe_model, [args.moe_layer], dim,
                                          rank=args.rank, e_max=args.e_max,
                                          top_k=args.top_k, route_feat_dim=4)
    moe_layer = injected[args.moe_layer]
    moe_model.to(device)

    # capture the FF input on the base at the same layer (token grid for routing)
    cap = FFInputCapture()
    base.transformer_blocks[args.moe_layer].ff.register_forward_pre_hook(cap)

    # activate the initial experts (fixed count at first)
    for e in range(args.n_experts):
        moe_layer.activate_expert(e)
    print(f"Activated {moe_layer.active_count} routed experts (+1 shared).", flush=True)

    # optional: init router rows from k-means of residual directions 
    if args.kmeans_init:
        init_router_from_kmeans(base, moe_model, moe_layer, train_dl, scheduler,
                                cap, args, device)

    #  optimizer: experts and router in separate param groups
    router_params = list(moe_layer.router.parameters())
    router_ids = {id(p) for p in router_params}
    expert_params = [p for p in trainable_parameters(injected) if id(p) not in router_ids]
    opt = torch.optim.AdamW([
        {"params": expert_params, "lr": args.lr},
        {"params": router_params, "lr": args.lr * args.router_lr_mult},
    ], weight_decay=0.0, betas=(0.9, 0.999))

    n_train = sum(p.numel() for p in expert_params) + sum(p.numel() for p in router_params)
    print(f"Trainable: {n_train/1e6:.3f}M (experts {sum(p.numel() for p in expert_params)/1e6:.3f}M "
          f"+ router {sum(p.numel() for p in router_params)/1e6:.3f}M)", flush=True)

    steps_per_epoch = len(train_dl) // args.accumulation_steps
    lr_sched = get_scheduler("cosine", optimizer=opt, num_warmup_steps=args.warmup_steps,
                             num_training_steps=args.epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda")

    def base_pass(latents, text_emb, timesteps, noise):
        noisy = scheduler.add_noise(latents, noise, timesteps)
        v_target = scheduler.get_velocity(latents, noise, timesteps)
        dummy = torch.zeros((latents.shape[0],), dtype=torch.long, device=device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            v_base = base(hidden_states=noisy, encoder_hidden_states=text_emb,
                          timestep=timesteps, class_labels=dummy).sample
        v_residual = (v_target.float() - v_base.float())          # [B,4,64,64]
        route_feat = make_routing_feature(cap.value, v_residual, args.patch)  # [B,T,4]
        return v_target, v_base, v_residual, noisy, dummy, route_feat

    best_val = float("inf")
    for epoch in range(args.epochs):
        moe_model.train()
        # router warmup: freeze router for the first few epochs so experts learn before beign assign
        router_frozen = epoch < args.router_warmup_epochs
        for p in moe_layer.router.parameters():
            p.requires_grad_(not router_frozen)

        run = 0.0; seen = 0
        util_accum = torch.zeros(args.e_max)
        opt.zero_grad(set_to_none=True)

        for step, (latents, text_emb, _lbl) in enumerate(tqdm(train_dl, desc=f"E{epoch+1}")):
            latents = latents.to(device); text_emb = text_emb.to(device)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, 1000, (bsz,), device=device).long()
            noise = torch.randn_like(latents)

            v_target, v_base, v_residual, noisy, dummy, route_feat = base_pass(
                latents, text_emb, timesteps, noise)

            # hand the routing feature to the MoE layer
            moe_layer.set_routing_feature(route_feat)

            with torch.amp.autocast("cuda"):
                v_moe = moe_model(hidden_states=noisy, encoder_hidden_states=text_emb,
                                  timestep=timesteps, class_labels=dummy).sample
                per_tok = (v_moe.float() - v_target.float()).pow(2).mean(dim=1, keepdim=True)  # [B,1,64,64]
                r_norm = v_residual.pow(2).mean(dim=1, keepdim=True).detach()

                # residual weighting
                w_tok = (r_norm / (r_norm.mean() + 1e-8)) ** args.residual_weight_pow
                w_tok = w_tok.clamp(1.0 / args.residual_weight_clip, args.residual_weight_clip)
                main = (w_tok * per_tok).mean()

                if args.guard_beta > 0:
                    thr = torch.quantile(r_norm.flatten(), args.guard_quantile)
                    well = (r_norm <= thr).float()
                    base_err = (v_base.float() - v_target.float()).pow(2).mean(dim=1, keepdim=True)
                    worse = F.relu(per_tok - base_err.detach())
                    guard = args.guard_beta * (well * worse).sum() / (well.sum() + 1e-8)
                else:
                    guard = torch.zeros((), device=device)

                # optional load-balance aux (off by default)
                if args.lb_beta > 0 and moe_layer.active_count > 1:
                    util = moe_layer.last_util[:args.e_max].to(device)
                    active = moe_layer.active_mask.float()
                    frac = util * active
                    frac = frac / (frac.sum() + 1e-8)
                    target = active / active.sum()
                    lb = args.lb_beta * ((frac - target) ** 2).sum()
                else:
                    lb = torch.zeros((), device=device)

                loss = main + guard + lb

            scaler.scale(loss / args.accumulation_steps).backward()
            run += loss.item() * bsz; seen += bsz
            util_accum += moe_layer.last_util[:args.e_max]

            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(expert_params + router_params, 1.0)
                scaler.step(opt); scaler.update(); lr_sched.step()
                opt.zero_grad(set_to_none=True)

            # spawn hook A (Stage 2): check the residual-saturation trigger here,
            #      using an EMA of r_norm among routed tokens per expert

            if (step + 1) % (200 * args.accumulation_steps) == 0:
                util = (util_accum / (step + 1))
                util_str = " ".join(f"{i}:{util[i]:.2f}" for i in range(args.e_max)
                                    if moe_layer.active_mask[i])
                print(f"  step {step+1} | loss {run/seen:.4f} | main {main.item():.4f} "
                      f"| guard {float(guard):.4f} | util[{util_str}]"
                      f"{' (router frozen)' if router_frozen else ''}", flush=True)

        train_loss = run / max(1, seen)

        #  validation: plain diffusion loss with routing active 
        val_loss = float("nan")
        if val_dl:
            moe_model.eval()
            vrun = 0.0; vn = 0
            g = torch.Generator(device=device).manual_seed(1234)
            with torch.no_grad():
                for latents, text_emb, _ in val_dl:
                    latents = latents.to(device); text_emb = text_emb.to(device)
                    bsz = latents.shape[0]
                    ts = torch.randint(0, 1000, (bsz,), device=device, generator=g).long()
                    noise = torch.randn(latents.shape, device=device, generator=g, dtype=latents.dtype)
                    _, _, _, noisy, dummy, route_feat = base_pass(latents, text_emb, ts, noise)
                    moe_layer.set_routing_feature(route_feat)
                    vt = scheduler.get_velocity(latents, noise, ts)
                    with torch.amp.autocast("cuda"):
                        vm = moe_model(hidden_states=noisy, encoder_hidden_states=text_emb,
                                       timestep=ts, class_labels=dummy).sample
                    vrun += F.mse_loss(vm.float(), vt.float()).item() * bsz; vn += bsz
            val_loss = vrun / max(1, vn)

        util = (util_accum / max(1, seen // args.physical_batch_size))
        util_report = {i: round(float(util[i]), 3) for i in range(args.e_max)
                       if moe_layer.active_mask[i]}
        print(f"Epoch {epoch+1} | train {train_loss:.4f} | val(plain) {val_loss:.4f} "
              f"| active={moe_layer.active_count} util={util_report}", flush=True)

        # spawn hook B (Stage 2): decide whether to activate a new expert at the
        #      epoch boundary (freeze-then-grow)

        adapter_state = {k: v.cpu() for k, v in moe_model.state_dict().items()
                         if any(t in k for t in ["shared_expert", "experts.", "router", "active_mask"])}
        ck = {"epoch": epoch+1, "adapter_state": adapter_state, "val_loss": val_loss,
              "args": vars(args), "moe_layer": args.moe_layer,
              "active_count": moe_layer.active_count}
        if shutil.disk_usage(args.checkpoint_path).free / 1e9 > 5:
            atomic_save(ck, os.path.join(args.checkpoint_path, "last.pt"))
            if val_loss < best_val:
                best_val = val_loss
                atomic_save(ck, os.path.join(args.checkpoint_path, "best_adapter.pt"))
                print(f"  new best val {best_val:.4f}", flush=True)


def init_router_from_kmeans(base, moe_model, moe_layer, dl, scheduler, cap, args, device):
    """Collect residual directions over a few batches, k-means to n_experts
    clusters, and point each active router row at a cluster centroid. Gives the
    router a warm, non-random start aligned to real residual structure."""
    print(f"[kmeans-init] collecting residual dirs over {args.kmeans_init_batches} batches...", flush=True)
    feats = []
    it = iter(dl)
    with torch.no_grad():
        for _ in range(args.kmeans_init_batches):
            try:
                latents, text_emb, _ = next(it)
            except StopIteration:
                break
            latents = latents.to(device); text_emb = text_emb.to(device)
            bsz = latents.shape[0]
            ts = torch.randint(0, 1000, (bsz,), device=device).long()
            noise = torch.randn_like(latents)
            noisy = scheduler.add_noise(latents, noise, ts)
            v_target = scheduler.get_velocity(latents, noise, ts)
            dummy = torch.zeros((bsz,), dtype=torch.long, device=device)
            with torch.amp.autocast("cuda"):
                v_base = base(hidden_states=noisy, encoder_hidden_states=text_emb,
                              timestep=ts, class_labels=dummy).sample
            rf = make_routing_feature(cap.value, (v_target.float() - v_base.float()), args.patch)
            feats.append(rf.reshape(-1, rf.shape[-1]).cpu())
    F_all = torch.cat(feats).numpy()
    km = KMeans(n_clusters=args.n_experts, n_init=10, random_state=0).fit(F_all)
    cents = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device)
    for e in range(args.n_experts):
        moe_layer.router.set_expert_row(e, F.normalize(cents[e], dim=0))
    print(f"[kmeans-init] router rows set from {args.n_experts} centroids.", flush=True)


if __name__ == "__main__":
    main()