"""
train_moe_optionA.py -- Predicted residual: route on a learned, inference-available
residual prediction.

Predicted residual focus on a different quantity. The residual is r = v_target - v_base.
At inference a small head regresses the residual direction from inference available inputs [hidden, v_base, t], 
and the router runs on that predicted residual.    
  - teacher target: true residual direction rdir = normalize(pool(v_target - v_base)).
  - predictor P: an MLP on inference features [h_l || v_base_pooled || temb]
    giving r_hat on the same token grid as rdir, trained with cosine to rdir.
    Inference-valid.
  - router: reads r_hat (the prediction), not the true residual
"""
import argparse
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from moe_lora import inject_moe_lora, trainable_parameters
from train_base import CachedLatentDataset, PromptBank, build_dit, LATENT_SCALE
from train_moe_stage0 import atomic_save, balanced_weights, load_frozen_base, load_manifest
from train_moe_stage1 import FFInputCapture, make_routing_feature


class ResidualPredictor(nn.Module):
    """
    Predict the residual direction per token from inference-available inputs
    [hidden_state || pooled v_base || timestep embedding]. The output lives on
    the same [B,T,route_dim] grid as the true residual direction and is what the
    router reads, so at inference we run this (all inputs available) instead of
    the oracle residual.
    """
    def __init__(self, hidden_dim, vbase_dim, temb_dim, out_dim, width=512):
        super().__init__()
        in_dim = hidden_dim + vbase_dim + temb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, out_dim),
        )

    def forward(self, hidden, vbase_tok, temb_tok):
        x = torch.cat([hidden, vbase_tok, temb_tok], dim=-1)
        return F.normalize(self.net(x), dim=-1)


def timestep_embedding(timesteps, dim):
    """Standard sinusoidal timestep embedding, [B] to [B, dim]."""
    half = dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(half, device=timesteps.device) / half)
    a = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(a), torch.sin(a)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb  # [B, dim]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_checkpoint", required=True)
    ap.add_argument("--cached_img_path", nargs="+", required=True)
    ap.add_argument("--val_cached_path", nargs="+", default=None)
    ap.add_argument("--bank_dir", required=True)
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--moe_layer", type=int, default=11)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--n_experts", type=int, default=4)
    ap.add_argument("--e_max", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=1)
    ap.add_argument("--patch", type=int, default=2)
    ap.add_argument("--route_dim", type=int, default=4,
                    help="dim of the (pooled) residual-direction routing space")
    ap.add_argument("--temb_dim", type=int, default=128)
    ap.add_argument("--pred_width", type=int, default=512)
    ap.add_argument("--pred_beta", type=float, default=1.0,
                    help="weight on the residual-prediction (teacher) loss")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--physical_batch_size", type=int, default=16)
    ap.add_argument("--accumulation_steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--router_lr_mult", type=float, default=0.5)
    ap.add_argument("--warmup_steps", type=int, default=300)
    ap.add_argument("--balance_alpha", type=float, default=0.5)
    ap.add_argument("--finding_dropout", type=float, default=0.10)
    ap.add_argument("--cfg_dropout", type=float, default=0.10)
    ap.add_argument("--max_drop", type=int, default=2)
    ap.add_argument("--residual_weight_pow", type=float, default=0.5)
    ap.add_argument("--residual_weight_clip", type=float, default=3.0)
    ap.add_argument("--guard_beta", type=float, default=0.5)
    ap.add_argument("--guard_quantile", type=float, default=0.5)
    ap.add_argument("--router_warmup_epochs", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_path, exist_ok=True)
    print(f"Option A (predicted-residual routing) | layer={args.moe_layer} "
          f"rank={args.rank} n_experts={args.n_experts}", flush=True)

    #  data 
    man = load_manifest(args.cached_img_path)
    bank = PromptBank(args.bank_dir)
    train_ds = CachedLatentDataset(man["file"].tolist(), bank, args.finding_dropout,
                                   args.cfg_dropout, args.max_drop, train=True)
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

    #  models 
    base = load_frozen_base(args.base_checkpoint, device)
    moe_model = load_frozen_base(args.base_checkpoint, device)
    dim = moe_model.transformer_blocks[0].ff.net[0].proj.in_features
    # the router reads the predicted residual direction, so route_feat_dim = route_dim
    moe_model, injected = inject_moe_lora(moe_model, [args.moe_layer], dim,
                                          rank=args.rank, e_max=args.e_max,
                                          top_k=args.top_k, route_feat_dim=args.route_dim)
    moe_layer = injected[args.moe_layer]
    moe_model.to(device)

    for e in range(args.n_experts):
        moe_layer.activate_expert(e)
    print(f"Activated {moe_layer.active_count} routed experts (+1 shared).", flush=True)

    # capture the layer-l hidden state on the frozen base (inference-available)
    cap_base = FFInputCapture()
    base.transformer_blocks[args.moe_layer].ff.register_forward_pre_hook(cap_base)

    # v_base is [B,4,64,64] and pool it to the token grid the router uses (32x32),
    # matching make_routing_feature's pooling. We pool v_base to [B,T,4] channels.
    vbase_dim = 4  # the 4 latent channels, pooled per token

    predictor = ResidualPredictor(hidden_dim=dim, vbase_dim=vbase_dim,
                                  temb_dim=args.temb_dim, out_dim=args.route_dim,
                                  width=args.pred_width).to(device)

    #  optimizer: experts + router + predictor 
    router_params = list(moe_layer.router.parameters())
    pred_params = list(predictor.parameters())
    special = {id(p) for p in router_params + pred_params}
    expert_params = [p for p in trainable_parameters(injected) if id(p) not in special]
    opt = torch.optim.AdamW([
        {"params": expert_params, "lr": args.lr},
        {"params": router_params, "lr": args.lr * args.router_lr_mult},
        {"params": pred_params, "lr": args.lr},
    ], weight_decay=0.0, betas=(0.9, 0.999))
    steps_per_epoch = len(train_dl) // args.accumulation_steps
    lr_sched = get_scheduler("cosine", optimizer=opt, num_warmup_steps=args.warmup_steps,
                             num_training_steps=args.epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda")

    def pool_to_tokens(x_bchw):
        """[B,C,64,64] to [B,T,C] on the 32x32=T grid (avg 2x2), matching
        make_routing_feature's pooling so the predictor output aligns with rdir."""
        pooled = F.avg_pool2d(x_bchw, kernel_size=args.patch, stride=args.patch)  # [B,C,32,32]
        return pooled.flatten(2).permute(0, 2, 1)  # [B,T,C]

    def base_pass(latents, text_emb, timesteps, noise):
        noisy = scheduler.add_noise(latents, noise, timesteps)
        v_target = scheduler.get_velocity(latents, noise, timesteps)
        dummy = torch.zeros((latents.shape[0],), dtype=torch.long, device=device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            v_base = base(hidden_states=noisy, encoder_hidden_states=text_emb,
                          timestep=timesteps, class_labels=dummy).sample
        v_residual = (v_target.float() - v_base.float())
        # true residual direction target grid [B,T,route_dim]
        rdir = make_routing_feature(cap_base.value, v_residual, args.patch)  # [B,T,route_dim]
        return v_target, v_base, v_residual, noisy, dummy, rdir

    best_val = float("inf")
    for epoch in range(args.epochs):
        moe_model.train(); predictor.train()
        router_frozen = epoch < args.router_warmup_epochs
        for p in moe_layer.router.parameters():
            p.requires_grad_(not router_frozen)

        run = 0.0; seen = 0; pred_cos_run = 0.0
        opt.zero_grad(set_to_none=True)

        for step, (latents, text_emb, _lbl) in enumerate(tqdm(train_dl, desc=f"E{epoch+1}")):
            latents = latents.to(device); text_emb = text_emb.to(device)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, 1000, (bsz,), device=device).long()
            noise = torch.randn_like(latents)
            v_target, v_base, v_residual, noisy, dummy, rdir = base_pass(
                latents, text_emb, timesteps, noise)

            hidden = cap_base.value.float()                    # [B,T,dim] inference-available
            vbase_tok = pool_to_tokens(v_base.float())         # [B,T,4]  inference-available
            temb = timestep_embedding(timesteps, args.temb_dim)  # [B,temb_dim]
            temb_tok = temb.unsqueeze(1).expand(-1, hidden.shape[1], -1)  # [B,T,temb_dim]

            with torch.amp.autocast("cuda"):
                # predict the residual direction from inference available inputs
                r_hat = predictor(hidden, vbase_tok, temb_tok)  # [B,T,route_dim]
                # route on the prediction (not the oracle residual)
                moe_layer.set_routing_feature(r_hat)
                v_moe = moe_model(hidden_states=noisy, encoder_hidden_states=text_emb,
                                  timestep=timesteps, class_labels=dummy).sample

                per_tok = (v_moe.float() - v_target.float()).pow(2).mean(dim=1, keepdim=True)
                rw = v_residual.pow(2).mean(dim=1, keepdim=True).detach()
                w_tok = (rw / (rw.mean() + 1e-8)) ** args.residual_weight_pow
                w_tok = w_tok.clamp(1.0 / args.residual_weight_clip, args.residual_weight_clip)
                main = (w_tok * per_tok).mean()

                if args.guard_beta > 0:
                    thr = torch.quantile(rw.flatten(), args.guard_quantile)
                    well = (rw <= thr).float()
                    base_err = (v_base.float() - v_target.float()).pow(2).mean(dim=1, keepdim=True)
                    worse = F.relu(per_tok - base_err.detach())
                    guard = args.guard_beta * (well * worse).sum() / (well.sum() + 1e-8)
                else:
                    guard = torch.zeros((), device=device)

                # teacher loss: pull the predicted residual direction toward the true one
                cos = (r_hat * rdir.detach()).sum(-1)          # [B,T] cosine per token
                pred_loss = args.pred_beta * (1.0 - cos).mean()

                loss = main + guard + pred_loss

            scaler.scale(loss / args.accumulation_steps).backward()
            run += loss.item() * bsz; seen += bsz
            pred_cos_run += cos.mean().item() * bsz

            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(expert_params + router_params + pred_params, 1.0)
                scaler.step(opt); scaler.update(); lr_sched.step()
                opt.zero_grad(set_to_none=True)

            if (step + 1) % (200 * args.accumulation_steps) == 0:
                us = " ".join(f"{i}:{moe_layer.last_util[i]:.2f}" for i in range(args.e_max)
                              if moe_layer.active_mask[i])
                print(f"  step {step+1} | loss {run/seen:.4f} | main {main.item():.4f} "
                      f"| pred_cos {pred_cos_run/seen:.3f} | util[{us}]"
                      f"{' (rf)' if router_frozen else ''}", flush=True)

        train_loss = run / max(1, seen)
        pred_cos_epoch = pred_cos_run / max(1, seen)

        #  validation: route on the predicted residual (inference path) 
        val_loss = float("nan"); val_ent = float("nan")
        if val_dl:
            moe_model.eval(); predictor.eval(); vrun = 0.0; vn = 0; ent_run = 0.0
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
                        # base pass to fill hidden + v_base (both inference-available)
                        vb = base(hidden_states=noisy, encoder_hidden_states=text_emb,
                                  timestep=ts, class_labels=dummy).sample
                        hidden = cap_base.value.float()
                        vbase_tok = pool_to_tokens(vb.float())
                        temb = timestep_embedding(ts, args.temb_dim)
                        temb_tok = temb.unsqueeze(1).expand(-1, hidden.shape[1], -1)
                        r_hat = predictor(hidden, vbase_tok, temb_tok)
                        moe_layer.set_routing_feature(r_hat)
                        vm = moe_model(hidden_states=noisy, encoder_hidden_states=text_emb,
                                       timestep=ts, class_labels=dummy).sample
                        # routing entropy (decisive vs blind)
                        logits = moe_layer.router(r_hat, moe_layer.active_mask)
                        probs = F.softmax(logits.float(), dim=-1).clamp_min(1e-9)
                        ent = (-(probs * probs.log()).sum(-1)).mean().item()
                    vrun += F.mse_loss(vm.float(), vt.float()).item() * bsz; vn += bsz
                    ent_run += ent * bsz
            val_loss = vrun / max(1, vn)
            val_ent = ent_run / max(1, vn)

        uniform_ent = float(np.log(max(1, moe_layer.active_count)))
        print(f"Epoch {epoch+1} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"| pred_cos {pred_cos_epoch:.3f} | val_route_entropy {val_ent:.3f} "
              f"(uniform {uniform_ent:.3f})", flush=True)

        state = {k: v.cpu() for k, v in moe_model.state_dict().items()
                 if any(t in k for t in ["shared_expert", "experts.", "router", "active_mask"])}
        state.update({f"predictor.{k}": v.cpu() for k, v in predictor.state_dict().items()})
        ck = {"epoch": epoch+1, "adapter_state": state, "val_loss": val_loss,
              "pred_cos": pred_cos_epoch, "val_route_entropy": val_ent,
              "mode": "optionA_predicted_residual", "args": vars(args)}
        if shutil.disk_usage(args.checkpoint_path).free / 1e9 > 5:
            atomic_save(ck, os.path.join(args.checkpoint_path, "last.pt"))
            if val_loss < best_val:
                best_val = val_loss
                atomic_save(ck, os.path.join(args.checkpoint_path, "best_adapter.pt"))
                print(f"  new best val {best_val:.4f}", flush=True)


if __name__ == "__main__":
    main()