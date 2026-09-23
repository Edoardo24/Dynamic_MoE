"""
train_moe_controls.py -- controls for the hidden-state distillation study.

Same setup as Stage 1_hidden. The one addition is --teacher_target, which
swaps what the router is distilled toward, so the students in the routing study
can be bracketed by a capability ceiling and a chance floor.

The residual is only used to build the distillation target. Routing at inference
reads the hidden state, exactly as in Stage 1_hidden
"""

import argparse
import os
import shutil

import numpy as np
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
from train_moe_stage0 import atomic_save, balanced_weights, load_frozen_base, load_manifest
from train_moe_stage1_oracle import FFInputCapture, make_routing_feature


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
    ap.add_argument("--epochs", type=int, default=40)
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
    ap.add_argument("--distill_beta", type=float, default=1.0,
                    help="weight on the router distillation (hidden-state -> residual-optimal)")
    ap.add_argument("--router_warmup_epochs", type=int, default=1)
    ap.add_argument("--kmeans_clusters_batches", type=int, default=40)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--teacher_target",
                    choices=["residual", "label", "shuffle", "content_cluster"],
                    default="residual",
                    help="residual=real teacher; label=coarse positive; "
                         "shuffle=chance floor; content_cluster=clean positive")
    ap.add_argument("--log_agreement", action="store_true")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_path, exist_ok=True)
    print(f"Stage 1b (hidden-routed) | layer={args.moe_layer} rank={args.rank} "
          f"n_experts={args.n_experts}", flush=True)

    # data 
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

    base = load_frozen_base(args.base_checkpoint, device)
    moe_model = load_frozen_base(args.base_checkpoint, device)
    dim = moe_model.transformer_blocks[0].ff.net[0].proj.in_features

    # the router reads the full hidden state here, so route_feat_dim = dim
    moe_model, injected = inject_moe_lora(moe_model, [args.moe_layer], dim,
                                          rank=args.rank, e_max=args.e_max,
                                          top_k=args.top_k, route_feat_dim=dim)
    moe_layer = injected[args.moe_layer]
    moe_model.to(device)

    # capture the layer-l hidden state on the frozen baseand use it for both
    # as the router input and (pooled) as the residual-direction target grid.
    cap_base = FFInputCapture()
    base.transformer_blocks[args.moe_layer].ff.register_forward_pre_hook(cap_base)

    for e in range(args.n_experts):
        moe_layer.activate_expert(e)
    print(f"Activated {moe_layer.active_count} routed experts (+1 shared).", flush=True)

    # build residual-direction cluster centroids (the supervision target) 
    # That is the same as routing Stage 1 made with the true residual, now used only to supervise
    centroids = build_residual_centroids(base, cap_base, train_dl, scheduler,
                                         args, device)  # [n_experts, 4]
    centroids = centroids.to(device)

    # content_cluster target (clean positive control)
    content_centroids = None
    if args.teacher_target == "content_cluster":
        content_centroids = build_hidden_centroids(base, cap_base, train_dl,
                                                    scheduler, args, device).to(device)

    router_params = list(moe_layer.router.parameters())
    router_ids = {id(p) for p in router_params}
    expert_params = [p for p in trainable_parameters(injected) if id(p) not in router_ids]
    opt = torch.optim.AdamW([
        {"params": expert_params, "lr": args.lr},
        {"params": router_params, "lr": args.lr * args.router_lr_mult},
    ], weight_decay=0.0, betas=(0.9, 0.999))
    print(f"Trainable: experts {sum(p.numel() for p in expert_params)/1e6:.3f}M "
          f"+ router {sum(p.numel() for p in router_params)/1e6:.3f}M "
          f"(router in_dim={dim})", flush=True)

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
        v_residual = (v_target.float() - v_base.float())
        # residual-direction target grid [B,T,4] (for the distillation target)
        res_dir = make_routing_feature(cap_base.value, v_residual, args.patch)
        return v_target, v_base, v_residual, noisy, dummy, res_dir

    best_val = float("inf")
    for epoch in range(args.epochs):
        moe_model.train()
        router_frozen = epoch < args.router_warmup_epochs
        for p in moe_layer.router.parameters():
            p.requires_grad_(not router_frozen)

        run = 0.0; seen = 0
        util_accum = torch.zeros(args.e_max)
        distill_acc = 0.0
        opt.zero_grad(set_to_none=True)

        for step, (latents, text_emb, lbl) in enumerate(tqdm(train_dl, desc=f"E{epoch+1}")):
            lbl = lbl.to(device)
            latents = latents.to(device); text_emb = text_emb.to(device)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, 1000, (bsz,), device=device).long()
            noise = torch.randn_like(latents)

            v_target, v_base, v_residual, noisy, dummy, res_dir = base_pass(
                latents, text_emb, timesteps, noise)

            # residual-optimal expert target: nearest centroid to each token's
            # residual direction (cosine, since both normalized).  [B,T]
            with torch.no_grad():
                sim = torch.einsum("btd,ed->bte", res_dir, F.normalize(centroids, dim=-1))
                residual_target = sim.argmax(-1)  # [B,T] residual-optimal
                B, T = residual_target.shape
                if args.teacher_target == "residual":
                    target_expert = residual_target
                elif args.teacher_target == "label":
                    # coarse positive: per-image label bucket (imbalanced)
                    idx = lbl.float().argmax(dim=1)
                    img_bucket = (idx % args.n_experts).long()
                    target_expert = img_bucket.view(B, 1).expand(B, T).contiguous()
                elif args.teacher_target == "content_cluster":
                    # clean positive: nearest hidden-state cluster (balanced, and a
                    # function of the router input by construction).
                    hf = cap_base.value.float()                      # [B,T,dim]
                    hn = F.normalize(hf, dim=-1)
                    cc = F.normalize(content_centroids, dim=-1)      # [n_experts,dim]
                    csim = torch.einsum("btd,ed->bte", hn, cc)
                    target_expert = csim.argmax(-1)                  # [B,T]
                else:  # shuffle
                    perm = torch.randperm(B * T, device=device)
                    target_expert = residual_target.reshape(-1)[perm].reshape(B, T)

            with torch.amp.autocast("cuda"):
                # The router reads the layer-l hidden state, available at inference. 
                # It is produced by layers 0..l-1, which are frozen and adapter-free,
                # so it is identical in the base pass and the MoE pass
                hidden_feat = cap_base.value.float()            # [B,T,dim] inference-available
                moe_layer.set_routing_feature(hidden_feat)
                v_moe = moe_model(hidden_states=noisy, encoder_hidden_states=text_emb,
                                  timestep=timesteps, class_labels=dummy).sample

                per_tok = (v_moe.float() - v_target.float()).pow(2).mean(dim=1, keepdim=True)
                r_norm = v_residual.pow(2).mean(dim=1, keepdim=True).detach()
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

                # distillation: hidden-state router logits toward the target assignment
                if not router_frozen:
                    logits = moe_layer.router(hidden_feat, moe_layer.active_mask)  # [B,T,e_max]
                    logits_active = logits[..., :args.n_experts]
                    distill = args.distill_beta * F.cross_entropy(
                        logits_active.reshape(-1, args.n_experts),
                        target_expert.reshape(-1))
                    if args.log_agreement and (step % 400 == 0):
                        with torch.no_grad():
                            pred = logits_active.reshape(-1, args.n_experts).argmax(-1)
                            tgt = target_expert.reshape(-1)
                            agree = (pred == tgt).float().mean().item()
                            p = torch.bincount(tgt, minlength=args.n_experts).float()
                            p = p / p.sum()
                            H = -(p * (p + 1e-9).log()).sum().item()
                            print(f"    [agree {agree:.3f} | H(target)={H:.4f} "
                                  f"| ln(n)={float(np.log(args.n_experts)):.4f}]", flush=True)
                else:
                    distill = torch.zeros((), device=device)

                loss = main + guard + distill

            scaler.scale(loss / args.accumulation_steps).backward()
            run += loss.item() * bsz; seen += bsz
            util_accum += moe_layer.last_util[:args.e_max]
            distill_acc += float(distill)

            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(expert_params + router_params, 1.0)
                scaler.step(opt); scaler.update(); lr_sched.step()
                opt.zero_grad(set_to_none=True)

            if (step + 1) % (200 * args.accumulation_steps) == 0:
                util = (util_accum / (step + 1))
                us = " ".join(f"{i}:{util[i]:.2f}" for i in range(args.e_max)
                              if moe_layer.active_mask[i])
                print(f"  step {step+1} | loss {run/seen:.4f} | main {main.item():.4f} "
                      f"| guard {float(guard):.4f} | distill {distill_acc/(step+1):.4f} "
                      f"| util[{us}]{' (rf)' if router_frozen else ''}", flush=True)

        train_loss = run / max(1, seen)

        #  validation: route on the hidden state (inference-available)
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
                        # base pass populates cap_base with the layer-l hidden state
                        _ = base(hidden_states=noisy, encoder_hidden_states=text_emb,
                                 timestep=ts, class_labels=dummy).sample
                        moe_layer.set_routing_feature(cap_base.value.float())
                        vm = moe_model(hidden_states=noisy, encoder_hidden_states=text_emb,
                                       timestep=ts, class_labels=dummy).sample
                    vrun += F.mse_loss(vm.float(), vt.float()).item() * bsz; vn += bsz
            val_loss = vrun / max(1, vn)

        util = (util_accum / max(1, seen // args.physical_batch_size))
        ur = {i: round(float(util[i]), 3) for i in range(args.e_max) if moe_layer.active_mask[i]}
        print(f"Epoch {epoch+1} | train {train_loss:.4f} | val(hidden-routed) {val_loss:.4f} "
              f"| distill {distill_acc/max(1,len(train_dl)):.4f} | util={ur}", flush=True)

        adapter_state = {k: v.cpu() for k, v in moe_model.state_dict().items()
                         if any(t in k for t in ["shared_expert", "experts.", "router", "active_mask"])}
        ck = {"epoch": epoch+1, "adapter_state": adapter_state, "val_loss": val_loss,
              "args": vars(args), "moe_layer": args.moe_layer,
              "route_feat_dim": dim, "routing": "hidden",
              "active_count": moe_layer.active_count}
        if shutil.disk_usage(args.checkpoint_path).free / 1e9 > 5:
            atomic_save(ck, os.path.join(args.checkpoint_path, "last.pt"))
            if val_loss < best_val:
                best_val = val_loss
                atomic_save(ck, os.path.join(args.checkpoint_path, "best_adapter.pt"))
                print(f"  new best val {best_val:.4f}", flush=True)


def build_hidden_centroids(base, cap_base, dl, scheduler, args, device):
    """K-means centroids of the hidden state h_l: balanced content clusters,
    the clean-positive distillation target
    """
    print(f"[content] collecting hidden states over {args.kmeans_clusters_batches} batches...", flush=True)
    feats = []
    it = iter(dl)
    with torch.no_grad():
        for _ in range(args.kmeans_clusters_batches):
            try:
                latents, text_emb, _ = next(it)
            except StopIteration:
                break
            latents = latents.to(device); text_emb = text_emb.to(device)
            bsz = latents.shape[0]
            ts = torch.randint(0, 1000, (bsz,), device=device).long()
            noise = torch.randn_like(latents)
            noisy = scheduler.add_noise(latents, noise, ts)
            dummy = torch.zeros((bsz,), dtype=torch.long, device=device)
            with torch.amp.autocast("cuda"):
                _ = base(hidden_states=noisy, encoder_hidden_states=text_emb,
                         timestep=ts, class_labels=dummy).sample
            h = cap_base.value.float()                    # [B,T,dim]
            feats.append(h.reshape(-1, h.shape[-1]).cpu())
    F_all = torch.cat(feats).numpy()
    # subsample for k-means speed (dim can be 1024)
    if F_all.shape[0] > 60000:
        idx = np.random.default_rng(0).choice(F_all.shape[0], 60000, replace=False)
        F_all = F_all[idx]
    km = KMeans(n_clusters=args.n_experts, n_init=10, random_state=0).fit(F_all)
    print(f"[content] built {args.n_experts} hidden-state centroids.", flush=True)
    return torch.tensor(km.cluster_centers_, dtype=torch.float32)


def build_residual_centroids(base, cap_base, dl, scheduler, args, device):
    """K-means centroids of the true residual directions. These define the
    residual optimal assignment used to supervise the hidden state router."""
    print(f"[centroids] collecting residual dirs over {args.kmeans_clusters_batches} batches...", flush=True)
    feats = []
    it = iter(dl)
    with torch.no_grad():
        for _ in range(args.kmeans_clusters_batches):
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
            rf = make_routing_feature(cap_base.value, (v_target.float() - v_base.float()), args.patch)
            feats.append(rf.reshape(-1, rf.shape[-1]).cpu())
    F_all = torch.cat(feats).numpy()
    km = KMeans(n_clusters=args.n_experts, n_init=10, random_state=0).fit(F_all)
    print(f"[centroids] built {args.n_experts} residual-direction centroids.", flush=True)
    return torch.tensor(km.cluster_centers_, dtype=torch.float32)


if __name__ == "__main__":
    main()