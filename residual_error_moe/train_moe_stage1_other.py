"""
train_moe_stage1c.py -- inference-safe routing via label-conditioned student
routers distilled from the residual (oracle) teacher.

  This file tests whether an inference-available signal Stage 1b lacked, the
  label, lets the student router recover the residual teacher's routing. 
  
--student_feature:
  label_only   : router reads [label_embed] broadcast to every token.  (ablation)
  label_hidden : router reads [h_l || label_embed].                     (candidate)
  static_label : no learned router
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

from data_preprocessing import LABEL_COLS, PATHOLOGY_COLS, NO_FINDING_COL
from moe_lora import inject_moe_lora, trainable_parameters
from train_base import CachedLatentDataset, PromptBank, build_dit, LATENT_SCALE
from train_moe_stage0 import atomic_save, balanced_weights, load_frozen_base, load_manifest
from train_moe_stage1_oracle import FFInputCapture, make_routing_feature


# Label to bucket static partition: head/med/tail by global
# frequency plus a healthy bucket (No Finding)
def build_label_buckets(manifest, n_buckets=4):
    """Return a function mapping label_vec[14] to a bucket id in {0..n_buckets-1}.
    Bucket 0 is healthy (No Finding dominant). Buckets 1..K-1 are head/med/tail
    by the rarest present pathology's frequency."""
    mat = manifest[LABEL_COLS].to_numpy().astype(np.float32)
    freq = np.maximum(mat.sum(0), 1.0)
    nf = LABEL_COLS.index(NO_FINDING_COL)
    path_idx = [LABEL_COLS.index(c) for c in PATHOLOGY_COLS]
    # frequency thresholds splitting pathologies into head/med/tail (3 tiers)
    path_freqs = sorted([freq[i] for i in path_idx])
    t1, t2 = np.percentile(path_freqs, [33, 66])

    def assign(label_vec):
        present = [i for i in path_idx if label_vec[i] == 1.0]
        if not present:  # no pathology, so healthy
            return 0
        rarest = min(freq[i] for i in present)
        if rarest <= t1:
            return 3  # tail
        elif rarest <= t2:
            return 2  # medium
        return 1      # head
    return assign


class LabelEmbed(nn.Module):
    """Multi-hot label vector [B,14] to a dense embedding [B, embed_dim]."""
    def __init__(self, n_labels, embed_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_labels, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, label_vec):  # [B,14] to [B,embed_dim]
        return self.proj(label_vec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_checkpoint", required=True)
    ap.add_argument("--cached_img_path", nargs="+", required=True)
    ap.add_argument("--val_cached_path", nargs="+", default=None)
    ap.add_argument("--bank_dir", required=True)
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--student_feature", required=True,
                    choices=["label_only", "label_hidden", "static_label"])
    ap.add_argument("--moe_layer", type=int, default=11)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--n_experts", type=int, default=4)
    ap.add_argument("--e_max", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=1)
    ap.add_argument("--patch", type=int, default=2)
    ap.add_argument("--label_embed_dim", type=int, default=128)
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
    ap.add_argument("--distill_beta", type=float, default=1.0)
    ap.add_argument("--router_warmup_epochs", type=int, default=1)
    ap.add_argument("--centroid_batches", type=int, default=40)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_path, exist_ok=True)
    print(f"Stage 1c | student_feature={args.student_feature} layer={args.moe_layer} "
          f"rank={args.rank} n_experts={args.n_experts}", flush=True)

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

    static_bucket = build_label_buckets(man, args.n_experts) if args.student_feature == "static_label" else None

    # models 
    base = load_frozen_base(args.base_checkpoint, device)
    moe_model = load_frozen_base(args.base_checkpoint, device)
    dim = moe_model.transformer_blocks[0].ff.net[0].proj.in_features

    # router input dim depends on the student feature
    if args.student_feature == "label_only":
        route_feat_dim = args.label_embed_dim
    elif args.student_feature == "label_hidden":
        route_feat_dim = dim + args.label_embed_dim
    else:  # static_label: router still exists but is unused
        route_feat_dim = args.label_embed_dim

    moe_model, injected = inject_moe_lora(moe_model, [args.moe_layer], dim, rank=args.rank,
                                          e_max=args.e_max, top_k=args.top_k,
                                          route_feat_dim=route_feat_dim)
    moe_layer = injected[args.moe_layer]
    moe_model.to(device)

    label_embed = LabelEmbed(len(LABEL_COLS), args.label_embed_dim).to(device) \
        if args.student_feature in ("label_only", "label_hidden") else None

    cap_base = FFInputCapture()
    base.transformer_blocks[args.moe_layer].ff.register_forward_pre_hook(cap_base)

    for e in range(args.n_experts):
        moe_layer.activate_expert(e)
    print(f"Activated {moe_layer.active_count} routed experts (+1 shared).", flush=True)

    # residual-direction centroids: the distillation target (what the true
    # residual would have routed each token to)
    centroids = None
    if args.student_feature in ("label_only", "label_hidden"):
        centroids = build_residual_centroids(base, cap_base, train_dl, scheduler,
                                             args, device)

    #  optimizer 
    router_params = list(moe_layer.router.parameters())
    if label_embed is not None:
        router_params += list(label_embed.parameters())
    router_ids = {id(p) for p in router_params}
    expert_params = [p for p in trainable_parameters(injected) if id(p) not in router_ids]
    opt = torch.optim.AdamW([
        {"params": expert_params, "lr": args.lr},
        {"params": router_params, "lr": args.lr * args.router_lr_mult},
    ], weight_decay=0.0, betas=(0.9, 0.999))
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
        res_dir = make_routing_feature(cap_base.value, v_residual, args.patch)  # [B,T,4]
        return v_target, v_base, v_residual, noisy, dummy, res_dir

    def build_student_feature(hidden, label_vec):
        """Assemble the router input for the current variant. label_vec [B,14]."""
        B, T, _ = hidden.shape
        if args.student_feature == "label_only":
            le = label_embed(label_vec)                       # [B, Le]
            return le.unsqueeze(1).expand(B, T, -1)           # broadcast to tokens
        else:  # label_hidden
            le = label_embed(label_vec).unsqueeze(1).expand(B, T, -1)
            return torch.cat([hidden, le], dim=-1)            # [B,T,dim+Le]

    best_val = float("inf")
    for epoch in range(args.epochs):
        moe_model.train()
        router_frozen = epoch < args.router_warmup_epochs
        for p in moe_layer.router.parameters():
            p.requires_grad_(not router_frozen)

        run = 0.0; seen = 0; distill_run = 0.0
        util_accum = torch.zeros(args.e_max)
        opt.zero_grad(set_to_none=True)

        for step, (latents, text_emb, labels) in enumerate(tqdm(train_dl, desc=f"E{epoch+1}")):
            latents = latents.to(device); text_emb = text_emb.to(device)
            label_vec = labels.to(device).float()             # [B,14]
            bsz = latents.shape[0]
            timesteps = torch.randint(0, 1000, (bsz,), device=device).long()
            noise = torch.randn_like(latents)

            v_target, v_base, v_residual, noisy, dummy, res_dir = base_pass(
                latents, text_emb, timesteps, noise)

            #  routing feature for stateic label 
            if args.student_feature == "static_label":
                # deterministic per-image bucket, giving a per-token constant assignment
                bucket = torch.tensor([static_bucket(label_vec[i].cpu().numpy())
                                       for i in range(bsz)], device=device)   # [B]
                moe_layer.set_hard_route(bucket)              # per-image hard top-1
                distill = torch.zeros((), device=device)
            else:
                hidden_feat = cap_base.value.float()          # [B,T,dim]
                feat = build_student_feature(hidden_feat, label_vec)
                moe_layer.set_routing_feature(feat)

            with torch.amp.autocast("cuda"):
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

                #  distillation: student router toward the residual-optimal expert 
                if args.student_feature in ("label_only", "label_hidden") and not router_frozen:
                    # target: nearest residual centroid per token (what the true
                    # residual would route to). res_dir [B,T,4], centroids [E,4].
                    with torch.no_grad():
                        sim = torch.einsum("btd,ed->bte", res_dir, centroids)  # [B,T,E]
                        target_expert = sim.argmax(-1)                          # [B,T]
                    logits = moe_layer.router(feat, moe_layer.active_mask)      # [B,T,e_max]
                    logits_active = logits[..., :args.n_experts]
                    distill = args.distill_beta * F.cross_entropy(
                        logits_active.reshape(-1, args.n_experts),
                        target_expert.reshape(-1))
                else:
                    distill = torch.zeros((), device=device)

                loss = main + guard + distill

            scaler.scale(loss / args.accumulation_steps).backward()
            run += loss.item() * bsz; seen += bsz
            distill_run += float(distill) * bsz
            util_accum += moe_layer.last_util[:args.e_max]

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
                      f"| distill {distill_run/seen:.4f} | util[{us}]"
                      f"{' (rf)' if router_frozen else ''}", flush=True)

        train_loss = run / max(1, seen)
        distill_epoch = distill_run / max(1, seen)

        #  validation: route on the same inference-available feature 
        val_loss = float("nan")
        if val_dl:
            moe_model.eval(); vrun = 0.0; vn = 0
            g = torch.Generator(device=device).manual_seed(1234)
            with torch.no_grad():
                for latents, text_emb, labels in val_dl:
                    latents = latents.to(device); text_emb = text_emb.to(device)
                    label_vec = labels.to(device).float()
                    bsz = latents.shape[0]
                    ts = torch.randint(0, 1000, (bsz,), device=device, generator=g).long()
                    noise = torch.randn(latents.shape, device=device, generator=g, dtype=latents.dtype)
                    noisy = scheduler.add_noise(latents, noise, ts)
                    vt = scheduler.get_velocity(latents, noise, ts)
                    dummy = torch.zeros((bsz,), dtype=torch.long, device=device)
                    with torch.amp.autocast("cuda"):
                        _ = base(hidden_states=noisy, encoder_hidden_states=text_emb,
                                 timestep=ts, class_labels=dummy).sample
                        if args.student_feature == "static_label":
                            bucket = torch.tensor([static_bucket(label_vec[i].cpu().numpy())
                                                   for i in range(bsz)], device=device)
                            moe_layer.set_hard_route(bucket)
                        else:
                            feat = build_student_feature(cap_base.value.float(), label_vec)
                            moe_layer.set_routing_feature(feat)
                        vm = moe_model(hidden_states=noisy, encoder_hidden_states=text_emb,
                                       timestep=ts, class_labels=dummy).sample
                    vrun += F.mse_loss(vm.float(), vt.float()).item() * bsz; vn += bsz
            val_loss = vrun / max(1, vn)

        util = (util_accum / max(1, seen // args.physical_batch_size))
        ur = {i: round(float(util[i]), 3) for i in range(args.e_max) if moe_layer.active_mask[i]}
        print(f"Epoch {epoch+1} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"| distill {distill_epoch:.4f} | util={ur}", flush=True)

        state = {k: v.cpu() for k, v in moe_model.state_dict().items()
                 if any(t in k for t in ["shared_expert", "experts.", "router", "active_mask"])}
        if label_embed is not None:
            state.update({f"label_embed.{k}": v.cpu() for k, v in label_embed.state_dict().items()})
        ck = {"epoch": epoch+1, "adapter_state": state, "val_loss": val_loss,
              "distill": distill_epoch, "student_feature": args.student_feature,
              "route_feat_dim": route_feat_dim, "args": vars(args)}
        if shutil.disk_usage(args.checkpoint_path).free / 1e9 > 5:
            atomic_save(ck, os.path.join(args.checkpoint_path, "last.pt"))
            if val_loss < best_val:
                best_val = val_loss
                atomic_save(ck, os.path.join(args.checkpoint_path, "best_adapter.pt"))
                print(f"  new best val {best_val:.4f}", flush=True)


def build_residual_centroids(base, cap_base, dl, scheduler, args, device):
    """K-means centroids of the residual direction, i.e. the distillation target
    space. A token's residual-optimal expert is its nearest centroid
    """
    print(f"[centroids] collecting residual dirs over {args.centroid_batches} batches...", flush=True)
    feats = []
    it = iter(dl)
    with torch.no_grad():
        for _ in range(args.centroid_batches):
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
    cents = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device)
    cents = F.normalize(cents, dim=-1)
    print(f"[centroids] built {args.n_experts} residual-direction centroids.", flush=True)
    return cents


if __name__ == "__main__":
    main()