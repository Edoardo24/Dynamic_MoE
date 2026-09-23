import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from diffusers.training_utils import EMAModel
from diffusers import Transformer2DModel, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKL as _AutoencoderKL
from transformers import AutoModel as _AutoModel, AutoImageProcessor as _AutoImageProcessor
from tqdm import tqdm
import re
import argparse
import diffusers

from moe_architecture import inject_moe_into_dit
from moe_orchestrator import MoEBufferManager, train_new_expert_offline
from moe_telemetry import (print_telemetry_report, sync_ema_shadow_for_expert,
                           sync_ema_shadow_for_router, echo_run_manifest)


@torch.no_grad()
def compute_poorly_served_mask(model, moe_module, latents, text_embeddings,
                               scheduler, device, ref_timestep, grid_h, grid_w,
                               keep_fraction=0.5):
    """
    Expert-performance filter: creates a mask deciding which tokens get buffered as
    spawn candidates because the current ensemble predicts them poorly. Computed with
    one extra forward at a fixed timestep, which keeps the error score comparable
    across batches. An adaptive threshold is used instead of a fixed hyperparameter.

    Returns a [B, T] bool mask marking the tokens the current ensemble predicts worst
    (top `keep_fraction` by per-token velocity error); these get buffered, the rest
    are dropped.
    """
    bsz = latents.shape[0]
    T = grid_h * grid_w

    # fixed reference timestep for every sample in the batch
    timesteps = torch.full((bsz,), int(ref_timestep), device=device, dtype=torch.long)
    noise = torch.randn_like(latents)
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    target = scheduler.get_velocity(latents, noise, timesteps)
    dummy_classes = torch.zeros((bsz,), dtype=torch.long, device=device)

    # ensure this scoring pass does mot itself trigger capture/filtering
    saved_psm = moe_module.latest_poorly_served_mask
    moe_module.latest_poorly_served_mask = None
    was_training = moe_module.training
    moe_module.eval()   # deterministic routing (no gumbel noise) for a clean score

    with torch.amp.autocast('cuda'):
        out = model(hidden_states=noisy_latents, encoder_hidden_states=text_embeddings,
                    timestep=timesteps, class_labels=dummy_classes).sample

    if was_training:
        moe_module.train()
    moe_module.latest_poorly_served_mask = saved_psm

    # per-token error: mean squared error over the 2x2 latent patch + channels.
    # out/target are [B, C, H, W] = [B, 4, 64, 64]; fold to per-token [B, T].
    err = (out.float() - target.float()) ** 2                     # [B,4,64,64]
    err = err.mean(dim=1)                                         # [B,64,64]
    err = err.view(bsz, grid_h, 2, grid_w, 2).mean(dim=(2, 4))    # [B,32,32] per token
    err = err.reshape(bsz, T)                                     # [B, T]

    # adaptive cutoff: keep the worst `keep_fraction` of tokens across the batch.
    # Use >= so ties at the quantile don't collapse the selection to near
    # zero when many tokens share similar error (guard against an empty mask).
    flat = err.reshape(-1)
    q = torch.quantile(flat, 1.0 - keep_fraction)
    poorly_served = err >= q                                      # [B, T] bool
    # Safety: if numerical ties made this select almost nothing, fall back to the
    # top-k tokens by error so the buffer is never fully starved.
    min_keep = max(1, int(keep_fraction * flat.numel() * 0.5))
    if int(poorly_served.sum()) < min_keep:
        k = max(1, int(keep_fraction * flat.numel()))
        thresh = torch.topk(flat, k, largest=True).values.min()
        poorly_served = err >= thresh
    return poorly_served


@torch.no_grad()
def run_validation(model, moe_module, val_dataloader, scheduler, device,
                   grid_h=32, grid_w=32, tail_fraction=0.10, seed=1234):
    """
    deterministic validation: here every epoch sees the SAME (timestep, noise) per batch, via a
    per-batch seeded generator, so epoch-to-epoch deltas are real.

    Returns (aggregate_mse, tail_mse) where tail_mse is the mean per-token error
    over the worst `tail_fraction` of tokens in each batch (a proxy for the
    long-tail objective that the aggregate MSE drowns out).
    """
    was_training = moe_module.training
    model.eval()
    moe_module.latest_poorly_served_mask = None
    T = grid_h * grid_w
    agg_sum, agg_n = 0.0, 0
    tail_sum, tail_n = 0.0, 0

    for b_idx, (val_latents, val_text) in enumerate(val_dataloader):
        val_latents = val_latents.to(device)
        val_text = val_text.to(device)
        bsz = val_latents.shape[0]
        g = torch.Generator(device=device).manual_seed(seed + b_idx)
        val_timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                                      (bsz,), device=device, generator=g).long()
        val_noise = torch.randn(val_latents.shape, device=device, generator=g,
                                dtype=val_latents.dtype)
        val_noisy = scheduler.add_noise(val_latents, val_noise, val_timesteps)
        val_target = scheduler.get_velocity(val_latents, val_noise, val_timesteps)
        val_classes = torch.zeros((bsz,), dtype=torch.long, device=device)

        with torch.amp.autocast('cuda'):
            out = model(hidden_states=val_noisy, encoder_hidden_states=val_text,
                        timestep=val_timesteps, class_labels=val_classes).sample

        err = (out.float() - val_target.float()) ** 2          # [B,C,H,W]
        agg_sum += err.mean().item() * bsz
        agg_n += bsz

        # per-token error -> worst-decile mean (long-tail proxy)
        per_tok = err.mean(dim=1)                               # [B,H,W]
        per_tok = per_tok.view(bsz, grid_h, 2, grid_w, 2).mean(dim=(2, 4))  # [B,32,32]
        flat = per_tok.reshape(-1)
        k = max(1, int(tail_fraction * flat.numel()))
        tail_sum += torch.topk(flat, k, largest=True).values.mean().item() * bsz
        tail_n += bsz

    if was_training:
        model.train()
    return agg_sum / max(1, agg_n), tail_sum / max(1, tail_n)


# same as train_phase0_base.py
class CachedLatentDataset(Dataset):
    def __init__(self, cache_dir):
        self.file_paths = glob.glob(os.path.join(cache_dir, "*.pt"))
        print(f"Found {len(self.file_paths)} cached features in {cache_dir}", flush=True)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx], weights_only=True, map_location="cpu")
        return data["latent"], data["text_embedding"]


def get_base_dit(pretrained_ckpt_path=None, sanity_check_layer=6):
    print("Initializing diffusers Transformer2DModel (DiT)...", flush=True)
    model = Transformer2DModel(
        sample_size=64, num_layers=12, patch_size=2, attention_head_dim=64,
        num_attention_heads=16, in_channels=4, out_channels=4,
        cross_attention_dim=4096, norm_type="ada_norm_zero",
        num_embeds_ada_norm=1, activation_fn="gelu-approximate",
    )

    # Loading pretrained base model
    if pretrained_ckpt_path and os.path.exists(pretrained_ckpt_path):
        print(f"Loading PRE-TRAINED weights from {pretrained_ckpt_path}...", flush=True)
        checkpoint = torch.load(pretrained_ckpt_path, map_location="cpu", weights_only=True)

        if "ema_state_dict" in checkpoint:
            print(" -> Restoring EMA (smoothed) weights via EMAModel.copy_to()")
            ema = EMAModel(model.parameters(), model_cls=Transformer2DModel,
                           model_config=model.config)
            ema.load_state_dict(checkpoint["ema_state_dict"])
            ema.copy_to(model.parameters())
        elif "model_state_dict" in checkpoint:
            print(" -> Restoring raw model_state_dict")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)

        # Sanity check on the layer that will actually receive the MoE injection
        w = model.transformer_blocks[sanity_check_layer].ff.net[0].proj.weight
        print(f" -> Layer-{sanity_check_layer} FFN weight std after load: "
              f"{w.std().item():.5f}", flush=True)
    else:
        print("[WARNING] No pre-trained checkpoint found. Random init.", flush=True)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached_img_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--base_pretrained_path", type=str, required=False, default=None)
    parser.add_argument("--layer_index", type=int, required=False, default=6)
    parser.add_argument("--max_experts", type=int, required=False, default=16)
    parser.add_argument("--rank", type=int, required=False, default=16)
    parser.add_argument("--entropy_threshold", type=float, required=False, default=1.0)
    # spawn_threshold is counted in tokens, not images.
    parser.add_argument("--spawn_threshold", type=int, required=False, default=40000)
    parser.add_argument("--epochs", type=int, required=False, default=50)
    parser.add_argument("--physical_batch_size", type=int, required=False, default=16)
    parser.add_argument("--accumulation_steps", type=int, required=False, default=4)
    parser.add_argument("--spawn_start_epoch", type=int, required=False, default=2)
    # Only attemp a spawn every N epochs (time reduction).
    parser.add_argument("--spawn_every", type=int, required=False, default=5)
    parser.add_argument("--top_k", type=int, default=2)
    # per-token clustering knobs
    parser.add_argument("--offline_batch_size", type=int, default=32)
    parser.add_argument("--min_tokens_per_cluster", type=int, default=2000)
    parser.add_argument("--min_contributing_images", type=int, default=30)
    parser.add_argument("--min_tokens_per_image", type=int, default=1)
    parser.add_argument("--min_silhouette", type=float, default=0.1)
    parser.add_argument("--redundancy_cos_threshold", type=float, default=0.9)
    # Aim the router at a freshly spawned expert's cluster
    parser.add_argument("--warm_start_router", action="store_true", default=True,
                        help="Point the new router row at the cluster on spawn (default on).")
    parser.add_argument("--no_warm_start_router", dest="warm_start_router",
                        action="store_false")
    parser.add_argument("--warm_start_strength", type=float, default=1.0)
    parser.add_argument("--select_metric", type=str, default="tail",
                        choices=["agg", "tail"])
    # impact experiment knobs
    parser.add_argument("--expert_volume", type=float, default=4.0)
    parser.add_argument("--offline_lr", type=float, default=3e-4)
    parser.add_argument("--offline_epochs", type=int, default=8)
    # experience performance filter knobs (see above)
    parser.add_argument("--use_perf_filter", action="store_true")
    parser.add_argument("--filter_keep_fraction", type=float, default=0.5)
    parser.add_argument("--filter_ref_timestep", type=int, default=500)
    parser.add_argument("--filter_min_experts", type=int, default=3)

    parser.add_argument("--router_mode", type=str, required=True,
                        choices=["skip", "no_skip", "shared"])
    #  spawn clustering feature space (supervisor's DINO/CLIP suggestion) 
    parser.add_argument("--cluster_method", type=str, default="kmeans",
                        choices=["kmeans", "hdbscan"])
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=250)
    parser.add_argument("--cluster_feature", type=str, default="internal",
                        choices=["internal", "dino", "clip"])
    parser.add_argument("--cluster_encoder_model", type=str, default="microsoft/rad-dino")
    parser.add_argument("--cluster_granularity", type=str, default="sample",
                        choices=["sample", "patch"])
    parser.add_argument("--cluster_patch_size", type=int, default=14)
    parser.add_argument("--cluster_vae_path", type=str, default=None)
    parser.add_argument("--skip_mode", type=str, default="compete", choices=["compete", "gate"])
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Phase 1 Training on: {device}", flush=True)

    full_dataset = CachedLatentDataset(args.cached_img_path)
    val_size = int(len(full_dataset) * 0.05)
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
                                              generator=generator)

    train_dataloader = DataLoader(train_dataset, batch_size=args.physical_batch_size,
                                  shuffle=True, num_workers=8, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.physical_batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear",
                              prediction_type="v_prediction")
    model = get_base_dit(args.base_pretrained_path,
                         sanity_check_layer=args.layer_index).to(device)

    print("Injecting Dynamic MoE-LoRA...", flush=True)
    # forward the arm, then prove it arrived.
    model = inject_moe_into_dit(
        model, layer_idx=args.layer_index, max_experts=args.max_experts,
        rank=args.rank, top_k=args.top_k, expert_volume=args.expert_volume,
        skip_mode=args.skip_mode,
        router_mode=args.router_mode,
    )
    moe_module = model.transformer_blocks[args.layer_index].ff.moe
    echo_run_manifest(args, moe_module)
    model.to(device)

    # freeze the backbone
    print("Freezing base DiT backbone...", flush=True)
    for param in model.parameters():
        param.requires_grad = False

    for param in moe_module.parameters():
        param.requires_grad = True

    # base_up/base_down are the DiT's original tensors, reused by the MoE
    moe_module.base_up.requires_grad_(False)
    moe_module.base_down.requires_grad_(False)
    # The shared expert is inert in 'skip'/'no_skip'; keep it frozen there so it
    # never enters the optimizer and never drifts.
    if args.router_mode != "shared":
        moe_module.shared_up.requires_grad_(False)
        moe_module.shared_down.requires_grad_(False)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_train:,}", flush=True)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)


    buffer_manager = MoEBufferManager(spawn_threshold=args.spawn_threshold,
                                      grid_h=32, grid_w=32,
                                      cluster_feature=args.cluster_feature,
                                      cluster_method=args.cluster_method,
                                      hdbscan_min_cluster_size=args.hdbscan_min_cluster_size)

    # Opt-in: attach a frozen DINO/CLIP encoder (+ VAE decoder) for spawn clustering.
    # Default ('internal') skips all of this and behaves exactly as before.
    if args.cluster_feature != "internal":
        if args.cluster_vae_path is None:
            raise ValueError("--cluster_vae_path is required when --cluster_feature != internal "
                             "(needed to decode buffered latents back to images).")
        print(f"[cluster] loading external encoder '{args.cluster_encoder_model}' + decode VAE...",
              flush=True)
        _vae = _AutoencoderKL.from_pretrained(args.cluster_vae_path).to(device).eval()
        _vae.requires_grad_(False)
        _enc = _AutoModel.from_pretrained(args.cluster_encoder_model).to(device).eval()
        _enc.requires_grad_(False)
        _proc = _AutoImageProcessor.from_pretrained(args.cluster_encoder_model)

        @torch.no_grad()
        def _encoder_fn(pixel_values):
            out = _enc(pixel_values=pixel_values)
            pooled = getattr(out, "pooler_output", None)
            return pooled if pooled is not None else out.last_hidden_state[:, 0]

        @torch.no_grad()
        def _encoder_patch_fn(pixel_values):
            # return all tokens; the buffer slices the trailing patch tokens
            return _enc(pixel_values=pixel_values).last_hidden_state

        _mean = getattr(_proc, "image_mean", [0.485, 0.456, 0.406])
        _std = getattr(_proc, "image_std", [0.229, 0.224, 0.225])
        _size = getattr(_proc, "size", {"height": 224, "width": 224})
        if isinstance(_size, dict):
            _res = _size.get("height", _size.get("shortest_edge", 224))
        else:
            _res = int(_size)
        # rad-dino's processor defaults to 518, which is ~5x the ViT work per image
        # for no benefit on this clustering. Cap at 224 (divisible by patch size 14).
        _res = min(int(_res), 224)
        buffer_manager.attach_external_encoder(
            _vae, _encoder_fn, _mean, _std, input_res=_res, name=args.cluster_feature,
            encoder_patch_fn=_encoder_patch_fn, patch_size=args.cluster_patch_size,
            granularity=args.cluster_granularity)
        print(f"[cluster] external spawn-clustering ACTIVE "
              f"({args.cluster_feature}/{args.cluster_granularity}, res={_res}).", flush=True)
    ema_model = EMAModel(model.parameters(), decay=0.9999,
                         model_cls=Transformer2DModel, model_config=model.config)
    ema_model.to(device)

    if hasattr(model, "set_use_memory_efficient_attention_xformers"):
        diffusers.utils.import_utils._pytorch_version = torch.__version__
        print("Relying on PyTorch 2.0+ SDPA for Memory Efficient Attention.", flush=True)

    total_train_steps = len(train_dataloader)
    effective_optimization_steps = total_train_steps // args.accumulation_steps

    lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=100,
                                 num_training_steps=args.epochs * effective_optimization_steps)

    start_epoch = 0
    best_val_loss = float('inf')
    scaler = torch.amp.GradScaler('cuda')

    checkpoint_files = glob.glob(os.path.join(args.checkpoint_path, "dit_epoch_*.pt"))
    if checkpoint_files:
        latest_checkpoint_path = max(
            checkpoint_files, key=lambda f: int(re.search(r'dit_epoch_(\d+)', f).group(1)))
        print(f"\n[Slurm Resume] Auto-detected latest checkpoint: "
              f"{latest_checkpoint_path}", flush=True)
        checkpoint = torch.load(latest_checkpoint_path, map_location='cpu', weights_only=True)

        # The three arms share state_dict shape (no conflict if shared, skip or no_skip are selected)
        ckpt_cfg = checkpoint.get('moe_config')
        if ckpt_cfg is not None:
            live_cfg = moe_module.config_summary()
            for key in ("router_mode", "skip_mode", "top_k", "arch_rev"):
                if str(ckpt_cfg.get(key)) != str(live_cfg.get(key)):
                    raise RuntimeError(
                        f"Checkpoint/arg mismatch on '{key}': checkpoint has "
                        f"{ckpt_cfg.get(key)!r}, this run wants {live_cfg.get(key)!r}. "
                        f"The arms share tensor shapes, so this would load silently "
                        f"and produce a run that is neither arm.")
            print(f"[Slurm Resume] moe_config matches: {ckpt_cfg}", flush=True)
        else:
            print("[Slurm Resume][WARN] checkpoint has no moe_config -- cannot verify "
                  "the arm matches. Pre-dates the config guard.", flush=True)

        model.load_state_dict(checkpoint['model_state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        model.to(device)
        ema_model.to(device)
        start_epoch = checkpoint['epoch']
        for _ in range(start_epoch * effective_optimization_steps):
            lr_scheduler.step()
    else:
        print("\n[Slurm Resume] No checkpoints found. Starting training from scratch.\n",
              flush=True)

    print("Starting Training Loop!", flush=True)


    # At init every expert's B close to 0 and the shared expert's
    # so moe_update==0 and shared_delta==0: the injected model is very close to 
    # the base model. One deterministic validation pass here is therefore
    # the exact base reference under the identical protocol

    base_agg = base_tail = None
    if start_epoch == 0:
        base_agg, base_tail = run_validation(
            model, moe_module, val_dataloader, scheduler, device,
            grid_h=buffer_manager.grid_h, grid_w=buffer_manager.grid_w)
        print(f"[BASELINE] base model (injected, untrained) | "
              f"agg val MSE = {base_agg:.5f} | tail(worst-10%) MSE = {base_tail:.5f}",
              flush=True)
        best_val_loss = base_tail if args.select_metric == "tail" else base_agg

    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---", flush=True)

        # add Gumbel temperature
        moe_module.current_gumbel_temp = max(0.3, 1.0 - (epoch / 30.0))
        # moe_module.current_gumbel_temp = 0.1

        # Attempt a spawn only on scheduled epochs (>= start, every spawn_every)
        do_spawn_this_epoch = (epoch >= args.spawn_start_epoch
                               and (epoch - args.spawn_start_epoch) % args.spawn_every == 0)

        if epoch < args.spawn_start_epoch:
            print(f"[Router Warmup] Spawning paused until Epoch "
                  f"{args.spawn_start_epoch + 1}.", flush=True)
        else:
            print(f"[Curriculum] Gumbel Routing Active. Temp: "
                  f"{moe_module.current_gumbel_temp:.3f}", flush=True)

        model.train()
        moe_module.reset_epoch_telemetry()
        train_epoch_loss = 0.0
        optimizer.zero_grad()

        for step, (latents, text_embeddings) in enumerate(
                tqdm(train_dataloader, desc=f"Train Epoch {epoch+1}")):
            latents, text_embeddings = latents.to(device), text_embeddings.to(device)
            bsz = latents.shape[0]

            # keep the true conditional embeddings for the scoring pass before
            # CFG dropout is applied below, otherwise ~10% of samples would be
            # scored with zeroed conditioning
            cond_text_embeddings = text_embeddings

            mask = (torch.rand(bsz, device=device) > 0.10).view(bsz, 1, 1)
            uncond_embedding = torch.zeros_like(text_embeddings)
            text_embeddings = torch.where(mask, text_embeddings, uncond_embedding)

            timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                                      (bsz,), device=device).long()
            noise = torch.randn_like(latents)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            target = scheduler.get_velocity(latents, noise, timesteps)
            dummy_classes = torch.zeros((bsz,), dtype=torch.long, device=device)

            # expert performance filter
            active_count_pre = int(moe_module.active_mask.sum().item())
            if (do_spawn_this_epoch
                    and active_count_pre >= args.filter_min_experts
                    and args.use_perf_filter):
                moe_module.latest_poorly_served_mask = compute_poorly_served_mask(
                    model, moe_module, latents, cond_text_embeddings, scheduler, device,
                    ref_timestep=args.filter_ref_timestep,
                    grid_h=buffer_manager.grid_h, grid_w=buffer_manager.grid_w,
                    keep_fraction=args.filter_keep_fraction)
            else:
                moe_module.latest_poorly_served_mask = None

            with torch.amp.autocast('cuda'):
                model_output = model(
                    hidden_states=noisy_latents, encoder_hidden_states=text_embeddings,
                    timestep=timesteps, class_labels=dummy_classes
                ).sample

                mse_loss = F.mse_loss(model_output, target)

                balance_loss = moe_module.latest_balance_loss
                if balance_loss is None:
                    balance_loss = torch.tensor(0.0, device=device)
                # router_loss = 0.0 * balance_loss
                router_loss = 1e-2 * balance_loss

                total_loss = (mse_loss + router_loss) / args.accumulation_steps

            scaler.scale(total_loss).backward()
            train_epoch_loss += (mse_loss.item() * args.accumulation_steps)

            if (step + 1) % args.accumulation_steps == 0 or (step + 1) == total_train_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
                ema_model.step(model.parameters())

            if (int(moe_module.active_mask.sum().item()) < moe_module.max_experts
                    and do_spawn_this_epoch):
                # one per-step call. The MoE already captured each unsure token's content-space
                # representation + (batch_row, flat_position) during the forward.
                # add_step pulls those out, deduplicates contributing images, and appends one row per unsure token.
                buffer_manager.add_step(latents, text_embeddings, moe_module)

            if (step + 1) % (50 * args.accumulation_steps) == 0:
                current_lr = lr_scheduler.get_last_lr()[0]
                print(f"Effective Step {(step+1)//args.accumulation_steps}/"
                      f"{effective_optimization_steps} | Train Loss: "
                      f"{(mse_loss.item() * args.accumulation_steps):.4f} | "
                      f"LR: {current_lr:.2e}", flush=True)

        avg_train_loss = train_epoch_loss / total_train_steps

        # validation
        ema_model.store(model.parameters())
        ema_model.copy_to(model.parameters())

        model.eval()
        # Clear the perf-filter mask before validation 
        moe_module.latest_poorly_served_mask = None
        avg_val_loss, tail_val_loss = run_validation(
            model, moe_module, val_dataloader, scheduler, device,
            grid_h=buffer_manager.grid_h, grid_w=buffer_manager.grid_w)

        base_note = ""
        if base_agg is not None:
            base_note = (f" | vs base: agg {avg_val_loss - base_agg:+.5f}, "
                         f"tail {tail_val_loss - base_tail:+.5f}")
        print(f"Epoch {epoch+1} Completed | Train Loss: {avg_train_loss:.4f} | "
              f"Val MSE: {avg_val_loss:.5f} | Tail(worst-10%) MSE: "
              f"{tail_val_loss:.5f}{base_note}", flush=True)

        # Restore the raw training weights before telemetry, spawning, and checkpointing after copy.
        ema_model.restore(model.parameters())

        # telemetry
        print_telemetry_report(moe_module, epoch)

        # end of epoch spawning eevnt
        active_count = int(moe_module.active_mask.sum().item())
        if active_count < moe_module.max_experts and do_spawn_this_epoch:
            if buffer_manager.is_ready_to_spawn():
                new_expert_idx = active_count
                train_new_expert_offline(
                    model=model, moe_module=moe_module, buffer_manager=buffer_manager,
                    new_expert_idx=new_expert_idx, scheduler=scheduler, device=device,
                    epochs=args.offline_epochs,
                    offline_batch_size=args.offline_batch_size,
                    offline_lr=args.offline_lr,
                    min_tokens_per_cluster=args.min_tokens_per_cluster,
                    min_contributing_images=args.min_contributing_images,
                    min_tokens_per_image=args.min_tokens_per_image,
                    min_silhouette=args.min_silhouette,
                    redundancy_cos_threshold=args.redundancy_cos_threshold,
                    warm_start_router=args.warm_start_router,
                    warm_start_strength=args.warm_start_strength,
                )
                # Only sync if the spawn actually happened (an abort leaves
                # active_mask untouched, so compare against the pre-spawn count).
                if int(moe_module.active_mask.sum().item()) > active_count:
                    n_sync = sync_ema_shadow_for_expert(
                        ema_model, model, moe_module, new_expert_idx)
                    # Warm-start edited the router row in place
                    n_r = sync_ema_shadow_for_router(ema_model, model, moe_module)
                    print(f"[EMA] Synced {n_sync} expert + {n_r} router shadow "
                          f"tensors for spawned Expert {new_expert_idx}. Without "
                          f"this, validation would run it dead for ~10k steps "
                          f"(decay=0.9999).", flush=True)
            else:
                print(f"[Curriculum] Not enough unsure tokens gathered this epoch to "
                      f"spawn (Buffer: {len(buffer_manager.tok_feat)}).", flush=True)
                buffer_manager.clear()   # start fresh next epoch

        os.makedirs(args.checkpoint_path, exist_ok=True)

        checkpoint_dict = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_loss': best_val_loss,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_tail_loss': tail_val_loss,
            'base_val_loss': base_agg,
            'base_tail_loss': base_tail,
            'moe_config': {**moe_module.config_summary(),
                           'layer_index': args.layer_index},
        }
        if (epoch + 1) % 2 == 0:
            torch.save(checkpoint_dict,
                       os.path.join(args.checkpoint_path, f"dit_epoch_{epoch+1}.pt"))

        # Select on the metric that matches the objective (tail)
        current_metric = tail_val_loss if args.select_metric == "tail" else avg_val_loss
        if current_metric < best_val_loss:
            best_val_loss = current_metric
            checkpoint_dict['best_val_loss'] = best_val_loss
            best_ckpt_path = os.path.join(args.checkpoint_path, "best_model.pt")
            torch.save(checkpoint_dict, best_ckpt_path)
            beat = ""
            if base_agg is not None:
                ref = base_tail if args.select_metric == "tail" else base_agg
                beat = f" (beats base by {ref - current_metric:+.5f} on {args.select_metric})"
            print(f"New best {args.select_metric} val loss = {current_metric:.5f}! "
                  f"Saved to {best_ckpt_path}{beat}", flush=True)


if __name__ == "__main__":
    main()