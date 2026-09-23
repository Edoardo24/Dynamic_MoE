"""
train_phase0_base.py -- Phase 0: pretrain the base text-conditioned DiT backbone.

Pipeline position:
    data_preprocessing.py
        -> DiT_diffusion_model.py        caches {latent, text_embedding} .pt files
        -> train_phase0_base.py          (this file) pretrains the full DiT from scratch
        -> train_phase1.py               freezes that DiT, trains MoE-LoRA on top

train_phase1.py does not train a backbone. It calls
    get_base_dit(base_pretrained_path)  ->  freezes every backbone parameter
    ->  injects the MoE-LoRA  ->  trains only the experts.
So it needs a pretrained backbone checkpoint to load.
"""

import os
import glob
import re
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from diffusers import Transformer2DModel, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel


class CachedLatentDataset(Dataset):
    def __init__(self, cache_dir):
        self.file_paths = glob.glob(os.path.join(cache_dir, "*.pt"))
        print(f"Found {len(self.file_paths)} cached features in {cache_dir}", flush=True)
        if len(self.file_paths) == 0:
            raise FileNotFoundError(
                f"No .pt files in {cache_dir}. Did the caching step "
                f"(DiT_diffusion_model.py) run on the MULTI-LABEL CSV and write here?")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx], weights_only=True, map_location="cpu")
        return data["latent"], data["text_embedding"]


# Model
def build_base_dit():
    print("Initializing diffusers Transformer2DModel (DiT) from scratch...", flush=True)
    model = Transformer2DModel(
        sample_size=64, num_layers=12, patch_size=2, attention_head_dim=64,
        num_attention_heads=16, in_channels=4, out_channels=4,
        cross_attention_dim=4096, norm_type="ada_norm_zero",
        num_embeds_ada_norm=1, activation_fn="gelu-approximate",
    )
    n = sum(p.numel() for p in model.parameters())
    print(f"  DiT parameters (all trainable in Phase 0): {n:,}", flush=True)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached_img_path", type=str, required=True,
                        help="Directory of cached {latent, text_embedding} .pt files "
                             "(the SAME cache train_phase1 consumes).")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Output directory for base backbone checkpoints.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--physical_batch_size", type=int, default=16)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="DiT pretraining conventionally uses 0. train_phase1 used "
                             "1e-4 for the LoRA phase -- that is a different regime.")
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--lr_schedule", type=str, default="cosine",
                        choices=["cosine", "constant_with_warmup"])
    parser.add_argument("--cfg_dropout_prob", type=float, default=0.10,
                        help="Fraction of samples whose text embedding is zeroed "
                             "(unconditional) for classifier-free guidance.")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=5,
                        help="Also write dit_epoch_N.pt every N epochs (for Slurm resume).")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Phase 0 (base DiT pretraining) on: {device}", flush=True)

    # data
    full_dataset = CachedLatentDataset(args.cached_img_path)
    val_size = int(len(full_dataset) * args.val_fraction)
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
                                              generator=generator)
    print(f"  train={train_size}  val={val_size}", flush=True)

    train_dataloader = DataLoader(train_dataset, batch_size=args.physical_batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.physical_batch_size,
                                shuffle=False, num_workers=max(1, args.num_workers // 2),
                                pin_memory=True)

    # scheduler / model / optim 
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear",
                              prediction_type="v_prediction")
    model = build_base_dit().to(device)

    # Phase 0 trains the entire model: no freezing, no MoE injection
    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    ema_model = EMAModel(model.parameters(), decay=args.ema_decay,
                         model_cls=Transformer2DModel, model_config=model.config)
    ema_model.to(device)

    total_train_steps = len(train_dataloader)
    effective_optimization_steps = total_train_steps // args.accumulation_steps
    lr_scheduler = get_scheduler(
        args.lr_schedule, optimizer=optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=args.epochs * effective_optimization_steps)

    scaler = torch.amp.GradScaler("cuda")
    start_epoch = 0
    best_val_loss = float("inf")

    # Slurm auto-resume
    checkpoint_files = glob.glob(os.path.join(args.checkpoint_path, "dit_epoch_*.pt"))
    if checkpoint_files:
        latest = max(checkpoint_files,
                     key=lambda f: int(re.search(r"dit_epoch_(\d+)", f).group(1)))
        print(f"\n[Slurm Resume] Auto-detected checkpoint: {latest}", flush=True)
        ckpt = torch.load(latest, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        ema_model.load_state_dict(ckpt["ema_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "best_val_loss" in ckpt:
            best_val_loss = ckpt["best_val_loss"]
        model.to(device)
        ema_model.to(device)
        start_epoch = ckpt["epoch"]
        for _ in range(start_epoch * effective_optimization_steps):
            lr_scheduler.step()
        print(f"[Slurm Resume] Resuming from epoch {start_epoch}.", flush=True)
    else:
        print("\n[Slurm Resume] No checkpoints found. Training from scratch.\n", flush=True)

    # Train
    print("Starting Phase 0 training loop!", flush=True)
    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---", flush=True)
        model.train()
        train_epoch_loss = 0.0
        optimizer.zero_grad()

        for step, (latents, text_embeddings) in enumerate(
                tqdm(train_dataloader, desc=f"Train Epoch {epoch+1}")):
            latents = latents.to(device)
            text_embeddings = text_embeddings.to(device)
            bsz = latents.shape[0]

            # Classifier-free guidance dropout: zero the whole text embedding for a
            # fraction of samples so the model learns p(x) as well as p(x|text).
            keep = (torch.rand(bsz, device=device) > args.cfg_dropout_prob).view(bsz, 1, 1)
            uncond = torch.zeros_like(text_embeddings)
            text_embeddings = torch.where(keep, text_embeddings, uncond)

            timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                                      (bsz,), device=device).long()
            noise = torch.randn_like(latents)
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            target = scheduler.get_velocity(latents, noise, timesteps)   # v-prediction
            dummy_classes = torch.zeros((bsz,), dtype=torch.long, device=device)

            with torch.amp.autocast("cuda"):
                model_output = model(
                    hidden_states=noisy_latents,
                    encoder_hidden_states=text_embeddings,
                    timestep=timesteps, class_labels=dummy_classes,
                ).sample
                mse_loss = F.mse_loss(model_output, target)
                total_loss = mse_loss / args.accumulation_steps

            scaler.scale(total_loss).backward()
            train_epoch_loss += mse_loss.item()

            if (step + 1) % args.accumulation_steps == 0 or (step + 1) == total_train_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
                ema_model.step(model.parameters())

            if (step + 1) % (50 * args.accumulation_steps) == 0:
                cur_lr = lr_scheduler.get_last_lr()[0]
                print(f"Effective Step {(step+1)//args.accumulation_steps}/"
                      f"{effective_optimization_steps} | Train Loss: "
                      f"{mse_loss.item():.4f} | LR: {cur_lr:.2e}", flush=True)

        avg_train_loss = train_epoch_loss / total_train_steps

        # calidation: under EMA weights (store raw, copy EMA in, restore after).
        ema_model.store(model.parameters())
        ema_model.copy_to(model.parameters())
        model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for val_latents, val_text in tqdm(val_dataloader, desc=f"Val Epoch {epoch+1}"):
                val_latents = val_latents.to(device)
                val_text = val_text.to(device)
                vb = val_latents.shape[0]
                vt = torch.randint(0, scheduler.config.num_train_timesteps,
                                   (vb,), device=device).long()
                vn = torch.randn_like(val_latents)
                vnoisy = scheduler.add_noise(val_latents, vn, vt)
                vtarget = scheduler.get_velocity(val_latents, vn, vt)
                vdummy = torch.zeros((vb,), dtype=torch.long, device=device)
                with torch.amp.autocast("cuda"):
                    vout = model(hidden_states=vnoisy, encoder_hidden_states=val_text,
                                 timestep=vt, class_labels=vdummy).sample
                    vloss = F.mse_loss(vout, vtarget)
                val_epoch_loss += vloss.item()
        avg_val_loss = val_epoch_loss / max(1, len(val_dataloader))

        # Restore the raw training weights before saving / optimizing further.
        ema_model.restore(model.parameters())

        print(f"Epoch {epoch+1} Completed | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}", flush=True)

        # checkpoint
        os.makedirs(args.checkpoint_path, exist_ok=True)
        checkpoint_dict = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "base_config": {
                "sample_size": 64, "num_layers": 12, "patch_size": 2,
                "attention_head_dim": 64, "num_attention_heads": 16,
                "in_channels": 4, "out_channels": 4, "cross_attention_dim": 4096,
                "norm_type": "ada_norm_zero", "num_embeds_ada_norm": 1,
                "activation_fn": "gelu-approximate",
                "prediction_type": "v_prediction",
            },
        }
        if (epoch + 1) % args.save_every == 0:
            torch.save(checkpoint_dict,
                       os.path.join(args.checkpoint_path, f"dit_epoch_{epoch+1}.pt"))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_dict["best_val_loss"] = best_val_loss
            best_path = os.path.join(args.checkpoint_path, "best_model.pt")
            torch.save(checkpoint_dict, best_path)
            print(f"New best validation loss! Saved to {best_path}", flush=True)

    print("\nPhase 0 complete. Point train_phase1.py at:", flush=True)
    print(f"  --base_pretrained_path {os.path.join(args.checkpoint_path, 'best_model.pt')}",
          flush=True)


if __name__ == "__main__":
    main()