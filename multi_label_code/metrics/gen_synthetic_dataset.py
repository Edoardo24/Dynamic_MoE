"""
gen_synthetic_dataset.py -- generate one self-contained synthetic H5 per model.

--per_class single-positive images for each of the 13 labels, so every
class (especially the rare ones) reaches the same count
"""

import os
import re
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import h5py

from diffusers import AutoencoderKL, Transformer2DModel, DPMSolverMultistepScheduler
from diffusers.training_utils import EMAModel
from transformers import T5Tokenizer, T5EncoderModel
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from moe_architecture import inject_moe_into_dit
from data_preprocessing import LABEL_COLS


def prompt_for_label(name):
    if name == "No Finding":
        return "A normal chest radiograph with no acute cardiopulmonary findings."
    return f"A chest radiograph demonstrating findings consistent with {name.lower()}."


def make_base_dit(device):
    return Transformer2DModel(
        sample_size=64, num_layers=12, patch_size=2, attention_head_dim=64,
        num_attention_heads=16, in_channels=4, out_channels=4,
        cross_attention_dim=4096, norm_type="ada_norm_zero",
        num_embeds_ada_norm=1, activation_fn="gelu-approximate").to(device)


def build_model(state_dict, moe_config, args, device):
    model = make_base_dit(device)
    moe_layers = set()
    for k in state_dict:
        m = re.search(r"transformer_blocks\.(\d+)\.ff\.moe\.", k)
        if m:
            moe_layers.add(int(m.group(1)))
    if not moe_layers:
        return model, False
    cfg = moe_config or {}
    router_mode = cfg.get("router_mode", args.router_mode)
    if router_mode is None:
        raise RuntimeError("Checkpoint has no moe_config['router_mode']; pass --router_mode.")
    for li in moe_layers:
        model = inject_moe_into_dit(
            model, layer_idx=li, max_experts=int(cfg.get("max_experts", args.max_experts)),
            rank=args.rank, top_k=int(cfg.get("top_k", args.top_k)),
            expert_volume=float(cfg.get("expert_volume", 1.0)),
            skip_mode=cfg.get("skip_mode", "compete"), router_mode=router_mode).to(device)
    return model, True


@torch.no_grad()
def generate_batch(model, prompts, tokenizer, text_encoder, vae, scheduler, args, device, seed):
    """Returns [B, S, S] uint8 grayscale in [0,255]."""
    bsz = len(prompts)
    scheduler.set_timesteps(args.num_inference_steps, device=device)
    with torch.amp.autocast("cuda"):
        ti = tokenizer(prompts, padding="max_length", max_length=77, truncation=True,
                       return_tensors="pt").to(device)
        ui = tokenizer([""] * bsz, padding="max_length", max_length=77, truncation=True,
                       return_tensors="pt").to(device)
        cond = text_encoder(ti.input_ids)[0].to(torch.float16)
        uncond = text_encoder(ui.input_ids)[0].to(torch.float16)
        text_emb = torch.cat([uncond, cond])

    dummy = torch.zeros((bsz * 2,), dtype=torch.long, device=device)
    g = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn((bsz, 4, 64, 64), generator=g, dtype=torch.float16, device=device)

    for t in scheduler.timesteps:
        inp = scheduler.scale_model_input(torch.cat([latents] * 2), t)
        with torch.amp.autocast("cuda"):
            pred = model(hidden_states=inp, encoder_hidden_states=text_emb,
                         timestep=t.unsqueeze(0).to(device), class_labels=dummy).sample
        pu, pt = pred.chunk(2)
        pred = pu + args.guidance_scale * (pt - pu)
        latents = scheduler.step(pred, t, latents).prev_sample

    latents = latents / 0.18215
    img = vae.decode(latents.to(torch.float32)).sample          # [B,3,512,512]
    img = (img / 2 + 0.5).clamp(0, 1).mean(dim=1, keepdim=True)  # grayscale [B,1,512,512]
    if args.img_size != img.shape[-1]:
        img = F.interpolate(img, size=(args.img_size, args.img_size),
                            mode="bilinear", align_corners=False)
    img = (img.squeeze(1) * 255.0).round().clamp(0, 255).to(torch.uint8)  # [B,S,S]
    return img.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_h5", required=True)
    ap.add_argument("--per_class", type=int, default=5000,
                    help="Images generated per label (all 13). Total = 13 x per_class.")
    ap.add_argument("--guidance_scale", type=float, default=2.0)
    ap.add_argument("--num_inference_steps", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=512, help="Stored resolution (match real H5).")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--vae_path", default="/home/vault/b143dc/b143dc55/mimic/models/sd-vae")
    ap.add_argument("--t5_path", default="/home/vault/b143dc/b143dc55/mimic/models/t5-xxl")
    ap.add_argument("--router_mode", default=None, choices=["skip", "no_skip", "shared"])
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--max_experts", type=int, default=16)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[load] VAE / T5 ...", flush=True)
    vae = AutoencoderKL.from_pretrained(args.vae_path, torch_dtype=torch.float32,
                                        local_files_only=True).to(device).eval()
    vae.enable_slicing()
    text_encoder = T5EncoderModel.from_pretrained(args.t5_path, torch_dtype=torch.float16,
                                                  local_files_only=True).to(device).eval()
    tokenizer = T5Tokenizer.from_pretrained(args.t5_path, use_fast=False, legacy=False,
                                            local_files_only=True)
    scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, beta_schedule="linear",
                                            prediction_type="v_prediction",
                                            algorithm_type="dpmsolver++")

    print(f"[load] checkpoint {args.checkpoint}", flush=True)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    moe_config = ckpt.get("moe_config") if isinstance(ckpt, dict) else None
    model, has_moe = build_model(state_dict, moe_config, args, device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("expert_volume") and name in missing:
                p.fill_(1.0)
    other_missing = [k for k in missing if not (k.endswith("expert_volume") or "time_embed" in k)]
    if other_missing:
        raise RuntimeError(f"Unexpected missing keys: {other_missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected extra keys: {unexpected}")
    if "ema_state_dict" in ckpt:
        try:
            ema = EMAModel(model.parameters(), decay=0.9999,
                           model_cls=Transformer2DModel, model_config=model.config)
            ema.load_state_dict(ckpt["ema_state_dict"])
            ema.copy_to(model.parameters())
            print("  applied EMA weights", flush=True)
        except RuntimeError as e:
            print(f"  [WARN] EMA skipped: {e}", flush=True)
    model.eval().to(dtype=torch.float16)
    print(f"  model ready ({'MoE' if has_moe else 'baseline'})", flush=True)

    #  build the flat (prompt, label) plan: per_class single-positive per label 
    C = len(LABEL_COLS)
    prompts, labels = [], []
    for c, name in enumerate(LABEL_COLS):
        vec = np.zeros(C, dtype=np.float32)
        vec[c] = 1.0
        prompts.extend([prompt_for_label(name)] * args.per_class)
        labels.extend([vec] * args.per_class)
    labels = np.stack(labels)
    total = len(prompts)
    print(f"[plan] {total} images ({args.per_class} x {C} labels) @ {args.img_size}px", flush=True)

    #  H5: allocate up front, fill incrementally 
    os.makedirs(os.path.dirname(args.out_h5) or ".", exist_ok=True)
    with h5py.File(args.out_h5, "w") as f:
        imgs_ds = f.create_dataset("images", shape=(total, args.img_size, args.img_size),
                                   dtype="uint8",
                                   chunks=(min(64, total), args.img_size, args.img_size))
        labs_ds = f.create_dataset("labels", shape=(total, C), dtype="float32")
        f.attrs["label_cols"] = ",".join(LABEL_COLS)
        labs_ds[:] = labels

        ptr = 0
        n_batches = (total + args.batch_size - 1) // args.batch_size
        for b in range(n_batches):
            sl = slice(b * args.batch_size, min((b + 1) * args.batch_size, total))
            imgs = generate_batch(model, prompts[sl], tokenizer, text_encoder, vae,
                                  scheduler, args, device, seed=args.seed + b)
            imgs_ds[sl] = imgs
            ptr += imgs.shape[0]
            if (b + 1) % 20 == 0 or (b + 1) == n_batches:
                print(f"[gen] {ptr}/{total} ({100*ptr/total:.1f}%)", flush=True)

    print(f"\nSaved synthetic set -> {args.out_h5}", flush=True)
    print("Point both metrics at it with --synth_h5.", flush=True)


if __name__ == "__main__":
    main()