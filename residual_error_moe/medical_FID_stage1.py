"""
medical_fid_stage1.py: FID comparison between synthetic and real images

The FID math, the domain DenseNet extractor, the matched-global-FID trick, and
the rare-class oversampling plan are unchanged.
"""

import argparse
import gc
import json
import math
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
from transformers import T5EncoderModel, T5Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import LABEL_COLS, PATHOLOGY_COLS, SUPPORT_COL, NO_FINDING_COL, \
    DEVICE_TOKEN, UNCOND_PROMPT, build_prompt, tokens_from_labels
from moe_lora import inject_moe_lora
from train_moe_stage0 import load_frozen_base
from train_moe_stage1 import FFInputCapture, make_routing_feature

LATENT_SCALE = 0.18215
VAE_PATH = "/home/vault/b143dc/b143dc55/mimic/models/sd-vae"
T5_PATH = "/home/vault/b143dc/b143dc55/mimic/models/t5-xxl"
DENSENET_PATH = "/home/vault/b143dc/b143dc55/mimic/models/densenet121-res224-mimic_ch.pt"


# return essential information of the real images
class RealRefDataset(Dataset):
    def __init__(self, h5_file_path, csv_df, transform):
        self.h5_file_path = h5_file_path
        self.transform = transform
        with h5py.File(h5_file_path, "r") as f:
            h5_paths = [str(p)[2:-1] for p in f["paths"][:]]
        self.p2i = {p: i for i, p in enumerate(h5_paths)}
        self.df = csv_df[csv_df["path"].isin(self.p2i.keys())].reset_index(drop=True)
        print(f"  matched {len(self.df)} real images.", flush=True)
        self.h5 = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_file_path, "r")
            self.imgs = self.h5["images"]
        row = self.df.iloc[i]
        arr = self.imgs[self.p2i[row["path"]]]
        img = self.transform(Image.fromarray(arr).convert("L"))
        label_vec = row[LABEL_COLS].to_numpy().astype("float32")
        label_vec = (label_vec == 1.0).astype("float32")
        prompt = build_prompt(tokens_from_labels(label_vec))
        return img, torch.from_numpy(label_vec), prompt, row["path"]


# Feature extractor and FID math
def get_extractor(device):
    if not os.path.exists(DENSENET_PATH):
        raise FileNotFoundError(DENSENET_PATH)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = torch.load(DENSENET_PATH, map_location="cpu")
    return model.to(device).eval().requires_grad_(False)


def calc_fid(real, synth, eps=1e-6):
    mu_r, mu_s = real.mean(0), synth.mean(0)
    sig_r, sig_s = np.cov(real, rowvar=False), np.cov(synth, rowvar=False)
    diff = mu_r - mu_s
    covmean, _ = linalg.sqrtm(sig_r.dot(sig_s), disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(sig_r.shape[0]) * eps
        covmean = linalg.sqrtm((sig_r + off).dot(sig_s + off))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sig_r) + np.trace(sig_s) - 2 * np.trace(covmean))


@torch.no_grad()
def extract_feats(img_1ch, extractor, device):
    # img_1ch: [B,1,H,W] in [0,1]. Resizes to 224, scales to xrv range
    img = F.interpolate(img_1ch, size=(224, 224), mode="bilinear", align_corners=False)
    img_xrv = img * 2048.0 - 1024.0
    feats = extractor.features(img_xrv)
    return F.adaptive_avg_pool2d(feats, (1, 1)).view(img.size(0), -1).cpu().numpy()


def build_oversample_plan(real_labels, target_per_class, max_repeat):
    n, C = real_labels.shape
    counts = real_labels.sum(0)
    desired = np.ones(C)
    for c in range(C):
        if counts[c] > 0:
            desired[c] = math.ceil(target_per_class / counts[c])
    desired = np.clip(desired, 1, max_repeat)
    extra = np.zeros(n, dtype=np.int64)
    for i in range(n):
        pos = np.where(real_labels[i] == 1.0)[0]
        if len(pos):
            extra[i] = max(0, int(desired[pos].max()) - 1)
    return extra


# Model loading
def load_base_only(base_ckpt, device):
    return load_frozen_base(base_ckpt, device).to(torch.float16), None, None


def load_stage1(base_ckpt, adapter_ckpt, moe_layer, rank, e_max, top_k, device):
    model = load_frozen_base(base_ckpt, device)
    dim = model.transformer_blocks[0].ff.net[0].proj.in_features
    model, injected = inject_moe_lora(model, [moe_layer], dim, rank=rank,
                                      e_max=e_max, top_k=top_k, route_feat_dim=4)
    layer = injected[moe_layer]
    ck = torch.load(adapter_ckpt, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(ck["adapter_state"], strict=False)
    real_missing = [k for k in missing if any(t in k for t in
                    ["shared_expert", "experts.", "router", "active_mask"])]
    if real_missing:
        raise RuntimeError(f"adapter keys missing: {real_missing[:6]}")
    print(f"  Stage1 adapter epoch {ck.get('epoch')} val {ck.get('val_loss'):.4f} "
          f"active {int(layer.active_mask.sum())}", flush=True)
    # capture the FF input on this model's base ff
    cap = FFInputCapture()
    model.transformer_blocks[moe_layer].ff.base_ff.register_forward_pre_hook(cap)
    
    return model.eval().to(device=device, dtype=torch.float16), layer, cap


# Generation
@torch.no_grad()
def generate_batch(model, moe_layer, cap, route_mode, prompts, vae, tokenizer,
                   text_encoder, scheduler, args, batch_idx, device):
    bsz = len(prompts)
    scheduler.set_timesteps(args.num_inference_steps, device=device)
    with torch.amp.autocast("cuda"):
        ti = tokenizer(prompts, padding="max_length", max_length=77,
                       truncation=True, return_tensors="pt").to(device)
        ui = tokenizer([UNCOND_PROMPT] * bsz, padding="max_length", max_length=77,
                       truncation=True, return_tensors="pt").to(device)
        cond = text_encoder(ti.input_ids)[0].to(torch.float16)
        uncond = text_encoder(ui.input_ids)[0].to(torch.float16)
        text_emb = torch.cat([uncond, cond])

    dummy = torch.zeros((bsz * 2,), dtype=torch.long, device=device)
    g = torch.Generator(device=device).manual_seed(1234 + batch_idx)
    latents = torch.randn((bsz, 4, 64, 64), generator=g, dtype=torch.float16, device=device)

    for t in scheduler.timesteps:
        inp = scheduler.scale_model_input(torch.cat([latents] * 2), t)
        t_in = t.unsqueeze(0).to(device)
        if moe_layer is not None:
            if route_mode == "shared_only":
                moe_layer.set_routing_feature(None)
            else:
                # production of the vpred and the cap.value. This runs on the sum of the basemodel 
                # and the shared expert
                moe_layer.set_routing_feature(None)
                with torch.amp.autocast("cuda"):
                    vpred = model(hidden_states=inp, encoder_hidden_states=text_emb,
                                  timestep=t_in, class_labels=dummy).sample
                # cap.value contains baseline method
                feat = make_routing_feature(cap.value, vpred.float(), args.patch)
                moe_layer.set_routing_feature(feat)
        with torch.amp.autocast("cuda"):
            pred = model(hidden_states=inp, encoder_hidden_states=text_emb,
                         timestep=t_in, class_labels=dummy).sample
        pu, pt = pred.chunk(2)
        pred = pu + args.guidance_scale * (pt - pu)
        latents = scheduler.step(pred, t, latents).prev_sample
        if torch.isnan(latents).any():
            print(f"  [CRITICAL] NaN at t={t}", flush=True)
            break

    img = vae.decode((latents / LATENT_SCALE).to(torch.float32)).sample
    img = (img / 2 + 0.5).clamp(0, 1).mean(dim=1, keepdim=True)
    return img


def evaluate_model(tag, model, moe_layer, cap, route_mode, gen_prompts, gen_labels,
                   gen_is_base, real_features, real_labels, extractor, vae, tokenizer,
                   text_encoder, scheduler, args, device):
    print(f"\n=== generating for {tag} (route_mode={route_mode}) ===", flush=True)
    synth = []
    nb = math.ceil(len(gen_prompts) / args.batch_size)
    for b in tqdm(range(nb), desc=tag):
        sl = slice(b * args.batch_size, (b + 1) * args.batch_size)
        img = generate_batch(model, moe_layer, cap, route_mode, gen_prompts[sl],
                             vae, tokenizer, text_encoder, scheduler, args, b, device)
        synth.append(extract_feats(img, extractor, device))
    synth = np.vstack(synth)

    base_synth = synth[gen_is_base]
    result = {"global_fid": calc_fid(real_features, base_synth), "per_class": {}}
    for c, name in enumerate(PATHOLOGY_COLS):
        # map PATHOLOGY_COLS index to LABEL_COLS index
        ci = LABEL_COLS.index(name)
        r_mask = real_labels[:, ci] == 1.0
        s_mask = gen_labels[:, ci] == 1.0
        n_r, n_s = int(r_mask.sum()), int(s_mask.sum())
        if n_r < 50 or n_s < 50:
            result["per_class"][name] = {"fid": None, "n_real": n_r, "n_synth": n_s, "note": "too few"}
            continue
        fid_c = calc_fid(real_features[r_mask], synth[s_mask])
        note = "ok" if min(n_r, n_s) >= args.min_bucket else "LOW_N_unreliable"
        result["per_class"][name] = {"fid": fid_c, "n_real": n_r, "n_synth": n_s, "note": note}

    nf = LABEL_COLS.index(NO_FINDING_COL)
    r_h = real_labels[:, nf] == 1.0
    s_h = gen_labels[:, nf] == 1.0
    result["healthy_fid"] = (calc_fid(real_features[r_h], synth[s_h])
                             if r_h.sum() >= 50 and s_h.sum() >= 50 else None)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_checkpoint", required=True)
    ap.add_argument("--adapter_checkpoint", required=True)
    ap.add_argument("--h5_path", required=True, help="TEST-folder shard (disjoint from training)")
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--route_mode", default="est_residual", choices=["est_residual", "shared_only"])
    ap.add_argument("--moe_layer", type=int, default=11)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--e_max", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=1)
    ap.add_argument("--patch", type=int, default=2)
    ap.add_argument("--num_real", type=int, default=10000)
    ap.add_argument("--target_per_class", type=int, default=2000)
    ap.add_argument("--max_repeat", type=int, default=12)
    ap.add_argument("--min_bucket", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_inference_steps", type=int, default=25)
    ap.add_argument("--guidance_scale", type=float, default=2.0)
    ap.add_argument("--out_json", default="fid_base_vs_stage1.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}", flush=True)

    csv_df = pd.read_csv(args.csv_path)
    for c in LABEL_COLS:
        if c not in csv_df.columns:
            raise ValueError(f"CSV missing label column: {c}")

    # real reference features and labels
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    ds = RealRefDataset(args.h5_path, csv_df, transform)
    n_real = min(args.num_real, len(ds))
    gen = torch.Generator().manual_seed(42)
    subset, _ = torch.utils.data.random_split(ds, [n_real, len(ds) - n_real], generator=gen)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    extractor = get_extractor(device)
    rf, rl, rp = [], [], []
    print("\n[phase1] real features...", flush=True)
    for img, lbl, prompt, _ in tqdm(loader, desc="real"):
        img = img.to(device)
        rf.append(extract_feats(img, extractor, device))
        rl.append(lbl.numpy()); rp.extend(prompt)
    real_features = np.vstack(rf)
    real_labels = np.vstack(rl)
    print(f" -> real {real_features.shape}, labels {real_labels.shape}", flush=True)

    # generation plan (both for baseline model and the shared versions)
    extra = build_oversample_plan(real_labels, args.target_per_class, args.max_repeat)
    gi, gb = [], []
    for i in range(len(rp)):
        gi.append(i); gb.append(True)
        for _ in range(int(extra[i])):
            gi.append(i); gb.append(False)
    gi = np.array(gi); gb = np.array(gb, dtype=bool)
    gen_prompts = [rp[i] for i in gi]
    gen_labels = real_labels[gi]
    print(f"[plan] base={gb.sum()} extra={(~gb).sum()} total={len(gen_prompts)}", flush=True)

    # shared VAE / T5 / scheduler
    vae = AutoencoderKL.from_pretrained(VAE_PATH, torch_dtype=torch.float32,
                                        local_files_only=True).to(device).eval()
    vae.enable_slicing()
    text_encoder = T5EncoderModel.from_pretrained(T5_PATH, torch_dtype=torch.float16,
                                                  local_files_only=True).to(device).eval()
    tokenizer = T5Tokenizer.from_pretrained(T5_PATH, use_fast=False, legacy=False,
                                            local_files_only=True)
    scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, beta_schedule="linear",
                                            prediction_type="v_prediction", algorithm_type="dpmsolver++")

    results = {}

    # base
    base_model, _, _ = load_base_only(args.base_checkpoint, device)
    results["base"] = evaluate_model("base", base_model, None, None, "none",
                                     gen_prompts, gen_labels, gb, real_features, real_labels,
                                     extractor, vae, tokenizer, text_encoder, scheduler, args, device)
    del base_model; gc.collect(); torch.cuda.empty_cache()

    # load and evaluate synth model
    moe_model, moe_layer, cap = load_stage1(args.base_checkpoint, args.adapter_checkpoint,
                                            args.moe_layer, args.rank, args.e_max, args.top_k, device)
    results["stage1"] = evaluate_model("stage1", moe_model, moe_layer, cap, args.route_mode,
                                       gen_prompts, gen_labels, gb, real_features, real_labels,
                                       extractor, vae, tokenizer, text_encoder, scheduler, args, device)

    # report
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.out_json}\n")

    def tail_mean(res):
        vals = [d["fid"] for d in res["per_class"].values() if d["fid"] is not None]
        return np.mean(vals) if vals else float("nan")

    print(f"{'':<24}{'base':>12}{'stage1':>12}{'delta':>12}")
    print(f"{'GLOBAL FID':<24}{results['base']['global_fid']:>12.3f}"
          f"{results['stage1']['global_fid']:>12.3f}"
          f"{results['stage1']['global_fid']-results['base']['global_fid']:>+12.3f}")
    hb, hs = results['base']['healthy_fid'], results['stage1']['healthy_fid']
    if hb and hs:
        print(f"{'HEALTHY FID':<24}{hb:>12.3f}{hs:>12.3f}{hs-hb:>+12.3f}")
    print(f"{'TAIL MEAN FID':<24}{tail_mean(results['base']):>12.3f}"
          f"{tail_mean(results['stage1']):>12.3f}"
          f"{tail_mean(results['stage1'])-tail_mean(results['base']):>+12.3f}")
    print(f"\n{'class':<24}{'base':>10}{'stage1':>10}{'delta':>10}  note")
    for name in PATHOLOGY_COLS:
        b = results["base"]["per_class"][name]
        s = results["stage1"]["per_class"][name]
        if b["fid"] is None or s["fid"] is None:
            print(f"{name:<24}{'n/a':>10}{'n/a':>10}{'':>10}  {b['note']}/{s['note']}")
        else:
            print(f"{name:<24}{b['fid']:>10.3f}{s['fid']:>10.3f}"
                  f"{s['fid']-b['fid']:>+10.3f}  {s['note']}")
    print("\nNegative delta = Stage 1 better (lower FID). Watch tail classes.")


if __name__ == "__main__":
    main()