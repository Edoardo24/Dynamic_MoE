"""
Two-stage feature extraction.

Stage A: text bank (run once, shard-independent). Embed every prompt the
training loop can construct,  each observed label set, its per-finding-dropout
subsets, orderings permutations of each, and the empty prompt "" (row 0) for
classifier-free guidance

Stage B: latents (run per shard). Save the VAE posterior (mean and
log-variance) so the training loop can resample z each epoch, along with the
14-dim label vector and the id group key for a patient-level split.
"""

import argparse
import glob
import json
import os
import pandas as pd
from itertools import combinations

import numpy as np
import torch
from diffusers import AutoencoderKL
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

from data_preprocessing import (
    DEVICE_TOKEN,
    LABEL_COLS,
    PATHOLOGY_COLS,
    SUPPORT_COL,
    UNCOND_PROMPT,
    build_prompt,
    get_mimic_dataloader,
    load_label_csv,
)

MAX_LEN = 77
T5_DIM = 4096


def subset_key(tokens):
    """Canonical, order independent key for a token subset."""
    return "|".join(sorted(tokens))


# Stage A: text bank
def enumerate_prompts(df, orderings=2, max_drop=2, seed=0):
    """
    Returns an ordered list of unique prompt strings. 
    Row 0 is always the unconditional prompt.
    """
    rng = np.random.default_rng(seed)

    label_mat = df[LABEL_COLS].to_numpy().astype(np.float32)
    support_idx = LABEL_COLS.index(SUPPORT_COL)

    # observed token sets, as canonical (sorted) tuples
    observed = set()
    for row in label_mat:
        toks = [PATHOLOGY_COLS[i].lower() for i in range(len(PATHOLOGY_COLS)) if row[i] == 1.0]
        if row[support_idx] == 1.0:
            toks.append(DEVICE_TOKEN)
        observed.add(tuple(sorted(toks)))

    print(f"[bank] {len(observed)} distinct label sets in the CSV.", flush=True)

    # every subset reachable by dropping up to `max_drop` tokens
    subsets = set()
    for s in observed:
        n = len(s)
        for k in range(max(0, n - max_drop), n + 1):
            subsets.update(combinations(s, k))

    print(f"[bank] {len(subsets)} distinct subsets after <= {max_drop} dropped token(s).", flush=True)

    prompts = [UNCOND_PROMPT]
    seen = {UNCOND_PROMPT: 0}
    # subset_key to its row indices (one per cached ordering)
    subset_rows = {}

    for s in sorted(subsets):
        variants = [list(s)]
        if len(s) >= 2 and orderings > 1:
            for _ in range(orderings - 1):
                v = list(s)
                rng.shuffle(v)
                variants.append(v)

        key = subset_key(s)
        rows = []
        for v in variants:
            p = build_prompt(v)
            if p not in seen:
                seen[p] = len(prompts)
                prompts.append(p)
            rows.append(seen[p])
        subset_rows[key] = sorted(set(rows))

    gb = len(prompts) * MAX_LEN * T5_DIM * 2 / 1e9
    print(f"[bank] {len(prompts)} unique prompts -> {gb:.2f} GB fp16 on disk.", flush=True)
    if gb > 40:
        raise RuntimeError("Prompt bank too large; lower --orderings or --max_drop.")
    return prompts, subset_rows


def build_text_bank(args, device):
    df = load_label_csv(args.csv_path, split=None)  # bank must cover TRAIN and TEST
    prompts, subset_rows = enumerate_prompts(df, orderings=args.orderings, max_drop=args.max_drop)

    os.makedirs(args.bank_dir, exist_ok=True)
    idx_path = os.path.join(args.bank_dir, "prompt_index.json")
    emb_path = os.path.join(args.bank_dir, "prompt_embeddings.fp16.npy")

    with open(idx_path, "w") as f:
        json.dump(
            {"prompts": prompts, "subset_rows": subset_rows,
             "max_len": MAX_LEN, "dim": T5_DIM, "uncond_row": 0},
            f,
        )

    print("[bank] loading T5...", flush=True)
    tokenizer = T5Tokenizer.from_pretrained(args.t5_path, use_fast=False, local_files_only=True)
    text_encoder = T5EncoderModel.from_pretrained(
        args.t5_path, torch_dtype=torch.float16,
        low_cpu_mem_usage=True, local_files_only=True,
    ).to(device).eval()
    text_encoder.requires_grad_(False)

    bank = np.lib.format.open_memmap(
        emb_path, mode="w+", dtype=np.float16, shape=(len(prompts), MAX_LEN, T5_DIM)
    )

    bs = args.text_batch_size
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), bs), desc="Embedding prompts"):
            chunk = prompts[i : i + bs]
            ti = tokenizer(chunk, padding="max_length", max_length=MAX_LEN,
                           truncation=True, return_tensors="pt").to(device)
            emb = text_encoder(ti.input_ids)[0]           # [b, 77, 4096] fp16
            bank[i : i + len(chunk)] = emb.cpu().numpy()

    bank.flush()
    del bank, text_encoder
    torch.cuda.empty_cache()
    print(f"[bank] wrote {emb_path} and {idx_path}", flush=True)


# Stage B: latents
def extract_latents(args, device):
    print("[latents] loading VAE...", flush=True)
    vae = AutoencoderKL.from_pretrained(
        args.vae_path, torch_dtype=torch.float16, local_files_only=True
    ).to(device).eval()
    vae.requires_grad_(False)

    dataloader = get_mimic_dataloader(
        csv_path=args.csv_path,
        h5_dir=args.h5_dir,
        shards=args.shards,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    total = len(dataloader.dataset)
    manifest = []

    with torch.no_grad():
        for idx, (images, labels, group_ids, paths) in enumerate(tqdm(dataloader, desc="Latents")):
            images = images.to(device, dtype=torch.float16)
            posterior = vae.encode(images).latent_dist

            # Store the posterior, not one frozen draw; the 0.18215 scaling is
            # applied at train time, after resampling.
            #
            # We persist logvar rather than std: std for a well-fit VAE latent
            # can be ~1e-6, which underflows fp16 (min normal ~6e-5) to zero and
            # silently disables resampling for those elements. logvar is clamped
            # by diffusers to [-30, 20] and is safe in fp16.
            mu = posterior.mean.cpu()          # [b, 4, 64, 64]
            logvar = posterior.logvar.cpu()

            for b in range(images.shape[0]):
                stem = os.path.splitext(os.path.basename(paths[b]))[0]
                torch.save(
                    {
                        "latent_mean": mu[b].to(torch.float16),
                        "latent_logvar": logvar[b].to(torch.float16),
                        "labels": labels[b].to(torch.float32),   # 14-dim, LABEL_COLS order
                        "group_id": int(group_ids[b]),           # patient/study key
                        "path": paths[b],
                    },
                    os.path.join(args.save_dir, stem + ".pt"),
                )
                manifest.append(
                    [stem + ".pt", paths[b], int(group_ids[b])] + labels[b].tolist()
                )

            if (idx + 1) % 20 == 0 or (idx + 1) == len(dataloader):
                done = min((idx + 1) * args.batch_size, total)
                print(f"[PROGRESS] {done}/{total} ({100 * done / total:.1f}%)", flush=True)

    # train_base.py do the patient-level split and the
    # balanced sampler weights without opening the files. path is the
    # globally unique MIMIC image key
    tag = args.manifest_tag or args.split.lower()
    man_path = os.path.join(args.save_dir, f"manifest_{tag}.csv")
    pd.DataFrame(manifest, columns=["file", "path", "group_id"] + LABEL_COLS).to_csv(
        man_path, index=False)
    print(f"[latents] wrote {man_path} ({len(manifest)} rows)", flush=True)
    print("[latents] done.", flush=True)


# Verify the disjointness that filename based exclusion can't give you.
def verify_disjoint(cache_dirs):
    sets, gsets = {}, {}
    for d in cache_dirs:
        mans = sorted(glob.glob(os.path.join(d, "manifest_*.csv")))
        if not mans:
            raise FileNotFoundError(f"no manifest_*.csv in {d}")
        df = pd.concat([pd.read_csv(m) for m in mans], ignore_index=True)
        sets[d] = set(df["path"])
        gsets[d] = set(df["group_id"])
        print(f"{d}: {len(df)} images, {len(gsets[d])} patients", flush=True)

    ok = True
    dirs = list(cache_dirs)
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            a, b = dirs[i], dirs[j]
            img_overlap = sets[a] & sets[b]
            pat_overlap = gsets[a] & gsets[b]
            if img_overlap:
                ok = False
                print(f"  [FAIL] {a} and {b} share {len(img_overlap)} IMAGES", flush=True)
            if pat_overlap:
                # not necessarily fatal, but worth knowing about
                print(f"  [WARN] {a} and {b} share {len(pat_overlap)} PATIENTS "
                      f"(same patient, different studies)", flush=True)
            if not img_overlap and not pat_overlap:
                print(f"  [OK]   {a} and {b} are image- and patient-disjoint", flush=True)

    if not ok:
        raise SystemExit("Partitions overlap. Fix before training.")
    print("\nAll partitions are image-disjoint.", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["text", "latents", "verify", "both"], default="both")
    p.add_argument("--csv_path", type=str, help="the FULL multi-label CSV")
    p.add_argument("--h5_dir", type=str,
                   help="ONE MIMIC folder (train | val | test). Shard names repeat across folders.")
    p.add_argument("--shards", nargs="*", default=None,
                   help="explicit shard filenames within --h5_dir; omit for all of them")
    p.add_argument("--save_dir", type=str, help="per-image latent cache")
    p.add_argument("--bank_dir", type=str, help="prompt embedding bank")
    p.add_argument("--manifest_tag", type=str, default=None,
                   help="suffix for manifest_<tag>.csv; defaults to the split name")
    p.add_argument("--vae_path", type=str, default="/home/vault/b143dc/b143dc55/mimic/models/sd-vae")
    p.add_argument("--t5_path", type=str, default="/home/vault/b143dc/b143dc55/mimic/models/t5-xxl")
    p.add_argument("--split", type=str, default="ALL",
                   help="TRAIN | VALIDATE | TEST | ALL. If each folder already contains only "
                        "one split, ALL is correct and the Split column is just a cross-check.")
    p.add_argument("--verify_dirs", nargs="*", default=[],
                   help="cache dirs to check for image/patient overlap (--stage verify)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--text_batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--orderings", type=int, default=2,
                   help="permutations of the finding list cached per subset")
    p.add_argument("--max_drop", type=int, default=2,
                   help="max findings droppable by per-finding dropout at train time")
    args = p.parse_args()

    if args.stage == "verify":
        if len(args.verify_dirs) < 2:
            raise ValueError("--stage verify needs at least two --verify_dirs")
        verify_disjoint(args.verify_dirs)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    if args.stage in ("text", "both"):
        if not args.csv_path or not args.bank_dir:
            raise ValueError("--csv_path and --bank_dir are required for the text stage.")
        build_text_bank(args, device)
    if args.stage in ("latents", "both"):
        if not all([args.h5_dir, args.save_dir, args.csv_path]):
            raise ValueError("--csv_path, --h5_dir and --save_dir are required for the latents stage.")
        extract_latents(args, device)


if __name__ == "__main__":
    main()