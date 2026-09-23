"""
MIMIC-CXR multi-label preprocessing.

Important aspects of the data formatting:
  - Label columns are matched by name, not by position. Slicing by index
    (row.iloc[4:-1]) drops Support Devices whenever the CSV column order shifts.
  - "No Finding" is a structural flag, not a prompt word
  - The Split column is respected, so we don't train on test images.
  - Images are forced to RGB, since the VAE expects in_channels=3.
"""

import glob
import os
import random

import h5py
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

# Label schema. The order here defines the index of the label vector.
PATHOLOGY_COLS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
]
NO_FINDING_COL = "No Finding"
SUPPORT_COL = "Support Devices"

# 14-dim vector: 12 pathologies + No Finding + Support Devices
LABEL_COLS = PATHOLOGY_COLS + [NO_FINDING_COL, SUPPORT_COL]

DEVICE_TOKEN = "support devices"


def build_prompt(tokens):
    """
    tokens: iterable of lowercase strings, possibly including DEVICE_TOKEN.

    Returns the prompt string that T5 will embed. extract_features.py and
    train_base.py
    """
    tokens = list(tokens)
    has_device = DEVICE_TOKEN in tokens
    findings = [t for t in tokens if t != DEVICE_TOKEN]

    if not findings:
        s = "a chest radiograph with no acute cardiopulmonary findings"
    else:
        s = "a chest radiograph demonstrating findings consistent with " + ", ".join(findings)

    if has_device:
        s += ", with support devices present"
    return s + "."


UNCOND_PROMPT = ""  # row 0 of the text bank (used for classifier-free guidance)


def tokens_from_labels(label_vec):
    """label_vec: length-14 array aligned to LABEL_COLS. Returns the prompt tokens."""
    toks = [PATHOLOGY_COLS[i].lower() for i in range(len(PATHOLOGY_COLS)) if label_vec[i] == 1.0]
    if label_vec[LABEL_COLS.index(SUPPORT_COL)] == 1.0:
        toks.append(DEVICE_TOKEN)
    return toks


def load_label_csv(csv_path, split=None, verify=True):
    df = pd.read_csv(csv_path)

    missing = [c for c in LABEL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required label columns: {missing}")

    if "Split" in df.columns:
        counts = df["Split"].value_counts().to_dict()
        print(f"  CSV Split counts: {counts}", flush=True)

    if split is not None and split != "ALL":
        if "Split" not in df.columns:
            raise ValueError("CSV has no 'Split' column but split filtering was requested.")
        df = df[df["Split"] == split].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(
                f"Split={split!r} matched 0 rows. Check the exact spelling against the "
                f"counts printed above (MIMIC uses TRAIN / VALIDATE / TEST)."
            )

    if verify:
        # 'No Finding' should be mutually exclusive with the pathologies.
        n_path = df[PATHOLOGY_COLS].sum(axis=1)
        bad = int(((df[NO_FINDING_COL] == 1.0) & (n_path > 0)).sum())
        if bad:
            print(f"  [WARN] {bad} rows have No Finding=1 alongside a pathology.", flush=True)
        orphan = int(((df[NO_FINDING_COL] == 0.0) & (n_path == 0)).sum())
        if orphan:
            print(f"  [WARN] {orphan} rows have neither No Finding nor any pathology.", flush=True)

    return df


class MIMICShardDataset(Dataset):
    def __init__(self, h5_file_path, csv_df, transform=None):
        self.h5_file_path = h5_file_path
        self.transform = transform

        shard_name = h5_file_path.split("/")[-1]
        with h5py.File(h5_file_path, "r") as f:
            h5_paths = [str(p)[2:-1] for p in f["paths"][:]]

        self.path_to_h5_index = {p: i for i, p in enumerate(h5_paths)}

        valid = csv_df[csv_df["path"].isin(self.path_to_h5_index.keys())].reset_index(drop=True)
        self.paths = valid["path"].tolist()
        self.group_ids = valid["id"].to_numpy().astype(np.int64)
        self.labels = valid[LABEL_COLS].to_numpy().astype(np.float32)

        print(f"  [{shard_name}] matched {len(self.paths)} / {len(h5_paths)} images.", flush=True)
        self.h5_file = None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if self.h5_file is None:  # lazy open, one handle per worker
            self.h5_file = h5py.File(self.h5_file_path, "r")
            self.images_dset = self.h5_file["images"]

        image_path = self.paths[index]
        img_array = self.images_dset[self.path_to_h5_index[image_path]]

        # The VAE has in_channels=3. Force RGB rather than relying on the H5
        # happening to store 3 channels.
        img_pil = Image.fromarray(img_array).convert("RGB")
        img_tensor = self.transform(img_pil) if self.transform else torch.from_numpy(img_array).float()

        return (
            img_tensor,
            torch.from_numpy(self.labels[index]),
            int(self.group_ids[index]),
            image_path,
        )


def get_mimic_dataloader(csv_path, h5_dir, shards=None, split=None,
                         batch_size=32, num_workers=4, image_size=512):
    """
        Loadin the different shards in the the dataset. The shards are splitted into multiple
        folders. Even if the shard name is the same if the dir is different no leakage
        is possible
    """
    transforms = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),                       # now in [0, 1]
        T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),  # now in [-1, 1]
    ])

    df = load_label_csv(csv_path, split=split)
    print(f"CSV rows after split={split!r} filter: {len(df)}", flush=True)

    if shards:
        h5_files = [os.path.join(h5_dir, s) for s in shards]
        for f in h5_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f)
    else:
        h5_files = sorted(glob.glob(os.path.join(h5_dir, "mimic_images_shard*.h5")))

    if not h5_files:
        raise RuntimeError(f"No shards selected in {h5_dir}.")
    print(f"Using {len(h5_files)} shard(s) from {h5_dir}: "
          f"{[os.path.basename(f) for f in h5_files]}", flush=True)

    datasets = []
    for h5_path in h5_files:
        ds = MIMICShardDataset(h5_path, df, transform=transforms)
        if len(ds) > 0:
            datasets.append(ds)

    full = ConcatDataset(datasets)
    print(f"Total images: {len(full)}", flush=True)

    return DataLoader(
        full,
        batch_size=batch_size,
        shuffle=False,          
        num_workers=num_workers,
        pin_memory=True,
    )