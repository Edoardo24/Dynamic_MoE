"""
data_preprocessing.py: MULTI-LABEL MIMIC-CXR preprocessing.

Single source of truth for:
  * LABEL_COLS: the 13 label columns (CSV columns 4:-1). Support Devices
                       (the last column) is intentionally DROPPED; No Finding is
                       KEPT as a normal label.
  * PATHOLOGY_COLS: the 12 actual pathologies (LABEL_COLS minus No Finding).
                       Used by train_classifier.py for the macro-F1 that the
                       thesis reports.
"""

import glob

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as T
from PIL import Image



# LABEL SCHEMA  (CSV columns:  Unnamed:0, id, path, Split, <labels...>)
# Columns 4:-1 == the 13 kept labels; column -1 (Support Devices) dropped.
LABEL_COLS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
]

# Pathologies only: macro-F1 is averaged over these. "No Finding" is a valid label
# (a prediction target) but not a pathology, so it is excluded from the average.
PATHOLOGY_COLS = [c for c in LABEL_COLS if c != "No Finding"]


def labels_to_prompt(label_vec):
    """
    label_vec: sequence aligned with LABEL_COLS, entries in {0.0, 1.0}.

    An image with no pathology present gets the normal radiograph prompt. Otherwise the
    active pathologies are listed. 
    """
    active = [LABEL_COLS[i] for i, v in enumerate(label_vec) if float(v) == 1.0]
    pathologies = [a for a in active if a in PATHOLOGY_COLS]
    if len(pathologies) == 0:
        return "A normal chest radiograph with no acute cardiopulmonary findings."
    findings_str = ", ".join(pathologies).lower()
    return f"A chest radiograph demonstrating findings consistent with {findings_str}."


class FilterMIMICDataset(Dataset):

    def __init__(self, h5_file_path, csv_df, path_col_name="path", transform=None):
        self.h5_file_path = h5_file_path
        self.transform = transform

        shard_name = h5_file_path.split("/")[-1]
        print(f"  [{shard_name}] Reading paths from H5 file...", flush=True)

        with h5py.File(h5_file_path, "r") as f:
            h5_path = [str(p)[2:-1] for p in f["paths"][:]]

        print(f"  [{shard_name}] Found {len(h5_path)} images. Building index map...", flush=True)
        self.path_to_h5_index = {path: index for index, path in enumerate(h5_path)}

        print(f"  [{shard_name}] Cross-referencing with CSV labels...", flush=True)
        # Fail if the CSV doesn't have the schema we expect
        missing = [c for c in ([path_col_name] + LABEL_COLS) if c not in csv_df.columns]
        if missing:
            raise ValueError(f"CSV missing expected columns: {missing}")

        self.valid_df = csv_df[csv_df[path_col_name].isin(self.path_to_h5_index.keys())]
        self.valid_df = self.valid_df.reset_index(drop=True)

        print(f"  [{shard_name}] Successfully matched {len(self.valid_df)} usable images.", flush=True)
        self.h5_file = None

    def __len__(self):
        return len(self.valid_df)

    def __getitem__(self, index):
        # Lazy open the H5 handle per worker.
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_file_path, "r")
            self.images_dset = self.h5_file["images"]

        row = self.valid_df.iloc[index]
        image_path = row["path"]
        h5_index = self.path_to_h5_index[image_path]

        img_array = self.images_dset[h5_index]
        img_pil = Image.fromarray(img_array)  

        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = torch.tensor(img_array, dtype=torch.float32)

        # Multi hot label vector selected by name (robust to column order / to
        # Support Devices sitting last), positive == 1.0. Uncertain (-1) and blank
        # (NaN) count as negative, per the chosen U-Zeros policy.
        label_vec = (row[LABEL_COLS].values.astype("float32") == 1.0).astype("float32")
        text_prompt = labels_to_prompt(label_vec)

        return img_tensor, text_prompt, image_path


def get_mimic_dataloader(csv_path, h5_dir, batch_size=32, num_workers=4,
                         shard_glob="mimic_images_shard*.h5"):
    
    # Geometrical transformation (RandomRotation removed to speed up extraction).
    train_transforms = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),                          # scales to [0.0, 1.0]
        T.Normalize(mean=[0.5], std=[0.5]),    # shifts to [-1.0, 1.0]
    ])

    df = pd.read_csv(csv_path)
     
    # Hard coded data preprocessing of the trainign shards
    # file_names = ['mimic_images_shard0000.h5', 'mimic_images_shard0001.h5', 'mimic_images_shard0002.h5', 'mimic_images_shard0004.h5']
    file_names = ['mimic_images_shard0003.h5']
    h5_files = [f"{h5_dir}/{fname}" for fname in file_names]
    dataset_list = []
    
    if len(h5_files) == 0:
        raise FileNotFoundError(f"No shards matching '{shard_glob}' in {h5_dir}")
    print(f"Found {len(h5_files)} shard(s) to process.", flush=True)

    dataset_list = []
    for h5_path in h5_files:
        print(f"Processing {h5_path.split('/')[-1]}...")
        shard_dataset = FilterMIMICDataset(
            h5_file_path=h5_path,
            csv_df=df,
            path_col_name="path",
            transform=train_transforms,
        )
        if len(shard_dataset) > 0:
            dataset_list.append(shard_dataset)

    full_train_dataset = ConcatDataset(dataset_list)

    return DataLoader(
        full_train_dataset,
        batch_size=batch_size,
        shuffle=False,           
        num_workers=num_workers,
        pin_memory=True,
    )