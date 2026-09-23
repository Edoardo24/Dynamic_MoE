import os
from huggingface_hub import snapshot_download

print("Downloading T5-XXL as raw files (No symlinks)...")
snapshot_download(
    repo_id="DeepFloyd/t5-v1_1-xxl",
    local_dir="/home/vault/b143dc/b143dc55/mimic/models/t5-xxl",
    local_dir_use_symlinks=False
)
print("T5-XXL downloaded perfectly!")

print("Downloading VAE as raw files...")
snapshot_download(
    repo_id="stabilityai/sd-vae-ft-ema",
    local_dir="/home/vault/b143dc/b143dc55/mimic/models/sd-vae",
    local_dir_use_symlinks=False
)
print("VAE downloaded perfectly!")