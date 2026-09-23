import os
import argparse
import torch
from diffusers import AutoencoderKL
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm 

# Import your custom dataloader
from data_preprocessing import get_mimic_dataloader

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # print("Loading VAE...")
    # vae_path = "/home/vault/b143dc/b143dc55/mimic/sd-vae"
    # vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16, local_files_only=True).to(device)
    print("Loading VAE...")
    vae_path = "/home/vault/b143dc/b143dc55/mimic/models/sd-vae" 
    vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16, local_files_only=True).to(device)
    vae.eval()
    vae.requires_grad_(False)

    print("Loading Text Encoder (ChexGen Native Spec)...")
    t5_path = "/home/vault/b143dc/b143dc55/mimic/models/t5-xxl"
    
    tokenizer = T5Tokenizer.from_pretrained(
        t5_path, 
        use_fast=False,        
        local_files_only=True  
    )
    
    text_encoder = T5EncoderModel.from_pretrained(
        t5_path, 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        device_map="cuda"
    )
    
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    # Get Dataloader
    print("Initializing Dataloader...")
    dataloader = get_mimic_dataloader(
        csv_path=args.csv_path, 
        h5_dir=args.h5_dir, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers
    )
    
    # Extraction Loop
    total_batches = len(dataloader)
    total_images = len(dataloader.dataset)
    print(f"Starting extraction loop! Total batches to process: {total_batches}", flush=True)
    
    with torch.no_grad():
        # Keep tqdm for interactive node testing, but use enumerate for batch logging
        for idx, (images, texts, paths) in enumerate(tqdm(dataloader, desc="Extracting")):
            images = images.to(device, dtype=torch.float16)
            
            # Extract Latents
            latents = vae.encode(images).latent_dist.sample() * 0.18215 
            
            # Extract Text
            text_inputs = tokenizer(texts, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
            text_embeddings = text_encoder(text_inputs.input_ids)[0]
            
            # Save to disk
            for b in range(images.shape[0]):
                # filename = os.path.basename(paths[b]).replace(".jpg", ".pt")
                filename = os.path.splitext(os.path.basename(paths[b]))[0] + ".pt"
                save_path = os.path.join(args.save_dir, filename)
                torch.save({
                    "latent": latents[b].cpu(),
                    "text_embedding": text_embeddings[b].cpu()
                }, save_path)
            
            # Periodic Progress Logging for SLURM logs
            if (idx + 1) % 10 == 0 or (idx + 1) == total_batches:
                processed_imgs = min((idx + 1) * args.batch_size, total_images)
                percentage = (processed_imgs / total_images) * 100
                print(f"[PROGRESS] Batch {idx + 1}/{total_batches} | Images: {processed_imgs}/{total_images} ({percentage:.2f}%) processed and saved.", flush=True)
                
    print("FINISHED! All features extracted and safely cached.", flush=True)
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and cache latents/text embeddings.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to mimic_single_label.csv")
    parser.add_argument("--h5_dir", type=str, required=True, help="Directory containing h5 shards")
    parser.add_argument("--save_dir", type=str, required=True, help="Where to save the .pt files")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    main(args)