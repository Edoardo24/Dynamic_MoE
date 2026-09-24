# Reproduction: content-space routing MoE with dynamic expert spawning

All commands are executed from the project directory. The following paths are
referenced throughout:

```bash
MIMIC=/path/to/mimic
CSV=$MIMIC/mimic.csv
CACHE=$MIMIC/cached
BASE=$MIMIC/checkpoints/base_multilabel/best_model.pt
RUNS=$MIMIC/moe_runs
SYNTH=$MIMIC/synth
DENSENET=$MIMIC/models/densenet121-res224-mimic_ch.pt
```

## Data preparation

Encode the VAE latents for the training shards:

```bash
python -u DiT_diffusion_model.py --csv_path $CSV --h5_dir $MIMIC/train --save_dir $CACHE/cached_train --batch_size 32 --num_workers 4
```

## Base model

Train the base DiT (no MoE):

```bash
python -u train_phase0_base.py --cached_img_path $CACHE/cached_train --checkpoint_path $MIMIC/checkpoints/base_multilabel --epochs 100 --physical_batch_size 16 --accumulation_steps 4 --lr 1e-4
```

## MoE training

The MoE freezes the base and injects routed experts at one transformer block.
Tokens are clustered in content space and a learned router dispatches them.
Experts are spawned online during training.

Main configuration (block 11, shared router, HDBSCAN clustering,
performance-filtered spawning):

```bash
python -u train_phase1.py --cached_img_path $CACHE/cached_train --checkpoint_path $RUNS/shared_hdbscan_block11 --base_pretrained_path $BASE --layer_index 11 --router_mode shared --cluster_method hdbscan --hdbscan_min_cluster_size 250 --max_experts 16 --rank 16 --top_k 3 --expert_volume 4.0 --epochs 75 --physical_batch_size 32 --accumulation_steps 4 --spawn_start_epoch 3 --use_perf_filter
```

Configuration with DINO clustering features and the full offline/spawn filter
settings:

```bash
python -u train_phase1.py --cached_img_path $CACHE/cached_train --checkpoint_path $RUNS/shared_dino_hdbscan --base_pretrained_path $BASE --layer_index 11 --router_mode shared --cluster_feature dino --cluster_method hdbscan --hdbscan_min_cluster_size 250 --cluster_vae_path $MIMIC/models/sd-vae --cluster_encoder_model $MIMIC/models/rad-dino --max_experts 16 --rank 16 --top_k 3 --expert_volume 4.0 --entropy_threshold 1.0 --spawn_threshold 80000 --epochs 75 --physical_batch_size 32 --accumulation_steps 4 --spawn_start_epoch 3 --offline_lr 3e-4 --offline_epochs 8 --use_perf_filter --filter_keep_fraction 0.5 --filter_ref_timestep 500 --filter_min_experts 3
```

The reported ablations vary, from the main configuration:

- injection block: `--layer_index` (0 to 11)
- router configuration: `--router_mode` (`shared`, `no_skip`)
- clustering: `--cluster_method` (`hdbscan`, `kmeans`) and `--cluster_feature`
  (latent, `dino`)
- spawning buffer: `--use_perf_filter` with `--filter_keep_fraction`,
  `--filter_ref_timestep`, `--filter_min_experts`, versus entropy-only spawning
  via `--entropy_threshold` and `--spawn_threshold`.

## Synthetic dataset generation

Generate one synthetic set per trained model (repeat with a different
`--checkpoint` and `--out_h5` for each model compared):

```bash
python -u gen_synthetic_dataset.py --checkpoint $RUNS/shared_hdbscan_block11/dit_epoch_74.pt --out_h5 $SYNTH/shared_hdbscan_block11.h5 --per_class 5000 --guidance_scale 2.0 --num_inference_steps 25 --batch_size 32 --img_size 512
```

## Evaluation

Balanced FID between the real reference and the synthetic set:

```bash
python -u balanced_medical_FID.py --real_h5 $MIMIC/val/mimic_images_shard0000.h5 --real_csv $CSV --synth_h5 $SYNTH/shared_hdbscan_block11.h5 --densenet $DENSENET --batch_size 64 --num_workers 8
```

Downstream classification utility (real training shard augmented with the
synthetic set, with `--with_control` adding the real-duplicated control condition):

```bash
python -u classifier_synth_gain.py --real_train_h5 $MIMIC/train/mimic_images_shard0004.h5 --real_csv $CSV --real_val_h5 $MIMIC/val/mimic_images_shard0000.h5 --real_test_h5 $MIMIC/test/mimic_images_shard0000.h5 --synth_h5 $SYNTH/shared_hdbscan_block11.h5 --with_control --epochs 15 --batch_size 64 --lr 1e-4 --num_workers 8
```
