# Reproduction
 
All commands are executed from the `residual_error_moe/` directory. The following
paths are referenced throughout:
 
```bash
MIMIC=/path/to/mimic
CACHE=$MIMIC/cached
CSV=$MIMIC/mimic.csv
RUNS=$MIMIC/moe_runs
BASE=$MIMIC/default_model_all_label/base_multilabel_alpha0.5/best_model.pt
CTRL_CACHE=/path/to/consolidated_train_cache
```
 
## Data preparation
 
Build the T5 prompt bank (once):
 
```bash
python -u extract_features.py --stage text --csv_path $CSV --bank_dir $CACHE/text_bank --orderings 2 --max_drop 2
```
 
Encode the VAE latents, one cache per shard:
 
```bash
python -u extract_features.py --stage latents --csv_path $CSV --h5_dir $MIMIC/train --shards mimic_images_shard0000.h5 --save_dir $CACHE/cache_s0000 --split TRAIN --manifest_tag s0000
python -u extract_features.py --stage latents --csv_path $CSV --h5_dir $MIMIC/train --shards mimic_images_shard0001.h5 --save_dir $CACHE/cache_s0001 --split TRAIN --manifest_tag s0001
python -u extract_features.py --stage latents --csv_path $CSV --h5_dir $MIMIC/train --shards mimic_images_shard0002.h5 --save_dir $CACHE/cache_s0002 --split TRAIN --manifest_tag s0002
python -u extract_features.py --stage latents --csv_path $CSV --h5_dir $MIMIC/val --save_dir $CACHE/cache_val --split VALIDATE --manifest_tag val
```
 
Verify that the training and validation partitions are image-disjoint:
 
```bash
python -u extract_features.py --stage verify --verify_dirs $CACHE/cache_s0000 $CACHE/cache_s0001 $CACHE/cache_s0002 $CACHE/cache_val
```
 
## Baseline
 
Train the frozen base DiT:
 
```bash
python -u train_base.py --cached_img_path $CACHE/cache_s0000 $CACHE/cache_s0001 $CACHE/cache_s0002 --val_cached_path $CACHE/cache_val --bank_dir $CACHE/text_bank --checkpoint_path $MIMIC/default_model_all_label/base_multilabel_alpha0.5 --epochs 100 --physical_batch_size 16 --accumulation_steps 4 --lr 1e-4 --weight_decay 0.0 --warmup_steps 500 --ema_decay 0.999 --balance_alpha 0.5 --finding_dropout 0.10 --cfg_dropout 0.10 --max_drop 2 --num_workers 8
```
 
Train the shared-adapter floor (no routing):
 
```bash
python -u train_moe_stage0.py --base_checkpoint $BASE --cached_img_path $CACHE/cache_s0000 $CACHE/cache_s0001 $CACHE/cache_s0002 --val_cached_path $CACHE/cache_val --bank_dir $CACHE/text_bank --checkpoint_path $RUNS/stage0_plain_r64 --moe_layers 11 --rank 64 --epochs 40 --residual_weight_pow 0 --guard_beta 0
```
 
## Gradient-conflict probe
 
```bash
python -u gradient_conflict_probe.py --base_checkpoint $BASE --cached_img_path $CACHE/cache_s0000 $CACHE/cache_s0001 $CACHE/cache_s0002 --bank_dir $CACHE/text_bank --moe_layer 11 --n_tokens 400 --k 4
```
 
## Routing experiments
 
Each is a fixed-count routed MoE (four experts) that differs only in the routing
signal.
 
Oracle routing on the true residual:
 
```bash
python -u train_moe_stage1_oracle.py --base_checkpoint $BASE --cached_img_path $CACHE/cache_s0000 $CACHE/cache_s0001 $CACHE/cache_s0002 --val_cached_path $CACHE/cache_val --bank_dir $CACHE/text_bank --checkpoint_path $RUNS/stage1_routed_r64_e4 --moe_layer 11 --rank 64 --n_experts 4 --top_k 1 --epochs 30
```
 
Routing on the hidden state:
 
```bash
python -u train_moe_stage1_hidden.py --base_checkpoint $BASE --cached_img_path $CACHE/cache_s0000 $CACHE/cache_s0001 $CACHE/cache_s0002 --val_cached_path $CACHE/cache_val --bank_dir $CACHE/text_bank --checkpoint_path $RUNS/stage1b_hidden --moe_layer 11 --rank 64 --n_experts 4 --top_k 1 --epochs 30
```
 
Routing on the label only:
 
```bash
python -u train_moe_stage1_other.py --base_checkpoint $BASE --cached_img_path $CACHE/cache_s0000 $CACHE/cache_s0001 $CACHE/cache_s0002 --val_cached_path $CACHE/cache_val --bank_dir $CACHE/text_bank --checkpoint_path $RUNS/stage1c_label_only --student_feature label_only --moe_layer 11 --rank 64 --n_experts 4 --top_k 1 --epochs 30
```
 
Routing on the hidden state and the label:
 
```bash
python -u train_moe_stage1_other.py --base_checkpoint $BASE --cached_img_path $CACHE/cache_s0000 $CACHE/cache_s0001 $CACHE/cache_s0002 --val_cached_path $CACHE/cache_val --bank_dir $CACHE/text_bank --checkpoint_path $RUNS/stage1c_label_hidden --student_feature label_hidden --moe_layer 11 --rank 64 --n_experts 4 --top_k 1 --epochs 30
```
 
Static label-bucket routing:
 
```bash
python -u train_moe_stage1_other.py --base_checkpoint $BASE --cached_img_path $CACHE/cache_s0000 $CACHE/cache_s0001 $CACHE/cache_s0002 --val_cached_path $CACHE/cache_val --bank_dir $CACHE/text_bank --checkpoint_path $RUNS/stage1c_static_label --student_feature static_label --moe_layer 11 --rank 64 --n_experts 4 --top_k 1 --epochs 30
```
 
Routing on a learned residual prediction:
 
```bash
python -u train_moe_optionA.py --base_checkpoint $BASE --cached_img_path $CACHE/cache_s0000 $CACHE/cache_s0001 $CACHE/cache_s0002 --val_cached_path $CACHE/cache_val --bank_dir $CACHE/text_bank --checkpoint_path $RUNS/optionA_predresid --moe_layer 11 --rank 64 --n_experts 4 --top_k 1 --epochs 30 --route_dim 4 --temb_dim 128 --pred_width 512 --pred_beta 1.0
```
 
## Controls
 
The controls use the hidden-state student and differ only in the distillation
target (`--teacher_target`).
 
Label-derived target:
 
```bash
python -u train_moe_controls.py --base_checkpoint $BASE --cached_img_path $CTRL_CACHE --val_cached_path $CACHE/cache_val --bank_dir $CACHE/text_bank --checkpoint_path $RUNS/control1_positive_label --teacher_target label --log_agreement --moe_layer 11 --rank 64 --n_experts 4 --top_k 1 --epochs 30 --physical_batch_size 32 --accumulation_steps 2 --num_workers 16
```
 
Content-cluster target:
 
```bash
python -u train_moe_controls.py --base_checkpoint $BASE --cached_img_path $CTRL_CACHE --val_cached_path $CACHE/cache_val --bank_dir $CACHE/text_bank --checkpoint_path $RUNS/control1b_content_cluster --teacher_target content_cluster --log_agreement --moe_layer 11 --rank 64 --n_experts 4 --top_k 1 --epochs 30 --physical_batch_size 32 --accumulation_steps 2 --num_workers 16
```
 
Shuffled target:
 
```bash
python -u train_moe_controls.py --base_checkpoint $BASE --cached_img_path $CTRL_CACHE --val_cached_path $CACHE/cache_val --bank_dir $CACHE/text_bank --checkpoint_path $RUNS/control2_shuffle_floor --teacher_target shuffle --log_agreement --moe_layer 11 --rank 64 --n_experts 4 --top_k 1 --epochs 30 --physical_batch_size 32 --accumulation_steps 2 --num_workers 16
```
 
## Evaluation
 
Compute FID between the real test reference and the generated samples for the base
model and the routed adapter:
 
```bash
python -u medical_FID_stage1.py --base_checkpoint $BASE --adapter_checkpoint $RUNS/stage1_routed_r64_e4/best_adapter.pt --h5_path $MIMIC/test/mimic_images_shard0000.h5 --csv_path $CSV --route_mode est_residual --moe_layer 11 --rank 64 --e_max 8 --top_k 1 --num_real 10000 --target_per_class 2000 --out_json fid_oracle.json
```
 
Measure the inference-time routing entropy of the deployed adapter:
 
```bash
python -u diff_est_vs_base.py --base_checkpoint $BASE --adapter_checkpoint $RUNS/stage1_routed_r64_e4/best_adapter.pt --moe_layer 11 --rank 64 --e_max 8 --top_k 1
```
