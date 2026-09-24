# Online update of expert assignment for long tail generation

A Dynamic Mixture of Experts Approach for Rare Pathology Synthesis.

Code for the master thesis by Edoardo Berardi Vittur (Friedrich-Alexander-
Universität Erlangen-Nürnberg). The task is text-conditioned chest X-ray generation on
MIMIC-CXR, with a focus on rare (long-tail) pathologies. A frozen diffusion
backbone is kept fixed and small Mixture-of-Experts LoRA adapters are added and
routed on top of it.

The theoretical background, the method derivation and the full results and
analysis are in the thesis. This repository holds the runnable code, organised
into two folders that correspond to the two lines of work described there.

## Repository structure

### `multi_lable_code/`: main method

The primary contribution. A frozen DiT backbone with LoRA experts injected at one
transformer block, routed in content space and grown online through dynamic
expert spawning. The folder covers the full pipeline: VAE latent caching, base
model training, MoE training with clustering and online spawning, synthetic
dataset generation and evaluation by FID and by downstream classification
(Macro-F1).

### `residual_error_moe/`: residual-routing study

The exploratory study behind the main method. It tests whether the base model's
residual-error direction can drive expert routing and whether any
inference-available signal (the hidden state, the label or a learned residual
prediction) can recover the routing that the true residual would produce. Each
signal is compared against a shared-adapter floor and two controls. This is the
investigation summarised in the Residual-Routing Study results (Table 2) of the
thesis.

Each folder has its own reproduction.md with the commands to run its experiments.

## Data and models

Both folders use MIMIC-CXR (the image shards and the multi-label CSV) and local
weights for the VAE, the T5 text encoder and a domain DenseNet-121 used by the
FID metric. The concrete paths are set inside each folder's scripts and its
`reproduction.md`.
