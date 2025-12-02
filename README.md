# Protein Folding 2D

Predicting DSSP8 secondary structure for the PS4 dataset using a mix of fine-tuned ESM2 encoders, lightweight transformers over pre-computed ESM embeddings, and CRF/contact-map variants. This repository started from the Lightning-Hydra template but now contains project-specific configs, scripts, and data utilities for the competition workflow.

## Model Snapshot

| Config | Input | Core Model | Notes | Leaderboard |
| :-- | :-- | :-- | :-- | :-- |
| `data=protein_1` / `model=protein_1` | Raw sequence | ESM2-t33 encoder with top 4 layers unfrozen + MLP head | CrossEntropy over 9 DSSP8 states (+pad) | ~0.745
| `data=protein_2` / `model=protein_2` | Sharded ESM2 token embeddings (5120-dim) | 12-layer Transformer (16 heads, RoPE, warmup-cosine LR) | Trains from frozen embeddings | ~0.784
| `data=protein_2_res` / `model=protein_2_crf_res` | Sharded embeddings + residue IDs | 4-layer Transformer + CRF + residual blocks | Best offline scores, used for submissions | —
| `data=protein_contact` / `model=protein_contact` | Sharded embeddings + residue IDs + contact map | Sequence + pairwise heads on contact maps | Default `train.yaml` selection | —

`num_classes` is 10 everywhere (9 DSSP8 classes plus padding index 0). See `configs/model/*.yaml` and `configs/data/*.yaml` for full hyperparameters.

## Repository Map

- `src/data/`: Lightning data modules for raw sequences, sharded embeddings, and contact maps.
- `src/models/`: Lightning modules plus transformer/CRF/contact heads under `components/`.
- `configs/`: Hydra configs for data, models, trainer, callbacks, and paths (default train uses `protein_contact`).
- `scripts/`: Utilities for embedding regeneration, contact-map creation, and submission generation.
- `data/`: Example artifacts (`ps4_data.csv`, `test/test.tsv`, sharded embeddings, contact maps).

## Setup

```bash
# from repo root
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# or: conda env create -f environment.yaml

# optional: export PROJECT_ROOT to keep hydra paths stable
export PROJECT_ROOT=$(pwd)
```

PyTorch installation may need to follow the instructions for your CUDA setup: https://pytorch.org/get-started/.

## Data Expectations

- `data/ps4_data.csv`: training metadata with columns `chain_id`, `first_res`, `input` (AA sequence), `dssp8` (labels).
- `data/test/test.tsv`: competition test IDs in `chain_aa3_position` format.
- `data/esm2_token_embeddings_sharded_fp16_sorted/`: sharded token embeddings (`.pt` files with `chain_ids`, `lengths`, `embeddings`).
- `data/contact_maps/*.pt`: optional contact-map tensors produced by `scripts/compute_contact_maps.py`.

If you need to rebuild assets:
- Token embeddings for the test set: `python3 scripts/regenerate_embeddings.py` (updates `data/esm2_embeddings_test_fixed.pt`).
- Contact maps from mmCIF files: `python3 scripts/compute_contact_maps.py --csv_path data/ps4_data.csv --mmcif_dir data/mmcif --output_dir data/contact_maps`.

## Training

Train with Hydra overrides from the repo root (all commands assume the `PROJECT_ROOT` env var or the `.project-root` file is present):

```bash
# Contact-map model (default)
python3 src/train.py

# Fine-tune ESM2 encoder on raw sequences
python3 src/train.py data=protein_1 model=protein_1 trainer.max_epochs=5

# Transformer on frozen sharded embeddings
python3 src/train.py data=protein_2 model=protein_2 trainer.max_epochs=10

# Residual Transformer + CRF on embeddings + residue IDs
python3 src/train.py data=protein_2_res model=protein_2_crf_res trainer.max_epochs=10
```

Common flags:
- Override data locations via `paths.data_dir=/custom/data/`.
- Enable testing after training with `test=True`.
- Switch loggers, callbacks, and trainers through the configs in `configs/logger`, `configs/callbacks`, and `configs/trainer`.

## Inference / Submissions

`ProteinResCRFLitModule` is used for the latest submissions. Adjust the paths at the top of `scripts/generate_submission_3.py` and run:

```bash
python3 scripts/generate_submission_3.py
```

The script expects:
- A checkpoint (`CHECKPOINT_PATH`) from your training run.
- Test embeddings (`EMBEDDINGS_PATH`, e.g., `data/esm2_embeddings_test_fixed.pt`).
- The original `data/test/test.tsv` to map predictions back to IDs.
- It writes a tab-separated file under `predictions/`.

Older submission flows exist in `generate_submission_1.py` and `generate_submission_2.py` if you need to reproduce previous baselines.

## Testing

Light smoke tests from the template remain available:

```bash
pytest
```

They cover MNIST examples only; adjust or extend them if you want automated checks for the protein pipelines.

## Notes

- The project still uses Hydra/Lightning conventions from the original template; configs drive most behavior.
- Padding index is 0 for inputs and labels; metrics ignore padding via `ignore_index=0`.
- Sharded embedding datasets stream tensors lazily to keep memory usage low; tune `shard_cache_size` in the data configs when training on different hardware.
