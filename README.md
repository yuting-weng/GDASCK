## 1. Introduction

This is a conditional generation framework for reaction-state data. The goal is to generate data that can be used to train downstream error-regression DNN models while preserving physical plausibility. The project currently uses a default three-stage training flow and supports a full loop from training and export to distribution comparison and evaluation.

Core goals:

- Keep generated samples as close as possible to the real data distribution
- Improve the generalization ability of downstream error-regression DNN models
- Ensure basic physical constraints during training and generation

---

## 2. Directory Structure

- `train.py`: unified command entry
- `configs/default.yaml`: default configuration
- `src/data`: data loading, splitting, BCT + normalization transforms
- `src/models`: Generator / Critic / QualityDNN
- `src/trainers`: GAN and quality model training logic
- `src/oracle`: Cantera single-step ground-truth interface
- `src/eval`: smoke tests, export, distribution visualization, capacity re-check scripts
- `dataset`: `.npy` files for training and evaluation
- `mechanism`: Cantera mechanism files
- `outputs`: all runtime outputs

---

## 3. Environment Setup (Windows)

### 3.1 Prerequisites

- OS: Windows
- Python: 3.10+ recommended
- GPU: CUDA available recommended (CPU is also supported)
- Conda (recommended)

### 3.2 Create and Activate Environment (recommended)

```powershell
conda create -n gan python=3.10 -y
conda activate gan
```

### 3.3 Install Dependencies

```powershell
pip install numpy scipy matplotlib scikit-learn pyyaml tqdm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install cantera
```

### 3.4 Verify Environment

```powershell
python -c "import torch, cantera, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'cantera', cantera.__version__)"
```

---

## 4. Data Preparation Requirements

The default configuration uses:

- Input data: `dataset/input_sample.npy`
- Regression target: `dataset/output_sample.npy`
- Mechanism file: `mechanism/.yaml`

Data requirements:

- `.npy` must be 2D arrays, with the first dimension as sample count
- `input_sample.npy` and `output_sample.npy` must have the same number of samples
- By default, input dimension 1 (temperature) is standardized only, without BCT (`disable_input_dim0_bct: true`)

---

## 5. Configuration Guide (default: `configs/default.yaml`)

Key parameters:

- `data`
  - `npy_path`: GAN training input
  - `batch_size`, `val_ratio`, `subset_size`
  - `condition_dim`: condition dimension
- `transform`
  - `use_bct`, `standardize`
  - `disable_input_dim0_bct`
- `model`
  - `latent_dim`
  - `generator_hidden_dims` / `critic_hidden_dims` / `quality_hidden_dims`
  - `generator.condition_encoder.enabled`
  - `critic.minibatch_discrimination.enabled`
- `optim`
  - `lr_g`, `lr_c`, `lr_quality`
- `train`
  - `use_three_stage`
  - `epochs_gan`, `epochs_quality`, `n_critic`
  - `wgan_gp_lambda`
  - `physics_species_bounds.enabled`
  - `three_stage.loss_balance.lambda_quality/lambda_phys/lambda_wgan/lambda_cond`
- `quality`
  - `regression_input_path`, `regression_target_path`
  - `oracle.mechanism_path/time_step/reference_pressure`
  - `hybrid.w_classifier/w_regression`
- `generate`
  - `target_size`, `sample_batch_size`
  - `output_path`
  - `filter.enable_double_step`, `filter.enable_qdot_screen`

---

## 6. Command Overview (`train.py`)

Unified format:

```powershell
python train.py --config <config_file> --device <auto|cpu|cuda> <subcommand> [args]
```

Subcommands:

- `train_gan`: train GAN (three-stage by default)
- `train_quality_dnn`: train quality model only
- `generate_dataset`: generate samples from a checkpoint and export plots

---

## 7. End-to-End Reproducible Workflow

All examples below are executed from the project root, and `--device cuda` is recommended.

### Step 1: Train Main Model

```powershell
python train.py --config configs/exp_modules_off_baseline.yaml --device cuda train_gan
```

For small-sample debugging:

```powershell
python train.py --config configs/exp_modules_off_baseline.yaml --device cuda train_gan --subset_size 4096
```

### Step 2: Export Generated Data

```powershell
python train.py --config configs/exp_modules_off_baseline.yaml --device cuda generate_dataset --gan_checkpoint outputs/train_gan_xxx/generator.pt --transform_stats outputs/train_gan_xxx/transform_stats.npz --target_size 60000
```

Notes:

- `--gan_checkpoint` points to the trained `generator.pt`
- `--transform_stats` should come from the same training run
- If not explicitly enabled, filtering remains disabled by default

### Step 3: Train Quality Model Only (optional)

```powershell
python train.py --config configs/exp_modules_off_baseline.yaml --device cuda train_quality_dnn --mode hybrid
```

Available modes: `classifier` / `error_regression` / `hybrid`.

---

## 8. Output Artifacts

Each run creates outputs in `outputs/<command>_<timestamp>/`. Common files:

- Configuration and transforms
  - `config_snapshot.json`
  - `transform_stats.npz`
  - `reg_input_transform_stats.npz`
  - `reg_target_transform_stats.npz`
- Model weights
  - `generator.pt`, `critic.pt`
  - `quality_regressor_pretrain.pt`, `quality_classifier_joint.pt`
- Logs
  - `gan_train_three_stage.jsonl` or `gan_train.jsonl`
  - `quality_*.jsonl`
- Generated data and plots
  - `generated/*.npy`
  - `generated/generation_summary.json`
  - `generated/plots/feature_hist_compare.png`
  - `generated/plots/pca2_compare.png`
