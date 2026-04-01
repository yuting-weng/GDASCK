# 框架

## 1. 简介

一个面向反应状态数据生成的条件生成框架，目标是在保持物理合理性的前提下，生成可用于下游误差回归模型训练的数据。项目当前默认使用三阶段训练流程，并支持从训练、导出到分布对比评估的完整闭环。

核心目标：

- 生成样本与真实数据分布尽量一致
- 生成样本能提升下游误差回归 DNN 的泛化能力
- 训练和生成过程满足基础物理约束

---

## 2. 模型结构与训练逻辑

### 2.1 核心模块

- `Generator`：根据噪声和条件信息生成状态向量
- `Critic`：WGAN-GP 判别器，用于提供对抗学习信号
- `QualityDNN`：
  - 二分类头：评估真实/生成分布一致性
  - 误差回归头：评估样本在 Oracle 下一步预测误差上的“难度”
- `Oracle`：基于 Cantera 的单步推进真值计算接口

### 2.2 三阶段训练（默认开启）

- 阶段 A：回归 DNN 预训练  
  使用 `dataset/input_sample.npy` 与 `dataset/output_sample.npy` 训练误差回归器。
- 阶段 B：GAN 主训练  
  训练 `generator + critic`（WGAN-GP）并联合质量分支。
- 阶段 C：融合评分引导  
  使用分类分数与误差回归分数的加权结果引导生成器优化。

默认混合评分权重：

- `quality.hybrid.w_classifier = 0.8`
- `quality.hybrid.w_regression = 0.2`

---

## 3. 目录结构

- `train.py`：统一命令入口
- `configs/default.yaml`：默认配置
- `src/data`：数据读取、划分、BCT+标准化变换
- `src/models`：Generator / Critic / QualityDNN
- `src/trainers`：GAN 与质量模型训练逻辑
- `src/oracle`：Cantera 单步真值接口
- `src/eval`：冒烟、导出、分布可视化、容量复核等评估脚本
- `dataset`：训练与评估用 `.npy` 数据
- `mechanism`：Cantera 机理文件
- `outputs`：所有运行产物目录

---

## 4. 环境部署（Windows）

### 4.1 前置条件

- 操作系统：Windows
- Python：建议 3.10+
- 建议 GPU：CUDA 可用（无 GPU 也可 CPU 运行）
- Conda（推荐）

### 4.2 创建并激活环境（推荐新建）

```powershell
conda create -n cpc_gan python=3.10 -y
conda activate cpc_gan
```

### 4.3 安装依赖

```powershell
pip install numpy scipy matplotlib scikit-learn pyyaml tqdm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install cantera
```

如果你已有可用环境（如 `D:\Anaconda\envs\peft\python.exe`），可直接使用，无需重复创建。

### 4.4 验证环境

```powershell
python -c "import torch, cantera, numpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'cantera', cantera.__version__)"
```

---

## 5. 数据准备要求

默认配置对应以下文件：

- 输入数据：`dataset/input_sample.npy`
- 回归目标：`dataset/output_sample.npy`
- 机理文件：`mechanism/Burke2012_s9r23.yaml`

数据要求：

- `.npy` 二维数组，第一维是样本数
- `input_sample.npy` 与 `output_sample.npy` 样本数需一致
- 默认输入第 1 维（温度）仅做标准化，不做 BCT（`disable_input_dim0_bct: true`）

---

## 6. 配置说明（默认配置：`configs/default.yaml`）

关键参数：

- `data`
  - `npy_path`：GAN 训练输入
  - `batch_size`、`val_ratio`、`subset_size`
  - `condition_dim`：条件维度
- `transform`
  - `use_bct`、`standardize`
  - `disable_input_dim0_bct`
- `model`
  - `latent_dim`
  - `generator_hidden_dims` / `critic_hidden_dims` / `quality_hidden_dims`
  - `generator.condition_encoder.enabled`
  - `critic.minibatch_discrimination.enabled`
- `optim`
  - `lr_g`、`lr_c`、`lr_quality`
- `train`
  - `use_three_stage`
  - `epochs_gan`、`epochs_quality`、`n_critic`
  - `wgan_gp_lambda`
  - `physics_species_bounds.enabled`
  - `three_stage.loss_balance.lambda_quality/lambda_phys/lambda_wgan/lambda_cond`
- `quality`
  - `regression_input_path`、`regression_target_path`
  - `oracle.mechanism_path/time_step/reference_pressure`
  - `hybrid.w_classifier/w_regression`
- `generate`
  - `target_size`、`sample_batch_size`
  - `output_path`
  - `filter.enable_double_step`、`filter.enable_qdot_screen`

---

## 7. 命令总览（`train.py`）

统一格式：

```powershell
python train.py --config <配置文件> --device <auto|cpu|cuda> <子命令> [参数]
```

子命令：

- `train_gan`：训练 GAN（默认三阶段）
- `train_quality_dnn`：仅训练质量模型
- `smoke_test`：端到端快速冒烟
- `generate_dataset`：使用已有 checkpoint 生成样本并输出分布图

---

## 8. 从零到可复现的一套完整流程

以下示例均在项目根目录执行，建议优先使用 `--device cuda`。

### 步骤 1：训练主模型（三阶段）

```powershell
python train.py --config configs/exp_modules_off_baseline.yaml --device cuda train_gan
```

如需小样本调试：

```powershell
python train.py --config configs/exp_modules_off_baseline.yaml --device cuda train_gan --subset_size 4096
```

### 步骤 2：导出生成数据（无筛选）

```powershell
python train.py --config configs/exp_modules_off_baseline.yaml --device cuda generate_dataset --gan_checkpoint outputs/train_gan_xxx/generator.pt --transform_stats outputs/train_gan_xxx/transform_stats.npz --target_size 60000
```

说明：

- `--gan_checkpoint` 指向训练得到的 `generator.pt`
- `--transform_stats` 建议使用同一次训练的 `transform_stats.npz`
- 未显式开启时，默认筛选项关闭

### 步骤 3：仅训练质量模型（可选）

```powershell
python train.py --config configs/exp_modules_off_baseline.yaml --device cuda train_quality_dnn --mode hybrid
```

可选模式：`classifier` / `error_regression` / `hybrid`。

---

## 9. 输出产物说明

每次运行会在 `outputs/<command>_<timestamp>/` 生成产物。常见文件：

- 配置与变换
  - `config_snapshot.json`
  - `transform_stats.npz`
  - `reg_input_transform_stats.npz`
  - `reg_target_transform_stats.npz`
- 模型参数
  - `generator.pt`, `critic.pt`
  - `quality_regressor_pretrain.pt`, `quality_classifier_joint.pt`
- 日志
  - `gan_train_three_stage.jsonl` 或 `gan_train.jsonl`
  - `quality_*.jsonl`
- 生成与图表
  - `generated/*.npy`
  - `generated/generation_summary.json`
  - `generated/plots/feature_hist_compare.png`
  - `generated/plots/pca2_compare.png`

---

## 11. 常见问题排查

- CUDA 不可用  
  使用 `python -c "import torch; print(torch.cuda.is_available())"` 检查，必要时改用 `--device cpu`。
- Cantera 相关报错  
  确认机理路径存在，并在当前环境执行 `python -c "import cantera"` 无报错。
- 生成与训练维度不一致  
  确保 `--transform_stats` 与 `--gan_checkpoint` 来自同一次训练。
- 样本筛选过严导致通过率低  
  在配置中保持 `generate.filter.enable_double_step=false`、`generate.filter.enable_qdot_screen=false` 进行基线导出。

---

## 12. 推荐执行顺序

1. 先 `smoke_test` 验证环境  
2. 再 `train_gan` 训练主模型  
3. 再 `generate_dataset` 导出样本与分布图  
4. 最后执行评估脚本做效果与容量复核

按以上流程可以完成从环境部署、模型训练、样本生成到效果评估的完整闭环。
