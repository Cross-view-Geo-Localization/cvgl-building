## Configuration

This project uses **typed-argument-parser (Tap)** for configuration.  
You can modify the default hyperparameters by passing them as command-line arguments.

---

### Environment & Paths

| Argument        | Type | Default                                             | Description                                                                 |
|----------------|------|-----------------------------------------------------|-----------------------------------------------------------------------------|
| `--exp_name`   | str  | `'Pretrain HGNetv2 on University dataset'`          | The name of the experiment.                                                 |
| `--exp_dir`    | str  | `'./SparK/pretrain/logs/hgnetv2_university'`        | Directory where logs and checkpoints will be saved. Automatically created. |
| `--data_path`  | str  | `'./data/University-Release'`                       | Path to the training dataset.                                               |
| `--init_weight`| str  | `''`                                                | Path to a checkpoint to initialize model weights only.                      |
| `--resume_from`| str  | `''`                                                | Path to resume full training (weights + optimizer + epoch).                 |

---

### Model & Architecture

| Argument        | Type  | Default                        | Description                                                                 |
|----------------|-------|--------------------------------|-----------------------------------------------------------------------------|
| `--model`      | str   | `'hgnetv2_b1.ssld_stage1...'`  | The encoder architecture to use.                                            |
| `--input_size` | int   | `224`                          | Input image resolution.                                                     |
| `--sbn`        | bool  | `True`                         | Enable Synchronized BatchNorm (disabled if not distributed).               |
| `--mask`       | float | `0.6`                          | SparK mask ratio (must be between 0 and 1).                                |

---

### Data Loading

| Argument                | Type | Default | Description                                |
|------------------------|------|---------|--------------------------------------------|
| `--bs`                 | int  | `32`    | Global batch size across all GPUs.         |
| `--dataloader_workers` | int  | `8`     | Number of CPU workers for data loading.    |

---

### Optimization & Pre-training

| Argument   | Type  | Default | Description                                                                 |
|------------|-------|---------|-----------------------------------------------------------------------------|
| `--ep`     | int   | `160`   | Total number of pre-training epochs.                                       |
| `--wp_ep`  | int   | `40`    | Number of warmup epochs.                                                   |
| `--base_lr`| float | `2e-4`  | Base learning rate (auto-scaled, see below).                              |
| `--opt`    | str   | `'lamb'`| Optimizer used for training.                                               |
| `--wd`     | float | `0.04`  | Weight decay factor.                                                       |
| `--wde`    | float | `0.2`   | Weight decay for specific layers (defaults to `--wd` if not specified).   |
| `--dp`     | float | `0.0`   | Dropout rate.                                                              |
| `--clip`   | float | `5.0`   | Gradient clipping threshold.                                              |
| `--ada`    | float | `0.0`   | Ada factor (auto: 0.95 for ResNet, 0.999 for ConvNeXt).                   |

---

### Auto-Calculated Variables

You **do not need to specify** the following parameters; they are automatically computed at runtime:

- **Learning Rate (`lr`)**: `base_lr * (global_batch_size / 256)` (Linear Scaling Rule)
- **Batch Size Per GPU (`batch_size_per_gpu`)**: `batch_size / num_gpus`