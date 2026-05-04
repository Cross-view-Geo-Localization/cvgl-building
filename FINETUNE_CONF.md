# Configuration Guide for MEAN Architecture

This project uses a YAML configuration file to manage training, evaluation, and dataset parameters. Below is a detailed breakdown of each section.

## Logging (`logging`)

Manages experiment tracking and identification.

| Parameter     | Type   | Description |
|---------------|--------|-------------|
| `project_name` | String | Name of the overarching project (e.g., "Drone CVGL") |
| `task_name`    | String | Name of the specific experiment or run (e.g., "Train SiameseNetwork_MEAN") |

## Model Architecture (`model`)

Defines the neural network architecture, backbone, and specific arguments for the MEAN modules (DEG, DSAM, FAT, MSF).

| Parameter                  | Type      | Description |
|----------------------------|-----------|-------------|
| `model_path`               | String    | Directory path to save or load model checkpoints |
| `model_name`               | String    | Base architecture wrapper (e.g., "SiameseNetwork_MEAN") |
| `model_args.model_name`    | String    | Specific backbone (e.g., "hgnetv2_b1.ssld_stage1_in22k_in1k") |
| `model_args.pretrained`    | Boolean   | Whether to load pretrained weights |
| `model_args.img_size`      | Integer   | Input image resolution |
| `model_args.num_classes`   | Integer   | Number of unique geographic identities (e.g., 701 for University-1652) |
| `model_args.block`         | Integer   | Number of views/partitions (set to 2 for DEG module) |
| `model_args.return_f`      | Boolean   | Return pre-classifier embeddings for metric-learning losses |
| `model_args.deg_dropout`   | Float     | Dropout rate for DEG module output |
| `model_args.proj_hid_mult` | Integer   | Hidden channel multiplier in DSAM (e.g., 2) |
| `model_args.proj_out_dim`  | Integer   | Output dimension of SparseMLP1D and FAT |
| `model_args.num_proj_layers` | Integer | Number of Conv1d layers in SparseMLP1D |
| `model_args.fat_temperature` | Float   | Softmax temperature for FAT module |
| `model_args.msf_out_dim`   | Integer   | Final output channel dimension of MSF module |

## Loss Function (`loss`)

The MEAN architecture uses a multi-loss strategy.

| Parameter | Type | Description |
|-----------|------|-------------|
| `losses.infoNCE.loss_name` | String | Global contrastive loss (e.g., "InfoNCE") |
| `losses.infoNCE.loss_args.label_smoothing` | Float | Label smoothing for global InfoNCE (e.g., 0.1) |
| `losses.DSA_loss.loss_name` | String | Contrastive loss on MSF-aligned features (e.g., "InfoNCE") |
| `losses.DSA_loss.loss_args.label_smoothing` | Float | Label smoothing for DSA loss (typically 0.0) |
| `losses.Triplet.loss_name` | String | Metric-learning loss on DEG part embeddings (e.g., "Triplet") |
| `losses.Triplet.loss_args.margin` | Float | Margin for Triplet loss (e.g., 0.3) |

## Data Preparation (`data`)

Handles dataset selection and input dimensions for drone-to-satellite matching.

| Parameter         | Type   | Description |
|-------------------|--------|-------------|
| `img_size`        | Integer | Base image size |
| `sat_img_size`    | List   | Target dimensions `[H, W]` for satellite images |
| `drone_img_size`  | List   | Target dimensions `[H, W]` for drone images |
| `dataset`         | String | Dataset split (`"U1652-D2S"` or `"U1652-S2D"`) |
| `data_folder`     | String | Root dataset directory (e.g., `"./data/University-Release"`) |

## Evaluation (`eval`)

Parameters for validation during training.

| Parameter              | Type      | Description |
|------------------------|-----------|-------------|
| `batch_size_eval`      | Integer   | Batch size for evaluation |
| `eval_every_n_epoch`   | Integer   | Evaluation frequency |
| `normalize_features`   | Boolean   | Normalize features before similarity calculation |
| `eval_reference_n`     | Integer   | Number of reference images (-1 = full gallery) |

## Training Parameters (`training`)

### General Training Settings

| Parameter            | Type      | Description |
|----------------------|-----------|-------------|
| `epochs`             | Integer   | Total training epochs |
| `batch_size`         | Integer   | Batch size per GPU (actual = 2 × batch_size in Siamese) |
| `handcraft_model`    | Boolean   | Must be `True` for MEAN |
| `return_f`           | Boolean   | Must match `model_args.return_f` |
| `mixed_precision`    | Boolean   | Enable FP16 training |
| `custom_sampling`    | Boolean   | Enable custom sampler |
| `sim_sample`         | Boolean   | Enable hard negative sampling |
| `seed`               | Integer   | Random seed |
| `verbose`            | Boolean   | Detailed training logs |

### Loss Weights

| Parameter          | Type   | Description |
|--------------------|--------|-------------|
| `weight_infonce`   | Float  | Weight for global InfoNCE loss |
| `weight_cls`       | Float  | Weight for classification loss |
| `weight_triplet`   | Float  | Weight for Triplet loss |
| `weight_dsa`       | Float  | Weight for DSA (MSF) loss |

### Optimizer & Learning Rate

| Parameter             | Type      | Description |
|-----------------------|-----------|-------------|
| `lr`                  | Float     | Base learning rate |
| `scheduler`           | String    | `"cosine"`, `"polynomial"`, `"constant"`, or `null` |
| `warmup_epochs`       | Integer   | Warmup epochs |
| `lr_end`              | Float     | Final LR (for polynomial scheduler) |
| `clip_grad`           | Float/Null| Gradient clipping norm |
| `decay_exclude_bias`  | Boolean   | Exclude biases from weight decay |
| `grad_checkpointing`  | Boolean   | Gradient checkpointing |

### Hardware & System

| Parameter             | Type    | Description |
|-----------------------|---------|-------------|
| `gpu_ids`             | List    | GPUs to use (e.g., `[0]`) |
| `device`              | String  | Device (`"cuda"`) |
| `num_workers`         | Integer | DataLoader workers (0 on Windows) |
| `cudnn_benchmark`     | Boolean | Enable cuDNN benchmark |
| `cudnn_deterministic` | Boolean | Deterministic mode |

### Augmentation & Checkpoints

| Parameter            | Type      | Description |
|----------------------|-----------|-------------|
| `prob_flip`          | Float     | Probability of horizontal flip |
| `checkpoint_start`   | String/Null | Path to resume weights (`null` = from scratch) |
| `zero_shot`          | Boolean   | Run evaluation before first epoch |

---

**Note**: Make sure `handcraft_model` and `return_f` are set correctly for the MEAN architecture to function properly.