## Configuration Guide

This project relies on a YAML configuration file to manage training, evaluation, and dataset parameters. Below is a detailed breakdown of each section within the configuration file to help you customize your runs.

### Logging (`logging`)
Manages experiment tracking and identification.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `project_name` | String | Name of the overarching project (e.g., `"Drone CVGL"`). |
| `task_name` | String | Name of the specific experiment or run (e.g., `"Train HGNetv2"`). |

### Model Architecture (`model`)
Defines the neural network architecture and its specific arguments.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `model_path` | String | Directory path to save or load the model checkpoints. |
| `model_name` | String | The base architecture wrapper (e.g., `"SiameseNetworkWithPretrainedModel"`). |
| `model_args.model_name` | String | The specific backbone used (e.g., `"hgnetv2_b1.ssld_stage1_in22k_in1k"`). |
| `model_args.pretrained` | Boolean | Whether to load default pretrained weights (`True` or `False`). |
| `model_args.img_size` | Integer | Input image resolution for the model. |
| `sparse` | Boolean | Enables or disables sparse representations. |

### Loss Function (`loss`)
Configures the objective function for training.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `loss_name` | String | The loss function used (e.g., `"InfoNCE"`). |
| `loss_args.label_smoothing` | Float | Applies label smoothing to prevent overconfidence (e.g., `0.1`). |

### Data Preparation (`data`)
Handles dataset selection, location, and input dimensions.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `img_size` | Integer | Base image size for processing. |
| `sat_img_size` | List | Target dimensions `[H, W]` for satellite images. |
| `drone_img_size` | List | Target dimensions `[H, W]` for drone images. |
| `dataset` | String | The dataset split to use (Options: `'U1652-D2S'` or `'U1652-S2D'`). |
| `data_folder` | String | Path to the root dataset directory. |

### Evaluation (`eval`)
Parameters controlling the validation process during training.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `batch_size_eval` | Integer | Batch size used during the evaluation loop. |
| `eval_every_n_epoch` | Integer | Frequency of evaluation (e.g., `1` means evaluate after every epoch). |
| `normalize_features` | Boolean | If `True`, normalizes extracted feature vectors before distance calculation. |
| `eval_reference_n` | Integer | Number of reference images to use. `-1` uses all available references. |

### Training Parameters (`training`)
Controls the learning process, optimization, and hardware utilization.

#### General Training Settings
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `epochs` | Integer | Total number of training epochs. |
| `batch_size` | Integer | Batch size per GPU. *Note: Actual batch size in a Siamese setup is `2 * batch_size`.* |
| `mixed_precision` | Boolean | Enables FP16 mixed precision training to save memory and speed up computation. |
| `custom_sampling` | Boolean | Enables a custom sampler instead of standard random sampling. |
| `sim_sample` | Boolean | Enables hard negative sampling for contrastive learning. |
| `seed` | Integer | Random seed for reproducibility. |

#### Optimizer & Learning Rate
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `lr` | Float | Base learning rate (e.g., `1e-4`). |
| `scheduler` | String | Learning rate scheduling policy (`"cosine"`, `"polynomial"`, `"constant"`, or `None`). |
| `warmup_epochs` | Integer | Number of epochs to gradually warm up the learning rate. |
| `lr_end` | Float | Final learning rate (specifically used for the `"polynomial"` scheduler). |
| `clip_grad` | Float/Null | Maximum gradient norm for clipping. Use `None` to disable. |
| `decay_exclude_bias` | Boolean | If `True`, excludes biases from weight decay. |
| `grad_checkpointing` | Boolean | Trades compute for memory by saving intermediate activations. |

#### Hardware & System
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `gpu_ids` | List | Specify which GPUs to use (e.g., `[0,]`). |
| `num_workers` | Integer | Number of CPU workers for data loading. *Set to `0` if running on Windows.* |
| `cudnn_benchmark` | Boolean | Enables cuDNN benchmark to optimize convolution algorithms (set to `True` for static input sizes). |
| `cudnn_deterministic` | Boolean | Forces deterministic cuDNN operations. Can negatively impact performance if `True`. |

#### Augmentation & Checkpoints
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `prob_flip` | Float | Probability of applying a simultaneous horizontal flip to both satellite and drone images. |
| `checkpoint_start` | String/Null | Path to custom pre-trained weights to resume or initialize training. Use `null` to train from scratch. |
| `zero_shot` | Boolean | If `True`, performs an evaluation run before training begins. |

***