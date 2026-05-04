# cvgl-building
Cross View Geo Localization for Drone-Satellite view of building

# Preparation for pre-training with University-Release & University-Release fine-tuning

## Pip dependencies

1. Prepare a python environment, e.g.:
```shell script
$ conda create -n spark python=3.11.5 -y
$ conda activate spark
```

2. Install requirements:
```shell script
$ pip install -r requirements.txt 
```



## University-Release preparation

Prepare the University-Release dataset
- assume the dataset is in `/cvgl-building/data/University-Release`
- it should look like this:
```
/cvgl-building/data/University-Release/:
    test/:
        gallery_drone/: 
            0/:
                a_lot_images.jpeg
            1/:
                a_lot_images.jpeg
        gallery_satellite/:
            0/:
                a_lot_images.jpeg
            1/:
                a_lot_images.jpeg
        query_drone/: 
            0/:
                a_lot_images.jpeg
            1/:
                a_lot_images.jpeg
        query_satellite/:
            0/:
                a_lot_images.jpeg
            1/:
                a_lot_images.jpeg
    train/:
        drone/:
            0839/:
                a_lot_images.jpeg
            0842/:
                a_lot_images.jpeg
        satellite:
            0839/:
                0839.jpeg
            0842/:
                0839.jpeg
```

# Pretraining with University-Release
## Configuration
For detailed configuration settings, please see [`PRETRAIN_CONF.md`](./PRETRAIN_CONF.md).
## Debug on 1 GPU

Use a small batch size `--bs=32` for avoiding OOM.

```shell script
CUDA_VISIBLE_DEVICES=0 python3 SparK/pretrain/main.py \
    --exp_dir=./logs/debug \
    --ep=1 \
    --data_path=./data/University-Release \
    --model=hgnetv2_b1.ssld_stage1_in22k_in1k \
    --bs=32
```

## Pretraining HGNetv2_b1 on University-Release
```shell script
CUDA_VISIBLE_DEVICES=0 python3 SparK/pretrain/main.py \
    --exp_dir=./logs/hgnetv2_university_pretrain \
    --ep=160 \
    --data_path=./data/University-Release \
    --model=hgnetv2_b1.ssld_stage1_in22k_in1k \
    --bs=128
```

## or Continue Pretraining from a Checkpoint
```shell script
CUDA_VISIBLE_DEVICES=0 python3 SparK/pretrain/main.py \
    --exp_dir=./logs/hgnetv2_university_pretrain_from_ep160 \
    --resume_from=your_last_exp_dir/hgnetv2_b1.ssld_stage1_in22k_in1k_withdecoder.pth \
    --ep=165 \ # greater than last ep
    --data_path=./data/University-Release \
    --model=hgnetv2_b1.ssld_stage1_in22k_in1k \
    --bs=128
```

## or Pretraining on multiple-dataset
```shell script
CUDA_VISIBLE_DEVICES=0 python3 SparK/pretrain/main.py \
    --exp_dir=./logs/hgnetv2_3data_100ep \
    --ep=100 \
    --data_path ./data/University-Release ./data/University160k/ ./data/AerialExtreMatchCustomed/ \
    --model=hgnetv2_b1.ssld_stage1_in22k_in1k \
    -bs=128
```
Alternatively, you can configure the parameters in the [`SparK/pretrain/utils/arg_util.py`](SparK/pretrain/utils/arg_util.py) file and run the training process using the following command:
```shell script
CUDA_VISIBLE_DEVICES=0 python3 SparK/pretrain/main.py
```
## Results
See files in your `--exp_dir` to track your experiment:

- `<model>_withdecoder.pth`: saves model and optimizer states, current epoch, current reconstruction loss, etc.; can be used to resume pretraining; can also be used for visualization in [SparK/pretrain/viz_reconstruction.ipynb](./SparK/pretrain/viz_reconstruction.ipynb)
- `<model>_without_decoder.pth`: can be used for downstream finetuning
- `pretrain_log.txt`: records some important information such as:
    - `git_commit_id`: git version
    - `cmd`: the command of this experiment
    
    It also reports the loss and remaining pretraining time.

- `tensorboard_log/`: saves a lot of tensorboard logs including loss values, learning rates, gradient norms and more things. Use `tensorboard --logdir /path/to/this/tensorboard_log/ --port 23333` for viz.
- `stdout_backup.txt` and `stderr_backup.txt`: backups stdout/stderr.

## Pre-trained Model Visualization
To assess the reconstruction capabilities and feature quality of the pre-trained model, you can run the provided visualization notebook:

* **File:** [`SparK/pretrain/viz_reconstruction.ipynb`](SparK/pretrain/viz_reconstruction.ipynb)

This allows you to qualitatively evaluate how well the model has learned to represent the data before you begin the fine-tuning process on the Drone-to-Satellite task.


# Fine-tuning with University-Release and Evaluation on SUES200
## Configuration
For detailed configuration settings, please see [`FINETUNE_CONF.md`](./FINETUNE_CONF.md).
## Fine-tuning
```shell script
CUDA_VISIBLE_DEVICES=0 python3 SparK/DroneCVGL/train.py --config config.yaml
```
The training checkpoints will be saved at: `./university_checkpoint/model_name/yymmdd_id/weights_e{ep}_{recall_1}.pth`

## Evaluation on SUES200
```
CUDA_VISIBLE_DEVICES=0 python3 SparK/DroneCVGL/eval_script/eval_sues200.py --config config.yaml
```

## Visualize Retrieval Results

To evaluate the model and export retrieval results (Top-K) to a `.csv` file, use the evaluation script located in the visualization directory. Run the following command:

```shell script
python3 SparK/DroneCVGL/visualize/eval_and_get_topk.py --config config.yaml
```

After generating the retrieval results, you can use the provided Jupyter Notebook to visualize the embeddings and analyze the model's performance:

1. Open the notebook: [`SparK/DroneCVGL/visualize/visualize_embedding.ipynb`](SparK/DroneCVGL/visualize/visualize_embedding.ipynb).
2. Run the cells to generate visual plots of the embedding space and qualitative retrieval results.
