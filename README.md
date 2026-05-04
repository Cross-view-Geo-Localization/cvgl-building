# MEAN
Multilevel Embedding and Alignment Network With Consistency and Invariance Learning for Cross-View Geo-Localization

# Preparation for pre-training with University-Release & University-Release fine-tuning

## Pip dependencies

1. Prepare a python environment, e.g.:
```shell script
$ conda create -n mean python=3.11.5 -y
$ conda activate mean
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

# Fine-tuning with University-Release and Evaluation on SUES200
## Configuration
For detailed configuration settings, please see [`FINETUNE_CONF.md`](./FINETUNE_CONF.md).
## Fine-tuning
```shell script
CUDA_VISIBLE_DEVICES=0 python3 MEAN/DroneCVGL/train.py --config MEAN_siamese.yaml
```
The training checkpoints will be saved at: `./university_checkpoint/model_name/yymmdd_id/weights_e{ep}_{recall_1}.pth`

## Evaluation on SUES200
```
CUDA_VISIBLE_DEVICES=0 python3 MEAN/DroneCVGL/eval_script/eval_sues200.py --config MEAN_siamese.yaml
```

## Visualize Retrieval Results

To evaluate the model and export retrieval results (Top-K) to a `.csv` file, use the evaluation script located in the visualization directory. Run the following command:

```shell script
python3 MEAN/DroneCVGL/visualize/eval_and_get_topk.py --config MEAN_siamese.yaml
```

After generating the retrieval results, you can use the provided Jupyter Notebook to visualize the embeddings and analyze the model's performance:

1. Open the notebook: [`MEAN/DroneCVGL/visualize/visualize_embedding.ipynb`](MEAN/DroneCVGL/visualize/visualize_embedding.ipynb).
2. Run the cells to generate visual plots of the embedding space and qualitative retrieval results.