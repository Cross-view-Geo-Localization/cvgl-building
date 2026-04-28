import os
import argparse
import sys
import torch
from torch.utils.data import DataLoader
import gc
import csv
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf



project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import using the full module path
from DroneCVGL.data.sues200 import SUES200DatasetEval, get_transforms
from DroneCVGL.core.metrics.sues200 import compute_mAP
from DroneCVGL.utils.predict import predict, ForwardMode
from DroneCVGL.utils.registry import build_model

from DroneCVGL.models.siamese_network import SiameseNetwork
from DroneCVGL.models.asymmetric_network import AsymmetricNetwork
from DroneCVGL.models.sinkhorn_siamese_network import SinkhornSiameseNetwork
from DroneCVGL.models.siamese_network_with_pretrained_model import SiameseNetworkWithPretrainedModel

from DroneCVGL.core.loss import InfoNCE

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pretrain.models import load_sparse_checkpoint_to_dense


def evaluate(config,
             model,
             query_loader,
             ref_loader,
             query_paths,
             ref_paths,
             csv_out="retrieval_result.csv",
             ranks=[1, 5, 10],
             cleanup=True):
    print("Extract Features:")
    img_features_query, ids_query = predict(config, model, query_loader, mode=ForwardMode.QUERY)
    img_features_ref, ids_ref = predict(config, model, ref_loader, mode=ForwardMode.REFERENCE)

    gl = ids_ref.cpu().numpy()
    ql = ids_query.cpu().numpy()

    print("Compute Scores:")
    CMC = torch.IntTensor(len(ids_ref)).zero_()
    ap = 0.0

    output_dir = os.path.dirname(csv_out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(csv_out, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        header = ["query_path"] + [f"top_{i}_ref" for i in range(1, 11)] + ["correct_ref_path"]
        csv_writer.writerow(header)

        for i in tqdm(range(len(ids_query)), desc="Evaluate TopK"):
            ap_tmp, CMC_tmp, sorted_index = eval_query(img_features_query[i], ql[i], img_features_ref, gl)

            if CMC_tmp[0] == -1:
                continue

            top10_idx = sorted_index[:10]
            top10_paths = [ref_paths[idx] for idx in top10_idx]

            good_indices = np.argwhere(gl == ql[i]).flatten()
            correct_path = ref_paths[good_indices[0]] if len(good_indices) > 0 else "None"

            row = [query_paths[i]] + top10_paths + [correct_path]
            csv_writer.writerow(row)

            CMC = CMC + CMC_tmp
            ap += ap_tmp

    AP = ap / len(ids_query) * 100
    CMC = CMC.float() / len(ids_query)

    top1 = max(1, round(len(ids_ref) * 0.01))

    metrics = [f"Recall@{i}: {CMC[i-1]*100:.4f}" for i in ranks]
    metrics.append(f"Recall@top1: {CMC[top1]*100:.4f}")
    metrics.append(f"AP: {AP:.4f}")

    print(' - '.join(metrics))

    if cleanup:
        del img_features_query, ids_query, img_features_ref, ids_ref
        gc.collect()

    return CMC[0].item()


def eval_query(qf, ql, gf, gl):
    score = gf @ qf.unsqueeze(-1)
    score = score.squeeze().cpu().numpy()

    index = np.argsort(score)[::-1]
    good_index = np.argwhere(gl == ql)
    junk_index = np.argwhere(gl == -1)

    ap, cmc = compute_mAP(index, good_index, junk_index)
    return ap, cmc, index


def main():
    parser = argparse.ArgumentParser(description="Visualize and get Top-K results")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.config) and not args.config.startswith("."):
        config_path = os.path.join(script_dir, "..", "config", args.config)
    else:
        config_path = args.config
    config = OmegaConf.load(config_path)

    config.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if os.name == 'nt':
        config.training.num_workers = 0

    query_folder = 'data/SUES-200-512x512/drone_view_512'
    ref_folder = 'data/SUES-200-512x512/satellite-view'

    print("\nModel:", config.model.model_name)
    model = build_model(config)

    if "sparse" in config.model and config.model.sparse is False:
        ckpt = load_sparse_checkpoint_to_dense(config.training.checkpoint_start, config.model.model_args.model_name)
        model.load_state_dict(ckpt, strict=False)
        print(f"[load_pretrained_hgnetv2_from_sparse] Loaded weights from {config.training.checkpoint_start}")
    elif config.training.checkpoint_start is not None:
        print("Start from:", config.training.checkpoint_start)
        ckpt = torch.load(config.training.checkpoint_start, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)

    if torch.cuda.device_count() > 1 and len(config.training.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.training.gpu_ids)

    model = model.to(config.training.device)

    data_config = model.get_config()
    print(data_config)
    query_mean, query_std = data_config['query']['mean'], data_config['query']['std']
    query_img_size = (config.data.drone_img_size, config.data.drone_img_size)

    val_transforms, _train_sat_transforms, _train_drone_transforms = get_transforms(
        img_size=query_img_size[0],
        mean=query_mean,
        std=query_std)

    query_dataset = SUES200DatasetEval(
        data_folder=query_folder,
        mode='drone',
        transforms=val_transforms)

    query_dataloader = DataLoader(
        query_dataset,
        batch_size=config.eval.batch_size_eval,
        num_workers=config.training.num_workers,
        shuffle=False,
        pin_memory=True)

    ref_dataset = SUES200DatasetEval(
        data_folder=ref_folder,
        mode='satellite',
        transforms=val_transforms,
        sample_ids=query_dataset.get_sample_ids(),
        ref_n=config.eval.eval_reference_n)

    ref_dataloader = DataLoader(
        ref_dataset,
        batch_size=config.eval.batch_size_eval,
        num_workers=config.training.num_workers,
        shuffle=False,
        pin_memory=True)

    query_image_paths = query_dataloader.dataset.images
    ref_image_paths = ref_dataloader.dataset.images

    output_path = os.path.join(script_dir, '..', 'retrieval_result', f"{config.model.model_name}_SUES200_top10.csv")

    evaluate(
        config=config,
        model=model,
        query_loader=query_dataloader,
        ref_loader=ref_dataloader,
        query_paths=query_image_paths,
        ref_paths=ref_image_paths,
        csv_out=output_path)


if __name__ == '__main__':
    main()
