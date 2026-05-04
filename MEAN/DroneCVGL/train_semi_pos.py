"""
train_semi_pos.py
=================
Training script for UAV1-style datasets using semi-positive pairs.

Key differences from train.py:
  - Uses UAV1DatasetTrain / UAV1DatasetEval  (data/uav1.py)
  - Calls train_one_epoch_semi_pos()  which uses IoU-weighted soft-target loss
  - Calls core/metrics/uav1.evaluate() which works with per-name GT dicts

Usage:
    conda run -n cvgl python train_semi_pos.py \
        --config config/sinkhorn_siamese.yaml \
        --data-root /path/to/UAV1 \
        --train-meta cross-area-drone2sate-train.json \
        --test-meta  cross-area-drone2sate-test.json \
        --save-root  ./uav1_checkpoint
"""

import os
import sys
import time
import shutil
import random
import argparse

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

from utils.logger import Logger
from utils.registry import build_model, build_loss
from utils.train_one_epoch import train_one_epoch_semi_pos
from data.transforms import get_transforms_train, get_transforms_val
from data.uav1 import UAV1DatasetTrain, UAV1DatasetEval
from core.metrics.uav1 import evaluate as evaluate_uav1

# Model / Loss registry — import to trigger registration
from models.siamese_network import SiameseNetwork                           
from models.siamese_network_max_avg import SiameseNetworkMaxAvg             
from models.asymmetric_network import AsymmetricNetwork                     
from models.sinkhorn_siamese_network import SinkhornSiameseNetwork         
from models.aspp import ASPPSinkhornSiameseNetwork                         
from core.loss import InfoNCE, ColBERTLoss                                  


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Config
    # -------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config", "mixer_sinkhorn_siamese.yaml")
    config = OmegaConf.load(config_path)
    print("="*60)
    print("Experiment Config")
    print(OmegaConf.to_yaml(config))
    print("="*60)

    # Derived values
    config.training.neighbour_select = config.training.batch_size
    config.training.neighbour_range  = config.training.batch_size * 2

    if os.name == "nt":
        config.training.num_workers = 0

    config.training.device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    seed = config.training.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark    = config.training.cudnn_benchmark
        torch.backends.cudnn.deterministic = config.training.cudnn_deterministic

    # -------------------------------------------------------------------------
    # Output directory
    # -------------------------------------------------------------------------
    run_name   = f"{config.model.model_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    model_path = os.path.join(args.save_root, run_name)
    os.makedirs(model_path, exist_ok=True)

    shutil.copyfile(os.path.abspath(__file__), os.path.join(model_path, "train_semi_pos.py"))
    shutil.copyfile(config_path,               os.path.join(model_path, "config.yaml"))
    sys.stdout = Logger(os.path.join(model_path, "log.txt"))

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    model      = build_model(config)
    data_cfg   = model.get_config()

    query_mean,  query_std  = data_cfg["query"]["mean"],     data_cfg["query"]["std"]
    ref_mean,    ref_std    = data_cfg["reference"]["mean"], data_cfg["reference"]["std"]
    drone_size  = (config.data.drone_img_size, config.data.drone_img_size) \
                   if isinstance(config.data.drone_img_size, int) \
                   else tuple(config.data.drone_img_size)
    sate_size   = (config.data.sat_img_size,   config.data.sat_img_size) \
                   if isinstance(config.data.sat_img_size, int) \
                   else tuple(config.data.sat_img_size)

    if config.training.grad_checkpointing:
        model.set_grad_checkpointing(True)

    if config.training.checkpoint_start is not None:
        print("Resume from:", config.training.checkpoint_start)
        ckpt = torch.load(config.training.checkpoint_start, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=False)

    model = model.to(config.training.device)

    # -------------------------------------------------------------------------
    # Transforms
    # -------------------------------------------------------------------------
    train_drone_tf, train_sate_tf = get_transforms_train(
        image_size_sat=sate_size[0],
        image_size_drone=drone_size[0],
        query_mean=query_mean, query_std=query_std,
        ref_mean=ref_mean,     ref_std=ref_std,
    )
    val_drone_tf, val_sate_tf = get_transforms_val(
        image_size_sat=sate_size[0],
        image_size_drone=drone_size[0],
        query_mean=query_mean, query_std=query_std,
        ref_mean=ref_mean,     ref_std=ref_std,
    )

    # -------------------------------------------------------------------------
    # Datasets & Dataloaders
    # -------------------------------------------------------------------------
    train_dataset = UAV1DatasetTrain(
        data_root=args.data_root,
        pairs_meta_file=args.train_meta,
        transforms_query=train_drone_tf,
        transforms_gallery=train_sate_tf,
        prob_flip=config.training.prob_flip,
        shuffle_batch_size=config.training.batch_size,
        mode=args.mode,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=not config.training.custom_sampling,
        pin_memory=True,
    )

    # ---- Eval: query (drone) ----
    query_dataset = UAV1DatasetEval(
        pairs_meta_file=args.test_meta,
        data_root=args.data_root,
        view="drone",
        mode="pos",
        transforms=val_drone_tf,
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=config.eval.batch_size_eval,
        num_workers=config.training.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    # ---- Eval: gallery (satellite) ----
    gallery_dataset = UAV1DatasetEval(
        pairs_meta_file=args.test_meta,
        data_root=args.data_root,
        view="sate",
        sate_img_dir=args.sate_dir,
        query_mode="D2S",
        transforms=val_sate_tf,
    )
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=config.eval.batch_size_eval,
        num_workers=config.training.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    # Convenience handles for GT name lists used by evaluate()
    query_names   = query_dataset.images_name
    gallery_names = gallery_dataset.images_name
    pairs_dict    = query_dataset.pairs_drone2sate_dict   # {drone_name: [sate_names]}

    if config.training.custom_sampling:
        train_dataset.shuffle()

    print(f"Train pairs       : {len(train_dataset)}")
    print(f"Query images      : {len(query_dataset)}")
    print(f"Gallery tiles     : {len(gallery_dataset)}")

    # -------------------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------------------
    loss_fn = build_loss(config)

    # -------------------------------------------------------------------------
    # AMP Scaler
    # -------------------------------------------------------------------------
    scaler = GradScaler(init_scale=2.0 ** 10) if config.training.mixed_precision else None

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------
    if config.training.decay_exclude_bias:
        decay_params, no_decay_params = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if len(p.shape) == 1 or n.endswith(".bias"):
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        wd = 0.01 if "convnext" in config.model.model_args.model_name else 1e-3
        optimizer = torch.optim.AdamW([
            {"params": decay_params,    "weight_decay": wd},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=config.training.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr)

    # -------------------------------------------------------------------------
    # LR Scheduler
    # -------------------------------------------------------------------------
    train_steps  = len(train_loader) * config.training.epochs
    warmup_steps = len(train_loader) * config.training.warmup_epochs
    print(f"Warmup steps: {warmup_steps}  |  Total train steps: {train_steps}")

    if config.training.scheduler == "polynomial":
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_training_steps=train_steps,
            lr_end=config.training.lr_end,
            power=1.5,
            num_warmup_steps=warmup_steps,
        )
    elif config.training.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=train_steps,
            num_warmup_steps=warmup_steps,
        )
    elif config.training.scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps)
    else:
        scheduler = None

    # -------------------------------------------------------------------------
    # Zero-shot Evaluation
    # -------------------------------------------------------------------------
    if config.zero_shot:
        print(f"\n{30*'-'}[ Zero Shot ]{30*'-'}")
        results = evaluate_uav1(
            config, model, query_loader, gallery_loader,
            query_names, gallery_names, pairs_dict,
            ranks=[1, 5, 10], cleanup=True,
        )
        print(results)

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    print("\nStarting Training...")
    best_score = 0.0

    for epoch in range(1, config.training.epochs + 1):
        print(f"\n{30*'-'}[ Epoch: {epoch} ]{30*'-'}")

        train_loss = train_one_epoch_semi_pos(
            config, model, train_loader,
            loss_fn, optimizer, scheduler, scaler,
        )
        print(f"Epoch: {epoch} | Train Loss = {train_loss:.4f}"
              f" | LR = {optimizer.param_groups[0]['lr']:.6f}")

        # ---- Periodic Evaluation ----
        if (epoch % config.eval.eval_every_n_epoch == 0) or epoch == config.training.epochs:
            print(f"\n{30*'-'}[ Evaluate ]{30*'-'}")

            results = evaluate_uav1(
                config, model, query_loader, gallery_loader,
                query_names, gallery_names, pairs_dict,
                ranks=[1, 5, 10], cleanup=True,
            )

            r1 = results["r1"]
            if r1 > best_score:
                best_score = r1
                ckpt_path  = os.path.join(
                    model_path, f"weights_e{epoch:02d}_{r1:.4f}.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"New best R@1: {r1:.4f} — checkpoint saved → {ckpt_path}")

        # ---- Reshuffle for next epoch ----
        if config.training.custom_sampling:
            train_dataset.shuffle()

    # ---- Save final weights ----
    final_path = os.path.join(model_path, "weights_end.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Best R@1: {best_score:.4f}")
    print(f"Final weights → {final_path}")
