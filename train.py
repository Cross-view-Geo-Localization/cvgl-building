import os
import shutil
import time
import sys
import torch
import random
import numpy as np
from torch.cuda.amp import GradScaler
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import (get_constant_schedule_with_warmup,
                          get_polynomial_decay_schedule_with_warmup,
                          get_cosine_schedule_with_warmup)

from utils.logger import Logger
from utils.registry import build_model, build_loss
from utils.train_one_epoch import train_one_epoch
from data.transforms import get_transforms_train, get_transforms_val
from data.university import U1652DatasetTrain, U1652DatasetEval
from core.metrics.university import evaluate, calc_sim

# MODELS and LOSSES
from models.siamese_network import SiameseNetwork
from models.siamese_network_GeM import SiameseNetworkGeM
from models.siamese_network_max_avg import SiameseNetworkMaxAvg
from models.asymmetric_network import AsymmetricNetwork
from models.sinkhorn_siamese_network import SinkhornSiameseNetwork
from models.aspp import ASPPSinkhornSiameseNetwork
from models.self_distillation_network import SelfDistillationNetwork
from models.supersalad import SuperSALADNetwork
from models.siamese_network_with_atttention import SiameseNetworkWithAttention
from models.dac import DAC
from core.loss import InfoNCE, ColBERTLoss, IntraInfoNCE, SelfDistillationLoss

if __name__ == "__main__":
    #-----------------------------------------------------------------------------#
    # Setup                                                                       #
    #-----------------------------------------------------------------------------#
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config", "base.yaml")
    config = OmegaConf.load(config_path)
    print("="*60)
    print("Experiment Config")
    print(OmegaConf.to_yaml(config))
    print("="*60)

    # dynamic values
    config.training.neighbour_select = config.training.batch_size
    config.training.neighbour_range = config.training.batch_size * 2

    if os.name == "nt":
        config.training.num_workers = 0

    config.training.device = "cuda" if torch.cuda.is_available() else "cpu"

    if config.data.dataset == 'U1652-D2S':
        config.data.query_folder_train = './data/University-Release/train/drone'
        config.data.reference_folder_train = './data/University-Release/train/satellite'
        config.data.query_folder_test = './data/University-Release/test/query_drone'
        config.data.reference_folder_test = './data/University-Release/test/gallery_satellite'
    elif config.data.dataset == 'U1652-S2D':
        config.data.query_folder_train = './data/University-Release/train/drone'
        config.data.reference_folder_train = './data/University-Release/train/satellite'
        config.data.query_folder_test = './data/University-Release/test/query_satellite'
        config.data.reference_folder_test = './data/University-Release/test/gallery_drone'


    # Set seeds for for reproducible training
    seed = config.training.seed
    random.seed(seed)
    np.random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn_benchmark_enabled = config.training.cudnn_benchmark
        torch.backends.cudnn.deterministic = config.training.cudnn_deterministic


    model_path = f"{config.model.model_path}/{config.model.model_name}/{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(model_path, exist_ok=True)
    shutil.copyfile(os.path.abspath(__file__), "{}/train.py".format(model_path))
    shutil.copyfile(os.path.join(script_dir, "data", "transforms.py"), "{}/transforms.py".format(model_path))
    shutil.copyfile(config_path, "{}/config.yaml".format(model_path))
    sys.stdout = Logger(os.path.join(model_path, 'log.txt'))


    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
    model = build_model(config)
    data_config = model.get_config()
    query_mean, query_std = data_config['query']['mean'], data_config['query']['std']
    query_img_size = (config.data.drone_img_size, config.data.drone_img_size)
    ref_mean, ref_std = data_config['reference']['mean'], data_config['reference']['std']
    ref_img_size = (config.data.sat_img_size, config.data.sat_img_size)

    if config.training.grad_checkpointing:
        model.set_grad_checkpointing(True)

    if config.training.checkpoint_start is not None:
        print("Start from:", config.training.checkpoint_start)
        ckpt = torch.load(config.training.checkpoint_start, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)

    model = model.to(config.training.device)


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#
    train_drone_transforms, train_sat_transforms = get_transforms_train(
        image_size_sat=config.data.sat_img_size,
        image_size_drone=config.data.drone_img_size,
        query_mean=query_mean,
        query_std=query_std,
        ref_mean=ref_mean,
        ref_std=ref_std)
    val_drone_transforms, val_sat_transforms = get_transforms_val(
        image_size_sat=config.data.sat_img_size,
        image_size_drone=config.data.drone_img_size,
        query_mean=query_mean,
        query_std=query_std,
        ref_mean=ref_mean,
        ref_std=ref_std)

    train_dataset = U1652DatasetTrain(
        config.data.query_folder_train,
        config.data.reference_folder_train,
        train_drone_transforms,
        train_sat_transforms,
        config.training.prob_flip,
        config.training.batch_size)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=not config.training.custom_sampling,
        pin_memory=True)

    query_dataset_test = U1652DatasetEval(config.data.query_folder_test, "query", val_drone_transforms)
    query_dataloader_test = DataLoader(query_dataset_test, batch_size=config.eval.batch_size_eval,
                                       num_workers=config.training.num_workers, shuffle=False, pin_memory=True)

    reference_dataset_test = U1652DatasetEval(
        config.data.reference_folder_test, "reference", val_sat_transforms,
        sample_ids=query_dataset_test.get_sample_ids(),
        reference_n=config.eval.eval_reference_n)
    reference_dataloader_test = DataLoader(reference_dataset_test, batch_size=config.eval.batch_size_eval,
                                           num_workers=config.training.num_workers, shuffle=False, pin_memory=True)

    reference_dataloader_train = None
    if config.training.sim_sample:
        reference_dataset_train = U1652DatasetEval(config.data.reference_folder_train, "reference", val_sat_transforms)
        reference_dataloader_train = DataLoader(reference_dataset_train, batch_size=config.eval.batch_size_eval,
                                                num_workers=config.training.num_workers, shuffle=False, pin_memory=True)

    if config.training.custom_sampling:
        train_dataloader.dataset.shuffle()

    print(f"Train dataset: {len(train_dataset)}")
    print(f"Query test: {len(query_dataset_test)}")
    print(f"Reference test: {len(reference_dataset_test)}")

    #-----------------------------------------------------------------------------#
    # Loss                                                                        #
    #-----------------------------------------------------------------------------#
    loss_fn = build_loss(config)

    if config.training.mixed_precision:
        scaler = GradScaler(init_scale=2.**10)
    else:
        scaler = None

    #-----------------------------------------------------------------------------#
    # Optimizer                                                                   #
    #-----------------------------------------------------------------------------#
    if config.training.decay_exclude_bias:
        decay_params, no_decay_params = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if len(p.shape) == 1 or n.endswith(".bias"):
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        wd = 0.01 if 'convnext' in config.model.model_args.model_name else 1e-3
        optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": wd},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=config.training.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr)

    #-----------------------------------------------------------------------------#
    # Scheduler                                                                   #
    #-----------------------------------------------------------------------------#
    train_steps = len(train_dataloader) * config.training.epochs
    warmup_steps = len(train_dataloader) * config.training.warmup_epochs

    print(f"Warmup Steps: {warmup_steps}  |  Total Train Steps: {train_steps}")

    if config.training.scheduler == "polynomial":
        print(f"Scheduler: polynomial - max LR: {config.training.lr} - end LR: {config.training.lr_end}")
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_training_steps=train_steps,
            lr_end=config.training.lr_end,
            power=1.5,
            num_warmup_steps=warmup_steps)
    elif config.training.scheduler == "cosine":
        print(f"Scheduler: cosine - max LR: {config.training.lr}")
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=train_steps,
            num_warmup_steps=warmup_steps)
    elif config.training.scheduler == "constant":
        print(f"Scheduler: constant - max LR: {config.training.lr}")
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    else:
        scheduler = None


    #-----------------------------------------------------------------------------#
    # Zero Shot Evaluation                                                        #
    #-----------------------------------------------------------------------------#
    if config.zero_shot:
        print(f"\n{30*'-'}[ Zero Shot ]{30*'-'}")
        results = evaluate(config, model, query_dataloader_test, reference_dataloader_test,
                           ranks=[1, 5, 10], step_size=1000, cleanup=True)
        print(results)


    #-----------------------------------------------------------------------------#
    # Train                                                                       #
    #-----------------------------------------------------------------------------#
    print("\nStarting Training...")
    best_score = 0.0
    sim_dict = None

    for epoch in range(1, config.training.epochs + 1):
        print(f"\n{30*'-'}[ Epoch: {epoch} ]{30*'-'}")

        train_loss = train_one_epoch(config, model, train_dataloader,
                                     loss_fn, optimizer, scheduler, scaler)

        print(f"Epoch: {epoch} | Train Loss = {train_loss:.4f} | LR = {optimizer.param_groups[0]['lr']:.6f}")

        # ------ Evaluate ------
        if (epoch % config.eval.eval_every_n_epoch == 0) or epoch == config.training.epochs:
            print(f"\n{30*'-'}[ Evaluate ]{30*'-'}")

            results = evaluate(config, model, query_dataloader_test, reference_dataloader_test,
                               ranks=[1, 5, 10], cleanup=True)

            r1 = results["r1"]

            if config.training.sim_sample:
                sim_dict = calc_sim(config, model, reference_dataloader_train,
                                    step_size=1000, cleanup=True)

            if r1 > best_score:
                best_score = r1
                torch.save(model.state_dict(),
                           f"{model_path}/weights_e{epoch:02d}_{r1:.4f}.pth")
                print(f"New best R@1: {r1:.4f} — checkpoint saved.")

        # ------ Reshuffle for next epoch ------
        if config.training.custom_sampling:
            if config.training.sim_sample and sim_dict is not None:
                train_dataloader.dataset.hard_negative_sampling_shuffle(
                    sim_dict,
                    neighbour_select=config.training.neighbour_select,
                    neighbour_range=config.training.neighbour_range)
            else:
                train_dataloader.dataset.shuffle()

    # Save final weights
    torch.save(model.state_dict(), f"{model_path}/weights_end.pth")
    print(f"\nTraining complete. Best R@1: {best_score:.4f}")