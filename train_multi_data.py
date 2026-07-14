import os
import sys
import time
import shutil
import random

import numpy as np
from tqdm import tqdm
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
from utils.train_one_epoch import compute_loss_and_backward
from data.transforms import get_transforms_train, get_transforms_val

from data.university import U1652DatasetTrain, U1652DatasetEval
from data.sues200 import SUES200DatasetTrain, SUES200DatasetEval
from data.visloc import VisLocDatasetTrain, VisLocDatasetEval
from core.metrics.visloc import evaluate as evaluate_visloc
from core.metrics.university import evaluate as evaluate_university, calc_sim as calc_sim_uni
from core.metrics.sues200 import evaluate as evaluate_sues200, calc_sim as calc_sim_sues

# Model / Loss registry — import to trigger registration
from models.siamese_network import SiameseNetwork                           
from models.siamese_network_max_avg import SiameseNetworkMaxAvg             
from models.asymmetric_network import AsymmetricNetwork                     
from models.sinkhorn_siamese_network import SinkhornSiameseNetwork         
from models.aspp import ASPPSinkhornSiameseNetwork                         
from core.loss import InfoNCE, ColBERTLoss, WeightedInfoNCE

def cycle(iterable):
    while True:
        for x in iterable:
            yield x


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Config
    # -------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config", "multi_data_sinkhorn.yaml")
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
    model_path = f"{config.model.model_path}/{config.model.model_name}/{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(model_path, exist_ok=True)

    shutil.copyfile(os.path.abspath(__file__), os.path.join(model_path, "train_multi_data.py"))
    shutil.copyfile(config_path,               os.path.join(model_path, "config.yaml"))
    shutil.copyfile(os.path.join(script_dir, "data", "transforms.py"), "{}/transforms.py".format(model_path))
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
        image_size_sat=sate_size,
        image_size_drone=drone_size,
        query_mean=query_mean, query_std=query_std,
        ref_mean=ref_mean,     ref_std=ref_std,
    )
    val_drone_tf, val_sate_tf = get_transforms_val(
        image_size_sat=sate_size,
        image_size_drone=drone_size,
        query_mean=query_mean, query_std=query_std,
        ref_mean=ref_mean,     ref_std=ref_std,
    )

    # -------------------------------------------------------------------------
    # Datasets & Dataloaders VisLoc
    # -------------------------------------------------------------------------
    visloc_train_dataset = VisLocDatasetTrain(
        data_root=config.data.visloc.data_folder,
        pairs_meta_file=config.data.visloc.train_pairs_meta_file,
        transforms_query=train_drone_tf,
        transforms_gallery=train_sate_tf,
        shuffle_batch_size=config.training.batch_size
    )
    visloc_train_loader = DataLoader(
        visloc_train_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=not config.training.custom_sampling,
        pin_memory=True,
    )

    # ---- Eval: query (drone) ----
    visloc_query_dataset_test = VisLocDatasetEval(
        pairs_meta_file=config.data.visloc.test_pairs_meta_file,
        data_root=config.data.visloc.data_folder,
        view="drone",
        mode="pos",
        transforms=val_drone_tf,
    )

    visloc_query_img_list = visloc_query_dataset_test.images_name
    visloc_pairs_drone2sate_dict = visloc_query_dataset_test.pairs_drone2sate_dict
    visloc_query_center_loc_xy_list = visloc_query_dataset_test.images_center_loc_xy

    visloc_query_loader = DataLoader(
        visloc_query_dataset_test,
        batch_size=config.eval.batch_size_eval,
        num_workers=config.training.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    # ---- Eval: gallery (satellite) ----
    visloc_gallery_dataset_test = VisLocDatasetEval(
        pairs_meta_file=config.data.visloc.test_pairs_meta_file,
        data_root=config.data.visloc.data_folder,
        view="sate",
        sate_img_dir=config.data.visloc.sate_img_dir,
        transforms=val_sate_tf,
    )

    visloc_gallery_img_list = visloc_gallery_dataset_test.images_name
    visloc_gallery_center_loc_xy_list = visloc_gallery_dataset_test.images_center_loc_xy
    visloc_gallery_topleft_loc_xy_list = visloc_gallery_dataset_test.images_topleft_loc_xy
    print('jyxjyx, test len', len(visloc_query_img_list), len(visloc_gallery_img_list), flush=True)

    visloc_gallery_loader = DataLoader(
        visloc_gallery_dataset_test,
        batch_size=config.eval.batch_size_eval,
        num_workers=config.training.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    if config.training.custom_sampling:
        visloc_train_dataset.shuffle()

    print(f"Train pairs VisLoc: {len(visloc_train_dataset)}")
    print(f"Query images VisLoc: {len(visloc_query_dataset_test)}")
    print(f"Gallery tiles VisLoc: {len(visloc_gallery_dataset_test)}")

    # -------------------------------------------------------------------------
    # Datasets & Dataloaders University-1652
    # -------------------------------------------------------------------------
    if config.data.university.dataset == 'U1652-D2S':
        config.data.university.query_folder_train = './data/University-Release/train/drone'
        config.data.university.reference_folder_train = './data/University-Release/train/satellite'
        config.data.university.query_folder_test = './data/University-Release/test/query_drone'
        config.data.university.reference_folder_test = './data/University-Release/test/gallery_satellite'
    elif config.data.university.dataset == 'U1652-S2D':
        config.data.university.query_folder_train = './data/University-Release/train/drone'
        config.data.university.reference_folder_train = './data/University-Release/train/satellite'
        config.data.university.query_folder_test = './data/University-Release/test/query_satellite'
        config.data.university.reference_folder_test = './data/University-Release/test/gallery_drone'

    uni_train_dataset = U1652DatasetTrain(
        config.data.university.query_folder_train,
        config.data.university.reference_folder_train,
        train_drone_tf,
        train_sate_tf,
        config.training.prob_flip,
        config.training.batch_size)
    uni_train_dataloader = DataLoader(
        uni_train_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=not config.training.custom_sampling,
        pin_memory=True)

    uni_query_dataset_test = U1652DatasetEval(config.data.university.query_folder_test, "query", val_drone_tf)
    uni_query_dataloader_test = DataLoader(uni_query_dataset_test, batch_size=config.eval.batch_size_eval,
                                           num_workers=config.training.num_workers, shuffle=False, pin_memory=True)

    uni_reference_dataset_test = U1652DatasetEval(
        config.data.university.reference_folder_test, "reference", val_sate_tf,
        sample_ids=uni_query_dataset_test.get_sample_ids(),
        reference_n=config.eval.eval_reference_n)
    uni_reference_dataloader_test = DataLoader(uni_reference_dataset_test, batch_size=config.eval.batch_size_eval,
                                               num_workers=config.training.num_workers, shuffle=False, pin_memory=True)

    uni_reference_dataloader_train = None
    if config.training.sim_sample:
        uni_reference_dataset_train = U1652DatasetEval(config.data.university.reference_folder_train, "reference", val_sate_tf)
        uni_reference_dataloader_train = DataLoader(uni_reference_dataset_train, batch_size=config.eval.batch_size_eval,
                                                num_workers=config.training.num_workers, shuffle=False, pin_memory=True)

    if config.training.custom_sampling:
        uni_train_dataloader.dataset.shuffle()

    print(f"Train dataset University: {len(uni_train_dataset)}")
    print(f"Query test University: {len(uni_query_dataset_test)}")
    print(f"Reference test University: {len(uni_reference_dataset_test)}")

    # -------------------------------------------------------------------------
    # Datasets & Dataloaders SUES-200
    # -------------------------------------------------------------------------
    sues_train_dataset = SUES200DatasetTrain(
        config.data.sues200.query_folder_train,
        config.data.sues200.reference_folder_train,
        train_drone_tf,
        train_sate_tf,
        config.training.prob_flip,
        config.training.batch_size)
    sues_train_dataloader = DataLoader(
        sues_train_dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=not config.training.custom_sampling,
        pin_memory=True)

    sues_query_dataset_test = SUES200DatasetEval(config.data.sues200.query_folder_test, "drone", val_drone_tf)
    sues_query_dataloader_test = DataLoader(sues_query_dataset_test, batch_size=config.eval.batch_size_eval,
                                            num_workers=config.training.num_workers, shuffle=False, pin_memory=True)

    sues_reference_dataset_test = SUES200DatasetEval(
        config.data.sues200.reference_folder_test, "satellite", val_sate_tf,
        sample_ids=sues_query_dataset_test.get_sample_ids())
    sues_reference_dataloader_test = DataLoader(sues_reference_dataset_test, batch_size=config.eval.batch_size_eval,
                                           num_workers=config.training.num_workers, shuffle=False, pin_memory=True)

    sues_reference_dataloader_train = None
    if config.training.sim_sample:
        sues_reference_dataset_train = SUES200DatasetEval(config.data.sues200.reference_folder_train, "satellite", val_sate_tf)
        sues_reference_dataloader_train = DataLoader(sues_reference_dataset_train, batch_size=config.eval.batch_size_eval,
                                                num_workers=config.training.num_workers, shuffle=False, pin_memory=True)

    if config.training.custom_sampling:
        sues_train_dataloader.dataset.shuffle()

    print(f"Train dataset SUES-200: {len(sues_train_dataset)}")
    print(f"Query test SUES-200: {len(sues_query_dataset_test)}")
    print(f"Reference test SUES-200: {len(sues_reference_dataset_test)}")

    # -------------------------------------------------------------------------
    # Dataloader Iterators
    # -------------------------------------------------------------------------
    visloc_iter = iter(cycle(visloc_train_loader))
    uni_iter = iter(cycle(uni_train_dataloader))
    sues_iter = iter(cycle(sues_train_dataloader))

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
    # Tính toán số Steps động (Dynamic Steps Calculation)
    # -------------------------------------------------------------------------
    train_datasets = [visloc_train_dataset, uni_train_dataset, sues_train_dataset]

    avg_dataset_size = sum(len(d) for d in train_datasets) / len(train_datasets)

    steps_per_virtual_epoch = max(1, int(avg_dataset_size / config.training.batch_size))

    total_train_steps = steps_per_virtual_epoch * config.training.epochs

    warmup_steps = int(steps_per_virtual_epoch * config.training.warmup_epochs)

    eval_every_n_steps = config.eval.eval_every_n_epoch * steps_per_virtual_epoch

    # -------------------------------------------------------------------------
    # LR Scheduler
    # -------------------------------------------------------------------------
    print(f"Warmup steps: {warmup_steps}  |  Total train steps: {total_train_steps} | Average Dataset Size: {avg_dataset_size} | Steps per Virtual Epochs: {steps_per_virtual_epoch}")

    if config.training.scheduler == "polynomial":
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_training_steps=total_train_steps,
            lr_end=config.training.lr_end,
            power=1.5,
            num_warmup_steps=warmup_steps,
        )
    elif config.training.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=total_train_steps,
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
    # if config.zero_shot:
    #     print(f"\n{30*'-'}[ Zero Shot ]{30*'-'}")
    #     results = evaluate_visloc(
    #         config=config,
    #         model=model,
    #         query_loader=query_loader,
    #         gallery_loader=gallery_loader, 
    #         query_list=query_img_list,
    #         gallery_list=gallery_img_list,
    #         query_center_loc_xy_list=query_center_loc_xy_list,
    #         gallery_center_loc_xy_list=gallery_center_loc_xy_list,
    #         gallery_topleft_loc_xy_list=gallery_topleft_loc_xy_list,
    #         pairs_dict=pairs_drone2sate_dict,
    #         ranks_list=[1, 5, 10],
    #         step_size=1000,
    #         cleanup=True
    #     )
    #     print(results)

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    print("\nStarting Training...")
    best_score = 0.0
    optimizer.zero_grad()

    pbar = tqdm(total=total_train_steps, desc="Training", dynamic_ncols=True)

    for step in range(1, total_train_steps + 1):
        model.train()
        total_step_loss = 0.0
        current_virtual_epoch = step // steps_per_virtual_epoch

        batch_visloc = next(visloc_iter)
        loss_visloc = compute_loss_and_backward(config, model, batch_visloc, loss_fn, scaler, data_weight=1.2, has_weights=True)
        total_step_loss += loss_visloc

        batch_uni = next(uni_iter)
        loss_uni = compute_loss_and_backward(config, model, batch_uni, loss_fn, scaler)
        total_step_loss += loss_uni

        batch_sues = next(sues_iter)
        loss_sues = compute_loss_and_backward(config, model, batch_sues, loss_fn, scaler)
        total_step_loss += loss_sues

        if scaler is not None:
            if config.training.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.training.clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            if config.training.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.training.clip_grad)
            optimizer.step()

        optimizer.zero_grad() 
        scheduler.step()      

        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({'Loss': f"{total_step_loss:.4f}", 'LR': f"{current_lr:.6f}"})
        pbar.update(1)

        if step % steps_per_virtual_epoch == 0 or step == total_train_steps:
            print(f"Step {step}/{total_train_steps} | Loss: {total_step_loss:.4f} | LR: {current_lr:.6f}")

        # ---- Periodic Evaluation ----
        if (step % eval_every_n_steps == 0) or step == total_train_steps:
            print(f"\n{30*'-'}[ Evaluate @ Step {step} ]{30*'-'}")

            visloc_eval_results = evaluate_visloc(
                config=config,
                model=model,
                query_loader=visloc_query_loader,
                gallery_loader=visloc_gallery_loader, 
                query_list=visloc_query_img_list,
                gallery_list=visloc_gallery_img_list,
                pairs_dict=visloc_pairs_drone2sate_dict,
                query_center_loc_xy_list=visloc_query_center_loc_xy_list,
                gallery_center_loc_xy_list=visloc_gallery_center_loc_xy_list,
                gallery_topleft_loc_xy_list=visloc_gallery_topleft_loc_xy_list,
                ranks_list=[1, 5, 10],
                step_size=1000,
                cleanup=True
            )
            uni_eval_results = evaluate_university(
                config=config,
                model=model,
                query_loader=uni_query_dataloader_test,
                reference_loader=uni_reference_dataloader_test, 
                ranks=[1, 5, 10],
                cleanup=True
            )
            sues_eval_results = evaluate_sues200(
                config=config,
                model=model,
                query_loader=sues_query_dataloader_test,
                ref_loader=sues_reference_dataloader_test, 
                ranks=[1, 5, 10],
                cleanup=True
            )

            r1_vis = visloc_eval_results if visloc_eval_results <= 1.0 else visloc_eval_results / 100.0
            r1_uni = uni_eval_results["r1"] if uni_eval_results["r1"] <= 1.0 else uni_eval_results["r1"] / 100.0
            r1_sues = sues_eval_results if sues_eval_results <= 1.0 else sues_eval_results / 100.0

            r1_test = (r1_vis * r1_uni * r1_sues) ** (1/3)

            if r1_test > best_score:
                best_score = r1_test
                ckpt_path  = os.path.join(
                    model_path, f"weights_e{step:02d}_{r1_test:.4f}.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"New best R@1: {r1_test:.4f} — checkpoint saved → {ckpt_path}")

            if config.training.sim_sample:
                uni_sim_dict = calc_sim_uni(config, model, uni_reference_dataloader_train,
                                    step_size=1000, cleanup=True)
                sues_sim_dict = calc_sim_sues(config, model, sues_reference_dataloader_train,
                                    step_size=1000, cleanup=True)

        # ---- Reshuffle for next epoch ----
        if config.training.custom_sampling and step % steps_per_virtual_epoch == 0:
            visloc_train_dataset.shuffle()
            if config.training.sim_sample and uni_sim_dict is not None:
                uni_train_dataset.hard_negative_sampling_shuffle(
                    uni_sim_dict,
                    neighbour_select=config.training.neighbour_select,
                    neighbour_range=config.training.neighbour_range)
                sues_train_dataset.hard_negative_sampling_shuffle(
                    sues_sim_dict,
                    neighbour_select=config.training.neighbour_select,
                    neighbour_range=config.training.neighbour_range
                )
            else:
                uni_train_dataset.shuffle()
                sues_train_dataset.shuffle()

    pbar.close()
    # ---- Save final weights ----
    final_path = os.path.join(model_path, "weights_end.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Best R@1: {best_score:.4f}")
    print(f"Final weights → {final_path}")