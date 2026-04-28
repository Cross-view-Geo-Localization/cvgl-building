"""
core/metrics/uav1.py
====================
Evaluation helpers for UAV1-style datasets.

The UAV1DatasetEval __getitem__ returns bare images (no label), so we use a
dedicated predict() that only iterates images, then pass the query/gallery name
lists explicitly to evaluate().

evaluate() signature:
    evaluate(config, model, query_loader, gallery_loader,
             query_names, gallery_names, pairs_drone2sate_dict,
             ranks=[1, 5, 10], cleanup=True)
             -> dict  {"r1", "r5", "r10", "r_top1", "AP"}
"""

import gc
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.amp import autocast
from tqdm import tqdm

from MEAN.DroneCVGL.utils.predict import ForwardMode

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def predict(config, model, dataloader, mode: ForwardMode):
    """
    Extract features from a UAV1DatasetEval dataloader.

    UAV1DatasetEval yields bare image tensors (no label), so we handle the
    iteration accordingly.

    Args:
        config:     OmegaConf config object (uses config.training / config.eval).
        model:      The retrieval model.
        dataloader: DataLoader wrapping UAV1DatasetEval.
        mode:       Forward mode (drone or satellite).

    Returns:
        img_features: Float32 tensor [N, D] kept on GPU.
    """
    train_cfg = config.training
    eval_cfg  = config.eval

    model.eval()
    time.sleep(0.1)

    bar = tqdm(dataloader, total=len(dataloader)) if train_cfg.verbose else dataloader

    features_list = []

    with torch.no_grad():
        for img in bar:
            with autocast(device_type="cuda", enabled=train_cfg.mixed_precision):
                img = img.to(train_cfg.device)
                feat = model(img, mode=mode)

                if eval_cfg.normalize_features:
                    feat = F.normalize(feat, dim=-1)

            features_list.append(feat.to(torch.float32))

    if train_cfg.verbose:
        bar.close()

    return torch.cat(features_list, dim=0)  # [N, D]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    config,
    model,
    query_loader,
    gallery_loader,
    query_names,
    gallery_names,
    pairs_drone2sate_dict,
    ranks=(1, 5, 10),
    cleanup=True,
):
    """
    Compute Recall@K, Recall@top1%, and mAP for a UAV1 evaluation split.

    Args:
        config:               OmegaConf config.
        model:                Retrieval model.
        query_loader:         DataLoader for UAV1DatasetEval(view='drone').
        gallery_loader:       DataLoader for UAV1DatasetEval(view='sate').
        query_names:          List[str] — dataset.images_name for query set.
        gallery_names:        List[str] — dataset.images_name for gallery set.
        pairs_drone2sate_dict: {drone_img_name: [sate_img_name, ...]}
                              Ground-truth matches from UAV1DatasetEval.
        ranks:                Recall ranks to compute.
        cleanup:              Free GPU tensors after evaluation.

    Returns:
        dict with keys 'r1', 'r5', 'r10', 'r_top1', 'AP' (all float, percent).
    """
    print("Extract Query Features:")
    q_features = predict(config, model, query_loader, mode=ForwardMode.QUERY)  # [Q, D]

    print("Extract Gallery Features:")
    g_features = predict(config, model, gallery_loader, mode=ForwardMode.REFERENCE)  # [G, D]

    # ---- Build name → index lookup for gallery ----------------------------
    gallery_name_to_idx = {name: i for i, name in enumerate(gallery_names)}

    # For the trajectory-aware scoring mask:
    # tiles from the same trajectory share the same prefix (str_i = first token of name)
    # e.g. "HoaLac_5_023_011.png" -> trajectory "HoaLac"
    # We create per-trajectory binary masks over the full gallery so that
    # a query only competes against gallery tiles from its own trajectory.
    G = len(gallery_names)
    traj_mask = {}
    for idx, name in enumerate(gallery_names):
        traj = name.split('_')[0]
        if traj not in traj_mask:
            traj_mask[traj] = np.zeros(G, dtype=np.float32)
        traj_mask[traj][idx] = 1.0

    # ---- Compute all scores -----------------------------------------------
    # [Q, G] similarity matrix (compute in chunks to save memory)
    print("Compute Similarity Matrix:")
    score_matrix = (q_features @ g_features.T).cpu().numpy()  # [Q, G]

    # ---- Per-query metrics ------------------------------------------------
    Q = len(query_names)
    cmc = np.zeros(G, dtype=np.float64)
    all_ap = []

    print("Compute Metrics:")
    for i in tqdm(range(Q)):
        drone_name = query_names[i]
        gt_sate_names = pairs_drone2sate_dict.get(drone_name, [])

        if len(gt_sate_names) == 0:
            continue  # no ground truth → skip

        # Map gt names to gallery indices
        gt_idx = np.array(
            [gallery_name_to_idx[s] for s in gt_sate_names if s in gallery_name_to_idx],
            dtype=np.int64,
        )
        if len(gt_idx) == 0:
            continue

        # Trajectory-aware masking: only rank within same trajectory tiles
        traj = drone_name.split('_')[0]
        mask = traj_mask.get(traj, np.ones(G, dtype=np.float32))
        score = score_matrix[i] * mask  # zero out other trajectories

        # Sort descending
        ranked_idx = np.argsort(score)[::-1]

        # Good-index binary indicator aligned to ranked_idx order
        good_indicator = np.isin(ranked_idx, gt_idx)

        # --- AP ---
        y_true   = good_indicator.astype(int)
        y_scores = np.arange(len(y_true), 0, -1)
        if y_true.sum() > 0:
            ap = average_precision_score(y_true, y_scores)
            all_ap.append(ap)

        # --- CMC: find rank of first correct match ---
        first_hit = np.where(good_indicator)[0]
        if len(first_hit) > 0:
            cmc[first_hit[0]:] += 1

    mAP = float(np.mean(all_ap)) if all_ap else 0.0
    cmc = cmc / Q  # normalise to recall rate

    # top 1%
    top1pct_k = max(1, round(G * 0.01))

    results = {}
    strings  = []
    for r in ranks:
        val = float(cmc[r - 1]) * 100
        results[f"r{r}"] = val
        strings.append(f"Recall@{r}: {val:.4f}")

    r_top1 = float(cmc[top1pct_k - 1]) * 100
    results["r_top1"] = r_top1
    results["AP"] = mAP * 100
    strings.append(f"Recall@top1%: {r_top1:.4f}")
    strings.append(f"AP: {mAP * 100:.4f}")

    print(" - ".join(strings))

    if cleanup:
        del q_features, g_features, score_matrix
        gc.collect()

    return results