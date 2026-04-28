
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp import autocast
from MEAN.DroneCVGL.core.cal_loss import cal_loss, cal_kl_loss, cal_triplet_loss
from MEAN.DroneCVGL.utils.predict import ForwardMode
 
 
# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────
 
class AverageMeter:
    """Tracks the running average of a scalar value."""
    def __init__(self):
        self.reset()
 
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
 
    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count
 
 
def _unwrap(model, use_multi_gpu: bool):
    """Return the raw model regardless of DataParallel / DistributedDataParallel wrapping."""
    return model.module if (use_multi_gpu and hasattr(model, "module")) else model
 
 
def _unpack_branch(output, return_f: bool):
    """
    Unpack one branch output from SiameseNetwork._branch_forward.
 
    return_f=False  →  (pfeat_align, y,        gap, part)
    return_f=True   →  (pfeat_align, cls_list, feat_list, gap, part)
 
    Returns
    -------
    pfeat_align : (B, msf_out_dim, H*W)   – DSA spatial features
    logit_list  : list[Tensor]             – classifier logits per head
    embed_list  : list[Tensor] | None      – pre-classifier embeddings (only when return_f=True)
    gap         : (B, C)                   – GAP descriptor used for infoNCE
    """
    if return_f:
        pfeat_align, logit_list, embed_list, gap, _part = output
    else:
        pfeat_align, logit_list, gap, _part = output
        embed_list = None
    return pfeat_align, logit_list, embed_list, gap
 
 
def _cls_loss(logit_list_q, logit_list_r, labels, criterion):
    """
    Average cross-entropy over all classifier heads (DEG part heads + global head).
 
    logit_list_*[i] : (B, num_classes) logit tensor for head i
    """
    total = torch.tensor(0.0, device=labels.device)
    n     = len(logit_list_q)
    for lq, lr in zip(logit_list_q, logit_list_r):
        total = total + cal_loss(lq, labels, criterion) + cal_loss(lr, labels, criterion)
    return total / max(n, 1)
 
 
def _triplet_feats(embed_list, gap):
    """
    Select the best available triplet embedding.
    - If embed_list exists (return_f=True): average the DEG part embeddings (all but last head).
    - Otherwise fall back to the GAP vector.
    """
    if embed_list is not None and len(embed_list) > 1:
        # embed_list[-1] is the global head; [:-1] are the DEG part heads
        return torch.stack(embed_list[:-1], dim=1).mean(dim=1)   # (B, num_bottleneck)
    return gap                                                      # (B, C)
 
 
# ──────────────────────────────────────────────────────────────────────────────
#  Main training loop
# ──────────────────────────────────────────────────────────────────────────────
 
def train_one_epoch(
    config,
    model,
    dataloader,
    loss_functions: dict,
    # Expected keys:
    #   "infoNCE"        – InfoNCE / contrastive loss  (always required)
    #   "Triplet"        – Triplet loss                (optional)
    #   "DSA_loss"       – Domain Space Alignment loss (optional)
    optimizer,
    epoch: int,
    train_steps_per: int,
    scheduler=None,
    scaler=None,
):
    """
    Single-epoch training loop for SiameseNetwork.
 
    Branch output mapping (SiameseNetwork._branch_forward)
    ───────────────────────────────────────────────────────
    return_f=False → (pfeat_align, [logit_0, logit_1, logit_global], gap, part)
    return_f=True  → (pfeat_align, [logit_0, ...], [embed_0, ...],  gap, part)
 
    Loss mapping
    ─────────────
    infoNCE  ← gap_q  vs  gap_r          (GAP descriptor, retrieval-ready)
    cls      ← logit_list per head        (cross-entropy, averaged over heads)
    triplet  ← DEG part embeds or gap     (metric learning)
    DSA      ← pfeat_align.mean(-1)       (domain alignment in spatial-pooled space)
    """
    model.train()
 
    losses_meter  = AverageMeter()
    criterion_cls = nn.CrossEntropyLoss()
 
    # ── Loss weights ───────────────────────────────────────────────────────────
    w_infonce = getattr(config.training, "weight_infonce", 1.0)
    w_cls     = getattr(config.training, "weight_cls",     0.0)
    w_triplet = getattr(config.training, "weight_triplet", 0.0)
    w_dsa     = getattr(config.training, "weight_dsa",     0.0)
 
    # ── Flags ──────────────────────────────────────────────────────────────────
    use_multi_gpu = (
        torch.cuda.device_count() >= 1
        and len(getattr(config.training, "gpu_ids", [])) >= 1
    )
    handcraft = getattr(config.training, "handcraft_model", False)
    return_f  = getattr(config.training, "return_f",        False)
 
    optimizer.zero_grad(set_to_none=True)
    bar  = tqdm(dataloader, total=len(dataloader))
    step = 1
 
    for query, reference, _ids, labels in bar:
        query     = query.to(config.training.device)
        reference = reference.to(config.training.device)
        labels    = labels.to(config.training.device)
 
        # ── Forward + loss ─────────────────────────────────────────────────────
        def forward_and_compute_loss():
 
            # ── 1. Forward pass ──────────────────────────────────────────────
            if handcraft:
                # SiameseNetwork returns (branch_q, branch_r) in TRAIN mode
                out_q, out_r = model(query, reference, mode=ForwardMode.TRAIN)
 
                # Unpack each branch into its semantic components
                pfeat_align_q, logit_list_q, embed_list_q, gap_q = _unpack_branch(out_q, return_f)
                pfeat_align_r, logit_list_r, embed_list_r, gap_r = _unpack_branch(out_r, return_f)
 
                # infoNCE descriptor: raw GAP vector (retrieval embedding)
                feat_q = gap_q                           # (B, C)
                feat_r = gap_r                           # (B, C)
 
                # Triplet embedding: DEG part embeds (or GAP fallback)
                feat_tri_q = _triplet_feats(embed_list_q, gap_q)
                feat_tri_r = _triplet_feats(embed_list_r, gap_r)
 
                # DSA descriptor: pool spatial dim → (B, msf_out_dim)
                feat_dsa_q = pfeat_align_q.mean(-1)
                feat_dsa_r = pfeat_align_r.mean(-1)
 
            else:
                # Plain backbone: returns (feat_q, feat_r) with no extras
                feat_q, feat_r = model(query, reference, mode=ForwardMode.TRAIN)
                logit_list_q = logit_list_r = None
                feat_tri_q   = feat_tri_r   = None
                feat_dsa_q   = feat_dsa_r   = None
 
            # ── 2. Temperature scales ────────────────────────────────────────
            raw = _unwrap(model, use_multi_gpu)
            scale        = raw.logit_scale.exp()
            scale_blocks = raw.logit_scale_blocks.exp() if hasattr(raw, "logit_scale_blocks") else scale
 
            # ── 3. InfoNCE loss (always active) ──────────────────────────────
            loss_infonce = loss_functions["infoNCE"](feat_q, feat_r, scale)
 
            # ── 4. Classification loss ────────────────────────────────────────
            # Averages cross-entropy across all classifier heads
            # (2 DEG-part heads + 1 global head = 3 heads total)
            loss_cls = torch.tensor(0.0, device=config.training.device)
            if w_cls > 0 and logit_list_q is not None:
                loss_cls = _cls_loss(logit_list_q, logit_list_r, labels, criterion_cls)
 
            # ── 5. Triplet loss ───────────────────────────────────────────────
            loss_triplet = torch.tensor(0.0, device=config.training.device)
            if w_triplet > 0 and feat_tri_q is not None and "Triplet" in loss_functions:
                loss_triplet = loss_functions["Triplet"](feat_tri_q, feat_tri_r)
 
            # ── 6. DSA loss ───────────────────────────────────────────────────
            # Operates on the MSF-aligned spatial features (pooled)
            loss_dsa = torch.tensor(0.0, device=config.training.device)
            if w_dsa > 0 and feat_dsa_q is not None and "DSA_loss" in loss_functions:
                loss_dsa = loss_functions["DSA_loss"](feat_dsa_q, feat_dsa_r, scale_blocks)
 
            # ── 7. Combined loss ──────────────────────────────────────────────
            loss_total = (
                w_infonce * loss_infonce
                + w_cls     * loss_cls
                + w_triplet * loss_triplet
                + w_dsa     * loss_dsa
            )
            return loss_total, loss_infonce, loss_cls, loss_triplet, loss_dsa
 
        # ── Training step (AMP or full-precision) ──────────────────────────────
        if scaler is not None:
            with autocast(device_type="cuda", enabled=config.training.mixed_precision):
                loss, l_infonce, l_cls, l_triplet, l_dsa = forward_and_compute_loss()
            scaler.scale(loss).backward()
            if config.training.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), config.training.clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, l_infonce, l_cls, l_triplet, l_dsa = forward_and_compute_loss()
            loss.backward()
            if config.training.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), config.training.clip_grad)
            optimizer.step()
 
        optimizer.zero_grad(set_to_none=True)
 
        if scheduler is not None:
            scheduler.step()
 
        losses_meter.update(loss.item())
 
        # ── Progress bar ───────────────────────────────────────────────────────
        bar.set_postfix(
            epoch    = epoch,
            loss     = f"{loss.item():.4f}",
            infonce  = f"{w_infonce * l_infonce.item():.4f}",
            cls      = f"{w_cls     * l_cls.item():.4f}",
            triplet  = f"{w_triplet * l_triplet.item():.4f}",
            dsa      = f"{w_dsa     * l_dsa.item():.4f}",
            avg      = f"{losses_meter.avg:.4f}",
            lr       = f"{optimizer.param_groups[0]['lr']:.6f}",
        )
 
        step += 1
 
    bar.close()
    return losses_meter.avg


def train_one_epoch_semi_pos(
    config,
    model,
    dataloader,
    loss_fn,
    optimizer,
    scheduler=None,
    scaler=None,
):
    """
    Training loop for UAV1-style semi-positive datasets.

    The UAV1DatasetTrain dataloader yields:
        (query_img, gallery_img, positive_weight)

    where *positive_weight* is the IoU score in [semi_threshold, 1].
    All weighting logic is encapsulated inside *loss_fn*, which should be an
    instance of WeightedInfoNCE.  The per-pair weights are forwarded directly
    as the ``positive_weights`` argument.

    Args:
        config:     OmegaConf config (uses config.training.*)
        model:      Retrieval model — forward(query, reference) returns
                    (feat_query, feat_reference).
        dataloader: DataLoader from UAV1DatasetTrain.
        loss_fn:    WeightedInfoNCE instance (or any loss accepting the
                    signature loss_fn(feat_q, feat_r, logit_scale, positive_weights)).
        optimizer:  Torch optimizer.
        scheduler:  LR scheduler stepped per batch (optional).
        scaler:     GradScaler for AMP (optional).

    Returns:
        float: mean training loss over the epoch.
    """
    model.train()
    losses = []

    device = config.training.device
    bar    = tqdm(dataloader, total=len(dataloader))
    optimizer.zero_grad(set_to_none=True)

    for query, gallery, weights in bar:
        query   = query.to(device)
        gallery = gallery.to(device)
        weights = weights.to(device, dtype=torch.float32)  # [B] IoU scores

        if scaler is not None:
            from torch.amp import autocast
            with autocast(device_type="cuda", enabled=config.training.mixed_precision):
                feat_q, feat_r = model(query, gallery)
                loss = loss_fn(feat_q, feat_r, model.logit_scale.exp(),
                               positive_weights=weights)
            scaler.scale(loss).backward()
            if config.training.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), config.training.clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            feat_q, feat_r = model(query, gallery)
            loss = loss_fn(feat_q, feat_r, model.logit_scale.exp(),
                           positive_weights=weights)
            loss.backward()
            if config.training.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), config.training.clip_grad)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())
        bar.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.6f}",
        )

    bar.close()
    return sum(losses) / len(losses)