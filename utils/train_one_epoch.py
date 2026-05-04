import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_one_epoch(config, model, dataloader, loss_fn, optimizer, scheduler=None, scaler=None):
    """Single-epoch training loop."""
    model.train()
    losses = []

    bar = tqdm(dataloader, total=len(dataloader))

    optimizer.zero_grad(set_to_none=True)

    for query, reference, _ids in bar:
        query = query.to(config.training.device)
        reference = reference.to(config.training.device)

        if scaler is not None:
            from torch.amp import autocast
            with autocast(device_type="cuda", enabled=config.training.mixed_precision):
                feat_q, feat_r = model(query, reference)
                loss = loss_fn(feat_q, feat_r, model.logit_scale.exp())

            scaler.scale(loss).backward()

            if config.training.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), config.training.clip_grad)

            scaler.step(optimizer)
            scaler.update()
        else:
            feat_q, feat_r = model(query, reference)
            loss = loss_fn(feat_q, feat_r, model.logit_scale.exp())
            loss.backward()

            if config.training.clip_grad:
                torch.nn.utils.clip_grad_value_(model.parameters(), config.training.clip_grad)

            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())
        bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")

    bar.close()
    return sum(losses) / len(losses)


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


def train_one_epoch_self_distill(
    config,
    model,
    dataloader,
    loss_fn,
    optimizer,
    scheduler=None,
    scaler=None,
):
    """
    Training loop for distillation with teacher features.

    The dataloader yields:
        (query_img, reference_img, index (for positive pair), labels (for classification))
    The loss_fn should be a custom loss that accepts the signature
    loss_fn(logits_list1, logits_list2, embed1, embed2, logit_scale, labels), where *labels* is a tensor of shape [B] containing the class labels for the query images.
    """

    model.train()
    losses = []

    device = config.training.device
    bar    = tqdm(dataloader, total=len(dataloader))
    optimizer.zero_grad(set_to_none=True)

    for query, reference, _ids, labels in bar:
        query     = query.to(device)
        reference = reference.to(device)
        labels    = labels.to(device)

        if scaler is not None:
            from torch.amp import autocast
            with autocast(device_type="cuda", enabled=config.training.mixed_precision):
                logits_list1, embed1, logits_list2, embed2 = model(query, reference)
                loss = loss_fn(logits_list1, logits_list2, embed1, embed2, model.logit_scale.exp(), labels)
            scaler.scale(loss).backward()
            if config.training.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(model.parameters(), config.training.clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits_list1, embed1, logits_list2, embed2 = model(query, reference)
            loss = loss_fn(logits_list1, logits_list2, embed1, embed2, model.logit_scale.exp(), labels)
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


def train_one_epoch_cross_distill(
    config,
    teacher,
    student,
    dataloader,
    loss_fn,
    cross_distill_loss_fn,
    optimizer,
    scheduler=None,
    scaler=None,
):
    """
    Training loop for cross-model distillation.

    The dataloader yields:
        (query_img, reference_img, index (for positive pair))
    The loss_fn should be a custom loss that accepts the signature
    loss_fn(s_logits, t_logits)
    """

    student.train()
    losses = []

    device = config.training.device
    bar    = tqdm(dataloader, total=len(dataloader))
    optimizer.zero_grad(set_to_none=True)

    for query, reference, _ids in bar:
        query     = query.to(device)
        reference = reference.to(device)

        if scaler is not None:
            from torch.amp import autocast
            with autocast(device_type="cuda", enabled=config.training.mixed_precision):
                s_embed1, s_embed2 = student(query, reference)
                t_embed1, t_embed2 = teacher(query, reference)
                loss = loss_fn(s_embed1, s_embed2, student.logit_scale.exp()) + cross_distill_loss_fn(s_embed1, s_embed2, t_embed1, t_embed2)

            scaler.scale(loss).backward()
            if config.training.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_value_(student.parameters(), config.training.clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            s_embed1, s_embed2 = student(query, reference)
            t_embed1, t_embed2 = teacher(query, reference)

            loss = loss_fn(s_embed1, s_embed2, student.logit_scale.exp()) + cross_distill_loss_fn(s_embed1, s_embed2, t_embed1, t_embed2)
            loss.backward()
            if config.training.clip_grad:
                torch.nn.utils.clip_grad_value_(student.parameters(), config.training.clip_grad)
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