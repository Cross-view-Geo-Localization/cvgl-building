MODELS = {}
LOSSES = {}

def register_model(name):
    def wrapper(cls):
        MODELS[name] = cls
        return cls
    return wrapper

def register_loss(name):
    def wrapper(cls):
        LOSSES[name] = cls
        return cls
    return wrapper

def build_model(config):
    return MODELS[config.model.model_name](**config.model.model_args)

def build_loss(config):
    """
    Đọc config.loss.losses và build dict:
    {
        "infoNCE":  InfoNCELoss(...),
        "DSA_loss": InfoNCELoss(...),
        "Triplet":  TripletLoss(...),
    }
    Chỉ build những loss nào có weight > 0 trong config.training.
    """
    weight_map = {
        "infoNCE":  getattr(config.training, "weight_infonce", 1.0),
        "DSA_loss": getattr(config.training, "weight_dsa",     0.0),
        "Triplet":  getattr(config.training, "weight_triplet", 0.0),
    }

    loss_functions = {}
    for loss_key, loss_cfg in config.loss.losses.items():
        if weight_map.get(loss_key, 0.0) <= 0.0:
            continue  # bỏ qua loss bị tắt, không tốn memory
        loss_functions[loss_key] = LOSSES[loss_cfg.loss_name](**loss_cfg.loss_args)

    return loss_functions