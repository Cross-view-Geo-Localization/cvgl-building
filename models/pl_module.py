import pytorch_lightning as pl
from torch import optim
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup
from omegaconf import OmegaConf

from utils.registry import build_model, build_loss
from core.loss import InfoNCE

class CVGLLightningModule(pl.LightningModule):
    def __init__(self, config, steps_per_epoch):
        '''
        config: OmegaConf config
        steps_per_epoch: the training steps per epoch
        '''
        super().__init__()
        config_dict = OmegaConf.to_container(config, resolve=True)
        self.save_hyperparameters(config_dict)
        self.config = config
        self.steps_per_epoch = steps_per_epoch

        self.model = build_model(config)
        if config.training.grad_checkpointing:
            self.model.set_grad_checkpointing(True)

        self.criterion = build_loss(config)

    def forward(self, query_img, reference_img):
        return self.model(query_img, reference_img)
    
    def training_step(self, batch, batch_idx):
        query_img, ref_img, _ = batch 
        query_feat, ref_feat = self(query_img, ref_img)
        loss = self.criterion(query_feat, ref_feat, self.model.logit_scale.exp())
        
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        # 1. Optimizer setup with Bias exclusion logic
        if self.config.training.decay_exclude_bias:
            decay_params, no_decay_params = [], []
            for n, p in self.model.named_parameters():
                if not p.requires_grad: continue
                if len(p.shape) == 1 or n.endswith(".bias"):
                    no_decay_params.append(p)
                else:
                    decay_params.append(p)
                    
            wd = 0.01 if 'convnext' in self.config.model.model_name else 1e-3 
            optimizer_parameters = [
                {"params": decay_params, "weight_decay": wd},
                {"params": no_decay_params, "weight_decay": 0.0},
            ]
            optimizer = optim.AdamW(optimizer_parameters, lr=self.config.training.lr)
        else:
            optimizer = optim.AdamW(self.model.parameters(), lr=self.config.training.lr)

        # 2. Scheduler setup
        train_steps = self.steps_per_epoch * self.config.training.epochs
        warmup_steps = self.steps_per_epoch * self.config.training.warmup_epochs

        if self.config.training.scheduler == "polynomial":
            scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                                    num_warmup_steps=warmup_steps,
                                                                    num_training_steps=train_steps,
                                                                    lr_end=self.config.lr_end,
                                                                    power=1.5)
        elif self.config.training.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=train_steps, num_warmup_steps=warmup_steps)
        elif self.config.training.scheduler == "constant":
            scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps)
        else:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", # Transformers schedulers step per batch, not per epoch
            }
        }