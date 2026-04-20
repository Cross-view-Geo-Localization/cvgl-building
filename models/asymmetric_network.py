import sys
import torch
import timm
import numpy as np
import torch.nn as nn

from utils.registry import register_model
from utils.predict import ForwardMode

sys.path.insert(0, "/home/tts26/sonh/scale-mae")
sys.path.insert(0, "/home/tts26/sonh/scale-mae/mae")
from mae import models_vit
from util.pos_embed import interpolate_pos_embed 

@register_model("AsymmetricNetwork")
class AsymmetricNetwork(nn.Module):
    def __init__(
        self,
        query_model_name,
        reference_model_name,
        pretrained=True,
        query_img_size=384,
        reference_img_size=224,
        embed_dim=768,
        freeze_reference=True
    ):
        super().__init__()
        self.ref_img_size = reference_img_size
        self.query_img_size = query_img_size
        self.query_model_name = query_model_name
        self.ref_model_name = reference_model_name
        self.freeze_reference = freeze_reference

        # ---------------------------------------------------------
        # SATELLITE BRANCH 
        # ---------------------------------------------------------
        self.reference_branch = models_vit.vit_large_patch16(
            img_size=reference_img_size, 
            num_classes=0, 
            global_pool=False,
            drop_path_rate=0.0
        )

        # Load checkpoint
        print(f"Loading pre-trained checkpoint from: {reference_model_name}")
        checkpoint = torch.load(reference_model_name, map_location='cpu', weights_only=False)
        checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
        state_dict = self.reference_branch.state_dict()

        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict.get(k, torch.empty(0)).shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        interpolate_pos_embed(self.reference_branch, checkpoint_model)
        msg = self.reference_branch.load_state_dict(checkpoint_model, strict=False)
        print(f"Scale MAE Loading Message: {msg}")

        ref_feat_dim = self.reference_branch.embed_dim

        self.reference_branch.eval()
        if self.freeze_reference:
            for param in self.reference_branch.parameters():
                param.requires_grad = False

        # ---------------------------------------------------------
        # QUERY BRANCH 
        # ---------------------------------------------------------
        if "vit" in query_model_name:
            # automatically change interpolate pos-encoding to img_size
            self.query_branch = timm.create_model(query_model_name, pretrained=pretrained, num_classes=0, img_size=query_img_size) 
        else:
            self.query_branch = timm.create_model(query_model_name, pretrained=pretrained, num_classes=0)

        with torch.no_grad():
            dummy_input = torch.zeros(2, 3, self.query_img_size, self.query_img_size)
            query_feat_dim = self.query_branch(dummy_input).shape[1]

        # Projection head
        self.query_proj = nn.Linear(query_feat_dim, embed_dim)
        self.ref_proj = nn.Linear(ref_feat_dim, embed_dim)
        self.dropout = nn.Dropout(0.4)

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_config(self):
        query_config = timm.data.resolve_model_data_config(self.query_branch)

        return {
            'reference': {
                'input_size': (3, self.ref_img_size, self.ref_img_size),
                'interpolation': 'bicubic', 
                'mean': (0.485, 0.456, 0.406), 
                'std': (0.229, 0.224, 0.225),  
            },
            'query': {
                'input_size': (3, self.query_img_size, self.query_img_size),
                'interpolation': query_config.get('interpolation', 'bicubic'),
                'mean': query_config.get('mean', (0.485, 0.456, 0.406)),
                'std': query_config.get('std', (0.229, 0.224, 0.225)),
            }
        }
    
    def set_grad_checkpointing(self, enable=True):
        self.query_branch.set_grad_checkpointing(enable)
        if not self.freeze_reference:
            if enable:
                if hasattr(self.reference_branch, 'set_grad_checkpointing'):
                    self.reference_branch.set_grad_checkpointing(enable)
            else:
                if hasattr(self.reference_branch, 'set_grad_checkpointing'):
                    self.reference_branch.set_grad_checkpointing(False)

    def _extract_features(self, x, input_res=None):
        if input_res is None:
            # Scale MAE expects a resolution tensor (GSD in meters/pixel).
            input_res = torch.ones(x.shape[0], device=x.device) * 0.3
            
        cls_token = self.reference_branch.forward_features(x, input_res=input_res)
        return cls_token
    
    def train(self, mode=True):
        super().train(mode)
        if self.freeze_reference:
            self.reference_branch.eval()
        return self
    
    def forward(self, image1, image2=None, mode=ForwardMode.TRAIN): 
        """
        Args:
            image1: The query image (or drone image).
            image2: The reference image (or satellite image). Required only in TRAIN mode.
            mode: ForwardMode Enum dictating the execution path.
        """
        if mode == ForwardMode.TRAIN: 
            if image2 is None:
                raise ValueError("Both image1 and image2 must be provided in TRAIN mode.")
            
            feat1 = self.query_branch(image1)
            if self.freeze_reference:
                with torch.no_grad():
                    feat2 = self._extract_features(image2)
            else:
                feat2 = self._extract_features(image2)

            feat1 = self.dropout(feat1)
            feat2 = self.dropout(feat2)

            feat1 = self.query_proj(feat1)
            feat2 = self.ref_proj(feat2)
            
            return feat1, feat2
            
        elif mode == ForwardMode.QUERY: 
            feat = self.query_branch(image1)
            feat = self.query_proj(feat)
            return feat
            
        elif mode == ForwardMode.REFERENCE: 
            feat = self._extract_features(image1)
            feat = self.ref_proj(feat)
            return feat
            
        else:
            raise ValueError(f"Invalid forward mode: {mode}.")
        
