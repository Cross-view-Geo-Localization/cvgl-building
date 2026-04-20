import torch
import timm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import register_model
from utils.predict import ForwardMode


@register_model("MobileGeo")
class MobileGeo(nn.Module):

    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=384,
                 h_dim=512,
                 num_classes=701):
                 
        super().__init__()
        
        self.img_size = img_size
        
        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.prediction_head = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.model.feature_info.channels[i], h_dim),
                nn.ReLU(),
                nn.Linear(h_dim, num_classes),
            ) for i in range(len(self.model.feature_info.channels))
        ])
        
        
    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return {
            'reference': {
                'input_size': (3, self.img_size, self.img_size),
                'interpolation': data_config.get('interpolation', 'bicubic'),
                'mean': data_config.get('mean', (0.485, 0.456, 0.406)),
                'std': data_config.get('std', (0.229, 0.224, 0.225)),
            },
            'query': {
                'input_size': (3, self.img_size, self.img_size),
                'interpolation': data_config.get('interpolation', 'bicubic'),
                'mean': data_config.get('mean', (0.485, 0.456, 0.406)),
                'std': data_config.get('std', (0.229, 0.224, 0.225)),
            }
        }
    
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

    def _extract_features(self, image):
        # features is a list of tensors [feat_stage1, feat_stage2, ...]
        features = self.model(image)
        
        logits_list = []
        for i in range(len(features)):
            logits = self.prediction_head[i](features[i])
            logits_list.append(logits)
            
        # Extract the final embedding for metric/contrastive learning from the deepest stage
        final_feat = features[-1]
        final_embedding = F.adaptive_avg_pool2d(final_feat, 1).flatten(1)
        
        return logits_list, final_embedding
        
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
            
            logits_list1, embed1 = self._extract_features(image1)
            logits_list2, embed2 = self._extract_features(image2)
            
            return logits_list1, embed1, logits_list2, embed2
            
        elif mode == ForwardMode.QUERY or mode == ForwardMode.REFERENCE:
            _, embed = self._extract_features(image1)
            return embed
            
        else:
            raise ValueError(f"Invalid forward mode: {mode}. Must be of type ForwardMode.")