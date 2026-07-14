import torch
import timm
import numpy as np
import torch.nn as nn
import torch.fft

from utils.registry import register_model
from utils.predict import ForwardMode
from models.sinkhorn_siamese_network import DustbinSinkhornPooling

class SpatialFrequencyFusion(nn.Module):
    def __init__(self, apply_log=True, shift_center=True):
        super().__init__()
        self.apply_log = apply_log
        self.shift_center = shift_center

    def forward(self, x):
        fft_complex = torch.fft.fftn(x, dim=(-2, -1), norm="orthor")
        if self.shift_center:
            fft_complex = torch.fft.fftshift(fft_complex, dim=(-2, -1))

        magnitude = torch.abs(fft_complex)

        if self.apply_log:
            magnitude = torch.log1p(magnitude + 1e-8)

        out = torch.cat([x, magnitude], dim=1) # (B, C*2, H, W)

        return out
    

@register_model("SFFNetwork")
class SFFNetwork(nn.Module):
    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=384,
                 num_prototypes=48,
                 num_iters=4,
                 dim_prototype=128,
                 epsilon=0.05):
                 
        super(SFFNetwork, self).__init__()
        
        self.img_size = img_size
        
        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True, img_size=img_size) 
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True, num_classes=0)
        
        self.sinkhorn = DustbinSinkhornPooling(
            feature_dim=dim_prototype,
            num_prototypes=num_prototypes,
            num_iters=num_iters,
            epsilon=epsilon
        )
        self.sff = SpatialFrequencyFusion()
    
        feat_dim = self.model.feature_info.channels()[-1]
        self.dim_reduce = nn.Sequential(
            nn.Conv2d(feat_dim, dim_prototype, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim_prototype)
        )

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        
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
            feat1 = self.sff(image1)
            feat2 = self.sff(image2)

            feat1 = self.model(feat1)[-1]
            feat2 = self.model(feat2)[-1]

            feat1 = self.dim_reduce(feat1)
            feat2 = self.dim_reduce(feat2)

            feat1 = self.sinkhorn(feat1)
            feat2 = self.sinkhorn(feat2)
            
            return feat1, feat2
            
        elif mode == ForwardMode.QUERY:
            feat = self.sff(image1)
            feat = self.model(feat)[-1]
            feat = self.dim_reduce(feat)
            return self.sinkhorn(feat)
            
        elif mode == ForwardMode.REFERENCE:
            feat = self.sff(image1)
            feat = self.model(feat)[-1] 
            feat = self.dim_reduce(feat)
            return self.sinkhorn(feat)
            
        else:
            raise ValueError(f"Invalid forward mode: {mode}. Must be of type ForwardMode.")
        
