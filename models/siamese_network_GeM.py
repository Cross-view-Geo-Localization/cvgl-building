import torch
import timm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import register_model
from utils.predict import ForwardMode

class GeMPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6, output_size=1):
        """
        Thêm output_size=1 để khớp với signature của AdaptiveAvgPool2d 
        (phòng trường hợp module cha truyền tham số này vào).
        """
        super(GeMPooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        
        if x.dim() == 4:
            # Đối với CNN: [B, C, H, W] -> [B, C, 1, 1]
            # SỬ DỤNG keepdim=True để mô phỏng chính xác hành vi của AdaptiveAvgPool2d(1)
            x = x.mean(dim=(-2, -1), keepdim=True)
        elif x.dim() == 3:
            # Đối với ViT: [B, N, C] -> [B, 1, C]
            x = x.mean(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unexpected feature dim: {x.dim()}")
            
        return x.pow(1.0 / self.p)
    

@register_model("SiameseNetworkGeM")
class SiameseNetworkGeM(nn.Module):

    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=384,
                 p_init=3.0):
                 
        super(SiameseNetworkGeM, self).__init__()
        
        self.img_size = img_size
        
        # 1. BỎ tham số global_pool='' để timm khởi tạo module SelectAdaptivePool2d
        if "vit" in model_name:
            self.model = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0, 
                img_size=img_size
            ) 
        else:
            self.model = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0
            )

        # 2. Duyệt qua các module và thay thế AdaptiveAvgPool2d bằng GeMPooling
        for name, module in self.model.named_modules():
            if module.__class__.__name__ == "SelectAdaptivePool2d":
                for child_name, child in module.named_children():
                    if isinstance(child, nn.AdaptiveAvgPool2d):
                        setattr(module, child_name, GeMPooling(p=p_init))
                        print(f"✅ Replaced {child_name} with GeMPooling in {name}")

        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def get_config(self):
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

    def extract_and_pool_features(self, x):
        """
        Do GeM đã được cấy trực tiếp vào bên trong, `self.model(x)` sẽ tự động 
        thực hiện tính GeM Pooling và Flatten. Output đã sẵn sàng là [B, C].
        """
        return self.model(x)
        
    def forward(self, image1, image2=None, mode=ForwardMode.TRAIN):
        if mode == ForwardMode.TRAIN:
            if image2 is None:
                raise ValueError("Both image1 and image2 must be provided in TRAIN mode.")
            
            feat1 = self.extract_and_pool_features(image1)
            feat2 = self.extract_and_pool_features(image2)
            
            return feat1, feat2
            
        elif mode in [ForwardMode.QUERY, ForwardMode.REFERENCE]:
            return self.extract_and_pool_features(image1)
            
        else:
            raise ValueError(f"Invalid forward mode: {mode}. Must be of type ForwardMode.")