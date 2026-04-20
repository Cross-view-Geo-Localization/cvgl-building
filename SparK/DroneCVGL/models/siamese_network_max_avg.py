import torch
import timm
import numpy as np
import torch.nn as nn

from utils.registry import register_model
from utils.predict import ForwardMode


class MixedAdaptivePool2d(nn.Module):
    """
    Thay thế AdaptiveAvgPool2d bằng weighted combination của avg + max pooling.
    Output shape giống hệt: [B, C, 1, 1]
    """
    def __init__(self, output_size=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size)
        self.max_pool = nn.AdaptiveMaxPool2d(output_size)
        # Learnable scalar, khởi tạo = 1.0 → weight ban đầu = 1.0 (thiên về avg)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dim() == 4:
            feat_avg = self.avg_pool(x)          # [B, C, 1, 1]
            feat_max = self.max_pool(x)          # [B, C, 1, 1]
        elif x.dim() == 3:
            feat_avg = x.mean(dim=1, keepdim=True).unsqueeze(-1)
            feat_max = x.max(dim=1, keepdim=True)[0].unsqueeze(-1)
        else:
            raise ValueError(f"Unexpected feature dim: {x.dim()}")

        weight = torch.clamp((self.alpha + 1.0) / 2.0, min=0.0, max=1.0)
        return weight * feat_avg + (1.0 - weight) * feat_max
    
def replace_adaptive_pool(model):
    """
    Duyệt toàn bộ model, thay thế AdaptiveAvgPool2d đầu tiên
    tìm thấy trong SelectAdaptivePool2d bằng MixedAdaptivePool2d.
    """
    for name, module in model.named_modules():
        if module.__class__.__name__ == "SelectAdaptivePool2d":
            # Thay đúng attribute 'pool' hoặc tên sub-module chứa AdaptiveAvgPool2d
            for child_name, child in module.named_children():
                if isinstance(child, nn.AdaptiveAvgPool2d):
                    setattr(module, child_name, MixedAdaptivePool2d(output_size=1))
                    print(f"✅ Replaced {child_name} in {name}")
    return model


@register_model("SiameseNetworkMaxAvg")
class SiameseNetworkMaxAvg(nn.Module):

    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=384):
                 
        super(SiameseNetworkMaxAvg, self).__init__()
        
        self.img_size = img_size
        
        # Thêm global_pool='' để lấy feature map thô (chưa qua pooling) từ backbone
        if "vit" in model_name:
            self.model = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0, 
                img_size=img_size,
                global_pool=''
            ) 
        else:
            self.model = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0,
                global_pool=''
            )

        # if "vit" in model_name:
        #     self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True, img_size=img_size) 
        # else:
        #     self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True, num_classes=0)

        # for name, module in self.model.named_modules():
        #     if module.__class__.__name__ == "SelectAdaptivePool2d":
        #         # Thay đúng attribute 'pool' hoặc tên sub-module chứa AdaptiveAvgPool2d
        #         for child_name, child in module.named_children():
        #             if isinstance(child, nn.AdaptiveAvgPool2d):
        #                 setattr(module, child_name, MixedAdaptivePool2d(output_size=1))
        #                 print(f"✅ Replaced {child_name} in {name}")
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # Khởi tạo 2 module pooling dành cho CNN feature maps
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
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

    def extract_and_pool_features(self, x):
        """
        Hàm phụ trợ để trích xuất feature từ backbone và áp dụng logic pool + concat.
        """
        features = self.model(x)
        
        if features.dim() == 4:
            feat_avg = self.avg_pool(features).flatten(1) # [B, C]
            feat_max = self.max_pool(features).flatten(1) # [B, C]
        elif features.dim() == 3:
            feat_avg = features.mean(dim=1)               # [B, C]
            feat_max = features.max(dim=1)[0]             # [B, C]
        else:
            raise ValueError(f"Unexpected feature dimension from backbone: {features.dim()}")
        
        weight = torch.clamp((self.alpha + 1.0) / 2.0, min=0.0, max=1.0)
        # weight = torch.sigmoid(self.alpha)
        
        f_approx = weight * feat_avg + (1.0 - weight) * feat_max
        
        return f_approx
    
        # # Nối đặc trưng average và max. Output shape: [B, 2*C]
        # return torch.cat([feat_avg, feat_max], dim=1)

        
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
            
            # Sử dụng hàm extract_and_pool_features thay vì gọi thẳng self.model
            feat1 = self.extract_and_pool_features(image1)
            feat2 = self.extract_and_pool_features(image2)
            
            return feat1, feat2
            
        elif mode == ForwardMode.QUERY:
            return self.extract_and_pool_features(image1)
            
        elif mode == ForwardMode.REFERENCE:
            return self.extract_and_pool_features(image1)
            
        else:
            raise ValueError(f"Invalid forward mode: {mode}. Must be of type ForwardMode.")
